# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 Flash Vision-Exp image preprocessing.

Faithful port of the official reference implementation shipped in
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (inference/image_processor.py and
the image-visibility helpers from inference/model.py).

This module is deliberately free of vLLM imports so the exact reference
semantics can be unit-tested standalone. All image sizing parameters are
read from the model config (vision_* keys) by the callers; nothing is
hard-coded to the DeepSeek-V4-Flash-Vision-Exp defaults.

Semantic image token types (order matters, matches the reference):
    IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)

The model carries these as *synthetic* input ids ``vocab_size + type``:
they never index the LM embedding table (multimodal placeholder machinery
scatters the real embeddings over those positions), but the decoder
generates per-position type/routing/visibility metadata from them exactly
like DeepSeek's reference Transformer does.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

IMAGE_START = 0
IMAGE_PAD = 1
IMAGE = 2
IMAGE_NEW_LINE = 3
IMAGE_END = 4
NUM_IMAGE_TOKEN_TYPES = 5

COMPRESS_PAD_TO = 4

# The chat-level placeholder string emitted for every OpenAI image content
# block; must match encoding/encoding_dsv4.py in the reference repo.
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


@dataclass
class ProcessedImage:
    """Everything the model and the prompt expander need for one image.

    - patches: float32 numpy array of shape (n_vit_h * n_vit_w, 3, patch, patch)
      in row-major (n_vit_h, n_vit_w) order, channel-first, values in [-1, 1].
    - n_vit_h / n_vit_w: ViT grid size in patches.
    - n_llm_h / n_llm_w: aligner output grid size (LM-token cells).
    """

    patches: "np.ndarray"
    n_vit_h: int
    n_vit_w: int
    n_llm_h: int
    n_llm_w: int


@dataclass
class ImageBlockLayout:
    """One image's expanded token block in the final prompt.

    - start: index of the first block token in the final expanded prompt.
    - compress_pad: number of leading IMAGE_PAD tokens before IMAGE_START
      (``COMPRESS_PAD_TO - 1 - start % COMPRESS_PAD_TO``; aligns the first grid
      token to the C4 compressor blocks). Offset-dependent; everything else in
      the block is content-only.
    - types: list of semantic types (one per placeholder slot, incl. pads).
    - perm: aligner-output permutation; block[types == IMAGE][i] receives the
      aligner output indexed by perm[i].
    - token_ids: the synthetic input ids (vocab_size + type) for the block.
    """

    start: int
    compress_pad: int
    types: list[int]
    perm: list[int]
    token_ids: list[int]


def grid_tokens(
    best_height: int, best_width: int, patch_size: int, downsample_ratio: int
) -> tuple[int, int, int]:
    """Number of LLM tokens the aligner grid occupies (N-layout, incl. row/align
    padding). Mirrors the reference ``grid_tokens``."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int, int]:
    """Largest patch-multiple size whose grid costs at most ``max_n_token``.

    Mirrors the reference ``solve_resize_ratio``."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    """Shrink the candidate patch-multiple size until the grid fits within
    ``max_n_token - (COMPRESS_PAD_TO - 1)``.

    Mirrors the reference ``safe_resize``."""
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def process_image(
    image: Image.Image,
    *,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
    min_pixels: int,
    max_wh_ratio: int | None,
) -> ProcessedImage:
    """Load and transform one PIL image into ViT patches.

    Faithful port of the reference ``load_image``: clamp the w/h ratio,
    upscale below ``min_pixels``, round up to a patch multiple, cap the grid
    cost through ``safe_resize``, pad to (127, 127, 127) gray (or resize
    directly for extreme aspect ratios), normalize to [-1, 1] and cut into
    (n_vit_h, n_vit_w) row-major patches.
    """
    image = image.convert("RGB")
    width, height = image.size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        downsample_ratio,
        max_n_token,
    )
    n_vit_h, n_vit_w = best_height // patch_size, best_width // patch_size
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    # (C, H/vit_h, p, W/vit_w, p) -> (vit_h, vit_w, C, p, p) -> flattened grid.
    patches = (
        x.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .transpose(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )
    return ProcessedImage(
        patches=np.ascontiguousarray(patches),
        n_vit_h=n_vit_h,
        n_vit_w=n_vit_w,
        n_llm_h=n_llm_h,
        n_llm_w=n_llm_w,
    )


def build_image_block(
    n_llm_h: int,
    n_llm_w: int,
    start_pos: int,
    vocab_size: int,
) -> ImageBlockLayout:
    """Build the N-layout token types (final order) and the aligner-row order
    for IMAGE slots.

    Faithful port of the reference ``build_image_block``. ``start_pos`` is
    the index of the block's first token in the final expanded prompt; the
    leading IMAGE_PAD count (``COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO``)
    aligns the first grid token to the C4 compressor blocks.
    """
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2

    types = [IMAGE] * n_llm_w + [IMAGE_NEW_LINE] * 1
    types = types * n_llm_h + [IMAGE_PAD] * (row_len * pad_h)

    arange = np.arange(rows * row_len).reshape(rows // 2, 2, row_len)
    order = arange.transpose(0, 2, 1).reshape(-1).tolist()

    image_idx = np.full((rows * row_len,), -1, dtype=np.int64)
    image_idx.reshape(rows, row_len)[:n_llm_h, :n_llm_w] = np.arange(
        n_llm_h * n_llm_w, dtype=np.int64
    ).reshape(n_llm_h, n_llm_w)
    perm = [int(v) for v in image_idx[order] if v >= 0]

    types = [IMAGE_PAD] * compress_pad + [IMAGE_START] + [types[i] for i in order]
    types += [IMAGE_PAD] * pad_last + [IMAGE_END]
    return ImageBlockLayout(
        start=start_pos,
        compress_pad=compress_pad,
        types=list(types),
        perm=perm,
        token_ids=[vocab_size + t for t in types],
    )


def get_image_visible(
    input_ids,
    vocab_size: int,
    max_image_tokens: int,
) -> tuple[list[int], list[int]]:
    """Per-token visible counts to the left/right within each
    ``[IMAGE_START, IMAGE_END]`` span.

    ``input_ids`` is the token stream; image block positions carry the
    synthetic ids ``vocab_size + type``. Mirrors ``get_image_visible`` in the
    reference ``inference/model.py``.
    """
    sentinels = np.asarray(input_ids)
    seqlen = sentinels.shape[-1]
    idx = np.arange(seqlen, dtype=np.int64)
    is_start = sentinels == vocab_size + IMAGE_START
    is_end = sentinels == vocab_size + IMAGE_END
    valid = (np.cumsum(is_start) > np.cumsum(is_end)) | is_end
    starts = np.maximum.accumulate(np.where(is_start, idx, 0))
    left = (idx - starts) * valid
    ends = np.minimum.accumulate(np.where(is_end, idx, seqlen)[::-1])[::-1]
    right = (ends - idx) * valid
    return (
        np.minimum(left, max_image_tokens - 1).tolist(),
        np.minimum(right, max_image_tokens).tolist(),
    )


def visible_window(
    pos: int,
    left: int,
    right: int,
    window_size: int,
    seq_start: int = 0,
) -> tuple[int, int]:
    """Sliding-window bounds [start, end] (inclusive, absolute positions) for
    one query at absolute position ``pos`` given its ``left``/``right``
    visibility counts.

    Combines the reference ``get_window_topk_idxs_visible`` semantics for a
    single token: the window reaches back to the span's IMAGE_START when the
    span is longer than the window, and bidirectionally up to its IMAGE_END
    when the query is inside the span. Outside spans ``left == right == 0``
    and this degenerates to the plain causal window.
    """
    left_add = max(left - (window_size - 1), 0)
    start = max(pos - (window_size - 1) - left_add, seq_start)
    end = pos + right
    return start, end


def get_image_visible_torch(
    input_ids: "torch.Tensor",
    vocab_size: int,
    max_image_tokens: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Torch port of the reference ``get_image_visible`` (inference/model.py).

    Operates directly on the flattened per-step ``input_ids`` tensor on its
    device (no host sync). Synthetic image ids are ``vocab_size + type``;
    returns per-token (left, right) visible counts, capped at
    ``max_image_tokens - 1`` / ``max_image_tokens`` exactly like the
    reference, as int32 tensors for the prefill combine kernel.
    """
    import torch

    seqlen = input_ids.shape[-1]
    idx = torch.arange(seqlen, dtype=torch.int32, device=input_ids.device)
    is_start = input_ids == vocab_size + IMAGE_START
    is_end = input_ids == vocab_size + IMAGE_END
    valid = (is_start.cumsum(-1) > is_end.cumsum(-1)) | is_end
    starts = torch.where(is_start, idx, 0).cummax(-1).values
    left = (idx - starts) * valid
    ends = (
        torch.where(is_end, idx, seqlen)
        .flip(-1)
        .cummin(-1)
        .values
        .flip(-1)
    )
    right = (ends - idx) * valid
    return (
        left.clamp(max=max_image_tokens - 1).to(torch.int32),
        right.clamp(max=max_image_tokens).to(torch.int32),
    )
