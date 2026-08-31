# SPDX-License-Identifier: Apache-2.0
"""Parity tests: vLLM's DeepSeek V4 Vision-Exp image preprocessing vs the
official reference implementation (vendored under ./reference/).

Covers (DeepSeek V4 config values):
  * resize / aspect-ratio clamp / min-pixels upscale / max-token cap
  * padding and normalization
  * patchification layout (n_vit_h, n_vit_w)
  * image block types/perm (build_image_block)
  * get_image_visible / get_window_topk_idxs_visible semantics
"""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from ._util import load_impl, load_reference

impl = load_impl("image_processing")
ref_ip = load_reference("image_processor")
ref_model = load_reference("model_excerpt")

# DeepSeek-V4-Flash-Vision-Exp config values (from the official config.json).
VOCAB_SIZE = 129280
REF_ARGS = SimpleNamespace(
    vision_patch_size=14,
    vision_downsample_ratio=3,
    vision_max_n_token=384,
    vision_min_pixels=147456,
    vision_max_wh_ratio=8,
)

TEST_SIZES = [
    (100, 100),
    (512, 380),
    (800, 200),
    (200, 800),
    (4000, 500),
    (50, 50),
    (1024, 768),
    (640, 480),
    (1, 1),
    (3000, 20),
    (19, 11),
]


def make_image(width: int, height: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def ref_load_image(data: bytes):
    return ref_ip.load_image({"data": data}, REF_ARGS)


def impl_process_image(data: bytes):
    with Image.open(io.BytesIO(data)) as img:
        return impl.process_image(
            img,
            patch_size=REF_ARGS.vision_patch_size,
            downsample_ratio=REF_ARGS.vision_downsample_ratio,
            max_n_token=REF_ARGS.vision_max_n_token,
            min_pixels=REF_ARGS.vision_min_pixels,
            max_wh_ratio=REF_ARGS.vision_max_wh_ratio,
        )


@pytest.mark.parametrize("width,height", TEST_SIZES)
def test_process_image_matches_reference(width: int, height: int):
    data = make_image(width, height)
    ref_patches, ref_n_vit_h, ref_n_vit_w, ref_n_llm_h, ref_n_llm_w = (
        ref_load_image(data)
    )
    got = impl_process_image(data)

    assert (got.n_vit_h, got.n_vit_w) == (ref_n_vit_h, ref_n_vit_w)
    assert (got.n_llm_h, got.n_llm_w) == (ref_n_llm_h, ref_n_llm_w)
    assert got.patches.shape == tuple(ref_patches.shape)

    # The reference computes patches in bf16; the port keeps fp32 until the
    # model converts. Compare after the same cast.
    ref_np = ref_patches.float().numpy()
    assert np.allclose(got.patches, ref_np, atol=2e-3)


def test_token_budget_is_respected():
    for width, height in TEST_SIZES:
        data = make_image(width, height)
        got = impl_process_image(data)
        _, _, num_tokens = impl.grid_tokens(
            got.n_vit_h * REF_ARGS.vision_patch_size,
            got.n_vit_w * REF_ARGS.vision_patch_size,
            REF_ARGS.vision_patch_size,
            REF_ARGS.vision_downsample_ratio,
        )
        # The grid may use at most max_n_token - (COMPRESS_PAD_TO - 1) slots
        # (safe_resize) so the full block incl. pads fits in max_n_token.
        assert num_tokens <= REF_ARGS.vision_max_n_token - (
            impl.COMPRESS_PAD_TO - 1
        )


@pytest.mark.parametrize("n_llm_h,n_llm_w", [(10, 10), (5, 20), (20, 5), (1, 1), (7, 12)])
@pytest.mark.parametrize("start_pos", [0, 1, 2, 3, 5, 17, 128])
def test_build_image_block_matches_reference(n_llm_h: int, n_llm_w: int, start_pos: int):
    ref_types, ref_perm = ref_ip.build_image_block(n_llm_h, n_llm_w, start_pos)
    got = impl.build_image_block(n_llm_h, n_llm_w, start_pos, VOCAB_SIZE)

    assert got.types == ref_types.tolist()
    assert got.perm == ref_perm.tolist()
    assert got.start == start_pos
    assert got.compress_pad == impl.COMPRESS_PAD_TO - 1 - start_pos % impl.COMPRESS_PAD_TO
    assert got.token_ids == [VOCAB_SIZE + t for t in got.types]

    # Invariants: exactly one START/END, IMAGE count == grid cells, perm is a
    # permutation of the grid cell indices.
    assert got.types.count(impl.IMAGE_START) == 1
    assert got.types.count(impl.IMAGE_END) == 1
    assert got.types.count(impl.IMAGE) == n_llm_h * n_llm_w
    assert sorted(got.perm) == list(range(n_llm_h * n_llm_w))


def _make_span_ids(types: list[int]) -> list[int]:
    return [VOCAB_SIZE + t for t in types]


@pytest.mark.parametrize("max_image_tokens", [8, 384])
def test_get_image_visible_matches_reference(max_image_tokens: int):
    # text text [block] text text, with a block built for a 2x3 llm grid.
    block = impl.build_image_block(2, 3, start_pos=5, vocab_size=VOCAB_SIZE)
    ids = [11, 12, 13, 14, 15] + _make_span_ids(block.types) + [42, 43, 44]

    ref_left, ref_right = ref_model.get_image_visible(
        torch.tensor([ids]), VOCAB_SIZE, max_image_tokens
    )
    got_left, got_right = impl.get_image_visible(ids, VOCAB_SIZE, max_image_tokens)
    assert got_left == ref_left[0].tolist()
    assert got_right == ref_right[0].tolist()

    # The torch port must agree with both (it is what the model runs).
    t_left, t_right = impl.get_image_visible_torch(
        torch.tensor(ids), VOCAB_SIZE, max_image_tokens
    )
    assert t_left.tolist() == ref_left[0].tolist()
    assert t_right.tolist() == ref_right[0].tolist()


def test_get_image_visible_text_only_is_zero():
    ids = list(range(50))
    left, right = impl.get_image_visible(ids, VOCAB_SIZE, 384)
    assert left == [0] * 50
    assert right == [0] * 50


@pytest.mark.parametrize(
    "window_size,grid",
    [
        (128, (6, 6)),  # window longer than the span: left_add == 0
        (8, (10, 10)),  # span longer than the window: left_add > 0
        (8, (3, 4)),  # odd grid (row padding) with a short window
    ],
)
def test_visible_window_matches_reference(window_size: int, grid: tuple[int, int]):
    max_image_tokens = 384
    block = impl.build_image_block(*grid, start_pos=10, vocab_size=VOCAB_SIZE)
    ids = [1] * 10 + _make_span_ids(block.types) + [2] * 5
    left_t, right_t = ref_model.get_image_visible(
        torch.tensor([ids]), VOCAB_SIZE, max_image_tokens
    )
    ref_matrix = ref_model.get_window_topk_idxs_visible(
        window_size, len(ids), left_t, right_t, max_image_tokens
    )[0]

    for pos in range(len(ids)):
        start, end = impl.visible_window(
            pos,
            int(left_t[0, pos]),
            int(right_t[0, pos]),
            window_size,
        )
        ref_row = ref_matrix[pos]
        ref_valid = ref_row[ref_row >= 0]
        assert list(range(start, end + 1)) == ref_valid.tolist()


def test_image_placeholder_string():
    assert impl.IMAGE_PLACEHOLDER == "<｜deepseek_image｜>"
    assert (
        impl.NUM_IMAGE_TOKEN_TYPES
        == len(
            {
                impl.IMAGE_START,
                impl.IMAGE_PAD,
                impl.IMAGE,
                impl.IMAGE_NEW_LINE,
                impl.IMAGE_END,
            }
        )
        == 5
    )
