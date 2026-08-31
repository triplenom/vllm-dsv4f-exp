# SPDX-License-Identifier: Apache-2.0
"""Verbatim excerpts from the official DeepSeek V4 Flash Vision-Exp
reference ``inference/model.py`` (Apache-2.0), used as the specification
for parity tests.

- ``get_image_visible`` / ``get_window_topk_idxs_visible``: copied
  unchanged (only the ``image_processor`` import was redirected to the
  vendored copy in this directory).
- ``gate_forward_reference``: the exact body of ``Gate.forward`` with
  ``self`` attributes passed as explicit arguments.
"""

import torch
import torch.nn.functional as F

from .image_processor import IMAGE_END, IMAGE_START


def get_image_visible(input_ids: torch.Tensor, vocab_size: int, max_image_tokens: int):
    """Per-token visible counts to the left/right within each [IMAGE_START, IMAGE_END] span."""
    seqlen = input_ids.size(1)
    idx = torch.arange(seqlen, dtype=torch.int32).unsqueeze(0)
    is_start = input_ids == vocab_size + IMAGE_START
    is_end = input_ids == vocab_size + IMAGE_END
    valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end
    starts = torch.where(is_start, idx, 0).cummax(1)[0]
    left = (idx - starts) * valid
    ends = torch.where(is_end, idx, seqlen).flip(1).cummin(1)[0].flip(1)
    right = (ends - idx) * valid
    return left.clamp(max=max_image_tokens - 1), right.clamp(max=max_image_tokens)


def get_window_topk_idxs_visible(window_size: int, seqlen: int, left: torch.Tensor, right: torch.Tensor,
                                 max_image_tokens: int):
    width = min(seqlen, window_size + max_image_tokens)
    idx = torch.arange(seqlen).unsqueeze(0)
    left_add = (left - (window_size - 1)).clamp(min=0)
    starts = (idx - (window_size - 1) - left_add).clamp(min=0)
    matrix = starts.unsqueeze(-1) + torch.arange(width)
    matrix = torch.where(matrix > (idx + right).unsqueeze(-1), -1, matrix)
    return matrix.int().contiguous()


def gate_forward_reference(
    x: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    weight: torch.Tensor,
    score_func: str,
    route_scale: float,
    topk: int,
    vocab_size: int,
    tid2eid: torch.Tensor | None,
    bias: torch.Tensor | None,
    bias_vl: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact body of the reference ``Gate.forward`` (self attrs as args)."""
    scores = F.linear(x.float(), weight.float())
    if score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    original_scores = scores
    image_mask = (input_ids >= vocab_size) if bias_vl is not None else None
    # Bias shifts scores for expert selection (topk) but does not affect routing weights.
    if tid2eid is not None:
        if image_mask is None:
            indices = tid2eid[input_ids]
        else:
            indices = tid2eid[torch.where(image_mask, 0, input_ids)]
            vl_indices = (scores + bias_vl).topk(topk, dim=-1)[1]
            indices = torch.where(image_mask.unsqueeze(-1), vl_indices.to(indices.dtype), indices)
    else:
        if image_mask is None:
            scores = scores + bias
        else:
            scores = scores + torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
        indices = scores.topk(topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    if score_func != "softmax":
        weights /= weights.sum(dim=-1, keepdim=True)
    weights *= route_scale
    return weights, indices
