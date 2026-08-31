# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 Flash Vision-Exp MoE routing.

Faithful port of the reference ``Gate.forward`` (inference/model.py):

- text positions route exactly like the text-only model (hash-table lookup
  in the first ``num_hash_layers`` layers, score-based top-k elsewhere);
- image positions (synthetic ids ``>= vocab_size``) never use the text hash
  table; they always use score-based selection with the vision-specific
  bias ``bias_vl``;
- the bias (text or vision) shifts expert *selection* only; final routing
  weights always come from the un-biased scores;
- scoring matches ``scoring_func`` (DeepSeek V4 uses ``sqrtsoftplus``),
  renormalization matches ``norm_topk_prob``, and the output is scaled by
  ``routed_scaling_factor``.

Everything is expressed with branch-free tensor ops (no ``.any()`` guards)
so the function can live inside the torch.compile region of the model.
"""

import torch
import torch.nn.functional as F


def image_token_mask(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Boolean mask of image positions from the synthetic id convention.

    Image block positions carry ids ``vocab_size + type`` with
    ``type in [IMAGE_START .. IMAGE_END]``; ordinary tokens are ``< vocab_size``.
    """
    return input_ids >= vocab_size


def compute_moe_scores(gating_output: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Raw routing scores from gate logits, matching the reference Gate."""
    if scoring_func == "softmax":
        return gating_output.float().softmax(dim=-1)
    if scoring_func == "sigmoid":
        return gating_output.float().sigmoid()
    return torch.sqrt(F.softplus(gating_output.float()))


def dsv4_vision_aware_topk(
    gating_output: torch.Tensor,
    *,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor | None,
    vision_correction_bias: torch.Tensor | None,
    tid2eid: torch.Tensor | None,
    input_ids: torch.Tensor,
    vocab_size: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert selection for DeepSeek V4 with Vision-Exp routing semantics.

    Reproduces ``Gate.forward`` from the reference for both hash-routing
    layers (``tid2eid is not None``) and score-routing layers. Callers with
    text-only checkpoints keep using the existing FusedMoE topk kernels
    instead; this path is entered only when ``vision_correction_bias``
    exists, i.e. for Vision-Exp-configured models.

    Args:
        gating_output: [M, n_experts] gate logits (fp32 in the model).
        scoring_func: "softmax" | "sigmoid" | "sqrtsoftplus" (same as MoE).
        e_score_correction_bias: text routing bias [n_experts] or None.
        vision_correction_bias: vision routing bias [n_experts] (must not be
            None on this code path).
        tid2eid: [vocab_size, topk] hash table at hash layers, else None.
        input_ids: [M] token ids (may include synthetic image ids).
        vocab_size: LM vocabulary size; ids >= vocab_size are image positions.
        topk: experts per token.
        renormalize: renormalize routing weights (norm_topk_prob).
        routed_scaling_factor: text/model route scale.

    Returns:
        (topk_weights [M, topk] fp32, topk_indices [M, topk] indices_dtype)
    """
    if vision_correction_bias is None:
        raise ValueError(
            "dsv4_vision_aware_topk requires a vision routing bias; "
            "use the standard FusedMoE topk path for text-only models."
        )

    scores = compute_moe_scores(gating_output, scoring_func)
    original_scores = scores
    image_mask = image_token_mask(input_ids, vocab_size)

    if tid2eid is not None:
        # Hash-routing layers: text positions keep the predetermined
        # tid2eid mapping (sentinel ids substituted with 0 before lookup,
        # exactly like the reference); image positions use score+bias_vl
        # top-k selection.
        safe_ids = torch.where(image_mask, 0, input_ids).long()
        hash_indices = tid2eid.index_select(0, safe_ids).to(torch.int64)
        vl_indices = torch.topk(
            scores + vision_correction_bias.to(scores.dtype), k=topk, dim=-1
        )[1]
        topk_indices = torch.where(
            image_mask.unsqueeze(-1),
            vl_indices,
            hash_indices,
        )
    else:
        # Score-routing layers: image positions replace the text bias with
        # the vision bias for selection only.
        if e_score_correction_bias is None:
            scores_for_choice = scores
        else:
            bias = torch.where(
                image_mask.unsqueeze(-1),
                vision_correction_bias.to(scores.dtype),
                e_score_correction_bias.to(scores.dtype),
            )
            scores_for_choice = scores + bias
        topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1)[1]

    topk_indices = topk_indices.to(indices_dtype)
    topk_weights = original_scores.gather(1, topk_indices.long())

    if scoring_func != "softmax" and renormalize:
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-20)
    topk_weights = topk_weights.float() * routed_scaling_factor
    return topk_weights, topk_indices
