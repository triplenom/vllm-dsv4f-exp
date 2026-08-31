# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 checkpoint weight-name mapping.

Lives in its own module (instead of ``nvidia/model.py``) so the mapping can
be unit-tested on CPU without importing the CUDA model implementation.
"""

import regex as re

from vllm.model_executor.models.utils import WeightsMapper


def is_dsv4_vision_weight(name: str) -> bool:
    """True for vision-tower/aligner weights at any prefix depth.

    Checkpoint/HF names arrive as ``vision.…`` / ``aligner.…`` and become
    ``model.vision.…`` / ``model.aligner.…`` after prefix mapping; inside
    ``DeepseekV4Model.load_weights`` the ``model.`` prefix is stripped again,
    so the relative names START with ``vision.``/``aligner.``. Match both
    forms. Their ``w1``/``w2`` tensors must never be treated as
    shared-expert ``gate_up_proj`` shards by the stacked-params mapping.
    """
    return (
        name.startswith("vision.")
        or name.startswith("aligner.")
        or ".vision." in name
        or ".aligner." in name
    )


def make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). FP8 linear and
        # shared experts use Fp8LinearMethod's block scales, which
        # register as ``weight_scale_inv``.
        scale_regex = {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
            # Vision-Exp: vision tower and aligner live on the model.
            "vision.": "model.vision.",
            "aligner.": "model.aligner.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
            # Vision-Exp routing bias ("bias_vl" does not end with "bias",
            # so the two suffixes never collide).
            ".ffn.gate.bias_vl": ".ffn.gate.e_score_correction_bias_vl",
            # Vision-Exp special image embeddings (bare names, no .weight).
            "image_start": "model.image_start",
            "image_end": "model.image_end",
            "image_newline": "model.image_newline",
            "image_pad": "model.image_pad",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )
