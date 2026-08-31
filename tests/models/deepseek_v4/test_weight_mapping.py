# SPDX-License-Identifier: Apache-2.0
"""Weight-name mapping tests for DeepSeek V4 Vision-Exp checkpoints.

Uses the real mapper factory (``vllm/models/deepseek_v4/weights.py``) and
representative names covering every tensor family present in the official
``model.safetensors.index.json`` of deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.
"""

import pytest

pytest.importorskip(
    "vllm.model_executor.models.utils", reason="needs the vLLM model utils"
)

from ._util import import_dsv4_submodule

make_deepseek_v4_weights_mapper = import_dsv4_submodule(
    "weights"
).make_deepseek_v4_weights_mapper

MAPPER = make_deepseek_v4_weights_mapper("fp4")

# (checkpoint_name, expected_mapped_name) — one per weight family in the
# actual Vision-Exp checkpoint index.
EXPECTED = [
    # Vision tower
    ("vision.patch_embed.proj.weight", "model.vision.patch_embed.proj.weight"),
    ("vision.patch_embed.proj.bias", "model.vision.patch_embed.proj.bias"),
    ("vision.blocks.0.attn.wqkv.weight", "model.vision.blocks.0.attn.wqkv.weight"),
    ("vision.blocks.0.attn.wqkv.bias", "model.vision.blocks.0.attn.wqkv.bias"),
    ("vision.blocks.31.attn.wo.weight", "model.vision.blocks.31.attn.wo.weight"),
    ("vision.blocks.31.attn.wo.bias", "model.vision.blocks.31.attn.wo.bias"),
    ("vision.blocks.3.mlp.w1.weight", "model.vision.blocks.3.mlp.w1.weight"),
    ("vision.blocks.3.mlp.w2.weight", "model.vision.blocks.3.mlp.w2.weight"),
    ("vision.blocks.0.norm1.weight", "model.vision.blocks.0.norm1.weight"),
    ("vision.blocks.0.norm2.weight", "model.vision.blocks.0.norm2.weight"),
    ("vision.norm.weight", "model.vision.norm.weight"),
    # Aligner
    ("aligner.w1.weight", "model.aligner.w1.weight"),
    ("aligner.w1.bias", "model.aligner.w1.bias"),
    ("aligner.w2.weight", "model.aligner.w2.weight"),
    ("aligner.w2.bias", "model.aligner.w2.bias"),
    # Special image embeddings (bare names, no .weight)
    ("image_start", "model.image_start"),
    ("image_end", "model.image_end"),
    ("image_newline", "model.image_newline"),
    ("image_pad", "model.image_pad"),
    # LM gate: text bias, vision bias, hash table
    ("layers.0.ffn.gate.weight", "model.layers.0.ffn.gate.weight"),
    ("layers.0.ffn.gate.bias", "model.layers.0.ffn.gate.e_score_correction_bias"),
    ("layers.0.ffn.gate.bias_vl", "model.layers.0.ffn.gate.e_score_correction_bias_vl"),
    ("layers.0.ffn.gate.tid2eid", "model.layers.0.ffn.gate.tid2eid"),
    ("layers.42.ffn.gate.bias_vl", "model.layers.42.ffn.gate.e_score_correction_bias_vl"),
    # Shared experts: w1/w3 -> gate_up_proj happens in the model's
    # stacked_params_mapping at load time (not in the WeightsMapper).
    ("layers.5.ffn.shared_experts.w1.weight", "model.layers.5.ffn.shared_experts.w1.weight"),
    ("layers.5.ffn.shared_experts.w2.weight", "model.layers.5.ffn.shared_experts.down_proj.weight"),
    # Experts (fp4: .scale -> .weight_scale)
    ("layers.5.ffn.experts.17.w1.weight", "model.layers.5.ffn.experts.17.w1.weight"),
    ("layers.5.ffn.experts.17.w1.scale", "model.layers.5.ffn.experts.17.w1.weight_scale"),
    # Attention
    ("layers.5.attn.wq_a.weight", "model.layers.5.attn.wq_a.weight"),
    ("layers.5.attn.wq_a.scale", "model.layers.5.attn.wq_a.weight_scale_inv"),
    ("layers.5.attn.attn_sink", "model.layers.5.attn.attn_sink"),
    ("layers.5.attn.compressor.ape", "model.layers.5.attn.compressor.ape"),
    ("layers.5.attn_norm.weight", "model.layers.5.attn_norm.weight"),
    # Top level
    ("embed.weight", "model.embed_tokens.weight"),
    ("norm.weight", "model.norm.weight"),
    ("head.weight", "lm_head.weight"),
    ("hc_head_fn", "model.hc_head_fn"),
    ("hc_head_base", "model.hc_head_base"),
    ("hc_head_scale", "model.hc_head_scale"),
    # MTP subtree (kept under model.mtp.*; skipped by the main loader)
    ("mtp.0.attn.wq_a.weight", "model.mtp.0.attn.wq_a.weight"),
    ("mtp.1.ffn.gate.bias_vl", "model.mtp.1.ffn.gate.e_score_correction_bias_vl"),
    ("mtp.2.markov_head.markov_w1.weight", "model.mtp.2.markov_head.markov_w1.weight"),
]


@pytest.mark.parametrize("name,expected", EXPECTED)
def test_weight_name_mapping(name: str, expected: str):
    assert MAPPER._map_name(name) == expected


def test_bias_vl_never_maps_to_text_bias():
    # ".ffn.gate.bias" and ".ffn.gate.bias_vl" must stay distinct.
    assert MAPPER._map_name("layers.0.ffn.gate.bias").endswith("e_score_correction_bias")
    assert MAPPER._map_name("layers.0.ffn.gate.bias_vl").endswith(
        "e_score_correction_bias_vl"
    )


def test_fp8_mapper_variant():
    mapper = make_deepseek_v4_weights_mapper("fp8")
    assert (
        mapper._map_name("layers.5.ffn.experts.17.w1.scale")
        == "model.layers.5.ffn.experts.17.w1.weight_scale_inv"
    )
    assert (
        mapper._map_name("aligner.w1.weight") == "model.aligner.w1.weight"
    )


is_dsv4_vision_weight = import_dsv4_submodule("weights").is_dsv4_vision_weight


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Relative names as seen by DeepseekV4Model.load_weights (the
        # "model." prefix is stripped by AutoWeightsLoader). Regression:
        # "aligner.w1.bias" reached the stacked gate_up_proj mapping because
        # the old guard only matched the dotted infix form.
        ("aligner.w1.weight", True),
        ("aligner.w1.bias", True),
        ("aligner.w2.weight", True),
        ("vision.blocks.3.mlp.w1.weight", True),
        ("vision.patch_embed.proj.weight", True),
        # Fully prefixed forms also match.
        ("model.vision.blocks.0.mlp.w2.weight", True),
        ("model.aligner.w1.bias", True),
        # Ordinary LM weights must NOT match.
        ("layers.0.ffn.shared_experts.w1.weight", False),
        ("layers.61.ffn.experts.5.w1.weight", False),
        ("embed_tokens.weight", False),
        ("layers.0.attn.wq_a.weight", False),
    ],
)
def test_is_dsv4_vision_weight(name, expected):
    assert is_dsv4_vision_weight(name) is expected
