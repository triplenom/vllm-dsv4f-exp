# SPDX-License-Identifier: Apache-2.0
"""Shape and parity tests for the DeepSeek V4 Vision-Exp ViT + Aligner port
(``vllm/models/deepseek_v4/vision.py``) against the vendored official
reference (./reference/vision.py).

Uses a tiny artificial config; both modules share the same parameter names,
so weights are copied across and outputs must match exactly.
"""

import torch
from types import SimpleNamespace

from ._util import load_impl, load_reference

impl = load_impl("vision")
ref = load_reference("vision")

# Tiny config exercising non-trivial shapes (odd grid, downsample padding).
ARGS = SimpleNamespace(
    vision_n_layers=2,
    vision_dim=64,
    vision_n_heads=4,
    vision_inter_dim=112,
    vision_patch_size=14,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=3,
    dim=128,  # LM hidden size (reference Aligner arg name)
)
HIDDEN_SIZE = ARGS.dim


def _build_pair():
    torch.manual_seed(0)
    ref_vit = ref.ViT(ARGS)
    impl_vit = impl.DeepseekV4ViT(ARGS)
    impl_vit.load_state_dict(ref_vit.state_dict())

    ref_aligner = ref.Aligner(ARGS)
    impl_aligner = impl.DeepseekV4Aligner(
        ARGS.vision_dim, HIDDEN_SIZE, ARGS.vision_downsample_ratio
    )
    impl_aligner.load_state_dict(ref_aligner.state_dict())
    return ref_vit, impl_vit, ref_aligner, impl_aligner


def test_module_structure():
    vit = impl.DeepseekV4ViT(ARGS)
    assert len(vit.blocks) == ARGS.vision_n_layers
    assert vit.patch_embed.proj.in_features == 3 * ARGS.vision_patch_size**2
    assert vit.patch_embed.proj.out_features == ARGS.vision_dim
    assert vit.norm.weight.shape == (ARGS.vision_dim,)

    aligner = impl.DeepseekV4Aligner(
        ARGS.vision_dim, HIDDEN_SIZE, ARGS.vision_downsample_ratio
    )
    assert aligner.w1.in_features == ARGS.vision_dim * ARGS.vision_downsample_ratio**2
    assert aligner.w1.out_features == HIDDEN_SIZE
    assert aligner.w2.out_features == HIDDEN_SIZE


def test_rope_tables_match_reference():
    for n_h, n_w in [(6, 6), (5, 8), (1, 1)]:
        ref_cos, ref_sin = ref.get_vision_cos_sin(
            n_h, n_w, ARGS.vision_dim // ARGS.vision_n_heads // 2,
            ARGS.vision_rope_theta,
        )
        got_cos, got_sin = impl.get_vision_cos_sin(
            n_h, n_w, ARGS.vision_dim // ARGS.vision_n_heads // 2,
            ARGS.vision_rope_theta,
        )
        torch.testing.assert_close(got_cos, ref_cos)
        torch.testing.assert_close(got_sin, ref_sin)


def test_vit_matches_reference():
    ref_vit, impl_vit, _, _ = _build_pair()
    n_h, n_w = 6, 9  # 54 patches
    patches = torch.randn(n_h * n_w, 3, ARGS.vision_patch_size, ARGS.vision_patch_size)
    with torch.no_grad():
        ref_out = ref_vit(patches, n_h, n_w)
        got_out = impl_vit(patches, n_h, n_w)
    assert got_out.shape == (n_h * n_w, ARGS.vision_dim)
    torch.testing.assert_close(got_out, ref_out, rtol=1e-4, atol=1e-5)


def test_aligner_matches_reference():
    _, _, ref_aligner, impl_aligner = _build_pair()
    n_h, n_w = 7, 5  # odd dims exercise the F.pad path
    x = torch.randn(n_h * n_w, ARGS.vision_dim)
    with torch.no_grad():
        ref_out = ref_aligner(x, n_h, n_w)
        got_out = impl_aligner(x, n_h, n_w)
    # ceil(7/3) * ceil(5/3) = 3 * 2 = 6 LM cells
    assert got_out.shape == (6, HIDDEN_SIZE)
    assert ref_out.shape == got_out.shape
    torch.testing.assert_close(got_out, ref_out, rtol=1e-4, atol=1e-5)


def test_vit_aligner_pipeline_output_dim():
    """The full encode path produces one LM-hidden vector per grid cell."""
    ref_vit, impl_vit, ref_aligner, impl_aligner = _build_pair()
    n_h, n_w = 6, 9
    patches = torch.randn(n_h * n_w, 3, ARGS.vision_patch_size, ARGS.vision_patch_size)
    with torch.no_grad():
        got = impl_aligner(impl_vit(patches, n_h, n_w), n_h, n_w)
        want = ref_aligner(ref_vit(patches, n_h, n_w), n_h, n_w)
    n_cells = -(-n_h // ARGS.vision_downsample_ratio) * (
        -(-n_w // ARGS.vision_downsample_ratio)
    )
    assert got.shape == (n_cells, HIDDEN_SIZE)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
