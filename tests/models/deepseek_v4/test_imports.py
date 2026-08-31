# Regression tests for import correctness in the DeepSeek V4 Vision-Exp
# integration. The Windows test environment loads modules through a stubbed
# package, so an import that only resolves on a real install (e.g.
# MultiModalEmbeddings from the wrong module) is only caught on Linux.
# These source-level checks pin the correct locations.

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
NVIDIA_MODEL = REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "nvidia" / "model.py"


def test_multimodal_embeddings_imported_from_interfaces():
    """MultiModalEmbeddings is a TypeAlias defined in
    vllm.model_executor.models.interfaces, NOT vllm.multimodal.inputs."""
    interfaces = (
        REPO_ROOT / "vllm" / "model_executor" / "models" / "interfaces.py"
    ).read_text(encoding="utf-8")
    assert "MultiModalEmbeddings" in interfaces

    src = NVIDIA_MODEL.read_text(encoding="utf-8")
    assert not re.search(
        r"from vllm\.multimodal\.inputs import.*MultiModalEmbeddings", src
    ), "MultiModalEmbeddings does not exist in vllm.multimodal.inputs"
    block = re.search(
        r"from vllm\.model_executor\.models\.interfaces import \((.*?)\)",
        src,
        re.DOTALL,
    )
    assert block is not None
    assert "MultiModalEmbeddings" in block.group(1)


def test_no_hardcoded_local_paths_or_credentials_in_new_modules():
    """New vision modules must not contain dev-machine paths or tokens."""
    pkg = REPO_ROOT / "vllm" / "models" / "deepseek_v4"
    for name in (
        "image_processing.py",
        "processing.py",
        "vision.py",
        "vision_routing.py",
        "weights.py",
    ):
        src = (pkg / name).read_text(encoding="utf-8")
        assert "import C:" not in src and "/tmp/" not in src
        assert "ghp_" not in src  # no credentials


def test_embed_input_ids_moves_is_multimodal_to_input_device():
    """Regression: the runner passes is_multimodal on CPU; combining it with
    CUDA input_ids in embed_input_ids crashed with a cross-device error on
    the very first request. The pad-mask computation must move the mask to
    the input_ids device first."""
    src = NVIDIA_MODEL.read_text(encoding="utf-8")
    pad_section = src[src.index("pad_mask") - 600 :]
    pad_section = pad_section[: pad_section.index("pad_mask") + 400]
    assert 'is_multimodal.to(device=input_ids.device' in pad_section


def test_proposer_mm_branch_guards_deepseek_v4():
    """Regression: with speculative decoding enabled, the base proposer's
    multimodal branch must not fire for text-only configs of a
    SupportsMultiModal model (DeepseekV4ForCausalLM is always
    SupportsMultiModal but only multimodal for vision configs), and must not
    assume the target config has image_token_index (DeepSeek V4 uses
    synthetic ids instead)."""
    src = (
        REPO_ROOT / "vllm" / "v1" / "spec_decode" / "llm_base_proposer.py"
    ).read_text(encoding="utf-8")
    assert (
        "if supports_multimodal(target_model) and self.supports_mm_inputs:"
        in src
    )
    assert 'hasattr(target_model.config, "image_token_index")' in src
