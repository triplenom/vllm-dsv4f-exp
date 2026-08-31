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
