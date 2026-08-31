# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for DeepSeek V4 Vision-Exp tests.

The production modules under ``vllm/models/deepseek_v4/`` that these tests
exercise (``image_processing``, ``vision``, ``vision_routing``) are
deliberately free of vLLM imports so they can be loaded standalone on CPU.
We load them by file path to avoid importing the full ``vllm`` package
(which requires CUDA/Triton and only exists on the GPU CI runners).
"""

import importlib
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DSV4_DIR = REPO_ROOT / "vllm" / "models" / "deepseek_v4"
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


def load_module(name: str, path: Path):
    """Import a python file by path without triggering package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_impl(module_name: str):
    """Load a DeepSeek V4 production module (e.g. "image_processing")."""
    return load_module(f"dsv4_impl_{module_name}", DSV4_DIR / f"{module_name}.py")


def import_dsv4_submodule(module_name: str):
    """Import a vllm.models.deepseek_v4 submodule without executing the
    package __init__ (which pulls in CUDA/Triton model code)."""
    import sys
    import types

    pkg_name = "vllm.models.deepseek_v4"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(DSV4_DIR)]
        sys.modules[pkg_name] = pkg
    return importlib.import_module(f"{pkg_name}.{module_name}")


def load_reference(module_name: str):
    """Load a vendored reference file (e.g. "image_processor", "vision").

    Reference modules live under a synthetic "dsv4_reference" package so
    their relative imports (e.g. model_excerpt -> image_processor) work.
    """
    import sys
    import types

    pkg_name = "dsv4_reference"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(REFERENCE_DIR)]
        sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.{module_name}", REFERENCE_DIR / f"{module_name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg_name}.{module_name}"] = mod
    spec.loader.exec_module(mod)
    return mod
