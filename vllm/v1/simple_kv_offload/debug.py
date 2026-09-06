# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime debug helper for SimpleCPUOffload."""

import os
import sys
from typing import Any


def is_simple_kv_debug_enabled() -> bool:
    """Check if SimpleCPU offload debug logging is enabled at runtime."""
    val = os.environ.get("VLLM_SIMPLE_KV_DEBUG", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def debug_log(fmt: str, *args: Any) -> None:
    """Emit a concise structured [SIMPLE_KV_DEBUG] line to stdout."""
    if not is_simple_kv_debug_enabled():
        return
    try:
        msg = fmt % args if args else fmt
    except Exception:
        msg = f"{fmt} {args}"
    sys.stdout.write(f"[SIMPLE_KV_DEBUG] {msg}\n")
    sys.stdout.flush()


def format_hash(bhash: Any) -> str:
    """Format a BlockHash or bytes into a compact 8-char hex string."""
    if bhash is None:
        return "None"
    if isinstance(bhash, bytes):
        return bhash.hex()[:8]
    if isinstance(bhash, tuple) and len(bhash) > 0 and isinstance(bhash[0], bytes):
        return bhash[0].hex()[:8]
    if hasattr(bhash, "hex"):
        return bhash.hex()[:8]
    return str(bhash)[:8]