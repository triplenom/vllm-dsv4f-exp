# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for upstream #49059 / commit 34020ad: empty SM120
sparse-MLA prefill ranges.

FULL_AND_PIECEWISE CUDA-graph padding (or a CPU-KV-restore step that schedules
a prefill-phase request with 0 new tokens) can produce a prefill chunk with
query_start == query_end. An empty query range must never reach the FlashInfer
sparse kernel.

These tests exercise the real SM120 production path
``DeepseekV4FlashInferSM120Attention._forward_prefill`` with the FlashInfer
kernel mocked, so the chunk-skipping logic can be observed without CUDA kernels.
The original b3/b5 behavior (no guard) invokes the kernel for empty chunks and
fails; with the ``if query_start == query_end: continue`` guard the empty
chunks are skipped and non-empty chunks still run at their correct offsets.
"""
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as flashinfer_mod


class _FakeSM120Attention:
    """Minimal stand-in for the SM120 attention module to invoke the unbound
    production method ``DeepseekV4FlashInferSM120Attention._forward_prefill``."""

    def __init__(self, prefill_chunk_size: int = 4) -> None:
        self.compress_ratio = 1
        self.PREFILL_CHUNK_SIZE = prefill_chunk_size
        self.scale = 1.0
        self.attn_sink = None
        self._prepare_query = lambda query, output: query
        self._as_sparse_cache = lambda cache: cache
        self._get_workspace = lambda device: torch.empty(0)


def _swa_metadata(query_start_loc_cpu, num_prefills, num_prefill_tokens):
    return SimpleNamespace(
        num_prefills=num_prefills,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefill_tokens=num_prefill_tokens,
        query_start_loc_cpu=torch.tensor(query_start_loc_cpu, dtype=torch.int64),
        prefill_swa_indices=torch.zeros(
            (num_prefill_tokens, 1, 128), dtype=torch.int32
        ),
        prefill_swa_lens=torch.ones(num_prefill_tokens, dtype=torch.int32),
    )


def _run_prefill(
    monkeypatch,
    query_start_loc_cpu,
    num_prefills,
    num_prefill_tokens,
    prefill_chunk_size=4,
):
    """Invoke ``_forward_prefill`` and return the kernel call log as a list of
    ``(num_query_rows, first_row_index)`` tuples. An empty chunk (shape 0) would
    record ``(0, None)``; the guard means it is never reached."""
    calls = []

    def fake_sparse_mla(**kwargs):
        query = kwargs["query"]
        first = int(query[0, 0, 0].item()) if query.shape[0] else None
        calls.append((query.shape[0], first))

    monkeypatch.setattr(
        flashinfer_mod,
        "flashinfer_trtllm_batch_decode_sparse_mla_dsv4",
        fake_sparse_mla,
    )
    # The production path inspects the forward context for the Vision-Exp image
    # check; provide a neutral context so the test does not need a real forward
    # pass.
    monkeypatch.setattr(
        flashinfer_mod,
        "get_forward_context",
        lambda: SimpleNamespace(dsv4_image_visible=None),
    )

    attn = _FakeSM120Attention(prefill_chunk_size=prefill_chunk_size)
    query = (
        torch.arange(num_prefill_tokens, dtype=torch.float32)
        .view(-1, 1, 1)
        .expand(num_prefill_tokens, 16, 512)
    )

    flashinfer_mod.DeepseekV4FlashInferSM120Attention._forward_prefill(
        attn,
        q=query,
        compressed_k_cache=None,
        swa_k_cache=torch.empty((1, 64, 584), dtype=torch.uint8),
        output=torch.empty_like(query),
        attn_metadata=None,
        swa_metadata=_swa_metadata(
            query_start_loc_cpu, num_prefills, num_prefill_tokens
        ),
    )
    return calls


def test_flashinfer_sparse_prefill_skips_cudagraph_padding_chunks(monkeypatch):
    """Upstream #49059 semantics: a CUDA-graph padding chunk with an empty query
    range must not reach the FlashInfer kernel, while the non-empty chunk still
    runs."""
    calls = _run_prefill(
        monkeypatch,
        query_start_loc_cpu=[0, 1, 1, 1, 1, 1],
        num_prefills=5,
        num_prefill_tokens=2,
        prefill_chunk_size=4,
    )
    # chunk 0 -> rows [0:1]; chunk 1 (padding) -> empty, skipped.
    assert calls == [(1, 0)]


def test_mixed_empty_non_empty_chunks_preserve_offsets(monkeypatch):
    """Mixed empty/non-empty chunks: empty chunks are skipped and non-empty
    chunks are invoked on the correct contiguous row offsets."""
    # request query lens [2, 3, 0, 0, 0, 0, 4] -> cumulative boundaries below.
    calls = _run_prefill(
        monkeypatch,
        query_start_loc_cpu=[0, 2, 5, 5, 5, 5, 5, 9],
        num_prefills=7,
        num_prefill_tokens=9,
        prefill_chunk_size=2,
    )
    # chunk 0 -> rows [0:5]; chunk 1+2 (all-empty) skipped; chunk 3 -> rows [5:9].
    assert calls == [(5, 0), (4, 5)]


def test_all_empty_prefill_skips_sparse_kernel(monkeypatch):
    """When every prefill chunk is an empty graph-padding range, the sparse
    kernel is never invoked."""
    calls = _run_prefill(
        monkeypatch,
        query_start_loc_cpu=[0, 0, 0, 0, 0],
        num_prefills=4,
        num_prefill_tokens=0,
        prefill_chunk_size=2,
    )
    assert calls == []


def test_multiple_empty_padding_chunks_are_safe(monkeypatch):
    """Several consecutive empty graph-padding chunks are all skipped safely."""
    # 12 requests, chunk size 3 -> 4 chunks; requests 3..8 are zero-length.
    # query lens [2, 3, 2, 0, 0, 0, 0, 0, 0, 3, 2, 3] -> cumulative boundaries:
    # [0, 2, 5, 7, 7, 7, 7, 7, 7, 7, 10, 12, 15]
    calls = _run_prefill(
        monkeypatch,
        query_start_loc_cpu=[0, 2, 5, 7, 7, 7, 7, 7, 7, 7, 10, 12, 15],
        num_prefills=12,
        num_prefill_tokens=15,
        prefill_chunk_size=3,
    )
    # chunks: [0:3] rows[0:7]; [3:6] empty; [6:9] empty; [9:12] rows[7:15].
    assert calls == [(7, 0), (8, 7)]
