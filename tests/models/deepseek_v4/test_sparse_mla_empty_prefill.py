# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for upstream #49059 / commit 34020ad: empty SM120
sparse-MLA prefill ranges.

FULL_AND_PIECEWISE CUDA-graph padding (or a CPU-KV-restore step that schedules
a prefill-phase request with 0 new tokens) can produce a prefill chunk with
query_start == query_end. An empty query range must never reach FlashInfer.
"""
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)


class _FakeSparseMLA:
    """Minimal stand-in for the sparse-MLA module to exercise forward_mha's
    zero-query filtering without a CUDA backend."""

    def __init__(self):
        self.masked_mha_available = True
        self._sparse_mla_force_dense_mha = False
        self._sparse_mla_force_masked_mha = False
        self.topk_indices_buffer = torch.zeros((3, 1), dtype=torch.int32)
        self.kv_lora_rank = 4
        self.v_head_dim = 4
        self.num_heads = 1
        self._run_masked_mha_calls = []

    def _slice_topk_per_req(self, topk_all, q_lens):
        out = []
        offset = 0
        for q_len in q_lens:
            out.append(topk_all[offset : offset + q_len])
            offset += q_len
        return out

    def _project_kv(self, kv_c_normed, k_pe):
        n = kv_c_normed.shape[0]
        return (
            torch.zeros((n, self.kv_lora_rank), dtype=torch.float32),
            torch.zeros((n, 1), dtype=torch.float32),
        )

    def _run_masked_mha(self, **kwargs):
        self._run_masked_mha_calls.append(kwargs)
        q = kwargs["q"]
        return torch.zeros_like(q)

    # Delegate to the production method with this object as self.
    forward_mha = SparseMLACommonImpl.forward_mha  # type: ignore[assignment]


def _metadata(query_lens, cu_offsets, max_qlen):
    return SimpleNamespace(
        query_lens_cpu=torch.tensor(query_lens, dtype=torch.int64),
        query_start_loc=torch.tensor(cu_offsets, dtype=torch.int64),
        max_query_len=max_qlen,
        chunked_context=None,
        topk_mask_workspace=None,
        block_table=torch.empty(0, dtype=torch.int32),
    )


def test_all_empty_prefill_skips_backend():
    """When every prefill query is zero-length, the sparse-MLA backend is not
    invoked."""
    attn = _FakeSparseMLA()
    metadata = _metadata([0, 0], [0, 0, 0], 0)
    q = torch.zeros((0, 16, 8), dtype=torch.float32)
    attn.forward_mha(
        q=q,
        kv_c_normed=torch.zeros((0, 4), dtype=torch.float32),
        k_pe=torch.zeros((0, 1), dtype=torch.float32),
        kv_c_and_k_pe_cache=None,
        attn_metadata=SimpleNamespace(
            prefill=metadata,
            num_decode_tokens=0,
            prefill_max_seq_len=8,
            topk_tokens=4,
        ),
        k_scale=torch.ones(1),
        output=torch.zeros((0, 16, 8), dtype=torch.float32),
    )
    assert attn._run_masked_mha_calls == [], "empty query range reached FlashInfer"


def test_mixed_batch_drops_zero_query_segment():
    """A batch with one empty and one non-empty query drops the empty segment:
    q_lens becomes [2] and cu_seqlens_q has no zero-length boundary."""
    attn = _FakeSparseMLA()
    metadata = _metadata([0, 2], [0, 0, 2], 2)
    q = torch.zeros((2, 16, 8), dtype=torch.float32)
    attn.forward_mha(
        q=q,
        kv_c_normed=torch.zeros((2, 4), dtype=torch.float32),
        k_pe=torch.zeros((2, 1), dtype=torch.float32),
        kv_c_and_k_pe_cache=None,
        attn_metadata=SimpleNamespace(
            prefill=metadata,
            num_decode_tokens=0,
            prefill_max_seq_len=8,
            topk_tokens=4,
        ),
        k_scale=torch.ones(1),
        output=torch.zeros((2, 16, 8), dtype=torch.float32),
    )
    assert len(attn._run_masked_mha_calls) == 1
    kwargs = attn._run_masked_mha_calls[0]
    assert kwargs["q_lens"] == [2]
    assert kwargs["cu_seqlens_q"].tolist() == [0, 2]
    assert kwargs["cu_seqlens_k"].tolist() == [0, 2]
    assert kwargs["max_seqlen_q"] == 2
    assert kwargs["max_seqlen_k"] == 2


def test_no_zero_query_is_unchanged_when_no_empty_present():
    """Non-empty prefill behaves exactly as before: q_lens and cu_seqlens_q are
    untouched."""
    attn = _FakeSparseMLA()
    metadata = _metadata([2, 3], [0, 2, 5], 3)
    q = torch.zeros((5, 16, 8), dtype=torch.float32)
    attn.forward_mha(
        q=q,
        kv_c_normed=torch.zeros((5, 4), dtype=torch.float32),
        k_pe=torch.zeros((5, 1), dtype=torch.float32),
        kv_c_and_k_pe_cache=None,
        attn_metadata=SimpleNamespace(
            prefill=metadata,
            num_decode_tokens=0,
            prefill_max_seq_len=5,
            topk_tokens=4,
        ),
        k_scale=torch.ones(1),
        output=torch.zeros((5, 16, 8), dtype=torch.float32),
    )
    assert len(attn._run_masked_mha_calls) == 1
    kwargs = attn._run_masked_mha_calls[0]
    assert kwargs["q_lens"] == [2, 3]
    assert kwargs["cu_seqlens_q"].tolist() == [0, 2, 5]
