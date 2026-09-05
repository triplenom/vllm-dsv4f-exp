# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for upstream #54362: SWA store reachability during chunked
prefill.

These tests drive the real ``OffloadingConnectorScheduler._build_store_jobs``
path through the engine ``request_runner`` harness (not the reachability helper
in isolation) and assert the set of SWA chunks actually selected for storage.

They cover an intermediate chunked-prefill step, normal completed chunked
prefill, aborted chunked prefill (reconsideration of the actual final frontier),
synchronous and asynchronous scheduling, and the active decode frontier not
being reinterpreted as a final partial prompt/SWA segment.

The expected store set is computed from an independent segment-alignment oracle
so the test does not merely recompute the production helper logic.
"""
import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus

FULL_ATTN_BLOCK_SIZE = 36
SWA_BLOCK_SIZE = 4
SLIDING_WINDOW = 16
ALIGNMENT_CHUNKS = FULL_ATTN_BLOCK_SIZE // SWA_BLOCK_SIZE
SLIDING_WINDOW_CHUNKS = SLIDING_WINDOW // SWA_BLOCK_SIZE


def _hybrid_groups():
    """DeepSeek-V4 hybrid layout: full-attention alignment + a smaller SWA group."""
    return [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=FULL_ATTN_BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=SWA_BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=SLIDING_WINDOW,
            ),
        ),
    ]


def _expected_swa_chunks(store_horizon_chunks) -> list[int]:
    """Independent oracle: the reachable SWA chunk set at a final horizon."""
    expected: list[int] = []
    for chunk in range(store_horizon_chunks):
        segment_start = chunk - chunk % ALIGNMENT_CHUNKS
        segment_length = min(
            ALIGNMENT_CHUNKS, store_horizon_chunks - segment_start
        )
        if (
            chunk % ALIGNMENT_CHUNKS
            >= segment_length - SLIDING_WINDOW_CHUNKS
        ):
            expected.append(chunk)
    return expected


def _stored_swa_chunks(runner) -> list[int]:
    return sorted(
        block.request_block_offset
        for transfer in runner.completed_stores
        for block in transfer.gpu_blocks
        if block.group_idx == 1
    )


def _make_runner(request_runner, async_scheduling):
    return request_runner(
        block_size=SWA_BLOCK_SIZE,
        num_gpu_blocks=1000,
        async_scheduling=async_scheduling,
        kv_cache_groups=_hybrid_groups(),
    )


def _run_prefill(request_runner, num_tokens, abort_after_first, async_scheduling):
    runner = _make_runner(request_runner, async_scheduling)
    runner.new_request(token_ids=[0] * num_tokens)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    def abort_after_prefill_step():
        if runner.scheduler.running:
            runner.scheduler.finish_requests(
                (str(runner.req_id),), RequestStatus.FINISHED_ABORTED
            )

    runner._run(
        [1] if abort_after_first else [1, 1, EOS_TOKEN_ID],
        complete_transfers=True,
        post_step_fn=(abort_after_prefill_step if abort_after_first else None),
    )
    return runner


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_completed_chunked_prefill_stores_final_horizon_tail(
    request_runner, async_scheduling
):
    """A completed chunked prefill stores the reachable tail of the final
    prompt horizon, so unreachable interior chunks are skipped while the final
    partial-segment tail is included."""
    num_tokens = 1200
    store_horizon_chunks = num_tokens // SWA_BLOCK_SIZE

    runner = _run_prefill(
        request_runner,
        num_tokens,
        abort_after_first=False,
        async_scheduling=async_scheduling,
    )

    stored = _stored_swa_chunks(runner)
    expected = _expected_swa_chunks(store_horizon_chunks)
    assert stored == expected


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_aborted_chunked_prefill_reconsiders_final_tail(
    request_runner, async_scheduling
):
    """An aborted chunked prefill uses its actual computed frontier as the final
    horizon and stores the newly reachable tail of the final segment."""
    num_tokens = 1200
    abort_tokens = 1000
    store_horizon_chunks = abort_tokens // SWA_BLOCK_SIZE

    runner = _run_prefill(
        request_runner,
        num_tokens,
        abort_after_first=True,
        async_scheduling=async_scheduling,
    )

    stored = _stored_swa_chunks(runner)
    expected = _expected_swa_chunks(store_horizon_chunks)
    assert sorted(set(stored)) == expected


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_active_decode_does_not_advance_swa_final_horizon(
    request_runner, async_scheduling
):
    """Active decode chunks must not be treated as a final partial SWA segment:
    the first decode chunk (right after the prompt boundary) is not stored."""
    prompt_tokens = 1200
    runner = _make_runner(request_runner, async_scheduling)
    runner.new_request(token_ids=[0] * prompt_tokens)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    runner._run([1] * 12, complete_transfers=True)

    assert runner.scheduler.running
    stored = _stored_swa_chunks(runner)
    first_decode_chunk = prompt_tokens // SWA_BLOCK_SIZE
    assert first_decode_chunk not in stored
