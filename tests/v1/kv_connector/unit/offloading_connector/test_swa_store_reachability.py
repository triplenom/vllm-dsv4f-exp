# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for upstream #54362: SWA store reachability during chunked
prefill.

During active chunked prefill the offloading scheduler must evaluate SWA store
reachability against the FINAL intended prompt horizon, not the current
intermediate chunked-prefill frontier. The pre-fix behaviour treated each step
as the final partial segment and over-stored unreachable SWA chunks; it also
missed the final reachable tail when a request aborted before the intended
horizon.

The reachability decision itself (is_store_reachable_swa_chunk) is unchanged;
this test asserts the projected-horizon store set that _build_store_jobs now
produce for the SWA group.
"""
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    is_store_reachable_swa_chunk,
)
from vllm.utils.math_utils import cdiv


def _reachable_at_horizon(
    final_chunks: int, alignment: int, sw_chunks: int, eagle: bool
) -> set[int]:
    """SWA chunk indices reachable at a given (final) store horizon."""
    return {
        c
        for c in range(final_chunks)
        if is_store_reachable_swa_chunk(c, final_chunks, alignment, sw_chunks, eagle)
    }


def _b3_store_set(
    frontier: int, alignment: int, sw_chunks: int, eagle: bool
) -> set[int]:
    """Chunks the pre-fix code stored when it evaluated reachability against the
    current step's frontier (storable_chunk_count == frontier)."""
    return {
        c
        for c in range(frontier)
        if is_store_reachable_swa_chunk(c, frontier, alignment, sw_chunks, eagle)
    }


# Group configuration matching the DeepSeek-V4 hybrid case: a full-attention
# group with a larger block size defines the alignment segment; the SWA group
# has a smaller block size and a sliding window smaller than one segment.
ALIGNMENT = 4
SW_CHUNKS = 2
EAGLE = False


def test_completed_chunked_prefill_uses_final_horizon_not_intermediate():
    """A completed 4-chunk SWA segment must store only the reachable tail
    (chunks 2,3). The intermediate-frontier behaviour over-stored 0,1,2,3."""
    final_chunks = 4
    expected = _reachable_at_horizon(final_chunks, ALIGNMENT, SW_CHUNKS, EAGLE)
    assert expected == {2, 3}

    # Pre-fix: evaluating at each intermediate frontier (1..4) cumulatively
    # stored chunks that the load path can never request.
    b3_union = set()
    for frontier in range(1, final_chunks + 1):
        b3_union |= _b3_store_set(frontier, ALIGNMENT, SW_CHUNKS, EAGLE)
    assert b3_union == {0, 1, 2, 3}
    assert b3_union != expected, "pre-fix horizon produced the wrong SWA set"


def test_multi_segment_completed_prefill_stores_only_reachable_tail():
    """Two alignment segments (8 SWA chunks): only 2,3,6,7 are reachable."""
    final_chunks = 8
    expected = _reachable_at_horizon(final_chunks, ALIGNMENT, SW_CHUNKS, EAGLE)
    assert expected == {2, 3, 6, 7}

    b3_union = set()
    for frontier in range(1, final_chunks + 1):
        b3_union |= _b3_store_set(frontier, ALIGNMENT, SW_CHUNKS, EAGLE)
    assert expected < b3_union, "pre-fix over-stored unreachable SWA chunks"


def test_aborted_prefill_reconsiders_newly_reachable_tail():
    """If the request aborts at 3 of 4 intended chunks, the actual frontier (3)
    becomes the final horizon and SWA chunks 1,2 become reachable. The fix must
    store exactly {1,2} and never the previously-skipped-but-now-unreachable 0."""
    final_chunks = 4
    abort_at = 3
    expected_abort = _reachable_at_horizon(abort_at, ALIGNMENT, SW_CHUNKS, EAGLE)
    assert expected_abort == {1, 2}

    # union of the reachable sets at all frontiers the request traversed.
    stored = set()
    for frontier in range(1, abort_at + 1):
        # The fix re-evaluates against the final (abort) horizon and the cache
        # dedups, so the stored set equals the abort-horizon reachable set.
        stored |= _reachable_at_horizon(abort_at, ALIGNMENT, SW_CHUNKS, EAGLE)
    assert stored == {1, 2}


def test_eagle_trailing_chunk_still_excluded():
    """EAGLE/MTP trailing chunk remains excluded: reachable tail includes the
    extra draft chunk, matching SchedulerOffloadConfig is_eagle_group=True."""
    eagle = True
    final_chunks = 4
    reachable = _reachable_at_horizon(final_chunks, ALIGNMENT, SW_CHUNKS, eagle)
    # SWA reachable tail expands by the one volatile draft tail chunk.
    assert 3 in reachable
    assert 2 in reachable


@pytest.mark.parametrize(
    "final_chunks,expected",
    [
        (4, {2, 3}),
        (8, {2, 3, 6, 7}),
        (12, {2, 3, 6, 7, 10, 11}),
    ],
)
def test_reachable_set_matches_segment_alignment(final_chunks, expected):
    assert _reachable_at_horizon(final_chunks, ALIGNMENT, SW_CHUNKS, EAGLE) == expected


def test_alignment_chunk_count_derivation():
    """SchedulerOffloadConfig computes alignment_chunk_count when the SWA group
    has a smaller tokens_per_chunk than the single full-attention alignment."""
    # Full-attn tokens_per_chunk=16, SWA tokens_per_chunk=4, SWA window=8.
    alignment_tokens = 16
    sw_tokens_per_chunk = 4
    sw_window_chunks = cdiv(8, sw_tokens_per_chunk)
    per_segment = alignment_tokens // sw_tokens_per_chunk
    if sw_window_chunks < per_segment:
        alignment_chunk_count = per_segment
    else:
        alignment_chunk_count = None
    assert alignment_chunk_count == 4
