# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative-width (num_speculative_tokens) regression tests.

Covers the fix that removed the artificial
``num_speculative_tokens >= dspark_block_size`` (5) hard failure:

* K3 must be accepted by the DSpark validation path (warning, not error);
* K5/K6 remain accepted;
* the real Markov block sampler runs request-major blocks of EXACTLY K
  draft positions for K in {3, 5, 6} (shape-level propagation proof);
* genuine invalid inputs (non-[batch, block, vocab] logits) still fail.

The sampler module (``vllm/v1/spec_decode/dspark_sampling.py``) is loaded
by file path with its heavy leaf dependencies stubbed; those stubs are not
exercised on the greedy path under test.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLING_PATH = (
    REPO_ROOT / "vllm" / "v1" / "spec_decode" / "dspark_sampling.py"
)
SPECULATIVE_PATH = REPO_ROOT / "vllm" / "config" / "speculative.py"


def _load_dspark_sampling():
    """Load the real sampler with stubbed heavy leaf modules."""
    stubs = {
        "vllm": types.ModuleType("vllm"),
        "vllm.logger": types.ModuleType("vllm.logger"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.sample": types.ModuleType("vllm.v1.sample"),
        "vllm.v1.sample.metadata": types.ModuleType("vllm.v1.sample.metadata"),
        "vllm.v1.sample.ops": types.ModuleType("vllm.v1.sample.ops"),
        "vllm.v1.sample.ops.topk_topp_sampler": types.ModuleType(
            "vllm.v1.sample.ops.topk_topp_sampler"
        ),
        "vllm.v1.sample.sampler": types.ModuleType("vllm.v1.sample.sampler"),
    }

    class _Logger:
        def warning_once(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    stubs["vllm.logger"].init_logger = lambda name: _Logger()

    class SamplingMetadata:  # type placeholder only
        pass

    stubs["vllm.v1.sample.metadata"].SamplingMetadata = SamplingMetadata
    ops = stubs["vllm.v1.sample.ops.topk_topp_sampler"]
    ops.apply_top_k_top_p = None
    ops.empty_exponential_noise_like = None
    ops.sample_with_exponential_noise = None
    stubs["vllm.v1.sample.sampler"]._SAMPLING_EPS = 1e-5

    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "dsv4_dspark_sampling_under_test", SAMPLING_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return mod


SAMPLER = _load_dspark_sampling()


def _greedy_metadata():
    return types.SimpleNamespace(all_greedy=True)


def _identity_bias(logits, prev_token_ids, step_idx):
    return logits


@pytest.mark.parametrize("width", [3, 5, 6])
def test_markov_sampler_produces_exactly_k_positions(width):
    """The real sampler emits exactly K draft positions per request for
    K = num_speculative_tokens, on request-major [batch, K, vocab] logits."""
    batch, vocab = 2, 11
    logits = torch.full((batch, width, vocab), -100.0)
    # Distinct argmax per (batch, step): token = batch * 10 + step.
    for b in range(batch):
        for s in range(width):
            logits[b, s, b * 5 + s] = 100.0

    tokens, probs = SAMPLER.sample_dspark_markov_block(
        logits,
        torch.tensor([7, 8]),  # first_prev_token_ids
        _identity_bias,
        _greedy_metadata(),
        return_probs=False,
    )
    assert probs is None
    assert tokens.shape == (batch, width)
    for b in range(batch):
        for s in range(width):
            assert tokens[b, s].item() == b * 5 + s


@pytest.mark.parametrize("width", [3, 5, 6])
def test_markov_chain_threads_prev_tokens_and_steps(width):
    """The bias fn must be called once per position, in order, with the
    previously sampled token (Markov chaining), for any width K."""
    calls = []

    def bias(logits, prev_token_ids, step_idx):
        calls.append((step_idx, prev_token_ids.clone()))
        return logits

    batch, vocab = 1, 9
    logits = torch.full((batch, width, vocab), -100.0)
    # step s always picks token s+1
    for s in range(width):
        logits[0, s, s + 1] = 100.0

    tokens, _ = SAMPLER.sample_dspark_markov_block(
        logits,
        torch.tensor([4]),
        bias,
        _greedy_metadata(),
        return_probs=False,
    )
    assert tokens.shape == (1, width)
    assert [c[0] for c in calls] == list(range(width))
    # step 0 sees the seed token; step s sees the token sampled at step s-1.
    assert calls[0][1].item() == 4
    for s in range(1, width):
        assert calls[s][1].item() == s  # token sampled at step s-1
    assert tokens[0].tolist() == list(range(1, width + 1))


def test_markov_sampler_rejects_non_block_logits():
    """Genuine shape validation still fires."""
    with pytest.raises(ValueError, match="batch, block, vocab"):
        SAMPLER.sample_dspark_markov_block(
            torch.zeros(2, 5),  # missing block dim
            torch.tensor([1, 2]),
            _identity_bias,
            _greedy_metadata(),
            return_probs=False,
        )


def test_dspark_width_validation_no_hard_minimum():
    """The config validation must no longer hard-fail
    num_speculative_tokens < dspark_block_size, must warn for the below-block
    case, and must not claim K>5 positions 'can never be accepted'."""
    src = SPECULATIVE_PATH.read_text(encoding="utf-8")
    assert "DSpark requires num_speculative_tokens >=" not in src
    assert "produce incorrect output" not in src
    assert "can never be accepted" not in src
    assert "is below the " in src and "checkpoint DSpark block size" in src
