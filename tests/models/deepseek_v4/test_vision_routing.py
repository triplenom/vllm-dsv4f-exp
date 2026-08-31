# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the Vision-Exp MoE routing port
(``vllm/models/deepseek_v4/vision_routing.py``) against the reference
``Gate.forward`` (vendored excerpt in ./reference/model_excerpt.py).

Covers:
  * hash-routing layers: text positions use tid2eid, image positions use
    score-based selection with bias_vl and never touch tid2eid;
  * score-routing layers: image positions select with bias_vl instead of the
    text bias;
  * routing weights always come from the un-biased scores;
  * text-only batches produce exactly the text routing;
  * DeepSeek V4 defaults: scoring_func=sqrtsoftplus, norm_topk_prob=True,
    routed_scaling_factor applied.
"""

import torch

from ._util import load_impl, load_reference

routing = load_impl("vision_routing")
ref_model = load_reference("model_excerpt")

VOCAB_SIZE = 1000
N_EXPERTS = 32
TOPK = 6
ROUTE_SCALE = 1.5


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # 3 text tokens, then an image block (types 0..4), then 2 text tokens.
    input_ids = torch.tensor(
        [17, 42, 3, VOCAB_SIZE + 0, VOCAB_SIZE + 2, VOCAB_SIZE + 2, VOCAB_SIZE + 4, 9, 800]
    )
    logits = torch.randn(input_ids.shape[0], N_EXPERTS, generator=g)
    bias = torch.randn(N_EXPERTS, generator=g)
    bias_vl = torch.randn(N_EXPERTS, generator=g)
    tid2eid = torch.randint(0, N_EXPERTS, (VOCAB_SIZE, TOPK), generator=g)
    return input_ids, logits, bias, bias_vl, tid2eid


def _impl_topk(logits, input_ids, bias, bias_vl, tid2eid):
    return routing.dsv4_vision_aware_topk(
        logits,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=bias,
        vision_correction_bias=bias_vl,
        tid2eid=tid2eid,
        input_ids=input_ids,
        vocab_size=VOCAB_SIZE,
        topk=TOPK,
        renormalize=True,
        routed_scaling_factor=ROUTE_SCALE,
    )


def _ref_topk(logits, input_ids, bias, bias_vl, tid2eid):
    """Reference Gate.forward with logits fed through an identity matmul."""
    return ref_model.gate_forward_reference(
        logits,
        input_ids,
        weight=torch.eye(N_EXPERTS),
        score_func="sqrtsoftplus",
        route_scale=ROUTE_SCALE,
        topk=TOPK,
        vocab_size=VOCAB_SIZE,
        tid2eid=tid2eid,
        bias=bias,
        bias_vl=bias_vl,
    )


def test_hash_layer_matches_reference():
    input_ids, logits, bias, bias_vl, tid2eid = _inputs()
    ref_weights, ref_indices = _ref_topk(logits, input_ids, bias, bias_vl, tid2eid)
    weights, indices = _impl_topk(logits, input_ids, bias, bias_vl, tid2eid)
    assert indices.tolist() == ref_indices.tolist()
    torch.testing.assert_close(weights, ref_weights.float())


def test_score_layer_matches_reference():
    input_ids, logits, bias, bias_vl, _ = _inputs()
    ref_weights, ref_indices = _ref_topk(logits, input_ids, bias, bias_vl, None)
    weights, indices = _impl_topk(logits, input_ids, bias, bias_vl, None)
    assert indices.tolist() == ref_indices.tolist()
    torch.testing.assert_close(weights, ref_weights.float())


def test_text_only_hash_layer_unchanged():
    """A pure-text batch at a hash layer must use tid2eid for every token."""
    input_ids, logits, bias, bias_vl, tid2eid = _inputs()
    text_ids = input_ids[input_ids < VOCAB_SIZE]
    _, indices = _impl_topk(logits[: len(text_ids)], text_ids, bias, bias_vl, tid2eid)
    assert indices.tolist() == tid2eid[text_ids].tolist()


def test_image_positions_do_not_use_tid2eid():
    """Hash layers: image positions must come from score selection."""
    input_ids, logits, bias, bias_vl, tid2eid = _inputs()
    image_ids = input_ids[input_ids >= VOCAB_SIZE]
    logits_img = logits[input_ids >= VOCAB_SIZE]
    _, indices = _impl_topk(logits, input_ids, bias, bias_vl, tid2eid)
    img_indices = indices[input_ids >= VOCAB_SIZE]

    scores = torch.sqrt(torch.nn.functional.softplus(logits_img.float()))
    expected = (scores + bias_vl).topk(TOPK, dim=-1)[1]
    assert img_indices.tolist() == expected.tolist()

    # And they must NOT equal the tid2eid[0] row that a naive port would use.
    tid_row = tid2eid[0].expand(len(image_ids), TOPK)
    assert not torch.equal(img_indices, tid_row)


def test_weights_come_from_unbiased_scores():
    """Selection uses bias (or bias_vl); weights gather the raw scores."""
    input_ids, logits, bias, bias_vl, _ = _inputs()
    weights, indices = _impl_topk(logits, input_ids, bias, bias_vl, None)
    scores = torch.sqrt(torch.nn.functional.softplus(logits.float()))
    raw = scores.gather(1, indices.long())
    raw = raw / raw.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(weights, raw.float() * ROUTE_SCALE)


def test_no_vision_bias_rejected():
    input_ids, logits, bias, _, tid2eid = _inputs()
    try:
        routing.dsv4_vision_aware_topk(
            logits,
            scoring_func="sqrtsoftplus",
            e_score_correction_bias=bias,
            vision_correction_bias=None,
            tid2eid=tid2eid,
            input_ids=input_ids,
            vocab_size=VOCAB_SIZE,
            topk=TOPK,
            renormalize=True,
            routed_scaling_factor=ROUTE_SCALE,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError when the vision bias is missing")


def test_image_mask_helper():
    ids = torch.tensor([1, 2, VOCAB_SIZE, VOCAB_SIZE + 4, 7])
    mask = routing.image_token_mask(ids, VOCAB_SIZE)
    assert mask.tolist() == [False, False, True, True, False]
