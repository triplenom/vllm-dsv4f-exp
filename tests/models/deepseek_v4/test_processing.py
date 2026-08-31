# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek V4 Vision-Exp multimodal processor
(``vllm/models/deepseek_v4/processing.py``).

Covers prompt encoding / placeholder layout semantics:
  * text-only prompts pass through unchanged;
  * an image placeholder expands into the synthetic image block;
  * text/image/text ordering is preserved;
  * multiple images expand in order with correct offsets;
  * the mm placeholder range length equals the trimmed block length;
  * text-only (0731-style) configs report no mm limits and reject images.

Runs without CUDA: the ``vllm.models.deepseek_v4`` package __init__ (which
pulls in Triton/CUDA modules) is stubbed out; the processing module's own
imports stay within the CPU-safe vLLM subset.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from ._util import import_dsv4_submodule, load_impl

_impl = load_impl("image_processing")
IMAGE_PLACEHOLDER = _impl.IMAGE_PLACEHOLDER
VOCAB_SIZE = 129280
PLACEHOLDER_ID = 128000


processing = pytest.importorskip(
    "vllm.multimodal.processing", reason="requires the vLLM multimodal framework"
)
processing = import_dsv4_submodule("processing")

from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        raise AssertionError("tests pass prompts as token id lists")

    def convert_tokens_to_ids(self, token):
        assert token == IMAGE_PLACEHOLDER
        return PLACEHOLDER_ID


def _make_processor(vision: bool = True):
    from transformers import PretrainedConfig

    hf_config = PretrainedConfig()
    hf_config.model_type = "deepseek_v4"
    hf_config.vocab_size = VOCAB_SIZE
    if vision:
        hf_config.vision_n_layers = 32
        hf_config.vision_dim = 1024
        hf_config.vision_n_heads = 16
        hf_config.vision_inter_dim = 2816
        hf_config.vision_patch_size = 14
        hf_config.vision_rope_theta = 10000.0
        hf_config.vision_downsample_ratio = 3
        hf_config.vision_max_n_token = 384
        hf_config.vision_min_pixels = 147456
        hf_config.vision_max_wh_ratio = 8

    mm_config = SimpleNamespace(mm_hasher_algorithm="sha256", enable_mm_embeds=False)

    class _FakeModelConfig:
        def __init__(self, hf_config, mm_config):
            self.hf_config = hf_config
            self.model = "deepseek-v4-test"
            self.multimodal_config = mm_config

        def get_multimodal_config(self):
            return self.multimodal_config

    model_config = _FakeModelConfig(hf_config, mm_config)
    from vllm.multimodal.processing.context import InputProcessingContext

    ctx = InputProcessingContext(model_config=model_config, tokenizer=_FakeTokenizer())
    info = processing.DeepseekV4VisionProcessingInfo(ctx)
    dummy = processing.DeepseekV4VisionDummyInputsBuilder(info)
    return processing.DeepseekV4VisionMultiModalProcessor(info, dummy)


def _make_image(width=64, height=48, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _mm_items(*images: Image.Image) -> MultiModalDataItems:
    return MultiModalDataItems({"image": ImageProcessorItems(list(images))})


def _apply(proc, prompt_ids, images=()):
    inputs = ProcessorInputs(prompt=list(prompt_ids), mm_data_items=_mm_items(*images))
    return inputs, proc.apply(inputs, TimingContext(enabled=False))


def test_text_only_prompt_passthrough():
    proc = _make_processor()
    _, out = _apply(proc, [5, 6, 7])
    assert out["prompt_token_ids"] == [5, 6, 7]
    assert out["mm_placeholders"] == {}


def test_single_image_expansion():
    proc = _make_processor()
    image = _make_image()
    _, out = _apply(proc, [10, PLACEHOLDER_ID, 20], images=[image])

    # Reference block for this image at start position 1.
    item = _impl.process_image(
        image,
        patch_size=14,
        downsample_ratio=3,
        max_n_token=384,
        min_pixels=147456,
        max_wh_ratio=8,
    )
    block = _impl.build_image_block(item.n_llm_h, item.n_llm_w, 1, VOCAB_SIZE)

    assert out["prompt_token_ids"] == [10] + block.token_ids + [20]
    (ph,) = out["mm_placeholders"]["image"]
    assert ph.offset == 1 + block.compress_pad
    assert ph.length == len(block.types) - block.compress_pad
    # The placeholder range covers exactly [IMAGE_START, IMAGE_END].
    trimmed = block.types[block.compress_pad :]
    assert trimmed[0] == _impl.IMAGE_START and trimmed[-1] == _impl.IMAGE_END
    assert len(trimmed) == ph.length


def test_text_image_text_ordering_and_multiple_images():
    proc = _make_processor()
    img1 = _make_image(64, 48, seed=1)
    img2 = _make_image(48, 64, seed=2)
    _, out = _apply(proc, [7, PLACEHOLDER_ID, 8, 9, PLACEHOLDER_ID, 11], images=[img1, img2])

    item1 = _impl.process_image(
        img1, patch_size=14, downsample_ratio=3, max_n_token=384,
        min_pixels=147456, max_wh_ratio=8,
    )
    block1 = _impl.build_image_block(item1.n_llm_h, item1.n_llm_w, 1, VOCAB_SIZE)
    second_start = 1 + len(block1.types) + 2
    item2 = _impl.process_image(
        img2, patch_size=14, downsample_ratio=3, max_n_token=384,
        min_pixels=147456, max_wh_ratio=8,
    )
    block2 = _impl.build_image_block(item2.n_llm_h, item2.n_llm_w, second_start, VOCAB_SIZE)

    expected = [7] + block1.token_ids + [8, 9] + block2.token_ids + [11]
    assert out["prompt_token_ids"] == expected

    phs = out["mm_placeholders"]["image"]
    assert len(phs) == 2
    assert phs[0].offset == 1 + block1.compress_pad
    assert phs[1].offset == second_start + block2.compress_pad
    # Blocks must not overlap.
    end0 = phs[0].offset + phs[0].length
    assert end0 <= phs[1].offset
    # Synthetic ids are strictly out of vocabulary and typed in order.
    for ph, block in zip(phs, (block1, block2)):
        trimmed = block.types[block.compress_pad :]
        span = out["prompt_token_ids"][ph.offset : ph.offset + ph.length]
        assert span == [VOCAB_SIZE + t for t in trimmed]


def test_mm_kwargs_fields_present():
    proc = _make_processor()
    image = _make_image()
    _, out = _apply(proc, [1, PLACEHOLDER_ID], images=[image])
    items = out["mm_kwargs"]["image"]
    data = items[0].get_data()
    patches = data[processing.K_PATCHES]
    types = data[processing.K_TYPES]
    perm = data[processing.K_PERM]
    # types/perm lengths are consistent with the placeholder.
    (ph,) = out["mm_placeholders"]["image"]
    assert len(types) == ph.length
    assert len(perm) == int((types == _impl.IMAGE).sum())
    assert patches.dim() == 4 and patches.shape[1] == 3


def test_hashes_present_and_distinct_per_content():
    proc = _make_processor()
    _, out1 = _apply(proc, [1, PLACEHOLDER_ID], images=[_make_image(seed=1)])
    _, out2 = _apply(proc, [1, PLACEHOLDER_ID], images=[_make_image(seed=2)])
    h1 = out1["mm_hashes"]["image"]
    h2 = out2["mm_hashes"]["image"]
    assert len(h1) == 1 and len(h2) == 1
    assert h1[0] != h2[0]


def test_text_only_config_rejects_images():
    proc = _make_processor(vision=False)
    info = proc.info
    assert info.get_supported_mm_limits() == {}
    with pytest.raises(ValueError, match="Vision-Exp"):
        _apply(proc, [1, PLACEHOLDER_ID], images=[_make_image()])


def test_placeholder_image_count_mismatch_rejected():
    proc = _make_processor()
    with pytest.raises(ValueError):
        _apply(proc, [1, PLACEHOLDER_ID, PLACEHOLDER_ID], images=[_make_image()])
    with pytest.raises(ValueError):
        _apply(proc, [1, PLACEHOLDER_ID], images=[_make_image(seed=1), _make_image(seed=2)])
