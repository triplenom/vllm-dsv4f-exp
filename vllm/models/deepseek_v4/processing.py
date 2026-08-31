# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 Flash Vision-Exp multimodal processing.

Wires DeepSeek's official image prompt semantics into the vLLM v1
multimodal framework:

- OpenAI interleaved text/image_url content blocks are represented by the
  single ``<｜deepseek_image｜>`` placeholder token at the chat-template
  level (see ``get_placeholder_str`` on the model class).
- Stateless processing here then expands every placeholder token into the
  full DeepSeek image block (synthetic ids ``vocab_size + types`` with
  compressor-aligned leading pads), mirroring the reference
  ``prepare_vl_inputs``.
- Per-image fields (ViT patches, grid dims, semantic ``types`` and the
  aligner permutation ``perm``) travel to the model as mm kwargs so the
  model's ``embed_multimodal`` produces the LM-hidden embeddings for the
  placeholder merge.

The vLLM placeholder range of an image covers ``[IMAGE_START, IMAGE_END]``
(the leading compressor-alignment pads are excluded): that part of the
block is fully determined by the image content, so encoder outputs cached
by content hash stay valid when the same image appears at different prompt
offsets. The leading IMAGE_PAD positions still carry their synthetic ids in
``prompt_token_ids`` (so routing/visibility see them); the model fills them
with the ``image_pad`` embedding directly in ``embed_input_ids``.

Processing is done on the whole prompt (never per-item cached): the
leading IMAGE_PAD count depends on each image's absolute offset in the
expanded prompt, so per-item prompt-update caching would be unsound. Item
content hashing (encoder cache / prefix-cache identity) still uses the
standard framework mm hashing untouched.
"""

from collections.abc import Mapping, Sequence

import torch
from transformers.feature_extraction_utils import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptUpdate,
)
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.parse import ImageSize

from .image_processing import (
    IMAGE_PLACEHOLDER,
    build_image_block,
    process_image,
)

logger = init_logger(__name__)

# mm kwargs keys consumed by DeepseekV4ForCausalLM.embed_multimodal.
K_PATCHES = "dsv4_patches"
K_GRID = "dsv4_grid"  # (num_images, 6): n_vit_h, n_vit_w, n_llm_h, n_llm_w, types_len, perm_len
K_TYPES = "dsv4_types"
K_PERM = "dsv4_perm"

FieldConfigs = dict[str, MultiModalFieldConfig]


def _vision_config_args(config) -> dict:
    """Resolve the vision parameters from the HF config (vision_* keys).

    Returns None when the config is a text-only DeepSeek V4 config (0731),
    in which case multimodal requests are rejected with a clear error.
    """
    if getattr(config, "vision_n_layers", 0) <= 0:
        return None
    return {
        "patch_size": config.vision_patch_size,
        "downsample_ratio": config.vision_downsample_ratio,
        "max_n_token": config.vision_max_n_token,
        "min_pixels": config.vision_min_pixels,
        "max_wh_ratio": config.vision_max_wh_ratio,
    }


class DeepseekV4VisionProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Text-only DeepSeek V4 configs (no vision_* fields) report no
        # modalities, so the framework treats them as pure text models.
        if _vision_config_args(self.ctx.get_hf_config()) is None:
            return {}
        return {"image": None}

    def get_image_placeholder(self) -> str:
        return IMAGE_PLACEHOLDER

    def get_image_placeholder_id(self) -> int:
        tokenizer = self.get_tokenizer()
        placeholder_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if placeholder_id is None or placeholder_id < 0:
            raise ValueError(
                f"Tokenizer is missing the image placeholder {IMAGE_PLACEHOLDER!r}. "
                "Image input requires the DeepSeek-V4-Flash-Vision-Exp tokenizer."
            )
        return int(placeholder_id)

    def get_vision_args(self) -> dict | None:
        return _vision_config_args(self.ctx.get_hf_config())

    def get_image_size_with_most_features(self) -> ImageSize:
        # Safe upper bound for profiling: larger inputs are shrunk by
        # safe_resize to at most vision_max_n_token placeholders anyway; a
        # 14*3*19x19-cell grid places ~19*20+3 tokens (see grid_tokens).
        return ImageSize(width=14 * 3 * 20, height=14 * 3 * 20)


class DeepseekV4VisionDummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV4VisionProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return IMAGE_PLACEHOLDER * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        max_image_size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
            )
        }


class DeepseekV4VisionMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV4VisionProcessingInfo]
):
    # Abstract-method implementations; the whole-prompt ``apply`` below
    # supersedes the generic per-item update machinery (the DeepSeek block
    # layouts are offset-dependent), so these are only exercised through
    # ``_prompt_token_ids`` / profiling fallbacks.
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            K_PATCHES: MultiModalFieldConfig.batched("image"),
            K_GRID: MultiModalFieldConfig.batched("image"),
            K_TYPES: MultiModalFieldConfig.batched("image"),
            K_PERM: MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []

    # -- whole-prompt processing -------------------------------------------

    def _prompt_token_ids(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, list):
            return list(prompt)
        return self.info.get_tokenizer().encode(prompt, add_special_tokens=False)

    def _process_images(self, mm_data_items) -> tuple[list, list[list[int]]]:
        """Run the DeepSeek image preprocessor on all image items in order.

        Returns (processed_images, per_image_fields_lists).
        """
        config = self.info.ctx.get_hf_config()
        args = _vision_config_args(config)
        if args is None:
            raise ValueError(
                "This DeepSeek V4 checkpoint/config does not include the "
                "Vision-Exp fields (vision_n_layers > 0). Image input requires "
                "the DeepSeek-V4-Flash-Vision-Exp model weights and config."
            )

        valid_mm_items = mm_data_items.select(
            {k for k, c in mm_data_items.get_all_counts().items() if c > 0}
        )
        processor_data, _ = self._get_hf_mm_data(valid_mm_items)
        # ProcessorBatchItems.get_processor_data() keys items by the plural
        # modality name ("images").
        images = processor_data.get("images", [])
        if not isinstance(images, list):
            images = [images]
        if not images:
            raise ValueError(
                "The prompt contains image placeholders but no image data was "
                "provided for them."
            )

        processed = [
            process_image(
                image,
                patch_size=args["patch_size"],
                downsample_ratio=args["downsample_ratio"],
                max_n_token=args["max_n_token"],
                min_pixels=args["min_pixels"],
                max_wh_ratio=args["max_wh_ratio"],
            )
            for image in images
        ]
        return processed

    def _encode_images_torch(self, processed) -> list[torch.Tensor]:
        """DeepSeek reference patch conversion: fp32 -> bf16, grid layout as-is."""
        return [
            torch.from_numpy(item.patches).to(torch.bfloat16)
            for item in processed
        ]

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        mm_count = inputs.mm_data_items.get_all_counts().get("image", 0)
        prompt_ids = self._prompt_token_ids(inputs.prompt)

        if mm_count == 0:
            mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
                BatchFeature(),
                self._get_mm_fields_config(
                    BatchFeature(), inputs.hf_processor_mm_kwargs
                ),
            )
            return mm_input(
                prompt_token_ids=prompt_ids,
                mm_kwargs=mm_kwargs,
                mm_hashes={},
                mm_placeholders={},
            )

        with timing_ctx.record("dsv4_apply_hf_processor"):
            config = self.info.ctx.get_hf_config()
            placeholder_id = self.info.get_image_placeholder_id()
            processed = self._process_images(inputs.mm_data_items)

            patches = self._encode_images_torch(processed)
            block_layouts: list = []
            out_ids: list[int] = []
            cursor = 0
            image_iter = iter(processed)
            patches_iter = iter(patches)
            for token in prompt_ids:
                if token != placeholder_id:
                    out_ids.append(token)
                    continue
                item = next(image_iter, None)
                _ = next(patches_iter, None)
                if item is None:
                    raise ValueError(
                        f"Found {mm_count} images but the prompt contains more "
                        f"{self.info.get_image_placeholder()!r} placeholders "
                        "than that; counts must match."
                    )
                layout = build_image_block(
                    item.n_llm_h, item.n_llm_w, len(out_ids), config.vocab_size
                )
                block_layouts.append(layout)
                out_ids.extend(layout.token_ids)
            if next(image_iter, None) is not None:
                raise ValueError(
                    "The prompt contains fewer image placeholders than the "
                    "number of provided images."
                )

            mm_fields = BatchFeature(
                {
                    K_PATCHES: [patch for patch in patches],
                    K_GRID: torch.tensor(
                        [
                            [
                                item.n_vit_h,
                                item.n_vit_w,
                                item.n_llm_h,
                                item.n_llm_w,
                                len(layout.types) - layout.compress_pad,
                                len(layout.perm),
                            ]
                            for item, layout in zip(processed, block_layouts)
                        ],
                        dtype=torch.int32,
                    ),
                    K_TYPES: [
                        torch.tensor(
                            layout.types[layout.compress_pad :], dtype=torch.int64
                        )
                        for layout in block_layouts
                    ],
                    K_PERM: [
                        torch.tensor(layout.perm, dtype=torch.int64)
                        for layout in block_layouts
                    ],
                }
            )
            mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
                mm_fields,
                self._get_mm_fields_config(
                    mm_fields, inputs.hf_processor_mm_kwargs
                ),
            )

            # The placeholder range excludes the leading compressor-alignment
            # pads (see the module docstring): [IMAGE_START, IMAGE_END].
            mm_placeholders: dict[str, list[PlaceholderRange]] = {
                "image": [
                    PlaceholderRange(
                        offset=layout.start + layout.compress_pad,
                        length=len(layout.types) - layout.compress_pad,
                    )
                    for layout in block_layouts
                ]
            }

            mm_hashes = inputs.get_mm_hashes(
                self.info.model_id,
                self.info.ctx.get_mm_config().mm_hasher_algorithm,
            )

        return mm_input(
            prompt_token_ids=out_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


__all__ = [
    "DeepseekV4VisionProcessingInfo",
    "DeepseekV4VisionDummyInputsBuilder",
    "DeepseekV4VisionMultiModalProcessor",
]
