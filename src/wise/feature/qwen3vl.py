#!/usr/bin/env python3

## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from functools import cached_property
import logging

import numpy as np
import torch
import torchvision.transforms.functional as TF
from qwen_vl_utils import process_vision_info
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel, Qwen3VLModel,
)

from wise.feature.feature_extractor import FeatureExtractor, Features, get_torch_device
from wise.feature.hf_models import get_model_info

logger = logging.getLogger(__name__)


# Defined at module level (not inside a function) so pickle can locate it by
# qualified name when DataLoader spawns worker processes.
# The Qwen3-VL-Embedding checkpoint was saved from a class whose inner model
# lives at self.model, giving every weight key a "model.*" prefix.  Loading
# the bare Qwen3VLModel via AutoModel (no prefix) means no keys match and the
# entire language model gets randomly initialised.  This wrapper restores the
# expected key structure without loading the unused LM head.
class _Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def forward(self, **kwargs):
        return self.model(**kwargs)


# Qwen3-VL uses 16-pixel patches; IMAGE_FACTOR = patch_size * spatial_merge_size (2).
# FRAME_MAX_PIXELS caps each individual frame; MAX_TOTAL_PIXELS caps the total across
# all frames in a segment, letting fetch_video do proportional smart-resize.
_IMAGE_PATCH_SIZE = 16
_IMAGE_FACTOR = _IMAGE_PATCH_SIZE * 2          # 32
_FRAME_MAX_PIXELS = 768 * _IMAGE_FACTOR ** 2   # 786 432 ≈ 896×896 per frame
_MAX_TOTAL_PIXELS = 10 * _FRAME_MAX_PIXELS     # 7 864 320 across all frames


class Qwen3VLEmbeddingFeatureExtractor(FeatureExtractor):
    """Segment-level video feature extractor using Qwen3-VL-Embedding models.

    Produces one embedding vector per video segment (multiple frames -> one vector).
    Supports video_segment and text modalities.
    """

    ID_PREFIX = "hf/Qwen/Qwen3-VL-Embedding/"

    # Frame-level and audio modalities are not supported
    preprocess_image = None
    extract_image_features = None
    extract_image_region_features = None
    preprocess_audio = None
    extract_audio_features = None

    class Config(FeatureExtractor.Config):
        # torch.compile is not beneficial for VLMs with dynamic visual token counts
        # and adds significant startup overhead for large models — disabled by default.
        compile: bool = False

    def __init__(
        self,
        model_id: str,
        *,
        device: str | torch.device | None = None,
        warmup: bool = False,
        config: "Qwen3VLEmbeddingFeatureExtractor.Config" = None,
        **kwargs,
    ):
        if config is None:
            config = self.Config()
        super().__init__(model_id, device=device)

        info = get_model_info(model_id)
        if info is None:
            raise ValueError(
                f"Unknown Qwen3-VL model id {model_id!r}. "
                "Check src/feature/hf_models.py REGISTRY."
            )
        self._hf_model_id = info["hf_model_id"]
        self.compile = config.compile

        if warmup:
            self.warmup()

    @cached_property
    def _model_and_processor(self):
        from transformers import AutoProcessor

        import logging as _logging
        logger.info(
            "Loading Qwen3-VL-Embedding model %s on %s", self._hf_model_id, self.DEVICE
        )
        # padding_side='right' is required for last-token pooling: with right-padding
        # the last real token is always at index (attention_mask.sum() - 1).
        processor = AutoProcessor.from_pretrained(
            self._hf_model_id, trust_remote_code=True, padding_side="right"
        )
        # Suppress harmless key-mismatch warnings from transformers: because
        # _Qwen3VLForEmbedding is a custom wrapper class (not the original
        # checkpoint class), from_pretrained logs spurious unexpected/missing-key
        # warnings even though the weights load correctly.
        _hf_logger = _logging.getLogger("transformers.modeling_utils")
        _prev_level = _hf_logger.level
        _hf_logger.setLevel(_logging.ERROR)
        try:
            model = _Qwen3VLForEmbedding.from_pretrained(
                self._hf_model_id,
                trust_remote_code=True,
                dtype=torch.float16 if self.DEVICE.type == "cuda" else torch.float32,
                device_map={"": self.DEVICE},
            )
        finally:
            _hf_logger.setLevel(_prev_level)
        model.eval()
        logger.info("Qwen3-VL-Embedding model loaded on %s", self.DEVICE)
        if self.compile and self.DEVICE.type == "cuda":
            logger.info("Compiling Qwen3-VL model with torch.compile (this may take several minutes)")
            model = torch.compile(model, mode="reduce-overhead")
        return model, processor

    @property
    def _model(self):
        return self._model_and_processor[0]

    @property
    def _processor(self):
        return self._model_and_processor[1]

    @staticmethod
    def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last-token pooling: take the final non-padding token's hidden state.

        Qwen3-VL is a causal decoder — position 0 (BOS) attends only to itself
        and carries no semantic content.  The last token of the assistant-prefix
        has attended to the full input sequence, making it the correct embedding.
        Requires right-padding so that the last real token sits at sum(mask)-1.
        """
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths
        ]

    def preprocess_video_segment(self, frames: torch.Tensor) -> torch.Tensor:
        """No-op: frames are kept on CPU until the processor converts them to PIL.

        Device transfer happens inside extract_video_segment_features after
        the processor has built the model inputs.
        """
        return frames

    @torch.inference_mode()
    def extract_video_segment_features(self, frames: torch.Tensor) -> Features:
        """Extract a single embedding for a video segment.

        Parameters
        ----------
        frames:
            Tensor of shape (N, C, H, W) — N frames from one segment, uint8 or float.

        Returns
        -------
        Features
            One embedding vector (shape (1, D)) representing the entire segment.
        """
        model, processor = self._model_and_processor

        frames_cpu = frames.cpu()
        if frames_cpu.dtype != torch.uint8:
            frames_u8 = (frames_cpu * 255).clamp(0, 255).to(torch.uint8)
        else:
            frames_u8 = frames_cpu

        pil_frames = [TF.to_pil_image(f) for f in frames_u8]

        # Pass all frames as a single 'video' entry so the model applies temporal
        # video encoding.  total_pixels lets fetch_video smart-resize proportionally
        # across frames.  do_resize=False in the processor call prevents a second
        # resize after process_vision_info has already handled it.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": pil_frames,
                        "total_pixels": _MAX_TOTAL_PIXELS,
                    }
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, image_patch_size=_IMAGE_PATCH_SIZE,
            return_video_metadata=True, return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos, video_metadata = None, None
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            padding=True,
            do_resize=False,
            **video_kwargs,
        )
        inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}

        outputs = model(**inputs)

        embedding = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embedding = embedding.float()
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return Features(vectors=embedding.cpu().numpy(), metadata=None)

    @torch.inference_mode()
    def extract_text_features(self, text_query: list[str]) -> np.ndarray:
        model, processor = self._model_and_processor

        # Each text query must go through apply_chat_template so it lands in
        # the same embedding space as video segments (both use the same format).
        texts = []
        for q in text_query:
            messages = [{"role": "user", "content": [{"type": "text", "text": q}]}]
            texts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}

        outputs = model(**inputs)

        embedding = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embedding = embedding.float()
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()

    def warmup(self):
        logger.info("Warming up Qwen3VLEmbeddingFeatureExtractor")
        dummy_frames = torch.zeros(4, 3, 224, 224, dtype=torch.uint8)
        self.extract_video_segment_features(dummy_frames)
        self.extract_text_features(["warmup"])
