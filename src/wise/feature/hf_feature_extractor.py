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

import logging
from functools import cached_property
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from wise.feature.feature_extractor import (
    FeatureExtractor,
    Features,
    get_torch_device,
)


logger = logging.getLogger(__name__)


class HFMultiModalFeatureExtractor(FeatureExtractor):
    """Feature extractor for Hugging Face models.

    This class is a placeholder for future implementations that will
    support Hugging Face models for feature extraction.
    """
    ID_PREFIX = 'hf/'
    DESCRIPTION = 'Hugging Face feature extractor'

    class Config(FeatureExtractor.Config):
        preprocessor_kwargs: dict[str, Any] = {}
        model_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        id,
        *,
        device: str | torch.device | None = None,
        warmup: bool = False,
        config: Config = Config(),
        **kwargs,
    ):
        if not id.startswith(self.ID_PREFIX):
            raise ValueError(f'Feature ID must start with {self.ID_PREFIX}, got {id}')

        id_tokens = id.split('/')
        assert len(id_tokens) == 4, f'Invalid feature ID format: {id}'

        model_name, _dataset = id[len(self.ID_PREFIX):].rsplit('/', 1)
        self.DEVICE = get_torch_device(device)

        self._config = config
        self.preprocessor_kwargs = config.preprocessor_kwargs
        self.model_kwargs = config.model_kwargs

        self.model_config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_config(self.model_config, **self.model_kwargs)

        self.logit_scale = getattr(model, "logit_scale", torch.tensor(0.0))
        self.logit_bias = getattr(model, "logit_bias", torch.tensor(0.0))

        if not hasattr(model, 'get_image_features'):
            self.extract_image_features = None

        if not hasattr(model, 'get_text_features'):
            self.extract_text_features = None

        if not hasattr(model, 'get_audio_features'):
            self.extract_audio_features = None

        if warmup:
            self.warmup()

        self.__model_name = model_name

    @cached_property
    def processor(self):
        return AutoProcessor.from_pretrained(self.__model_name, use_fast=True)

    @cached_property
    def model(self):
        logger.info(f'Initialising model {self.ID_PREFIX} - {self.__model_name} (device={self.DEVICE})')
        model = AutoModel.from_pretrained(self.__model_name, config=self.model_config, device_map=f'{self.DEVICE}', **self.model_kwargs)
        model.eval()
        if self._config.compile:
            available_backends = torch._dynamo.list_backends()
            backend = "inductor"
            if "tensorrt" in available_backends:
                backend = "tensorrt"
            logger.info(f"Compiling model with backend {backend}")
            model = torch.compile(model, mode="reduce-overhead", backend=backend)
        return model

    def preprocess_image(self, images):
        return images

    def preprocess_audio(self, audio):
        return audio

    def preprocess_text(self, text):
        return text

    @torch.inference_mode()
    def extract_image_features(self, images: torch.Tensor) -> list[Features]:

        """Extracts features from pre-processed images.

        Parameters
        ----------
        images : torch.Tensor
            A tensor containing pre-processed images.

        Returns
        -------
        list[Features]
            One :class:`Feature` object per input image.
        """
        inputs = self.processor(images=images, return_tensors='pt', **self.preprocessor_kwargs).to(self.DEVICE)
        outputs = self.model.get_image_features(**inputs)
        outputs = outputs / torch.linalg.norm(outputs, dim=-1, keepdim=True)  # Normalize features
        outputs = outputs.cpu().numpy()
        feature_vectors = list(np.expand_dims(outputs, axis=1))
        return [Features(vectors=x, metadata=None) for x in feature_vectors]

    @torch.inference_mode()
    def extract_text_features(self, text_query: list[str]) -> np.ndarray:
        """Extracts features from text.

        Parameters
        ----------
        text_query : list[str]
            A list of strings representing the text queries.

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape (n, d) where n is the number of text queries
            and d is the feature dimension.
        """
        inputs = self.processor(text=text_query, return_tensors='pt', **self.preprocessor_kwargs).to(self.DEVICE)
        outputs = self.model.get_text_features(**inputs)
        outputs = outputs / torch.linalg.norm(outputs, dim=-1, keepdim=True)  # Normalize features
        outputs = outputs.cpu().numpy()

        return outputs

    @torch.inference_mode()
    def extract_audio_features(self, audio: torch.Tensor) -> np.ndarray:
        """Extracts features from text.

        Parameters
        ----------
        audio : torch.Tensor
            A tensor containing audio samples

        Returns
        -------
        list of Features
            A list of `Features` objects, one for each audio sample in the input tensor.
        """
        inputs = self.processor(audio=audio, return_tensors='pt', **self.preprocessor_kwargs).to(self.DEVICE)
        outputs = self.model.get_audio_features(**inputs)
        outputs = outputs / torch.linalg.norm(outputs, dim=-1, keepdim=True)  # Normalize features
        outputs = outputs.cpu().numpy()
        feature_vectors = list(np.expand_dims(outputs, axis=1))
        return [Features(vectors=x, metadata=None) for x in feature_vectors]

    def warmup(self):
        """Warm up the model by running a dummy forward pass."""
        logger.info('Warming up model')

        self.extract_image_features(torch.rand((1, 3, 224, 224)))
        self.extract_text_features(['dummy text'])

    def transform_faiss_distances_hook(self, dist: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            dist_tensor = torch.from_numpy(dist)
            dist_tensor = dist_tensor * self.logit_scale.exp() + self.logit_bias
            return dist_tensor.detach().numpy()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    extractor = HFMultiModalFeatureExtractor('hf/openai/clip-vit-base-patch32', warmup=True)
    image_features = extractor.extract_image_features(torch.rand((1, 3, 224, 224)))
    text_features = extractor.extract_text_features(['Hello, world!'])
    print(image_features, text_features)
