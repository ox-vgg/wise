from functools import cached_property
import logging
import open_clip
import torch
import numpy as np
from typing import List, Union
from PIL import Image
import torchvision.transforms.functional as F
from collections.abc import Iterable

from .feature_extractor import FeatureExtractor, Features, get_torch_device

logger = logging.getLogger(__name__)

class MlfoundationOpenClip(FeatureExtractor):
    """
    Feature extractors created by ML Foundation's open clip models
    see https://github.com/mlfoundations/open_clip

    code portions sourced from:
    https://gitlab.com/vgg/wise/wise/-/blob/a4499c57d3136a859cb03c839538394665867382/src/inference.py

    see FeatureExtractor.py for documentation of API
    """

    ID_PREFIX = 'mlfoundations/open_clip/'
    DESCRIPTION = 'See https://github.com/mlfoundations/open_clip'

    ## CLIP supports text and image (no audio)
    preprocess_audio = None
    extract_audio_features = None

    def __init__(
        self, id, device: str | torch.device | None = None, warmup: bool = False
    ):
        if not id.startswith(self.ID_PREFIX):
            raise ValueError(f'feature id cannot start with {id} and must start with {self.ID_PREFIX}')
        id_tokens = id.split('/')

        assert len(id_tokens) == 4
        if (id_tokens[2], id_tokens[3]) not in open_clip.list_pretrained():
            raise ValueError(f'Model ({id_tokens[2]}, {id_tokens[3]}) not available in {self.ID_PREFIX}')
        self.pretrained_model_name = id_tokens[2]
        self.pretraining_dataset = id_tokens[3]

        self.DEVICE = get_torch_device(device)

        if warmup:
            self.warmup()

    @cached_property
    def _models(self):
        logger.info(f'Initialising model {self.ID_PREFIX} - {self.pretrained_model_name} ({self.pretraining_dataset}, device={self.DEVICE})')
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.pretrained_model_name,
            pretrained=self.pretraining_dataset,
            device=self.DEVICE
        )
        model.eval()
        return model, preprocess

    @cached_property
    def tokenizer(self):
        return open_clip.get_tokenizer(self.pretrained_model_name)

    @property
    def model(self):
        _model, _ = self._models
        return _model

    @property
    def preprocess(self):
        _, _preprocess = self._models
        return _preprocess

    @property
    def input_image_size(self):
        _input_image_size = self.model.visual.image_size
        if isinstance(_input_image_size, Iterable):
            if isinstance(_input_image_size, str):
                _input_image_size = int(_input_image_size)
                _input_image_size = (_input_image_size, _input_image_size)
            else:
                _input_image_size = tuple(_input_image_size)[:2]
        elif isinstance(_input_image_size, int):
            _input_image_size = (_input_image_size, _input_image_size)
        else:
            raise NotImplementedError

        return _input_image_size

    @cached_property
    def output_dim(self):
        """  Warmup the GPU with these models and find the output_dim reliably
        There seems to be no other API in open_clip repo to get the output_dim,
        than running the model
        """
        logger.info('Warming up model and calculating output dimensions')
        random_image = torch.rand( (1, 3,) + (self.input_image_size) )
        model_image_input = self.preprocess_image(random_image)
        model_image_features = self.extract_image_features(model_image_input)
        model_text_input = ['some random text']
        model_text_features  = self.extract_text_features(model_text_input)
        assert model_image_features[0].vectors.shape[1] == model_text_features.shape[1]
        return model_text_features.shape[1]

    def preprocess_image(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            result = torch.stack([self.preprocess(im) for im in images], dim=0).to(device=self.DEVICE)
            return result
        elif isinstance(images, torch.Tensor) and len(images.shape) == 4:
            result = torch.stack([self.preprocess(F.to_pil_image(im)) for im in images], dim=0).to(device=self.DEVICE)
            return result

        else:
            raise ValueError('all input to preprocess_image() must be an instance of torch.Tensor or PIL.Image')

    @torch.inference_mode()
    def extract_image_features(self, images: torch.Tensor) -> list[Features]:
        if isinstance(images, torch.Tensor):
            model_input = images.to(device=self.DEVICE)
        else:
            raise ValueError('input to extract_features() must be an instance of torch.Tensor')

        model_output = self.model.encode_image(model_input).float()
        model_output /= torch.linalg.norm(model_output, dim=-1, keepdims=True)
        model_output = model_output.cpu().numpy()
        feature_vectors = list(np.expand_dims(model_output, axis=1))
        return [Features(vectors=x, metadata=None) for x in feature_vectors]

    @torch.inference_mode()
    def extract_text_features(self, text_query: List[str]) -> np.ndarray:
        model_input = self.tokenizer(text_query).to(device=self.DEVICE)
        model_output = self.model.encode_text(model_input).float()
        model_output /= torch.linalg.norm(model_output, dim=-1, keepdims=True)
        return model_output.cpu().numpy()

    def warmup(self):
        # calculating the output dim does the warmup anyway
        _ = self.output_dim
        return
