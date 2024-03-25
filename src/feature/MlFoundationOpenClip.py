import open_clip
import torch
import numpy as np
from typing import List
from PIL import Image
from collections.abc import Iterable

from FeatureExtractor import FeatureExtractor

class MlFoundationOpenClip(FeatureExtractor):
    """
    Feature extractors created by ML Foundation's open clip models
    see https://github.com/mlfoundations/open_clip

    code portions sourced from:
    https://gitlab.com/vgg/wise/wise/-/blob/a4499c57d3136a859cb03c839538394665867382/src/inference.py

    see FeatureExtractor.py for documentation of API
    """

    ID_PREFIX = 'mlfoundations/open_clip:'
    DESCRIPTION = 'See https://github.com/mlfoundations/open_clip'

    def __init__(self, id):
        if not id.startswith(self.ID_PREFIX):
            raise ValueError('feature id cannot start with {id} and must start with {self.id_prefix}')
        id_tokens = id.split(':')

        assert len(id_tokens) == 3
        if (id_tokens[1], id_tokens[2]) not in open_clip.list_pretrained():
            raise ValueError(f'Model ({id_tokens[1]}, {id_tokens[2]}) not available in {ID_PREFIX}')
        self.pretrained_model_name = id_tokens[1]
        self.pretraining_dataset = id_tokens[2]

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.pretrained_model_name,
                                                                               pretrained=self.pretraining_dataset,
                                                                               device=self.DEVICE)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.pretrained_model_name)

        # query model to get input image size and output feature dimension
        self._find_input_image_size()
        self._find_output_dim()

    def _find_input_image_size(self):
        self.input_image_size = self.model.visual.image_size
        if isinstance(self.input_image_size, Iterable):
            if isinstance(self.input_image_size, str):
                self.input_image_size = int(input_image_size)
                self.input_image_size = (self.input_image_size, self.input_image_size)
            else:
                self.input_image_size = tuple(self.input_image_size)[:2]
        elif isinstance(self.input_image_size, int):
            self.input_image_size = (self.input_image_size, self.input_image_size)
        else:
            raise NotImplementedError

    def _find_output_dim(self):
        """  Warmup the GPU with these models and find the output_dim reliably
        There seems to be no other API in open_clip repo to get the output_dim,
        than running the model
        """
        if not hasattr(self, 'output_dim'):
            input_image = Image.new("RGB", self.input_image_size)
            model_image_input = self.preprocess_image([input_image])
            model_image_features = self.extract_image_features(model_image_input)
            model_text_input = ['some random text']
            model_text_features  = self.extract_text_features(model_text_input)
            assert model_image_features.shape[1] == model_text_features.shape[1]
            self.output_dim = model_image_features.shape[1]

    def get_output_dim(self):
        return self.output_dim

    def get_input_image_size(self):
        return self.input_image_size

    def preprocess_image(self, images: List[Image.Image]) -> torch.Tensor:
        if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            result = torch.stack([self.preprocess(im) for im in images], dim=0).to(device=self.DEVICE)
            return result
        else:
            raise ValueError('all input to preprocess_image() must be an instance of PIL.Image')

    def extract_image_features(self, images: torch.Tensor) -> np.ndarray:
        if isinstance(images, torch.Tensor):
            model_input = images.to(device=self.DEVICE)
        else:
            raise ValueError('input to extract_features() must be an instance of torch.Tensor')

        with torch.no_grad():
            model_output = self.model.encode_image(model_input).float()
            model_output /= torch.linalg.norm(model_output, dim=-1, keepdims=True)
            return model_output.cpu().numpy()

    def extract_text_features(self, text_query: List[str]) -> np.ndarray:
        with torch.no_grad():
            model_input = self.tokenizer(text_query).to(device=self.DEVICE)
            model_output = self.model.encode_text(model_input).float()
            model_output /= torch.linalg.norm(model_output, dim=-1, keepdims=True)
            return model_output.cpu().numpy()
