from functools import cached_property
import logging
from pathlib import Path
import open_clip
import torch
import numpy as np
from typing import List, Union, Any
from PIL import Image
import torchvision.transforms.functional as F
from collections.abc import Iterable

from .feature_extractor import (
    FeatureExtractor,
    Features,
    MultiModalModel,
)

logger = logging.getLogger(__name__)


def _load_openclip_model(
    model_name: str,
    pretrained: str | None = None,
    device: str | torch.device = "cpu",
    **kwargs,
):
    """Load the model and preprocess function."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device, **kwargs
    )
    model.eval()
    return model, preprocess


class MlfoundationsOpenClipModel(MultiModalModel):
    
    @cached_property       
    def model(self):
        logger.info(
            f"Initialising model mlfoundations/openclip model - {self.model_id} ({self.pretraining_dataset}, device={self.DEVICE})"
        )
        model, _ = _load_openclip_model(
            self.model_id,
            pretrained=self.pretraining_dataset,
            device=self.DEVICE,
            **self.model_kwargs,
        )
        model.eval()
        if self.compile:
            available_backends = torch._dynamo.list_backends()
            backend = "inductor"
            if "tensorrt" in available_backends:
                backend = "tensorrt"
            logger.info(f"Compiling model with backend {backend}")
            model.compile(mode="reduce-overhead", backend=backend)
        return model

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

    @torch.inference_mode()
    def get_image_features(self, **kwargs) -> torch.Tensor:
        """Extract image features from the model."""
        images = kwargs.get("images", None)
        if not isinstance(images, torch.Tensor):
            raise ValueError(
                "Image tensor input is required for image feature extraction."
            )
        return self.model.encode_image(images.to(self.DEVICE)).float()

    @torch.inference_mode()
    def get_text_features(self, **kwargs) -> torch.Tensor:
        """Extract text features from the model."""
        text = kwargs.get("input_ids", None)
        if not isinstance(text, torch.Tensor):
            raise ValueError(
                "tokenized text input is required for text feature extraction."
            )
        return self.model.encode_text(text.to(self.DEVICE)).float()

    get_audio_features = None  # OpenClip does not support audio features

    def export_to_onnx(
        self, save_path: Path, visual_inputs: tuple, text_inputs: tuple, **kwargs
    ):
        """Export the model to ONNX format."""
        model = self.model
        model.eval()

        logger.info(f"Exporting vision model...")
        output_path = Path(f"{save_path}--image")
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "model.onnx"

        dynamic_axes = {
            "images": {
                0: "batch_size",
            },
            "embeddings": {
                0: "batch_size",
            },
        }
        with torch.inference_mode():
            torch.onnx.export(
                model.visual,
                visual_inputs,
                output_path,
                input_names=["images"],
                output_names=[
                    "embeddings",
                ],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                opset_version=17,
            )
        logger.info(f"Successfully exported vision model to {output_path}")

        class CustomTextEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                return self.model.encode_text(input_ids)

        logger.info(f"Exporting text model")
        output_path = Path(f"{save_path}--text")
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "model.onnx"

        dynamic_axes = {
            "input_ids": {
                0: "batch_size",
            },
            "embeddings": {
                0: "batch_size",
            },
        }
        text_model = CustomTextEncoder(model)
        text_model.eval()
        with torch.inference_mode():
            torch.onnx.export(
                text_model,
                text_inputs,
                output_path,
                input_names=["input_ids"],
                output_names=[
                    "embeddings",
                ],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                opset_version=17,
            )
        logger.info(f"Successfully exported text model to {output_path}")


class MlfoundationOpenClipFeatureExtractor(FeatureExtractor):
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

    class Config(FeatureExtractor.Config):
        model_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        id,
        *,
        device: str | torch.device | None = None,
        warmup: bool = False,
        config = Config(),
        **kwargs,
    ):
        super().__init__(id, device=device)
        id_tokens = id.split('/')

        assert len(id_tokens) == 4
        if (id_tokens[2], id_tokens[3]) not in open_clip.list_pretrained():
            raise ValueError(f'Model ({id_tokens[2]}, {id_tokens[3]}) not available in {self.ID_PREFIX}')

        self.pretrained_model_name = id_tokens[2]
        self.pretraining_dataset = id_tokens[3]

        self.model_kwargs = config.model_kwargs
        self.compile = config.compile

        _model, self.preprocess = _load_openclip_model(
            self.pretrained_model_name,
            pretrained=self.pretraining_dataset,
            device="cpu",
            **self.model_kwargs,
        )

        self.tokenizer = open_clip.get_tokenizer(self.pretrained_model_name)
        with torch.inference_mode():
            self.logit_scale = (
                _model.logit_scale.detach()
                if hasattr(_model, "logit_scale") and _model.logit_scale is not None
                else torch.tensor(0.0)
            )
            self.logit_bias = (
                _model.logit_bias.detach()
                if hasattr(_model, "logit_bias") and _model.logit_bias is not None
                else torch.tensor(0.0)
            )
        del _model  # we only needed it to get the preprocess function
        if warmup:
            self.warmup()

    @cached_property
    def tokenizer(self):
        return open_clip.get_tokenizer(self.pretrained_model_name)

    @cached_property
    def model(self):
        model = MlfoundationsOpenClipModel(
            model_id=self.pretrained_model_name,
            pretraining_dataset=self.pretraining_dataset,
            device=self.DEVICE,
            compile=self.compile,
            **self.model_kwargs,
        )
        return model

    @property
    def input_image_size(self):
        return self.model.input_image_size

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

    def preprocess_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError(
                "input to preprocess_text() must be an instance of str or List[str]"
            )

        return {"input_ids": self.tokenizer(text)}

    @torch.inference_mode()
    def extract_image_features(self, images: torch.Tensor) -> list[Features]:
        if not isinstance(images, torch.Tensor):
            raise ValueError('input to extract_features() must be an instance of torch.Tensor')

        model_output = self.model.get_image_features(images=images).float()
        model_output /= torch.linalg.norm(model_output, dim=-1, keepdims=True)
        model_output = model_output.cpu().numpy()
        feature_vectors = list(np.expand_dims(model_output, axis=1))
        return [Features(vectors=x, metadata=None) for x in feature_vectors]

    @torch.inference_mode()
    def extract_text_features(self, text_query: List[str]) -> np.ndarray:
        model_input = self.preprocess_text(text_query)
        model_output = self.model.get_text_features(**model_input).float()
        model_output /= torch.linalg.norm(model_output, dim=-1, keepdims=True)
        return model_output.cpu().numpy()

    def warmup(self):
        # calculating the output dim does the warmup anyway
        _ = self.output_dim
        return

    def transform_faiss_distances_hook(self, dist: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            dist_tensor = torch.from_numpy(dist)
            dist_tensor = dist_tensor * self.logit_scale.exp() + self.logit_bias

            return dist_tensor.detach().numpy()
