import logging
from pathlib import Path
from typing import List, Union
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import clip

IS_CUDA = torch.cuda.is_available()

AVAILABLE_MODELS = clip.available_models()

logger = logging.getLogger(__name__)


def _load_clip(model_name: str):

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model - {model_name}, available models: {AVAILABLE_MODELS}"
        )
    logger.info(f"Loading CLIP (model: {model_name})...")
    model, preprocess = clip.load(model_name)

    if IS_CUDA:
        model.cuda()
    model.eval()
    logger.info("Loaded")

    return model, preprocess


def setup_clip(model_name: str = "ViT-B/32"):

    model, preprocess = _load_clip(model_name)
    input_dim = model.visual.input_resolution
    output_dim = model.visual.output_dim
    mean_tensor = (
        torch.Tensor([0.48145466, 0.4578275, 0.40821073])
        .reshape(3, 1, 1)
        .repeat(1, input_dim, input_dim)
    )

    def _preprocess(p: Union[Path, BytesIO]):
        try:
            with Image.open(p) as im:
                return preprocess(im)
        except Exception as e:
            logger.warning(f"warning: failed to process {p} - {e}")
            return mean_tensor

    def extract_image_features(images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():

            _input = torch.stack([preprocess(im) for im in images], dim=0)

            if IS_CUDA:
                _input = _input.cuda()

            output = model.encode_image(_input).float()
            output /= torch.linalg.norm(output, dim=-1, keepdims=True)

            if IS_CUDA:
                output = output.cpu()

            return output.numpy()

    def extract_text_features(
        queries: List[str],
    ) -> np.ndarray:

        with torch.no_grad():

            text_tokens = clip.tokenize(queries)
            if IS_CUDA:
                text_tokens = text_tokens.cuda()

            output = model.encode_text(text_tokens).float()
            output /= torch.linalg.norm(output, dim=-1, keepdims=True)

            if IS_CUDA:
                output = output.cpu()

            return output.numpy()

    return output_dim, extract_image_features, extract_text_features


class LinearBinaryClassifier(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
