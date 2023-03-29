import enum
import logging
from typing import List, Tuple, cast
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Compose
import open_clip

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AVAILABLE_MODELS = open_clip.list_pretrained(as_str=True)


class _CLIPModel(str, enum.Enum):
    pass


CLIPModel = _CLIPModel("CLIPModel", {x: x for x in AVAILABLE_MODELS})


def _load_clip(clip_model: CLIPModel):

    model_name, pretrained = clip_model.value.split(":", 1)
    logger.info(f"Loading CLIP (model: {model_name})...")
    model, preprocess = cast(
        Tuple[open_clip.CLIP, Compose],
        open_clip.create_model_from_pretrained(
            model_name, pretrained=pretrained, device=DEVICE
        ),
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    logger.info("Loaded")

    return model, preprocess, tokenizer


def setup_clip(model_name: CLIPModel = "ViT-B-32:openai"):

    model, preprocess, tokenizer = _load_clip(model_name)
    output_dim = model.visual.output_dim

    def extract_image_features(images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():

            _input = torch.stack([preprocess(im) for im in images], dim=0).to(
                device=DEVICE
            )
            output = model.encode_image(_input).float()
            output /= torch.linalg.norm(output, dim=-1, keepdims=True)

            return output.cpu().numpy()

    def extract_text_features(
        queries: List[str],
    ) -> np.ndarray:

        with torch.no_grad():

            text_tokens = tokenizer(queries).to(device=DEVICE)

            output = model.encode_text(text_tokens).float()
            output /= torch.linalg.norm(output, dim=-1, keepdims=True)

            return output.cpu().numpy()

    return output_dim, extract_image_features, extract_text_features


class LinearBinaryClassifier(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
