from collections.abc import Iterable
import enum
import logging
from typing import List, Tuple, cast, Union
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Compose
from torch.utils.data import Dataset
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
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    logger.info("Loaded")

    return model, preprocess, tokenizer


def setup_clip(model_name: CLIPModel = "ViT-B-32:openai"):

    model, preprocess, tokenizer = _load_clip(model_name)
    input_image_size = model.visual.image_size

    if isinstance(input_image_size, Iterable):
        if isinstance(input_image_size, str):
            input_image_size = int(input_image_size)
            input_image_size = (input_image_size, input_image_size)
        else:
            input_image_size = tuple(input_image_size)[:2]
    elif isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    else:
        raise NotImplementedError

    def get_output_dim():
        """
        Warmup the GPU with these models
        and find the output_dim reliably
        There seems to be no other API in open_clip repo to
        get the output_dim, than running the model
        """
        im_features = extract_image_features([Image.new("RGB", input_image_size)])
        text_features = extract_text_features(["dummy"])
        assert im_features.shape[1] == text_features.shape[1]
        return im_features.shape[1]

    def extract_image_features(images: Union[torch.Tensor, List[Image.Image]]) -> np.ndarray:
        if isinstance(images, torch.Tensor):
            _input = images.to(device=DEVICE)
        elif isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            _input = torch.stack([preprocess(im) for im in images], dim=0).to(device=DEVICE)

        with torch.no_grad():
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

    output_dim = get_output_dim()
    return output_dim, preprocess, extract_image_features, extract_text_features


class LinearBinaryClassifier(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
