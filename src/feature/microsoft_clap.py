from functools import cached_property
import logging
from msclap import CLAP
import torch
import numpy as np
from typing import List
from collections.abc import Iterable

from .feature_extractor import FeatureExtractor, get_torch_device

logger = logging.getLogger(__name__)

class MicrosoftClap(FeatureExtractor):
    """
    Audio feature extractors created by Microsoft's CLAP project
    see https://github.com/microsoft/CLAP/

    see FeatureExtractor.py for documentation of API
    """

    ID_PREFIX = 'microsoft/clap/'
    DESCRIPTION = 'See https://github.com/microsoft/CLAP'

    ## CLAP supports text and audio (no image)
    preprocess_image = None
    extract_image_features = None

    def __init__(
        self, id, device: str | torch.device | None = None, warmup: bool = False
    ):
        if not id.startswith(self.ID_PREFIX):
            raise ValueError(f'feature id cannot start with {id} and must start with {self.ID_PREFIX}')
        id_tokens = id.split('/')

        assert len(id_tokens) == 4
        if id_tokens[2] not in CLAP.model_name:
            raise ValueError(f'Model version {id_tokens[2]} is not available. Available models are {CLAP.model_name.keys()}')
        self.version = id_tokens[2]
        self.DEVICE = get_torch_device(device)

        if warmup:
            self.warmup()

    @cached_property
    def model(self):
        use_cuda = self.DEVICE.type == 'cuda'
        logger.info(f'Initialising model {self.ID_PREFIX} ({self.version}, use_cuda={use_cuda})')
        return CLAP(version=self.version, use_cuda=use_cuda)

    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        # CLAP accepts (1xN_samples)
        if audio.shape[0] > 2:
            audio = torch.transpose(audio, 0, 1)
        # the CLAP model only accepts single channel audio
        if audio.shape[0] != 1:
            audio = torch.mean(audio, 0, keepdim=True)
        return self.model.default_collate([audio])

    def preprocess_text(self, text: str) -> str:
        return self.model.preprocess_text(text)

    @torch.inference_mode()
    def extract_audio_features(self, preprocessed_audio: torch.Tensor) -> np.ndarray:
        preprocessed_audio = preprocessed_audio.reshape(
            preprocessed_audio.shape[0], preprocessed_audio.shape[2]).to(device=self.DEVICE)
        audio_embeddings = self.model.clap.audio_encoder(preprocessed_audio)[0]
        audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
        return audio_embeddings.cpu().numpy()

    @torch.inference_mode()
    def extract_text_features(self, text: List[str]) -> np.ndarray:
        preprocessed_text = self.model.preprocess_text(text)
        text_embeddings = self.model.clap.caption_encoder(preprocessed_text)
        text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
        return text_embeddings.cpu().numpy()

    def warmup(self):
        logger.info("Warming up model")
        random_audio = torch.rand((1, 192_000))
        preprocessed_audio = self.preprocess_audio(random_audio)
        audio_embedding = self.extract_audio_features(preprocessed_audio)
        text_embedding = self.extract_text_features(["some random text"])
        assert audio_embedding.shape[1] == text_embedding.shape[1]
        return
