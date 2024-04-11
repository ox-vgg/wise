from msclap import CLAP
import torch
import numpy as np
from typing import List
from collections.abc import Iterable

from .feature_extractor import FeatureExtractor

class MicrosoftClap(FeatureExtractor):
    """
    Audio feature extractors created by Microsoft's CLAP project
    see https://github.com/microsoft/CLAP/

    see FeatureExtractor.py for documentation of API
    """

    ID_PREFIX = 'microsoft/clap/'
    DESCRIPTION = 'See https://github.com/microsoft/CLAP'

    def __init__(self, id):
        if not id.startswith(self.ID_PREFIX):
            raise ValueError(f'feature id cannot start with {id} and must start with {self.ID_PREFIX}')
        id_tokens = id.split('/')

        assert len(id_tokens) == 4
        if id_tokens[2] not in CLAP.model_name:
            raise ValueError(f'Model version {id_tokens[2]} is not available. Available models are {CLAP.model_name.keys()}')
        self.version = id_tokens[2]
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLAP(version=self.version, use_cuda=self.DEVICE)

    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.model.default_collate(audio)

    def preprocess_text(self, text: str) -> str:
        return self.model.preprocess_text(text)

    def extract_audio_features(self, preprocessed_audio: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2]).to(device=self.DEVICE)
            return self.model.clap.audio_encoder(preprocessed_audio)[0].cpu()

    def extract_text_features(self, preprocessed_text: List[str]) -> np.ndarray:
        with torch.no_grad():
            self.model.clap.caption_encoder(preprocessed_text)
