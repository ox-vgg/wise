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
        # CLAP accepts (1xN_samples)
        if audio.shape[0] > 2:
            audio = torch.transpose(audio, 0, 1)
        # the CLAP model only accepts single channel audio
        if audio.shape[0] != 1:
            audio = torch.mean(audio, 0, keepdim=True)
        return self.model.default_collate([audio])

    def preprocess_text(self, text: str) -> str:
        return self.model.preprocess_text(text)

    def extract_audio_features(self, preprocessed_audio: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2]).to(device=self.DEVICE)
            audio_embeddings = self.model.clap.audio_encoder(preprocessed_audio)[0]
            audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
            return audio_embeddings.cpu().numpy()

    def extract_text_features(self, text: List[str]) -> np.ndarray:
        preprocessed_text = self.model.preprocess_text(text)
        with torch.no_grad():
            text_embeddings = self.model.clap.caption_encoder(preprocessed_text)
            text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
            return text_embeddings.cpu().numpy()
