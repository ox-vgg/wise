#!/usr/bin/env python3

import io
import logging
from tempfile import NamedTemporaryFile
from .exceptions import ModalityNotSupportedError, FeatureExtractorNotFoundError, NoFeaturesFoundError
from src.feature import FeatureExtractor, FeatureExtractorFactory
from pydantic import BaseModel
import numpy as np
from PIL import Image
import torch
from torch.hub import download_url_to_file
import torchaudio

logger = logging.getLogger(__name__)


def initialize_feature_extractors(
    feature_extractor_ids: list[str], config: dict
) -> dict[str, FeatureExtractor]:
    """
    Initialize feature extractors based on the provided feature_extractor_ids
    """
    feature_extractors = {}
    for _id in feature_extractor_ids:
        if _id in feature_extractors:
            logger.warning(f"Feature extractor {_id} is already initialized. Ignoring duplicate.")
            continue

        feature_extractor = FeatureExtractorFactory(_id, config)
        feature_extractors[_id] = feature_extractor

    return feature_extractors


def load_audio(x: list[io.BytesIO]) -> torch.Tensor:
    # TODO add support for loading multiple audio files
    if len(x) == 0:
        raise ValueError("No audio file was specified")
    elif len(x) > 1:
        raise NotImplementedError("Please specify 1 audio file only")
    
    target_sample_rate = 48_000 # TODO set this based on model?
    audio_file = x[0]
    waveform, original_sample_rate = torchaudio.load(audio_file)
    waveform = torchaudio.functional.resample(waveform, orig_freq=original_sample_rate, new_freq=target_sample_rate)
    return waveform

class EmbeddingConfig(BaseModel):
    query_prefix: str
    text_queries_weight: float = 2.0
    negative_queries_weight: float = 0.2

class EmbeddingService:
    def __init__(self, feature_extractors: dict[str, FeatureExtractor]):
        self.feature_extractors = feature_extractors

    @classmethod
    def from_ids(
        cls, feature_extractor_ids: list[str], config: dict
    ) -> "EmbeddingService":
        feature_extractors = initialize_feature_extractors(
            feature_extractor_ids, config
        )
        return cls(feature_extractors)

    def transform_distances(self, feature_extractor_id: str, distances: list) -> list:
        feature_extractor = self.feature_extractors.get(feature_extractor_id)
        if feature_extractor is None:
            raise FeatureExtractorNotFoundError(f"Feature extractor {feature_extractor_id} not initialized!")
        ret = feature_extractor.transform_faiss_distances_hook(np.array(distances))
        return ret.tolist()

    def transform_internal_image_queries(self, feature_extractor_id: str, image_query: np.ndarray) -> np.ndarray:
        feature_extractor = self.feature_extractors.get(feature_extractor_id)
        if feature_extractor is None:
            raise FeatureExtractorNotFoundError(f"Feature extractor {feature_extractor_id} not initialized!")

        return feature_extractor.transform_internal_image_queries_hook(image_query)

    def embed(
        self,
        feature_extractor_id: str,
        config: EmbeddingConfig,
        q: list[dict[str, np.ndarray | bytes | str]],
    ) -> np.ndarray:

        feature_vectors = []
        weights = []

        feature_extractor = self.feature_extractors.get(feature_extractor_id)
        if feature_extractor is None:
            raise FeatureExtractorNotFoundError(f"Feature extractor {feature_extractor_id} not initialized!")

        def extract_text_features(text: list[str]) -> np.ndarray:
            if feature_extractor.extract_text_features is None:
                raise ModalityNotSupportedError("text modality not supported")
            return feature_extractor.extract_text_features(text)

        def extract_image_features(images: list[Image.Image]) -> np.ndarray:
            if feature_extractor.extract_image_features is None:
                raise ModalityNotSupportedError("image modality not supported")
            assert len(images) == 1
            features = feature_extractor.extract_image_features(
                feature_extractor.preprocess_image(images)
            )[0]
            if not len(features.vectors):
                raise NoFeaturesFoundError("no features found on image")
            if len(features.vectors) > 1:
                logger.debug("multiple features found, will return vector for the top feature only")
            return features.vectors[0:1]

        def extract_audio_features(audio: torch.Tensor) -> np.ndarray:
            if feature_extractor.extract_audio_features is None:
                raise ModalityNotSupportedError("audio modality not supported")
            return feature_extractor.extract_audio_features(
                feature_extractor.preprocess_audio(audio)
            )

        for query_dict in q:
            query = query_dict["val"]
            feature_vector = None
            if query_dict['modality'] == 'image':
                if isinstance(query, bytes):
                    with Image.open(io.BytesIO(query)) as im:
                        im = im.convert('RGB')
                        feature_vector = extract_image_features([im])
                        weights.append(
                            config.negative_queries_weight
                            if query_dict["sign"] == "negative"
                            else 1
                        )
                elif isinstance(query, np.ndarray):
                    feature_vector = query
                    weights.append(
                        config.negative_queries_weight
                        if query_dict["sign"] == "negative"
                        else 1
                    )
                elif query.startswith(("http://", "https://")):
                    logger.info("Downloading %s to file", query)
                    with NamedTemporaryFile() as tmpfile:
                        download_url_to_file(query, tmpfile.name)
                        with Image.open(tmpfile.name) as im:
                            im = im.convert('RGB')
                            feature_vector = extract_image_features([im])
                            weights.append(
                                config.negative_queries_weight
                                if query_dict["sign"] == "negative"
                                else 1
                            )
            elif query_dict['modality'] == 'audio':
                if isinstance(query, bytes):
                    im = io.BytesIO(query)
                    feature_vector = extract_audio_features(load_audio([im]))
                    weights.append(
                        config.negative_queries_weight
                        if query_dict["sign"] == "negative"
                        else 1
                    )
                elif isinstance(query, np.ndarray):
                    feature_vector = query
                    weights.append(
                        config.negative_queries_weight
                        if query_dict["sign"] == "negative"
                        else 1
                    )
                elif query.startswith(("http://", "https://")):
                    logger.info("Downloading", query, "to file")
                    with NamedTemporaryFile() as tmpfile:
                        download_url_to_file(query, tmpfile.name)
                        with open(tmpfile.name, mode='rb') as f:
                            file_bytes_io = io.BytesIO(f.read())
                            feature_vector = extract_audio_features(load_audio([file_bytes_io]))
                            weights.append(
                                config.negative_queries_weight
                                if query_dict["sign"] == "negative"
                                else 1
                            )
            elif query_dict['modality'] == 'text':
                if config.query_prefix:
                    prefixed_queries = f"{config.query_prefix} {query.strip()}".strip()
                else:
                    prefixed_queries = query.strip()
                feature_vector = extract_text_features([prefixed_queries])
                weights.append(
                    config.text_queries_weight
                    * (  # assign higher weight to natural language queries
                        config.negative_queries_weight
                        if query_dict["sign"] == "negative"
                        else 1
                    )
                )
            else:
                raise ValueError(f"Unsupported modality: {query_dict['modality']}")

            if query_dict["sign"] == "negative":
                feature_vector = -feature_vector
            feature_vectors.append(feature_vector)
        weights = np.array(weights, dtype=np.float32)
        average_features = np.average(feature_vectors, axis=0, weights=weights)
        average_features /= np.linalg.norm(average_features, axis=-1, keepdims=True)
        return average_features
