#!/usr/bin/env python3

## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import io
import logging
from tempfile import NamedTemporaryFile
from .exceptions import ModalityNotSupportedError, FeatureExtractorNotFoundError, NoFeaturesFoundError
from src.feature import FeatureExtractor, FeatureExtractorFactory
from pydantic import BaseModel, HttpUrl
import numpy as np
from PIL import Image
import torch
from torch.hub import download_url_to_file
import torchaudio
import typing
from fastapi import HTTPException

from ...common import MediaQueryTerm, Query, TextQueryTerm, VectorQueryTerm

logger = logging.getLogger(__name__)


def _is_HttpUrl(obj) -> bool:
    ## XXX: drop this function when we depend on pydantic>=2.10.
    ## Before pydantic 2.10, HttpUrl implementation was a subscripted
    ## generic so we couldn't just use instance, see
    ## https://github.com/pydantic/pydantic/pull/10766
    if type(HttpUrl) is type:
        return isinstance(obj, HttpUrl)
    else:
        return isinstance(obj, typing.get_args(HttpUrl)[0])


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
        q: Query,
    ) -> np.ndarray:

        feature_vectors = []
        weights = []

        feature_extractor = self.feature_extractors.get(feature_extractor_id)
        if feature_extractor is None:
            raise FeatureExtractorNotFoundError(f"Feature extractor {feature_extractor_id} not initialized!")

        def extract_text_features(text: str) -> np.ndarray:
            if feature_extractor.extract_text_features is None:
                raise ModalityNotSupportedError("text modality not supported")
            return feature_extractor.extract_text_features([text])

        def extract_image_features(image: Image.Image) -> np.ndarray:
            if feature_extractor.extract_image_features is None:
                raise ModalityNotSupportedError("image modality not supported")
            features = feature_extractor.extract_image_features(
                feature_extractor.preprocess_image([image])
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

        for qterm in q:
            feature_vector = None
            if isinstance(qterm, VectorQueryTerm):
                feature_vector = qterm.vector
            elif isinstance(qterm, MediaQueryTerm):
                if qterm.qtype == "visual":
                    if isinstance(qterm.src, bytes):
                        with Image.open(io.BytesIO(qterm.src)) as im:
                            im = im.convert('RGB')
                            feature_vector = extract_image_features(im)
                    elif _is_HttpUrl(qterm.src):
                        logger.info("Downloading %s to file", qterm.src)
                        with NamedTemporaryFile() as tmpfile:
                            download_url_to_file(qterm.src, tmpfile.name)
                            with Image.open(tmpfile.name) as im:
                                im = im.convert('RGB')
                                feature_vector = extract_image_features(im)
                    else:
                        raise HTTPException(
                            400, {"message": "Unhandled query term"}
                        )
                elif qterm.qtype == "audio":
                    if isinstance(qterm.src, bytes):
                        au = io.BytesIO(qterm.src)
                        feature_vector = extract_audio_features(load_audio([au]))
                    elif _is_HttpUrl(qterm.src):
                        logger.info("Downloading", qterm.src, "to file")
                        with NamedTemporaryFile() as tmpfile:
                            download_url_to_file(qterm.src, tmpfile.name)
                            with open(tmpfile.name, mode='rb') as f:
                                file_bytes_io = io.BytesIO(f.read())
                                feature_vector = extract_audio_features(
                                    load_audio([file_bytes_io])
                                )
                    else:
                        raise HTTPException(
                            400, {"message": "Unhandled qterm"}
                        )
                else:
                    raise HTTPException(
                        400, {"message": "Unhandled qtype '%s'" % qterm.qtype}
                    )

            elif isinstance(qterm, TextQueryTerm):
                if config.query_prefix:
                    prefixed_queries = f"{config.query_prefix} {qterm.txt.strip()}".strip()
                else:
                    prefixed_queries = qterm.txt.strip()

                feature_vector = extract_text_features(prefixed_queries)

            else:
                raise ValueError(f"Unsupported query term: {type(qterm)}")

            if qterm.is_negative:
                feature_vector = -feature_vector
            feature_vectors.append(feature_vector)


            if qterm.is_negative:
                weights.append(config.negative_queries_weight)
            else:
                weights.append(1)
            ## Assign different weight to natural language queries
            if isinstance(qterm, TextQueryTerm):
                weights[-1] *= config.text_queries_weight

        weights = np.array(weights, dtype=np.float32)
        average_features = np.average(feature_vectors, axis=0, weights=weights)
        average_features /= np.linalg.norm(average_features, axis=-1, keepdims=True)
        return average_features
