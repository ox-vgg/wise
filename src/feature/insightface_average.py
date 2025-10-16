#!/usr/bin/env python3

## Copyright (C) 2025 University of Oxford

import logging

import numpy as np
import torch

from .feature_extractor import FeatureExtractor, Features
from .insightface import InsightFaceFeatureExtractor


_logger = logging.getLogger(__name__)


class InsightFaceAverageFeatureExtractor(FeatureExtractor):
    """Feature extractor to find images with multiple people.

    This feature extractor is based on InsightFaceFeatureExtractor but
    specialised to find images with multiple specific people.  It
    stores one feature vector per image which is the average of all
    face feature vectors.

    This works well enough to find images where both of the query
    people appear.

    """

    ## InsightFace supports image only (no audio and no text)
    preprocess_text = None
    extract_text_features = None
    preprocess_audio = None
    extract_audio_features = None

    def __init__(self, feature_id: str, *args, **kwargs):
        feature_id_parts = feature_id.split("/")
        assert (
            len(feature_id) == 4
            and feature_id_parts[0] == "deepinsight"
            and feature_id_parts[1] == "insightface-average",
            (
                f"Invalid feature-id: '{feature_id}', an example of a valid"
                f" feature-id is 'deepinsight/insightface-average/buffalo_l/_'"
            ),
        )
        feature_id_parts[1] = "insightface"  # remove the "-average" suffix
        self._extractor = InsightFaceFeatureExtractor(
            "/".join(feature_id_parts), *args, **kwargs
        )

    def preprocess_image(self, *args, **kwargs):
        return self._extractor.preprocess_image(*args, **kwargs)

    def extract_image_features(self, images: torch.Tensor) -> list[Features]:
        ## Average the vectors and set metadata to None
        face_features = self._extractor.extract_image_features(images)
        for features in face_features:
            features.metadata = None
            if features.vectors.shape[0] > 1:
                average_vector = np.average(
                    features.vectors, axis=0, keepdims=True
                )
                average_vector /= np.linalg.norm(average_vector, keepdims=True)
                features.vectors = average_vector
        return face_features

    def warmup(self, *args, **kwargs):
        return self._extractor.warmup(*args, **kwargs)
