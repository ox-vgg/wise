from dataclasses import dataclass
from typing import Any, List, Union

from PIL import Image
import torch
import numpy as np
import sqlalchemy as sa


@dataclass
class Features:
    """Object for extracted features vectors as well as any metadata.

    Attributes
    ----------
    vectors
        A 2D numpy ndarray, the first dimension being the number of
        features (in a single image).  The length of the second
        dimension is the dimensionality of the embedding and dependent
        on the model used by the `FeatureExtractor`.

    metadata
        A free for all place, specific to each feature extractor.
        This is the metadata for all feature vectors so it might make
        sense for this to be a list but each `FeatureExtractor` is
        free to use in whatever manner.  This will be the argument to
        :meth:`FeatureExtractor.add_to_vector_metadata_table`.

    """
    metadata: Any  # free for use by concrete FeatureExtractor
    vectors: np.ndarray  # shaped (n-features, embedding-length)


class FeatureExtractor:
    """ABC for extractor of feature vectors from audio, image, and text.

    If a subclass will not support specific modalities, e.g., the
    model does not handle audio, set the methods for that modality to
    `None` (see :py:exc:`NotImplementedError`).

    """

    def __init__(self):
        raise NotImplementedError

    def create_vector_metadata_table(self, db_engine: sa.Engine) -> None:
        """Create if needed a table for these features metadata.
        """
        pass  # default to no-op

    def add_to_vector_metadata_table(
        self, conn:sa.Connection, vid: list[int], metadata: Any
    ) -> None:
        """Add vector metadata to the database.

        Parameters
        ----------
        conn
        ----
            Connection for the insert (if any).
        vid
            List of the ids in the vectors table.
        metadata
            The `metadata` attributed of the `Features` returned by
            :meth:`.extract_image_features`.

        """
        pass  # default to no-op

    def preprocess_image(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """ Preprocess media to prepare it for feature extraction

        Parameters
        ----------
        images : a list of PIL.Image where each element represents an image

        Returns
        -------
        torch.Tensor
            a torch tensor representing pre-processed images
        """
        raise NotImplementedError

    def extract_image_features(self, images: torch.Tensor) -> list[Features]:
        """Extracts features from pre-processed images

        Parameters
        ----------
        images : a torch.Tensor containing pre-processed images

        Returns
        -------
        list[Features]
            One :class:`Feature` object per input image.

        """
        raise NotImplementedError

    def preprocess_text(self, text: str) -> str:
        raise NotImplementedError

    def extract_text_features(self, text_query: List[str]) -> np.ndarray:
        """ Extracts features from text

        Parameters
        ----------
        text_query : a list of strings

        Returns
        -------
        np.ndarray
            a numpy ndarray containing extracted feature vectors
        """
        raise NotImplementedError

    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract_audio_features(self, preprocessed_audio: torch.Tensor) -> np.ndarray:
        raise NotImplementedError
