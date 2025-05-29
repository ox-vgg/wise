from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Union

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


class BBoxXYWH(NamedTuple):
    """Bounding box in (x0, y0, w, h) coordinates relative to image size.

    Coordinates (x0, y0, w, h) in range [0, 1], relative to the size
    of the image.

    """
    x: float
    y: float
    w: float
    h: float


@dataclass
class FeatureExtMetadata:
    """Optional feature metadata that WISE "core" knows how to handle.

    Individual :class:`FeatureExtractor` keep whatever metadata they
    want for each feature, in whatever format they want, in their own
    separate db table.  Those are the Ext metadata, and separate from
    the vector metadata common to all `FeatureExtractor` and stored in
    the "core" vectors table.

    This is what :meth:`FeatureExtractor.get_vector_metadata` returns.

    All Ext metadata attributes are optional because feature extractors
    are not required to compute, store, or return any of them.

    """
    bbox: Optional[BBoxXYWH] = None


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
        self, conn: sa.Connection, vid: list[int], metadata: Any
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

    def get_vector_metadata(
        self, conn: sa.Connection, vid: list[int]
    ) -> list[FeatureExtMetadata]:
        """Get Ext vector metadata from the database.

        Since `FeatureExtractor` are not required to have this
        metadata, default to return a list with none of it.
        Subclasses may overload this if they want.

        Parameters
        ----------
        conn
        ----
            Connection for the insert (if any).
        vid
            List of the ids in the vectors table.

        Returns
        -------
        list[FeatureExtMetadata]
            One for each of input `vid` and in the same order.

        """
        return [FeatureExtMetadata() for _ in range(len(vid))]

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

    def transform_internal_image_queries_hook(self, vec: np.ndarray) -> np.ndarray:
        """Hook method to transform internal image queries. This hook is useful for
        models like OWLv2 which requires feature vectors to be 'augmented'. See
        src/feature/transformers_owlv2.py::extract_image_features() for an example.

        This method can be overridden by subclasses to apply specific 
        transformations to internal image queries. By default, it does nothing.

        Parameters
        ----------
        vec : np.ndarray
            The input feature vector, with shape (num_queries, vector_dim).

        Returns
        -------
        np.ndarray
            The transformed feature vector, with shape
            (num_queries, vector_dim).
        """
        return vec

    def transform_faiss_distances_hook(self, dist: np.ndarray) -> np.ndarray:
        """Hook method to transform Faiss distance scores. This is useful for models
        like OWLv2 which requires a sigmoid function to be applied to the similarity
        scores returned by Faiss. This transformation ensures that similarity scores
        are in the range of 0 and 1 and therefore ready for the frontend.

        This method can be overridden by subclasses to apply specific 
        transformations to the distance scores returned by Faiss. By default, 
        it does nothing.

        Parameters
        ----------
        dist : np.ndarray
            The input Faiss distance scores, with shape (num_queries, k).

        Returns
        -------
        np.ndarray
            The transformed Faiss distance scores, with shape (num_queries, k).
        """
        return dist

    def warmup(self):
        """
        Warmup method to be implemented by subclasses
        Useful when the models are lazy loaded and if someone wants to
        eagerly load them and allocate memory beforehand
        """
        pass


def get_torch_device(device: str | torch.device | None = None):
    """
    Get the torch device object for use in feature extractors.
    Useful when the calling code wants to override placement, or use other special accelerators

    torch.device -> torch.device
    "" or None -> cuda if available else cpu
    any other string -> torch.device(string)
    """
    if isinstance(device, torch.device):
        return device
    
    _default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _device = device or _default_device
    return torch.device(_device)
