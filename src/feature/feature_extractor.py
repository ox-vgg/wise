from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Type
from pydantic import BaseModel, ConfigDict
from PIL import Image
import torch
import numpy as np
import sqlalchemy as sa


class FeatureExtractorConfig(BaseModel):
    """Configuration for a feature extractor.
    Feature extractor implementations can extend this class to add
    additional configuration parameters and validation.

    The base feature extractor class will use this configuration to configure
    the feature extractor instance.

    This class is expected to be defined within the feature extractor implementation as class attribute `Config`

    Attributes
    ----------
    device : str | torch.device | None
        The device to use for the feature extractor. If None, it defaults to 'cuda'
        if available, otherwise 'cpu'. It can also be a string representing a specific
        device (e.g., 'cuda:0', 'cpu', etc.).

    warmup : bool
        Whether to warm up the feature extractor. This is useful for models that are lazy loaded
        and if someone wants to eagerly load them and allocate memory beforehand.

    compile : bool
        Whether to compile the model using `torch.compile()`. This can improve performance
        for some models, but may not be supported for all models or devices.

    """

    model_config = ConfigDict(from_attributes=True)

    device: str | None = None
    warmup: bool = False
    compile: bool = True  # whether to compile the model using torch.compile()


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


@dataclass(kw_only=True)
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


def check_config_in_init_args(cls: Type["FeatureExtractor"]) -> bool:
    """Check if the class constructor accepts a config argument."""
    parameters = inspect.signature(cls.__init__).parameters
    if "config" not in parameters:
        return False
    param = parameters["config"]
    return issubclass(param.annotation, FeatureExtractorConfig) and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class MultiModalModel(ABC):
    """Base class for multi-modal models.

    This class is used to define the interface for multi-modal models that can
    extract features from images, text, and audio. Subclasses should implement
    the methods to extract features for each modality.

    """

    def __init__(
        self,
        model_id: str,
        pretraining_dataset: str | None = None,
        device: str | torch.device | None = None,
        compile: bool = True,
        **kwargs,
    ):
        self.model_id = model_id
        self.pretraining_dataset = pretraining_dataset
        self.DEVICE = get_torch_device(device)
        self.compile = compile
        self.model_kwargs = kwargs

    @abstractmethod
    def get_image_features(self, **kwargs):
        """Extracts image features."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_text_features(self, **kwargs):
        """Extracts text features."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_audio_features(self, **kwargs):
        """Extracts audio features."""
        raise NotImplementedError("Subclasses must implement this method.")

    def export_to_onnx(self, output_path: str):
        """Export the model to ONNX format."""
        raise NotImplementedError(
            "No implementation for ONNX export found for this model"
        )


class FeatureExtractor:
    """ABC for extractor of feature vectors from audio, image, and text.

    If a subclass will not support specific modalities, e.g., the
    model does not handle audio, set the methods for that modality to
    `None` (see :py:exc:`NotImplementedError`).

    """
    ID_PREFIX = None
    _vector_metadata_table: sa.Table | None = None
    class Config(FeatureExtractorConfig):
        """Configuration for the feature extractor."""

        pass

    @classmethod
    def from_config(
        cls,
        model_id: str,
        config: dict[str, Any] = {},
    ) -> FeatureExtractor:
        """Create a feature extractor instance from the given configuration."""
        _config = cls.Config.model_validate(config)

        return cls(
            model_id,
            device=get_torch_device(_config.device),
            warmup=_config.warmup,
            config=_config,
        )

    def __init__(
        self,
        model_id: str,
        *,
        device: str | torch.device | None = None,
    ):
        if self.ID_PREFIX is None:
            raise ValueError(
                "FeatureExtractor.ID_PREFIX must be set to a non-empty string by the subclass"
            )

        if not model_id.startswith(self.ID_PREFIX):
            raise ValueError(
                f"feature id cannot start with {model_id} and must start with {self.ID_PREFIX}"
            )
        self.DEVICE = get_torch_device(device)

    @classmethod
    def create_vector_metadata_table(cls, db_engine: sa.Engine) -> None:
        """Create if needed a table for these features metadata.
        """
        pass  # default to no-op

    @classmethod
    def add_to_vector_metadata_table(
        cls, conn: sa.Connection, vid: list[int], metadata: Any
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

    @classmethod
    def get_vector_metadata(
        cls, conn: sa.Connection, vid: list[int]
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

    def preprocess_image(self, images: torch.Tensor | list[Image.Image]) -> torch.Tensor:
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

    def extract_text_features(self, text_query: list[str]) -> np.ndarray:
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
