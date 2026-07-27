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

import functools
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Type

import numpy as np
import sqlalchemy as sa
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torchvision.ops import box_iou

try:
    ## Importing torch_tensorrt registers 'tensorrt' as a custom torch
    ## compile backend which we use as the default if available.
    import torch_tensorrt  # pylint: disable=unused-import
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)


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


def _box_iou_xywh(boxes1_xywh: Tensor, boxes2_xywh: Tensor) -> Tensor:
    """box_iou implementation to support torchvision <0.24.

    torchvision 0.24 added the fmt option to box_iou to support boxes
    in xywh format (torchvision<0.24 only supported the xyxy format).
    We are currently stuck on torchvision<0.24 (see comments on #175)
    so we do it ourselves.

    """
    boxes1_xyxy = boxes1_xywh.detach().clone()
    boxes1_xyxy[:, 2:] = boxes1_xyxy[:, :2] + boxes1_xyxy[:, 2:]
    boxes2_xyxy = boxes2_xywh.detach().clone()
    boxes2_xyxy[:, 2:] = boxes2_xyxy[:, :2] + boxes2_xyxy[:, 2:]
    return box_iou(boxes1_xyxy, boxes2_xyxy)


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

    def _load_torch_model(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement this method.")

    ## TODO: instead of having _*_torch_model (to handle cases where
    ## model is not a nn.Module), maybe have a separate class for
    ## those.  Or maybe, those classes should be returning nn.Module
    ## (currently, this is only insightface).
    @functools.cache
    def _build_torch_model(self) -> nn.Module:
        """Template method for models that are `torch.nn.Module`.

        Models that are nn.Module need to overload `_load_torch_model`
        method and its `model` property only needs to call this
        method.

        """
        logger.info(
            "Initialising model %s on device '%s'", self.model_id, self.DEVICE
        )
        model = self._load_torch_model()
        model = model.to(self.DEVICE)
        model.eval()
        if self.compile:
            available_backends = torch._dynamo.list_backends()
            if "tensorrt" in available_backends:
                backend = "tensorrt"
            else:
                logger.warning(
                    "torch_tensorrt is not installed.  Models will be compiled"
                    " with inductor backend which may be less performant."
                )
                backend = "inductor"
            logger.info("Compiling model with backend %s", backend)
            model.compile(mode="reduce-overhead", backend=backend)
        return model

    @property
    @abstractmethod
    def model(self):
        raise NotImplementedError("Subclasses must implement this property.")

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
    ) -> "FeatureExtractor":
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
        """Create if needed a table for these features metadata."""
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

    def preprocess_image(
        self, images: torch.Tensor | list[Image.Image]
    ) -> torch.Tensor:
        """Preprocess media to prepare it for feature extraction

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

    def preprocess_image_region(
        self, image: torch.Tensor | Image.Image, region: BBoxXYWH
    ) -> torch.Tensor:
        """Preprocess image to call :meth:`extract_image_region_features`.

        Args:
            image: unlike the other `preprocess_*` methods, this takes
                only one image to be used with one region.
            region: the bounding box that defines the image region
                that will be processed.

        .. seealso::

            :meth:`_preprocess_image_region_crop` and
            :meth:`_preprocess_image_region_nocrop` for drop-in
            implementations of this method.

        """
        raise NotImplementedError

    def extract_image_region_features(
        self, image: torch.Tensor, region: BBoxXYWH
    ) -> Features:
        """Extract features for the given image region.

        The meaning of "image region" is up to each FeatureExtractor
        concrete implementation and should be done in concert with
        :meth:`preprocess_image_region`:

        * For feature extractors that return a single vector
          representing the visual content, it may make sense to simply
          crop the image and do the same as `extract_image_features`.
          In that case, consider using
          :meth:`_preprocess_image_region_crop` with
          :meth:`extract_image_features`.

        * For feature extractors that use a detector, it may make more
          sense to process the whole image and then select one of the
          regions using the `region` argument as hint.  In that case,
          consider using :meth:`_preprocess_image_region_nocrop` and
          look into
          :meth:`_extract_image_region_features_highest_iou`.

        """
        raise NotImplementedError

    def extract_video_segment_features(self, frames: torch.Tensor) -> Features:
        """Extract a single embedding for a multi-frame video segment.

        Parameters
        ----------
        frames : torch.Tensor
            Tensor of shape (N, C, H, W) — N frames from one temporal segment.

        Returns
        -------
        Features
            One embedding vector representing the entire segment.
        """
        raise NotImplementedError

    def preprocess_text(self, text: str) -> str:
        raise NotImplementedError

    def extract_text_features(self, text_query: list[str]) -> np.ndarray:
        """Extracts features from text

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

    def extract_audio_features(
        self, preprocessed_audio: torch.Tensor
    ) -> np.ndarray:
        raise NotImplementedError

    def transform_internal_image_queries_hook(
        self, vec: np.ndarray
    ) -> np.ndarray:
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

    def _preprocess_image_region_crop(
        self, image: torch.Tensor | Image.Image, region: BBoxXYWH
    ) -> torch.Tensor:
        """Implementation of :meth:`preprocess_image_region` that crops image."""
        if isinstance(image, torch.Tensor):
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("expect Tensor image to be RGB in CHW order")
            crop = F.crop(
                image,
                round(region.y * image.shape[1]),
                round(region.x * image.shape[2]),
                round(region.h * image.shape[1]),
                round(region.w * image.shape[2]),
            )
            return self.preprocess_image(crop.unsqueeze(0))
        elif isinstance(image, Image.Image):
            crop = image.crop(
                round(region.x * image.width),
                round(region.y * image.height),
                round((region.x + region.w) * image.width),
                round((region.y + region.h) * image.height),
            )
            return self.preprocess_image([crop])
        else:
            raise TypeError("unexpected input images of type %s" % type(image))

    def _preprocess_image_region_nocrop(
        self, image: torch.Tensor | Image.Image, region: BBoxXYWH
    ) -> torch.Tensor:
        """Implementation of :meth:`preprocess_image_region` that ignores region."""
        if isinstance(image, torch.Tensor):
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("expect Tensor image to be RGB in CHW order")
            return self.preprocess_image(image.unsqueeze(0))
        elif isinstance(image, Image.Image):
            return self.preprocess_image([image])
        else:
            raise TypeError("unexpected input images of type %s" % type(image))

    def _extract_image_region_features_highest_iou(
        self, image: torch.Tensor, region: BBoxXYWH
    ) -> Features:
        """Implementation of :meth:`extract_image_region_features` that returns region with highest IoU."""
        features = self.extract_image_features(image)
        assert len(features) == 1
        ## FIXME: once we can depend on torchvision>0.24, we can
        ## replace _box_iou_xywh() with box_iou(..., fmt="xywh")
        iou = _box_iou_xywh(
            torch.stack([torch.Tensor(x.bbox) for x in features[0].metadata]),
            torch.Tensor(region).unsqueeze(0),
        )
        iou_max_idx = iou.argmax()
        return Features(
            vectors=features[0].vectors[[iou_max_idx], :],
            metadata=[features[0].metadata[iou_max_idx]],
        )


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

    _default_device = "cuda" if torch.cuda.is_available() else "cpu"
    _device = device or _default_device
    return torch.device(_device)
