#!/usr/bin/env python3

## Copyright (C) 2025 University of Oxford

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import NamedTuple, Union

import numpy as np
import PIL.Image
import sqlalchemy as sa
import torch

## Importing InsightFace requires some care:
##
##   1. The Python package has an undeclared dependency on onnxruntime
##      (see https://github.com/deepinsight/insightface/pull/2437).  I
##      guess this is because there are two distributions providing
##      the package (onnxruntime and onxxruntime-gpu) and there is no
##      support on requirements.txt to list alternative distributions.
##      We import it ourselves to provide a cleaner error message.
##
##   2. cv2 needs to be imported before onnxruntime (and therefore,
##      before insightface), because otherwise onnxruntime loads an
##      older version of libstdc++ which is incompatible with newer
##      python-opencv versions (you get a "version `GLIBCXX_x.y.zz'
##      not found" error).  (at this point, torch or numpy might
##      actually have already imported cv2).
##
##   3. InsightFace imports albumentations which by default call PyPI
##      to check if it's the latest version and prints a message if
##      not.  We need to set env variable to disable that.

# isort: off
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # disable check for version
import cv2  # import before onnxruntime
import onnxruntime  # import before insightface for cleaner error
import insightface.app
# isort: on

from .feature_extractor import FeatureExtractor, Features


_logger = logging.getLogger(__name__)


def rgb_nchw_to_bgr_nhwc(images: torch.Tensor) -> torch.Tensor:
    assert images.shape[1] == 3
    images = images.flip(1)  # RGB -> BGR
    images = images.transpose(1, 3).transpose(1, 2)  # (N,C,H,W) -> (N,H,W,C)
    return images


def rgb_pil_to_bgr_hwc_tensor(pil_img: PIL.Image.Image) -> torch.Tensor:
    assert pil_img.mode == "RGB"
    tensor_img = torch.as_tensor(np.array(pil_img, copy=True))
    tensor_img = tensor_img.view(pil_img.size[1], pil_img.size[0], 3)
    tensor_img = tensor_img.flip(2)  # RGB -> BGR
    return tensor_img


def pil_img_list_to_nhwc_tensor(images: list[PIL.Image.Image]) -> torch.Tensor:
    if len(images) == 0:
        return torch.empty([0, 1024, 768, 3])

    assert len({x.size for x in images}) in [0, 1], \
        "multiple PIL images of different sizes"
    return torch.stack([rgb_pil_to_bgr_hwc_tensor(x) for x in images])


class BBox(NamedTuple):
    x: float
    y: float
    w: float
    h: float


@dataclass
class FaceFeatureMetadata:
    detection_score: float
    bbox: BBox
    age: int
    is_male: bool

    @classmethod
    def from_app_face(cls, face: insightface.app.common.Face, img: np.ndarray):
        assert img.ndim == 3
        img_w = float(img.shape[1])
        img_h = float(img.shape[0])
        bbox = face.bbox.copy()
        bbox[(0, 2),] /= img_w
        bbox[(1, 3),] /= img_h
        bbox[(2, 3),] -= bbox[(0, 1),]
        return cls(
            detection_score=face.det_score,
            bbox=BBox(*bbox),
            age=face.age,
            is_male=(face.sex == "M"),
        )

    def to_sql_values(self, vector_id: int):
        return {
            "vector_id": vector_id,
            "detection_score" : self.detection_score,
            "bbox_x": self.bbox.x,
            "bbox_y": self.bbox.y,
            "bbox_w": self.bbox.w,
            "bbox_h": self.bbox.h,
            "age": self.age,
            "is_male": self.is_male,
        }


class InsightFaceFeatureExtractor(FeatureExtractor):

    ## InsightFace supports image only (no audio and no text)
    preprocess_text = None
    extract_text_features = None
    preprocess_audio = None
    extract_audio_features = None

    def __init__(self, feature_id: str):
        _logger.info("initialising feature extractor for %s", feature_id)
        feature_id_parts = feature_id.split("/")
        assert (
            len(feature_id_parts) == 4
            and feature_id_parts[0] == "insightface"
            and feature_id_parts[1] == "_"
            and feature_id_parts[3] == "_"
        )
        model_name = feature_id_parts[2]
        ## XXX: investigate allowed_modules arg to FaceAnalysis

        ## InsightFace and GPU
        ##
        ## ONNX models are loaded when FaceAnalysis is constructed.
        ## Whether the models are loaded in CPU or GPU (or others) is
        ## dependent on the available execution providers (see
        ## https://onnxruntime.ai/docs/execution-providers/) and any
        ## particular model dependency.
        ##
        ## For CUDA, the user needs to have ONNX runtime built with
        ## CUDA (onnxruntime-gpu on PyPI).  This can be checked with
        ## onnxruntime.get_available_providers().  But cudnn is also
        ## needed and that is only checked after loading the model.
        ## So we do nothing and check afterwards if the CUDA provider
        ## is available to the models.
        ##
        ## The ctx_id argument to prepare is undocumented, but does
        ## the following (checked from reading the source).  If
        ## negative, uses CPU even if the CUDA provider is available.
        ## If non-negative, it uses the default preference which is
        ## CUDA if available and CPU if not.  The ctx_id is NOT the
        ## index for the GPU.  So we just leave ctx_id set to zero to
        ## use the ONNX possible preference default (which seem
        ## reasonable).

        ## FaceAnalysis() and FaceAnalysis.prepare() print to stdout
        ## which mess up our own stdout so we throw it away.
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                self._app = insightface.app.FaceAnalysis(
                    model_name,
                    allowed_modules=["detection", "recognition", "genderage"],
                )
                self._app.prepare(
                    ctx_id=0,
                    det_thresh=0.5,
                    det_size=(640, 640),
                )

        ## Check if all models have the CUDA execution model available
        ## to them.
        for task_name, model in self._app.models.items():
            task_providers = model.session.get_providers()
            _logger.debug(
                "InsightFace '%s' task has '%s' execution providers available",
                task_name,
                task_providers,
            )
            if "CUDAExecutionProvider" not in task_providers:
                _logger.warning(
                    "CUDA provider not available for '%s' task", task_name
                )

        recognition_outputs = self._app.models[
            "recognition"
        ].session.get_outputs()[0]

        assert (
            len(recognition_outputs.shape) == 2
            and recognition_outputs.shape[0] == 1
        )
        self._embedding_size = recognition_outputs.shape[-1]

        ## It should be possible to get the numpy type from the
        ## onxxruntime NodeArg type but I couldn't figure it out.  So
        ## just hardcode float which seems to be the case for all
        ## models anyway.
        assert recognition_outputs.type == "tensor(float)"
        self._embedding_dtype = np.float32

    def create_vector_metadata_table(self, db_engine: sa.Engine) -> None:
        db_metadata_obj = sa.MetaData()
        db_metadata_obj.reflect(db_engine)
        self._vector_metadata_table = sa.Table(
            "vector_metadata_insightface",
            db_metadata_obj,
            sa.Column(
                "vector_id",
                sa.ForeignKey("vectors.id", ondelete="cascade"),
                nullable=False,
            ),
            sa.Column("detection_score", sa.Float, nullable=False),
            ## InsightFace returns (x0, y0, x1, y1) in absolute
            ## coordinates and we convert to (x, y, w, h) in relative
            ## coordinates because that's what frontend uses.
            sa.Column("bbox_x", sa.Float, nullable=False),  # [0.0-1.0]
            sa.Column("bbox_y", sa.Float, nullable=False),  # [0.0-1.0]
            sa.Column("bbox_w", sa.Float, nullable=False),  # [0.0-1.0]
            sa.Column("bbox_h", sa.Float, nullable=False),  # [0.0-1.0]
            sa.Column("age", sa.Integer, nullable=False),
            ## InsightFace returns gender/sex with two options only,
            ## so we use is_male so we can use boolean type.
            sa.Column("is_male", sa.Boolean, nullable=False),
            keep_existing=True,
        )
        db_metadata_obj.create_all(db_engine)

    def add_to_vector_metadata_table(
        self, conn:sa.Connection, vid: list[int], metadata: list[FaceFeatureMetadata]
    ) -> None:
        ## This condition is needed because insert with an empty list
        ## does an insert with NULLs instead of "nothing".  See
        ## https://github.com/sqlalchemy/sqlalchemy/discussions/9645
        if len(metadata) > 0:
            conn.execute(
                sa.insert(self._vector_metadata_table),
                [x.to_sql_values(vid) for vid, x in zip(vid, metadata)],
            )

    def preprocess_image(
        self, images: Union[torch.Tensor, list[PIL.Image.Image]]
    ) -> torch.Tensor:
        _logger.debug("preprocessing images of type %s", type(images))
        if isinstance(images, torch.Tensor):
            if images.ndim != 4 or images.shape[1] != 3:
                raise Exception("expect tensor images to be RGB in NCHW order")
            return rgb_nchw_to_bgr_nhwc(images)
        elif isinstance(images, list):
            if not all([isinstance(x, PIL.Image.Image) for x in images]):
                raise Exception("expect list images to all be PIL Image")
            elif not all([x.mode == "RGB" for x in images]):
                raise Exception("expect list of PIL images to be in RGB mode")
            return pil_img_list_to_nhwc_tensor(images)
        else:
            raise Exception("unexpected input images of type %s" % type(images))

    @torch.inference_mode
    def extract_image_features(self, images: torch.Tensor) -> list[Features]:
        _logger.debug("extracting image features from a %s", type(images))
        features: list[Features] = []
        for image in images:
            ## NB: undocumented but `get()` expects a numpy ndarray,
            ##     of shape `(H, W, C)`, and mode BRG.  The mode seems
            ##     less (not?) important for detection but has an
            ##     impact on recognition models, namely gender
            ##     recognition.
            ## NB: undocumented but `get()` returns faces ordered by
            ##     detection score.  We want to preserve that order.
            ##     This is so that when someone makes a search with
            ##     multiple faces, the face used for the search is the
            ##     one we are most confident about (the big frontal
            ##     and centre face instead of a small, barely
            ##     noticeable, face in the background).
            faces = self._app.get(image.numpy())
            _logger.debug("found %d faces", len(faces))

            feature_vectors = np.empty(
                (len(faces), self._embedding_size), dtype=self._embedding_dtype
            )
            feature_metadata: list[FaceFeatureMetadata] = []
            for i, face in enumerate(faces):
                feature_vectors[i] = face.normed_embedding
                feature_metadata.append(
                    FaceFeatureMetadata.from_app_face(face, image)
                )
            features.append(
                Features(vectors=feature_vectors, metadata=feature_metadata)
            )

        return features
