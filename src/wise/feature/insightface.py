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

from __future__ import annotations

import contextlib
import logging
import os
import os.path
from dataclasses import dataclass
from functools import cached_property

import huggingface_hub
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
##
##   4. torch must be import before onnxruntime so that onnxruntime
##      picks up the CUDA and cuDNN from torch and eliminate the need
##      for separate cuDNN installation, see
##      https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch

# isort: off
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # disable check for version
import cv2  # import before onnxruntime
import onnxruntime  # import before insightface for cleaner error
import insightface.app
# isort: on

from ..db import project_metadata_obj
from .feature_extractor import (
    BBoxXYWH,
    FeatureExtMetadata,
    FeatureExtractor,
    Features,
    MultiModalModel,
)


_logger = logging.getLogger(__name__)


def _default_insightface_models_dir():
    root = os.path.expanduser("~/.insightface")
    return os.path.join(root, "models")


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


@dataclass
class FaceInferenceResponse:
    """
    Represents the response from the InsightFace inference.
    Contains the embeddings, scores, boxes, age, gender
    """

    embeddings: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    age: np.ndarray
    is_male: np.ndarray

    def to_app_face(self) -> list[insightface.app.common.Face]:
        """
        Converts the inference response to a Faces object.
        """
        metadata = []
        for score, box, age, is_male in zip(
            self.scores, self.boxes, self.age, self.is_male
        ):
            _face = insightface.app.common.Face(
                det_score=score.item(),
                bbox=box,
                age=age.item(),
                gender=int(is_male)
            )
            metadata.append(_face)
        return metadata

    @classmethod
    def from_tensor(cls, **outputs):
        """
        Converts the outputs from the InsightFace model to a FaceInferenceResponse.
        """
        embeddings = outputs["embeddings"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        boxes = outputs["boxes"].cpu().numpy()
        age = outputs["age"].cpu().numpy()
        is_male = outputs["is_male"].cpu().numpy()
        return cls(
            embeddings=embeddings, scores=scores, boxes=boxes, age=age, is_male=is_male
        )


@dataclass(kw_only=True)
class FaceFeatureMetadata(FeatureExtMetadata):
    bbox: BBoxXYWH
    detection_score: float
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
            bbox=BBoxXYWH(*bbox),
            age=face.age,
            is_male=(face.sex == "M"),
        )

    @classmethod
    def from_sql_values(cls, row: dict):
        return cls(
            detection_score=row["detection_score"],
            bbox=BBoxXYWH(
                row["bbox_x"],
                row["bbox_y"],
                row["bbox_w"],
                row["bbox_h"],
            ),
            age=row["age"],
            is_male=row["is_male"],
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


class InsightFaceModel(MultiModalModel):
    @cached_property
    def model(self):
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

        # TODO - based on device, pass the current provider and provider options as kwargs

        ## XXX: InsightFace is the Python package and comes with its
        ## own model_zoo.  But it is also possible to use it with
        ## weights from elsewhere, namely Hugging Face Hub which has
        ## AuraFace (which is Apache 2).  The model id has four four
        ## components (although at this point we already discarded the
        ## first two elements).  We first assume that we are using
        ## InsightFace own model_zoo (in which case we only use the
        ## third element --- the fourth is ignored).  If that fails,
        ## then we try the Hugging Face Hub with both elements.

        _logger.info(f"Initialising model {self.model_id}")
        insightface_model_id = self.model_id.split("/", maxsplit=1)[0]
        huggingface_hub_model_id = self.model_id

        def get_app(model_id):
            return insightface.app.FaceAnalysis(
                model_id,
                allowed_modules=["detection", "recognition", "genderage"],
            )

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                _logger.info(
                    "Trying first with a model named '%s'",
                    insightface_model_id
                )
                try:
                    _app = get_app(insightface_model_id)
                except:
                    _logger.info(
                        "model '%s' failed; trying '%s' from huggingface_hub",
                        insightface_model_id,
                        huggingface_hub_model_id
                    )
                    huggingface_hub.snapshot_download(
                        huggingface_hub_model_id,
                        local_dir=os.path.join(
                            _default_insightface_models_dir(),
                            huggingface_hub_model_id
                        ),
                    )
                    _app = get_app(huggingface_hub_model_id)
                _app.prepare(
                    ctx_id=0,
                    det_thresh=0.5,
                    det_size=(640, 640),
                )

        ## Check if all models have the CUDA execution model available
        ## to them.
        for task_name, model in _app.models.items():
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

        return _app

    @property
    def embedding_size(self) -> int:
        """
        Returns the size of the embedding vector for the recognition model.
        """
        recognition_outputs = self.model.models["recognition"].session.get_outputs()[0]
        assert len(recognition_outputs.shape) == 2 and recognition_outputs.shape[0] == 1
        assert recognition_outputs.type == "tensor(float)"
        return recognition_outputs.shape[-1]

    @torch.inference_mode()
    def get_image_features(self, **kwargs):
        image = kwargs.get("image", None)
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image input is required for image feature extraction.")
        faces = self.model.get(image.numpy())
        n_faces = len(faces)

        if len(faces) == 0:
            return {
                "embeddings": torch.empty(
                    (n_faces, self.embedding_size), dtype=torch.float32
                ),
                "scores": torch.empty((n_faces,), dtype=torch.float32),
                "boxes": torch.empty((n_faces, 4), dtype=torch.float32),
                "age": torch.empty((n_faces,), dtype=torch.int32),
                "is_male": torch.empty((n_faces,), dtype=torch.bool),
            }

        embeddings = np.stack([face.normed_embedding for face in faces], axis=0)
        scores = np.array([face.det_score for face in faces], dtype=np.float32)
        boxes = np.array([face.bbox for face in faces], dtype=np.float32)
        age = np.array([face.age for face in faces], dtype=np.int32)
        is_male = np.array([face.sex == "M" for face in faces], dtype=bool)

        return {
            "embeddings": torch.from_numpy(embeddings),
            "scores": torch.from_numpy(scores),
            "boxes": torch.from_numpy(boxes),
            "age": torch.from_numpy(age),
            "is_male": torch.from_numpy(is_male),
        }

    get_text_features = None  # InsightFace does not support text features
    get_audio_features = None  # InsightFace does not support audio features


class InsightFaceFeatureExtractor(FeatureExtractor):
    """Feature extractor for InsightFace.

    This feature extractor uses `InsightFace <https://insightface.ai>`
    Python package (see also, `on github
    <https://github.com/deepinsight/insightface>` and `on PYPI
    <https://pypi.org/project/insightface/>`).  It can use the models
    from InsightFace's own model zoo or compatible models from Hugging
    Face Hub.

    The `feature_id` argument is a four component string split with
    the `"/"` character.  The initial two components are used
    internally by Wise to identify this feature extractor and
    therefore `feature_id` must start with
    `"deepinsight/insightface/"`.  The remaining two components are
    used to identify the actual model used.  This feature extractor
    first tries to fetch a model with InsightFace's own model zoo
    which only need one value while the fourth element is ignored.
    For example, to use InsightFace's `buffalo_l` model, use::

        deepinsight/insightface/buffalo_l/_  # the last element is ignored

    If that fails, it uses both remaining components to download model
    weights from Hugging Face Hub.  For example, to use AuraFace-v1
    from FAL, use::

        deepinsight/insightface/fal/AuraFace-v1

    """

    ## InsightFace supports image only (no audio and no text)
    preprocess_text = None
    extract_text_features = None
    preprocess_audio = None
    extract_audio_features = None

    _vector_metadata_table = sa.Table(
        "vector_metadata_insightface",
        project_metadata_obj,
        sa.Column(
            "vector_id",
            sa.Integer,
            sa.ForeignKey("vectors.id", ondelete="cascade"),
            nullable=False,
            index=True,
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

    def __init__(
        self,
        feature_id: str,
        *,
        warmup: bool = False,
        **kwargs,
    ):
        _logger.info("initialising feature extractor for %s", feature_id)
        feature_id_parts = feature_id.split("/")
        assert (
            len(feature_id_parts) == 4
            and feature_id_parts[0] == "deepinsight"
            and feature_id_parts[1] == "insightface"
        ), f"Invalid feature-id: {feature_id}, an example of a valid feature-id is 'deepinsight/insightface/buffalo_l/_unknown'"
        self.model_name = "/".join(feature_id_parts[2:])

        self._embedding_dtype = np.float32

        if warmup:
            self.warmup()

    @classmethod
    def create_vector_metadata_table(cls, db_engine: sa.Engine) -> None:
        cls._vector_metadata_table.create(db_engine, checkfirst=True)

    @classmethod
    def add_to_vector_metadata_table(
        cls, conn:sa.Connection, vid: list[int], metadata: list[FaceFeatureMetadata]
    ) -> None:
        ## This condition is needed because insert with an empty list
        ## does an insert with NULLs instead of "nothing".  See
        ## https://github.com/sqlalchemy/sqlalchemy/discussions/9645
        if len(metadata) > 0:
            conn.execute(
                sa.insert(cls._vector_metadata_table),
                [x.to_sql_values(vid) for vid, x in zip(vid, metadata)],
            )

    @classmethod
    def get_vector_metadata(
        cls, conn: sa.Connection, vid: list[int]
    ) -> list[FeatureExtMetadata]:
        c = cls._vector_metadata_table.c
        res = conn.execute(
            sa.select(
                c.detection_score,
                c.bbox_x,
                c.bbox_y,
                c.bbox_w,
                c.bbox_h,
                c.age,
                c.is_male,
            )
            .where(c.vector_id.in_(vid))
            .order_by(sa.case({x: i for i, x in enumerate(vid)}, value=c.vector_id))
        )
        res = [FaceFeatureMetadata.from_sql_values(x) for x in res.mappings()]
        assert len(vid) == len(res)
        return res

    def preprocess_image(
        self, images: torch.Tensor | list[PIL.Image.Image]
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

    @cached_property
    def model(self) -> InsightFaceModel:
        """
        Returns the InsightFace model instance.
        """
        return InsightFaceModel(
            self.model_name,
        )

    def extract_image_features(self, images: torch.Tensor) -> list[Features]:
        _logger.debug("extracting image features from a %s", type(images))
        features: list[Features] = []
        for image in images:
            ## NB: undocumented but `get()` expects a numpy ndarray,
            ##     of shape `(H, W, C)`, and mode BGR.  The mode seems
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
            outputs = self.model.get_image_features(image=image)
            response = FaceInferenceResponse.from_tensor(**outputs)
            faces = response.to_app_face()
            _logger.debug("found %d faces", len(faces))

            feature_vectors = response.embeddings
            feature_metadata: list[FaceFeatureMetadata] = []
            for i, face in enumerate(faces):
                feature_metadata.append(
                    FaceFeatureMetadata.from_app_face(face, image)
                )
            features.append(
                Features(vectors=feature_vectors, metadata=feature_metadata)
            )

        return features


    def preprocess_image_region(
        self, image: torch.Tensor | PIL.Image.Image, region: BBoxXYWH
    ) -> torch.Tensor:
        return self._preprocess_image_region_nocrop(image, region)

    def extract_image_region_features(
        self, image: torch.Tensor, region: BBoxXYWH
    ) -> Features:
        return self._extract_image_region_features_highest_iou(image, region)


    def warmup(self):
        random_image = torch.rand((1, 3, 768, 1024))
        features = self.extract_image_features(self.preprocess_image(random_image))
        assert features[0].vectors.shape[1] == 512
        return
