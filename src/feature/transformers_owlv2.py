## Copyright (C) 2025 University of Oxford

## Parts of the code on this file are based on the code for the
## transformers.models.owlv2.modelling_owlv2 module from the
## transformers Python package distribution version 4.44.2 (see
## comments on the code below for more details).  The original code
## had the following notice:
##
## Copyright 2023 Google AI and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from functools import cached_property
import logging
from pathlib import Path
from typing import Any
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from typing import Union
from PIL import Image
import sqlalchemy as sa

from .feature_extractor import (
    BBoxXYWH,
    FeatureExtMetadata,
    FeatureExtractor,
    Features,
    get_torch_device,
    MultiModalModel,
)
from ..db import project_metadata_obj

logger = logging.getLogger(__name__)

def flatten_patch_features(feature_map: torch.Tensor) -> torch.Tensor:
    assert feature_map.ndim == 4
    batch_sz, num_patches_h, num_patches_w, hidden_dim = feature_map.shape
    return feature_map.reshape((batch_sz, num_patches_h * num_patches_w, hidden_dim))


def sort_by_objectness(objectness_scores, embeds, pred_boxes):
    assert (
        objectness_scores.ndim == 2
        and embeds.ndim == 3
        and pred_boxes.ndim == 3
        and objectness_scores.shape[-1] == embeds.shape[-2]
        and objectness_scores.shape[-1] == pred_boxes.shape[-2]
        and objectness_scores.shape[-2] == embeds.shape[-3]
        and objectness_scores.shape[-2] == pred_boxes.shape[-3]
    )
    sort_idx = torch.argsort(objectness_scores, descending=True)
    objectness_scores = torch.take_along_dim(
        objectness_scores, sort_idx, dim=1
    )
    embeds = torch.take_along_dim(
        embeds, sort_idx.unsqueeze(-1).expand_as(embeds), dim=1
    )
    pred_boxes = torch.take_along_dim(
        pred_boxes, sort_idx.unsqueeze(-1).expand_as(pred_boxes), dim=1
    )
    return (objectness_scores, embeds, pred_boxes)


def owlv2_bbox_to_xywh(
        owlv2_bbox: np.ndarray, im_width: int, im_height: int
    ) -> np.ndarray:
    """Convert bbox coordinates from OWLv2 format to XYWH.

    The original bounding box coordinates from OWLv2 are normalized
    between 0 and 1, based on a *square-padded* version of the input
    image, and in the (x_center, y_center, width, height) format.

    This function converts those coordinates to the (x0, y0, w, h)
    format relative to the original image size.
    """
    x_center, y_center, width, height = owlv2_bbox

    # Get coordinates of top left corner
    x0 = x_center - width/2
    y0 = y_center - height/2

    # Adjust bounding box coordinates to account for square padding applied
    # to the input image
    if im_width > im_height:
        y0 *= im_width/im_height
        height *= im_width/im_height
    else:
        x0 *= im_height/im_width
        width *= im_height/im_width
    return np.array([x0, y0, width, height])


def get_object_features(model: Owlv2ForObjectDetection, images: torch.Tensor):

    # --- Code below is adapted from Owlv2ForObjectDetection.forward() source code ---
    batch_feature_map, _ = model.image_embedder(
        images
    )  # shape of batch_feature_map: (B, 60, 60, 768)
    batch_image_feats = flatten_patch_features(
        batch_feature_map
    )  # shape: (B, 3600, 768)

    _, batch_image_class_embeds = model.class_predictor(batch_image_feats)
    # Normalize image features
    # shape of batch_image_class_embeds: (B, 3600, 512)
    batch_image_class_embeds = batch_image_class_embeds / (
        torch.linalg.norm(batch_image_class_embeds, dim=-1, keepdim=True) + 1e-6
    )

    # Apply a learnable shift and scale to logits
    batch_logit_shift = model.class_head.logit_shift(
        batch_image_feats
    )  # shape: (B, 3600, 1)
    batch_logit_scale = model.class_head.logit_scale(batch_image_feats)
    batch_logit_scale = (
        model.class_head.elu(batch_logit_scale) + 1
    )  # shape: (B, 3600, 1)

    # Augment image embeddings
    batch_image_class_embeds = (
        torch.concat([batch_image_class_embeds, batch_logit_shift], axis=-1)
        * batch_logit_scale
    )  # shape: (B, 3600, 513)

    # Predict objectness
    batch_objectness_logits = model.objectness_predictor(batch_image_feats)
    batch_objectness_scores = torch.sigmoid(batch_objectness_logits)  # shape: (B, 3600)

    # Predict object boxes
    batch_pred_boxes = model.box_predictor(
        batch_image_feats, batch_feature_map
    )  # shape: (B, 3600, 4)

    return (
        batch_objectness_scores,
        batch_image_class_embeds,
        batch_pred_boxes,
    )


def get_text_embeddings(
    model: Owlv2ForObjectDetection, input_ids, attention_mask
) -> torch.Tensor:
    # --- Code below is based on Owlv2Model.forward() source code ---
    text_outputs = model.owlv2.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    text_embeds = text_outputs[1]  # shape: (N, 512)
    text_embeds = model.owlv2.text_projection(text_embeds)  # shape: (N, 512)
    # normalized features
    text_embeds = text_embeds / (
        torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
    )

    text_embeds = torch.concat(
        [text_embeds, torch.ones_like(text_embeds[..., :1])], axis=-1
    )  # shape: (N, 513)

    return text_embeds


@dataclass(kw_only=True)
class OWLv2FeatureMetadata(FeatureExtMetadata):
    bbox: BBoxXYWH
    objectness_score: float

    @classmethod
    def from_owlv2(cls, owlv2_bbox: np.ndarray, objectness_score: np.floating, im_width: int, im_height: int):
        """
        Creates a OWLv2FeatureMetadata instance with the bounding box and
        objectness score of a single object (patch) detected by OWLv2. The
        bounding box coordinates are also converted as follows:
        - The original bounding box coordinates from OWLv2 are normalized
          between 0 and 1, based on a *square-padded* version of the input
          image
        - This method converts the coordinates so that they are normalized
          relative to the original image dimensions instead of the
          square-padded image
        - This method also converts the coordinates from the
          (x_center, y_center, width, height) format returned by OWLv2, to the
          (x0, y0, width, height) format used by the frontend

        Parameters
        ----------
        owlv2_bbox : np.ndarray
            A numpy array with 4 elements, representing the predicted box
            coordinates for a given object (image patch), in the form
            (x_center, y_center, width, height) normalized by
            max(width, height) to be between 0 and 1.
        objectness_score : float
            The objectness score between 0 and 1 for the given object
        im_width : int
            The original image width (before square padding and resizing)
        im_height : int
            The original image height (before square padding and resizing)
        """
        x0, y0, width, height = owlv2_bbox_to_xywh(
            owlv2_bbox, im_width, im_height
        )
        return cls(
            objectness_score=objectness_score.item(),
            bbox=BBoxXYWH(x0.item(), y0.item(), width.item(), height.item()),
        )

    @classmethod
    def from_sql_values(cls, row: dict):
        return cls(
            objectness_score=row["objectness_score"],
            bbox=BBoxXYWH(
                row["bbox_x"],
                row["bbox_y"],
                row["bbox_w"],
                row["bbox_h"],
            ),
        )

    def to_sql_values(self, vector_id: int):
        return {
            "vector_id": vector_id,
            "objectness_score" : self.objectness_score,
            "bbox_x": self.bbox.x,
            "bbox_y": self.bbox.y,
            "bbox_w": self.bbox.w,
            "bbox_h": self.bbox.h,
        }


class TransformersOWLv2Model(MultiModalModel):
    """
    A MultiModalModel wrapper for the OWLv2 model from HuggingFace Transformers.
    This class is used to load the OWLv2 model and perform inference on it.
    """ 
    @cached_property
    def model(self) -> Owlv2ForObjectDetection:
        """
        Returns the OWLv2ForObjectDetection model instance.
        """
        logger.info(f"Initialising OWLv2 model {self.model_id} on device {self.DEVICE}")
        _model = Owlv2ForObjectDetection.from_pretrained(
            self.model_id, **self.model_kwargs
        ).to(self.DEVICE)
        _model.eval()
        if self.compile:
            available_backends = torch._dynamo.list_backends()
            backend = "inductor"
            if "tensorrt" in available_backends:
                backend = "tensorrt"
            logger.info(f"Compiling model with backend {backend}")
            _model.compile(mode="reduce-overhead", backend=backend)
        return _model

    def get_image_features(self, **kwargs):
        images = kwargs.get("images", None)
        if not isinstance(images, torch.Tensor):
            raise ValueError(
                "Images tensor input is required for image feature extraction."
            )

        scores, embeddings, boxes = get_object_features(
            self.model, images.to(self.DEVICE)
        )

        return {
            "scores": scores.float(),
            "embeddings": embeddings.float(),
            "boxes": boxes.float(),
        }

    def get_text_features(self, **kwargs) -> torch.Tensor:
        """
        Extract text features from the OWLv2 model.
        """
        input_ids = kwargs.get("input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if input_ids is None or attention_mask is None:
            raise ValueError(
                "Input IDs and attention mask are required for text feature extraction."
            )
        return get_text_embeddings(
            self.model,
            input_ids=input_ids.to(self.DEVICE),
            attention_mask=attention_mask.to(self.DEVICE),
        ).float()

    get_audio_features = None  # OWLv2 does not support audio features

    def export_to_onnx(
        self, save_path: str, visual_inputs: tuple, text_inputs: tuple, **kwargs
    ):
        """
        Export the OWLv2 model to ONNX format.
        This method is not implemented for OWLv2 as it is not straightforward.
        """

        model = self.model
        model.eval()

        logger.info(f"Exporting vision model...")
        output_path = Path(f"{save_path}--image")
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "model.onnx"

        class CustomOWLv2VisionModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images):
                return get_object_features(self.model, images)

        custom_model = CustomOWLv2VisionModel(model)
        custom_model.eval()
        # custom_model(*visual_inputs)

        # Dynamo export has some issues - cannot set batch size to dynamic
        # see https://github.com/pytorch/pytorch/issues/122321
        batch_size = torch.export.Dim("batch_size")
        dynamic_shapes = {
            "images": {0: batch_size},
        }

        torch.onnx.export(
            custom_model,
            visual_inputs,
            output_path,
            input_names=["images"],
            output_names=["scores", "embeddings", "boxes"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            verify=True,
            do_constant_folding=True,
            opset_version=20,
            report=True,
        )

        logger.info(f"Successfully exported vision model to {output_path}")

        class CustomOWLv2TextModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask):
                return get_text_embeddings(model, input_ids, attention_mask)

        custom_text_model = CustomOWLv2TextModel(model)
        custom_text_model.eval()
        # custom_text_model(*text_inputs)

        logger.info(f"Exporting text model")
        output_path = Path(f"{save_path}--text")
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "model.onnx"

        batch_size = torch.export.Dim("batch_size")
        dynamic_shapes = {
            "input_ids": {0: batch_size},
            "attention_mask": {0: batch_size},
        }
        torch.onnx.export(
            custom_text_model,
            text_inputs,
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_shapes=dynamic_shapes,
            do_constant_folding=True,
            opset_version=20,
            dynamo=True,
            verify=True,
        )
        logger.info(f"Successfully exported text model to {output_path}")


class TransformersOWLv2FeatureExtractor(FeatureExtractor):
    """
    Feature extractor based on the HuggingFace Transformers implementation of
    OWLv2 (from Google) for open-vocabulary object detection
    https://huggingface.co/docs/transformers/en/model_doc/owlv2

    see FeatureExtractor.py for documentation of API
    """

    ID_PREFIX = 'transformers/owlv2/'
    DESCRIPTION = 'See https://huggingface.co/docs/transformers/en/model_doc/owlv2'

    ## OWLv2 does not support audio
    preprocess_audio = None
    extract_audio_features = None

    _vector_metadata_table = sa.Table(
        "vector_metadata_owlv2",
        project_metadata_obj,
        sa.Column(
            "vector_id",
            sa.Integer,
            sa.ForeignKey("vectors.id", ondelete="cascade"),
            nullable=False,
            index=True,
        ),
        sa.Column("objectness_score", sa.Float, nullable=False),
        ## we store normalized (x0, y0, w, h) coordinates
        ## because that's what the frontend uses.
        sa.Column("bbox_x", sa.Float, nullable=False),  # [0.0-1.0]
        sa.Column("bbox_y", sa.Float, nullable=False),  # [0.0-1.0]
        sa.Column("bbox_w", sa.Float, nullable=False),  # [0.0-1.0]
        sa.Column("bbox_h", sa.Float, nullable=False),  # [0.0-1.0]
        keep_existing=True,
    )

    class Config(FeatureExtractor.Config):
        objectness_threshold: float = 0.02
        preprocessor_kwargs: dict[str, Any] = {}
        model_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        id: str,
        *,
        device: str | torch.device | None = None,
        warmup: bool = False,
        config: Config = Config(),
        **kwargs,
    ):
        """
        Parameters
        ----------
        id : str
            Set this to `transformers/owlv2/google/` followed by one of the following:
            - `owlv2-base-patch16`
            - `owlv2-base-patch16-finetuned`
            - `owlv2-base-patch16-ensemble`
            - `owlv2-large-patch14`
            - `owlv2-large-patch14-finetuned`
            - `owlv2-large-patch14-ensemble`

            Example: `transformers/owlv2/google/owlv2-base-patch16-ensemble`
        objectness_threshold : float
            During feature extraction, all objects with an objectness score
            below this threshold are discarded to limit the number of feature
            vectors that need to be stored and indexed
        """
        if not id.startswith(self.ID_PREFIX):
            raise ValueError(f'feature id cannot start with {id} and must start with {self.ID_PREFIX}')
        id_tokens = id.split('/')

        assert len(id_tokens) == 4
        self.model_name = id[len(self.ID_PREFIX):] # remove ID_PREFIX from id string

        self.DEVICE = get_torch_device(device)
        self.objectness_threshold = config.objectness_threshold
        self.model_kwargs = config.model_kwargs
        self.compile = config.compile

        if warmup:
            self.warmup()

    @cached_property
    def model(self):
        model = TransformersOWLv2Model(
            model_id=self.model_name,
            device=self.DEVICE,
            pretraining_dataset=None,  # OWLv2 does not use pretraining dataset
            compile=self.compile,
            **self.model_kwargs,
        )
        return model

    @cached_property
    def processor(self):
        return Owlv2Processor.from_pretrained(self.model_name)

    @classmethod
    def create_vector_metadata_table(cls, db_engine: sa.Engine) -> None:
        cls._vector_metadata_table.create(bind=db_engine, checkfirst=True)

    @classmethod
    def add_to_vector_metadata_table(
        cls, conn:sa.Connection, vid: list[int], metadata: list[OWLv2FeatureMetadata]
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
            sa.select(c.objectness_score, c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h)
            .where(c.vector_id.in_(vid))
            .order_by(sa.case({x: i for i, x in enumerate(vid)}, value=c.vector_id))
        )
        res = [OWLv2FeatureMetadata.from_sql_values(x) for x in res.mappings()]
        assert len(vid) == len(res)
        return res

    def preprocess_image(self, images: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError("expect tensor images to be RGB in NCHW order")
        elif isinstance(images, list):
            if not all([isinstance(x, Image.Image) for x in images]):
                raise TypeError("expect list images to all be PIL Image")
            images = torch.stack([pil_to_tensor(img) for img in images])
        else:
            raise TypeError("unexpected input images of type %s" % type(images))
        return images

    @torch.inference_mode()
    def extract_image_features(
        self,
        images: torch.Tensor,
    ) -> list[Features]:
        """
        Extract features/embeddings, objectness scores, and box coordinates for each patch in
        a batch of input images

        Parameters
        ----------
        images : torch.Tensor
            A batch of preprocessed images with shape (B, H, W, C) where B denotes the
            batch size and H, W, C denote the height, width, and number of channels respectively
        return_augmented_features : bool, optional
            Whether to return 'augmented' feature vectors or not (True by default)

            In OWL-ViT and OWLv2, there is a learnable shift and scale value for each
            patch token that is applied after computing the dot products between
            the image patch tokens and the text embeddings (see link below):
            https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/models/owlv2/modeling_owlv2.py#L1269-L1273

            This makes it difficult to implement fast lookups with vector databases
            or approximate nearest neighbour search algorithms. Fortunately, there
            is a solution to this which involves augmenting the vectors as described
            in the links below:
            https://github.com/google-research/scenic/issues/831

            https://github.com/google-research/scenic/pull/891/files

        Returns
        -------
        list of Features
            A list of `Features` objects, one for each image in the input batch.
            Each `Features` object contains these two attributes:

            - vectors: the patch features/embeddings computed by OWLv2 for a
              given image. This is a numpy array of shape (P, D) where P is the
              number of image patches (3600 by default), and D is the embedding
              dimension (default 513). If `return_augmented_features = False`
              then the last dimension is 512.
            - metadata: a list of `OWLv2FeatureMetadata` objects containing the
              predicted bounding box coordinates and objectness score for each
              patch in a given image.
        """
        if not isinstance(images, torch.Tensor):
            raise ValueError('input to extract_features() must be an instance of torch.Tensor')

        # Save original image sizes in a list before they get resized
        orig_sizes = [(image.shape[2], image.shape[1]) for image in images] # list of (width, height) tuples
        # Preprocess image (including resizing)
        images = self.processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]  # shape: (B, C, 960, 960)

        outputs = self.model.get_image_features(images=images)
        batch_objectness_scores = outputs["scores"]
        batch_image_class_embeds = outputs["embeddings"]
        batch_pred_boxes = outputs["boxes"]

        # Having the feature for the "most object" first is important for
        # searching with images since only the first feature is used.
        batch_objectness_scores, batch_image_class_embeds, batch_pred_boxes = (
            sort_by_objectness(
                batch_objectness_scores,
                batch_image_class_embeds,
                batch_pred_boxes,
            )
        )

        # Convert to numpy
        batch_image_class_embeds = batch_image_class_embeds.cpu().numpy()
        batch_objectness_scores = batch_objectness_scores.cpu().numpy()
        batch_pred_boxes = batch_pred_boxes.cpu().numpy()

        return self.construct_features(
            batch_image_class_embeds,
            batch_objectness_scores,
            batch_pred_boxes,
            orig_sizes,
        )

    def construct_features(
        self,
        batch_image_class_embeds: np.ndarray,
        batch_objectness_scores: np.ndarray,
        batch_pred_boxes: np.ndarray,
        orig_sizes: list[tuple[int, int]],
    ) -> list[Features]:
        features: list[Features] = []
        for (
            patchwise_image_class_embeds,
            patchwise_objectness_scores,
            patchwise_pred_boxes,
            orig_size,
        ) in zip(batch_image_class_embeds, batch_objectness_scores, batch_pred_boxes, orig_sizes):
            # Filter boxes by objectness score
            remaining_mask = patchwise_objectness_scores >= self.objectness_threshold
            patchwise_image_class_embeds = patchwise_image_class_embeds[remaining_mask]
            patchwise_objectness_scores = patchwise_objectness_scores[remaining_mask]
            patchwise_pred_boxes = patchwise_pred_boxes[remaining_mask]

            feature_metadata = [
                OWLv2FeatureMetadata.from_owlv2(pred_box, objectness_score, *orig_size)
                for pred_box, objectness_score in zip(patchwise_pred_boxes, patchwise_objectness_scores)
            ]
            features.append(
                Features(vectors=patchwise_image_class_embeds, metadata=feature_metadata)
            )
        return features

    def preprocess_text(self, text_query: list[str]) -> dict:
        """
        Preprocess text queries for the OWLv2 model.

        Parameters
        ----------
        text_query : list of str
            A list of text queries to preprocess

        Returns
        -------
        dict
            A dictionary containing the preprocessed text inputs ready for the model.
        """
        return self.processor(text=text_query, return_tensors="pt")

    @torch.inference_mode()
    def extract_text_features(
        self,
        text_query: list[str],
        return_augmented_features: bool = True
    ) -> np.ndarray:
        """
        Extract text features/embeddings for a list of text queries

        Parameters
        ----------
        text_query : list of str
            A list of text queries of objects to search for
        return_augmented_features : bool, optional
            Whether to return 'augmented' feature vectors or not (True by default)

            In OWL-ViT and OWLv2, there is a learnable shift and scale value for each
            patch token that is applied after computing the dot products between
            the image patch tokens and the text embeddings (see link below):
            https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/models/owlv2/modeling_owlv2.py#L1269-L1273

            This makes it difficult to implement fast lookups with vector databases
            or approximate nearest neighbour search algorithms. Fortunately, there
            is a solution to this which involves augmenting the vectors as described
            in the links below:
            https://github.com/google-research/scenic/issues/831

            https://github.com/google-research/scenic/pull/891/files

        Returns
        -------
        np.ndarray
            The features/embeddings of the text queries computed with the OWLv2 text
            encoder. Shape is (N, D) where N is the number of text queries and D is
            the embedding dimension (default 513). If
            `return_augmented_features = False` then the last dimension is 512.
        """
        inputs = self.preprocess_text(text_query)
        text_embeds = self.model.get_text_features(**inputs)
        return text_embeds.cpu().numpy() # shape: (N, 513)

    def transform_internal_image_queries_hook(self, vec: np.ndarray) -> np.ndarray:
        ## Change the vector augmentation of the internal feature vector

        # Un-augment feature vector
        # Original shape of vec: (N, 769)
        vec = vec[:, :-1]  # Remove extra logit_shift element (new shape: (N, 768))
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)  # Normalize vector to undo logit_scale

        # Re-augment
        vec = np.concatenate(
            [vec, np.ones_like(vec[..., :1])], axis=-1
        )  # Shape: (N, 769)
        return vec

    def transform_faiss_distances_hook(self, dist: np.ndarray) -> np.ndarray:
        # Apply sigmoid transformation to the logits (dot products) from Faiss.
        return 1. / (1. + np.exp(-dist))

    def warmup(self):
        random_image = torch.rand((1, 3, 512, 512))
        image_features = self.extract_image_features(
            self.preprocess_image(random_image)
        )
        text_features = self.extract_text_features(["some random text"])
        assert image_features[0].vectors.shape[1] == text_features.shape[1]
        return
