from dataclasses import dataclass
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
import numpy as np
from typing import Union
from PIL import Image
import sqlalchemy as sa

from .feature_extractor import BBoxXYWH, FeatureExtMetadata, FeatureExtractor, Features


def flatten_patch_features(feature_map: torch.Tensor) -> torch.Tensor:
    assert feature_map.ndim == 4
    batch_sz, num_patches_h, num_patches_w, hidden_dim = feature_map.shape
    return feature_map.reshape((batch_sz, num_patches_h * num_patches_w, hidden_dim))


class ImageBatchTensorWithOrigSizes(torch.Tensor):
    """
    Subclass of torch.Tensor representing a preprocessed (resized) batch of
    images, with a custom attribute `orig_sizes` storing the original image
    sizes as a list of (width, height) tuples
    """
    orig_sizes: list[tuple[int, int]]

    def __new__(cls, data, orig_sizes: list[tuple[int, int]]):
        # Create a new tensor instance
        obj = torch.as_tensor(data).as_subclass(cls)
        # Set custom attribute
        obj.orig_sizes = orig_sizes
        return obj

    def __repr__(self):
        return f"ImageBatchTensorWithOrigSizes(data={super().__repr__()}, orig_sizes={self.orig_sizes})"

    # Override `clone()` method to preserve orig_sizes
    def clone(self, *args, **kwargs):
        cloned = super().clone(*args, **kwargs).as_subclass(ImageBatchTensorWithOrigSizes)
        cloned.orig_sizes = self.orig_sizes
        return cloned

    # Override `to()` method to preserve orig_sizes
    def to(self, *args, **kwargs):
        new_obj = super().to(*args, **kwargs)
        if new_obj is self:
            return self
        return ImageBatchTensorWithOrigSizes(new_obj, self.orig_sizes)

@dataclass
class OWLv2FeatureMetadata:
    objectness_score: float
    bbox: BBoxXYWH

    @classmethod
    def from_owlv2(cls, owlv2_bbox: np.ndarray, objectness_score: float, im_width: int, im_height: int):
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

        return cls(
            objectness_score=objectness_score,
            bbox=BBoxXYWH(x0, y0, width, height),
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


class TransformersOWLv2(FeatureExtractor):
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

    def __init__(self, id: str, objectness_threshold=0.02):
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
        model_name = id[len(self.ID_PREFIX):] # remove ID_PREFIX from id string

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Owlv2ForObjectDetection = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.DEVICE)
        self.processor = Owlv2Processor.from_pretrained(model_name)

        self.model.eval()

        self.objectness_threshold = objectness_threshold

        # get input image size
        self.input_image_size = (self.processor.image_processor.size['height'], self.processor.image_processor.size['width'])

    def get_input_image_size(self):
        return self.input_image_size

    def create_vector_metadata_table(self, db_engine: sa.Engine) -> None:
        db_metadata_obj = sa.MetaData()
        db_metadata_obj.reflect(db_engine)
        self._vector_metadata_table = sa.Table(
            "vector_metadata_owlv2",
            db_metadata_obj,
            sa.Column(
                "vector_id",
                sa.ForeignKey("vectors.id", ondelete="cascade"),
                nullable=False,
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
        db_metadata_obj.create_all(db_engine)

    def add_to_vector_metadata_table(
        self, conn:sa.Connection, vid: list[int], metadata: list[OWLv2FeatureMetadata]
    ) -> None:
        ## This condition is needed because insert with an empty list
        ## does an insert with NULLs instead of "nothing".  See
        ## https://github.com/sqlalchemy/sqlalchemy/discussions/9645
        if len(metadata) > 0:
            conn.execute(
                sa.insert(self._vector_metadata_table),
                [x.to_sql_values(vid) for vid, x in zip(vid, metadata)],
            )

    def get_vector_metadata(
        self, conn: sa.Connection, vid: list[int]
    ) -> list[FeatureExtMetadata]:
        c = self._vector_metadata_table.c
        res = conn.execute(
            sa.select(c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h)
            .where(c.vector_id.in_(vid))
            .order_by(
                sa.case({x: i for i, x in enumerate(vid)}, value=c.vector_id)
            )
        )
        res = [FeatureExtMetadata(BBoxXYWH(*x)) for x in res]
        assert len(vid) == len(res)
        return res

    def preprocess_image(self, images: Union[torch.Tensor, list[Image.Image]]) -> ImageBatchTensorWithOrigSizes:
        # Save original image sizes in a list
        orig_sizes = [] # list of (width, height) tuples
        if isinstance(images, torch.Tensor):
            orig_sizes = [(image.shape[2], image.shape[1]) for image in images]
        elif isinstance(images, Image.Image):
            orig_sizes = [image.size for image in images]
        else:
            raise TypeError("`images` must be either a tensor or a PIL Image")

        preprocessed_images = self.processor(images=images, return_tensors="pt")['pixel_values'].to(self.DEVICE) # shape: (B, C, 960, 960)
        return ImageBatchTensorWithOrigSizes(preprocessed_images, orig_sizes)

    @torch.inference_mode()
    def extract_image_features(
        self,
        images: ImageBatchTensorWithOrigSizes,
        return_augmented_features: bool = True
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
        if not isinstance(images, ImageBatchTensorWithOrigSizes):
            raise ValueError('input to extract_features() must be an instance of ImageBatchTensorWithOrigSizes')

        # --- Code below is adapted from Owlv2ForObjectDetection.forward() source code ---
        batch_feature_map, _ = self.model.image_embedder(images) # shape of batch_feature_map: (B, 60, 60, 768)
        batch_image_feats = flatten_patch_features(batch_feature_map) # shape: (B, 3600, 768)

        _, batch_image_class_embeds = self.model.class_predictor(batch_image_feats)
        # Normalize image features
        # shape of batch_image_class_embeds: (B, 3600, 512)
        batch_image_class_embeds = batch_image_class_embeds / (torch.linalg.norm(batch_image_class_embeds, dim=-1, keepdim=True) + 1e-6)

        if return_augmented_features:
            # Apply a learnable shift and scale to logits
            batch_logit_shift = self.model.class_head.logit_shift(batch_image_feats) # shape: (B, 3600, 1)
            batch_logit_scale = self.model.class_head.logit_scale(batch_image_feats)
            batch_logit_scale = self.model.class_head.elu(batch_logit_scale) + 1 # shape: (B, 3600, 1)

            # Augment image embeddings
            batch_image_class_embeds = (
                torch.concat([batch_image_class_embeds, batch_logit_shift], axis=-1) * batch_logit_scale
            ) # shape: (B, 3600, 513)

        # Predict objectness
        batch_objectness_logits = self.model.objectness_predictor(batch_image_feats)
        batch_objectness_scores = torch.sigmoid(batch_objectness_logits) # shape: (B, 3600)

        # Predict object boxes
        batch_pred_boxes = self.model.box_predictor(batch_image_feats, batch_feature_map) # shape: (B, 3600, 4)

        # Convert to numpy
        batch_image_class_embeds = batch_image_class_embeds.cpu().numpy()
        batch_objectness_scores = batch_objectness_scores.cpu().numpy()
        batch_pred_boxes = batch_pred_boxes.cpu().numpy()

        features: list[Features] = []
        for (
            patchwise_image_class_embeds,
            patchwise_objectness_scores,
            patchwise_pred_boxes,
            orig_size,
        ) in zip(batch_image_class_embeds, batch_objectness_scores, batch_pred_boxes, images.orig_sizes):
            # Filter boxes by objectness score
            remaining_indices = patchwise_objectness_scores >= self.objectness_threshold
            patchwise_image_class_embeds = patchwise_image_class_embeds[remaining_indices]
            patchwise_objectness_scores = patchwise_objectness_scores[remaining_indices]
            patchwise_pred_boxes = patchwise_pred_boxes[remaining_indices]

            feature_metadata = [
                OWLv2FeatureMetadata.from_owlv2(pred_box, objectness_score, *orig_size)
                for pred_box, objectness_score in zip(patchwise_pred_boxes, patchwise_objectness_scores)
            ]
            features.append(
                Features(vectors=patchwise_image_class_embeds, metadata=feature_metadata)
            )
        return features

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
        inputs = self.processor(text=text_query, return_tensors="pt").to(self.DEVICE)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # --- Code below is based on Owlv2Model.forward() source code ---
        text_outputs = self.model.owlv2.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs[1] # shape: (N, 512)
        text_embeds = self.model.owlv2.text_projection(text_embeds) # shape: (N, 512)
        # normalized features
        text_embeds = text_embeds / (torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True) + 1e-6)

        if return_augmented_features:
            text_embeds = torch.concat(
                [text_embeds, torch.ones_like(text_embeds[..., :1])], axis=-1
            ) # shape: (N, 513)

        return text_embeds.cpu().numpy() # shape: (N, 513)
