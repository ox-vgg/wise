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

import enum
import logging
from collections.abc import Callable

import numpy as np
import open_clip
import torch
import torchvision.transforms.v2 as transforms_v2

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = open_clip.list_pretrained(as_str=True) + ["internvideo"]

INTERNVIDEO_MEAN = np.array(
    [0.48145466, 0.4578275, 0.40821073], dtype=np.float32
)
INTERNVIDEO_STD = np.array(
    [0.26862954, 0.26130258, 0.27577711], dtype=np.float32
)


def squeeze_(x):
    return x.squeeze(0)


def unsqueeze_(x):
    return x.unsqueeze(0)


def permute_(x):
    return x.permute(1, 0, 2, 3)


def get_input_transform_for_model(clip_model):
    if clip_model == "internvideo":
        # Internvideo preprocessing
        return transforms_v2.Compose(
            [
                # Convert chunk to tensor to overcome issue with
                # .numpy() method in tensor subclasses
                transforms_v2.Resize(224),
                transforms_v2.CenterCrop(224),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(
                    mean=INTERNVIDEO_MEAN.tolist(),
                    std=INTERNVIDEO_STD.tolist(),
                ),
                # C x B x H x W
                permute_,
            ]
        )

    model_name, _ = clip_model.value.split(":", 1)
    logger.info("Loading CLIP (model: %s)...", model_name)
    model = open_clip.create_model(model_name, None)
    preprocess = open_clip.transform.image_transform_v2(
        open_clip.transform.PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return transforms_v2.Compose(
        [
            # Convert chunk to tensor to overcome issue with
            # .numpy() method in tensor subclasses
            squeeze_,
            transforms_v2.ToPILImage(),
            preprocess,
            unsqueeze_,
        ]
    )


class _CLIPModel(str, enum.Enum):
    pass


CLIPModel = _CLIPModel(
    "CLIPModel", {x: x for x in AVAILABLE_MODELS} | {"None": None}
)


def _preprocess(
    preprocess_fn: Callable, x: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    return torch.stack([preprocess_fn(xi) for xi in x])
