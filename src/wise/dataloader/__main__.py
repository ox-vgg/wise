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
import itertools
import logging
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.utils.data as torch_data
import torchvision.transforms.v2 as transforms_v2
import typer
from tqdm import tqdm

from wise.dataloader.dataset import get_dataset, get_metadata_for_valid_files
from wise.dataloader.utils import get_files_from_directory_with_extensions

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


CLIPModel = Enum(
    "CLIPModel", {x: x for x in AVAILABLE_MODELS} | {"None": None}
)


def get_input_transform_for_model(clip_model: CLIPModel):
    if clip_model is CLIPModel.internvideo:
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


def _preprocess(
    preprocess_fn: Callable, x: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    return torch.stack([preprocess_fn(xi) for xi in x])


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    app = typer.Typer()
    app_state = {"verbose": True}

    @app.callback()
    def base(verbose: bool = False):
        """
        Dataloader demo app

        Usage: python3 -m src INPUT_FILE --model CLIP_MODEL
        TODO: Fill this
        """
        app_state["verbose"] = verbose
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
        )
        global logger
        logger = logging.getLogger()

    @app.command()
    def run(
        media_dir_list: list[Path] = typer.Argument(
            ...,
            file_okay=False,
            dir_okay=True,
            exists=True,
            readable=True,
            help="Path to input folder of media files",
        ),
        media_include: list[str] = typer.Option(
            default=["*"],
            help="regular expression to include certain media files",
        ),
        model: CLIPModel = typer.Option(
            "ViT-B-32:openai",
            help="Pass in a open_clip model string (or) internvideo",
        ),
        thumbnails: bool = typer.Option(
            True, help="Flag to control thumbnail extraction"
        ),
        num_workers: int | None = typer.Option(
            None, help="number of data loading workers"
        ),
    ):
        """
        Dummy CLI to test the dataloader

        Example:
        # Loading one video based on clip preprocessing
        python3 -m wise.dataloder run data/Shazam.mkv --model "ViT-L-14:openai"

        # With a directory of videos
        python3 -m wise.dataloder run data/ --model "ViT-L-14:openai"

        # With internvideo
        python3 -m wise.dataloder run data/ --model "internvideo"
        """

        # Define output stream options based on model.
        # Every 0.5 seconds, we read 8 frames chunk for internvideo, and 1 for clip

        ## If `--model None`, then typer assigns model the None value
        ## instead of the None enum, so do it here.
        if model is None:
            model = CLIPModel(model)

        audio_sampling_rate = 48_000  # (48 kHz)

        video_frame_rate = 2  # fps
        video_frames_per_chunk = 8  # frames
        segment_length = (
            video_frames_per_chunk / video_frame_rate
        )  # frames / fps = seconds

        # If this is not an integer, may cause drift?
        audio_frames_per_chunk = int(
            round(audio_sampling_rate * segment_length)
        )  # audio frames spanning the same segment length as video

        # get preprocessing function from feature extractor
        logger.debug("Getting preprocessing function")

        frame_preprocess = None
        if model is not CLIPModel(None):
            preprocess = get_input_transform_for_model(model)
            if model is not CLIPModel.internvideo:
                preprocess = functools.partial(_preprocess, preprocess)
            frame_preprocess = {model: preprocess}

        params = {
            "video_frames_per_chunk": video_frames_per_chunk,
            "video_frame_rate": video_frame_rate,
            "video_preprocessing_function_map": frame_preprocess,
            "audio_samples_per_chunk": audio_frames_per_chunk,
            "audio_sampling_rate": audio_sampling_rate,
            "audio_preprocessing_function_map": None,
            "image_preprocessing_function_map": frame_preprocess,
            "offset": None,
            "thumbnails": thumbnails,
        }

        # Get metadata to write into the media table
        input_files = list(
            itertools.chain.from_iterable(
                get_files_from_directory_with_extensions(
                    media_dir, media_include
                )
                for media_dir in media_dir_list
            )
        )
        metadata, _ = get_metadata_for_valid_files(input_files)
        stream = torch_data.ChainDataset(get_dataset(metadata, params))

        # Construct the dataloader
        num_workers = max(0, num_workers or 0)
        loader = torch_data.DataLoader(
            stream,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers else None,
        )
        logger.info(f"Iterating over {len(metadata)} file(s)")
        for mid, chunks in tqdm(loader):

            for media_chunk_type, chunk in chunks.items():
                if chunk is None:
                    continue
                if isinstance(chunk, dict):
                    for feature_extractor_id in chunk:
                        if chunk[model] is None:
                            continue  # end of stream indicator
                        logger.debug(
                            [
                                {
                                    media_chunk_type: (
                                        (
                                            f"List length: {len(chunk[model].tensor)} | Shapes: {[t.shape for t in chunk[feature_extractor_id].tensor]}"
                                            if isinstance(
                                                chunk[
                                                    feature_extractor_id
                                                ].tensor,
                                                list,
                                            )
                                            else chunk[
                                                feature_extractor_id
                                            ].tensor.shape
                                        ),
                                        chunk[feature_extractor_id].pts,
                                    )
                                }
                            ]
                        )
                else:
                    logger.debug(
                        [
                            {
                                media_chunk_type: (
                                    (
                                        f"List length: {len(chunk.tensor)} | Shapes: {[t.shape for t in chunk.tensor]}"
                                        if isinstance(chunk.tensor, list)
                                        else chunk.tensor.shape
                                    ),
                                    chunk.pts,
                                )
                            }
                        ]
                    )
            pass

    app()
