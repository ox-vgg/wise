#!/usr/bin/env python3

from __future__ import annotations
import logging
import functools
from pathlib import Path
import itertools
from typing import Optional

from ._extra import (
    CLIPModel,
    get_input_transform_for_model,
    _preprocess,
)
from .utils import get_files_from_directory_with_extensions
from .dataset import get_metadata_for_valid_files, get_dataset
import torch
import torch.utils.data as torch_data
import typer
from tqdm import tqdm


logger = logging.getLogger(__name__)


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
            default=["*"], help="regular expression to include certain media files"
        ),
        model: Optional[CLIPModel] = typer.Option(
            "ViT-B-32:openai", help="Pass in a open_clip model string (or) internvideo"
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
        python3 -m src run data/Shazam.mkv --model "ViT-L-14:openai"

        # With a directory of videos
        python3 -m src run data/ --model "ViT-L-14:openai"

        # With internvideo
        python3 -m src run data/ --model "internvideo"
        """

        # Define output stream options based on model.
        # Every 0.5 seconds, we read 8 frames chunk for internvideo, and 1 for clip

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
        if model != "None":
            preprocess = get_input_transform_for_model(model)
            if model != "internvideo":
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
                get_files_from_directory_with_extensions(media_dir, media_include)
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
                                                chunk[feature_extractor_id].tensor, list
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
