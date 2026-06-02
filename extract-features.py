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

import argparse
import os
import time
from pathlib import Path
import pprint

from typing import Literal
import torch
import torch.utils.data as torch_data
from tqdm import tqdm
import numpy as np
import logging

import sqlalchemy as sa
from wise.enums import BaseStrEnum
from wise.dataloader.dataset import MediaChunk
from wise.dataloader import get_dataset, get_metadata_for_valid_files, DatasetPayload
from wise.data_models import SourceMediaType, MediaChunkType
from wise.dataloader.utils import get_files_from_directory_with_extensions
from wise.wise_project import WiseProject
from wise.feature.feature_extractor import FeatureExtractor
from wise.feature.feature_extractor_factory import (
    FeatureExtractorFactory,
    get_canonical_feature_extractor_id,
)
from wise.feature.store import (
    FeatureStore,
    FeatureStoreFactory,
    FeatureStoreType,
)

from wise.data_models import (
    MediaMetadata,
    SourceCollection,
    VectorMetadata,
    ThumbnailMetadata,
    MediaType,
    ModalityType,
    SourceCollectionType,
)
from wise.repository import (
    SourceCollectionRepo,
    MediaRepo,
    VectorRepo,
    ThumbnailRepo,
)
from wise.dataloader.shot import ShotStream

from wise.config import APIConfig

class ExtractFeatureMode(BaseStrEnum):
    create = "create"
    add_feature_extractor = "add_feature_extractor"
    add_media = "add_media"

def initialise_feature_extractors(
    project: WiseProject,
    feature_extractor_ids: dict[ModalityType, list],
    feature_extractor_config: dict[str, dict],
    feature_store_type: FeatureStoreType,
    shard_max_count: int,
    db_engine: sa.Engine,
) -> tuple[
    dict[ModalityType, dict[str, FeatureExtractor]],
    dict[ModalityType, dict[str, FeatureStore]],
]:
    ## 3. Prepare for feature extraction and storage
    logger.info(f"Initialising feature extractor")

    feature_extractors = {}
    feature_stores = {}

    for modality_type, feature_extractor_id_map in feature_extractor_ids.items():
        feature_extractors[modality_type] = {}
        feature_stores[modality_type] = {}
        for feature_extractor_id in feature_extractor_id_map:
            ## 3.1 Initialise feature extractor
            logger.info(f"Initialising {feature_extractor_id} for {modality_type}")

            # Check if we already have an instance of this feature extractor (could be local / triton)
            canonical_feature_extractor_id = get_canonical_feature_extractor_id(
                feature_extractor_id
            )
            if canonical_feature_extractor_id in feature_extractors[modality_type]:
                logger.warning(
                    f"Feature extractor {feature_extractor_id} for {modality_type} already exists, re-using previous instance."
                )
                continue

            instance = FeatureExtractorFactory(
                feature_extractor_id, feature_extractor_config
            )
            feature_extractor_id = canonical_feature_extractor_id

            feature_extractors[modality_type][feature_extractor_id] = instance
            feature_extractors[modality_type][
                feature_extractor_id
            ].create_vector_metadata_table(db_engine)

            ## 3.2 Create folders to store features, metadata and search index
            project.create_features_dir(feature_extractor_id)

            ## 3.3 Initialise feature store to store features
            try:
                store = FeatureStoreFactory.load_store(
                    modality_type, project.features_dir(feature_extractor_id)
                )
            except ValueError:
                store = FeatureStoreFactory.create_store(
                    feature_store_type,
                    modality_type,
                    project.features_dir(feature_extractor_id),
                )
            store.enable_write(shard_maxcount=shard_max_count)
            feature_stores[modality_type][feature_extractor_id] = store

    return feature_extractors, feature_stores

def get_dataset_params(feature_extractors: dict[ModalityType, dict[str, FeatureExtractor]], thumbnails: bool) -> dict:
    ## dataset
    ## TODO move parameters to args / config
    audio_sampling_rate = 48_000  # (48 kHz)
    video_frame_rate = 2  # fps
    video_frames_per_chunk = 8  # frames
    segment_length = video_frames_per_chunk / video_frame_rate  # frames / fps = seconds
    audio_segment_length = segment_length  # seconds
    audio_frames_per_chunk = int(round(audio_sampling_rate * audio_segment_length))

    params = {
        "video_frames_per_chunk": (
            video_frames_per_chunk if ModalityType.VIDEO in feature_extractors else 0
        ),
        "video_frame_rate": video_frame_rate,
        "video_preprocessing_function_map": (
            {
                feature_extractor_id: feature_extractors[ModalityType.VIDEO][
                    feature_extractor_id
                ].preprocess_image
                for feature_extractor_id in feature_extractors.get(
                    ModalityType.VIDEO, {}
                )
            }
            if ModalityType.VIDEO in feature_extractors
            else None
        ),
        "audio_samples_per_chunk": (
            audio_frames_per_chunk if ModalityType.AUDIO in feature_extractors else 0
        ),
        "audio_sampling_rate": audio_sampling_rate,
        "audio_preprocessing_function_map": (
            {
                feature_extractor_id: feature_extractors[ModalityType.AUDIO][
                    feature_extractor_id
                ].preprocess_audio
                for feature_extractor_id in feature_extractors.get(
                    ModalityType.AUDIO, {}
                )
            }
            if ModalityType.AUDIO in feature_extractors
            else None
        ),
        "image_preprocessing_function_map": (
            {
                feature_extractor_id: feature_extractors[ModalityType.IMAGE][
                    feature_extractor_id
                ].preprocess_image
                for feature_extractor_id in feature_extractors.get(
                    ModalityType.IMAGE, {}
                )
            }
            if ModalityType.IMAGE in feature_extractors
            else None
        ),
        "offset": None,
        "thumbnails": thumbnails,
    }

    logger.info(f"Dataset parameters: {pprint.pformat(params)}")
    return params, segment_length

def get_dataset_stream(all_metadata: list[DatasetPayload], params: dict, use_shots: bool):
    uniform_stream = torch_data.ChainDataset(
        get_dataset(all_metadata, params)
    )
    if use_shots:
        logger.info("Extracting features from center frame of each shot in videos")
        shots = project.get_shots()
        if shots is None or len(shots) == 0:
            logger.error(
                "No shots found in the project.\nTo perform shot-based sampling of video frames for feature extraction:\n"
                "1. Generate a shots.csv text file (e.g. using TransNetV2) in a format like this:\n"
                "   id,media_id,timestamp,end_timestamp\n"
                "   1,1,0.000,1.567\n"
                "   2,1,1.600,3.533\n"
                "   ... (where, media_id is the ID of the video in the WISE project)\n"
                "2. Add it to a WISE project as follows:\n"
                "   python3 media-metadata.py import-shots ... --from-csv shots.csv"
            )
            exit(1)
        stream = ShotStream(uniform_stream, shots, params)
    else:
        logger.info("Using fixed frame sampling for feature extraction")
        stream = uniform_stream

    return stream

def get_dataloader(stream: torch.utils.data.Dataset, num_workers: int):
    logger.info(f"Initializing data loader with {num_workers} workers ...")

    prefetch_factor = None
    persistent_workers = False
    if num_workers > 0:
        prefetch_factor = 4
        persistent_workers = True

    av_data_loader = torch_data.DataLoader(
        stream,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return av_data_loader


def process_media_files(media_dir: Path, db_engine, media_files: list[Path]):
    # Get metadata and media datasets corresponding to the files
    metadata, unknown_files = get_metadata_for_valid_files(media_files)
    if len(unknown_files) > 0:
        logger.warning(
            f"Skipping {len(unknown_files)} invalid media files in directory {media_dir}'"
        )
        logger.debug("\n".join(map(str, unknown_files)))

    # Add metadata to database
    dataset_payload: list[DatasetPayload] = []
    logger.info(f"Writing metadata to database...")
    with tqdm(total=len(metadata)) as pbar, db_engine.begin() as conn:
        # Add each folder to source collection table
        data = SourceCollection(location=str(media_dir), type=SourceCollectionType.DIR)
        media_source_collection = SourceCollectionRepo.create(conn, data=data)

        for media_metadata in metadata:
            # Get metadata for each file and add it to media table
            # Get media_path relative to
            media_path = media_metadata.path
            _metadata = MediaRepo.create(
                conn,
                data=MediaMetadata(
                    source_collection_id=media_source_collection.id,
                    path=os.path.relpath(media_path, media_source_collection.location),
                    media_type=media_metadata.media_type,
                    checksum=media_metadata.md5sum,
                    size_in_bytes=os.path.getsize(media_path),
                    date_modified=os.path.getmtime(media_path),
                    format=media_metadata.format,
                    width=media_metadata.width,
                    height=media_metadata.height,
                    num_frames=media_metadata.num_frames,
                    duration=media_metadata.duration or 0,
                ),
            )
            # extra_metadata = ExtraMediaMetadata(
            #     media_id=_metadata.id,
            #     metadata={
            #         "fps": media_metadata.fps,
            #     }
            #     | media_metadata.extra,
            # )
            # MediaMetadataRepo.create(conn, data=extra_metadata)
            dataset_payload.append(
                DatasetPayload(_metadata.id, media_path, _metadata.media_type)
            )
            pbar.update(1)


    # return metadata and datasets to be chained
    return dataset_payload


def validate_args(args):
    if args.num_workers <= 0:
        args.num_workers = 0

    if args.media_files_from and args.media_include_list:
        raise ValueError(
            "--media-files-from and --media-include options are"
            " mutually exclusive"
        )
    if args.media_files_from and len(args.media_dir_list) > 1:
        raise ValueError(
            "--media-files-from option can't be used with more than"
            " one MEDIA_DIR"
        )

    # sanity check: remove duplicate entries in command line args
    if not args.media_include_list:
        setattr(args, 'media_include_list', ['*'])
    else:
        unique_media_include_list = list(set(args.media_include_list))
        setattr(args, 'media_include_list', unique_media_include_list)

    if len(args.media_dir_list) > 1:
        unique_media_dir_list = list(set(args.media_dir_list))
        if len(unique_media_dir_list) != len(args.media_dir_list):
            logger.warning(
                "Ignoring duplicated MEDIA_DIR"
            )
        setattr(args, 'media_dir_list', unique_media_dir_list)

    assert all(Path(x).is_dir() for x in args.media_dir_list), \
        "All values for media_dir_list must be directories"

    # Feature Extractor IDs
    # Set default for {image,audio,video}_feature_id_map only if the argument was not provided
    if args.video_feature_id_map is None and args.image_feature_id_map is None and args.audio_feature_id_map is None:
        args.video_feature_id_map = ["mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli"]
        args.image_feature_id_map = ["mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli"]
        args.audio_feature_id_map = ["microsoft/clap/2023/four-datasets"]
    else:
        # If any feature extractor ids are provided, do not use the default values for the missing ones
        if args.video_feature_id_map is None:
            args.video_feature_id_map = []
        if args.image_feature_id_map is None:
            args.image_feature_id_map = []
        if args.audio_feature_id_map is None:
            args.audio_feature_id_map = []

    # remove duplicate entries in feature extractor ids
    unique_video_feature_ids = list(set(args.video_feature_id_map or []))
    unique_image_feature_ids = list(set(args.image_feature_id_map or []))
    unique_audio_feature_ids = list(set(args.audio_feature_id_map or []))
    setattr(args, 'video_feature_id_map', unique_video_feature_ids)
    setattr(args, 'image_feature_id_map', unique_image_feature_ids)
    setattr(args, 'audio_feature_id_map', unique_audio_feature_ids)

    return args

def get_mode(args):
    mode = None
    if not Path(args.project_dir).exists():
        mode = ExtractFeatureMode.create
    else:
        logger.info(f"Project directory {args.project_dir} already exists.")
        if len(args.media_dir_list) == 0:
            mode = ExtractFeatureMode.add_feature_extractor
        else:
            mode = ExtractFeatureMode.add_media
    logger.debug(f"Operating in {mode} mode")
    return mode

def get_feature_extractor_ids_from_args(args):
    feature_extractor_ids: dict[ModalityType, list] = {}
    if args.video_feature_id_map:
        feature_extractor_ids[ModalityType.VIDEO] = args.video_feature_id_map
    if args.image_feature_id_map:
        feature_extractor_ids[ModalityType.IMAGE] = args.image_feature_id_map
    if args.audio_feature_id_map:
        feature_extractor_ids[ModalityType.AUDIO] = args.audio_feature_id_map
    return feature_extractor_ids

def get_feature_extractor_ids_from_project(project: WiseProject):
    project_assets = project.discover_assets()
    feature_extractor_ids: dict[ModalityType, list] = {}
    for modality_type, feature_extractor_id_list in project_assets.items():
        ## project_assets uses string for keys, convert to
        ## ModalityType (see also merge request !127).
        modality_type = ModalityType(modality_type)
        if modality_type not in [ModalityType.IMAGE, ModalityType.VIDEO, ModalityType.AUDIO]:
            continue
        feature_extractor_ids[modality_type] = list(feature_extractor_id_list.keys())
    return feature_extractor_ids

def get_feature_extractor_ids(mode: ExtractFeatureMode, project: WiseProject, args):

    feature_extractor_ids = get_feature_extractor_ids_from_args(args)

    if mode == ExtractFeatureMode.create:
        return feature_extractor_ids
    
    project_feature_extractor_ids = get_feature_extractor_ids_from_project(project)

    # Add feature extractor mode
    # Remove feature extractor ids that already exist in the project
    feature_extractor_ids_copy = feature_extractor_ids.copy()
    for (modality_type, feature_extractor_id_list) in feature_extractor_ids_copy.items():
        if modality_type not in project_feature_extractor_ids:
            # project does not have any feature extractors for this modality type
            continue

        for feature_extractor_id in feature_extractor_id_list:
            _feature_extractor_id = get_canonical_feature_extractor_id(
                feature_extractor_id
            )

            if _feature_extractor_id in project_feature_extractor_ids[modality_type]:
                logger.warning(
                    f"Feature extractor {_feature_extractor_id} for {modality_type} already exists in the project. Skipping."
                )
                feature_extractor_ids[modality_type].remove(feature_extractor_id)
        if len(feature_extractor_ids[modality_type]) == 0:
            del feature_extractor_ids[modality_type]

    if mode == ExtractFeatureMode.add_media:
        if len(feature_extractor_ids) > 0:
            logger.warning(
                "A project can only be updated with new media files using the existing feature extractors in the project. Ignoring the following feature extractor ids.\n"
                f"{pprint.pformat(feature_extractor_ids)}\n"
                "To add new feature extractors, first update the project with new media files and then call this script again without any new media files."
            )
        # Use existing feature extractor ids in the project, ignoring any provided in the command line args
        return project_feature_extractor_ids
    
    return feature_extractor_ids

def get_media_files_for_dataset(mode: ExtractFeatureMode, project: WiseProject, args):
    """Initialise internal metadata database with valid files."""
    logger.info("Initialising internal metadata database")
    all_metadata: list[DatasetPayload] = []
    if mode == ExtractFeatureMode.add_feature_extractor:
        metadata = project.get_media_files()
        all_metadata.extend(metadata)
    else:
        media_files: list[Path] = []
        if args.media_files_from:
            assert len(args.media_dir_list) == 1  # checked in validate_args
            media_dir = Path(args.media_dir_list[0])

            logger.info(f"Reading filepaths to be included from '{args.media_files_from}'")
            with open(args.media_files_from, "rt") as fh:
                ## Remove *only* the \n at the end of each line
                media_files = [media_dir / line[:-1] for line in fh if len(line) > 1]
            metadata = process_media_files(media_dir, db_engine, media_files)
            all_metadata.extend(metadata)
        else:
            for media_dir in args.media_dir_list:
                media_dir = Path(media_dir)
                media_files = sorted(
                    get_files_from_directory_with_extensions(
                        media_dir, args.media_include_list
                    )
                )
                metadata = process_media_files(media_dir, db_engine, media_files)
                all_metadata.extend(metadata)

    return all_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="extract-features",
        description="Initialise a WISE project by extractng features from images, audio and videos.",
        epilog="For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/",
    )
    parser.add_argument(
        "media_dir_list",
        nargs='*',
        help="process images and video from this folder (an existing WISE project will be updated if this is not provided)",
        default=[],
    )

    parser.add_argument(
        "--media-include",
        required=False,
        action="append",
        dest="media_include_list",
        default=[],
        type=str,
        help="regular expression to include certain media files",
    )

    ## TODO: implement alternative --media-files0-from option or a
    ## `--from0` which modifies the behaviour of --*-from options.
    parser.add_argument(
        "--media-files-from",
        required=False,
        type=str,
        help="media files to extract features from; one file path per line (relative to MEDIA_DIR)",
    )

    parser.add_argument(
        "--shard-maxcount",
        required=False,
        type=int,
        default=2048,
        help="max number of entries in each feature store shard (for faiss store, using shard-maxcount=1e6 results in 4GB files with 1024-dim features)",
    )

    parser.add_argument(
        "--num-workers",
        required=False,
        type=int,
        default=0,
        help="number of workers used by data loader",
    )

    parser.add_argument(
        "--feature-store",
        required=False,
        type=str,
        default="faiss",
        dest="feature_store_type",
        choices=sorted([x.value for x in FeatureStoreType]),
        help="extracted features are stored using this data structure",
    )

    parser.add_argument(
        "--image-feature-id",
        required=False,
        action="append",
        dest="image_feature_id_map",
        type=str,
        help="use one or more feature extractors for images",
    )

    parser.add_argument(
        "--video-feature-id",
        required=False,
        action="append",
        dest="video_feature_id_map",
        type=str,
        help="use one or more feature extractors for video frames",
    )

    parser.add_argument(
        "--audio-feature-id",
        required=False,
        action="append",
        dest="audio_feature_id_map",
        type=str,
        help="use one or more feature extractors for audio samples",
    )

    # TODO: Temporarily disabling this feature
    # Need to implement it as a parameter in the get_dataset call
    # parser.add_argument(
    #     "--skip-audio-feature-extraction",
    #     required=False,
    #     action="store_true",
    #     help="skip the extraction of audio features (for videos)"
    # )

    parser.add_argument(
        "--project-dir",
        required=True,
        type=str,
        help="folder where all project assets are stored",
    )

    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Automatically answer yes to all prompts"
    )

    parser.add_argument(
        "--use-shots",
        action="store_true",
        help="[EXPERIMENTAL] extract features only from middle frame of each shot in the video instead of sampling at fixed intervals (e.g. 2 fps)",
    )

    parser.add_argument(
        "--thumbnails", default=True, action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--enable-autocast",
        action="store_true",
        help="enable automatic mixed precision (AMP) for faster feature extraction (disabled by default as some feature extractors like MS CLAP are not compatible)",
    )

    args = parser.parse_args()
    config = APIConfig(project_dir=Path(args.project_dir), command='extract_features')

    feature_extractor_config = config.feature_extractor_config
    args = validate_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    if args.num_workers > 0:
        torch.multiprocessing.set_start_method("spawn")

    # If project doesn't exist, create it and set up the feature extractors based on input argument

    # If project exists,
    # if media_dir_list is not empty
    #  if feature extractors are provided, then ignore the extra user input with a warning.
    #  if feature extractors are not provided, then use the existing feature extractors in the project.
    # if media_dir_list is empty
    #  if feature extractors are provided, then use them if they don't exist in the project and extract features for existing media files
    #  if feature extractors are not provided, then exit with an error.

    # Check if an update of an existing project is requested
    mode = get_mode(args)
    if mode != ExtractFeatureMode.create:
        if not args.yes:
            answer = input(f'Do you want to update it? [y/N]: ')
            if answer.lower() != 'y':
                logger.info('Aborting...')
                exit(1)
        if mode == ExtractFeatureMode.add_media:
            logger.info(
                f"Updating existing project {args.project_dir} with new media files ..."
            )
        else:
            logger.info(
                f"Updating existing project {args.project_dir} with new feature extractor(s) ..."
            )
    else:
        logger.info(f"Creating new project {args.project_dir} ...")

    project = WiseProject(args.project_dir, create_project=True, db_kwargs={'echo': False}, thumbsdb_kwargs={'echo': False})
    db_engine = project.db_engine
    thumbs_engine = project.thumbsdb_engine

    start_time = time.time()

    ## Prepare a list of requested feature extractors
    feature_extractor_ids = get_feature_extractor_ids(mode, project, args)

    # Feature extractor ids can be empty in add_feautre_extractor case if the user provided feature extractor ids that already exist in the project
    if (
        len(feature_extractor_ids) == 0
        and mode == ExtractFeatureMode.add_feature_extractor
    ):
        logger.info("No new feature extractors specified. Nothing to do.")
        exit(0)

    ## 1. Initialise internal metadata database with valid files
    print('Initialising internal metadata database')
    all_metadata = get_media_files_for_dataset(mode, project, args)

    if len(all_metadata) == 0:
        logger.info("No valid media files found. Nothing to do.")
        exit(0)

    # Get the set of media types present in the input media files
    media_types_present: set[SourceMediaType] = set(x.media_type for x in all_metadata)

    # Remove feature extractor ids for modalities that are not present in the input media files
    if SourceMediaType.VIDEO not in media_types_present and SourceMediaType.AV not in media_types_present:
        feature_extractor_ids.pop(ModalityType.VIDEO, None)

    if SourceMediaType.IMAGE not in media_types_present:
        feature_extractor_ids.pop(ModalityType.IMAGE, None)

    if SourceMediaType.AUDIO not in media_types_present and SourceMediaType.AV not in media_types_present:
        feature_extractor_ids.pop(ModalityType.AUDIO, None)

    if len(feature_extractor_ids) == 0:
        if mode == ExtractFeatureMode.add_feature_extractor:
            logger.info(
                "No new feature extractors specified. Nothing to do."
            )
        else:
            logger.info(
                "No feature extractors matching the relevant modality of the media files specified. Nothing to do."
            )
        exit(0)

    feature_extractors, feature_stores = initialise_feature_extractors(
        project,
        feature_extractor_ids,
        feature_extractor_config,
        FeatureStoreType(args.feature_store_type),
        args.shard_maxcount,
        db_engine,
    )

    params, segment_length = get_dataset_params(feature_extractors, args.thumbnails)
    stream = get_dataset_stream(all_metadata, params, args.use_shots)
    av_data_loader = get_dataloader(stream, args.num_workers)

    audio_sampling_rate = params['audio_sampling_rate']
    video_frame_rate = params['video_frame_rate']
    audio_segment_length = segment_length
    audio_frames_per_chunk = int(round(audio_sampling_rate * audio_segment_length))

    MAX_BULK_INSERT = 1024
    with (
        db_engine.connect() as conn,
        thumbs_engine.connect() as thumbs_conn,
        tqdm(desc="Feature extraction") as pbar,
        torch.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=args.enable_autocast)
    ):
        mid: str | int # type annotation
        chunks: dict[
            MediaChunkType, dict[str, MediaChunk | None] | MediaChunk | None
        ]  # type annotation

        def handle_chunk(
            chunk: MediaChunk, media_type: MediaChunkType, feature_extractor_id: str
        ):
            segment_tensor = chunk.tensor
            segment_pts = chunk.pts

            if segment_tensor is None or segment_tensor.shape[0] == 0:
                logger.warning(
                    f"Skipping empty segment for media_id={mid}, media_type={media_type}, feature_extractor_id={feature_extractor_id}"
                )
                return

            feature_extractor = feature_extractors[media_type][feature_extractor_id]
            feature_store = feature_stores[media_type][feature_extractor_id]

            if media_type == "image" or media_type == "video":
                segment_feature = feature_extractor.extract_image_features(
                    segment_tensor
                )
                pbar.update(segment_tensor.shape[0])

            elif media_type == "audio":
                if segment_tensor.shape[2] < audio_frames_per_chunk:
                    # we discard any malformed audio segments
                    return
                segment_feature = feature_extractor.extract_audio_features(
                    segment_tensor
                )
                pbar.update(segment_tensor.shape[0])
            else:
                raise ValueError(f"Unknown media_type {media_type}")

            # TODO: Update based on model - internvideo might need end timestamp, whereas clip might not
            if media_type == MediaType.VIDEO or media_type == MediaType.IMAGE:
                for frame_idx, frame_features in enumerate(segment_feature):
                    vector_ids: list[int] = []
                    if type(segment_pts) is list and args.use_shots:
                        frame_timestamp = segment_pts[frame_idx]
                    else:
                        frame_timestamp = segment_pts + frame_idx * (
                            1 / video_frame_rate
                        )

                    for frame_single_vector in frame_features.vectors:
                        feature_metadata = VectorRepo.create(
                            conn,
                            data=VectorMetadata(
                                modality=media_type,
                                feature_extractor_id=feature_extractor_id,
                                media_id=mid,
                                timestamp=frame_timestamp,
                            ),
                        )
                        feature_store.add(
                            feature_metadata.id,
                            np.expand_dims(frame_single_vector, axis=0),
                        )
                        vector_ids.append(feature_metadata.id)
                    feature_extractor.add_to_vector_metadata_table(
                        conn, vector_ids, frame_features.metadata
                    )
            else:
                # Add whole segment
                _start_time = segment_pts
                _end_time = segment_pts + audio_segment_length
                feature_metadata = VectorRepo.create(
                    conn,
                    data=VectorMetadata(
                        modality=media_type,
                        feature_extractor_id=feature_extractor_id,
                        media_id=mid,
                        timestamp=_start_time,
                        end_timestamp=_end_time,
                    ),
                )
                feature_store.add(feature_metadata.id, segment_feature)

        for idx, (mid, chunks) in enumerate(av_data_loader):
            for media_type in chunks:
                if media_type not in feature_extractors or chunks[media_type] is None:
                    # This is a single chunk, not a dictionary of chunks
                    logger.debug(
                        f"Skipping empty / irrelevant chunk for media_id={mid}, media_type={media_type}"
                    )
                    continue
                if isinstance(chunks[media_type], MediaChunk):
                    # We are somehow reading a chunk without any feature extractor ids - must be thumbnails or reading audio / video while only requesting video / audio features
                    continue
                for feature_extractor_id in chunks[media_type]:
                    _chunk = chunks[media_type][feature_extractor_id]
                    if (
                        _chunk is None
                        or feature_extractor_id not in feature_extractors[media_type]
                    ):
                        logger.debug(
                            f"Skipping empty / irrelevant chunk for media_id={mid}, media_type={media_type}, feature_extractor_id={feature_extractor_id}"
                        )
                        continue
                    handle_chunk(_chunk, media_type, feature_extractor_id)

            if 'thumbnails' in chunks and chunks['thumbnails'] is not None:
                # Handle thumbnails
                _thumb_jpegs = chunks['thumbnails'].tensor
                _thumb_pts = chunks['thumbnails'].pts

                # Store in thumbnail store
                # (thumbnail will be N x 3 x 192 x W)
                for thumb_index in range(len(_thumb_jpegs)):
                    if type(_thumb_pts) is list and args.use_shots:
                        thumb_timestamp = _thumb_pts[thumb_index]
                    else:
                        thumb_timestamp = _thumb_pts + thumb_index * (1 / video_frame_rate)
                    # convert thumb tensor to jpeg
                    thumbnail_metadata = ThumbnailRepo.create(
                        thumbs_conn,
                        data=ThumbnailMetadata(
                            media_id=mid,
                            timestamp=thumb_timestamp,
                            content=bytes(_thumb_jpegs[thumb_index].numpy().data),
                        ),
                    )
            if idx % MAX_BULK_INSERT == 0:
                conn.commit()
                thumbs_conn.commit()

        conn.commit()
        thumbs_conn.commit()

    del av_data_loader

    for id in feature_stores:
        for feature_extractor_id in feature_stores[id]:
            feature_stores[id][feature_extractor_id].close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Feature extraction completed in {elapsed_time:.0f} sec ({elapsed_time/60:.2f} min)"
    )
