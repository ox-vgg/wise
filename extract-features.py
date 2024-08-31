import argparse
import os
import time
from pathlib import Path
from typing import Dict, Literal
import torch.utils.data as torch_data
from tqdm import tqdm
import numpy as np
import logging

from src.dataloader.dataset import MediaChunk
from src.dataloader import  get_dataset, get_metadata_for_valid_files, DatasetPayload
from src.dataloader.streamreader import SourceMediaType, MediaChunkType
from src.wise_project import WiseProject
from src.feature.feature_extractor import FeatureExtractor
from src.feature.feature_extractor_factory import FeatureExtractorFactory
from src.feature.store.feature_store import FeatureStore
from src.feature.store.feature_store_factory import FeatureStoreFactory
from src import db
from src.data_models import (
    MediaMetadata,
    SourceCollection,
    ExtraMediaMetadata,
    VectorMetadata,
    ThumbnailMetadata,
    MediaType,
    ModalityType,
    SourceCollectionType,
)
from src.repository import (
    SourceCollectionRepo,
    MediaRepo,
    VectorRepo,
    MediaMetadataRepo,
    ThumbnailRepo,
)


def get_files_from_directory_with_extensions(dir: Path, extensions: list[str]):
    return (x for ext in extensions for x in dir.rglob(ext) if x.is_file())

def initialise_feature_extractors(
    project: WiseProject,
    feature_extractor_ids: dict[ModalityType, str],
    feature_store_type: Literal['webdataset', 'numpy'],
    shard_max_count: int,
    shard_max_size: int
) -> tuple[dict[ModalityType, FeatureExtractor], dict[ModalityType, FeatureStore]]:
    ## 3. Prepare for feature extraction and storage
    logger.info(f"Initialising feature extractor")

    feature_extractors = {}
    feature_stores = {}

    for modality_type, feature_extractor_id in feature_extractor_ids.items():
        ## 3.1 Initialise feature extractor
        feature_extractors[modality_type] = FeatureExtractorFactory(
            feature_extractor_id
        )
        print(f"Using {feature_extractor_id} for {modality_type}")

        ## 3.2 Create folders to store features, metadata and search index
        project.create_features_dir(feature_extractor_id)

        ## 3.3 Initialise feature store to store features
        feature_stores[modality_type] = FeatureStoreFactory.create_store(
            feature_store_type,
            modality_type,
            project.features_dir(feature_extractor_id),
        )
        feature_stores[modality_type].enable_write(
            shard_max_count, shard_max_size
        )

    return feature_extractors, feature_stores

def process_media_dir(media_dir: Path, db_engine, include_extensions: list[str] = ['*']):

    # Get files matching extensions
    input_files = list(get_files_from_directory_with_extensions(media_dir, include_extensions))

    # Get metadata and media datasets corresponding to the files
    metadata, unknown_files = get_metadata_for_valid_files(input_files)
    if len(unknown_files) > 0:
        logger.info(f'Skipping {len(unknown_files)} files that are not valid media in directory "{media_dir}"')
        logger.debug("\n".join(map(str, unknown_files)))

    # Add metadata to database
    dataset_payload: list[DatasetPayload] = []
    with tqdm(total=len(metadata)) as pbar, db_engine.begin() as conn:
        # Add each folder to source collection table
        data = SourceCollection(
            location=str(media_dir), type=SourceCollectionType.DIR
        )
        media_source_collection = SourceCollectionRepo.create(conn, data=data)

        for media_metadata in metadata:
            # Get metadata for each file and add it to media table
            # Get media_path relative to
            media_path = media_metadata.path
            _metadata = MediaRepo.create(
                conn,
                data=MediaMetadata(
                    source_collection_id=media_source_collection.id,
                    path=os.path.relpath(
                        media_path, media_source_collection.location
                    ),
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
            dataset_payload.append(DatasetPayload(_metadata.id, media_path, _metadata.media_type))
            pbar.update(1)


    # return metadata and datasets to be chained
    return dataset_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="extract-features",
        description="Initialise a WISE project by extractng features from images, audio and videos.",
        epilog="For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/",
    )
    parser.add_argument(
        "media_dir_list",
        nargs='+',
        help="process images and video from this folder",
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

    parser.add_argument(
        "--shard-maxcount",
        required=False,
        type=int,
        default=2048,
        help="max number of entries in each shard of webdataset tar",
    )

    parser.add_argument(
        "--shard-maxsize",
        required=False,
        type=int,
        default=20 * 1024 * 1024,  # tar overheads results in 25MB shards
        help="max size (in bytes) of each shard of webdataset tar",
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
        default="webdataset",
        dest="feature_store_type",
        choices=["webdataset", "numpy"],
        help="extracted features are stored using this data structure",
    )

    parser.add_argument(
        "--video-feature-id",
        required=False,
        type=str,
        default="mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k",
        help="use this feature extractor for video frames",
    )

    parser.add_argument(
        "--audio-feature-id",
        required=False,
        type=str,
        default="microsoft/clap/2023/four-datasets",
        help="use this feature extractor for audio samples",
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
        "--thumbnails", default=True, action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # sanity check: remove duplicate entries in command line args
    n_extension = len(args.media_include_list)
    if n_extension == 0:
        setattr(args, 'media_include_list', ['*'])
    else:
        unique_media_include_list = list(set(args.media_include_list))
        setattr(args, 'media_include_list', unique_media_include_list)

    if len(args.media_dir_list) > 1:
        unique_media_dir_list = list(set(args.media_dir_list))
        setattr(args, 'media_dir_list', unique_media_dir_list)

    assert all(Path(x).is_dir() for x in args.media_dir_list), "All values for media_dir_list must be directories"

    # we need a non-existing folder to initialise a new WISE project
    if Path(args.project_dir).exists():
        raise ValueError(f'project_dir {args.project_dir} already exists')

    # TODO: allow adding new files to an existing project
    project = WiseProject(args.project_dir, create_project=True)
    db_engine = db.init_project(project.dburi, echo=False)
    thumbs_engine = db.init_thumbs(project.thumbs_uri, echo=False)

    start_time = time.time()

    ## 1. Initialise internal metadata database with valid files
    print('Initialising internal metadata database')
    all_metadata: list[DatasetPayload] = []
    for media_dir in args.media_dir_list:
        metadata = process_media_dir(Path(media_dir), db_engine, args.media_include_list)
        all_metadata.extend(metadata)

    # Get the set of media types present in the input media files
    media_types_present: set[SourceMediaType] = set(x.media_type for x in all_metadata)

    ## 5. extract video and audio features
    feature_extractor_ids: dict[ModalityType, str] = {}
    if SourceMediaType.VIDEO in media_types_present or SourceMediaType.AV in media_types_present:
        feature_extractor_ids[ModalityType.VIDEO] = args.video_feature_id
    if SourceMediaType.IMAGE in media_types_present:
        feature_extractor_ids[ModalityType.IMAGE] = args.video_feature_id
    if SourceMediaType.AUDIO in media_types_present or SourceMediaType.AV in media_types_present:
        # TODO: temporary disable - if not args.skip_audio_feature_extraction:
        feature_extractor_ids[ModalityType.AUDIO] = args.audio_feature_id

    feature_extractors, feature_stores = initialise_feature_extractors(
        project,
        feature_extractor_ids,
        args.feature_store_type,
        args.shard_maxcount, args.shard_maxsize,
    )

    ## dataset
    audio_sampling_rate = 48_000  # (48 kHz)
    video_frame_rate = 2  # fps
    video_frames_per_chunk = 8  # frames
    segment_length = video_frames_per_chunk / video_frame_rate  # frames / fps = seconds
    audio_segment_length = segment_length  # seconds
    audio_frames_per_chunk = int(round(audio_sampling_rate * audio_segment_length))

    params = {
        "video_frames_per_chunk": video_frames_per_chunk,
        "video_frame_rate": video_frame_rate,
        "video_preprocessing_function": feature_extractors[ModalityType.VIDEO].preprocess_image if ModalityType.VIDEO in feature_extractors else None,

        "audio_samples_per_chunk": audio_frames_per_chunk,
        "audio_sampling_rate": audio_sampling_rate,
        "audio_preprocessing_function": feature_extractors[ModalityType.AUDIO].preprocess_audio if ModalityType.AUDIO in feature_extractors else None,

        "image_preprocessing_function": feature_extractors[ModalityType.IMAGE].preprocess_image if ModalityType.IMAGE in feature_extractors else None,

        "offset": None,
        "thumbnails": args.thumbnails
    }
    stream = torch_data.ChainDataset(
        get_dataset(all_metadata, params)
    )
    print(f"Initializing data loader with {args.num_workers} workers ...")
    av_data_loader = torch_data.DataLoader(
        stream, batch_size=None, num_workers=args.num_workers
    )
    MAX_BULK_INSERT = 8192
    with db_engine.connect() as conn, thumbs_engine.connect() as thumbs_conn, tqdm(desc="Feature extraction") as pbar:
        mid: str | int # type annotation
        chunks: Dict[MediaChunkType, MediaChunk | None] # type annotation
        for idx, (mid, chunks) in enumerate(av_data_loader):
            for media_type in feature_extractor_ids:
                if media_type not in chunks:
                    continue
                segment_tensor = chunks[media_type].tensor
                segment_pts = chunks[media_type].pts

                if media_type == "image" or media_type == "video":
                    segment_feature = feature_extractors[
                        media_type
                    ].extract_image_features(segment_tensor)
                elif media_type == "audio":
                    if segment_tensor.shape[2] < audio_frames_per_chunk:
                        # we discard any malformed audio segments
                        continue
                    segment_feature = feature_extractors[
                        media_type
                    ].extract_audio_features(segment_tensor)
                else:
                    raise ValueError("Unknown media_type {media_type}")

                # TODO: Update based on model - internvideo might need end timestamp, whereas clip might not
                if media_type == MediaType.VIDEO or media_type == MediaType.IMAGE:
                    for i in range(len(segment_feature)):
                        feature_metadata = VectorRepo.create(
                            conn,
                            data=VectorMetadata(
                                modality=media_type,
                                media_id=mid,
                                timestamp=segment_pts + i * (1 / video_frame_rate),
                            ),
                        )
                        feature_stores[media_type].add(
                            feature_metadata.id,
                            np.expand_dims(segment_feature[i], axis=0),
                        )
                else:
                    # Add whole segment
                    _start_time = segment_pts
                    _end_time = segment_pts + audio_segment_length
                    feature_metadata = VectorRepo.create(
                        conn,
                        data=VectorMetadata(
                            modality=media_type,
                            media_id=mid,
                            timestamp=_start_time,
                            end_timestamp=_end_time,
                        ),
                    )
                    feature_stores[media_type].add(
                        feature_metadata.id, segment_feature
                    )

            if 'thumbnails' in chunks:
                # Handle thumbnails
                _thumb_jpegs = chunks['thumbnails'].tensor
                _thumb_pts = chunks['thumbnails'].pts

                # Store in thumbnail store
                # (thumbnail will be N x 3 x 192 x W)
                for i in range(len(_thumb_jpegs)):
                    # convert thumb tensor to jpeg
                    thumbnail_metadata = ThumbnailRepo.create(
                        thumbs_conn,
                        data=ThumbnailMetadata(
                            media_id=mid,
                            timestamp=_thumb_pts + i * 0.5,
                            content=bytes(_thumb_jpegs[i].numpy().data),
                        ),
                    )

            # Update progress bar
            _media = chunks.get('video') or chunks.get('audio') or chunks.get('image')
            if _media is not None:
                pbar.update(_media.tensor.shape[0])

            if idx % MAX_BULK_INSERT == 0:
                conn.commit()
                thumbs_conn.commit()

        conn.commit()
        thumbs_conn.commit()

    for id in feature_stores:
        store = feature_stores[id]
        store.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Feature extraction completed in {elapsed_time:.0f} sec ({elapsed_time/60:.2f} min)"
    )
