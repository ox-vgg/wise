import argparse
import glob
import os
import time
from pathlib import Path
from typing import Dict, List
import torch.utils.data as torch_data
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from src.dataloader import AVDataset, get_media_metadata
from src.dataloader.utils import VIDEO_EXTENSIONS, is_valid_image, is_valid_video
from src.wise_project import WiseProject
from src.feature.feature_extractor_factory import FeatureExtractorFactory
from src.feature.store.feature_store_factory import FeatureStoreFactory
from src import db
from src.data_models import (
    MediaMetadata,
    SourceCollection,
    ExtraMediaMetadata,
    VectorMetadata,
    ThumbnailMetadata,
    MediaType,
    SourceCollectionType,
)
from src.repository import (
    SourceCollectionRepo,
    MediaRepo,
    VectorRepo,
    MediaMetadataRepo,
    ThumbnailRepo,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="extract-features",
        description="Initialise a WISE project by extractng features from images, audio and videos.",
        epilog="For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/",
    )
    parser.add_argument(
        "--media-dir",
        required=True,
        action="append",
        dest="media_dir_list",
        type=str,
        help="source images and video from this folder",
    )

    parser.add_argument(
        "--media-include",
        required=False,
        action="append",
        dest="media_include_list",
        default=[f"*.{ext}" for ext in VIDEO_EXTENSIONS],
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
        help="use this feature extractor for video frames",
    )

    parser.add_argument(
        "--audio-feature-id",
        required=False,
        type=str,
        help="use this feature extractor for audio samples",
    )

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

    # sanity check: remove duplicate entries in command line args
    if len(args.media_include_list) > 1:
        unique_media_include_list = list(set(args.media_include_list))
        setattr(args, 'media_include_list', unique_media_include_list)
    if len(args.media_dir_list) > 1:
        unique_media_dir_list = list(set(args.media_dir_list))
        setattr(args, 'media_dir_list', unique_media_dir_list)

    # we need a non-existing folder to initialise a new WISE project
    if Path(args.project_dir).exists():
        raise ValueError(f'project_dir {args.project_dir} already exists')

    # TODO: allow adding new files to an existing project
    project = WiseProject(args.project_dir, create_project=True)
    db_engine = db.init_project(project.dburi, echo=False)
    thumbs_engine = db.init_thumbs(project.thumbs_uri, echo=False)

    start_time = time.time()

    ## 1. Check validity of input files
    print("Checking for invalid media files")
    # Create dictionary of media files; key: media_dir, value: list of filepaths
    valid_media_files: defaultdict[str, List[str]] = defaultdict(list)
    for media_dir in args.media_dir_list:
        for media_include in args.media_include_list:
            media_search_dir = os.path.join(media_dir, "**/" + media_include)
            media_paths = list(glob.iglob(pathname=media_search_dir, recursive=True))
            media_paths = [Path(x) for x in media_paths]
            media_paths = [str(x) for x in tqdm(media_paths) if x.is_file() and is_valid_video(x)]
            if len(media_paths) > 0:
                valid_media_files[media_dir] += media_paths

    if len(valid_media_files) == 0:
        raise ValueError("No valid media files found")

    ## 2. Initialise internal metadata database
    print('Initialising internal metadata database')
    media_filelist: Dict[int, str] = {} # maps media ids to filepaths
    for media_dir, media_paths in valid_media_files.items():
        with db_engine.begin() as conn:
            # Add each folder to source collection table
            data = SourceCollection(
                location=media_dir, type=SourceCollectionType.DIR
            )
            media_source_collection = SourceCollectionRepo.create(conn, data=data)
        
            for media_path in media_paths:
                # Get metadata for each file and add it to media table
                # Get media_path relative to
                media_metadata = get_media_metadata(media_path)
                metadata = MediaRepo.create(
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
                        duration=media_metadata.duration,
                    ),
                )
                # extra_metadata = ExtraMediaMetadata(
                #     media_id=metadata.id,
                #     metadata={
                #         "fps": media_metadata.fps,
                #     }
                #     | media_metadata.extra,
                # )
                # MediaMetadataRepo.create(conn, data=extra_metadata)
                media_filelist[metadata.id] = media_path


    ## 3. Prepare for feature extraction and storage
    print(f"Extracting features from {len(media_filelist)} files")
    feature_extractor_id_list = {}
    if hasattr(args, 'video_feature_id') and args.video_feature_id is not None:
        feature_extractor_id_list['video'] = args.video_feature_id
    else:
        feature_extractor_id_list['video'] = 'mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k'
    if hasattr(args, 'audio_feature_id') and args.audio_feature_id is not None:
        feature_extractor_id_list['audio'] = args.audio_feature_id
    else:
        feature_extractor_id_list['audio'] = 'microsoft/clap/2023/four-datasets'

    feature_extractor_list = {}
    feature_store_dir_list = {}
    feature_store_list = {}

    for media_type in feature_extractor_id_list:
        feature_extractor_id = feature_extractor_id_list[media_type]

        ## 3.1 Initialise feature extractor
        feature_extractor_list[media_type] = FeatureExtractorFactory(
            feature_extractor_id
        )
        print(f"Using {feature_extractor_id} for {media_type}")

        ## 3.2 Create folders to store features, metadata and search index
        project.create_features_dir(feature_extractor_id)

        ## 3.3 Initialise feature store to store features
        feature_store_list[media_type] = FeatureStoreFactory.create_store(
            args.feature_store_type,
            media_type,
            project.features_dir(feature_extractor_id),
        )
        feature_store_list[media_type].enable_write(
            args.shard_maxcount, args.shard_maxsize
        )

    ## 4. Initialise data loader
    audio_sampling_rate = 48_000  # (48 kHz)
    video_frame_rate = 2  # fps
    video_frames_per_chunk = 8  # frames
    segment_length = video_frames_per_chunk / video_frame_rate  # frames / fps = seconds
    audio_segment_length = segment_length  # seconds
    audio_frames_per_chunk = int(round(audio_sampling_rate * audio_segment_length))

    stream = AVDataset(
        media_filelist,
        video_frames_per_chunk=video_frames_per_chunk,
        video_frame_rate=video_frame_rate,
        video_preprocessing_function=feature_extractor_list["video"].preprocess_image,
        audio_samples_per_chunk=audio_frames_per_chunk,
        audio_sample_rate=audio_sampling_rate,
        audio_preprocessing_function=feature_extractor_list["audio"].preprocess_audio,
        offset=None,
        thumbnails=args.thumbnails,
    )

    ## 5. extract video and audio features
    print(f"Initializing data loader with {args.num_workers} workers ...")
    av_data_loader = torch_data.DataLoader(
        stream, batch_size=None, num_workers=args.num_workers
    )
    MAX_BULK_INSERT = 8192
    with db_engine.connect() as conn, thumbs_engine.connect() as thumbs_conn, tqdm(desc="Feature extraction") as pbar:
        for idx, (mid, video, audio, *rest) in enumerate(av_data_loader):
            media_segment = {"video": video, "audio": audio}
            for media_type in feature_extractor_id_list:
                if media_segment[media_type] is None:
                    continue
                segment_tensor = media_segment[media_type].tensor
                segment_pts = media_segment[media_type].pts

                if media_type == "image" or media_type == "video":
                    segment_feature = feature_extractor_list[
                        media_type
                    ].extract_image_features(segment_tensor)
                elif media_type == "audio":
                    if segment_tensor.shape[2] < audio_frames_per_chunk:
                        # we discard any malformed audio segments
                        continue
                    segment_feature = feature_extractor_list[
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
                        feature_store_list[media_type].add(
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
                    feature_store_list[media_type].add(
                        feature_metadata.id, segment_feature
                    )

            if len(rest) > 0:
                # Handle thumbnails
                _thumbnails = rest[0]
                if _thumbnails == None:
                    continue

                _thumb_jpegs = _thumbnails.tensor
                _thumb_pts = _thumbnails.pts

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
            _media = video or audio
            if _media is not None:
                pbar.update(_media.tensor.shape[0])

            if idx % MAX_BULK_INSERT == 0:
                conn.commit()
                thumbs_conn.commit()

        conn.commit()
        thumbs_conn.commit()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Feature extraction completed in {elapsed_time:.0f} sec ({elapsed_time/60:.2f} min)"
    )
