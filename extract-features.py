import typer
import glob
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch.utils.data as torch_data
from tqdm import tqdm
import numpy as np

from src.dataloader import AVDataset, get_media_metadata
from src.wise_project import WiseProject
from src.feature.feature_extractor_factory import FeatureExtractorFactory
from src.feature.store.feature_store_factory import FeatureStoreFactory, FeatureStoreType
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
    ThumbnailMetadataRepo,
)

app = typer.Typer()

@app.command(
    help="Initialise a WISE project by extracting features from images, audio and videos.",
    epilog="For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/",
    no_args_is_help=True,
)
def main(
    media_dir_list: List[Path] = typer.Option(
        ...,
        "--media-dir",
        dir_okay=True,
        file_okay=False,
        exists=True,
        help="Source images and videos from this folder",
    ),
    project_dir: Path = typer.Option(
        dir_okay=True,
        file_okay=False,
        help="Folder where all project assets are stored",
    ),
    media_include_list: List[str] = typer.Option(
        ["*.mkv"],
        "--media-include",
        help="Regular expression to include certain media files",
    ),
    num_workers: int = typer.Option(
        0,
        help="Number of workers (subprocesses) used by DataLoader. If set to 0, the main process is used for data loading.",
        # TODO add code to automatically determine the number of workers to use if this value is omitted
    ),
    shard_maxcount: int = typer.Option(
        2048,
        help="Max number of entries in each shard of webdataset tar",
    ),
    shard_maxsize: int = typer.Option(
        20 * 1024 * 1024,  # tar overheads results in 25MB shards
        help="Max size (in bytes) of each shard of webdataset tar",
    ),
    feature_store_type: FeatureStoreType = typer.Option(
        "webdataset",
        "--feature-store",
        help="Extracted features are stored using this data structure",
    ),
    thumbnails: bool = typer.Option(True),
):
    # TODO: allow adding new files to an existing project
    project = WiseProject(str(project_dir), create_project=True)

    DB_SCHEME = "sqlite+pysqlite://"
    DB_URI = f"{DB_SCHEME}/{project_dir}/{project_dir.stem}.db"
    db_engine = db.init_project(DB_URI, echo=False)

    start_time = time.time()

    ## 1. Create a list of input files
    media_filelist: Dict[int, str] = {}

    for media_dir in media_dir_list:
        with db_engine.begin() as conn:
            # Add each folder to source collection table
            data = SourceCollection(
                location=str(media_dir), type=SourceCollectionType.IMAGE_DIR
            )
            media_source_collection = SourceCollectionRepo.create(conn, data=data)
            for media_include in media_include_list:
                media_search_dir = os.path.join(media_dir, "**/" + media_include)
                for media_path in glob.iglob(pathname=media_search_dir, recursive=True):
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
                            hash=media_metadata.md5sum,
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

    print(f"Extracting features from {len(media_filelist)} files")

    ## 2. Prepare for feature extraction and storage
    feature_extractor_id_list = {
        "audio": "microsoft/clap/2023/four-datasets",
        "video": "mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k",
    }
    feature_extractor_list = {}
    feature_store_dir_list = {}
    feature_store_list = {}

    for media_type in feature_extractor_id_list:
        feature_extractor_id = feature_extractor_id_list[media_type]

        ## 2.1 Initialise feature extractor
        feature_extractor_list[media_type] = FeatureExtractorFactory(
            feature_extractor_id
        )
        print(f"Using {feature_extractor_id} for {media_type}")

        ## 2.2 Create folders to store features, metadata and search index
        project.create_features_dir(feature_extractor_id)

        ## 2.3 Initialise feature store to store features
        feature_store_list[media_type] = FeatureStoreFactory.create_store(
            feature_store_type,
            media_type,
            project.features_dir(feature_extractor_id),
        )
        feature_store_list[media_type].enable_write(
            shard_maxcount, shard_maxsize
        )

    if thumbnails:
        feature_store_list["thumbnails"] = FeatureStoreFactory.create_store(
            feature_store_type,
            "thumbs",
            project.thumbnail_dir(),
        )
        feature_store_list["thumbnails"].enable_write(
            shard_maxcount, shard_maxsize
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
        thumbnails=thumbnails,
    )

    ## 5. extract video and audio features
    print(f"Initializing data loader with {num_workers} workers ...")
    av_data_loader = torch_data.DataLoader(
        stream, batch_size=None, num_workers=num_workers
    )
    MAX_BULK_INSERT = 8192
    with db_engine.connect() as conn:
        for idx, (mid, video, audio, *rest) in enumerate(tqdm(av_data_loader)):
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
                    for i in range(segment_feature.shape[0]):
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
                            np.reshape(
                                segment_feature[i], (1, segment_feature.shape[1])
                            ),
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

                _thumb_tensor = _thumbnails.tensor
                _thumb_pts = _thumbnails.pts

                # Store in thumbnail store
                # (thumbnail will be N x 3 x 192 x W)
                for i in range(_thumb_tensor.shape[0]):
                    thumbnail_metadata = ThumbnailMetadataRepo.create(
                        conn,
                        data=ThumbnailMetadata(
                            media_id=mid,
                            timestamp=_thumb_pts + i * 0.5,
                        ),
                    )
                    feature_store_list["thumbnails"].add(
                        thumbnail_metadata.id,
                        _thumb_tensor[i],
                    )

            if idx % MAX_BULK_INSERT == 0:
                conn.commit()

        conn.commit()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Feature extraction completed in {elapsed_time:.0f} sec. or {elapsed_time/60:.2f} min."
    )

if __name__ == "__main__":
    app()
