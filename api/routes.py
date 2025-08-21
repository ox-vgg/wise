from contextlib import ExitStack
import datetime
import time
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, Union, BinaryIO, Any
import io
import itertools
import functools
import logging
from pathlib import Path
import tarfile
from PIL import Image
import math
import numpy as np
from collections import defaultdict
import torch
import torchaudio
from numpy import ndarray, array, zeros, average, expand_dims, float32
from numpy.random import default_rng
from numpy.linalg import norm
from tempfile import NamedTemporaryFile
from torch.hub import download_url_to_file
from fastapi import APIRouter, HTTPException, Query, File, Form, Request, status
from fastapi.responses import (
    Response,
    FileResponse,
    PlainTextResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import field_validator, BaseModel, ConfigDict
import typer
import csv
import json
import os
import sqlalchemy as sa
from webvtt import Caption, WebVTT

from config import APIConfig
from src.index.search_index_factory import SearchIndexFactory
from src.index.search_index import SearchIndex
from src import db
from src.repository import (
    SourceCollectionRepo,
    MediaRepo,
    VectorRepo,
    ThumbnailRepo,
    # query_by_timestamp,
    get_featured_images,
    get_full_metadata_batch,
    get_media_counts_by_media_type,
    get_project_total_duration,
    get_thumbnail_by_timestamp,
    get_related_vectors_rows,
)
from src.data_models import MediaMetadata, MediaType, ModalityType, SourceCollectionType, VectorAndMediaMetadata, VideoShot
from src.enums import IndexType
from src.utils import convert_uint8array_to_base64
from src.wise_project import WiseProject
from src.feature import FeatureExtractorFactory
from src.feature.feature_extractor import FeatureExtMetadata
from src.search.fts import FTSSearch, WISEFTSQuery
from src.dataloader import AVDataset

import faiss
import json

logger = logging.getLogger(__name__)


def raise_(ex):
    raise ex

# TODO config
NUM_THUMBNAILS_PER_PARTITION = 300


def timedelta_to_vtt_timestamp(dt: datetime.timedelta):
    """
    Convert the timedelta python object to a HH:MM:SS.sss string
    Required by the VTT Captions
    """
    _seconds = dt.seconds

    hours = (dt.days * 24) + (_seconds // 3600)
    _seconds = _seconds % 3600

    minutes = _seconds // 60
    seconds = _seconds % 60

    milliseconds = dt.microseconds // 1000

    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def get_thumbnail_count_for_media_id(thumbs_conn: sa.Connection, _video_media_id: int):
    _thumbs_table = db.thumbnails_table
    num_thumbs = thumbs_conn.execute(
        sa.select(sa.func.count(_thumbs_table.c.id)).where(
            _thumbs_table.c.media_id == _video_media_id
        )
    ).scalar_one()

    if num_thumbs == 0:
        raise ValueError(f"no thumbnails for media id {_video_media_id}")

    return num_thumbs


def get_thumbnail_size(thumbs_conn: sa.Connection, _video_media_id: int):
    # Assumes all thumbnails have the same size
    _thumbs_table = db.thumbnails_table
    one_thumb = thumbs_conn.execute(
        sa.select(_thumbs_table.c.content)
        .where(_thumbs_table.c.media_id == _video_media_id)
        .limit(1)
    ).scalar_one()

    with Image.open(io.BytesIO(one_thumb)) as im:
        w, h = im.size

    return w, h


def get_thumbnails(
    thumbs_conn: sa.Connection,
    _video_media_id: int,
    num_seconds_per_image: int,
    partition_id: int | None = None,
):
    _thumbs_table = db.thumbnails_table
    # # For videos longer than 30 minutes, reduce the frequency of thumbnails to 1 every 4 seconds
    # num_seconds_per_image = 2 if num_thumbs < (2 * 30 * 60) else 4
    num_images_per_partition = (
        NUM_THUMBNAILS_PER_PARTITION if partition_id is not None else None
    )
    offset = partition_id * num_images_per_partition if partition_id is not None else 0
    cols = [
        _thumbs_table.c.id,
        _thumbs_table.c.timestamp,
    ]
    if partition_id is not None:
        # add thumbnail data as well
        cols.append(_thumbs_table.c.content)

    stmt = (
        sa.select(*cols)
        .where(
            sa.and_(
                _thumbs_table.c.media_id == _video_media_id,
                (10 * _thumbs_table.c.timestamp) % (10 * num_seconds_per_image) == 0,
            )
        )
        .order_by(_thumbs_table.c.timestamp)
        .offset(offset)
        .limit(num_images_per_partition)
    )
    return thumbs_conn.execute(stmt).all()


def get_thumbnail_spritesheet(
    thumbs_conn: sa.Connection,
    _video_media_id: int,
    num_seconds_per_image: int,
    partition_id: int,
):
    all_thumbs = list(
        get_thumbnails(
            thumbs_conn, _video_media_id, num_seconds_per_image, partition_id
        )
    )
    num_thumbs = len(all_thumbs)
    if num_thumbs == 0:
        raise ValueError(
            f"No thumbnails found for media {_video_media_id} and parition {partition_id}"
        )
    # Get thumbnails
    with Image.open(io.BytesIO(all_thumbs[0].content)) as im:
        w, h = im.size  # Assumes all thumbnails have the same size

    # Create storyboard
    num_columns = 10
    num_rows = math.ceil(num_thumbs / num_columns)
    storyboard = Image.new("RGB", (w * num_columns, h * num_rows))
    for idx, thumb in enumerate(all_thumbs):
        x = (idx % num_columns) * w
        y = (idx // num_columns) * h
        with Image.open(io.BytesIO(thumb.content)) as _thumb:
            storyboard.paste(_thumb, (x, y))

    return storyboard


def get_webvtt_spritesheet(
    thumbs_conn: sa.Connection, _video_media_id: int, num_seconds_per_image: int = 2
):
    all_thumbs = list(
        get_thumbnails(thumbs_conn, _video_media_id, num_seconds_per_image)
    )
    num_thumbs = len(all_thumbs)
    if num_thumbs == 0:
        raise ValueError(f"No thumbnails found for media {_video_media_id}!")

    w, h = get_thumbnail_size(thumbs_conn, _video_media_id)
    # Create storyboard
    num_columns = 10
    vtt = WebVTT()
    next_timestamp = datetime.timedelta(seconds=0)
    for thumb_idx, thumb in enumerate(all_thumbs):
        partition_id = thumb_idx // NUM_THUMBNAILS_PER_PARTITION
        idx = thumb_idx % NUM_THUMBNAILS_PER_PARTITION
        x = (idx % num_columns) * w
        y = (idx // num_columns) * h

        current_timestamp = datetime.timedelta(seconds=thumb.timestamp)
        next_timestamp = current_timestamp + datetime.timedelta(
            seconds=num_seconds_per_image
        )
        vtt.captions.append(
            Caption(
                timedelta_to_vtt_timestamp(current_timestamp),
                timedelta_to_vtt_timestamp(next_timestamp),
                f"storyboard/{_video_media_id}/{partition_id}.jpg#xywh={x},{y},{w},{h}",
            )
        )

    return vtt.content


class WiseFrontendUserException(Exception):
    """An exception whose message can be sent to the user.

    Exceptions by default will only send an "Internal server error"
    message to the user.  This separate class enables us to catch only
    some with a message meant to the frontend user.
    """
    pass


def send_bytes_range_requests(
    file_obj: BinaryIO, start: int, end: int, chunk_size: int = 10_000
):
    """Send a file in chunks using Range Requests specification RFC7233

    `start` and `end` parameters are inclusive due to specification
    """
    with file_obj as f:
        f.seek(start)
        while (pos := f.tell()) <= end:
            read_size = min(chunk_size, end + 1 - pos)
            yield f.read(read_size)


def _get_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    def _invalid_range():
        return HTTPException(
            status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail=f"Invalid request range (Range:{range_header!r})",
        )

    try:
        h = range_header.replace("bytes=", "").split("-")
        start = int(h[0]) if h[0] != "" else 0
        end = int(h[1]) if h[1] != "" else file_size - 1
    except ValueError:
        raise _invalid_range()

    if start > end or start < 0 or end > file_size - 1:
        raise _invalid_range()
    return start, end


def get_project_router(config: APIConfig):
    if config.project_dir is None:
        raise typer.BadParameter("project_dir is missing!")

    project = WiseProject(config.project_dir, create_project=False)
    if config.thumbnail_project_dir:
        try:
            WiseProject(config.thumbnail_project_dir, create_project=False)
        except Exception as e:
            logging.error(e)
            raise typer.BadParameter(
                f'Project from thumbnails "{config.thumbnail_project_dir}" not found!'
            )

    project_name = config.project_dir.stem
    router = APIRouter(prefix=f"/{project_name}", tags=[f"{project_name}"])
    search_router, search_router_info = _get_search_router(config)
    router.include_router(_get_project_data_router(config, search_router_info))
    router.include_router(_get_report_image_router(config))
    router.include_router(search_router)

    return router


def _get_project_data_router(config: APIConfig, search_router_info: Dict[str, Any]):
    """
    Returns a router with API routes for reading the project data

    Provides
    - /media/{_media_id} -> Access the original image/video/audio file from URL / disk
    - /thumbs/{_id} -> Read the thumbnail as bytes from dataset
    - /storyboard/{_media_id} -> Get a storyboard (a set of thumbnails used for the timeline hover previews in the video player UI)
    - /metadata/{_media_id} -> Read the metadata associated with a media file
    - /info -> Read the project level metadata
    """

    project = WiseProject(config.project_dir, create_project=False)
    project_assets = project.discover_assets()

    router_cm = ExitStack()
    router = APIRouter(
        on_shutdown=[lambda: print("shutting down") and router_cm.close()],
    )
    project_engine = project.db_engine
    thumbs_engine = project.thumbsdb_engine

    if config.redirect_media_url_by_path and config.redirect_media_url_num_components < 1:
        raise ValueError(
            "redirect_media_url_num_components must be greater than 0 when redirect_media_url_by_path is True"
        )

    # Pre-compute project info
    with project_engine.connect() as conn, thumbs_engine.connect() as thumbs_conn:
        num_vectors = VectorRepo.get_count(conn)
        num_media_files = MediaRepo.get_count(conn)
        media_file_counts = get_media_counts_by_media_type(conn)
        num_thumbs = ThumbnailRepo.get_count(thumbs_conn)
        total_duration = get_project_total_duration(conn)

    @router.api_route(
        "/media/{media_id}",
        response_class=Union[FileResponse, StreamingResponse],
        responses={404: {"content": "text/plain"}, 302: {}},
        methods=['GET', 'HEAD'],
    )
    def get_media_file(media_id: int, request: Request):
        """
        Returns a media file given the media_id.
        If the requested file is an image, a FileResponse is returned.
        If the requested file is a video or audio file, a StreamingResponse is returned using Range Requests of a given file
        See: https://github.com/tiangolo/fastapi/discussions/7718#discussioncomment-5143493
        """
        with project_engine.connect() as conn:
            metadata = MediaRepo.get(conn, media_id)
            if metadata is None:
                return PlainTextResponse(
                    status_code=404, content=f"{media_id} not found!"
                )
            # TODO (WISE 2) get source URI from imported_metadata table
            # # Send the source_uri if present, or try to read from source
            # # we read from
            # # Maybe do a HEAD request to check existence before redirect
            # # so that we can try to serve the file from disk if present?
            # if metadata.source_uri and is_valid_uri(metadata.source_uri):
            #     return RedirectResponse(metadata.source_uri, status_code=302)

            source_collection = SourceCollectionRepo.get(conn, metadata.source_collection_id)
            if source_collection is None:
                return PlainTextResponse(
                    status_code=404, content=f"{media_id} not found!"
                )

            file_path = Path(source_collection.location) / metadata.path

            if config.redirect_media_url_by_path:
                num_components = min(config.redirect_media_url_num_components, len(file_path.parts) - 1)
                parts = (config.redirect_media_url_prefix,) + file_path.parts[-num_components:]
                return RedirectResponse(f"{'/'.join(parts)}", status_code=302)

            if metadata.media_type in {MediaType.VIDEO, MediaType.AV, MediaType.AUDIO}:
                file_size = file_path.stat().st_size
                range_header = request.headers.get("range")

                content_type = f"{metadata.media_type.value}/{metadata.format}" if metadata.media_type == MediaType.AUDIO else f"video/mp4"
                headers = {
                    "content-type": content_type,
                    "accept-ranges": "bytes",
                    "content-length": str(file_size),
                    "access-control-expose-headers": (
                        "content-type, accept-ranges, content-length, "
                        "content-range"
                    ),
                }
                start = 0
                end = file_size - 1
                status_code = status.HTTP_200_OK

                if range_header is not None:
                    start, end = _get_range_header(range_header, file_size)
                    size = end - start + 1
                    headers["content-length"] = str(size)
                    headers["content-range"] = f"bytes {start}-{end}/{file_size}"
                    status_code = status.HTTP_206_PARTIAL_CONTENT

                return StreamingResponse(
                    send_bytes_range_requests(open(file_path, mode="rb"), start, end),
                    headers=headers,
                    status_code=status_code,
                )
            else:
                # Image files

                # Look up the source_collections table and find the location and type
                # Handle case where we read the media file from disk, but it may not be there

                location = Path(source_collection.location)

                if source_collection.type == SourceCollectionType.DIR:
                    # metadata.source_uri will be None, so we have to search for it on disk
                    file_path = location / metadata.path
                    if file_path.is_file():
                        return FileResponse(
                            file_path, media_type=f"image/{metadata.format.lower()}"
                        )
                    return PlainTextResponse(
                        status_code=404, content=f"{media_id} not found!"
                    )

                # Try to extract from local file if present
                if not location.is_file() or not tarfile.is_tarfile(location):
                    return PlainTextResponse(
                        status_code=404, content=f"{media_id} not found!"
                    )
                try:
                    file_iter = get_file_from_tar(location, metadata.path.lstrip("#"))
                    return StreamingResponse(
                        file_iter, media_type=f"image/{metadata.format.lower()}"
                    )
                except Exception as e:
                    logger.exception(f"Exception when reading image {media_id}")
                    return PlainTextResponse(
                        status_code=404, content=f"{media_id} not found!"
                    )

    @router.get(
        "/thumbnail",
        response_class=Response,
        responses={200: {"content": "image/jpeg"}, 404: {"content": "text/plain"}},
    )
    def get_thumbnail(media_id: int, timestamp: float, high_res: bool = False):
        # Get a thumbnail given a thumbnail id
        if high_res:
            return get_high_res_thumbnail(media_id, timestamp)
        else:
            with thumbs_engine.connect() as thumbs_conn:
                thumbnail = get_thumbnail_by_timestamp(
                    thumbs_conn, media_id=media_id, timestamp=timestamp
                )
                if thumbnail is None:
                    if config.use_shots:
                        # If no thumbnail found, try to get a high-res thumbnail from the original video
                        return get_high_res_thumbnail(media_id, timestamp)
                    else:
                        raise HTTPException(status_code=404, detail=f"Thumbnail not found!")
                return Response(
                    content=thumbnail,
                    media_type="image/jpeg",
                    status_code=200,
                )

    def get_high_res_thumbnail(media_id: int, timestamp: float):
        # seek the original video
        with project_engine.connect() as conn:
            metadata = MediaRepo.get(conn, media_id)
            if metadata is None:
                return PlainTextResponse(
                    status_code=404, content=f"{media_id} not found!"
                )
            # TODO (WISE 2) get source URI from imported_metadata table
            # # Send the source_uri if present, or try to read from source
            # # we read from
            # # Maybe do a HEAD request to check existence before redirect
            # # so that we can try to serve the file from disk if present?
            # if metadata.source_uri and is_valid_uri(metadata.source_uri):
            #     return RedirectResponse(metadata.source_uri, status_code=302)

            source_collection = SourceCollectionRepo.get(
                conn, metadata.source_collection_id
            )
            if source_collection is None:
                return PlainTextResponse(
                    status_code=404, content=f"{media_id} not found!"
                )

            file_path = Path(source_collection.location) / metadata.path

            audio_sampling_rate = 48_000  # (48 kHz)
            video_frame_rate = 2  # fps
            video_frames_per_chunk = 8  # frames
            segment_length = (
                video_frames_per_chunk / video_frame_rate
            )  # frames / fps = seconds
            audio_segment_length = segment_length  # seconds
            audio_frames_per_chunk = int(
                round(audio_sampling_rate * audio_segment_length)
            )
            offset = 4 * ((timestamp) // 4)
            ## dataset
            stream = AVDataset(
                [str(file_path)],
                video_frames_per_chunk=video_frames_per_chunk,
                video_frame_rate=video_frame_rate,
                audio_samples_per_chunk=audio_frames_per_chunk,
                audio_sample_rate=audio_sampling_rate,
                offset=offset,
                thumbnails=True,
            )
            for _, (mid, chunks) in enumerate(stream):
                video = chunks["video"]
                logger.debug(f"media_id: {mid}, pts: {video.pts}")
                if not video:
                    break

                if video.pts != offset:
                    continue

                if video.pts > timestamp:
                    break

                n = int((timestamp - video.pts) // 0.5)
                logger.debug(f"index: {n}, pts: {video.pts}, ts: {timestamp}")
                arr = video.tensor[n].numpy().astype(np.uint8).transpose(1, 2, 0)
                with Image.fromarray(arr) as im:
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=90)
                    return Response(
                        content=buf.getvalue(),
                        media_type="image/jpeg",
                        status_code=200,
                    )
            raise HTTPException(
                status_code=404,
                detail=f"Thumbnail for media_id {media_id} and timestamp {timestamp} not found!",
            )

    @router.get(
        "/storyboard/{_video_media_id}/{_partition}.jpg",
        response_class=Response,
        responses={
            200: {"content": "image/jpeg"},
            404: {"content": "application/json"},
        },
    )
    def get_storyboard_image(_video_media_id: int, _partition: int):
        # TODO config
        # For videos longer than 30 minutes, reduce the frequency of thumbnails to 1 every 4 seconds
        num_seconds_per_image = 2 if num_thumbs < (2 * 30 * 60) else 4
        try:
            with thumbs_engine.connect() as thumbs_conn:
                storyboard = get_thumbnail_spritesheet(
                    thumbs_conn, _video_media_id, num_seconds_per_image, _partition
                )
                buffered = io.BytesIO()
                storyboard.save(buffered, format="JPEG", quality=70)
                return Response(
                    content=buffered.getvalue(),
                    media_type="image/jpeg",
                    status_code=200,
                    headers={
                        "Cache-Control": "public, max-age=86400",
                    },
                )
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Thumbnails not found for media_id={_video_media_id} partition: {_partition}!",
            )

    @router.get(
        "/storyboard/{_video_media_id}.vtt",
        response_class=JSONResponse,
        responses={
            200: {"content": "text/vtt"},
            404: {"content": "application/json"},
        },
    )
    def get_storyboard(_video_media_id: int):
        """
        Generate JSON storyboard for a given video (as per this documentation: https://www.vidstack.io/docs/player/core-concepts/loading?styling=default-theme#json).
        A storyboard image (like this example: https://media-files.vidstack.io/storyboard.jpg)
        is generated based on the existing thumbnails of the video, and is included in the response.
        This is used for the preview thumbnails in the frontend UI when hovering over the timeline in the video player.
        """
        # For videos longer than 30 minutes, reduce the frequency of thumbnails to 1 every 4 seconds
        num_seconds_per_image = 2 if num_thumbs < (2 * 30 * 60) else 4
        try:
            with thumbs_engine.connect() as thumbs_conn:
                vtt_content = get_webvtt_spritesheet(
                    thumbs_conn, _video_media_id, num_seconds_per_image
                )
                return Response(
                    content=vtt_content,
                    status_code=200,
                    media_type="text/vtt",
                    headers={
                        "Cache-Control": "public, max-age=86400",
                    },
                )
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Thumbnails not found for media_id={_video_media_id}!",
            )

    @router.get(
        "/metadata/{_id}",
        response_model=MediaMetadata,
        response_model_exclude=set(["id", "source_collection_id", "size_in_bytes", "date_modified"]),
        responses={200: {"content": "application/json"}},
    )
    def get_metadata(_id: int):
        with project_engine.connect() as conn:
            metadata = MediaRepo.get(conn, _id)
            if metadata is None:
                raise HTTPException(status_code=404, detail=f"Metadata not found!")
            return metadata

    models = {
        media_type: [
            feature_extractor_id for feature_extractor_id in project_assets[media_type]
        ] for media_type in project_assets
    }

    @router.get("/info")
    def get_info():
        return {
            "project_name": config.project_dir.stem,
            "models": models,
            "search_targets": search_router_info['search_targets'],
            "shot_based_filters": search_router_info['shot_based_filters'],
            "num_vectors": num_vectors,
            "num_media_files": num_media_files, # Total number of media files
            "media_file_counts": media_file_counts, # Number of media files by media type
            "total_duration": total_duration,
        }

    return router


def _get_report_image_router(config: APIConfig):
    router_cm = ExitStack()
    router = APIRouter(
        on_shutdown=[lambda: print("shutting down") and router_cm.close()],
    )

    @router.post("/report")
    def report_image(
        file_queries: List[bytes] = File([]),
        url_queries: List[str] = Form([]),
        text_queries: List[str] = Form([]),
        sourceURI: str = Form(),
        reasons: List[str] = Form([]),
    ):
        # TODO implement code to store data in database
        # For now, we are saving the reports in a CSV file
        report_filename = "data/reported_images.csv"
        fieldnames = [
            "text_queries",
            "url_queries",
            "file_queries",
            "sourceURI",
            "reasons",
        ]

        # Write header row if the file doesn't exist
        if not os.path.exists(report_filename):
            os.makedirs(os.path.dirname(report_filename), exist_ok=True)
            with open(report_filename, "a", newline="") as report_file:
                csv.writer(report_file).writerow(fieldnames)

        # Write data row
        with open(report_filename, "a", newline="") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=fieldnames)
            writer.writerow(
                {
                    "text_queries": json.dumps(text_queries),
                    "url_queries": json.dumps(url_queries),
                    "file_queries": json.dumps(
                        # to prevent the CSV file from getting too large, we store a placeholder text ('uploaded image')
                        # instead of storing the image file
                        ["uploaded image" for _ in file_queries]
                    ),
                    "sourceURI": sourceURI,
                    "reasons": json.dumps(reasons),
                }
            )

        return PlainTextResponse(status_code=200, content="Image has been reported")

    return router


def _get_search_router(config: APIConfig):
    project = WiseProject(config.project_dir)
    index_type = IndexType[config.index_type]

    # Metadata for a video/audio/image file, to be sent to the frontend
    class MediaInfo(BaseModel):
        id: str
        filename: str
        width: int
        height: int
        media_type: str
        format: str
        duration: float
        title: str = ""
        external_metadata: dict = {}

    # A subclass of MediaInfo for images
    class ImageInfo(MediaInfo):
        pass

    # A subclass of MediaInfo for videos
    class VideoInfo(MediaInfo):
        timeline_hover_thumbnails: str

    class BBoxXYWH(BaseModel):
        x: float
        y: float
        w: float
        h: float

        @field_validator("x", "y", "w", "h")
        @classmethod
        def round_bbox(cls, v):
            return round(v, config.precision)

    class VectorInfo(BaseModel):
        vector_id: str
        media_id: str
        link: str
        thumbnail: str
        bbox: Optional[BBoxXYWH] = None

        @field_validator("bbox", mode="before")
        @classmethod
        def cast_bbox(cls, v):
            if v is None:
                return v
            elif isinstance(v, BBoxXYWH):
                return v
            else:  # v is the NamedTuple in feature_extractor module
                return BBoxXYWH(**{k: v for (k, v) in zip('xywh', v)})

    class VectorResult(VectorInfo):
        distance: float

        @field_validator("distance")
        @classmethod
        def round_distance(cls, v):
            return round(v, config.precision)

    class ImageVector(VectorResult):
        pass

    class VideoSegment(VectorResult):
        ts: float
        te: float
        thumbnail_ts: float

    class VideoAudioResults(BaseModel):
        total: int # maximum number of unmerged_windows that can be returned
        unmerged_windows: List[VideoSegment] # e.g. 7-second windows
        merged_windows: List[VideoSegment] # shots (for edited videos) or merged segments (for unedited videos)
        videos: Dict[str, VideoInfo]

    class VideoResults(BaseModel):
        total: int # maximum number of unmerged_windows that can be returned
        unmerged_windows: List[VideoSegment] # frames (CLIP) or unmerged 4-second segments (InternVideo/LanguageBind)
        merged_windows: List[VideoSegment] # shots (for edited videos) or merged segments (for unedited videos)
        videos: Dict[str, VideoInfo]

    class ImageResults(BaseModel):
        total: int # maximum number of images that can be returned e.g. min(1000, num_images_in_project)
        vectors: List[ImageVector]
        images: Dict[str, ImageInfo]

    class SearchResponse(BaseModel):
        time: float # backend search time in seconds
        video_audio_results: Optional[VideoAudioResults] # search results from audio stream of video files
        video_results: Optional[VideoResults] # search results from video stream of video files
        image_results: Optional[ImageResults] # search results from image files

    _prefix = {
        MediaType.IMAGE: config.query_prefix.strip(),
        MediaType.VIDEO: config.query_prefix.strip(),
        MediaType.AV: "This is the sound of", # TODO add this to config
        MediaType.AUDIO: "This is the sound of",
    }
    project_assets = project.discover_assets()
    project_engine = project.db_engine
    thumbs_engine = project.thumbsdb_engine
    external_metadata_tables = db.reflect_external_metadata(project_engine)

    shots_table = db.shots_table

    if config.use_shots:

        with project_engine.connect() as conn:
            shots_count = conn.execute(sa.select(sa.func.count(shots_table.c.id))).scalar()

        if shots_count == 0:
            logger.warning('use_shots is set to True, but shots table is empty! Please make sure to populate the shots table before using this feature.')

    def merge_close_segments(_keyframes: List[VideoSegment]):
        """
        Takes a list of segments of a media file and merges them if they are close - within 4 seconds of each other
        The merged segment is represented by the best matching segment based on distance
        """
        merged_segments: List[VideoSegment] = []
        start = None
        current = None
        best = None
        for k in _keyframes:
            if start is None:
                # Start a new group
                start = k
                current = k
                best = k

            elif (k.ts - current.te) <= 4:
                current = k
                if current.distance > best.distance:
                    best = current

            else:
                merged_segments.append(
                    VideoSegment(
                        vector_id=best.vector_id,
                        media_id=best.media_id,
                        ts=start.ts,
                        te=current.te,
                        link=f"media/{best.media_id}#t={start.ts},{current.te}",
                        distance=best.distance,
                        thumbnail=best.thumbnail,
                        thumbnail_ts=best.thumbnail_ts,
                        bbox=best.bbox,
                    )
                )
                start = k
                current = k
                best = k

        if start is not None:
            merged_segments.append(
                VideoSegment(
                    vector_id=best.vector_id,
                    media_id=best.media_id,
                    ts=start.ts,
                    te=current.te,
                    link=f"media/{best.media_id}#t={start.ts},{current.te}",
                    distance=best.distance,
                    thumbnail=best.thumbnail,
                    thumbnail_ts=best.thumbnail_ts,
                    bbox=best.bbox,
                )
            )

        return merged_segments

    def get_shots_from_segments(
            segments: List[VideoSegment],
            merge_function: Callable[[list[VideoSegment]], list[VideoSegment]] = merge_close_segments
        ):
        """
        Functions that takes a list of segments and returns a list of merged segments
        based on the merge function passed in

        The merge function by default merges close segments
        """
        # Sort by video_id, timestamp
        sorted_segments = sorted(segments, key=lambda x: (x.media_id, x.ts))

        # for each key, apply merge logic
        all_merged_segments = []
        for _, g in itertools.groupby(sorted_segments, key=lambda x: x.media_id):
            merged_segments = merge_function(list(g))
            all_merged_segments.extend(merged_segments)

        # sort the merged segments by distance
        all_merged_segments = sorted(
            all_merged_segments,
            key=lambda x: x.distance,
            reverse=True,
        )
        return all_merged_segments

    # configure lookup functions
    def query_shot_by_timestamp(conn, *, media_id: int, timestamp: float):
        # Join the table and query by dataset_path, and return the id

        start_timestamp_expr = (timestamp + 0.2) >= shots_table.c.ts
        end_timestamp_expr = timestamp < shots_table.c.te
        dataset_expr = shots_table.c.media_id == media_id
        stmt = sa.select(shots_table.c.id, shots_table.c.media_id, shots_table.c.ts, shots_table.c.te).where(
            (dataset_expr & start_timestamp_expr & end_timestamp_expr)
        )
        result = conn.execute(stmt)
        for row in result.mappings():
            yield VideoShot.model_validate(row)

    def keyframes_to_shots(_keyframes: List[VideoSegment]):
        """
        Get Shot corresponding to a keyframe
        """
        with project_engine.connect() as shots_conn:
            # Assuming segment maps to only one shot (no duplicates)
            return list(
                next(
                    query_shot_by_timestamp(
                        shots_conn, media_id=int(x.media_id), timestamp=x.ts
                    ),
                    None,
                )
                for x in _keyframes
            )

    def get_shots_from_keyframes(_keyframes: List[VideoSegment]):
        """
        Get unique shots from list of keyframes belonging to a single video
        """
        # Input are keyframes from same video
        shots = keyframes_to_shots(_keyframes)

        # Ignore where segment doesn't have a shot mapping
        shots_iter = filter(lambda x: x[0], zip(shots, _keyframes))

        shots_list = []
        for _, g in itertools.groupby(shots_iter, key=lambda x: x[0].id):
            shot_group, keyframes_group = list(zip(*g))
            _shot = shot_group[0]

            # Computing best matching segment and best thumbnail to represent the segment with
            # For clip case, it will be same. For internvideo it will be different
            best_segment = sorted(
                keyframes_group, key=lambda x: x.distance, reverse=True
            )[0]
            shots_list.append(
                VideoSegment(
                    vector_id=best_segment.vector_id,
                    media_id=best_segment.media_id,
                    ts=_shot.ts,
                    te=_shot.te,
                    link=f"media/{best_segment.media_id}#t={_shot.ts},{_shot.te}",
                    distance=best_segment.distance,
                    thumbnail=best_segment.thumbnail,
                    thumbnail_ts=best_segment.thumbnail_ts,
                    bbox=best_segment.bbox,
                )
            )
        return shots_list

    def construct_video_search_response(
        search_in: MediaType,
        top_dist: List[float],
        all_metadata: List[VectorAndMediaMetadata],
        all_ext_metadata: list[FeatureExtMetadata],
        get_thumbs_fn: Callable[[List[VectorAndMediaMetadata]], Iterable[Tuple[str, float]]],
        merge_function: Callable[[list[VideoSegment]], list[VideoSegment]],
    ):
        videos = {}
        shots = []
        segments = []
        for _dist, _metadata, _ext_metadata, (_thumb, _) in zip(
            top_dist,
            all_metadata,
            all_ext_metadata,
            get_thumbs_fn(all_metadata),
        ):
            video_id = str(_metadata.media_id)
            if video_id not in videos:
                videos[video_id] = VideoInfo(
                    id=video_id,
                    filename=_metadata.path,
                    width=_metadata.width,
                    height=_metadata.height,
                    media_type=_metadata.media_type,
                    format=_metadata.format,
                    duration=_metadata.duration,
                    timeline_hover_thumbnails=f"storyboard/{video_id}.vtt",
                    external_metadata=_metadata.external_metadata,
                )
            ts = _metadata.timestamp
            te = _metadata.end_timestamp
            if ts is None:
                logger.error(f"ts is None for vector {_metadata.id}")
            if te is None:
                te = ts

            if ts == te:
                te = ts + 4.0

            segment = VideoSegment(
                vector_id=str(_metadata.id),
                media_id=video_id,
                ts=float(ts),
                te=float(te),
                link=f"media/{video_id}#t={ts},{te}", # f"{_metadata.source_uri if _metadata.source_uri else f'media/{video_id}{_metadata.path}'}",
                distance=_dist,
                thumbnail=_thumb,
                thumbnail_ts=float(ts),
                bbox=_ext_metadata.bbox,
            )

            segments.append(segment)

        shots = get_shots_from_segments(segments, merge_function=merge_function)

        if search_in == MediaType.VIDEO:
            return VideoResults(
                total=300, # TODO change this
                unmerged_windows=segments,
                merged_windows=shots,
                videos=videos,
            )
        elif search_in == MediaType.AV:
            return VideoAudioResults(
                total=300, # TODO change this
                unmerged_windows=segments,
                merged_windows=shots,
                videos=videos,
            )
        else:
            raise ValueError("`search_in` must be either `MediaType.VIDEO` or `MediaType.AV`")

    def construct_image_search_response(
        top_dist: List[float],
        all_metadata: List[VectorAndMediaMetadata],
        all_ext_metadata: list[FeatureExtMetadata],
        get_thumbs_fn: Callable[[List[VectorAndMediaMetadata]], Iterable[Tuple[str, float]]],
    ):
        images = {}
        image_vectors = []
        for _dist, _metadata, _ext_metadata, (_thumb, _) in zip(
            top_dist,
            all_metadata,
            all_ext_metadata,
            get_thumbs_fn(all_metadata),
        ):
            image_id = str(_metadata.media_id)
            if image_id not in images:
                images[image_id] = ImageInfo(
                    id=image_id,
                    filename=_metadata.path,
                    width=_metadata.width,
                    height=_metadata.height,
                    media_type=_metadata.media_type,
                    format=_metadata.format,
                    duration=_metadata.duration,
                    external_metadata=_metadata.external_metadata,
                )

            image_vector = ImageVector(
                vector_id=str(_metadata.id),
                media_id=image_id,
                link=f"media/{image_id}",
                distance=_dist,
                thumbnail=_thumb,
                bbox=_ext_metadata.bbox,
            )
            image_vectors.append(image_vector)

        return ImageResults(
            total=300, # TODO change this
            vectors=image_vectors,
            images=images,
        )

    def construct_search_response(
        top_dist: List[float],
        top_ids: List[int],
        get_metadata_fn: Callable[[List[int]], List[VectorAndMediaMetadata]],
        get_ext_metadata_fn: Callable[[List[int]], list[FeatureExtMetadata]],
        get_thumbs_fn: Callable[[List[VectorAndMediaMetadata]], Iterable[Tuple[str, float]]],
        merge_function: Callable[[list[VideoSegment]], list[VideoSegment]] = merge_close_segments,
        search_in: MediaType = None,
    ):
        all_metadata = get_metadata_fn(top_ids)
        all_ext_metadata = get_ext_metadata_fn(top_ids)
        video_audio_results = None
        video_results = None
        image_results = None
        if search_in is None or search_in == MediaType.IMAGE:
            image_indices = [i for i, x in enumerate(all_metadata) if x.modality == ModalityType.IMAGE]
            if len(image_indices) > 0:
                image_top_dist = [top_dist[i] for i in image_indices]
                image_all_metadata = [all_metadata[i] for i in image_indices]
                image_ext_metadata = [all_ext_metadata[i] for i in image_indices]
                image_results = construct_image_search_response(image_top_dist, image_all_metadata, image_ext_metadata, get_thumbs_fn)
        if search_in is None or search_in == MediaType.VIDEO:
            video_indices = [i for i, x in enumerate(all_metadata) if x.modality == ModalityType.VIDEO]
            if len(video_indices) > 0:
                video_top_dist = [top_dist[i] for i in video_indices]
                video_all_metadata = [all_metadata[i] for i in video_indices]
                video_ext_metadata = [all_ext_metadata[i] for i in video_indices]
                video_results = construct_video_search_response(MediaType.VIDEO, video_top_dist, video_all_metadata, video_ext_metadata, get_thumbs_fn, merge_function)
        if search_in is None or search_in == MediaType.AV:
            av_indices = [i for i, x in enumerate(all_metadata) if x.modality == ModalityType.AUDIO and x.media_type == MediaType.AV]
            if len(av_indices) > 0:
                av_top_dist = [top_dist[i] for i in av_indices]
                av_all_metadata = [all_metadata[i] for i in av_indices]
                av_ext_metadata = [all_ext_metadata[i] for i in av_indices]
                video_audio_results = construct_video_search_response(MediaType.AV, av_top_dist, av_all_metadata, av_ext_metadata, get_thumbs_fn, merge_function)
        if search_in is not None and search_in not in [MediaType.IMAGE, MediaType.VIDEO, MediaType.AV]:
            raise NotImplementedError("`search_in` must be either `MediaType.IMAGE`, `MediaType.VIDEO`, or `MediaType.AV`. Support for `MediaType.AUDIO` is not available yet")

        return SearchResponse(
            time=0.0, # Dummy value to be overwritten by the @add_response_time decorator function
            video_audio_results=video_audio_results,
            video_results=video_results,
            image_results=image_results,
        )

    def _get_query_features(
        query_prefix: str,
        q: List[Dict[str, Union[ndarray, bytes, str]]],
        extract_features_from_text: Callable[[List[str]], ndarray] = None,
        extract_features_from_image: Callable[[List[Image.Image]], ndarray] = None,
        extract_features_from_audio: Callable[[List[torch.Tensor]], ndarray] = None,
    ) -> ndarray:
        feature_vectors = []
        weights = []

        for query_dict in q:
            query = query_dict["val"]
            feature_vector = None
            if query_dict['modality'] == 'image':
                if isinstance(query, bytes):
                    with Image.open(io.BytesIO(query)) as im:
                        im = im.convert('RGB')
                        feature_vector = extract_features_from_image([im])
                        weights.append(
                            config.negative_queries_weight
                            if query_dict["sign"] == "negative"
                            else 1
                        )
                elif isinstance(query, ndarray):
                    feature_vector = query
                    weights.append(
                        config.negative_queries_weight
                        if query_dict["sign"] == "negative"
                        else 1
                    )
                elif query.startswith(("http://", "https://")):
                    logger.info("Downloading %s to file", query)
                    with NamedTemporaryFile() as tmpfile:
                        download_url_to_file(query, tmpfile.name)
                        with Image.open(tmpfile.name) as im:
                            im = im.convert('RGB')
                            feature_vector = extract_features_from_image([im])
                            weights.append(
                                config.negative_queries_weight
                                if query_dict["sign"] == "negative"
                                else 1
                            )
            elif query_dict['modality'] == 'audio':
                if isinstance(query, bytes):
                    im = io.BytesIO(query)
                    feature_vector = extract_features_from_audio([im])
                    weights.append(
                        config.negative_queries_weight
                        if query_dict["sign"] == "negative"
                        else 1
                    )
                elif isinstance(query, ndarray):
                    feature_vector = query
                    weights.append(
                        config.negative_queries_weight
                        if query_dict["sign"] == "negative"
                        else 1
                    )
                elif query.startswith(("http://", "https://")):
                    logger.info("Downloading", query, "to file")
                    with NamedTemporaryFile() as tmpfile:
                        download_url_to_file(query, tmpfile.name)
                        with open(tmpfile.name, mode='rb') as f:
                            file_bytes_io = io.BytesIO(f.read())
                            feature_vector = extract_features_from_audio([file_bytes_io])
                            weights.append(
                                config.negative_queries_weight
                                if query_dict["sign"] == "negative"
                                else 1
                            )
            elif query_dict['modality'] == 'text':
                if query_prefix:
                    prefixed_queries = f"{query_prefix} {query.strip()}".strip()
                else:
                    prefixed_queries = query.strip()
                feature_vector = extract_features_from_text([prefixed_queries])
                weights.append(
                    config.text_queries_weight
                    * (  # assign higher weight to natural language queries
                        config.negative_queries_weight
                        if query_dict["sign"] == "negative"
                        else 1
                    )
                )
            else:
                raise ValueError(f"Unsupported modality: {query_dict['modality']}")

            if query_dict["sign"] == "negative":
                feature_vector = -feature_vector
            feature_vectors.append(feature_vector)
        weights = array(weights, dtype=float32)
        average_features = average(feature_vectors, axis=0, weights=weights)
        average_features /= norm(average_features, axis=-1, keepdims=True)
        return average_features

    """
    Load all available search indices by default
    `search_indices` is a dictionary of SearchIndex objects, where the key is the
    feature_extractor_id and value is a SearchIndex object
    """
    search_indices: dict[str, dict[str, SearchIndex]] = {}
    active_search_targets: dict[str, list[str]] = {}

    db_inspector = sa.inspect(project_engine)
    fts_search_index = None
    if db_inspector.has_table(db._WISE_FTS_TABLE):
        db.project_metadata_obj.reflect(bind=project_engine, only=[db._WISE_FTS_TABLE])
        fts_search_index = FTSSearch(project, db.project_metadata_obj)
        # search_indices[MediaType.VIDEO] = {'wise/metadata': fts_search_index}
        # active_search_targets[MediaType.VIDEO] = ['wise/metadata']

    feature_extractors = {}
    for media_type in project_assets:
        if media_type not in {MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO}:
            # Added to ensure projects created with older versions
            # remain compatible (TODO: remove this in the future)
            logger.warning(
                f"Media type {media_type} is not supported. "
                "Please use IMAGE, VIDEO, or AUDIO media types."
            )
            continue
        for feature_extractor_id in project_assets[media_type]:
            if media_type not in search_indices:
                search_indices[media_type] = {}
                active_search_targets[media_type] = []

            if feature_extractor_id not in feature_extractors:
                feature_extractors[feature_extractor_id] = FeatureExtractorFactory(
                    feature_extractor_id,
                    warmup=config.mode != 'development',
                )
            search_indices[media_type][feature_extractor_id] = SearchIndexFactory(
                media_type,
                feature_extractor_id,
                project_assets[media_type][feature_extractor_id],
                feature_extractors[feature_extractor_id],
            )
            asset = project_assets[media_type][feature_extractor_id]
            index_type_to_load = config.index_type

            if index_type_to_load:
                # check if the preferred index type is available
                preferred_index_filename = search_indices[media_type][feature_extractor_id].get_index_filename(index_type_to_load)
                if not os.path.exists(preferred_index_filename):
                    logger.warning(f"Index file not found for preferred index type {index_type_to_load}. Will try to load any other available index.")
                    index_type_to_load = None

            if not index_type_to_load:
                # load any available index
                available_indices = [
                    f for f in asset['index_files'] if f.endswith('.faiss')
                ]
                if available_indices:
                    # extract index type from filename, e.g. "video-IndexFlatIP.faiss" -> "IndexFlatIP"
                    index_type_to_load = Path(available_indices[0]).stem.split('-')[1]
                    logger.info(f"Loading available index of type {index_type_to_load}")
                else:
                    logger.error(f"No index files found for {media_type} and {feature_extractor_id}")
                    del search_indices[media_type][feature_extractor_id]
                    continue

            logger.info(f"Loading faiss index from {search_indices[media_type][feature_extractor_id].get_index_filename(index_type_to_load)}")
            if not search_indices[media_type][feature_extractor_id].load_index(index_type_to_load, project_engine):
                print(f'failed to load {media_type} index: {feature_extractor_id}')
                del search_indices[media_type][feature_extractor_id]
                continue
            active_search_targets[media_type].append(feature_extractor_id)
            if hasattr(search_indices[media_type][feature_extractor_id].index, "nprobe"):
                # See https://github.com/facebookresearch/faiss/blob/43d86e30736ede853c384b24667fc3ab897d6ba9/faiss/IndexIVF.h#L184C8-L184C42
                search_indices[media_type][feature_extractor_id].index.parallel_mode = 1
                search_indices[media_type][feature_extractor_id].index.nprobe = getattr(config, "nprobe", 32)

                if not search_indices[media_type][feature_extractor_id].is_internal_search_supported:
                    logger.info(
                        "This faiss index does not support internal search. To enable "
                        "internal search, please re-create the index by running "
                        f"`python create-index.py --project-dir \"{config.project_dir}\" --media-type {media_type} --index-type {search_indices[media_type][feature_extractor_id].index_type} --overwrite`",
                    )
        # TODO: Fix this to handle audio when support gets added
        if fts_search_index is not None and media_type in {MediaType.IMAGE, MediaType.VIDEO}:
            search_indices[media_type]['wise/metadata'] = fts_search_index
            active_search_targets[media_type].append('wise/metadata')  

    is_audio_only_project = (
        MediaType.VIDEO not in search_indices
        and MediaType.AUDIO in search_indices
        and len(active_search_targets[MediaType.AUDIO]) > 0
    )
    if fts_search_index is not None and is_audio_only_project:
        search_indices[MediaType.AUDIO]['wise/metadata'] = fts_search_index
        active_search_targets[MediaType.AUDIO].append('wise/metadata')  

    # sort active search targets based on user defined order in config.search_target_order
    search_target_order = ["open_clip", "insightface", "owlv2", "clap", "wise/metadata"]
    if hasattr(config, "search_target_order") and config.search_target_order:
        search_target_order = config.search_target_order
    for media_type in active_search_targets:
        def sort_key(x):
            for i, partial in enumerate(search_target_order):
                if partial in x:
                    return i
            return len(search_target_order)
        active_search_targets[media_type].sort(key=sort_key)

    preferred_order = [MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO]
    active_search_targets = {
        x: active_search_targets[x]
        for x in sorted(
            active_search_targets.keys(), key=lambda x: preferred_order.index(x)
        )
    }
    logger.info("Loaded the following search indices:\n%s", json.dumps(active_search_targets, indent=4))

    # Get counts
    with project_engine.connect() as conn:
        num_vectors = VectorRepo.get_count(conn)

    # Shot property (e.g. shot_scale, camera_motion, etc.) based filters
    shot_based_filters = {}
    if config.use_shots and db_inspector.has_table(db.shots_table.name):
        # check if the shots table contains a column named "shot_scale"
        colnames = [col["name"] for col in db_inspector.get_columns(db.shots_table.name)]
        if 'shot_scale' in colnames:
            if not db_inspector.has_table('vectors_to_shots_map'):
                raise ValueError("vectors_to_shots_map table not found! Please run the import shots script as follows:"
                                 "Please run \"python3 media-metadata.py import-shot-scale ...\"")
            # find the distinct values of this column
            with project_engine.connect() as conn:
                shot_scales = conn.execute(sa.text("select distinct(shot_scale) from shots ORDER BY shot_scale")).fetchall()
            shot_scales = [row[0] for row in shot_scales if row[0] is not None]
            if shot_scales:
                logger.info("shot_scale filter enabled with values =%s", shot_scales)
                shot_based_filters["shot_scale"] = {
                    "name": "Shot Scale",
                    "description": "Filter by the scale (or size) of the shot in edited videos.",
                    "options": shot_scales
                }

            db.project_metadata_obj.reflect(bind=project_engine, only=['vectors_to_shots_map'])
            vectors_to_shots_map = db.project_metadata_obj.tables['vectors_to_shots_map']
    router_cm = ExitStack()

    def _thumbs_with_score(conn: sa.Connection, dist: List[float], thumbnails_to_send: int):
        def _thumbnail_url(_m: VectorAndMediaMetadata):
            return f"thumbnail?media_id={_m.media_id}&timestamp={_m.timestamp}"

        def _thumbnail(_conn: sa.Connection, _m: VectorAndMediaMetadata):
            thumbnail = get_thumbnail_by_timestamp(
                _conn, media_id=_m.id, timestamp=_m.timestamp
            )
            return convert_uint8array_to_base64(thumbnail)

        def inner(vector_and_media_metadata_list: List[VectorAndMediaMetadata]):
            thumbs = [
                (
                    _thumbnail(conn, vector_and_media_metadata)
                    if i < thumbnails_to_send
                    else _thumbnail_url(vector_and_media_metadata)
                )
                for i, vector_and_media_metadata in enumerate(
                    vector_and_media_metadata_list
                )
            ]
            return zip(thumbs, dist)

        return inner

    if config.thumbnail_project_dir:
        # TODO update the code below
        raise NotImplementedError()
        # # from project load up the model
        # thumbnail_project_tree = WiseProjectTree(config.thumbnail_project_id)
        # thumbnail_project_engine = db.init_project(thumbnail_project_tree.dburi)

        # # TODO Big assumption - all datasets were written with same model name
        # # Should Read / Write to project db instead
        # # Get model name
        # thumbnail_vds = thumbnail_project_tree.latest
        # thumbnail_model_name = CLIPModel[get_model_name(thumbnail_vds)]

        # (
        #     _,
        #     _,
        #     extract_image_features_for_thumbnail,
        #     extract_text_features_for_thumbnail,
        # ) = setup_clip(thumbnail_model_name)

        # # load the feature and thumbnail reader
        # _thumbnail_feature_reader, _thumbnail_thumbs_reader = [
        #     router_cm.enter_context(get_h5reader(thumbnail_vds)(x))
        #     for x in (H5Datasets.IMAGE_FEATURES, H5Datasets.THUMBNAILS)
        # ]

        # get_query_features_for_thumbnail = functools.partial(
        #     _get_query_features,
        #     extract_image_features_for_thumbnail,
        #     extract_text_features_for_thumbnail,
        #     "this is a photo of",
        # )

        # # set up the functions
        # def _get_matching_thumbnails(
        #     conn: sa.Connection,
        #     _q: Dict[str, Union[str, ndarray, bytes]],
        #     _dist,
        # ):
        #     def inner(_ids: List[int]):
        #         # Read metadata from current project, get timestamp
        #         id_map = {}
        #         with thumbnail_project_engine.connect() as tconn:
        #             for m in get_records(conn, [1 + x for x in _ids]):
        #                 # Make range query to thumbnail project to get equivalent thumbnail ids
        #                 ts = m.metadata.get(
        #                     "start_timestamp", m.metadata.get("timestamp", 0)
        #                 )
        #                 te = m.metadata.get(
        #                     "end_timestamp", m.metadata.get("timestamp", 9999)
        #                 )

        #                 tmid = query_by_timestamp(
        #                     tconn, location=m.location, timestamp=[ts, te]
        #                 )
        #                 id_map[m.id - 1] = [(x - 1) for x in tmid]

        #         # Get features from h5
        #         thumbnail_ids = [x for v in id_map.values() for x in v]
        #         thumbnail_features = _thumbnail_feature_reader(thumbnail_ids)

        #         # compute dot product
        #         query_feature = get_query_features_for_thumbnail(_q)

        #         offset = 0

        #         thumbnail_id_map = []
        #         own_thumbnail_ids = []
        #         own_counter = 0
        #         external_thumbnail_ids = []
        #         external_counter = 0
        #         final_scores = []
        #         # TODO Fails with IndexIVF
        #         for i, (k, v) in enumerate(id_map.items()):
        #             if len(v) == 0:
        #                 # Thumbnail not found in the other project
        #                 # Use own thumbnail
        #                 own_thumbnail_ids.append(k)
        #                 thumbnail_id_map.append(("own", own_counter))
        #                 final_scores.append(_dist[i])
        #                 own_counter += 1
        #             else:
        #                 start = offset
        #                 end = offset + len(v)

        #                 tdist, tids = brute_force_search(
        #                     [np.stack(thumbnail_features[start:end])],
        #                     query_feature,
        #                     top_k=1,
        #                 )
        #                 tidx = int(tids[0, 0])
        #                 final_scores.append(round(float(tdist[0, 0]), 3))
        #                 external_thumbnail_ids.append(v[tidx])
        #                 thumbnail_id_map.append(("external", external_counter))
        #                 external_counter += 1

        #                 offset = end

        #         # get thumbnails
        #         if len(own_thumbnail_ids) == 0:
        #             return list(
        #                 zip(
        #                     _thumbnail_thumbs_reader(external_thumbnail_ids),
        #                     final_scores,
        #                 )
        #             )
        #         elif len(external_thumbnail_ids) == 0:
        #             return list(zip(_thumbs_reader(own_thumbnail_ids), final_scores))

        #         # Both cases
        #         own_thumbnails = _thumbs_reader(own_thumbnail_ids)
        #         external_thumbnails = _thumbnail_thumbs_reader(external_thumbnail_ids)
        #         _thumbnail_container = lambda x: (
        #             external_thumbnails if x == "external" else own_thumbnails
        #         )
        #         return list(
        #             zip(
        #                 [_thumbnail_container(x)[idx] for (x, idx) in thumbnail_id_map],
        #                 final_scores,
        #             )
        #         )

        #     return inner

        # thumbs_reader = _get_matching_thumbnails
    else:
        thumbs_reader = _thumbs_with_score

    router = APIRouter(
        on_shutdown=[lambda: print("shutting down") and router_cm.close()],
    )

    def reconstruct_internal_img_feature(search_index: SearchIndex, vector_ids: List[int]) -> List[Union[ndarray, bytes]]:
        reconstructed_features = search_index.index.reconstruct_batch(vector_ids)
        features_list = []
        for i in range(0, reconstructed_features.shape[0]):
            features_list.append( expand_dims(reconstructed_features[i,], axis=0) )
        return features_list

    def load_internal_images(vector_ids: List[int]) -> List[Union[ndarray, bytes]]:
        # a list of ndarrays (feature vectors) or bytes (from image file)
        internal_images_loaded = []
        with project_engine.connect() as conn:
            for image_id in vector_ids:
                # Try to read feature vector from h5 dataset
                try:
                    # TODO (WISE 2) update the code below
                    with get_h5reader(vds_path)(
                        H5Datasets.IMAGE_FEATURES
                    ) as image_features_reader:
                        # This is an np.ndarray of shape: (output_dim,) e.g. (768,)
                        image_features = image_features_reader([image_id])[0]
                        # Add batch dimension so the shape becomes (1, output_dim) e.g. (1, 768)
                        image_features = expand_dims(image_features, axis=0)  
                        internal_images_loaded.append(image_features)
                        continue
                except Exception:
                    logger.info(
                        f"Could not retrieve feature vector for image {image_id} from h5 dataset. Attempting to re-compute features from original image"
                    )
                    pass

                # Fallback: read the original image from disk and re-compute the features
                # Get metadata from media and source_collections table to locate the file
                metadata = MediaRepo.get(conn, image_id)
                if metadata is None:
                    raise FileNotFoundError(
                        f"Image {image_id} not found in metadata database"
                    )
                source_collection = SourceCollectionRepo.get(conn, metadata.source_collection_id)
                if source_collection is None:
                    raise LookupError(f"Source collection not found for image {image_id}")

                location = Path(source_collection.location)

                if source_collection.type == SourceCollectionType.DIR:
                    # Try to read image from disk if present
                    # metadata.source_uri will be None, so we have to search for it on disk
                    file_path = location / metadata.path
                    if file_path.is_file():
                        with open(file_path, "rb") as f:
                            internal_images_loaded.append(f.read())
                    else:
                        raise FileNotFoundError(
                            f"Image file for image {image_id} does not exist or is not a regular file"
                        )
                else:
                    # Try to extract from local file if present
                    if not location.is_file() or not tarfile.is_tarfile(location):
                        raise FileNotFoundError(
                            f"WebDataset tar file (for image {image_id}) does not exist or is not a tar file"
                        )
                    try:
                        with tarfile.open(location, "r") as t:
                            buf = t.extractfile(metadata.path.lstrip("#"))
                            internal_images_loaded.append(buf.read())
                    except Exception as e:
                        logger.exception(f"Exception when reading image {image_id}")
                        raise FileNotFoundError(
                            f"Error extracting image {image_id} from WebDataset tar file"
                        )
        return internal_images_loaded

    def add_response_time(func: Callable[..., Awaitable[SearchResponse]]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            response = await func(*args, **kwargs)
            end_time = time.perf_counter()
            response.time = end_time - start_time
            return response

        return wrapper

    # Generate a list of random featured images for each modality and feature extractor
    ids: dict[str: dict[str: list[int]]] = {}
    with project_engine.connect() as conn:
        for modality in search_indices:
            ids[modality] = {}
            for feature_extractor_id in search_indices[modality]:
                this_ids = get_featured_images(
                    conn, modality, feature_extractor_id, config.use_shots
                )

                # Select a random subset of up to 10000 image ids (for performance reasons)
                default_rng(seed=42).shuffle(this_ids)
                ids[modality][feature_extractor_id] = this_ids[:10000]
                del this_ids

    @router.get("/featured", response_model=SearchResponse)
    @add_response_time
    async def handle_get_featured(
        # The "media type" used for "featured_in" does not actually
        # refer to the media_type in the database.  It is actually
        # closer, but not the same, to the frontend viewModality but
        # we use MediaType because it uses a subset of its keys.  This
        # is just a convenience to get the values checked.
        featured_in: MediaType = Query(),
        feature_extractor_id: str = Query(),
        start: int = Query(0, ge=0, le=980),
        end: int = Query(20, gt=0, le=1000),
        thumbnails_to_send: int = Query(0),
        # This seed is used to randomly select the set of images used for the featured images
        random_seed: int = Query(123),
    ):
        if featured_in == MediaType.AV:
            modality = ModalityType.AUDIO
        else:
            modality = ModalityType(featured_in)
        with project_engine.connect() as conn, thumbs_engine.connect() as thumbs_conn:
            # Select up to 1000 random image ids, using the specified random seed, from the set of 10000 ids
            if feature_extractor_id == "wise/metadata":
                # use other feature_extractor_id from the modality because "wise/metadata"
                # is a FTS search index and it does not support "feature_extractor.get_vector_metadata()"
                other_ids = [fid for fid in active_search_targets[modality] if fid != "wise/metadata"]
                if not other_ids:
                    selected_ids = [] # i.e featured images not available
                else:
                    selected_ids = ids[modality][other_ids[0]].copy()
            else:
                selected_ids = ids[modality][feature_extractor_id].copy()
            default_rng(seed=random_seed).shuffle(selected_ids)
            selected_ids = selected_ids[:1000]

            # Use 0 as a filler value for the distance array since this is not relevant for the featured images
            dist = [0.0] * len(selected_ids)

            _get_metadata = functools.partial(get_full_metadata_batch, conn, external_metadata_tables=external_metadata_tables)
            search_index = search_indices[modality][feature_extractor_id]
            if feature_extractor_id == "wise/metadata":
                # ensure that other_ids[0] (i.e. feature_extractor_id) specific vector metadata
                # are not shown in the featured images
                _get_ext_metadata = lambda _: [FeatureExtMetadata()] * len(selected_ids)
            else:
                _get_ext_metadata = functools.partial(
                    search_index.feature_extractor.get_vector_metadata, conn
                )
            get_thumbs = _thumbs_with_score(thumbs_conn, dist[start:end], thumbnails_to_send)
            response = construct_search_response(
                top_dist=dist[start:end],
                top_ids=selected_ids[start:end],
                get_metadata_fn=_get_metadata,
                get_ext_metadata_fn=_get_ext_metadata,
                get_thumbs_fn=get_thumbs,
            )

        return response

    @router.get(
        "/related-vectors/{_vector_id}",
        response_model=list[VectorInfo],
        responses={200: {"content": "application/json"}},
    )
    def get_related_vectors(_vector_id: int):
        with project_engine.connect() as conn:
            related_rows = list(get_related_vectors_rows(conn, _vector_id))
            if not related_rows:
                vectors_ext_metadata = []
            else:
                feature_extractor = search_indices[related_rows[0].modality][
                    related_rows[0].feature_extractor_id
                ].feature_extractor

                vectors_ext_metadata = (
                    feature_extractor.get_vector_metadata(
                        conn, [v.id for v in related_rows]
                    )
                )

        vectors_info = []
        for row, extm in zip(related_rows, vectors_ext_metadata):
            vectors_info.append(
                VectorInfo(
                    vector_id=str(row.id),
                    media_id=str(row.media_id),
                    link=f"media/{row.media_id}#t={row.timestamp},{row.end_timestamp}",
                    thumbnail=f"thumbnail?media_id={row.media_id}&timestamp={row.timestamp}",
                    bbox=extm.bbox,
                )
            )
        return vectors_info

    @router.post("/search", response_model=SearchResponse)
    @add_response_time
    async def handle_post_search_multimodal(
        # Which media type to search on
        # "video" refers to the visual stream of videos, "av" refers to the audio stream of videos
        # "audio" refers to pure audio files, and "image" refers to images
        search_in: MediaType = Query(),
        feature_extractor_id: str = Query(),
        # Positive queries
        text_queries: List[str] = Query(default=[]),
        image_file_queries: List[bytes] = File([]),  # user-uploaded images
        audio_file_queries: List[bytes] = File([]),  # user-uploaded audio files
        image_url_queries: List[str] = Form([]),  # URLs to online images
        audio_url_queries: List[str] = Form([]),  # URLs to online audio files
        internal_image_queries: List[int] = Query(default=[]),  # ids to internal images
        # Negative queries
        negative_text_queries: List[str] = Query(default=[]),
        negative_image_file_queries: List[bytes] = File([]),  # user-uploaded images
        negative_audio_file_queries: List[bytes] = File(
            []
        ),  # user-uploaded audio files
        negative_image_url_queries: List[str] = Form([]),  # URLs to online images
        negative_audio_url_queries: List[str] = Form([]),  # URLs to online audio files
        negative_internal_image_queries: List[int] = Query(
            default=[]
        ),  # ids to internal images
        # Other parameters
        start: int = Query(0, ge=0, le=980),
        end: int = Query(20, gt=0, le=1000),
        thumbnails_to_send: int = Query(0),
        shot_scale: str = Query(None),
        metadata_filter: List[str] = Query(default=[]),
        add_prefix: bool = Query(True)
    ):
        """
        Handles queries sent by POST request. This endpoint can handle file queries, URL queries (i.e. URL to an image), and/or text queries.
        Multimodal queries (i.e. images + text) are performed by computing a weighted sum of the feature vectors of the
        input images/text, and then using this as the query vector.
        """
        if shot_scale is not None:
            try:
                shot_scale = json.loads(shot_scale)
            except Exception:
                raise HTTPException(400, {
                    "message": "shot_scale must be a JSON array string"
                })
        media_type = 'audio' if search_in == MediaType.AV else search_in
        if media_type not in search_indices:
            raise HTTPException(400, {
                "message": f"No search index exists for this modality: {media_type}"
            })
        search_index = search_indices[media_type][feature_extractor_id]

        def extract_text_features(text: List[str]) -> ndarray:
            if search_index.feature_extractor.extract_text_features is None:
                raise WiseFrontendUserException("text modality not supported")
            return search_index.feature_extractor.extract_text_features(text)

        def extract_image_features(images: List[Image.Image]) -> ndarray:
            if search_index.feature_extractor.extract_image_features is None:
                raise WiseFrontendUserException("image modality not supported")
            assert len(images) == 1
            features = search_index.feature_extractor.extract_image_features(
                search_index.feature_extractor.preprocess_image(images)
            )[0]
            if not len(features.vectors):
                raise WiseFrontendUserException("no features found on image")
            if len(features.vectors) > 1:
                logger.debug("multiple features found, will return vector for the top feature only")
            return features.vectors[0:1]

        def load_audio(x: List[io.BytesIO]) -> torch.Tensor:
            # TODO add support for loading multiple audio files
            if len(x) == 0:
                raise ValueError("No audio file was specified")
            elif len(x) > 1:
                raise NotImplementedError("Please specify 1 audio file only")

            target_sample_rate = 48_000 # TODO set this based on model?
            audio_file = x[0]
            waveform, original_sample_rate = torchaudio.load(audio_file)
            waveform = torchaudio.functional.resample(waveform, orig_freq=original_sample_rate, new_freq=target_sample_rate)
            return waveform

        def extract_audio_features(audio: List[io.BytesIO]) -> ndarray:
            if search_index.feature_extractor.extract_audio_features is None:
                raise WiseFrontendUserException("audio modality not supported")
            return search_index.feature_extractor.extract_audio_features(
                search_index.feature_extractor.preprocess_audio(load_audio(audio))
            )

        if internal_image_queries or negative_internal_image_queries:
            if search_index.is_internal_search_supported:
                try:
                    # reconstruct features from faiss index
                    internal_image_queries = reconstruct_internal_img_feature(search_index, internal_image_queries)
                    negative_internal_image_queries = reconstruct_internal_img_feature(search_index, negative_internal_image_queries)
                except Exception as e:
                    logger.exception(e)
                    return PlainTextResponse(
                        status_code=500, content=f"Error processing internal search query"
                    )
            else:
                logger.exception(
                    "This faiss index does not support internal search. To enable "
                    "internal search, please re-create the index by running "
                    f"`python create-index.py --project-dir \"{config.project_dir}\" --media-type {media_type} --index-type {search_index.index_type} --overwrite`",
                )
                return PlainTextResponse(
                    status_code=500, content=f"Internal search not supported in this project"
                )

            # Apply hook to transform internal image query vectors
            internal_image_queries = [
                search_index.feature_extractor.transform_internal_image_queries_hook(x)
                for x in internal_image_queries
            ]
            negative_internal_image_queries = [
                search_index.feature_extractor.transform_internal_image_queries_hook(x)
                for x in negative_internal_image_queries
            ]

        for tq in text_queries:
            if tq.strip() in config.query_blocklist:
                message = (
                    "One of the search terms you entered has been blocked"
                    if len(text_queries) > 1
                    else "The search term you entered has been blocked"
                )
                raise HTTPException(403, {"message": message})

        q = [dict(sign="positive", modality="text", val=query) for query in text_queries]
        q += [dict(sign="positive", modality="image", val=query) for query in (
            image_file_queries + image_url_queries + internal_image_queries
        )]
        q += [dict(sign="positive", modality="audio", val=query) for query in (
            audio_file_queries + audio_url_queries
        )]

        q += [dict(sign="negative", modality="text", val=query) for query in negative_text_queries]
        q += [dict(sign="negative", modality="image", val=query) for query in (
            negative_image_file_queries + negative_image_url_queries + negative_internal_image_queries
        )]
        q += [dict(sign="negative", modality="audio", val=query) for query in (
            negative_audio_file_queries + negative_audio_url_queries
        )]

        if len(q) == 0:
            raise HTTPException(400, {"message": "Missing search query"})
        elif len(q) > 5:
            raise HTTPException(400, {"message": "Too many query items"})

        if feature_extractor_id == 'wise/metadata':
            # ASR search
            if start > end:
                raise HTTPException(
                    400, {"message": "'start' cannot be greater than 'end'"}
                )

            # TODO escape special characters
            text = " ".join(text_queries)
            q = WISEFTSQuery.model_validate({"$match": text})
            return asr_search(q, search_index, search_in, start, end)

        if search_in == MediaType.IMAGE:
            if len([query for query in q if query['modality'] == 'audio']) > 0:
                raise HTTPException(400, {
                    "message": "Cannot search on images using an audio query"
                })
        elif search_in == MediaType.VIDEO:
            if len([query for query in q if query['modality'] == 'audio']) > 0:
                raise HTTPException(400, {
                    "message": "Cannot search on visual stream of video files using an audio query"
                })
        elif search_in == MediaType.AUDIO or search_in == MediaType.AV:
            if len([query for query in q if query['modality'] == 'image']) > 0:
                raise HTTPException(400, {
                    "message": "Cannot search on audio using an image query"
                })

        end = min(end, num_vectors)
        if start > end:
            raise HTTPException(
                400, {"message": "'start' cannot be greater than 'end'"}
            )

        filter_specs = None
        if shot_scale is not None and len(shot_scale) > 0:
            filter_specs = {"shot_scale_query": {"$in": shot_scale}}
        if metadata_filter:
            if filter_specs is None:
                filter_specs = {}
            filter_specs |= {
                "metadata_query": WISEFTSQuery.model_validate(
                    {"$match": " ".join(metadata_filter)}
                )
            }
        return similarity_search(
            q,
            search_in=search_in,
            search_index=search_index,
            start=start,
            end=end,
            feature_extractor_id=feature_extractor_id,
            thumbnails_to_send=thumbnails_to_send,
            extract_text_features=extract_text_features,
            extract_image_features=extract_image_features,
            extract_audio_features=extract_audio_features,
            get_ext_metadata=search_index.feature_extractor.get_vector_metadata,
            filter_specs=filter_specs,
            add_prefix=add_prefix
        )

    def similarity_search(
        q: List[Dict[str, Union[ndarray, bytes, str]]],
        search_in: MediaType,
        search_index: SearchIndex,
        start: int,
        end: int,
        feature_extractor_id: str,
        thumbnails_to_send: int = 0,
        extract_text_features: Callable[[List[str]], ndarray] = None,
        extract_image_features: Callable[[List[Image.Image]], ndarray] = None,
        extract_audio_features: Callable[[List[io.BytesIO]], ndarray] = None,
        get_ext_metadata: Callable[[list[int]], list[FeatureExtMetadata]] = None,
        filter_specs: Dict[str, Any] = None,
        add_prefix: bool = True,
    ):
        prefix = _prefix[search_in] if add_prefix else ""
        features = _get_query_features(prefix, q, extract_text_features, extract_image_features, extract_audio_features)
        if filter_specs is not None:
            filtered_ids = get_filtered_ids(filter_specs, search_in, feature_extractor_id)
            sel = faiss.IDSelectorBatch(filtered_ids)
            if search_index.index_type == 'IndexFlatIP':
                params = faiss.SearchParameters(sel=sel)
            elif search_index.index_type == 'IndexIVFFlat':
                params = faiss.SearchParametersIVF(sel=sel, nprobe=search_index.index.nprobe)
            else:
                raise HTTPException(400, {
                    "message": f"filter_specs does not support index type : {search_index.index_type}"
                })
            dist, ids = search_index.index.search(features, end, params=params)
        else:
            dist, ids = search_index.index.search(features, end)

        # Apply hook to transform Faiss distance scores
        dist = search_index.feature_extractor.transform_faiss_distances_hook(dist)

        top_ids, top_dist = ids[0, start:end], dist[0, start:end]

        valid_indices = [i for i, x in enumerate(top_ids) if x != -1]

        valid_ids = [int(top_ids[x]) for x in valid_indices]
        valid_dist = [float(top_dist[x]) for x in valid_indices]

        if len(valid_ids) == 0:
            return SearchResponse(
                time=0.0,
                video_audio_results=None,
                video_results=None,
                image_results=None,
            )
        # supports shots
        is_shot_merge_supported = config.use_shots and search_in == MediaType.VIDEO
        with project_engine.connect() as conn, thumbs_engine.connect() as thumbs_conn:
            _get_metadata = functools.partial(get_full_metadata_batch, conn, external_metadata_tables=external_metadata_tables)
            _get_ext_metadata = functools.partial(get_ext_metadata, conn)

            get_thumbs = thumbs_reader(thumbs_conn, valid_dist, thumbnails_to_send)

            response = construct_search_response(
                top_dist=valid_dist,
                top_ids=valid_ids,
                get_metadata_fn=_get_metadata,
                get_ext_metadata_fn=_get_ext_metadata,
                get_thumbs_fn=get_thumbs,
                merge_function=get_shots_from_keyframes if is_shot_merge_supported else merge_close_segments,
                search_in=search_in,
            )

        return response

    def get_filtered_ids(filter_specs: Dict[str, Any], search_in: MediaType, feature_extractor_id: str) -> np.ndarray:
        id_constraint = None
        media_type = 'audio' if search_in == MediaType.AV else search_in
        metadata_query = filter_specs.get("metadata_query", None)
        if metadata_query:
            with project_engine.connect() as conn:
                media_ids_cte = search_indices[media_type]["wise/metadata"].search(
                    conn, metadata_query, ids_only=True
                )
                if media_ids_cte is None:
                    return np.array([], dtype=np.int64)

                vector_ids = (
                    conn.execute(
                        sa.select(db.vectors_table.c.id).select_from(
                            media_ids_cte.join(
                                db.vectors_table,
                                sa.and_(
                                    db.vectors_table.c.media_id == media_ids_cte.c.media_id,
                                    db.vectors_table.c.modality == media_type,
                                    db.vectors_table.c.feature_extractor_id == feature_extractor_id,
                                )
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                id_constraint = np.array(vector_ids, dtype=np.int64)

        shot_scale_query = filter_specs.get("shot_scale_query", None)
        if shot_scale_query:
            shot_scales = shot_scale_query["$in"]
            with project_engine.connect() as conn:
                stmt = (
                    sa.select(db.vectors_table.c.id)
                    .select_from(
                        db.shots_table.join(
                            vectors_to_shots_map,
                            sa.and_(
                                db.shots_table.c.id == vectors_to_shots_map.c.shot_id,
                                db.shots_table.c.media_id
                                == vectors_to_shots_map.c.media_id,
                            ),
                        ).join(
                            db.vectors_table,
                            vectors_to_shots_map.c.vector_id == db.vectors_table.c.id,
                        )
                    )
                    .where(
                        sa.and_(
                            db.vectors_table.c.modality == media_type,
                            db.vectors_table.c.feature_extractor_id
                            == feature_extractor_id,
                            db.shots_table.c.shot_scale.in_(shot_scales),
                        )
                    )
                )
                result = conn.execute(stmt).scalars().all()
                shot_scale_constraint = np.array(result, dtype=np.int64)
                id_constraint = (
                    shot_scale_constraint
                    if id_constraint is None
                    else np.intersect1d(id_constraint, shot_scale_constraint)
                )
        return id_constraint

    def asr_search(
        q: WISEFTSQuery,
        search_index: FTSSearch,
        search_in: MediaType,
        start: int,
        end: int,
        thumbnails_to_send: int = 0,
    ):
        with project_engine.connect() as conn, thumbs_engine.connect() as thumbs_conn:

            all_metadata = search_index.search(conn, q, start, end)
            n_results = len(all_metadata)
            if n_results == 0:
                return SearchResponse(
                    time=0.0,
                    video_audio_results=None,
                    video_results=None,
                    image_results=None,
                )
            dist = list([-x for x in range(1, n_results + 1)])
            get_thumbs = thumbs_reader(thumbs_conn, iter(dist), thumbnails_to_send)
            _get_metadata_fn = lambda _: all_metadata
            _get_ext_metadata_fn = lambda _: [FeatureExtMetadata()] * n_results
            get_thumbs = thumbs_reader(thumbs_conn, iter(dist), thumbnails_to_send)
            response = construct_search_response(
                dist,
                None, 
                _get_metadata_fn,
                _get_ext_metadata_fn,
                get_thumbs,
                search_in=search_in
            )
            return response

    search_router_info = {
        "search_targets": active_search_targets,
        "shot_based_filters": shot_based_filters,
    }
    return router, search_router_info
