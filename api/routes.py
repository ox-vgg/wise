from contextlib import ExitStack
import time
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, Union, BinaryIO
import io
import itertools
import functools
import logging
from pathlib import Path
import tarfile
from PIL import Image
import math
import numpy as np
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
from pydantic import field_validator, BaseModel
import typer
import csv
import json
import os
import sqlalchemy as sa

from config import APIConfig
from src.search_index import SearchIndex
from src import db
from src.repository import (
    SourceCollectionRepo,
    MediaRepo,
    VectorRepo,
    ThumbnailRepo,
    # query_by_timestamp,
    get_featured_images,
    get_full_metadata_batch,
    get_thumbnail_by_timestamp,
)
from src.data_models import MediaMetadata, MediaType, SourceCollectionType, VectorAndMediaMetadata
from src.inference import setup_clip, CLIPModel
from src.enums import IndexType
from src.search import read_index, brute_force_search
from src.utils import convert_uint8array_to_base64
from src.wise_project import WiseProject

logger = logging.getLogger(__name__)


def raise_(ex):
    raise ex


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
    router.include_router(_get_project_data_router(config))
    router.include_router(_get_report_image_router(config))
    router.include_router(_get_search_router(config))

    return router


def _get_project_data_router(config: APIConfig):
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
    project_engine = db.init_project(project.dburi)
    thumbs_engine = db.init_thumbs(project.thumbs_uri)

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

            if metadata.media_type in {MediaType.VIDEO, MediaType.AV, MediaType.AUDIO}:
                file_size = file_path.stat().st_size
                range_header = request.headers.get("range")

                content_type = f"{metadata.media_type.value}/{metadata.format}" if metadata.media_type == MediaType.AUDIO else f"video/mp4"
                headers = {
                    "content-type": content_type,
                    "accept-ranges": "bytes",
                    "content-encoding": "identity",
                    "content-length": str(file_size),
                    "access-control-expose-headers": (
                        "content-type, accept-ranges, content-length, "
                        "content-range, content-encoding"
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
                            file_path, media_type=f"media/{metadata.format.lower()}"
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
                        file_iter, media_type=f"media/{metadata.format.lower()}"
                    )
                except Exception as e:
                    logger.exception(f"Exception when reading image {media_id}")
                    return PlainTextResponse(
                        status_code=404, content=f"{media_id} not found!"
                    )

    @router.get(
        "/thumbs/{_id}",
        response_class=Response,
        responses={200: {"content": "image/jpeg"}, 404: {"content": "text/plain"}},
    )
    def get_thumbnail(_id: int):
        # Get a thumbnail given a thumbnail id
        with thumbs_engine.connect() as thumbs_conn:
            thumbnail_metadata = ThumbnailRepo.get(thumbs_conn, _id)
            if thumbnail_metadata is None:
                raise HTTPException(status_code=404, detail=f"Thumbnail not found!")
            return Response(
                content=thumbnail_metadata.content,
                media_type="image/jpeg",
                status_code=200,
            )
    
    @router.get(
        "/storyboard/{_video_media_id}",
        response_class=JSONResponse,
        responses={200: {"content": "application/json"}, 404: {"content": "application/json"}},
    )
    def get_storyboard(_video_media_id: int):
        """
        Generate JSON storyboard for a given video (as per this documentation: https://www.vidstack.io/docs/player/core-concepts/loading?styling=default-theme#json).
        A storyboard image (like this example: https://media-files.vidstack.io/storyboard.jpg)
        is generated based on the existing thumbnails of the video, and is included in the response.
        This is used for the preview thumbnails in the frontend UI when hovering over the timeline in the video player.
        """
        with thumbs_engine.connect() as thumbs_conn:
            thumbnail_rows = ThumbnailRepo.list_by_column_match(
                thumbs_conn,
                column_to_match="media_id",
                value_to_match=_video_media_id,
                select_columns=("id", "timestamp", "content"),
                order_by_column="timestamp"
            )
            thumbnail_rows = list(thumbnail_rows)
            if len(thumbnail_rows) == 0:
                raise HTTPException(status_code=404, detail=f"Thumbnails not found for media_id={_video_media_id}!")

            thumbnail_rows = thumbnail_rows[::4] # Get every 4th item in the list
                                            # (i.e. 1 thumbnail per 2 seconds of video if the sampling rate was 2fps)
                                            # TODO make this change based on sampling rate
            
            # Get thumbnails
            ids = [thumbnail_row['id'] for thumbnail_row in thumbnail_rows]
            thumbs = [Image.open(io.BytesIO(thumbnail_row['content'])) for thumbnail_row in thumbnail_rows]
            w, h = thumbs[0].size # Assumes all thumbnails have the same size

            # Create storyboard
            num_columns = 10
            num_rows = math.ceil(len(thumbs) / num_columns)
            storyboard = Image.new('RGB', (w*num_columns, h*num_rows))
            tiles = []
            for idx, (thumb, thumbnail_row) in enumerate(zip(thumbs, thumbnail_rows)):
                x = (idx % num_columns) * w
                y = (idx // num_columns) * h
                storyboard.paste(thumbs[idx], (x, y))
                tiles.append({
                    "startTime": thumbnail_row['timestamp'],
                    "x": x,
                    "y": y
                })
            buffered = io.BytesIO()
            storyboard.save(buffered, format='JPEG')

            response = {
                "url": convert_uint8array_to_base64(buffered.getvalue()),
                "tileWidth": w,
                "tileHeight": h,
                "tiles": tiles
            }
            return JSONResponse(status_code=200, content=response)

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

    @router.get("/info")
    def get_info():
        with project_engine.connect() as conn:
            num_vectors = VectorRepo.get_count(conn)
            num_media_files = MediaRepo.get_count(conn)
        return {
            "project_name": config.project_dir.stem,
            "models": {
                media_type: [
                    feature_extractor_id for feature_extractor_id in project_assets[media_type]
                ] for media_type in project_assets
            },
            "num_vectors": num_vectors,
            "num_media_files": num_media_files,
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
        caption: str = ""
        copyright: str = ""

    # A search result containing the metadata fields from MediaInfo, as well as additional fields like `thumbnail` and `distance`
    class MediaSearchResult(MediaInfo):
        link: str
        thumbnail: str
        distance: Optional[float] = None

        @field_validator("distance")
        @classmethod
        def round_distance(cls, v):
            return round(v, config.precision)

    # A subclass of MediaSearchResult for images
    class ImageSearchResult(MediaSearchResult):
        pass

    # A subclass of MediaSearchResult for pure audio files
    class AudioSearchResult(MediaSearchResult):
        pass

    # A subclass of MediaSearchResult for videos
    class VideoSearchResult(MediaSearchResult):
        timeline_hover_thumbnails: str

    # An audio or video segment
    class MediaSegment(BaseModel):
        vector_id: str
        media_id: str
        ts: float
        te: float
        link: str
        distance: float

        @field_validator("distance")
        @classmethod
        def round_distance(cls, v):
            return round(v, config.precision)

    # A subclass of MediaSegment for pure audio files
    class AudioSegment(MediaSegment):
        pass

    # A subclass of MediaSegment for videos
    class VideoSegment(MediaSegment):
        thumbnail: str
        thumbnail_score: float

        @field_validator("thumbnail_score")
        @classmethod
        def round_distance(cls, v):
            return round(v, config.precision)

    class AudioResults(BaseModel):
        total: int # maximum number of audio results that can be returned
        unmerged_windows: List[AudioSegment] # e.g. 7-second windows
        audios: List[AudioSearchResult]

    class VideoAudioResults(BaseModel):
        total: int # maximum number of unmerged_windows that can be returned
        unmerged_windows: List[VideoSegment] # e.g. 7-second windows
        merged_windows: List[VideoSegment] # shots (for edited videos) or merged segments (for unedited videos)
        videos: Dict[str, VideoSearchResult]

    class VideoResults(BaseModel):
        total: int # maximum number of unmerged_windows that can be returned
        unmerged_windows: List[VideoSegment] # frames (CLIP) or unmerged 4-second segments (InternVideo/LanguageBind)
        merged_windows: List[VideoSegment] # shots (for edited videos) or merged segments (for unedited videos)
        videos: Dict[str, VideoSearchResult]

    class ImageResults(BaseModel):
        total: int # maximum number of images that can be returned e.g. min(1000, num_images_in_project)
        results: List[ImageSearchResult]

    class SearchResponse(BaseModel):
        time: float # backend search time in seconds
        audio_results: Optional[AudioResults] # search results from pure audio files
        video_audio_results: Optional[VideoAudioResults] # search results from audio stream of video files
        video_results: Optional[VideoResults] # search results from video stream of video files
        image_results: Optional[ImageResults] # search results from image files


    def merge_close_segments(video_id: int, _keyframes: List[VideoSegment]):
        merged_segments: List[VideoSegment] = []
        start = None
        current = None
        best_thumbnail = None
        best_thumbnail_score = 0
        best_segment_score = 0
        for k in _keyframes:
            if start is None:
                # Start a new group
                start = k
                current = k
                best_thumbnail = k.thumbnail
                best_thumbnail_score = k.thumbnail_score
                best_segment_score = k.distance

            elif (k.ts - current.te) <= 4:
                current = k
                if current.thumbnail_score > best_thumbnail_score:
                    best_thumbnail_score = current.thumbnail_score
                    best_thumbnail = current.thumbnail
                if current.distance > best_segment_score:
                    best_segment_score = current.distance

            else:
                merged_segments.append(
                    VideoSegment(
                        vector_id=start.vector_id,
                        media_id=start.media_id,
                        ts=start.ts,
                        te=current.te,
                        link=f"videos/{start.media_id}#t={start.ts},{current.te}",
                        distance=best_segment_score,
                        thumbnail=best_thumbnail,
                        thumbnail_score=best_thumbnail_score,
                    )
                )
                start = k
                current = k
                best_thumbnail_score = k.thumbnail_score
                best_thumbnail = k.thumbnail
                best_segment_score = k.distance

        if start is not None:
            merged_segments.append(
                VideoSegment(
                    vector_id=start.vector_id,
                    media_id=start.media_id,
                    ts=start.ts,
                    te=current.te,
                    link=f"videos/{start.media_id}#t={start.ts},{current.te}",
                    distance=best_segment_score,
                    thumbnail=best_thumbnail,
                    thumbnail_score=best_thumbnail_score,
                )
            )

        return merged_segments

    def get_shots_from_segments(segments: List[VideoSegment]):
        # Sort by video_id, timestamp
        sorted_segments = sorted(segments, key=lambda x: (x.media_id, x.ts))

        # for each key, merge keyframes with <= 4s gap and keep track of best thumbnail per video
        best_thumbnail = {}
        all_merged_segments = []
        for vid, g in itertools.groupby(sorted_segments, key=lambda x: x.media_id):
            merged_segments = merge_close_segments(vid, list(g))
            best_thumbnail[vid] = sorted(
                merged_segments, key=lambda x: x.thumbnail_score, reverse=True
            )[0]
            all_merged_segments.extend(merged_segments)

        # sort the merged segments by distance
        all_merged_segments = sorted(
            all_merged_segments,
            key=lambda x: x.distance,
            reverse=True,
        )
        return all_merged_segments, best_thumbnail

    def construct_video_search_response(
        top_dist: List[float],
        top_ids: List[int],
        get_metadata_fn: Callable[[List[int]], List[VectorAndMediaMetadata]],
        get_thumbs_fn: Callable[[List[VectorAndMediaMetadata]], Iterable[Tuple[str, float]]],
    ):
        videos = {}
        shots = []
        segments = []
        all_metadata = get_metadata_fn(top_ids)
        for _dist, _metadata, (_thumb, _thumb_score) in zip(
            top_dist,
            all_metadata,
            get_thumbs_fn(all_metadata),
        ):
            video_id = str(_metadata.media_id)
            if video_id not in videos:
                videos[video_id] = VideoSearchResult(
                    id=video_id,
                    link=f"media/{video_id}",
                    filename=_metadata.path,
                    width=_metadata.width,
                    height=_metadata.height,
                    media_type=_metadata.media_type,
                    format=_metadata.format,
                    duration=_metadata.duration,
                    thumbnail="",
                    timeline_hover_thumbnails=f"storyboard/{video_id}",
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
                link=f"videos/{video_id}#t={ts},{te}", # f"{_metadata.source_uri if _metadata.source_uri else f'videos/{video_id}{_metadata.path}'}",
                distance=_dist,
                thumbnail=_thumb,
                thumbnail_score=_thumb_score,
            )

            segments.append(segment)

        shots, best_thumbnails = get_shots_from_segments(segments)
        for v in videos:
            videos[v].thumbnail = best_thumbnails[v].thumbnail

        video_results = VideoResults(
            total=300, # TODO change this
            unmerged_windows=segments,
            merged_windows=shots,
            videos=videos,
        )

        return SearchResponse(
            time=0.0, # TODO change this
            audio_results=None,
            video_audio_results=None,
            video_results=video_results,
            image_results=None,
        )

    def _get_query_features(
        extract_features_from_image: Callable[[List[Image.Image]], ndarray],
        extract_features_from_text: Callable[[List[str]], ndarray],
        query_prefix: str,
        q: List[Dict[str, Union[ndarray, bytes, str]]],
    ) -> ndarray:
        feature_vectors = []
        weights = []

        for query_dict in q:
            query = query_dict["val"]
            feature_vector = None
            if isinstance(query, bytes):
                with Image.open(io.BytesIO(query)) as im:
                    im = im.convert('RGB')
                    feature_vector = extract_features_from_image([im])
                    weights.append(
                        config.negative_queries_weight
                        if query_dict["type"] == "negative"
                        else 1
                    )
            elif isinstance(query, ndarray):
                feature_vector = query
                weights.append(
                    config.negative_queries_weight
                    if query_dict["type"] == "negative"
                    else 1
                )
            elif query.startswith(("http://", "https://")):
                logger.info("Downloading", query, "to file")
                with NamedTemporaryFile() as tmpfile:
                    download_url_to_file(query, tmpfile.name)
                    with Image.open(tmpfile.name) as im:
                        im = im.convert('RGB')
                        feature_vector = extract_features_from_image([im])
                        weights.append(
                            config.negative_queries_weight
                            if query_dict["type"] == "negative"
                            else 1
                        )
            else:
                prefixed_queries = f"{query_prefix} {query.strip()}".strip()
                feature_vector = extract_features_from_text(prefixed_queries)
                weights.append(
                    config.text_queries_weight
                    * (  # assign higher weight to natural language queries
                        config.negative_queries_weight
                        if query_dict["type"] == "negative"
                        else 1
                    )
                )
            if query_dict["type"] == "negative":
                feature_vector = -feature_vector
            feature_vectors.append(feature_vector)
        weights = array(weights, dtype=float32)
        average_features = average(feature_vectors, axis=0, weights=weights)
        average_features /= norm(average_features, axis=-1, keepdims=True)

        return average_features

    _prefix = config.query_prefix.strip()
    project_assets = project.discover_assets()
    project_engine = db.init_project(project.dburi)
    thumbs_engine = db.init_thumbs(project.thumbs_uri)

    media_type = 'video' # TODO change this later
    feature_extractor_id_list = list(project_assets[media_type].keys())
    feature_extractor_id = feature_extractor_id_list[0] # TODO: allow users to select the feature
    
    # Load the faiss search index
    index_dir = project_assets[media_type][feature_extractor_id]['index_dir']
    search_index = SearchIndex(media_type,
                                feature_extractor_id,
                                index_dir)
    logger.info(f"Loading faiss index from {search_index.get_index_filename(config.index_type)}")
    if not search_index.load_index(config.index_type):
        raise RuntimeError("Unable to load faiss index")
    if hasattr(search_index.index, "nprobe"):
        # See https://github.com/facebookresearch/faiss/blob/43d86e30736ede853c384b24667fc3ab897d6ba9/faiss/IndexIVF.h#L184C8-L184C42
        search_index.index.parallel_mode = 1
        search_index.index.nprobe = getattr(config, "nprobe", 32)

    if hasattr(config, 'index_use_direct_map') and config.index_use_direct_map == 1:
        try:
            logger.info(f"Enabling direct map on search index for faster internal search.")
            search_index.index.make_direct_map(True)
        except Exception as e:
            logger.info(f"Search index does not support direct map, falling back to using saved features for internal search (slower)")
    else:
        logger.info(f"Direct map on search index can be enabled by setting index_use_direct_map=1 in config.py. This speeds up internal search.")

    # Get counts
    with project_engine.connect() as conn:
        num_vectors = VectorRepo.get_count(conn)

    extract_text_features = search_index.feature_extractor.extract_text_features
    extract_image_features: Callable[[List[Image.Image]], ndarray] = lambda x: search_index.feature_extractor.extract_image_features(
        search_index.feature_extractor.preprocess_image(x)
    )

    router_cm = ExitStack()

    def _thumbs_with_score(conn: sa.Connection, dist: List[float], thumbnails_to_send: int):
        def inner(vector_and_media_metadata_list: List[VectorAndMediaMetadata]):
            thumbs = [
                get_thumbnail_by_timestamp(
                    conn,
                    media_id=vector_and_media_metadata.media_id,
                    timestamp=vector_and_media_metadata.timestamp,
                    get_id_only=i >= thumbnails_to_send
                )
                for i, vector_and_media_metadata
                in enumerate(vector_and_media_metadata_list)
            ]
            thumbs = [
                (
                    f"thumbs/{_thumb}" if isinstance(_thumb, int)
                    else convert_uint8array_to_base64(_thumb)
                ) for _thumb in thumbs
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

    get_query_features = functools.partial(
        _get_query_features, extract_image_features, extract_text_features, _prefix
    )
    router = APIRouter(
        on_shutdown=[lambda: print("shutting down") and router_cm.close()],
    )

    def reconstruct_internal_img_feature(vector_ids: List[int]) -> List[Union[ndarray, bytes]]:
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

    # Create a random array of featured images (1 per video)
    with project_engine.connect() as conn:
        ids = get_featured_images(conn)

        # Select a random subset of up to 10000 image ids (for performance reasons)
        default_rng(seed=42).shuffle(ids)
        ids = ids[:10000]

    @router.get("/featured", response_model=SearchResponse)
    @add_response_time
    async def handle_get_featured(
        start: int = Query(0, ge=0, le=980),
        end: int = Query(20, gt=0, le=1000),
        thumbnails_to_send: int = Query(0),
        # This seed is used to randomly select the set of images used for the featured images
        random_seed: int = Query(123),
    ):
        with project_engine.connect() as conn, thumbs_engine.connect() as thumbs_conn:
            # Select up to 1000 random image ids, using the specified random seed, from the set of 10000 ids
            selected_ids = ids.copy()
            default_rng(seed=random_seed).shuffle(selected_ids)
            selected_ids = selected_ids[:1000]

            # Use 0 as a filler value for the distance array since this is not relevant for the featured images
            dist = [0.0] * len(selected_ids)

            _get_metadata = functools.partial(get_full_metadata_batch, conn)
            # def _get_metadata(_id: int):
            #     m = MetadataRepo.get(conn, int(_id) + 1)
            #     if m is None:
            #         raise RuntimeError()

            #     if m.metadata.get("title") is None:
            #         d = DatasetRepo.get(conn, int(m.dataset_id))
            #         if d is None:
            #             raise RuntimeError()
            #         video_filename = Path(d.location).name
            #         m.metadata["title"] = video_filename

            #     return m

            get_thumbs = _thumbs_with_score(thumbs_conn, dist[start:end], thumbnails_to_send)
            response = construct_video_search_response(
                dist[start:end],
                selected_ids[start:end],
                _get_metadata,
                get_thumbs,
            )

        return response

    @router.get("/search", response_model=SearchResponse)
    @add_response_time
    async def handle_get_search(
        q: List[str] = Query(default=[]),
        start: int = Query(0, ge=0, le=980),
        end: int = Query(20, gt=0, le=1000),
        thumbnails_to_send: int = Query(0),
    ):
        if len(q) == 0:
            raise HTTPException(400, {"message": "Missing search query"})

        end = min(end, num_vectors)
        if start > end:
            raise HTTPException(
                400, {"message": "'start' cannot be greater than 'end'"}
            )
        # if (end - start) > 50 and thumbs == True:
        #     raise HTTPException(
        #         400,
        #         {
        #             "message": "Cannot return more than 50 results at a time when thumbs=1"
        #         },
        #     )
    
        for query in q:
            if query.strip() in config.query_blocklist:
                message = (
                    "One of the search terms you entered has been blocked"
                    if len(q) > 1
                    else "The search term you entered has been blocked"
                )
                raise HTTPException(403, {"message": message})

        q = [dict(type="positive", val=query) for query in q]

        return similarity_search(
            q=q, start=start, end=end, thumbnails_to_send=thumbnails_to_send
        )

    @router.post("/search", response_model=SearchResponse)
    @add_response_time
    async def handle_post_search_multimodal(
        # Positive queries
        text_queries: List[str] = Query(default=[]),
        file_queries: List[bytes] = File([]),  # user-uploaded images
        url_queries: List[str] = Form([]),  # URLs to online images
        internal_image_queries: List[int] = Query(default=[]),  # ids to internal images
        # Negative queries
        negative_text_queries: List[str] = Query(default=[]),
        negative_file_queries: List[bytes] = File([]),  # user-uploaded images
        negative_url_queries: List[str] = Form([]),  # URLs to online images
        negative_internal_image_queries: List[int] = Query(
            default=[]
        ),  # ids to internal images
        # Other parameters
        start: int = Query(0, ge=0, le=980),
        end: int = Query(20, gt=0, le=1000),
        thumbnails_to_send: int = Query(0),
    ):
        """
        Handles queries sent by POST request. This endpoint can handle file queries, URL queries (i.e. URL to an image), and/or text queries.
        Multimodal queries (i.e. images + text) are performed by computing a weighted sum of the feature vectors of the
        input images/text, and then using this as the query vector.
        """
        try:
            if not hasattr(search_index.index, 'direct_map') or search_index.index.direct_map.type == search_index.index.direct_map.NoMap:
                # load saved features from HDF file (slower)
                internal_image_queries = load_internal_images(internal_image_queries)
                negative_internal_image_queries = load_internal_images(negative_internal_image_queries)
            else:
                # reconstruct features from faiss index (faster)
                internal_image_queries = reconstruct_internal_img_feature(internal_image_queries)
                negative_internal_image_queries = reconstruct_internal_img_feature(negative_internal_image_queries)
        except Exception as e:
            logger.exception(e)
            return PlainTextResponse(
                status_code=500, content=f"Error processing internal image queries"
            )

        for tq in text_queries:
            if tq.strip() in config.query_blocklist:
                message = (
                    "One of the search terms you entered has been blocked"
                    if len(text_queries) > 1
                    else "The search term you entered has been blocked"
                )
                raise HTTPException(403, {"message": message})

        q = file_queries + url_queries + text_queries + internal_image_queries
        q = [dict(type="positive", val=query) for query in q]

        negative_q = (
            negative_file_queries
            + negative_url_queries
            + negative_text_queries
            + negative_internal_image_queries
        )
        q = q + [dict(type="negative", val=query) for query in negative_q]

        if len(q) == 0:
            raise HTTPException(400, {"message": "Missing search query"})
        elif len(q) > 5:
            raise HTTPException(400, {"message": "Too many query items"})

        end = min(end, num_vectors)
        if start > end:
            raise HTTPException(
                400, {"message": "'start' cannot be greater than 'end'"}
            )
        # if (end - start) > 50 and thumbs == True:
        #     raise HTTPException(
        #         400,
        #         {
        #             "message": "Cannot return more than 50 results at a time when thumbs=1"
        #         },
        #     )

        return similarity_search(
            q,
            start=start,
            end=end,
            thumbnails_to_send=thumbnails_to_send,
        )

    def similarity_search(
        q: List[Dict[str, Union[ndarray, bytes, str]]],
        start: int,
        end: int,
        thumbnails_to_send: int = 0,
    ):
        features = get_query_features(q)
        dist, ids = search_index.index.search(features, end)

        top_ids, top_dist = ids[0, start:end], dist[0, start:end]

        valid_indices = [i for i, x in enumerate(top_ids) if x != -1]

        valid_ids = [int(top_ids[x]) for x in valid_indices]
        valid_dist = [float(top_dist[x]) for x in valid_indices]

        with project_engine.connect() as conn, thumbs_engine.connect() as thumbs_conn:
            _get_metadata = functools.partial(get_full_metadata_batch, conn)
            # def _get_metadata(_id: int):
            #     m = MediaRepo.get(conn, int(_id))
            #     if m is None:
            #         raise RuntimeError()

            #     if m.metadata.get("title") is None:
            #         d = DatasetRepo.get(conn, int(m.dataset_id))
            #         if d is None:
            #             raise RuntimeError()
            #         video_filename = Path(d.location).name
            #         m.metadata["title"] = video_filename

            #     return m

            get_thumbs = thumbs_reader(thumbs_conn, valid_dist, thumbnails_to_send)

            response = construct_video_search_response(
                valid_dist,
                valid_ids,
                _get_metadata,
                get_thumbs,
            )

        return response

    return router
