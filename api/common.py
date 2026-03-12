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

import base64
import functools
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Callable, Literal, Optional

import numpy as np
from fastapi import HTTPException, Request, Response, UploadFile
from fastapi.routing import APIRoute
from pydantic import (
    BaseModel,
    HttpUrl,
    PlainSerializer,
    TypeAdapter,
    field_serializer,
    field_validator,
    ConfigDict
)

from config import APIConfig


PRECISION = 5
def round_float_(v: float) -> float:
    return round(v, PRECISION)

round_float = Annotated[float, PlainSerializer(round_float_, when_used='json-unless-none')]


class BBoxXYWH(BaseModel):
    x: round_float
    y: round_float
    w: round_float
    h: round_float


class NPArray(BaseModel):
    """Utility to convert between numpy arrays and json for HTTP requests.
    """
    content: str
    shape: list[int]
    # assume we only exchange float32 arrays for now

    @classmethod
    def from_array(cls, x: np.ndarray) -> "NPArray":
        np_bytes = x.tobytes()
        base64_encoded = base64.b64encode(np_bytes)
        return cls(
            content=base64_encoded.decode('ascii'),
            shape=list(x.shape)
        )

    def to_array(self) -> np.ndarray:
        bytes_from_b64 = base64.b64decode(self.content.encode('ascii'))
        arr = np.frombuffer(bytes_from_b64, dtype=np.float32)
        arr = arr.reshape(self.shape)
        return arr


class BaseQueryTerm(BaseModel):
    """Base class for one query term.

    Each query term must have its own unique term ID across a query.
    The term ID is used to map it to files in multipart/form-data
    requests.  The term_id can also be useful later to send back term
    specific data on the response.

    See `MediaQueryTerm`, `VectorQueryTerm`, and `TextQueryTerm` for
    concrete "implementations".

    """
    term_id: str
    is_negative: bool


class MediaQueryTerm(BaseQueryTerm):
    """A query term based on a media file (image, audio, or video).

    This provides "cropping" around space and time: `bbox` for x and y
    dimensions and ts/te for time, thus providing an API to search
    with image regions and video segments.

    Attributes:
        src: bytes for image, int for media id, URL string for URL to
            download media file.
        qtype: in the case of an audiovisual file, whether to use the
             audio or video streams for the query.
        bbox: region to use for search in XYWH format with origin in
            the top left corner.  If missing, uses the whole image.
        ts/te: time start and time end, defaulting, respectively, to
            the start and end of the video file.  If ts and te have
            the same value then it uses a single frame, otherwise it's
            a video segment.

    """
    src: HttpUrl | bytes | int
    qtype: Literal["audio", "visual"]
    bbox: Optional[BBoxXYWH] = None
    ts: Optional[float] = None
    te: Optional[float] = None

class VectorIdQueryTerm(BaseQueryTerm):
    vector_id: str  # str because format is {shard_id}/{media_id}/{vector_id}

class VectorQueryTerm(BaseQueryTerm):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector: np.ndarray

    @field_validator("vector", mode="before")
    @classmethod
    def cast_vector(cls, v):
        if isinstance(v, dict):  # v is NPArray from json
            return NPArray.model_validate(v).to_array()
        else:
            return v

    @field_serializer('vector', mode='plain')
    def serialize_vector(self, value: np.ndarray) -> NPArray:
        return NPArray.from_array(self.vector)


class TextQueryTerm(BaseQueryTerm):
    txt: str


Query = list[MediaQueryTerm | TextQueryTerm | VectorQueryTerm | VectorIdQueryTerm]


## Query's are an HTTP multipart/form-data request where the files
## (images, audio, videos) go on a separate key.  The files filenames
## must match the corresponding MediaQueryTerm term_id.  So we have
## intermediary types with the `*InForm` suffix, to validate what we
## got on the request before getting the "final" MediaQueryTerm.
##
## One alternative evaluated was having gzipped requests in JSON
## format with the uploaded files in base64.  That would lead to
## requests of roughly the same size.  However, we choose the
## multipart approach instead of gzip+base64+json because we decided
## it would be simpler to maintain with lower CPU usage in the client.

class MediaQueryTermInForm(MediaQueryTerm):
    src: HttpUrl | int | None  # None means bytes in another form part

    @classmethod
    def from_MediaQueryTerm(cls, q: MediaQueryTerm):
        if isinstance(q.src, bytes):
            return cls(**q.model_dump(exclude="src"), src=None)
        else:
            return cls(**q.model_dump())


QueryTermInForm = MediaQueryTermInForm | TextQueryTerm | VectorQueryTerm | VectorIdQueryTerm
QueryTermInFormAdapter = TypeAdapter(QueryTermInForm)
QueryInForm = list[MediaQueryTermInForm | TextQueryTerm | VectorQueryTerm | VectorIdQueryTerm]

def merge_multipart_query_form(
    query_form: list[str], query_form_files: list[UploadFile]
) -> Query:
    query_form = [QueryTermInFormAdapter.validate_json(x) for x in query_form]
    term_ids = {x.term_id for x in query_form}
    if len(term_ids) != len(query_form):
        raise HTTPException(
            400, {"message": "query terms must have unique term_id"}
        )

    ## We hijack the form-data filename to use as term_id
    filename_to_file = {x.filename: x for x in query_form_files}
    if len(filename_to_file) != len(query_form_files):
        raise HTTPException(
            400, {"message": "query files must have unique filenames"}
        )
    if any([x not in term_ids for x in filename_to_file.keys()]):
        raise HTTPException(
            400, {"message": "query files must have query term with matching term_id"}
        )

    query = []
    for term_form in query_form:
        if isinstance(term_form, MediaQueryTermInForm):
            if term_form.src is None:  # get file from query_file
                query.append(
                    MediaQueryTerm(
                        **term_form.model_dump(exclude="src"),
                        src=filename_to_file[term_form.term_id].file.read(),
                    )
                )
            else:
                query.append(MediaQueryTerm(**term_form.model_dump()))
        else:
            query.append(term_form)
    return query


def build_response_query(original: Query, processed: Query) -> QueryInForm:
    """Build the query to be included in the search response.

    The original query is processed for the search.  We need to
    process it back to be included in the response, e.g., strip the
    media content, and convert vector query back into the vector id
    query.

    """
    assert [x.term_id for x in original] == [x.term_id for x in processed]
    r: QueryInForm = []
    for o, p in zip(original, processed):
        if isinstance(o, VectorIdQueryTerm):
            assert isinstance(p, VectorQueryTerm)
            r.append(o)
        elif isinstance(o, MediaQueryTerm):
            r.append(MediaQueryTermInForm.from_MediaQueryTerm(p))
        else:
            r.append(p)
    return r


def parse_old_api_query(
    # Positive queries
    text_queries: list[str],
    image_file_queries: list[bytes],  # user-uploaded images
    audio_file_queries: list[bytes],  # user-uploaded audio files
    image_url_queries: list[HttpUrl],  # URLs to online images
    audio_url_queries: list[HttpUrl],  # URLs to online audio files
    internal_image_queries: list[str],  # ids to internal images
    # Negative queries
    negative_text_queries: list[str],
    negative_image_file_queries: list[bytes],  # user-uploaded images
    negative_audio_file_queries: list[bytes],  # user-uploaded audio files
    negative_image_url_queries: list[HttpUrl],  # URLs to online images
    negative_audio_url_queries: list[HttpUrl],  # URLs to online audio files
    negative_internal_image_queries: list[str],  # ids to internal images
) -> Query:
    """Convert from the *_queries values from API into the "internal" form.
    """
    q = []
    q += [
        TextQueryTerm(term_id=str(uuid.uuid4()), is_negative=False, txt=val)
        for val in text_queries
    ]
    q += [
        MediaQueryTerm(term_id=str(uuid.uuid4()), is_negative=False, src=val, qtype="visual")
        for val in image_file_queries
    ]
    q += [
        MediaQueryTerm(term_id=str(uuid.uuid4()), is_negative=False, src=HttpUrl(val), qtype="visual")
        for val in image_url_queries
    ]
    q += [
        VectorQueryTerm(term_id=str(uuid.uuid4()), is_negative=False, vector_id=val)
        for val in internal_image_queries
    ]
    q += [
        MediaQueryTerm(term_id=str(uuid.uuid4()), is_negative=False, src=val, qtype="audio")
        for val in audio_file_queries
    ]
    q += [
        MediaQueryTerm(term_id=str(uuid.uuid4()), is_negative=False, src=HttpUrl(val), qtype="audio")
        for val in audio_url_queries
    ]
    q += [
        TextQueryTerm(term_id=str(uuid.uuid4()), is_negative=True, txt=val)
        for val in negative_text_queries
    ]
    q += [
        MediaQueryTerm(term_id=str(uuid.uuid4()), is_negative=True, src=val, qtype="visual")
        for val in negative_image_file_queries
    ]
    q += [
        MediaQueryTerm(term_id=str(uuid.uuid4()), is_negative=True, src=HttpUrl(val), qtype="visual")
        for val in negative_image_url_queries
    ]
    q += [
        VectorQueryTerm(term_id=str(uuid.uuid4()), is_negative=True, vector_id=val)
        for val in negative_internal_image_queries
    ]
    q += [
        MediaQueryTerm(term_id=str(uuid.uuid4()), is_negative=True, src=val, qtype="audio")
        for val in negative_audio_file_queries
    ]
    q += [
        MediaQueryTerm(term_id=str(uuid.uuid4()), is_negative=True, src=HttpUrl(val), qtype="audio")
        for val in negative_audio_url_queries
    ]
    return q


class VectorInfo(BaseModel):
    vector_id: str
    media_id: str
    link: str
    thumbnail: str
    bbox: Optional[BBoxXYWH] = None

    @field_validator("bbox", mode="before")
    @classmethod
    def cast_bbox(cls, v):
        if isinstance(v, tuple):  # v is the NamedTuple in feature_extractor module
           return BBoxXYWH(**{k: v for (k, v) in zip('xywh', v)})
        else:
            return v

class VectorResult(VectorInfo):
    distance: round_float

# Metadata for a video/audio/image file, to be sent to the frontend
class MediaInfo(BaseModel):
    id: str
    filename: str
    width: int
    height: int
    media_type: str
    format: str
    duration: round_float
    title: str = ""
    external_metadata: dict = {}

# A subclass of MediaInfo for images
class ImageInfo(MediaInfo):
    pass

# A subclass of MediaInfo for videos
class VideoInfo(MediaInfo):
    timeline_hover_thumbnails: str

class ImageVector(VectorResult):
    pass

class VideoSegment(VectorResult):
    ts: round_float
    te: round_float
    thumbnail_ts: round_float

class VideoAudioResults(BaseModel):
    total: int # maximum number of unmerged_windows that can be returned
    unmerged_windows: list[VideoSegment] # e.g. 7-second windows
    merged_windows: list[VideoSegment] # shots (for edited videos) or merged segments (for unedited videos)
    videos: dict[str, VideoInfo]

class VideoResults(BaseModel):
    total: int # maximum number of unmerged_windows that can be returned
    unmerged_windows: list[VideoSegment] # frames (CLIP) or unmerged 4-second segments (InternVideo/LanguageBind)
    merged_windows: list[VideoSegment] # shots (for edited videos) or merged segments (for unedited videos)
    videos: dict[str, VideoInfo]

class ImageResults(BaseModel):
    total: int # maximum number of images that can be returned e.g. min(1000, num_images_in_project)
    vectors: list[ImageVector]
    images: dict[str, ImageInfo]

class FaceTextShardResults(BaseModel):
    video_audio_results: Optional[VideoAudioResults] = None
    video_results: Optional[VideoResults] = None
    image_results: Optional[ImageResults] = None

class SearchResponse(BaseModel):
    time: float # backend search time in seconds
    query: Query | QueryInForm
    video_audio_results: Optional[VideoAudioResults] # search results from audio stream of video files
    video_results: Optional[VideoResults] # search results from video stream of video files
    image_results: Optional[ImageResults] # search results from image files

class FaceTextSearchResponse(BaseModel):
    time: float # backend search time in seconds
    face_results: FaceTextShardResults
    text_results: FaceTextShardResults

def split_query_terms(query: Query):
    text_queries: list[str] = []
    negative_text_queries: list[str] = []
    image_file_queries: list[bytes] = []
    image_url_queries: list[HttpUrl] = []
    internal_image_queries: list[str] = []
    negative_image_file_queries: list[bytes] = []
    negative_image_url_queries: list[HttpUrl] = []
    negative_internal_image_queries: list[str] = []
    audio_file_queries: list[bytes] = []
    audio_url_queries: list[HttpUrl] = []
    negative_audio_file_queries: list[bytes] = []
    negative_audio_url_queries: list[HttpUrl] = []

    for term in query:
        if isinstance(term, TextQueryTerm):
            if term.is_negative:
                negative_text_queries.append(term.txt)
            else:
                text_queries.append(term.txt)
        elif isinstance(term, MediaQueryTerm):
            if term.qtype == "visual":
                if isinstance(term.src, bytes):
                    (negative_image_file_queries if term.is_negative else image_file_queries).append(term.src)
                elif isinstance(term.src, HttpUrl):
                    (negative_image_url_queries if term.is_negative else image_url_queries).append(term.src)
            elif term.qtype == "audio":
                if isinstance(term.src, bytes):
                    (negative_audio_file_queries if term.is_negative else audio_file_queries).append(term.src)
                elif isinstance(term.src, HttpUrl):
                    (negative_audio_url_queries if term.is_negative else audio_url_queries).append(term.src)
        elif isinstance(term, VectorQueryTerm):
            if term.is_negative:
                negative_internal_image_queries.append(term.vector_id)
            else:
                internal_image_queries.append(term.vector_id)

    return (
        text_queries,
        negative_text_queries,
        image_file_queries,
        image_url_queries,
        internal_image_queries,
        negative_image_file_queries,
        negative_image_url_queries,
        negative_internal_image_queries,
        audio_file_queries,
        audio_url_queries,
        negative_audio_file_queries,
        negative_audio_url_queries,
    )


def patch_precision(config: APIConfig):

    # TODO
    # Instead of monkey-patching, move the serialisation part before sending the response
    # or handle it frontend
    global PRECISION
    PRECISION = config.precision

def add_response_time(func: Callable[..., Awaitable[SearchResponse]]):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        response = await func(*args, **kwargs)
        end_time = time.perf_counter()
        response.time = end_time - start_time
        return response

    return wrapper


class CachedBodyRequest(Request):
    """Request which caches body before parsing form.

    `form()` calls stream() directly to construct `FormData` without
    storing the request content.  `body()` also calls `stream()` but
    stores the content which `stream()` then uses if available.  This
    Request class calls `body()` before `form()` to ensure that the
    stream raw content is kept.  See
    https://github.com/Kludex/starlette/discussions/1933

    This is needed in the POST search with metadata route which
    forwards the request after parsing the form arguments.

    """
    async def form(self, *args, **kwargs):
        await super().body()
        return await super().form(*args, **kwargs)

class CachedBodyRoute(APIRoute):
    """Route that ensures that Request cache body, see CachedBodyRequest
    """
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def cache_body_route_handler(request: Request) -> Response:
            request = CachedBodyRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return cache_body_route_handler
