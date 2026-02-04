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

from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, Optional, TypedDict
import base64
import functools
import time
import numpy as np
from pydantic import BaseModel, PlainSerializer, field_validator
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


class InternalQTerm(TypedDict):
    sign: Literal["positive", "negative"]
    modality: Literal["image", "audio", "text"]
    val: bytes | str | np.ndarray


def api_query_to_internal_q(
    # Positive queries
    text_queries: list[str],
    image_file_queries: list[bytes],  # user-uploaded images
    audio_file_queries: list[bytes],  # user-uploaded audio files
    image_url_queries: list[str],  # URLs to online images
    audio_url_queries: list[str],  # URLs to online audio files
    internal_image_queries: list[str],  # ids to internal images
    # Negative queries
    negative_text_queries: list[str],
    negative_image_file_queries: list[bytes],  # user-uploaded images
    negative_audio_file_queries: list[bytes],  # user-uploaded audio files
    negative_image_url_queries: list[str],  # URLs to online images
    negative_audio_url_queries: list[str],  # URLs to online audio files
    negative_internal_image_queries: list[str],  # ids to internal images
) -> list[InternalQTerm]:
    """Convert from the *_queries values from API into the "internal" form.
    """
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
        if v is None:
            return v
        elif isinstance(v, BBoxXYWH):
            return v
        elif isinstance(v, dict):
            return BBoxXYWH(**{k: v[k] for k in ['x', 'y', 'w', 'h']})
        else:  # v is the NamedTuple in feature_extractor module
            return BBoxXYWH(**{k: v for (k, v) in zip('xywh', v)})
        
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

class SearchResponse(BaseModel):
    time: float # backend search time in seconds
    video_audio_results: Optional[VideoAudioResults] # search results from audio stream of video files
    video_results: Optional[VideoResults] # search results from video stream of video files
    image_results: Optional[ImageResults] # search results from image files

class NPArray(BaseModel):
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
