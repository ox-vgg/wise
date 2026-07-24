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

import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, dataclasses


class SourceCollectionType(Enum):
    DIR = "dir"
    WEBDATASET = "webdataset"


class MediaChunkType(Enum):
    AUDIO = "audio"
    VIDEO = "video"
    THUMBNAILS = "thumbnails"
    IMAGE = "image"


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    AV = "av"

    @classmethod
    def from_modality(cls, modality_type: "ModalityType"):
        return cls[modality_type.name]


class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

    @classmethod
    def from_media(cls, media_type: MediaType | MediaChunkType):
        return cls[media_type.name]


class SourceCollection(BaseModel):
    id: Optional[int] = None
    location: str
    type: SourceCollectionType
    model_config = ConfigDict(from_attributes=True)


class MediaMetadata(BaseModel):
    id: Optional[int] = None
    source_collection_id: int
    path: str
    checksum: bytes
    size_in_bytes: int
    date_modified: datetime.datetime
    media_type: MediaType
    format: str
    width: int
    height: int
    num_frames: int
    duration: float
    model_config = ConfigDict(from_attributes=True)


class MediaMetadataWithSource(MediaMetadata):
    source_collection: SourceCollection

    @property
    def full_path(self) -> Path:
        return Path(self.source_collection.location) / self.path


class VectorMetadata(BaseModel):
    id: Optional[int] = None
    modality: ModalityType
    feature_extractor_id: Optional[str] = None
    media_id: int
    timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None


class VectorAndMediaMetadata(VectorMetadata, MediaMetadata):
    external_metadata: dict = {}
    pass


class ThumbnailMetadata(BaseModel):
    id: Optional[int] = None
    media_id: int
    timestamp: Optional[float] = None
    content: bytes


class ExtraMediaMetadata(BaseModel):
    media_id: int
    external_id: Optional[str] = None
    metadata: dict[str, Any]  # TODO: narrow the type


class VideoShot(BaseModel):
    id: int
    media_id: int
    ts: float
    te: float
    shot_scale: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


@dataclasses.dataclass
class DatasetPayload(object):
    id: Any
    path: str
    media_type: MediaType
