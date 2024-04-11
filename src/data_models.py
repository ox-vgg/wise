import enum
from pydantic import ConfigDict, BaseModel
from typing import Optional, Dict, Any
import datetime


class SourceCollectionType(str, enum.Enum):
    IMAGE_DIR = "image_dir"
    WEBDATASET = "webdataset"


class MediaType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class QueryType(str, enum.Enum):
    NATURAL_LANGUAGE_QUERY = "NATURAL_LANGUAGE_QUERY"
    IMAGE_QUERY = "IMAGE_QUERY"
    IMAGE_CLASSIFICATION_QUERY = "IMAGE_CLASSIFICATION_QUERY"


# TODO
class MediaInfo(BaseModel):
    id: str
    filename: str
    media_type: str
    width: int
    height: int
    format: str
    duration: float
    title: str = ""
    caption: str = ""
    copyright: str = ""

class BaseSourceCollection(BaseModel):
    id: Optional[str] = None
    location: str
    type: SourceCollectionType

class SourceCollection(BaseSourceCollection):
    id: str
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class MediaMetadata(BaseModel):
    id: Optional[str] = None
    source_collection_id: str
    path: str
    md5sum: bytes
    size_in_bytes: int
    date_modified: datetime.datetime
    format: str
    width: int
    height: int
    num_frames: int
    duration: float
    model_config = ConfigDict(from_attributes=True)


class VectorMetadata(BaseModel):
    id: int
    modality: MediaType
    media_id: str
    timestamp: Optional[float]
    end_timestamp: Optional[float]


class Project(BaseModel):
    id: str
    version: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


class URL(str):
    pass
