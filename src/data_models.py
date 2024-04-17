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
    AV = "av"


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


class SourceCollection(BaseModel):
    id: Optional[int] = None
    location: str
    type: SourceCollectionType
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class MediaMetadata(BaseModel):
    id: Optional[int] = None
    source_collection_id: int
    media_type: MediaType
    path: str
    hash: bytes
    size_in_bytes: int
    date_modified: datetime.datetime
    format: str
    width: int
    height: int
    num_frames: int
    duration: float
    model_config = ConfigDict(from_attributes=True)


class VectorMetadata(BaseModel):
    id: Optional[int] = None
    modality: MediaType
    media_id: int
    timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None


class Project(BaseModel):
    id: str
    version: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


class ExtraMediaMetadata(BaseModel):
    media_id: int
    external_id: Optional[str] = None
    metadata: Dict[str, Any]  # TODO: narrow the type


class URL(str):
    pass
