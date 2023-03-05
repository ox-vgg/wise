import enum
from pydantic import BaseModel
from typing import Optional, Dict, Any


class DatasetType(str, enum.Enum):
    IMAGE_DIR = "image_dir"
    WEBDATASET = "webdataset"


class QueryType(str, enum.Enum):
    NATURAL_LANGUAGE_QUERY = "NATURAL_LANGUAGE_QUERY"
    IMAGE_QUERY = "IMAGE_QUERY"
    IMAGE_CLASSIFICATION_QUERY = "IMAGE_CLASSIFICATION_QUERY"


class ImageInfo(BaseModel):
    filename: str
    width: int
    height: int
    title: str = ""
    caption: str = ""
    copyright: str = ""


class BaseDataset(BaseModel):
    id: Optional[int] = None
    location: str
    type: DatasetType


class DatasetCreate(BaseDataset):
    pass


class Dataset(BaseDataset):
    id: int

    class Config:
        orm_mode = True
        use_enum_values = True


class ImageMetadata(BaseModel):
    id: Optional[int] = None
    dataset_id: int = -1
    dataset_row: Optional[int] = None
    path: str
    size_in_bytes: int
    format: str
    width: int = -1
    height: int = -1
    source_uri: Optional[str] = None
    metadata: Dict[str, Any]  # TODO: tighter type

    class Config:
        orm_mode = True


class Project(BaseModel):
    id: str

    class Config:
        orm_mode = True


class URL(str):
    pass
