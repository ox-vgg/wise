import enum
from pydantic import BaseModel
from typing import Optional, Dict, Any


class ImageInfo(BaseModel):
    filename: str
    width: int
    height: int
    title: str = ""
    caption: str = ""
    copyright: str = ""


class ImageMetadata(BaseModel):
    id: Optional[int] = None
    dataset_id: int = -1
    path: str
    size_in_bytes: int
    format: str
    width: int = -1
    height: int = -1
    source_uri: str
    metadata: Dict[str, Any]  # TODO: tighter type

    class Config:
        orm_mode = True


class URL(str):
    pass
