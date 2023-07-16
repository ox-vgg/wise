from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Set


class APIConfig(BaseSettings):
    hostname: str = "0.0.0.0"
    port: int = 9670
    top_k: int = 10
    precision: int = 3
    query_prefix: str = "This is a photo of a"
    index_type: str = "IndexFlatIP"
    nprobe: int = 1024
    query_blocklist: Set[str] = set()
    project_id: str
