from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Set


class APIConfig(BaseSettings):
    mode: Literal['production', 'development'] = 'production'
    hostname: str = "0.0.0.0"
    port: int = 9670
    top_k: int = 10
    precision: int = 3
    query_prefix: str = "This is a photo of a"
    text_queries_weight: float = 2.0
    negative_queries_weight: float = 0.2
    index_type: str = "IndexFlatIP"
    nprobe: int = 1024
    query_blocklist: Set[str] = set()
    project_id: str
