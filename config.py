from typing import Optional
from pydantic import BaseSettings


class APIConfig(BaseSettings):
    hostname: str = "0.0.0.0"
    port: int = 9670
    top_k: int = 10
    precision: int = 3
    query_prefix: str = "This is a photo of a"
    project_id: str
