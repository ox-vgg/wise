from typing import Optional
from pydantic import BaseSettings


class APIConfig(BaseSettings):
    port: int = 8000
    top_k: int = 10
    precision: int = 3
    query_prefix: str = "This is a photo of a"
    project_id: Optional[str] = None
