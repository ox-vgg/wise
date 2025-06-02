from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Set, Optional, Dict
from pathlib import Path

class APIConfig(BaseSettings):
    mode: Literal['production', 'development'] = 'production'
    listen_address: str = "0.0.0.0"
    port: int = 9670
    top_k: int = 10
    precision: int = 3
    query_prefix: str = "This is a photo of a"
    text_queries_weight: float = 2.0
    negative_queries_weight: float = 0.2
    index_type: str = "IndexFlatIP"
    nprobe: int = 1024
    query_blocklist: Set[str] = set()
    project_dir: Path
    thumbnail_project_dir: Optional[Path] = None # "condensed-movies-roberta-2013"

    # If you want to serve the media files from a different static file server,
    # set redirect_media_url_by_path to True to redirect the media urls from 
    # /media/{media_id} to {config.redirect_media_url_prefix}/media/{file_path}
    #
    # For example, requests to http://server:port/prefix/project-name/media/1 gets
    # redirected to http://another_server/another_prefix/[path]/[to]/filename.mp4
    # The file_path can either be just the filename (media.path) or
    # redirect_media_url_num_components from the end of the absolute path
    # i.e. (source_collection.location / media.path)
    redirect_media_url_by_path: bool = False
    redirect_media_url_prefix: str = "."
    redirect_media_url_num_components: int = 1

    # flag to configure if shots must be used
    # if the flag is set to True, wise will look for shots table and use it, and raise an error if it is not found
    use_shots: bool = False

    # define the order in which search targets (or feature_extractor_id) are listed
    # this order is used by the frontend to display the search targets in the UI
    search_target_order: list[str] = ["open_clip", "insightface", "owlv2", "clap", "wise/metadata"]
