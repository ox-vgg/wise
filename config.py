from typing_extensions import Self
from pydantic import model_validator

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from typing import Literal, Optional
from pathlib import Path

class APIConfig(BaseSettings):

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    model_config = SettingsConfigDict(
        yaml_file="wise_config.yaml", env_file_encoding="utf-8"
    )
    project_dir: Path
    command: Literal['serve', 'create_index', 'extract_features', 'search']

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
    query_blocklist: set[str] = set()
    project_dir: Path
    remote_projects: set[str] = set()
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

    # enable profiling for development mode
    enable_profiling: bool = False

    # feature extractor configuration
    # key must be the feature extractor id
    # value is a dictionary with the configuration for the feature extractor
    # e.g. {"open_clip": {"device": "cuda:0", "warmup": True}}
    feature_extractor_config: dict[str, dict] = {"default": {}}

    @model_validator(mode='after')
    def validate_feature_extractor_config(self) -> Self:
        if 'default' not in self.feature_extractor_config:
            self.feature_extractor_config['default'] = {}

        if 'warmup' not in self.feature_extractor_config['default']:
            if self.command == 'serve':
                # warmup in serve when not in development mode
                self.feature_extractor_config['default']['warmup'] = self.mode != 'development'

            elif self.command == 'extract_features':
                # warmup in extract_features always
                self.feature_extractor_config['default']['warmup'] = True

            else:
                # default to False if not set
                self.feature_extractor_config['default']['warmup'] = False

        return self

    @model_validator(mode="after")
    def check_project(self) -> Self:
        if self.remote_projects:
            # remote projects are provided, no need to check local project dir
            return self

        # Local project dir must be provided and must exist for all commands except 'extract_features'
        if self.command != "extract_features" and not (
            self.project_dir.exists() and self.project_dir.is_dir()
        ):
            raise ValueError(
                f"Local project does not exist or is not a directory: {self.project_dir}"
            )

        return self

    @model_validator(mode="after")
    def check_redirect_config(self) -> Self:
        if (
            self.redirect_media_url_by_path
            and self.redirect_media_url_num_components < 1
        ):
            raise ValueError(
                "redirect_media_url_num_components must be greater than 0 when redirect_media_url_by_path is True"
            )
        return self
