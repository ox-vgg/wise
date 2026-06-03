#!/usr/bin/env python3

## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from pathlib import Path
from typing import Literal, Optional

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from typing_extensions import Self


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
    #
    # The redirect_media_url_num_components controls the number of
    # filepath parts to use.  A positive counts the number of elements
    # from the filename, a negative counts from the root of the
    # filepath.  For example:
    #
    #    Filepath                num_components = 3   num_components = -2
    #    ---------------------   ------------------   -------------------
    #    /srv/data/c/d/e.jpg     c/d/e.jpg            c/d/e.jpg
    #    /srv/data/c/f.jpg       data/c/f.jpg         c/f.jpg
    #    /srv/data/g/x/y/z.jpg   x/y/z.jpg            g/x/y/z.jpg
    #
    # So, if all your media have roughly the same number of elements
    # in the source_collection path, use a negative number to remove
    # it.  If instead, all you media have the same number of parts in
    # the media.path, use a positive number.
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

    # Facets enable users to explore a dataset based on anchors such as as people,
    # locations, acoustic events, etc. The facets must be first defined using the
    # tools contained in scripts/explore/ folder.
    # See docs/Explore.md for more details.
    enable_facets: bool = False

    # feature extractor configuration
    # key must be the feature extractor id
    # value is a dictionary with the configuration for the feature extractor
    # e.g. {"open_clip": {"device": "cuda:0", "warmup": True}}
    feature_extractor_config: dict[str, dict] = {"default": {}}

    # Optional: adaptive k settings for KNN in face+text queries.
    # Example:
    # {
    #   "deepinsight/insightface/buffalo_l/_unknown": {
    #     "k_default": 500,
    #     "k_expanded": 2000,
    #     "score_threshold": 0.4
    #   }
    # }
    face_text_search_options: dict[str, dict] = {
        "deepinsight/insightface/buffalo_l/_unknown": {
            "k_default": 500,
            "k_expanded": 2000,
            "score_threshold": 0.4,
        }
    }
    # Preferred text feature extractor id for face+text queries.
    # If unset, CLIP-based models are preferred when available.
    face_text_search_text_embedder: Optional[str] = None
    # Reciprocal Rank Fusion parameters for face+text ranking.
    # face_low_weight : weight for face search results with score less than threshold
    face_text_search_rrf: dict = {
        "k": 60,
        "face_weight": 1.0,
        "text_weight": 1.0,
        "face_low_weight": 0.0,
    }
    # Target k for text search in face+text mode (before capping by available vectors).
    face_text_search_text_k: int = 500
    # Optional: override IVF nprobe for face+text text search when using ID constraints
    # as the default nprobe (which is small) may not be sufficient when the search is
    # constrained to a small subset of the index (e.g. all faces matching a specific person).
    # Set to None to disable, higher values may improve recall at the cost of latency.
    face_text_search_nprobe_target: Optional[int] = 4096

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
