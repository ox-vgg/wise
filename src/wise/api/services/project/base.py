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

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce

from pydantic import BaseModel, Field

from wise.data_models import MediaType


class Filter(BaseModel):
    name: str
    description: str
    options: list

class ShotBasedFilters(BaseModel):
    shot_scale: Filter
    # extend with more shot filters like camera motion etc.

class ProjectInfo(BaseModel):
    name: str = Field(alias="project_name")
    num_vectors: int # total number of vectors in the project
    num_media_files: int # total number of media files in the project
    num_thumbnails: int
    num_shots: int
    media_file_counts: dict[MediaType, int] # e.g., {"image": 100, "video": 50, "audio": 20}
    total_duration: float # in seconds
    models: dict[MediaType, list[str]]
    shot_based_filters: ShotBasedFilters | None = None
    search_targets: dict[MediaType, list[str]] = {}
    enable_facets: bool = False

    def normalized(self) -> "ProjectInfo":
        """Return a copy with deterministic ordering for list fields."""
        info = self.model_copy(deep=True)
        info.models = {
            media_type: sorted(model_list)
            for media_type, model_list in info.models.items()
        }
        # Do not sort search_targets here, as their order is explicitly
        # defined by config.search_target_order and sorted in the LocalProjectService.
        info.search_targets = {
            media_type: targets
            for media_type, targets in info.search_targets.items()
        }
        return info

    @classmethod
    def reduce(cls, name, info: list["ProjectInfo"]) -> "ProjectInfo":
        def merge_(a: ProjectInfo, b: ProjectInfo) -> ProjectInfo:
            a.num_vectors += b.num_vectors
            a.num_media_files += b.num_media_files
            a.num_thumbnails += b.num_thumbnails
            a.total_duration += b.total_duration
            a.num_shots += b.num_shots
            for media_type, count in b.media_file_counts.items():
                if media_type in a.media_file_counts:
                    a.media_file_counts[media_type] += count
                else:
                    a.media_file_counts[media_type] = count
            for media_type, model_list in b.models.items():
                if media_type in a.models:
                    a.models[media_type] = list(dict.fromkeys(a.models[media_type] + model_list))
                else:
                    a.models[media_type] = model_list

            for media_type, targets in b.search_targets.items():
                if media_type in a.search_targets:
                    a.search_targets[media_type] = list(dict.fromkeys(a.search_targets[media_type] + targets))
                else:
                    a.search_targets[media_type] = targets

            a.shot_based_filters = a.shot_based_filters or b.shot_based_filters
            a.enable_facets = a.enable_facets or b.enable_facets
            return a

        merged = reduce(merge_, info)
        merged.name = name
        return merged

class WiseProjectService(ABC):
    @abstractmethod
    def __init__(self, project_uri: str):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def info(self) -> ProjectInfo:
        raise NotImplementedError
