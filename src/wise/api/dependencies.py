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

import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from wise.config import APIConfig

from .services.embedding import EmbeddingService
from .services.project import (
    LocalWiseProjectService,
    ProjectInfo,
    RemoteWiseProjectService,
)
from .services.search import LocalSearchService, RemoteSearchService


logger = logging.getLogger(__name__)

ProjectServiceType = LocalWiseProjectService | dict[str, RemoteWiseProjectService]
SearchServiceType = LocalSearchService | RemoteSearchService

# module globals
config: APIConfig = None
project_service: ProjectServiceType = None
project_info: ProjectInfo = None
embedding_service: EmbeddingService = None
search_service: SearchServiceType = None

def get_config():
    if config is None:
        raise ValueError("Config not initialized - call init method before using other methods in the module")
    return config

ConfigDep = Annotated[APIConfig, Depends(get_config)]

def get_project_service(config: ConfigDep):
    global project_service
    if project_service is None:
        if config.project_dir is None:
            raise ValueError("project_dir is missing!")

        def init_project_service(config: APIConfig):
            if config.remote_projects:
                remote_project_services = [
                    RemoteWiseProjectService(p, config) for p in config.remote_projects
                ]
                remote_project_services = {p.name: p for p in remote_project_services}
                _project_service = remote_project_services
            else:

                project_path = Path(config.project_dir)
                if not project_path.exists() or not project_path.is_dir():
                    raise ValueError(
                        f"Local path does not exist or is not a directory: {project_path}"
                    )
                from wise.wise_project import WiseProject

                from .services.project import LocalWiseProjectService
                project = WiseProject(project_path, read_only=True)
                project.load_search_indices(config.index_type, config.nprobe)
                _project_service = LocalWiseProjectService(project, config)

            return _project_service

        project_service = init_project_service(config)

    return project_service

ProjectServiceDep = Annotated[ProjectServiceType, Depends(get_project_service)]

def get_project_info(config: ConfigDep, project_service: ProjectServiceDep):
    global project_info
    if project_info is None:
        def init_project_info(project_service: ProjectServiceType):
            if isinstance(project_service, LocalWiseProjectService):
                return project_service.info()
            # remote projects
            name = config.project_dir.name
            return RemoteWiseProjectService.get_info(project_service, name)

        project_info = init_project_info(project_service)
    return project_info

ProjectInfoDep = Annotated[ProjectInfo, Depends(get_project_info)]


def get_embedding_service(config: ConfigDep, project_info: ProjectInfoDep):
    global embedding_service
    if embedding_service is None:
        def init_embedding_service(project_info: ProjectInfo):
            active_search_targets = project_info.search_targets
            logger.info(
                "Loaded the following search indices:\n%s",
                json.dumps(active_search_targets, indent=4),
            )

            feature_extractor_ids = list(
                dict.fromkeys(
                    [
                        x
                        for media_type in active_search_targets
                        for x in active_search_targets[media_type]
                        if "metadata" not in x
                    ]
                )
            )  # unique values
            return EmbeddingService.from_ids(
                feature_extractor_ids, config.feature_extractor_config
            )
        embedding_service = init_embedding_service(project_info)
    return embedding_service


EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]


def get_search_service(
    project_service: ProjectServiceDep, embedding_service: EmbeddingServiceDep
):
    global search_service
    if search_service is None:
        def init_search_service(project_service: ProjectServiceType, embedding_service: EmbeddingService):
            if isinstance(project_service, LocalWiseProjectService):
                return LocalSearchService(project_service, embedding_service)
            # remote projects
            return RemoteSearchService(project_service, embedding_service)
        search_service = init_search_service(project_service, embedding_service)
    return search_service


SearchServiceDep = Annotated[SearchServiceType, Depends(get_search_service)]

def init(_config: APIConfig):
    global config
    config = _config

    project_service = get_project_service(config)
    project_info = get_project_info(config, project_service)
    embedding_service = get_embedding_service(config, project_info)
    _ = get_search_service(project_service, embedding_service)
    logger.info("Initialized dependencies module")
