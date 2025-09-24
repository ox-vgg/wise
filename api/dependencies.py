import logging
import json
from typing import Annotated
from fastapi import Depends
from config import APIConfig

from .services.project import LocalWiseProjectService, RemoteWiseProjectService, ProjectInfo
from .services.embedding import EmbeddingService
from .services.search import LocalSearchService, RemoteSearchService


logger = logging.getLogger(__name__)

ProjectServiceType = LocalWiseProjectService | dict[str, RemoteWiseProjectService]
SearchServiceType = LocalSearchService | RemoteSearchService

# module globals
config: APIConfig
project_service: ProjectServiceType
project_info: ProjectInfo
embedding_service: EmbeddingService
search_service: SearchServiceType

def get_config():
    if config is None:
        raise ValueError("Config not initialized - call init method before using other methods in the module")
    return config

ConfigDep = Annotated[APIConfig, Depends(get_config)]

def get_project_service(config: ConfigDep):
    if project_service is None:
        if config.project_dir is None:
            raise ValueError("project_dir is missing!")
        
        def init_project_service(config: APIConfig):
            if config.remote_projects:
                remote_project_services = [
                    RemoteWiseProjectService(p, config) for p in config.remote_projects
                ]
                remote_project_services = {p.name: p for p in remote_project_services}
                project_service = remote_project_services
            else:

                from pathlib import Path
                project_path = Path(config.project_dir)
                if not project_path.exists() or not project_path.is_dir():
                    raise ValueError(
                        f"Local path does not exist or is not a directory: {project_path}"
                    )
                from .services.project import WiseProject, LocalWiseProjectService
                project = WiseProject(project_path)
                project.load_search_indices(config.index_type, config.nprobe)
                project_service = LocalWiseProjectService(project, config)
            
            return project_service

        global project_service
        project_service = init_project_service(config)
    
    return project_service

ProjectServiceDep = Annotated[ProjectServiceType, Depends(get_project_service)]

def get_project_info(config: ConfigDep, project_service: ProjectServiceDep):
    if project_info is None:
        def init_project_info(project_service: ProjectServiceType):
            if isinstance(project_service, LocalWiseProjectService):
                return project_service.info()
            # remote projects
            name = config.project_dir.name
            return RemoteWiseProjectService.get_info(project_service, name)
        
        global project_info
        project_info = init_project_info(project_service)
    return project_info

ProjectInfoDep = Annotated[ProjectInfo, Depends(get_project_info)]

def get_embedding_service(project_info: ProjectInfoDep):
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
            embedding_service = EmbeddingService.from_ids(feature_extractor_ids)
            
            return embedding_service

        global embedding_service
        embedding_service = init_embedding_service(project_info)
    return embedding_service

EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]

def get_search_service(
        project_service: ProjectServiceDep, 
        embedding_service = EmbeddingServiceDep):
    if search_service is None:
        def init_search_service(project_service: ProjectServiceType, embedding_service: EmbeddingService):
            if isinstance(project_service, LocalWiseProjectService):
                return LocalSearchService(project_service, embedding_service)
            # remote projects
            return RemoteSearchService(project_service, embedding_service)
        global search_service
        search_service = init_search_service(project_service, embedding_service)
    return search_service

SearchServiceDep = Annotated[SearchServiceType, Depends(get_search_service)]

def init(_config: APIConfig):
    global config
    config = _config

    project_service = get_project_service(config)
    project_info = get_project_info(config, project_service)
    embedding_service = get_embedding_service(project_info)
    _ = get_search_service(project_service, embedding_service)
    logger.info("Initialized dependencies module")
    