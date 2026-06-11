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

import logging

from fastapi import APIRouter, HTTPException, Request

from wise.api import common
from wise.api.dependencies import ProjectInfoDep, ProjectServiceDep


logger = logging.getLogger(__name__)


router = APIRouter()

@router.get(
    "/related-vectors/{_vector_id}",
    response_model=list[common.VectorInfo],
    responses={200: {"content": "application/json"}},
)
async def get_related_vectors(_vector_id: int, media_id: str, projects: ProjectServiceDep):
    shard_id, _ = media_id.rsplit('/', 2)
    project = projects.get(shard_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project shard {shard_id} not found!")
    
    response = await project.related_vectors(_vector_id)
    return response


@router.api_route(
    "/shard/{shard_id}/{full_path:path}",
    methods=["GET", "HEAD"],
)
async def forward(shard_id: str, full_path: str, request: Request,  projects: ProjectServiceDep):
    """
    Forward the request to a remote project based on the project_id
    """
    logger.info('Forwarding request to remote shard %s for path %s', shard_id, full_path)
    
    project = projects.get(shard_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project shard {shard_id} not found!")
    
    response = await project.forward(full_path, request)
    return response

@router.get("/info")
async def get_info(info: ProjectInfoDep):
    return info.normalized().model_dump(by_alias=True)
