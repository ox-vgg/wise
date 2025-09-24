import logging

from .. import common


from fastapi import HTTPException, APIRouter, Request
from fastapi.responses import (
    StreamingResponse,
)
from ..dependencies import ProjectServiceDep, ProjectInfoDep


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
    response_class=StreamingResponse,
    methods=['GET', 'HEAD'],
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
    return info.model_dump(by_alias=True)
