from contextlib import ExitStack
from typing import Dict, List
import io
import logging
from pathlib import Path
import tarfile
from PIL import Image
from fastapi import APIRouter, HTTPException, Query, File
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel, validator
import typer

from config import APIConfig
from src import db
from src.projects import (
    get_wise_db_uri,
    get_wise_features_dataset_path,
    get_wise_project_db_uri,
    get_wise_thumbs_dataset_path,
)
from src.repository import WiseProjectsRepo, MetadataRepo, DatasetRepo
from src.data_models import ImageInfo, DatasetType, Dataset
from src.ioutils import (
    get_dataset_reader,
    get_thumbs_reader,
    is_valid_uri,
    get_file_from_tar,
)
from src.inference import setup_clip
from src.search import brute_force_search

logger = logging.getLogger(__name__)


def raise_(ex):
    raise ex


def get_image_router(config: APIConfig):

    if config.project_id is None:
        raise typer.BadParameter("project id is missing!")

    engine = db.init(get_wise_db_uri())
    with engine.connect() as conn:
        project = WiseProjectsRepo.get(conn, config.project_id)
        if project is None:
            raise typer.BadParameter(f"Project {config.project_id} not found!")

    project_id = project.id
    router = APIRouter(
        prefix=f"/images/{project_id}",
        tags=["images"],
    )

    project_engine = db.init_project(get_wise_project_db_uri(project_id))

    @router.get(
        "/{image_id}",
        response_class=FileResponse,
        responses={404: {"content": "text/plain"}, 302: {}},
    )
    def get_image(image_id: int):
        with project_engine.connect() as conn:
            metadata = MetadataRepo.get(conn, image_id)
            if metadata is None:
                return PlainTextResponse(
                    status_code=404, content=f"{image_id} not found!"
                )
            # Send the source_uri if present, or try to read from source
            # we read from
            # Maybe do a HEAD request to check existence before redirect
            # so that we can try to serve the file from disk if present?
            if metadata.source_uri and is_valid_uri(metadata.source_uri):
                return RedirectResponse(metadata.source_uri, status_code=302)

            # Look up the dataset table and find the location and type
            dataset = DatasetRepo.get(conn, metadata.dataset_id)
            if dataset is None:
                return PlainTextResponse(
                    status_code=404, content=f"{image_id} not found!"
                )

            location = Path(dataset.location)

            # Handle case where we read the image from disk, but it may not be there
            if dataset.type == DatasetType.IMAGE_DIR:
                # metadata.source_uri will be None, so we have to search for it on disk
                file_path = location / metadata.path
                if file_path.is_file():
                    return FileResponse(
                        file_path, media_type=f"image/{metadata.format.lower()}"
                    )
                return PlainTextResponse(
                    status_code=404, content=f"{image_id} not found!"
                )

            # Try to extract from local file if present
            if not location.is_file() or not tarfile.is_tarfile(location):
                return PlainTextResponse(
                    status_code=404, content=f"{image_id} not found!"
                )
            try:
                file_iter = get_file_from_tar(location, metadata.path.lstrip("#"))
                return StreamingResponse(
                    file_iter, media_type=f"image/{metadata.format.lower()}"
                )
            except Exception as e:
                logger.exception(f"Exception when reading image {image_id}")
                return PlainTextResponse(
                    status_code=404, content=f"{image_id} not found!"
                )

    return router


def get_search_router(config: APIConfig):
    if config.project_id is None:
        raise typer.BadParameter("project id is missing!")

    engine = db.init(get_wise_db_uri())
    with engine.connect() as conn:
        project = WiseProjectsRepo.get(conn, config.project_id)
        if project is None:
            raise typer.BadParameter(f"Project {config.project_id} not found!")

    project_id = project.id
    image_features = get_wise_features_dataset_path(
        project_id,
        "features",
        "images",
    )

    class SearchResponse(BaseModel):
        link: str
        thumbnail: str
        distance: float
        info: ImageInfo

        @validator("distance")
        def round_distance(cls, v):
            return round(v, config.precision)

    def make_response(queries, dist, ids, get_metadata_fn, get_thumbs_fn):
        return {
            _q: [
                SearchResponse(
                    thumbnail=_thumb,
                    link=f"{_metadata.source_uri if _metadata.source_uri else f'images/{project_id}/{_metadata.id}'}",
                    distance=_dist,
                    info=ImageInfo(
                        filename=_metadata.path,
                        width=_metadata.width,
                        height=_metadata.height,
                    ),
                )
                for _dist, _metadata, _thumb in zip(
                    top_dist,
                    map(
                        lambda _id: get_metadata_fn(_id),
                        top_ids,
                    ),
                    get_thumbs_fn(top_ids),
                )
            ]
            for _q, top_dist, top_ids in zip(queries, dist, ids)
        }

    _prefix = config.query_prefix.strip()
    model_name, num_files, reader = get_dataset_reader(image_features, driver="family")
    _, extract_image_features, extract_text_features = setup_clip(model_name)
    project_engine = db.init_project(get_wise_project_db_uri(project_id))
    thumbs = get_wise_thumbs_dataset_path(project_id)
    router_cm = ExitStack()
    thumbs_reader = router_cm.enter_context(get_thumbs_reader(thumbs, driver="family"))

    router = APIRouter(
        tags=["search"],
        on_shutdown=[lambda: print("shutting down") and router_cm.close()],
    )

    @router.get("/search", response_model=Dict[str, List[SearchResponse]])
    async def natural_language_search(
        q: List[str] = Query(
            default=[],
        ),
        top_k: int = Query(config.top_k, gt=0, le=min(200, num_files)),
    ):
        if len(q) == 0:
            raise HTTPException(
                400, {"message": "Must be called with search query term"}
            )

        prefixed_queries = [f"{_prefix} {x.strip()}".strip() for x in q]
        text_features = extract_text_features(prefixed_queries)
        dist, ids = brute_force_search(
            (x for x, _ in reader()), text_features, top_k=top_k
        )

        with project_engine.connect() as conn:

            def get_metadata(_id):
                m = MetadataRepo.get(conn, int(_id) + 1)
                if m is None:
                    raise RuntimeError()
                return m

            response = make_response(q, dist, ids, get_metadata, thumbs_reader)

        return response

    @router.post("/search", response_model=Dict[str, List[SearchResponse]])
    async def image_search(
        q: bytes = File(),
        top_k: int = Query(config.top_k, gt=0, le=min(200, num_files)),
    ):
        with Image.open(io.BytesIO(q)) as im:
            query_features = extract_image_features([im])
            dist, ids = brute_force_search(
                (x for x, _ in reader()), query_features, top_k=top_k
            )

        with project_engine.connect() as conn:

            def get_metadata(_id):
                m = MetadataRepo.get(conn, int(_id) + 1)
                if m is None:
                    raise RuntimeError()
                return m

            response = make_response(["image"], dist, ids, get_metadata, thumbs_reader)
        return response

    return router
