from typing import Dict, List
import io
from PIL import Image
from fastapi import APIRouter, HTTPException, Query, File
from pydantic import BaseModel, validator
import typer
from config import APIConfig
from src import db
from src.projects import (
    get_wise_db_uri,
    get_wise_features_dataset_path,
    get_wise_project_db_uri,
)
from src.repository import WiseProjectsRepo, MetadataRepo
from src.data_models import ImageInfo
from src.ioutils import get_dataset_reader
from src.inference import setup_clip
from src.search import brute_force_search


def raise_(ex):
    raise ex


def get_search_router(config: APIConfig):
    if config.project_id is None:
        raise typer.BadParameter("project id is missing!")

    router = APIRouter(
        tags=["search"],
    )
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

    def make_response(queries, dist, ids, get_metadata_fn):
        return {
            _q: [
                SearchResponse(
                    thumbnail=f"{_metadata.source_uri}",
                    link=f"{_metadata.source_uri}",
                    distance=_dist,
                    info=ImageInfo(
                        filename=_metadata.path,
                        width=_metadata.width,
                        height=_metadata.height,
                    ),
                )
                for _dist, _metadata in zip(
                    top_dist,
                    map(
                        lambda _id: get_metadata_fn(_id),
                        top_ids,
                    ),
                )
            ]
            for _q, top_dist, top_ids in zip(queries, dist, ids)
        }

    _prefix = config.query_prefix.strip()
    model_name, num_files, reader = get_dataset_reader(image_features, driver="family")
    extract_image_features, extract_text_features = setup_clip(model_name)
    project_engine = db.init_project(get_wise_project_db_uri(project_id))

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

            response = make_response(q, dist, ids, get_metadata)

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

            response = make_response(["image"], dist, ids, get_metadata)
        return response

    return router
