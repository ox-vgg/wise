from contextlib import ExitStack
from typing import Dict, List
import io
import logging
from pathlib import Path
import tarfile
from PIL import Image
from fastapi import APIRouter, HTTPException, Query, File
from fastapi.responses import (
    Response,
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel, validator
import typer

from config import APIConfig
from src import db
from src.projects import WiseTree, WiseProjectTree
from src.repository import WiseProjectsRepo, MetadataRepo, DatasetRepo
from src.data_models import ImageInfo, ImageMetadata, DatasetType, Dataset
from src.ioutils import (
    H5Datasets,
    get_h5reader,
    get_model_name,
    get_counts,
    is_valid_uri,
    get_file_from_tar,
)
from src.inference import setup_clip, CLIPModel
from src.enums import IndexType
from src.search import read_index
from src.utils import convert_uint8array_to_base64

logger = logging.getLogger(__name__)


def raise_(ex):
    raise ex


def get_project_router(config: APIConfig):
    if config.project_id is None:
        raise typer.BadParameter("project id is missing!")

    engine = db.init(WiseTree.dburi)
    with engine.connect() as conn:
        project = WiseProjectsRepo.get(conn, config.project_id)
        if project is None:
            raise typer.BadParameter(f"Project {config.project_id} not found!")

    project_id = project.id
    router = APIRouter(prefix=f"/{project_id}", tags=[f"{project_id}"])
    router.include_router(_get_project_data_router(config))
    router.include_router(_get_search_router(config))

    return router


def _get_project_data_router(config: APIConfig):
    """
    Returns a router with API routes for reading the project data

    Provides
    - /images/{_id} -> Access the original image from URL / disk
    - /thumbs/{_id} -> Read the thumbnail as bytes from dataset
    - /metadata/{_id} -> Read the metadata associated with the specific sample
    - /info -> Read the project level metadata
    """

    project_id = config.project_id
    project_tree = WiseProjectTree(project_id)
    vds_path = project_tree.latest

    router_cm = ExitStack()
    router = APIRouter(
        on_shutdown=[lambda: print("shutting down") and router_cm.close()],
    )
    thumbs_reader = router_cm.enter_context(
        get_h5reader(vds_path)(H5Datasets.THUMBNAILS)
    )
    project_engine = db.init_project(project_tree.dburi)

    @router.get(
        "/images/{image_id}",
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

    @router.get(
        "/thumbs/{_id}",
        response_class=Response,
        responses={200: {"content": "image/jpeg"}, 404: {"content": "text/plain"}},
    )
    def get_thumbnail(_id: int):
        try:
            return Response(
                content=bytes(thumbs_reader([_id])[0]),
                media_type="image/jpeg",
                status_code=200,
            )
        except IndexError:
            return PlainTextResponse(status_code=404, content=f"{_id} not found!")
        except Exception as e:
            logger.error(f"Failed to get thumbnail {_id}, {e}")
            raise HTTPException(status_code=500)

    @router.get(
        "/metadata/{_id}",
        response_model=ImageMetadata,
        response_model_exclude=set(["id", "dataset_id", "dataset_row"]),
        responses={200: {"content": "application/json"}},
    )
    def get_metadata(_id: int):
        with project_engine.connect() as conn:
            metadata = MetadataRepo.get(conn, _id)
            if metadata is None:
                raise HTTPException(status_code=404, detail=f"Metadata not found!")
            return metadata

    @router.get("/info")
    def get_info():
        model_name = get_model_name(vds_path)
        counts = get_counts(vds_path)
        return {
            "id": project_id,
            "model": model_name,
            "num_images": counts[H5Datasets.IMAGE_FEATURES],
        }

    return router


def _get_search_router(config: APIConfig):

    project_id = config.project_id
    project_tree = WiseProjectTree(project_id)
    index_type = IndexType[config.index_type]

    class SearchResponse(BaseModel):
        link: str
        thumbnail: str
        distance: float
        info: ImageInfo

        @validator("distance")
        def round_distance(cls, v):
            return round(v, config.precision)

    def make_basic_response(queries, dist, ids, get_metadata_fn):
        return {
            _q: [
                SearchResponse(
                    thumbnail='',
                    link=f"{_metadata.source_uri if _metadata.source_uri else f'images/{_metadata.id}'}",
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

    def make_full_response(queries, dist, ids, get_metadata_fn, get_thumbs_fn):
        return {
            _q: [
                SearchResponse(
                    thumbnail=convert_uint8array_to_base64(_thumb),
                    link=f"{_metadata.source_uri if _metadata.source_uri else f'images/{_metadata.id}'}",
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
    project_engine = db.init_project(project_tree.dburi)

    # TODO Big assumption - all datasets were written with same model name
    # Should Read / Write to project db instead

    # Get model name
    vds_path = project_tree.latest
    model_name = CLIPModel[get_model_name(vds_path)]

    # load the feature search index
    index_filename = project_tree.index(index_type)
    logger.info(f"Loading faiss index from {index_filename}")
    index = read_index(index_filename)
    if hasattr(index, "nprobe"):
        index.nprobe = getattr(config, 'nprobe', 32)

    # Get counts
    counts = get_counts(vds_path)
    assert counts[H5Datasets.IMAGE_FEATURES] == counts[H5Datasets.IDS]
    num_files = counts[H5Datasets.IMAGE_FEATURES]

    _, _, extract_image_features, extract_text_features = setup_clip(model_name)

    router_cm = ExitStack()
    thumbs_reader = router_cm.enter_context(
        get_h5reader(vds_path)(H5Datasets.THUMBNAILS)
    )

    router = APIRouter(
        on_shutdown=[lambda: print("shutting down") and router_cm.close()],
    )

    @router.get("/search", response_model=Dict[str, List[SearchResponse]])
    async def natural_language_search(
        q: List[str] = Query(
            default=[],
        ),
        start: int = Query(0, gt=-1, le=980),
        end: int = Query(20, gt=0, le=1000),
        thumbs: int = 1
    ):
        if len(q) == 0:
            raise HTTPException(400, {"message": "missing search query"})

        end = min(end, num_files)
        if start > end:
            raise HTTPException(
                400, {"message": "'start' cannot be greater than 'end'"}
            )
        if (end - start) > 50 and thumbs == 1:
            raise HTTPException(
                400, {"message": "cannot return more than 50 results at a time when thumbs=1"}
            )
        prefixed_queries = [f"{_prefix} {x.strip()}".strip() for x in q]
        text_features = extract_text_features(prefixed_queries)

        dist, ids = index.search(text_features, end)
        with project_engine.connect() as conn:
            def get_metadata(_id):
                m = MetadataRepo.get(conn, int(_id))
                if m is None:
                    raise RuntimeError()
                return m

            if thumbs == 0:
                response = make_basic_response(q,
                                               dist[[0], start:end],
                                               ids[[0], start:end],
                                               get_metadata
                )
            else:
                response = make_full_response(q,
                                              dist[[0], start:end],
                                              ids[[0], start:end],
                                              get_metadata,
                                              thumbs_reader
                )
        return response

    @router.post("/search", response_model=Dict[str, List[SearchResponse]])
    async def image_search(
        q: bytes = File(),
        start: int = Query(0, gt=-1, le=980),
        end: int = Query(20, gt=0, le=1000),
    ):
        end = min(end, num_files)
        if start > end:
            raise HTTPException(
                400, {"message": "'start' cannot be greater than 'end'"}
            )
        if (end - start) > 50:
            raise HTTPException(
                400, {"message": "cannot return more than 50 results at a time"}
            )
        with Image.open(io.BytesIO(q)) as im:
            query_features = extract_image_features([im])
            dist, ids = index.search(query_features, end)

        with project_engine.connect() as conn:

            def get_metadata(_id):
                m = MetadataRepo.get(conn, int(_id))
                if m is None:
                    raise RuntimeError()
                return m

            response = make_response(
                ["image"],
                dist[[0], start:end],
                ids[[0], start:end],
                get_metadata,
                thumbs_reader,
            )
        return response

    return router
