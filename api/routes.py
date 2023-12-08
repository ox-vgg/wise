from contextlib import ExitStack
from typing import Dict, List, Union
import io
import logging
from pathlib import Path
import tarfile
from PIL import Image
from numpy import ndarray, array, zeros, average, expand_dims, float32
from numpy.random import default_rng
from numpy.linalg import norm
from tempfile import NamedTemporaryFile
from torch.hub import download_url_to_file
from fastapi import APIRouter, HTTPException, Query, File, Form
from fastapi.responses import (
    Response,
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import field_validator, BaseModel
import typer
import csv
import json
import os

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
    router.include_router(_get_report_image_router(config))
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


def _get_report_image_router(config: APIConfig):
    router_cm = ExitStack()
    router = APIRouter(
        on_shutdown=[lambda: print("shutting down") and router_cm.close()],
    )

    @router.post("/report")
    def report_image(file_queries: List[bytes] = File([]), url_queries: List[str] = Form([]), text_queries: List[str] = Form([]),
                     sourceURI: str = Form(), reasons: List[str] = Form([])):
        # TODO implement code to store data in database
        # For now, we are saving the reports in a CSV file
        report_filename = 'data/reported_images.csv'
        fieldnames = ['text_queries', 'url_queries', 'file_queries', 'sourceURI', 'reasons']

        # Write header row if the file doesn't exist
        if not os.path.exists(report_filename):
            with open(report_filename, 'a', newline='') as report_file:
                csv.writer(report_file).writerow(fieldnames)

        # Write data row
        with open(report_filename, 'a', newline='') as report_file:
            writer = csv.DictWriter(report_file, fieldnames=fieldnames)
            writer.writerow({
                'text_queries': json.dumps(text_queries),
                'url_queries': json.dumps(url_queries),
                'file_queries': json.dumps(
                    # to prevent the CSV file from getting too large, we store a placeholder text ('uploaded image')
                    # instead of storing the image file
                    ['uploaded image' for _ in file_queries]),
                'sourceURI': sourceURI,
                'reasons': json.dumps(reasons)
            })

        return PlainTextResponse(
            status_code=200, content="Image has been reported"
        )

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

        @field_validator("distance")
        @classmethod
        def round_distance(cls, v):
            return round(v, config.precision)

    def make_basic_response(queries, dist, ids, get_metadata_fn):
        return make_full_response(queries, dist, ids, get_metadata_fn, get_thumbs_fn=None)

    def make_full_response(queries, dist, ids, get_metadata_fn, get_thumbs_fn=None):
        valid_indices = [[i for i, x in enumerate(top_ids) if x != -1] for top_ids in ids]
        valid_ids = [[top_ids[x] for x in indices] for indices, top_ids in zip(valid_indices, ids)]
        valid_dist = [[top_dist[x] for x in indices] for indices, top_dist in zip(valid_indices, dist)]

        if get_thumbs_fn is None:
            get_thumbs_fn = lambda x: [None for _ in range(len(x))]

        return {
            _q: [
                SearchResponse(
                    thumbnail=convert_uint8array_to_base64(_thumb) if _thumb is not None else "",
                    link=f"{_metadata.source_uri if _metadata.source_uri else f'images/{_metadata.id}'}",
                    distance=_dist,
                    info=ImageInfo(
                        id=str(_metadata.id),
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
            for _q, top_dist, top_ids in zip(queries, valid_dist, valid_ids)
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
    index = read_index(index_filename, readonly=True)
    if hasattr(index, "nprobe"):
        # See https://github.com/facebookresearch/faiss/blob/43d86e30736ede853c384b24667fc3ab897d6ba9/faiss/IndexIVF.h#L184C8-L184C42
        index.parallel_mode = 1
        index.nprobe = getattr(config, "nprobe", 32)

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

    def load_internal_images(image_ids: List[int]) -> List[Union[ndarray, bytes]]:
        if len(image_ids) == 0:
            return []
        internal_images_loaded = [] # a list of ndarrays (feature vectors) or bytes (from image file)
        with project_engine.connect() as conn:
            for image_id in image_ids:
                # Try to read feature vector from h5 dataset
                try:
                    with get_h5reader(vds_path)(H5Datasets.IMAGE_FEATURES) as image_features_reader:
                        image_features = image_features_reader([image_id])[0] # This is an np.ndarray of shape: (output_dim,) e.g. (768,)
                        image_features = expand_dims(image_features, axis=0) # Add batch dimension so the shape becomes (1, output_dim) e.g. (1, 768)
                        internal_images_loaded.append(image_features)
                        continue
                except Exception:
                    logger.info(f"Could not retrieve feature vector for image {image_id} from h5 dataset. Attempting to re-compute features from original image")
                    pass
                
                metadata = MetadataRepo.get(conn, image_id)
                if metadata is None:
                    raise FileNotFoundError(f"Image {image_id} not found in metadata database")

                # Look up the dataset table and find the location and type
                dataset = DatasetRepo.get(conn, metadata.dataset_id)
                if dataset is None:
                    raise LookupError(f"Dataset not found for image {image_id}")

                location = Path(dataset.location)

                if dataset.type == DatasetType.IMAGE_DIR:
                    # Try to read image from disk if present
                    # metadata.source_uri will be None, so we have to search for it on disk
                    file_path = location / metadata.path
                    if file_path.is_file():
                        with open(file_path, 'rb') as f:
                            internal_images_loaded.append(f.read())
                    else:
                        raise FileNotFoundError(f"Image file for image {image_id} does not exist or is not a regular file")
                else:
                    # Try to extract from local file if present
                    if not location.is_file() or not tarfile.is_tarfile(location):
                        raise FileNotFoundError(f"WebDataset tar file (for image {image_id}) does not exist or is not a tar file")
                    try:
                        with tarfile.open(location, "r") as t:
                            buf = t.extractfile(metadata.path.lstrip("#"))
                            internal_images_loaded.append(buf.read())
                    except Exception as e:
                        logger.exception(f"Exception when reading image {image_id}")
                        raise FileNotFoundError(f"Error extracting image {image_id} from WebDataset tar file")
        return internal_images_loaded

    # Create a random array of featured images
    with project_engine.connect() as conn:
        # Get all image ids from the metadata table
        ids = array([row['id'] for row in MetadataRepo.get_columns(conn, ('id',))])

        # Select a random subset of up to 10000 image ids (for performance reasons)
        default_rng(seed=42).shuffle(ids)
        ids = ids[:10000]

    @router.get('/featured', response_model=Dict[str, List[SearchResponse]])
    async def handle_get_featured(
        start: int = Query(0, ge=0, le=980),
        end: int = Query(20, gt=0, le=1000),
        thumbs: bool = Query(True),
        random_seed: int = Query(123) # This seed is used to randomly select the set of images used for the featured images
    ):
        with project_engine.connect() as conn:
            # Select up to 1000 random image ids, using the specified random seed, from the set of 10000 ids
            selected_ids = ids.copy()
            default_rng(seed=random_seed).shuffle(selected_ids)
            selected_ids = selected_ids[:1000]
            selected_ids = expand_dims(selected_ids, axis=0)

            dist = zeros(selected_ids.shape) # Use 0 as a filler value for the distance array since this is not relevant for the featured images

            def get_metadata(_id):
                m = MetadataRepo.get(conn, int(_id))
                if m is None:
                    raise RuntimeError()
                return m

            if not thumbs:
                response = make_basic_response(
                    ['featured'], dist[[0], start:end], selected_ids[[0], start:end], get_metadata
                )
            else:
                response = make_full_response(
                    ['featured'],
                    dist[[0], start:end],
                    selected_ids[[0], start:end],
                    get_metadata,
                    thumbs_reader,
                )
        return response


    @router.get("/search", response_model=Dict[str, List[SearchResponse]])
    async def handle_get_search(
        q: List[str] = Query(default=[]),
        start: int = Query(0, ge=0, le=980),
        end: int = Query(20, gt=0, le=1000),
        thumbs: bool = Query(True),
    ):
        if len(q) == 0:
            raise HTTPException(400, {"message": "Missing search query"})

        end = min(end, num_files)
        if start > end:
            raise HTTPException(
                400, {"message": "'start' cannot be greater than 'end'"}
            )
        if (end - start) > 50 and thumbs == True:
            raise HTTPException(
                400,
                {
                    "message": "Cannot return more than 50 results at a time when thumbs=1"
                },
            )

        if len(q) == 1 and q[0].startswith(("http://", "https://")):
            query = q[0]
            logger.info("Downloading", query, "to file")
            with NamedTemporaryFile() as tmpfile:
                download_url_to_file(query, tmpfile.name)
                with Image.open(tmpfile.name) as im:
                    query_features = extract_image_features([im])

            return similarity_search(q=["image"], features=query_features, start=start, end=end, thumbs=thumbs)
        else:
            for query in q:
                if query.strip() in config.query_blocklist:
                    message = "One of the search terms you entered has been blocked" if len(q) > 1 else "The search term you entered has been blocked"
                    raise HTTPException(
                        403, {"message": message}
                    )
            prefixed_queries = [f"{_prefix} {x.strip()}".strip() for x in q]
            text_features = extract_text_features(prefixed_queries)
            return similarity_search(q=q, features=text_features, start=start, end=end, thumbs=thumbs)

    @router.post("/search", response_model=Dict[str, List[SearchResponse]])
    async def handle_post_search_multimodal(
        # Positive queries
        text_queries: List[str] = Query(default=[]),
        file_queries: List[bytes] = File([]), # user-uploaded images
        url_queries: List[str] = Form([]), # URLs to online images
        internal_image_queries: List[int] = Query(default=[]), # ids to internal images

        # Negative queries
        negative_text_queries: List[str] = Query(default=[]),
        negative_file_queries: List[bytes] = File([]), # user-uploaded images
        negative_url_queries: List[str] = Form([]), # URLs to online images
        negative_internal_image_queries: List[int] = Query(default=[]), # ids to internal images

        # Other parameters
        start: int = Query(0, ge=0, le=980),
        end: int = Query(20, gt=0, le=1000),
        thumbs: bool = Query(True),
    ):
        """
        Handles queries sent by POST request. This endpoint can handle file queries, URL queries (i.e. URL to an image), and/or text queries.
        Multimodal queries (i.e. images + text) are performed by computing a weighted sum of the feature vectors of the
        input images/text, and then using this as the query vector.
        """
        try:
            internal_image_queries = load_internal_images(internal_image_queries)
            negative_internal_image_queries = load_internal_images(negative_internal_image_queries)
        except Exception as e:
            logger.exception(e)
            return PlainTextResponse(
                status_code=500, content=f"Error processing internal image queries"
            )

        q = file_queries + url_queries + text_queries + internal_image_queries
        q = [dict(type='positive', val=query) for query in q]

        negative_q = negative_file_queries + negative_url_queries + negative_text_queries + negative_internal_image_queries
        q = q + [dict(type='negative', val=query) for query in negative_q]

        if len(q) == 0:
            raise HTTPException(400, {"message": "Missing search query"})
        elif len(q) > 5:
            raise HTTPException(400, {"message": "Too many query items"})

        end = min(end, num_files)
        if start > end:
            raise HTTPException(
                400, {"message": "'start' cannot be greater than 'end'"}
            )
        if (end - start) > 50 and thumbs == True:
            raise HTTPException(
                400, {"message": "Cannot return more than 50 results at a time when thumbs=1"}
            )

        feature_vectors = []
        weights = []
        for query_dict in q:
            query = query_dict['val']
            feature_vector = None
            if isinstance(query, bytes):
                with Image.open(io.BytesIO(query)) as im:
                    feature_vector = extract_image_features([im])
                    weights.append(config.negative_queries_weight if query_dict['type'] == 'negative' else 1)
            elif isinstance(query, ndarray):
                feature_vector = query
                weights.append(config.negative_queries_weight if query_dict['type'] == 'negative' else 1)
            elif query.startswith(("http://", "https://")):
                logger.info("Downloading", query, "to file")
                with NamedTemporaryFile() as tmpfile:
                    download_url_to_file(query, tmpfile.name)
                    with Image.open(tmpfile.name) as im:
                        feature_vector = extract_image_features([im])
                        weights.append(config.negative_queries_weight if query_dict['type'] == 'negative' else 1)
            else:
                if query.strip() in config.query_blocklist:
                    message = "One of the search terms you entered has been blocked" if len(q) > 1 else "The search term you entered has been blocked"
                    raise HTTPException(
                        403, {"message": message}
                    )
                prefixed_queries = f"{_prefix} {query.strip()}".strip()
                feature_vector = extract_text_features(prefixed_queries)
                weights.append(
                    config.text_queries_weight * # assign higher weight to natural language queries
                    (config.negative_queries_weight if query_dict['type'] == 'negative' else 1)
                )
            if query_dict['type'] == 'negative':
                feature_vector = -feature_vector
            feature_vectors.append(feature_vector)
        weights = array(weights, dtype=float32)
        average_features = average(feature_vectors, axis=0, weights=weights)
        average_features /= norm(average_features, axis=-1, keepdims=True)
        return similarity_search(q=["multimodal"], features=average_features, start=start, end=end, thumbs=thumbs)

    def similarity_search(q: List[str], features: ndarray, start: int, end: int, thumbs: bool):
        dist, ids = index.search(features, end)
        with project_engine.connect() as conn:
            def get_metadata(_id):
                m = MetadataRepo.get(conn, int(_id))
                if m is None:
                    raise RuntimeError()
                return m


            if not thumbs:
                response = make_basic_response(
                    q, dist[[0], start:end], ids[[0], start:end], get_metadata
                )
            else:
                response = make_full_response(
                    q,
                    dist[[0], start:end],
                    ids[[0], start:end],
                    get_metadata,
                    thumbs_reader,
                )
        return response

    return router
