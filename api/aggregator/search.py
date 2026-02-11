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
from typing import Annotated, Literal, cast


from .. import common
from ..services.embedding import EmbeddingConfig
from ..dependencies import (
    APIConfig,
    ConfigDep, 
    EmbeddingService,
    EmbeddingServiceDep,
    RemoteSearchService,
    SearchServiceDep
)

from src.data_models import MediaType, ModalityType

from fastapi import APIRouter, Query, Form, HTTPException, Request, UploadFile, File
from pydantic import HttpUrl



logger = logging.getLogger(__name__)


def get_prefix(config):
    return {
        MediaType.IMAGE: config.query_prefix.strip(),
        MediaType.VIDEO: config.query_prefix.strip(),
        MediaType.AV: "This is the sound of", # TODO add this to config
        MediaType.AUDIO: "This is the sound of",
    }


router = APIRouter(route_class=common.CachedBodyRoute)

@router.get("/featured", response_model=common.SearchResponse)
@common.add_response_time
async def handle_get_featured(
    search_service: SearchServiceDep,
    # The "media type" used for "featured_in" does not actually
    # refer to the media_type in the database.  It is actually
    # closer, but not the same, to the frontend viewModality but
    # we use MediaType because it uses a subset of its keys.  This
    # is just a convenience to get the values checked.
    featured_in: MediaType = Query(),
    feature_extractor_id: str = Query(),
    start: int = Query(0, ge=0, le=980),
    end: int = Query(20, gt=0, le=1000),
    thumbnails_to_send: int = Query(0),
    # This seed is used to randomly select the set of images used for the featured images
    random_seed: int = Query(123),
):
    # modality = ModalityType.AUDIO if featured_in == MediaType.AV else ModalityType(featured_in) 
    response = await cast(RemoteSearchService, search_service).featured(
        featured_in, feature_extractor_id, start, end, random_seed
    )
    return response


async def replace_vector_ids_with_search_embeddings(
    search_service,
    embedding_service,
    media_type,
    feature_extractor_id,
    q: list[common.InternalQTerm],
) -> list[common.InternalQTerm]:
    ## Pick up queries for internal vectors
    q_idx = []
    vector_ids = []
    for i, x in enumerate(q):
        if x["modality"] != "text" and isinstance(x["val"], str):
            q_idx.append(i)
            vector_ids.append(x)
    if not vector_ids:
        return q

    ## Reconstruct features from faiss index
    embeddings = await search_service.reconstruct_vectors(
        media_type, feature_extractor_id, vector_ids
    )
    ## Apply hook to transform internal image query vectors
    search_embeddings = [
        embedding_service.transform_internal_image_queries(feature_extractor_id, x)
        for x in embeddings
    ]

    new_q = q.copy()
    for idx, embedding in zip(q_idx, search_embeddings):
        new_q[idx] = q[idx] | {"val": embedding}
    return new_q

async def _search(
    config: APIConfig,
    embedding_service: EmbeddingService,
    search_service: RemoteSearchService,
    # Which media type to search on
    # "video" refers to the visual stream of videos, "av" refers to the audio stream of videos
    # "audio" refers to pure audio files, and "image" refers to images
    request: Request,
    search_in: MediaType,
    feature_extractor_id: str,
    # Query
    q: list[common.InternalQTerm],
    # Other parameters
    start: int,
    end: int,
    thumbnails_to_send: int,
    shot_scale: list[int],
    metadata_filter: list[str],
    add_prefix: bool,
    search_endpoint: Literal["/search", "/search2"] = "/search"
):
    if search_in == MediaType.IMAGE:
        if len([query for query in q if query['modality'] == 'audio']) > 0:
            raise HTTPException(400, {
                "message": "Cannot search on images using an audio query"
            })
    elif search_in == MediaType.VIDEO:
        if len([query for query in q if query['modality'] == 'audio']) > 0:
            raise HTTPException(400, {
                "message": "Cannot search on visual stream of video files using an audio query"
            })
    elif search_in == MediaType.AUDIO or search_in == MediaType.AV:
        if len([query for query in q if query['modality'] == 'image']) > 0:
            raise HTTPException(400, {
                "message": "Cannot search on audio using an image query"
            })

    if feature_extractor_id == 'wise/metadata':
        response = await search_service.search(request, endpoint=search_endpoint)
        return response

    media_type = MediaType.AUDIO if search_in == MediaType.AV else search_in

    q = await replace_vector_ids_with_search_embeddings(
        search_service, embedding_service, media_type, feature_extractor_id, q
    )

    prefix = get_prefix(config)[search_in] if add_prefix else ""
    embedding_config = EmbeddingConfig(
        query_prefix=prefix,
        text_queries_weight=config.text_queries_weight,
        negative_queries_weight=config.negative_queries_weight,
    )
    features = embedding_service.embed(feature_extractor_id, embedding_config, q)
    search_response = await search_service.search_with_feature(
        features,
        search_in=search_in,
        feature_extractor_id=feature_extractor_id,
        start=start,
        end=end,
        thumbnails_to_send=thumbnails_to_send,
        shot_scale=shot_scale,
        metadata_filter=metadata_filter,
    )

    return search_response

@router.post("/search", response_model=common.SearchResponse)
@common.add_response_time
async def handle_post_search(
    config: ConfigDep,
    embedding_service: EmbeddingServiceDep,
    search_service: SearchServiceDep,
    # Which media type to search on
    # "video" refers to the visual stream of videos, "av" refers to the audio stream of videos
    # "audio" refers to pure audio files, and "image" refers to images
    request: Request,
    search_in: MediaType = Query(),
    feature_extractor_id: str = Query(),
    # Positive queries
    text_queries: list[str] = Query(default=[]),
    image_file_queries: list[bytes] = File([]),  # user-uploaded images
    audio_file_queries: list[bytes] = File([]),  # user-uploaded audio files
    image_url_queries: list[HttpUrl] = Form([]),  # URLs to online images
    audio_url_queries: list[HttpUrl] = Form([]),  # URLs to online audio files
    internal_image_queries: list[str] = Query(default=[]),  # ids to internal images
    # Negative queries
    negative_text_queries: list[str] = Query(default=[]),
    negative_image_file_queries: list[bytes] = File([]),  # user-uploaded images
    negative_audio_file_queries: list[bytes] = File(
        []
    ),  # user-uploaded audio files
    negative_image_url_queries: list[HttpUrl] = Form([]),  # URLs to online images
    negative_audio_url_queries: list[HttpUrl] = Form([]),  # URLs to online audio files
    negative_internal_image_queries: list[str] = Query(
        default=[]
    ),  # ids to internal images
    # Other parameters
    start: int = Query(0, ge=0, le=980),
    end: int = Query(20, gt=0, le=1000),
    thumbnails_to_send: int = Query(0),
    shot_scale: list[int] = Query(default=[]),
    metadata_filter: list[str] = Query(default=[]),
    add_prefix: bool = Query(True)
):
    """
    Handles queries sent by POST request. This endpoint can handle file queries, URL queries (i.e. URL to an image), and/or text queries.
    Multimodal queries (i.e. images + text) are performed by computing a weighted sum of the feature vectors of the
    input images/text, and then using this as the query vector.
    """
    q = common.api_query_to_internal_q_old(
        text_queries,
        image_file_queries,
        audio_file_queries,
        image_url_queries,
        audio_url_queries,
        internal_image_queries,
        negative_text_queries,
        negative_image_file_queries,
        negative_audio_file_queries,
        negative_image_url_queries,
        negative_audio_url_queries,
        negative_internal_image_queries,
    )

    if len(q) == 0:
        raise HTTPException(400, {"message": "Missing search query"})
    elif len(q) > 5:
        raise HTTPException(400, {"message": "Too many query items"})
    
    response = await _search(
        config,
        embedding_service,
        cast(RemoteSearchService, search_service),
        request,
        search_in,
        feature_extractor_id,
        q,
        start,
        end,
        thumbnails_to_send,
        shot_scale,
        metadata_filter,
        add_prefix
    )
    return response

@router.post("/search2", response_model=common.SearchResponse)
@common.add_response_time
async def handle_post_search2(
    config: ConfigDep,
    embedding_service: EmbeddingServiceDep,
    search_service: SearchServiceDep,
    # Which media type to search on
    # "video" refers to the visual stream of videos, "av" refers to the audio stream of videos
    # "audio" refers to pure audio files, and "image" refers to images
    request: Request,
    search_in: MediaType = Query(),
    feature_extractor_id: str = Query(),
    # Query
    query_term: Annotated[list[str], Form()] = [],
    query_file: list[UploadFile] = [],
    # Other parameters
    start: int = Query(0, ge=0, le=980),
    end: int = Query(20, gt=0, le=1000),
    thumbnails_to_send: int = Query(0),
    shot_scale: list[int] = Query(default=[]),
    metadata_filter: list[str] = Query(default=[]),
    add_prefix: bool = Query(True)
):
    """
    Handles queries sent by POST request. This endpoint can handle file queries, URL queries (i.e. URL to an image), and/or text queries.
    Multimodal queries (i.e. images + text) are performed by computing a weighted sum of the feature vectors of the
    input images/text, and then using this as the query vector.
    """
    if len(query_term) == 0:
        raise HTTPException(400, {"message": "Missing search query"})
    elif len(query_term) > 5:
        raise HTTPException(400, {"message": "Too many query items"})

    q = common.api_query_to_internal_q(query_term, query_file)

    response = await _search(
        config,
        embedding_service,
        cast(RemoteSearchService, search_service),
        request,
        search_in,
        feature_extractor_id,
        q,
        start,
        end,
        thumbnails_to_send,
        shot_scale,
        metadata_filter,
        add_prefix,
        search_endpoint="/search2"
    )
    return response
