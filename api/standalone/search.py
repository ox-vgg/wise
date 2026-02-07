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

import itertools
import json
import logging
import functools
from collections.abc import Callable, Iterable
from config import APIConfig
from .. import common
from ..common import VideoSegment
from ..services.embedding import EmbeddingConfig

from src.data_models import MediaType, ModalityType, VectorAndMediaMetadata
from src.search.fts import WISEFTSQuery

from src.feature.feature_extractor import FeatureExtMetadata
from src.wise_project import WiseProject

import numpy as np
from fastapi import APIRouter, Query, File, Form, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import HttpUrl

from ..dependencies import ConfigDep, ProjectServiceDep, ProjectInfoDep, EmbeddingServiceDep, SearchServiceDep

logger = logging.getLogger(__name__)

def merge_close_segments(_keyframes: list[VideoSegment]):
    """
    Takes a list of segments of a media file and merges them if they are close - within 4 seconds of each other
    The merged segment is represented by the best matching segment based on distance
    """
    merged_segments: list[VideoSegment] = []
    start = None
    current = None
    best = None
    for k in _keyframes:
        if start is None:
            # Start a new group
            start = k
            current = k
            best = k

        elif (k.ts - current.te) <= 4:
            current = k
            if current.distance > best.distance:
                best = current

        else:
            merged_segments.append(
                common.VideoSegment(
                    vector_id=best.vector_id,
                    media_id=best.media_id,
                    ts=start.ts,
                    te=current.te,
                    link=f"media/{best.media_id}#t={start.ts},{current.te}",
                    distance=best.distance,
                    thumbnail=best.thumbnail,
                    thumbnail_ts=best.thumbnail_ts,
                    bbox=best.bbox,
                )
            )
            start = k
            current = k
            best = k

    if start is not None:
        merged_segments.append(
            common.VideoSegment(
                vector_id=best.vector_id,
                media_id=best.media_id,
                ts=start.ts,
                te=current.te,
                link=f"media/{best.media_id}#t={start.ts},{current.te}",
                distance=best.distance,
                thumbnail=best.thumbnail,
                thumbnail_ts=best.thumbnail_ts,
                bbox=best.bbox,
            )
        )

    return merged_segments

def get_shots_from_segments(
        segments: list[VideoSegment],
        merge_function: Callable[[list[VideoSegment]], list[VideoSegment]] = merge_close_segments
    ):
    """
    Functions that takes a list of segments and returns a list of merged segments
    based on the merge function passed in

    The merge function by default merges close segments
    """
    # Sort by video_id, timestamp
    sorted_segments = sorted(segments, key=lambda x: (x.media_id, x.ts))

    # for each key, apply merge logic
    all_merged_segments = []
    for _, g in itertools.groupby(sorted_segments, key=lambda x: x.media_id):
        merged_segments = merge_function(list(g))
        all_merged_segments.extend(merged_segments)

    # sort the merged segments by distance
    all_merged_segments = sorted(
        all_merged_segments,
        key=lambda x: x.distance,
        reverse=True,
    )
    return all_merged_segments

def keyframes_to_shots(_keyframes: list[VideoSegment], project: WiseProject):
    """
    Get Shot corresponding to a keyframe
    """
   
    # Assuming segment maps to only one shot (no duplicates)
    return list(
        next(
            project.shot(media_id=int(x.media_id), timestamp=x.ts),
            None,
        )
        for x in _keyframes
    )

def get_shots_from_keyframes(project: WiseProject, _keyframes: list[VideoSegment]):
    """
    Get unique shots from list of keyframes belonging to a single video
    """
    # Input are keyframes from same video
    shots = keyframes_to_shots(_keyframes, project)

    # Ignore where segment doesn't have a shot mapping
    shots_iter = filter(lambda x: x[0], zip(shots, _keyframes))

    shots_list = []
    for _, g in itertools.groupby(shots_iter, key=lambda x: x[0].id):
        shot_group, keyframes_group = list(zip(*g))
        _shot = shot_group[0]

        # Computing best matching segment and best thumbnail to represent the segment with
        # For clip case, it will be same. For internvideo it will be different
        best_segment = sorted(
            keyframes_group, key=lambda x: x.distance, reverse=True
        )[0]
        shots_list.append(
            common.VideoSegment(
                vector_id=best_segment.vector_id,
                media_id=best_segment.media_id,
                ts=_shot.ts,
                te=_shot.te,
                link=f"media/{best_segment.media_id}#t={_shot.ts},{_shot.te}",
                distance=best_segment.distance,
                thumbnail=best_segment.thumbnail,
                thumbnail_ts=best_segment.thumbnail_ts,
                bbox=best_segment.bbox,
            )
        )
    return shots_list

def construct_video_search_response(
    search_in: MediaType,
    top_dist: list[float],
    all_metadata: list[VectorAndMediaMetadata],
    all_ext_metadata: list[FeatureExtMetadata],
    all_thumbs: Iterable[str],
    merge_function: Callable[[list[VideoSegment]], list[VideoSegment]],
):
    videos = {}
    shots = []
    segments = []
    for _dist, _metadata, _ext_metadata, _thumb in zip(
        top_dist,
        all_metadata,
        all_ext_metadata,
        all_thumbs,
    ):
        video_id = str(_metadata.media_id)
        if video_id not in videos:
            videos[video_id] = common.VideoInfo(
                id=video_id,
                filename=_metadata.path,
                width=_metadata.width,
                height=_metadata.height,
                media_type=_metadata.media_type,
                format=_metadata.format,
                duration=_metadata.duration,
                timeline_hover_thumbnails=f"storyboard/{video_id}.vtt",
                external_metadata=_metadata.external_metadata,
            )
        ts = _metadata.timestamp
        te = _metadata.end_timestamp
        if ts is None:
            logger.error(f"ts is None for vector {_metadata.id}")
        if te is None:
            te = ts

        if ts == te:
            te = ts + 4.0

        segment = common.VideoSegment(
            vector_id=str(_metadata.id),
            media_id=video_id,
            ts=float(ts),
            te=float(te),
            link=f"media/{video_id}#t={ts},{te}", # f"{_metadata.source_uri if _metadata.source_uri else f'media/{video_id}{_metadata.path}'}",
            distance=_dist,
            thumbnail=_thumb,
            thumbnail_ts=float(ts),
            bbox=_ext_metadata.bbox,
        )

        segments.append(segment)

    shots = get_shots_from_segments(segments, merge_function=merge_function)

    if search_in == MediaType.VIDEO:
        return common.VideoResults(
            total=300, # TODO change this
            unmerged_windows=segments,
            merged_windows=shots,
            videos=videos,
        )
    elif search_in == MediaType.AV:
        return common.VideoAudioResults(
            total=300, # TODO change this
            unmerged_windows=segments,
            merged_windows=shots,
            videos=videos,
        )
    else:
        raise ValueError("`search_in` must be either `MediaType.VIDEO` or `MediaType.AV`")

def construct_image_search_response(
    top_dist: list[float],
    all_metadata: list[VectorAndMediaMetadata],
    all_ext_metadata: list[FeatureExtMetadata],
    all_thumbs: Iterable[str],
):
    images = {}
    image_vectors = []
    for _dist, _metadata, _ext_metadata, _thumb in zip(
        top_dist,
        all_metadata,
        all_ext_metadata,
        all_thumbs,
    ):
        image_id = str(_metadata.media_id)
        if image_id not in images:
            images[image_id] = common.ImageInfo(
                id=image_id,
                filename=_metadata.path,
                width=_metadata.width,
                height=_metadata.height,
                media_type=_metadata.media_type,
                format=_metadata.format,
                duration=_metadata.duration,
                external_metadata=_metadata.external_metadata,
            )

        image_vector = common.ImageVector(
            vector_id=str(_metadata.id),
            media_id=image_id,
            link=f"media/{image_id}",
            distance=_dist,
            thumbnail=_thumb,
            bbox=_ext_metadata.bbox,
        )
        image_vectors.append(image_vector)

    return common.ImageResults(
        total=300, # TODO change this
        vectors=image_vectors,
        images=images,
    )

def construct_search_response(
    top_dist: list[float],
    all_metadata: list[VectorAndMediaMetadata],
    all_ext_metadata: list[FeatureExtMetadata],
    all_thumbs: Iterable[str],
    merge_function: Callable[[list[VideoSegment]], list[VideoSegment]] = merge_close_segments,
    search_in: MediaType = None,
):

    video_audio_results = None
    video_results = None
    image_results = None
    if search_in is None or search_in == MediaType.IMAGE:
        image_indices = [i for i, x in enumerate(all_metadata) if x.modality == ModalityType.IMAGE]
        if len(image_indices) > 0:
            image_top_dist = [top_dist[i] for i in image_indices]
            image_all_metadata = [all_metadata[i] for i in image_indices]
            image_ext_metadata = [all_ext_metadata[i] for i in image_indices]
            image_thumbs = [all_thumbs[i] for i in image_indices]
            image_results = construct_image_search_response(image_top_dist, image_all_metadata, image_ext_metadata, image_thumbs)
    if search_in is None or search_in == MediaType.VIDEO:
        video_indices = [i for i, x in enumerate(all_metadata) if x.modality == ModalityType.VIDEO]
        if len(video_indices) > 0:
            video_top_dist = [top_dist[i] for i in video_indices]
            video_all_metadata = [all_metadata[i] for i in video_indices]
            video_ext_metadata = [all_ext_metadata[i] for i in video_indices]
            video_thumbs = [all_thumbs[i] for i in video_indices]
            video_results = construct_video_search_response(MediaType.VIDEO, video_top_dist, video_all_metadata, video_ext_metadata, video_thumbs, merge_function)
    if search_in is None or search_in == MediaType.AV:
        av_indices = [i for i, x in enumerate(all_metadata) if x.modality == ModalityType.AUDIO and x.media_type == MediaType.AV]
        if len(av_indices) > 0:
            av_top_dist = [top_dist[i] for i in av_indices]
            av_all_metadata = [all_metadata[i] for i in av_indices]
            av_ext_metadata = [all_ext_metadata[i] for i in av_indices]
            av_thumbs = [all_thumbs[i] for i in av_indices]
            video_audio_results = construct_video_search_response(MediaType.AV, av_top_dist, av_all_metadata, av_ext_metadata, av_thumbs, merge_function)
    if search_in is not None and search_in not in [MediaType.IMAGE, MediaType.VIDEO, MediaType.AV]:
        raise NotImplementedError("`search_in` must be either `MediaType.IMAGE`, `MediaType.VIDEO`, or `MediaType.AV`. Support for `MediaType.AUDIO` is not available yet")

    return common.SearchResponse(
        time=0.0, # Dummy value to be overwritten by the @add_response_time decorator function
        video_audio_results=video_audio_results,
        video_results=video_results,
        image_results=image_results,
    )

def get_prefix(config: APIConfig):
    return {
        MediaType.IMAGE: config.query_prefix.strip(),
        MediaType.VIDEO: config.query_prefix.strip(),
        MediaType.AV: "This is the sound of", # TODO add this to config
        MediaType.AUDIO: "This is the sound of",
    }
    

router = APIRouter()
@router.get(
    "/vectors",
    response_model=common.NPArray,
    responses={200: {"content": "application/json"}, 500: {"content": "text/plain"}},
)
def reconstruct_vectors(
    config: ConfigDep,
    project_info: ProjectInfoDep,
    search_service: SearchServiceDep,
    search_in: MediaType = Query(),
    feature_extractor_id: str = Query(),
    internal_ids: list[int] = Query(default=[]),  # ids to internal images
):
    media_type = MediaType.AUDIO if search_in == MediaType.AV else search_in
    search_targets = project_info.search_targets

    if media_type not in search_targets:
        raise HTTPException(400, {
            "message": f"No search index exists for this modality: {media_type}"
        })
    
    if not internal_ids:
        vectors = np.array([])
        response = common.NPArray.from_array(vectors)
        return response
    
    if search_service.is_internal_search_supported(media_type, feature_extractor_id):
        try:
            # reconstruct features from faiss index
            vectors = search_service.reconstruct_vectors(media_type, feature_extractor_id, internal_ids)
        except Exception as e:
            logger.exception(e)
            return PlainTextResponse(
                status_code=500, content=f"Error processing internal search query"
            )
    else:
        index_type = search_service.get_search_index_type(media_type, feature_extractor_id)
        logger.exception(
            "This faiss index does not support internal search. To enable "
            "internal search, please re-create the index by running "
            f"`python create-index.py --project-dir \"{config.project_dir}\" --media-type {media_type} --index-type {index_type} --overwrite`",
        )
        return PlainTextResponse(
            status_code=500, content=f"Internal search not supported in this project"
        )
    
    vectors = np.concatenate(vectors, axis=0)
    response = common.NPArray.from_array(vectors)
    return response


def build_filter_specs(shot_scale: list[int], metadata_filter: list[str]):
    filter_specs = {}
    if len(shot_scale) > 0:
        filter_specs["shot_scale_query"] = {"$in": shot_scale}
    if metadata_filter:
        filter_specs["metadata_query"] = WISEFTSQuery.model_validate(
            {"$match": " ".join(metadata_filter)}
        )
    return filter_specs


@router.post("/search_with_feature", response_model=common.SearchResponse)
@common.add_response_time
async def handle_post_search_feature(
    config: ConfigDep,
    project_info: ProjectInfoDep,
    project_service: ProjectServiceDep,
    search_service: SearchServiceDep,
    feature: common.NPArray,
    # Which media type to search on
    # "video" refers to the visual stream of videos, "av" refers to the audio stream of videos
    # "audio" refers to pure audio files, and "image" refers to images
    search_in: MediaType = Query(),
    feature_extractor_id: str = Query(),
    # Other parameters
    start: int = Query(0, ge=0, le=980),
    end: int = Query(20, gt=0, le=1000),
    thumbnails_to_send: int = Query(0),
    shot_scale: list[int] = Query(default=[]),
    metadata_filter: list[str] = Query(default=[]),
):
    media_type = MediaType.AUDIO if search_in == MediaType.AV else search_in
    search_targets = project_info.search_targets
    if media_type not in search_targets:
        raise HTTPException(400, {
            "message": f"No search index exists for this modality: {media_type}"
        })

    if feature_extractor_id == 'wise/metadata':
        raise HTTPException(400, {
            "message": "`wise/metadata` feature extractor cannot be used for feature-based search. Please use a different feature extractor."
        })

        end = min(end, project_info.num_vectors)
    if start > end:
        raise HTTPException(
            400, {"message": "'start' cannot be greater than 'end'"}
        )

    filter_specs = build_filter_specs(shot_scale, metadata_filter)

    vectors = feature.to_array()
    search_output = search_service.search_with_feature(
        vectors,
        media_type=media_type,
        feature_extractor_id=feature_extractor_id,
        start=start,
        end=end,
        filter_specs=filter_specs
    )

    if len(search_output.ids) == 0:
        return common.SearchResponse(
            time=0.0,
            video_audio_results=None,
            video_results=None,
            image_results=None,
        )

    all_thumbs = project_service.get_thumbnail_reader(thumbnails_to_send)(search_output.metadata)
    _get_shots_from_keyframes = functools.partial(get_shots_from_keyframes, project_service.wise_project)
    # supports shots
    is_shot_merge_supported = config.use_shots and search_in == MediaType.VIDEO
    response = construct_search_response(
        top_dist=search_output.distances,
        all_metadata=search_output.metadata,
        all_ext_metadata=search_output.ext_metadata,
        all_thumbs=all_thumbs,
        merge_function=_get_shots_from_keyframes if is_shot_merge_supported else merge_close_segments,
        search_in=search_in,
    )

    return response


def get_search_embeddings_for_internal_queries(
    search_service,
    embedding_service,
    media_type,
    feature_extractor_id,
    internal_queries: list[str],
) -> list[np.ndarray]:
    if not internal_queries:
        return internal_queries

    def handle_internal_id(_id: str):
        *_, vector_id = _id.rsplit("/", 1)
        return int(vector_id)

    internal_queries = list(map(handle_internal_id, internal_queries))

    # reconstruct features from faiss index
    internal_queries = search_service.reconstruct_vectors(
        media_type, feature_extractor_id, internal_queries
    )
    # Apply hook to transform internal image query vectors
    internal_queries = [
        embedding_service.transform_internal_image_queries(feature_extractor_id, x)
        for x in internal_queries
    ]
    return internal_queries


@router.post("/search", response_model=common.SearchResponse)
@common.add_response_time
async def handle_post_search_multimodal(
    config: ConfigDep,
    project_info: ProjectInfoDep,
    project_service: ProjectServiceDep,
    embedding_service: EmbeddingServiceDep,
    search_service: SearchServiceDep,
    # Which media type to search on
    # "video" refers to the visual stream of videos, "av" refers to the audio stream of videos
    # "audio" refers to pure audio files, and "image" refers to images
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
    media_type = MediaType.AUDIO if search_in == MediaType.AV else search_in
    search_targets = project_info.search_targets
    if media_type not in search_targets:
        raise HTTPException(400, {
            "message": f"No search index exists for this modality: {media_type}"
        })

    q = [dict(sign="positive", modality="text", val=query) for query in text_queries]
    
    if len(q) > 5:
        raise HTTPException(400, {"message": "Too many query items"})

    if feature_extractor_id == 'wise/metadata':
        if len(q) == 0:
            raise HTTPException(400, {"message": "Missing search query"})

        if (image_file_queries
            or audio_file_queries
            or image_url_queries
            or audio_url_queries
            or internal_image_queries
            or negative_text_queries
            or negative_image_file_queries
            or negative_audio_file_queries
            or negative_image_url_queries
            or negative_audio_url_queries
            or negative_internal_image_queries):
            raise HTTPException(400, {
                "message": "`wise/metadata` feature extractor can only be used with `text_queries`."
            })

        # ASR search
        if start > end:
            raise HTTPException(
                400, {"message": "'start' cannot be greater than 'end'"}
            )
        
        # TODO escape special characters
        text = " ".join(text_queries)
        q = WISEFTSQuery.model_validate({"$match": text})
        search_output = search_service.asr_search(q, media_type, start, end)
        if len(search_output.ids) == 0:
            return common.SearchResponse(
                time=0.0,
                video_audio_results=None,
                video_results=None,
                image_results=None,
            )
        all_thumbs = project_service.get_thumbnail_reader(thumbnails_to_send)(search_output.metadata)
        response = construct_search_response(
            search_output.distances,
            search_output.metadata,
            search_output.ext_metadata,
            all_thumbs,
            search_in=search_in
        )

        return response

    if (
        (internal_image_queries or negative_internal_image_queries)
        and not search_service.is_internal_search_supported(media_type, feature_extractor_id)
    ):
        index_type = search_service.get_search_index_type(media_type, feature_extractor_id)
        logger.exception(
            "This faiss index does not support internal search. To enable "
            "internal search, please re-create the index by running "
            f"`python create-index.py --project-dir \"{config.project_dir}\" "
            f"--media-type {media_type} --index-type {index_type} --overwrite`"
        )
        return PlainTextResponse(
            status_code=500,
            content=f"Internal search not supported in this project"
        )

    try:
        internal_image_queries = get_search_embeddings_for_internal_queries(
            search_service,
            embedding_service,
            media_type,
            feature_extractor_id,
            internal_image_queries,
        )
        negative_internal_image_queries = get_search_embeddings_for_internal_queries(
            search_service,
            embedding_service,
            media_type,
            feature_extractor_id,
            negative_internal_image_queries,
        )
    except Exception as e:
        logger.exception(e)
        return PlainTextResponse(
            status_code=500, content=f"Error processing internal search query"
        )

    q = common.api_query_to_internal_q(
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

    end = min(end, project_info.num_vectors)
    if start > end:
        raise HTTPException(
            400, {"message": "'start' cannot be greater than 'end'"}
        )

    filter_specs = build_filter_specs(shot_scale, metadata_filter)

    _prefix = get_prefix(config)
    prefix = _prefix[search_in] if add_prefix else ""
    embedding_config = EmbeddingConfig(
        query_prefix=prefix,
        text_queries_weight=config.text_queries_weight,
        negative_queries_weight=config.negative_queries_weight,
    )
    
    search_output = search_service.search(
        q,
        embedding_config=embedding_config,
        media_type=media_type,
        feature_extractor_id=feature_extractor_id,
        start=start,
        end=end,
        filter_specs=filter_specs
    )

    if len(search_output.ids) == 0:
        return common.SearchResponse(
            time=0.0,
            video_audio_results=None,
            video_results=None,
            image_results=None,
        )
    all_thumbs = project_service.get_thumbnail_reader(thumbnails_to_send)(search_output.metadata)
    _get_shots_from_keyframes = functools.partial(get_shots_from_keyframes, project_service.wise_project)
    # supports shots
    is_shot_merge_supported = config.use_shots and search_in == MediaType.VIDEO
    response = construct_search_response(
        top_dist=search_output.distances,
        all_metadata=search_output.metadata,
        all_ext_metadata=search_output.ext_metadata,
        all_thumbs=all_thumbs,
        merge_function=_get_shots_from_keyframes if is_shot_merge_supported else merge_close_segments,
        search_in=search_in,
    )

    return response

@router.get("/featured", response_model=common.SearchResponse)
@common.add_response_time
async def handle_get_featured(
    project_service: ProjectServiceDep,
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
    modality = ModalityType.AUDIO if featured_in == MediaType.AV else ModalityType(featured_in) 
    search_output = search_service.featured(
        modality, feature_extractor_id, start, end, random_seed
    )
    all_thumbs = project_service.get_thumbnail_reader(thumbnails_to_send)(search_output.metadata)
    response = construct_search_response(
        top_dist=search_output.distances,
        all_metadata=search_output.metadata,
        all_ext_metadata=search_output.ext_metadata,
        all_thumbs=all_thumbs,
    )

    return response
