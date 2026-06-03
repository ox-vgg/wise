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

import asyncio
import functools
import logging
from typing import Literal

import numpy as np
from fastapi import Request

from wise.api import common
from wise.api.services.embedding import EmbeddingService
from wise.api.services.project import RemoteWiseProjectService
from wise.data_models import MediaType


logger = logging.getLogger(__name__)

def merge_response(a: common.SearchResponse, b: common.SearchResponse) -> common.SearchResponse:
    if a.image_results and b.image_results:
        a.image_results.total += b.image_results.total
        a.image_results.vectors.extend(b.image_results.vectors)
        a.image_results.images.update(b.image_results.images)
    elif b.image_results:
        a.image_results = b.image_results
    
    if a.video_results and b.video_results:
        a.video_results.total += b.video_results.total
        a.video_results.unmerged_windows.extend(b.video_results.unmerged_windows)
        a.video_results.merged_windows.extend(b.video_results.merged_windows)
        a.video_results.videos.update(b.video_results.videos)
    elif b.video_results:
        a.video_results = b.video_results

    if a.video_audio_results and b.video_audio_results:
        a.video_audio_results.total += b.video_audio_results.total
        a.video_audio_results.unmerged_windows.extend(b.video_audio_results.unmerged_windows)
        a.video_audio_results.merged_windows.extend(b.video_audio_results.merged_windows)
        a.video_audio_results.videos.update(b.video_audio_results.videos)
    elif b.video_audio_results:
        a.video_audio_results = b.video_audio_results

    return a

def sort_response(response: common.SearchResponse) -> common.SearchResponse:
    if response.image_results:
        response.image_results.vectors.sort(key=lambda x: x.distance, reverse=True)
    if response.video_results:
        response.video_results.unmerged_windows.sort(key=lambda x: x.distance, reverse=True)
        response.video_results.merged_windows.sort(key=lambda x: x.distance, reverse=True)
    if response.video_audio_results:
        response.video_audio_results.unmerged_windows.sort(key=lambda x: x.distance, reverse=True)
        response.video_audio_results.merged_windows.sort(key=lambda x: x.distance, reverse=True)
    return response

class RemoteSearchService:
    def __init__(self, remote_projects: dict[str, RemoteWiseProjectService], embedding_service: EmbeddingService):
        self.project_services = remote_projects
        self.embedding_service = embedding_service
    
    async def featured(
            self,
            media_type: MediaType,
            feature_extractor_id: str, 
            start: int, 
            end: int,
            random_seed: int = 42,
        ):

        all_responses = await asyncio.gather(*[
            project_service.featured(
                media_type, feature_extractor_id, start, end, random_seed
            ) for project_service in self.project_services.values()
        ])
        response = functools.reduce(merge_response, all_responses)
        response = sort_response(response)
        return response
    
    async def search(
            self,
            request: Request,
            endpoint: Literal["/search", "/search2"] = "/search"
        ) -> common.SearchResponse:

        all_responses = await asyncio.gather(*[
            project_service.search(
                request,
                endpoint=endpoint,
            ) for project_service in self.project_services.values()
        ])
        response = functools.reduce(merge_response, all_responses)
        response = sort_response(response)
        return response
    
    async def search_with_feature(
            self,
            features: np.ndarray,
            search_in: MediaType,
            feature_extractor_id: str, 
            start: int, 
            end: int,
            thumbnails_to_send: int = 0,
            shot_scale: list[int] | None = None,
            metadata_filter: list[str] = [],
        ) -> common.SearchResponse:

        all_responses = await asyncio.gather(*[
            project_service.search_with_feature(
                features, search_in, feature_extractor_id, start, end, thumbnails_to_send, shot_scale, metadata_filter
            ) for project_service in self.project_services.values()
        ])
        response = functools.reduce(merge_response, all_responses)
        response = sort_response(response)
        return response

    async def reconstruct_vectors(self, media_type: MediaType, feature_extractor_id: str, internal_ids: list[str]) -> list[np.ndarray]:
        async def handle_internal_id(internal_id: str) -> np.ndarray:
            # internal_id is of the form <project_name>_<vector_id>
            parts = internal_id.rsplit("/", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid internal_id: {internal_id}")
            project_id, _, vector_id = parts
            
            project_service = self.project_services.get(project_id)
            if project_service is None:
                raise ValueError(f"Project {project_id} not found")
            
            vectors = await project_service.reconstruct_vectors(
                media_type, feature_extractor_id, [vector_id]
            )
            return vectors[0]

        all_vectors = await asyncio.gather(*[
            handle_internal_id(v) for v in internal_ids
        ])
        
        return all_vectors
