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

from functools import cached_property
from typing import Literal

import httpx
import numpy as np
from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from wise.api import common
from wise.api.services.project.base import ProjectInfo, WiseProjectService
from wise.config import APIConfig
from wise.data_models import MediaType, VectorAndMediaMetadata


class RemoteWiseProjectService(WiseProjectService):
    def __init__(self, project_uri: str, config: APIConfig):
        self.project_uri = project_uri
        # Initialize other necessary attributes here
        self.config = config

    @cached_property
    def client(self):
        return httpx.AsyncClient(base_url=self.project_uri, timeout=30.0)

    @cached_property
    def sync_client(self):
        return httpx.Client(base_url=self.project_uri, timeout=30.0)
    @property
    def name(self) -> str:
        return self.project_uri.strip('/').split("/")[-1]

    @property
    def info(self) -> ProjectInfo:
        # Implement logic to retrieve project info from remote service

        resp = self.sync_client.get("/info")
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("Failed to retrieve project info from remote service")
        # Parse and return ProjectInfo object
        return ProjectInfo.model_validate(data)

    async def forward(self, full_path: str, request: Request):
        # see https://github.com/fastapi/fastapi/discussions/7382#discussioncomment-5136466
        url = httpx.URL(
            path=full_path,
            query=request.url.query.encode("utf-8")
        )
        rp_req = self.client.build_request(
            request.method, url,
            headers=request.headers.raw,
            content=request.stream()
        )
        rp_resp = await self.client.send(rp_req, stream=True)
        return StreamingResponse(
            rp_resp.aiter_raw(),
            status_code=rp_resp.status_code,
            headers=rp_resp.headers,
            background=BackgroundTask(rp_resp.aclose),
        )

    async def featured(
            self,
            media_type: MediaType,
            feature_extractor_id: str,
            start: int,
            end: int,
            random_seed: int = 42,
        ):
        resp = await self.client.get(
            "/featured",
            params={
                "featured_in": media_type.value,
                "feature_extractor_id": feature_extractor_id,
                "start": start,
                "end": end,
                "random_seed": random_seed,
            }
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("Failed to retrieve featured items from remote service")

        response =  common.SearchResponse.model_validate(data)
        response = self.modify_response(response)

        return response

    async def search(
            self,
            request: Request,
            endpoint: Literal['/search', '/search2']= "/search"
    ):
        url = httpx.URL(
            path=endpoint,
            query=request.url.query.encode("utf-8")
        )
        rp_req = self.client.build_request(
            request.method, url,
            headers=request.headers.raw,
            content=await request.body()
        )
        rp_resp = await self.client.send(rp_req)
        rp_resp.raise_for_status()
        data = rp_resp.json()
        if not data:
            raise ValueError("Failed to retrieve search results from remote service")
        response =  common.SearchResponse.model_validate(data)
        response = self.modify_response(response)

        return response

    async def search_with_feature(
            self,
            features: np.ndarray,
            search_in: MediaType,
            feature_extractor_id: str,
            start: int = 0,
            end: int = 20,
            thumbnails_to_send: int = 0,
            shot_scale: list[int] | None = None,
            metadata_filter: list[str] = [],
        ):
        vector_qterm = common.VectorQueryTerm(
            term_id='_',
            is_negative=False,
            vector=features,
        )

        resp = await self.client.post(
            "/search_with_feature",
            params={
                "search_in": search_in.value,
                "feature_extractor_id": feature_extractor_id,
                "start": start,
                "end": end,
                "thumbnails_to_send": thumbnails_to_send,
                "shot_scale": shot_scale,
                "metadata_filter": metadata_filter,
            },
            json=vector_qterm.model_dump()
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("Failed to retrieve search results from remote service")
        response =  common.SearchResponse.model_validate(data)
        response = self.modify_response(response)

        return response

    async def related_vectors(self, vector_id: int) -> list[common.VectorInfo]:
        resp = await self.client.get(
            f"/related-vectors/{vector_id}",
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError("Failed to retrieve featured items from remote service")
        response =  [common.VectorInfo.model_validate(d) for d in data]
        response = self.modify_vector_info(response)
        return response

    async def reconstruct_vectors(self, media_type: MediaType, feature_extractor_id: str, internal_ids: list[str]) -> list[np.ndarray]:
        resp = await self.client.get(
            "/vectors",
            params={
                "search_in": media_type.value,
                "feature_extractor_id": feature_extractor_id,
                "internal_ids": internal_ids,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("Failed to retrieve vectors from remote service")

        arr_obj = common.NPArray.model_validate(data)  # validate the response
        vectors = arr_obj.to_array()
        return [np.expand_dims(x, axis=0) for x in vectors]

    def modify_vector_info(self, vectors: list[common.VectorInfo]) -> list[common.VectorInfo]:
        for item in vectors:
            item.media_id = f"{self.name}/{item.media_id}"
            item.link = f"shard/{self.name}/{item.link}"
            item.thumbnail = f"shard/{self.name}/{item.thumbnail}"

        return vectors
    def modify_image_results(self, results: common.ImageResults) -> common.ImageResults:
        images = {}
        for item in results.images.values():
            item.id = f"{self.name}/{item.id}"
            images[item.id] = item

        results.images = images
        results.vectors = self.modify_vector_info(results.vectors)
        return results

    def modify_video_results(self, results: common.VideoResults | common.VideoAudioResults) -> common.VideoResults | common.VideoAudioResults:
        videos = {}
        for item in results.videos.values():
            item.id = f"{self.name}/{item.id}"
            item.timeline_hover_thumbnails = f"shard/{self.name}/{item.timeline_hover_thumbnails}"
            videos[item.id] = item
        results.videos = videos

        results.unmerged_windows = self.modify_vector_info(results.unmerged_windows)
        results.merged_windows = self.modify_vector_info(results.merged_windows)
        return results

    def modify_response(self, response: common.SearchResponse) -> common.SearchResponse:
        # Modify the response to adjust thumbnail URLs
        if response.image_results is not None:
            response.image_results = self.modify_image_results(response.image_results)

        if response.video_results is not None:
            response.video_results = self.modify_video_results(response.video_results)

        if response.video_audio_results is not None:
            response.video_audio_results = self.modify_video_results(response.video_audio_results)

        return response

    def get_thumbnail_reader(self):
        def _thumbnail_url(_m: VectorAndMediaMetadata):
            return f"shard/{self.name}/thumbnail?media_id={_m.media_id}&timestamp={_m.timestamp}"

        def inner(vector_and_media_metadata_list: list[VectorAndMediaMetadata]):
            thumbs = list(map(
                _thumbnail_url,
                vector_and_media_metadata_list
            ))
            return thumbs

        return inner

    @classmethod
    def get_info(cls, projects: dict[str, "RemoteWiseProjectService"], name: str):
        infos = [p.info for p in projects.values()]
        merged_info = ProjectInfo.reduce(name, infos)
        merged_info.enable_facets = False
        return merged_info
