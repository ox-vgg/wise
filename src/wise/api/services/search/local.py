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

import faiss
import numpy as np

from wise.data_models import MediaType
from wise.feature.feature_extractor import FeatureExtMetadata
from wise.search.fts import WISEFTSQuery
from wise.wise_project import WiseProject

from ...common import Query
from ..embedding import EmbeddingConfig, EmbeddingService
from ..project import LocalWiseProjectService
from .base import SearchOutput
from .exceptions import UnknownSearchIndexError


logger = logging.getLogger(__name__)

class LocalSearchService:
    def __init__(self, project_service: LocalWiseProjectService, embedding_service: EmbeddingService):
        self.project_service = project_service
        self.embedding_service = embedding_service

        self.wise_project: WiseProject = project_service.wise_project
        self.search_indices = self.wise_project.load_search_indices()
        logger.info('search indices: %s', self.search_indices)
        self._featured_ids = project_service.featured_vectors_for_targets()

    def is_internal_search_supported(self, media_type: MediaType, feature_id: str) -> bool:
        search_index = self.search_indices[media_type][feature_id]
        return search_index.is_internal_search_supported

    def get_search_index_type(self, media_type: MediaType, feature_id: str) -> str:
        search_index = self.search_indices[media_type][feature_id]
        return search_index.index_type

    def reconstruct_vectors(self, media_type: MediaType, feature_id: str, vector_ids: list[int]) -> list[np.ndarray]:
        search_index = self.search_indices[media_type][feature_id]
        reconstructed_features = search_index.index.reconstruct_batch(vector_ids)
        features_list = []
        for i in range(0, reconstructed_features.shape[0]):
            features_list.append( np.expand_dims(reconstructed_features[i,], axis=0) )
        return features_list

    def search_with_feature(
        self,
        features: np.ndarray,
        media_type: MediaType,
        feature_extractor_id: str,
        start: int,
        end: int,
        filter_specs: dict,
        vector_id_constraint: np.ndarray | None = None,
        nprobe_override: int | None = None,
    ):
        search_index = self.search_indices[media_type][feature_extractor_id]
        if not filter_specs and vector_id_constraint is None:
            # search on full index without any filtering
            logger.info(
                "Performing search - media_type=%s, feature_extractor_id=%s",
                media_type,
                feature_extractor_id,
            )
            dist, ids = search_index.index.search(features, end)
        else:
            # search needs to be constrained by filter_specs and/or vector_id_constraint
            filtered_ids = None
            if filter_specs:
                logger.info(
                    "Applying filters for search - media_type=%s, feature_extractor_id=%s, filter_specs=%s",
                    media_type,
                    feature_extractor_id,
                    filter_specs,
                )
                filtered_ids = self.filter_vectors(
                    media_type, feature_extractor_id, filter_specs
                )
            if vector_id_constraint is not None:
                logger.info(
                    "Applying vector ID constraint for search - media_type=%s, feature_extractor_id=%s, num_ids=%d",
                    media_type,
                    feature_extractor_id,
                    len(vector_id_constraint),
                )
                vector_id_constraint = np.array(vector_id_constraint, dtype=np.int64)
                if filtered_ids is None:
                    filtered_ids = vector_id_constraint
                else:
                    filtered_ids = np.intersect1d(filtered_ids, vector_id_constraint)
            if filtered_ids is None or filtered_ids.size == 0:
                logger.info(
                    "No valid vector IDs after applying filters and constraints for search"
                )
                return SearchOutput()
            sel = faiss.IDSelectorBatch(filtered_ids)
            if search_index.index_type == 'IndexFlatIP':
                params = faiss.SearchParameters(sel=sel)
            elif search_index.index_type == 'IndexIVFFlat':
                nprobe = search_index.index.nprobe
                if nprobe_override is not None and nprobe_override > 0:
                    nlist = getattr(search_index.index, "nlist", None)
                    if nlist is not None:
                        nprobe = min(nprobe_override, nlist)
                    else:
                        nprobe = nprobe_override
                params = faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)
            else:
                raise UnknownSearchIndexError(f"Unknown index type: {search_index.index_type}")
            logger.info(
                "Performing filtered search - media_type=%s, feature_extractor_id=%s, num_filtered_ids=%d",
                media_type,
                feature_extractor_id,
                len(filtered_ids),
            )
            dist, ids = search_index.index.search(features, end, params=params)

        top_ids, top_dist = ids[0, start:end], dist[0, start:end]

        valid_ids_mask = top_ids != -1
        valid_ids = top_ids[valid_ids_mask].tolist()
        valid_dist = top_dist[valid_ids_mask].tolist()

        # Apply hook to transform Faiss distance scores
        valid_dist = self.embedding_service.transform_distances(feature_extractor_id, valid_dist)
        if not valid_ids:
            return SearchOutput()

        all_metadata = self.project_service.get_vector_and_media_metadata_for_ids(valid_ids)
        all_ext_metadata = self.project_service.get_vector_ext_metadata_for_ids(feature_extractor_id, valid_ids)

        return SearchOutput(
            ids=valid_ids,
            distances=valid_dist,
            metadata=all_metadata,
            ext_metadata=all_ext_metadata
        )

    def search(
        self,
        q: Query,
        embedding_config: EmbeddingConfig,
        media_type: MediaType,
        feature_extractor_id: str,
        start: int,
        end: int,
        filter_specs: dict,
        vector_id_constraint: np.ndarray | None = None
    ):
        features = self.embedding_service.embed(feature_extractor_id, embedding_config, q)
        return self.search_with_feature(
            features,
            media_type,
            feature_extractor_id,
            start,
            end,
            filter_specs,
            vector_id_constraint
        )

    def asr_search(self, q: WISEFTSQuery, media_type: MediaType, start: int, end: int):
        search_index = self.search_indices[media_type]["wise/metadata"]
        project_engine = self.wise_project.db_engine
        with project_engine.connect() as conn:
            all_metadata = search_index.search(conn, q, start, end)
            dist = list([-x for x in range(1, len(all_metadata) + 1)])
        n_results = len(all_metadata)
        all_ext_metadata = [FeatureExtMetadata()] * n_results
        return SearchOutput(
            ids=[x.id for x in all_metadata],
            distances=dist,
            metadata=all_metadata,
            ext_metadata=all_ext_metadata
        )

    def filter_vectors(self, media_type: MediaType, feature_extractor_id: str, filter_spec: dict) -> np.ndarray:
        id_constraint = np.array([], dtype=np.int64)
        project_engine = self.wise_project.db_engine

        metadata_query = filter_spec.get("metadata_query", None)
        if metadata_query:
            with project_engine.connect() as conn:
                media_ids = self.search_indices[media_type]["wise/metadata"].search(
                    conn, metadata_query, ids_only=True
                )
                if not media_ids:
                    return id_constraint

                vector_ids = self.wise_project.get_vector_ids(media_ids, media_type, feature_extractor_id)
                id_constraint = np.array(vector_ids, dtype=np.int64)

        shot_scale_query = filter_spec.get("shot_scale_query", None)
        if shot_scale_query:
            shot_scales = shot_scale_query["$in"]
            with project_engine.connect() as conn:

                result = self.wise_project.get_vector_ids_for_shot_scale(
                    shot_scales, media_type, feature_extractor_id,
                )
                shot_scale_constraint = np.array(result, dtype=np.int64)
                id_constraint = (
                    shot_scale_constraint
                    if len(id_constraint) == 0
                    else np.intersect1d(id_constraint, shot_scale_constraint)
                )
        return id_constraint

    def featured(
            self,
            media_type: MediaType,
            feature_extractor_id: str,
            start: int,
            end: int,
            random_seed: int = 42,
        ):

        # Select up to 1000 random image ids, using the specified random seed, from the set of 10000 ids
        if feature_extractor_id == "wise/metadata":
            # use other feature_extractor_id from the modality because "wise/metadata"
            # is a FTS search index and it does not support "feature_extractor.get_vector_metadata()"
            other_ids = [fid for fid in self._featured_ids[media_type] if fid != "wise/metadata"]
            if not other_ids:
                selected_ids = [] # i.e featured images not available
            else:
                selected_ids = self._featured_ids[media_type][other_ids[0]].copy()
        else:
            selected_ids = self._featured_ids[media_type][feature_extractor_id].copy()

        np.random.default_rng(seed=random_seed).shuffle(selected_ids)
        selected_ids = selected_ids[start:end]

        # Use 0 as a filler value for the distance array since this is not relevant for the featured images
        dist = [0.0] * len(selected_ids)

        if feature_extractor_id == "wise/metadata":
            # ensure that other_ids[0] (i.e. feature_extractor_id) specific vector metadata
            # are not shown in the featured images
            all_ext_metadata = [FeatureExtMetadata()] * len(selected_ids)
        else:
            all_ext_metadata = self.project_service.get_vector_ext_metadata_for_ids(feature_extractor_id, selected_ids)

        return SearchOutput(
            ids=selected_ids,
            distances=dist,
            metadata=self.project_service.get_vector_and_media_metadata_for_ids(selected_ids),
            ext_metadata=all_ext_metadata
        )
