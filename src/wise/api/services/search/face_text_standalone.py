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

from __future__ import annotations

from wise.data_models import MediaType

from ...common import MediaQueryTerm, Query, TextQueryTerm, VectorIdQueryTerm
from .face_text import run_face_text_search


def describe_face_text_query(
    q: Query,
    *,
    feature_extractor_id: str,
    embedding_service,
) -> tuple[bool, bool, bool]:
    has_text_queries = any(isinstance(term, TextQueryTerm) for term in q)
    has_image_queries = any(
        (isinstance(term, MediaQueryTerm) and term.qtype == "visual")
        or isinstance(term, VectorIdQueryTerm)
        for term in q
    )
    extractor = embedding_service.feature_extractors.get(feature_extractor_id)
    supports_text = extractor is not None and extractor.extract_text_features is not None
    return has_text_queries, has_image_queries, supports_text


def resolve_face_text_embedder(
    q: Query,
    *,
    media_type: MediaType,
    feature_extractor_id: str,
    search_targets: dict[MediaType, list[str]],
    embedding_service,
    preferred_id: str | None,
) -> tuple[str | None, bool, bool, bool]:
    has_text_queries, has_image_queries, supports_text = describe_face_text_query(
        q,
        feature_extractor_id=feature_extractor_id,
        embedding_service=embedding_service,
    )
    if not (has_text_queries and has_image_queries and not supports_text):
        return None, has_text_queries, has_image_queries, supports_text

    from .face_text import get_face_text_embedder_id

    text_feature_extractor_id = get_face_text_embedder_id(
        media_type,
        search_targets,
        embedding_service,
        preferred_id=preferred_id,
        excluded_ids={feature_extractor_id},
    )
    return text_feature_extractor_id, has_text_queries, has_image_queries, supports_text


async def run_face_text_search_standalone(
    q: Query,
    *,
    media_type: MediaType,
    feature_extractor_id: str,
    text_feature_extractor_id: str,
    embedding_config,
    filter_specs: dict,
    start: int,
    end: int,
    num_vectors: int,
    face_text_options: dict | None,
    text_k_target: int,
    rrf_config: dict,
    face_score_threshold: float | None,
    text_search_nprobe_override: int | None,
    embedding_service,
    search_service,
    project_service,
):
    async def _search_face(features, start_idx, end_idx, _filter_specs):
        return search_service.search_with_feature(
            features,
            media_type=media_type,
            feature_extractor_id=feature_extractor_id,
            start=start_idx,
            end=end_idx,
            filter_specs=_filter_specs,
        )

    async def _search_text(features, start_idx, end_idx, _filter_specs, vector_id_constraint=None):
        return search_service.search_with_feature(
            features,
            media_type=media_type,
            feature_extractor_id=text_feature_extractor_id,
            start=start_idx,
            end=end_idx,
            filter_specs=_filter_specs,
            vector_id_constraint=vector_id_constraint,
            nprobe_override=text_search_nprobe_override,
        )

    def _resolve_text_vector_ids(face_search_output):
        if media_type == MediaType.VIDEO:
            face_media_ts_pairs = list(dict.fromkeys(
                (meta.media_id, meta.timestamp)
                for meta in face_search_output.metadata
                if meta.timestamp is not None
            ))
            if face_media_ts_pairs:
                face_media_ids, face_timestamps = zip(*face_media_ts_pairs)
                return project_service.wise_project.get_vector_ids(
                    list(face_media_ids),
                    media_type,
                    text_feature_extractor_id,
                    timestamps=list(face_timestamps),
                )
            return []
        face_media_ids = [meta.media_id for meta in face_search_output.metadata]
        unique_media_ids = list(dict.fromkeys(face_media_ids))
        return project_service.wise_project.get_vector_ids(
            unique_media_ids, media_type, text_feature_extractor_id
        )

    def _build_text_ranked_keys(text_search_output):
        return [(meta.media_id, meta.timestamp) for meta in text_search_output.metadata]

    def _build_face_items(face_search_output, text_key_set):
        items: list[
            tuple[tuple[int, float | None], float, tuple[int, object, object]]
        ] = []
        for vid, dist, meta, ext in zip(
            face_search_output.ids,
            face_search_output.distances,
            face_search_output.metadata,
            face_search_output.ext_metadata,
        ):
            frame_key = (meta.media_id, meta.timestamp)
            if frame_key not in text_key_set:
                continue
            items.append((frame_key, dist, (vid, meta, ext)))
        return items

    return await run_face_text_search(
        q,
        media_type=media_type,
        feature_extractor_id=feature_extractor_id,
        text_feature_extractor_id=text_feature_extractor_id,
        embedding_config=embedding_config,
        filter_specs=filter_specs,
        start=start,
        end=end,
        num_vectors=num_vectors,
        face_text_options=face_text_options,
        text_k_target=text_k_target,
        rrf_config=rrf_config,
        face_score_threshold=face_score_threshold,
        embed_fn=embedding_service.embed,
        search_face_fn=_search_face,
        search_text_fn=_search_text,
        get_face_distances_fn=lambda output: output.distances,
        build_text_ranked_keys_fn=_build_text_ranked_keys,
        build_face_items_fn=_build_face_items,
        resolve_text_vector_ids_fn=_resolve_text_vector_ids,
    )
