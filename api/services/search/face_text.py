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

from typing import Hashable, Iterable, TypeVar

from src.data_models import MediaType
from ...common import MediaQueryTerm, Query, TextQueryTerm

T = TypeVar("T")


def get_face_text_embedder_id(
    media_type: MediaType,
    search_targets: dict[MediaType, list[str]],
    embedding_service,
    preferred_id: str | None = None,
    excluded_ids: set[str] | None = None,
) -> str | None:
    excluded_ids = excluded_ids or set()

    if preferred_id and preferred_id not in excluded_ids:
        extractor = embedding_service.feature_extractors.get(preferred_id)
        if extractor is not None and extractor.extract_text_features is not None:
            return preferred_id

    for candidate_id in search_targets.get(media_type, []):
        if candidate_id in excluded_ids or candidate_id == "wise/metadata":
            continue
        if "clip" not in candidate_id.lower():
            continue
        extractor = embedding_service.feature_extractors.get(candidate_id)
        if extractor is None:
            continue
        if extractor.extract_text_features is not None:
            return candidate_id

    return None


def _build_rank_map(ranked_keys: Iterable[Hashable]) -> dict[Hashable, int]:
    return {key: rank for rank, key in enumerate(ranked_keys, start=1)}


def _rrf_score(
    key: Hashable,
    face_rank: dict[Hashable, int],
    text_rank: dict[Hashable, int],
    k: int,
    face_weight: float,
    text_weight: float,
    face_score_map: dict[Hashable, float] | None = None,
    face_score_threshold: float | None = None,
    face_low_weight: float = 0.0,
) -> float:
    face_r = face_rank.get(key)
    text_r = text_rank.get(key)
    score = 0.0
    if face_r is not None:
        weight = face_weight
        if face_score_threshold is not None and face_score_map is not None:
            face_score = face_score_map.get(key)
            if face_score is not None and face_score < face_score_threshold:
                weight = face_low_weight
        score += weight * (1.0 / (k + face_r))
    if text_r is not None:
        score += text_weight * (1.0 / (k + text_r))
    return score


def _fuse_ranked_keys(
    face_ranked_keys: list[Hashable],
    text_ranked_keys: list[Hashable],
    *,
    k: int,
    face_weight: float,
    text_weight: float,
    face_score_map: dict[Hashable, float] | None = None,
    face_score_threshold: float | None = None,
    face_low_weight: float = 0.0,
) -> list[Hashable]:
    face_rank_map = _build_rank_map(face_ranked_keys)
    text_rank_map = _build_rank_map(text_ranked_keys)

    def sort_key(key: Hashable) -> tuple[float, int]:
        score = _rrf_score(
            key,
            face_rank_map,
            text_rank_map,
            k,
            face_weight,
            text_weight,
            face_score_map=face_score_map,
            face_score_threshold=face_score_threshold,
            face_low_weight=face_low_weight,
        )
        return (-score, face_rank_map.get(key, 10**9))

    return sorted(face_ranked_keys, key=sort_key)


def fuse_face_text_frame_results(
    face_items: list[tuple[Hashable, float, T]],
    text_ranked_keys: list[Hashable],
    *,
    k: int,
    face_weight: float,
    text_weight: float,
    start: int,
    end: int,
    face_score_threshold: float | None = None,
    face_low_weight: float = 0.0,
) -> list[tuple[T, int]]:
    text_key_set = set(text_ranked_keys)
    face_by_key: dict[Hashable, list[tuple[float, T]]] = {}
    face_ranked_keys: list[Hashable] = []
    seen_face_keys = set()

    for key, distance, payload in face_items:
        if key not in text_key_set:
            continue
        face_by_key.setdefault(key, []).append((distance, payload))
        if key not in seen_face_keys:
            face_ranked_keys.append(key)
            seen_face_keys.add(key)

    face_score_map = None
    if face_score_threshold is not None:
        face_score_map = {
            key: max(distance for distance, _ in matches)
            for key, matches in face_by_key.items()
        }

    fused_keys = _fuse_ranked_keys(
        face_ranked_keys,
        text_ranked_keys,
        k=k,
        face_weight=face_weight,
        text_weight=text_weight,
        face_score_map=face_score_map,
        face_score_threshold=face_score_threshold,
        face_low_weight=face_low_weight,
    )

    fused_payloads: list[T] = []
    for key in fused_keys:
        matches = face_by_key.get(key)
        if not matches:
            continue
        _, best_payload = max(matches, key=lambda item: item[0])
        fused_payloads.append(best_payload)

    fused_payloads = fused_payloads[start:end]
    total = len(fused_payloads)
    return [(payload, total - idx) for idx, payload in enumerate(fused_payloads)]

async def run_face_text_search(
    q: Query | None = None,
    *,
    media_type: MediaType,
    feature_extractor_id: str,
    text_feature_extractor_id: str,
    embedding_config,
    face_q: Query | None = None,
    text_q: Query | None = None,
    filter_specs: dict,
    start: int,
    end: int,
    num_vectors: int,
    face_text_options: dict | None,
    text_k_target: int,
    rrf_config: dict,
    face_score_threshold: float | None = None,
    embed_fn,
    search_face_fn,
    search_text_fn,
    get_face_distances_fn,
    build_text_ranked_keys_fn,
    build_face_items_fn,
    resolve_text_vector_ids_fn=None,
) -> tuple[object, object | None, list[tuple[T, int]]]:
    if q is not None:
        face_q = [
            item for item in q
            if isinstance(item, MediaQueryTerm) and item.qtype == "visual"
        ]
        text_q = [item for item in q if isinstance(item, TextQueryTerm)]
    if face_q is None or text_q is None:
        raise ValueError("Either q or both face_q and text_q must be provided.")
    # Decide the face-search k (adaptive if configured, otherwise use the request end).
    # Adaptively set the `k` value in K-Nearest Neighbors search based on threshold
    # which is known to roughly delineate more accurate vs less accurate matches.
    # If the last result in top-k matches is below the threshold, we expand k to get
    # more results.
    face_k = end
    if face_text_options:
        face_k_default = int(face_text_options.get("k_default", end))
        face_k_expanded = int(face_text_options.get("k_expanded", face_k_default))
        face_score_threshold = face_text_options.get("score_threshold", None)
        face_k = min(face_k_default, num_vectors)
    else:
        face_k_default = None
        face_k_expanded = None
        face_score_threshold = None

    # Run face search to obtain the initial candidate set based solely on face features.
    face_features = embed_fn(feature_extractor_id, embedding_config, face_q)
    face_output = await search_face_fn(face_features, 0, face_k, filter_specs)

    face_distances = get_face_distances_fn(face_output)
    if (
        face_text_options
        and face_k_default
        and face_score_threshold is not None
        and len(face_distances) >= face_k_default
    ):
        score_at_k = face_distances[face_k_default - 1]
        if score_at_k > face_score_threshold and face_k_expanded and face_k_expanded > face_k:
            face_k = min(face_k_expanded, num_vectors)
            # If the kth face score is strong, expand k and re-run face search.
            face_output = await search_face_fn(face_features, 0, face_k, filter_specs)
            face_distances = get_face_distances_fn(face_output)

    if not face_distances:
        return face_output, None, []

    # Map face results to vector IDs for the text extractor. For video,
    # constrain by (media_id, timestamp) so the text search is aligned to
    # the face-matching frames; for images, media_id alone is sufficient.
    vector_id_constraint = None
    if resolve_text_vector_ids_fn is not None:
        vector_id_constraint = resolve_text_vector_ids_fn(face_output)
        if not vector_id_constraint:
            return face_output, None, []

    # Perform text search but constrain it such that the results are limited
    # to the vector IDs obtained from the face search.
    text_features = embed_fn(text_feature_extractor_id, embedding_config, text_q)
    text_k = text_k_target
    if vector_id_constraint is not None:
        text_k = min(text_k_target, len(vector_id_constraint))
    text_output = await search_text_fn(
        text_features,
        0,
        text_k,
        filter_specs,
        vector_id_constraint=vector_id_constraint,
    )
    text_ranked_keys = build_text_ranked_keys_fn(text_output)
    if not text_ranked_keys:
        return face_output, text_output, []

    # Capture frame keys ranked by text search (media_id, timestamp).
    # Keep only face results whose frames survived the text filter.
    face_items = build_face_items_fn(face_output, set(text_ranked_keys))

    # Fuse face and text rankings at the frame level using RRF.
    # RRF rewards frames that appear high in either ranking or moderately high in both.
    rrf_k = int(rrf_config.get("k", 60))
    face_rrf_weight = float(rrf_config.get("face_weight", 1.0))
    text_rrf_weight = float(rrf_config.get("text_weight", 2.0))
    face_low_weight = float(rrf_config.get("face_low_weight", 0.0))
    fused = fuse_face_text_frame_results(
        face_items,
        text_ranked_keys,
        k=rrf_k,
        face_weight=face_rrf_weight,
        text_weight=text_rrf_weight,
        start=start,
        end=end,
        face_score_threshold=face_score_threshold,
        face_low_weight=face_low_weight,
    )
    return face_output, text_output, fused
