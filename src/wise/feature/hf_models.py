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

# Segment parameters for Qwen3-VL-Embedding:
#   segment_duration=8s, segment_num_frames=16  →  2 fps (denser temporal sampling
#     than the model's 1 fps default; 16 frames is well within the 64-frame budget)
#   segment_overlap=4s  →  50% overlap ensures any 4-second action falls fully inside
#     at least one segment, avoiding missed retrievals at segment boundaries
REGISTRY: dict[str, dict] = {
    "hf/Qwen/Qwen3-VL-Embedding/2B": {
        "hf_model_id": "Qwen/Qwen3-VL-Embedding-2B",
        "modalities": ["video_segment", "text"],
        "segment_duration": 8.0,
        "segment_overlap": 4.0,
        "segment_num_frames": 16,
    },
    "hf/Qwen/Qwen3-VL-Embedding/8B": {
        "hf_model_id": "Qwen/Qwen3-VL-Embedding-8B",
        "modalities": ["video_segment", "text"],
        "segment_duration": 8.0,
        "segment_overlap": 4.0,
        "segment_num_frames": 16,
    },
}


def get_model_info(feature_id: str) -> dict | None:
    return REGISTRY.get(feature_id)


def is_segment_level_extractor(feature_id: str) -> bool:
    info = get_model_info(feature_id)
    if info is None:
        return False
    return "video_segment" in info.get("modalities", [])


def get_segment_params(feature_id: str) -> dict:
    info = get_model_info(feature_id)
    if info is None:
        raise KeyError(f"No registry entry for feature_id={feature_id!r}")
    return {
        "segment_duration": info["segment_duration"],
        "segment_overlap": info["segment_overlap"],
        "segment_num_frames": info["segment_num_frames"],
    }
