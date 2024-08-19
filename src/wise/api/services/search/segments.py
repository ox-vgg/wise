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

from ... import common


def merge_close_segments(_keyframes: list[common.VideoSegment], threshold: float = 4):
    """
    Takes a list of segments of a media file and merges them if they are close - within 4 seconds of each other
    The merged segment is represented by the best matching segment based on distance
    """
    merged_segments: list[common.VideoSegment] = []
    start = None
    current = None
    best = None
    for k in _keyframes:
        if start is None:
            # Start a new group
            start = k
            current = k
            best = k

        elif (k.ts - current.te) <= threshold:
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
        segments: list[common.VideoSegment],
        merge_function=merge_close_segments
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
