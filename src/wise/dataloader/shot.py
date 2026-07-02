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

import numpy as np
import torch.utils.data as torch_data

from wise.data_models import MediaChunkType

logger = logging.getLogger(__name__)


class ShotStream(torch_data.IterableDataset):
    def __init__(self, dataset, shots, params):
        super().__init__()
        self.dataset = dataset
        self.shots = shots  # precomputed shot boundaries for each media item contained in the dataset
        self.params = params
        self.stream_hooks = (
            {}
        )  # stores the indices of the frames in each chunk that correspond to the shot centers
        self.stream_hooks_metadata = (
            {}
        )  # stores metadata for each hook contained in "stream_hooks" (useful for debugging)
        self.stream_chunk_timestamps = (
            {}
        )  # stores the expected timestamp of frames in each chunk retrieved by the DataLoader (useful for sanity checks)
        self.precompute_stream_hooks()

    def precompute_stream_hooks(self):
        frame_rate = self.params["video_frame_rate"]
        frame_rate_inv = 1 / frame_rate
        frame_rate_by_2 = frame_rate / 2
        video_frames_per_chunk = self.params["video_frames_per_chunk"]
        self.stream_hooks = {}
        self.stream_hooks_metadata = {}
        self.stream_chunk_timestamps = {}
        self.media_last_chunk_index = {}
        for media_id in self.shots:
            start_times = [shot["start_time"] for shot in self.shots[media_id]]
            end_times = [shot["end_time"] for shot in self.shots[media_id]]
            shot_true_centers = [
                (start + end) / 2 for start, end in zip(start_times, end_times)
            ]
            # The stream only contains uniformly sampled frames which may not correspond to the true
            # center of the shot. So we compute the approximate center which is closest to the true center.
            shot_approx_centers = [
                round((start + end) * frame_rate_by_2) * frame_rate_inv
                for start, end in zip(start_times, end_times)
            ]
            video_frame_timestamps = [
                frame_time
                for frame_time in np.arange(
                    start_times[0], end_times[-1], frame_rate_inv
                )
            ]

            current_chunk_index = 0
            self.stream_hooks[media_id] = {}
            self.stream_hooks_metadata[media_id] = {}
            self.stream_chunk_timestamps[media_id] = {}
            for shot_index, approx_center in enumerate(shot_approx_centers):
                # check if approx_center appears in video_frame_timestamps corresponding to the current chunk
                approx_center_index = None
                approx_center_found = False
                while (
                    approx_center_index is None
                    and current_chunk_index < len(video_frame_timestamps)
                ):
                    chunk_timestamps = video_frame_timestamps[
                        current_chunk_index
                        * video_frames_per_chunk : (current_chunk_index + 1)
                        * video_frames_per_chunk
                    ]
                    approx_center_index = (
                        chunk_timestamps.index(approx_center)
                        if approx_center in chunk_timestamps
                        else None
                    )
                    if approx_center_index is None:
                        current_chunk_index += 1
                        continue  # approx_center for this shot not found in current chunk (maybe it is in the next chunk)
                    else:
                        if (
                            current_chunk_index
                            not in self.stream_hooks[media_id]
                        ):
                            self.stream_hooks[media_id][
                                current_chunk_index
                            ] = []
                            self.stream_hooks_metadata[media_id][
                                current_chunk_index
                            ] = []
                            self.stream_chunk_timestamps[media_id][
                                current_chunk_index
                            ] = chunk_timestamps
                            self.media_last_chunk_index[media_id] = (
                                current_chunk_index
                            )
                        self.stream_hooks[media_id][
                            current_chunk_index
                        ].append(approx_center_index)
                        self.stream_hooks_metadata[media_id][
                            current_chunk_index
                        ].append(
                            {
                                "start_time": start_times[shot_index],
                                "end_time": end_times[shot_index],
                                "true_center": shot_true_centers[shot_index],
                                "approx_center": approx_center,
                                "approx_center_chunk_index": current_chunk_index,
                                "approx_center_index": approx_center_index,
                            }
                        )
                        approx_center_found = True
                        break  # continue search for the next shot
                if not approx_center_found:
                    logger.warning(
                        "Skipping media_id=%d, shot=%d, time=%f:%f and"
                        " approximate center %f as it was not found in stream"
                        " generated by the DataLoader",
                        media_id,
                        shot_index,
                        start_times[shot_index],
                        end_times[shot_index],
                        approx_center,
                    )
        logger.info(
            "Precomputed stream hooks for %d media items.",
            len(self.stream_hooks),
        )

    def __iter__(self):
        self.seen_chunk_indices = {}
        for item in iter(self.dataset):
            media_id, chunks = item

            if media_id not in self.stream_hooks:
                logger.warning(
                    "Skipping media_id=%d as it does not have any shots",
                    media_id,
                )
                continue
            if media_id not in self.seen_chunk_indices:
                self.seen_chunk_indices[media_id] = 0
            else:
                self.seen_chunk_indices[media_id] += 1
            current_chunk_index = self.seen_chunk_indices[media_id]

            if (
                MediaChunkType.VIDEO in chunks
                and chunks[MediaChunkType.VIDEO] is not None
            ):
                for feature_extractor_id in chunks[MediaChunkType.VIDEO]:
                    if (
                        chunks[MediaChunkType.VIDEO][feature_extractor_id]
                        is None
                    ):
                        logger.warning(
                            "Ignoring empty chunk: media_id=%d,"
                            " current_chunk_index=%d",
                            media_id,
                            current_chunk_index,
                        )
                        continue
                    # chunks[MediaChunkType.VIDEO][feature_extractor_id].tensor is a torch tensor of shape [BATCH,CHANNEL,WIDTH,HEIGHT]
                    # We retain only batch indices contained in self.stream_hooks[media_id][chunk_index]
                    if (
                        current_chunk_index in self.stream_hooks[media_id]
                        and self.stream_hooks[media_id][current_chunk_index]
                        is not None
                    ):
                        # Sanity check 1: ensure that the DataLoader timestamp matches our pre-computed timestamp
                        chunk_first_pts = self.stream_chunk_timestamps[
                            media_id
                        ][current_chunk_index][0]
                        if (
                            chunks[MediaChunkType.VIDEO][
                                feature_extractor_id
                            ].pts
                            != chunk_first_pts
                        ):
                            logger.warning(
                                "Unexpected chunks: pts=%s does not match"
                                " expected %s",
                                chunks[MediaChunkType.VIDEO][
                                    feature_extractor_id
                                ].pts,
                                chunk_first_pts,
                            )
                            continue
                        # Sanity check 2: ensure that the DataLoader batch size matches the number of timestamps (or frames) in our pre-computed chunk
                        # The DataLoader may return fewer frames than expected. If this happens, we can safely ignore the missing frames as each
                        # batch contains the `pts` of the first frame in the chunk.
                        chunk_frame_count = chunks[MediaChunkType.VIDEO][
                            feature_extractor_id
                        ].tensor.shape[0]
                        if (
                            len(
                                self.stream_chunk_timestamps[media_id][
                                    current_chunk_index
                                ]
                            )
                            != chunk_frame_count
                        ):
                            logger.warning(
                                "Ignoring unexpected batch size: expected=%d"
                                " but got=%d for media_id=%d, chunk_index=%d,"
                                " pts=%s",
                                len(
                                    self.stream_chunk_timestamps[media_id][
                                        current_chunk_index
                                    ]
                                ),
                                chunk_frame_count,
                                media_id,
                                current_chunk_index,
                                chunk_first_pts,
                            )
                            # ignore the frames missing in this batch
                            self.stream_chunk_timestamps[media_id][
                                current_chunk_index
                            ] = self.stream_chunk_timestamps[media_id][
                                current_chunk_index
                            ][
                                :chunk_frame_count
                            ]
                            retained_indices = []
                            for chunk_frame_index in self.stream_hooks[
                                media_id
                            ][current_chunk_index]:
                                if chunk_frame_index < chunk_frame_count:
                                    retained_indices.append(chunk_frame_index)
                            self.stream_hooks[media_id][
                                current_chunk_index
                            ] = retained_indices
                        # update the tensor and pts to retain only the frames corresponding to the shots
                        chunks[MediaChunkType.VIDEO][
                            feature_extractor_id
                        ].tensor = chunks[MediaChunkType.VIDEO][
                            feature_extractor_id
                        ].tensor[
                            self.stream_hooks[media_id][current_chunk_index]
                        ]
                        chunks[MediaChunkType.VIDEO][
                            feature_extractor_id
                        ].pts = []
                        for chunk_frame_index in self.stream_hooks[media_id][
                            current_chunk_index
                        ]:
                            chunks[MediaChunkType.VIDEO][
                                feature_extractor_id
                            ].pts.append(
                                self.stream_chunk_timestamps[media_id][
                                    current_chunk_index
                                ][chunk_frame_index]
                            )
                    else:
                        # This chunk does not contain any frames corresponding to the shots
                        chunks[MediaChunkType.VIDEO][
                            feature_extractor_id
                        ] = None
            # If thumbnails are present, we also filter them based on the retained frames
            if (
                MediaChunkType.THUMBNAILS in chunks
                and chunks[MediaChunkType.THUMBNAILS] is not None
            ):
                if (
                    current_chunk_index in self.stream_hooks[media_id]
                    and self.stream_hooks[media_id][current_chunk_index]
                    is not None
                ):
                    thumb_indices = self.stream_hooks[media_id][
                        current_chunk_index
                    ]
                    chunks[MediaChunkType.THUMBNAILS].tensor = [
                        chunks[MediaChunkType.THUMBNAILS].tensor[i]
                        for i in thumb_indices
                    ]
                    chunks[MediaChunkType.THUMBNAILS].pts = chunks[
                        MediaChunkType.VIDEO
                    ][feature_extractor_id].pts
                else:
                    chunks[MediaChunkType.THUMBNAILS] = None
            yield item
