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
import logging
from collections.abc import Callable, Generator
from typing import Any

import torch
import torch.utils.data as torch_data
import torchvision as tv
from torchaudio.io import StreamReader

from wise.data_models import MediaChunkType
from wise.dataloader.dataset import MediaChunk

logger = logging.getLogger(__name__)

THUMBNAIL_FPS = 2


def _encode_jpeg_frames(frames: torch.Tensor) -> list[torch.Tensor]:
    """Encode (N, C, H, W) uint8 RGB tensor to a list of JPEG byte tensors."""
    return [
        tv.io.encode_jpeg(frames[i], quality=80)
        for i in range(frames.shape[0])
    ]


class VideoSegmentDataset(torch_data.IterableDataset):
    """Iterable dataset that yields overlapping video segments.

    Each iteration yields:
        (media_id, {MediaChunkType.VIDEO: {extractor_id: MediaChunk}, ...})

    The VIDEO MediaChunk contains all frames for one segment as a
    (N, C, H, W) tensor (after preprocessing).  MediaChunk.pts is the
    segment start time in seconds.

    Thumbnail frames (2 fps JPEG) are emitted under MediaChunkType.THUMBNAILS
    for the non-overlapping stride portion of each segment only, to avoid
    duplicating thumbnails across overlapping windows.  The very first segment
    of each video emits thumbnails for its full duration.
    """

    def __init__(
        self,
        input_files: dict[str, str],
        segment_duration: float,
        segment_overlap: float,
        num_frames_per_segment: int,
        preprocessing_function_map: dict[str, Callable],
        thumbnails: bool = True,
    ):
        super().__init__()
        self._filelist: dict[str, str] = input_files
        self._segment_duration = segment_duration
        self._segment_overlap = segment_overlap
        self._stride = segment_duration - segment_overlap
        self._num_frames = num_frames_per_segment
        self._preprocessing_function_map = preprocessing_function_map
        self._thumbnails = thumbnails

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_frames_from_segment(
        self, path: str, start: float
    ) -> tuple[torch.Tensor, list[float]] | None:
        """Decode up to self._num_frames frames starting at `start` seconds.

        Uses frames_per_chunk=1 so that every chunk carries its own PTS.
        torchaudio only exposes the start timestamp of the first frame when
        frames_per_chunk > 1, making per-frame timestamps unavailable without
        frame-rate arithmetic. Reading one frame at a time gives an accurate
        PTS for each frame directly from the decoder.

        Returns (frames, pts_list) where frames is (N, C, H, W) uint8 and
        pts_list contains one decoder-provided timestamp per frame,
        or None if no frames could be decoded.
        """
        target_fps = self._num_frames / self._segment_duration

        reader = StreamReader(path)
        reader.add_basic_video_stream(
            frames_per_chunk=1,
            frame_rate=target_fps,
            format="rgb24",
        )
        reader.seek(start)

        frame_list: list[torch.Tensor] = []
        pts_list: list[float] = []
        for (chunk,) in reader.stream():
            if chunk is None:
                continue
            frame_list.append(torch.as_tensor(chunk))
            pts_list.append(chunk.pts)
            if len(frame_list) >= self._num_frames:
                break

        if not frame_list:
            return None

        return torch.cat(frame_list, dim=0), pts_list

    def _make_thumbnail_frames(
        self, path: str, start: float, end: float
    ) -> list[torch.Tensor] | None:
        """Decode 2-fps thumbnail frames for the interval [start, end)."""
        duration = end - start
        if duration <= 0:
            return None

        reader = StreamReader(path)
        reader.add_basic_video_stream(
            frames_per_chunk=max(1, int(duration * THUMBNAIL_FPS) + 2),
            frame_rate=THUMBNAIL_FPS,
            format="rgb24",
            height=192,
            width=-2,
        )
        reader.seek(start)

        jpegs: list[torch.Tensor] = []
        elapsed = 0.0
        for (chunk,) in reader.stream():
            if chunk is None:
                continue
            t = torch.as_tensor(
                chunk
            )  # (T, C, H, W) — torchaudio outputs NCHW for rgb24
            elapsed += t.shape[0] / THUMBNAIL_FPS
            encoded = _encode_jpeg_frames(t)
            jpegs.extend(encoded)
            if elapsed >= duration:
                break

        return jpegs if jpegs else None

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _iter_file(
        self, media_id: str, path: str
    ) -> Generator[tuple[str, dict], Any, None]:
        segment_start = 0.0
        first_segment = True
        n_segments = 0

        while True:
            result = self._sample_frames_from_segment(path, segment_start)
            if result is None:
                # No frames returned — past end of file
                break

            frames, pts_list = result
            segment_end = pts_list[-1]

            # Skip trailing remnants too short to embed meaningfully. The overlap
            # guarantee ensures this content was already covered by the previous segment.
            # Always process the first segment so short videos are not dropped entirely.
            if (
                not first_segment
                and (segment_end - pts_list[0]) < self._segment_overlap
            ):
                break

            # Apply per-extractor preprocessing
            video_chunks: dict[str, MediaChunk] = {}
            for (
                extractor_id,
                preprocess_fn,
            ) in self._preprocessing_function_map.items():
                if preprocess_fn is not None:
                    preprocessed = preprocess_fn(frames)
                else:
                    preprocessed = frames
                video_chunks[extractor_id] = MediaChunk(
                    tensor=preprocessed, pts=pts_list[0], end_pts=pts_list[-1]
                )

            chunks: dict[MediaChunkType, Any] = {
                MediaChunkType.VIDEO: video_chunks,
            }

            # Thumbnails: emit for the stride portion only (avoid duplicates)
            if self._thumbnails:
                if first_segment:
                    thumb_start = pts_list[0]
                    thumb_end = segment_end
                else:
                    thumb_start = pts_list[0] + self._segment_overlap
                    thumb_end = segment_end

                if thumb_end > thumb_start:
                    thumb_jpegs = self._make_thumbnail_frames(
                        path, thumb_start, thumb_end
                    )
                    if thumb_jpegs:
                        chunks[MediaChunkType.THUMBNAILS] = MediaChunk(
                            tensor=thumb_jpegs,
                            pts=thumb_start,
                        )

            yield media_id, chunks
            n_segments += 1

            segment_start += self._stride
            first_segment = False

        if n_segments == 0:
            logger.warning("No segments decoded from %r", path)

    def _get_media_iterator(
        self, id_list: list[str]
    ) -> Generator[tuple[str, dict], Any, None]:
        for media_id in id_list:
            path = self._filelist[media_id]
            try:
                yield from self._iter_file(media_id, path)
            except Exception:
                logger.exception(
                    'Exception when processing "%s: %s"', media_id, path
                )

    def __iter__(self) -> Generator[tuple[str, dict], Any, None]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self._get_media_iterator(list(self._filelist.keys()))

        n_workers = worker_info.num_workers
        worker_id = worker_info.id
        file_list = list(
            itertools.islice(self._filelist.keys(), worker_id, None, n_workers)
        )
        return self._get_media_iterator(file_list)
