import itertools
import logging
from pathlib import Path
from uuid import uuid4, UUID
from typing import List, Dict, Callable, Optional, Union
from .streamreader import (
    SourceMediaType,
    StreamOutputOptions,
    BasicAudioStreamOutputOptions,
    BasicVideoStreamOutputOptions,
    get_media_type,
    get_media_info,
    get_stream_duration,
    get_stream_reader,
)
from .utils import md5
from pydantic import dataclasses, ConfigDict
import torch
import torch.utils.data as torch_data

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MediaMetadata(object):
    path: str
    md5sum: str
    media_type: SourceMediaType
    format: str
    width: int
    height: int
    num_frames: int
    duration: Optional[float]
    fps: Optional[float]
    extra: Dict
    id: UUID = dataclasses.Field(default_factory=uuid4)


def get_media_metadata(url: str):
    # TODO: Update the code to handle remote path
    # Only md5sum will be a problem.

    # TODO: How to handle subtitle

    # Get stream metadata
    video_stream_info, audio_stream_info = get_media_info(
        url, guess_missing_video_info=True
    )

    # Get media type
    media_type = get_media_type(video_stream_info, audio_stream_info)

    # Get md5sum for file.
    # TODO Update code to ignore for remote file
    # Alternative, create a io reader, and compute hash at a level above
    md5sum = md5(url)

    # Get media type
    # Based on media type, get other fields
    # fps, duration, width, height, channels, sample_rate, extra

    # Must be one of image, audio-only, video-only or av
    if media_type == SourceMediaType.IMAGE:
        # image
        # TODO check for iptc, exif and other metadata
        return MediaMetadata(
            path=url,
            md5sum=md5sum,
            media_type=media_type,
            format=video_stream_info.codec,
            width=video_stream_info.width,
            height=video_stream_info.height,
            duration=None,
            num_frames=1,
            fps=None,
            extra={},
        )
    elif media_type == SourceMediaType.VIDEO:
        # video
        duration = get_stream_duration(video_stream_info)

        return MediaMetadata(
            path=url,
            md5sum=md5sum,
            media_type=media_type,
            format=video_stream_info.codec,
            width=video_stream_info.width,
            height=video_stream_info.height,
            duration=duration,
            num_frames=video_stream_info.num_frames,
            fps=video_stream_info.frame_rate,
            extra={},
        )
    elif media_type == SourceMediaType.AUDIO:
        # Audio-only
        duration = get_stream_duration(audio_stream_info)
        return MediaMetadata(
            path=url,
            md5sum=md5sum,
            media_type=media_type,
            format=audio_stream_info.codec,
            width=-1,
            height=-1,
            duration=duration,
            num_frames=-1,
            fps=None,
            extra={
                "channels": audio_stream_info.num_channels,
                "sample_rate": audio_stream_info.sample_rate,
            },
        )
    elif media_type == SourceMediaType.AV:
        # Both are present, classify it as video
        duration = get_stream_duration(video_stream_info)
        return MediaMetadata(
            path=url,
            md5sum=md5sum,
            media_type=media_type,
            format=f"{video_stream_info.codec}/{audio_stream_info.codec}",
            width=video_stream_info.width,
            height=video_stream_info.height,
            duration=duration,
            num_frames=video_stream_info.num_frames,
            fps=video_stream_info.frame_rate,
            extra={
                "channels": audio_stream_info.num_channels,
                "sample_rate": audio_stream_info.sample_rate,
            },
        )

    else:
        raise NotImplementedError(f"Unknown media type - {media_type}")


def IdentityTransform(x: torch.Tensor):
    return x


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MediaChunk:
    tensor: torch.Tensor
    pts: float


def get_segment_length(opts: StreamOutputOptions):
    frames_per_chunk = opts.frames_per_chunk
    rate = (
        opts.frame_rate
        if isinstance(opts, BasicVideoStreamOutputOptions)
        else opts.sample_rate
    )
    if rate is None:
        # Need to use the input media info to calculate the rate
        raise NotImplementedError()
    return frames_per_chunk / rate


class MediaDataset(torch_data.IterableDataset):
    """
    MediaDataset is a custom pytorch iterable-style dataset for reading
    a collection of media files and converting them into torch
    tensors. The provided transformation corresponding to the output stream
    is applied to the output tensors before returning to the user

    See __main__.py for usage

    Inputs:
    - input_files: Map of media_id to file location
    - output_stream_opts: Options to configure the output stream from the StreamReader
    - transforms: corresponding transforms to apply to the output stream
    - offset: Optional offset to remove from the beginning of each media file. Will result in no output if trying to seek an image

    VIDEO -> Applied to the image / video clip (after stacking frames specified by frames_per_chunk for video_clip)
    AUDIO -> Applied to the audio clip (after stacking samples specified by frames_per_chunk)
    """

    def __init__(
        self,
        input_files: Union[List[Path], Dict[str, Path]],
        output_stream_opts=List[StreamOutputOptions],
        transforms: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        offset: Optional[float] = None,
    ):
        super(MediaDataset).__init__()

        self._filelist = (
            input_files
            if isinstance(input_files, dict)
            else dict(enumerate(input_files))
        )
        self._transforms = (
            transforms
            if transforms is not None
            else [IdentityTransform for _ in range(len(self._output_stream_opts))]
        )
        self._output_stream_opts = output_stream_opts
        self._offset = offset or 0.0
        # verify if length of output_stream_opts and transform matches up
        assert len(self._transforms) == len(self._output_stream_opts)

    def _get_media_iterator(self, id_list: List[Union[str, int]]):
        stream_segment_lengths = [
            get_segment_length(x) for x in self._output_stream_opts
        ]
        for _id in id_list:
            path = self._filelist[_id]
            try:
                reader = get_stream_reader(str(path), self._output_stream_opts)

                reader.seek(self._offset)
                for sidx, c in enumerate(reader.stream()):
                    # Might contain 1 or many output streams. Apply the corresponding transform
                    stream_outputs = tuple(
                        (
                            MediaChunk(
                                tensor=stream_transform(torch.Tensor(stream_chunk)),
                                pts=sidx
                                * stream_segment_lengths[
                                    idx
                                ],  # Only available in Pytorch 2.0+ stream_chunk.pts,
                            )
                            if stream_chunk is not None
                            else None
                        )
                        for idx, (stream_chunk, stream_transform) in enumerate(
                            zip(c, self._transforms)
                        )
                    )
                    yield _id, stream_outputs
            except Exception:
                logger.exception(f'Exception when processing "{_id}: {path}"')

    def __iter__(
        self,
    ):
        """
        Creates the iterator used by the dataloader

        Note:
        Dataloader creates a copy of this dataset in each worker
        Based on worker_info object, we shard the input file list so that
        each worker works on a part of the list. The dataloader will then
        collate the inputs to form a batch if requested.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Same process
            # Construct iterator with all files
            return self._get_media_iterator(self._filelist.keys())

        print(worker_info)
        n_workers = worker_info.num_workers
        worker_id = worker_info.id

        # Split files by worker and yield the worker's shard
        # Each worker will process the video file with index in (w_id, w_id + n, w_id + 2n, ...)
        file_list = itertools.islice(self._filelist.keys(), worker_id, None, n_workers)

        return self._get_media_iterator(file_list)


class AudioDataset(MediaDataset):

    def __init__(
        self,
        input_files: Union[List[Path], Dict[str, Path]],
        samples_per_chunk: int,
        *,
        preprocessing_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sample_rate: Optional[int] = None,
        offset: Optional[float] = None,
    ):

        stream_opts = [
            BasicAudioStreamOutputOptions(
                frames_per_chunk=samples_per_chunk,
                sample_rate=sample_rate,
            )
        ]
        transforms = [preprocessing_function]
        super(AudioDataset, self).__init__(
            input_files=input_files,
            output_stream_opts=stream_opts,
            transforms=transforms,
            offset=offset,
        )

    def __iter__(self):
        _iter = super(AudioDataset, self).__iter__()
        for _id, (chunk,) in _iter:
            yield _id, chunk


class VideoDataset(MediaDataset):

    def __init__(
        self,
        input_files: Union[List[Path], Dict[str, Path]],
        frames_per_chunk: int,
        *,
        preprocessing_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        frame_rate: Optional[int] = None,
        offset: Optional[float] = None,
    ):

        stream_opts = [
            BasicVideoStreamOutputOptions(
                frames_per_chunk=frames_per_chunk, frame_rate=frame_rate
            )
        ]
        transforms = [preprocessing_function]
        super(VideoDataset, self).__init__(
            input_files=input_files,
            output_stream_opts=stream_opts,
            transforms=transforms,
            offset=offset,
        )

    def __iter__(self):
        _iter = super(VideoDataset, self).__iter__()
        for _id, (chunk,) in _iter:
            yield _id, chunk


class ImageDataset(MediaDataset):

    def __init__(
        self,
        input_files: Union[List[Path], Dict[str, Path]],
        *,
        preprocessing_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):

        stream_opts = [
            BasicVideoStreamOutputOptions(frames_per_chunk=1, frame_rate=None)
        ]
        transforms = [preprocessing_function]
        super(ImageDataset, self).__init__(
            input_files=input_files,
            output_stream_opts=stream_opts,
            transforms=transforms,
            offset=None,
        )

    def __iter__(self):
        _iter = super(ImageDataset, self).__iter__()
        for _id, (chunk,) in _iter:
            yield _id, chunk


class AVDataset(MediaDataset):

    def __init__(
        self,
        input_files: Union[List[Path], Dict[str, Path]],
        video_frames_per_chunk: int,
        audio_samples_per_chunk: int,
        *,
        audio_preprocessing_function: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        video_preprocessing_function: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        video_frame_rate: Optional[int] = None,
        audio_sample_rate: Optional[int] = None,
        offset: Optional[float] = None,
    ):

        stream_opts = [
            BasicVideoStreamOutputOptions(
                frames_per_chunk=video_frames_per_chunk,
                frame_rate=video_frame_rate,
            ),
            BasicAudioStreamOutputOptions(
                frames_per_chunk=audio_samples_per_chunk,
                sample_rate=audio_sample_rate,
            ),
        ]
        transforms = [
            (
                video_preprocessing_function
                if video_preprocessing_function is not None
                else IdentityTransform
            ),
            (
                audio_preprocessing_function
                if audio_preprocessing_function is not None
                else IdentityTransform
            ),
        ]
        super(AVDataset, self).__init__(
            input_files=input_files,
            output_stream_opts=stream_opts,
            transforms=transforms,
            offset=offset,
        )

    def __iter__(self):
        _iter = super(AVDataset, self).__iter__()
        for _id, (video_chunk, audio_chunk) in _iter:
            yield _id, video_chunk, audio_chunk
