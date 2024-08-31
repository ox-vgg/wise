import itertools
import logging
import json
from pathlib import Path
from uuid import uuid4, UUID
from typing import List, Dict, Callable, Optional, Union, Generator, Tuple, Any, overload, Literal
from .streamreader import (
    SourceMediaType,
    MediaChunkType,
    StreamOutputOptions,
    BasicImageStreamOutputOptions,
    BasicAudioStreamOutputOptions,
    BasicVideoStreamOutputOptions,
    BasicThumbnailStreamOutputOptions,
    get_media_type,
    get_media_info,
    get_stream_duration,
    get_stream_reader,
    get_media_chunk_type,
)
from .utils import get_mimetype_and_media_type_for_file, md5, MediaMimetype, get_media_type_from_mimetype, get_mime_type
from pydantic import dataclasses, ConfigDict
import torch
import torchvision as tv
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

@dataclasses.dataclass
class DatasetPayload(object):
    id: Any
    path: str
    media_type: SourceMediaType

def get_media_metadata(url: str, media_type_from_mimetype: MediaMimetype = None, mimetype: str = None):
    # TODO: Update the code to handle remote path
    # Only md5sum will be a problem.

    # TODO: How to handle subtitle

    if not media_type_from_mimetype or not mimetype:
        media_type_from_mimetype, mimetype, _ = get_mimetype_and_media_type_for_file(url)

    # Get stream metadata
    video_stream_info, audio_stream_info = get_media_info(
        url, guess_missing_video_info=True
    )

    # Get media type
    media_type = get_media_type(video_stream_info, audio_stream_info, media_type_from_mimetype)

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


def JpegTransform(x: torch.Tensor):
    return [tv.io.encode_jpeg(t, quality=80) for t in x]


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MediaChunk:
    tensor: torch.Tensor | list[torch.Tensor]
    pts: float

def get_segment_lengths(stream, stream_opts):
    segment_lengths = []
    for i, opts in enumerate(stream_opts):
        output_stream_info = stream.get_out_stream_info(i)
        frames_per_chunk = opts.frames_per_chunk
        rate = (
            output_stream_info.frame_rate
            if isinstance(opts, BasicVideoStreamOutputOptions)
            else output_stream_info.sample_rate
        )
        if rate is None:
            # Need to use the input media info to calculate the rate
            raise NotImplementedError()

        if rate == 0:
            segment_lengths.append(0)
        else:
            segment_lengths.append(frames_per_chunk / rate)
    return segment_lengths


def validate_segment_lengths_from_options(stream_opts: List[StreamOutputOptions]):
    # Each stream should be aligned, or we will just miss one or the other.
    segment_length = None
    for opts in stream_opts:
        frames = opts.frames_per_chunk
        rate = (
            opts.frame_rate
            if isinstance(opts, (BasicVideoStreamOutputOptions, BasicImageStreamOutputOptions))
            else opts.sample_rate
        )

        if rate is None:
            # Need stream
            rate = 0

        if rate == 0:
            _segment_length = 0
        else:
            _segment_length = frames / rate

        if segment_length == None:
            segment_length = _segment_length
        elif abs(segment_length - _segment_length) > 1e-2:
            raise ValueError(
                "Output streams have different rates configured. Will result in missing data!"
            )
    return segment_length


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
        input_files: Union[List[str], Dict[str, str]],
        output_stream_opts=List[StreamOutputOptions],
        transforms: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        offset: Optional[float] = None,
        thumbnails: bool = True,
    ):
        super(MediaDataset).__init__()

        self._filelist: Dict[str | int, str] = (
            input_files
            if isinstance(input_files, dict)
            else dict(enumerate(input_files))
        )
        self._transforms = (
            transforms
            if transforms is not None
            else [IdentityTransform for _ in range(len(self._output_stream_opts))]
        )

        self._segment_length = validate_segment_lengths_from_options(output_stream_opts)
        if self._segment_length is None:
            # No output stream configured, default to 4s for the sake of thumbnails
            self._segment_length = 4

        # Handle thumbnail for video and image
        # TODO Works only when the input streams are synced with the thumbnail stream
        assert self._segment_length % 0.5 == 0
        self._thumbnails = thumbnails
        self._thumbnail_opts = BasicThumbnailStreamOutputOptions(
            frames_per_chunk=self._segment_length * 2 if self._segment_length else 1,
            frame_rate=2 if self._segment_length else None,
            width=-2,
            height=192,
        )

        self._output_stream_opts = output_stream_opts
        self._offset = offset or 0.0
        # verify if length of output_stream_opts and transform matches up
        assert len(self._transforms) == len(self._output_stream_opts)

    def _get_media_iterator(self, id_list: List[Union[str, int]]) -> Generator[Tuple[str | int, Dict[MediaChunkType, MediaChunk | None]], Any, None]:
        for _id in id_list:
            path = self._filelist[_id]
            try:
                stream_transforms = list(self._transforms)
                output_stream_opts = list(self._output_stream_opts)
                if self._thumbnails:
                    logger.debug("Adding thumbnails stream")
                    output_stream_opts.append(self._thumbnail_opts)
                    stream_transforms.append(JpegTransform)

                # Read the frames from starting offset
                reader = get_stream_reader(path, output_stream_opts)
                # reader.seek(self._offset)


                media_chunk_types = [get_media_chunk_type(opts) for opts in output_stream_opts]

                for c in reader.stream():
                    # Might contain 1 or many output streams. Apply the corresponding transform
                    media_chunks = {
                        media_chunk_type: (
                            MediaChunk(
                                tensor=stream_transform(torch.Tensor(stream_chunk)),
                                pts=stream_chunk.pts,
                            )
                            if stream_chunk is not None
                            else None
                        )
                        for (stream_chunk, stream_transform, media_chunk_type) in zip(
                            c, stream_transforms, media_chunk_types
                        )
                    }
                    yield _id, media_chunks
            except Exception:
                logger.exception(f'Exception when processing "{_id}: {path}"')

    def __iter__(self) -> Generator[Tuple[str | int, Dict[MediaChunkType, MediaChunk | None]], Any, None]:
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
            return self._get_media_iterator(list(self._filelist.keys()))

        print(worker_info)
        n_workers = worker_info.num_workers
        worker_id = worker_info.id

        # Split files by worker and yield the worker's shard
        # Each worker will process the video file with index in (w_id, w_id + n, w_id + 2n, ...)
        file_list = itertools.islice(self._filelist.keys(), worker_id, None, n_workers)

        return self._get_media_iterator(list(file_list))


class AudioDataset(MediaDataset):

    def __init__(
        self,
        input_files: Union[List[str], Dict[str, str]],
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
            thumbnails=False,
        )


class VideoDataset(MediaDataset):

    def __init__(
        self,
        input_files: Union[List[str], Dict[str, str]],
        frames_per_chunk: int,
        *,
        preprocessing_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        frame_rate: Optional[int] = None,
        offset: Optional[float] = None,
        thumbnails: bool = True,
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
            thumbnails=thumbnails,
        )


class ImageDataset(MediaDataset):

    def __init__(
        self,
        input_files: Union[List[str], Dict[str, str]],
        *,
        preprocessing_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        thumbnails: bool = True,
    ):

        stream_opts = [
            BasicImageStreamOutputOptions(frames_per_chunk=1)
        ]
        transforms = [preprocessing_function]
        super(ImageDataset, self).__init__(
            input_files=input_files,
            output_stream_opts=stream_opts,
            transforms=transforms,
            offset=None,
            thumbnails=thumbnails,
        )


class AVDataset(MediaDataset):

    def __init__(
        self,
        input_files: Union[List[str], Dict[str, str]],
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
        thumbnails: bool = True,
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
            thumbnails=thumbnails,
        )


def get_metadata_for_valid_files(paths: list[Path]):
    """
    Given a list of file paths
    - filter files with media mimetypes (image/*, video/*, audio/*)
    - get metadata for each file if possible
    - return a list of metadata and list of unknown_files

    TODO: Accept URLs, filebuffers in the future
    """
    # get the mimetypes and media types for each file
    media_files = [get_mimetype_and_media_type_for_file(x) for x in paths]
    # separate the files with an unknown MIME type
    unknown_files = [p for (_, media_type, p) in media_files if media_type == MediaMimetype.unknown]
    known_files = [(mimetype, media_type, p) for (mimetype, media_type, p) in media_files if media_type != MediaMimetype.unknown]

    media_metadata: list[MediaMetadata] = []
    # for each file, try to open the file and get its metadata
    # skip the ones that fail
    for mimetype, media_type, p in known_files:
        try:
            metadata = get_media_metadata(str(p), media_type, mimetype)
            media_metadata.append(metadata)
        except Exception:
            logger.exception(f'Exception while reading file - {p}, skipping')
            unknown_files.append(p)

    return media_metadata, unknown_files

@overload
def _get_dataset(
    input_files: List[str] | Dict[str, str],
    media_type: Literal[SourceMediaType.AV],
    *,
    video_frames_per_chunk: int,
    audio_samples_per_chunk: int,
    video_frame_rate: int | None = None,
    audio_sampling_rate: int | None = None,
    video_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    audio_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    image_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    offset: float | None,
    thumbnails: bool = True,
) -> MediaDataset: ...

@overload
def _get_dataset(
    input_files: List[str] | Dict[str, str],
    media_type: Literal[SourceMediaType.VIDEO],
    *,
    video_frames_per_chunk: int,
    audio_samples_per_chunk: int = -1,
    video_frame_rate: int | None = None,
    audio_sampling_rate: int | None = None,
    video_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    audio_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    image_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    offset: float | None,
    thumbnails: bool = True,
) -> MediaDataset: ...

@overload
def _get_dataset(
    input_files: List[str] | Dict[str, str],
    media_type: Literal[SourceMediaType.AUDIO],
    *,
    audio_samples_per_chunk: int,
    video_frames_per_chunk: int = -1,
    video_frame_rate: int | None = None,
    audio_sampling_rate: int | None = None,
    video_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    audio_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    image_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    offset: float | None,
    thumbnails: bool = True,
) -> MediaDataset: ...

@overload
def _get_dataset(
    input_files: List[str] | Dict[str, str],
    media_type: Literal[SourceMediaType.IMAGE],
    *,
    audio_samples_per_chunk: int = -1,
    video_frames_per_chunk: int = -1,
    video_frame_rate: int | None = None,
    audio_sampling_rate: int | None = None,
    video_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    audio_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    image_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    offset: float | None,
    thumbnails: bool = True,
) -> MediaDataset: ...

def _get_dataset(
        input_files: List[str] | Dict[str, str],
        media_type: SourceMediaType,
        video_frames_per_chunk: int,
        audio_samples_per_chunk: int,
        video_frame_rate: int | None = None,
        audio_sampling_rate: int | None = None,
        video_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        audio_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        image_preprocessing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        offset: float | None = None,
        thumbnails: bool = True ):
    if media_type == SourceMediaType.AV:
        stream = AVDataset(
            input_files,
            video_frames_per_chunk=video_frames_per_chunk,
            video_frame_rate=video_frame_rate,
            video_preprocessing_function=video_preprocessing_function,
            audio_samples_per_chunk=audio_samples_per_chunk,
            audio_sample_rate=audio_sampling_rate,
            audio_preprocessing_function=audio_preprocessing_function,
            offset=offset,
            thumbnails=thumbnails,
        )
    elif media_type == SourceMediaType.VIDEO:
        stream = VideoDataset(
            input_files,
            frames_per_chunk=video_frames_per_chunk,
            preprocessing_function=video_preprocessing_function,
            frame_rate=video_frame_rate,
            offset=offset,
            thumbnails=thumbnails
        )

    elif media_type == SourceMediaType.AUDIO:
        stream = AudioDataset(
            input_files,
            samples_per_chunk=audio_samples_per_chunk,
            sample_rate=audio_sampling_rate,
            preprocessing_function=audio_preprocessing_function,
            offset=offset,
        )
    elif media_type == SourceMediaType.IMAGE:
        stream = ImageDataset(
            input_files,
            preprocessing_function=image_preprocessing_function,
            thumbnails=thumbnails
        )
    else:
        raise ValueError(f'Unknown media_type: {media_type}')
    return stream

def get_dataset(media_metadata: list[DatasetPayload], params: Dict[str, Any]):
    # sort and group (by media_type - image/video/audio/av)
    sort_func = lambda x: x.media_type
    sorted_metadata = sorted(media_metadata, key=sort_func)
    datasets = [
        _get_dataset({x.id: x.path for x in g}, k, **params)
        for k, g in itertools.groupby(sorted_metadata, key=sort_func)
    ]
    return datasets


def is_valid_media_file(p: Path):
    """
    Quicker, but non-exhaustive check.
    Can find if the streamreader recognizes the file, but doesn't ensure it can be iterated over
    """
    media_type = get_media_type_from_mimetype(get_mime_type(p))
    if media_type == MediaMimetype.unknown:
        return False

    try:
        get_media_info(str(p))
        return True
    except Exception:
        logger.warning(f'Skipping invalid video file: {p}')
        return False
