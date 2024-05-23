from datetime import datetime, timedelta
import enum
import logging
import math
from typing import List, Optional, Literal, TypeVar

from pydantic import dataclasses
from dataclasses import asdict
from torchaudio.io import StreamReader

logger = logging.getLogger(__name__)


class SourceMediaType(str, enum.Enum):
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    AV = "av"


@dataclasses.dataclass
class BaseStreamOutputOptions(object):
    frames_per_chunk: int
    buffer_chunk_size: int = 3
    stream_index: Optional[int] = None


@dataclasses.dataclass
class BasicVideoStreamOutputOptions(BaseStreamOutputOptions):
    format: Optional[Literal["rgb24", "bgr24", "yuv420p", "gray"]] = "rgb24"
    frame_rate: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

    # TODO
    # Add decoder and decoder_option, hw_accel
    # https://pytorch.org/audio/stable/generated/torio.io.StreamingMediaDecoder.html#add-basic-video-stream


@dataclasses.dataclass
class BasicAudioStreamOutputOptions(BaseStreamOutputOptions):
    format: Optional[Literal["u8p", "s16p", "s32p", "s64p", "fltp", "dblp"]] = "fltp"
    sample_rate: Optional[int] = None

    # TODO
    # Add decoder and decoder_option, hw_accel
    # https://pytorch.org/audio/stable/generated/torio.io.StreamingMediaDecoder.html#add-basic-audio-stream


StreamOutputOptions = TypeVar("StreamOutputOptions", bound=BaseStreamOutputOptions)

# Subset of `ffmpeg -decoders` output
IMAGE_CODECS = (
    "bmp",
    "exr",
    "hdr",
    "jpeg2000",
    "jpegls",
    "pbm",
    "pgm",
    "png",
    "ppm",
    "tiff",
    "webp",
)


def convert_duration_string_to_seconds(duration_str: Optional[str]):
    """
    Function to convert duration present in video metadata into seconds

    Expected Behaviour
    None -> 0 (float)
    HH:MM:SS.[ssssss] (str) -> T.ssssss seconds (float)
    T.[ssssss] (str) -> T.ssssss seconds (float)
    """
    if duration_str is None:
        return 0

    # TODO the duration string may be in a random format

    # The duration string may be millisecond, microsecond, nanosecond or may not have it at all?
    duration_second, subsecond = str(duration_str).split(".", 1)

    # Pad right to min len 6 and truncate to handle the differences
    microsecond = f"{subsecond:<06}"[:6]

    if ":" in str(duration_str):
        time_obj = datetime.strptime(f"{duration_second}.{microsecond}", "%H:%M:%S.%f")
    else:
        time_obj = (
            datetime.min
            + timedelta(seconds=float(duration_second), microseconds=float(microsecond))
        ).time()

    return (
        time_obj.microsecond / 1e6
        + time_obj.second
        + 60 * (time_obj.minute + (60 * time_obj.hour))
    )


def _get_stream_duration(stream_info):
    """
    Looking for duration metadata in the source stream
    """
    return stream_info.metadata.get(
        "DURATION", stream_info.metadata.get("duration", "0.0")
    )


get_stream_duration = lambda x: convert_duration_string_to_seconds(
    _get_stream_duration(x)
)


def _update_video_info(streamer: StreamReader, video_stream_info):
    """
    Internally used to figure out fps, n_images, and duration if not present in the metadata

    DO NOT try to re-use the streamer object - might have reached end of stream after this method
    TODO: Verify if the metadata actually matches the n_images
    """
    duration_str = get_stream_duration(video_stream_info)
    duration = convert_duration_string_to_seconds(duration_str)

    n_images: Optional[int] = video_stream_info.num_frames or None
    fps: Optional[float] = video_stream_info.frame_rate

    logger.debug(
        f"Initial values - FPS: {fps}, duration: {duration}, n_images: {n_images}"
    )
    if n_images is None:
        logger.debug("Guessing n_images")
        # Guess from duration and fps
        streamer.add_basic_video_stream(1)

        if duration != 0 and fps is not None:
            n_images = math.floor(duration * fps)

            # Use the guess to seek and find out actual n_images
            streamer.seek(math.floor(duration))
            for _ in streamer.stream():
                n_images += 1

        else:
            # No choice but to scan entire video (or do a expand - contract search)
            n_images = 0
            for _ in streamer.stream():
                n_images += 1

        logger.debug(f"Num images: {n_images}")

    if not fps:
        # Guess from n_images and duration
        if duration:
            fps = n_images / duration
        logger.debug(f"FPS: {fps}")

    if not duration:
        # Guess from fps and n_images
        if fps:
            duration = n_images / fps
        logger.debug(f"Duration: {duration}")

    video_stream_info.frame_rate = fps
    video_stream_info.num_frames = n_images
    video_stream_info.metadata["DURATION"] = duration
    return video_stream_info


def get_media_info(url: str, guess_missing_video_info: bool = False):
    # Load stream reader with path
    streamer = StreamReader(url)

    # Get media metadata (we will only check the default stream for video and audio)
    video_stream = streamer.default_video_stream
    audio_stream = streamer.default_audio_stream

    video_stream_info = (
        streamer.get_src_stream_info(video_stream) if video_stream is not None else None
    )
    audio_stream_info = (
        streamer.get_src_stream_info(audio_stream) if audio_stream is not None else None
    )
    if video_stream_info and guess_missing_video_info:
        video_stream_info = _update_video_info(streamer, video_stream_info)

    return video_stream_info, audio_stream_info


def get_stream_reader(url: str, output_stream_opts: List[BaseStreamOutputOptions] = []):
    streamer = StreamReader(url)

    logger.debug(f"StreamReader: (metadata) {streamer.get_metadata()}")
    logger.debug(f"StreamReader: {url} contains {streamer.num_src_streams} streams")
    for i in range(streamer.num_src_streams):
        logger.debug(f"{i}: {streamer.get_src_stream_info(i)}")

    for opts in output_stream_opts:
        # Add output video stream
        if not isinstance(
            opts, (BasicVideoStreamOutputOptions, BasicAudioStreamOutputOptions)
        ):
            raise TypeError(f"Unknown stream options type - {type(output_stream_opts)}")

        _stream_opts = asdict(opts)

        if isinstance(opts, BasicVideoStreamOutputOptions):
            streamer.add_basic_video_stream(**_stream_opts)
        else:
            streamer.add_basic_audio_stream(**_stream_opts)

    logger.debug(f"StreamReader: {streamer.num_out_streams} output stream(s)")
    for i in range(streamer.num_out_streams):
        logger.debug(f"Output stream {i}: {streamer.get_out_stream_info(i)}")

    return streamer


def get_media_type(video_stream_info, audio_stream_info):
    """
    Based on the default video / audio stream info, infer whether the source file is either
        - IMAGE
        - VIDEO Only
        - AUDIO Only
        - A/V
    """
    # Raise error if no stream in media (might be redundant, torchaudio might complain already)
    if video_stream_info is None and audio_stream_info is None:
        raise ValueError(f"No media streams found!")

    # Must be one of image, audio-only, video-only or av
    elif audio_stream_info is None:
        # Either video-only or image
        if video_stream_info.codec in IMAGE_CODECS:
            # image
            # TODO check for iptc, exif and other metadata
            return SourceMediaType.IMAGE
        else:
            return SourceMediaType.VIDEO
    elif video_stream_info is None:
        return SourceMediaType.AUDIO
    else:
        return SourceMediaType.AV
