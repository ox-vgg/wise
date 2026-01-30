#!/usr/bin/env python3

import enum
import hashlib
from pathlib import Path
import logging

import filetype
import magic

logger = logging.getLogger(__name__)

MATCHERS = {
    x.MIME: x
    for x in (
        filetype.image_matchers + filetype.video_matchers + filetype.audio_matchers
    )
}

SEEN_EXTENSIONS = {x.EXTENSION: x.MIME for x in MATCHERS.values()}


class MediaMimetype(str, enum.Enum):
    image = "image"
    video = "video"
    audio = "audio"
    unknown = "unknown"


def md5(path: str):
    file_hash = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(4096):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def get_mime_type(p: Path):
    """
    TODO: Very minor optimisation - prioritise the matchers list based on previously seen values?
    """
    # Must be a file!
    # First check based on extension (only last part of extension, no .tar.gz type ones!)
    extension = p.suffix.lstrip(".")
    guessed_matcher = MATCHERS.get(
        SEEN_EXTENSIONS.get(extension, "unknown"),
        None,
    )
    if guessed_matcher is not None and filetype.match(p, (guessed_matcher,)):
        mimetype = guessed_matcher.mime
        SEEN_EXTENSIONS[extension] = mimetype
        return mimetype

    # If not present, try checking all media matchers
    m = filetype.match(p, MATCHERS.values())

    if m is not None:
        mimetype = m.mime
        # extension possibly unseen, keep it for next time
        SEEN_EXTENSIONS[extension] = mimetype
        logger.debug(f"Adding {extension} for {mimetype} to known list")
        return mimetype

    # Fallback to python-magic
    logger.debug("Falling back to magic...")
    mimetype = magic.from_file(p, mime=True)
    logger.debug(f"Mime for {p} from magic: {mimetype}")
    return mimetype


def get_media_type_from_mimetype(mimetype: str) -> MediaMimetype:
    if mimetype.startswith("image/"):
        return MediaMimetype.image

    if mimetype.startswith("audio/"):
        return MediaMimetype.audio

    if mimetype.startswith("video/"):
        return MediaMimetype.video

    return MediaMimetype.unknown


def get_mimetype_and_media_type_for_file(p: Path):
    # p must be a file. TODO implement a contract check

    mimetype = get_mime_type(p)
    media_type = get_media_type_from_mimetype(mimetype)
    return (mimetype, media_type, p)


def get_files_from_directory_with_extensions(dir: Path, extensions: list[str]):
    return (x for ext in extensions for x in dir.rglob(ext) if x.is_file())


Identity = lambda *args, **kwargs: (args, kwargs)
NoOp = lambda *args, **kwargs: None
