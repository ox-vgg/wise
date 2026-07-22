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

import hashlib
import logging
from enum import Enum
from pathlib import Path

import filetype
import magic

logger = logging.getLogger(__name__)

MATCHERS = {
    x.MIME: x
    for x in (
        filetype.image_matchers
        + filetype.video_matchers
        + filetype.audio_matchers
    )
}

SEEN_EXTENSIONS = {x.EXTENSION: x.MIME for x in MATCHERS.values()}


class MediaMimetype(Enum):
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
        logger.debug("Adding '%s' for '%s' to known list", extension, mimetype)
        return mimetype

    # Fallback to python-magic
    logger.debug("Falling back to magic...")
    mimetype = magic.from_file(p, mime=True)
    logger.debug("Mime for '%s' from magic: '%s'", p, mimetype)
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
