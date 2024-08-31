import enum
import hashlib
from pathlib import Path
import logging

import magic

logger = logging.getLogger(__name__)

class MediaMimetype(str, enum.Enum):
    image = 'image'
    video = 'video'
    audio = 'audio'
    unknown = 'unknown'

def md5(path: str):
    file_hash = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(4096):
            file_hash.update(chunk)

    return file_hash.hexdigest()

def get_mime_type(p: Path):
    return magic.from_file(p, mime=True)

def get_media_type_from_mimetype(mimetype: str) -> MediaMimetype:
    if mimetype.startswith('image/'):
        return MediaMimetype.image

    if mimetype.startswith('audio/'):
        return MediaMimetype.audio

    if mimetype.startswith('video/'):
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
