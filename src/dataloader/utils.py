import hashlib
from pathlib import Path
from PIL import Image
import mimetypes


def md5(path: str):
    file_hash = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(4096):
            file_hash.update(chunk)

    return file_hash.hexdigest()


VIDEO_EXTENSIONS = ("mp4", "webm", "mkv")


def is_valid_image(p: Path):
    try:
        with Image.open(p) as im:
            im.verify()
            return True
    except Exception:
        return False


def is_valid_video(p: Path):
    return mimetypes.guess_type(p)[0].startswith('video')


Identity = lambda *args, **kwargs: (args, kwargs)
NoOp = lambda *args, **kwargs: None
