import hashlib
from pathlib import Path


def md5(path: str):
    file_hash = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(4096):
            file_hash.update(chunk)

    return file_hash.hexdigest()


Identity = lambda *args, **kwargs: (args, kwargs)
NoOp = lambda *args, **kwargs: None
