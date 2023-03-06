from base64 import b64encode
import itertools


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def batched(iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch


BASE64JPEGPREFIX = b"data:image/jpeg;charset=utf-8;base64,"

convert_uint8array_to_base64 = lambda x: (
    BASE64JPEGPREFIX + b64encode(bytes(x))
).decode("utf-8")
