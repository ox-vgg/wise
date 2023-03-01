import enum
import io
import typing
from typing import (
    List,
    Literal,
    Optional,
    Generator,
    Tuple,
    cast,
    Callable,
    Iterator,
    Any,
)
from base64 import b64encode
from contextlib import contextmanager
from pathlib import Path
import tarfile
import numpy as np
import h5py
from PIL import Image, IptcImagePlugin
from tqdm import tqdm
import webdataset as wds
from .utils import argsort
from .data_models import ImageInfo, ImageMetadata


class EmptyDatasetException(Exception):
    pass


def is_valid_image(p: Path):

    try:
        with Image.open(p) as im:
            im.verify()
            return True
    except Exception:
        return False


def is_valid_webdataset_source(p: str):
    try:
        next(iter(wds.WebDataset(p).to_tuple("__key__", "jpg;jpeg", "json")))
        return True
    except Exception as e:
        print(f"{p} is not a valid data source - {e}")
        return False


def get_valid_webdataset_tar_from_folder(folder: Path):
    return (
        str(x)
        for x in folder.rglob("*")
        if x.is_file() and tarfile.is_tarfile(x) and is_valid_webdataset_source(str(x))
    )


# Create iterator over images in folder
def get_valid_images_from_folder(folder: Path):
    image_files = (x for x in folder.rglob("*") if x.is_file() and is_valid_image(x))
    for image in image_files:
        with Image.open(image) as im:
            w, h = im.size
            format = im.format or "UNKNOWN"
            # Load the image into memory to prevent seeking after generator ends
            im.load()
            metadata = ImageMetadata(
                path=str(image.relative_to(folder)),
                size_in_bytes=image.stat().st_size,
                format=format,
                width=w,
                height=h,
                source_uri=image.absolute().as_uri(),
                metadata={},
            )
            yield im, metadata


# Create iterator over images in webdataset
def get_valid_images_from_webdataset(url: str):
    ds = wds.WebDataset(url).decode("pil")
    for sample in ds:
        k = sample["__key__"]
        im_key = next((a for a in ["jpg", "jpeg"] if a in sample), None)
        if im_key is None:
            continue

        im: Image.Image = sample[im_key]
        w, h = im.size
        im.load()
        metadata = sample.get("json", {})
        yield im, ImageMetadata(
            path=f"{url}#{k}.{im_key}",
            size_in_bytes=-1,
            format="jpeg",
            width=w,
            height=h,
            source_uri=metadata.get("url", f"{url}#{k}.{im_key}"),
            metadata={},
        )


def does_hdf5_file_exists(path: Path, **kwargs) -> bool:
    try:
        with _get_features_dataset(path, mode="r", **kwargs) as _:
            return True
    except FileNotFoundError as e:
        print(f"FileNotFound {path} - {e}")
        return False


class H5_ENCODING(str, enum.Enum):
    ARRAY = "array"
    BYTES = "ascii"
    STRING = "utf-8"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        return H5_ENCODING.ARRAY


def _get_h5_dataset(f: h5py.File, key, **kwargs):
    if not f.mode == "r":
        # Write mode
        if key not in f:
            f.create_dataset(key, **kwargs)
    return typing.cast(h5py.Dataset, f[key])


@contextmanager
def _get_thumbs_dataset(path: Path, *, mode: Literal["r", "w", "a"] = "r", **kwargs):
    with h5py.File(path, mode=mode, **kwargs) as f:
        # Get h5 dataset with key in read mode, create if not exists in write mode
        # Parameters are used only in write mode, no effect in read mode
        yield _get_h5_dataset(
            f,
            "thumbnails",
            shape=(0,),
            dtype=h5py.vlen_dtype(np.dtype("uint8")),
            maxshape=(None,),
        )


@contextmanager
def _get_features_dataset(
    path: Path,
    *,
    mode: Literal["r", "w", "a"] = "r",
    n_dim: Optional[int] = None,
    **kwargs,
) -> Generator[Tuple[h5py.Dataset, h5py.Dataset], None, None]:
    with h5py.File(path, mode, **kwargs) as f:
        if mode != "r":
            # Write enabled
            if n_dim is None:
                raise ValueError("n_dim cannot be None in write mode")

        yield _get_h5_dataset(
            f, "features", shape=(0, n_dim), maxshape=(None, n_dim)
        ), _get_h5_dataset(
            f,
            "files",
            shape=(0,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            maxshape=(None,),
        )


def _append_array_to_dataset(dataset: h5py.Dataset, arr: np.ndarray):
    shape_diff = len(dataset.shape) - len(arr.shape)
    if shape_diff > 1:
        raise ValueError(
            f"Unsupported array shape - must be {dataset.shape[1:]} or {('N',) + dataset.shape[1:]}"
        )

    iarr = arr
    if shape_diff == 1:
        iarr = np.expand_dims(arr, axis=0)

    n = dataset.shape[0]
    b = iarr.shape[0]

    # Resize
    dataset.resize(n + b, axis=0)

    # Write
    dataset[-b:, ...] = iarr


def _dataset_iterator(ds: h5py.Dataset, batch_size: int = 1024):
    data_len = ds.len()
    dtype = H5_ENCODING(ds.dtype).name

    s_idx, e_idx = 0, 0
    while s_idx < data_len:
        e_idx = s_idx + batch_size

        if dtype == H5_ENCODING.STRING:
            yield [x.decode("utf-8") for x in ds[s_idx:e_idx]]
        elif dtype == H5_ENCODING.BYTES:
            yield ds[s_idx:e_idx, ...].tolist()
        else:
            yield ds[s_idx:e_idx, ...]

        s_idx = e_idx


def get_dataset_reader(
    dataset: Path,
    *,
    batch_size: int = 1024,
    **kwargs,
) -> Tuple[str, int, Callable[[], Generator[Tuple[np.ndarray, List[str]], None, None]]]:
    def _reader():
        with _get_features_dataset(dataset, mode="r", **kwargs) as (ds, fs):
            features_iterator = typing.cast(
                Generator[np.ndarray, None, None],
                _dataset_iterator(ds, batch_size=batch_size),
            )
            file_ids_iterator = typing.cast(
                Generator[List[str], None, None],
                _dataset_iterator(fs, batch_size=batch_size),
            )
            yield from zip(features_iterator, file_ids_iterator)

    with _get_features_dataset(dataset, mode="r", **kwargs) as (ds, _):
        model_name: str = cast(str, ds.attrs["model"])
        return model_name, ds.len(), _reader


def generate_thumbnail(im: Image.Image):
    """
    Generate thumbnail image and returns bytes
    Modifies Image in-place, make sure to pass copy if needed
    """
    with io.BytesIO() as buf:
        im.thumbnail((192, 192), resample=Image.BILINEAR)
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


BASE64JPEGPREFIX = b"data:image/jpeg;charset=utf-8;base64,"


@contextmanager
def get_thumbs_reader(dataset: Path, **kwargs):

    with _get_thumbs_dataset(dataset, mode="r", **kwargs) as ds:

        def reader(indices: List[int]):
            sort_indices = argsort(indices)
            sorted_indices = [indices[i] for i in sort_indices]
            unsort_indices = argsort(sort_indices)

            # Fetch thumbnails from hdf5 dataset
            thumbnails_arr: List[List[int]] = ds[sorted_indices, ...].tolist()

            # Unsort the result + convert to base64 -> this will now be in original indices order
            return [
                (BASE64JPEGPREFIX + b64encode(bytes(thumbnails_arr[i]))).decode("utf-8")
                for i in unsort_indices
            ]

        yield reader


@contextmanager
def get_thumbs_writer(
    dataset: Path,
    *,
    mode: Literal["w", "a"] = "w",
    **kwargs,
):
    with _get_thumbs_dataset(dataset, mode=mode, **kwargs) as ts:

        def writer(images: List[Image.Image]):
            # Generate thumbnail and write to h5
            _append_array_to_dataset(
                ts,
                np.array(
                    [
                        np.frombuffer(generate_thumbnail(im), dtype=np.uint8)
                        for im in images
                    ],
                    dtype=object,
                ),
            )

        yield writer


def write_dataset(
    dataset: Path,
    inputs: Generator[Tuple[np.ndarray, List[str]], None, None],
    model_name: str,
    *,
    mode: Literal["w", "a"] = "w",
    write_size: int = 1024,
    **kwargs,
):
    """
    Features is a 2D array of shape (len(file_ids), n_dim)
    """

    # Read first array from generator to get shape
    _input = next(inputs, None)
    if _input is None:
        # Nothing to write
        raise EmptyDatasetException()

    arr, file_ids = _input
    n_dim = arr.shape[-1]

    with _get_features_dataset(dataset, n_dim=n_dim, mode=mode, **kwargs) as (
        ds,
        fs,
    ), tqdm() as pbar:
        if ds.shape[-1] != n_dim:
            raise ValueError("Feature dimension mismatch")

        if ds.attrs.get("model", model_name) != model_name:
            raise ValueError("Model name mismatch")

        ds.attrs["model"] = model_name

        # Write remaining arrays
        buf_arr = arr
        buf_file_ids = file_ids
        for _arr, _file_ids in inputs:
            if buf_arr.shape[0] > write_size:
                _append_array_to_dataset(ds, buf_arr)
                _append_array_to_dataset(fs, np.array(buf_file_ids))
                pbar.update(buf_arr.shape[0])

                buf_arr = _arr
                buf_file_ids = _file_ids
            else:
                buf_arr = np.concatenate((buf_arr, _arr), axis=0)
                buf_file_ids.extend(_file_ids)

        _append_array_to_dataset(ds, buf_arr)
        _append_array_to_dataset(fs, np.array(buf_file_ids))
        pbar.update(buf_arr.shape[0])

        print(f"Done - wrote features (for {model_name}) with shape: {ds.shape}")

    pass


ENCODING_MAP = {
    "CP_1252": "cp1252",
    "CP_UTF8": "utf-8",
}


def get_image_info(p: Path, basedir: Optional[Path] = None):
    # See https://gist.github.com/bhaskarkc/abcbc4a35229815bd6ce4ab7372748f9

    with Image.open(p) as im:
        w, h = im.size
        iminfo = ImageInfo(
            filename=str(p.relative_to(basedir) if basedir else p),
            width=w,
            height=h,
            title=p.name,
        )

        # Add iptc
        iptc = IptcImagePlugin.getiptcinfo(im)
        if not iptc:
            return iminfo

        encoding = (iptc.get((2, 183)) or b"utf-8").decode()

        encoding = ENCODING_MAP.get(encoding, "utf-8")

        iminfo.title = (iptc.get((2, 85)) or iminfo.title.encode()).decode(encoding)
        iminfo.copyright = (iptc.get((2, 116)) or b"").decode(encoding)
        iminfo.caption = (iptc.get((2, 120)) or b"").decode(encoding)

        return iminfo
