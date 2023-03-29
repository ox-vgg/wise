from contextlib import contextmanager
import enum
from functools import partial
import io
import json
import logging
import typing
from typing import (
    List,
    Literal,
    Optional,
    Union,
    Generator,
    overload,
    Tuple,
    Callable,
    Dict,
    Any,
)

from pathlib import Path
import tarfile
import numpy as np
import h5py
from PIL import Image, IptcImagePlugin
import webdataset as wds
import httpx

from .utils import argsort
from .data_models import ImageInfo, ImageMetadata

logger = logging.getLogger(__name__)


class EmptyDatasetException(Exception):
    pass


class H5Datasets(str, enum.Enum):
    IMAGE_FEATURES = "features/image"
    METADATA_FEATURES = "features/metadata"
    IDS = "ids"
    THUMBNAILS = "thumbnails"


def is_valid_uri(uri: str):
    try:
        with httpx.stream("HEAD", uri, follow_redirects=True) as r:
            r.raise_for_status()
            logger.info(f'"{uri}" - {r.headers}')
            return True
    except Exception as e:
        logger.warning(f'Request to URI "{uri}" failed - {e}')
        return False


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
        logger.warning(f"{p} is not a valid data source - {e}")
        return False


def get_valid_webdataset_tar_from_folder(folder: Path):
    return (
        str(x)
        for x in folder.rglob("*.tar")
        if x.is_file() and tarfile.is_tarfile(x) and is_valid_webdataset_source(str(x))
    )


def get_file_from_tar(location: Path, key: str):
    with tarfile.open(location, "r") as t:
        buf = t.extractfile(key)
        if buf:
            yield from buf
    return None


# Create iterator over images in folder
def get_valid_images_from_folder(
    folder: Path, error_handler: Callable[[Dict[str, Any]], None]
):
    image_files = (x for x in folder.rglob("*") if x.is_file() and is_valid_image(x))
    for image in image_files:
        try:
            with Image.open(image) as im:
                w, h = im.size
                format = im.format or "UNKNOWN"
                # Load the image into memory to prevent seeking after generator ends
                im.load()
                relative_path = image.relative_to(folder)

                metadata = ImageMetadata(
                    path=str(relative_path),
                    size_in_bytes=image.stat().st_size,
                    format=format,
                    width=w,
                    height=h,
                    source_uri=None,
                    metadata={
                        "title": relative_path.stem,
                        "description": relative_path.stem,
                        "author": "",
                        "datetime": "",
                        "license": "",
                    },
                )
                yield im, metadata
        except Exception as e:
            logging.error(f"Error {image} - ({type(e)}){e}")
            _, *suffix_key = image.suffix.split(".", 1)
            image_key = image.relative_to(folder)
            image_key = image_key.parent / image_key.stem
            sample = {
                "__key__": str(image_key),
                "__url__": str(folder),
                f'{".".join(suffix_key)}': image.read_bytes(),
                "json": json.dumps(
                    {
                        "url": image.resolve().as_uri(),
                        "error_message": f"({e.__class__.__name__}) - {e}",
                    }
                ).encode("utf-8"),
            }
            error_handler(sample)


# Create iterator over images in webdataset
def get_valid_images_from_webdataset(url: str, error_handler):
    ds = wds.WebDataset(url)
    for sample in ds:
        k = sample["__key__"]
        # TODO Handle other formats as well
        im_key = next((a for a in ["jpg", "jpeg"] if a in sample), None)
        if im_key is None:
            # Unknown sample - Ignore for now, raise exception later
            continue
        try:
            im_sample = sample[im_key]
            with Image.open(io.BytesIO(im_sample)) as im:
                w, h = im.size
                im.load()
                metadata = json.loads(sample.get("json", b"{}"))
                yield im, ImageMetadata(
                    path=f"{url}#{k}.{im_key}",
                    size_in_bytes=len(im_sample),
                    format="JPEG",
                    width=w,
                    height=h,
                    source_uri=metadata.get("url", None),
                    metadata={
                        "title": metadata.get("title", ""),
                        "description": metadata.get("description", ""),
                        "author": metadata.get("author", ""),
                        "datetime": metadata.get("datetime", ""),
                        "license": metadata.get("license", ""),
                    },
                )
        except Exception as e:
            logging.error(f"Error {url}#{k}.{im_key} - ({e.__class__.__name__}){e}")
            error_handler(sample)


def does_wise_hdf5_exists(path: Path, **kwargs) -> bool:
    try:
        with _get_dataset(path, name=list(H5Datasets), mode="r", **kwargs) as _:
            return True
    except FileNotFoundError as e:
        logger.warning(f"FileNotFound {path} - {e}")
        return False


class H5_ENCODING(str, enum.Enum):
    ARRAY = "array"
    BYTES = "ascii"
    STRING = "utf-8"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        return H5_ENCODING.ARRAY


ENCODING_FNS = {
    H5_ENCODING.ARRAY: lambda x: x,
    H5_ENCODING.STRING: lambda x: np.array(x),
    H5_ENCODING.BYTES: lambda x: np.array(
        [np.frombuffer(im, dtype=np.uint8) for im in x],
        dtype=object,
    ),
}

DECODING_FNS: Dict[
    H5_ENCODING, Callable[[np.ndarray], Union[np.ndarray, List[str], List[bytes]]]
] = {
    H5_ENCODING.ARRAY: lambda arr: arr,
    H5_ENCODING.STRING: lambda arr: [x.decode("utf-8") for x in arr],
    H5_ENCODING.BYTES: lambda arr: arr.tolist(),
}


def _get_h5_dataset(f: h5py.File, key: H5Datasets, **kwargs):
    if not f.mode == "r":
        # Write mode
        if key not in f:
            f.create_dataset(key.value, **kwargs)
    return typing.cast(h5py.Dataset, f[key])


def _get_initial_shape_and_dtype_for_dataset(
    name: H5Datasets, n_dim: Optional[int] = None
):
    shape = (0,)
    maxshape = (None,)

    if name in [H5Datasets.IMAGE_FEATURES, H5Datasets.METADATA_FEATURES]:
        return {"shape": (0, n_dim), "dtype": np.float32, "maxshape": (None, n_dim)}

    if name == H5Datasets.THUMBNAILS:
        return {
            "shape": shape,
            "dtype": h5py.vlen_dtype(np.dtype("uint8")),
            "maxshape": maxshape,
        }

    if name == H5Datasets.IDS:
        return {
            "shape": shape,
            "dtype": h5py.string_dtype(encoding="utf-8"),
            "maxshape": maxshape,
        }

    raise NotImplementedError()


def _is_features_dataset(name: H5Datasets):
    return name in [H5Datasets.IMAGE_FEATURES, H5Datasets.METADATA_FEATURES]


@overload
def _get_dataset(
    path: Path,
    name: H5Datasets,
    *,
    mode: Literal["r", "w", "a"] = "r",
    n_dim: Optional[int] = None,
    **kwargs,
) -> Generator[h5py.Dataset, None, None]:
    ...


@overload
def _get_dataset(
    path: Path,
    name: List[H5Datasets],
    *,
    mode: Literal["r", "w", "a"] = "r",
    n_dim: Optional[int] = None,
    **kwargs,
) -> Generator[Tuple[h5py.Dataset], None, None]:
    ...


@contextmanager
def _get_dataset(
    path: Path,
    name: Union[H5Datasets, List[H5Datasets]],
    *,
    mode: Literal["r", "w", "a"] = "r",
    n_dim: Optional[int] = None,
    **kwargs,
):
    """
    Duplicate name keys will be ignored
    """
    if isinstance(name, list):
        if len(name) == 0:
            raise ValueError("Name of at least one dataset required")

        _names = list(dict.fromkeys(name))
    else:
        _names = [name]

    if mode != "r":
        # Write enabled
        if any(_is_features_dataset(x) for x in _names) and n_dim is None:
            raise ValueError(
                "n_dim cannot be None for image / metadata features in write mode"
            )

    with h5py.File(path, mode=mode, **kwargs) as f:
        if isinstance(name, list):
            yield tuple(
                _get_h5_dataset(
                    f, n, **_get_initial_shape_and_dtype_for_dataset(n, n_dim)
                )
                for n in _names
            )
        else:
            yield _get_h5_dataset(
                f, name, **_get_initial_shape_and_dtype_for_dataset(name, n_dim)
            )


def _append_array_to_dataset(ds: h5py.Dataset, arr: np.ndarray):
    shape_diff = len(ds.shape) - len(arr.shape)
    if shape_diff > 1:
        raise ValueError(
            f"Unsupported array shape - must be {ds.shape[1:]} or {('N',) + ds.shape[1:]}"
        )

    iarr = arr
    if shape_diff == 1:
        iarr = np.expand_dims(arr, axis=0)

    n = ds.shape[0]
    b = iarr.shape[0]

    # Resize
    ds.resize(n + b, axis=0)

    # Write
    ds[-b:, ...] = iarr


def _rollback_dataset_by_nrows(dataset: h5py.Dataset, n_count: int):
    if n_count > dataset.len():
        dataset.resize(n_count, axis=0)


def _get_dataset_type(ds: h5py.Dataset):
    dtype = ds.dtype

    if h5py.check_string_dtype(dtype):
        return H5_ENCODING.STRING
    if h5py.check_vlen_dtype(dtype):
        return H5_ENCODING.BYTES
    return H5_ENCODING.ARRAY


def _dataset_iterator(ds: h5py.Dataset, batch_size: int = 1024):
    data_len = ds.len()
    _decode_fn = DECODING_FNS[_get_dataset_type(ds)]

    s_idx, e_idx = 0, 0
    while s_idx < data_len:
        e_idx = s_idx + batch_size
        yield _decode_fn(ds[s_idx:e_idx, ...])

        s_idx = e_idx


# Public functions


def generate_thumbnail(im: Image.Image):
    """
    Generate thumbnail image and returns bytes
    Modifies Image in-place, make sure to pass copy if needed
    """
    with io.BytesIO() as buf:
        im.thumbnail((192, 192), resample=Image.BILINEAR)
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def get_shapes(dataset: Path, **kwargs):
    _names = list(H5Datasets)
    with _get_dataset(dataset, _names, mode="r", **kwargs) as ds_arr:
        return {x: y.shape for x, y in zip(_names, ds_arr)}


def get_counts(dataset: Path, **kwargs):
    """
    Returns the counts of all datasets as
    dictionary

    TODO: Raises error if the dataset is scalar
    """
    _names = list(H5Datasets)
    with _get_dataset(dataset, _names, mode="r", **kwargs) as ds_arr:
        return {x: y.len() for x, y in zip(_names, ds_arr)}


def get_model_name(dataset: Path, **kwargs):
    """
    Returns the model name stored on the FEATURES datasets
    """
    with _get_dataset(dataset, H5Datasets.IMAGE_FEATURES, mode="r", **kwargs) as ds:
        return ds.attrs.get("model", None)


def get_h5iterator(dataset: Path, *, batch_size: int = 1024, **kwargs):
    """
    Returns a function that yields values from the chosen dataset
    based on the batch size.
    """

    def _reader(name: H5Datasets):
        if name not in H5Datasets:
            raise ValueError(f"Only {[x.name for x in H5Datasets]} are allowed")

        with _get_dataset(dataset, name, mode="r", **kwargs) as ds:
            yield from _dataset_iterator(ds, batch_size=batch_size)

    return _reader


def get_h5reader(dataset: Path, **kwargs):
    """
    Returns a contextmanager that allows you to open a
    dataset and random access rows based on leading index
    """

    def indexed_reader(
        ds: h5py.Dataset,
        decode_fn: Callable[[np.ndarray], Union[np.ndarray, List[str], List[bytes]]],
        indices: List[int],
    ):
        sort_indices = argsort(indices)
        sorted_indices = [indices[i] for i in sort_indices]
        unsort_indices = argsort(sort_indices)

        # Fetch thumbnails from hdf5 dataset
        arr = decode_fn(ds[sorted_indices, ...])

        # Unsort the result + this will now be in original indices order
        return [arr[i] for i in unsort_indices]

    @contextmanager
    def _reader(name: H5Datasets):
        if name not in H5Datasets:
            raise ValueError(f"Only {[x.name for x in H5Datasets]} are allowed")

        with _get_dataset(dataset, name, mode="r", **kwargs) as ds:
            decoding_fn = DECODING_FNS[_get_dataset_type(ds)]
            yield partial(indexed_reader, ds, decoding_fn)

    return _reader


@contextmanager
def get_h5writer(
    dataset: Path,
    names: Union[H5Datasets, List[H5Datasets]],
    model_name: Optional[str] = None,
    n_dim: Optional[int] = None,
    mode: Literal["w", "a"] = "w",
    **kwargs,
):
    """
    yields a writer function that allows you to write
    to the datasets (passed as names) and rollback all
    of them together in case of exception.

    The returned function expects the arrays that are
    written to the datasets to have the same length
    in the outer most dimension, provided in the same order
    as names
    """
    if isinstance(names, list):
        if len(names) == 0:
            raise ValueError("Name of at least one dataset required")

        _names = list(dict.fromkeys(names))
    else:
        _names = [names]

    if not set(H5Datasets).issuperset(set(_names)):
        raise ValueError(f"Only {[x.name for x in H5Datasets]} are allowed")

    is_features_dataset_requested = any(_is_features_dataset(x) for x in _names)
    if is_features_dataset_requested:
        if model_name is None:
            raise ValueError(
                "Model Name must be provided when writing to features dataset"
            )

    with _get_dataset(dataset, _names, mode=mode, n_dim=n_dim, **kwargs) as ds_arr:
        _encoding_fns = list(ENCODING_FNS[_get_dataset_type(_ds)] for _ds in ds_arr)

        num_datasets = len(ds_arr)

        if is_features_dataset_requested:
            all_features_ds = [
                ds_arr[i] for i, x in enumerate(_names) if _is_features_dataset(x)
            ]

            if any(x.shape[-1] != n_dim for x in all_features_ds):
                raise ValueError("Feature dimension mismatch")

            if any(
                x.attrs.get("model", model_name) != model_name for x in all_features_ds
            ):
                raise ValueError("Model name mismatch")

            for _features_ds in all_features_ds:
                _features_ds.attrs["model"] = model_name

        def writer(*args):
            if len(args) != num_datasets:
                raise ValueError(
                    f"Expected {num_datasets} arrays to write, Got {len(args)}"
                )

            current_size = (x.len() for x in ds_arr)
            try:
                for _arr, _ds, _fn in zip(args, ds_arr, _encoding_fns):
                    _append_array_to_dataset(_ds, _fn(_arr))

            except Exception as e:
                for _ds, _nrows in zip(ds_arr, current_size):
                    _rollback_dataset_by_nrows(_ds, _nrows)
                raise e

        yield writer


def concat_h5datasets(sources, output: Path, **kwargs):
    # Find number of rows across datasets
    total = {}
    model_name = None
    features_dim = None

    # Make sure we have the same model name across sources
    for s in sources:
        if model_name is None:
            model_name = get_model_name(s, **kwargs)
        elif model_name != get_model_name(s, **kwargs):
            raise ValueError("Expected model_name to be same in all data sources")

        if features_dim is None:
            features_dim = get_shapes(s, **kwargs).get(
                H5Datasets.IMAGE_FEATURES, (None, None)
            )[1]

        elif (
            features_dim
            != get_shapes(s, **kwargs).get(H5Datasets.IMAGE_FEATURES, (None, None))[1]
        ):
            raise ValueError("All feature arrays in source to have same shape!")

        counts = get_counts(s, **kwargs)
        for k, v in counts.items():
            if k not in H5Datasets:
                continue

            if k not in total:
                total[k] = 0
            total[k] += v

    datasets = list(H5Datasets)
    layouts = {}
    for d in datasets:
        params = _get_initial_shape_and_dtype_for_dataset(d, n_dim=features_dim)
        layouts[d] = h5py.VirtualLayout(
            shape=(total[d],) + params["shape"][1:], dtype=params.get("dtype")
        )
    SIDX = 0
    for s in sources:
        with _get_dataset(s, datasets) as ds_arr:
            EIDX = SIDX + ds_arr[0].len()
            for d, ds in zip(datasets, ds_arr):
                vsource = h5py.VirtualSource(ds)
                layouts[d][SIDX:EIDX, ...] = vsource
            SIDX = EIDX

    with h5py.File(output, "w", libver="latest") as f:
        for _k, _layout in layouts.items():
            vds = f.create_virtual_dataset(_k, _layout)
            if _k == H5Datasets.IMAGE_FEATURES:
                vds.attrs["model"] = model_name


def truncate_h5(dataset: Path, **kwargs):
    with h5py.File(dataset, mode="w", **kwargs) as _:
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
