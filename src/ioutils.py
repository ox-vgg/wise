import typing
from typing import List, Literal, Optional, Generator, Tuple, cast, Callable
from contextlib import contextmanager
from pathlib import Path
import tarfile
import numpy as np
import h5py
from PIL import Image, IptcImagePlugin
from tqdm import tqdm
import webdataset as wds

from .data_models import ImageInfo, ImageMetadata


def is_valid_image(p: Path):

    try:
        with Image.open(p) as im:
            im.verify()
            return True
    except Exception:
        return False


def is_valid_webdataset_tarfile(p: str):
    if tarfile.is_tarfile(p) == False:
        return False

    try:
        next(iter(wds.WebDataset(p).to_tuple("__key__", "jpg;jpeg", "json")))
        return True
    except Exception as e:
        print(f"{p} is not a valid data source - {e}")
        return False


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


@contextmanager
def _get_dataset(
    path: Path, mode: Literal["r", "w"] = "r", n_dim: Optional[int] = None
) -> Generator[Tuple[h5py.Dataset, h5py.Dataset], None, None]:
    with h5py.File(path, mode) as f:
        if mode != "r":
            # Write enabled
            if n_dim is None:
                raise ValueError("n_dim cannot be None in write mode")
            if "features" not in f:
                f.create_dataset("features", shape=(0, n_dim), maxshape=(None, n_dim))

            if "files" not in f:
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset("files", shape=(0,), dtype=dt, maxshape=(None,))

        yield typing.cast(h5py.Dataset, f["features"]), typing.cast(
            h5py.Dataset, f["files"]
        )


def _write_array_to_dataset(dataset: h5py.Dataset, arr: np.ndarray):
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


def read_dataset(dataset: Path) -> Tuple[np.ndarray, List[str], str]:
    with _get_dataset(dataset, "r") as (ds, fs):
        model_name: str = cast(str, ds.attrs["model"])
        files = [x.decode("utf-8") for x in fs[:]]
        features: np.ndarray = ds[:, ...]
        print(f"Read ({features.shape}) features (model: {model_name})")
        return features, files, model_name


def write_dataset(
    dataset: Path,
    features: Generator[np.ndarray, None, None],
    file_names: List[Path],
    model_name: str,
):
    """
    Features is a 2D array of shape (len(file_names), n_dim)
    """

    # Read first array from generator to get shape
    arr = next(features)
    n_dim = arr.shape[-1]

    with _get_dataset(dataset, "w", n_dim=n_dim) as (ds, fs), tqdm(
        total=len(file_names)
    ) as pbar:

        ds.attrs["model"] = model_name

        # Write the first array
        _write_array_to_dataset(ds, arr)
        pbar.update(arr.shape[0])

        # Write remaining arrays
        for _arr in features:
            _write_array_to_dataset(ds, _arr)
            pbar.update(_arr.shape[0])

        print("writing file names")
        _write_array_to_dataset(fs, np.array([str(x) for x in file_names]))

        print(f"Done - wrote features ({model_name}) with shape: {ds.shape}")

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
