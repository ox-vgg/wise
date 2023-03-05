import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import webdataset as wds
from src import ioutils
import pytest


def get_1x1_pixel_array():
    return np.array([[[255, 255, 255]]], dtype=np.uint8)


def get_1x1_pixel_image_buffer():
    with io.BytesIO() as buf:
        im = Image.fromarray(get_1x1_pixel_array())
        im.save(buf, format="JPEG")
        return buf.getvalue()


@pytest.fixture(scope="session")
def image_dir_source(tmp_path_factory):
    image_source = tmp_path_factory.mktemp("images")
    image_source.mkdir(exist_ok=True)

    buf = get_1x1_pixel_image_buffer()
    valid_image = image_source / "valid.jpg"
    valid_image.write_bytes(buf)

    invalid_image = image_source / "invalid.jpg"
    invalid_image.write_bytes(buf[:-1])

    return image_source


@pytest.fixture(scope="session")
def valid_image(image_dir_source):
    return image_dir_source / "valid.jpg"


@pytest.fixture(scope="session")
def invalid_image(image_dir_source):
    return image_dir_source / "invalid.jpg"


@pytest.fixture(scope="session")
def wds_dir(tmp_path_factory, image_dir_source):
    wds_dir: Path = tmp_path_factory.mktemp("wds")

    valid_image = image_dir_source / "valid.jpg"

    wds_tar = wds_dir / "valid.tar"
    with wds.TarWriter(str(wds_tar), encoder=False) as sink:
        sink.write({"__key__": "valid", "jpg": valid_image.read_bytes(), "json": b"{}"})

    invalid_wds_tar = wds_dir / "invalid.tar"
    with wds.TarWriter(str(invalid_wds_tar), encoder=False) as sink:
        sink.write({"__key__": "invalid", "json": b"{}"})

    return wds_dir


@pytest.fixture(scope="session")
def valid_wds(wds_dir):
    return wds_dir / "valid.tar"


@pytest.fixture(scope="session")
def invalid_wds(wds_dir):
    return wds_dir / "invalid.tar"


class TestValidationFunctions:
    """
    Assuming you are connected to the internet
    These tests should run fine
    """

    def test_should_return_true(self):

        assert ioutils.is_valid_uri("https://google.com") == True

    def test_should_return_false(self):
        assert ioutils.is_valid_uri("http://thisisarandomtest.wut") == False

    def test_should_return_true_on_valid_image(self, valid_image: Path):
        assert ioutils.is_valid_image(valid_image) == True

    def test_should_return_false_on_invalid_image(self):
        assert ioutils.is_valid_image(Path("app.py")) == False

    def test_should_return_true_on_valid_wds_source(self, valid_wds):
        assert ioutils.is_valid_webdataset_source(str(valid_wds))

    def test_should_return_false_on_invalid_tar(self, invalid_wds):
        assert ioutils.is_valid_webdataset_source(str(invalid_wds)) == False
        assert ioutils.is_valid_webdataset_source("app.py") == False


class TestSourceCollection:
    def test_should_pick_valid_wds_tar_files(self, wds_dir):
        sources = list(ioutils.get_valid_webdataset_tar_from_folder(wds_dir))
        assert str(wds_dir / "valid.tar") in sources
        assert str(wds_dir / "invalid.tar") not in sources

    def test_should_extract_file_from_tar(self, wds_dir):
        jsonbytes = b"".join(
            ioutils.get_file_from_tar(wds_dir / "valid.tar", "valid.json")
        )
        assert (len(jsonbytes)) == 2
        assert json.loads(jsonbytes) == {}
        # TODO Should raise exception cases are pending

    def test_should_return_images_from_folder(self, image_dir_source):
        paths = [
            x.path
            for _, x in ioutils.get_valid_images_from_folder(
                image_dir_source, lambda _: None
            )
        ]

        assert set(paths) == set(
            [
                "valid.jpg",
            ]
        )
        assert ["invalid.jpg"] not in paths

    def test_should_return_images_from_webdataset(self, valid_wds):
        paths = [
            x.path
            for _, x in ioutils.get_valid_images_from_webdataset(
                str(valid_wds), lambda _: None
            )
        ]

        assert [f"{valid_wds}#valid.jpg"] == paths


class TestH5Dataset:
    def test_dataset_writer(self, tmp_path: Path):
        h5_dataset = tmp_path / "test.h5"
        dummy_arr = np.arange(100).reshape(10, 10)

        with ioutils.get_h5writer(
            h5_dataset,
            names=[
                ioutils.H5Datasets.FEATURES,
                ioutils.H5Datasets.THUMBNAILS,
                ioutils.H5Datasets.IDS,
            ],
            n_dim=10,
            model_name="test",
        ) as write_fn:
            write_fn(dummy_arr, [b"abc"] * 0, ["1"] * 1)

        # assert file with all datasets exits
        assert ioutils.does_wise_hdf5_exists(h5_dataset)

        reader = ioutils.get_h5iterator(h5_dataset)
        model_name = ioutils.get_model_name(h5_dataset)
        counts_after_write = ioutils.get_counts(h5_dataset)

        # Test counts and attribute
        assert counts_after_write == {
            ioutils.H5Datasets.FEATURES: 10,
            ioutils.H5Datasets.IDS: 1,
            ioutils.H5Datasets.THUMBNAILS: 0,
        }
        assert model_name == "test"

        # Test iterator
        read_arr = np.concatenate(list(reader(ioutils.H5Datasets.FEATURES)), axis=0)
        assert np.array_equal(read_arr, dummy_arr)

        # Test indexed read
        indexed_reader = ioutils.get_h5reader(h5_dataset)
        with indexed_reader(ioutils.H5Datasets.FEATURES) as _reader:
            assert np.array_equal(dummy_arr, np.array(_reader(list(range(10)))))
