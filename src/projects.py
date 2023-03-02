import itertools
import re
from pathlib import Path
from typing import Optional, Literal

path_replace_pattern = re.compile(r"[^\w\-_\. ]")
DB_SCHEME = "sqlite+pysqlite://"


def get_wise_folder() -> Path:
    # Get OS USER folder
    home_folder = Path.home()

    # Create a .wise folder if it doesn't exist
    wise_folder = home_folder / ".wise"
    wise_folder.mkdir(exist_ok=True)

    # Return
    return wise_folder


def get_wise_db_uri():
    wise_db = get_wise_folder() / "wise.db"
    return f"{DB_SCHEME}/{wise_db.absolute()}"


def get_wise_project_db_uri(project_id: str):
    return (
        f"{DB_SCHEME}/{get_wise_project_folder(project_id).absolute()}/{project_id}.db"
    )


def get_wise_project_folder(project_id: str) -> Path:
    wise_folder = get_wise_folder()
    return wise_folder / "projects" / project_id


def get_wise_features_dataset_path(
    project_id: str,
    l1: Literal["features", "index"],
    l2: Literal["images", "metadata"],
    extra: Optional[str] = None,
) -> Path:
    return (
        get_wise_project_folder(project_id)
        / f"{l1}"
        / f"{l2}"
        / f"{l1}{('-' + path_replace_pattern.sub('_',extra)) if extra else ''}-%05d.h5"
    )


def get_wise_thumbs_dataset_path(project_id: str) -> Path:
    return get_wise_project_folder(project_id) / "thumbs" / "thumbs-%05d.h5"


def create_wise_project_tree(project_id: str):

    # Create folder structure to hold the data
    # - features/
    #   - {images, metadata}/[MODEL]-%03d.h5
    # - index/
    #   - {images, metadata}/
    #       - train.index
    #       - batch-%03d.index
    #       - merged.index
    # - thumbs/
    #   - thumbs-%05d.h5
    base_dir = get_wise_project_folder(project_id)

    # Make the thumbs dir
    thumbs_dir = base_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Make the features, index tree
    for x, y in itertools.product(["features", "index"], ["images", "metadata"]):
        (base_dir / x / y).mkdir(parents=True, exist_ok=True)

    return base_dir
