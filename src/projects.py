import re
from pathlib import Path
import shutil
from typing import Optional

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


def get_wise_project_index_folder(project_id: str) -> Path:
    return get_wise_project_folder(project_id) / "index"

def get_wise_project_virtual_h5dataset(project_id: str) -> Path:
    return get_wise_project_folder(project_id) / f"{project_id}.h5"


def get_wise_project_h5dataset(project_id: str, dataset_id: str):
    return get_wise_project_folder(project_id) / "data" / f"{dataset_id.zfill(5)}.h5"


def delete_wise_project_h5dataset(
    project_id: str, dataset_id: str, missing_ok: bool = True
):
    h5dataset = get_wise_project_h5dataset(project_id, dataset_id)
    h5dataset.unlink(missing_ok=missing_ok)


def create_wise_project_tree(project_id: str, destination_dir: Optional[Path] = None):

    # Create folder structure to hold the data
    # - data/[DATASET-ID.zfill(5)].h5
    # - index/
    #   - {...}/
    #       - train.index
    #       - batch-%03d.index
    #       - merged.index
    base_dir = get_wise_project_folder(project_id)

    if destination_dir:
        # $HOME/.wise/projects/{PROJECT_ID} -> /path/to/destination/{PROJECT_ID}
        project_dir = (destination_dir / project_id).resolve()
        if project_dir.is_dir():
            shutil.rmtree(project_dir)
        project_dir.mkdir()
        warning_file = project_dir / "DANGER.txt"
        warning_file.write_text(
            "DO NOT STORE ANYTHING IMPORTANT HERE\n\nMANAGED BY WISE CLI.\n\nCONTENTS MIGHT BE DELETED!",
            encoding="utf-8",
        )
        base_dir.symlink_to(project_dir, target_is_directory=True)

    # Make the data, index tree
    for x in ["data", "index"]:
        (base_dir / x).mkdir(parents=True, exist_ok=True)

    return base_dir


def delete_wise_project_tree(project_id: str):
    project_folder = get_wise_project_folder(project_id)
    if project_folder.is_dir():
        # A valid directory - could be a link
        if project_folder.is_symlink():
            # Remove what the link points to
            shutil.rmtree(project_folder.resolve())

            # remove the link
            project_folder.unlink()
        else:
            # remove the directory tree
            shutil.rmtree(project_folder)
    else:
        # Dangling symlink
        project_folder.unlink()
