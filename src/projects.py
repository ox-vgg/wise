import re
from pathlib import Path
import shutil
from typing import Optional

from .enums import IndexType

path_replace_pattern = re.compile(r"[^\w\-_\. ]")
DB_SCHEME = "sqlite+pysqlite://"
WARNING_TEXT = "DO NOT STORE ANYTHING IMPORTANT HERE\n\nMANAGED BY WISE CLI.\n\nCONTENTS MIGHT BE DELETED!"


class _WiseTree:
    def __init__(self, base: Path = Path.home()):
        self.__root = None
        self.__base = base

    # will only create the directories on the first invocation
    @property
    def root(self):
        if not self.__root:
            base_folder = self.__base / ".wise"
            base_folder.mkdir(exist_ok=True)

            projects_folder = base_folder / "projects"
            projects_folder.mkdir(exist_ok=True)

            self.__root = base_folder

        return self.__root

    @property
    def projects(self):
        return self.root / "projects"

    @property
    def db(self):
        return self.root / "wise.db"

    @property
    def dburi(self):
        return f"{DB_SCHEME}/{self.db.absolute()}"


WiseTree = _WiseTree()


class WiseProjectTree:
    """
    Create folder structure to hold the data
    - data/[DATASET-ID.zfill(5)].h5
    - thumbs/[DATASET-ID.zfill(5)].h5
    - versions/[v1, v2, v3,...].h5
    - index/
      - {...}/
          - train.index
          - batch-%03d.index
          - merged.index
    """

    @classmethod
    def create(cls, _id: str, destination: Optional[Path] = None):
        tree = cls(_id)
        location = tree.location
        if not location.exists():
            # Project doesn't exist yet (no valid symlink / directory under .wise folder)
            if destination:
                # $HOME/.wise/projects/{PROJECT_ID} -> /path/to/destination/{PROJECT_ID}
                project_dir = (destination / _id).resolve()
                if project_dir.is_dir():
                    raise ValueError(
                        f"Destination '{destination}' already contains a directory named '{_id}'. Please remove it to continue."
                    )
                project_dir.mkdir()

                warning_file = project_dir / "DANGER.txt"
                warning_file.write_text(
                    WARNING_TEXT,
                    encoding="utf-8",
                )
                location.symlink_to(project_dir, target_is_directory=True)
        else:
            # Location exists - maybe a symlink or a directory
            # Ignores destination argument
            pass
        # Make the project dir and the tree inside if it doesn't exist
        for x in ["data", "thumbs", "index", "versions"]:
            (location / x).mkdir(parents=True, exist_ok=True)

        return tree

    def __init__(self, _id: str):
        """
        Get a project tree under wise folder
        """
        self._id = _id

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, _):
        raise ValueError("Modifying ID is disallowed - Please create a new instance")

    @property
    def location(self):
        return WiseTree.projects / self._id

    def _latest(self):
        return self.location / f"{self._id}.h5"

    @property
    def latest(self):
        return self._latest().resolve()

    def index(self, index_type: IndexType):
        return self.location / "index" / f"{index_type.value}.faiss"

    def features(self, dataset_id: str):
        return self.location / "data" / f"{dataset_id.zfill(5)}.h5"

    def thumbs(self, dataset_id: str):
        return self.location / "thumbs" / f"{dataset_id.zfill(5)}.h5"

    def version(self, version: int, *, relative: bool = True):
        v = Path("versions") / f"v{version}.h5"
        return v if relative else (self.location / v)

    @property
    def dburi(self):
        return f"{DB_SCHEME}/{self.location.absolute()}/{self._id}.db"

    def update_version(self, version: int):
        latest = self._latest()

        # Older versions of the app may write it to a file, and may not have the
        # 'versions' directory
        if latest.is_file() and not latest.is_symlink():
            # Move it as version 0 and link to new version
            _old_path = self.version(0, relative=False)
            _old_path.parent.mkdir(exist_ok=True)
            latest.rename(_old_path)

        latest.unlink(missing_ok=True)
        latest.symlink_to(self.version(version))

        return self.location / latest.readlink()

    def _delete(self):
        project_folder = self.location
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
            project_folder.unlink(missing_ok=True)

    def _delete_dataset(self, dataset_id: str, missing_ok: bool = True):
        features = self.features(dataset_id)
        thumbs = self.thumbs(dataset_id)

        features.unlink(missing_ok=missing_ok)
        thumbs.unlink(missing_ok=missing_ok)

    def delete(self, dataset_id: Optional[str] = None, missing_ok: bool = True):
        if dataset_id:
            return self._delete_dataset(dataset_id, missing_ok=missing_ok)

        return self._delete()
