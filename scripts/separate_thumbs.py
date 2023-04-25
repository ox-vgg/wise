import logging
import typer

from src import db
from src.projects import WiseProjectTree, WiseTree
from src.repository import WiseProjectsRepo, DatasetRepo
from src import ioutils

import h5py
from tqdm import tqdm

app = typer.Typer()
app_state = {"verbose": True}
logger = logging.getLogger()


@app.callback()
def base(verbose: bool = False):
    """
    WISE CLI
    Search through collections of images with Text / Image
    """
    app_state["verbose"] = verbose
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )
    global logger, engine
    logger = logging.getLogger()
    engine = db.init(WiseTree.dburi, echo=app_state["verbose"])


@app.command()
def run(
    project_id: str = typer.Argument(..., help="Wise project id to split thumbnails")
):
    with engine.begin() as conn:
        project = WiseProjectsRepo.get(conn, project_id)
        if not project:
            raise typer.BadParameter(f"Project {project_id} not found!")

    # Re-create the tree - this will create inner folders if they dont exist in the
    # existing location
    project_tree = WiseProjectTree.create(project.id)
    dataset_engine = db.init_project(project_tree.dburi, echo=app_state["verbose"])

    # split data into features and thumbnails
    datasets = []
    with dataset_engine.connect() as conn:
        for x in tqdm(DatasetRepo.list(conn)):
            d = project_tree.features(str(x.id))
            t = project_tree.thumbs(str(x.id))

            # Copy thumbnails over to new file
            with h5py.File(d, mode="a") as old_f, h5py.File(
                t,
                mode="a",
            ) as new_f:
                if ioutils.H5Datasets.THUMBNAILS in old_f:
                    old_f.copy(old_f[ioutils.H5Datasets.THUMBNAILS], new_f)
                    del old_f[ioutils.H5Datasets.THUMBNAILS]

    # Update all virtual versions to point to the correct set.
    current_version = project.version or 0

    if current_version == 0:
        raise RuntimeError("Unknown state - expected project to have a version")
    logger.info(f"Current version: {current_version}")
    for v in tqdm(range(1, current_version + 1)):
        _version = project_tree.version(v, relative=False)

        # Get virtual sources and get dataset id from them
        with h5py.File(_version, "r") as _vf:
            dataset_ids = [
                x.file_name.rsplit("/", 1)[1].replace(".h5", "")
                for x in _vf[ioutils.H5Datasets.THUMBNAILS].virtual_sources()
            ]

        sources = [
            d
            for _id in dataset_ids
            for d in (project_tree.features(str(_id)), project_tree.thumbs(str(_id)))
        ]
        ioutils.concat_h5datasets(sources, _version)
    project_tree.update_version(current_version)


if __name__ == "__main__":
    app()
