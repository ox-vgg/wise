import enum
import functools
import itertools
import math
import logging
from pathlib import Path

from tempfile import NamedTemporaryFile
from threading import Lock
import typing
from typing import List, Union, Tuple, Literal, Optional, Dict, Any, Callable

import typer

from torch.hub import download_url_to_file
from torch import Tensor

import numpy as np
from PIL import Image
import braceexpand
from tqdm import tqdm
import webdataset as wds

from src import db
from src.enums import IndexType
from src.ioutils import (
    H5Datasets,
    get_model_name,
    get_shapes,
    get_counts,
    get_h5iterator,
    get_h5writer,
    is_valid_webdataset_source,
    get_valid_webdataset_tar_from_folder,
    get_dataloader,
    concat_h5datasets,
)

from src.data_models import (
    Dataset,
    DatasetType,
    URL,
    QueryType,
    Project,
    DatasetCreate,
    ImageMetadata,
)
from src.inference import setup_clip, CLIPModel
from src.search import (
    write_index,
    get_index,
    brute_force_search,
)
from src.repository import WiseProjectsRepo, DatasetRepo, MetadataRepo
from src.projects import WiseTree, WiseProjectTree

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


def parse_webdataset_url(wds_url: str):
    """
    Parses the user input of webdataset urls
    and splits them into individual URLs if needed

    Example
    - a/{001..010}.tar -> [a/001.tar, a/002.tar, ....]
    - a.tar,b.tar -> [a.tar, b.tar]
    - a/{001..010}.tar,b.tar -> [a/001.tar, a/002.tar, ..., a/010.tar, b.tar]
    """

    # - split at comma
    splits = wds_url.split(",")

    # - strip leading / trailing space
    stripped = map(lambda x: x.strip(), splits)

    # - apply braceexpand
    expanded = map(lambda x: braceexpand.braceexpand(x), stripped)

    return list(itertools.chain.from_iterable(expanded))


# Split input arguments into respective category of folders and webdataset urls
def parse_and_validate_input_datasets(input_dataset: List[str]):
    """
    Validates user input for data sources

    Should contain either a directory / webdataset url

    If a directory is provided, all valid webdataset tar files
    inside are selected recursively as sources.
    """
    # At least one input must be passed
    if len(input_dataset) == 0:
        raise typer.BadParameter("Input dataset cannot be empty")

    # Now we have a list[Folder | WebDataset]
    processed: List[Union[Path, str]] = []
    for dataset in input_dataset:
        if (p := Path(dataset)).is_dir():
            # dataset is a directory
            processed.append(p)

            # Extend with all valid webdataset tar files found inside folder
            processed.extend(sorted(get_valid_webdataset_tar_from_folder(p)))
        else:
            # We don't know what it is, we will try reading it with webdataset
            processed.extend(parse_webdataset_url(dataset))

    # - deduplicate
    num_sources = len(processed)
    processed = list(dict.fromkeys(processed))

    if len(processed) != num_sources:
        print("Found duplicate sources. Considering only one of them")

    # - raise error on invalid URLs
    for p in processed:
        if not isinstance(p, Path) and not is_valid_webdataset_source(p):
            raise typer.BadParameter(
                f"Invalid source - {p} is not a valid WebDataset Source"
            )
    return processed


def parse_query_parameter(
    queries: List[str],
) -> Union[
    Tuple[Literal[QueryType.NATURAL_LANGUAGE_QUERY], List[str]],
    Tuple[Literal[QueryType.IMAGE_QUERY], Union[URL, Path]],
    Tuple[Literal[QueryType.IMAGE_CLASSIFICATION_QUERY], Path],
]:
    """
    Parse the passed search queries and return based on conditions
    - If 2 or more -> Consider it as list of string queries
    - If 1 and is completely made of alphabets -> Consider as string query
    - If 1 and is a directory / file -> Consider as path
    - Else invalid -> raise typer.BadParameter
    """
    if len(queries) == 0:
        raise typer.BadParameter("Query cannot be empty")

    if len(queries) > 1:
        # This is a list, so consider it as list of multiple string queries
        return QueryType.NATURAL_LANGUAGE_QUERY, queries

    # queries has 1 element
    query = queries[0]

    # query is url
    if query.startswith(("http://", "https://")):
        return QueryType.IMAGE_QUERY, URL(query)

    if query.replace(" ", "").isalpha():
        return QueryType.NATURAL_LANGUAGE_QUERY, queries

    qpath = Path(query)

    if qpath.is_dir():
        return QueryType.IMAGE_CLASSIFICATION_QUERY, qpath

    if qpath.is_file():
        return QueryType.IMAGE_QUERY, qpath

    raise typer.BadParameter("Unknown query parameter, expected List[str] | URL | Path")


def overwrite():
    # if does_hdf5_file_exists(save_features, driver="family"):
    #     typer.confirm(
    #         "Are you sure you want to overwrite existing dataset?",
    #         abort=True,
    #         default=True,
    #     )
    #     print(f"Overwriting - {save_features}")
    pass


def _create_virtual_dataset(project: Project, sources: List[Path]):
    project_id = project.id
    project_tree = WiseProjectTree(project_id)
    old_version = project.version or 0

    next_version = 1 + old_version
    output = project_tree.update_version(next_version)

    try:
        concat_h5datasets(sources, output)
    except Exception as e:
        logger.error(f"Error creating virtual dataset from sources - {e}")
        logger.error("Rolling back...")

        # Delete the file if it was created
        output.unlink(missing_ok=True)

        # Update the symlink back to old version
        project_tree.update_version(old_version)
        raise e
    return next_version


def _extract_image_and_metadata_features(
    extract_image_features,
    extract_text_features,
    images: Union[Tensor, List[Image.Image]],
    metadata: List[ImageMetadata],
):
    image_features = extract_image_features(images)
    if extract_text_features is None:
        return (image_features,)

    # A callable is pass
    metadata_features = extract_text_features(
        [x.metadata.get("description", "") for x in metadata]
    )
    return image_features, metadata_features


# Function to convert input source to DatasetModel
def _convert_input_source_to_dataset(input_source: Union[str, Path]):
    return (
        DatasetCreate(location=str(input_source.resolve()), type=DatasetType.IMAGE_DIR)
        if isinstance(input_source, Path)
        else DatasetCreate(
            location=(
                str(x.resolve())
                if (x := Path(input_source)).is_file()
                else input_source
            ),
            type=DatasetType.WEBDATASET,
        )
    )


def add_dataset(
    dataset: Dataset,
    image_transform,
    extract_image_and_metadata_features,
    features_writer_fn,
    thumbs_writer_fn,
    *,
    db_engine,
    offset: int = 0,
    error_handler=lambda sample: None,
    batch_size: int = 1,
    num_workers: int = None,
):
    # Get dataset iterator
    dataloader = get_dataloader(
        dataset, image_transform, error_handler, batch_size, num_workers
    )

    # Keep track of records written in this dataset
    count_ = 0
    with tqdm() as pbar, db_engine.connect() as conn:
        for images, metadata, thumbs in dataloader:
            extracted_features = extract_image_and_metadata_features(images, metadata)

            row_count = len(images)
            h5_row_ids = [count_ + i for i in range(row_count)]

            with conn.begin():
                metadata = [
                    MetadataRepo.create(
                        conn,
                        data=x.copy(
                            update={
                                "dataset_row": row_id,
                                "id": offset + row_id,
                            }
                        ),
                    )
                    for row_id, x in zip(h5_row_ids, metadata)
                ]

                metadata_row_ids = [str(offset + x) for x in h5_row_ids]
                features_writer_fn(metadata_row_ids, *extracted_features)
                thumbs_writer_fn(thumbs)
            count_ += row_count

            # Udpate progress bar
            pbar.update(row_count)
    # Return updated offset
    return offset + count_


def _update(
    project: Project,
    input_datasets: List[Union[str, Path]],
    model_name: CLIPModel,
    *,
    db_engine,
    mode: Literal["init", "update"] = "init",
    handle_failed_sample=lambda sample: None,
    continue_on_error: bool = False,
    include_metadata_features: bool = False,
    batch_size: int = 1,
    num_workers: int = None,
):
    """
    Appends new data sources to project

    Creates a Virtual Dataset with the project's data sources
    at the end for unified access
    """
    project_id = project.id
    project_tree = WiseProjectTree(project_id)

    n_dim, image_transform, extract_image_features, extract_text_features = setup_clip(
        model_name
    )
    extract_image_and_metadata_features = functools.partial(
        _extract_image_and_metadata_features,
        extract_image_features,
        extract_text_features if include_metadata_features else None,
    )

    thumbs_h5dataset = H5Datasets.THUMBNAILS
    features_h5datasets = [
        H5Datasets.IDS,
        H5Datasets.IMAGE_FEATURES,
    ]
    if include_metadata_features:
        features_h5datasets.append(H5Datasets.METADATA_FEATURES)

    count_: int = 0
    if mode == "update":
        vds_path = project_tree.latest
        count_ = get_counts(vds_path)[H5Datasets.IMAGE_FEATURES]

    added_datasets = []
    failed_sources = []
    _exception = None

    _input_datasets = iter(input_datasets)

    for dataset in _input_datasets:
        logger.info(f"Processing - {dataset}")
        dataset_obj = None
        try:
            # Add data source to table
            with db_engine.begin() as conn:
                dataset_obj = DatasetRepo.create(
                    conn,
                    data=_convert_input_source_to_dataset(dataset),
                )

            # Get dataset path
            features_path = project_tree.features(str(dataset_obj.id))
            thumbs_path = project_tree.thumbs(str(dataset_obj.id))

            # Get writer fn
            with get_h5writer(
                features_path,
                features_h5datasets,
                model_name=model_name.value,
                n_dim=n_dim,
                mode="w",
            ) as features_writer_fn, get_h5writer(
                thumbs_path, thumbs_h5dataset, mode="w"
            ) as thumbs_writer_fn:
                # Process each item in data source and write to table + hdf5
                count_ = add_dataset(
                    dataset_obj,
                    image_transform,
                    extract_image_and_metadata_features,
                    features_writer_fn,
                    thumbs_writer_fn,
                    db_engine=db_engine,
                    offset=count_,
                    error_handler=handle_failed_sample,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            added_datasets.append(features_path)
            added_datasets.append(thumbs_path)
        except (KeyboardInterrupt, Exception) as e:
            logger.error(f'Error while processing data source "{dataset}" - {e}')
            # Delete the h5 datasets, table entry, and add to failed summary.
            if dataset_obj:
                # Delete dataset
                project_tree.delete(str(dataset_obj.id))

                # Delete the Dataset -> Which deletes all metadata
                # Later, we can just continue from where we left off instead of deleting.
                with db_engine.begin() as conn:
                    DatasetRepo.delete(conn, dataset_obj.id)

            failed_sources.append(dataset)
            if isinstance(e, KeyboardInterrupt) or (not continue_on_error):
                _exception = e
                failed_sources.extend(_input_datasets)
                break

    # init mode, re-raise the exception and clear out the project tree
    if _exception:
        raise _exception

    return added_datasets, failed_sources


@app.command()
def init(
    sources: List[str] = typer.Option(
        ..., "--source", help="List[DirPath | WebDataset compatible URL]"
    ),
    model: CLIPModel = typer.Option("ViT-B-32:openai", help="CLIP Model to use"),  # type: ignore
    store_in: Optional[Path] = typer.Option(
        None,
        writable=True,
        exists=True,
        dir_okay=True,
        file_okay=False,
        help="Directory to save the output files to",
    ),
    continue_on_error: bool = typer.Option(
        False, help="Continue processing when encountered with errors if possible"
    ),
    include_metadata_features: bool = typer.Option(
        False, help="Extract features from metadata in addition to the image"
    ),
    project_id: str = typer.Argument(..., help="Name of the project"),
    batch_size: int = typer.Option(
        1,
        help="Batch size that would fit your RAM/GPURAM",
    ),
    num_workers: int = typer.Option(
        None,
        help="Number of subprocesses to use for data loading (for PyTorch Dataloader). If set to 0, the main process is used for data loading. If omitted, num_workers will be automatically determined.",
    ),
):
    """
    Initialise WISE Project

    - Create project
    - For each data source, extract features, thumbnails and metadata
      and store it
    - Failed samples are written to wise_failedsamples_{project_id}.tar
    - If a dataset write fails, the associate files are deleted
    - If continue on error is True, other data sources will be processed
    - Any other exception, the project is not created.
    """
    # Setup
    # Validates the input sources and converts them into a list
    # containing directories, valid webdatset tar files in those directories
    # and parses other strings as webdataset urls and checks if it can be opened
    input_datasets = parse_and_validate_input_datasets(sources)

    _sources = "\n\t".join(sources)
    logger.info(f"Processsing data sources - {_sources}")

    # Try creating project id, it will fail if not unique.
    # TODO Translate the exception
    with engine.begin() as conn:
        project = WiseProjectsRepo.create(conn, data=Project(id=project_id))

    failures_path = f"wise_init_failedsamples_{project_id}.tar"
    failed_datasources = []
    dataset_engine = None

    try:
        # Create project tree, get paths to the HDF5 files
        project_tree = WiseProjectTree.create(project_id, destination=store_in)

        dataset_engine = db.init_project(project_tree.dburi, echo=app_state["verbose"])

        with wds.TarWriter(failures_path, encoder=False) as sink:
            failed_sample_writer_lock = Lock()

            def handle_failed_sample(sample: Dict[str, Any]):
                failed_sample_writer_lock.acquire()
                sink.write(sample)
                failed_sample_writer_lock.release()

            added, failed_datasources = _update(
                project,
                input_datasets,
                model,
                db_engine=dataset_engine,
                mode="init",
                handle_failed_sample=handle_failed_sample,
                continue_on_error=continue_on_error,
                include_metadata_features=include_metadata_features,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            if len(added) > 0:
                logger.info("Creating Virtual dataset...")
                with engine.begin() as conn:
                    new_version = _create_virtual_dataset(project, added)
                    WiseProjectsRepo.update(
                        conn,
                        project_id,
                        data=project.copy(update={"version": new_version}),
                    )
                logger.info("Done")

            if failed_datasources:
                logger.error(
                    f'Project "{project_id}" was created, but the following data sources failed'
                )
                logger.error("\n\t".join(failed_datasources))
                logger.error(
                    "Call the update command after fixing the datasource for errors"
                )

    except (KeyboardInterrupt, Exception):
        logger.exception(f"Initialising project {project_id} failed!")
        delete_project = typer.confirm("Delete associated project files?", default=True)
        project_tree = WiseProjectTree(project_id)
        if delete_project:
            with engine.begin() as conn:
                WiseProjectsRepo.delete(conn, project_id)
            project_tree.delete()

            raise typer.Exit(1)

        if dataset_engine:
            logger.info("Creating virtual dataset...")
            with dataset_engine.connect() as conn:
                datasets = [
                    d
                    for x in DatasetRepo.list(conn)
                    for d in (
                        project_tree.features(str(x.id)),
                        project_tree.thumbs(str(x.id)),
                    )
                ]
            with engine.begin() as conn:
                new_version = _create_virtual_dataset(project, datasets)
                WiseProjectsRepo.update(
                    conn, project_id, data=project.copy(update={"version": new_version})
                )
            logger.info("Done")
        logger.error(f'Files for "{project_id}" left in {project_tree.location}')
        raise typer.Exit(1)


@app.command()
def update(
    sources: List[str] = typer.Option(
        ..., "--source", help="List[DirPath | WebDataset compatible URL]"
    ),
    continue_on_error: bool = typer.Option(
        False, help="Continue processing when encountered with errors if possible"
    ),
    project_id: str = typer.Argument(..., help="Name of the project"),
    batch_size: int = typer.Option(
        1,
        help="Batch size that would fit your RAM/GPURAM",
    ),
    num_workers: int = typer.Option(
        None,
        help="Number of subprocesses to use for data loading (for PyTorch Dataloader). If set to 0, the main process is used for data loading. If omitted, num_workers will be automatically determined.",
    ),
):
    """
    Update WISE Project

    - For each data source, extract the features, thumbnails and metadata
      of images and append it to the project dataset
    - Failed samples are written to wise_failed_update_{project_id}.tar
    - If writing fails while processing a data source, the associated files are deleted
    - If continue on error is True, other data sources will be processed
    - Any other exception, the project is rolled back to initial state.
    """
    # Setup
    # Validates the input sources and converts them into a list
    # containing directories, valid webdatset tar files in those directories
    # and parses other strings as webdataset urls and checks if it can be opened
    input_datasets = parse_and_validate_input_datasets(sources)

    _sources = "\n\t".join(sources)
    logger.info(f"Processsing data sources - {_sources}")

    # Try creating project id, it will fail if not unique.
    # TODO Translate the exception
    with engine.begin() as conn:
        project = WiseProjectsRepo.get(conn, project_id)
        if not project:
            raise typer.BadParameter(f"Project {project_id} not found!")

    project_tree = WiseProjectTree(project_id)
    dataset_engine = db.init_project(project_tree.dburi, echo=app_state["verbose"])

    vds_path = project_tree.latest
    model_name = CLIPModel[get_model_name(vds_path)]
    counts = get_counts(vds_path)

    # Check if include_metadata_eatures was enabled in init command
    include_metadata_features = H5Datasets.METADATA_FEATURES in counts

    failures_path = f"wise_update_failedsamples_{project_id}.tar"
    added = []
    failed_datasources = []
    try:
        with wds.TarWriter(failures_path, encoder=False) as sink:

            def handle_failed_sample(sample: Dict[str, Any]):
                sink.write(sample)

            _, failed_datasources = _update(
                project,
                input_datasets,
                model_name,
                db_engine=dataset_engine,
                mode="update",
                batch_size=batch_size,
                handle_failed_sample=handle_failed_sample,
                continue_on_error=continue_on_error,
                include_metadata_features=include_metadata_features,
                num_workers=num_workers
            )
            if failed_datasources:
                logger.error(
                    f'Project "{project_id}" was updated, but the following data sources failed'
                )
                logger.error("\n\t".join(failed_datasources))
    except Exception:
        logger.exception(f"Updating project {project_id} failed!")
        raise typer.Exit(1)

    finally:
        # Recreate the new version
        logger.info("Creating Virtual Dataset...")
        with dataset_engine.connect() as conn:
            datasets = [
                d
                for x in DatasetRepo.list(conn)
                for d in (
                    project_tree.features(str(x.id)),
                    project_tree.thumbs(str(x.id)),
                )
            ]
        with engine.begin() as conn:
            new_version = _create_virtual_dataset(project, datasets)
            WiseProjectsRepo.update(
                conn, project_id, data=project.copy(update={"version": new_version})
            )
        logger.info("Done")


@app.command()
def delete(
    project_id: str = typer.Argument(..., help="Name of the project"),
    force: bool = typer.Option(False, "-f", help="Force delete"),
):
    """
    Delete the project and associated files
    """
    with engine.begin() as conn:
        project = WiseProjectsRepo.get(conn, project_id)
        if not project:
            raise typer.BadParameter(f"Project {project_id} not found!")

        delete_project = (
            project_id
            if force
            else typer.prompt("Please type the project name again to confirm")
        )
        if delete_project != project_id:
            logger.error(f"Not deleting {project_id}")
            raise typer.Abort()

        WiseProjectTree(project_id).delete()
        with engine.begin() as conn:
            WiseProjectsRepo.delete(conn, project_id)


class FEATURES(str, enum.Enum):
    IMAGE = "image_features"
    METADATA = "metadata_features"


@app.command()
def search(
    project_id: str = typer.Argument(..., help="Name of the project"),
    top_k: int = typer.Option(5, help="Top-k results to retrieve"),
    prefix: str = typer.Option(
        "This is a photo of a", help="Prefix to attach to all natural language queries"
    ),
    using: FEATURES = typer.Option(
        FEATURES.IMAGE,
        help="Select whether image or metadata features must be used for search",
    ),
    batch_size: int = typer.Option(1, help="Batch size to extract features"),
    queries: List[str] = typer.Argument(
        ...,
        help="Search query/queries. Can be a text (natural language) query, a path to an image, or a path to a directory of images",
    ),
):
    query_type, parsed_queries = parse_query_parameter(queries)
    engine = db.init(WiseTree.dburi, echo=app_state["verbose"])
    with engine.connect() as conn:
        project = WiseProjectsRepo.get(conn, project_id)
        if project is None:
            raise typer.BadParameter(f"Project {project_id} not found!")

    project_tree = WiseProjectTree(project_id)
    vds_path = project_tree.latest
    model_name = CLIPModel[get_model_name(vds_path)]
    counts = get_counts(vds_path)
    assert counts[H5Datasets.IMAGE_FEATURES] == counts[H5Datasets.IDS]

    if using == FEATURES.METADATA:
        if H5Datasets.METADATA_FEATURES not in counts:
            raise typer.BadParameter(
                "Project does not have metadata features to use. Re-initialise the project with --include-metadata-features flag"
            )
        assert counts[H5Datasets.IDS] == counts[H5Datasets.METADATA_FEATURES]

    num_files = counts[H5Datasets.IMAGE_FEATURES]
    reader = get_h5iterator(vds_path)
    all_features = lambda: reader(
        H5Datasets.IMAGE_FEATURES
        if using == FEATURES.IMAGE
        else H5Datasets.METADATA_FEATURES
    )

    _, _, extract_image_features, extract_text_features = setup_clip(model_name)
    top_k = min(top_k, num_files)

    dist, ids = None, None
    if (
        query_type == QueryType.IMAGE_QUERY
        or query_type == QueryType.NATURAL_LANGUAGE_QUERY
    ):
        if query_type == QueryType.NATURAL_LANGUAGE_QUERY:
            prefixed_queries = [
                f"{prefix.strip()} {x.strip()}".strip()
                for x in typing.cast(List[str], parsed_queries)
            ]
            print("Processing queries:\n", prefixed_queries)
            query_features = extract_text_features(prefixed_queries)
        else:
            # IMAGE_QUERY
            print(f"Processing image query - {parsed_queries}")
            if isinstance(parsed_queries, URL):
                print("Downloading", parsed_queries, "to file")
                with NamedTemporaryFile() as tmpfile:
                    download_url_to_file(parsed_queries, tmpfile.name)
                    with Image.open(tmpfile.name) as im:
                        query_features = extract_image_features([im])

            elif isinstance(parsed_queries, Path):
                with Image.open(parsed_queries) as im:
                    query_features = extract_image_features([im])

            else:
                raise NotImplementedError

        dist, ids = brute_force_search(all_features(), query_features, top_k)

    # Classification query
    else:
        pass
        # TODO update to use dataloader later
        # def process(query_images_folder: Path):
        #     for batch in batched(
        #         get_valid_images_from_folder(query_images_folder, lambda _: None),
        #         batch_size,
        #     ):

        #         images, _ = zip(*batch)
        #         features = extract_image_features(images)

        #         yield features

        # parsed_queries = typing.cast(Path, parsed_queries)
        # query_image_features = np.concatenate(list(process(parsed_queries)), axis=0)
        # # TODO Classification query loads all arrays into memory. need to refactor
        # features = np.concatenate(list(all_features()), axis=0)
        # dist, ids = classification_based_query(features, query_image_features, top_k)

    if ids is not None:
        project_engine = db.init_project(project_tree.dburi, echo=app_state["verbose"])
        with project_engine.connect() as conn:
            for query, result_dist, result_ids in zip(
                (
                    parsed_queries
                    if isinstance(parsed_queries, list)
                    else [parsed_queries]
                ),
                dist,
                ids,
            ):
                print(
                    query,
                )
                for d, k in zip(result_dist, result_ids):
                    # Sqlite row id when not set explicitly starts at 1
                    print(f"Dist: {d:.5f}", MetadataRepo.get(conn, int(k)))


@app.command()
def serve(
    project_id: str = typer.Argument(None, help="Name of the project"),
    theme_asset_dir: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=True,
        file_okay=False,
        help="Static HTML assets related to the user interface are served from this folder",
    ),
    index_type: Optional[IndexType] = typer.Option(
        None, help="The faiss index to use for serving"
    ),
    query_blocklist: Path = typer.Option(
        None,
        '--query-blocklist',
        '--query-blacklist',
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="A text file containing a list of words/phrases (each separated by a line break) that users should be blocked from searching. When the user enters a query that matches one of the terms in the blocklist, an error message will be returned",
    )
):
    from api import serve

    project_tree = WiseProjectTree(project_id)
    if index_type:
        index_filename = project_tree.index(index_type)
        if not index_filename.exists():
            raise typer.BadParameter(
                f"Index not found at {index_filename}. Use the 'index' command to create an index."
            )
    # If index_type is None, it will be read from the config

    serve(project_id, theme_asset_dir, index_type.value if index_type else None, query_blocklist)


@app.command()
def index(
    project_id: str = typer.Argument(..., help="Name of the project"),
    index_type: IndexType = typer.Option(
        IndexType.IndexFlatIP, help="The faiss index name"
    ),
    using: FEATURES = typer.Option(
        FEATURES.IMAGE, help="Specify the feature set to build the index with"
    ),
):
    """
    Creates / Updates the current search index for the project
    """
    with engine.connect() as conn:
        project = WiseProjectsRepo.get(conn, project_id)
    if project is None:
        raise typer.BadParameter(f"Project {project_id} not found!")

    project_tree = WiseProjectTree(project_id)
    vds_path = project_tree.latest
    n_dim = get_shapes(vds_path)[H5Datasets.IMAGE_FEATURES][1]

    counts = get_counts(vds_path)
    assert counts[H5Datasets.IMAGE_FEATURES] == counts[H5Datasets.IDS]
    if using == FEATURES.METADATA:
        if H5Datasets.METADATA_FEATURES not in counts:
            raise typer.BadParameter(
                "Project does not have metadata features to use. Re-initialise the project with --include-metadata-features flag"
            )
        assert counts[H5Datasets.IDS] == counts[H5Datasets.METADATA_FEATURES]

    num_files = counts[H5Datasets.IMAGE_FEATURES]
    read_batch_size = 1024
    reader = get_h5iterator(vds_path, batch_size=read_batch_size)
    features_set = (
        H5Datasets.IMAGE_FEATURES
        if using == FEATURES.IMAGE
        else H5Datasets.METADATA_FEATURES
    )
    all_features = lambda: reader(features_set)
    cell_count = 10 * round(math.sqrt(num_files))
    faiss_index = get_index(index_type, n_dim, cell_count)

    if index_type == IndexType.IndexIVFFlat:
        # Train stage
        train_count = min(num_files, 100 * cell_count)
        num_batches = math.ceil(train_count / read_batch_size)
        _train_features = functools.reduce(
            lambda a, x: (a.append(x), a)[1],
            itertools.islice(all_features(), num_batches),
            [],
        )
        train_features = np.concatenate(_train_features)
        assert not faiss_index.is_trained
        logger.info("Finding clusters from samples...")
        faiss_index.train(train_features)
        assert faiss_index.is_trained
        logger.info("Done")

        del train_features

    with tqdm(total=num_files) as pbar:
        for batch in all_features():
            faiss_index.add(batch)
            pbar.update(batch.shape[0])

    index_filename = project_tree.index(index_type)
    logger.info(f"Saving faiss index to {index_filename}...")
    write_index(faiss_index, index_filename)
    logger.info("Done!")


if __name__ == "__main__":
    app()
