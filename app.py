import enum
import itertools
from pathlib import Path

import typing
from typing import List, Union, Tuple, Literal, Optional, Dict, Any, Callable
from tempfile import NamedTemporaryFile

import logging

import typer

from src.inference import setup_clip, AVAILABLE_MODELS
from torch.hub import download_url_to_file

from src.ioutils import (
    H5Datasets,
    get_model_name,
    get_counts,
    get_h5iterator,
    get_h5writer,
    generate_thumbnail,
    get_valid_images_from_folder,
    get_valid_images_from_webdataset,
    is_valid_webdataset_source,
    get_valid_webdataset_tar_from_folder,
    concat_h5datasets,
)

import numpy as np
from PIL import Image
import braceexpand
from tqdm import tqdm
import webdataset as wds
from src import db
from src.data_models import (
    Dataset,
    DatasetType,
    URL,
    QueryType,
    Project,
    DatasetCreate,
    ImageMetadata,
)
from src.inference import setup_clip, AVAILABLE_MODELS
from src.utils import batched
from src.search import brute_force_search, classification_based_query
from src.repository import WiseProjectsRepo, DatasetRepo, MetadataRepo
from src.projects import (
    get_wise_db_uri,
    get_wise_project_db_uri,
    get_wise_project_folder,
    get_wise_project_h5dataset,
    get_wise_project_virtual_h5dataset,
    delete_wise_project_h5dataset,
    create_wise_project_tree,
    delete_wise_project_tree,
)
from src.exceptions import EmptyDatasetException


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
    global logger
    logger = logging.getLogger()


class _CLIPModel(str, enum.Enum):
    pass


CLIPModel = _CLIPModel("CLIPModel", {x: x for x in AVAILABLE_MODELS})


def parse_webdataset_url(wds_url: str):
    # - split at comma
    splits = wds_url.split(",")

    # - strip leading / trailing space
    stripped = map(lambda x: x.strip(), splits)

    # - apply braceexpand
    expanded = map(lambda x: braceexpand.braceexpand(x), stripped)

    return list(itertools.chain.from_iterable(expanded))


def get_dataset_iterator(
    dataset: Dataset, handle_failed_sample=Callable[[Dict[str, Any]], None]
):
    def update_id(metadata: ImageMetadata):
        return metadata.copy(update={"dataset_id": dataset.id})

    def update_path(metadata: ImageMetadata):
        return metadata.copy(
            update={"path": metadata.path.replace(dataset.location, "")}
        )

    if dataset.type == DatasetType.WEBDATASET:
        yield from map(
            lambda x: (x[0], update_path(update_id(x[1]))),
            get_valid_images_from_webdataset(dataset.location, handle_failed_sample),
        )

    elif dataset.type == DatasetType.IMAGE_DIR:
        yield from map(
            lambda x: (x[0], update_id(x[1])),
            get_valid_images_from_folder(Path(dataset.location), handle_failed_sample),
        )

    else:
        raise NotImplementedError


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
            processed.extend(get_valid_webdataset_tar_from_folder(p))
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


@app.command()
def init(
    batch_size: int = typer.Option(
        1,
        help="Batch size that would fit your RAM/GPURAM",
    ),
    model: CLIPModel = typer.Option("ViT-B/32", help="CLIP Model to use"),  # type: ignore
    sources: List[str] = typer.Option(
        ..., "--source", help="List[DirPath | WebDataset compatible URL]"
    ),
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
    project_id: str = typer.Argument(..., help="Name of the project"),
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
    engine = db.init(get_wise_db_uri(), echo=app_state["verbose"])

    with engine.begin() as conn:
        WiseProjectsRepo.create(conn, data=Project(id=project_id))

    failed_datasources = []
    try:
        # Create project tree, get paths to the HDF5 files
        project_folder = create_wise_project_tree(project_id, store_in)
        dataset_engine = db.init_project(
            get_wise_project_db_uri(project_id), echo=app_state["verbose"]
        )
        model_name = model.value
        n_dim, extract_features, _ = setup_clip(model_name)

        def create_virtual_dataset():
            with dataset_engine.connect() as conn:
                datasets = DatasetRepo.list(conn)

                # Get all dataset sources
                dataset_sources = [
                    get_wise_project_h5dataset(project_id, str(x.id)) for x in datasets
                ]

            output = get_wise_project_virtual_h5dataset(project_id)
            concat_h5datasets(dataset_sources, n_dim, output)

        # Function to transform images and metadata into required formats
        def process(batch: List[Tuple[Image.Image, ImageMetadata]]):
            images, metadata = zip(*batch)
            features = extract_features(images)
            thumbs = [generate_thumbnail(x) for x in images]

            return metadata, features, thumbs

        def create_dataset(dataset: Union[Path, str]):
            # Add dataset to table and get dataset id
            payload = (
                DatasetCreate(
                    location=str(dataset.resolve()), type=DatasetType.IMAGE_DIR
                )
                if isinstance(dataset, Path)
                else DatasetCreate(
                    location=(
                        str(x.resolve()) if (x := Path(dataset)).is_file() else dataset
                    ),
                    type=DatasetType.WEBDATASET,
                )
            )
            with dataset_engine.begin() as conn:
                dataset_obj = DatasetRepo.create(
                    conn,
                    data=payload,
                )
                return dataset_obj

        # Function to handle data source
        # Iterate -> Batch -> Transform -> Write
        def handle_dataset(
            dataset: Dataset, offset: int = 0, error_handler=lambda sample: None
        ):
            # Get dataset path
            features_path = get_wise_project_h5dataset(project_id, str(dataset.id))

            # Get dataset iterator
            dataset_iterator = get_dataset_iterator(dataset, error_handler)

            # Keep track of records written in this dataset
            count_ = 0
            with dataset_engine.connect() as conn, get_h5writer(
                features_path,
                list(H5Datasets),
                model_name=model_name,
                n_dim=n_dim,
                mode="w",
            ) as writer_fn, tqdm() as pbar:
                for batch in batched(dataset_iterator, batch_size):
                    metadata, features, thumbs = process(batch)

                    row_count = len(batch)
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
                        writer_fn(features, metadata_row_ids, thumbs)
                    count_ += row_count

                    # Udpate progress bar
                    pbar.update(row_count)
            # Return updated offset
            return offset + count_

        failures_path = f"wise_failedsamples_{project_id}.tar"

        with dataset_engine.connect() as conn, wds.TarWriter(
            failures_path, encoder=False
        ) as sink:

            def handle_failed_sample(sample: Dict[str, Any]):
                sink.write(sample)

            count_ = 0
            for dataset in input_datasets:
                logger.info(f"Processing - {dataset}")
                dataset_obj = None
                try:
                    # Add data source to table
                    dataset_obj = create_dataset(dataset)

                    # Process each item in data source and write to table + hdf5
                    count_ = handle_dataset(
                        dataset_obj, offset=count_, error_handler=handle_failed_sample
                    )

                except Exception as e:
                    logger.error(
                        f'Error while processing data source "{dataset}" - {e}'
                    )
                    # Delete the data source, truncate h5, and add to failed summary.
                    if dataset_obj:
                        # Delete dataset
                        delete_wise_project_h5dataset(project_id, str(dataset_obj.id))

                        # Delete the Dataset -> Which deletes all metadata
                        # Later, we can just continue from where we left off instead of deleting.
                        with conn.begin():
                            DatasetRepo.delete(conn, dataset_obj.id)

                        failed_datasources.append(dataset)
                    if not continue_on_error:
                        raise e

            logger.info("Creating Virtual Dataset across all sources")
            create_virtual_dataset()
            logger.info("Done")
    except Exception:
        logger.exception(f"Initialising project {project_id} failed!")
        delete_project = typer.confirm("Delete associated project files?", default=True)
        if not delete_project:
            logger.error(
                f'Files for "{project_id}" left in {get_wise_project_folder(project_id)}'
            )
            return typer.Exit(1)

        with engine.begin() as conn:
            WiseProjectsRepo.delete(conn, project_id)

        delete_wise_project_tree(project_id)

        return typer.Exit(1)

    if failed_datasources:
        logger.error(
            f'Project "{project_id}" was created, but the following data sources failed'
        )
        logger.error("\n\t".join(failed_datasources))

    typer.Exit(0)


@app.command()
def search(
    project_id: str = typer.Argument(..., help="Name of the project"),
    top_k: int = typer.Option(5, help="Top-k results to retrieve"),
    prefix: str = typer.Option(
        "This is a photo of a", help="Prefix to attach to all natural language queries"
    ),
    batch_size: int = typer.Option(1, help="Batch size to extract features"),
    queries: List[str] = typer.Argument(
        ...,
        help="Search query/queries. Can be a text (natural language) query, a path to an image, or a path to a directory of images",
    ),
):
    query_type, parsed_queries = parse_query_parameter(queries)
    engine = db.init(get_wise_db_uri(), echo=app_state["verbose"])
    with engine.connect() as conn:
        project = WiseProjectsRepo.get(conn, project_id)
        if project is None:
            raise typer.BadParameter(f"Project {project_id} not found!")

    vds_path = get_wise_project_virtual_h5dataset(project_id)
    model_name = get_model_name(vds_path)
    counts = get_counts(vds_path)

    assert counts[H5Datasets.FEATURES] == counts[H5Datasets.IDS]
    num_files = counts[H5Datasets.FEATURES]

    reader = get_h5iterator(vds_path)
    all_features = lambda: reader(H5Datasets.FEATURES)

    _, extract_image_features, extract_text_features = setup_clip(model_name)
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

    else:
        # Classification query
        def process(query_images_folder: Path):
            for batch in batched(
                get_valid_images_from_folder(query_images_folder, lambda _: None),
                batch_size,
            ):

                images, _ = zip(*batch)
                features = extract_image_features(images)

                yield features

        parsed_queries = typing.cast(Path, parsed_queries)
        query_image_features = np.concatenate(list(process(parsed_queries)), axis=0)
        # TODO Classification query loads all arrays into memory. need to refactor
        features = np.concatenate(list(all_features()), axis=0)
        dist, ids = classification_based_query(features, query_image_features, top_k)

    if ids is not None:
        project_engine = db.init_project(
            get_wise_project_db_uri(project_id), echo=app_state["verbose"]
        )
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
    project_id: Optional[str] = typer.Argument(None, help="Name of the project"),
):
    from api import serve

    serve(project_id)


if __name__ == "__main__":
    app()
