import enum
import itertools
from pathlib import Path

import shutil
import typing
from typing import List, Union, Tuple, Literal, Optional
from tempfile import NamedTemporaryFile

import typer
from src.ioutils import write_dataset
from src.inference import setup_clip, AVAILABLE_MODELS
from torch.hub import download_url_to_file

from src.ioutils import (
    write_dataset,
    get_dataset_reader,
    get_thumbs_writer,
    get_valid_images_from_folder,
    get_valid_images_from_webdataset,
)

import numpy as np
from PIL import Image
import braceexpand
from src import db
from src.data_models import Dataset, DatasetType, URL, QueryType, Project, DatasetCreate
from src.inference import setup_clip, AVAILABLE_MODELS
from src.utils import batched
from src.search import brute_force_search, classification_based_query
from src.repository import WiseProjectsRepo, DatasetRepo, MetadataRepo
from src.projects import (
    get_wise_db_uri,
    get_wise_project_db_uri,
    get_wise_folder,
    get_wise_project_folder,
    create_wise_project_tree,
    get_wise_features_dataset_path,
    get_wise_thumbs_dataset_path,
)
from api import main


app = typer.Typer()


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


def get_dataset_iterator(dataset: Dataset):
    if dataset.type == DatasetType.WEBDATASET:
        for im, metadata in get_valid_images_from_webdataset(dataset.location):
            # add dataset id
            metadata.dataset_id = dataset.id

            # remove location from path
            metadata.path = metadata.path.replace(dataset.location, "")

            yield im, metadata

    elif dataset.type == DatasetType.IMAGE_DIR:
        for im, metadata in get_valid_images_from_folder(Path(dataset.location)):
            # add dataset id
            metadata.dataset_id = dataset.id

            yield im, metadata


# Split input arguments into respective category of folders and webdataset urls
def parse_and_validate_input_datasets(input_dataset: List[str]):
    # At least one input must be passed
    if len(input_dataset) == 0:
        raise typer.BadParameter("Input dataset cannot be empty")

    # Now we have a list[Folder | WebDataset]
    processed = []
    for dataset in input_dataset:
        if (p := Path(dataset)).is_dir():
            processed.append(p)
        else:
            processed.extend(parse_webdataset_url(dataset))

    # - deduplicate
    processed = list(dict.fromkeys(processed))

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
    project_id: str = typer.Argument(..., help="Name of the project"),
):
    # Setup
    input_datasets = parse_and_validate_input_datasets(sources)

    # Try creating project id, it will fail if not unique.
    # TODO Translate the exception
    engine = db.init(get_wise_db_uri())
    try:
        with engine.begin() as conn:
            WiseProjectsRepo.create(conn, data=Project(id=project_id))

            # Create project tree, get paths to the HDF5 files
            create_wise_project_tree(project_id)

            save_features = get_wise_features_dataset_path(
                project_id,
                "features",
                "images",
            )
            save_thumbs = get_wise_thumbs_dataset_path(project_id)
            model_name = model.value
            extract_features, _ = setup_clip(model_name)
            dataset_engine = db.init_project(get_wise_project_db_uri(project_id))
            with get_thumbs_writer(
                save_thumbs, mode="w", driver="family"
            ) as thumbs_writer, dataset_engine.begin() as conn:
                # Generator to transform images and metadata into required shape, write thumbnails
                def process(dataset_iterator):
                    for batch in batched(dataset_iterator, batch_size):

                        images, metadata = zip(*batch)
                        features = extract_features(images)
                        metadata = [MetadataRepo.create(conn, data=x) for x in metadata]
                        metadata = list(
                            map(
                                lambda x: str(x.id) if x.id is not None else x.path,
                                metadata,
                            )
                        )
                        thumbs_writer(images)

                        yield features, metadata

                for dataset in input_datasets:
                    # Add dataset to table and get dataset id

                    payload = DatasetCreate(
                        location=str(dataset),
                        type=DatasetType.IMAGE_DIR
                        if isinstance(dataset, Path)
                        else DatasetType.WEBDATASET,
                    )
                    dataset_obj = DatasetRepo.create(
                        conn,
                        data=payload,
                    )
                    write_dataset(
                        save_features,
                        process(get_dataset_iterator(dataset_obj)),
                        model_name,
                        write_size=1024,
                        mode="a",
                        driver="family",  # writes 2 gb files
                    )
    except Exception as e:
        print(e)
        project_folder = get_wise_project_folder(project_id)
        if project_folder.is_dir():
            shutil.rmtree(project_folder)
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
    engine = db.init(get_wise_db_uri())
    with engine.connect() as conn:
        project = WiseProjectsRepo.get(conn, project_id)
        if project is None:
            raise typer.BadParameter(f"Project {project_id} not found!")

    image_features = get_wise_features_dataset_path(
        project_id,
        "features",
        "images",
    )
    model_name, num_files, reader = get_dataset_reader(image_features, driver="family")
    extract_image_features, extract_text_features = setup_clip(model_name)
    file_ids = []

    def get_features():
        for feature, _file_ids in reader():
            file_ids.extend(_file_ids)
            yield feature

    top_k = min(top_k, num_files)

    ids = None
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

        dist, ids = brute_force_search(get_features(), query_features, top_k)
        print(dist, [[file_ids[x] for x in top_ids] for top_ids in ids])
    else:
        # Classification query
        def process(query_images_folder: Path):
            for batch in batched(
                get_valid_images_from_folder(query_images_folder), batch_size
            ):

                images, _ = zip(*batch)
                features = extract_image_features(images)

                yield features

        parsed_queries = typing.cast(Path, parsed_queries)
        query_image_features = np.concatenate(list(process(parsed_queries)), axis=0)
        # TODO Classification query loads all arrays into memory. need to refactor
        features = np.concatenate(list(get_features()), axis=0)
        scores, ids = classification_based_query(features, query_image_features, top_k)
        print(scores, [[file_ids[x] for x in top_ids] for top_ids in ids])

    if ids is not None:
        project_engine = db.init_project(get_wise_project_db_uri(project_id))
        with project_engine.connect() as conn:
            for query, result_ids in zip(
                (
                    parsed_queries
                    if isinstance(parsed_queries, list)
                    else [parsed_queries]
                ),
                ids,
            ):
                print(query)
                for k in result_ids:
                    # Sqlite row id when not set explicitly starts at 1
                    print(MetadataRepo.get(conn, int(k + 1)))


@app.command()
def serve(
    project_id: Optional[str] = typer.Argument(None, help="Name of the project"),
):
    main(project_id)


if __name__ == "__main__":
    app()
