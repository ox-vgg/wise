import enum
from pathlib import Path
from typing import List, Union
import typer
from src.ioutils import write_dataset
from src.inference import setup_clip, AVAILABLE_MODELS
from src.schemas import URL
import urllib
from torch.hub import download_url_to_file
import numpy as np
import os

from src.search import (
    build_search_index,
    prepare_search,
    similarity_based_query,
    classification_based_query
)
from api import main

app = typer.Typer()


class _CLIPModel(str, enum.Enum):
    pass


CLIPModel = _CLIPModel("CLIPModel", {x: x for x in AVAILABLE_MODELS})


def parse_query_parameter(queries: List[str]) -> Union[List[str], Path]:
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
        return queries

    # queries has 1 element
    query = queries[0]

    # query is url
    if query.startswith(('http://', 'https://')):
        return URL(query)

    if query.replace(" ", "").isalpha():
        return queries

    qpath = Path(query)

    if qpath.is_dir() or qpath.is_file():
        return qpath

    raise typer.BadParameter("Unknown query parameter, expected List[str] | Path")


@app.command()
def extract_features(
    batch_size: int = typer.Option(
        1,
        help="Batch size that would fit your RAM/GPURAM",
    ),
    model: CLIPModel = typer.Option("ViT-B/32", help="CLIP Model to use"),  # type: ignore
    images_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory of images to index",
    ),
    save_to: Path = typer.Argument(
        ..., exists=False, help="File to write the extracted features to"
    ),
):
    if save_to.is_file():
        typer.confirm(
            "Are you sure you want to overwrite existing dataset?",
            abort=True,
            default=True,
        )
        print(f"Overwriting - {save_to}")

    files = sorted([x for x in images_dir.rglob("*") if x.is_file()])
    num_files = len(files)

    print(
        f'Processing: {num_files} files, ({[str(x) for x in files[:min(num_files, 10)]]}{"..." if num_files > 10 else ""})'
    )

    model_name = model.value

    extract_features, _ = setup_clip(model_name)

    features = extract_features(files, batch_size=batch_size)
    files_rel = [x.relative_to(images_dir) for x in files]
    write_dataset(save_to, features, files_rel, model_name)

    typer.Exit(0)


@app.command()
def search(
    dataset: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Extracted features .npy file",
    ),
    top_k: int = typer.Option(5, help="Top-k results to retrieve"),
    prefix: str = typer.Option(
        "This is a photo of a", help="Prefix to attach to all natural language queries"
    ),
    queries: List[str] = typer.Argument(..., help="Search query/queries. Can be a text (natural language) query, a path to an image, or a path to a directory of images"),
):
    parsed_queries = parse_query_parameter(queries)

    if isinstance(parsed_queries, list) and all(isinstance(x, str) for x in parsed_queries):
        query_type = 'NATURAL_LANGUAGE_QUERY'
    elif isinstance(parsed_queries, Path) or isinstance(parsed_queries, URL):
        if isinstance(parsed_queries, URL):
            print('Downloading', parsed_queries, 'to file')
            filename = Path(urllib.parse.unquote(parsed_queries).split('?')[0]).name
            download_url_to_file(parsed_queries, filename)
            parsed_queries = Path(filename)
            query_type = 'IMAGE_QUERY'
        elif os.path.isfile(parsed_queries):
            query_type = 'IMAGE_QUERY'
        elif os.path.isdir(parsed_queries):
            query_type = 'IMAGE_CLASSIFICATION_QUERY'
        else:
            raise ValueError("The query provided is neither a file nor a directory")
    else:
        raise NotImplementedError

    features, index, model_name, files, extract_image_features, extract_text_features = prepare_search(dataset, query_type)
    top_k = min(top_k, len(files))
    if query_type == 'NATURAL_LANGUAGE_QUERY' or query_type == 'IMAGE_QUERY':
        dist, ids = similarity_based_query(index, extract_image_features, extract_text_features, top_k, prefix, parsed_queries, query_type)
        print(dist, [[Path(files[x]).name for x in top_ids] for top_ids in ids])
    else: # query_type ==  IMAGE_CLASSIFICATION_QUERY
        scores, ids = classification_based_query(features, extract_image_features, top_k, parsed_queries)
        print(scores, [Path(files[x]).name for x in ids])



@app.command()
def serve():
    main()


if __name__ == "__main__":
    app()
