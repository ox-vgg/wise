import enum
from pathlib import Path
from typing import List, Union
import typer
from src.ioutils import write_dataset, read_dataset
from src.inference import setup_clip, AVAILABLE_MODELS, LinearBinaryClassifier
from src.schemas import URL
import urllib
import torch
import numpy as np
import os
from tqdm import tqdm

from src.search import (
    build_search_index,
    search_dataset,
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

def similarity_based_query(dataset: Path, features: np.ndarray, files: List[str], model_name: str,
    top_k: int, prefix: str, parsed_queries: Union[List[str], Path], query_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Process a similarity-based query (could be either a natural language query or a nearest neighbor image query)"""
    index = build_search_index(features)

    # Convert query to embedding
    extract_image_features, extract_text_features = setup_clip(model_name)

    if query_type == 'NATURAL_LANGUAGE_QUERY':
        prefixed_queries = [f"{prefix.strip()} {x.strip()}".strip() for x in parsed_queries]
        print("Processing queries:\n", prefixed_queries)
        query_features = extract_text_features(prefixed_queries)
    elif query_type == 'IMAGE_QUERY':
        print("Processing image query")
        query_features = extract_image_features(parsed_queries)
        query_features = next(query_features)

    dist, ids = search_dataset(index, query_features, top_k=top_k)
    return dist, ids
    
def classification_based_query(dataset: Path, features: np.ndarray, files: List[str], model_name: str,
    top_k: int, query_images_directory: Path, num_training_iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Process a classification-based query, where a binary classifier is trained on a set of query images,
    and is used to determine which images in the dataset belong to the same category as the query images"""
    extract_image_features, _ = setup_clip(model_name)
    query_images_filenames = [os.path.join(query_images_directory, filename) for filename in os.listdir(query_images_directory)]
    query_images_features = extract_image_features(query_images_filenames, batch_size = 16)
    query_images_features = np.concatenate(list(query_images_features))

    # Define classifier
    linear_classifier = LinearBinaryClassifier(embedding_dim=features.shape[1])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(linear_classifier.parameters(), lr=5)

    # Train classifier
    print("Training classifier")
    for i in tqdm(range(num_training_iterations)):
        negative_features = features[np.random.randint(features.shape[0], size=32), :] # select 32 random examples from the features array
        neg_batch_len, pos_batch_len = len(negative_features), len(query_images_features)
        batch_features = torch.concat([torch.tensor(negative_features), torch.tensor(query_images_features)])

        optimizer.zero_grad()
        logits = linear_classifier(batch_features)
        labels = torch.concat([torch.zeros(neg_batch_len), torch.ones(pos_batch_len)]).unsqueeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f'Step {i+1}/{num_training_iterations} - loss: {loss.item():.5f}')

    print('Computing scores using classifier')
    # Use the classifier to compute a score (between 0-1) for all images in the dataset
    scores = linear_classifier(torch.tensor(features, dtype=torch.float32))
    scores = scores.squeeze() # output: 1-D vector
    scores, ids = torch.topk(scores, k=top_k)
    scores, ids = scores.cpu().detach().numpy(), ids.cpu().detach().numpy()
    return scores, ids


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
            torch.hub.download_url_to_file(parsed_queries, filename)
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

    features, files, model_name = read_dataset(dataset)
    top_k = min(top_k, len(files))
    if query_type == 'NATURAL_LANGUAGE_QUERY' or query_type == 'IMAGE_QUERY':
        dist, ids = similarity_based_query(dataset, features, files, model_name, top_k, prefix, parsed_queries, query_type)
        print(dist, [[Path(files[x]).name for x in top_ids] for top_ids in ids])
    else: # query_type ==  IMAGE_CLASSIFICATION_QUERY
        scores, ids = classification_based_query(dataset, features, files, model_name, top_k, parsed_queries)
        print(scores, [Path(files[x]).name for x in ids])



@app.command()
def serve():
    main()


if __name__ == "__main__":
    app()
