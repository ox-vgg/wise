import numpy as np
from pathlib import Path
from typing import List, Union
import os
import torch
from tqdm import tqdm
import faiss
from src.ioutils import read_dataset
from src.inference import setup_clip, LinearBinaryClassifier


def build_search_index(features: np.ndarray) -> faiss.IndexFlatIP:
    n_dim = features.shape[-1]
    print(f"Building faiss index ({n_dim})")
    index = faiss.IndexFlatIP(n_dim)
    index.add(features)
    print(f"Index built.")
    return index


def prepare_search(dataset: Path, query_type: str = None):
    features, files, model_name = read_dataset(dataset)

    extract_image_features, extract_text_features = setup_clip(model_name)

    if query_type == 'IMAGE_CLASSIFICATION_QUERY':
        index = None
    else:
        index = build_search_index(features)
    
    return features, index, model_name, files, extract_image_features, extract_text_features


def similarity_based_query(index: faiss.IndexFlatIP, extract_image_features, extract_text_features,
    top_k: int, prefix: str, parsed_queries: Union[List[str], Path], query_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Process a similarity-based query (could be either a natural language query or a nearest neighbor image query)"""
    # Convert query to embedding
    if query_type == 'NATURAL_LANGUAGE_QUERY':
        prefixed_queries = [f"{prefix.strip()} {x.strip()}".strip() for x in parsed_queries]
        print("Processing queries:\n", prefixed_queries)
        query_features = extract_text_features(prefixed_queries)
    elif query_type == 'IMAGE_QUERY':
        print("Processing image query")
        query_features = extract_image_features(parsed_queries)
        query_features = next(query_features)

    dist, ids = index.search(x=query_features, k=top_k)
    return dist, ids


def classification_based_query(features: np.ndarray, extract_image_features,
    top_k: int, query_images_directory: Path, num_training_iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Process a classification-based query, where a binary classifier is trained on a set of query images,
    and is used to determine which images in the dataset belong to the same category as the query images"""
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