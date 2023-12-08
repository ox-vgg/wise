import enum
import logging
from pathlib import Path
from typing import Iterator
import numpy as np
import torch
from tqdm import tqdm
import faiss
from faiss.contrib.exhaustive_search import knn

from .enums import IndexType
from .inference import LinearBinaryClassifier

logger = logging.getLogger(__name__)


def write_index(index, path: Path):
    faiss.write_index(index, str(path))


def read_index(path: Path, readonly: bool = False, memory_mapped: bool = False):
    flags = 0
    flags = (flags | faiss.IO_FLAG_READ_ONLY) if readonly else flags
    flags = (flags | faiss.IO_FLAG_MMAP) if memory_mapped else flags

    return faiss.read_index(str(path), flags)


def get_index(type: IndexType, n_dim: int, *args):
    index = faiss.IndexFlatIP(n_dim)
    if type == IndexType.IndexIVFFlat:
        quantizer = index
        index = faiss.IndexIVFFlat(quantizer, n_dim, *args)
    if type == IndexType.IndexIVFPQ:
        quantizer = index
        if len(args) < 3:
            raise ValueError("Insufficient args passed to create a IVFPQ index")
        nlist, m, nbits = args[:3]
        index_ivfpq = faiss.IndexIVFPQ(quantizer, n_dim, nlist, m, nbits)
        opq_matrix = faiss.OPQMatrix(n_dim, m)

        index = faiss.IndexPreTransform(opq_matrix, index_ivfpq)
    return index


def build_search_index(features: np.ndarray) -> faiss.IndexFlatIP:
    n_dim = features.shape[-1]
    logger.info(f"Building faiss index ({n_dim})")
    index = faiss.IndexFlatIP(n_dim)
    index.add(features)
    logger.info(f"Index built.")
    return index


def search_dataset(index, query, top_k):
    dist, ids = index.search(x=query, k=top_k)
    return dist, ids


def brute_force_search(
    db_iterator: Iterator[np.ndarray], queries: np.ndarray, top_k: int = 3
):
    """
    Function that bypasses indexing to avoid double copy
    in index, in cases of IndexFlat structures as there
    is no clustering / quantisation etc.

    Allows for searching large datasets using iterators
    instead of loading it all at once. Combines the results
    using faiss ResultHeap Datastructure.

    See [Faiss Wiki](https://github.com/facebookresearch/faiss/wiki/Brute-force-search-without-an-index) \
        and [Faiss Tests](https://github.com/facebookresearch/faiss/blob/72f52af6e4ecf485fd127cbf838b4339d265e0fe/tests/test_contrib.py#L110)

    """

    nqueries = queries.shape[0]

    # Keep_max is true since we are looking at Inner Product
    result = faiss.ResultHeap(nqueries, k=top_k, keep_max=True)

    # We are not re-using the functions above since
    # we don't want to build the index
    offset = 0
    for batch in db_iterator:
        nbatch = batch.shape[0]

        D, I = knn(queries, batch, top_k, metric=faiss.METRIC_INNER_PRODUCT)
        I += offset

        result.add_result(D, I)
        offset += nbatch

    result.finalize()
    return result.D, result.I


def classification_based_query(
    features: np.ndarray, query_features, top_k: int, num_training_iterations: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Process a classification-based query, where a binary classifier is trained on a set of query images,
    and is used to determine which images in the dataset belong to the same category as the query images
    """

    # Define classifier
    linear_classifier = LinearBinaryClassifier(embedding_dim=features.shape[1])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(linear_classifier.parameters(), lr=5)

    # Train classifier
    logger.info("Training classifier")
    for i in tqdm(range(num_training_iterations)):
        negative_features = features[
            np.random.randint(features.shape[0], size=32), :
        ]  # select 32 random examples from the features array
        neg_batch_len, pos_batch_len = len(negative_features), len(query_features)
        batch_features = torch.concat(
            [torch.tensor(negative_features), torch.tensor(query_features)]
        )

        optimizer.zero_grad()
        logits = linear_classifier(batch_features)
        labels = torch.concat(
            [torch.zeros(neg_batch_len), torch.ones(pos_batch_len)]
        ).unsqueeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        logger.info(f"Step {i+1}/{num_training_iterations} - loss: {loss.item():.5f}")

    logger.info("Computing scores using classifier")
    # Use the classifier to compute a score (between 0-1) for all images in the dataset
    with torch.no_grad():
        scores = linear_classifier(torch.tensor(features, dtype=torch.float32))
        scores = scores.squeeze()  # output: 1-D vector
        scores, ids = torch.topk(scores, k=top_k)
    scores, ids = scores.cpu().detach().numpy(), ids.cpu().detach().numpy()
    return scores, ids
