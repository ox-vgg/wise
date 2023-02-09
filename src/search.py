import numpy as np
import faiss


def build_search_index(features: np.ndarray) -> faiss.IndexFlatIP:
    n_dim = features.shape[-1]
    print(f"Building faiss index ({n_dim})")
    index = faiss.IndexFlatIP(n_dim)
    index.add(features)
    print(f"Index built.")
    return index


def search_dataset(index: faiss.IndexFlatIP, features: np.ndarray, top_k: int = 3):

    dist, ids = index.search(x=features, k=top_k)

    return dist, ids
