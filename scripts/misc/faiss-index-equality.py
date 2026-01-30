#!/usr/bin/env python3

## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import faiss
import numpy as np
import argparse

def index_to_array(index):
    """Extract all vectors from a FAISS index into a numpy array."""
    ntotal = index.ntotal
    xb = []
    for i in range(ntotal):
        xb.append(index.reconstruct(index.id_map.at(i)))
    return np.array(xb)

def main():
    parser = argparse.ArgumentParser(
        description="Check if two FAISS indices contain the same feature vectors"
    )
    parser.add_argument("index1", help="Path to first FAISS index file")
    parser.add_argument("index2", help="Path to second FAISS index file")
    args = parser.parse_args()

    # Load both indices
    index1 = faiss.read_index(args.index1)
    index2 = faiss.read_index(args.index2)

    # Convert both to arrays
    vecs1 = index_to_array(index1)
    vecs2 = index_to_array(index2)

    print(f"Index1 vectors: {vecs1.shape}, Index2 vectors: {vecs2.shape}")

    # Check if same number of vectors
    if vecs1.shape != vecs2.shape:
        print("Indices have different sizes.")
        return

    # Sort rows for consistent order before comparison
    vecs1_sorted = np.array(sorted(map(tuple, vecs1)))
    vecs2_sorted = np.array(sorted(map(tuple, vecs2)))

    # Compare with tolerance
    if np.allclose(vecs1_sorted, vecs2_sorted, rtol=1e-5, atol=1e-7):
        print("Indices contain the same set of feature vectors.")
    else:
        print("Indices differ in their feature vectors.")

if __name__ == "__main__":
    main()
