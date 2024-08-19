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

import functools
from pathlib import Path
import glob
import random
import logging

from .feature_store import FeatureStore
from ...utils import batched

import numpy as np
import faiss

logger = logging.getLogger(__name__)

MAX_CACHE_SIZE = 32


def load_faiss_index(filename: str) -> faiss.Index:
    return faiss.read_index(filename, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)


class FaissStore(FeatureStore):
    """
    Feature store that uses Faiss for storing and retrieving feature vectors.

    When writing, creates a new Flat Index + IDMap and appends feature vectors to it.

    A new shard is created when the current shard contains shard_maxcount entries or
    when the close method is called or when object goes out of scope (__del__)
    """
    EXTENSION = "faiss"

    def __init__(self, store_name: str, store_data_dir: str):
        # Initialize Faiss-specific components here
        # e.g., self.index = faiss.IndexFlatL2(dimension)
        self.store_name = store_name
        self.store_data_dir = store_data_dir

        self._prefix = str(Path(self.store_data_dir) / f"{self.store_name}-")
        self._pattern = self._prefix + '%06d.' + self.EXTENSION

        logger.debug(
            f"FaissStore: store_name={self.store_name}, store_data_dir={self.store_data_dir}, pattern={self._pattern}"
        )

        self.load_faiss_index = functools.lru_cache(maxsize=MAX_CACHE_SIZE)(
            load_faiss_index
        )
        self._reset()
        self.enable_read()

    def _reset(self):
        _vector_id_to_shard_location: dict[int, str] = {}
        self.load_faiss_index.cache_clear()

        self._filenames = self._get_current_filenames()
        self._current_shard_idx = len(self._filenames)
        self._current_shard = None
        self._count = 0
        self._dim = None

        for filename in self._filenames:
            index = load_faiss_index(filename)
            N = index.ntotal
            self._count += N
            if self._dim is None:
                self._dim = index.d
            elif self._dim != index.d:
                raise ValueError(f'Stored features have different feature dimensions in shard - {filename} '
                                 f'(Expected: {self._dim}, Got: {index.d}) '
                                 '- this project is likely to be corrupt')

            feature_ids = faiss.vector_to_array(index.id_map)
            for feature_id in feature_ids:
                _vector_id_to_shard_location[int(feature_id)] = filename

        self._vector_id_to_shard_location = _vector_id_to_shard_location

        if self._dim is not None:
            self._current_shard = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))

        logger.debug("FaissStore reset: count=%d, dim=%s", self._count, self._dim)

    def _get_current_filenames(self):
        pattern = f"{self._prefix}*.{self.EXTENSION}"
        return sorted(glob.iglob(pathname=pattern, recursive=False))

    @property
    def filenames(self):
        return self._filenames

    @property
    def feature_count(self):
        ntotal = self._count

        if self._current_shard is not None:
            ntotal += self._current_shard.ntotal

        return ntotal

    @property
    def feature_dim(self):
        return self._dim

    def enable_write(self, shard_maxcount=1e6, overwrite: bool = False):
        # At around 1 million vectors of 1024 dimensions, the file size is around 4GB
        # Enable write mode for the Faiss index
        if overwrite:
            logger.info("Overwriting existing Faiss store")
            for filename in self._filenames:
                Path(filename).unlink(missing_ok=True)
            self._reset()

        if shard_maxcount < 1:
            raise ValueError("shard max count must be positive integer")

        self.shard_maxcount = int(shard_maxcount)

    def enable_read(self, shard_shuffle=False, shuffle_values=False, shuffle_bufsize=10000):
        # TODO - handle shuffle parameters
        self.shard_shuffle = shard_shuffle
        self.shuffle_values = shuffle_values
        self.shuffle_bufsize = shuffle_bufsize

        logger.debug(
            "FaissStore read: shard_shuffle=%s, shuffle_values=%s, shuffle_bufsize=%d",
            shard_shuffle,
            shuffle_values,
            shuffle_bufsize,
        )

    def save_current_shard(self):
        if self._current_shard is None or self._current_shard.ntotal == 0:
            return

        filename = self._pattern % self._current_shard_idx
        logger.info(
            f"Saving active shard to {filename} with {self._current_shard.ntotal} vectors"
        )
        faiss.write_index(self._current_shard, filename)

        feature_ids = faiss.vector_to_array(self._current_shard.id_map)
        for feature_id in feature_ids:
            self._vector_id_to_shard_location[int(feature_id)] = filename

        self._filenames.append(filename)
        self._current_shard_idx += 1
        self._count += self._current_shard.ntotal

        self._current_shard = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))

        logger.debug("FaissStore: count=%d", self.feature_count)

    def add(self, _ids: int | list[int], features: np.ndarray):
        """
        Add features with associated IDs to the Faiss index.
        Expects features to be 1 x N and id to be a single integer, or
        features to be M x N and id to be a list of M integers.
        """

        if isinstance(_ids, int):
            return self.add([_ids], features)

        if not (isinstance(_ids, list) and all(isinstance(i, int) for i in _ids)):
            raise ValueError("ID must be an integer or a list of integers")

        if len(_ids) == 0:
            raise ValueError("ID list cannot be empty")

        if len(features.shape) != 2:
            raise ValueError(f"Features must be a 2D array (Got: {features.shape})")

        if features.shape[0] != len(_ids):
            raise ValueError(
                f"Feature count and ID count mismatch (Features: {features.shape[0]}, IDs: {len(_ids)})"
            )

        # Add features to the Faiss index
        if self.feature_dim is None:
            self._dim = features.shape[1]
            self._current_shard = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))

        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch (Expected: {self.feature_dim}, Got: {features.shape[1]})"
            )

        if self._current_shard.ntotal + features.shape[0] <= self.shard_maxcount:
            ids_array = np.array(_ids, dtype=np.int64)
            self._current_shard.add_with_ids(features, ids_array)
            # Update vector ID to shard location mapping
            for feature_id in _ids:
                self._vector_id_to_shard_location[int(feature_id)] = "current"
            return

        # add upto maxcount and rollover
        space_left = self.shard_maxcount - self._current_shard.ntotal
        if space_left > 0:
            self.add(_ids[:space_left], features[:space_left])

        self.save_current_shard()
        self.add(_ids[space_left:], features[space_left:])

    def __iter__(self):
        # Iterate over the Faiss index
        for feature_ids, features in self.iter_batch():
            yield from zip(feature_ids, np.vsplit(features, features.shape[0]))

    def iter_batch(self, batch_size = 512):
        # Iterate over the Faiss index in batches
        _filelist = self.filenames.copy()

        if self._current_shard is not None and self._current_shard.ntotal > 0:
            _filelist.append("current")

        if self.shard_shuffle:
            random.shuffle(_filelist)

        for filename in _filelist:
            if filename == "current":
                index = self._current_shard
            else:
                index = load_faiss_index(filename)

            N = index.ntotal
            feature_ids = faiss.vector_to_array(index.id_map)
            features = index.reconstruct_batch(feature_ids)
            index_list = list(range(0, N))

            if self.shuffle_values:
                random.shuffle(index_list)

            for batch_indices in batched(index_list, batch_size):
                batch_feature_ids = feature_ids[batch_indices]
                batch_features = features[batch_indices, :]
                yield batch_feature_ids, batch_features

    def __getitem__(self, id: int):
        # Random access to features by ID
        if (
            id not in self._vector_id_to_shard_location
            or self._vector_id_to_shard_location[id] == "current"
        ):
            # try the current shard
            try:
                feature_vector = self._current_shard.reconstruct(id)
                return feature_vector
            except:
                raise KeyError(f'Feature ID {id} not found in the store')

        filename = self._vector_id_to_shard_location[id]

        # Access through the cached load_faiss_index function
        index = self.load_faiss_index(filename)
        feature_vector = index.reconstruct_n(id, 1)
        return feature_vector

    def close(self):
        # Close any resources if necessary
        self.save_current_shard()
        self.load_faiss_index.cache_clear()

    def __del__(self):
        self.close()
