import os
import numpy as np
from pathlib import Path
import glob
import random

from ...utils import batched
from .feature_store import FeatureStore

class NumpySaveStore(FeatureStore):
    def __init__(self, store_name, store_data_dir):
        """
        Store all data in the specified directory as numpy .npy binary file
        """
        self.store_name = store_name
        self.store_data_dir = Path(store_data_dir)
        self.vector_id_to_shard_location = None

        self._npz_filenames = []
        self._reset()
        # by default enable read
        self.enable_read()

    @property
    def npz_filenames(self):
        return self._npz_filenames

    def _get_current_npz_filenames(self):
        npz_pattern = self.store_data_dir / (self.store_name + '-*.npz')
        return sorted(glob.iglob(pathname=npz_pattern.as_posix(), recursive=False))

    def _reset(self):

        # compute feature dimension and feature count
        self._feature_count = 0
        self.feature_dim = None
        self._npz_filenames = self._get_current_npz_filenames()

        for npz_filename in self.npz_filenames:
            payload = np.load(npz_filename)
            feature_id_list = payload['feature_id']
            self._feature_count += feature_id_list.shape[0]
            features_list = payload['features']

            if len(features_list[0].shape) == 1:
                _feature_dim = features_list[0].shape[0]
            elif len(features_list[0].shape) == 2:
                _feature_dim = features_list[0].shape[1]
            else:
                raise ValueError(f'unrecognized feature shape {features_list[0].shape}')

            if self.feature_dim is None:
                self.feature_dim = _feature_dim
            elif self.feature_dim != _feature_dim:
                raise ValueError(
                    f'Stored features have different feature dimensions in shard - {npz_filename} '
                    f'(Expected: {self.feature_dim}, Got: {_feature_dim}) '
                    '- this project is likely to be corrupt'
                )

        self.current_shard_index = len(self.npz_filenames) # the next shard to save
        self.shard_feature_index = 0    

    @property
    def feature_count(self):
        return self._feature_count + self.shard_feature_index

    def enable_write(self, shard_maxcount = 100_000, shard_maxsize = 3 * 1024 ** 3, verbose=0, overwrite: bool = False):
        if shard_maxcount < 1:
            raise ValueError('shard max count must be positive integer')

        if shard_maxsize < 1 and shard_maxsize != -1:
            raise ValueError('shard max size must be set to positive integer (or) -1')

        self.shard_maxcount = int(shard_maxcount)
        self.shard_maxsize = int(shard_maxsize)
        self.verbose = verbose

        if overwrite:
            # delete
            for npz_filename in self.npz_filenames:
                Path(npz_filename).unlink(missing_ok=True)

            # reset internal state
            self._reset()

        if self.feature_dim is not None:
            self.shard_features = np.ndarray((self.shard_maxcount, self.feature_dim),
                                                dtype=np.float32)
            self.shard_feature_id = np.ndarray((self.shard_maxcount), dtype=np.int32)

    def enable_read(self, shard_shuffle=False, shuffle_values=False, shuffle_bufsize=10000):
        self.shard_shuffle = shard_shuffle
        self.shuffle_values = shuffle_values
        self.shuffle_bufsize = shuffle_bufsize        

    def add(self, id, features):
        if self.feature_dim is None:
            self.feature_dim = features.shape[1]
            self.shard_features = np.ndarray((self.shard_maxcount, self.feature_dim),
                                             dtype=np.float32)
            self.shard_feature_id = np.ndarray((self.shard_maxcount), dtype=np.int32)

            self.shard_feature_index = 0
            self.current_shard_index = 0
        if self.feature_dim != features.shape[1]:
            raise ValueError(f'feature dimension cannot change and must be {self.feature_dim}')
        if features.shape[0] != 1:
            raise ValueError(f'cannot add {features.shape[0]} features, only one feature can be added at a time')

        if self.shard_feature_index == self.shard_maxcount:
            # create a new shard
            self.save_current_shard()
            self.add(id, features)
        else:
            self.shard_features[self.shard_feature_index] = features
            self.shard_feature_id[self.shard_feature_index] = id
            self.shard_feature_index += 1

    def save_current_shard(self):
        if self.shard_feature_index:
            current_shard_id = f'{self.store_name}-{self.current_shard_index:06d}'
            current_shard_filename = self.store_data_dir / current_shard_id
            np.savez(
                current_shard_filename,
                feature_id=self.shard_feature_id[:self.shard_feature_index],
                features=self.shard_features[:self.shard_feature_index]
            )
            if self.verbose:
                print(f'saved {self.shard_feature_index} features to shard {current_shard_filename}')

            self._feature_count += self.shard_feature_index
            self.shard_feature_index = 0
            self.current_shard_index += 1
            self._npz_filenames.append(current_shard_filename)

    def __iter__(self):
        for feature_ids, feature_vectors in self.iter_batch(batch_size=1):
            # feature_ids[0] is of type np.int32 and feature_vectors is a numpy array of shape (1, feature_dim)
            yield feature_ids[0], feature_vectors

    def iter_batch(self, batch_size=512):
        # TODO: the shuffling and batching needs to be improved.
        # reservoir shuffling can be used
        # batching can be done across files when end of a shard is reached
        # in fact, the batch can be moved outside of this class as batched(instance) if __iter__ method
        # is implemented properly

        _file_list = self.npz_filenames.copy()
        if self.shard_shuffle:
            _file_list = random.shuffle(_file_list)

        for npz_filename in _file_list:
            payload = np.load(npz_filename)
            feature_ids_array = payload['feature_id']
            features_array = payload['features']
            N = feature_ids_array.shape[0]
            index_list = range(0, N)
            if self.shuffle_values:
                random.shuffle(index_list)
            for batch_indices in batched(index_list, batch_size):
                feature_ids = feature_ids_array[batch_indices]
                feature_vectors = features_array[batch_indices,:] # shape: (batch_size, feature_dim)
                yield feature_ids, feature_vectors

    def enable_random_access(self):
        """
        Enables random access to the NumpySaveStore for the internal search feature.
        Once enabled, feature vectors can be accessed directly using their vector id.

        Usage example:
        ```
        store = NumpySaveStore(...)
        store.enable_read()
        store.enable_random_access()
        
        # Access a feature vector with an id of 123
        vector = store[123]
        ```
        """
        # a dictionary with key: vector id and value: tuple(shard filename, array index within the shard)
        vector_id_to_shard_location: dict[int, tuple[str, int]] = {}

        for npz_filename in self.npz_filenames:
            payload = np.load(npz_filename)
            feature_ids_array = payload['feature_id'] # shape: (2048,)
            for array_index, feature_id in enumerate(feature_ids_array):
                vector_id_to_shard_location[feature_id] = (npz_filename, array_index)
        self.vector_id_to_shard_location = vector_id_to_shard_location

    def __getitem__(self, vector_id: int) -> np.ndarray:
        """
        Access a feature vector with its vector id. The `enable_random_access()`
        method needs to be called first in order to enable this.

        Usage example:
        ```
        store = NumpySaveStore(...)
        store.enable_read()
        store.enable_random_access()
        
        # Access a feature vector with an id of 123
        vector = store[123]
        ```
        """
        if not self.vector_id_to_shard_location:
            raise Exception("Please run `store.enable_random_access()` on this feature store first")
        npz_filename, array_index = self.vector_id_to_shard_location[vector_id]
        payload = np.load(npz_filename, mmap_mode="r")
        feature_vector = payload['features'][[array_index]] # shape: (1, feature_dim)
        return feature_vector

    def close(self):
        self.save_current_shard()

    def __del__(self):
        if hasattr(self, 'shard_feature_index'):
            if self.shard_feature_index != 0:
                self.close()
