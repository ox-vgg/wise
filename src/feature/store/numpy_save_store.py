import os
import numpy as np
from pathlib import Path
import glob
import random

from .feature_store import FeatureStore

class NumpySaveStore(FeatureStore):
    def __init__(self, store_name, store_data_dir):
        """
        Store all data in the specified directory as numpy .npy binary file
        """
        self.store_name = store_name
        self.store_data_dir = Path(store_data_dir)

    def enable_write(self, shard_maxcount, shard_maxsize, verbose=0):
        self.shard_maxcount = shard_maxcount
        self.shard_maxsize = shard_maxsize
        self.verbose = verbose
        self.current_shard_index = -1

    def enable_read(self, shard_shuffle=False, shuffle_values=False, shuffle_bufsize=10000):
        self.shard_shuffle = shard_shuffle
        self.shuffle_values = shuffle_values
        self.shuffle_bufsize = shuffle_bufsize
        self.shard_filename_list = self.store_data_dir
        npz_pattern = self.store_data_dir / (self.store_name + '-*.npz')
        self.npz_filename_list = []
        for npz_filename in glob.iglob(pathname=npz_pattern.as_posix(), recursive=False):
            self.npz_filename_list.append(npz_filename)
        if self.shard_shuffle:
            random.shuffle(self.npz_filename_list)
        else:
            self.npz_filename_list.sort()

        # compute number of features
        self.feature_count = 0
        for npz_filename in self.npz_filename_list:
            payload = np.load(npz_filename)
            feature_id_list = payload['feature_id']
            features_list = payload['features']
            self.feature_count += feature_id_list.shape[0]

        # compute feature dimension
        for npz_filename in self.npz_filename_list:
            payload = np.load(npz_filename)
            features_list = payload['features']
            if len(features_list[0].shape) == 1:
                self.feature_dim = features_list[0].shape[0]
            elif len(features_list[0].shape) == 2:
                self.feature_dim = features_list[0].shape[1]
            else:
                raise ValueError('unrecognized feature shape {features_list[0].shape}')
            break

    def add(self, id, features):
        if self.current_shard_index == -1:
            self.feature_dim = features.shape[1]
            self.shard_features = np.ndarray((self.shard_maxcount, self.feature_dim),
                                             dtype=np.float32)
            self.shard_feature_id = np.ndarray((self.shard_maxcount), dtype=np.int32)

            self.shard_feature_index = 0
            self.current_shard_index = 0
        if self.feature_dim != features.shape[1]:
            raise ValueError('feature dimension cannot change and must be {self.feature_dim}')
        if features.shape[0] != 1:
            raise ValueError('cannot add {features.shape[0]} features, only one feature can be added at a time')

        if self.shard_feature_index == self.shard_maxcount:
            # create a new shard
            self.save_current_shard()
            self.add(id, features)
        else:
            self.shard_features[self.shard_feature_index] = features
            self.shard_feature_id[self.shard_feature_index] = id
            self.shard_feature_index += 1

    def save_current_shard(self):
        current_shard_id = '%s-%06d' % (self.store_name, self.current_shard_index)
        current_shard_filename = self.store_data_dir / current_shard_id
        np.savez(current_shard_filename, feature_id=self.shard_feature_id, features=self.shard_features)
        if self.verbose:
            print(f'saved {self.shard_feature_index} features to shard {current_shard_filename}')
        self.current_shard_index += 1
        self.shard_feature_index = 0

    def __iter__(self):
        for npz_filename in self.npz_filename_list:
            payload = np.load(npz_filename)
            feature_id_list = payload['feature_id']
            features_list = payload['features']
            N = feature_id_list.shape[0]
            if self.shuffle_values:
                index_list = random.sample(range(0, N),N)
            else:
                index_list = range(0, N)
            for i in index_list:
                feature_id = feature_id_list[i]
                feature_vector = np.take(features_list, [i], 0) # to return (1,N) instead of (N,)
                yield feature_id, feature_vector

    def close(self):
        if self.shard_feature_index != 0:
            new_feature_id = np.delete(self.shard_feature_id, range(self.shard_feature_index,self.shard_maxcount), 0)
            new_features   = np.delete(self.shard_features, range(self.shard_feature_index,self.shard_maxcount), 0)
            self.shard_feature_id = new_feature_id
            self.shard_features   = new_features
            self.save_current_shard()
            self.shard_feature_index = 0

    def __del__(self):
        if hasattr(self, 'shard_feature_index'):
            if self.shard_feature_index != 0:
                self.close();
