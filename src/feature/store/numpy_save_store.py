import os
import numpy as np

from feature_store import FeatureStore

class NumpySaveStore(FeatureStore):
    def __init__(self, store_name, store_data_dir):
        """
        Store all data in the specified directory as numpy .npy binary file
        """
        self.store_name = store_name
        self.store_data_dir = store_data_dir
        self.EXTENSION = 'npy'
        self.store_data_filename = os.path.join(self.store_data_dir,
                                                self.store_name + '.' + self.EXTENSION)
    def add(self, new_features):
        if os.path.exists(self.store_data_filename):
            with open(self.store_data_filename, 'rb') as f:
                old_features = np.load(f)
                assert(old_features.shape[1] == new_features.shape[1])
                all_features = np.concatenate((old_features, new_features), axis=0)
        else:
            all_features = new_features
        with open(self.store_data_filename, 'wb') as f:
            np.save(f, all_features)
        return all_features.shape

    def load(self, start_index, count):
        with open(self.store_data_filename, 'rb') as f:
            all_features = np.load(f)
            if start_index == 0 and count == -1:
                return all_features
            else:
                return all_features[start_index:(start_index+count),:]
