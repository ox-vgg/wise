import os
import numpy as np
import webdataset as wds

from .feature_store import FeatureStore

class WebdatasetStore(FeatureStore):
    def __init__(self, store_name, store_data_dir, shard_maxcount, shard_maxsize):
        """
        Store all data in the specified directory as numpy .npy binary file

        Parameters
        ----------
        store_name : string
            a string prefix that gets added to all files generated by this store
        store_data_dir : string
            all assets of this store are saved in this folder
        shard_maxcount : int
            maximum number of records per shard
        shard_maxsize : int
            maximum size of each shard

        References:
        [1] https://github.com/webdataset/webdataset/blob/main/examples/wds-notes.ipynb
        [2] https://webdataset.github.io/webdataset/api/webdataset/writer.html
        """
        self.store_name = store_name
        self.store_data_dir = store_data_dir
        self.EXTENSION = 'tar'
        self.shard_maxcount = shard_maxcount
        self.shard_maxsize = shard_maxsize
        self.store_data_filename = os.path.join(self.store_data_dir,
                                                self.store_name + '-%06d.' + self.EXTENSION)
        self.shardWriter = wds.ShardWriter(pattern=self.store_data_filename,
                                           maxcount=self.shard_maxcount,
                                           maxsize=self.shard_maxsize)
        self.shardWriter.verbose = 0

    def add(self, id, features):
        self.shardWriter.write({
            '__key__': ('%10d' % id), # needs to be a string
            'features.pyd': features
        })

    def load(self, start_index, count):
        print(f'Loading features from {self.store_data_filename}')
        # TODO

    def __del__(self):
        self.shardWriter.close()
