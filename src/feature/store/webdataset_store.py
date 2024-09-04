import os
import numpy as np
import webdataset as wds
import glob
import io
import sys
import tarfile

from .feature_store import FeatureStore

class WebdatasetStore(FeatureStore):
    def __init__(self, store_name, store_data_dir):
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
        self.store_data_filename = os.path.join(self.store_data_dir,
                                                self.store_name + '-%06d.' + self.EXTENSION)
        self.feature_count = -1
        self.feature_dim = -1

    def enable_write(self, shard_maxcount, shard_maxsize, verbose=0):
        self.shard_maxcount = shard_maxcount
        self.shard_maxsize = shard_maxsize
        self.shardWriter = wds.ShardWriter(pattern=self.store_data_filename,
                                           maxcount=self.shard_maxcount,
                                           maxsize=self.shard_maxsize)
        self.shardWriter.verbose = verbose

    def enable_read(self, shard_shuffle=False, shuffle_values=False, shuffle_bufsize=10000):
        self.shard_shuffle = shard_shuffle
        self.shuffle_values = shuffle_values
        self.shuffle_bufsize = shuffle_bufsize

        # Load the feature dimension
        wds_tar_prefix = os.path.join(self.store_data_dir, self.store_name + '-')
        wds_tar_pattern = wds_tar_prefix + '*.tar'
        self.tar_index = [sys.maxsize, -1]
        self.tar_index_str = ['', '']
        for tar_filename in glob.iglob(pathname=wds_tar_pattern, recursive=False):
            tar_filename_tok = tar_filename.split(wds_tar_prefix)
            tar_index = tar_filename_tok[1].split('.tar')[0]
            if int(tar_index) < self.tar_index[0]:
                self.tar_index[0] = int(tar_index)
                self.tar_index_str[0] = tar_index
            if int(tar_index) > self.tar_index[1]:
                self.tar_index[1] = int(tar_index)
                self.tar_index_str[1] = tar_index
        tar_index_range = '{%s..%s}' % (self.tar_index_str[0], self.tar_index_str[1])
        self.wds_src_url = wds_tar_prefix + tar_index_range + '.tar'
        temp_shard_reader = wds.WebDataset(self.wds_src_url,
                                           shardshuffle=False,
                                           repeat=False)
        for payload in temp_shard_reader:
            feature_vector = np.load(io.BytesIO(payload['features.pyd']), allow_pickle=True)
            self.feature_dim = feature_vector.shape[1]
            break
        temp_shard_reader.close()

        # Fast method for counting the total number of features
        # We assume that tar files with the same filesize have the same number of features.
        # The process of counting the number of files (features) within a tar is slow,
        # whereas filesizes can be computed much faster.
        # Therefore, after we have counted the number of features of a given tar file,
        # we skip the counting process for all the other tar files with the same filesize
        self.feature_count = 0
        filesize_to_count_mapping = {} # key: filesize of tar file; value: number of features in tar file
        for tar_filename in glob.iglob(pathname=wds_tar_pattern, recursive=False):
            filesize = os.stat(tar_filename).st_size
            if filesize not in filesize_to_count_mapping:
                with tarfile.open(tar_filename) as f:
                    # Count the number of files in the tar (assuming each file is a feature vector)
                    filesize_to_count_mapping[filesize] = sum(1 for member in f if member.isreg())
            self.feature_count += filesize_to_count_mapping[filesize]        

    def add(self, id, features):
        if not self.shardWriter:
            raise ValueError('enable_write() must be activated before invoking add() method')
        self.shardWriter.write({
            '__key__': ('%010d' % id), # needs to be a string
            'features.pyd': features
        })


    def __iter__(self):
        if self.shuffle_values:
            shard_reader = wds.WebDataset(self.wds_src_url,
                                          shardshuffle=self.shard_shuffle,
                                          repeat=False).shuffle(self.shuffle_bufsize)
        else:
            shard_reader = wds.WebDataset(self.wds_src_url,
                                          shardshuffle=self.shard_shuffle,
                                          repeat=False)
        for payload in shard_reader:
            feature_id = int(payload['__key__'])
            feature_vector = np.load(io.BytesIO(payload['features.pyd']), allow_pickle=True)
            yield feature_id, feature_vector
    
    def iter_batch(self, batch_size=512):
        if self.shuffle_values:
            shard_reader = wds.WebDataset(self.wds_src_url,
                                          shardshuffle=self.shard_shuffle,
                                          repeat=False).shuffle(self.shuffle_bufsize)
        else:
            shard_reader = wds.WebDataset(self.wds_src_url,
                                          shardshuffle=self.shard_shuffle,
                                          repeat=False)
        
        def numpy_decoder(key, value):
            assert key.endswith('features.pyd'), f"Unexpected key: {key}"
            assert isinstance(value, bytes), f"Unexpected type: {type(value)}"
            return np.load(io.BytesIO(value), allow_pickle=True)

        shard_reader = (
            shard_reader
            .decode(numpy_decoder)
            .to_tuple("__key__", "features.pyd")
            .map_tuple(
                int, # convert key to int
                lambda x: x.squeeze(axis=0), # change shape of numpy array from (1, d) to (d,)
            )
            .batched(batch_size)
        )
        yield from shard_reader

    def close(self):
        self.shardWriter.close()

    def __del__(self):
        if hasattr(self, 'shardWriter') and hasattr(self.shardWriter, 'tarstream'):
            self.shardWriter.close()