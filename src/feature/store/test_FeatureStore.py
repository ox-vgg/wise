import unittest
import torch
import tempfile
import numpy as np

from WebdatasetStore import WebdatasetStore

class TestFeatureExtractorFactory(unittest.TestCase):
    def setUp(self):
        pass

    def test_feature_store(self):
        #temp_store_dir = tempfile.TemporaryDirectory()
        temp_store_dir = '/home/tlm/data/wise/tmp/'
        print(f'Using temporary feature store {temp_store_dir}')
        store_name = 'test-store'
        shard_maxcount = 256
        shard_maxsize = 10*1024*1024 # 10 MB
        featureStore = WebdatasetStore(store_name, temp_store_dir, shard_maxcount, shard_maxsize)

        feature_dim = 512
        feature_count = 8096
        for feature_index in range(0, feature_count, shard_maxcount):
            print(f'Writing {feature_index} to {feature_index+shard_maxcount} to feature store')
            rand_features = torch.rand((shard_maxcount, feature_dim))
            print(f'rand_features.shape = {rand_features.shape}')
            featureStore.add(rand_features)

        #temp_store_dir.cleanup()
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
