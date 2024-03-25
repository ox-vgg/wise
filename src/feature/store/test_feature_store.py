import unittest
import torch
import tempfile
import numpy as np

from webdataset_store import WebdatasetStore
from numpy_save_store import NumpySaveStore

class TestFeatureExtractorFactory(unittest.TestCase):
    def setUp(self):
        pass

    def test_numpy_save_store(self):
        with tempfile.TemporaryDirectory() as temp_store_dir:
            store_name = 'test-store'
            shard_maxcount = 256
            shard_maxsize = 10*1024*1024 # 10 MB
            featureStore1 = NumpySaveStore(store_name, temp_store_dir)

            feature_dim = 512
            feature_count = 2048
            for feature_index in range(0, feature_count, shard_maxcount):
                print(f'Writing {feature_index} to {feature_index+shard_maxcount} to feature store')
                rand_features = torch.rand((shard_maxcount, feature_dim))
                featureStore1.add(rand_features)

            print(f'Checking features saved in {temp_store_dir}')
            featureStore2 = NumpySaveStore(store_name, temp_store_dir)
            features = featureStore2.load(start_index=0, count=-1)
            self.assertEqual(features.shape, (feature_count, feature_dim))
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
