import unittest
import torch
import tempfile
import numpy as np

from store.webdataset_store import WebdatasetStore
from store.numpy_save_store import NumpySaveStore

class TestFeatureExtractorFactory(unittest.TestCase):
    def setUp(self):
        self.store_name = 'test-store'
        self.feature_dim = 512
        self.feature_count = 2048

    def test_numpy_save_store(self):
        with tempfile.TemporaryDirectory() as temp_store_dir:
            shard_maxcount = 256
            featureStore1 = NumpySaveStore(self.store_name, temp_store_dir)

            feature_count = 2048
            for feature_index in range(0, self.feature_count, shard_maxcount):
                rand_features = torch.rand((shard_maxcount, self.feature_dim))
                featureStore1.add(rand_features)

            featureStore2 = NumpySaveStore(self.store_name, temp_store_dir)
            features = featureStore2.load(start_index=0, count=-1)
            self.assertEqual(features.shape, (self.feature_count, self.feature_dim))

    def test_webdataset_store(self):
        with tempfile.TemporaryDirectory() as temp_store_dir:
            shard_maxcount = 256
            shard_maxsize = 1024 # 1 KB
            featureStore = WebdatasetStore(self.store_name, temp_store_dir, shard_maxcount, shard_maxsize)
            for feature_index in range(0, self.feature_count, shard_maxcount):
                rand_features = torch.rand((shard_maxcount, self.feature_dim))
                featureStore.add(rand_features)
            # TODO: test load operation as well

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
