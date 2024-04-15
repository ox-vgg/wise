import unittest
import torch
import tempfile
import numpy as np

from .webdataset_store import WebdatasetStore
from .numpy_save_store import NumpySaveStore

class TestFeatureExtractorFactory(unittest.TestCase):
    def setUp(self):
        self.store_name = 'test-store'
        self.feature_dim = 512
        self.feature_count = 2048

    def test_numpy_save_store(self):
        pass
        with tempfile.TemporaryDirectory() as temp_store_dir:
            shard_maxcount = 256
            featureStore1 = NumpySaveStore(self.store_name, temp_store_dir)

            feature_count = 2048
            for feature_index in range(0, self.feature_count, shard_maxcount):
                rand_features = torch.rand((shard_maxcount, self.feature_dim))
                featureStore1.add(feature_index, rand_features)

            featureStore2 = NumpySaveStore(self.store_name, temp_store_dir)
            features = featureStore2.load(start_index=0, count=-1)
            self.assertEqual(features.shape, (self.feature_count, self.feature_dim))

    def test_webdataset_store_batch_write(self):
        with tempfile.TemporaryDirectory() as temp_store_dir:
            shard_maxcount = 3
            shard_maxsize = 256 # bytes
            write_store = WebdatasetStore(self.store_name, temp_store_dir)
            featureA = np.array([[1,2,3,4]])
            featureB = np.array([[5,6,7,8]])
            featureC = np.array([[9,10,11,12]])

            feature0 = np.concatenate((featureA, featureB, featureC), axis=0)
            feature3 = np.concatenate((featureC, featureB, featureA), axis=0)

            write_store.enable_write(shard_maxcount, shard_maxsize)
            write_store.add(0, feature0)
            write_store.add(3, feature3)
            write_store.close()
            del write_store

            read_store = WebdatasetStore(self.store_name, temp_store_dir)
            read_store.enable_read(shard_shuffle=False, shuffle_values=False)

            for feature_id, feature_vector in read_store:
                if int(feature_id) == 0:
                    self.assertTrue(np.all(np.equal(feature_vector, feature0)))
                if int(feature_id) == 3:
                    self.assertTrue(np.all(np.equal(feature_vector, feature3)))
    def test_webdataset_store_read_order(self):
        with tempfile.TemporaryDirectory() as temp_store_dir:
            shard_maxcount = 3
            shard_maxsize = 256 # bytes
            write_store = WebdatasetStore(self.store_name, temp_store_dir)
            featureA = np.array([[1,2,3,4]])
            featureB = np.array([[5,6,7,8]])
            featureC = np.array([[9,10,11,12]])

            feature0 = np.concatenate((featureA, featureB, featureC), axis=0)
            feature3 = np.concatenate((featureC, featureB, featureA), axis=0)

            write_store.enable_write(shard_maxcount, shard_maxsize)
            write_store.add(0, feature0)
            write_store.add(3, feature3)
            write_store.add(6, featureA)
            write_store.add(7, featureB)
            write_store.add(8, featureC)
            write_store.close()
            del write_store

            read_store = WebdatasetStore(self.store_name, temp_store_dir)
            read_store.enable_read(shard_shuffle=False, shuffle_values=False)

            read_feature_id = []
            for feature_id, feature_vector in read_store:
                read_feature_id.append( int(feature_id) )
            self.assertEqual(read_feature_id, [0, 3, 6, 7, 8])
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
