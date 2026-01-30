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

import unittest
import torch
import tempfile
import numpy as np
from pathlib import Path

from .faiss_store import FaissStore

class TestFeatureExtractorFactory(unittest.TestCase):
    def setUp(self):
        self.store_name = 'test-store'
        self.feature_dim = 512

    def test_faiss_store(self):
        with tempfile.TemporaryDirectory() as temp_store_dir:
            store = FaissStore(self.store_name, temp_store_dir)
            featureA = np.array([[1, 2, 3, 4]])
            featureB = np.array([[5, 6, 7, 8]])
            featureC = np.array([[9, 10, 11, 12]])

            feature0 = np.concatenate((featureA, featureB, featureC), axis=0)
            feature3 = np.concatenate((featureC, featureB, featureA), axis=0)

            store.enable_write()

            store.add([0, 1, 2], feature0)
            self.assertEqual(store.feature_count, 3)
            self.assertEqual(store.feature_dim, 4)

            store.close()
            self.assertEqual(
                store.filenames,
                [
                    f"{temp_store_dir}/{self.store_name}-000000.faiss",
                ],
            )
            store.enable_write()
            store.add([3, 4, 5], feature3)
            store.close()

            self.assertEqual(store.feature_count, 6)
            self.assertEqual(store.feature_dim, 4)
            self.assertEqual(
                store.filenames,
                [
                    f"{temp_store_dir}/{self.store_name}-000000.faiss",
                    f"{temp_store_dir}/{self.store_name}-000001.faiss",
                ],
            )

            read_feature_id = []
            for feature_id, feature_vector in store:
                read_feature_id.append(int(feature_id))
                if int(feature_id) < 3:
                    self.assertTrue(
                        np.all(np.equal(feature_vector, feature0[feature_id]))
                    )
                else:
                    self.assertTrue(
                        np.all(np.equal(feature_vector, feature3[feature_id - 3]))
                    )

            self.assertEqual(read_feature_id, [0, 1, 2, 3, 4, 5])

            # Read vectors in a different order
            self.assertTrue(np.array_equal(store[2], featureC))
            self.assertTrue(np.array_equal(store[3], featureC))

            self.assertTrue(np.array_equal(store[0], featureA))
            self.assertTrue(np.array_equal(store[5], featureA))

            self.assertTrue(np.array_equal(store[1], featureB))
            self.assertTrue(np.array_equal(store[4], featureB))

            # print cache info to check cache access
            # should print something like
            # CacheInfo(hits=4, misses=2, maxsize=32, current_size=2)
            print(store.load_faiss_index.cache_info())

            # check overwrite
            store.enable_write(shard_maxcount=1, overwrite=True)
            self.assertEqual(store.feature_count, 0)
            self.assertEqual(store.feature_dim, None)

            store.add([0, 1, 2], feature0)
            self.assertEqual(store.feature_count, 3)
            self.assertEqual(store.feature_dim, 4)

            store.close()
            self.assertEqual(
                store.filenames,
                [
                    f"{temp_store_dir}/{self.store_name}-000000.faiss",
                    f"{temp_store_dir}/{self.store_name}-000001.faiss",
                    f"{temp_store_dir}/{self.store_name}-000002.faiss",
                ],
            )
            del store


if __name__ == '__main__':
    unittest.main()
