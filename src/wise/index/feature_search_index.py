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

import itertools
import logging
import math
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from wise.data_models import ModalityType
from wise.feature.store.feature_store_factory import FeatureStoreFactory
from wise.index.search_index import SearchIndex


logger = logging.getLogger(__name__)


class FeatureSearchIndex(SearchIndex):

    def __init__(
        self, modality_type: str, asset_id, asset,
    ):
        self.modality_type = ModalityType(modality_type)
        self.feature_extractor_id = asset_id

        assert 'features_dir' in asset, "features_dir missing in assets"
        self.features_dir = Path(asset['features_dir'])

        assert 'index_dir' in asset, "index_dir missing in assets"
        self.index_dir = Path(asset['index_dir'])

        self.prompt = {
            'image':'This is a photo of a ',
            'video':'This is a photo of a ',
            'audio':'this is the sound of '
        }

    def get_index_filename(self, index_type):
        return self.index_dir / (self.modality_type.value + '-' + index_type + '.faiss')

    def create_index(self, index_type, overwrite=False):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_fn = self.get_index_filename(index_type)
        if index_fn.exists() and overwrite is False:
            logger.warning(
                "%s index for %s already exists",
                index_type,
                self.modality_type,
            )
            return
        self.index_type = index_type

        feature_store = FeatureStoreFactory.load_store(
            self.modality_type, self.features_dir
        )
        feature_store.enable_read(shard_shuffle = False)

        feature_count = feature_store.feature_count
        feature_dim   = feature_store.feature_dim

        index = faiss.IndexFlatIP(feature_dim)
        if index_type == 'IndexFlatIP':
            # IndexFlatIP does not support index.add_with_ids() therefore we use IndexIdMap2
            # see https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing
            # We use IndexIdMap2 instead of IndexIdMap because it supports reconstructing
            # the original vectors from their IDs (for internal search)
            index = faiss.IndexIDMap2(index)
        if index_type == 'IndexIVFFlat':
            quantizer = index
            if feature_count < 200000:
                cell_count = 3 * round(math.sqrt(feature_count))
            else:
                cell_count = 10 * round(math.sqrt(feature_count))
            train_count = min(feature_count, 100 * cell_count)
            index = faiss.IndexIVFFlat(quantizer, feature_dim, cell_count, faiss.METRIC_INNER_PRODUCT)
            index.set_direct_map_type(faiss.DirectMap.Hashtable) # Hashtable needed to support non-sequential ids

            logger.info(
                "Loading a random sample of %d features from %d features",
                train_count,
                feature_count,
            )
            shuffled_features = FeatureStoreFactory.load_store(
                self.modality_type, self.features_dir
            )
            shuffled_features.enable_read(shard_shuffle=True, shuffle_values=True)

            train_features = np.ndarray((train_count, feature_dim), dtype=np.float32)
            for i, (feature_id, feature_vector) in tqdm(
                enumerate(itertools.islice(shuffled_features, train_count)),
                total=train_count
            ):
                train_features[i,:] = feature_vector

            assert not index.is_trained
            logger.info(
                "Training %s faiss index with %d features with %d clusters",
                index_type,
                train_count,
                cell_count,
            )
            index.train(train_features)
            assert index.is_trained

        logger.info("Adding feature vectors to index")
        with tqdm(total=feature_count) as pbar:
            for feature_ids_batch, feature_vectors_batch in feature_store.iter_batch():
                index.add_with_ids(feature_vectors_batch, feature_ids_batch)
                pbar.update(len(feature_ids_batch))

        faiss.write_index(index, index_fn.as_posix())
        logger.info("Saved index to '%s'", index_fn)

    def load_index(self, index_type):
        self.index_type = index_type
        index_fn = self.get_index_filename(index_type)
        if not index_fn.exists():
            logger.error(
                (
                    "Index '%s' does not exist;"
                    " use `python -m wise create-index` to create it."
                ),
                index_fn,
            )
            return False
        self.index = faiss.read_index(index_fn.as_posix(), faiss.IO_FLAG_READ_ONLY)
        return True

    @property
    def is_internal_search_supported(self):
        """
        Checks if the faiss index supports internal search (i.e. reconstructing
        vectors from their ids). This should be enabled by default for new
        projects created using our latest code, but older projects might not
        support this, so we need to perform some checks.
        """
        if self.index_type == 'IndexFlatIP':
            # In previous versions of our code, we were using faiss.IndexIDMap,
            # which doesn't support internal search. Therefore we need to check
            # if faiss.IndexIDMap2 (rather than faiss.IndexIDMap) is being used
            return isinstance(self.index, faiss.IndexIDMap2)
        elif self.index_type == 'IndexIVFFlat':
            # Check if the direct map was enabled
            return hasattr(self.index, 'direct_map') and not self.index.direct_map.no()
