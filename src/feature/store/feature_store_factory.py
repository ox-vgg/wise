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

from pathlib import Path
import enum

from .faiss_store import FaissStore
from ...data_models import ModalityType

class FeatureStoreType(enum.Enum):
    FAISS = "faiss"

class FeatureStoreFactory:
    @classmethod
    def create_store(
        cls,
        feature_store_type: FeatureStoreType,
        modality_type: ModalityType,
        features_dir,
    ):
        store_name_prefix = modality_type.value
        if feature_store_type is FeatureStoreType.FAISS:
            return FaissStore(store_name_prefix, features_dir)
        else:
            raise ValueError(f'unknown feature_store_type {feature_store_type}')

    @classmethod
    def load_store(cls, modality_type: ModalityType, features_dir):
        features_dir = Path(features_dir) # convert type in case features_dir is a string
        store_name_prefix = modality_type.value

        # infer the store type
        shard_suffixes = set([p.suffix for p in features_dir.glob(store_name_prefix + '-*')])
        if len(shard_suffixes) == 0:
            raise ValueError(f"found no feature store files '{features_dir}{store_name_prefix}-*'")
        elif len(shard_suffixes) > 1:
            raise ValueError(f"failed to infer type of '{features_dir}/{store_name_prefix}-*' feature store files because there are multiple file types present ({shard_suffixes})")

        shard_suffix = shard_suffixes.pop()
        if shard_suffix == ".faiss":
            return FaissStore(store_name_prefix, features_dir)
        else:
            raise ValueError(f'unknown store containing shard filenames with extension {shard_suffix}')
