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

class FeatureStoreType(str, enum.Enum):
    FAISS = "faiss"

class FeatureStoreFactory:
    @classmethod
    def create_store(cls, feature_store_type: FeatureStoreType, modality_type, features_dir):
        if feature_store_type == FeatureStoreType.FAISS:
            return FaissStore(modality_type, features_dir)
        else:
            raise ValueError(f'unknown feature_store_type {feature_store_type}')

    @classmethod
    def load_store(cls, modality_type, features_dir):
        features_dir = Path(features_dir) # convert type in case features_dir is a string

        # infer the store type
        shard_suffixes = set([p.suffix for p in features_dir.glob(modality_type + '-*')])
        if len(shard_suffixes) == 0:
            raise ValueError(f'found no feature store files in {features_dir} for type {modality_type}')
        elif len(shard_suffixes) > 1:
            raise ValueError(f'failed to infer type of {modality_type} feature store in {features_dir} because there are multiple file types present ({shard_suffixes})')

        shard_suffix = shard_suffixes.pop()
        if shard_suffix == ".faiss":
            return FaissStore(modality_type, features_dir)
        else:
            raise ValueError(f'unknown store containing shard filenames with extension {shard_suffix}')
