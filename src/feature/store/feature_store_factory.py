#!/usr/bin/env python3

from pathlib import Path
import enum

from .faiss_store import FaissStore

class FeatureStoreType(str, enum.Enum):
    FAISS = "faiss"

class FeatureStoreFactory:
    @classmethod
    def create_store(cls, feature_store_type: FeatureStoreType, media_type, features_dir):
        if feature_store_type == FeatureStoreType.FAISS:
            return FaissStore(media_type, features_dir)
        else:
            raise ValueError(f'unknown feature_store_type {feature_store_type}')

    @classmethod
    def load_store(cls, media_type, features_dir):
        features_dir = Path(features_dir) # convert type in case features_dir is a string

        # infer the store type
        shard_suffixes = set([p.suffix for p in features_dir.glob(media_type + '-*')])
        if len(shard_suffixes) == 0:
            raise ValueError(f'found no feature store files in {features_dir} for type {media_type}')
        elif len(shard_suffixes) > 1:
            raise ValueError(f'failed to infer type of {media_type} feature store in {features_dir} because there are multiple file types present ({shard_suffixes})')

        shard_suffix = shard_suffixes.pop()
        if shard_suffix == ".faiss":
            return FaissStore(media_type, features_dir)
        else:
            raise ValueError(f'unknown store containing shard filenames with extension {shard_suffix}')
