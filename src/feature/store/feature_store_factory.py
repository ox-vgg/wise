import glob
from pathlib import Path
import enum

from .webdataset_store import WebdatasetStore
from .numpy_save_store import NumpySaveStore
from .faiss_store import FaissStore

class FeatureStoreType(str, enum.Enum):
    WEBDATASET = "webdataset"
    NUMPY = "numpy"
    FAISS = "faiss"

class FeatureStoreFactory:
    @classmethod
    def create_store(cls, feature_store_type: FeatureStoreType, media_type, features_dir):
        if feature_store_type == FeatureStoreType.WEBDATASET:
            return WebdatasetStore(media_type, features_dir)
        elif feature_store_type == FeatureStoreType.NUMPY:
            return NumpySaveStore(media_type, features_dir)
        elif feature_store_type == FeatureStoreType.FAISS:
            return FaissStore(media_type, features_dir)
        else:
            raise ValueError(f'unknown feature_store_type {feature_store_type}')

    @classmethod
    def load_store(cls, media_type, features_dir):
        features_dir = Path(features_dir) # convert type in case features_dir is a string

        # infer the store type
        shard_ext_set = set()
        shard_file_pattern = features_dir / (media_type + '-*.*')
        for filename in glob.iglob(pathname=shard_file_pattern.as_posix(), recursive=False):
            suffix = Path(filename).suffix
            shard_ext_set.add(suffix)
        if len(shard_ext_set) == 0:
            raise ValueError(f'found no feature store files in {features_dir} for type {media_type}')
        elif len(shard_ext_set) > 1:
            raise ValueError(f'failed to infer type of {media_type} feature store in {features_dir} because there are multiple file types present ({shard_ext_set})')

        shard_suffix = shard_ext_set.pop()
        if shard_suffix == '.tar':
            return WebdatasetStore(media_type, features_dir)
        elif shard_suffix == '.npz':
            return NumpySaveStore(media_type, features_dir)
        elif shard_suffix == ".faiss":
            return FaissStore(media_type, features_dir)
        else:
            raise ValueError(f'unknown store containing shard filenames with extension {shard_suffix}')
