import glob
from pathlib import Path
import enum

from .webdataset_store import WebdatasetStore
from .numpy_save_store import NumpySaveStore

class FeatureStoreType(str, enum.Enum):
    WEBDATASET = "webdataset"
    NUMPY = "numpy"

class FeatureStoreFactory:
    @classmethod
    def create_store(cls, feature_store_type: FeatureStoreType, media_type, features_dir):
        if feature_store_type == FeatureStoreType.WEBDATASET:
            return WebdatasetStore(media_type, features_dir)
        elif feature_store_type == FeatureStoreType.NUMPY:
            return NumpySaveStore(media_type, features_dir)
        else:
            raise ValueError(f'unknown feature_store_type {feature_store_type}')

    @classmethod
    def load_store(cls, media_type, features_dir):
        # infer the store type
        shard_ext_list = []
        shard_file_pattern = features_dir / (media_type + '-*.*')
        for filename in glob.iglob(pathname=shard_file_pattern.as_posix(), recursive=False):
            print(filename)
            suffix = Path(filename).suffix
            if suffix not in shard_ext_list:
                shard_ext_list.append(suffix)
        print(shard_ext_list)
        if len(shard_ext_list) != 1:
            raise ValueError(f'failed to infer type of {media_type} feature store in {features_dir}')
        if shard_ext_list[0] == '.tar':
            return WebdatasetStore(media_type, features_dir)
        elif shard_ext_list[0] == '.npz':
            return NumpySaveStore(media_type, features_dir)
        else:
            raise ValueError(f'unknown store containing shard filenames with extension {shard_ext_list[0]}')
