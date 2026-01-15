# Feature Store
The FeatureStore provides storage for features extracted from video frames and audio samples.
WISE currently supports the following two types of FeatureStore:

 - WebdatasetStore : stores features across multiple `.tar` files

The [WebdatasetStore][src.feature.store.webdataset_store.WebdatasetStore]
uses the [Webdataset](https://webdataset.github.io/webdataset/) format
to store extracted features across multiple shards maitained as `.tar`
files.

Here is an example of how the features stores in these stores can be accessed.

```python
# source: src/search_index.py
from pathlib import Path
from .feature.store.feature_store_factory import FeatureStoreFactory

...
media_type = 'video' # or 'audio'
feature_dir = Path('/data/projects/Kinetics-7/store/mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k/features/')
feature_store = FeatureStoreFactory.load_store(media_type, feature_dir)
feature_store.enable_read(shard_shuffle = False)

for feature_id, feature_vector in feature_store:
    print(f'feature_id={feature_id}, feature = {feature_vector.shape}')

...
```
