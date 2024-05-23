# Feature Store
The FeatureStore provides storage for features extracted from video frames and audio samples.
WISE currently supports the following two types of FeatureStore:

 - NumpySaveStore  : stores features across multiple `.npz` files 
 - WebdatasetStore : stores features across multiple `.tar` files

The [NumpySaveStore](src/feature/store/numpy_save_store.py) uses [`numpy.savez()`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)
to store extracted features as numpy ndarray in `.npz` files. The [WebdatasetStore](src/feature/store/webdataset_store.py)
uses the [Webdataset](https://webdataset.github.io/webdataset/) format to store extracted features across multiple shards maitained as `.tar` files.

Here is an example of how the features stores in these stores can be accessed.

```
# source: src/search_index.py
import numpy as np

from .feature.feature_extractor_factory import FeatureExtractorFactory
from .feature.store.feature_store_factory import FeatureStoreFactory

...
media_type = 'video' # or 'audio'
feature_dir = '/data/projects/Kinetics-7/store/mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k/features/'
feature_store = FeatureStoreFactory.load_store(media_type, feature_dir)
feature_store.enable_read(shard_shuffle = False)

for feature_id, feature_vector in feature_store:
    print(f'feature_id={feature_id}, feature = {feature_vector.shape}'

...
```
