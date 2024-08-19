# Feature Store

The FeatureStore provides storage for features extracted from video
frames and audio samples.  The
[FaissStore][wise.feature.store.faiss_store.FaissStore] uses the
[faiss](https://ai.meta.com/tools/faiss/) index.

Here is an example of how the features stores in these stores can be accessed.

```python
# source: src/wise/search_index.py
from pathlib import Path
from .feature.store.feature_store_factory import FeatureStoreFactory
from ...data_models import ModalityType

...
modality_type = ModalityType.VIDEO  # or AUDIO, IMAGE, or TEXT
feature_dir = Path('/data/projects/Kinetics-7/store/mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k/features/')
feature_store = FeatureStoreFactory.load_store(modality_type, feature_dir)
feature_store.enable_read(shard_shuffle = False)

for feature_id, feature_vector in feature_store:
    print(f'feature_id={feature_id}, feature = {feature_vector.shape}')

...
```
