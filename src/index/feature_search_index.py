import faiss
from tqdm import tqdm
from pathlib import Path
import numpy as np
import math
import itertools

from .search_index import SearchIndex
from ..feature.feature_extractor_factory import FeatureExtractorFactory
from ..feature.store.feature_store_factory import FeatureStoreFactory
from ..feature.store.webdataset_store import WebdatasetStore

class FeatureSearchIndex(SearchIndex):
    def __init__(self, media_type, asset_id, asset):
        self.media_type = media_type
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
        return self.index_dir / (self.media_type + '-' + index_type + '.faiss')

    def create_index(self, index_type, overwrite=False):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_fn = self.get_index_filename(index_type)
        if index_fn.exists() and overwrite is False:
            print(f'{index_type} for {self.media_type} already exists')
            return
        self.index_type = index_type

        feature_store = FeatureStoreFactory.load_store(self.media_type, self.features_dir)
        feature_store.enable_read(shard_shuffle = False)

        feature_count = feature_store.feature_count
        feature_dim   = feature_store.feature_dim

        index = faiss.IndexFlatIP(feature_dim)
        if index_type == 'IndexFlatIP':
            # IndexFlatIP does not support index.add_with_ids() therefore we use IndexIdMap
            # see https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing
            index_for_id_map = index
            index = faiss.IndexIDMap(index_for_id_map)
        if index_type == 'IndexIVFFlat':
            quantizer = index
            if feature_count < 200000:
                cell_count = 3 * round(math.sqrt(feature_count))
            else:
                cell_count = 10 * round(math.sqrt(feature_count))
            train_count = min(feature_count, 100 * cell_count)
            index = faiss.IndexIVFFlat(quantizer, feature_dim, cell_count, faiss.METRIC_INNER_PRODUCT)

            print(f'  loading a random sample of {train_count} features from {feature_count} features ...')
            shuffled_features = WebdatasetStore(self.media_type, self.features_dir)
            shuffled_features.enable_read(shard_shuffle=True)

            train_features = np.ndarray((train_count, feature_dim), dtype=np.float32)
            for i, (feature_id, feature_vector) in tqdm(
                enumerate(itertools.islice(shuffled_features, train_count)),
                total=train_count
            ):
                train_features[i,:] = feature_vector

            assert not index.is_trained
            print(f'  training {index_type} faiss index with {train_count} features with {cell_count} clusters ...')
            index.train(train_features)
            assert index.is_trained

        print('Adding feature vectors to index')
        with tqdm(total=feature_count) as pbar:
            for feature_ids_batch, feature_vectors_batch in feature_store.iter_batch():
                index.add_with_ids(feature_vectors_batch, feature_ids_batch)
                pbar.update(len(feature_ids_batch))

        faiss.write_index(index, index_fn.as_posix())
        print(f'  saved index to {index_fn}')

    def is_index_loaded(self):
        return hasattr(self, 'index')

    def load_index(self, index_type):
        index_fn = self.get_index_filename(index_type)
        if not index_fn.exists():
            print(f'  index {index_fn} does not exist')
            print(f'  use create-index.py script to create an index')
            False
        self.index = faiss.read_index(index_fn.as_posix(), faiss.IO_FLAG_READ_ONLY)
        self.feature_extractor = FeatureExtractorFactory(self.feature_extractor_id)
        return True

    def search(self, media_type, query, topk=5, query_type='text'):
        if query_type != 'text':
            raise ValueError('query_type={query_type} not implemented')

        if media_type == 'audio':
            if isinstance(query, str):
                media_query_text = [query]
            else:
                media_query_text = [ (self.prompt[media_type] + x) for x in query]
        else:
            media_query_text = [ (self.prompt[media_type] + query) ]

        query_features = self.feature_extractor.extract_text_features(media_query_text)
        dist, ids  = self.index.search(query_features, topk)
        return dist[0], ids[0]
