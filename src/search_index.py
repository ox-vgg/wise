import faiss
from tqdm import tqdm
from pathlib import Path

from .feature.feature_extractor_factory import FeatureExtractorFactory
from .feature.store.webdataset_store import WebdatasetStore

class SearchIndex:
    def __init__(self, media_type, feature_extractor_id, index_dir, feature_dir=None):
        self.media_type = media_type
        self.feature_extractor_id = feature_extractor_id
        self.index_dir = index_dir
        self.feature_dir = feature_dir

        self.prompt = {
            'image':'This is a photo of a ',
            'video':'This is a photo of a ',
            'audio':'this is the sound of '
        }

    def get_index_filename(self, index_type):
        return self.index_dir / (self.media_type + '-' + index_type + '.faiss')

    def create_index(self, index_type, overwrite=False):
        index_fn = self.get_index_filename(index_type)
        if index_fn.exists() and overwrite is False:
            print(f'  index {index_fn} already exists')
            return
        if self.feature_dir is None:
            print(f'  feature_dir is missing')
            return
        self.index_type = index_type

        # TODO: Implement a FeatureStoreFactory to load features from any type of store
        feature_store = WebdatasetStore(self.media_type, self.feature_dir)
        feature_store.enable_read(shard_shuffle = False)

        feature_count = feature_store.feature_count
        feature_dim   = feature_store.feature_dim

        index = faiss.IndexFlatIP(feature_dim)
        if index_type == 'IndexIVFFlat':
            quantizer = index
            cell_count = 10 * round(math.sqrt(feature_count))
            train_count = min(feature_count, 100 * cell_count)
            index = faiss.IndexIVFFlat(quantizer, feature_dim, cell_count)

            print(f'  loading a random sample of {train_count} features from {feature_count} features ...')
            shuffled_features = WebdatasetStore(media_type, feature_dir)
            shuffled_features.enable_read(shard_shuffle=True)

            train_features = np.ndarray((train_count, feature_dim), dtype=np.float32)
            feature_index = 0
            for feature_id, feature_vector in shuffled_features:
                train_features[feature_index,:] = feature_vector
                feature_index += 1

            assert not index.is_trained
            print(f'  training {args.index_type} faiss index with {train_count} features ...')
            index.train(train_features)
            assert index.is_trained

        with tqdm(total=feature_count) as pbar:
            for feature_id, feature_vector in feature_store:
                index.add(feature_vector)
                pbar.update(1)

        faiss.write_index(index, index_fn.as_posix())
        print(f'  saved index to {index_fn}')

    def is_index_loaded(self):
        return hasattr(self, 'index')

    def load_index(self, index_type):
        index_fn = self.get_index_filename(index_type)
        if not index_fn.exists():
            print(f'  index {index_fn} does not exist')
            return
        self.index = faiss.read_index(index_fn.as_posix(), faiss.IO_FLAG_READ_ONLY)
        self.feature_extractor = FeatureExtractorFactory(self.feature_extractor_id)

    def search(self, media_type, query, topk=5, query_type='text'):
        if query_type != 'text':
            raise ValueError('query_type={query_type} not implemented')

        if media_type == 'audio':
            media_query_text = [ (self.prompt[media_type] + x) for x in query]
        else:
            media_query_text = [ (self.prompt[media_type] + query) ]
        print(f'Querying {media_type} with "{media_query_text}"')
        query_features = self.feature_extractor.extract_text_features(media_query_text)
        dist, ids  = self.index.search(query_features, topk)
        return dist[0], ids[0]

        
