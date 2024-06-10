class SearchIndex:
    """
    Represents various types of search indices. For example,
    the search index for audiovisual features is implemented
    in FeatureSearchIndex while the search index for text metadata
    is implemented in SqldbSearchIndex.
    """
    def __init__(self, media_type, asset_id, assets):
        raise NotImplementedError

    def get_index_filename(self, index_type):
        raise NotImplementedError

    def create_index(self, index_type, overwrite=False):
        raise NotImplementedError

    def is_index_loaded(self):
        raise NotImplementedError

    def load_index(self, index_type):
        raise NotImplementedError

    def search(self, media_type, query, topk=5, query_type='text'):
        raise NotImplementedError
