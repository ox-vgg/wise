class FeatureStore:
    def __init__(self, store_name, store_data_dir):
        raise NotImplementedError

    def add(self, index, features):
        raise NotImplementedError

    # TODO: implement IterableDataset
    # def __iter__(self):
    # def __next__(self):

    # random access
    def load(self, start_index=0, count=-1):
        raise NotImplementedError
