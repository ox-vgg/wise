from pathlib import Path

class WiseProject:
    def __init__(self, project_dir: Path, create_project=False):
        self.project_dir = Path(project_dir)
        self.store_dir = self.project_dir / 'store'
        self.media_dir = self.project_dir / 'media'
        self.media_type_list = ['image', 'video', 'audio']

        if not self.project_dir.exists():
            if create_project:
                # create the root folders
                self.store_dir.mkdir(parents=True, exist_ok=True)
                self.media_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f'project folder {self.project_dir} does not exist')

    def store_dir(self):
        return self.store_dir

    def media_dir(self):
        return self.media_dir

    def features_root(self, feature_extractor_id):
        return self.store_dir / feature_extractor_id

    def features_dir(self, feature_extractor_id):
        return self.features_root(feature_extractor_id) / 'features'

    def create_features_dir(self, feature_extractor_id):
        features_store = self.features_dir(feature_extractor_id)
        if not features_store.exists():
            features_store.mkdir(parents=True, exist_ok=True)
        return features_store

    def index_dir(self, feature_extractor_id):
        return self.features_root(feature_extractor_id) / 'index'

    def create_index_dir(self, feature_extractor_id):
        index_store = self.features_root(feature_extractor_id) / 'index'
        if not index_store.exists():
            index_store.mkdir(parents=True, exist_ok=True)
        return index_store

    def discover_assets(self):
        self.assets = {}
        # 1. find all feature-extractor-id
        for feature_dir in self.store_dir.glob('*/*/*/*/features/'):
            feature_extractor_id = str(feature_dir.relative_to(self.store_dir).parent)
            available_media_types = []
            for feature_data in feature_dir.glob('*.*'):
                media_type = str(feature_data.stem).split('-')[0]
                if media_type not in available_media_types:
                    available_media_types.append(media_type)
            for media_type in available_media_types:
                if media_type not in self.assets:
                    self.assets[media_type] = {}
                if feature_extractor_id not in self.assets[media_type]:
                    self.assets[media_type][feature_extractor_id] = {}
        # 2. locate all assets related to each feature-extractor-id
        for media_type in self.assets:
            for feature_extractor_id in self.assets[media_type]:
                features_root = self.store_dir / feature_extractor_id
                features_dir = features_root / 'features'
                self.assets[media_type][feature_extractor_id]['features_root'] = features_root
                self.assets[media_type][feature_extractor_id]['features_dir'] = features_dir
                self.assets[media_type][feature_extractor_id]['features_files'] = []
                for feature_data in features_dir.glob(media_type + '-*.*'):
                        self.assets[media_type][feature_extractor_id]['features_files'].append(feature_data.name)

                index_dir = features_root / 'index'
                self.assets[media_type][feature_extractor_id]['index_dir'] = index_dir
                self.assets[media_type][feature_extractor_id]['index_files'] = []
                if not index_dir.exists():
                    continue
                for index_data in index_dir.glob(media_type + '-*.faiss'):
                    self.assets[media_type][feature_extractor_id]['index_files'].append(index_data.name)

        return self.assets
            
