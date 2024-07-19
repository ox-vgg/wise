from pathlib import Path
import sqlite3

DB_SCHEME = "sqlite+pysqlite://"

class WiseProject:
    def __init__(self, project_dir: Path, create_project=False):
        self.project_dir = Path(project_dir)
        self.store_dir = self.project_dir / "store"
        self.media_dir = self.project_dir / "media"
        self.metadata_dir = self.project_dir / "metadata"
        self.media_type_list = ["image", "video", "audio"]

        if not self.project_dir.exists():
            if create_project:
                # create the root folders
                self.store_dir.mkdir(parents=True, exist_ok=True)
                self.media_dir.mkdir(parents=True, exist_ok=True)
                self.metadata_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f"project folder {self.project_dir} does not exist")

    @property
    def thumbs_uri(self):
        return f"{DB_SCHEME}/{self.project_dir.absolute()}/thumbs.db"

    @property
    def dburi(self):
        return f"{DB_SCHEME}/{self.metadata_dir.absolute()}/internal.db"

    def metadata_db_table(self, metadata_id, extension='.sqlite'):
        metadata_id_tok = metadata_id.split('/')
        assert len(metadata_id_tok) == 3, 'metadata_id must be in "FOLDER_NAME/DB_NAME/TABLE_NAME" format'
        metadata_db_dir = self.metadata_dir / metadata_id_tok[0]
        metadata_db_dir.mkdir(parents=True, exist_ok=True)
        metadata_db = metadata_db_dir / (metadata_id_tok[1] + extension)
        metadata_table = metadata_id_tok[2]
        return metadata_db, metadata_table

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
        """
        Find the location of all assets based on known structure of WISE project folder tree

        Returns:
        Here is an example of the returned data structure
        {
          "video": {
            "mlfoundations/open_clip/xlm/laion5b": {
              "features_root": "/data/wise/...",
              "features_dir": "/data/wise/.../features",
              "features_files": [
                "video-000000.tar",
                ...
              ],
              "index_dir": "/data/wise/.../index",
              "index_files": [
                "video-IndexFlatIP.faiss",
                ...
              ]
            }
          },
          "audio": {
            "microsoft/clap/2023/four-datasets": {
              "features_root": "/data/wise/...",
              "features_dir": "/data/wise/.../features",
              "features_files": [
                "audio-000000.tar",
                ...
              ],
              "index_dir": "/data/wise/.../index",
              "index_files": [
                "audio-IndexFlatIP.faiss",
                ...
              ]
            }
          },
          "metadata": {
            "EpicKitchens-100/retrieval_annotations/test": "/data/wise/.../metadata/EpicKitchens-100/retrieval_annotations.db",
            "EpicKitchens-100/retrieval_annotations/train": "/data/wise/.../metadata/EpicKitchens-100/retrieval_annotations.db"
          }
        }
        """
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
                self.assets[media_type][feature_extractor_id]['features_root'] = str(features_root)
                self.assets[media_type][feature_extractor_id]['features_dir'] = str(features_dir)
                self.assets[media_type][feature_extractor_id]['features_files'] = []
                for feature_data in features_dir.glob(media_type + '-*.*'):
                    self.assets[media_type][feature_extractor_id]['features_files'].append(feature_data.name)
                self.assets[media_type][feature_extractor_id]['features_files'].sort()

                index_dir = features_root / 'index'
                self.assets[media_type][feature_extractor_id]['index_dir'] = str(index_dir)
                self.assets[media_type][feature_extractor_id]['index_files'] = []
                if not index_dir.exists():
                    continue
                for index_data in index_dir.glob(media_type + '-*.faiss'):
                    self.assets[media_type][feature_extractor_id]['index_files'].append(index_data.name)
                self.assets[media_type][feature_extractor_id]['index_files'].sort()

        # 3. locate all assets related to metadata
        self.assets['metadata'] = {}
        for metadata_db in self.metadata_dir.glob('*/*.sqlite'):
            metadata_db_rel_path = metadata_db.relative_to(self.metadata_dir)
            assert len(metadata_db_rel_path.parts) == 2, f"unexpected {metadata_db_rel_path}, should be of form FOLDER_NAME/DB_NAME"
            metadata_id_prefix = str(metadata_db_rel_path.parent / metadata_db_rel_path.stem)
            with sqlite3.connect( str(metadata_db) ) as sqlite_connection:
                cursor = sqlite_connection.cursor()
                for row in cursor.execute(f'SELECT name FROM sqlite_master WHERE type="table"'):
                    table_name = row[0]
                    if '_fts' not in table_name:
                        metadata_id = metadata_id_prefix + '/' + table_name
                        self.assets['metadata'][metadata_id] = {
                            'metadata_db': str(metadata_db),
                            'metadata_db_type': 'sqlite',
                            'metadata_table': table_name
                        }
        return self.assets
