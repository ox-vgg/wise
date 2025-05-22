import argparse
import json
import logging

from pathlib import Path


#from src.dataloader import AVDataset
#from src.search_index import SearchIndex
from src.wise_project import WiseProject
from src.index.search_index_factory import SearchIndexFactory
from src.search.fts import FTSSearch
from src import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='create-index',
                                     description='Create a nearest neighbour search index for features extracted from images and videos.',
                                     epilog='For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/')
    parser.add_argument('--media-type',
                        required=False,
                        action='append',
                        choices=['audio', 'video', 'metadata', 'image'],
                        help='create search index only for these media types; applies to all by default ')

    parser.add_argument('--index-type',
                        required=False,
                        default='IndexFlatIP',
                        choices=['IndexFlatIP', 'IndexIVFFlat'],
                        type=str,
                        help='the type of faiss index for feature vectors, metadata defaults to full-text-search index')

    parser.add_argument('--overwrite',
                        required=False,
                        action='store_true',
                        default=False,
                        help='overwrite existing index file')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='folder where all project assets are stored')

    parser.add_argument('--fts-config', help='json file representing the config for building the FTS5 index')
    args = parser.parse_args()

    project = WiseProject(args.project_dir)
    project_assets = project.discover_assets()
    media_type_list = list(project_assets.keys())
    if args.media_type is not None:
        media_type_list = list(args.media_type)

    for media_type in media_type_list:
        if media_type == 'metadata':
            if not args.fts_config:
                raise ValueError('--fts-config must be a valid json file to index metadata')
            
            fts_config = Path(args.fts_config)
            if not fts_config.exists():
                raise ValueError('--fts-config must be a valid json file to index metadata')
            
            if not args.overwrite and project.fts_config_file.exists():
                logger.info('not overwriting existing metadata index')
                continue

            with fts_config.open() as f:
                fts_tables_columns = json.load(f)

            logger.info(f'creating fts index with config {fts_tables_columns}')
            project_engine = db.init_project(project.dburi)
            db.reflect_external_metadata(project_engine)            
            try:
                project.fts_config_file.write_text(json.dumps(fts_tables_columns, sort_keys=True))
                fts_index = FTSSearch(project, db.project_metadata_obj)
                with project_engine.begin() as conn:
                    fts_index.build_index(conn)

                logger.info('Successfully created fts5 index for metadata')
            except Exception:
                logging.exception('failed to create metadat index')
                project.fts_config_file.unlink(missing_ok=True)     
        else:
            for asset_id in project_assets[media_type]:
                asset = project_assets[media_type][asset_id]
                search_index = SearchIndexFactory(media_type, asset_id, asset)
                search_index.create_index(args.index_type, args.overwrite)
