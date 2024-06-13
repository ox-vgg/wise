import sqlite3
import faiss
from tqdm import tqdm
from pathlib import Path
import numpy as np
import math

from .search_index import SearchIndex
from ..feature.feature_extractor_factory import FeatureExtractorFactory
from ..feature.store.feature_store_factory import FeatureStoreFactory
from ..feature.store.webdataset_store import WebdatasetStore

class SqliteSearchIndex(SearchIndex):
    def __init__(self, media_type, asset_id, asset):
        self.media_type = media_type
        self.metadata_id = asset_id

        assert 'metadata_db_type' in asset, "features_dir missing in assets"
        assert asset['metadata_db_type'] == 'sqlite', "SqliteSearchIndex only supports the SQLite database engine"

        assert 'metadata_db' in asset, "metadata_db missing in assets"
        self.metadata_db = str(asset['metadata_db'])

        assert 'metadata_table' in asset, "metadata_table missing in assets"
        self.metadata_table = str(asset['metadata_table'])
        self.metadata_table_fts = f'{self.metadata_table}_fts'

    def get_index_filename(self, index_type):
        return self.metadata_db + '::' + self.metadata_table

    def sqlite_table_exists(self, db, table):
        if Path(db).exists():
            with sqlite3.connect( str(db) ) as sqlite_connection:
                cursor = sqlite_connection.cursor()
                for row in cursor.execute(f'SELECT COUNT(*) FROM sqlite_master WHERE type="table" AND name="{table}"'):
                    if row[0] == 1:
                        return True
                    else:
                        return False
        return False

    def create_index(self, index_type, overwrite=False):
        index_type = "FTS5" # use full text search version 5 of sqlite by default
        self.metadata_table_fts = f'{self.metadata_table}_fts'
        if self.sqlite_table_exists(self.metadata_db, self.metadata_table_fts) and not overwrite:
            print(f'{index_type} index for {self.media_type} already exists')
            return
        print(f'Creating metadata index for {self.metadata_id}')

        with sqlite3.connect(self.metadata_db) as sqlite_connection:
            cursor = sqlite_connection.cursor()

            ## 0. Drop existing fts tables
            exiting_fts_table_list = []
            for row in cursor.execute(f'SELECT name FROM sqlite_master WHERE type="table"'):
                table_name = row[0]
                if f'{self.metadata_table}_fts' in table_name:
                    exiting_fts_table_list.append(table_name)

            cursor.execute(f'BEGIN TRANSACTION')
            if len(exiting_fts_table_list):
                for table_name in exiting_fts_table_list:
                    cursor.execute(f'DROP TABLE IF EXISTS {table_name}')

            ## 1. Find the column names that correspond to metadata
            ## (i.e. column names that do not begin with "__")
            metadata_colnames = []
            cursor.execute(f'SELECT * from {self.metadata_table} LIMIT 1')
            for column_data in cursor.description:
                colname = column_data[0]
                if not colname.startswith('__'):
                    metadata_colnames.append(colname)
            value_placeholders = ','.join( ['?'] * len(metadata_colnames) )
            metadata_colnames_csv = ','.join(metadata_colnames)

            ## 2. Create table to store full text search index
            sql = f'CREATE VIRTUAL TABLE {self.metadata_table_fts} USING fts5({metadata_colnames_csv})'
            cursor.execute(sql)

            ## 3. Create an index of all the metadata columns
            fts_data = []
            for row in cursor.execute(f'SELECT {metadata_colnames_csv} FROM {self.metadata_table}'):
                fts_data.append(row)
            sql = f'INSERT INTO {self.metadata_table_fts}({metadata_colnames_csv}) VALUES ({value_placeholders})'
            cursor.executemany(sql, fts_data)
            cursor.execute(f'END TRANSACTION')

    def is_index_loaded(self):
        return hasattr(self, 'index')

    def load_index(self, index_type):
        exists = self.sqlite_table_exists(self.metadata_db, self.metadata_table_fts)

        if self.sqlite_table_exists(self.metadata_db, self.metadata_table_fts):
            with sqlite3.connect('file:' + str(self.metadata_db) + '?mode=ro', uri=True) as sqlite_connection:
                self.index = sqlite3.connect(':memory:')
                sqlite_connection.backup(self.index)
                return True
        else:
            print(f'missing metadata index')
            print(f'use create-index.py script to create a FTS search index')
            return False

    def search(self, media_type, query, topk=5, query_type='text'):
        if query_type != 'text':
            raise ValueError('query_type={query_type} not implemented')
        assert media_type == 'metadata', 'SqliteSearchIndex only supports metadata search'

        cursor = self.index.cursor()
        sql = f'''SELECT __filename, __starttime, __stoptime, rank FROM {self.metadata_table}
        JOIN {self.metadata_table_fts} ON {self.metadata_table_fts}.rowid = {self.metadata_table}.rowid
        WHERE {self.metadata_table_fts} MATCH "{query}"
        ORDER BY rank LIMIT {topk}'''

        match_filename_list = []
        match_pts_list = []
        match_score_list = []
        for row in cursor.execute(sql):
            starttime = float(row[1])
            stoptime = float(row[2])
            score = float(row[3])
            match_filename_list.append(row[0])
            match_pts_list.append([starttime, stoptime])
            match_score_list.append(score)
        return {
            'match_filename_list': match_filename_list,
            'match_pts_list': match_pts_list,
            'match_score_list': match_score_list,
        }
