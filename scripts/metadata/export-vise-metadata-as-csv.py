#!/usr/bin/env python3

## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import argparse
import sys
from pathlib import Path
import csv
import json
import sqlite3
import os

from wise.wise_project import WiseProject
from wise import db
from wise.repository import MediaRepo

import sqlalchemy as sa

def import_metadata_from_vise(vise_metadata_db, vise_join_colname, project_dir, csv_filename):
    if not sqlite_table_exists(vise_metadata_db, 'file_metadata'):
        raise ValueError(f'file_metadata table does not exist in {vise_metadata_db}')

    project = WiseProject(project_dir, create_project=False)
    project_assets = project.discover_assets()
    if len(project_assets) == 0:
        print(f'failed to load assets from {project_dir}')
        sys.exit(1)
    db_engine = project.db_engine
    db_inspector = sa.inspect(db_engine)

    metadata, colnames = load_metadata_from_sqlite(vise_metadata_db)
    print(f'loaded {len(metadata)} rows from {vise_metadata_db} file_metadata table')

    # add 'media_id' corresponding to each metadata row
    resolve_media_path(db_engine, metadata, vise_join_colname)
    colnames.insert(0, 'media_id')

    print(f'Exporting metadata as a CSV with following columns: {colnames}')
    export_media_metadata(db_engine, colnames, metadata, csv_filename)
    print(f'exported {len(metadata)} rows to {csv_filename}')

def sqlite_table_exists(sqlite_db, table_name):
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def load_metadata_from_sqlite(vise_metadata_db):
    print(f'Loading metadata from SQLite database {vise_metadata_db} ...')
    all_metadata = []
    conn = sqlite3.connect(vise_metadata_db)
    cursor = conn.cursor()

    # Assuming the table name is 'file_metadata' in the SQLite database
    cursor.execute("SELECT * FROM file_metadata")
    rows = cursor.fetchall()
    
    colnames = [description[0] for description in cursor.description]
    
    for row in rows:
        metadata = {}
        for i, colname in enumerate(colnames):
            metadata[colname] = row[i]
        all_metadata.append(metadata)

    conn.close()
    return all_metadata, colnames

def resolve_media_path(db_engine, metadata, vise_join_colname):
    with db_engine.connect() as conn:
        failed_media_path = []
        for i in range(0, len(metadata)):
            media_path = metadata[i][vise_join_colname]
            media_metadata = MediaRepo.get_row_by_column_match(conn,
                                                               column_name_to_match='path',
                                                               column_value=media_path)
            if media_metadata:
                metadata[i]['media_id'] = media_metadata.id
            else:
                metadata[i]['media_id'] = -1
                failed_media_path.append(media_path)
        if failed_media_path:
            print(f'failed to resolve media_path for these file paths: {failed_media_path}')
            raise ValueError(f'failed to resolved media_path for {len(failed_media_path)} metadata rows')

def export_media_metadata(db_engine, colnames, metadata, csv_filename):
    print(f'Exporting media metadata to CSV file {csv_filename} ...')
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=colnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in metadata:
            writer.writerow(row)
    print(f'Finished exporting media metadata to {csv_filename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import metadata from VGG Image Search Engine (VISE) project")
    parser.add_argument("--vise-metadata-db", type=str, required=True, help="Path to the SQLite database created by VISE")
    parser.add_argument("--vise-join-colname", type=str, required=True, help="The column that maps to media_path")
    
    parser.add_argument('--out-csv-file',
                        required=True,
                        type=str,
                        help='Path to the output CSV file')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='WISE project folder (a new table will be created in metadata/internal.db')
    parser.add_argument
    args = parser.parse_args()
    import_metadata_from_vise(args.vise_metadata_db,
                             args.vise_join_colname,
                             args.project_dir,
                             args.out_csv_file)
