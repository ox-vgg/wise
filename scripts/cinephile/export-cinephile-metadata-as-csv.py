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

# TODO: update when src is available as a module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Ensure the src directory is in sys.path for wise_project import
print(f'Adding {project_root}/src to sys.path')
sys.path.append(project_root)

from src.wise_project import WiseProject
from src import db
from src.repository import MediaRepo

import sqlalchemy as sa
from tqdm import tqdm

def import_metadata_from_json_files(json_dir, project_dir, csv_filename):
    project = WiseProject(project_dir, create_project=False)
    project_assets = project.discover_assets()
    if len(project_assets) == 0:
        print(f'failed to load assets from {project_dir}')
        sys.exit(1)

    metadata = {}
    for json_file in Path(json_dir).rglob("*.json"):
        with open(json_file, 'r') as f:
            media_path = json_file.relative_to(json_dir).with_suffix('.mp4')
            metadata[media_path] = json.load(f)
            metadata[media_path]['media_path'] = str(media_path)

    if not metadata:
        print(f'No metadata found in JSON files in {json_dir}')
        sys.exit(1)

    # Process and export metadata to CSV
    db_engine = project.db_engine

    # add 'media_id' corresponding to each entry
    resolve_media_path(db_engine, metadata)
    print(f'resolved media_id for {len(metadata)} metadata rows')

    colnames = [
        'media_id',
        'media_path',
        'guid',
        'link',
        'provider',
        'title',
        'type',
        'year',
        'country',
        'language',
        'dataProvider',
        'dcContributor',
        'dcDescription',
        'edmTimespanLabel',
        'edmPreview',
        'edmPlaceLatitude',
        'edmPlaceLongitude',
        'edmPlaceLabel',
        'edmPlaceAltLabel',
        'edmDatasetName',
        'edmConceptLabel',
    ]
    export_media_metadata(db_engine, colnames, metadata, csv_filename)
    print(f'exported {len(metadata)} rows to {csv_filename}')

def resolve_media_path(db_engine, metadata):
    with db_engine.connect() as conn:
        failed_media_path = []
        for media_path in metadata:
            media_metadata = MediaRepo.get_row_by_column_match(conn,
                                                               column_name_to_match='path',
                                                               column_value=str(media_path))
            if media_metadata:
                metadata[media_path]['media_id'] = media_metadata.id
            else:
                metadata[media_path]['media_id'] = -1
                failed_media_path.append(media_path)
        if failed_media_path:
            print(f'failed to resolve media_path for these file paths: {failed_media_path}')
            raise ValueError(f'failed to resolved media_path for {len(failed_media_path)} metadata rows')

def export_media_metadata(db_engine, colnames, metadata, csv_filename):
    print(f'Exporting media metadata to CSV file {csv_filename} ...')
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
        # Use csv.QUOTE_ALL to ensure quoted fields are handled, preserving newlines and quotes
        writer = csv.DictWriter(
            csv_file,
            fieldnames=colnames,
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            doublequote=True,
            lineterminator='\n'
        )
        writer.writeheader()
        for media_path in tqdm(metadata, desc="Exporting metadata"):
            row = {}
            for colname in colnames:
                if colname not in metadata[media_path]:
                    row[colname] = ''
                    continue
                value = metadata[media_path][colname]
                if isinstance(value, list):
                    if all(str(item).strip().endswith('.') for item in value):
                        cell = ' '.join(str(item) for item in value)
                    else:
                        for item in value:
                            # some items may contain information in the 'def' key
                            if isinstance(item, dict) and 'def' in item:
                                cell = ', '.join(str(item['def']) for item in value)
                            else:
                                cell = ', '.join(str(item) for item in value)
                elif isinstance(value, dict):
                    items = [f"{k}: {v}" for k, v in value.items()]
                    if all(str(item).strip().endswith('.') for item in items):
                        cell = ' '.join(items)
                    else:
                        cell = ', '.join(items)
                else:
                    cell = str(value)
                row[colname] = cell
            writer.writerow(row)
    print(f'Finished exporting media metadata to {csv_filename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Cinephile metadata from JSON files to a CSV file.",
        epilog=(
            "More details about the Cinephile metadata are available at "
            "https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html"
        )
    )

    parser.add_argument('--json-dir',
                        required=True,
                        type=str,
                        help='Path to the directory containing JSON files with metadata')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='WISE project folder (a new table will be created in metadata/internal.db')

    parser.add_argument('--out-csv-file',
                        required=True,
                        type=str,
                        help='Path to the output CSV file')

    args = parser.parse_args()
    import_metadata_from_json_files(args.json_dir,
                                     args.project_dir,
                                     args.out_csv_file)
