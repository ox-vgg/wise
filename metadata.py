"""

Table of Contents
  A. Command line interface (CLI) parser and handler
  B. Import metadata

"""

import argparse
import sys
from pathlib import Path
import csv
import json
import sqlite3

from src.wise_project import WiseProject
from src import db

from src.data_models import (
    MediaMetadata,
    SourceCollection,
    ExtraMediaMetadata,
    VectorMetadata,
    MediaType,
    SourceCollectionType,
)
from src.repository import (
    SourceCollectionRepo,
    MediaRepo,
    VectorRepo,
    MediaMetadataRepo,
)

##
## A. Command line interface (CLI) parser and handler
##

def main():
    parser = argparse.ArgumentParser(prog='metadata',
                                     description='Manage metadata associated with media contained in a WISE project',
                                     epilog='''Notes: Each column in the input CSV file can be referenced using column name (e.g. "filename"). A column can be composed by combining two or more columns. For example, --col-filename "{participant_id}/videos/{video_id}.MP4" will construct filename using values taken from "participant_id" and "video_id" before matching it to one of the existing media files in the WISE project.''')

    parser.add_argument('command',
                        choices=['import'],
                        nargs='?',
                        help='various modes of operation supported by the metadata script')

    parser.add_argument('--from-csv',
                        required=False,
                        type=str,
                        help='a CSV filename, must have a following column header as the first line')

    parser.add_argument('--metadata-name',
                        required=False,
                        type=str,
                        help='a unique id associated with all the metadata; append if already exists')

    parser.add_argument('--col-metadata-id',
                        required=False,
                        type=str,
                        help='column containing the unique id of each row of metadata (see notes)')

    parser.add_argument('--col-filename',
                        required=False,
                        type=str,
                        help='filename column that maps to existing media files in the WISE project (see notes)')

    parser.add_argument('--col-starttime',
                        required=False,
                        type=str,
                        help='column containing start time of a temporal segment (see notes)')

    parser.add_argument('--col-stoptime',
                        required=False,
                        type=str,
                        help='column containing stop time of the temporal segment (see notes)')

    parser.add_argument('--col-metadata',
                        required=False,
                        action='append',
                        type=str,
                        help='column(s) containing metadata of the temporal segment (see notes)')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='folder where all project assets are stored')

    args = parser.parse_args()

    if(args.command == 'import'):
        import_metadata(args)
    else:
        print(f'unknown command {args.command}')

##
## B. Import metadata
##

def import_metadata(args):
    project = WiseProject(args.project_dir, create_project=False)
    project_assets = project.discover_assets()
    if len(project_assets) == 0:
        print(f'failed to load assets from {args.project_dir}')
        sys.exit(1)
    db_engine = db.init_project(project.dburi, echo=False)

    if args.from_csv:
        csv_filename = Path(args.from_csv)
        if not csv_filename.exists():
            print(f'csv does not exist: {csv_filename}')
        metadata = load_metadata_from_csv(args.from_csv, args)

    metadata_count = len(metadata)
    if metadata_count == 0:
        print(f'metadata not found')
        return

    valid_metadata = get_valid_metadata(metadata, db_engine)

    # 2. Count the timestamps that lie within the valid range of existing media's duration
    metadata_db = project.metadata_filename()
    metadata_table = args.metadata_name
    add_metadata(metadata_db, metadata_table, valid_metadata)

def load_metadata_from_csv(csv_filename, args):
    metadata_store = []
    with open(csv_filename, 'r') as csv_file:
        if not csv.Sniffer().has_header(csv_file.read(2048)):
            print(f'csv file must have a header row')
            sys.exit(1)
        csv_file.seek(0)
        dialect = csv.Sniffer().sniff(csv_file.read(2048))
        csv_file.seek(0)
        reader = csv.DictReader(csv_file, dialect=dialect)
        colnames = reader.fieldnames

        for row in reader:
            try:
                metadata_id = get_csv_row_col_value(row, args.col_metadata_id)
                filename = get_csv_row_col_value(row, args.col_filename)
                starttime = get_csv_row_col_value(row, args.col_starttime)
                stoptime = get_csv_row_col_value(row, args.col_stoptime)
                metadata = {}
                for col_id in args.col_metadata:
                    metadata[col_id] = row[col_id]
                metadata_store.append( {
                    'filename': filename,
                    'metadata_id': metadata_id,
                    'anchor': 'media_segment', # media, media_segment, image_region, media_region, ...
                    'starttime': time2sec(starttime),
                    'stoptime': time2sec(stoptime),
                    'metadata': metadata
                })
            except ex:
                print(f'Error parsing row: {row}')
    return metadata_store

def get_valid_metadata(metadata, db_engine):
    mismatched_timestamp_count = 0
    missing_filename_list = []
    mismatch_filename_count = 0
    valid_metadata = []
    with db_engine.connect() as conn:
        for i in range(0, len(metadata)):
            filename = metadata[i]['filename']
            media_metadata = MediaRepo.get_row_by_column_match(conn,
                                                               column_name_to_match='path',
                                                               column_value=filename)
            if media_metadata:
                duration = float(media_metadata.duration)
                starttime = metadata[i]['starttime']
                stoptime = metadata[i]['stoptime']
                if starttime < 0 or starttime >= duration and stoptime < 0 and stoptime >= duration:
                    mismatched_timestamp_count += 1
                    print(f'Discarding malformed media segment: {row}')
                else:
                    valid_metadata.append( metadata[i] )
            else:
                if filename not in missing_filename_list:
                    missing_filename_list.append(filename)
                mismatch_filename_count += 1

    print(f'Adding {len(valid_metadata)} rows of metadata (discarded {len(metadata) - len(valid_metadata)} rows)')
    if mismatch_filename_count:
        print(f'  - there were {mismatch_filename_count} rows in the input metadata file whose filename were not found in WISE project')
        print(f'  - missing filenames: {missing_filename_list}')
    if mismatched_timestamp_count:
        print(f'  - discarding {mismatched_timestamp_count} row that contained malformed media segment')

    return valid_metadata

def add_metadata(metadata_db, metadata_table, metadata):
    print(f'add_metadata(): db={metadata_db}, table={metadata_table}, metadata={len(metadata)}')
    # TODO

##
## Helper functions
##
def get_csv_row_col_value(row, col_id):
    if '{' in col_id and '}' in col_id:
        col_value = col_id.format(**row)
    else:
        col_value = row[col_id]
    return col_value

def time2sec(time):
    if isinstance(time, int) or isinstance(time, float):
        return float(time)
    if isinstance(time, str):
        if ':' in time:
            return hhmmss_to_sec(time)
        else:
            try:
                time_sec = float(time)
                return time_sec
            except ex:
                print(ex)

def hhmmss_to_sec(hhmmss):
    tok = hhmmss.split(':')
    assert len(tok) == 3
    hh = int(tok[0])
    mm = int(tok[1])
    ssms_tok = tok[2].split('.')
    ss = int(ssms_tok[0])
    ms = int(ssms_tok[1])
    sec = hh*60*60 + mm*60 + ss + ms/100.0
    return float(sec)

if __name__ == '__main__':
    main()
