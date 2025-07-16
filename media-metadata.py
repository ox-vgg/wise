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
from src.metadata_type import MetadataType

from src.data_models import (
    MediaMetadata,
    SourceCollection,
    ExtraMediaMetadata,
    VectorMetadata,
    MediaType,
    SourceCollectionType,
    VideoShot
)
from src.repository import (
    SourceCollectionRepo,
    MediaRepo,
    VectorRepo,
    MediaMetadataRepo,
    VideoShotsRepo,
)
from src import db
import sqlalchemy as sa
from tqdm import tqdm
import bisect
from collections import defaultdict

##
## A. Command line interface (CLI) parser and handler
##

def main():
    parser = argparse.ArgumentParser(prog='media-metadata',
                                     description='Manage metadata associated with media files contained in a WISE project',
                                     epilog='''
                                     Notes: Each column in the input CSV file can be referenced using column name (e.g. "filename").
                                     A column can be composed by combining two or more columns. For example,
                                     --col-filename "{participant_id}/videos/{video_id}.MP4" will construct filename using values
                                     taken from "participant_id" and "video_id" before matching it to one of the existing media
                                     files in the WISE project.''')

    parser.add_argument('command',
                        choices=['import', 'import-shots', 'import-shot-scale'],
                        nargs='?',
                        help='various modes of operation supported by the metadata script')

    parser.add_argument('--metadata-id',
                        required=False,
                        type=str,
                        help='imported metadata will be uniquely identified in WISE project using this id')

    parser.add_argument('--from-csv',
                        required=False,
                        type=str,
                        help='a CSV file containing metadata (must have column header and one of the columns must be media_id or media_path)')

    parser.add_argument('--metadata-type',
                        choices=['media', 'frame', 'segment', 'region'],
                        nargs='?',
                        help='WISE supports the following four types of metadata: [1] media file (media_id), [2] video frame (media_id, timestamp), [3] temporal segment (media_id, timestamp, end_timestamp), or [4] spatial region (media_id, timestamp, region_id)')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='folder where all project assets are stored')

    args = parser.parse_args()

    if(args.command == 'import'):
        import_media_metadata(args)
    elif(args.command == 'import-shots'):
        import_shots(args)
    elif(args.command == 'import-shot-scale'):
        import_shot_scale(args)
    else:
        print(f'unknown command {args.command}')

##
## Import Shots
##

def import_shots(args):
    project = WiseProject(args.project_dir, create_project=False, db_kwargs={'echo': False})
    db_engine = project.db_engine

    def add_shots(_metadata):
        with db_engine.begin() as conn:
            # delete all existing shots
            VideoShotsRepo.delete_all(conn)

        with db_engine.connect() as conn:
            for idx, m in enumerate(tqdm(_metadata), start=1):
                VideoShotsRepo.create(
                    conn, 
                    data=VideoShot(
                        id=m['id'],
                        media_id=m['media_id'],
                        ts=m['timestamp'],
                        te=m['end_timestamp'],
                    )
                )
                if (idx % 1024) == 0:
                    conn.commit()
            conn.commit()

            db_inspector = sa.inspect(db_engine)
            if db_inspector.has_table('vectors_to_shots_map'):
                # drop the table if it exists
                print('dropping existing vectors_to_shots_map table ...')
                db.project_metadata_obj.drop_table('vectors_to_shots_map')

            print('creating vectors_to_shots_map table ...')
            sqlalchemy_metadata = sa.MetaData()
            sqlalchemy_metadata.reflect(bind=db_engine)
            vectors_to_shots_map = sa.Table(
                'vectors_to_shots_map',
                sqlalchemy_metadata,
                sa.Column('vector_id', sa.Integer, sa.ForeignKey('vectors.id', ondelete="CASCADE"), primary_key=True, nullable=False),
                sa.Column('shot_id', sa.Integer, nullable=False),
                sa.Column('media_id', sa.Integer, nullable=False),
                sa.ForeignKeyConstraint(['shot_id', 'media_id'], ['shots.id', 'shots.media_id'], ondelete="CASCADE"),
            )
            sqlalchemy_metadata.create_all(db_engine)

            # Count total vectors to process for progress bar
            result = conn.execute(sa.text("SELECT COUNT(*) FROM vectors"))
            total_vectors = result.scalar() or 0
            print(f'Populating vectors_to_shots_map table for {total_vectors} vectors (takes a while) ...')

            # Fetch all vectors and shots into memory for mapping
            vectors = conn.execute(sa.text("SELECT id, media_id, timestamp FROM vectors")).fetchall()
            shots = conn.execute(sa.text("SELECT id, media_id, ts, te FROM shots")).fetchall()

            # Build a lookup for shots by media_id for efficient search
            shots_by_media = defaultdict(list)
            for shot in shots:
                shots_by_media[shot.media_id].append(shot)

            # Prepare insert statement
            insert_stmt = sa.text("""
                INSERT INTO vectors_to_shots_map (vector_id, shot_id, media_id)
                VALUES (:vector_id, :shot_id, :media_id)
            """)

            # Progress bar for mapping
            for vector in tqdm(vectors, desc="Mapping vectors to shots", unit="vector"):
                media_id = vector.media_id
                timestamp = vector.timestamp
                for shot in shots_by_media.get(media_id, []):
                    if shot.ts <= timestamp <= shot.te:
                        conn.execute(insert_stmt, {
                            "vector_id": vector.id,
                            "shot_id": shot.id,
                            "media_id": media_id
                        })
                        break  # Each vector maps to at most one shot

            conn.commit()
            print('Creating indices on vectors_to_shots_map ...')
            conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_vectors_vector_id ON vectors_to_shots_map (vector_id);"))
            conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_vectors_shot_and_media_id ON vectors_to_shots_map (shot_id, media_id);"))
            conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_vectors_modality_feat_media ON vectors (modality, feature_extractor_id, media_id)"))

    csv_filename = Path(args.from_csv)
    if not csv_filename.exists():
        raise ValueError(f'csv file does not exist: {csv_filename}')
    
    csv_colnames = get_csv_header(csv_filename)
    if 'media_id' not in csv_colnames and 'media_path' not in csv_colnames:
        raise ValueError('media_id or media_path columns missing from CSV')
    
    if 'id' not in csv_colnames or 'timestamp' not in csv_colnames or 'end_timestamp' not in csv_colnames:
        raise ValueError('id / timestamp / end_timestamp columns missing from CSV - make sure the correct script was used to geenrate the shots csv')
    
    metadata = load_metadata_from_csv(args.from_csv, args)
    if 'media_path' in csv_colnames:
        resolve_media_path(db_engine, metadata)


    add_shots(metadata)


##
## B. Import metadata
##

def import_media_metadata(args):
    project = WiseProject(args.project_dir, create_project=False, db_kwargs={'echo': False})
    project_assets = project.discover_assets()
    if len(project_assets) == 0:
        print(f'failed to load assets from {args.project_dir}')
        sys.exit(1)
    db_engine = project.db_engine
    db_inspector = sa.inspect(db_engine)

    metadata_tablename = project.metadata_tablename(args.metadata_id)
    if db_inspector.has_table(metadata_tablename):
        raise ValueError(f'metadata "{args.metadata_id}" already exists')

    if not args.from_csv:
        raise ValueError('--from-csv must point to a file containing metadata')

    csv_filename = Path(args.from_csv)
    if not csv_filename.exists():
        raise ValueError(f'csv file does not exist: {csv_filename}')

    csv_colnames = get_csv_header(csv_filename)
    if 'media_id' not in csv_colnames and 'media_path' not in csv_colnames:
        raise ValueError('media_id or media_path columns missing from CSV')

    metadata = load_metadata_from_csv(args.from_csv, args)
    if 'media_path' in csv_colnames:
        resolve_media_path(db_engine, metadata)

    add_media_metadata(db_engine, metadata_tablename, csv_colnames, metadata)

def get_csv_header(csv_filename):
    with open(csv_filename, 'r') as csv_file:
        data_sample = csv_file.read(1024)
        csv_file.seek(0)
        dialect = csv.Sniffer().sniff(sample=data_sample, delimiters=',')
        reader = csv.DictReader(csv_file, dialect=dialect)
        colnames = reader.fieldnames
        return colnames

def load_metadata_from_csv(csv_filename, args):
    print(f'Loading metadata from CSV file {csv_filename} ...')
    all_metadata = []
    with open(csv_filename, 'r') as csv_file:
        data_sample = csv_file.read(1024)
        csv_file.seek(0)
        dialect = csv.Sniffer().sniff(sample=data_sample, delimiters=',')

        reader = csv.DictReader(csv_file, dialect=dialect)

        all_metadata = [row for row in reader]
    return all_metadata

def resolve_media_path(db_engine, metadata):
    with db_engine.connect() as conn:
        failed_count = 0
        for i in range(0, len(metadata)):
            media_path = metadata[i]['media_path']
            media_metadata = MediaRepo.get_row_by_column_match(conn,
                                                               column_name_to_match='path',
                                                               column_value=media_path)
            if media_metadata:
                metadata[i]['media_id'] = media_metadata.id
            else:
                metadata[i]['media_id'] = -1
                failed_count += 1
        if failed_count:
            raise ValueError(f'failed to resolved media_path for {failed_count} metadata rows')

def add_media_metadata(db_engine, metadata_tablename, csv_colnames, media_metadata):
    colnames = []
    ## All external metadata must have these columns:
    ## media_id, timestamp, end_timestamp, vector_id
    ##
    ## We create an index on media_id column so that full text search results
    ## on metadata can be resolved to parent media files quickly.
    colnames.append( sa.Column('media_id',
                               sa.Integer,
                               sa.ForeignKey("media.id", ondelete="CASCADE"),
                               index=True,
                               nullable=False) )
    colnames.append( sa.Column('timestamp',
                               sa.Numeric(6,2),
                               nullable=True) )
    colnames.append( sa.Column('end_timestamp',
                               sa.Numeric(6,2),
                               nullable=True) )
    colnames.append( sa.Column('vector_id',
                               sa.Integer,
                               nullable=True) )
    for csv_colname in csv_colnames:
        if csv_colname == 'media_id' or csv_colname == 'media_path':
            continue
        colnames.append( sa.Column(csv_colname,
                                   sa.String,
                                   nullable=True) )

    sqlalchemy_metadata = sa.MetaData()
    sqlalchemy_metadata.reflect(bind=db_engine)
    metadata_table = sa.Table(metadata_tablename,
                            sqlalchemy_metadata,
                            *colnames)
    sqlalchemy_metadata.create_all(db_engine)
    with db_engine.connect() as conn:
        conn.execute(metadata_table.insert(), media_metadata)
        conn.commit()
    print(f'inserted {len(media_metadata)} rows into table {metadata_tablename}')

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

def metadata_exist(metadata_db, metadata_table):
    if metadata_db.exists():
        with sqlite3.connect( str(metadata_db) ) as sqlite_connection:
            cursor = sqlite_connection.cursor()
            res = cursor.execute(f'SELECT COUNT(*) FROM sqlite_master WHERE type="table" AND name="{metadata_table}"')
            if res == (1,):
                return True
    return False

##
## Import Shot Scale
## e.g. shot_scale \in { 0:'extreme close-up', 1:'close-up', 2:'medium shot', 3:'full shot', 4:'long shot'}
##
def import_shot_scale(args):
    project = WiseProject(args.project_dir, create_project=False, db_kwargs={'echo': False})
    db_engine = project.db_engine
    thumbsdb_engine = project.thumbsdb_engine

    def add_shot_scale(metadata):
        # 1. Load the shot_scale class for each thumbnail
        thumbnail_id_to_shot_scale = {}
        for row in metadata:
            thumbnail_id = int(row.get('thumbnail_id'))
            if thumbnail_id in thumbnail_id_to_shot_scale:
                raise ValueError(f'duplicate thumbnail_id of {thumbnail_id} found in CSV metadata')
            thumbnail_id_to_shot_scale[thumbnail_id] = int(row.get('shot_scale'))
        print(f'Loaded {len(thumbnail_id_to_shot_scale)} thumbnail_id to shot_scale mappings from CSV metadata')

        # 2. Group thumbnails by media
        thumbs_by_media = {}
        with thumbsdb_engine.connect() as thumbs_conn:
            result = thumbs_conn.execute(sa.text("SELECT id, media_id, timestamp FROM thumbnails ORDER BY media_id, timestamp"))
            for row in result:
                media_id = int(row.media_id)
                thumb_id = int(row.id)
                if media_id not in thumbs_by_media:
                    thumbs_by_media[media_id] = []
                thumbs_by_media[media_id].append((thumb_id, float(row.timestamp)))
        print(f'Loaded {len(thumbs_by_media)} thumbnails grouped by media')

        # 3. Compute the shot_scale for each shot
        with db_engine.connect() as conn:
            result = conn.execute(sa.text("SELECT id, media_id, ts, te FROM shots"))
            shot_to_scale = []
            for row in result:
                shot_id = int(row.id)
                shot_ts = float(row.ts)
                shot_te = float(row.te)
                media_id = int(row.media_id)
                # for each shot, find the shot_scale for each thumbnail
                thumbnails = thumbs_by_media.get(media_id, [])
                shot_scales = []
                thumb_timestamps = [ts for _, ts in thumbnails]
                # Find left and right indices for thumbnails within [shot_ts, shot_te] using binary search
                left = bisect.bisect_left(thumb_timestamps, shot_ts)
                right = bisect.bisect_right(thumb_timestamps, shot_te)
                for thumb_id, thumb_ts in thumbnails[left:right]:
                    if thumb_id in thumbnail_id_to_shot_scale:
                        shot_scales.append(thumbnail_id_to_shot_scale[thumb_id])
                # find the most common shot_scale for this shot
                if shot_scales:
                    most_common_scale = max(set(shot_scales), key=shot_scales.count)
                    shot_to_scale.append((media_id, shot_id, most_common_scale))
                    #print(f"media_id={media_id}, shot_id={shot_id}, shot-window=({shot_ts} to {shot_te}), shot_scale: {most_common_scale}")
                else:
                    print(f'No thumbnails found for shot {shot_id} in media {media_id} within the shot window ({shot_ts} to {shot_te})')
            # Insert/Update shot_scale for each shot in the shots table
            if shot_to_scale:
                # Add shot_scale column if it doesn't exist
                shots_columns = [col['name'] for col in sa.inspect(conn).get_columns('shots')]
                if 'shot_scale' not in shots_columns:
                    conn.execute(sa.text("ALTER TABLE shots ADD COLUMN shot_scale INTEGER"))
                for media_id, shot_id, shot_scale in shot_to_scale:
                    conn.execute(
                        sa.text("UPDATE shots SET shot_scale = :shot_scale WHERE id = :id and media_id = :media_id"),
                        {"shot_scale": shot_scale, "id": shot_id, "media_id": media_id}
                    )
                conn.commit()
                print(f"Updated shot_scale for {len(shot_to_scale)} shots.")

    csv_filename = Path(args.from_csv)
    if not csv_filename.exists():
        raise ValueError(f'csv file does not exist: {csv_filename}')

    csv_colnames = get_csv_header(csv_filename)
    if 'thumbnail_id' not in csv_colnames and 'shot_scale' not in csv_colnames:
        raise ValueError('thumbnail_id or shot_scale columns missing from CSV')
    shot_scale_metadata = load_metadata_from_csv(args.from_csv, args)
    add_shot_scale(shot_scale_metadata)

if __name__ == '__main__':
    main()
