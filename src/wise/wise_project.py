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

import io
import itertools
import logging
import os
import shutil
import sqlite3
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from functools import partial

from wise.utils import batched

import numpy as np
import sqlalchemy as sa
from PIL import Image
from tqdm import tqdm

from wise import db as wise_db
from wise.data_models import (
    DatasetPayload,
    MediaMetadataWithSource,
    MediaType,
    ModalityType,
    SourceCollection,
    VectorAndMediaMetadata,
    VideoShot,
)
from wise.dataloader import AVDataset
from wise.feature.feature_extractor_factory import get_feature_extractor_class
from wise.feature.store import (
    FeatureStore,
    FeatureStoreFactory,
    FeatureStoreType,
)
from wise.index.search_index import SearchIndex
from wise.index.search_index_factory import SearchIndexFactory
from wise.repository import (
    SourceCollectionRepo,
)
from wise.search.fts import FTSSearch


logger = logging.getLogger(__name__)
DB_SCHEME = "sqlite+pysqlite://"


def get_cte_from_ids(ids: list[int], label="media_id"):
    """
    Create a CTE from a list of media_ids using the values expression in SQLAlchemy
    """
    cte = (
        sa.values(
            sa.column("rank", sa.Integer),
            sa.column(label, sa.Integer),
        )
        .data([(i, m) for i, m in enumerate(ids)])
        .cte("cte")
    )

    return cte


# utility to scan columns from a table in batches, either as a row tuple or dict
def batch_select(
    conn: sa.Connection,
    stmt,
    batch_size: int | None = None,
    as_dict: bool = False,
):
    _conn = conn
    if batch_size is not None:
        _conn = conn.execution_options(stream_results=True)

    result = _conn.execute(stmt)

    while rows := result.fetchmany(batch_size):
        if as_dict:
            yield [r._asdict() for r in rows]
        else:
            yield rows


# utility to insert a batch of rows into a table, and optionally return the inserted ids
def batch_insert(
    conn: sa.Connection, table: sa.Table, batch: list[dict], return_ids: bool = True
):
    stmt = table.insert()
    if return_ids:
        stmt = stmt.returning(table.c.id)

    result = conn.execute(
        stmt,
        batch,
    )
    if return_ids:
        return result.scalars().all()


# utility to query a table by a column with a list of values, preserving the order of the input list
def batch_query_by_column(
    conn: sa.Connection, table: sa.Table, column: str, vals: list
):
    if column not in table.c:
        raise ValueError(f"column {column} not found in table {table.name}")

    id_ordering = sa.case(
        {_id: index for index, _id in enumerate(vals)},
        value=table.c.id,
    )

    stmt = sa.select(table).where(table.c[column].in_(vals)).order_by(id_ordering)
    return conn.execute(stmt).mappings().all()


class WiseProject:
    """
    WISE Project class that encapsulates all operations related to a WISE project folder.


    """
    def __init__(self, project_dir: Path, *, create_project=False, read_only=False, **kwargs):
        self.project_dir = Path(project_dir)
        self._store_dir = self.project_dir / "store"
        self._media_dir = self.project_dir / "media"
        self.metadata_dir = self.project_dir / "metadata"
        self.media_type_list = ["image", "video", "audio"]
        self.read_only = read_only

        if create_project and read_only:
            raise ValueError("options create_project and read_only are mutually exclusive")

        if not self.project_dir.exists():
            if create_project:
                # create the root folders
                self.store_dir.mkdir(parents=True, exist_ok=True)
                self.media_dir.mkdir(parents=True, exist_ok=True)
                self.metadata_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f"project folder {self.project_dir} does not exist")

        self._db_kwargs = kwargs.get('db_kwargs', {})
        self._thumbsdb_kwargs = kwargs.get('thumbsdb_kwargs', {})

        self._search_indices = None

    @property
    def name(self) -> str:
        """
        Return the project directory name.

        """
        return self.project_dir.name

    @property
    def thumbs_uri(self) -> str:
        """
        Return the SQLAlchemy URI for the thumbnails database.

        """
        if self.read_only:
            return f"{DB_SCHEME}/file:{self.project_dir.absolute()}/thumbs.db?mode=ro&uri=true"
        return f"{DB_SCHEME}/{self.project_dir.absolute()}/thumbs.db"

    @property
    def dburi(self) -> str:
        """
        Return the SQLAlchemy URI for the main project database.
        """
        if self.read_only:
            return f"{DB_SCHEME}/file:{self.metadata_dir.absolute()}/internal.db?mode=ro&uri=true"
        return f"{DB_SCHEME}/{self.metadata_dir.absolute()}/internal.db"

    @cached_property
    def db_engine(self):
        """
        Return the SQLAlchemy engine for the main project database.
        """
        return wise_db.init_project(self.dburi, **self._db_kwargs)

    @cached_property
    def thumbsdb_engine(self):
        """
        Return the SQLAlchemy engine for the thumbnails database.
        """
        return wise_db.init_thumbs(self.thumbs_uri, **self._thumbsdb_kwargs)

    @cached_property
    def db_inspector(self):
        return sa.inspect(self.db_engine)

    @property
    def search_indices(self):
        return self._search_indices

    @property
    def fts_config_file(self) -> Path:
        return self.metadata_dir / 'fts_config.json'

    def metadata_db_table(self, metadata_id: str, extension='.sqlite') -> tuple[Path, str]:
        metadata_id_tok = metadata_id.split('/')
        assert len(metadata_id_tok) == 3, 'metadata_id must be in "FOLDER_NAME/DB_NAME/TABLE_NAME" format'
        metadata_db_dir = self.metadata_dir / metadata_id_tok[0]
        metadata_db_dir.mkdir(parents=True, exist_ok=True)
        metadata_db = metadata_db_dir / (metadata_id_tok[1] + extension)
        metadata_table = metadata_id_tok[2]
        return metadata_db, metadata_table

    def metadata_tablename(self, metadata_id: str) -> str:
        return "metadata-" + metadata_id

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    @property
    def media_dir(self) -> Path:
        return self._media_dir

    def features_root(self, feature_extractor_id: str) -> Path:
        return self.store_dir / feature_extractor_id

    def features_dir(self, feature_extractor_id: str) -> Path:
        return self.features_root(feature_extractor_id) / 'features'

    def create_features_dir(self, feature_extractor_id: str) -> Path:
        features_store = self.features_dir(feature_extractor_id)
        if not features_store.exists():
            features_store.mkdir(parents=True, exist_ok=True)
        return features_store

    def index_dir(self, feature_extractor_id: str) -> Path:
        return self.features_root(feature_extractor_id) / 'index'

    def create_index_dir(self, feature_extractor_id: str) -> Path:
        index_store = self.features_root(feature_extractor_id) / 'index'
        if not index_store.exists():
            index_store.mkdir(parents=True, exist_ok=True)
        return index_store

    @property
    def supported_modality_types_and_features(self) -> dict[ModalityType, set[str]]:
        with self.db_engine.connect() as conn:
            results = conn.execute(
                sa.select(
                    wise_db.vectors_table.c.modality,
                    wise_db.vectors_table.c.feature_extractor_id
                ).distinct()
            ).all()
        ## Do sorting in Python because it is very very slow in SQL
        ## (GROUP BY + ORDER BY) for very large tables.  Not sure why
        ## it is so slow but my guess is that SQL does not know that
        ## there are only a few modality+feature_extractor_id pairs.
        ## Maybe once https://gitlab.com/vgg/wise/wise/-/issues/138,
        ## then doing the whole thing in SQL will be just as fast.
        results = sorted(
            sorted(results, key=lambda x: x[1]), key=lambda x: x[0]
        )

        _assets = None
        _supported = defaultdict(set)
        for g, vals in itertools.groupby(results, key = lambda x: x[0]):
            feature_extractor_ids = set([v[1] for v in vals])
            if "" in feature_extractor_ids:
                # Fallback - find it by globbing
                if _assets is None:
                    _assets = self.discover_assets()
                feature_extractor_ids = set(_assets[g].keys())

            _supported[g].update(feature_extractor_ids)
        return _supported

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
        metadata_assets = {}
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
                        metadata_assets[metadata_id] = {
                            'metadata_db': str(metadata_db),
                            'metadata_db_type': 'sqlite',
                            'metadata_table': table_name
                        }
        if metadata_assets:
            self.assets['metadata'] = metadata_assets
        return self.assets

    def get_media_files(self) -> list[DatasetPayload]:
        media_files = []
        with self.db_engine.connect() as conn:
            stmt = (
                sa.select(
                    wise_db.media_table.c.id,
                    (wise_db.source_collections_table.c.location + '/' + wise_db.media_table.c.path).label('media_path'),
                    wise_db.media_table.c.media_type
                )
                .select_from(
                    wise_db.media_table.join(
                        wise_db.source_collections_table,
                        wise_db.media_table.c.source_collection_id == wise_db.source_collections_table.c.id
                    )
                )
            )
            rows = conn.execute(stmt)

            for row in rows:
                media_files.append(
                    DatasetPayload(row.id, row.media_path, row.media_type)
                )
        return media_files

    def get_shots(self) -> list[dict]:
        logger.info("Fetching shots from '%s' ...", self.dburi)
        shots = {}
        with self.db_engine.connect() as conn:
            stmt = sa.select(
                wise_db.shots_table.c.media_id,
                wise_db.shots_table.c.ts,
                wise_db.shots_table.c.te,
                wise_db.shots_table.c.id.label('shot_id')
            ).order_by(
                wise_db.shots_table.c.media_id,
                wise_db.shots_table.c.ts
            )
            rows = conn.execute(stmt)
            for row in rows:
                if row.media_id not in shots:
                    shots[row.media_id] = []
                shots[row.media_id].append({
                    'start_time': row.ts,
                    'end_time': row.te,
                    'shot_id': row.shot_id
                })
        return shots

    @property
    def num_media(self) -> int:
        with self.db_engine.connect() as conn:
            result = conn.execute(
                sa.select(sa.func.count(wise_db.media_table.c.id))
            ).scalar_one()

        return result

    @property
    def num_vectors(self) -> int:
        with self.db_engine.connect() as conn:
            result = conn.execute(
                sa.select(sa.func.count(wise_db.vectors_table.c.id))
            ).scalar_one()

        return result

    @property
    def num_thumbnails(self) -> int:
        with self.thumbsdb_engine.connect() as conn:
            result = conn.execute(
                sa.select(sa.func.count(wise_db.thumbnails_table.c.id))
            ).scalar_one()

        return result

    @property
    def total_duration(self) -> float:
        with self.db_engine.connect() as conn:
            ## COALESCE(SUM(...), 0.0) to cover the case of no media
            ## in the project (otherwise SUM([]) -> NULL).
            result = conn.execute(
                sa.select(
                    sa.func.coalesce(
                        sa.func.sum(wise_db.media_table.c.duration), 0.0
                    )
                )
            ).scalar_one()

        return result

    @property
    def media_file_counts(self) -> dict[MediaType, int]:
        """
        Get the number of media files for each media type (image, video, audio).
        Note that the "av" and "video" media types are both counted as "video".
        """
        with self.db_engine.connect() as conn:
            results = (
                conn.execute(
                    sa.select(
                        wise_db.media_table.c.media_type,
                        sa.func.count(wise_db.media_table.c.id),
                    ).group_by(wise_db.media_table.c.media_type)
                )
                .tuples()
                .all()
            )
            media_counts = {media_type: count for media_type, count in results}
            if MediaType.AV in media_counts:
                media_counts[MediaType.VIDEO] = (
                    media_counts.get(MediaType.VIDEO, 0) + media_counts[MediaType.AV]
                )
                del media_counts[MediaType.AV]
            return media_counts

    @property
    def num_shots(self) -> int:
        with self.db_engine.connect() as conn:
            result = conn.execute(
                sa.select(sa.func.count(wise_db.shots_table.c.id))
            ).scalar_one()

        return result

    def external_metadata_tables(self):
        return wise_db.reflect_external_metadata(self.db_engine)

    def enable_fts(self):
        _table = wise_db.project_metadata_obj.tables.get(wise_db._WISE_FTS_TABLE)
        if _table is not None:
            logger.debug(f"{wise_db._WISE_FTS_TABLE} table already loaded")
            return True

        if self.db_inspector.has_table(wise_db._WISE_FTS_TABLE):
            wise_db.project_metadata_obj.reflect(
                bind=self.db_engine, only=[wise_db._WISE_FTS_TABLE]
            )
            logger.debug(f"Loaded {wise_db._WISE_FTS_TABLE} table")
            return True
        return False

    def enable_shot_scale(self):
        _table = wise_db.project_metadata_obj.tables.get("vectors_to_shots_map")
        if _table is not None:
            logger.debug(f"vectors_to_shots_map table already loaded")
            return True

        if not self.db_inspector.has_table(wise_db.shots_table.name):
            logger.warning("No Shots table found. Cannot enable shot_scale filtering.")
            return False

        colnames = [
            col["name"]
            for col in self.db_inspector.get_columns(wise_db.shots_table.name)
        ]
        if "shot_scale" not in colnames:
            logger.warning(
                "No shot_scale column found in Shots table. Cannot enable shot_scale filtering."
            )
            return False

        if not self.db_inspector.has_table("vectors_to_shots_map"):
            logger.warning(
                "No vectors_to_shots_map table found. Cannot enable shot_scale filtering."
            )
            return False

        wise_db.project_metadata_obj.reflect(
            bind=self.db_engine, only=["vectors_to_shots_map"]
        )
        return True

    def shot_scales(self):
        """Get all unique shot scales from the shots table."""
        if not self.enable_shot_scale():
            raise ValueError(
                "vectors_to_shots_map table not found! Please run the import shots script as follows:"
                'Please run "python3 -m wise media-metadata import-shot-scale ..."'
            )

        with self.db_engine.connect() as conn:
            shot_scales = (
                conn.execute(
                    sa.text(
                        "select distinct(shot_scale) from shots ORDER BY shot_scale"
                    )
                )
                .scalars()
                .all()
            )
            shot_scales = list(filter(None, shot_scales))
            return shot_scales

    def num_thumbnails_for_media_id(self, media_id: int):
        _thumbs_table = wise_db.thumbnails_table
        with self.thumbsdb_engine.connect() as thumbs_conn:
            num_thumbs = thumbs_conn.execute(
                sa.select(sa.func.count(_thumbs_table.c.id)).where(
                    _thumbs_table.c.media_id == media_id
                )
            ).scalar_one()

            if num_thumbs == 0:
                raise ValueError(f"no thumbnails for media id {media_id}")

            return num_thumbs

    def thumbnail_size_for_media_id(self, media_id: int):
        # Assumes all thumbnails have the same size
        _thumbs_table = wise_db.thumbnails_table
        with self.thumbsdb_engine.connect() as thumbs_conn:
            one_thumb = thumbs_conn.execute(
                sa.select(_thumbs_table.c.content)
                .where(_thumbs_table.c.media_id == media_id)
                .limit(1)
            ).scalar_one()

            with Image.open(io.BytesIO(one_thumb)) as im:
                w, h = im.size

            return w, h

    def get_thumbnails(
        self,
        _video_media_id: int,
        num_seconds_per_image: int,
        partition_id: int | None = None,
        num_thumbs_per_partition: int = 300,
    ):
        _thumbs_table = wise_db.thumbnails_table
        # # For videos longer than 30 minutes, reduce the frequency of thumbnails to 1 every 4 seconds
        # num_seconds_per_image = 2 if num_thumbs < (2 * 30 * 60) else 4
        num_images_per_partition = (
            num_thumbs_per_partition if partition_id is not None else None
        )
        offset = (
            partition_id * num_images_per_partition if partition_id is not None else 0
        )
        cols = [
            _thumbs_table.c.id,
            _thumbs_table.c.timestamp,
        ]
        if partition_id is not None:
            # add thumbnail data as well
            cols.append(_thumbs_table.c.content)

        stmt = (
            sa.select(*cols)
            .where(
                sa.and_(
                    _thumbs_table.c.media_id == _video_media_id,
                    (10 * _thumbs_table.c.timestamp) % (10 * num_seconds_per_image)
                    == 0,
                )
            )
            .order_by(_thumbs_table.c.timestamp)
            .offset(offset)
            .limit(num_images_per_partition)
        )
        with self.thumbsdb_engine.connect() as thumbs_conn:
            return thumbs_conn.execute(stmt).all()

    def metadata(self, media_id: str):
        stmt = sa.select(wise_db.media_table, wise_db.source_collections_table).select_from(
            wise_db.media_table.join(
                wise_db.source_collections_table,
                wise_db.media_table.c.source_collection_id == wise_db.source_collections_table.c.id,
            )
        ).where(
            wise_db.media_table.c.id == media_id
        )
        with self.db_engine.connect() as conn:
            row = conn.execute(stmt).first()
            if row is None:
                return None

            media_metadata = {
                col.name: row._mapping[col] for col in wise_db.media_table.c
            }
            source_collection = {
                col.name: row._mapping[col]
                for col in wise_db.source_collections_table.c
            }

            media_metadata = MediaMetadataWithSource.model_validate(
                {**media_metadata, "source_collection": source_collection}
            )
            return media_metadata

    def thumbnail(
        self,
        media_id: str,
        timestamp: float,
        get_id_only: bool = False,
        highres: bool = False,
    ) -> bytes | int | None:
        """
        Get the thumbnail from a video given a `media_id` and a `timestamp` (finds the first thumbnail between `timestamp - 0.25` and `timestamp + 2`).

        Parameters
        ----------
        media_id : int
            Media id of the video file you want to get the thumbnail from
        timestamp : float
            Timestamp within the video
        get_id_only : bool, optional
            If set to True, the integer id of the matching thumbnail is returned.
            If set to False (default), the raw bytes of the thumbnail is returned.
        highres : bool, optional
            If set to True, retrieves a high-resolution thumbnail by seeking the video file directly.
        Returns
        -------
        bytes | int | None
            Returns the raw bytes of the thumbnail in JPEG format.
            If `get_id_only` was set to True, then the integer id of the thumbnail is returned instead.
            If no thumbnail was found, the return value is None.
        """
        # TODO Convert timestamp search interval to a project configuration and pass it down
        _thumbs_table = wise_db.thumbnails_table
        if get_id_only or not highres:
            if highres:
                logger.warning(
                    "highres thumbnail retrieval is not compatible with get_id_only=True, ignoring highres flag"
                )
            start_timestamp_expr = _thumbs_table.c.timestamp >= timestamp - 0.25
            end_timestamp_expr = _thumbs_table.c.timestamp <= timestamp + 2
            stmt = (
                sa.select(
                    _thumbs_table.c.content if not get_id_only else _thumbs_table.c.id
                )
                .where(_thumbs_table.c.media_id == media_id)
                .where((start_timestamp_expr & end_timestamp_expr))
                .order_by(_thumbs_table.c.timestamp)
            )
            with self.thumbsdb_engine.connect() as conn:
                result = conn.execute(stmt)
                return result.scalar()

        # highres thumbnail retrieval
        media_metadata = self.metadata(media_id)
        if media_metadata is None:
            return None

        file_path = media_metadata.full_path

        # TODO should be in project config / read from db after extract-features
        audio_sampling_rate = 48_000  # (48 kHz)
        video_frame_rate = 2  # fps
        video_frames_per_chunk = 8  # frames
        segment_length = (
            video_frames_per_chunk / video_frame_rate
        )  # frames / fps = seconds
        audio_segment_length = segment_length  # seconds
        audio_frames_per_chunk = int(round(audio_sampling_rate * audio_segment_length))
        offset = 4 * ((timestamp) // 4)

        stream = AVDataset(
            [str(file_path)],
            video_frames_per_chunk=video_frames_per_chunk,
            audio_samples_per_chunk=audio_frames_per_chunk,
            video_frame_rate=video_frame_rate,
            audio_sample_rate=audio_sampling_rate,
            offset=offset,
            thumbnails=True,
        )
        for _, (mid, chunks) in enumerate(stream):
            video = chunks["video"]
            logger.debug(f"media_id: {mid}, pts: {video.pts}")
            if not video:
                break

            if video.pts != offset:
                continue

            if video.pts > timestamp:
                break

            n = int((timestamp - video.pts) // 0.5)
            logger.debug(f"index: {n}, pts: {video.pts}, ts: {timestamp}")
            arr = video.tensor[n].numpy().astype(np.uint8).transpose(1, 2, 0)
            with Image.fromarray(arr) as im:
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=90)
                return buf.getvalue()

    def shot(self, media_id: int, timestamp: float):
        # Join the table and query by dataset_path, and return the id
        shots_table = wise_db.shots_table
        start_timestamp_expr = (timestamp + 0.2) >= shots_table.c.ts
        end_timestamp_expr = timestamp < shots_table.c.te
        dataset_expr = shots_table.c.media_id == media_id
        stmt = sa.select(
            shots_table.c.id, shots_table.c.media_id, shots_table.c.ts, shots_table.c.te
        ).where((dataset_expr & start_timestamp_expr & end_timestamp_expr))
        with self.db_engine.connect() as conn:
            result = conn.execute(stmt)
            for row in result.mappings():
                yield VideoShot.model_validate(row)

    def related_vectors(self, vector_id: int):
        """Get all other vectors for same image/timestamp, modality, and feature extractor."""
        _vtable = wise_db.vectors_table
        subq = sa.select(_vtable).where(_vtable.c.id == vector_id).subquery()
        stmt = (
            sa.select(_vtable)
            .join(
                subq,
                (
                    (_vtable.c.media_id == subq.c.media_id)
                    & (_vtable.c.timestamp == subq.c.timestamp)
                    & (_vtable.c.modality == subq.c.modality)
                    & (_vtable.c.feature_extractor_id == subq.c.feature_extractor_id)
                ),
            )
            .where(_vtable.c.id != vector_id)
        )
        with self.db_engine.connect() as conn:
            return list(conn.execute(stmt))

    def get_vector_media_metadata_for_ids(
        self, ids: list[int], external_metadata_tables=[]
    ) -> list[VectorAndMediaMetadata]:
        """
        Get the vector and media metadata for a batch of vector ids.

        Parameters
        ----------
        ids : list of int
            List of vector ids

        Returns
        -------
        list of VectorAndMediaMetadata
            Returns a list of VectorAndMediaMetadata objects (a combination of the
            VectorMetadata and corresponding MediaMetadata). Each item in the list
            corresponds to the an id from the input `ids`.

        Raises
        ------
        RuntimeError
            If the metadata for one or more ids could not be retrieved, e.g. due to the ids being invalid.
        """
        _vtable = wise_db.vectors_table
        _mtable = wise_db.media_table

        ordering = sa.case(
            {id: index for index, id in enumerate(ids)},
            value=_vtable.c.id,
        )

        stmt1 = (
            sa.select(_vtable.c, _mtable.c)
            .select_from(_vtable.join(_mtable))
            .where(_vtable.c.id.in_(ids))
            .order_by(ordering)
        )
        with self.db_engine.connect() as conn:
            res1 = conn.execute(stmt1)
            res1 = [
                VectorAndMediaMetadata.model_validate(row) for row in res1.mappings()
            ]

        if len(external_metadata_tables) == 0:
            if len(res1) != len(ids):
                raise RuntimeError(
                    f"Unable to retrieve metadata for all ids. Retrieved metadata for {len(res1)}/{len(ids)} ids"
                )
            return res1

        ## collect external metadata
        from_clause = _vtable.join(_mtable)
        for external_metadata_table in external_metadata_tables:
            from_clause = from_clause.join(external_metadata_table)
        external_metadata_colnames = []
        for external_metadata_table in external_metadata_tables:
            for col in external_metadata_table.columns:
                if col.name not in [
                    "media_id",
                    "timestamp",
                    "end_timestamp",
                    "vector_id",
                ]:
                    external_metadata_colnames.append(col)
        stmt2 = (
            sa.select(*external_metadata_colnames)
            .select_from(from_clause)
            .where(_vtable.c.id.in_(ids))
            .order_by(ordering)
        )

        with self.db_engine.connect() as conn:
            res2 = conn.execute(stmt2)
            for m, r in zip(res1, res2.mappings()):
                m.external_metadata = r
        if len(res1) != len(ids):
            raise RuntimeError(
                f"Unable to retrieve metadata for all ids. Retrieved metadata for {len(res1)}/{len(ids)} ids"
            )
        return res1

    def get_vector_ids(
        self,
        media_ids: list[int],
        modality: MediaType,
        feature_extractor_id: str,
        timestamps: list[float] | None = None,
    ):
        """Get vector ids for the given media ids constraints.

        If timestamps are provided (aligned to media_ids), return vector ids
        that match both media_id and timestamp.
        """
        if not media_ids:
            return []

        _vtable = wise_db.vectors_table

        if timestamps is not None:
            if len(timestamps) != len(media_ids):
                raise ValueError("media_ids and timestamps must have the same length")

            cte = (
                sa.values(
                    sa.column("rank", sa.Integer),
                    sa.column("media_id", sa.Integer),
                    sa.column("timestamp", sa.Float),
                )
                .data([(i, mid, ts) for i, (mid, ts) in enumerate(zip(media_ids, timestamps))])
                .cte("cte")
            )
            stmt = sa.select(_vtable.c.id).select_from(
                cte.join(
                    _vtable,
                    sa.and_(
                        _vtable.c.media_id == cte.c.media_id,
                        _vtable.c.timestamp == cte.c.timestamp,
                        _vtable.c.modality == modality,
                        _vtable.c.feature_extractor_id == feature_extractor_id,
                    ),
                )
            ).order_by(cte.c.rank)
        else:
            media_cte = get_cte_from_ids(media_ids, label="media_id")
            stmt = sa.select(_vtable.c.id).select_from(
                media_cte.join(
                    _vtable,
                    sa.and_(
                        _vtable.c.media_id == media_cte.c.media_id,
                        _vtable.c.modality == modality,
                        _vtable.c.feature_extractor_id == feature_extractor_id,
                    ),
                )
            ).order_by(media_cte.c.rank)

        with self.db_engine.connect() as conn:
            return conn.execute(stmt).scalars().all()

    def get_vector_ids_for_shot_scale(
        self, shot_scales: list[int], modality: MediaType, feature_extractor_id: str
    ):
        """Get vector ids for the given shot scale constraints."""

        if not self.enable_shot_scale():
            raise ValueError(
                "Unable to use shot_scale filtering, missing tables or columns. Make sure to import shots and shot_scales before using this method."
            )
        vectors_to_shots_map = wise_db.project_metadata_obj.tables[
            "vectors_to_shots_map"
        ]

        vtable = wise_db.vectors_table
        shots_table = wise_db.shots_table

        stmt = (
            sa.select(vtable.c.id)
            .select_from(
                shots_table.join(
                    vectors_to_shots_map,
                    sa.and_(
                        shots_table.c.id == vectors_to_shots_map.c.shot_id,
                        shots_table.c.media_id == vectors_to_shots_map.c.media_id,
                    ),
                ).join(
                    vtable,
                    sa.and_(
                        vtable.c.id == vectors_to_shots_map.c.vector_id,
                        vtable.c.media_id == shots_table.c.media_id,
                        vtable.c.modality == modality,
                        vtable.c.feature_extractor_id == feature_extractor_id,
                    ),
                )
            )
            .where(
                shots_table.c.shot_scale.in_(shot_scales),
            )
        )
        with self.db_engine.connect() as conn:
            return conn.execute(stmt).scalars().all()

    def get_vector_ext_metadata_for_ids(
        self, feature_extractor_id: str, ids: list[int]
    ) -> list:
        """
        Get the external metadata for a batch of vector ids.

        Args:
            feature_extractor_id: The Feature Extractor ID associated with the vectors.
            ids: A list of vector IDs for which to retrieve the external metadata.

        Returns:
            A list of external metadata corresponding to the provided vector IDs.

        Raises:
            ValueError: If the provided feature_extractor_id is unknown.

        """
        cls = get_feature_extractor_class(feature_extractor_id)
        if cls is None:
            raise ValueError(f"unknown feature extractor id {feature_extractor_id}")

        with self.db_engine.connect() as conn:
            return cls.get_vector_metadata(conn, ids)

    def load_search_indices(
        self, preferred_index_type: str = "IndexFlatIP", default_nprobe: int = 32
    ):
        """
        Load all available search indices by default
        `search_indices` is a dictionary of SearchIndex objects, where the key is the
        feature_extractor_id and value is a SearchIndex object
        """
        if self.search_indices is not None:
            return self.search_indices

        search_indices: dict[str, dict[str, SearchIndex]] = {}
        project_assets = self.discover_assets()

        fts_search_index = None
        if self.enable_fts():
            fts_search_index = FTSSearch(self, wise_db.project_metadata_obj)

        for media_type in project_assets:
            if media_type not in {MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO}:
                # Added to ensure projects created with older versions
                # remain compatible (TODO: remove this in the future)
                logger.warning(
                    f"Media type {media_type} is not supported. "
                    "Please use IMAGE, VIDEO, or AUDIO media types."
                )
                continue
            for feature_extractor_id in project_assets[media_type]:
                if media_type not in search_indices:
                    search_indices[media_type] = {}

                search_indices[media_type][feature_extractor_id] = SearchIndexFactory(
                    media_type,
                    feature_extractor_id,
                    project_assets[media_type][feature_extractor_id],
                )
                asset = project_assets[media_type][feature_extractor_id]
                index_type_to_load = preferred_index_type

                if index_type_to_load:
                    # check if the preferred index type is available
                    preferred_index_filename = search_indices[media_type][
                        feature_extractor_id
                    ].get_index_filename(index_type_to_load)
                    if not os.path.exists(preferred_index_filename):
                        logger.warning(
                            f"Index file not found for preferred index type {index_type_to_load}. Will try to load any other available index."
                        )
                        index_type_to_load = None

                if not index_type_to_load:
                    # load any available index
                    available_indices = [
                        f for f in asset["index_files"] if f.endswith(".faiss")
                    ]
                    if available_indices:
                        # extract index type from filename, e.g. "video-IndexFlatIP.faiss" -> "IndexFlatIP"
                        index_type_to_load = Path(available_indices[0]).stem.split("-")[
                            1
                        ]
                        logger.info(
                            f"Loading available index of type {index_type_to_load}"
                        )
                    else:
                        logger.error(
                            f"No index files found for {media_type} and {feature_extractor_id}"
                        )
                        del search_indices[media_type][feature_extractor_id]
                        continue

                logger.info(
                    f"Loading faiss index from {search_indices[media_type][feature_extractor_id].get_index_filename(index_type_to_load)}"
                )
                if not search_indices[media_type][feature_extractor_id].load_index(
                    index_type_to_load
                ):
                    logger.error(
                        "Failed to load %s index: %s",
                        media_type,
                        feature_extractor_id,
                    )
                    del search_indices[media_type][feature_extractor_id]
                    continue
                if hasattr(
                    search_indices[media_type][feature_extractor_id].index, "nprobe"
                ):
                    # See https://github.com/facebookresearch/faiss/blob/43d86e30736ede853c384b24667fc3ab897d6ba9/faiss/IndexIVF.h#L184C8-L184C42
                    search_indices[media_type][
                        feature_extractor_id
                    ].index.parallel_mode = 1
                    search_indices[media_type][
                        feature_extractor_id
                    ].index.nprobe = default_nprobe

                    if not search_indices[media_type][
                        feature_extractor_id
                    ].is_internal_search_supported:
                        logger.info(
                            "This faiss index does not support internal search. To enable "
                            "internal search, please re-create the index by running "
                            f'`python -m wise create-index --project-dir "{self.project_dir}" --media-type {media_type} --index-type {search_indices[media_type][feature_extractor_id].index_type} --overwrite`',
                        )
            # TODO: Fix this to handle audio when support gets added
            if fts_search_index is not None and media_type in {
                MediaType.IMAGE,
                MediaType.VIDEO,
            }:
                search_indices[media_type]["wise/metadata"] = fts_search_index

        is_audio_only_project = (
            MediaType.VIDEO not in search_indices
            and MediaType.AUDIO in search_indices
            and len(search_indices[MediaType.AUDIO]) > 0
        )
        if fts_search_index is not None and is_audio_only_project:
            search_indices[MediaType.AUDIO]["wise/metadata"] = fts_search_index

        self._search_indices = search_indices
        return search_indices

    def _merge(self, other: "WiseProject", dry_run: bool = True):
        """
        WARNING: EXPERIMENTAL

        Merges another wise project with this one.

        Doesnt support metadata tables - use sqlite3 dump and import for now

        Expects the two projects to be of the same schema (migrations are applied before merging)

        """
        # sanity checks - TODO
        NO_ID = {'id': None}
        supported_assets = other.supported_modality_types_and_features

        # Create the required tables for feature extractor metadata
        for modality_type in supported_assets:
            for feature_extractor_id in supported_assets[modality_type]:
                feature_extractor_cls = get_feature_extractor_class(
                    feature_extractor_id
                )
                feature_extractor_cls.create_vector_metadata_table(self.db_engine)

        # Merge source collections by location and type
        def handle_source_collection(
            conn: sa.Connection, other_source_collection: SourceCollection
        ):
            # Find the source collection by location and type in the destination project
            existing_source_collection_id = conn.execute(
                sa.select(wise_db.source_collections_table.c.id).where(
                    wise_db.source_collections_table.c.location
                    == other_source_collection.location,
                    wise_db.source_collections_table.c.type
                    == other_source_collection.type,
                )
            ).scalar_one_or_none()

            # if not found, create new. Else return existing id
            if existing_source_collection_id is None:
                logger.debug(
                    f"could not find source collection {other_source_collection} - copying over"
                )
                # none match, create new
                existing_source_collection = SourceCollectionRepo.create(
                    conn, data=other_source_collection.model_copy(update=NO_ID)
                )
                existing_source_collection_id = existing_source_collection.id

            else:
                logger.debug(
                    f"found existing source collection at id - {existing_source_collection_id}"
                )

            return other_source_collection.id, existing_source_collection_id

        # Update source collection
        # if an equivalent source collection exists, use that id. Else create a new one and use that id
        source_collection_id_map = {}
        with (
            self.db_engine.connect() as conn,
            other.db_engine.connect() as other_conn,
            tqdm(
                total=SourceCollectionRepo.get_count(other_conn),
                desc="Source collections",
            ) as pbar,
        ):
            _handle_source_collection = partial(handle_source_collection, conn)
            for batch_sc in batched(SourceCollectionRepo.list(other_conn), 16):
                source_collection_id_map |= dict(
                    map(_handle_source_collection, batch_sc)
                )
                if not dry_run:
                    conn.commit()
                pbar.update(len(batch_sc))

        logger.info("Updated source collection map")
        logger.debug(f"{source_collection_id_map}")

        # find matching media by checksum, path, size_in_bytes and the new source_collection_id
        # if found, map the media_id to the existing media_id. Else return id to be copied over
        def get_match_media_fn():
            # columns to filter and match against
            media_columns = [
                "id",
                "source_collection_id",
                "path",
                "checksum",
                "size_in_bytes",
            ]
            media_row_query = sa.select(
                *[wise_db.media_table.c[x] for x in media_columns]
            )

            # left join to find matching media
            def get_matching_media_stmt(cte: sa.CTE):
                return sa.select(cte.c.id, wise_db.media_table.c.id).select_from(
                    cte.join(
                        wise_db.media_table,
                        onclause=sa.and_(
                            wise_db.media_table.c.checksum == cte.c.checksum,
                            wise_db.media_table.c.path == cte.c.path,
                            wise_db.media_table.c.source_collection_id
                            == cte.c.source_collection_id,
                            wise_db.media_table.c.size_in_bytes == cte.c.size_in_bytes,
                        ),
                        isouter=True,
                    )
                )

            # run is a function that
            # iterates over an incoming batch of media from the "other" project and
            # returns a mapping of other media_id to existing media_id in the current project
            # using the prepared statement above. if no match is found, the existing media_id will be None
            # and the row will be copied over to the current project in the next step
            run = wise_db.prepare_filter_stmt(
                wise_db.media_table,
                media_columns,
                get_matching_media_stmt,
            )

            def find_matching_media(conn: sa.Connection, batch_media: list):
                media_id_map = {}
                _needs_copy = []
                mapped_batch_media = [
                    (x[0], source_collection_id_map[x[1]]) + tuple(x[2:])
                    for x in batch_media
                ]
                _id_map = run(conn, mapped_batch_media).all()
                assert len(_id_map) == len(batch_media)
                for other_media_id, existing_media_id in _id_map:
                    if existing_media_id is not None:
                        media_id_map[other_media_id] = existing_media_id
                    else:
                        _needs_copy.append(other_media_id)

                return media_id_map, _needs_copy

            return media_row_query, find_matching_media

        def get_handle_new_media_fn():
            def get_media_by_ids_stmt(cte: sa.CTE):
                return (
                    sa.select(wise_db.media_table)
                    .select_from(
                        cte.join(
                            wise_db.media_table,
                            onclause=wise_db.media_table.c.id == cte.c.id,
                        )
                    )
                    .order_by(cte.c.rank)
                )

            # the prepared statement takes the ids as input and binds it to the cte
            # used in the get_media_by_ids_stmt
            run = wise_db.prepare_filter_stmt(
                wise_db.media_table,
                ["id"],
                get_media_by_ids_stmt,
                include_ordering=True,
            )

            def get_shots_by_media_ids_stmt(cte: sa.CTE):
                return (
                    sa.select(wise_db.shots_table)
                    .select_from(
                        cte.join(
                            wise_db.shots_table,
                            onclause=wise_db.shots_table.c.media_id == cte.c.media_id,
                        )
                    )
                    .order_by(cte.c.rank)
                )

            run_shots = wise_db.prepare_filter_stmt(
                wise_db.shots_table,
                ["media_id"],
                get_shots_by_media_ids_stmt,
                include_ordering=True,
            )

            def handle_new_media(
                conn: sa.Connection,
                other_conn: sa.Connection,
                other_media_ids: list[int],
            ):
                other_media = (
                    run(other_conn, [(x,) for x in other_media_ids]).mappings().all()
                )
                other_media = [
                    dict(x)
                    | NO_ID
                    | {
                        "source_collection_id": source_collection_id_map[
                            x["source_collection_id"]
                        ]
                    }
                    for x in other_media
                ]

                # new media
                new_media_ids = batch_insert(conn, wise_db.media_table, other_media)

                # query all shots and map to new media ids
                media_id_map: dict[int, int] = dict(zip(other_media_ids, new_media_ids))

                shots = (
                    run_shots(other_conn, [(x,) for x in other_media_ids])
                    .mappings()
                    .all()
                )

                if shots:
                    shots = [
                        x | {"media_id": media_id_map[x["media_id"]]} for x in shots
                    ]
                    batch_insert(conn, wise_db.shots_table, shots, return_ids=False)

                return media_id_map

            return handle_new_media

        media_id_map = {}
        media_count = other.num_media
        batch_size = 2048

        with (
            self.db_engine.connect() as conn,
            other.db_engine.connect() as other_conn,
            tqdm(
                total=media_count,
                desc="Processing Media",
            ) as pbar,
        ):
            _needs_copy = []
            media_row_query, find_matching_media = get_match_media_fn()
            for batch_media in batch_select(
                other_conn, media_row_query, batch_size=batch_size
            ):
                # map the source collection ids to the new project
                media_id_map_batch, needs_copy_batch = find_matching_media(
                    conn, batch_media
                )
                media_id_map |= media_id_map_batch
                _needs_copy.extend(needs_copy_batch)
                pbar.update(len(batch_media))

        with (
            self.db_engine.connect() as conn,
            other.db_engine.connect() as other_conn,
            tqdm(
                total=len(_needs_copy),
                desc="Copying Media",
            ) as pbar,
        ):
            handle_new_media = get_handle_new_media_fn()
            for batch_media in batched(_needs_copy, batch_size):
                media_id_map |= handle_new_media(conn, other_conn, batch_media)
                if not dry_run:
                    conn.commit()
                pbar.update(len(batch_media))

        logger.info("Updated media map")
        logger.debug(f"{media_id_map}")

        def get_matching_thumbnails_fn():
            thumbnail_columns = [
                "id",
                "media_id",
                "timestamp",
            ]
            thumbs_row_query = sa.select(
                *[wise_db.thumbnails_table.c[x] for x in thumbnail_columns]
            )
            def get_thumbnail_ids_to_copy_stmt(cte: sa.CTE):
                return (
                    sa.select(cte.c.id)
                    .select_from(
                        cte.join(
                            wise_db.thumbnails_table,
                            onclause=sa.and_(
                                wise_db.thumbnails_table.c.media_id == cte.c.media_id,
                                wise_db.thumbnails_table.c.timestamp == cte.c.timestamp,
                            ),
                            isouter=True,
                        )
                    )
                    .where(wise_db.thumbnails_table.c.id == None)
                )

            run = wise_db.prepare_filter_stmt(
                wise_db.thumbnails_table,
                thumbnail_columns,
                get_thumbnail_ids_to_copy_stmt,
            )

            def get_thumbnail_ids_to_copy(conn: sa.Connection, batch_thumbnails: list):
                mapped_batch_thumbnails = [
                    (x[0], media_id_map[x[1]], x[2]) for x in batch_thumbnails
                ]
                _needs_copy = run(conn, mapped_batch_thumbnails).scalars().all()
                return _needs_copy

            return thumbs_row_query, get_thumbnail_ids_to_copy

        def get_handle_new_thumbnail_fn():

            def get_thumbnails_by_ids_stmt(cte: sa.CTE):
                return (
                    sa.select(wise_db.thumbnails_table)
                    .select_from(
                        cte.join(
                            wise_db.thumbnails_table,
                            onclause=wise_db.thumbnails_table.c.id == cte.c.id,
                        )
                    )
                    .order_by(cte.c.rank)
                )

            run = wise_db.prepare_filter_stmt(
                wise_db.thumbnails_table,
                ["id"],
                get_thumbnails_by_ids_stmt,
                include_ordering=True,
            )

            def handle_new_thumbnail(
                thumbs_conn: sa.Connection,
                other_thumbs_conn: sa.Connection,
                other_thumbnail_ids: list,
            ):

                other_thumbnails = (
                    run(other_thumbs_conn, [(x,) for x in other_thumbnail_ids])
                    .mappings()
                    .all()
                )

                other_thumbnails = [
                    dict(x) | NO_ID | {"media_id": media_id_map[x["media_id"]]}
                    for x in other_thumbnails
                ]
                batch_insert(
                    thumbs_conn,
                    wise_db.thumbnails_table,
                    other_thumbnails,
                    return_ids=False,
                )

            return handle_new_thumbnail

        batch_size = 2048
        thumbnail_count = other.num_thumbnails
        with (
            self.thumbsdb_engine.connect() as thumbs_conn,
            other.thumbsdb_engine.connect() as other_thumbs_conn,
            tqdm(
                total=thumbnail_count,
                desc="Processing Thumbnails",
            ) as pbar,
        ):
            _needs_copy = []
            thumbnail_row_query, get_thumbnail_ids_to_copy = (
                get_matching_thumbnails_fn()
            )
            for batch_thumbnail in batch_select(
                other_thumbs_conn, thumbnail_row_query, batch_size=batch_size
            ):
                _needs_copy_batch = get_thumbnail_ids_to_copy(
                    thumbs_conn, batch_thumbnail
                )
                _needs_copy.extend(_needs_copy_batch)
                pbar.update(len(batch_thumbnail))

        with (
            self.thumbsdb_engine.connect() as thumbs_conn,
            other.thumbsdb_engine.connect() as other_thumbs_conn,
            tqdm(
                total=len(_needs_copy),
                desc="Copying Thumbnails",
            ) as pbar,
        ):
            handle_new_thumbnail = get_handle_new_thumbnail_fn()
            for batch_thumbs in batched(_needs_copy, batch_size):
                handle_new_thumbnail(thumbs_conn, other_thumbs_conn, batch_thumbs)
                if not dry_run:
                    thumbs_conn.commit()
                pbar.update(len(batch_thumbs))

        logger.info("thumbnails copied over")

        def get_last_vector_timestamps(conn: sa.Connection):
            vector_timestamp_limits = {}

            result = conn.execute(
                sa.select(
                    wise_db.vectors_table.c.media_id,
                    wise_db.vectors_table.c.modality,
                    wise_db.vectors_table.c.feature_extractor_id,
                    sa.func.max(wise_db.vectors_table.c.timestamp),
                )
                .group_by(
                    wise_db.vectors_table.c.media_id,
                    wise_db.vectors_table.c.modality,
                    wise_db.vectors_table.c.feature_extractor_id,
                )
                .order_by(wise_db.vectors_table.c.media_id.asc())
            )
            for media_id, modality, feature_extractor_id, timestamp in result.all():
                vector_timestamp_limits[(media_id, modality, feature_extractor_id)] = (
                    timestamp
                )

            return vector_timestamp_limits

        with self.db_engine.connect() as conn:
            min_time_stamp_per_media_id = get_last_vector_timestamps(conn)
        logger.debug(f"vector timestamp - {min_time_stamp_per_media_id}")

        def copy_vectors(
            conn: sa.Connection,
            other_conn: sa.Connection,
            media_type: ModalityType,
            feature_extractor_id: str,
            other_store: FeatureStore,
        ):
            logger.info(
                f"copying for feature_extractor - {feature_extractor_id} ({media_type})"
            )
            feature_count = other_store.feature_count
            self.create_features_dir(feature_extractor_id)
            try:
                store = FeatureStoreFactory.load_store(
                    media_type, self.features_dir(feature_extractor_id)
                )
            except ValueError:
                store = FeatureStoreFactory.create_store(
                    FeatureStoreType.FAISS,
                    media_type,
                    self.features_dir(feature_extractor_id),
                )
            store.enable_write()

            feature_extractor_cls = get_feature_extractor_class(feature_extractor_id)

            if store.__class__.__name__ == "FaissFeatureStore":

                def add_to_store(feature_ids, features):
                    store.add(feature_ids, features)

            else:

                def add_to_store(feature_ids, features):
                    for fid, feat in zip(feature_ids, features):
                        store.add(fid, np.expand_dims(feat, axis=0))

            total_copied = 0
            batch_size = 4096

            def get_handle_new_vectors_fn():
                def get_vectors_by_ids_stmt(cte: sa.CTE):
                    return (
                        sa.select(wise_db.vectors_table)
                        .select_from(
                            cte.join(
                                wise_db.vectors_table,
                                onclause=wise_db.vectors_table.c.id == cte.c.id,
                            )
                        )
                        .order_by(cte.c.rank)
                    )

                run = wise_db.prepare_filter_stmt(
                    wise_db.vectors_table,
                    ["id"],
                    get_vectors_by_ids_stmt,
                    include_ordering=True,
                )

                def handle_new_vectors(
                    conn: sa.Connection,
                    other_conn: sa.Connection,
                    other_vector_ids: np.ndarray,
                    other_features: np.ndarray,
                ):
                    feature_id_list = other_vector_ids.tolist()
                    other_vectors = (
                        run(other_conn, [(x,) for x in feature_id_list])
                        .mappings()
                        .all()
                    )
                    other_vectors = {r["id"]: dict(r) for r in other_vectors}

                    other_vectors = {
                        k: v
                        | {
                            "media_id": media_id_map[v["media_id"]],
                            "feature_extractor_id": feature_extractor_id,
                        }
                        for k, v in other_vectors.items()
                    }

                    _needs_copy = []
                    _copy_idx = []
                    for idx, feature_id in enumerate(feature_id_list):
                        if feature_id not in other_vectors:
                            logger.debug("skipping missing vector")
                            continue

                        vector = other_vectors[feature_id]
                        new_media_id = vector["media_id"]
                        min_ts = min_time_stamp_per_media_id.get(
                            (new_media_id, media_type, feature_extractor_id),
                            min_time_stamp_per_media_id.get(
                                (new_media_id, media_type, ""), -1
                            ),
                        )
                        if vector["timestamp"] <= min_ts:
                            continue

                        _needs_copy.append(feature_id)
                        _copy_idx.append(idx)

                    if _needs_copy:
                        # bulk insert and get new ids
                        vectors_to_insert = [
                            other_vectors[oid] | NO_ID for oid in _needs_copy
                        ]
                        new_vector_ids = batch_insert(
                            conn, wise_db.vectors_table, vectors_to_insert
                        )
                        # copy to store
                        add_to_store(new_vector_ids, other_features[_copy_idx])

                        ext_vector_metadata = feature_extractor_cls.get_vector_metadata(
                            other_conn, _needs_copy
                        )
                        feature_extractor_cls.add_to_vector_metadata_table(
                            conn, new_vector_ids, ext_vector_metadata
                        )
                    return len(_needs_copy)

                return handle_new_vectors

            handle_new_vectors = get_handle_new_vectors_fn()
            try:
                with (
                    tqdm(total=feature_count, desc="Processing vectors") as pbar,
                ):
                    for feature_ids, features in other_store.iter_batch(batch_size):

                        total_copied += handle_new_vectors(
                            conn, other_conn, feature_ids, features
                        )
                        if not dry_run:
                            conn.commit()

                        pbar.update(len(feature_ids))
            finally:
                store.close()
            logger.info(
                f"copied {total_copied} vectors over for feature extractor - {feature_extractor_id} ({media_type})"
            )

        with (
            self.db_engine.connect() as conn,
            other.db_engine.connect() as other_conn,
        ):
            for media_type in supported_assets:
                for feature_extractor_id in supported_assets[media_type]:
                    other_store = FeatureStoreFactory.load_store(
                        media_type, other.features_dir(feature_extractor_id)
                    )
                    copy_vectors(
                        conn, other_conn, media_type, feature_extractor_id, other_store
                    )
                    if not dry_run:
                        conn.commit()

        # merge tables in database
    def merge(self, other: "WiseProject", dry_run: bool = True):

        # Source collection
        with self.db_engine.connect() as conn, self.thumbsdb_engine.connect() as thumbs_conn:
            last_collection_id = conn.execute(sa.select(sa.func.max(wise_db.source_collections_table.c.id))).scalar_one_or_none()
            last_media_id = conn.execute(sa.select(sa.func.max(wise_db.media_table.c.id))).scalar_one_or_none()
            last_vector_id = conn.execute(sa.select(sa.func.max(wise_db.vectors_table.c.id))).scalar_one_or_none()
            last_thumbnail_id = thumbs_conn.execute(sa.select(sa.func.max(wise_db.thumbnails_table.c.id))).scalar_one_or_none()

        supported_assets = self.discover_assets()
        supported_assets.pop('metadata', None)

        def cleanup_features():
            other_assets = other.discover_assets()
            for media_type in other_assets:
                for feature_extractor_id in other_assets[media_type]:
                    if media_type not in supported_assets:
                        for p in self.features_dir(feature_extractor_id).rglob(f'{media_type}-*'):
                            logger.info(f'deleting - {p}')
                            p.unlink(missing_ok=True)

                    elif feature_extractor_id not in supported_assets[media_type]:
                        logger.info(
                            f"deleting directory of {feature_extractor_id} from project"
                        )
                        shutil.rmtree(self.features_root(feature_extractor_id))

                    else:
                        for p in self.features_dir(feature_extractor_id).rglob(f'{media_type}-*'):
                            if p.name not in supported_assets[media_type][feature_extractor_id]['features_files']:
                                logger.info(f'deleting - {p}')
                                p.unlink(missing_ok=True)

        def cleanup_tables():

            def delete_rows_from_id(table: sa.Table, _id):
                if _id is None:
                    return

                logger.info(f"deleting rows from {table.name} with id > {_id}")
                with self.db_engine.connect() as conn:
                    res = conn.execute(
                        sa.delete(
                            table
                        ).where(
                            table.c.id > _id
                        )
                    )
                    if dry_run and res.rowcount > 0:
                        extra_rows = []
                        for r in conn.execute(sa.select(table.c.id).where(table.c.id > _id)).scalar_one():
                            extra_rows.append(r)
                        logger.error(f'ERROR - dry run inserted rows in {table} - {extra_rows}')
                    else:
                        conn.commit()

            delete_rows_from_id(wise_db.source_collections_table, last_collection_id)
            delete_rows_from_id(wise_db.media_table, last_media_id)
            delete_rows_from_id(wise_db.vectors_table, last_vector_id)
            delete_rows_from_id(wise_db.thumbnails_table, last_thumbnail_id)

        try:
            self._merge(other, dry_run)
            if dry_run:
                cleanup_features()
        except (Exception, KeyboardInterrupt):
            logger.exception('Error in merge, rolling back - dont interrupt')
            cleanup_features()
            cleanup_tables()

        # sanity checks - TODO


if __name__ == "__main__":
    # temp cli for merging projects
    import typer

    app = typer.Typer()
    app_state = {"verbose": True}

    @app.callback()
    def base(verbose: bool = False):
        """
        APP CLI
        Add description about the CLI
        """
        app_state["verbose"] = verbose
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
        )
        global logger
        logger = logging.getLogger()

    @app.command()
    def merge(
        other_projects: list[Path] = typer.Argument(
            ..., help="the source project directory to merge from"
        ),
        into: Path = typer.Option(..., help="the target project directory"),
        dry_run: bool = typer.Option(False, help="if set, will not make any changes"),
    ):
        """
        Merges another wise project into this one

        WARNING: EXPERIMENTAL - use at your own risk
        """
        verbose = app_state["verbose"]
        project = WiseProject(
            into,
            create_project=True,
            db_kwargs={"echo": verbose},
            thumbsdb_kwargs={"echo": verbose},
        )
        for _other_project_dir in other_projects:
            other_project = WiseProject(
                _other_project_dir,
                db_kwargs={"echo": verbose},
                thumbsdb_kwargs={"echo": verbose},
            )

            project.merge(other_project, dry_run=dry_run)
            if dry_run:
                logger.info("Dry run - no changes made")
            else:
                logger.info("Merge complete")

    if __name__ == "__main__":
        app()
