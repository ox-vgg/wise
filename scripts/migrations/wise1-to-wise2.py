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

"""Migrate WISE 1 projects to WISE 2.

This script migrates WISE projects (data/features, index, thumbnails,
and internal database) from version 1 to version 2.  There are many
different versions of WISE 1 and 2 and the database does not specify
schema version so beware.

After migration, it will probably be required to modify the WISE 2
database with the specifics of each project.  For example, WISE 2 has
no support for webdataset.  Alternatively, this script can be
modified.

This script requires Python 3.12 or later.

"""

import argparse
import datetime
import logging
import sqlite3
import sys
from itertools import batched  # requires python 3.12+
from pathlib import Path

import faiss
import h5py
import tqdm

from wise.data_models import (
    MediaMetadata,
    MediaType,
    ModalityType,
    SourceCollection,
    SourceCollectionType,
    VectorMetadata,
)
from wise.feature.store import (
    FeatureStoreFactory,
    FeatureStoreType,
)
from wise.repository import MediaRepo, SourceCollectionRepo, VectorRepo
from wise.wise_project import WiseProject

logger = logging.getLogger(__name__)


class Wise1Project:
    def __init__(self, project_dir: Path):
        self._project_dir = project_dir
        self._project_name = self._project_dir.name
        self._project_db = self._project_dir / (self._project_name + ".db")
        self._project_store = self._project_dir / (self._project_name + ".h5")
        self._index_dir = self._project_dir / "index"
        if not self._project_dir.exists():
            raise ValueError(f"'{self._project_dir}' does not exist")
        if not self._project_dir.is_dir():
            raise ValueError(f"'{self._project_dir}' is not a directory")
        if not self._index_dir.exists():
            raise ValueError(f"'{self._index_dir}' does not exist")
        if not self._index_dir.is_dir():
            raise ValueError(f"'{self._index_dir}' is not a directory")
        if not self._project_db.exists():
            raise ValueError(f"'{self._project_db}' does not exist")
        if not self._project_store.exists():
            raise ValueError(f"'{self._project_store}' does not exist")
        with h5py.File(self._project_store, "r") as h5fh:
            assert set(h5fh.keys()) == {"features", "ids", "thumbnails"}
            assert set(h5fh["/features"].keys()) == {"image"}
            assert len(h5fh["/ids"].shape) == 1
            assert len(h5fh["/features/image"].shape) == 2
            assert h5fh["/ids"].shape[0] == h5fh["/features/image"].shape[0]
            n_features = h5fh["/ids"].shape[0]
            logger.debug("Project has %d features/vectors", n_features)
            model = h5fh["/features/image"].attrs["model"]
            logger.debug("Project model is '%s'", model)
            self._feature_extractor_id = (
                "mlfoundations/openclip/" + model.replace(":", "/")
            )

        with sqlite3.connect(self._project_db) as conn:
            n_rows = next(conn.execute("SELECT Count(*) FROM metadata"))[0]
        if n_rows != n_features:
            raise ValueError(
                f"Number of features ({n_features} does not match"
                f" database ({n_rows})"
            )
        for index_file in self._index_dir.glob("*.faiss"):
            index = faiss.read_index(
                str(index_file), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )
            if index.ntotal != n_features:
                raise ValueError(
                    f"Number of features ({n_features} does not match"
                    f" index '{index_file}' ({index.ntotal})"
                )

    def migrate_features_to_wise2(
        self, wise2_project: WiseProject, shard_maxcount: int
    ):
        wise2_project.create_features_dir(self._feature_extractor_id)
        wise2_store = FeatureStoreFactory.create_store(
            FeatureStoreType.FAISS,
            ModalityType.IMAGE,
            wise2_project.features_dir(self._feature_extractor_id),
        )
        wise2_store.enable_write(shard_maxcount=shard_maxcount)

        with h5py.File(self._project_store, "r") as h5fh:
            for i in tqdm.trange(
                0,
                len(h5fh["/ids"]),
                shard_maxcount,
                desc="Migrating feature store",
                unit_scale=shard_maxcount,
            ):
                ids = [int(x) for x in h5fh["/ids"][i : i + shard_maxcount]]
                features = h5fh["/features/image"][i : i + shard_maxcount]
                wise2_store.add(ids, features)

    def migrate_index_to_wise2(self, wise2_project: WiseProject):
        wise2_assets = wise2_project.discover_assets()
        wise2_index_dir = Path(
            wise2_assets[MediaType.IMAGE][self._feature_extractor_id][
                "index_dir"
            ]
        )
        wise2_index_dir.mkdir(parents=True, exist_ok=False)
        for wise1_index_file in self._index_dir.glob("*.faiss"):
            wise2_index_file = wise2_index_dir / wise1_index_file.name
            if wise2_index_file.exists():
                raise RuntimeError(
                    f"index file '{wise2_index_file}' already exists"
                )
            logger.info(
                "Copying '%s' index file to '%s'",
                wise1_index_file,
                wise2_index_file,
            )
            wise2_index_file.write_bytes(wise1_index_file.read_bytes())

    def _migrate_datasets_table(self, wise2_project: WiseProject):
        wise1_conn = sqlite3.connect(self._project_db)
        stmt = """
            SELECT
                id,
                location,
                type
            FROM
                datasets
        """
        source_collections = []
        for dataset in wise1_conn.execute(stmt):
            assert dataset[2] == "WEBDATASET"
            source_collections.append(
                SourceCollection(
                    id=dataset[0],
                    location=dataset[1],
                    type=SourceCollectionType.WEBDATASET,
                )
            )
        with wise2_project.db_engine.connect() as wise2_conn:
            SourceCollectionRepo.create_many(
                wise2_conn, data=source_collections
            )
            wise2_conn.commit()

    def _migrate_metadata_table(
        self, wise2_project: WiseProject, max_insert: int
    ):
        epoch_time = datetime.datetime.fromtimestamp(0)
        with (
            sqlite3.connect(self._project_db) as wise1_conn,
            wise2_project.db_engine.connect() as wise2_conn,
        ):
            wise1_select = """
                SELECT
                    id,
                    dataset_id,
                    path,
                    size_in_bytes,
                    format,
                    width,
                    height
                FROM
                    metadata
            """
            n_rows = next(wise1_conn.execute("SELECT Count(*) FROM metadata"))[
                0
            ]
            for rows_batch in batched(
                tqdm.tqdm(
                    wise1_conn.execute(wise1_select),
                    desc="Migrating metadata table",
                    total=n_rows,
                ),
                n=max_insert,
            ):
                media_data = []
                vector_data = []
                for row in rows_batch:
                    media_data.append(
                        MediaMetadata(
                            id=row[0],
                            source_collection_id=row[1],
                            path=row[2],
                            checksum=b"",  # there was no checksums in wise 1
                            size_in_bytes=row[3],
                            date_modified=epoch_time,  # there was no date in Wise 1
                            media_type=MediaType.IMAGE,  # wise 1 only supported images
                            format=row[4].lower(),
                            width=row[5],
                            height=row[6],
                            num_frames=1,  # wise 1 only supported images
                            duration=0.0,  # wise 1 only supported images
                        )
                    )
                    ## Wise 1 had no separate vector and media tables
                    ## so we use the same id on both tables.
                    vector_data.append(
                        VectorMetadata(
                            id=row[0],
                            modality=ModalityType.IMAGE,  # wise 1 only supported images
                            feature_extractor_id=self._feature_extractor_id,
                            media_id=row[0],
                            timestamp=0.0,  # wise 1 only supported images
                        )
                    )
                MediaRepo.create_many(wise2_conn, data=media_data)
                VectorRepo.create_many(wise2_conn, data=vector_data)
                wise2_conn.commit()

    def migrate_database_to_wise2(
        self, wise2_project: WiseProject, max_insert: int
    ):
        self._migrate_datasets_table(wise2_project)
        self._migrate_metadata_table(wise2_project, max_insert)

    def migrate_thumbnails_to_wise2(self, wise2_project: WiseProject):
        raise NotImplementedError("migration of thumbnails is not implemented")


def main(argv: list[str]) -> int:
    logging.basicConfig()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--logging-level",
        action="store",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set logging level",
    )
    parser.add_argument(
        "--shard-maxcount",
        type=int,
        default=2048,
        help="Max number of entries in each feature store shard",
    )
    parser.add_argument(
        "--db-maxinsert",
        type=int,
        default=2048,
        help="Max number of rows inserted in database per transaction",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip migration of feature store",
    )
    parser.add_argument(
        "--skip-index", action="store_true", help="Skip the migration of index"
    )
    parser.add_argument(
        "--skip-internal-db",
        action="store_true",
        help="Skip migration of the internal metadata database",
    )
    parser.add_argument(
        "--skip-thumbnails",
        action="store_true",
        help="Skip migration thumbnails database",
    )
    parser.add_argument("wise1_project_dir", type=Path)
    parser.add_argument("wise2_project_dir", type=Path)
    args = parser.parse_args(argv[1:])

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.logging_level.upper()))

    if not args.skip_thumbnails:  # error early if this is required
        raise Exception("migration of thumbnails is not yet implemented")

    wise1 = Wise1Project(args.wise1_project_dir)
    wise2 = WiseProject(args.wise2_project_dir, create_project=True)

    if args.skip_features:
        logger.info("--skip-features: not migrating data / features store")
    else:
        wise1.migrate_features_to_wise2(wise2, args.shard_maxcount)

    if args.skip_index:
        logger.info("--skip-index: not migrating index")
    else:
        wise1.migrate_index_to_wise2(wise2)

    if args.skip_internal_db:
        logger.info("--skip-internal-db: not migrating internal database")
    else:
        wise1.migrate_database_to_wise2(wise2, args.db_maxinsert)

    if args.skip_thumbnails:
        logger.info("--skip-thumbnails: not migrating thumbnails database")
    else:
        wise1.migrate_thumbnails_to_wise2(wise2)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
