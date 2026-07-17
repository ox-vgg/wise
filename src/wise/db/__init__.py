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

"""Internal Metadata Database

The internal metadata database is an SQLite database storing
information about the source collections (i.e. input folders or
webdatasets), media files (e.g. images, videos, or audio files),
vectors, and extra metadata.  The database file is located in
`metadata/internal.db` within the project folder.

WISE uses [SQLAlchemy](https://www.sqlalchemy.org/) (Core API) to
interact with this database, and
[Pydantic](https://docs.pydantic.dev/latest/) to parse / validate the
data going into the DB.

The diagram below shows the database schema.  The table structures are
defined in the `db.tables` module and the validation models are
defined in the `data_models` module.

[![Database diagram](../../docs/assets/WISE%202%20internal%20metadata.svg)](https://dbdiagram.io/d/WISE-2-internal-metadata-65f3512eb1f3d4062cf6be68)

In addition to the tables defined above, WISE stores the thumbnails as
well in a separate database (`thumbs.db`) within the project folder.
Thumbnails are explained in more detail in the [Thumbnails section of
the documentation](../../docs/Thumbnails.md).

WISE uses the [repository
pattern](https://www.cosmicpython.com/book/chapter_02_repository.html)
to abstract the DB access.  The CRUD methods are defined in the
`repository.base` module and the repository for each metadata table is
instantiated in the `repository` module.

"""

import logging
import time
from sqlite3 import Connection as SQLite3Connection

from sqlalchemy import Engine, MetaData, create_engine, event

from wise.db.base import project_metadata_obj, thumbs_metadata_obj
from wise.db.tables import (
    featured_table,
    imported_metadata_table,
    media_table,
    shots_table,
    source_collections_table,
    thumbnails_table,
    vectors_table,
)
from wise.db.utils import prepare_filter_stmt

logger = logging.getLogger(__name__)

_WISE_FTS_TABLE = "metadata_fts"
_WISE_ASR_TABLE = "metadata-asr"
__wise_tables = [_WISE_FTS_TABLE, _WISE_ASR_TABLE]

## We require sqlite version 3.35.0 (released on 2021-03-12) because
## that's when sqlite added support for the RETURNING clause on INSERT
## (used in SQLAlchemyRepository.create).
_SQLITE_VERSION_REQUIREMENT = (3, 35, 0)


def before_cursor_execute(
    conn, cursor, statement, parameters, context, executemany
):
    del cursor, parameters, context, executemany
    conn.info.setdefault("query_start_time", []).append(time.time())
    logger.info("Start Query: %s", statement)


def after_cursor_execute(
    conn, cursor, statement, parameters, context, executemany
):
    del cursor, statement, parameters, context, executemany
    total = time.time() - conn.info["query_start_time"].pop(-1)
    logger.info("Query Complete!")
    logger.info("Total Time: %f", total)


def check_sqlite_requirements(dbapi_connection, connection_record):
    ## This is equivalent to `engine.dialect.server_version_info` but
    ## we don't have the engine object here.
    cursor = dbapi_connection.cursor()
    res = cursor.execute("SELECT sqlite_version() AS version;")
    driver_version = res.fetchone()[0]
    logger.debug("Found sqlite driver version %s", driver_version)
    driver_version_parts = [int(x) for x in driver_version.split(".")]
    assert len(driver_version_parts) == 3
    for req, ver in zip(_SQLITE_VERSION_REQUIREMENT, driver_version_parts):
        if ver > req:
            break
        elif ver < req:
            msg = "WISE requires sqlite %d.%d.%d or later"
            logger.critical(msg, *_SQLITE_VERSION_REQUIREMENT)
            raise Exception(msg % _SQLITE_VERSION_REQUIREMENT)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    del connection_record
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")

        # TODO: Move these to config?
        # use WAL mode, allows N reader and 1 writer
        # caveat: sqlite doesn't play nicely on network disks
        # cursor.execute(
        #    "PRAGMA journal_mode=WAL"
        # )

        # 5 second timeout for trying to get a lock
        # cursor.execute(
        #    "PRAGMA busy_timeout = 5000"
        # )

        # force sqlite to store temporary tables and indexes in memory, not on disk
        # cursor.execute("PRAGMA temp_store = MEMORY")

        cursor.close()


def _init(dburi: str, metadata_obj: MetaData, **kwargs) -> Engine:
    profile = kwargs.pop("profile", False)

    engine = create_engine(dburi, **kwargs)
    event.listen(engine, "first_connect", check_sqlite_requirements)
    if profile:
        event.listen(engine, "before_cursor_execute", before_cursor_execute)
        event.listen(engine, "after_cursor_execute", after_cursor_execute)

    if "mode=ro" not in dburi:
        metadata_obj.create_all(engine)
    return engine


def init_project(dburi: str, **kwargs) -> Engine:
    return _init(dburi, project_metadata_obj, **kwargs)


def init_thumbs(dburi: str, **kwargs) -> Engine:
    return _init(dburi, thumbs_metadata_obj, **kwargs)


def reflect_external_metadata(db_engine):
    project_metadata_obj.reflect(
        bind=db_engine, only=lambda x, _: x != _WISE_FTS_TABLE
    )

    return [
        v
        for k, v in project_metadata_obj.tables.items()
        if k.startswith("metadata-") and k not in __wise_tables
    ]
