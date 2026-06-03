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

from sqlite3 import Connection as SQLite3Connection

from sqlalchemy import Engine, MetaData, create_engine, event

from wise.db.base import project_metadata_obj, thumbs_metadata_obj
from wise.db.tables import (
    imported_metadata_table,
    media_table,
    shots_table,
    source_collections_table,
    thumbnails_table,
    vectors_table,
)


_WISE_FTS_TABLE = 'metadata_fts'
_WISE_ASR_TABLE = 'metadata-asr'
__wise_tables = [_WISE_FTS_TABLE, _WISE_ASR_TABLE]


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _init(dburi: str, metadata_obj: MetaData, **kwargs) -> Engine:
    engine = create_engine(dburi, **kwargs)
    if "mode=ro" not in dburi:
        metadata_obj.create_all(engine)
    return engine


def init_project(dburi: str, **kwargs) -> Engine:
    return _init(dburi, project_metadata_obj, **kwargs)


def init_thumbs(dburi: str, **kwargs) -> Engine:
    return _init(dburi, thumbs_metadata_obj, **kwargs)

def reflect_external_metadata(db_engine):
    project_metadata_obj.reflect(bind=db_engine, only=lambda x, _: x != _WISE_FTS_TABLE)

    return [ v for k, v in project_metadata_obj.tables.items() if k.startswith('metadata-') and k not in __wise_tables]
