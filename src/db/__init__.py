from sqlite3 import Connection as SQLite3Connection
from sqlalchemy import create_engine, Engine, MetaData, event
from .base import thumbs_metadata_obj, project_metadata_obj
from .tables import (
    source_collections_table,
    media_table,
    vectors_table,
    imported_metadata_table,
    thumbnails_table,
    shots_table
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
    metadata_obj.create_all(engine)
    return engine


def init_project(dburi: str, **kwargs) -> Engine:
    return _init(dburi, project_metadata_obj, **kwargs)


def init_thumbs(dburi: str, **kwargs) -> Engine:
    return _init(dburi, thumbs_metadata_obj, **kwargs)

def reflect_external_metadata(db_engine):
    project_metadata_obj.reflect(bind=db_engine, only=lambda x, _: x != _WISE_FTS_TABLE)

    return [ v for k, v in project_metadata_obj.tables.items() if k.startswith('metadata-') and k not in __wise_tables]
