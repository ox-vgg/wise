from sqlalchemy import create_engine, Engine, MetaData
from .base import thumbs_metadata_obj, project_metadata_obj
from .tables import (
    source_collections_table,
    media_table,
    vectors_table,
    imported_metadata_table,
    thumbnails_table,
)

_WISE_FTS_TABLE = 'metadata_fts'
_WISE_ASR_TABLE = 'metdata-asr'
__wise_tables = [_WISE_FTS_TABLE, _WISE_ASR_TABLE]

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