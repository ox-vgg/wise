from sqlalchemy import create_engine, Engine, MetaData
from .base import wise_metadata_obj, project_metadata_obj
from .tables import dataset_table, metadata_table, project_table


def _init(dburi: str, metadata_obj: MetaData, **kwargs) -> Engine:
    engine = create_engine(dburi, **kwargs)
    metadata_obj.create_all(engine)
    return engine


def init(dburi: str, **kwargs) -> Engine:
    return _init(dburi, wise_metadata_obj, **kwargs)


def init_project(dburi: str, **kwargs) -> Engine:
    return _init(dburi, project_metadata_obj, **kwargs)
