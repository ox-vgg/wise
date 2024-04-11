from .base import SQLAlchemyRepository
from .projects import WiseProjectsSQLAlchemyRepository
from ..data_models import Project, SourceCollection, BaseSourceCollection, MediaMetadata, VectorMetadata
from .. import db


WiseProjectsRepo = WiseProjectsSQLAlchemyRepository(db.project_table, Project)
SourceCollectionRepo = SQLAlchemyRepository[SourceCollection, BaseSourceCollection, SourceCollection](
    db.source_collections_table, SourceCollection
)
MediaRepo = SQLAlchemyRepository[MediaMetadata, MediaMetadata, MediaMetadata](
    db.media_table, MediaMetadata
)
VectorRepo = SQLAlchemyRepository[VectorMetadata, VectorMetadata, VectorMetadata](
  db.vectors_table, VectorMetadata
)
