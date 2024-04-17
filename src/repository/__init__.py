from .base import SQLAlchemyRepository
from .projects import WiseProjectsSQLAlchemyRepository

# from .extra_metadata import ExtraMediaMetadataSQLAlchemyRepository
from ..data_models import (
    Project,
    SourceCollection,
    MediaMetadata,
    VectorMetadata,
    ExtraMediaMetadata,
)
from .. import db


WiseProjectsRepo = WiseProjectsSQLAlchemyRepository(db.project_table, Project)
SourceCollectionRepo = SQLAlchemyRepository[
    SourceCollection, SourceCollection, SourceCollection
](db.source_collections_table, SourceCollection)
MediaRepo = SQLAlchemyRepository[MediaMetadata, MediaMetadata, MediaMetadata](
    db.media_table, MediaMetadata
)
VectorRepo = SQLAlchemyRepository[VectorMetadata, VectorMetadata, VectorMetadata](
    db.vectors_table, VectorMetadata
)
MediaMetadataRepo = SQLAlchemyRepository[
    ExtraMediaMetadata, ExtraMediaMetadata, ExtraMediaMetadata
](db.imported_metadata_table, ExtraMediaMetadata)
