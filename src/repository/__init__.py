from typing import List, Optional
import sqlalchemy as sa

from .base import SQLAlchemyRepository
from .projects import WiseProjectsSQLAlchemyRepository

# from .extra_metadata import ExtraMediaMetadataSQLAlchemyRepository
from ..data_models import (
    Project,
    SourceCollection,
    MediaMetadata,
    VectorAndMediaMetadata,
    VectorMetadata,
    ExtraMediaMetadata,
    ThumbnailMetadata,
)
from .. import db


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

ThumbnailRepo = SQLAlchemyRepository[
    ThumbnailMetadata, ThumbnailMetadata, ThumbnailMetadata
](db.thumbnails_table, ThumbnailMetadata)


_vtable = db.vectors_table
_mtable = db.media_table
_thumbs_table = db.thumbnails_table

def get_full_metadata_batch(conn: sa.Connection, ids: List[int]) -> List[VectorAndMediaMetadata]:
    # TODO add docstring
    # ids refers to vector ids
    ordering = sa.case(
        {id: index for index, id in enumerate(ids)},
        value=_vtable.c.id,
    )
    stmt = (
        sa.select(_vtable.c, _mtable.c)
        .select_from(_vtable.join(_mtable))
        .where(_vtable.c.id.in_(ids))
        .order_by(ordering)
    )
    res = conn.execute(stmt)

    return [VectorAndMediaMetadata.model_validate(row) for row in res.mappings()]

def get_thumbnail_by_timestamp(conn: sa.Connection, *, media_id: int, timestamp: float) -> Optional[bytes]:
    """
    Get the thumbnail for a given `media_id` and `timestamp` (finds the first thumbnail within between `timestamp` and `timestamp + 4`)
    If no thumbnail was found, the return value is None
    """
    start_timestamp_expr = _thumbs_table.c.timestamp >= timestamp
    end_timestamp_expr = _thumbs_table.c.timestamp < timestamp + 4
    stmt = (
        sa.select(_thumbs_table.c.content)
        .where(_thumbs_table.c.media_id == media_id)
        .where((start_timestamp_expr & end_timestamp_expr))
        .order_by(_thumbs_table.c.timestamp)
    )
    result = conn.execute(stmt)
    return result.scalar()

def get_featured_images(conn: sa.Connection) -> List[int]:
    # Get the ids of the 12th second from each video
    stmt = (
        sa.select(_vtable.c.id)
        .select_from(_vtable.join(_mtable))
        .where(_vtable.c.timestamp >= 12)
        .where(_vtable.c.timestamp < 13)
    )
    return conn.execute(stmt).scalars().all()

# def query_by_timestamp(conn, *, location: str, timestamp: Tuple[int, int]):
#     # Join the table and query by dataset_path, and return the id

#     start_timestamp_expr = _vtable.c["timestamp"] >= timestamp[0]
#     end_timestamp_col = "end_timestamp" if "end_timestamp" in _vtable.c else "timestamp"
#     end_timestamp_expr = _vtable.c[end_timestamp_col] < timestamp[1]
#     dataset_subquery = sa.select(_mtable).where(_mtable.c.location == location).cte()
#     stmt = (
#         sa.select(_vtable.c.id)
#         .join_from(_vtable, dataset_subquery)
#         .where((start_timestamp_expr & end_timestamp_expr))
#     )
#     result = conn.execute(stmt)

#     return [row[0] for row in result]
