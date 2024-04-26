from typing import List, Optional
import sqlalchemy as sa

from .base import SQLAlchemyRepository

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
    """
    Get the vector and media metadata for a batch of vector ids.

    Parameters
    ----------
    conn : sqlalchemy.Connection
        Database connection for the internal metadata database
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
    res = [VectorAndMediaMetadata.model_validate(row) for row in res.mappings()]
    if len(res) != len(ids):
        raise RuntimeError(f"Unable to retrieve metadata for all ids. Retrieved metadata for {len(res)}/{len(ids)} ids")
    return res

def get_thumbnail_by_timestamp(conn: sa.Connection, *, media_id: int, timestamp: float) -> Optional[bytes]:
    """
    Get the thumbnail from a video given a `media_id` and a `timestamp` (finds the first thumbnail between `timestamp` and `timestamp + 4`).

    Parameters
    ----------
    conn : sqlalchemy.Connection
        Database connection for the thumbnail database
    media_id : int
        Media id of the video file you want to get the thumbnail from
    timestamp : float
        Timestamp within the video

    Returns
    -------
    Optional[bytes]
        Returns the raw bytes of the thumbnail in JPEG format.
        If no thumbnail was found, the return value is None.
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
    """
    Get a set of featured images to be shown on the frontend.
    Returns a list of vector ids of the 12th second from each video.

    Parameters
    ----------
    conn : sqlalchemy.Connection
        Database connection for the internal metadata database
    
    Returns
    -------
    list of int
        List of vector ids from the 12th second from each video
    """
    stmt = (
        sa.select(_vtable.c.id)
        .select_from(_vtable.join(_mtable))
        .where(_vtable.c.timestamp >= 12)
        .where(_vtable.c.timestamp < 12.5)
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
