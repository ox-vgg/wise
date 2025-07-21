from typing import List, Optional, Union
import sqlalchemy as sa

from .base import SQLAlchemyRepository

# from .extra_metadata import ExtraMediaMetadataSQLAlchemyRepository
from ..data_models import (
    MediaType,
    ModalityType,
    Project,
    SourceCollection,
    MediaMetadata,
    VectorAndMediaMetadata,
    VectorMetadata,
    ExtraMediaMetadata,
    ThumbnailMetadata,
    VideoShot
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

VideoShotsRepo = SQLAlchemyRepository[VideoShot, VideoShot, VideoShot](db.shots_table, VideoShot)


_vtable = db.vectors_table
_mtable = db.media_table
_thumbs_table = db.thumbnails_table

def get_full_metadata_batch(conn: sa.Connection, ids: List[int], external_metadata_tables = []) -> List[VectorAndMediaMetadata]:
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

    stmt1 = (
        sa.select(_vtable.c, _mtable.c)
        .select_from(_vtable.join(_mtable))
        .where(_vtable.c.id.in_(ids))
        .order_by(ordering)
    )
    res1 = conn.execute(stmt1)
    res1 = [VectorAndMediaMetadata.model_validate(row) for row in res1.mappings()]
    if len(external_metadata_tables) == 0:
        if len(res1) != len(ids):
            raise RuntimeError(f"Unable to retrieve metadata for all ids. Retrieved metadata for {len(res1)}/{len(ids)} ids")
        return res1

    ## collect external metadata
    from_clause = _vtable.join(_mtable)
    for external_metadata_table in external_metadata_tables:
        from_clause = from_clause.join(external_metadata_table)
    external_metadata_colnames = []
    for external_metadata_table in external_metadata_tables:
        for col in external_metadata_table.columns:
            if col.name not in ['media_id', 'timestamp', 'end_timestamp', 'vector_id']:
                external_metadata_colnames.append(col)
    stmt2 = (
        sa.select(*external_metadata_colnames)
        .select_from(from_clause)
        .where(_vtable.c.id.in_(ids))
        .order_by(ordering)
    )

    res2 = conn.execute(stmt2)
    for m, r in zip(res1, res2.mappings()):
        m.external_metadata = r

    if len(res1) != len(ids):
        raise RuntimeError(f"Unable to retrieve metadata for all ids. Retrieved metadata for {len(res1)}/{len(ids)} ids")
    return res1

def get_thumbnail_by_timestamp(conn: sa.Connection, *, media_id: int, timestamp: float, get_id_only: bool = False) -> Optional[Union[bytes, int]]:
    """
    Get the thumbnail from a video given a `media_id` and a `timestamp` (finds the first thumbnail between `timestamp - 0.25` and `timestamp + 2`).

    Parameters
    ----------
    conn : sqlalchemy.Connection
        Database connection for the thumbnail database
    media_id : int
        Media id of the video file you want to get the thumbnail from
    timestamp : float
        Timestamp within the video
    get_id_only : bool, optional
        If set to True, the integer id of the matching thumbnail is returned.
        If set to False (default), the raw bytes of the thumbnail is returned.

    Returns
    -------
    Optional[Union[bytes, int]]
        Returns the raw bytes of the thumbnail in JPEG format.
        If `get_id_only` was set to True, then the integer id of the thumbnail is returned instead.
        If no thumbnail was found, the return value is None.
    """
    # TODO Convert timestamp search interval to a project configuration and pass it down
    start_timestamp_expr = _thumbs_table.c.timestamp >= timestamp - 0.25
    end_timestamp_expr = _thumbs_table.c.timestamp <= timestamp + 2
    stmt = (
        sa.select(_thumbs_table.c.content if not get_id_only else _thumbs_table.c.id)
        .where(_thumbs_table.c.media_id == media_id)
        .where((start_timestamp_expr & end_timestamp_expr))
        .order_by(_thumbs_table.c.timestamp)
    )
    result = conn.execute(stmt)
    return result.scalar()

def get_featured_images(
    conn: sa.Connection, modality: ModalityType, feature_extractor_id: str,
    use_shots: bool = False
) -> List[int]:
    """
    Get a set of featured images to be shown on the frontend.
    Returns a list of vector ids of the 4th second from each video.

    Parameters
    ----------
    conn : sqlalchemy.Connection
        Database connection for the internal metadata database

    modality: ModalityType
        Modality to which to limit the search.

    feature_extractor_id: string
        Feature extractor id to which to limit the search.

    Returns
    -------
    list of int
        List of vector ids from the 4th second from each video
    """
    where_clause = sa.and_(
        _vtable.c.modality == modality,
        _vtable.c.feature_extractor_id == feature_extractor_id
    )
    if modality != ModalityType.IMAGE:
        # Get the vector id from the 4th second of each video (non-image)
        if use_shots:
            # If using shots, we need to join with the shots table
            where_clause = sa.and_(
                where_clause,
                _vtable.c.timestamp >= 4,
                _vtable.c.timestamp < 20,
            )
        else:
            where_clause = sa.and_(
                where_clause,
                _vtable.c.timestamp >= 4,
                _vtable.c.timestamp < 4.5,
            )

    stmt = (
        sa.select(_vtable.c.id)
        .where(where_clause)
    )
    return conn.execute(stmt).scalars().all()

def get_project_total_duration(conn: sa.Connection) -> Optional[float]:
    """
    Get the total duration (in seconds) of all the video/audio files in the project
    """
    return conn.execute(sa.select(sa.sql.func.sum(_mtable.c.duration))).scalar()

def get_media_counts_by_media_type(conn: sa.Connection) -> dict[MediaType, int]:
    """
    Get the number of media files for each media type (image, video, audio).
    Note that the "av" and "video" media types are both counted as "video".
    """
    results = conn.execute(sa.select(_mtable.c.media_type, sa.func.count(_mtable.c.id)).group_by(_mtable.c.media_type)).tuples().all()
    media_counts = {
        media_type: count for media_type, count in results
    }
    if MediaType.AV in media_counts:
        media_counts[MediaType.VIDEO] = media_counts.get(MediaType.VIDEO, 0) + media_counts[MediaType.AV]
        del media_counts[MediaType.AV]
    return media_counts

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


def get_related_vectors_rows(conn: sa.Connection, vid: int):
    """Get all other vectors for same image/timestamp, modality, and feature extractor.
    """
    subq = sa.select(_vtable).where(_vtable.c.id == vid).subquery()
    stmt = (
        sa.select(_vtable)
        .join(
            subq,
            (
                (_vtable.c.media_id == subq.c.media_id)
                & (_vtable.c.timestamp == subq.c.timestamp)
                & (_vtable.c.modality == subq.c.modality)
                & (_vtable.c.feature_extractor_id == subq.c.feature_extractor_id)
            )
        )
        .where(_vtable.c.id != vid)
    )
    return conn.execute(stmt)
