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

import sqlalchemy as sa

from .. import db
from ..data_models import (
    ExtraMediaMetadata,
    MediaMetadata,
    ModalityType,
    SourceCollection,
    ThumbnailMetadata,
    VectorMetadata,
    VideoShot,
)
from .base import SQLAlchemyRepository


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


def get_featured_images(
    conn: sa.Connection,
    modality: ModalityType,
    feature_extractor_id: str,
) -> list[int]:
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
    # select a random vector from each media id that matches the modality and feature_extractor_id
    # and is from the 4th second of the video
    where_clause = sa.and_(
        _vtable.c.modality == modality,
        _vtable.c.feature_extractor_id == feature_extractor_id
    )

    # select a random list of media ids from the media table
    random_vector = (
        sa.select(_vtable.c.id)
        .where(
            sa.and_(
                _vtable.c.media_id == _mtable.c.id,
                where_clause,
            )
        )
        .order_by(sa.func.random())
        .limit(5)
        .scalar_subquery()
    )

    stmt = sa.select(random_vector).select_from(_mtable)
    return conn.execute(stmt).scalars().all()
