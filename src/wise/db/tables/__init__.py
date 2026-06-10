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

from wise.data_models import MediaType, ModalityType, SourceCollectionType
from wise.db.base import (
    facets_metadata_obj,
    project_metadata_obj,
    thumbs_metadata_obj,
)


source_collections_table = sa.Table(
    "source_collections",
    project_metadata_obj,
    sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
    sa.Column("location", sa.Unicode(1024), nullable=False),
    sa.Column("type", sa.Enum(SourceCollectionType), nullable=False),
)

media_table = sa.Table(
    "media",
    project_metadata_obj,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column(
        "source_collection_id",
        sa.Integer,
        sa.ForeignKey("source_collections.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("path", sa.Unicode(1024), nullable=False),
    sa.Column("checksum", sa.LargeBinary(128), index=True, nullable=True),
    sa.Column("size_in_bytes", sa.Integer, nullable=True),
    sa.Column("date_modified", sa.DateTime(True), nullable=True),
    sa.Column("media_type", sa.Enum(MediaType), nullable=False),
    sa.Column("format", sa.String(5), nullable=False),
    sa.Column("width", sa.Integer, nullable=False),
    sa.Column("height", sa.Integer, nullable=False),
    sa.Column("num_frames", sa.Integer, nullable=True),  # only applies to video files
    sa.Column("duration", sa.Float, nullable=True),  # only applies to video files
)

vectors_table = sa.Table(
    "vectors",
    project_metadata_obj,
    sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
    sa.Column("modality", sa.Enum(ModalityType), index=True, nullable=False),
    sa.Column("feature_extractor_id", sa.Unicode(255), index=True, nullable=False),
    sa.Column(
        "media_id",
        sa.Integer,
        sa.ForeignKey("media.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    ),
    sa.Column("timestamp", sa.Float(), index=True, nullable=True),
    sa.Column("end_timestamp", sa.Float(), index=True, nullable=True),
)

imported_metadata_table = sa.Table(
    "imported_metadata",
    project_metadata_obj,
    sa.Column(
        "media_id",
        sa.Integer,
        sa.ForeignKey("media.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("external_id", sa.Unicode(2048), nullable=True),
    sa.Column("metadata", sa.JSON, nullable=False, default={}),
)

thumbnails_table = sa.Table(
    "thumbnails",
    thumbs_metadata_obj,
    sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
    sa.Column(
        "media_id",
        sa.Integer,
        index=True,
        nullable=False,
    ),
    sa.Column("timestamp", sa.Float(), index=True, nullable=True),
    sa.Column("content", sa.LargeBinary(), nullable=False),
    sa.schema.Index("ix_thumbnails_media_id_and_timestamp", "media_id", "timestamp"),
)

shots_table = sa.Table(
    "shots",
    project_metadata_obj,
    sa.Column("id", sa.Integer, nullable=False, primary_key=True),
    sa.Column(
        "media_id",
        sa.Integer,
        sa.ForeignKey("media.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        primary_key=True,
    ),
    sa.Column("ts", sa.Float, nullable=False, index=True),
    sa.Column("te", sa.Float, nullable=False, index=True),
    sa.Column("shot_scale", sa.Integer, default=None, index=True),
)
