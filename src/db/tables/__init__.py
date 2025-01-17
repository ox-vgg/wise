import sqlalchemy as sa
from ..base import project_metadata_obj, thumbs_metadata_obj
from ...data_models import SourceCollectionType, MediaType, ModalityType

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
    sa.Column("checksum", sa.LargeBinary(128), nullable=True),
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
    sa.Column("modality", sa.Enum(ModalityType), nullable=False),
    sa.Column(
        "media_id",
        sa.Integer,
        sa.ForeignKey("media.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("timestamp", sa.Float(), nullable=True),
    sa.Column("end_timestamp", sa.Float(), nullable=True),
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
