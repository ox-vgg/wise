import sqlalchemy as sa
from ..base import wise_metadata_obj, project_metadata_obj
from ...data_models import SourceCollectionType, MediaType

project_table = sa.Table(
    "projects",
    wise_metadata_obj,
    sa.Column("id", sa.Unicode(256), primary_key=True),
    sa.Column("version", sa.Integer, nullable=False, server_default="0"),
)

source_collections_table = sa.Table(
    "source_collections",
    project_metadata_obj,
    sa.Column("id", sa.String(8), primary_key=True, nullable=False),
    sa.Column("location", sa.Unicode(1024), nullable=False),
    sa.Column("type", sa.Enum(SourceCollectionType), nullable=False),
)

media_table = sa.Table(
    "media",
    project_metadata_obj,
    sa.Column("id", sa.String(10), primary_key=True, nullable=False),
    sa.Column(
      "source_collection_id",
      sa.String(8),
      sa.ForeignKey("source_collections.id", ondelete="CASCADE"),
      nullable=False,
    ),
    sa.Column("path", sa.Unicode(1024), nullable=False),
    sa.Column("md5sum", sa.LargeBinary(128), nullable=True),
    sa.Column("size_in_bytes", sa.Integer, nullable=True),
    sa.Column("date_modified", sa.DateTime(True), nullable=True),
    sa.Column("media_type", sa.Enum(MediaType), nullable=False),
    sa.Column("format", sa.String(5), nullable=False),
    sa.Column("width", sa.Integer, nullable=False),
    sa.Column("height", sa.Integer, nullable=False),
    sa.Column("num_frames", sa.Integer, nullable=True), # only applies to video files
    sa.Column("duration", sa.Float, nullable=True), # only applies to video files

    # TODO (WISE 2) remove references to these old columns in other files:
    # sa.Column("source_uri", sa.Unicode(4096), nullable=True),
    # sa.Column("metadata", sa.JSON, nullable=False, default={}),
)

vectors_table = sa.Table(
    "vectors",
    project_metadata_obj,
    sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
    sa.Column("modality", sa.Enum(MediaType), nullable=False),
    sa.Column(
        "media_id",
        sa.String(10),
        sa.ForeignKey("media.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("timestamp", sa.Float(), nullable=True),
    sa.Column("end_timestamp", sa.Float(), nullable=True),
)

imported_metadata = sa.Table(
    "imported_metadata",
    project_metadata_obj,
    sa.Column(
        "media_id",
        sa.String(10),
        sa.ForeignKey("media.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("external_id", sa.Unicode(2048), nullable=True),
    sa.Column("metadata", sa.JSON, nullable=False, default={}),
)

# TODO (WISE 2) remove references to old metadata_table
# metadata_table = sa.Table(
#     "metadata",
#     project_metadata_obj,
#     sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
#     sa.Column(
#         "dataset_id",
#         sa.Integer,
#         sa.ForeignKey("source_collections.id", ondelete="CASCADE"),
#         nullable=False,
#     ),
#     sa.Column("dataset_row", sa.Integer, nullable=False),
#     sa.Column("path", sa.Unicode(1024), nullable=False),
#     sa.Column("size_in_bytes", sa.Integer, nullable=False),
#     sa.Column("format", sa.String(5), nullable=False),
#     sa.Column("width", sa.Integer, default=-1, nullable=False),
#     sa.Column("height", sa.Integer, default=-1, nullable=False),
#     sa.Column("source_uri", sa.Unicode(4096), nullable=True),
#     sa.Column("metadata", sa.JSON, nullable=False, default={}),
# )
