import sqlalchemy as sa
from ..base import wise_metadata_obj, project_metadata_obj
from ...data_models import DatasetType

project_table = sa.Table(
    "projects",
    wise_metadata_obj,
    sa.Column("id", sa.Unicode(256), primary_key=True),
    sa.Column("version", sa.Integer, nullable=False, server_default="0"),
)

dataset_table = sa.Table(
    "datasets",
    project_metadata_obj,
    sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
    sa.Column("location", sa.Unicode(1024), nullable=False),
    sa.Column("type", sa.Enum(DatasetType)),
)

metadata_table = sa.Table(
    "metadata",
    project_metadata_obj,
    sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
    sa.Column(
        "dataset_id",
        sa.Integer,
        sa.ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("dataset_row", sa.Integer, nullable=False),
    sa.Column("path", sa.Unicode(1024), nullable=False),
    sa.Column("size_in_bytes", sa.Integer, nullable=False),
    sa.Column("format", sa.String(5), nullable=False),
    sa.Column("width", sa.Integer, default=-1, nullable=False),
    sa.Column("height", sa.Integer, default=-1, nullable=False),
    sa.Column("source_uri", sa.Unicode(4096), nullable=True),
    sa.Column("metadata", sa.JSON, nullable=False, default={}),
)

# TODO Add thumbs table
