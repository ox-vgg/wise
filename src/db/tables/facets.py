import sqlalchemy as sa
from ..base import facets_metadata_obj
from . import vectors_table

facets_table = sa.Table(
    "facets",
    facets_metadata_obj,
    sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
    sa.Column("name", sa.Unicode(255), nullable=False, unique=True),
    sa.Column("feature_extractor_id", sa.Unicode(255), nullable=False),
)

facet_metadata_table = sa.Table(
    "facet_metadata",
    facets_metadata_obj,
    sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
    sa.Column(
        "vector_id",
        sa.Integer,
        sa.ForeignKey(vectors_table.c.id, ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    sa.Column("cluster_id", sa.Integer, nullable=False, index=True),
)

cluster_metadata_table = sa.Table(
    "cluster_metadata",
    facets_metadata_obj,
    sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
    sa.Column("cluster_id", sa.Integer, nullable=False, index=True, unique=True),
    sa.Column("facet_id", sa.Integer, sa.ForeignKey("facets.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("cluster_label", sa.Unicode(255), nullable=False),
    sa.Column("metadata_json", sa.JSON, nullable=False, default={}),
)