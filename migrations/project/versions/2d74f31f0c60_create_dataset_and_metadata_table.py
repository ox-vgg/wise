"""create dataset and metadata table

Revision ID: 2d74f31f0c60
Revises:
Create Date: 2023-02-16 18:12:58.326036

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2d74f31f0c60"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
        sa.Column("location", sa.Unicode(1024), nullable=False),
    )
    op.create_table(
        "metadata",
        sa.Column("id", sa.Integer, autoincrement=True, primary_key=True),
        sa.Column(
            "dataset_id", sa.Integer, sa.ForeignKey("datasets.id"), nullable=False
        ),
        sa.Column("path", sa.Unicode(1024), nullable=False),
        sa.Column("size_in_bytes", sa.Integer, nullable=False),
        sa.Column("format", sa.String(5), nullable=False),
        sa.Column("width", sa.Integer, default=-1, nullable=False),
        sa.Column("height", sa.Integer, default=-1, nullable=False),
        sa.Column("source_uri", sa.Unicode(4096), nullable=True),
        sa.Column("metadata", sa.JSON, nullable=False, default={}),
    )


def downgrade() -> None:
    op.drop_table("metadata")
    op.drop_table("datasets")
