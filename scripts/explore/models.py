import enum
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import TypeDecorator, BLOB
import numpy as np

Base = declarative_base()

class NumpyArray(TypeDecorator):
    impl = BLOB

    def process_bind_param(self, value, dialect):
        if value is not None:
            return value.astype(np.float32).tobytes()
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return np.frombuffer(value, dtype=np.float32)
        return None

class Facet(Base):
    __tablename__ = 'facets'
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String(255), nullable=False, unique=True)
    feature_extractor_id = sa.Column(sa.String(255), nullable=False)

class FacetMetadataSchema(Base):
    __tablename__ = 'facet_metadata_schema'
    __table_args__ = (sa.UniqueConstraint('facet_id', 'key_name', name='uq_facet_key'),)
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    facet_id = sa.Column(sa.Integer, sa.ForeignKey('facets.id', ondelete='CASCADE'), nullable=False)
    key_name = sa.Column(sa.String(255), nullable=False)
    data_type = sa.Column(sa.String(50), nullable=False, default="string") # string, number, date

class ClusterStatus(enum.Enum):
    draft = "draft"
    reviewed = "reviewed"
    published = "published"

class Cluster(Base):
    __tablename__ = 'clusters'
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    facet_id = sa.Column(sa.Integer, sa.ForeignKey('facets.id', ondelete='CASCADE'), nullable=False)
    cluster_label = sa.Column(sa.String(255), nullable=True)
    metadata_json = sa.Column(sa.JSON, nullable=False, default={})
    status = sa.Column(sa.Enum(ClusterStatus), nullable=False, default=ClusterStatus.draft)
    is_starred = sa.Column(sa.Boolean, nullable=False, default=False)

class Assignment(Base):
    __tablename__ = 'assignments'
    __table_args__ = (sa.UniqueConstraint('vector_id', 'cluster_id', name='uq_vector_cluster'),)
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    vector_id = sa.Column(sa.Integer, nullable=False, index=True)
    cluster_id = sa.Column(sa.Integer, sa.ForeignKey('clusters.id', ondelete='CASCADE'), nullable=False, index=True)
    confidence_score = sa.Column(sa.Float, nullable=True)
    is_manual_override = sa.Column(sa.Boolean, nullable=False, default=False)

class KnownFaceCluster(Base):
    __tablename__ = 'known_face_clusters'
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    cluster_id = sa.Column(sa.Integer, sa.ForeignKey('clusters.id', ondelete='CASCADE'))
    vector_id = sa.Column(sa.Integer, index=True)
    centroid = sa.Column(NumpyArray, nullable=False)
