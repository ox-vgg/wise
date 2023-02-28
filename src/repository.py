from .crud import SQLAlchemyRepository
from .data_models import Project, Dataset, DatasetCreate, ImageMetadata
from . import db

WiseProjectsRepo = SQLAlchemyRepository[Project, Project, Project](
    db.project_table, Project
)
DatasetRepo = SQLAlchemyRepository[Dataset, DatasetCreate, Dataset](
    db.dataset_table, Dataset
)
MetadataRepo = SQLAlchemyRepository[ImageMetadata, ImageMetadata, ImageMetadata](
    db.metadata_table, ImageMetadata
)
