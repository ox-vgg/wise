from pydantic import BaseModel
from src.data_models import VectorAndMediaMetadata
from src.feature.feature_extractor import FeatureExtMetadata

class SearchOutput(BaseModel):
    ids: list[int | None] = []
    distances: list[float] = []
    metadata: list[VectorAndMediaMetadata] = []
    ext_metadata: list[FeatureExtMetadata] = []
