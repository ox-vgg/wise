from .feature_search_index import FeatureSearchIndex
from .sqlite_search_index import SqliteSearchIndex
from ..feature.feature_extractor import FeatureExtractor


def SearchIndexFactory(
    media_type, asset_id, asset, feature_extractor: FeatureExtractor | None = None
):
    """
    Create search index based on media_type

    Parameters
    ----------
    media_type : str
         can be ['audio', 'video', 'metadata']
    media_assets : dict
         see src/wise_project.py::discover_assets()

    """
    if media_type in ['audio', 'video', 'image']:
        if feature_extractor is None:
            raise ValueError(
                f"Feature extractor must be provided for media type {media_type}"
            )
        return FeatureSearchIndex(media_type, asset_id, asset, feature_extractor)
    elif media_type == 'metadata':
        return SqliteSearchIndex(media_type, asset_id, asset)
    else:
        raise ValueError(f'Unknown media_type {media_type}')
