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

from .feature_search_index import FeatureSearchIndex
from .sqlite_search_index import SqliteSearchIndex


def SearchIndexFactory(media_type, asset_id, asset):
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
        return FeatureSearchIndex(media_type, asset_id, asset)
    elif media_type == 'metadata':
        return SqliteSearchIndex(media_type, asset_id, asset)
    else:
        raise ValueError(f'Unknown media_type {media_type}')
