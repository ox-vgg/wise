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


## FIXME: modality_type is a str because it may have the value
## "metadata" but ideally it would be a ModalityType.
def SearchIndexFactory(modality_type: str, asset_id, asset):
    """
    Create search index based on modality_type

    Parameters
    ----------
    modality_type : str
         can be ['audio', 'video', 'image', 'metadata']
    media_assets : dict
         see src/wise_project.py::discover_assets()

    """
    if modality_type in ['audio', 'video', 'image']:
        return FeatureSearchIndex(modality_type, asset_id, asset)
    elif modality_type == 'metadata':
        return SqliteSearchIndex(modality_type, asset_id, asset)
    else:
        raise ValueError(f'Unknown modality_type {modality_type}')
