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

class SearchIndex:
    """
    Represents various types of search indices. For example,
    the search index for audiovisual features is implemented
    in FeatureSearchIndex while the search index for text metadata
    is implemented in SqldbSearchIndex.
    """
    def __init__(self, modality_type, asset_id, assets):
        raise NotImplementedError

    def get_index_filename(self, index_type):
        raise NotImplementedError

    def create_index(self, index_type, overwrite=False):
        raise NotImplementedError

    def is_index_loaded(self):
        raise NotImplementedError

    def load_index(self, index_type):
        raise NotImplementedError

    def search(self, media_type, query, topk=5, query_type='text'):
        raise NotImplementedError
