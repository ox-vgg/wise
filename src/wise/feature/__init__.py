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

"""Feature Extractor

Contains implementation for feature extractor and feature store.

"""

from .feature_extractor import BBoxXYWH as BBoxXYWH
from .feature_extractor import FeatureExtMetadata as FeatureExtMetadata
from .feature_extractor import FeatureExtractor as FeatureExtractor
from .feature_extractor import FeatureExtractorConfig as FeatureExtractorConfig
from .feature_extractor import Features as Features
from .feature_extractor import get_torch_device as get_torch_device
from .feature_extractor_factory import (
    FeatureExtractorFactory as FeatureExtractorFactory,
)
from .feature_extractor_factory import (
    get_canonical_feature_extractor_id as get_canonical_feature_extractor_id,
)
from .feature_extractor_factory import (
    get_feature_extractor_class as get_feature_extractor_class,
)
