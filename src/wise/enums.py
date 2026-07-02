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

import enum


class ContainsEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        if type(item) == cls:
            return enum.EnumMeta.__contains__(cls, item)
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseStrEnum(str, enum.Enum, metaclass=ContainsEnumMeta):
    pass


class IndexType(BaseStrEnum):
    IndexFlatIP = "IndexFlatIP"
    IndexIVFFlat = "IndexIVFFlat"
    IndexIVFPQ = "IndexIVFPQ"
