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

from base64 import b64encode
import itertools


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def batched(iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch


BASE64JPEGPREFIX = b"data:image/jpeg;charset=utf-8;base64,"

convert_uint8array_to_base64 = lambda x: (
    BASE64JPEGPREFIX + b64encode(bytes(x))
).decode("utf-8")
