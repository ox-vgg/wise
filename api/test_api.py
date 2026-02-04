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

import os.path
import tempfile
import unittest

from fastapi.testclient import TestClient

from api import create_app
from config import APIConfig
from src.wise_project import WiseProject


class TestWithEmptyProject(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        ## project_dir is a subdir of tmp_dir because it must not
        ## exist before creating WiseProject (otherwise the
        ## subdirectories will not be created).
        self.project_dir = os.path.join(self.tmp_dir.name, "wise-test-project")
        self.project = WiseProject(self.project_dir, create_project=True)

        self.api_config = APIConfig(
            project_dir=self.project_dir,
            command="serve",
        )
        self.assets_dir = self.project_dir
        self.app = create_app(self.api_config, self.project_dir)
        self.client = TestClient(self.app)

    def test_get_info(self):
        response = self.client.get("/wise-test-project/info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "project_name": "wise-test-project",
                "num_vectors": 0,
                "num_media_files": 0,
                "num_thumbnails": 0,
                "num_shots": 0,
                "media_file_counts": {},
                "total_duration": 0.0,
                "models": {},
                "shot_based_filters": None,
                "search_targets": {},
            },
        )

    def test_get_nonexistent_media(self):
        resp = self.client.get("/wise-test-project/media/1")
        self.assertEqual(resp.status_code, 404)
        ## XXX: this returns plain text but maybe should return json
        ## like the others when it errors?
        self.assertEqual(resp.content, b"1 not found!")

    def test_get_related_vectors_to_nonexistent_vector(self):
        resp = self.client.get("/wise-test-project/related-vectors/1")
        ## XXX: this succeeds but maybe it should 404?
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    def test_get_metadata_to_nonexistent_media(self):
        resp = self.client.get("/wise-test-project/metadata/1")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["detail"], "Metadata not found!")

    def test_get_thumbnail_to_nonexistent_media(self):
        resp = self.client.get("/wise-test-project/thumbnail/1")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["detail"], "Not Found")

    def tearDown(self):
        self.tmp_dir.cleanup()
