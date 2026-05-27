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
from io import BytesIO

from fastapi import UploadFile
from fastapi.testclient import TestClient
from pydantic import HttpUrl

import api.common
from api import create_app
from api.common import MediaQueryTerm, TextQueryTerm
from config import APIConfig
from src.wise_project import WiseProject


class TestWithEmptyProject(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        ## project_dir is a subdir of tmp_dir because it must not
        ## exist before creating WiseProject (otherwise the
        ## subdirectories will not be created).
        self.project_dir = os.path.join(self.tmp_dir.name, "wise-test-project")

        ## The database files must exist later during serve() which
        ## will create another instance of WiseProject in readonly
        ## mode.  So we "get" the db engines here to trigger the db
        ## creation.  See https://gitlab.com/vgg/wise/wise/-/work_items/231
        project = WiseProject(self.project_dir, create_project=True, read_only=False)
        _ = project.db_engine
        _ = project.thumbsdb_engine

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


class TestHandlingMultipartForm(unittest.TestCase):
    def test_merge_with_file(self):
        query_form = [
            '{"term_id": "bill", "is_negative": true, "txt": "vi"}',
            '{"term_id": "rms", "is_negative": false, "src": null, "qtype": "visual"}',
        ]
        query_form_files = [
            UploadFile(
                file=BytesIO(bytes([101, 109, 97, 99, 115])),
                filename="rms",
                headers={"content-type": "application/octet-stream"},
            )
        ]
        expected_query = [
            TextQueryTerm(term_id="bill", is_negative=True, txt="vi"),
            MediaQueryTerm(
                term_id="rms",
                is_negative=False,
                src=bytes([101, 109, 97, 99, 115]),
                qtype="visual",
            ),
        ]

        self.assertEqual(
            api.common.merge_multipart_query_form(query_form, query_form_files),
            expected_query,
        )

    def test_parsing_internal_vector_id(self):
        query_form = [
            '{"term_id": "foo", "is_negative": true, "txt": "crane"}',
            '{"term_id": "bar", "is_negative": false, "vector_id": "0/0/0"}',
        ]
        expected_q = [
            api.common.TextQueryTerm(
                term_id="foo", is_negative=True, txt="crane"
            ),
            api.common.VectorIdQueryTerm(
                term_id="bar", is_negative=False, vector_id="0/0/0"
            )
        ]
        self.assertEqual(
            api.common.merge_multipart_query_form(query_form, []), expected_q
        )

    def test_parsing_url(self):
        url = "http://www.example.com/960px-Flagstone_3_(20767991).jpg"
        query_form = [
            '{"term_id": "gnu", "is_negative": false, "src": "%s", "qtype": "visual"}' % url,
        ]
        expected_q = [
            api.common.MediaQueryTerm(
                term_id="gnu",
                is_negative=False,
                src=HttpUrl(url),
                qtype="visual",
            )
        ]
        self.assertEqual(
            api.common.merge_multipart_query_form(query_form, []), expected_q
        )
