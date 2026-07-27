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

import tempfile
import unittest
from pathlib import Path

from wise.wise_project import WiseProject


class TestCreateProject(unittest.TestCase):
    def test_init_from_empty_directory(self):
        """Check we can create a project from an existing empty directory."""
        with tempfile.TemporaryDirectory() as project_dir:
            project = WiseProject(
                project_dir, create_project=True, read_only=False
            )
            self.assertTrue(
                (Path(project_dir) / "metadata" / "internal.db").exists()
            )

    def test_fail_from_filled_directory(self):
        """Check it fails if create is called on a directory with files."""
        with tempfile.TemporaryDirectory() as project_dir:
            fh = open(Path(project_dir) / "file-stamp", "w")
            fh.close()
            with self.assertRaisesRegex(ValueError, "not empty"):
                project = WiseProject(
                    project_dir, create_project=True, read_only=False
                )
