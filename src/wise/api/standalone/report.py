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

from fastapi import APIRouter, File, Form
from fastapi.responses import PlainTextResponse
import os
import csv
import json

router = APIRouter()

@router.post("/report")
def report_image(
    file_queries: list[bytes] = File([]),
    url_queries: list[str] = Form([]),
    text_queries: list[str] = Form([]),
    sourceURI: str = Form(),
    reasons: list[str] = Form([]),
):
    # TODO implement code to store data in database
    # For now, we are saving the reports in a CSV file
    report_filename = "data/reported_images.csv"
    fieldnames = [
        "text_queries",
        "url_queries",
        "file_queries",
        "sourceURI",
        "reasons",
    ]

    # Write header row if the file doesn't exist
    if not os.path.exists(report_filename):
        os.makedirs(os.path.dirname(report_filename), exist_ok=True)
        with open(report_filename, "a", newline="") as report_file:
            csv.writer(report_file).writerow(fieldnames)

    # Write data row
    with open(report_filename, "a", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writerow(
            {
                "text_queries": json.dumps(text_queries),
                "url_queries": json.dumps(url_queries),
                "file_queries": json.dumps(
                    # to prevent the CSV file from getting too large, we store a placeholder text ('uploaded image')
                    # instead of storing the image file
                    ["uploaded image" for _ in file_queries]
                ),
                "sourceURI": sourceURI,
                "reasons": json.dumps(reasons),
            }
        )

    return PlainTextResponse(status_code=200, content="Image has been reported")
