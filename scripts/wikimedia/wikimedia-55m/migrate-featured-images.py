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

"""Imported the featured images from the original wise 1 wikimedia demo.

The featured images in the original WISE 1 demo were selected from a
`featured_images.json` file at the root of the repository.  This file
was a list of objects like:

```
{
  'row_num': 17810,
  'img_title': '"Everything is Going to be Alright" artwork, Christchurch Art Gallery, Christchurch, New Zealand.jpg',
  'orig_width': 4342,
  'orig_height': 1995,
  'original_download_url': 'https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/%22Everything_is_Going_to_be_Alright%22_artwork%2C_Christchurch_Art_Gallery%2C_Christchurch%2C_New_Zealand.jpg&width=975',
  'ImageDescription': '"Everything is Going to be Alright" artwork by Martin Creed, Christchurch Art Gallery, Christchurch, New Zealand',
  'DateTimeOriginal': 'Taken on\xa019 April 2020, 19:14:32',
  'Artist': '<a href="//commons.wikimedia.org/wiki/User:Podzemnik" title="User:Podzemnik">Michal Klajban</a>',
  'LicenseShortName': 'CC BY-SA 4.0',
  'Credit': '<span class="int-own-work" lang="en">Own work</span>',
  'UsageTerms': 'Creative Commons Attribution-Share Alike 4.0',
}
```

None of these mapped to the database, so these featured images are not
necessarily images in the actual wise project.  Some are though.

The featured images in WISE 2 require images (vectors actually, the
difference is important for faces and objects) that are indexed.  The
approach taken in this scrip twas to compare `original_download_url`
in the json file with `source_uri` in the WISE 2 database.  Featured
images that are not part of the project are ignored.

This script can be used as example to migrate other projects with
featured_images or how to import from other lists.

"""

import argparse
import json
import logging
import re
import sqlite3
import sys
import urllib.parse
from pathlib import Path

import sqlalchemy as sa

import wise.db
from wise.wise_project import WiseProject

logger = logging.getLogger(__name__)


def parse_wikimedia_uri(uri):
    ## Both source_uri in the database and original_download_url in
    ## featured_image.json are "quoted" but different (the urls in
    ## featured_image.json have more characters quoted) so we unquote
    ## both to simplify comparison.
    commons_redirect_file_url = "https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/"
    assert uri.startswith(commons_redirect_file_url)
    fname = uri[len(commons_redirect_file_url) :]
    m = re.search(r"&width=\d+$", fname)
    assert m
    fname = fname[: m.start()]
    fname = urllib.parse.unquote(fname)
    return fname


def get_wise1_paths(db_fpath):
    conn = sqlite3.connect(db_fpath)
    stmt = """
        SELECT id, source_uri
        FROM metadata
        WHERE source_uri IS NOT NULL
    """
    fname_to_id = dict()
    for mid, uri in conn.execute(stmt):
        assert uri not in fname_to_id
        fname_to_id[parse_wikimedia_uri(uri)] = mid
    logger.info("There are %d media with name", len(fname_to_id))
    return fname_to_id


def get_wise1_featured_fnames(featured_fpath):
    with open(featured_fpath, "r") as fh:
        objs = json.load(fh)
    featured_fnames = {
        parse_wikimedia_uri(x["original_download_url"]) for x in objs
    }
    assert len(objs) == len(featured_fnames)
    logger.info("There are %d featured images", len(featured_fnames))
    return featured_fnames


def get_wise1_featured_ids(wise1_db_fpath, featured_fpath):
    fname_to_id = get_wise1_paths(wise1_db_fpath)
    featured_fnames = get_wise1_featured_fnames(featured_fpath)
    featured_ids = set()
    for fname in featured_fnames:
        maybe_id = fname_to_id.get(fname)
        if maybe_id is not None:
            featured_ids.add(maybe_id)
        else:
            logger.info("Failed to find id for name '%s'", fname)

    logger.info(
        "Found ids for %d images (out of %d)",
        len(featured_ids),
        len(featured_fnames),
    )
    return featured_ids


def main(argv: list[str]) -> int:
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logging-level",
        action="store",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set logging level",
    )
    parser.add_argument("wise1_featured_fpath", type=Path)
    parser.add_argument("wise1_db_fpath", type=Path)
    parser.add_argument("wise2_project_dir", type=Path)
    args = parser.parse_args(argv[1:])

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.logging_level.upper()))

    featured_ids = get_wise1_featured_ids(
        args.wise1_db_fpath, args.wise1_featured_fpath
    )

    wise2_project = WiseProject(args.wise2_project_dir, create_project=False)
    with wise2_project.db_engine.connect() as conn:
        cur = conn.execute(
            sa.insert(wise.db.featured_table),
            [{"vector_id": x} for x in featured_ids],
        )
        conn.commit()
        assert cur.rowcount == len(featured_ids)
        logger.info("Inserted %d featured vectors in database", cur.rowcount)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
