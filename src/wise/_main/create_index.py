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

import argparse
import json
import logging
from pathlib import Path

from wise import db
from wise.config import APIConfig
from wise.data_models import ModalityType
from wise.index.feature_search_index import FeatureSearchIndex
from wise.search.fts import FTSSearch
from wise.wise_project import WiseProject

logger = logging.getLogger(__name__)


def create_fts_index(project, args):
    if not args.fts_config:
        raise ValueError(
            "--fts-config must be a valid json file to index metadata"
        )

    fts_config = Path(args.fts_config)
    if not fts_config.exists():
        raise ValueError(
            "--fts-config must be a valid json file to index metadata"
        )

    if not args.overwrite and project.fts_config_file.exists():
        logger.info("not overwriting existing metadata index")
        return

    with fts_config.open() as f:
        fts_tables_columns = json.load(f)

    logger.info("Creating fts index with config %s", fts_tables_columns)
    project_engine = project.db_engine
    db.reflect_external_metadata(project_engine)
    try:
        project.fts_config_file.write_text(
            json.dumps(fts_tables_columns, sort_keys=True)
        )
        fts_index = FTSSearch(project, db.project_metadata_obj)
        with project_engine.begin() as conn:
            fts_index.build_index(conn)

        logger.info("Successfully created fts5 index for metadata")
    except Exception:
        logging.exception("failed to create metadat index")
        project.fts_config_file.unlink(missing_ok=True)


def main(argv: list[str]):
    logging.basicConfig(
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="create-index",
        description="Create a nearest neighbour search index for features extracted from images and videos.",
        epilog="For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/",
    )

    parser.add_argument(
        "--logging-level",
        action="store",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set logging level",
    )

    parser.add_argument(
        "--media-type",
        default=[],
        required=False,
        action="append",
        choices=["audio", "video", "image"],
        help="alias to --modality-type; kept for backwards compatibility",
    )

    parser.add_argument(
        "--modality-type",
        default=[],
        required=False,
        action="append",
        choices=["audio", "video", "image"],
        help="create search index only for these media types; applies to all by default ",
    )

    parser.add_argument(
        "--index-type",
        required=False,
        default="IndexFlatIP",
        choices=["IndexFlatIP", "IndexIVFFlat"],
        type=str,
        help="the type of faiss index for feature vectors, metadata defaults to full-text-search index",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        default=False,
        help="overwrite existing index file",
    )

    parser.add_argument(
        "--project-dir",
        required=True,
        type=str,
        help="folder where all project assets are stored",
    )

    parser.add_argument(
        "--feature-id",
        required=False,
        type=str,
        help="the id of the feature to create an index for",
    )

    parser.add_argument(
        "--fts-config",
        help="json file representing the config for building the FTS5 index",
    )
    args = parser.parse_args(argv[1:])
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.logging_level.upper()))

    config = APIConfig(project_dir=args.project_dir, command="create_index")

    project = WiseProject(args.project_dir)
    project_assets = project.discover_assets()
    logger.debug(
        "Discovered assets for modality types: %s", list(project_assets.keys())
    )

    modality_types_wanted = set()
    modality_types_wanted.update([ModalityType(x) for x in args.media_type])
    modality_types_wanted.update([ModalityType(x) for x in args.modality_type])
    if not modality_types_wanted:
        modality_types_wanted.update(list(project_assets.keys()))
        logger.debug(
            "No modality types specified on command line; defaulting to all available '%s'",
            modality_types_wanted,
        )

    if args.fts_config:
        create_fts_index(project, args)

    if not modality_types_wanted:
        logger.info("No index requested for any modality type")
    for modality_type in modality_types_wanted:
        logger.info("Creating index for modality types: %s", modality_type)
        feature_extractor_id_list = list(project_assets[modality_type].keys())
        if args.feature_id:
            if args.feature_id not in feature_extractor_id_list:
                raise ValueError(
                    f"feature id {args.feature_id} not found for modality type {modality_type}"
                )
            feature_extractor_id_list = [args.feature_id]

        for feature_extractor_id in feature_extractor_id_list:
            asset = project_assets[modality_type][feature_extractor_id]
            search_index = FeatureSearchIndex(
                modality_type, feature_extractor_id, asset
            )
            search_index.create_index(args.index_type, args.overwrite)
