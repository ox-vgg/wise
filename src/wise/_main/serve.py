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
import logging
import sys
from pathlib import Path

from wise.enums import IndexType


def _arg_type_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"'{path}' does not exist")
    elif path.is_dir():
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory")


def main(argv: list[str]):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Serve the REST API and frontend UI for WISE.",
        epilog=(
            "For more details about WISE, visit"
            " https://www.robots.ox.ac.uk/~vgg/software/wise/"
        ),
    )
    parser.add_argument(
        "--theme-asset-dir",
        default=Path("frontend/dist"),
        type=_arg_type_dir,
        help=(
            "Static HTML assets related to the user interface are"
            " served from this folder"
        ),
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default=None,
        choices=IndexType.__members__.keys(),
        help="The faiss index to use for serving"
    )
    parser.add_argument(
        "--proxy-root-path",
        type=str,
        default="",
        help="The root path where the app is being served behind a proxy",
    ),
    parser.add_argument(
        "--project-dir",
        type=_arg_type_dir,
        required=True,
        help="Project directory path",
    )
    args = parser.parse_args(argv[1:])

    # ensure that the frontend assets are built
    if not (args.theme_asset_dir / 'index.html').exists():
        raise FileNotFoundError(
            f"Frontend assets not found at {args.theme_asset_dir}. "
            "Please build the frontend assets using `npm install && npm run build`."
        )
    from wise.api import serve

    serve(
        args.project_dir,
        args.theme_asset_dir,
        args.index_type,
        proxy_root_path=args.proxy_root_path,
    )


if __name__ == "__main__":
    main(sys.argv)
