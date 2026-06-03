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

import logging
from pathlib import Path
from typing import Optional

import typer

from wise.enums import IndexType


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
)

app = typer.Typer()
@app.command(
    help="Serve the REST API and frontend UI for WISE.",
    epilog="For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/",
    no_args_is_help=True,
)
def main(
    project_dir: Path = typer.Option(
        help="Project directory path"
    ),
    theme_asset_dir: Path = typer.Option(
        "frontend/dist",
        exists=True,
        dir_okay=True,
        file_okay=False,
        help=(
            "Static HTML assets related to the user interface are served "
            "from this folder."
        ),
    ),
    index_type: Optional[IndexType] = typer.Option(
        None,
        help="The faiss index to use for serving"
    ),
    proxy_root_path: str = typer.Option(
        "",
        help="The root path where the app is being served behind a proxy",
    ),
):
    # ensure that the frontend assets are built
    if not Path(theme_asset_dir / 'index.html').exists():
        raise FileNotFoundError(
            f"Frontend assets not found at {theme_asset_dir}. "
            "Please build the frontend assets using `npm install && npm run build`."
        )
    from wise.api import serve

    serve(
        project_dir,
        theme_asset_dir,
        index_type.value if index_type else None,
        proxy_root_path=proxy_root_path,
    )


if __name__ == "__main__":
    app()
