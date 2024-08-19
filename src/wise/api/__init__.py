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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional, TypedDict

from fastapi import FastAPI, Request, APIRouter
from fastapi.staticfiles import StaticFiles
import uvicorn

from wise.api.config import APIConfig
from . import common
from . import dependencies
from pathlib import Path

logger = logging.getLogger(__name__)

def log_custom_format(message: str):
    """Log a message with a custom format (green text, bold)"""
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[92m" # green color
    BOLD_SEQ = "\033[1m"
    logger.info(
        f'{COLOR_SEQ}{BOLD_SEQ}{message}{RESET_SEQ}',
    )


class State(TypedDict):
    config: APIConfig


def setup_routers(config: APIConfig):
    dependencies.init(config)

    from .standalone import report_router

    if config.remote_projects:
        from .aggregator import project_router
        from .aggregator import search_router
    else:
        from .standalone import project_router
        from .standalone import search_router

    project_info = dependencies.project_info
    project_name = project_info.name

    router = APIRouter(prefix=f"/{project_name}", tags=[f"{project_name}"])
    router.include_router(project_router)
    router.include_router(report_router)
    router.include_router(search_router)
    return router


def create_app(config: APIConfig, theme_asset_dir: Path):
    # Apply precision monkey patching
    common.patch_precision(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[State]:
        # Startup code
        yield {"config": config}
        # Shutdown code

    app = FastAPI(
        title="Wise API Server",
        summary="Multi-modal search engine for audiovisual collections",
        description="""
        The WISE API Server provides access to a multi-modal search engine for large-scale audiovisual collections.
        It supports searching using text queries, image queries, and sketch queries, as well as browsing and retrieving media content.
        """,
        contact={
            "name": "WISE Team",
            "email": "vgg-webmasters@robots.ox.ac.uk",
            "url": "https://www.robots.ox.ac.uk/~vgg/software/wise/",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        lifespan=lifespan,
        root_path_in_servers=False,
    )

    # Enable CORS for development mode
    # If you are running a dev server for the frontend React app,
    # this allows the frontend dev server on a different port to access the backend
    if config.mode == 'development':
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        if config.enable_profiling:
            from pyinstrument import Profiler
            from pyinstrument.renderers.html import HTMLRenderer

            @app.middleware("http")
            async def profile_request(request: Request, call_next):
                if request.query_params.get("profile", False):
                    with Profiler(interval=0.001, async_mode="enabled") as profiler:
                        response = await call_next(request)
                    with open(f"profile.html", "w") as out:
                        out.write(profiler.output(renderer=HTMLRenderer()))
                    return response
                else:
                    return await call_next(request)

    logger.info(f"Loading html user interface from {theme_asset_dir}")
    app.include_router(setup_routers(config))

    app.mount(
        f"/{config.project_dir.name}/",
        StaticFiles(directory=theme_asset_dir, html=True),
        name="assets",
    )
    log_custom_format(
        f"Open http://{config.listen_address}:{config.port}/{config.project_dir.name}/ in your browser"
    )

    return app


def serve(
    project_dir: Path,
    theme_asset_dir: Path,
    index_type: Optional[str] = None,
    proxy_root_path: str = "",
):
    options = {"command": "serve"}
    options = options | ({"project_dir": project_dir} if project_dir else {})
    if index_type:
        options.update({"index_type": index_type})

    config = APIConfig.model_validate(options)  # type: ignore

    app = create_app(config, theme_asset_dir)
    uvicorn.run(
        app,
        host=config.listen_address,
        port=config.port,
        log_level="info",
        root_path=proxy_root_path,
    )
