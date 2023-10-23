import logging

from typing import Optional, Callable
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import APIConfig
from .routes import get_project_router

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

def create_app(config: APIConfig, theme_asset_dir: Path, callback: Callable = None):
    app = FastAPI()
    app.state.config = config

    @app.on_event("startup")
    async def startup():
        app.include_router(get_project_router(config))
        logger.info(f"Loading html user interface from {theme_asset_dir}")
        app.mount(
            f"/{config.project_id}/",
            StaticFiles(directory=theme_asset_dir, html=True),
            name="assets",
        )
        log_custom_format(f'Open http://{config.hostname}:{config.port}/{config.project_id}/ in your browser')
        if callback:
            callback()

    @app.on_event("shutdown")
    async def shutdown():
        pass

    return app


def serve(
    project_id: str,
    theme_asset_dir: Path,
    index_type: Optional[str] = None,
    query_blocklist_file: Path = None,
    callback: Callable = None # You can pass in a callback function to be called when the server has started
):
    options = {"project_id": project_id} if project_id else {}
    if index_type:
        options.update({"index_type": index_type})
    if query_blocklist_file:
        query_blocklist = []
        with open(query_blocklist_file, 'r') as f:
            for line in f:
                term = line.strip()
                if term:
                    query_blocklist.append(term)
        options.update({"query_blocklist": set(query_blocklist)})

    config = APIConfig.model_validate(options)  # type: ignore

    app = create_app(config, theme_asset_dir, callback)
    uvicorn.run(app, host=config.hostname, port=config.port, log_level="info")
