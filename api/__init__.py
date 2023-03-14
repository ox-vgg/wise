from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import APIConfig
from .routes import get_project_router

from pathlib import Path

def create_app(config: APIConfig, theme_asset_dir, index_type):
    app = FastAPI()
    app.state.config = config

    @app.on_event("startup")
    async def startup():
        app.include_router(get_project_router(config, index_type))
        print('Loading html user interface from %s' % (str(theme_asset_dir)))
        app.mount(
            f"/{config.project_id}/",
            StaticFiles(directory=theme_asset_dir, html=True),
            name="assets",
        )

    @app.on_event("shutdown")
    async def shutdown():
        pass

    return app


def serve(project_id: str, theme_asset_dir: Path, index_type: str):
    options = {"project_id": project_id} if project_id else {}
    config = APIConfig.parse_obj(options)  # type: ignore

    app = create_app(config, theme_asset_dir, index_type)
    uvicorn.run(app, port=config.port, log_level="info")
