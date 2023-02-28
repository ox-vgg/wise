from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import APIConfig
from routes import get_search_router


def create_app(config: APIConfig):
    app = FastAPI()
    app.state.config = config
    # TODO Mount dataset source directories as static
    # and return URL accordingly
    app.mount("/public", StaticFiles(directory="public"), name="public")

    @app.on_event("startup")
    async def startup():
        app.include_router(get_search_router(config))

    @app.on_event("shutdown")
    async def shutdown():
        pass

    return app


def main(project_id: Optional[str] = None):
    config = APIConfig(project_id=project_id)
    app = create_app(config)
    uvicorn.run(app, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
