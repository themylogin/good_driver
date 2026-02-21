from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .config import Mode, get_mode
from .static_files import mount_frontend


def create_app() -> FastAPI:
    mode = get_mode()
    app = FastAPI(title="Good Driver")

    if mode == Mode.DEV:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(api_router)

    if mode != Mode.DEV:
        mount_frontend(app)

    return app
