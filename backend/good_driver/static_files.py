from starlette.requests import Request
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles

from .config import get_frontend_dist_path


class SPAStaticFiles(StaticFiles):
    """Serve static files with SPA fallback to index.html."""

    async def get_response(self, path: str, scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except Exception:
            return await super().get_response("index.html", scope)


def mount_frontend(app):
    dist = get_frontend_dist_path()
    if dist.is_dir() and (dist / "index.html").exists():
        app.mount("/", SPAStaticFiles(directory=str(dist), html=True), name="spa")
