# Good Driver

Cross-platform desktop application with a web UI frontend.

- **Backend**: Python / FastAPI (async)
- **Frontend**: React / TypeScript / Vite
- **Desktop**: pywebview + PyInstaller

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node.js 20+
- **Linux**: GTK WebKit (`gir1.2-webkit2-4.1` package, usually pre-installed on Ubuntu/Fedora)

## Setup

```bash
make install
```

This installs both Python dependencies (via `uv sync`) and Node.js dependencies (via `npm install`).

## Development

Run backend and frontend in separate terminals:

```bash
# Terminal 1 — API server with hot reload (port 8000)
make dev-backend

# Terminal 2 — Vite dev server with HMR (port 5173)
make dev-frontend
```

Open http://localhost:5173 in your browser. API calls (`/api/*`) are proxied to the backend automatically via Vite's dev server proxy.

## Testing

```bash
make test
```

Tests use `pytest` with `pytest-asyncio` (auto mode) and `httpx` for async ASGI transport.

## Desktop App

Run locally (builds frontend, starts pywebview window):

```bash
make run-desktop
```

Build a standalone binary:

```bash
make build-desktop
```

The binary is output to `dist/good_driver`.

## All Make Commands

| Command | Description |
|---|---|
| `make install` | Install Python and Node.js dependencies |
| `make dev-backend` | Run FastAPI with hot reload on port 8000 |
| `make dev-frontend` | Run Vite dev server with HMR on port 5173 |
| `make test` | Run pytest |
| `make build-frontend` | Build React app and copy to `backend/frontend_dist/` |
| `make run-desktop` | Build frontend + launch pywebview window |
| `make build-desktop` | Build a standalone binary to `dist/good_driver` |

## Project Structure

```
backend/                  Python backend (FastAPI)
  pyproject.toml          Dependencies and project config (uv)
  good_driver/
    app.py                FastAPI application factory
    config.py             Mode detection and path helpers
    static_files.py       SPA static file serving with index.html fallback
    api/
      __init__.py         APIRouter aggregating all route modules
      health.py           GET /api/health — starter endpoint
  frontend_dist/          Built frontend assets (populated by make build-frontend)
  tests/
    test_health.py        Async API tests using httpx ASGI transport

frontend/                 React frontend (Vite + TypeScript)
  vite.config.ts          Dev proxy /api → localhost:8000, React plugin
  src/
    App.tsx               Root component
    main.tsx              Entry point

desktop/
  launcher.py             pywebview + uvicorn launcher
  good_driver.spec        PyInstaller one-file build spec
```

## Architecture

### Runtime Modes

The app runs in three modes, detected by `backend/good_driver/config.py`:

| Mode | How it's activated | Behavior |
|---|---|---|
| **dev** | Default (no env var) | Backend serves API only. Frontend runs on Vite dev server (port 5173) with proxy to backend. CORS enabled for `localhost:5173`. |
| **desktop** | `GOOD_DRIVER_MODE=desktop` or PyInstaller frozen binary | Backend serves API + built frontend static files. pywebview opens a native OS window. Port is dynamically assigned. |
| **production** | `GOOD_DRIVER_MODE=production` | Backend serves API + built frontend static files as a regular web app (future SaaS deployment). |

### How the Desktop Launcher Works

1. Finds a free port by binding to port 0
2. Starts uvicorn (FastAPI) in a daemon thread on that port
3. Polls `GET /api/health` until the server is ready
4. Opens a pywebview native window pointing to `http://127.0.0.1:{port}`
5. When the window is closed, the process exits (daemon thread dies automatically)

### How Frontend Assets Are Bundled

- `make build-frontend` runs `vite build` and copies output from `frontend/dist/` to `backend/frontend_dist/`
- In desktop/production mode, FastAPI mounts `frontend_dist/` as static files with SPA fallback (all non-API routes serve `index.html`)
- PyInstaller bundles `backend/frontend_dist/` into the binary via the `datas` option in `good_driver.spec`; at runtime, assets are extracted to a temp directory (`sys._MEIPASS`)

### Adding New API Endpoints

1. Create a new route module in `backend/good_driver/api/` (e.g., `users.py`)
2. Register its router in `backend/good_driver/api/__init__.py`
3. All routes are prefixed under `/api/`

### Linux: Snap Environment Note

If you run the desktop app from a terminal inside a snap-based IDE (e.g., VS Code installed via snap), WebKit may crash due to library conflicts. This does not affect:
- Running from a regular terminal
- The built binary
- End users

As a workaround, launch from a clean environment:
```bash
env -i HOME="$HOME" PATH="$PATH" DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
  DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
  make run-desktop
```
