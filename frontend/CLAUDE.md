# Frontend Debug

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Node.js with npm


## Start dev servers

If `uv` is not in PATH, find it:
```
which uv || ls ~/.local/bin/uv ~/.cargo/bin/uv ~/snap/*/current/.local/bin/uv 2>/dev/null
```
Then `export PATH="<dir>:$PATH"` for the session.

Run from the repo root:
```
make dev-backend   # FastAPI on :8000
make dev-frontend  # Vite on :5173
```

Then open Chrome at http://localhost:5173/
