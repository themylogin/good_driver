.PHONY: install dev-backend dev-frontend build-frontend build-desktop run-desktop test download-model

install:
	cd backend && uv sync
	cd frontend && npm install

dev-backend:
	cd backend && uv run uvicorn good_driver.app:create_app --factory --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

build-frontend:
	cd frontend && npm run build
	rm -rf backend/frontend_dist/*
	cp -r frontend/dist/* backend/frontend_dist/

build-desktop: build-frontend
	cd backend && uv run pyinstaller ../desktop/good_driver.spec --distpath ../dist --workpath ../build --noconfirm

run-desktop: build-frontend
	cd backend && uv run python ../desktop/launcher.py

test:
	cd backend && uv run pytest

download-model:
	cd backend && uv run python -m good_driver.model
