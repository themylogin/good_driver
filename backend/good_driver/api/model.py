"""API endpoints for model file management."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from good_driver.model import MODEL_NAME, RESOURCES_URL, extract_model, get_model_path

router = APIRouter(prefix="/model")


@router.get("/status")
async def model_status():
    path = get_model_path()
    return {"exists": path.exists(), "name": MODEL_NAME}


@router.post("/download")
async def download_model():
    """Stream model download progress as Server-Sent Events.

    Events:
      {"downloaded": <bytes>, "total": <bytes>}   — download progress
      {"status": "extracting"}                     — extraction started
      {"status": "done"}                           — finished successfully
      {"error": "<message>"}                       — something went wrong
    """
    async def generate():
        import httpx

        target = get_model_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        archive = target.parent / "resources.tar.gz"

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                async with client.stream("GET", RESOURCES_URL) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    with archive.open("wb") as f:
                        async for chunk in response.aiter_bytes(65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            yield f"data: {json.dumps({'downloaded': downloaded, 'total': total})}\n\n"

            yield f"data: {json.dumps({'status': 'extracting'})}\n\n"
            await asyncio.to_thread(extract_model, archive, target)
            archive.unlink(missing_ok=True)
            yield f"data: {json.dumps({'status': 'done'})}\n\n"

        except Exception as e:
            archive.unlink(missing_ok=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
