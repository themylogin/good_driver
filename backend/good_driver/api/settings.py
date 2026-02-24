import json
from pathlib import Path

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/settings")

SETTINGS_FILENAME = "settings.json"
DEFAULT_SETTINGS = {
    "osrm_url": "http://localhost:5000",
    "nominatim_url": "http://localhost:8080",
}


class Settings(BaseModel):
    osrm_url: str = "http://localhost:5000"
    nominatim_url: str = "http://localhost:8080"


class CheckRequest(BaseModel):
    url: str
    service: str  # "osrm" or "nominatim"


@router.get("")
async def get_settings(directory: str):
    path = Path(directory) / SETTINGS_FILENAME
    if path.exists():
        data = json.loads(path.read_text())
        return {**DEFAULT_SETTINGS, **data}
    return DEFAULT_SETTINGS


@router.put("")
async def save_settings(directory: str, settings: Settings):
    path = Path(directory) / SETTINGS_FILENAME
    path.write_text(json.dumps(settings.model_dump(), indent=2))
    return {"ok": True}


@router.post("/check")
async def check_url(req: CheckRequest):
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            if req.service == "nominatim":
                url = req.url.rstrip("/") + "/status"
                resp = await client.get(url)
                if resp.status_code == 200:
                    # Plain-text "OK" response
                    if resp.text.strip() == "OK":
                        return {"ok": True, "message": "OK"}
                    # JSON response (some Nominatim versions)
                    try:
                        data = resp.json()
                        if data.get("status") == 0:
                            return {"ok": True, "message": data.get("message", "OK")}
                        return {"ok": False, "message": data.get("message", "Unknown error")}
                    except Exception:
                        return {"ok": True, "message": resp.text.strip()}
                return {"ok": False, "message": f"HTTP {resp.status_code}"}
            elif req.service == "osrm":
                url = req.url.rstrip("/") + "/nearest/v1/driving/0,0"
                resp = await client.get(url)
                if resp.status_code == 200:
                    return {"ok": True, "message": "OK"}
                return {"ok": False, "message": f"HTTP {resp.status_code}"}
            else:
                return {"ok": False, "message": "Unknown service type"}
        except httpx.ConnectError:
            return {"ok": False, "message": "Connection refused"}
        except httpx.TimeoutException:
            return {"ok": False, "message": "Timed out"}
        except Exception as e:
            return {"ok": False, "message": str(e)}
