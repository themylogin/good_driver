from fastapi import APIRouter

from good_driver.config import get_mode

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "mode": get_mode().value}


@router.get("/hardware")
async def hardware():
    from good_driver.api.calibrate import get_hardware_info

    return get_hardware_info()
