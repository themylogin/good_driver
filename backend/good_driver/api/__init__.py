from fastapi import APIRouter

from .calibrate import router as calibrate_router
from .health import router as health_router

router = APIRouter(prefix="/api")
router.include_router(health_router)
router.include_router(calibrate_router)
