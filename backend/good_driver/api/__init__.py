from fastapi import APIRouter

from .calibrate import router as calibrate_router
from .footage import router as footage_router
from .health import router as health_router
from .model import router as model_router
from .settings import router as settings_router

router = APIRouter(prefix="/api")
router.include_router(health_router)
router.include_router(settings_router)
router.include_router(calibrate_router)
router.include_router(footage_router)
router.include_router(model_router)
