from fastapi import APIRouter

from .safety_index import router as safety_index_router
from .speed_distribution import router as speed_distribution_router
from .top_speeding import router as top_speeding_router

router = APIRouter(prefix="/analytics")
router.include_router(speed_distribution_router)
router.include_router(top_speeding_router)
router.include_router(safety_index_router)
