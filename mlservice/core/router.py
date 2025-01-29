from fastapi import APIRouter
from .upload_routes import router as upload_router

router = APIRouter()
router.include_router(upload_router)