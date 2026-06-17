from fastapi import APIRouter

from .knowledge import router as knowledge_router
from .chat import router as chat_router
from .plans import router as plans_router
from .quizzes import router as quizzes_router
from .progress import router as progress_router

router = APIRouter()
router.include_router(knowledge_router, prefix="/knowledge", tags=["knowledge"])
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(plans_router, prefix="/plans", tags=["plans"])
router.include_router(quizzes_router, prefix="/quizzes", tags=["quizzes"])
router.include_router(progress_router, prefix="/progress", tags=["progress"])
