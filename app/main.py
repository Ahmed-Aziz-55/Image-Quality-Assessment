"""
app/main.py

FastAPI entry point for the Image Quality Assessment API.

Run with:
    uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI

from app.core.logging_config import setup_logging
from app.routers.quality import router as quality_router

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Quality Assessment API", version="1.0.0")
app.include_router(quality_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
