"""
app/routers/quality.py

Endpoint for uploading an image and getting a quality assessment back.
Unlike VisionSeek's /search (which queries a pre-built index), this
endpoint processes whatever image the caller uploads — there's no
underlying dataset here.
"""

import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.quality.assessor import QualityAssessor
from app.schemas.quality import CheckResult, QualityAssessmentResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Built once at import time — the assessor holds no state per-request,
# so a single shared instance is safe and avoids re-creating three
# detector objects on every call.
assessor = QualityAssessor()


@router.post("/assess", response_model=QualityAssessmentResponse)
async def assess_quality(file: UploadFile = File(...)) -> QualityAssessmentResponse:
    contents = await file.read()

    # Decode the uploaded bytes into an OpenCV image without saving to
    # disk first — np.frombuffer + cv2.imdecode reads directly from memory.
    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image — unsupported or corrupt file")

    try:
        result = assessor.assess(img)
    except Exception as e:
        logger.error(f"Quality assessment failed for '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail="Quality assessment failed") from e

    return QualityAssessmentResponse(
        filename=file.filename,
        passed=result["passed"],
        blur=CheckResult(flagged=result["blur"]["is_blurry"], score=result["blur"]["score"]),
        darkness=CheckResult(flagged=result["darkness"]["is_dark"], score=result["darkness"]["score"]),
        glare=CheckResult(flagged=result["glare"]["has_glare"], score=result["glare"]["score"]),
    )
