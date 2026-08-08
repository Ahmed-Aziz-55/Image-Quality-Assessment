"""
app/routers/quality.py

Endpoint for uploading an image and getting a quality assessment back,
including the trained Suitability Model's Suitable/Not Suitable verdict.
"""

import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.quality.assessor import QualityAssessor
from app.schemas.quality import CheckResult, QualityAssessmentResponse, SuitabilityResult

logger = logging.getLogger(__name__)
router = APIRouter()

# Built once at import time -- loads all 8 detectors + the trained
# SuitabilityModel. Holds no per-request state, so a single shared
# instance is safe and avoids reloading the model on every call.
assessor = QualityAssessor()


@router.post("/assess", response_model=QualityAssessmentResponse)
async def assess_quality(file: UploadFile = File(...)) -> QualityAssessmentResponse:
    contents = await file.read()

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
        suitability=SuitabilityResult(
            label=result["suitability"]["label"],
            confidence=result["suitability"]["confidence"],
        ),
        passed=result["passed"],
        blur=CheckResult(flagged=result["blur"]["is_blurry"], score=result["blur"]["score"]),
        darkness=CheckResult(flagged=result["darkness"]["is_dark"], score=result["darkness"]["score"]),
        glare=CheckResult(flagged=result["glare"]["has_glare"], score=result["glare"]["score"]),
        overexposure=CheckResult(
            flagged=result["overexposure"]["is_overexposed"], score=result["overexposure"]["score"]
        ),
        resolution=CheckResult(
            flagged=result["resolution"]["is_low_resolution"], score=float(result["resolution"]["score"])
        ),
        motion=CheckResult(flagged=result["motion"]["has_motion_blur"], score=result["motion"]["score"]),
        occlusion=CheckResult(flagged=result["occlusion"]["has_occlusion"], score=result["occlusion"]["score"]),
        framing=CheckResult(flagged=result["framing"]["is_poorly_framed"], score=result["framing"]["score"]),
    )
