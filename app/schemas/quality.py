"""
app/schemas/quality.py

Response schema for the quality assessment API. Mirrors the dict shape
returned by QualityAssessor.assess().
"""

from pydantic import BaseModel


class CheckResult(BaseModel):
    flagged: bool
    score: float


class SuitabilityResult(BaseModel):
    label: str  # "Suitable" or "Not Suitable"
    confidence: float


class QualityAssessmentResponse(BaseModel):
    filename: str
    suitability: SuitabilityResult
    passed: bool
    blur: CheckResult
    darkness: CheckResult
    glare: CheckResult
    overexposure: CheckResult
    resolution: CheckResult
    motion: CheckResult
    occlusion: CheckResult
    framing: CheckResult
