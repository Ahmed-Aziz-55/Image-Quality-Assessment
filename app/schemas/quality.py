"""
app/schemas/quality.py

Response schema for the quality assessment API. Mirrors the dict shape
returned by QualityAssessor.assess(), but as an explicit Pydantic model
so the API contract is documented and validated.
"""

from pydantic import BaseModel


class CheckResult(BaseModel):
    flagged: bool
    score: float


class QualityAssessmentResponse(BaseModel):
    filename: str
    passed: bool
    blur: CheckResult
    darkness: CheckResult
    glare: CheckResult
