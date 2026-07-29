"""
app/quality/assessor.py

Combines BlurDetector, DarknessDetector, and GlareDetector into a single
entry point. Loads an image once and runs all three checks against it,
returning a unified result.
"""

import logging

import cv2

from app.quality.blur_detector import BlurDetector
from app.quality.darkness_detector import DarknessDetector
from app.quality.glare_detector import GlareDetector

logger = logging.getLogger(__name__)


class QualityAssessor:
    """
    Runs blur, darkness, and glare detection against an image and reports
    a combined pass/fail verdict plus each individual score.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        darkness_threshold: float = 50.0,
        glare_brightness_cutoff: int = 250,
        glare_area_threshold: float = 0.08,
    ):
        self.blur_detector = BlurDetector(threshold=blur_threshold)
        self.darkness_detector = DarknessDetector(threshold=darkness_threshold)
        self.glare_detector = GlareDetector(
            brightness_cutoff=glare_brightness_cutoff,
            area_threshold=glare_area_threshold,
        )

    def assess_path(self, image_path: str) -> dict:
        """
        Loads an image from disk and runs all quality checks. Returns a
        dict with a top-level 'passed' verdict and per-check details.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {
                "image_path": image_path,
                "loaded": False,
                "passed": False,
                "reason": "failed to load image",
            }

        return self.assess(img, image_path=image_path)

    def assess(self, img, image_path: str | None = None) -> dict:
        """
        Runs all quality checks against an already-loaded image
        (as a numpy array, e.g. from cv2.imread or an uploaded file
        decoded with cv2.imdecode).
        """
        is_blurry, blur_score = self.blur_detector.is_blurry(img)
        is_dark, dark_score = self.darkness_detector.is_dark(img)
        has_glare, glare_score = self.glare_detector.has_glare(img)

        passed = not (is_blurry or is_dark or has_glare)

        return {
            "image_path": image_path,
            "loaded": True,
            "passed": passed,
            "blur": {"is_blurry": is_blurry, "score": round(blur_score, 2)},
            "darkness": {"is_dark": is_dark, "score": round(dark_score, 2)},
            "glare": {"has_glare": has_glare, "score": round(glare_score, 4)},
        }
