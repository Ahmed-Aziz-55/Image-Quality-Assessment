"""
app/quality/glare_detector.py

Detects glare (overexposed, blown-out highlights) by measuring what
fraction of an image's pixels are near-white (saturated). Unlike overall
brightness, glare is usually localized to a small region (e.g. a flash
reflection), so a whole-image average would miss it — counting the
proportion of extreme-bright pixels catches localized overexposure that
an average would dilute away.

No training or labeled data required — deterministic, explainable formula.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GlareDetector:
    """
    Flags an image as having glare if the fraction of near-white pixels
    exceeds a configurable threshold.
    """

    def __init__(self, brightness_cutoff: int = 240, area_threshold: float = 0.05):
        self.brightness_cutoff = brightness_cutoff
        self.area_threshold = area_threshold

    def compute_score(self, image: np.ndarray) -> float:
        """
        Returns the fraction (0.0-1.0) of pixels at or above
        brightness_cutoff. Higher = more glare.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        bright_pixels = np.sum(gray >= self.brightness_cutoff)
        total_pixels = gray.size
        return float(bright_pixels / total_pixels)

    def has_glare(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Returns (has_glare, score). has_glare is True when the fraction
        of near-white pixels exceeds self.area_threshold.
        """
        score = self.compute_score(image)
        return score > self.area_threshold, score
