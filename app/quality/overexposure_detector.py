"""
app/quality/overexposure_detector.py

Detects overexposed (too bright, overall) images by measuring average
brightness — the mirror of DarknessDetector. A high mean grayscale value
indicates the whole image is overexposed.

This is intentionally different from GlareDetector: glare is a LOCALIZED
saturated region (e.g. a flash reflection covering a small area), while
overexposure here is an OVERALL exposure problem across the whole image.
GlareDetector counts the fraction of near-white pixels (catches small,
extreme bright spots); this detector uses whole-image mean brightness
(catches the image being uniformly too bright, even without any single
region being fully saturated).

No training or labeled data required — deterministic, explainable formula.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OverexposureDetector:
    """
    Flags an image as overexposed if its mean grayscale brightness
    exceeds a configurable threshold (0-255 scale).
    """

    def __init__(self, threshold: float = 220.0):
        self.threshold = threshold

    def compute_score(self, image: np.ndarray) -> float:
        """
        Returns mean grayscale brightness (0-255). Higher = brighter /
        more likely overexposed.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        return float(gray.mean())

    def is_overexposed(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Returns (is_overexposed, score). is_overexposed is True when
        score exceeds self.threshold.
        """
        score = self.compute_score(image)
        return score > self.threshold, score
