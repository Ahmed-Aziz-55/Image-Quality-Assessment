"""
app/quality/darkness_detector.py

Detects underexposed (too dark) images by measuring average brightness.
Converts to grayscale (0=black, 255=white) and takes the mean pixel
value — images with a low mean are dark/underexposed.

No training or labeled data required — deterministic, explainable formula.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DarknessDetector:
    """
    Flags an image as too dark if its mean grayscale brightness falls
    below a configurable threshold (0-255 scale).
    """

    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold

    def compute_score(self, image: np.ndarray) -> float:
        """
        Returns mean grayscale brightness (0-255). Lower = darker.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        return float(gray.mean())

    def is_dark(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Returns (is_dark, score). is_dark is True when score falls
        below self.threshold.
        """
        score = self.compute_score(image)
        return score < self.threshold, score
