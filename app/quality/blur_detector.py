"""
app/quality/blur_detector.py

Detects blur in an image using the variance of the Laplacian — a classic,
widely-used no-reference blur metric. Sharp images have strong, well-defined
edges, so the Laplacian (an edge-detection operator) produces high-variance
output. Blurred images have soft, smeared edges, so the Laplacian output
has low variance.

No training or labeled data required — this is a deterministic, explainable
formula, not a learned model.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BlurDetector:
    """
    Flags an image as blurry if its Laplacian variance falls below a
    configurable threshold.
    """

    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold

    def compute_score(self, image: np.ndarray) -> float:
        """
        Returns the raw Laplacian variance for an image (BGR or grayscale
        numpy array, as loaded by OpenCV). Higher = sharper.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def is_blurry(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Returns (is_blurry, score). is_blurry is True when score falls
        below self.threshold.
        """
        score = self.compute_score(image)
        return score < self.threshold, score
