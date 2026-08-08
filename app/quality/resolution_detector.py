"""
app/quality/resolution_detector.py

Flags images with insufficient pixel dimensions for reliable downstream
computer-vision processing. Purely a metadata check — no pixel content
is analyzed, only width x height.

No training or labeled data required — deterministic threshold on a
directly-measurable property.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ResolutionDetector:
    """
    Flags an image as low-resolution if its total pixel count (width x
    height) falls below a configurable threshold.
    """

    def __init__(self, min_pixels: int = 256 * 256):
        self.min_pixels = min_pixels

    def compute_score(self, image: np.ndarray) -> int:
        """
        Returns total pixel count (width x height). Higher = more
        resolution.
        """
        height, width = image.shape[:2]
        return width * height

    def is_low_resolution(self, image: np.ndarray) -> tuple[bool, int]:
        """
        Returns (is_low_resolution, score). is_low_resolution is True
        when total pixel count falls below self.min_pixels.
        """
        score = self.compute_score(image)
        return score < self.min_pixels, score
