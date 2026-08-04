"""
app/quality/occlusion_detector.py

Best-effort heuristic occlusion detector: flags images where a large
contiguous region has both low color variance (uniform/flat) AND low
edge density (little texture/detail) — a rough proxy for "something is
blocking part of the frame" (e.g. a finger over the lens, an object
covering part of the subject).

IMPORTANT LIMITATION: no general-purpose labeled occlusion dataset was
available to validate this heuristic (see Decisions.md for the reasoning
behind not using a narrow, domain-mismatched dataset such as CMU_KO8,
which covers only texture-less kitchen objects under controlled
clutter). This detector's accuracy on arbitrary real-world photos is
UNVERIFIED and should be treated as a low-confidence signal, not an
authoritative detector. Legitimate low-texture content (clear sky, plain
walls, calm water, out-of-focus backgrounds) can trigger false positives.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OcclusionDetector:
    """
    Divides the image into a grid of blocks and flags occlusion if a
    large-enough fraction of blocks are both low-variance (flat color)
    and low-edge-density (little detail).
    """

    def __init__(
        self,
        grid_size: int = 12,
        variance_threshold: float = 50.0,
        edge_density_threshold: float = 0.02,
        block_fraction_threshold: float = 0.15,
    ):
        self.grid_size = grid_size
        self.variance_threshold = variance_threshold
        self.edge_density_threshold = edge_density_threshold
        self.block_fraction_threshold = block_fraction_threshold

    def compute_score(self, image: np.ndarray) -> float:
        """
        Returns the fraction (0.0-1.0) of grid blocks flagged as both
        flat-colored and low-detail — the suspected-occlusion area
        fraction. Higher = more of the image looks blocked.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape
        block_h = height // self.grid_size
        block_w = width // self.grid_size

        if block_h == 0 or block_w == 0:
            return 0.0  # image too small to grid meaningfully

        edges = cv2.Canny(gray, 100, 200)

        suspicious_blocks = 0
        total_blocks = 0

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y0, y1 = row * block_h, (row + 1) * block_h
                x0, x1 = col * block_w, (col + 1) * block_w

                block_gray = gray[y0:y1, x0:x1]
                block_edges = edges[y0:y1, x0:x1]

                variance = float(block_gray.var())
                edge_density = float(np.sum(block_edges > 0)) / block_edges.size

                total_blocks += 1
                if variance < self.variance_threshold and edge_density < self.edge_density_threshold:
                    suspicious_blocks += 1

        return suspicious_blocks / total_blocks if total_blocks > 0 else 0.0

    def has_occlusion(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Returns (has_occlusion, score). has_occlusion is True when the
        suspected-occlusion area fraction exceeds block_fraction_threshold.
        """
        score = self.compute_score(image)
        return score > self.block_fraction_threshold, score
