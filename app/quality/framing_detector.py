"""
app/quality/framing_detector.py

Heuristic-based detector for poor image composition/framing, using a
rule-of-thirds-style grid over Canny edge output.

Design note (see Decisions.md):
This detector is included as an 8th INPUT FEATURE for the Suitability
Model. It does NOT contribute to ground-truth label generation, because
there is no reliable way to synthetically inject a "poor framing" defect
with a known severity level -- VisionSeek/Flickr30k images carry no
subject-location metadata to inject against. The model will learn to
weight this feature on its own (possibly near-zero if it turns out to
be noisy).

Two signals are combined, mirroring how MotionDetector combines
anisotropy + weak-direction energy rather than relying on a single
number:
  1. center_occupancy     -- fraction of total edge energy sitting in
                              the center grid cell. Low value suggests
                              no clear subject near the frame center.
  2. border_concentration -- fraction of total edge energy sitting in
                              the outer ring of grid cells. High value
                              suggests the subject is crowding or
                              exiting the frame (a classic "cut off"
                              framing defect).

compute_score() reports border_concentration as the primary score (the
more diagnostic of the two for a single-number summary); has_poor_framing()
flags True if EITHER signal crosses its threshold.

No training or labeled data required -- deterministic, explainable formula.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FramingDetector:
    """
    Divides the image into a rule-of-thirds-style grid and flags poor
    framing if the center cell has too little edge energy (no clear
    central subject) or the outer ring has too much (subject crowding/
    exiting the frame).
    """

    def __init__(
        self,
        grid_size: int = 3,
        center_occupancy_threshold: float = 0.05,
        border_concentration_threshold: float = 0.6,
        canny_low: int = 50,
        canny_high: int = 150,
    ):
        self.grid_size = grid_size
        self.center_occupancy_threshold = center_occupancy_threshold
        self.border_concentration_threshold = border_concentration_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high

    def _grid_edge_fractions(self, image: np.ndarray) -> tuple[float, float, bool]:
        """
        Returns (center_occupancy, border_concentration, has_edges).
        has_edges is False if the image has no detectable edges at all
        (or is too small to grid) -- in that case center_occupancy and
        border_concentration are both 0.0 but should NOT be interpreted
        as "no subject in center," since there's nothing to judge.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        height, width = edges.shape
        cell_h = height // self.grid_size
        cell_w = width // self.grid_size

        total_edge_pixels = np.count_nonzero(edges)
        if total_edge_pixels == 0 or cell_h == 0 or cell_w == 0:
            # No detectable edges (e.g. a heavily blurred/darkened image
            # has destroyed edge structure) -- this is NOT evidence of
            # good framing. Returning 0.0 here previously caused a
            # confound: severely degraded (Not Suitable) images had no
            # edges, scored as "good framing" (0.0 -> normalizes to
            # best-case), producing a spurious positive correlation
            # between poor framing score and the Suitable label (see
            # Decisions.md). Returning the detector's own threshold
            # instead makes FeatureExtractor normalize this to a NEUTRAL
            # 0.5 -- "we genuinely can't judge framing here" rather than
            # "framing is good."
            return 0.0, self.border_concentration_threshold, False

        center_edge_pixels = 0
        border_edge_pixels = 0
        center_idx = self.grid_size // 2

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y0 = row * cell_h
                y1 = (row + 1) * cell_h if row < self.grid_size - 1 else height
                x0 = col * cell_w
                x1 = (col + 1) * cell_w if col < self.grid_size - 1 else width

                cell_edges = np.count_nonzero(edges[y0:y1, x0:x1])

                if row == center_idx and col == center_idx:
                    center_edge_pixels += cell_edges
                if row == 0 or row == self.grid_size - 1 or col == 0 or col == self.grid_size - 1:
                    border_edge_pixels += cell_edges

        center_occupancy = center_edge_pixels / total_edge_pixels
        border_concentration = border_edge_pixels / total_edge_pixels

        return center_occupancy, border_concentration, True

    def compute_score(self, image: np.ndarray) -> float:
        """
        Returns border_concentration (0.0-1.0) as the primary single-
        number score. Higher = more edge energy concentrated at the
        frame's outer ring, suggesting the subject is crowding/exiting
        the frame.
        """
        _, border_concentration, _ = self._grid_edge_fractions(image)
        return border_concentration

    def has_poor_framing(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Returns (has_poor_framing, score). Flags True when EITHER:
        - center_occupancy falls below self.center_occupancy_threshold
          (no clear subject centrally), OR
        - border_concentration exceeds self.border_concentration_threshold
          (subject crowding/exiting the frame).
        Never flags an image with no detectable edges at all -- that's a
        job for other detectors (Blur/Darkness), not this one.
        score is border_concentration (see compute_score).
        """
        center_occupancy, border_concentration, has_edges = self._grid_edge_fractions(image)

        if not has_edges:
            return False, border_concentration

        is_center_empty = center_occupancy < self.center_occupancy_threshold
        is_border_heavy = border_concentration > self.border_concentration_threshold

        return bool(is_center_empty or is_border_heavy), border_concentration