"""
app/quality/motion_detector.py

Detects motion blur (directional blur from camera/subject movement,
distinct from out-of-focus blur) using Sobel gradient anisotropy.

Concept: motion blur smears detail in one specific direction, so gradient
energy in THAT direction drops sharply while the perpendicular direction
stays largely intact. Out-of-focus blur weakens gradients roughly equally
in all directions (isotropic) — both stay low together, or both stay
high together. Measuring both (a) the ratio between horizontal and
vertical gradient energy (anisotropy) AND (b) whether the weaker
direction's energy is genuinely low (not just relatively weaker) avoids
false positives on naturally directional but still-sharp content (e.g.
horizontal stripes in a genuinely sharp photo).

No training or labeled data required — deterministic, explainable formula.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Flags an image as having motion blur if its Sobel gradient energy is
    strongly anisotropic (imbalanced between horizontal and vertical
    directions) AND the weaker direction's own energy is low (indicating
    that direction is genuinely blurred, not just weaker by comparison).
    """

    def __init__(self, anisotropy_threshold: float = 3.0, weak_direction_energy_threshold: float = 2000.0):
        self.anisotropy_threshold = anisotropy_threshold
        self.weak_direction_energy_threshold = weak_direction_energy_threshold

    def _directional_energies(self, image: np.ndarray) -> tuple[float, float]:
        """
        Returns (mean_energy_x, mean_energy_y) — per-pixel average
        squared Sobel gradient in each direction. Using the mean (not
        sum) makes this comparable across different image sizes.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        energy_x = float(np.mean(sobel_x ** 2))
        energy_y = float(np.mean(sobel_y ** 2))

        return energy_x, energy_y

    def compute_score(self, image: np.ndarray) -> float:
        """
        Returns the anisotropy ratio: max(energy_x, energy_y) /
        min(energy_x, energy_y). Near 1.0 = balanced (isotropic). High =
        one direction much weaker than the other (directional blur
        signature).
        """
        energy_x, energy_y = self._directional_energies(image)

        if energy_x == 0 or energy_y == 0:
            return 1.0

        return max(energy_x, energy_y) / min(energy_x, energy_y)

    def has_motion_blur(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Returns (has_motion_blur, score). Flags True only when BOTH:
        - the anisotropy ratio exceeds the threshold (one direction much
          weaker than the other), AND
        - the weaker direction's own energy is genuinely low (it is
          actually blurred, not just weaker relative to a very sharp
          perpendicular direction).
        """
        energy_x, energy_y = self._directional_energies(image)
        score = self.compute_score(image)

        weaker_direction_energy = min(energy_x, energy_y)

        is_anisotropic = score > self.anisotropy_threshold
        is_weak_direction_blurred = weaker_direction_energy < self.weak_direction_energy_threshold

        return (is_anisotropic and is_weak_direction_blurred), score
