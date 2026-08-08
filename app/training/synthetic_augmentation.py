"""
app/training/synthetic_augmentation.py

Synthetic Augmentation Engine for the Suitability Model's training data.

GOLDEN RULE (enforced by construction): this module only ever consults
severity_table.py and degradation_engine.py -- it NEVER calls any
detector from app/quality/. The ground-truth label is decided purely
from what was injected (see ground_truth_generator.py), not from what
any detector measures. This keeps labels and features statistically
independent, avoiding circular reasoning / label leakage.

RESOLUTION SPECIAL CASE (see Decisions.md for full writeup): the shared
degradation_engine.apply_low_resolution() downscales THEN upscales back
to the original dimensions -- correct for the pilot script's side-by-side
visual comparisons, but it means ResolutionDetector (which only measures
width x height) can never detect it, since dimensions are always
restored. For TRAINING DATA generation specifically, resolution is
applied here as a genuine downscale with NO upscale-back, so the
resulting image is actually smaller and ResolutionDetector can measure
the real degradation. degradation_engine.py itself is left unchanged
(still used as-is by the pilot script).
"""

import random

import cv2

from app.pilot.degradation_engine import DEGRADATION_LADDERS
from app.training.severity_table import INJECTABLE_DEFECTS, SEVERITY_RANGES, SEVERITY_TIERS


def _apply_true_downscale(image, downscale_factor: float):
    """
    Downscales the image by downscale_factor and returns it AT THE
    SMALLER SIZE (no upscale back) -- unlike
    degradation_engine.apply_low_resolution(), which restores original
    dimensions. Used only here, for training-data generation, so
    ResolutionDetector's width x height check can actually see the
    degradation.
    """
    downscale_factor = max(0.05, min(1.0, downscale_factor))
    height, width = image.shape[:2]
    small_h = max(1, int(height * downscale_factor))
    small_w = max(1, int(width * downscale_factor))
    return cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)


class SyntheticAugmentationEngine:
    """
    Applies 0-3 randomly chosen defects (from the 7 injectable types), at
    a randomly chosen severity tier each, to a clean image.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def augment(self, image):
        """
        Returns (degraded_image, injections). injections is a list of
        dicts: {"defect": str, "severity_tier": str, "param_value": float}
        -- one entry per defect actually applied. An empty list means the
        image was left clean (a valid, intentional outcome so the
        training set includes genuinely clean examples).

        NOTE: if "resolution" is among the chosen defects, the returned
        image will have SMALLER dimensions than the input (see module
        docstring) -- callers must not assume output shape matches input
        shape when resolution degradation may have been applied.
        """
        num_defects = self._rng.randint(0, 3)
        chosen_defects = self._rng.sample(INJECTABLE_DEFECTS, k=num_defects)

        result = image.copy()
        injections = []

        for defect in chosen_defects:
            tier = self._rng.choice(SEVERITY_TIERS)
            low, high = SEVERITY_RANGES[defect][tier]
            param_value = self._rng.uniform(low, high)

            if defect == "resolution":
                result = _apply_true_downscale(result, param_value)
            else:
                apply_fn, _ladder = DEGRADATION_LADDERS[defect]
                result = apply_fn(result, param_value)

            injections.append({
                "defect": defect,
                "severity_tier": tier,
                "param_value": param_value,
            })

        return result, injections
