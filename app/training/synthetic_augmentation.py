"""
app/training/synthetic_augmentation.py

Synthetic Augmentation Engine for the Suitability Model's training data.

GOLDEN RULE (enforced by construction): this module only ever consults
severity_table.py and degradation_engine.py -- it NEVER calls any
detector from app/quality/. The ground-truth label is decided purely
from what was injected (see ground_truth_generator.py), not from what
any detector measures. This keeps labels and features statistically
independent, avoiding circular reasoning / label leakage.
"""

import random

from app.pilot.degradation_engine import DEGRADATION_LADDERS
from app.training.severity_table import INJECTABLE_DEFECTS, SEVERITY_RANGES, SEVERITY_TIERS


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
        """
        num_defects = self._rng.randint(0, 3)
        chosen_defects = self._rng.sample(INJECTABLE_DEFECTS, k=num_defects)

        result = image.copy()
        injections = []

        for defect in chosen_defects:
            tier = self._rng.choice(SEVERITY_TIERS)
            low, high = SEVERITY_RANGES[defect][tier]
            param_value = self._rng.uniform(low, high)

            apply_fn, _ladder = DEGRADATION_LADDERS[defect]
            result = apply_fn(result, param_value)

            injections.append({
                "defect": defect,
                "severity_tier": tier,
                "param_value": param_value,
            })

        return result, injections
