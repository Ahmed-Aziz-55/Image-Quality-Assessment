"""
app/training/ground_truth_generator.py

Applies the agreed combination rule to SyntheticAugmentationEngine's
injection metadata to produce a ground-truth Suitable/Not Suitable
label. This is the ONLY place the label is decided, and it looks ONLY
at injection metadata (never at detector output) -- see the golden
rule in synthetic_augmentation.py.

Combination rule (see project Decisions.md):
    "Not Suitable" if (any injected defect at High severity)
                    OR (2 or more injected defects at Medium severity)
    else "Suitable"

Note this rule only ever counts injected (synthetic) defects. Poor
Framing is never injected and therefore never affects this label --
consistent with the design decision that Framing is a feature-only
signal for the Suitability Model, not a labeling input.
"""

LABEL_SUITABLE = "Suitable"
LABEL_NOT_SUITABLE = "Not Suitable"


def generate_label(injections: list[dict]) -> str:
    """
    injections: the list returned by SyntheticAugmentationEngine.augment(),
    e.g. [{"defect": "blur", "severity_tier": "high", "param_value": 6.2}, ...]

    Returns LABEL_SUITABLE or LABEL_NOT_SUITABLE per the combination rule.
    """
    tiers = [injection["severity_tier"] for injection in injections]

    has_high = "high" in tiers
    medium_count = tiers.count("medium")

    if has_high or medium_count >= 2:
        return LABEL_NOT_SUITABLE
    return LABEL_SUITABLE