"""
app/quality/assessor.py

Combines all 8 quality detectors (via FeatureExtractor) with the trained
SuitabilityModel into a single entry point, returning both the trained
Suitable/Not Suitable verdict and each individual detector's raw result.

Design note: this class does NOT expose per-detector threshold overrides
(unlike an earlier version). SuitabilityModel was trained on features
produced by FeatureExtractor's fixed, default detector configuration --
allowing different thresholds here would create a train/serve mismatch
(the model would see features computed differently from what it learned
on). To retune a detector's threshold, change it in FeatureExtractor,
then regenerate the training dataset and retrain the model.
"""

import logging

import cv2

from app.quality.feature_extractor import FeatureExtractor, FEATURE_NAMES
from app.quality.suitability_model import SuitabilityModel

logger = logging.getLogger(__name__)


class QualityAssessor:
    """
    Runs all 8 quality checks (via FeatureExtractor) and the trained
    SuitabilityModel against an image, returning a unified result.
    """

    def __init__(self, model_path: str = "models/suitability_logreg.joblib"):
        self.extractor = FeatureExtractor()
        self.suitability_model = SuitabilityModel(model_path=model_path)

    def assess_path(self, image_path: str) -> dict:
        """
        Loads an image from disk and runs all quality checks. Returns a
        dict with a top-level trained 'suitability' verdict, a legacy
        'passed' boolean (mirrors suitability.label), and per-detector
        raw results.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {
                "image_path": image_path,
                "loaded": False,
                "passed": False,
                "reason": "failed to load image",
            }

        return self.assess(img, image_path=image_path)

    def assess(self, img, image_path: str | None = None) -> dict:
        """
        Runs all 8 quality checks + the trained SuitabilityModel against
        an already-loaded image (as a numpy array, e.g. from cv2.imread
        or an uploaded file decoded with cv2.imdecode).
        """
        extraction = self.extractor.extract(img)
        raw = extraction["_raw"]  # {name: (flag, raw_score)}
        feature_vector = [extraction["features"][name] for name in FEATURE_NAMES]

        label, confidence = self.suitability_model.predict(feature_vector)

        is_blurry, blur_score = raw["blur"]
        is_dark, dark_score = raw["darkness"]
        has_glare, glare_score = raw["glare"]
        is_overexposed, overexposure_score = raw["overexposure"]
        has_motion_blur, motion_score = raw["motion"]
        has_occlusion, occlusion_score = raw["occlusion"]
        is_low_resolution, resolution_score = raw["resolution"]
        is_poorly_framed, framing_score = raw["framing"]

        return {
            "image_path": image_path,
            "loaded": True,
            "suitability": {"label": label, "confidence": round(confidence, 4)},
            "passed": label == "Suitable",
            "blur": {"is_blurry": is_blurry, "score": round(blur_score, 2)},
            "darkness": {"is_dark": is_dark, "score": round(dark_score, 2)},
            "glare": {"has_glare": has_glare, "score": round(glare_score, 4)},
            "overexposure": {
                "is_overexposed": is_overexposed,
                "score": round(overexposure_score, 2),
            },
            "resolution": {
                "is_low_resolution": is_low_resolution,
                "score": resolution_score,
            },
            "motion": {"has_motion_blur": has_motion_blur, "score": round(motion_score, 2)},
            "occlusion": {"has_occlusion": has_occlusion, "score": round(occlusion_score, 4)},
            "framing": {"is_poorly_framed": is_poorly_framed, "score": round(float(framing_score), 4)},
        }
