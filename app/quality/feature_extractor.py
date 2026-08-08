"""
app/quality/feature_extractor.py

Runs all 8 quality detectors against an image and returns a normalized
0-1 "severity" feature vector for the Suitability Model.

Each detector's raw score uses a different scale and direction (e.g.
Blur's Laplacian variance vs. Occlusion's 0-1 block fraction). This
module maps every raw score onto a common 0-1 scale via piecewise-linear
interpolation through three reference points:
  - worst-case anchor  -> 1.0 (as bad as it gets)
  - the detector's own configured pass/fail threshold -> 0.5 (so the
    midpoint always matches the detector's existing boolean flag)
  - best-case anchor   -> 0.0 (as good as it gets)

IMPORTANT CAVEAT: the best/worst anchors below are reasonable guesses,
NOT empirically measured against real images (unlike the thresholds
themselves, which were calibrated/tested in Decisions.md). They only
affect how "graded" a score feels past the threshold -- the 0.5 midpoint
is always exactly at the detector's real, tested threshold regardless.
"""

import logging

import numpy as np

from app.quality.blur_detector import BlurDetector
from app.quality.darkness_detector import DarknessDetector
from app.quality.glare_detector import GlareDetector
from app.quality.overexposure_detector import OverexposureDetector
from app.quality.motion_detector import MotionDetector
from app.quality.occlusion_detector import OcclusionDetector
from app.quality.resolution_detector import ResolutionDetector
from app.quality.framing_detector import FramingDetector

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "blur", "darkness", "glare", "overexposure",
    "motion", "occlusion", "resolution", "framing",
]


def _normalize(raw_score: float, threshold: float, best_anchor: float, worst_anchor: float) -> float:
    """
    Piecewise-linear map: worst_anchor -> 1.0, threshold -> 0.5,
    best_anchor -> 0.0. Works regardless of whether "higher raw score is
    worse" (worst_anchor > threshold) or "lower is worse" (worst_anchor
    < threshold) -- points are sorted before interpolation. Clamped to
    [0, 1] outside the anchor range.
    """
    xs = [worst_anchor, threshold, best_anchor]
    ys = [1.0, 0.5, 0.0]
    if xs[0] > xs[-1]:
        xs = xs[::-1]
        ys = ys[::-1]
    return float(np.clip(np.interp(raw_score, xs, ys), 0.0, 1.0))


class FeatureExtractor:
    """
    Runs all 8 detectors and returns a normalized 0-1 feature vector,
    in FEATURE_NAMES order.
    """

    def __init__(self):
        self.blur = BlurDetector()
        self.darkness = DarknessDetector()
        self.glare = GlareDetector()
        self.overexposure = OverexposureDetector()
        self.motion = MotionDetector()
        self.occlusion = OcclusionDetector()
        self.resolution = ResolutionDetector()
        self.framing = FramingDetector()

        # (threshold, best_anchor, worst_anchor) per detector.
        # threshold values match each detector's own tested default.
        # best/worst anchors are UNVALIDATED reference guesses (see
        # module docstring) -- one-line rationale per entry below.
        self._bounds = {
            # lower variance = blurrier; 300 = comfortably sharp, 0 = fully flat/blurred
            "blur": (self.blur.threshold, 300.0, 0.0),
            # lower brightness = darker; 150 = comfortably lit, 0 = pure black
            "darkness": (self.darkness.threshold, 150.0, 0.0),
            # higher saturated-fraction = more glare; 0.0 = none, 0.3 = severe
            "glare": (self.glare.area_threshold, 0.0, 0.3),
            # higher brightness = more overexposed; 130 = normal, 255 = fully blown out
            "overexposure": (self.overexposure.threshold, 130.0, 255.0),
            # higher anisotropy = more directional (motion) blur; 1.0 = isotropic, 20 = extreme
            "motion": (self.motion.anisotropy_threshold, 1.0, 20.0),
            # higher block-fraction = more occlusion; 0.0 = none, 0.5 = half the frame
            "occlusion": (self.occlusion.block_fraction_threshold, 0.0, 0.5),
            # FEWER pixels = worse (inverted direction vs. the others);
            # 1920x1080 = comfortably high-res, 10,000px = tiny/unusable
            "resolution": (self.resolution.min_pixels, 1920 * 1080, 10_000),
            # higher border-concentration = worse framing; 0.0 = centered, 1.0 = fully at edges
            "framing": (self.framing.border_concentration_threshold, 0.0, 1.0),
        }

    def extract(self, image: np.ndarray) -> dict:
        """
        Returns a dict of {feature_name: normalized_0_1_score}, plus the
        raw (flag, raw_score) per detector under "_raw" for debugging/
        transparency.
        """
        raw_results = {
            "blur": self.blur.is_blurry(image),
            "darkness": self.darkness.is_dark(image),
            "glare": self.glare.has_glare(image),
            "overexposure": self.overexposure.is_overexposed(image),
            "motion": self.motion.has_motion_blur(image),
            "occlusion": self.occlusion.has_occlusion(image),
            "resolution": self.resolution.is_low_resolution(image),
            "framing": self.framing.has_poor_framing(image),
        }

        features = {}
        for name in FEATURE_NAMES:
            _flag, raw_score = raw_results[name]
            threshold, best_anchor, worst_anchor = self._bounds[name]
            features[name] = _normalize(raw_score, threshold, best_anchor, worst_anchor)

        return {"features": features, "_raw": raw_results}

    def extract_vector(self, image: np.ndarray) -> list[float]:
        """Returns just the normalized feature values, in FEATURE_NAMES order."""
        result = self.extract(image)
        return [result["features"][name] for name in FEATURE_NAMES]
