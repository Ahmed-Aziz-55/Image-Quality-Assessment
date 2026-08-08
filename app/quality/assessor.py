"""
app/quality/assessor.py

Combines all 8 quality detectors (Blur, Darkness, Glare, Overexposure,
Resolution, Motion, Occlusion, Framing) into a single entry point. Loads
an image once and runs every check against it, returning a unified
pass/fail verdict plus each individual score.
"""

import logging

import cv2

from app.quality.blur_detector import BlurDetector
from app.quality.darkness_detector import DarknessDetector
from app.quality.glare_detector import GlareDetector
from app.quality.overexposure_detector import OverexposureDetector
from app.quality.resolution_detector import ResolutionDetector
from app.quality.motion_detector import MotionDetector
from app.quality.occlusion_detector import OcclusionDetector
from app.quality.framing_detector import FramingDetector

logger = logging.getLogger(__name__)


class QualityAssessor:
    """
    Runs all 8 quality checks against an image and reports a combined
    pass/fail verdict plus each individual score.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        darkness_threshold: float = 50.0,
        glare_brightness_cutoff: int = 250,
        glare_area_threshold: float = 0.08,
        overexposure_threshold: float = 220.0,
        min_pixels: int = 256 * 256,
        motion_anisotropy_threshold: float = 3.0,
        motion_weak_direction_energy_threshold: float = 2000.0,
        occlusion_grid_size: int = 12,
        occlusion_variance_threshold: float = 50.0,
        occlusion_edge_density_threshold: float = 0.02,
        occlusion_block_fraction_threshold: float = 0.15,
        framing_grid_size: int = 3,
        framing_center_occupancy_threshold: float = 0.05,
        framing_border_concentration_threshold: float = 0.6,
    ):
        self.blur_detector = BlurDetector(threshold=blur_threshold)
        self.darkness_detector = DarknessDetector(threshold=darkness_threshold)
        self.glare_detector = GlareDetector(
            brightness_cutoff=glare_brightness_cutoff,
            area_threshold=glare_area_threshold,
        )
        self.overexposure_detector = OverexposureDetector(threshold=overexposure_threshold)
        self.resolution_detector = ResolutionDetector(min_pixels=min_pixels)
        self.motion_detector = MotionDetector(
            anisotropy_threshold=motion_anisotropy_threshold,
            weak_direction_energy_threshold=motion_weak_direction_energy_threshold,
        )
        self.occlusion_detector = OcclusionDetector(
            grid_size=occlusion_grid_size,
            variance_threshold=occlusion_variance_threshold,
            edge_density_threshold=occlusion_edge_density_threshold,
            block_fraction_threshold=occlusion_block_fraction_threshold,
        )
        self.framing_detector = FramingDetector(
            grid_size=framing_grid_size,
            center_occupancy_threshold=framing_center_occupancy_threshold,
            border_concentration_threshold=framing_border_concentration_threshold,
        )

    def assess_path(self, image_path: str) -> dict:
        """
        Loads an image from disk and runs all quality checks. Returns a
        dict with a top-level 'passed' verdict and per-check details.
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
        Runs all 8 quality checks against an already-loaded image (as a
        numpy array, e.g. from cv2.imread or an uploaded file decoded
        with cv2.imdecode).
        """
        is_blurry, blur_score = self.blur_detector.is_blurry(img)
        is_dark, dark_score = self.darkness_detector.is_dark(img)
        has_glare, glare_score = self.glare_detector.has_glare(img)
        is_overexposed, overexposure_score = self.overexposure_detector.is_overexposed(img)
        is_low_resolution, resolution_score = self.resolution_detector.is_low_resolution(img)
        has_motion_blur, motion_score = self.motion_detector.has_motion_blur(img)
        has_occlusion, occlusion_score = self.occlusion_detector.has_occlusion(img)
        is_poorly_framed, framing_score = self.framing_detector.has_poor_framing(img)

        passed = not (
            is_blurry
            or is_dark
            or has_glare
            or is_overexposed
            or is_low_resolution
            or has_motion_blur
            or has_occlusion
            or is_poorly_framed
        )

        return {
            "image_path": image_path,
            "loaded": True,
            "passed": passed,
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
            "framing": {"is_poorly_framed": is_poorly_framed, "score": round(framing_score, 4)},
        }
