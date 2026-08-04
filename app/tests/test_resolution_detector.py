import numpy as np
import pytest

from app.quality.resolution_detector import ResolutionDetector


def test_high_resolution_not_flagged():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = ResolutionDetector(min_pixels=480 * 480)
    is_low, score = detector.is_low_resolution(image)

    assert is_low is False
    assert score == 1280 * 720


def test_low_resolution_flagged():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    detector = ResolutionDetector(min_pixels=480 * 480)
    is_low, score = detector.is_low_resolution(image)

    assert is_low is True
    assert score == 160 * 120


def test_exact_threshold_not_flagged():
    """Boundary check: exactly at threshold should NOT count as low-res
    (strictly '<', not '<=')."""
    image = np.zeros((480, 480, 3), dtype=np.uint8)
    detector = ResolutionDetector(min_pixels=480 * 480)
    is_low, score = detector.is_low_resolution(image)

    assert is_low is False
    assert score == 480 * 480
