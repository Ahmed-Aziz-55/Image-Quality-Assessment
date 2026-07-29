import numpy as np
import pytest

from app.quality.darkness_detector import DarknessDetector


@pytest.fixture
def bright_image():
    return np.full((300, 300), 220, dtype=np.uint8)


@pytest.fixture
def dark_image():
    return np.full((300, 300), 15, dtype=np.uint8)


def test_bright_image_not_flagged_as_dark(bright_image):
    detector = DarknessDetector(threshold=50.0)
    is_dark, score = detector.is_dark(bright_image)

    assert is_dark is False
    assert score == pytest.approx(220.0)


def test_dark_image_flagged_as_dark(dark_image):
    detector = DarknessDetector(threshold=50.0)
    is_dark, score = detector.is_dark(dark_image)

    assert is_dark is True
    assert score == pytest.approx(15.0)


def test_score_at_exact_threshold_not_flagged():
    """Boundary check: score equal to threshold should NOT count as dark
    (the check is strictly '<', not '<=')."""
    image = np.full((300, 300), 50, dtype=np.uint8)
    detector = DarknessDetector(threshold=50.0)
    is_dark, score = detector.is_dark(image)

    assert is_dark is False
    assert score == pytest.approx(50.0)
