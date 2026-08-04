import numpy as np
import pytest

from app.quality.overexposure_detector import OverexposureDetector


@pytest.fixture
def normal_image():
    return np.full((300, 300), 130, dtype=np.uint8)


@pytest.fixture
def overexposed_image():
    return np.full((300, 300), 230, dtype=np.uint8)


def test_normal_image_not_flagged(normal_image):
    detector = OverexposureDetector(threshold=220.0)
    is_over, score = detector.is_overexposed(normal_image)

    assert is_over is False
    assert score == pytest.approx(130.0)


def test_overexposed_image_flagged(overexposed_image):
    detector = OverexposureDetector(threshold=220.0)
    is_over, score = detector.is_overexposed(overexposed_image)

    assert is_over is True
    assert score == pytest.approx(230.0)


def test_naturally_bright_scene_not_flagged():
    """Regression test for the false-positive issue found on a real
    snowy-scene photo (see Decisions.md #8) — a bright-but-not-clipped
    image (mean ~207) should not be flagged at the tuned threshold."""
    image = np.full((300, 300), 207, dtype=np.uint8)
    detector = OverexposureDetector(threshold=220.0)
    is_over, score = detector.is_overexposed(image)

    assert is_over is False
