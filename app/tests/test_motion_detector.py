import cv2
import numpy as np
import pytest

from app.quality.motion_detector import MotionDetector


@pytest.fixture
def sharp_image():
    img = np.zeros((300, 300), dtype=np.uint8)
    img[::10, :] = 255
    img[:, ::10] = 255
    return img


@pytest.fixture
def motion_blurred_image(sharp_image):
    kernel = np.zeros((15, 15))
    kernel[7, :] = 1.0 / 15
    return cv2.filter2D(sharp_image, -1, kernel)


@pytest.fixture
def focus_blurred_image(sharp_image):
    return cv2.GaussianBlur(sharp_image, (15, 15), 0)


def test_sharp_image_not_flagged(sharp_image):
    detector = MotionDetector(anisotropy_threshold=3.0, weak_direction_energy_threshold=2000.0)
    has_motion, score = detector.has_motion_blur(sharp_image)

    assert has_motion is False
    assert score == pytest.approx(1.0)


def test_motion_blur_flagged(motion_blurred_image):
    detector = MotionDetector(anisotropy_threshold=3.0, weak_direction_energy_threshold=2000.0)
    has_motion, score = detector.has_motion_blur(motion_blurred_image)

    assert has_motion is True
    assert score > 3.0


def test_focus_blur_not_flagged_as_motion(focus_blurred_image):
    """Regression test: isotropic (out-of-focus) blur should NOT be
    flagged as motion blur — its gradient energy drops roughly equally
    in both directions, so anisotropy stays low (see Decisions.md #9)."""
    detector = MotionDetector(anisotropy_threshold=3.0, weak_direction_energy_threshold=2000.0)
    has_motion, score = detector.has_motion_blur(focus_blurred_image)

    assert has_motion is False
    assert score == pytest.approx(1.0)
