import cv2
import numpy as np
import pytest

from app.quality.blur_detector import BlurDetector


@pytest.fixture
def sharp_image():
    img = np.zeros((300, 300), dtype=np.uint8)
    img[::10, :] = 255
    img[:, ::10] = 255
    return img


@pytest.fixture
def blurred_image(sharp_image):
    return cv2.GaussianBlur(sharp_image, (25, 25), 0)


def test_sharp_image_not_flagged_as_blurry(sharp_image):
    detector = BlurDetector(threshold=100.0)
    is_blurry, score = detector.is_blurry(sharp_image)

    assert is_blurry is False
    assert score > 100.0


def test_blurred_image_flagged_as_blurry(blurred_image):
    detector = BlurDetector(threshold=100.0)
    is_blurry, score = detector.is_blurry(blurred_image)

    assert is_blurry is True
    assert score < 100.0


def test_sharp_score_much_higher_than_blurred(sharp_image, blurred_image):
    detector = BlurDetector(threshold=100.0)
    _, sharp_score = detector.is_blurry(sharp_image)
    _, blurred_score = detector.is_blurry(blurred_image)

    assert sharp_score > blurred_score
