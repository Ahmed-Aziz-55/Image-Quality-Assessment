import numpy as np
import pytest

from app.quality.occlusion_detector import OcclusionDetector


@pytest.fixture
def textured_image():
    np.random.seed(42)
    return np.random.randint(0, 255, (300, 300), dtype=np.uint8)


@pytest.fixture
def occluded_image(textured_image):
    img = textured_image.copy()
    img[50:200, 50:200] = 128  # large flat block, ~25% of image
    return img


def test_textured_image_not_flagged(textured_image):
    detector = OcclusionDetector()
    has_occ, score = detector.has_occlusion(textured_image)

    assert has_occ is False
    assert score == pytest.approx(0.0)


def test_large_flat_block_flagged(occluded_image):
    detector = OcclusionDetector()
    has_occ, score = detector.has_occlusion(occluded_image)

    assert has_occ is True
    assert score > 0.15


def test_small_flat_region_not_flagged():
    """A small flat patch (below block_fraction_threshold) should not
    trigger occlusion — regression test for over-sensitivity."""
    np.random.seed(1)
    img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    img[0:30, 0:30] = 128  # small patch, ~1% of image

    detector = OcclusionDetector()
    has_occ, score = detector.has_occlusion(img)

    assert has_occ is False
