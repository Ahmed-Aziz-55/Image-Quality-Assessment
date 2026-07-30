import numpy as np
import pytest

from app.quality.glare_detector import GlareDetector


@pytest.fixture
def normal_image():
    return np.full((300, 300), 100, dtype=np.uint8)


@pytest.fixture
def glare_image():
    img = np.full((300, 300), 100, dtype=np.uint8)
    img[50:150, 50:150] = 255  # 100x100 bright patch = 11.11% of image
    return img


def test_normal_image_not_flagged_as_glare(normal_image):
    detector = GlareDetector(brightness_cutoff=250, area_threshold=0.08)
    has_glare, score = detector.has_glare(normal_image)

    assert has_glare is False
    assert score == pytest.approx(0.0)


def test_large_bright_patch_flagged_as_glare(glare_image):
    detector = GlareDetector(brightness_cutoff=250, area_threshold=0.08)
    has_glare, score = detector.has_glare(glare_image)

    assert has_glare is True
    assert score == pytest.approx(0.1111, abs=0.001)


def test_small_bright_patch_not_flagged():
    """A small bright patch (below area_threshold) should not trigger a
    glare flag — regression test for the false-positive issue found on
    real sunlit-wall photos (see Decisions.md #3)."""
    img = np.full((300, 300), 100, dtype=np.uint8)
    img[0:20, 0:20] = 255  # 20x20 patch = 0.44% of image — small

    detector = GlareDetector(brightness_cutoff=250, area_threshold=0.08)
    has_glare, score = detector.has_glare(img)

    assert has_glare is False
