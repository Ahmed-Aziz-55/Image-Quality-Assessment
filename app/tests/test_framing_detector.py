import numpy as np
import cv2

from app.quality.framing_detector import FramingDetector


def _blank_canvas(size=300):
    return np.full((size, size, 3), 255, dtype=np.uint8)


def test_centered_subject_not_flagged():
    """A clearly-drawn subject sitting in the middle of the frame should pass."""
    img = _blank_canvas(300)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), thickness=3)
    detector = FramingDetector()
    has_poor_framing, score = detector.has_poor_framing(img)
    assert has_poor_framing is False


def test_subject_hugging_edges_flagged():
    """A subject whose edges concentrate in the outer ring (crowding/
    exiting the frame border) should be flagged as poorly framed."""
    img = _blank_canvas(300)
    cv2.rectangle(img, (5, 5), (295, 295), (0, 0, 0), thickness=8)
    detector = FramingDetector()
    has_poor_framing, score = detector.has_poor_framing(img)
    assert has_poor_framing is True
    assert score > 0.6


def test_blank_image_not_flagged():
    """An image with no edges at all (uniform color) shouldn't be flagged --
    that's a job for other detectors, not framing."""
    img = _blank_canvas(300)
    detector = FramingDetector()
    has_poor_framing, score = detector.has_poor_framing(img)
    assert has_poor_framing is False
    # Score is now the detector'''s own threshold (not 0.0) when no edges
    # are detected -- this makes FeatureExtractor normalize it to a
    # NEUTRAL 0.5, rather than the old behavior of returning 0.0 (which
    # incorrectly normalized to "good framing"). See Decisions.md for
    # the confound this fixed in the Suitability Model.
    assert score == detector.border_concentration_threshold