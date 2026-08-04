import numpy as np
import cv2
import pytest

from app.quality.framing_detector import detect_poor_framing


def _blank_canvas(size=300):
    return np.full((size, size, 3), 255, dtype=np.uint8)


def test_centered_subject_not_flagged():
    """A clearly-drawn subject sitting in the middle of the frame should pass."""
    img = _blank_canvas(300)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), thickness=3)
    result = detect_poor_framing(img)
    assert result["is_poorly_framed"] is False
    assert result["center_occupancy"] > 0.05


def test_subject_cut_off_at_edge_flagged():
    """A subject whose edges are concentrated in the outer ring / crossing
    the frame border should be flagged as poorly framed."""
    img = _blank_canvas(300)
    # Rectangle that hugs the border on all sides (a few px inside, so Canny
    # can actually see it -- edges drawn exactly on the image boundary
    # produce no detectable gradient) -> all edge energy lands in the outer
    # grid ring, none in the center cell.
    cv2.rectangle(img, (5, 5), (295, 295), (0, 0, 0), thickness=8)
    result = detect_poor_framing(img)
    assert result["is_poorly_framed"] is True
    assert result["border_concentration"] > 0.6


def test_blank_image_not_flagged():
    """An image with no edges at all (uniform color) shouldn't be flagged --
    that's a job for other detectors, not framing."""
    img = _blank_canvas(300)
    result = detect_poor_framing(img)
    assert result["is_poorly_framed"] is False
    assert result["center_occupancy"] == 0.0
    assert result["border_concentration"] == 0.0