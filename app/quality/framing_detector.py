"""
Poor Framing Detector

Heuristic-based detector for poor image composition/framing.

Design note (see Decisions.md):
This detector is included as an 8th INPUT FEATURE for the Suitability Model.
It does NOT contribute to ground-truth label generation, because there is no
reliable way to synthetically inject a "poor framing" defect with a known
severity level — VisionSeek/Flickr30k images carry no subject-location
metadata to inject against. The model will learn to weight this feature on
its own (possibly near-zero if it turns out to be noisy).

Approach: a rule-of-thirds-style grid (default 3x3) over Canny edge output.
Two signals are combined:
  1. center_occupancy    -- fraction of total edge energy sitting in the
                             center cell. Low value suggests the subject
                             isn't in/near the frame center at all (or there
                             is no clear subject).
  2. border_concentration -- fraction of total edge energy sitting in the
                             outer ring of cells. A high value suggests the
                             subject is crowding or exiting the frame edges
                             (a classic "cut off" framing defect).

An image is flagged as poorly framed if EITHER signal crosses its threshold.
"""

import cv2
import numpy as np


def detect_poor_framing(
    image,
    grid_size: int = 3,
    center_occupancy_threshold: float = 0.05,
    border_concentration_threshold: float = 0.6,
    canny_low: int = 50,
    canny_high: int = 150,
):
    """
    Detect poor framing using a rule-of-thirds-style grid analysis.

    Parameters
    ----------
    image : np.ndarray
        BGR image as loaded by cv2.imread (grayscale also accepted).
    grid_size : int
        Number of grid cells per side (default 3x3).
    center_occupancy_threshold : float
        Minimum fraction of total edge energy required in the center cell
        for the image to be considered "has a visible subject centrally."
        Below this, image is flagged.
    border_concentration_threshold : float
        If the fraction of total edge energy in the outermost ring of cells
        exceeds this, the image is flagged (subject likely crowds/exits frame).
    canny_low, canny_high : int
        Canny edge detector thresholds.

    Returns
    -------
    dict:
        is_poorly_framed: bool
        center_occupancy: float
        border_concentration: float
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, canny_low, canny_high)

    h, w = edges.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    total_edge_pixels = np.count_nonzero(edges)
    if total_edge_pixels == 0:
        # No edges at all -- nothing to assess framing against. Don't flag;
        # this is a job for the Blur/Darkness detectors, not this one.
        return {
            "is_poorly_framed": False,
            "center_occupancy": 0.0,
            "border_concentration": 0.0,
        }

    center_edge_pixels = 0
    border_edge_pixels = 0
    center_idx = grid_size // 2

    for row in range(grid_size):
        for col in range(grid_size):
            y0 = row * cell_h
            y1 = (row + 1) * cell_h if row < grid_size - 1 else h
            x0 = col * cell_w
            x1 = (col + 1) * cell_w if col < grid_size - 1 else w

            cell_edges = np.count_nonzero(edges[y0:y1, x0:x1])

            if row == center_idx and col == center_idx:
                center_edge_pixels += cell_edges
            if row == 0 or row == grid_size - 1 or col == 0 or col == grid_size - 1:
                border_edge_pixels += cell_edges

    center_occupancy = center_edge_pixels / total_edge_pixels
    border_concentration = border_edge_pixels / total_edge_pixels

    is_poorly_framed = bool(
        center_occupancy < center_occupancy_threshold
        or border_concentration > border_concentration_threshold
    )

    return {
        "is_poorly_framed": is_poorly_framed,
        "center_occupancy": round(float(center_occupancy), 4),
        "border_concentration": round(float(border_concentration), 4),
    }