"""
app/pilot/degradation_engine.py

Applies each of the 7 injectable defect types (everything except Poor
Framing, which has no ground-truth injection path) at a controlled,
continuous severity parameter. Used for:
  1. The pilot experiment -- generating severity ladders to visually
     validate where Mild/Medium/High actually break.
  2. Later, the Synthetic Augmentation Engine for the Suitability
     Model's training data.

Each function takes a clean BGR image (as loaded by cv2.imread) and a
single severity parameter matching the units in the severity table in
the project status doc (Gaussian sigma, % brightness change, kernel
length px, patch area %, downscale factor). All functions return an
image of the SAME dimensions as the input (resolution degradation
downscales then upscales back, simulating detail loss while keeping
the image comparable side-by-side with the others).

No thresholds are hardcoded here -- severity is a free parameter so the
pilot experiment can sweep it and find where Mild/Medium/High actually
sit, rather than baking in unvalidated guesses.
"""

import random

import cv2
import numpy as np


def apply_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian blur. sigma matches the severity table's "Gaussian sigma"
    units directly (1-2 mild, 3-5 medium, 6+ high -- per the placeholder
    table, pending pilot validation).
    """
    if sigma <= 0:
        return image.copy()
    # kernel size must be odd and large enough to represent the given sigma
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)


def apply_darkness(image: np.ndarray, reduction_pct: float) -> np.ndarray:
    """
    Reduces brightness by reduction_pct (0-100). E.g. reduction_pct=25
    scales all pixel values to 75% of their original value.
    """
    factor = max(0.0, 1.0 - (reduction_pct / 100.0))
    darkened = image.astype(np.float32) * factor
    return np.clip(darkened, 0, 255).astype(np.uint8)


def apply_overexposure(image: np.ndarray, increase_pct: float) -> np.ndarray:
    """
    Increases brightness by increase_pct (0-100+). Mirrors apply_darkness:
    scales pixel values up and clips at 255, so higher increase_pct washes
    out more of the image (true overexposure look, not just a linear add).
    """
    factor = 1.0 + (increase_pct / 100.0)
    brightened = image.astype(np.float32) * factor
    return np.clip(brightened, 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, kernel_length: int, angle_degrees: float = 0.0) -> np.ndarray:
    """
    Directional motion blur via a line-shaped convolution kernel.
    kernel_length is in pixels, matching the severity table's "kernel
    length px" units. angle_degrees controls blur direction (0 = horizontal).
    """
    if kernel_length <= 1:
        return image.copy()

    kernel_length = int(kernel_length)
    kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
    kernel[kernel_length // 2, :] = 1.0

    if angle_degrees != 0.0:
        center = (kernel_length / 2 - 0.5, kernel_length / 2 - 0.5)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_length, kernel_length))

    kernel /= kernel.sum() if kernel.sum() != 0 else 1.0
    return cv2.filter2D(image, -1, kernel)


def apply_occlusion(image: np.ndarray, patch_area_pct: float, seed: int | None = None) -> np.ndarray:
    """
    Pastes a solid-colored rectangular patch covering approximately
    patch_area_pct (0-100) of the image area, at a random position. The
    fill color is sampled from a random location in the image itself, so
    the patch looks like a plausible occluding object rather than an
    obviously synthetic flat rectangle.
    """
    if patch_area_pct <= 0:
        return image.copy()

    rng = random.Random(seed)
    result = image.copy()
    height, width = image.shape[:2]

    target_area = (patch_area_pct / 100.0) * height * width
    # keep patch roughly square-ish but allow some aspect variation
    aspect = rng.uniform(0.6, 1.6)
    patch_h = int(round((target_area / aspect) ** 0.5))
    patch_w = int(round(patch_h * aspect))
    patch_h = min(patch_h, height)
    patch_w = min(patch_w, width)

    y0 = rng.randint(0, max(0, height - patch_h))
    x0 = rng.randint(0, max(0, width - patch_w))

    fill_y = rng.randint(0, height - 1)
    fill_x = rng.randint(0, width - 1)
    fill_color = image[fill_y, fill_x].tolist()

    result[y0:y0 + patch_h, x0:x0 + patch_w] = fill_color
    return result


def apply_low_resolution(image: np.ndarray, downscale_factor: float) -> np.ndarray:
    """
    Downscales the image by downscale_factor (0-1, e.g. 0.5 = half
    linear resolution) then upscales back to the original dimensions,
    simulating detail loss while keeping the output directly comparable
    (same shape) to the other degraded variants.
    """
    downscale_factor = max(0.05, min(1.0, downscale_factor))
    height, width = image.shape[:2]
    small_h = max(1, int(height * downscale_factor))
    small_w = max(1, int(width * downscale_factor))

    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_glare(image: np.ndarray, patch_area_pct: float, seed: int | None = None) -> np.ndarray:
    """
    Pastes a bright, near-white saturated patch (simulating a flash
    reflection or lens glare) covering approximately patch_area_pct
    (0-100) of the image area, with soft edges so it doesn't look like a
    hard-edged synthetic rectangle.
    """
    if patch_area_pct <= 0:
        return image.copy()

    rng = random.Random(seed)
    height, width = image.shape[:2]

    target_area = (patch_area_pct / 100.0) * height * width
    radius = int(round((target_area / np.pi) ** 0.5))
    radius = max(1, min(radius, min(height, width) // 2))

    cy = rng.randint(radius, max(radius, height - radius))
    cx = rng.randint(radius, max(radius, width - radius))

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.3)
    mask = np.clip(mask, 0, 1)[:, :, None]

    white = np.full_like(image, 255)
    result = image.astype(np.float32) * (1 - mask) + white.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8)


# Maps defect name -> (function, default ladder of severity values to sweep)
DEGRADATION_LADDERS = {
    "blur": (apply_blur, [0.5, 1, 2, 3, 4, 5, 6, 8, 10]),
    "darkness": (apply_darkness, [10, 20, 30, 40, 50, 60, 70]),
    "overexposure": (apply_overexposure, [10, 20, 30, 40, 50, 60, 70]),
    "motion": (apply_motion_blur, [2, 4, 6, 8, 10, 12, 15, 18, 22]),
    "occlusion": (apply_occlusion, [2, 5, 8, 12, 15, 20, 25]),
    "resolution": (apply_low_resolution, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
    "glare": (apply_glare, [2, 5, 8, 12, 15, 20]),
}