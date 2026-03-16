"""Synthetic image enhancement utilities.

Author: David
"""

import random
from itertools import combinations
from typing import Iterable, Tuple

import cv2
import numpy as np

_AUG_MARKS = ("B", "M", "N", "L", "S", "C")


def _build_augmentation_combinations() -> Tuple[Tuple[str, ...], list[int]]:
    combos: list[Tuple[str, ...]] = []
    for r in range(1, len(_AUG_MARKS) + 1):
        combos.extend(combinations(_AUG_MARKS, r))

    def valid_combo(combo: Iterable[str]) -> bool:
        combo_set = set(combo)
        invalid_pairs = [{"B", "M"}, {"L", "S"}, {"L", "C"}, {"M", "C"}, {"S", "C"}]
        return not any(pair.issubset(combo_set) for pair in invalid_pairs)

    combos = [combo for combo in combos if valid_combo(combo)]
    combos.insert(0, ("",))

    weights = [len(combos) // 2] + [1] * (len(combos) - 1)
    return tuple(combos), weights


_AUG_COMBOS, _AUG_WEIGHTS = _build_augmentation_combinations()


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def generate_blur(image: np.ndarray, kernel_min: int = 3, kernel_max: int = 7) -> np.ndarray:
    """Apply gaussian blur with a random odd kernel size."""
    kernel_min, kernel_max = sorted((kernel_min, kernel_max))
    candidates = [k for k in range(kernel_min, kernel_max + 1) if k % 2 == 1]
    if not candidates:
        candidates = [3]
    k = random.choice(candidates)
    return cv2.GaussianBlur(image, (k, k), 0)


def generate_blur_motion(image: np.ndarray, kernel_min: int, kernel_max: int) -> np.ndarray:
    """Apply linear motion blur with random direction and kernel size."""
    kernel_min, kernel_max = sorted((kernel_min, kernel_max))
    candidates = [k for k in range(kernel_min, kernel_max + 1) if k % 2 == 1]
    if not candidates:
        candidates = [3]
    kernel_size = random.choice(candidates)

    angle = np.random.randint(1, 25) * 15
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2
    indices = np.arange(kernel_size)

    x = np.round(center + np.cos(np.deg2rad(angle)) * (indices - center)).astype(int)
    y = np.round(center + np.sin(np.deg2rad(angle)) * (indices - center)).astype(int)
    kernel[y, x] = 1.0
    kernel /= kernel_size

    return cv2.filter2D(image, -1, kernel)


def gaussian_noise(image: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """Add gaussian noise to a grayscale image."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return _as_uint8(noisy)


def simulate_copy_effect(
    image: np.ndarray,
    contrast_factor: float = 1.0,
    brightness_delta: int = 50,
) -> np.ndarray:
    """Simulate faded copy quality by reducing brightness range."""
    adjusted = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=-brightness_delta)
    upper = random.randint(200, 230)
    return np.clip(adjusted, brightness_delta, upper).astype(np.uint8)


def generate_photo_light(
    image: np.ndarray,
    direction: int = -1,
    darkness: float = 0.7,
) -> np.ndarray:
    """Simulate uneven phone-capture lighting."""
    if direction == -1:
        direction = np.random.randint(0, 4)

    low_value = int(np.min(image)) + 20
    high_value = int(np.max(image)) - 20
    if high_value <= low_value:
        return image

    diff = max(0, high_value - low_value)
    max_intensity = int(min(diff * darkness, 80))
    mask = np.zeros_like(image, dtype=np.int16)

    if direction in (2, 3):
        for x in range(image.shape[1]):
            intensity = int(x / image.shape[1] * max_intensity)
            mask[:, x] = max_intensity - intensity if direction == 2 else intensity
    else:
        for y in range(image.shape[0]):
            intensity = int(y / image.shape[0] * max_intensity)
            mask[y, :] = max_intensity - intensity if direction == 0 else intensity

    return cv2.subtract(image, mask.astype(np.uint8), dtype=cv2.CV_8U)


def check_image_type(image: np.ndarray) -> bool:
    """Heuristically identify scan-like images by bright percentile."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim > 2 else image

    h, w = gray.shape
    if h > 5 and w > 5:
        gray = gray[int(0.2 * h) : int(0.8 * h), int(0.2 * w) : int(0.8 * w)]

    h, w = gray.shape
    if h <= 0 or w <= 0:
        return False

    ys = np.random.randint(0, h, size=100)
    xs = np.random.randint(0, w, size=100)
    sample = gray[ys, xs]
    return float(np.percentile(sample, 80)) > 235


def enhance_process_image(image: np.ndarray) -> np.ndarray:
    """Apply randomized enhancement steps used during OCR training."""
    image = _as_uint8(image)

    combo = random.choices(_AUG_COMBOS, weights=_AUG_WEIGHTS, k=1)[0]
    mark = "".join(combo)
    if not mark:
        return image

    enhanced = image
    is_scan_image = check_image_type(enhanced)

    if "B" in mark:
        enhanced = generate_blur(enhanced, 3, 5 if is_scan_image else 3)

    if "M" in mark:
        enhanced = generate_blur_motion(enhanced, 3, 3)

    if "N" in mark:
        sigma = 10 if is_scan_image and ("M" in mark or "B" in mark) else 5 if is_scan_image else 3
        enhanced = gaussian_noise(enhanced, sigma=sigma)

    if "L" in mark:
        darkness = 0.1 if is_scan_image else 0.3
        enhanced = generate_photo_light(enhanced, direction=-1, darkness=darkness)

    if "C" in mark:
        brightness = random.randint(20, 50) if is_scan_image else random.randint(10, 20)
        enhanced = simulate_copy_effect(enhanced, contrast_factor=1.0, brightness_delta=brightness)

    return _as_uint8(enhanced)
