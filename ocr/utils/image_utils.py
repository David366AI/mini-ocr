"""Image preprocessing utilities for OCR training.

Author: David
"""

from typing import Dict, Tuple

import cv2
import numpy as np
import torch


def recognition_transform(
    image: np.ndarray,
    label: str,
    mean: float,
    std: float,
    alphabet_dict: Dict[str, int],
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize an image and encode its text label.

    Args:
        image: Grayscale image array.
        label: Ground-truth text.
        mean: Normalization mean.
        std: Normalization std.
        alphabet_dict: Character-to-index mapping (1-based for CTC).
        max_seq_len: Max label sequence length.

    Returns:
        image_tensor: Shape (1, H, W), float32.
        label_tensor: Shape (max_seq_len,), long, padded with -1.
        label_len_tensor: Shape (1,), long.
    """
    image = np.expand_dims(image, axis=0).astype(np.float32)
    if np.max(image) > 1.0:
        image = image / 255.0
    image = (image - mean) / std
    image_tensor = torch.from_numpy(image.copy())

    label_encode = np.full(max_seq_len, -1, dtype=np.float32)
    encoded_len = min(len(label), max_seq_len)

    for i, letter in enumerate(label[:encoded_len]):
        if letter not in alphabet_dict:
            raise KeyError(f"Character '{letter}' not found in alphabet_dict")
        label_encode[i] = alphabet_dict[letter]

    label_tensor = torch.from_numpy(label_encode).long()
    label_len_tensor = torch.from_numpy(np.array([encoded_len])).long()
    return image_tensor, label_tensor, label_len_tensor


def resize_gray_image(image: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    """Resize a grayscale image to target height and pad/crop width.

    The image keeps aspect ratio by scaling with height first, then:
    - right-pad with white if width is smaller than target,
    - hard-cap to target width if larger.
    """
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image with shape (H, W), got {image.shape}")

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    if h == input_h and w == input_w:
        return image

    scale = input_h / float(h)
    new_w = max(1, int(round(w * scale)))
    new_w = min(new_w, input_w)

    resized = cv2.resize(image, (new_w, input_h), interpolation=cv2.INTER_LINEAR)

    if new_w < input_w:
        delta_w = input_w - new_w
        resized = cv2.copyMakeBorder(
            resized,
            top=0,
            bottom=0,
            left=0,
            right=delta_w,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )

    return resized
