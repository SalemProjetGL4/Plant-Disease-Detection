from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


def apply_notebook_preprocessing(img_bgr: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """
    Apply the same preprocessing used in images_preprocessing.ipynb (section 7).
    """
    if img_bgr is None:
        raise ValueError("Input image is empty.")

    denoised = cv2.medianBlur(img_bgr, 3)

    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    sharpened = cv2.addWeighted(
        enhanced,
        1.2,
        cv2.GaussianBlur(enhanced, (0, 0), 1.2),
        -0.2,
        0,
    )

    return cv2.resize(sharpened, out_size, interpolation=cv2.INTER_AREA)


def _to_tensor(image_rgb: np.ndarray, mean_values: Sequence[float], std_values: Sequence[float]) -> torch.Tensor:
    tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(list(mean_values)).view(3, 1, 1)
    std = torch.tensor(list(std_values)).view(3, 1, 1)
    return (tensor - mean) / std


def preprocess_pil_image(
    pil_img: Image.Image,
    image_size: int,
    mean_values: Iterable[float],
    std_values: Iterable[float],
) -> torch.Tensor:
    image_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    processed_bgr = apply_notebook_preprocessing(img_bgr, (image_size, image_size))
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    return _to_tensor(processed_rgb, mean_values, std_values)
