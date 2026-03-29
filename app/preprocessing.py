from __future__ import annotations

import cv2
import numpy as np
import torch


def decode_uploaded_image(file_bytes: np.ndarray) -> np.ndarray:
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode the image. Please upload a valid image file.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(
    image_rgb: np.ndarray,
    image_size: int,
    mean_values: list[float],
    std_values: list[float],
) -> tuple[torch.Tensor, np.ndarray]:
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1)

    mean_tensor = torch.tensor(mean_values).view(3, 1, 1)
    std_tensor = torch.tensor(std_values).view(3, 1, 1)
    normalized = (tensor - mean_tensor) / std_tensor

    return normalized.unsqueeze(0), resized
