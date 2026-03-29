from __future__ import annotations

import torch


def predict_probabilities(model: torch.nn.Module, input_tensor: torch.Tensor, device_name: str) -> torch.Tensor:
    device = torch.device(device_name)
    with torch.no_grad():
        outputs = model(input_tensor.to(device))
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu()
    return probabilities


def build_thinking_steps(
    original_shape: tuple[int, int, int],
    image_size: int,
    mean_values: list[float],
    std_values: list[float],
    top1_label: str,
    top1_confidence: float,
) -> list[str]:
    return [
        f"Input decoded with OpenCV in RGB format, shape = {original_shape}.",
        f"Image resized to {image_size} x {image_size}.",
        f"Pixel values scaled to [0, 1], then normalized with mean={mean_values} and std={std_values}.",
        "Model produced logits for each disease class, converted to probabilities with softmax.",
        f"Top prediction selected as '{top1_label}' with confidence {top1_confidence * 100:.2f}%.",
    ]
