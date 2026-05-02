import os
import json
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from config import DEVICE


def load_classes(path: str = "classes.json") -> list[str] | None:
    """
    Load class names from a JSON file.
    Returns None if the file does not exist.
    """
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_model(num_classes: int) -> nn.Module:
    """
    Build EfficientNet-B0 with a custom classification head.
    Weights are NOT loaded here — call load_weights() separately.
    """
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, num_classes),
    )
    return model


def load_weights(model: nn.Module, model_path: str) -> bool:
    """
    Load saved weights into the model in-place.
    Returns True if weights were found and loaded, False otherwise.
    """
    if not os.path.exists(model_path):
        return False
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    return True


def get_model(model_path: str, num_classes: int) -> tuple[nn.Module, bool]:
    """
    Convenience function: build model, attempt to load weights,
    move to device, set to eval mode.

    Returns:
        model       — ready-to-use nn.Module
        loaded      — True if weights were found and loaded
    """
    model = build_model(num_classes)
    loaded = load_weights(model, model_path)
    model.to(DEVICE).eval()
    return model, loaded
