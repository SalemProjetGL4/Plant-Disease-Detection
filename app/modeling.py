from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models

from app.config import BACKBONES


def clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


def build_backbone(backbone_name: str, num_classes: int) -> nn.Module:
    backbone_key = backbone_name.lower()

    if backbone_key == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone_key == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone_key == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone_key == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if backbone_key == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    if backbone_key == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported backbone '{backbone_name}'. Choose one of: {', '.join(BACKBONES)}"
    )


@st.cache_resource(show_spinner=False)
def load_model(
    model_path: str,
    device_name: str,
    selected_backbone: str,
    num_classes: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device(device_name)

    try:
        jit_model = torch.jit.load(str(model_file), map_location=device)
        jit_model.eval()
        return jit_model, {
            "load_type": "torchscript",
            "backbone": "torchscript",
            "missing_keys": [],
            "unexpected_keys": [],
            "class_names": [],
        }
    except Exception as torchscript_error:
        torchscript_error_text = str(torchscript_error)

    checkpoint = torch.load(str(model_file), map_location=device)

    if isinstance(checkpoint, nn.Module):
        checkpoint.to(device)
        checkpoint.eval()
        return checkpoint, {
            "load_type": "serialized_module",
            "backbone": checkpoint.__class__.__name__,
            "missing_keys": [],
            "unexpected_keys": [],
            "class_names": [],
        }

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint format is unsupported. Use TorchScript, a serialized module, "
            "or a state_dict checkpoint dictionary."
        )

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    if state_dict is None and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint

    if state_dict is None:
        raise ValueError(
            "Could not find a state_dict in this checkpoint. "
            f"TorchScript load also failed with: {torchscript_error_text}"
        )

    class_names_from_ckpt = checkpoint.get("class_names", [])
    if isinstance(class_names_from_ckpt, tuple):
        class_names_from_ckpt = list(class_names_from_ckpt)
    if not isinstance(class_names_from_ckpt, list):
        class_names_from_ckpt = []

    inferred_num_classes = checkpoint.get("num_classes")
    if isinstance(inferred_num_classes, int) and inferred_num_classes > 1:
        num_classes = inferred_num_classes
    elif class_names_from_ckpt:
        num_classes = len(class_names_from_ckpt)

    inferred_backbone = checkpoint.get("arch") or checkpoint.get("model_name") or selected_backbone
    model = build_backbone(str(inferred_backbone), num_classes)

    cleaned_state_dict = clean_state_dict(state_dict)
    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, {
        "load_type": "state_dict",
        "backbone": str(inferred_backbone),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "class_names": class_names_from_ckpt,
    }
