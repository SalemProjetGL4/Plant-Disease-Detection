from __future__ import annotations

DEFAULT_IMAGE_SIZE = 128
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
BACKBONES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_small",
    "densenet121",
]
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
