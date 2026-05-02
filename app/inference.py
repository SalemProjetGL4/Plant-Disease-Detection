import torch
import numpy as np
from PIL import Image

from config import DEVICE, infer_transform


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """
    Apply evaluation-style preprocessing (resize + normalize) and add batch dimension.
    Returns a tensor of shape (1, 3, 224, 224) on DEVICE.
    """
    return infer_transform(pil_img).unsqueeze(0).to(DEVICE)


def predict(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """
    Run a forward pass and return softmax probabilities as a numpy array.

    Args:
        model  — trained model in eval mode
        tensor — preprocessed image tensor (1, 3, H, W)

    Returns:
        probs — numpy array of shape (num_classes,)
    """
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


def top_k_results(probs: np.ndarray, classes: list[str], k: int = 5) -> list[dict]:
    """
    Return the top-k predictions as a list of dicts with keys:
        index, name, confidence
    """
    top_idx = np.argsort(probs)[::-1][:k]
    return [
        {
            "index":      int(i),
            "name":       classes[i],
            "confidence": float(probs[i]),
        }
        for i in top_idx
    ]
