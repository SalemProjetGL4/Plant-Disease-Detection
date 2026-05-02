import torch
import numpy as np
import cv2
from PIL import Image

from config import IMG_SIZE


class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.
    Hooks into model.features[8] — the last MBConv block.

    Usage:
        gc  = GradCAM(model)
        cam = gc.generate(tensor, class_idx=None)
    """

    def __init__(self, model: torch.nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None

        target_layer = model.features[8]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a given class.

        Args:
            tensor    — preprocessed image tensor (1, 3, H, W), requires_grad not needed
            class_idx — target class index; if None, uses the predicted class

        Returns:
            cam — normalized heatmap in [0, 1], shape (H', W') where H'≈7
        """
        self.model.zero_grad()
        tensor = tensor.clone().requires_grad_(True)
        out = self.model(tensor)

        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(
    pil_img:  Image.Image,
    cam:      np.ndarray,
    alpha:    float = 0.45,
    colormap: int   = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Blend a Grad-CAM heatmap onto the original image.

    Args:
        pil_img  — original PIL image (any size)
        cam      — normalized heatmap array in [0, 1]
        alpha    — heatmap opacity (0 = invisible, 1 = fully opaque)
        colormap — OpenCV colormap constant (e.g. cv2.COLORMAP_JET)

    Returns:
        blended PIL image at IMG_SIZE × IMG_SIZE
    """
    img_np  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    cam_rs  = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heat    = cv2.applyColorMap(np.uint8(255 * cam_rs), colormap)
    heat    = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blend   = (alpha * heat + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(blend)
