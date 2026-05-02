import torch
from torchvision import transforms

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

IMAGENET_MEAN = torch.tensor(MEAN).view(3, 1, 1)
IMAGENET_STD  = torch.tensor(STD).view(3, 1, 1)

# ── Inference transform ───────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
