import json
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

from config    import IMG_SIZE
from model     import get_model, load_classes
from inference import preprocess, predict
from gradcam   import GradCAM, overlay_heatmap
from ui        import (
    inject_css,
    render_header,
    render_uploader,
    render_image_panel,
    render_info_panel,
    render_cam_legend,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = "models/2 stage fine tuned model.pth"
DEFAULT_CLASSES_PATH = "classes.json"
DEFAULT_NUM_CLASSES = 38
DEFAULT_TOP_K = 5
DEFAULT_CAM_ALPHA = 0.45


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return path


def prettify_class_name(raw_name: str) -> str:
    pretty = raw_name.replace("___", " - ").replace("__", " ").replace("_", " ")
    return " ".join(pretty.split())


def load_classes_from_dataset(dataset_root: str) -> list[str] | None:
    train_dir = Path(dataset_root) / "train"
    if not train_dir.exists():
        return None
    class_dirs = [d.name for d in train_dir.iterdir() if d.is_dir()]
    return sorted(class_dirs) if class_dirs else None


def infer_classes_from_datasets() -> list[str] | None:
    candidates = [
        "datasets/preprocessed/New Plant Diseases",
        "datasets/New Plant Diseases",
        "datasets/New Plant Diseases Dataset(Augmented)",
        "datasets/preprocessed/New Plant Diseases Dataset(Augmented)",
    ]
    for root in candidates:
        classes = load_classes_from_dataset(root)
        if classes:
            return classes
    return None


def load_mapping_from_json(path: str) -> dict[str, str] | None:
    file_path = resolve_project_path(path)
    if not file_path.exists():
        return None

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return {str(item): prettify_class_name(str(item)) for item in payload}
    return None


def try_write_mapping(path: str, mapping: dict[str, str]) -> None:
    file_path = resolve_project_path(path)
    if file_path.exists():
        return
    try:
        file_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=True), encoding="utf-8")
    except OSError:
        return

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿", 
    layout="wide",
)
inject_css()

# ── Config (no sidebar) ───────────────────────────────────────────────────────
model_path = str(resolve_project_path(DEFAULT_MODEL_PATH))
classes_path = str(resolve_project_path(DEFAULT_CLASSES_PATH))
num_classes_fallback = DEFAULT_NUM_CLASSES
top_k = DEFAULT_TOP_K
cam_alpha = DEFAULT_CAM_ALPHA

# ── Load classes (match evaluation ordering) ─────────────────────────────────
resolved_classes_path = classes_path
class_mapping = load_mapping_from_json(resolved_classes_path)
dataset_class_names = infer_classes_from_datasets()

if dataset_class_names:
    class_names = dataset_class_names
elif class_mapping:
    class_names = sorted(class_mapping.keys())
else:
    class_names = load_classes(resolved_classes_path) or []

if class_names:
    if not class_mapping:
        class_mapping = {name: prettify_class_name(name) for name in class_names}
        try_write_mapping(resolved_classes_path, class_mapping)
    class_labels = [class_mapping.get(name, prettify_class_name(name)) for name in class_names]
else:
    class_names = [f"Classe {i}" for i in range(num_classes_fallback)]
    class_labels = class_names

num_classes = len(class_names)

# ── Load model (cached by Streamlit) ─────────────────────────────────────────
@st.cache_resource
def get_cached_model(model_path, num_classes):
    return get_model(model_path, num_classes)

model, model_loaded = get_cached_model(model_path, num_classes)
gradcam = GradCAM(model)

# ── Header ────────────────────────────────────────────────────────────────────
render_header(model_loaded, model_path)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = render_uploader()
if uploaded is None:
    st.info("👆 Chargez une image pour commencer.")
    st.stop()

# ── Inference ─────────────────────────────────────────────────────────────────
pil_img   = Image.open(uploaded).convert("RGB")
tensor    = preprocess(pil_img)
probs     = predict(model, tensor)

pred_idx  = int(np.argmax(probs))
pred_name = class_labels[pred_idx]
pred_conf = float(probs[pred_idx])

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
cam         = gradcam.generate(tensor, class_idx=pred_idx)
original    = pil_img.resize((IMG_SIZE, IMG_SIZE))
cam_overlay = overlay_heatmap(pil_img, cam, alpha=cam_alpha)

# ── Layout ────────────────────────────────────────────────────────────────────
col_img, col_info = st.columns([1.1, 1], gap="large")

with col_img:
    show_cam = render_image_panel(original, cam_overlay)

with col_info:
    render_info_panel(
        pred_name = pred_name,
        pred_conf = pred_conf,
        probs     = probs,
        classes   = class_labels,
        top_k     = top_k,
        filename  = uploaded.name,
        img_size  = pil_img.size,
    )

if show_cam:
    render_cam_legend()
