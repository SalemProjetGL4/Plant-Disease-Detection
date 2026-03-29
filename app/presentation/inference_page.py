from __future__ import annotations

import numpy as np
import streamlit as st
import torch

from app.config import BACKBONES, DEFAULT_IMAGE_SIZE, DEFAULT_MEAN, DEFAULT_STD, SUPPORTED_IMAGE_TYPES
from app.data_utils import (
    find_default_dataset_root,
    list_model_candidates,
    load_class_names,
    parse_three_floats,
    pretty_label,
)
from app.inference import build_thinking_steps, predict_probabilities
from app.modeling import load_model
from app.preprocessing import decode_uploaded_image, preprocess_image
from app.presentation.results import render_inference_results


def render_inference_page() -> None:
    st.subheader("Inference")
    st.caption("Upload an image, configure settings, then run prediction.")

    uploaded_image = st.file_uploader("Drop or upload a plant image", type=SUPPORTED_IMAGE_TYPES, key="inference_upload")
    if uploaded_image is None:
        st.info("Upload an image to continue.")
        return

    file_bytes = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
    try:
        image_rgb = decode_uploaded_image(file_bytes)
    except Exception as error:  # noqa: BLE001
        st.error(f"Image decoding failed: {error}")
        return

    st.image(image_rgb, caption="Uploaded image", use_container_width=True)

    st.markdown("### Inference settings")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        device_name = st.selectbox("Device", ["cuda", "cpu"], index=0, key="inference_device")
    else:
        st.caption("CUDA not available. Using CPU.")

    image_size = st.slider(
        "Input image size",
        min_value=64,
        max_value=512,
        value=DEFAULT_IMAGE_SIZE,
        step=32,
        key="inference_image_size",
    )

    mean_text = st.text_input(
        "Normalize mean (comma-separated)",
        value=", ".join(str(v) for v in DEFAULT_MEAN),
        key="inference_mean",
    )
    std_text = st.text_input(
        "Normalize std (comma-separated)",
        value=", ".join(str(v) for v in DEFAULT_STD),
        key="inference_std",
    )
    mean_values = parse_three_floats(mean_text, DEFAULT_MEAN)
    std_values = parse_three_floats(std_text, DEFAULT_STD)

    selected_backbone = st.selectbox(
        "Checkpoint backbone (for state_dict)",
        BACKBONES,
        index=0,
        key="inference_backbone",
    )

    default_dataset = find_default_dataset_root()
    dataset_root = st.text_input(
        "Dataset root (must contain train)",
        value=str(default_dataset) if default_dataset else "datasets/New Plant Diseases Dataset(Augmented)",
        key="inference_dataset_root",
    )

    classes = load_class_names(dataset_root)
    st.caption(f"Detected classes from dataset: {len(classes)}")

    model_candidates = list_model_candidates()
    default_model = model_candidates[0] if model_candidates else ""
    model_path = st.text_input("Model checkpoint path", value=default_model, key="inference_model_path")

    top_k = st.slider("Top-K probabilities", min_value=3, max_value=10, value=5, key="inference_top_k")

    if not st.button("Run prediction", type="primary", use_container_width=True, key="run_prediction"):
        return

    if not model_path.strip():
        st.error("Please provide a model checkpoint path.")
        return

    classes_for_inference = classes

    try:
        with st.spinner("Loading model..."):
            model, model_meta = load_model(
                model_path=model_path.strip(),
                device_name=device_name,
                selected_backbone=selected_backbone,
                num_classes=max(len(classes_for_inference), 2),
            )

        if model_meta.get("class_names"):
            classes_for_inference = model_meta["class_names"]

        input_tensor, resized_image = preprocess_image(
            image_rgb=image_rgb,
            image_size=image_size,
            mean_values=mean_values,
            std_values=std_values,
        )
        probabilities = predict_probabilities(model, input_tensor, device_name)

    except Exception as error:  # noqa: BLE001
        st.error(f"Inference failed: {error}")
        st.caption(
            "If your checkpoint is state_dict-based, ensure backbone and class count match training. "
            "TorchScript checkpoints usually work without extra settings."
        )
        return

    top_k = min(top_k, probabilities.numel())
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    predicted_idx = int(top_indices[0].item())
    predicted_confidence = float(top_probs[0].item())

    if 0 <= predicted_idx < len(classes_for_inference):
        predicted_raw_label = classes_for_inference[predicted_idx]
    else:
        predicted_raw_label = f"class_{predicted_idx}"

    predicted_label = pretty_label(predicted_raw_label)
    is_healthy = "healthy" in predicted_raw_label.lower()

    thinking_steps = build_thinking_steps(
        original_shape=image_rgb.shape,
        image_size=image_size,
        mean_values=mean_values,
        std_values=std_values,
        top1_label=predicted_label,
        top1_confidence=predicted_confidence,
    )

    render_inference_results(
        image_rgb=image_rgb,
        resized_image=resized_image,
        predicted_label=predicted_label,
        predicted_confidence=predicted_confidence,
        is_healthy=is_healthy,
        model_meta=model_meta,
        device_name=device_name,
        thinking_steps=thinking_steps,
        top_probs=top_probs,
        top_indices=top_indices,
        classes_for_inference=classes_for_inference,
    )
