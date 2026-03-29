from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch

from app.data_utils import pretty_label


def _build_probability_rows(
    top_probs: torch.Tensor,
    top_indices: torch.Tensor,
    classes_for_inference: list[str],
) -> pd.DataFrame:
    class_rows = []
    for rank, (score, class_index) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        if 0 <= class_index < len(classes_for_inference):
            class_name = pretty_label(classes_for_inference[class_index])
        else:
            class_name = f"class_{class_index}"

        class_rows.append(
            {
                "rank": rank,
                "class": class_name,
                "probability": float(score),
                "confidence_percent": round(float(score) * 100, 3),
            }
        )

    return pd.DataFrame(class_rows)


def render_inference_results(
    image_rgb: np.ndarray,
    resized_image: np.ndarray,
    predicted_label: str,
    predicted_confidence: float,
    is_healthy: bool,
    model_meta: dict[str, Any],
    device_name: str,
    thinking_steps: list[str],
    top_probs: torch.Tensor,
    top_indices: torch.Tensor,
    classes_for_inference: list[str],
) -> None:
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.image(image_rgb, caption="Input image", use_container_width=True)
        st.image(resized_image, caption="Model input after resizing", use_container_width=True)

    with col_right:
        st.subheader("Prediction Result")
        st.metric("Predicted class", predicted_label)
        st.metric("Confidence", f"{predicted_confidence * 100:.2f}%")
        st.progress(int(predicted_confidence * 100))
        st.write(f"Plant status: {'Healthy' if is_healthy else 'Diseased'}")

        st.caption(
            f"Model type: {model_meta['load_type']} | Backbone: {model_meta['backbone']} | Device: {device_name}"
        )
        if model_meta["missing_keys"] or model_meta["unexpected_keys"]:
            st.warning(
                "Checkpoint mismatch detected. Missing keys: "
                f"{len(model_meta['missing_keys'])}, Unexpected keys: {len(model_meta['unexpected_keys'])}."
            )

    st.subheader("Model Thinking")
    for step in thinking_steps:
        st.write(f"- {step}")

    probs_df = _build_probability_rows(top_probs, top_indices, classes_for_inference)
    st.dataframe(probs_df, use_container_width=True, hide_index=True)
    st.bar_chart(data=probs_df.set_index("class")["probability"], use_container_width=True)

    st.subheader("Explainable AI")
    st.info(
        "Next step: add Grad-CAM heatmaps so you can visualize which image regions drove the prediction. "
        "This UI is prepared for that extension in a future update."
    )
