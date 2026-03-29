from __future__ import annotations

import streamlit as st

from app.presentation.inference_page import render_inference_page
from app.presentation.notebook_browser import render_notebook_preview
from app.presentation.sidebar import render_sidebar


def run_app() -> None:
    st.set_page_config(page_title="Plant Disease Inference", layout="wide")

    sidebar_state = render_sidebar()

    if sidebar_state.active_page == "inference":
        st.title("Plant Disease Inference UI")
        st.caption("Upload a leaf image and run prediction.")
        render_inference_page()
    else:
        render_notebook_preview()
