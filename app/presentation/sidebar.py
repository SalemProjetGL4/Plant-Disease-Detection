from __future__ import annotations

import streamlit as st
from app.presentation.notebook_browser import render_sidebar_notebook_browser
from app.presentation.types import SidebarState


def _inject_sidebar_button_styles() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            border: none !important;
            box-shadow: none !important;
            justify-content: flex-start !important;
            text-align: left !important;
            padding-left: 0.35rem !important;
            border-radius: 0.5rem !important;
        }
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
            justify-content: flex-start !important;
            text-align: left !important;
        }
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] > div,
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] > div > span {
            width: 100% !important;
            justify-content: flex-start !important;
            text-align: left !important;
        }
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p {
            width: 100% !important;
            text-align: left !important;
            margin: 0 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button p {
            text-align: left !important;
            width: 100% !important;
            margin: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> SidebarState:
    with st.sidebar:
        _inject_sidebar_button_styles()
        st.header("Navigation")
        st.divider()

        if "active_page" not in st.session_state:
            st.session_state["active_page"] = "notebook_preview"

        render_sidebar_notebook_browser(notebooks_dir="notebooks")

        if st.button("Run inference", use_container_width=True, type="primary"):
            st.session_state["active_page"] = "inference"

    return SidebarState(
        active_page=st.session_state["active_page"],
        selected_notebook_path=st.session_state.get("selected_notebook_path"),
    )
