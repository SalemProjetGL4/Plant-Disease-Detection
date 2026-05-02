import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
        .main { background-color: #f8faf8; }
        .stApp { max-width: 1200px; margin: 0 auto; }
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07);
            border-left: 4px solid #2e7d32;
            margin-bottom: 0.5rem;
        }
        .metric-label { font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
        .metric-value { font-size: 1.6rem; font-weight: 600; color: #1b5e20; }
        .metric-sub   { font-size: 0.85rem; color: #888; margin-top: 2px; }
        .tag {
            display: inline-block;
            background: #e8f5e9;
            color: #2e7d32;
            border-radius: 20px;
            padding: 2px 12px;
            font-size: 0.78rem;
            font-weight: 500;
            margin: 2px;
        }
        .tag.warn { background:#fff8e1; color:#f57f17; }
        .section-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 1.2rem 0 0.5rem;
        }
        div[data-testid="stImage"] img { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

def render_header(model_loaded: bool, model_path: str):
    st.markdown("# 🌿 Plant Disease Classifier")
    st.markdown("Chargez une image de feuille pour obtenir la prédiction et visualiser l'explication Grad-CAM.")
    if not model_loaded:
        st.warning(f"⚠️ Modèle introuvable à `{model_path}`. Les prédictions utiliseront des poids aléatoires.")
    st.markdown("---")


# ── Upload ────────────────────────────────────────────────────────────────────

def render_uploader() -> "st.UploadedFile | None":
    return st.file_uploader(
        "Déposez une image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )


# ── Image panel ───────────────────────────────────────────────────────────────

def render_image_panel(
    original: Image.Image,
    cam_overlay: Image.Image,
) -> bool:
    """
    Render the left column: toggle + image display.
    Returns True if Grad-CAM is currently toggled on.
    """
    st.markdown('<div class="section-title">Visualisation</div>', unsafe_allow_html=True)
    show_cam = st.toggle("Afficher Grad-CAM", value=False)

    if show_cam:
        st.image(cam_overlay, caption="Grad-CAM — zones d'attention du modèle", use_container_width=True)
        with st.expander("Comparer côte à côte"):
            c1, c2 = st.columns(2)
            c1.image(original,    caption="Original",  use_container_width=True)
            c2.image(cam_overlay, caption="Grad-CAM",  use_container_width=True)
    else:
        st.image(original, caption="Image originale", use_container_width=True)

    return show_cam


# ── Info panel ────────────────────────────────────────────────────────────────

def confidence_color(conf: float) -> str:
    if conf >= 0.85: return "#2e7d32"
    if conf >= 0.60: return "#f57f17"
    return "#c62828"


def render_info_panel(
    pred_name: str,
    pred_conf: float,
    probs:     np.ndarray,
    classes:   list[str],
    top_k:     int,
    filename:  str,
    img_size:  tuple[int, int],
):
    color = confidence_color(pred_conf)

    st.markdown('<div class="section-title">Prédiction</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
        <div class="metric-label">Classe prédite</div>
        <div class="metric-value" style="color:{color}">{pred_name.replace("_", " ")}</div>
        <div class="metric-sub">Confiance : <b style="color:{color}">{pred_conf*100:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pred_conf)

    st.markdown(f'<div class="section-title">Top {top_k} prédictions</div>', unsafe_allow_html=True)
    pred_idx = int(np.argmax(probs))
    top_idx  = np.argsort(probs)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(5, top_k * 0.52))
    colors_bar = ["#2e7d32" if i == pred_idx else "#a5d6a7" for i in top_idx]
    bars = ax.barh(
        [classes[i].replace("___", " — ").replace("__", " ").replace("_", " ") for i in top_idx[::-1]],
        [probs[i] * 100 for i in top_idx[::-1]],
        color=colors_bar[::-1],
        height=0.6,
        edgecolor="none",
    )
    ax.set_xlabel("Confiance (%)", fontsize=9)
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    for bar, val in zip(bars, [probs[i] * 100 for i in top_idx[::-1]]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8, color="#444")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    w, h = img_size
    low_conf = pred_conf < 0.6
    st.markdown('<div class="section-title">Infos image</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <span class="tag">{w} × {h} px</span>
    <span class="tag">{filename}</span>
    <span class="tag {'warn' if low_conf else ''}">{'⚠️ Confiance faible' if low_conf else '✓ Confiance élevée'}</span>
    """, unsafe_allow_html=True)


# ── Grad-CAM legend ───────────────────────────────────────────────────────────

def render_cam_legend():
    st.markdown("---")
    st.markdown("### 🔍 Lecture du Grad-CAM")
    c1, c2, c3 = st.columns(3)
    c1.markdown("🔴 **Rouge / Orange**\n\nZones auxquelles le modèle accorde le plus d'importance pour cette prédiction.")
    c2.markdown("🟡 **Jaune / Vert**\n\nZones contribuant modérément à la décision.")
    c3.markdown("🔵 **Bleu**\n\nZones ignorées — arrière-plan, bords, zones non discriminantes.")
