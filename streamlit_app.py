"""Streamlit demo for the lawn mower perception models.

Upload a yard photo and see YOLOv8 obstacle detection and/or lawn
segmentation overlaid on it, with a confidence-threshold slider.
Deploy on Streamlit Community Cloud with main file: streamlit_app.py
"""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from mower_perception import Detector, format_detections_table, legend_markdown, visualize

EXAMPLES_DIR = Path(__file__).parent / "examples" / "inputs"

st.set_page_config(page_title="Lawn Mower Perception", page_icon="🌱", layout="wide")


@st.cache_resource
def load_detector() -> Detector:
    return Detector()


st.title("Lawn Mower Perception 🌱")
st.markdown(
    "YOLOv8 obstacle detection + lawn segmentation from an autonomous mower capstone. "
    "Upload a yard photo, pick a mode, and adjust the confidence threshold. Trained on a "
    "small custom-labeled dataset — a random backyard photo may need a lower confidence "
    "threshold than a photo similar to the training set."
)
with st.expander("What am I looking at?"):
    st.markdown(legend_markdown())

col_in, col_out = st.columns(2)

with col_in:
    uploaded = st.file_uploader("Yard photo", type=["jpg", "jpeg", "png"])
    example_files = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
    example_choice = st.selectbox("...or pick a sample photo", ["(none)"] + [p.name for p in example_files])
    mode = st.radio("Mode", ["Detection", "Segmentation", "Both"], index=2, horizontal=True)
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05)

image = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
elif example_choice != "(none)":
    image = Image.open(EXAMPLES_DIR / example_choice).convert("RGB")

if image is not None:
    detector = load_detector()
    image_bgr = np.array(image)[:, :, ::-1]
    result = detector.predict(image_bgr, mode=mode.lower(), conf=conf)
    annotated_rgb = visualize(image_bgr, result)[:, :, ::-1]

    with col_in:
        st.image(image, caption="Input", width="stretch")
    with col_out:
        st.image(annotated_rgb, caption="Annotated result", width="stretch")
        st.text(format_detections_table(result))
else:
    st.info("Upload a photo or pick a sample to get started.")
