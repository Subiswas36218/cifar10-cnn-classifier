"""
Streamlit frontend for the CIFAR-10 CNN Classifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import streamlit as st

from src.inference import InferenceEngine

st.set_page_config(
    page_title="CIFAR-10 CNN Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 CIFAR-10 CNN Classifier")

st.markdown("""
Upload an image and let the trained PyTorch CNN predict the most likely
CIFAR-10 class.

**Supported Classes**

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck
""")


# Load model only once
@st.cache_resource
def load_engine() -> InferenceEngine:
    return InferenceEngine()


engine = load_engine()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True,
    )

    with st.spinner("Running inference..."):

        result = engine.predict(image)

    st.success("Prediction completed!")

    st.subheader("Prediction")

    st.metric(
        label="Predicted Class",
        value=result["prediction"],
    )

    st.metric(
        label="Confidence",
        value=f"{result['confidence']:.2%}",
    )

    st.subheader("Top-5 Predictions")

    for prediction in result["top_k"]:

        st.progress(prediction["confidence"])

        st.write(f"**{prediction['class']}** " f"({prediction['confidence']:.2%})")

with st.expander("Model Information"):

    st.write("""
Architecture:
- CNN Backbone
- Batch Normalization
- ReLU Activation
- MaxPooling
- Adaptive Average Pooling
- Fully Connected Classifier

Framework:
- PyTorch

Dataset:
- CIFAR-10
""")

st.divider()

st.caption("Built with PyTorch • FastAPI • Streamlit")
