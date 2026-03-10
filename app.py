import ultralytics
import torch
import streamlit as st

st.write("Ultralytics version:", ultralytics.__version__)
st.write("Torch version:", torch.__version__)

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Apple Leaf Disease Segmentation", layout="wide")

st.title("Apple Leaf Disease Segmentation")
st.write("Upload an image and run YOLOv12 segmentation on the leaf.")

@st.cache_resource
def load_model():
    return YOLO("models/apple_leaf_seg_best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
image_size = st.slider("Image Size", 320, 1280, 640, 32)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Run Segmentation"):
        with st.spinner("Running inference..."):
            results = model.predict(
                source=image_np,
                imgsz=image_size,
                conf=conf_threshold
            )

            annotated_image = results[0].plot()

        with col2:
            st.subheader("Segmented Output")
            st.image(annotated_image, use_container_width=True)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            st.subheader("Predictions")
            boxes = results[0].boxes
            names = model.names

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                st.write(f"- {names[cls_id]}: {conf:.3f}")
        else:
            st.warning("No detections found.")