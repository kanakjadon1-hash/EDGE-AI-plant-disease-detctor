
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# Load model
MODEL_PATH = "model/plant_disease_model.keras"
DATASET_PATH = "dataset"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Get class names from dataset folders
class_names = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

# Page design
st.title("🌿 Plant Disease Detector")
st.write("### Edge-AI Enabled Real-Time Plant Disease Classification")
st.write(
    "Upload a tomato leaf image to predict its disease "
    "using our trained MobileNetV2 model."
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded leaf image", use_container_width=True)

    if st.button("🔍 Detect Disease", type="primary"):
        with st.spinner("Analyzing leaf image..."):
            model = load_model()

            # Preprocess image exactly as in training
            resized_image = image.resize((224, 224))
            image_array = np.array(resized_image).astype("float32") / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Predict disease
            predictions = model.predict(image_array, verbose=0)
            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_index]) * 100

        st.success("Analysis complete!")
        st.subheader("Prediction Result")
        st.write(f"**Predicted class:** {class_names[predicted_index]}")
        st.metric("Model confidence", f"{confidence:.2f}%")
        st.progress(confidence / 100)

        st.info(
            "This is an AI prediction, not a confirmed diagnosis. "
            "Results may be unreliable for poor-quality images "
            "or leaf diseases outside the training classes."
        )

st.divider()
st.caption("Plant Disease Detector | MobileNetV2 | AI Microproject")
