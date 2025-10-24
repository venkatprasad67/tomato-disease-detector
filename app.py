
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("tomato_mobilenetv2_best.keras")

# Define the class labels
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus"
]

# Disease info
disease_info = {
    "Tomato_Bacterial_spot": {
        "description": "Caused by *Xanthomonas campestris*, leads to small dark spots on leaves and fruits.",
        "symptoms": "Brown-black leaf spots with yellow halos; fruits get scabby patches.",
        "treatment": [
            "Remove infected leaves and fruits.",
            "Spray copper-based bactericides.",
            "Avoid overhead watering; maintain proper spacing."
        ]
    },
    "Tomato__Target_Spot": {
        "description": "Fungal disease caused by *Corynespora cassiicola*, common in humid conditions.",
        "symptoms": "Dark concentric rings ('target-like' spots) on leaves.",
        "treatment": [
            "Use fungicides with chlorothalonil or mancozeb.",
            "Remove affected leaves; improve ventilation."
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies.",
        "symptoms": "Upward curling and yellowing of leaves, stunted growth, low yield.",
        "treatment": [
            "Remove infected plants immediately.",
            "Use neem oil or sticky traps to control whiteflies.",
            "Plant virus-resistant varieties."
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Highly contagious viral infection affecting tomato leaves and fruit.",
        "symptoms": "Mottled light/dark green leaves, leaf distortion, poor fruit yield.",
        "treatment": [
            "Destroy infected plants.",
            "Sterilize tools with 10% bleach.",
            "Avoid smoking near plants."
        ]
    }
}

# Streamlit UI
st.title("🍅 Tomato Leaf Disease Detector")
st.write("Upload a tomato leaf image to identify the disease and get treatment suggestions.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.subheader(f"🌿 Prediction: {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Disease Info
    info = disease_info[predicted_class]
    st.write(f"### 🦠 Description")
    st.write(info['description'])
    st.write(f"### ⚠️ Symptoms")
    st.write(info['symptoms'])
    st.write(f"### 💊 Treatment")
    for step in info['treatment']:
        st.write(f"- {step}")
