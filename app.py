import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert import Interpreter

st.set_page_config(page_title="🍅 Tomato Disease Detector", layout="centered")
st.title("🍅 Tomato Leaf Disease Detector (TFLite Runtime)")
st.write("Upload a tomato leaf image to identify the disease and get information on how to treat it.")

# 🧠 Load the TFLite model and labels
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="tomato_mobilenetv2_best.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

interpreter = load_model()
labels = load_labels()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 🩺 Disease info
disease_info = {
    "Tomato_Bacterial_spot": {
        "Description": "Caused by bacteria leading to small dark spots on leaves and fruit.",
        "Symptoms": "Black or brown water-soaked lesions on leaves and fruit.",
        "Treatment": "Use copper-based fungicides, remove infected leaves, avoid overhead watering."
    },
    "Tomato__Target_Spot": {
        "Description": "Fungal disease caused by Corynespora cassiicola, forming concentric rings.",
        "Symptoms": "Circular lesions with yellow halos, starting from lower leaves.",
        "Treatment": "Apply fungicides like mancozeb or chlorothalonil and prune affected leaves."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "Description": "Viral infection spread by whiteflies, causing leaf curling and stunted growth.",
        "Symptoms": "Upward leaf curling, yellowing, reduced fruit yield.",
        "Treatment": "Control whiteflies and use resistant varieties."
    },
    "Tomato__Tomato_mosaic_virus": {
        "Description": "Virus causing mottled and distorted leaves, reducing yield.",
        "Symptoms": "Mosaic patterns, twisted leaves, poor fruit set.",
        "Treatment": "Disinfect tools, avoid handling after smoking, use resistant seeds."
    }
}

# 📤 Upload Image
uploaded_file = st.file_uploader("Upload a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 🧩 Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

    # 🔍 Prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    label = labels[pred_idx]

    # 📊 Display Results
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: **{confidence:.2f}%**")

    info = disease_info.get(label)
    if info:
        st.markdown("### 🩺 Disease Information")
        st.write(f"**Description:** {info['Description']}")
        st.write(f"**Symptoms:** {info['Symptoms']}")
        st.write(f"**Treatment:** {info['Treatment']}")
