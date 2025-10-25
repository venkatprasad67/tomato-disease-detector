import gradio as gr
import numpy as np
from PIL import Image
from ai_edge_litert import Interpreter

# Load the TFLite model
interpreter = Interpreter(model_path="tomato_mobilenetv2_best.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Disease information
disease_info = {
    "Tomato_Bacterial_spot": {
        "Description": "Bacterial disease causing small dark lesions on leaves and fruit.",
        "Symptoms": "Black or brown water-soaked spots.",
        "Treatment": "Use copper fungicide, remove infected parts, avoid overhead watering."
    },
    "Tomato__Target_Spot": {
        "Description": "Fungal disease forming concentric rings on leaves.",
        "Symptoms": "Circular lesions with yellow halos.",
        "Treatment": "Use fungicides (mancozeb/chlorothalonil), prune affected leaves."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "Description": "Viral infection spread by whiteflies.",
        "Symptoms": "Upward leaf curling, yellowing, stunted growth.",
        "Treatment": "Control whiteflies, plant resistant varieties."
    },
    "Tomato__Tomato_mosaic_virus": {
        "Description": "Virus causing mottled leaves and poor yield.",
        "Symptoms": "Mosaic patterns, twisted leaves, smaller fruits.",
        "Treatment": "Disinfect tools, use resistant varieties."
    }
}

def predict(image):
    # Preprocess
    image = image.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    label = labels[pred_idx]

    info = disease_info.get(label, {"Description": "N/A", "Symptoms": "", "Treatment": ""})

    return f"""
    **Prediction:** {label}  
    **Confidence:** {confidence:.2f}%  

    🩺 **Description:** {info['Description']}  
    ⚠️ **Symptoms:** {info['Symptoms']}  
    💊 **Treatment:** {info['Treatment']}
    """

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Markdown(),
    title="🍅 Tomato Disease Detector",
    description="Upload a tomato leaf image to detect the disease and learn about its treatment.",
)

app.launch()
