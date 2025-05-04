#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Set page config immediately after imports (only once)
st.set_page_config(page_title="Nail Disease Detector", page_icon="🧠", layout="wide")

# --- Load models once ---
@st.cache_resource
def load_models():
    efficientnet = load_model('efficientnetv0_nail_disease.h5')
    mobilenetv2 = load_model('nail_disease_retrained_mobilenetv2.h5')
    densenet = load_model('densenet201_nail_disease.h5')
    return efficientnet, mobilenetv2, densenet

efficientnet_model, mobilenetv2_model, densenet_model = load_models()

# --- Class labels ---
class_labels = ['Acral_Lentiginous_Melanoma', 'Healthy_Nail', 'Onychogryphosis', 'blue_finger', 'clubbing', 'pitting']

# --- Preprocessing function ---
def load_and_preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# --- Prediction function ---
def predict(model, img):
    processed_img = load_and_preprocess_image(img)
    prediction = model.predict(processed_img)
    return prediction

# --- Custom CSS for background and styling ---
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
# Replace this line with a local image from your PC
st.sidebar.image("C:\\Users\\PC\\OneDrive\\Pictures\\4534941.png", width=120)
st.sidebar.title("🔍 Navigation")
model_choice = st.sidebar.selectbox("Select Model:", ["EfficientNet", "MobileNetV2", "DenseNet"])
st.sidebar.markdown("---")
st.sidebar.info("📤 Upload a **nail image** in JPG format to predict disease.")

# --- Main Area ---
st.title("🩺 Nail Disease Detection App")
st.markdown("#### Powered by Deep Learning models: EfficientNet, MobileNetV2, DenseNet201")

uploaded_image = st.file_uploader("Upload a JPG Image", type=["jpg", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="🖼️ Uploaded Image", width=350)

    if st.button("🚀 Predict"):
        if model_choice == "EfficientNet":
            prediction = predict(efficientnet_model, img)
        elif model_choice == "MobileNetV2":
            prediction = predict(mobilenetv2_model, img)
        elif model_choice == "DenseNet":
            prediction = predict(densenet_model, img)

        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
else:
    st.warning("👆 Please upload a valid nail image!")

# --- Footer ---
st.markdown("---")
st.markdown("<center>Made with ❤️ by Diya And Bhavpreet | 2025</center>", unsafe_allow_html=True)


# In[ ]:





# In[ ]:




