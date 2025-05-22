# app.py
import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

# Load the best model, scaler, and class names
model_path = 'models/final_best_model2.pkl'
scaler_path = 'models/scaler2.pkl'
class_names_path = 'models/class_names2.pkl'

best_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
class_names = joblib.load(class_names_path)

# Streamlit App
st.title('🍎 Fruit and Vegetable Classification')
st.write('Upload an image to classify it.')

# Image upload
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write('')
    st.write('Classifying...')

    # Preprocess the image
    img = np.array(image)

    # Convert image to RGB if needed
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize and normalize
    img = cv2.resize(img, (64, 64))
    img = img / 255.0

    # Flatten and scale
    img = img.reshape(1, -1)
    img_scaled = scaler.transform(img)

    # Predict
    try:
        prediction = best_model.predict(img_scaled)
        predicted_class = class_names[int(prediction[0])]
        st.write(f'### ✅ Prediction: `{predicted_class}`')
    except Exception as e:
        st.error(f"Error during prediction: {e}")