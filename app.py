import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# ✅ Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ✅ Load trained model
model = tf.keras.models.load_model('digit_classifier_final_10.h5')

# ✅ Page setup
st.set_page_config(page_title="Digit Classifier by Engr Aamir", page_icon="🔢", layout="wide")

# ✅ Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧠 Digit Classifier by Engr Aamir")
    st.write("Upload one or more digit images to get predictions using deep learning.")

    uploaded_files = st.file_uploader("📥 Upload digit images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.markdown("#### 📂 Uploaded Files")
        for file in uploaded_files:
            st.markdown(f"- {file.name}")

with col2:
    if 'uploaded_files' in locals() and uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown("### 🖼️ Uploaded Image")
            image = Image.open(uploaded_file).convert("L")
            st.image(image, width=150)

            # Resize and preprocess
            img_resized = image.resize((24, 24))
            img_array = np.array(img_resized)
            img_normalized = img_array / 255.0
            img_input = img_normalized.reshape(1, 24, 24, 1)

            # Prediction
            st.markdown("### 🔍 Prediction Result")
            prediction = model.predict(img_input)
            predicted_digit = np.argmax(prediction)
            st.success(f"🎯 Predicted Digit: **{predicted_digit}**")

            # Probabilities
            st.markdown("### 📊 Prediction Probabilities")
            st.bar_chart(prediction[0])
            st.markdown("---")
    else:
        st.info("👉 Upload images from the left to see predictions here.")
