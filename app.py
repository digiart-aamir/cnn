import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# ✅ Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ✅ Load your trained model (trained on 24x24 grayscale images)
model = tf.keras.models.load_model('digit_classifier_final_10.h5')

# ✅ Streamlit UI
st.set_page_config(page_title="Digit Recognition", page_icon="✍️")
st.title("🧠 Handwritten Digit Classifier")
st.write("Upload a **24x24 grayscale** digit image for prediction.")

# ✅ Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption='Uploaded Image', width=150)

    # ✅ Resize to 24x24 (match model input)
    img_resized = image.resize((24, 24))
    img_array = np.array(img_resized)

    # ✅ Normalize and reshape
    img_normalized = img_array / 255.0
    img_input = img_normalized.reshape(1, 24, 24, 1)

    # ✅ Make prediction
    prediction = model.predict(img_input)
    predicted_digit = np.argmax(prediction)

    # ✅ Show result
    st.success(f"Predicted Digit: **{predicted_digit}**")
    st.bar_chart(prediction[0])

else:
    st.info("Upload a digit image to get prediction.")
