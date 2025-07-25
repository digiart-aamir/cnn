import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageOps
import tensorflow as tf

# ✅ Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ✅ Load model
model = tf.keras.models.load_model('digit_classifier_final_10.h5')

# ✅ Flask setup
app = Flask(__name__)

# ✅ Home Route
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message="No selected file")

    # ✅ Read and preprocess the image
    image = Image.open(file).convert("L")
    image = image.resize((24, 24))
    img_array = np.array(image)
    img_normalized = img_array / 255.0
    img_input = img_normalized.reshape(1, 24, 24, 1)

    # ✅ Predict
    prediction = model.predict(img_input)
    predicted_digit = np.argmax(prediction)

    # ✅ Send result to page
    confidence = [f"{p:.2%}" for p in prediction[0]]

    return render_template('index.html',
                           message=f"Predicted Digit: {predicted_digit}",
                           confidence=confidence,
                           digit=predicted_digit)

# ✅ Run app
if __name__ == '__main__':
    app.run(debug=True)
