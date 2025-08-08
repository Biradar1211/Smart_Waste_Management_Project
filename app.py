import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ✅ Load the trained model
model = tf.keras.models.load_model("waste_classifier_model.h5")  # or .h5

# ✅ Class names in the same order as training
class_names = ['hazardous', 'organic', 'plastic', 'recyclable']

# ✅ Streamlit UI
st.title("♻️ Smart Waste Classifier")
st.write("Upload an image of waste to classify it into one of the categories.")

# ✅ File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ✅ Preprocess image
    img = image.resize((128, 128))  # same size as training
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # ✅ Show result
    st.subheader("Prediction:")
    st.success(f"This waste is *{predicted_class.upper()}* with *{confidence * 100:.2f}%* confidence.")
