
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model and labels
model = tf.keras.models.load_model("resnet50_model.h5")
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f]

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("ResNet-50 Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]

    st.success(f"Predicted Class: **{predicted_class}**")
