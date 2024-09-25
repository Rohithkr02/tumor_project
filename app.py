import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model(r'D:\rohith\Duk\programs\dl\deep-learning\CNN\tumor_detection\results\model.h5')

# Set the title of the app
st.title("Brain Tumor Detection")

# Instruction for the user
st.write("Upload an MRI image and the model will predict if it shows a brain tumor or not.")

# Function to make predictions
def make_prediction(img, model):
    img = img.resize((128, 128))  # Resize the image to 128x128 pixels
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)  # Add batch dimension
    res = model.predict(input_img)
    if res[0][0] > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor"

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Open the image using PIL
    img = Image.open(uploaded_file)

    # Preprocess and make prediction
    result = make_prediction(img, model)

    # Display the result
    st.write(f"Prediction: {result}")