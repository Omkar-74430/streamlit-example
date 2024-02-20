# libs
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up Streamlit
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model_path = r"C:\project\fabric_authentication_model2.h5"
model = tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img

# Function to make predictions
def predict(image):
    result = model.predict(image)
    return result

# Streamlit app
st.title("Fabric Authentication System using Deep Learning")
st.sidebar.title("Options")

# Apply custom styles
st.markdown(
    """
    <style>
        body {
            color: #333;
            background: linear-gradient(to right, #ff6666, #ff8c66, #ffb366, #ffd966, #ffff66, #d9ff66, #b3ff66);
            font-family: 'Arial', sans-serif;
        }
        .reportview-container {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        .sidebar .sidebar-content {
            background: #ffffff;
        }
        .sidebar .sidebar-content .block-container {
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = preprocess_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success("Image successfully uploaded!")

    # Predict option
    if st.button("Predict"):
        # Make predictions
        predictions = predict(image)
        result_prob = predictions[0][0]

        # Display results
        if result_prob > 0.5:
            st.success(f"Probability of the cloth being handmade is {result_prob * 100:.2f}%")
            st.write('It is Handmade')
        else:
            st.error(f"Probability of the cloth being handmade is {result_prob * 100:.2f}%")
            st.write('It is Machine-made cloth')

