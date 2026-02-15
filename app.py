import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the model (Cached so it doesn't reload on every click)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('Mymodel.keras')

model = load_my_model()

# 2. Define the class labels 
# (Keras flow_from_directory sorts labels alphabetically)
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- UI Setup ---
st.set_page_config(page_title="Emotion Detector", page_icon="🎭")
st.title("🎭 Facial Emotion Recognition")
st.write("Upload a photo of a face, and I'll tell you how they're feeling.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # --- Preprocessing ---
    # 1. Convert to RGB (to match ImageDataGenerator's default)
    # 2. Resize to 48x48
    # 3. Convert to array and rescale (1./255)
    img = image.convert('RGB')
    img = img.resize((48, 48))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # --- Prediction ---
    if st.button('Predict Emotion'):
        with st.spinner('Analyzing facial features...'):
            prediction = model.predict(img_array)
            result_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.subheader(f"Result: **{classes[result_index]}**")
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")

            # Fun feedback
            if classes[result_index] == 'Happy':
                st.balloons()
