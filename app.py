import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# 1. Load the model
# We use st.cache_resource so the model stays in memory
@st.cache_resource
def load_my_model():
    try:
        # compile=False is used because we only need the model for inference (predictions)
        return tf.keras.models.load_model('Mymodel.keras', compile=False)
    except Exception as e:
        st.error(f"Error loading the model file: {e}")
        return None

model = load_my_model()

# 2. Define the class labels
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- UI Setup ---
st.set_page_config(page_title="Emotion Detector", page_icon="🎭")
st.title("🎭 Facial Emotion Recognition")
st.write("Upload an image of a face to detect the emotion.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # --- Preprocessing ---
    # 1. Convert to Grayscale (standard for FER-2013 models)
    img = image.convert('L') 
    # 2. Resize to the dimensions the model was trained on
    img = img.resize((48, 48))
    # 3. Normalize pixel values to [0, 1]
    img_array = np.array(img) / 255.0
    # 4. Expand dimensions to (1, 48, 48, 1) -> (Batch, Height, Width, Channels)
    img_array = np.expand_dims(img_array, axis=-1) 
    img_array = np.expand_dims(img_array, axis=0) 

    # --- Prediction ---
    if st.button('Predict Emotion'):
        if model is not None:
            with st.spinner('Analyzing...'):
                prediction = model.predict(img_array)
                result_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                # Display Results
                st.markdown("---")
                st.subheader(f"Detected Emotion: **{classes[result_index]}**")
                st.write(f"Confidence: **{confidence:.2f}%**")
                
                # Show probability bar chart
                probs_df = pd.DataFrame(prediction[0], index=classes, columns=['Confidence'])
                st.bar_chart(probs_df)

                # Visual Celebration
                if classes[result_index] == 'Happy':
                    st.balloons()
                elif classes[result_index] == 'Surprise':
                    st.snow()
        else:
            st.error("Model not loaded. Please check the model file and requirements.")

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<h4 style='text-align: center; color: grey;'>Made by Muhammad Ali Kahoot</h4>", 
    unsafe_allow_html=True
)
