import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# 1. Load the model (Cached for performance)
@st.cache_resource
def load_my_model():
    # Adding compile=False skips loading the optimizer/loss state
    return tf.keras.models.load_model('Mymodel.keras', compile=False)

model = load_my_model()

# 2. Define the class labels
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- UI Setup ---
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

st.title("🎭 Facial Emotion Recognition")
st.markdown("---")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a clear photo of a face", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # --- Preprocessing ---
    # Most FER-2013 models use 48x48 Grayscale ('L')
    img = image.convert('L') 
    img = img.resize((48, 48))
    img_array = np.array(img) / 255.0
    
    # Ensure shape is (1, 48, 48, 1) to match Keras input requirements
    img_array = np.expand_dims(img_array, axis=-1) 
    img_array = np.expand_dims(img_array, axis=0) 

    # --- Prediction ---
    if st.button('Predict Emotion'):
        with st.spinner('Analyzing facial features...'):
            prediction = model.predict(img_array)
            result_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            # Main Result
            st.subheader(f"Result: **{classes[result_index]}**")
            st.progress(min(int(confidence), 100))
            st.write(f"Confidence Score: {confidence:.2f}%")

            # --- Visualizing All Emotions ---
            st.write("### Emotion Probability Distribution")
            chart_data = pd.DataFrame(
                prediction[0], 
                index=classes, 
                columns=['Probability']
            )
            st.bar_chart(chart_data)

            # Fun feedback
            if classes[result_index] == 'Happy':
                st.balloons()
            elif classes[result_index] == 'Surprise':
                st.snow()

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: grey;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        Made by <b>Muhammad Ali Kahoot</b>
    </div>
    """, 
    unsafe_allow_html=True
)
