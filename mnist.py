import os
import cv2
import numpy as np
import tensorflow as tf

import streamlit as st
from PIL import Image

import tempfile

def classify_digit(model, image_np):
    # Check the number of channels and convert only if needed
    if len(image_np.shape) == 3:  # Color image (e.g., RGB or RGBA)
        if image_np.shape[2] == 4:  # RGBA -> Grayscale
            gray_img = cv2.cvtColor(image_np, cv2.COLOR_BGRA2GRAY)
        else:  # RGB -> Grayscale
            gray_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image_np  # Image is already grayscale

    # Resize image to 28x28 pixels (the size expected by the model)
    gray_img_resized = cv2.resize(gray_img, (28, 28))

    # Invert the image (black digits on white background)
    img = np.invert(np.array([gray_img_resized]))
    
    # Normalize the image to match the training data format
    img = img / 255.0

    # Reshape the image to match the input shape of the model (1, 28, 28)
    img = img.reshape(1, 28, 28)

    # Get the prediction from the model
    prediction = model.predict(img)
    return prediction

# Function to resize the image for display
def resize_image(image, target_size):
    img = Image.open(image)
    resized_image = img.resize(target_size)
    return resized_image

st.set_page_config('Digit Recognition', page_icon='🔢')

# Title, caption, and markdown
st.title('Handwritten Digit Recognition 🔢')
st.caption('by Somisetty Alekhya')

st.markdown(r'''This simple application is designed to recognize a number from 0-9 from a PNG file with a resolution of 28x28 pixels. 
            While it may not achieve 100% accuracy, its performance is consistently high.''')
st.subheader('Have fun giving it a try!!! 😊')

# Upload the image
uploaded_image = st.file_uploader('Insert a picture of a number from 0-9', type='png')

if uploaded_image is not None:
    # Convert uploaded image to a NumPy array
    image_np = np.array(Image.open(uploaded_image))

    # Save the image temporarily if necessary
    temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_image.png')
    cv2.imwrite(temp_image_path, image_np)

    # Resize the image for display
    resized_image = resize_image(uploaded_image, (300, 300))

    col1, col2, col3 = st.columns(3)
    
    # Display the image in the center column
    with col2:
        st.image(resized_image)

    # Button to trigger prediction
    submit = st.button('Predict')

    if submit:
        # Load the model
        model = tf.keras.models.load_model('handwrittendigit.model')

        # Use the model to predict the uploaded image
        prediction = classify_digit(model, image_np)

        # Display the prediction result
        st.subheader('Prediction Result')
        st.success(f'The digit is probably a {np.argmax(prediction)}')

        # Clean up the temporary image file
        os.remove(temp_image_path)
