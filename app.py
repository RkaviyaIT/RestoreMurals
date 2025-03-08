import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pywt
import random

# Streamlit app title
st.title("Image Restoration Viewer")

# Function to process the uploaded image
def process_image(uploaded_file):
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original_image is None:
        st.error("Error: Could not load image!")
        return None, None, None

    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    ### **1️ Detect Edges Using Canny and Wavelet Transform**
    canny_edges = cv2.Canny(gray, 50, 150)
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    wavelet_edges = cv2.addWeighted(np.abs(LH), 0.5, np.abs(HL), 0.5, 0)
    wavelet_edges = cv2.addWeighted(wavelet_edges, 0.8, np.abs(HH), 0.2, 0)
    wavelet_edges = cv2.resize(wavelet_edges, (gray.shape[1], gray.shape[0]))

    # Merge Edge Maps
    final_edges = cv2.addWeighted(wavelet_edges.astype(np.float32), 0.4, canny_edges.astype(np.float32), 0.6, 0)
    final_edges = cv2.normalize(final_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ### **2️ Create Binary Damage Mask**
    _, damage_mask = cv2.threshold(final_edges, 120, 255, cv2.THRESH_BINARY)

    ### **3️ Apply Blurring for "Wear and Tear" Effect**
    blurred_image = cv2.GaussianBlur(original_image, (21, 21), 15)
    damaged_image = original_image.copy()
    damaged_image[damage_mask == 255] = blurred_image[damage_mask == 255]

    ### **4️ Introduce Random Missing Parts (Tears, Holes)**
    for _ in range(10):
        x, y = random.randint(0, original_image.shape[1] - 50), random.randint(0, original_image.shape[0] - 50)
        w, h = random.randint(30, 120), random.randint(30, 120)
        damaged_image[y:y+h, x:x+w] = (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150))

    ### **5️ Add Random Color Degradation**
    for _ in range(5):  
        x, y = random.randint(0, original_image.shape[1] - 50), random.randint(0, original_image.shape[0] - 50)
        w, h = random.randint(50, 150), random.randint(50, 150)
        damaged_image[y:y+h, x:x+w, 0] = np.clip(damaged_image[y:y+h, x:x+w, 0] + 40, 0, 255)
        damaged_image[y:y+h, x:x+w, 1] = np.clip(damaged_image[y:y+h, x:x+w, 1] - 30, 0, 255)

    return original_image, damaged_image, damage_mask

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the uploaded image
    original_image, damaged_image, damage_mask = process_image(uploaded_file)

    if original_image is not None:
        # Convert images to PIL format for display
        original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        damaged_image_pil = Image.fromarray(cv2.cvtColor(damaged_image, cv2.COLOR_BGR2RGB))

        # Convert damage mask to a 3-channel image for display
        damage_mask_rgb = cv2.cvtColor(damage_mask, cv2.COLOR_GRAY2RGB)
        damage_mask_pil = Image.fromarray(damage_mask_rgb)

        # Display images
        st.subheader("Uploaded Image(Damaged Image)")
        st.image(damaged_image_pil, caption="Damaged Image", use_column_width=True)

        st.subheader("Damage Mask")
        st.image(damage_mask_pil, caption="Damage Mask", use_column_width=True)

        st.subheader("Restored Image")
        st.image(original_image_pil, caption="Restored Image", use_column_width=True)
