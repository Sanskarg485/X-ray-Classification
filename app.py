import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn

# Load model (already loaded in previous cell)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device) # Use the model loaded in the previous cell
model.eval()

# Class labels
class_names = ["COVID", "Pneumonia", "Normal"]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure 3-channel
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Streamlit UI
st.title("🩺 Chest X-ray Classifier")
st.write("Upload a chest X-ray image to classify it as COVID, Pneumonia, or Normal.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]
        st.success(f"🧠 Prediction: **{label}**")
