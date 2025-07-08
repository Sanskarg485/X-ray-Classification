import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)

# Replace this path with the correct one to your .pth file
MODEL_PATH = (r"D:\Users\Sanskar Gupta\Downloads\Covid19-Pneumonia-Normal Chest X-Ray Images Dataset\model.pth")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Class labels
    class_names = ["COVID", "Pneumonia", "Normal"]

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure 3-channel
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # For 3 channels
    ])

    # Streamlit UI
    st.title("🩺 Chest X-ray Classifier")
    st.write("Upload a chest X-ray image to classify it as COVID, Pneumonia, or Normal.")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray', use_column_width=True)

        # Preprocess and predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = class_names[predicted.item()]
            st.success(f"🧠 Prediction: **{label}**")
