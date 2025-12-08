# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 22:49:04 2025

@author: hbukke
"""

import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import io

checkpoint = torch.load("daimler_parts_classifier.pth", map_location="cpu")
class_names = checkpoint["class_names"]

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(model.classifier[1].in_features, len(class_names))
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ---------- TRANSFORMS ----------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- STREAMLIT UI ----------
st.title("Daimler Parts Classifier")
st.write("Upload an image to classify its part type.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Selected Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_label = class_names[pred.item()]

    st.success(f"Predicted Class: **{predicted_label}**") 

