import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.title("Daimler Parts Detector (YOLO)")

# ---------------------------
# Load YOLO model
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")     # <-- put your YOLO trained model file here

model = load_model()


# ---------------------------
# Input (camera or upload)
# ---------------------------
img_file = st.camera_input("Take a picture")

if img_file is None:
    img_file = st.file_uploader("Or upload an image", type=["jpg","jpeg","png"])


# ---------------------------
# Run detection
# ---------------------------
if img_file:
    # Read image using PIL
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy for YOLO
    img_np = np.array(image)

    # Save temp image because YOLO expects a file path or numpy
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Run prediction
    results = model.predict(temp_path, conf=0.25)

    # Render annotated image using PIL
    annotated = results[0].plot()   # returns numpy array with boxes drawn

    st.image(annotated, caption="YOLO Detection", use_column_width=True)

    # Show detected classes
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    if detected_classes:
        st.success(f"Detected: {', '.join(detected_classes)}")
    else:
        st.warning("No objects detected.")
