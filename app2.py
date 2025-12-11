os.environ["YOLO_OPENCV"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# Mock cv2 module to prevent import errors
import sys, types
cv2 = types.ModuleType("cv2")
sys.modules["cv2"] = cv2

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.title("Daimler Parts Detector (YOLO Model)")
st.write("Upload or capture an image to detect parts.")

# ------------------ LOAD YOLO MODEL ------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")   # update path if needed
    return model

model = load_model()

# ------------------ INPUT ------------------
img_file = st.camera_input("Take a picture")

if img_file is None:
    img_file = st.file_uploader("Or upload from gallery", type=["jpg", "jpeg", "png"])

if img_file is not None:
    # Streamlit gives image as BytesIO → save temporarily
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(img_file.getvalue())
    temp.flush()

    # Display image
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ------------------ RUN YOLO DETECTION ------------------
    results = model.predict(temp.name, conf=0.25)

    # results[0] contains detection info
    res = results[0]

    # Save image with bounding boxes
    boxed_img_path = temp.name.replace(".jpg", "_pred.jpg")
    res.save(filename=boxed_img_path)

    st.image(boxed_img_path, caption="Detections", use_column_width=True)

    # ------------------ DISPLAY DETECTED LABELS ------------------
    st.subheader("Detected Parts:")

    if len(res.boxes) == 0:
        st.warning("No objects detected.")
    else:
        for box in res.boxes:
            cls = int(box.cls[0])
            label = res.names[cls]
            conf = float(box.conf[0]) * 100

            st.write(f"🔹 **{label}** ({conf:.2f}% confidence)")


