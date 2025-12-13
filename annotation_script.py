"""
1. Select the first class (e.g., "pothole") by pressing 0.

2. Draw a box around a pothole in the image.

3. Switch class by pressing 1 for crack.

4. Draw a box around a crack.

5. Press s to save the annotations for that image and move to the next image.

"""

import cv2
import os

# === CONFIG ===
base_dataset_folder = "dataset"

image_folder = os.path.join(base_dataset_folder, "images")
output_folder = os.path.join(base_dataset_folder, "labels")
os.makedirs(output_folder, exist_ok=True)

classes = ["Adjustable Wrench", "Steering Wheel", "Throttle Body"]

# === Globals ===
drawing = False
ix_resized, iy_resized = -1, -1
boxes_resized = []
current_class = 0

# Max display size
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720


def draw_current_class_banner(img):
    """Draws the current class name at top-left of the window."""
    banner = img.copy()
    return banner


def draw_rectangle(event, x, y, flags, param):
    global drawing, ix_resized, iy_resized, img_display, img_display_copy, boxes_resized

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix_resized, iy_resized = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_display_copy = draw_current_class_banner(img_display.copy())
        cv2.rectangle(img_display_copy, (ix_resized, iy_resized), (x, y), (0, 255, 0), 2)

        # Draw class name above the box
        cv2.putText(img_display_copy, classes[current_class], (ix_resized, iy_resized - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, y_min = min(ix_resized, x), min(iy_resized, y)
        x_max, y_max = max(ix_resized, x), max(iy_resized, y)

        boxes_resized.append([x_min, y_min, x_max, y_max, current_class])

        img_display_copy = draw_current_class_banner(img_display.copy())
        cv2.rectangle(img_display_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw class name above the box
        cv2.putText(img_display_copy, classes[current_class], (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def resize_for_display(img):
    h, w = img.shape[:2]
    scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    return resized, scale


def save_yolo(file_path, boxes_resized, img_w, img_h, scale):
    with open(file_path, "w") as f:
        for box in boxes_resized:
            x_min_r, y_min_r, x_max_r, y_max_r, cls = box

            # Scale back to original
            x_min = int(x_min_r / scale)
            y_min = int(y_min_r / scale)
            x_max = int(x_max_r / scale)
            y_max = int(y_max_r / scale)

            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# === Start labeling ===
image_files = sorted([f for f in os.listdir(image_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

for img_name in image_files:
    boxes_resized = []

    img = cv2.imread(os.path.join(image_folder, img_name))
    if img is None:
        print("Error loading:", img_name)
        continue

    img_h, img_w = img.shape[:2]

    img_display, scale = resize_for_display(img)
    img_display_copy = draw_current_class_banner(img_display.copy())

    cv2.namedWindow("Labeling", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Labeling", draw_rectangle)

    print(f"\nAnnotating: {img_name}")
    print("[s] Save | [n] Skip | [c] Clear | [0-9] Change Class")

    while True:
        cv2.imshow("Labeling", img_display_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            break

        elif key == ord('s'):
            save_yolo(os.path.join(output_folder, img_name.rsplit(".", 1)[0] + ".txt"),
                      boxes_resized, img_w, img_h, scale)
            print("Saved.")
            break

        elif key == ord('c'):
            boxes_resized = []
            img_display_copy = draw_current_class_banner(img_display.copy())
            print("Cleared.")

        elif key in [ord(str(i)) for i in range(len(classes))]:
            current_class = int(chr(key))
            img_display_copy = draw_current_class_banner(img_display.copy())
            print("Class →", classes[current_class])

    cv2.destroyAllWindows()
