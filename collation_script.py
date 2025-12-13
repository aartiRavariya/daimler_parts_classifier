import os
import shutil

base_dataset_folder = "dataset"
classes = ["Adjustable Wrench", "Steering wheel", "Throttle Body"]

images_output_folder = os.path.join(base_dataset_folder, "images")
labels_output_folder = os.path.join(base_dataset_folder, "labels")

os.makedirs(images_output_folder, exist_ok=True)
os.makedirs(labels_output_folder, exist_ok=True)

counter = 1  # global counter to ensure uniqueness

for class_name in classes:
    class_folder = os.path.join(base_dataset_folder, class_name)

    if not os.path.isdir(class_folder):
        print(f"Skipping missing folder: {class_folder}")
        continue

    for filename in os.listdir(class_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            src = os.path.join(class_folder, filename)

            # create guaranteed unique filename
            ext = filename.rsplit(".", 1)[1]
            new_filename = f"{class_name.replace(' ', '_')}_{counter}.{ext}"
            counter += 1

            dest = os.path.join(images_output_folder, new_filename)

            shutil.copy(src, dest)
            print(f"Copied → {new_filename}")

print("\n🎉 Safe collation completed with NO overwritten files!")
