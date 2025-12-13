# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 18:34:13 2025

@author: aravariya
"""

import os
import shutil

base_dataset_folder = "dataset"

class_folders = [
    ("Throttle_Body", "Throttle body new dataset 12 dec"),
    ("Adjustable_Wrench", "Adjustable wrench new dataset 12 dec"),
]

images_output_folder = os.path.join(base_dataset_folder, "images")
os.makedirs(images_output_folder, exist_ok=True)

counter = 85  # GLOBAL counter starts here

for class_name, source_folder in class_folders:

    src_folder = os.path.join(base_dataset_folder, source_folder)

    if not os.path.isdir(src_folder):
        print(f"❌ Folder not found: {src_folder}")
        continue

    for filename in sorted(os.listdir(src_folder)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            src_path = os.path.join(src_folder, filename)
            ext = filename.rsplit(".", 1)[1]

            new_filename = f"{class_name}_{counter}.{ext}"
            dest_path = os.path.join(images_output_folder, new_filename)

            if os.path.exists(dest_path):
                print(f"⚠️ Skipping existing file: {new_filename}")
                counter += 1
                continue

            shutil.copy(src_path, dest_path)
            print(f"✅ Copied → {new_filename}")

            counter += 1

print("\n🎉 Collation completed with continuous numbering!")
