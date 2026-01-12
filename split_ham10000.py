import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Load metadata
metadata_path = "HAM10000_metadata.csv"
df = pd.read_csv(metadata_path)

# Combine image folders
image_dir1 = "HAM10000_images_part_1"
image_dir2 = "HAM10000_images_part_2"

# Output directories
output_base = "dataset"
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")

# Create base directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Class labels
classes = df["dx"].unique()

# Create subfolders
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

# Process each class
for cls in classes:
    df_cls = df[df["dx"] == cls]
    image_files = df_cls["image_id"].values

    # Add .jpg extension
    image_files = [f"{img_id}.jpg" for img_id in image_files]

    # Train-test split
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    # Copy files
    for f in train_files:
        src = os.path.join(image_dir1, f) if os.path.exists(os.path.join(image_dir1, f)) else os.path.join(image_dir2, f)
        dst = os.path.join(train_dir, cls, f)
        shutil.copyfile(src, dst)

    for f in val_files:
        src = os.path.join(image_dir1, f) if os.path.exists(os.path.join(image_dir1, f)) else os.path.join(image_dir2, f)
        dst = os.path.join(val_dir, cls, f)
        shutil.copyfile(src, dst)

print("✅ Dataset split into 'dataset/train/' and 'dataset/val/' successfully!")