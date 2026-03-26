import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Load annotations
df = pd.read_csv("annotations.csv")

# Split 80% train, 20% test — stratified so each class is balanced
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["class"])

# Save split CSVs
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train set: {len(train_df)} images")
print(train_df["class"].value_counts())
print(f"\nTest set: {len(test_df)} images")
print(test_df["class"].value_counts())

# Create train/test folder structure
for split in ["train", "test"]:
    for cls in ["tv", "remote", "wine_bottle"]:
        os.makedirs(os.path.join("dataset", split, cls), exist_ok=True)

# Copy images to train/test folders
def copy_images(dataframe, split_name):
    for _, row in dataframe.iterrows():
        src = os.path.join("images", row["filename"])
        dst = os.path.join("dataset", split_name, row["filename"])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: {src} not found!")

print("\nCopying train images...")
copy_images(train_df, "train")
print("Copying test images...")
copy_images(test_df, "test")

print("\nDone! Folder structure:")
for split in ["train", "test"]:
    for cls in ["tv", "remote", "wine_bottle"]:
        path = os.path.join("dataset", split, cls)
        count = len([f for f in os.listdir(path) if not f.startswith('.')])
        print(f"  dataset/{split}/{cls}/ — {count} images")

print("\nUpload the 'dataset' folder + train.csv + test.csv to Kaggle!")