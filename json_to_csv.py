import os
import json
import pandas as pd

base_dir = "images"

folders = {
    "tv": "tv",
    "remote": "remote",
    "wine_bottle": "wine_bottle"
}

rows = []

for folder, label in folders.items():
    annotations_dir = os.path.join(base_dir, folder, "annotations")
    images_dir = os.path.join(base_dir, folder)
    
    if not os.path.exists(annotations_dir):
        print(f"Annotations folder not found: {annotations_dir}")
        continue
    
    json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    print(f"{folder}/ — {len(json_files)} annotation files found")
    
    for jf in sorted(json_files):
        json_path = os.path.join(annotations_dir, jf)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        filename = data.get("imagePath", "")
        # Clean up the filename — get just the file name
        filename = os.path.basename(filename)
        
        for shape in data.get("shapes", []):
            points = shape["points"]
            label_name = shape["label"]
            
            # LabelMe rectangle gives two corner points
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            xmin = int(min(x_coords))
            ymin = int(min(y_coords))
            xmax = int(max(x_coords))
            ymax = int(max(y_coords))
            
            # Store path relative to base: e.g. tv/tv_01.jpg
            filepath = f"{folder}/{filename}"
            
            rows.append({
                "filename": filepath,
                "class": label_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

df = pd.DataFrame(rows)
df.to_csv("annotations.csv", index=False)

print(f"\nTotal annotations: {len(df)}")
print(f"\nClass distribution:")
print(df["class"].value_counts())
print(f"\nSaved to annotations.csv")
print("\nFirst 5 rows:")
print(df.head())