import os

# Update this path if your images folder is somewhere else
base_dir = "images"

folders = {
    "tv": "tv",
    "remote": "remote",
    "wine_bottle": "wine"
}

for folder, prefix in folders.items():
    folder_path = os.path.join(base_dir, folder)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    
    # Get all image files
    extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    files.sort()
    
    print(f"\n{folder}/ — {len(files)} images found")
    
    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        # Convert all to .jpg extension name (doesn't re-encode, just renames)
        new_name = f"{prefix}_{i:02d}{ext}"
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f"  {filename} -> {new_name}")

print("\nDone! All images renamed.")