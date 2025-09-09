import os
import json

folder_path = 'raw' # <--- Edit this
output_file = 'images.json' # <--- Edit this

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

# Get all image files from the folder
image_files = []
if os.path.exists(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(image_extensions):
            image_files.append(f"{folder_path}/{filename}")

# Write to JSON file
with open(output_file, 'w') as f:
    json.dump(image_files, f, indent=2)

print(f"Created {output_file} with {len(image_files)} images")
