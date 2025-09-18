import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ----------------------------
# PARAMETERS
# ----------------------------
image_folder = r"C:\Users\praga\OneDrive\Desktop\Cog\Bouncinator\Synthetic"
csv_file = r"C:\Users\praga\OneDrive\Desktop\Cog\Bouncinator\experiment2_ratings.csv"
participant_id = input("Enter participant/run name (e.g., 'P1'): ")

# ----------------------------
# 1. READ AND SHUFFLE IMAGES
# ----------------------------
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(".png")]
random.shuffle(image_files)

# ----------------------------
# 2. COLLECT RATINGS
# ----------------------------
ratings = []

for idx, filename in enumerate(image_files, start=1):
    filepath = os.path.join(image_folder, filename)
    img = Image.open(filepath)
    
    plt.imshow(np.array(img), cmap="gray")
    plt.title(f"Image {idx}/{len(image_files)}")
    plt.axis("off")
    plt.show()

    # Get rating 1-5 from participant
    while True:
        try:
            rating = float(input("Enter rating (1–5): "))
            if 1 <= rating <= 5:
                break
            else:
                print("Rating must be between 1 and 5.")
        except:
            print("Invalid input. Enter a number 1–5.")
    
    ratings.append(rating)

# ----------------------------
# 3. SAVE RATINGS TO CSV
# ----------------------------
# If CSV exists, append a new column, otherwise create
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(index=image_files)  # index = image filenames

# Add current participant's ratings as a new column
df[participant_id] = ratings

# Save updated CSV
df.to_csv(csv_file, index_label="image_file")
print(f"Saved ratings for {participant_id} in {csv_file}")