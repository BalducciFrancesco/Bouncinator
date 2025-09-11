import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_folder = "C:/Users/praga/OneDrive/Desktop/Cog/Bouncinator/gray_images"

images = []
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path)  # grayscale
        #img = img.resize((200, 200))  # ensure same size
        img_array = np.array(img, dtype=np.float32) / 255.0
        images.append(img_array)

dataset = np.stack(images, axis=0)  # shape: (N, H, W)

mean_image = np.mean(dataset, axis=0)  # shape: (H, W)
centered_dataset = dataset - mean_image

plt.imshow(mean_image, cmap="gray")
plt.title("Average Image")
plt.axis("off")
plt.show()