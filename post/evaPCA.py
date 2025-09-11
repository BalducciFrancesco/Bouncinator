import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path_to_folder = os.path.expanduser("~/Documents/Bouncinator/gray_images")

#load the images and resize them to 100x100
target_size = (100, 100)

images = []
filenames = []

for fname in sorted(os.listdir(path_to_folder)):
    filepath = os.path.join(path_to_folder, fname) #join it so it can find the image each time
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    if img is None: 
        continue
    img_small = cv.resize(img, target_size, interpolation=cv.INTER_AREA)
    images.append(img_small.flatten())
    filenames.append(fname)

# Convert list to numpy array
X = np.array(images) #convert to numpy array. n_samples x n_pixels/features
H, W, = target_size[1], target_size[0]
print ("Shape of data matrix X:", X.shape)

#centering the data by subtracting the mean image
mean_img = X.mean(axis=0)
X_centered = X - mean_img 
print("Centered X shape: ", X_centered.shape)

#visulize the mean image
H, W = target_size
plt.imshow(mean_img.reshape(H, W), cmap="gray")
plt.title("Mean image")
plt.axis("off")
plt.show()

n_keep = min(X_centered.shape[0], 150) #caping number of PC at 150 or number of samples, whichever is smaller
pca = PCA(n_components=n_keep, svd_solver="randomized", random_state=0)
scores = pca.fit_transform(X_centered) #project the centered data onto the principal components
components = pca.components_ #principal axes in feature space
explained_variance = pca.explained_variance_ratio_ #variance explained by each of the selected components

print("Scores shape: ", scores.shape)
print("First 10 variance ratios:", explained_variance[:10])

mean_2d = mean_img.reshape(H, W)

def show_pc (i):
    pc = components[i].reshape(H, W)
    s = scores[:, i] #???

    smin, smax = float(s.min()), float(s.max())

    img_min = mean_2d + smin*pc
    img_mid = mean_2d
    img_max = mean_2d + smax*pc

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, im, title in zip(axs, [img_min, img_mid, img_max],[f"PC{i+1} @ min score", "Mean image", f"PC{i+1} @ max score"]):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout
    plt.show()

for i in range(10):
    show_pc(i)



