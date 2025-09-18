import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.interpolate import interp1d

# ----------------------------
# 1. CONFIGURATION
# ----------------------------
image_folder = "C:/Users/praga/OneDrive/Desktop/Cog/Bouncinator/gray_images"
csv_files = [
    "C:/Users/praga/OneDrive/Desktop/Cog/Bouncinator/post/s250062.csv",
    "C:/Users/praga/OneDrive/Desktop/Cog/Bouncinator/post/s250129.csv",
    "C:/Users/praga/OneDrive/Desktop/Cog/Bouncinator/post/s250200.csv"
]
image_size = (200, 200)
num_pca_components = 40
target_score_column = 'Score1'
threshold_variance = 0.90

# ----------------------------
# 2. LOAD IMAGE NAMES AND MERGE SCORES
# ----------------------------
image_data = pd.DataFrame({
    "Image": [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
})

for csv_file in csv_files:
    scores_df = pd.read_csv(csv_file)
    scores_df['Image'] = scores_df['Image'].apply(lambda x: os.path.basename(x))
    image_data = image_data.merge(scores_df, on="Image", how="left")

# ----------------------------
# 3. LOAD AND NORMALIZE IMAGES
# ----------------------------
images = []
valid_indices = []

for idx, row in image_data.iterrows():
    path = os.path.join(image_folder, row['Image'])
    if os.path.exists(path):
        try:
            img = Image.open(path).convert("L")
            img = img.resize(image_size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    else:
        print(f"Warning: {path} not found, skipping.")

image_data = image_data.iloc[valid_indices].reset_index(drop=True)
dataset = np.stack(images, axis=0)
N, H, W = dataset.shape

# ----------------------------
# 4. CENTER DATASET AND PCA
# ----------------------------
mean_image = np.mean(dataset, axis=0)
centered_dataset = dataset - mean_image
X_flat = centered_dataset.reshape(N, H*W)

# PCA via SVD
U, S, Vt = np.linalg.svd(X_flat, full_matrices=False)
k = min(num_pca_components, len(S))
X_pca = U[:, :k] * S[:k]
top_components = Vt[:k, :]

# Attach PCA scores to dataframe
for i in range(k):
    image_data[f'PC{i+1}'] = X_pca[:, i]

# ----------------------------
# 5. VARIANCE EXPLAINED
# ----------------------------
explained_variance = (S**2)/(N-1)
explained_variance_ratio = explained_variance / explained_variance.sum()
cumulative_variance = np.cumsum(explained_variance_ratio)
num_components_90 = np.argmax(cumulative_variance >= threshold_variance) + 1
pc_columns = [f'PC{i+1}' for i in range(num_components_90)]
print(f"Using {num_components_90} PCs to explain {threshold_variance*100:.0f}% variance.")

# ----------------------------
# 5b. VISUALIZE PCA COMPONENTS (MIN, MEAN, MAX SCORE)
# ----------------------------
num_to_show = min(10, k)  # show first 10 PCs

for i in range(num_to_show):
    pc_vector = top_components[i].reshape(H, W)
    pc_scores = X_pca[:, i]

    min_score = pc_scores.min()
    max_score = pc_scores.max()

    # Create three reconstructions
    imgs = [
        mean_image + min_score * pc_vector,
        mean_image,
        mean_image + max_score * pc_vector
    ]

    # Plot side-by-side
    plt.figure(figsize=(8, 3))
    for j, img in enumerate(imgs):
        plt.subplot(1, 3, j+1)
        plt.imshow(np.clip(img, 0, 1), cmap="gray")
        if j == 0:
            plt.title(f"PC{i+1}\nMin score")
        elif j == 1:
            plt.title("Mean image")
        else:
            plt.title("Max score")
        plt.axis("off")

    plt.suptitle(f"Effect of PC{i+1}", fontsize=14)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 6. PREPARE DATA FOR REGRESSION
# ----------------------------
df_model = image_data.dropna(subset=[target_score_column])
available_pcs = [pc for pc in pc_columns if pc in df_model.columns]
X = df_model[available_pcs].values
y = df_model[target_score_column].values
y_norm = (y - y.min()) / (y.max() - y.min())

# ----------------------------
# 7. FORWARD FEATURE SELECTION
# ----------------------------
lr = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
sfs = SequentialFeatureSelector(
    lr,
    n_features_to_select="auto",
    direction="forward",
    scoring="r2",
    cv=cv,
    n_jobs=-1
)
sfs.fit(X, y_norm)
selected_mask = sfs.get_support()
selected_features = [pc for pc, keep in zip(available_pcs, selected_mask) if keep]
print("Selected PCs:", selected_features)

# Fit final model on selected PCs
X_selected = df_model[selected_features].values
final_model = LinearRegression()
final_model.fit(X_selected, y_norm)

for pc_name in selected_features:
    i = int(pc_name.replace("PC", "")) - 1  # PC index (0-based)
    pc_vector = top_components[i].reshape(H, W)
    pc_scores = X_pca[:, i]

    min_score = pc_scores.min()
    max_score = pc_scores.max()

    # Reconstructions at min, mean, max
    imgs = [
        mean_image + min_score * pc_vector,
        mean_image,
        mean_image + max_score * pc_vector
    ]

    # Plot side-by-side
    plt.figure(figsize=(8, 3))
    for j, img in enumerate(imgs):
        plt.subplot(1, 3, j+1)
        plt.imshow(np.clip(img, 0, 1), cmap="gray")
        if j == 0:
            plt.title(f"{pc_name}\nMin score")
        elif j == 1:
            plt.title("Mean image")
        else:
            plt.title("Max score")
        plt.axis("off")

    var_explained = explained_variance_ratio[i] * 100
    plt.suptitle(f"Effect of {pc_name} (Explains {var_explained:.2f}% variance)", fontsize=14)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 8. INTERPOLATE PC SCORES BY RATING
# ----------------------------
mean_pc_per_rating = df_model.groupby(target_score_column)[selected_features].mean()
pc_interp = {}
for pc in selected_features:
    interp = interp1d(mean_pc_per_rating.index, mean_pc_per_rating[pc], kind='linear', fill_value='extrapolate')
    pc_interp[pc] = interp

def get_pc_scores_for_rating(r):
    return np.array([pc_interp[pc](r) for pc in selected_features])

# ----------------------------
# 9. GENERATE SYNTHETIC FACES USING FIRST 5 SELECTED PCs
# ----------------------------
ratings_to_generate = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
synthetic_faces = []

# Take only first 5 selected PCs
selected_indices = [int(pc.replace("PC", "")) - 1 for pc in selected_features[:8]]

for r in ratings_to_generate:
    pc_scores = get_pc_scores_for_rating(r)

    reconstruction = mean_image.flatten().copy()

    # Use only the first 5 selected PCs
    for score, idx in zip(pc_scores[:8], selected_indices):
        reconstruction += score * top_components[idx]

    img = reconstruction.reshape(H, W)
    img = np.clip(img, 0, 1)
    synthetic_faces.append((r, img))

# ----------------------------
# 10. PLOT SYNTHETIC FACES
# ----------------------------
plt.figure(figsize=(20, 4))
for i, (rating, face) in enumerate(synthetic_faces):
    plt.subplot(1, len(synthetic_faces), i+1)
    plt.imshow(face, cmap="gray")
    plt.title(f"Rating {rating}")
    plt.axis("off")
plt.suptitle("Synthetic Faces Using First 8 Selected PCs", fontsize=16)
plt.show()

# ----------------------------
# PARAMETERS
# ----------------------------
output_folder = r"C:\Users\praga\OneDrive\Desktop\Cog\Bouncinator\Synthetic"
os.makedirs(output_folder, exist_ok=True)

num_repeats = 10  # each synthetic image shown 10 times
ratings_to_generate = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]

# ----------------------------
# GENERATE AND SAVE IMAGES
# ----------------------------
synthetic_faces_files = []

for r in ratings_to_generate:
    pc_scores = get_pc_scores_for_rating(r)  # from your previous code
    reconstruction = mean_image.flatten().copy()
    
    # Use first 5 selected PCs
    selected_indices = [int(pc.replace("PC", "")) - 1 for pc in selected_features[:5]]
    for score, idx in zip(pc_scores[:5], selected_indices):
        reconstruction += score * top_components[idx]
    
    img_array = np.clip(reconstruction.reshape(H, W), 0, 1)
    img = Image.fromarray(np.uint8(img_array * 255))
    
    # Save multiple repeats
    for repeat in range(1, num_repeats + 1):
        filename = f"{r:.1f}-{repeat}.png"  # e.g., "2.5-3.png"
        filepath = os.path.join(output_folder, filename)
        img.save(filepath)
        synthetic_faces_files.append(filepath)

print(f"Saved {len(synthetic_faces_files)} synthetic images to '{output_folder}'")
