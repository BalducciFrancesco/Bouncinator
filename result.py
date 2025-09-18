import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------
# PARAMETERS
# ----------------------------
csv_file = r"C:\Users\praga\OneDrive\Desktop\Cog\Bouncinator\experiment2_ratings.csv"

# ----------------------------
# 1. LOAD CSV
# ----------------------------
df = pd.read_csv(csv_file, usecols=['image_file.1', '250129', 's250062', 's250200'])
df = df.rename(columns={'image_file.1': 'image_file'})

# Extract predicted rating from filename
df['predicted_rating'] = df['image_file'].astype(str).apply(lambda x: float(x.split('-')[0]))

participants = ['250129', 's250062', 's250200']

# Melt dataframe for plotting
df_melt = df.melt(id_vars='predicted_rating', value_vars=participants,
                  var_name='participant', value_name='rating')
df_melt['rating'] = pd.to_numeric(df_melt['rating'], errors='coerce')

# ----------------------------
# 2. SETUP SUBPLOTS
# ----------------------------
num_participants = len(participants)
cols = 2
rows = int(np.ceil(num_participants / cols))

fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
axes = axes.flatten()

# ----------------------------
# 3. PLOT BOX PLOTS AND CALCULATE SPEARMAN ρ
# ----------------------------
rho_table = {}

for i, participant in enumerate(participants):
    ax = axes[i]
    sns.boxplot(x='predicted_rating', y='rating', data=df_melt[df_melt['participant']==participant], ax=ax)
    ax.set_title(f"Participant {participant}")
    ax.set_xlabel("Predicted Rating")
    ax.set_ylabel("Participant Rating (1–5)")
    
    # Spearman correlation
    participant_data = df_melt[df_melt['participant']==participant]
    rho = participant_data['predicted_rating'].corr(participant_data['rating'], method='spearman')
    rho_table[participant] = rho

# Remove unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ----------------------------
# 4. PRINT SPEARMAN ρ
# ----------------------------
print("Spearman's ρ for each participant:")
for participant, rho in rho_table.items():
    print(f"{participant}: ρ = {rho:.3f}")
