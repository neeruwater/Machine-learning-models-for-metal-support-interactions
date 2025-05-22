import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("Final_EadsOH.csv")  # Replace with actual file name

# Select input features
input_features = [
    'Delta Bader', 'ICOBI-L1', 'ICOHP-L1', 'Dipole Moment',
    'Work Function', 'd-band center', 'd-band width', 'Strain'
]
df_inputs = df[input_features]

# Compute Pearson correlation matrix
corr_matrix = df_inputs.corr(method='pearson')
corr_matrix_rounded = corr_matrix.round(3)

# Save full correlation matrix to CSV
corr_matrix_rounded.to_csv("pearson_correlation_matrix_all_features.csv")
print(" full Pearson correlation matrix saved to 'pearson_correlation_matrix_all_features.csv'")

# Extract high-correlation pairs (|r| > 0.75)
threshold = 0.75
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > threshold:
            high_corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': round(corr_val, 3)
            })

# Save highly correlated pairs
high_corr_df = pd.DataFrame(high_corr_pairs)
high_corr_df.to_csv("highly_correlated_feature_pairs.csv", index=False)
print(" highly correlated feature pairs saved to 'highly_correlated_feature_pairs.csv'")

# Step 7: Plot and save the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='magma', interpolation='nearest')
plt.colorbar(label="Pearson r")

# Tick labels
plt.xticks(np.arange(len(input_features)), input_features, rotation=45, ha='right')
plt.yticks(np.arange(len(input_features)), input_features)
plt.title("Pearson Correlation Matrix of Input Features", fontsize=14)
plt.tight_layout()

# Save the heatmap
plt.savefig("pearson_correlation_heatmap.png", dpi=300)
print(" heatmap saved to 'pearson_correlation_heatmap.png'")

# Print summary
print("\n=== Highly Correlated Feature Pairs (|r| > 0.75) ===")
print(high_corr_df)

