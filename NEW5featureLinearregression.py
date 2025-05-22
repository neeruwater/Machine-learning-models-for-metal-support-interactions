import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import normalize

# Load dataset
df = pd.read_csv("cleaned_5_EadsOH.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Define columns
feature_cols = ["Delta Bader", "ICOBI-L1", "Work Function", "d-band center", "Strain"]
target_col = "E_ads(OH)"
meta_cols = ["Metal", "Support", "Termination", "Number of Metal Layers"]

metadata = df[meta_cols].reset_index(drop=True)
X_raw = df[feature_cols].values
y = df[target_col].values

# Apply L2 normalization (per sample)
X = normalize(X_raw, norm='l2', axis=1)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=2)
model = LinearRegression()

pred_all = np.zeros_like(y)
residuals_all = np.zeros_like(y)
mae_scores = []
r2_scores = []

coefs_all = []
intercepts_all = []

# Train/test loop
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    
    # Store fold coefficients
    coefs_all.append(model.coef_)
    intercepts_all.append(model.intercept_)
    pred_all[test_idx] = preds
    residuals_all[test_idx] = preds - y[test_idx]

    mae_scores.append(mean_absolute_error(y[test_idx], preds))
    r2_scores.append(r2_score(y[test_idx], preds))

avg_coefs = np.mean(coefs_all, axis=0)
avg_intercept = np.mean(intercepts_all)

# predictions
results_df = metadata.copy()
results_df["E_ads(OH)_true"] = y
results_df["E_ads(OH)_pred"] = pred_all
results_df["Residual"] = residuals_all
results_df.to_csv("linear_regression_predictions.csv", index=False)

# Save averaged coefficients
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Average Coefficient": avg_coefs
})
coef_df.loc[len(coef_df.index)] = ["Intercept", avg_intercept]
coef_df.to_csv("linear_model_coefficients.csv", index=False)



# Improved Parity Plot
mae_text = f"MAE = {np.mean(np.abs(residuals_all)):.2f} eV"

# Manually define colors for key metals
custom_colors = {
    "Au(111)": "red",
    "Ag(111)": "skyblue",
    "Pt(111)": "green"
}

# Fallback color for other metals
default_color = "gray"

unique_metals = metadata["Metal"].unique()

plt.figure(figsize=(6, 6))

# Scatter each metal group with specified or fallback color
for metal in unique_metals:
    idx = metadata["Metal"] == metal
    color = custom_colors.get(metal, default_color)
    plt.scatter(
        y[idx], pred_all[idx],
        label=metal,
        color=color,
        edgecolor='k',
        alpha=0.85
    )

# Draw black parity line
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', linewidth=1)

# Axis labels with units and bold font
plt.xlabel(r"DFT Adsorption energy (OH$^*$), eV", fontsize=12, fontweight='bold')
plt.ylabel(r"Predicted Adsorption energy (OH$^*$), eV", fontsize=12, fontweight='bold')

# Annotate MAE inside the plot
plt.text(0.05, 0.95, mae_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.legend(title="Metal", fontsize=10, title_fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=11)
plt.grid(False)
plt.tight_layout()
plt.savefig("parity_plot_linear.png", dpi=300)
plt.close()



# Improved Residual Histogram
plt.figure(figsize=(8, 5))
plt.hist(residuals_all, bins=10, color='skyblue', edgecolor='k')

plt.xlabel("Residual (Predicted − Actual)", fontsize=12, fontweight='bold')
plt.ylabel("Count", fontsize=12, fontweight='bold')

plt.tick_params(axis='both', which='major', labelsize=11)
plt.grid(False)
plt.tight_layout()
plt.savefig("residual_histogram_linear.png", dpi=300)
plt.close()


# Improved Coefficient Bar Plot
avg_coefs_series = pd.Series(avg_coefs, index=feature_cols)

plt.figure(figsize=(8, 4))
avg_coefs_series.plot(kind='barh', color='steelblue', edgecolor='k')

plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.xlabel("Linear Regression Coefficient", fontsize=12, fontweight='bold')
plt.ylabel("Feature", fontsize=12, fontweight='bold')

plt.tick_params(axis='both', which='major', labelsize=11)
plt.grid(False)
plt.tight_layout()
plt.savefig("linear_coefficients.png", dpi=300)
plt.close()



# Metrics
avg_mae = np.mean(mae_scores)
avg_r2 = np.mean(r2_scores)
percent_within_0_2ev = 100 * np.sum(np.abs(residuals_all) <= 0.2) / len(residuals_all)

print("\n=== Linear Regression Equation ===")
eqn = " + ".join([f"({c:.3f})*{f}" for c, f in zip(avg_coefs, feature_cols)])
print(f"E_ads(OH) = {eqn} + ({avg_intercept:.3f})\n")

# Save report
report_lines = [
    "=== Linear Regression Evaluation (5-Fold Cross-Validation with L2 Normalization) ===",
    f"Average MAE: {avg_mae:.3f} eV",
    f"Average R² : {avg_r2:.3f}",
    f"% of predictions within ±0.2 eV: {percent_within_0_2ev:.1f}%",
    "",
    "Linear Model (Average Coefficients):",
    f"E_ads(OH) = {eqn} + ({avg_intercept:.3f})",
    "",
    "Files generated:",
    "- linear_regression_predictions.csv",
    "- linear_model_coefficients.csv",
    "- parity_plot_linear.png",
    "- residual_histogram_linear.png",
    "- linear_coefficients.png"
]



with open("linear_regression_report.txt", "w") as f:
    f.write("\n".join(report_lines))

# Print summary
print("\n".join(report_lines))

