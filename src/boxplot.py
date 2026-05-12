"""
A01 - California Housing Boxplot
OPIM 5512 | Dr. Dave Wanik
Loads the California Housing dataset and saves a boxplot to figs/boxplot.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# ── 1. Load data ──────────────────────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(df.describe().round(2))

# ── 2. Create output directory ────────────────────────────────────────────────
os.makedirs("figs", exist_ok=True)

# ── 3. Build boxplot ──────────────────────────────────────────────────────────
cols_to_plot = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "MedHouseVal"]
labels = [
    "Median\nIncome",
    "House\nAge",
    "Avg\nRooms",
    "Avg\nBedrooms",
    "Median House\nValue ($100k)",
]

fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(14, 6))
fig.suptitle("California Housing Dataset — Feature Distributions", fontsize=14, fontweight="bold", y=1.01)

colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

for ax, col, label, color in zip(axes, cols_to_plot, labels, colors):
    data = df[col].dropna()

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.5),
        flierprops=dict(marker="o", color=color, alpha=0.3, markersize=3),
    )

    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.75)

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.yaxis.set_tick_params(length=3)

    # annotate median value
    median_val = data.median()
    ax.text(
        1, median_val, f" {median_val:.2f}",
        va="center", ha="left", fontsize=8, color="black"
    )

plt.tight_layout()

# ── 4. Save figure ────────────────────────────────────────────────────────────
output_path = os.path.join("figs", "boxplot.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {output_path}")
plt.show()
