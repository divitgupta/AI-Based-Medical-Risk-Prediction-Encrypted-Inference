"""
Exploratory Data Analysis — generates all EDA plots and statistics.

Run:
    python -m src.eda
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import BREAST_CANCER_CSV, HEART_DISEASE_CSV, PLOTS_DIR


def eda_breast_cancer():
    """Generate EDA plots for the Breast Cancer dataset."""
    print("\n📊 EDA: Breast Cancer Wisconsin")
    df = pd.read_csv(BREAST_CANCER_CSV)
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"\n  Class distribution:")
    print(df["target"].value_counts().rename({0: "malignant", 1: "benign"}))

    # ── 1. Class distribution bar chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["target"].value_counts()
    bars = ax.bar(["Malignant (0)", "Benign (1)"], counts.values,
                  color=["#e74c3c", "#2ecc71"], edgecolor="black")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", fontweight="bold")
    ax.set_title("Breast Cancer – Class Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_bc_class_dist.png", dpi=150)
    plt.close(fig)

    # ── 2. Correlation heatmap (mean features only) ───────────────────
    mean_cols = [c for c in df.columns if c.startswith("mean")]
    corr = df[mean_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Breast Cancer – Correlation Matrix (Mean Features)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_bc_corr_matrix.png", dpi=150)
    plt.close(fig)

    # ── 3. Feature distributions (mean features) ─────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, col in zip(axes.flat, mean_cols):
        for label, color in [(0, "#e74c3c"), (1, "#2ecc71")]:
            subset = df[df["target"] == label][col]
            ax.hist(subset, bins=25, alpha=0.6, color=color,
                    label=f"{'Malig' if label==0 else 'Benign'}")
        ax.set_title(col.replace("mean ", ""), fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle("Breast Cancer – Feature Distributions by Class", fontsize=13)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_bc_feat_dists.png", dpi=150)
    plt.close(fig)

    # ── 4. Box plots of top-5 most separating features ───────────────
    # Rank by absolute difference in mean between classes
    diffs = {}
    for col in mean_cols:
        m0 = df[df["target"] == 0][col].mean()
        m1 = df[df["target"] == 1][col].mean()
        diffs[col] = abs(m0 - m1)
    top5 = sorted(diffs, key=diffs.get, reverse=True)[:5]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, top5):
        sns.boxplot(x="target", y=col, data=df, ax=ax,
                    palette={0: "#e74c3c", 1: "#2ecc71", "0": "#e74c3c", "1": "#2ecc71"})
        ax.set_xticklabels(["Malignant", "Benign"])
        ax.set_title(col, fontsize=9)
    plt.suptitle("Top-5 Most Discriminative Features", fontsize=12)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_bc_top5_boxplots.png", dpi=150)
    plt.close(fig)

    print("  ✅ Saved EDA plots for Breast Cancer")


def eda_heart_disease():
    """Generate EDA plots for the Heart Disease dataset."""
    print("\n📊 EDA: Heart Disease (Cleveland)")
    df = pd.read_csv(HEART_DISEASE_CSV)
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"\n  Class distribution:")
    print(df["target"].value_counts().rename({0: "No Disease", 1: "Disease"}))

    # ── 1. Class distribution bar chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["target"].value_counts().sort_index()
    bars = ax.bar(["No Disease (0)", "Disease (1)"], counts.values,
                  color=["#3498db", "#e74c3c"], edgecolor="black")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", fontweight="bold")
    ax.set_title("Heart Disease – Class Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_hd_class_dist.png", dpi=150)
    plt.close(fig)

    # ── 2. Correlation heatmap ────────────────────────────────────────
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Heart Disease – Correlation Matrix")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_hd_corr_matrix.png", dpi=150)
    plt.close(fig)

    # ── 3. Continuous feature distributions ───────────────────────────
    cont = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, col in zip(axes, cont):
        for label, color in [(0, "#3498db"), (1, "#e74c3c")]:
            subset = df[df["target"] == label][col]
            ax.hist(subset, bins=20, alpha=0.6, color=color,
                    label=f"{'No Dis' if label==0 else 'Disease'}")
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)
    plt.suptitle("Heart Disease – Continuous Feature Distributions", fontsize=12)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_hd_cont_dists.png", dpi=150)
    plt.close(fig)

    # ── 4. Categorical feature counts by target ──────────────────────
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    n_cat = len(cat_cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cat_cols):
        pd.crosstab(df[col], df["target"]).plot(kind="bar", ax=ax,
                    color=["#3498db", "#e74c3c"], edgecolor="black")
        ax.set_title(col, fontsize=10)
        ax.legend(["No Disease", "Disease"], fontsize=7)
        ax.set_xlabel("")
    plt.suptitle("Heart Disease – Categorical Features by Target", fontsize=12)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_hd_cat_dists.png", dpi=150)
    plt.close(fig)

    print("  ✅ Saved EDA plots for Heart Disease")


if __name__ == "__main__":
    eda_breast_cancer()
    eda_heart_disease()
    print(f"\nAll EDA plots saved to: {PLOTS_DIR}")
