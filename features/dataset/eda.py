"""Exploratory Data Analysis (EDA), biological zero auditing, and statistical summaries."""

from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from features.dataset.loader import (
    BIOLOGICAL_ZERO_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def audit_biological_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics and quantify biologically invalid zero values across features."""
    rows = []
    n_total = len(df)

    for col in FEATURE_COLUMNS:
        series = df[col]
        is_bio_zero_col = col in BIOLOGICAL_ZERO_COLUMNS
        zero_count = int((series == 0).sum()) if is_bio_zero_col else 0
        zero_pct = (zero_count / n_total) * 100.0 if is_bio_zero_col else 0.0

        valid_series = series.replace(0, np.nan) if is_bio_zero_col else series

        rows.append(
            {
                "Feature": col,
                "Raw_Mean": round(float(series.mean()), 2),
                "Raw_SD": round(float(series.std()), 2),
                "Raw_Min": round(float(series.min()), 2),
                "Raw_Max": round(float(series.max()), 2),
                "Invalid_Zero_Count": zero_count,
                "Invalid_Zero_Pct": round(zero_pct, 2),
                "Valid_Median": round(float(valid_series.median()), 2),
                "Valid_IQR": round(
                    float(valid_series.quantile(0.75) - valid_series.quantile(0.25)),
                    2,
                ),
                "Valid_Mean": round(float(valid_series.mean()), 2),
                "Valid_SD": round(float(valid_series.std()), 2),
            }
        )

    return pd.DataFrame(rows)


def generate_eda_figures(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    figures_dir: Path,
) -> Dict[str, Path]:
    """Generate publication-grade EDA charts (300 DPI PNG and vector PDF)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
        }
    )

    # Figure 1: Class Distribution & Biologically Invalid Zero Audit
    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel (a): Class Imbalance
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Non-Diabetic (0)", "Diabetic (1)"]
    colors = ["#2b5c8f", "#d95f02"]
    bars = axes[0].bar(
        labels,
        class_counts.values,
        color=colors,
        width=0.55,
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0].set_title("(a) Target Class Distribution (N = 768)", fontweight="bold")
    axes[0].set_ylabel("Number of Patients")
    axes[0].set_ylim(0, 580)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for bar, count in zip(bars, class_counts.values):
        pct = (count / len(df)) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            count + 12,
            f"{count} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Panel (b): Biologically Impossible Zero Values
    bio_df = audit_df[audit_df["Feature"].isin(BIOLOGICAL_ZERO_COLUMNS)].sort_values(
        "Invalid_Zero_Pct", ascending=True
    )
    bars2 = axes[1].barh(
        bio_df["Feature"],
        bio_df["Invalid_Zero_Pct"],
        color="#c0392b",
        edgecolor="black",
        linewidth=0.8,
        height=0.55,
    )
    axes[1].set_title(
        "(b) Biologically Impossible Zero Values Encoded as Missing",
        fontweight="bold",
    )
    axes[1].set_xlabel("Percentage of Samples (%)")
    axes[1].set_xlim(0, 68)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)
    for bar, (_, row) in zip(bars2, bio_df.iterrows()):
        axes[1].text(
            row["Invalid_Zero_Pct"] + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['Invalid_Zero_Count'])} ({row['Invalid_Zero_Pct']:.1f}%)",
            va="center",
            ha="left",
            fontweight="bold",
        )

    fig1.tight_layout()
    fig1_png = figures_dir / "fig1_class_and_missing_audit.png"
    fig1_pdf = figures_dir / "fig1_class_and_missing_audit.pdf"
    fig1.savefig(fig1_png, dpi=300, bbox_inches="tight")
    fig1.savefig(fig1_pdf, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Correlation Matrix & Key Feature Distributions by Outcome
    clean_df = df.copy()
    for col in BIOLOGICAL_ZERO_COLUMNS:
        clean_df[col] = clean_df[col].replace(0, np.nan)
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    corr = clean_df.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-0.3,
        vmax=1.0,
        cbar=True,
        ax=axes[0],
        annot_kws={"size": 7.5},
    )
    axes[0].set_title(
        "(a) Clinical Feature Correlation Matrix (Post-Imputation)",
        fontweight="bold",
    )

    # Panel (b): Standardized Boxplots of Top 4 Features by Outcome
    top_features = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]
    norm_df = clean_df[top_features + [TARGET_COLUMN]].copy()
    for col in top_features:
        norm_df[col] = (norm_df[col] - norm_df[col].mean()) / norm_df[col].std()

    melted = norm_df.melt(
        id_vars=TARGET_COLUMN,
        value_vars=top_features,
        var_name="Feature",
        value_name="Z_Score",
    )
    melted["Feature"] = melted["Feature"].replace(
        {"DiabetesPedigreeFunction": "Diabetes\nPedigree"}
    )
    melted["Diagnosis"] = melted[TARGET_COLUMN].map(
        {0: "Non-Diabetic (0)", 1: "Diabetic (1)"}
    )
    sns.boxplot(
        data=melted,
        x="Feature",
        y="Z_Score",
        hue="Diagnosis",
        order=["Glucose", "BMI", "Age", "Diabetes\nPedigree"],
        hue_order=["Non-Diabetic (0)", "Diabetic (1)"],
        palette={"Non-Diabetic (0)": "#2b5c8f", "Diabetic (1)": "#d95f02"},
        ax=axes[1],
        fliersize=2.5,
    )
    axes[1].set_title(
        "(b) Standardized Distribution of Key Predictors by Outcome",
        fontweight="bold",
    )
    axes[1].set_ylabel("Standardized Value (Z-Score)")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left")

    fig2.tight_layout()
    fig2_png = figures_dir / "fig2_correlation_heatmap.png"
    fig2_pdf = figures_dir / "fig2_correlation_heatmap.pdf"
    fig2.savefig(fig2_png, dpi=300, bbox_inches="tight")
    fig2.savefig(fig2_pdf, bbox_inches="tight")
    plt.close(fig2)

    return {
        "fig1_png": fig1_png,
        "fig1_pdf": fig1_pdf,
        "fig2_png": fig2_png,
        "fig2_pdf": fig2_pdf,
    }
