"""Large-type, single-message charts for the presentation slides (read from metrics_summary.csv)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS_CSV = ROOT / "artifacts" / "results" / "metrics_summary.csv"
OUT_DIR = ROOT / "artifacts" / "figures" / "slides"

MODELS = ["Decision Tree", "Random Forest", "XGBoost", "SVM"]
STAGES = [
    ("Exp1_Baseline", "Exp. 1: Baseline", "#9aa0a6"),
    ("Exp2_SMOTE", "Exp. 2: + SMOTE", "#4c78a8"),
    ("Exp3_Tuned", "Exp. 3: + SMOTE + Tuning", "#e4572e"),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _value(df: pd.DataFrame, model: str, stage: str, column: str) -> float:
    return float(df[(df["Model"] == model) & (df["Stage"] == stage)][column].iloc[0])


def plot_sensitivity_by_stage(df: pd.DataFrame) -> None:
    """Grouped bars: test sensitivity per model across the three stages."""
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.26
    x = np.arange(len(MODELS))
    for i, (key, label, color) in enumerate(STAGES):
        vals = [_value(df, m, key, "Test_Recall_Sensitivity") for m in MODELS]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["Decision\nTree", "Random\nForest", "XGBoost", "SVM"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Sensitivity (recall on diabetics)")
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "s_sensitivity_by_stage")


def plot_missed_patients(df: pd.DataFrame) -> None:
    """Missed diabetic patients (false negatives out of 54): baseline vs tuned."""
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.36
    x = np.arange(len(MODELS))
    for i, (key, label, color) in enumerate([STAGES[0], STAGES[2]]):
        vals = [_value(df, m, key, "FN") for m in MODELS]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{int(v)}", ha="center", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(["Decision\nTree", "Random\nForest", "XGBoost", "SVM"])
    ax.set_ylabel("Missed diabetics (of 54)")
    ax.set_ylim(0, 34)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "s_missed_patients")


def plot_overfit_gap(df: pd.DataFrame) -> None:
    """Train-test F1 gap: baseline vs tuned (lower is better)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.36
    x = np.arange(len(MODELS))
    for i, (key, label, color) in enumerate([STAGES[0], STAGES[2]]):
        vals = [_value(df, m, key, "Gap_Train_Test_F1") for m in MODELS]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Decision\nTree", "Random\nForest", "XGBoost", "SVM"])
    ax.set_ylabel("Train $-$ test F1 (lower = less overfit)")
    ax.set_ylim(0, 0.58)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "s_overfit_gap")


def main() -> None:
    _style()
    df = pd.read_csv(METRICS_CSV)
    df["Stage"] = df["Stage"].astype(str)
    plot_sensitivity_by_stage(df)
    plot_missed_patients(df)
    plot_overfit_gap(df)
    print(f"Slide figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
