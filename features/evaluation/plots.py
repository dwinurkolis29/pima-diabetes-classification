"""Publication-grade evaluation plots: Confusion Matrices, ROC/PR Curves, Bias-Variance Learning Curves, and Feature Importance."""

from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

from features.dataset.loader import FEATURE_COLUMNS


MODEL_COLORS = {
    "Decision Tree": "#e67e22",
    "Random Forest": "#27ae60",
    "XGBoost": "#c0392b",
    "SVM": "#2980b9",
}


def generate_evaluation_figures(
    metrics_df: pd.DataFrame,
    fitted_pipelines: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
    random_state: int = 42,
) -> Dict[str, Path]:
    """Generate Figures 3, 4, 5, and 6 in 300 DPI PNG and vector PDF formats."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 300,
        }
    )

    model_order = ["Decision Tree", "Random Forest", "XGBoost", "SVM"]

    # -------------------------------------------------------------------------
    # FIGURE 3: 2x2 Confusion Matrices for Tuned Models (Exp3_Tuned)
    # -------------------------------------------------------------------------
    fig3, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(model_order):
        ax = axes_flat[idx]
        pipe = fitted_pipelines["Exp3_Tuned"][model_name]
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1_val = (2 * tp) / (2 * tp + fp + fn)

        annot = np.array(
            [
                [
                    f"TN = {tn}\n({tn / (tn + fp) * 100:.1f}%)",
                    f"FP = {fp}\n({fp / (tn + fp) * 100:.1f}%)",
                ],
                [
                    f"FN = {fn}\n({fn / (tp + fn) * 100:.1f}%)",
                    f"TP = {tp}\n({tp / (tp + fn) * 100:.1f}%)",
                ],
            ]
        )

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred: Non-Diabetic (0)", "Pred: Diabetic (1)"],
            yticklabels=["Actual: Non-Diabetic (0)", "Actual: Diabetic (1)"],
            annot_kws={"size": 10, "fontweight": "bold"},
            linewidths=1.2,
            linecolor="black",
        )
        ax.set_title(
            f"({chr(97 + idx)}) {model_name} (Tuned)\n"
            f"Sens: {sens:.3f} | Spec: {spec:.3f} | F1: {f1_val:.3f}",
            fontweight="bold",
        )

    fig3.tight_layout()
    fig3_png = figures_dir / "fig3_confusion_matrices.png"
    fig3_pdf = figures_dir / "fig3_confusion_matrices.pdf"
    fig3.savefig(fig3_png, dpi=300, bbox_inches="tight")
    fig3.savefig(fig3_pdf, bbox_inches="tight")
    plt.close(fig3)

    # -------------------------------------------------------------------------
    # FIGURE 4: ROC Curves and Precision-Recall Curves (Test Set)
    # -------------------------------------------------------------------------
    fig4, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel (a): ROC Curves
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (AUC = 0.500)")
    for model_name in model_order:
        pipe = fitted_pipelines["Exp3_Tuned"][model_name]
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(
            fpr,
            tpr,
            lw=2.2,
            color=MODEL_COLORS[model_name],
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )
    axes[0].set_title("(a) Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right")

    # Panel (b): Precision-Recall Curves
    pos_rate = float(y_test.mean())
    axes[1].axhline(
        pos_rate,
        color="k",
        linestyle="--",
        lw=1.2,
        label=f"No-Skill Baseline (Prevalence = {pos_rate:.3f})",
    )
    for model_name in model_order:
        pipe = fitted_pipelines["Exp3_Tuned"][model_name]
        y_prob = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)
        axes[1].plot(
            rec,
            prec,
            lw=2.2,
            color=MODEL_COLORS[model_name],
            label=f"{model_name} (PR-AUC = {pr_auc:.3f})",
        )
    axes[1].set_title("(b) Precision-Recall (PR) Curves on Hold-Out Test Set", fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="lower left")

    fig4.tight_layout()
    fig4_png = figures_dir / "fig4_roc_pr_curves.png"
    fig4_pdf = figures_dir / "fig4_roc_pr_curves.pdf"
    fig4.savefig(fig4_png, dpi=300, bbox_inches="tight")
    fig4.savefig(fig4_pdf, bbox_inches="tight")
    plt.close(fig4)

    # -------------------------------------------------------------------------
    # FIGURE 5: Learning Curves (Bias-Variance Trade-off: Baseline vs Tuned)
    # -------------------------------------------------------------------------
    fig5, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    axes_flat = axes.flatten()
    cv_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_sizes_frac = np.linspace(0.2, 1.0, 5)

    for idx, model_name in enumerate(model_order):
        ax = axes_flat[idx]
        base_pipe = fitted_pipelines["Exp1_Baseline"][model_name]
        tuned_pipe = fitted_pipelines["Exp3_Tuned"][model_name]

        # Baseline learning curve
        sizes_b, train_b, val_b = learning_curve(
            base_pipe,
            X_train,
            y_train,
            cv=cv_lc,
            train_sizes=train_sizes_frac,
            scoring="f1",
            n_jobs=-1,
        )
        # Tuned learning curve
        sizes_t, train_t, val_t = learning_curve(
            tuned_pipe,
            X_train,
            y_train,
            cv=cv_lc,
            train_sizes=train_sizes_frac,
            scoring="f1",
            n_jobs=-1,
        )

        ax.plot(
            sizes_b,
            np.mean(train_b, axis=1),
            "--",
            color="#7f8c8d",
            lw=1.5,
            label="Baseline Train F1",
        )
        ax.plot(
            sizes_b,
            np.mean(val_b, axis=1),
            ":",
            color="#7f8c8d",
            lw=1.8,
            label="Baseline CV-Val F1",
        )

        color = MODEL_COLORS[model_name]
        train_t_mean = np.mean(train_t, axis=1)
        val_t_mean = np.mean(val_t, axis=1)
        val_t_std = np.std(val_t, axis=1)

        ax.plot(
            sizes_t,
            train_t_mean,
            "o--",
            color=color,
            lw=1.8,
            label="Tuned Train F1",
        )
        ax.plot(
            sizes_t,
            val_t_mean,
            "s-",
            color=color,
            lw=2.2,
            label="Tuned CV-Val F1",
        )
        ax.fill_between(
            sizes_t,
            val_t_mean - val_t_std,
            val_t_mean + val_t_std,
            color=color,
            alpha=0.15,
        )

        gap_base = float(np.mean(train_b[-1]) - np.mean(val_b[-1]))
        gap_tuned = float(train_t_mean[-1] - val_t_mean[-1])
        ax.set_title(
            f"({chr(97 + idx)}) {model_name}\n"
            f"Overfit Gap: Baseline={gap_base:.3f} -> Tuned={gap_tuned:.3f}",
            fontweight="bold",
        )
        ax.set_xlabel("Training Fold Sample Size")
        ax.set_ylabel("F1-Score")
        ax.set_ylim(0.40, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", ncol=2, fontsize=7.5)

    fig5.tight_layout()
    fig5_png = figures_dir / "fig5_learning_curves_bias_variance.png"
    fig5_pdf = figures_dir / "fig5_learning_curves_bias_variance.pdf"
    fig5.savefig(fig5_png, dpi=300, bbox_inches="tight")
    fig5.savefig(fig5_pdf, bbox_inches="tight")
    plt.close(fig5)

    # -------------------------------------------------------------------------
    # FIGURE 6: Experimental Progression & Permutation Feature Importance
    # -------------------------------------------------------------------------
    fig6, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # Panel (a): F1 and Sensitivity Progression Across Stages
    stage_labels = {
        "Exp1_Baseline": "Exp 1: Baseline",
        "Exp2_SMOTE": "Exp 2: SMOTE",
        "Exp3_Tuned": "Exp 3: Tuned+SMOTE",
    }
    x = np.arange(len(model_order))
    width = 0.25
    stage_colors = ["#95a5a6", "#3498db", "#2c3e50"]

    for s_idx, (stage_key, stage_title) in enumerate(stage_labels.items()):
        sub = (
            metrics_df[metrics_df["Stage"] == stage_key]
            .set_index("Model")
            .loc[model_order]
        )
        bars = axes[0].bar(
            x + (s_idx - 1) * width,
            sub["Test_F1"],
            width=width,
            label=f"{stage_title} (Test F1)",
            color=stage_colors[s_idx],
            edgecolor="black",
            linewidth=0.7,
        )
        for bar, val, rec in zip(
            bars, sub["Test_F1"], sub["Test_Recall_Sensitivity"]
        ):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.006,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
            )
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                val - 0.035,
                f"R:{rec:.2f}",
                ha="center",
                va="top",
                fontsize=6.5,
                color="white",
                fontweight="bold",
            )

    axes[0].set_title(
        "(a) Test F1-Score & Recall (R) Across Experimental Stages",
        fontweight="bold",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_order)
    axes[0].set_ylabel("Test F1-Score")
    axes[0].set_ylim(0.45, 0.85)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left")

    # Panel (b): Permutation Feature Importance (Test F1 impact) across Tuned Models
    perm_records = []
    for model_name in model_order:
        pipe = fitted_pipelines["Exp3_Tuned"][model_name]
        perm = permutation_importance(
            pipe,
            X_test,
            y_test,
            scoring="f1",
            n_repeats=30,
            random_state=random_state,
            n_jobs=-1,
        )
        imp = np.maximum(perm.importances_mean, 0)
        imp_norm = imp / imp.sum() if imp.sum() > 0 else imp
        for feat, val in zip(FEATURE_COLUMNS, imp_norm):
            perm_records.append(
                {"Feature": feat, "Model": model_name, "Relative_Importance": val}
            )

    perm_df = pd.DataFrame(perm_records)
    feat_order = (
        perm_df.groupby("Feature")["Relative_Importance"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    sns.barplot(
        data=perm_df,
        y="Feature",
        x="Relative_Importance",
        hue="Model",
        order=feat_order,
        palette=MODEL_COLORS,
        ax=axes[1],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].set_title(
        "(b) Relative Permutation Feature Importance (Test F1 Impact)",
        fontweight="bold",
    )
    axes[1].set_xlabel("Normalized Importance Weight")
    axes[1].set_ylabel("")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)
    axes[1].legend(loc="lower right")

    fig6.tight_layout()
    fig6_png = figures_dir / "fig6_feature_importance.png"
    fig6_pdf = figures_dir / "fig6_feature_importance.pdf"
    fig6.savefig(fig6_png, dpi=300, bbox_inches="tight")
    fig6.savefig(fig6_pdf, bbox_inches="tight")
    plt.close(fig6)

    return {
        "fig3_png": fig3_png,
        "fig4_png": fig4_png,
        "fig5_png": fig5_png,
        "fig6_png": fig6_png,
    }
