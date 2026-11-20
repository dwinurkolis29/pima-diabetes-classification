"""Single entrypoint to execute data loading, EDA, 3-stage ML experiments, and publication figure generation."""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from features.dataset.eda import audit_biological_zeros, generate_eda_figures
from features.dataset.loader import create_stratified_split, load_pima_dataset
from features.evaluation.plots import generate_evaluation_figures
from features.experiments.runner import run_all_experiments


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    artifacts_dir = root_dir / "artifacts"
    raw_csv_path = artifacts_dir / "data" / "pima_diabetes_raw.csv"
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures"

    print("[1/4] Loading Pima Indians Diabetes dataset and creating 80:20 stratified split...")
    df = load_pima_dataset(raw_csv_path)
    X_train, X_test, y_train, y_test = create_stratified_split(df, test_size=0.20, random_state=42)
    print(f"      Total samples: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")

    print("[2/4] Auditing biologically invalid zeros and generating EDA figures...")
    audit_df = audit_biological_zeros(df)
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(results_dir / "eda_feature_audit.csv", index=False)
    generate_eda_figures(df, audit_df, figures_dir)

    print("[3/4] Running 3-stage experimental benchmark across Decision Tree, Random Forest, XGBoost, and SVM...")
    metrics_df, fitted_pipelines, _ = run_all_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        results_dir=results_dir,
        random_state=42,
    )

    print("[4/4] Generating evaluation plots (Confusion Matrices, ROC/PR, Learning Curves, Feature Importance)...")
    generate_evaluation_figures(
        metrics_df=metrics_df,
        fitted_pipelines=fitted_pipelines,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        figures_dir=figures_dir,
        random_state=42,
    )

    print("\n=== EXPERIMENTAL BENCHMARK SUMMARY (TEST SET) ===")
    cols = [
        "Model",
        "Stage",
        "Test_Accuracy",
        "Test_Precision",
        "Test_Recall_Sensitivity",
        "Test_Specificity",
        "Test_F1",
        "Test_ROC_AUC",
        "Gap_Train_Test_F1",
    ]
    print(metrics_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
