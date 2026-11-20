"""Three-stage experimental runner comparing Decision Tree, Random Forest, XGBoost, and SVM."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features.evaluation.metrics import evaluate_fitted_pipeline
from features.experiments.pipelines import build_experiment_pipeline


def run_all_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results_dir: Path,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Run Exp 1 (Baseline), Exp 2 (Imbalance Handling via SMOTE), and Exp 3 (Hyperparameter Tuning + Imbalance)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = round(neg_count / pos_count, 4)

    all_metrics = []
    fitted_pipelines: Dict[str, Dict[str, Any]] = {
        "Exp1_Baseline": {},
        "Exp2_SMOTE": {},
        "Exp3_Tuned": {},
    }
    best_params_summary: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # STAGE 1: BASELINE (Zero-Leakage Imputation + Scaling, Default Params, No Imbalance Handling)
    # -------------------------------------------------------------------------
    baseline_models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "SVM": SVC(probability=True, random_state=random_state),
    }

    for model_name, clf in baseline_models.items():
        pipe = build_experiment_pipeline(
            clf, use_smote=False, random_state=random_state
        )
        metrics = evaluate_fitted_pipeline(
            model_name,
            "Exp1_Baseline",
            pipe,
            X_train,
            y_train,
            X_test,
            y_test,
            cv,
        )
        all_metrics.append(metrics)
        fitted_pipelines["Exp1_Baseline"][model_name] = pipe

    # -------------------------------------------------------------------------
    # STAGE 2: IMBALANCE HANDLING (SMOTE inside training folds, Default Classifier Params)
    # -------------------------------------------------------------------------
    smote_models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "SVM": SVC(probability=True, random_state=random_state),
    }

    for model_name, clf in smote_models.items():
        pipe = build_experiment_pipeline(
            clf,
            use_smote=True,
            smote_sampling_strategy=1.0,
            random_state=random_state,
        )
        metrics = evaluate_fitted_pipeline(
            model_name,
            "Exp2_SMOTE",
            pipe,
            X_train,
            y_train,
            X_test,
            y_test,
            cv,
        )
        all_metrics.append(metrics)
        fitted_pipelines["Exp2_SMOTE"][model_name] = pipe

    # -------------------------------------------------------------------------
    # STAGE 3: HYPERPARAMETER TUNING + IMBALANCE HANDLING (5-Fold Stratified GridSearchCV optimizing F1)
    # -------------------------------------------------------------------------
    tuning_configs = {
        "Decision Tree": (
            build_experiment_pipeline(
                DecisionTreeClassifier(random_state=random_state),
                use_smote=True,
                random_state=random_state,
            ),
            {
                "smote__sampling_strategy": [0.85, 1.0],
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [3, 4, 5, 6],
                "classifier__min_samples_split": [10, 20, 30],
                "classifier__min_samples_leaf": [5, 10, 15],
                "classifier__ccp_alpha": [0.0, 0.005, 0.01],
            },
        ),
        "Random Forest": (
            build_experiment_pipeline(
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                use_smote=True,
                random_state=random_state,
            ),
            {
                "smote__sampling_strategy": [0.85, 1.0],
                "classifier__n_estimators": [150, 250],
                "classifier__max_depth": [5, 7, 9],
                "classifier__min_samples_split": [8, 12],
                "classifier__min_samples_leaf": [3, 6],
                "classifier__max_features": ["sqrt", 0.5],
            },
        ),
        "XGBoost": (
            build_experiment_pipeline(
                XGBClassifier(
                    random_state=random_state,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
                use_smote=True,
                random_state=random_state,
            ),
            {
                "smote__sampling_strategy": [0.85, 1.0],
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [2, 3, 4],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8, 1.0],
                "classifier__reg_lambda": [1.0, 5.0],
                "classifier__min_child_weight": [2, 5],
            },
        ),
        "SVM": (
            build_experiment_pipeline(
                SVC(kernel="rbf", probability=True, random_state=random_state),
                use_smote=True,
                random_state=random_state,
            ),
            {
                "smote__sampling_strategy": [0.9, 1.0],
                "smote__k_neighbors": [3, 5, 7],
                "classifier__C": [0.25, 0.5, 1.0, 2.0, 4.0],
                "classifier__gamma": ["scale", 0.03, 0.05, 0.08, 0.12],
            },
        ),
    }

    for model_name, (base_pipe, param_grid) in tuning_configs.items():
        grid = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_

        best_params_summary[model_name] = {
            "best_cv_f1": round(float(grid.best_score_), 4),
            "best_params": grid.best_params_,
            "scale_pos_weight_reference": scale_pos_weight,
        }

        metrics = evaluate_fitted_pipeline(
            model_name,
            "Exp3_Tuned",
            best_pipe,
            X_train,
            y_train,
            X_test,
            y_test,
            cv,
        )
        all_metrics.append(metrics)
        fitted_pipelines["Exp3_Tuned"][model_name] = best_pipe

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = results_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    best_params_path = results_dir / "best_hyperparameters.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params_summary, f, indent=2)

    return metrics_df, fitted_pipelines, best_params_summary
