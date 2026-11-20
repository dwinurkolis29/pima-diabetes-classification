"""Evaluation metrics for binary medical classification including Specificity, Sensitivity, G-Mean, and Bootstrap CIs."""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer


def specificity_score(y_true, y_pred) -> float:
    """Calculate Specificity (True Negative Rate = TN / (TN + FP))."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for Test F1-Score and ROC-AUC."""
    rng = np.random.RandomState(random_state)
    f1_samples = []
    auc_samples = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        f1_samples.append(f1_score(y_true[indices], y_pred[indices], zero_division=0))
        auc_samples.append(roc_auc_score(y_true[indices], y_prob[indices]))

    f1_ci = (
        float(np.percentile(f1_samples, 2.5)),
        float(np.percentile(f1_samples, 97.5)),
    )
    auc_ci = (
        float(np.percentile(auc_samples, 2.5)),
        float(np.percentile(auc_samples, 97.5)),
    )
    return f1_ci, auc_ci


def evaluate_fitted_pipeline(
    model_name: str,
    stage_name: str,
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, Any]:
    """Evaluate a fitted pipeline across Training, 5-Fold Stratified CV, and Hold-out Test sets."""
    # Cross-validation scores on training set
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "specificity": make_scorer(specificity_score),
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_res = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    # Fit on full training set and evaluate on Train and Test
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]

    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()

    train_acc = float(accuracy_score(y_train, y_train_pred))
    train_rec = float(recall_score(y_train, y_train_pred, zero_division=0))
    train_spec = specificity_score(y_train, y_train_pred)
    train_f1 = float(f1_score(y_train, y_train_pred, zero_division=0))
    train_auc = float(roc_auc_score(y_train, y_train_prob))

    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec = float(precision_score(y_test, y_test_pred, zero_division=0))
    test_rec = float(recall_score(y_test, y_test_pred, zero_division=0))
    test_spec = specificity_score(y_test, y_test_pred)
    test_f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    test_gmean = float(np.sqrt(test_rec * test_spec))
    test_auc = float(roc_auc_score(y_test, y_test_prob))
    test_prauc = float(average_precision_score(y_test, y_test_prob))

    f1_ci, auc_ci = bootstrap_confidence_interval(
        y_test.to_numpy(),
        y_test_pred,
        y_test_prob,
    )

    cv_f1_mean = float(np.mean(cv_res["test_f1"]))
    cv_f1_std = float(np.std(cv_res["test_f1"]))
    cv_train_f1_mean = float(np.mean(cv_res["train_f1"]))
    cv_auc_mean = float(np.mean(cv_res["test_roc_auc"]))
    cv_auc_std = float(np.std(cv_res["test_roc_auc"]))
    cv_rec_mean = float(np.mean(cv_res["test_recall"]))
    cv_spec_mean = float(np.mean(cv_res["test_specificity"]))

    return {
        "Model": model_name,
        "Stage": stage_name,
        "Train_Accuracy": round(train_acc, 4),
        "Train_Recall": round(train_rec, 4),
        "Train_Specificity": round(train_spec, 4),
        "Train_F1": round(train_f1, 4),
        "Train_ROC_AUC": round(train_auc, 4),
        "CV_Train_F1_Mean": round(cv_train_f1_mean, 4),
        "CV_F1_Mean": round(cv_f1_mean, 4),
        "CV_F1_SD": round(cv_f1_std, 4),
        "CV_Recall_Mean": round(cv_rec_mean, 4),
        "CV_Specificity_Mean": round(cv_spec_mean, 4),
        "CV_ROC_AUC_Mean": round(cv_auc_mean, 4),
        "CV_ROC_AUC_SD": round(cv_auc_std, 4),
        "Test_Accuracy": round(test_acc, 4),
        "Test_Precision": round(test_prec, 4),
        "Test_Recall_Sensitivity": round(test_rec, 4),
        "Test_Specificity": round(test_spec, 4),
        "Test_F1": round(test_f1, 4),
        "Test_GMean": round(test_gmean, 4),
        "Test_ROC_AUC": round(test_auc, 4),
        "Test_PR_AUC": round(test_prauc, 4),
        "Test_F1_CI95_Low": round(f1_ci[0], 4),
        "Test_F1_CI95_High": round(f1_ci[1], 4),
        "Test_AUC_CI95_Low": round(auc_ci[0], 4),
        "Test_AUC_CI95_High": round(auc_ci[1], 4),
        "Gap_Train_Test_F1": round(train_f1 - test_f1, 4),
        "Gap_CVTrain_CVVal_F1": round(cv_train_f1_mean - cv_f1_mean, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
