"""Dataset loading, canonical schema standardization, and stratified splitting for Pima Indians Diabetes."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

CANONICAL_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

FEATURE_COLUMNS = CANONICAL_COLUMNS[:-1]
TARGET_COLUMN = "Outcome"

# Features where a recorded value of 0 is biologically impossible in living patients
BIOLOGICAL_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

OPENML_RENAME_MAP = {
    "preg": "Pregnancies",
    "plas": "Glucose",
    "pres": "BloodPressure",
    "skin": "SkinThickness",
    "insu": "Insulin",
    "mass": "BMI",
    "pedi": "DiabetesPedigreeFunction",
    "age": "Age",
    "class": "Outcome",
}

FALLBACK_CSV_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)


def load_pima_dataset(raw_csv_path: Path) -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset from local cache, OpenML (ID 37), or public mirror."""
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_csv_path.exists():
        df = pd.read_csv(raw_csv_path)
        if list(df.columns) == CANONICAL_COLUMNS and len(df) == 768:
            return df

    try:
        dataset = fetch_openml(data_id=37, as_frame=True, parser="auto")
        df = dataset.frame.copy()
        df = df.rename(columns=OPENML_RENAME_MAP)
        df["Outcome"] = (
            df["Outcome"]
            .astype(str)
            .map({"tested_positive": 1, "tested_negative": 0, "1": 1, "0": 0})
            .astype(int)
        )
        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[CANONICAL_COLUMNS]
    except Exception:
        df = pd.read_csv(FALLBACK_CSV_URL, header=None, names=CANONICAL_COLUMNS)
        df["Outcome"] = df["Outcome"].astype(int)

    df.to_csv(raw_csv_path, index=False)
    return df


def create_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform an 80:20 stratified train-test split prior to any imputation, scaling, or resampling."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
