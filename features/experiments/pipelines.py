"""Zero-leakage preprocessing transformers and imblearn Pipeline builders."""

from typing import List, Optional
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from features.dataset.loader import BIOLOGICAL_ZERO_COLUMNS


class BiologicalZeroToNanTransformer(BaseEstimator, TransformerMixin):
    """Replace biologically impossible zero values with np.nan inside CV folds to prevent data leakage."""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = (
            self.columns if self.columns is not None else BIOLOGICAL_ZERO_COLUMNS
        )
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            for col in self.columns_:
                if col in X_out.columns:
                    X_out[col] = X_out[col].replace(0, np.nan)
            return X_out.to_numpy(dtype=float)

        X_arr = np.array(X, dtype=float, copy=True)
        # Indices of BIOLOGICAL_ZERO_COLUMNS in canonical order: [1, 2, 3, 4, 5]
        for idx in [1, 2, 3, 4, 5]:
            if idx < X_arr.shape[1]:
                X_arr[X_arr[:, idx] == 0, idx] = np.nan
        return X_arr


def build_experiment_pipeline(
    classifier,
    use_smote: bool = False,
    smote_sampling_strategy: float = 1.0,
    smote_k_neighbors: int = 5,
    random_state: int = 42,
) -> ImbPipeline:
    """Construct a strict zero-leakage pipeline: Zero->NaN -> MedianImputer -> StandardScaler -> (Optional SMOTE) -> Classifier."""
    steps = [
        ("zero_to_nan", BiologicalZeroToNanTransformer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    sampling_strategy=smote_sampling_strategy,
                    k_neighbors=smote_k_neighbors,
                    random_state=random_state,
                ),
            )
        )
    steps.append(("classifier", classifier))
    return ImbPipeline(steps)
