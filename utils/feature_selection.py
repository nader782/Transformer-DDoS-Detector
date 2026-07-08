"""
feature_selection.py
--------------------
Random-Forest-based feature selection using Gini Impurity reduction.

All fitting is performed on TRAINING data only.
"""

import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def select_features_rf(
    X_train: np.ndarray,
    y_train_raw: pd.Series,
    feature_names: list,
    n_features: int = 20,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple:
    """Train a Random Forest on *X_train* / *y_train_raw* and select the top
    *n_features* most important features (by mean decrease in Gini impurity).

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training feature matrix.
    y_train_raw : pd.Series
        Raw (un-encoded) training labels.
    feature_names : list[str]
        Column names corresponding to the columns of *X_train*.
    n_features : int
        Number of top features to select.
    n_estimators : int
        Number of trees in the Random Forest.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Parallel jobs for sklearn (-1 = all cores).
    verbose : bool
        Print progress messages.

    Returns
    -------
    selected_features : list[str]
        Names of the top *n_features* selected features.
    importances_df : pd.DataFrame
        Full feature-importance table sorted descending.
    rf_model : RandomForestClassifier
        The fitted model (can be inspected or discarded).
    """
    if verbose:
        print(
            f"[feature_selection] Training Random Forest with {n_estimators} trees …")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rf.fit(X_train, y_train_raw)

    importances = rf.feature_importances_
    importances_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    selected_features = importances_df.head(n_features)["feature"].tolist()

    if verbose:
        print(f"[feature_selection] Top {n_features} features selected:")
        print(importances_df.head(n_features).to_string(index=False))

    return selected_features, importances_df, rf


def save_selected_features(selected_features: list, path: str) -> None:
    """Persist *selected_features* list to a JSON file."""
    with open(path, "w") as f:
        json.dump(selected_features, f, indent=2)
    print(f"[feature_selection] Selected features saved → {path}")


def load_selected_features(path: str) -> list:
    """Load the previously saved list of selected feature names."""
    with open(path, "r") as f:
        features = json.load(f)
    print(
        f"[feature_selection] Loaded {len(features)} selected features from {path}")
    return features
