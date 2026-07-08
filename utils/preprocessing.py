"""
preprocessing.py
----------------
Data cleaning, feature scaling, and label encoding utilities.

Design principle: ALL fitting operations (scaler, label binarizer) must be
called ONLY on the training split to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def remove_duplicates_and_nulls(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove duplicate rows, rows with NaN values, and rows with inf values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    verbose : bool
        Print counts of removed rows.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a reset index.
    """
    initial_shape = df.shape

    # Drop exact duplicate rows
    df = df.drop_duplicates()
    after_dup = df.shape

    # Replace ±inf with NaN so they are caught by dropna
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(inplace=True)
    after_na = df.shape

    df.reset_index(drop=True, inplace=True)

    if verbose:
        removed_dup = initial_shape[0] - after_dup[0]
        removed_na = after_dup[0] - after_na[0]
        print(
            f"[preprocessing] Original rows : {initial_shape[0]:>10,}\n"
            f"[preprocessing] Removed duplicates : {removed_dup:>10,}\n"
            f"[preprocessing] Removed NaN/Inf : {removed_na:>10,}\n"
            f"[preprocessing] Remaining rows : {after_na[0]:>10,}"
        )

    return df


# ---------------------------------------------------------------------------
# Feature / Label splitting
# ---------------------------------------------------------------------------

def split_features_labels(df: pd.DataFrame, label_col: str = "Label"):
    """Separate feature matrix X and label series y from *df*.

    Returns
    -------
    X : pd.DataFrame
        Feature columns only.
    y : pd.Series
        Label column.
    feature_names : list[str]
        List of feature column names.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y, list(X.columns)


# ---------------------------------------------------------------------------
# Scaling  (fit on train ONLY)
# ---------------------------------------------------------------------------

def fit_scaler(X_train: pd.DataFrame) -> MinMaxScaler:
    """Fit a MinMaxScaler on *X_train* and return it.

    The scaler must be fitted ONLY on training data to prevent leakage.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    return scaler


def transform_features(scaler: MinMaxScaler, X: pd.DataFrame) -> np.ndarray:
    """Apply a *fitted* scaler to feature matrix X.

    Parameters
    ----------
    scaler : MinMaxScaler
        A scaler already fitted on training data.
    X : pd.DataFrame
        Feature matrix to scale (train or test).

    Returns
    -------
    np.ndarray
        Scaled values in [0, 1].
    """
    return scaler.transform(X)


# ---------------------------------------------------------------------------
# Label encoding  (one-hot)
# ---------------------------------------------------------------------------

def fit_label_encoder(y_train: pd.Series) -> LabelBinarizer:
    """Fit a LabelBinarizer on *y_train* and return it.

    The encoder must be fitted ONLY on training data to prevent leakage.
    """
    lb = LabelBinarizer()
    lb.fit(y_train)
    return lb


def encode_labels(encoder: LabelBinarizer, y: pd.Series) -> np.ndarray:
    """Apply a *fitted* LabelBinarizer to label series y.

    Returns a one-hot encoded numpy array.
    Binary case: shape (n, 2) via manual stacking.
    Multiclass: shape (n, n_classes).
    """
    encoded = encoder.transform(y)

    # LabelBinarizer returns shape (n, 1) for binary; expand to (n, 2) so
    # the representation is always full one-hot, consistent with the doc spec.
    if encoded.shape[1] == 1:
        encoded = np.hstack([encoded, 1 - encoded])

    return encoded


def get_label_mapping(encoder: LabelBinarizer) -> dict:
    """Return a human-readable dict mapping class name → one-hot vector."""
    classes = encoder.classes_
    mapping = {}
    for cls in classes:
        vec = encode_labels(encoder, pd.Series([cls]))[0]
        mapping[cls] = vec.tolist()
    return mapping
