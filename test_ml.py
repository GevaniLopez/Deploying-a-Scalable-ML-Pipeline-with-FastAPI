import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import pytest

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    performance_on_categorical_slice,
)

CSV_PATH = Path("data/census.csv")

# if raw csv not present, skip tests and resume
pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason="census.csv not available in CI (tracked via DVC)"
)


# Paths and constants
DATA_PATH = str(CSV_PATH)
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def _load_small_df(n=400):
    """Load a small sample to keep the tests quick."""
    df = pd.read_csv(DATA_PATH)
    n = min(n, len(df))
    # keep the class balance for the split
    return df.sample(n=n, random_state=0)


def test_train_test_split_shapes():
    """
    Train/test split produces non-empty partitions
    that add back to the original size.
    """
    df = _load_small_df()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[LABEL]
    )
    assert len(train_df) > 0 and len(test_df) > 0
    assert len(train_df) + len(test_df) == len(df)


def test_train_model_and_inference_shapes_types():
    """
    Trains a small model, runs inference, and checks output shape/type.
    """
    df = _load_small_df()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[LABEL]
    )

    X_train, y_train, enc, lb = process_data(
        X=train_df, categorical_features=CAT_FEATURES,
        label=LABEL, training=True
    )
    model = train_model(X_train, y_train)

    X_test, y_test, _, _ = process_data(
        X=test_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=enc,
        lb=lb,
    )

    preds = inference(model, X_test)
    assert preds.shape[0] == y_test.shape[0]
    assert np.issubdtype(preds.dtype, np.number)


def test_compute_model_metrics_and_slice_function():
    """
    Metrics are in [0,1] and slice function returns reasonable values
    for a real slice.
    """
    df = _load_small_df()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[LABEL]
    )

    X_train, y_train, enc, lb = process_data(
        X=train_df, categorical_features=CAT_FEATURES,
        label=LABEL, training=True
    )
    model = train_model(X_train, y_train)

    X_test, y_test, _, _ = process_data(
        X=test_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=enc,
        lb=lb,
    )

    # Core metrics sanity
    preds = inference(model, X_test)
    p, r, f1 = compute_model_metrics(y_test, preds)
    for m in (p, r, f1):
        assert isinstance(m, float) or hasattr(m, "item")
        m_val = float(m)  # this part handles the np.float64
        assert 0.0 <= m_val <= 1.0

    # Slice metrics on one real category value
    feat = "education"
    val = test_df[feat].iloc[0]
    p, r, f1 = performance_on_categorical_slice(
        data=test_df,
        column_name=feat,
        slice_value=val,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        encoder=enc,
        lb=lb,
        model=model,
    )
    for m in (p, r, f1):
        assert isinstance(m, float)
        assert 0.0 <= m <= 1.0
