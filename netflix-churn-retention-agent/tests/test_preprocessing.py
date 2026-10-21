"""Automated unit tests for preprocessing, feature engineering, and data loader modules.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_generator import generate_synthetic_churn_data
from src.data_loader import validate_dataset
from src.feature_engineering import engineer_features
from src.preprocessing import ChurnDataPreprocessor


def test_feature_engineering_adds_behavioral_signals():
    """Verifies that engineer_features generates all required behavioral features."""
    raw = generate_synthetic_churn_data(200, random_seed=42)
    eng = engineer_features(raw)

    expected_features = [
        "watch_time_decline_pct",
        "engagement_trend",
        "inactivity_indicator",
        "payment_risk_indicator",
        "complaint_risk_indicator",
        "marketing_engagement_rate",
        "subscriber_tenure_group",
        "viewing_intensity",
    ]
    for feat in expected_features:
        assert feat in eng.columns, f"Missing engineered feature: {feat}"


def test_division_by_zero_safety():
    """Ensures feature engineering handles edge-case zero activity without division by zero or NaN."""
    raw = generate_synthetic_churn_data(100, random_seed=42)
    # Inject extreme edge zeros
    raw.loc[0, "watch_hours_previous_30_days"] = 0.0
    raw.loc[0, "watch_hours_last_30_days"] = 0.0
    raw.loc[0, "marketing_emails_received"] = 0
    raw.loc[0, "active_days_last_30_days"] = 0

    eng = engineer_features(raw)
    assert not np.isnan(eng.loc[0, "watch_time_decline_pct"])
    assert not np.isinf(eng.loc[0, "watch_time_decline_pct"])
    assert not np.isnan(eng.loc[0, "marketing_engagement_rate"])
    assert not np.isnan(eng.loc[0, "viewing_intensity"])


def test_preprocessor_fit_transform():
    """Tests that ChurnDataPreprocessor fits on training data and transforms test data."""
    raw = generate_synthetic_churn_data(300, random_seed=42)
    preproc = ChurnDataPreprocessor()
    df_train, df_test, y_tr, y_te = preproc.split_data(raw, test_size=0.20)

    X_train = preproc.fit_transform(df_train)
    X_test = preproc.transform(df_test)

    assert X_train.shape[0] == len(df_train)
    assert X_test.shape[0] == len(df_test)
    assert X_train.shape[1] == X_test.shape[1]
    assert preproc.is_fitted
    assert len(preproc.feature_names_out) == X_train.shape[1]


def test_data_validation_accepts_valid_data():
    """Ensures validator accepts valid schema."""
    raw = generate_synthetic_churn_data(200, random_seed=42)
    is_valid, errs, warns, stats = validate_dataset(raw)
    assert is_valid
    assert len(errs) == 0
    assert stats["total_rows"] == 200


def test_data_validation_rejects_empty_data():
    """Ensures validator rejects empty or malformed dataset."""
    is_valid, errs, warns, stats = validate_dataset(pd.DataFrame())
    assert not is_valid
    assert len(errs) > 0
