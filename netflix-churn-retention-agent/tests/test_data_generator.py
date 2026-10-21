"""Automated unit tests for synthetic data generation module.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import REQUIRED_COLUMNS
from src.data_generator import generate_synthetic_churn_data


def test_synthetic_data_generation_row_count():
    """Verifies that generated synthetic dataset satisfies the minimum record count."""
    df = generate_synthetic_churn_data(num_records=3000, random_seed=42)
    assert len(df) == 3000
    assert isinstance(df, pd.DataFrame)


def test_synthetic_data_schema_completeness():
    """Ensures all required columns are present in the generated dataset."""
    df = generate_synthetic_churn_data(num_records=500, random_seed=42)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_reproducibility_with_seed():
    """Confirms that identical seeds produce identical subscriber records."""
    df1 = generate_synthetic_churn_data(num_records=500, random_seed=123)
    df2 = generate_synthetic_churn_data(num_records=500, random_seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_records():
    """Confirms that different seeds produce distinct datasets."""
    df1 = generate_synthetic_churn_data(num_records=200, random_seed=1)
    df2 = generate_synthetic_churn_data(num_records=200, random_seed=2)
    assert not df1["watch_hours_last_30_days"].equals(df2["watch_hours_last_30_days"])


def test_realistic_churn_rate():
    """Ensures churn target is binary and churn rate is in a realistic subscription range."""
    df = generate_synthetic_churn_data(num_records=1000, random_seed=42)
    churn_rate = df["churned"].mean()
    assert set(df["churned"].unique()).issubset({0, 1})
    assert 0.10 <= churn_rate <= 0.35, f"Unexpected churn rate: {churn_rate:.2%}"


def test_domain_bounds_and_validity():
    """Validates numerical domain bounds on synthetic features."""
    df = generate_synthetic_churn_data(num_records=500, random_seed=42)
    assert (df["monthly_subscription_fee"] > 0).all()
    assert (df["watch_hours_last_30_days"] >= 0).all()
    assert (df["active_days_last_30_days"] >= 0).all()
    assert (df["active_days_last_30_days"] <= 31).all()
    assert (df["days_since_last_activity"] >= 0).all()
    assert (df["payment_failures_last_90_days"] >= 0).all()
