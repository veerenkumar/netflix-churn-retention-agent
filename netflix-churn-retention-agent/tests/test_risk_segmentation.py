"""Automated unit tests for customer risk segmentation and revenue exposure calculations.
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
from src.risk_segmentation import (
    filter_subscribers,
    get_segment_summary,
    segment_customers,
)


def test_segment_threshold_boundaries():
    """Verifies that risk tiers are correctly assigned across threshold boundaries."""
    raw = generate_synthetic_churn_data(100, random_seed=42).head(3)
    # Test boundary probabilities
    probs = np.array([0.75, 0.45, 0.15])
    segmented = segment_customers(raw, probs, threshold_high=0.70, threshold_low=0.40)

    assert segmented.iloc[0]["risk_category"] == "High Risk"
    assert segmented.iloc[1]["risk_category"] == "Medium Risk"
    assert segmented.iloc[2]["risk_category"] == "Low Risk"


def test_invalid_thresholds_raise_error():
    """Ensures threshold ordering violations raise ValueError."""
    raw = generate_synthetic_churn_data(100, random_seed=42).head(5)
    probs = np.array([0.5] * 5)
    with pytest.raises(ValueError):
        # threshold_low > threshold_high is invalid
        segment_customers(raw, probs, threshold_high=0.40, threshold_low=0.70)


def test_active_subscribers_isolation():
    """Verifies that is_active_subscriber separates prospective active from historical churned."""
    raw = generate_synthetic_churn_data(200, random_seed=42)
    probs = np.random.uniform(0.1, 0.9, size=200)
    segmented = segment_customers(raw, probs)

    churned_count = (raw["churned"] == 1).sum()
    active_count = (raw["churned"] == 0).sum()

    assert (segmented["is_active_subscriber"] == False).sum() == churned_count
    assert (segmented["is_active_subscriber"] == True).sum() == active_count


def test_segment_summary_metrics():
    """Verifies segment summary aggregations."""
    raw = generate_synthetic_churn_data(200, random_seed=42)
    probs = np.random.uniform(0.1, 0.9, size=200)
    segmented = segment_customers(raw, probs)
    summary = get_segment_summary(segmented, active_only=True)

    assert summary["total_subscribers"] == int(segmented["is_active_subscriber"].sum())
    assert summary["total_mrr"] > 0
    assert summary["total_mrr_at_risk"] <= summary["total_mrr"]
    assert (
        summary["high_risk_count"] + summary["medium_risk_count"] + summary["low_risk_count"]
        == summary["total_subscribers"]
    )
