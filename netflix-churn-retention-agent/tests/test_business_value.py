"""Automated unit tests for business value and ROI prioritization calculations.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.business_value import (
    aggregate_strategy_prioritization,
    calculate_business_value,
    get_business_value_kpis,
)


def test_business_value_formula():
    """Verifies that Expected Net Benefit = (incremental_prob * expected_margin) - cost."""
    df = pd.DataFrame(
        [
            {
                "customer_id": "NFLX-VAL-1",
                "monthly_subscription_revenue": 20.0,
                "churn_probability": 0.80,
                "estimated_intervention_cost": 2.0,
                "priority_level": "Urgent",
                "recommended_action": "Frictionless Payment Resolution",
            }
        ]
    )
    # Margin = 20 * 0.75 * 6 = 90.0
    # Incremental retention prob = 0.80 * 0.20 = 0.16
    # Gross benefit = 0.16 * 90 = 14.40
    # Net benefit = 14.40 - 2.0 = 12.40
    val_df = calculate_business_value(
        df,
        contribution_margin_ratio=0.75,
        horizon_months=6,
        incremental_retention_uplift=0.20,
    )
    assert np.isclose(val_df.iloc[0]["expected_contribution_margin"], 90.0, atol=0.01)
    assert np.isclose(val_df.iloc[0]["incremental_retention_probability"], 0.16, atol=0.001)
    assert np.isclose(val_df.iloc[0]["expected_incremental_net_benefit"], 12.40, atol=0.01)


def test_aggregate_strategy_prioritization():
    """Verifies that strategy aggregation computes rank and portfolio metrics."""
    df = pd.DataFrame(
        [
            {
                "customer_id": f"NFLX-{i}",
                "monthly_subscription_revenue": 15.0,
                "churn_probability": 0.70 if i < 3 else 0.20,
                "revenue_at_risk": 10.5 if i < 3 else 3.0,
                "estimated_intervention_cost": 1.5,
                "priority_level": "High",
                "recommended_action": "Action A" if i < 3 else "Action B",
                "expected_contribution_margin": 67.5,
                "expected_incremental_net_benefit": 7.95 if i < 3 else 0.39,
            }
            for i in range(6)
        ]
    )
    agg = aggregate_strategy_prioritization(df)
    assert len(agg) == 2
    assert "strategic_rank" in agg.columns
    # Action A has higher total net benefit, so it should rank #1
    assert agg.iloc[0]["recommended_action"] == "Action A"
    assert agg.iloc[0]["strategic_rank"] == 1


def test_business_value_kpis():
    """Verifies executive KPI calculations."""
    df = pd.DataFrame(
        [
            {
                "estimated_intervention_cost": 10.0,
                "expected_incremental_net_benefit": 50.0,
            },
            {
                "estimated_intervention_cost": 5.0,
                "expected_incremental_net_benefit": 25.0,
            },
        ]
    )
    kpis = get_business_value_kpis(df)
    assert kpis["total_campaign_cost"] == 15.0
    assert kpis["total_expected_net_benefit"] == 75.0
    assert kpis["portfolio_roi_pct"] == 500.0
