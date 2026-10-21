"""Automated unit tests for personalized retention strategy recommendation engine.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retention_strategy import (
    generate_retention_recommendations,
    recommend_retention_action_for_subscriber,
)


def test_payment_failure_triggers_payment_resolution():
    """Confirms that subscribers with payment failures receive Frictionless Payment Resolution."""
    row = pd.Series(
        {
            "customer_id": "NFLX-PAY-1",
            "payment_failures_last_90_days": 2,
            "customer_complaints_last_90_days": 0,
            "support_tickets_last_90_days": 0,
            "days_since_last_activity": 2,
            "watch_time_decline_pct": 5.0,
            "subscription_plan": "Standard",
            "risk_category": "High Risk",
        }
    )
    rec = recommend_retention_action_for_subscriber(row)
    assert rec["recommended_action"] == "Frictionless Payment Resolution"
    assert rec["priority_level"] == "Urgent"


def test_complaint_triggers_vip_care():
    """Confirms that complaints trigger VIP Priority Care Outreach."""
    row = pd.Series(
        {
            "customer_id": "NFLX-CARE-1",
            "payment_failures_last_90_days": 0,
            "customer_complaints_last_90_days": 2,
            "support_tickets_last_90_days": 4,
            "days_since_last_activity": 1,
            "watch_time_decline_pct": 0.0,
            "subscription_plan": "Standard",
            "risk_category": "High Risk",
        }
    )
    rec = recommend_retention_action_for_subscriber(row)
    assert rec["recommended_action"] == "VIP Priority Care Outreach"
    assert rec["priority_level"] == "Urgent"


def test_prolonged_inactivity_triggers_win_back():
    """Confirms that inactive subscribers receive Win-Back campaign."""
    row = pd.Series(
        {
            "customer_id": "NFLX-INACT-1",
            "payment_failures_last_90_days": 0,
            "customer_complaints_last_90_days": 0,
            "support_tickets_last_90_days": 0,
            "days_since_last_activity": 25,
            "watch_time_decline_pct": 10.0,
            "subscription_plan": "Standard",
            "risk_category": "High Risk",
        }
    )
    rec = recommend_retention_action_for_subscriber(row)
    assert rec["recommended_action"] == "Targeted Win-Back & Trending Highlight"


def test_healthy_low_risk_receives_standard_perks():
    """Confirms that healthy low-risk subscribers receive standard loyalty perks, not discounts."""
    row = pd.Series(
        {
            "customer_id": "NFLX-HEALTHY-1",
            "payment_failures_last_90_days": 0,
            "customer_complaints_last_90_days": 0,
            "support_tickets_last_90_days": 0,
            "days_since_last_activity": 1,
            "watch_time_decline_pct": -10.0,
            "subscription_plan": "Standard",
            "risk_category": "Low Risk",
        }
    )
    rec = recommend_retention_action_for_subscriber(row)
    assert rec["recommended_action"] == "Standard Loyalty & Perks"
    assert rec["priority_level"] == "Standard"


def test_batch_recommendations_schema():
    """Verifies that batch recommendation generation outputs complete required fields."""
    sample_df = pd.DataFrame(
        [
            {
                "customer_id": f"NFLX-{i}",
                "payment_failures_last_90_days": i % 2,
                "customer_complaints_last_90_days": 0,
                "support_tickets_last_90_days": 0,
                "days_since_last_activity": i * 5,
                "watch_time_decline_pct": float(i * 10),
                "subscription_plan": "Standard",
                "risk_category": "Medium Risk",
            }
            for i in range(5)
        ]
    )
    recs = generate_retention_recommendations(sample_df)
    assert len(recs) == 5
    for col in [
        "recommended_action",
        "business_rationale",
        "suggested_campaign",
        "priority_level",
        "estimated_intervention_cost",
        "follow_up_metric",
        "advisory_status",
    ]:
        assert col in recs.columns
