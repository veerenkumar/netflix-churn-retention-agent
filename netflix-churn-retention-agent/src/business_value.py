"""Business value quantification, intervention ROI, and strategic prioritization module.
"""
from pathlib import Path
import sys
from typing import Any, Dict
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DEFAULT_CONTRIBUTION_MARGIN,
    DEFAULT_EXPECTED_UPLIFT,
    DEFAULT_LIFETIME_HORIZON_MONTHS,
)

BUSINESS_SCENARIO_DISCLAIMER = (
    "SCENARIO NOTICE: Financial projections below are modeled business scenarios based on "
    "configured gross margin, evaluation horizon, and assumed incremental retention uplift. "
    "They do not guarantee financial returns until confirmed via randomized holdout A/B experiments."
)


def calculate_business_value(
    df: pd.DataFrame,
    contribution_margin_ratio: float = DEFAULT_CONTRIBUTION_MARGIN,
    horizon_months: int = DEFAULT_LIFETIME_HORIZON_MONTHS,
    incremental_retention_uplift: float = DEFAULT_EXPECTED_UPLIFT,
) -> pd.DataFrame:
    """Calculates granular unit economics, expected incremental net benefit, and ROI for each

    candidate retention intervention.

    Formula:
        Expected Incremental Net Benefit =
            (incremental_retention_prob × expected_contribution_margin) - intervention_cost
    """
    df_val = df.copy()

    # Ensure required columns exist
    if "monthly_subscription_revenue" in df_val.columns:
        mrr = df_val["monthly_subscription_revenue"].astype(float)
    else:
        mrr = df_val["monthly_subscription_fee"].astype(float)
        df_val["monthly_subscription_revenue"] = mrr

    churn_prob = df_val.get("churn_probability", pd.Series(0.20, index=df_val.index)).astype(float)
    cost = df_val.get("estimated_intervention_cost", pd.Series(1.0, index=df_val.index)).astype(
        float
    )

    # 1. Expected Contribution Margin over evaluation horizon (not gross revenue!)
    expected_margin = mrr * float(contribution_margin_ratio) * int(horizon_months)
    df_val["expected_contribution_margin"] = np.round(expected_margin, 2)

    # 2. Incremental Retention Probability
    # IMPORTANT: We DO NOT treat churn probability as the probability the intervention saves them.
    # Rather, incremental saved probability is a fraction (uplift) of the at-risk probability.
    incremental_saved_prob = np.clip(churn_prob * float(incremental_retention_uplift), 0.0, 1.0)
    df_val["incremental_retention_probability"] = np.round(incremental_saved_prob, 4)

    # 3. Expected Incremental Gross Benefit
    gross_benefit = incremental_saved_prob * expected_margin
    df_val["expected_gross_incremental_benefit"] = np.round(gross_benefit, 2)

    # 4. Expected Incremental Net Benefit ($)
    net_benefit = gross_benefit - cost
    df_val["expected_incremental_net_benefit"] = np.round(net_benefit, 2)

    # 5. Estimated ROI Multiple
    safe_cost = np.maximum(cost, 0.05)
    df_val["expected_roi_ratio"] = np.round((net_benefit / safe_cost), 2)

    # Priority Rank Score: combines Net Benefit with Urgency Weight
    priority_weights = {"Urgent": 1.4, "High": 1.2, "Medium": 1.0, "Standard": 0.5}
    weight_series = (
        df_val["priority_level"].map(priority_weights).fillna(1.0)
        if "priority_level" in df_val.columns
        else 1.0
    )
    df_val["prioritization_score"] = np.round(net_benefit * weight_series, 2)

    return df_val


def aggregate_strategy_prioritization(df_val: pd.DataFrame) -> pd.DataFrame:
    """Aggregates subscriber-level economics up to the campaign / strategy level for

    executive prioritization and budgeting decisions.
    """
    if "recommended_action" not in df_val.columns:
        return pd.DataFrame()

    summary = (
        df_val.groupby("recommended_action")
        .agg(
            eligible_subscribers=("customer_id", "count"),
            priority_level=("priority_level", "first"),
            avg_churn_prob=("churn_probability", "mean"),
            total_mrr=("monthly_subscription_revenue", "sum"),
            total_mrr_at_risk=("revenue_at_risk", "sum"),
            total_intervention_cost=("estimated_intervention_cost", "sum"),
            total_expected_margin_horizon=("expected_contribution_margin", "sum"),
            total_expected_net_benefit=("expected_incremental_net_benefit", "sum"),
        )
        .reset_index()
    )

    # Format percentages and rounding
    summary["avg_churn_risk_pct"] = np.round(summary["avg_churn_prob"] * 100, 1)
    summary["portfolio_roi_pct"] = np.round(
        (
            summary["total_expected_net_benefit"]
            / np.maximum(summary["total_intervention_cost"], 1.0)
        )
        * 100,
        1,
    )
    summary["total_mrr"] = np.round(summary["total_mrr"], 2)
    summary["total_mrr_at_risk"] = np.round(summary["total_mrr_at_risk"], 2)
    summary["total_intervention_cost"] = np.round(summary["total_intervention_cost"], 2)
    summary["total_expected_net_benefit"] = np.round(summary["total_expected_net_benefit"], 2)

    # Sort by Expected Incremental Net Benefit descending
    summary = summary.sort_values(by="total_expected_net_benefit", ascending=False)
    summary["strategic_rank"] = range(1, len(summary) + 1)

    return summary


def get_business_value_kpis(df_val: pd.DataFrame) -> Dict[str, Any]:
    """Computes headline portfolio economics for the executive overview."""
    total_cost = float(df_val["estimated_intervention_cost"].sum())
    total_net_benefit = float(df_val["expected_incremental_net_benefit"].sum())
    overall_roi = round((total_net_benefit / max(total_cost, 1.0)) * 100, 1)

    return {
        "total_campaign_cost": round(total_cost, 2),
        "total_expected_net_benefit": round(total_net_benefit, 2),
        "portfolio_roi_pct": overall_roi,
        "avg_net_benefit_per_user": round(float(df_val["expected_incremental_net_benefit"].mean()), 2),
    }


if __name__ == "__main__":
    from src.data_generator import generate_synthetic_churn_data
    from src.feature_engineering import engineer_features
    from src.retention_strategy import generate_retention_recommendations
    from src.risk_segmentation import segment_customers

    raw = generate_synthetic_churn_data(200, random_seed=42)
    eng = engineer_features(raw)
    seg = segment_customers(eng, np.random.uniform(0.1, 0.9, size=200))
    recs = generate_retention_recommendations(seg)
    val = calculate_business_value(recs)
    agg = aggregate_strategy_prioritization(val)
    print("Aggregated Strategy Priorities:")
    print(
        agg[
            [
                "strategic_rank",
                "recommended_action",
                "eligible_subscribers",
                "total_intervention_cost",
                "total_expected_net_benefit",
                "portfolio_roi_pct",
            ]
        ]
    )
