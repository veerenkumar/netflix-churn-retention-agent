"""Customer risk segmentation and revenue-at-risk quantification module.
"""
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_MEDIUM_RISK_THRESHOLD,
    TARGET_COLUMN,
)


def segment_customers(
    df: pd.DataFrame,
    churn_probabilities: np.ndarray,
    threshold_high: float = DEFAULT_HIGH_RISK_THRESHOLD,
    threshold_low: float = DEFAULT_MEDIUM_RISK_THRESHOLD,
) -> pd.DataFrame:
    """Assigns churn risk categories, bounds probabilities, and quantifies revenue at risk.

    IMPORTANT: Flags active subscribers vs historical churners so only prospective active
    subscribers are surfaced for active retention interventions.

    Args:
        df: Subscriber DataFrame.
        churn_probabilities: Model-estimated churn probabilities [0, 1].
        threshold_high: Boundary above which risk is 'High Risk' (e.g. 0.70).
        threshold_low: Boundary separating 'Low Risk' and 'Medium Risk' (e.g. 0.40).

    Returns:
        pd.DataFrame: Scored DataFrame with risk columns.
    """
    if not (0.0 < threshold_low < threshold_high < 1.0):
        raise ValueError(
            f"Invalid thresholds: Must satisfy 0.0 < threshold_low ({threshold_low}) < threshold_high ({threshold_high}) < 1.0"
        )

    scored_df = df.copy()
    probs = np.clip(np.asarray(churn_probabilities, dtype=float), 0.0, 1.0)
    scored_df["churn_probability"] = np.round(probs, 4)

    # Classify Risk Tiers
    conditions = [
        scored_df["churn_probability"] >= threshold_high,
        (scored_df["churn_probability"] >= threshold_low)
        & (scored_df["churn_probability"] < threshold_high),
        scored_df["churn_probability"] < threshold_low,
    ]
    categories = ["High Risk", "Medium Risk", "Low Risk"]
    scored_df["risk_category"] = np.select(conditions, categories, default="Low Risk")

    # Quantify Monthly Revenue at Risk
    if "monthly_subscription_revenue" in scored_df.columns:
        mrr = scored_df["monthly_subscription_revenue"].astype(float)
    elif "monthly_subscription_fee" in scored_df.columns:
        mrr = scored_df["monthly_subscription_fee"].astype(float)
        scored_df["monthly_subscription_revenue"] = mrr
    else:
        mrr = pd.Series(14.99, index=scored_df.index)
        scored_df["monthly_subscription_revenue"] = mrr

    scored_df["revenue_at_risk"] = np.round(mrr * scored_df["churn_probability"], 2)

    # Segment Active Prospective vs Historical Inactive Subscribers
    if TARGET_COLUMN in scored_df.columns:
        scored_df["is_active_subscriber"] = (scored_df[TARGET_COLUMN] == 0).astype(bool)
    else:
        # Default all prospective input rows to active if unlabelled
        scored_df["is_active_subscriber"] = True

    return scored_df


def get_segment_summary(
    scored_df: pd.DataFrame,
    active_only: bool = True,
) -> Dict[str, Any]:
    """Generates executive summary KPI counts and revenue aggregations across risk tiers.

    Args:
        scored_df: Scored customer dataset with risk categories.
        active_only: Whether to restrict metrics to prospective active subscribers.

    Returns:
        Dict[str, Any]: Summary metrics across tiers.
    """
    if active_only and "is_active_subscriber" in scored_df.columns:
        df_target = scored_df[scored_df["is_active_subscriber"]].copy()
    else:
        df_target = scored_df.copy()

    total_subscribers = len(df_target)
    if total_subscribers == 0:
        return {
            "total_subscribers": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "high_risk_pct": 0.0,
            "medium_risk_pct": 0.0,
            "low_risk_pct": 0.0,
            "total_mrr": 0.0,
            "total_mrr_at_risk": 0.0,
            "high_risk_mrr_at_risk": 0.0,
        }

    high_risk_mask = df_target["risk_category"] == "High Risk"
    med_risk_mask = df_target["risk_category"] == "Medium Risk"
    low_risk_mask = df_target["risk_category"] == "Low Risk"

    high_count = int(high_risk_mask.sum())
    med_count = int(med_risk_mask.sum())
    low_count = int(low_risk_mask.sum())

    total_mrr = float(df_target["monthly_subscription_revenue"].sum())
    total_at_risk = float(df_target["revenue_at_risk"].sum())
    high_at_risk = float(df_target.loc[high_risk_mask, "revenue_at_risk"].sum())

    return {
        "total_subscribers": total_subscribers,
        "high_risk_count": high_count,
        "medium_risk_count": med_count,
        "low_risk_count": low_count,
        "high_risk_pct": round((high_count / total_subscribers) * 100, 1),
        "medium_risk_pct": round((med_count / total_subscribers) * 100, 1),
        "low_risk_pct": round((low_count / total_subscribers) * 100, 1),
        "total_mrr": round(total_mrr, 2),
        "total_mrr_at_risk": round(total_at_risk, 2),
        "high_risk_mrr_at_risk": round(high_at_risk, 2),
    }


def filter_subscribers(
    df: pd.DataFrame,
    active_only: bool = True,
    risk_categories: Optional[List[str]] = None,
    plans: Optional[List[str]] = None,
    tenure_groups: Optional[List[str]] = None,
    search_id: Optional[str] = None,
) -> pd.DataFrame:
    """Filters scored subscriber records by user-specified dashboard criteria."""
    filtered = df.copy()

    if active_only and "is_active_subscriber" in filtered.columns:
        filtered = filtered[filtered["is_active_subscriber"]]

    if risk_categories and "risk_category" in filtered.columns:
        filtered = filtered[filtered["risk_category"].isin(risk_categories)]

    if plans and "subscription_plan" in filtered.columns:
        filtered = filtered[filtered["subscription_plan"].isin(plans)]

    if tenure_groups and "subscriber_tenure_group" in filtered.columns:
        filtered = filtered[filtered["subscriber_tenure_group"].isin(tenure_groups)]

    if search_id and "customer_id" in filtered.columns:
        filtered = filtered[
            filtered["customer_id"].astype(str).str.contains(search_id.strip(), case=False)
        ]

    return filtered


if __name__ == "__main__":
    from src.data_generator import generate_synthetic_churn_data

    sample = generate_synthetic_churn_data(200, random_seed=42)
    fake_probs = np.random.uniform(0.1, 0.95, size=200)
    segmented = segment_customers(sample, fake_probs, threshold_high=0.70, threshold_low=0.40)
    summary = get_segment_summary(segmented, active_only=True)
    print("Segment summary for active subscribers:")
    print(summary)
