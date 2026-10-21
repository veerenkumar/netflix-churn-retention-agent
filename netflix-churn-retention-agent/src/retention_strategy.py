"""Personalized retention strategy recommendation engine.
"""
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import STRATEGY_CATALOG


def recommend_retention_action_for_subscriber(row: pd.Series) -> Dict[str, Any]:
    """Applies deterministic, explainable business rules to recommend tailored retention

    interventions based strictly on the subscriber's observed behavioral pattern.
    """
    customer_id = str(row.get("customer_id", "Unknown"))
    risk_cat = str(row.get("risk_category", "Low Risk"))
    genre = str(row.get("preferred_content_genre", "Top Trending"))
    plan = str(row.get("subscription_plan", "Standard"))
    decline_pct = float(row.get("watch_time_decline_pct", 0.0))
    days_inactive = int(row.get("days_since_last_activity", 0))
    pay_failures = int(row.get("payment_failures_last_90_days", 0))
    complaints = int(row.get("customer_complaints_last_90_days", 0))
    tickets = int(row.get("support_tickets_last_90_days", 0))
    prev_cancels = int(row.get("previous_cancellations", 0))
    daily_watch = float(row.get("average_daily_watch_hours", 2.0))

    # Priority 1: Involuntary payment friction (highest operational urgency)
    if pay_failures >= 1:
        strategy_key = "Frictionless Payment Resolution"
        config_data = STRATEGY_CATALOG[strategy_key]
        priority = "Urgent" if (risk_cat == "High Risk" or pay_failures >= 2) else "High"
        rationale = (
            f"Detected {pay_failures} billing failure(s) in past 90 days. "
            "Involuntary churn is preventable through proactive payment method updates."
        )
        campaign = (
            "Smart in-app billing modal + card expiration email reminder with zero streaming lockout."
        )
        indicators = f"Payment failures: {pay_failures}"

    # Priority 2: Service dissatisfaction / Complaints
    elif complaints >= 1 or tickets >= 3:
        strategy_key = "VIP Priority Care Outreach"
        config_data = STRATEGY_CATALOG[strategy_key]
        priority = "Urgent"
        rationale = (
            f"Subscriber filed {complaints} complaint(s) and {tickets} support ticket(s). "
            "Dissatisfaction requires white-glove human intervention."
        )
        campaign = "Dedicated customer care outreach, billing audit, and priority issue resolution."
        indicators = f"Complaints: {complaints}, Support tickets: {tickets}"

    # Priority 3: Prolonged inactivity (Dormancy risk)
    elif days_inactive >= 14:
        strategy_key = "Targeted Win-Back & Trending Highlight"
        config_data = STRATEGY_CATALOG[strategy_key]
        priority = "Urgent" if days_inactive >= 21 else "High"
        rationale = (
            f"Inactive for {days_inactive} consecutive days. "
            "Subscriber risks perceiving subscription as wasted recurring expense."
        )
        campaign = (
            f"Personalized 'What you missed' digest highlighting top 3 trending {genre} releases."
        )
        indicators = f"Days inactive: {days_inactive}"

    # Priority 4: High risk with prior cancellation history
    elif prev_cancels >= 1 and risk_cat == "High Risk":
        strategy_key = "High-Touch Loyalty Onboarding"
        config_data = STRATEGY_CATALOG[strategy_key]
        priority = "Urgent"
        rationale = (
            f"Subscriber has cancelled {prev_cancels} time(s) historically and shows elevated churn risk. "
            "Requires active retention anchoring."
        )
        campaign = f"Exclusive behind-the-scenes content and customized {genre} series roadmap."
        indicators = f"Previous cancellations: {prev_cancels}, High churn probability"

    # Priority 5: Declining engagement (Viewing drop)
    elif decline_pct >= 20.0:
        strategy_key = "Personalized Genre Re-engagement"
        config_data = STRATEGY_CATALOG[strategy_key]
        priority = "High" if decline_pct >= 40.0 else "Medium"
        rationale = (
            f"Watch hours fell by {decline_pct:.1f}% over the last 30 days. "
            "Signals content fatigue and viewing routine breakdown."
        )
        campaign = (
            f"Dynamic algorithmic watchlist highlighting 5 critically-acclaimed {genre} titles."
        )
        indicators = f"Watch time drop: {decline_pct:.1f}%"

    # Priority 6: Potential price/tier mismatch on Premium
    elif plan == "Premium" and daily_watch < 0.8:
        strategy_key = "Plan Right-Sizing Advisory"
        config_data = STRATEGY_CATALOG[strategy_key]
        priority = "Medium"
        rationale = (
            f"Subscriber pays $22.99 for Premium tier but only streams {daily_watch:.1f} hrs/day. "
            "Right-sizing preserves subscriber relationship rather than losing them entirely."
        )
        campaign = "Advisory suggestion to right-size to Standard ($15.49) or Ad-supported ($6.99) tier."
        indicators = f"Premium plan with low watch hours ({daily_watch:.1f} hrs/day)"

    # Default / Low Risk: Healthy customer engagement
    else:
        strategy_key = "Standard Loyalty & Perks"
        config_data = STRATEGY_CATALOG[strategy_key]
        priority = "Standard"
        rationale = (
            "Healthy streaming volume and steady engagement. "
            "Discounts are unnecessary and counter-productive."
        )
        campaign = "Platform loyalty milestones, sneak-peek trailers, and 4K feature reminders."
        indicators = "Consistent engagement, no payment or complaint friction"

    return {
        "customer_id": customer_id,
        "risk_category": risk_cat,
        "main_supporting_indicators": indicators,
        "recommended_action": strategy_key,
        "business_rationale": rationale,
        "suggested_campaign": campaign,
        "priority_level": priority,
        "estimated_intervention_cost": config_data["cost"],
        "follow_up_metric": config_data["follow_up_metric"],
        "advisory_status": "Advisory (Requires Human Approval)",
    }


def generate_retention_recommendations(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Generates structured, advisory retention recommendations for all eligible subscribers."""
    recs: List[Dict[str, Any]] = []

    for _, row in scored_df.iterrows():
        rec = recommend_retention_action_for_subscriber(row)
        recs.append(rec)

    recs_df = pd.DataFrame(recs)

    # Merge recommendations with key subscriber metrics for unified table view
    out_df = scored_df.copy()
    for col in [
        "recommended_action",
        "business_rationale",
        "suggested_campaign",
        "priority_level",
        "estimated_intervention_cost",
        "follow_up_metric",
        "advisory_status",
        "main_supporting_indicators",
    ]:
        out_df[col] = recs_df[col].values

    return out_df


if __name__ == "__main__":
    from src.data_generator import generate_synthetic_churn_data
    from src.feature_engineering import engineer_features
    from src.risk_segmentation import segment_customers

    df_test = generate_synthetic_churn_data(100, random_seed=42)
    df_eng = engineer_features(df_test)
    fake_probs = np.random.uniform(0.1, 0.9, size=100)
    df_seg = segment_customers(df_eng, fake_probs)
    recs = generate_retention_recommendations(df_seg)
    print("Sample recommendations:")
    print(
        recs[
            [
                "customer_id",
                "risk_category",
                "recommended_action",
                "priority_level",
                "estimated_intervention_cost",
            ]
        ].head()
    )
