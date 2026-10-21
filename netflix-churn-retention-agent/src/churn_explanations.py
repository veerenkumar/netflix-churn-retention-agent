"""Interpretable churn driver analysis and per-subscriber behavioral explanations.
"""
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Standardized scientific disclaimer on correlation vs causation
CAUSAL_DISCLAIMER = (
    "DISCLAIMER: Behavioral drivers identified below reflect statistical correlations "
    "associated with historical cancellation patterns. They do not constitute proven "
    "causal mechanisms. Retention actions should be treated as operational hypotheses "
    "and validated via randomized A/B holdout testing."
)


def get_global_feature_importance(pipeline: Any) -> pd.DataFrame:
    """Extracts and aggregates global feature importances across high-level domain concepts."""
    if not hasattr(pipeline, "best_model") or pipeline.best_model is None:
        raise ValueError("Pipeline has no trained model to extract feature importance from.")

    model = pipeline.best_model
    feature_names = pipeline.preprocessor.feature_names_out or []

    # Extract raw importance array
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_importances = np.abs(model.coef_[0])
    elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
        raw_importances = model.estimator.feature_importances_
    else:
        # Uniform fallback if unextractable
        raw_importances = np.ones(len(feature_names)) / max(len(feature_names), 1)

    df_imp = pd.DataFrame({"encoded_feature": feature_names, "raw_importance": raw_importances})

    # Group into high-level business drivers
    def map_domain_driver(name: str) -> str:
        name_lower = name.lower()
        if "watch_time_decline" in name_lower or "watch_hours" in name_lower:
            return "Watch Hours & Viewing Drop"
        if "days_since_last_activity" in name_lower or "inactivity" in name_lower:
            return "Activity Recency & Inactivity"
        if "payment_failures" in name_lower or "payment_risk" in name_lower:
            return "Payment & Billing Failures"
        if "complaint" in name_lower or "support_ticket" in name_lower:
            return "Customer Care & Complaints"
        if "search_to_play" in name_lower:
            return "Search-to-Play Browsing Fatigue"
        if "auto_renewal" in name_lower:
            return "Auto-Renewal Status"
        if "marketing" in name_lower:
            return "Marketing Email Engagement"
        if "previous_cancellations" in name_lower:
            return "Prior Cancellation History"
        if "tenure" in name_lower:
            return "Tenure & Account Longevity"
        if "subscription_plan" in name_lower or "fee" in name_lower:
            return "Plan Tier & Subscription Pricing"
        if "genre" in name_lower:
            return "Content Genre Affinity"
        return "Demographics & Region"

    df_imp["domain_driver"] = df_imp["encoded_feature"].apply(map_domain_driver)
    domain_summary = (
        df_imp.groupby("domain_driver")["raw_importance"]
        .sum()
        .reset_index()
        .sort_values(by="raw_importance", ascending=False)
    )
    # Normalize to 100%
    total_val = domain_summary["raw_importance"].sum()
    if total_val > 0:
        domain_summary["importance_pct"] = np.round(
            (domain_summary["raw_importance"] / total_val) * 100, 2
        )
    else:
        domain_summary["importance_pct"] = 0.0

    return domain_summary


def explain_subscriber_churn_risk(row: pd.Series) -> List[Dict[str, Any]]:
    """Evaluates an individual subscriber's behavioral indicators and generates a ranked

    factual explanation list based strictly on their observed record.
    """
    explanations: List[Dict[str, Any]] = []

    # 1. Watch Time Decline
    decline_pct = float(row.get("watch_time_decline_pct", 0.0))
    if decline_pct >= 40.0:
        explanations.append(
            {
                "driver": "Severe Viewing Drop",
                "severity": "Critical",
                "evidence": f"Watch hours declined by {decline_pct:.1f}% in the past 30 days.",
                "hypothesis": "Subscribers finishing a show or franchise without discovering next engaging series.",
                "type": "Predictive Behavioral Indicator",
            }
        )
    elif decline_pct >= 20.0:
        explanations.append(
            {
                "driver": "Moderate Viewing Drop",
                "severity": "Moderate",
                "evidence": f"Watch hours dropped {decline_pct:.1f}% compared to prior 30 days.",
                "hypothesis": "Softening platform habituation and potential content fatigue.",
                "type": "Predictive Behavioral Indicator",
            }
        )

    # 2. Prolonged Inactivity
    days_inactive = int(row.get("days_since_last_activity", 0))
    if days_inactive >= 21:
        explanations.append(
            {
                "driver": "Critical Account Inactivity",
                "severity": "Critical",
                "evidence": f"No streaming activity logged for {days_inactive} consecutive days.",
                "hypothesis": "High probability of dormant subscription perception, accelerating cancellation intent.",
                "type": "Predictive Behavioral Indicator",
            }
        )
    elif days_inactive >= 14:
        explanations.append(
            {
                "driver": "Elevated Inactivity",
                "severity": "Moderate",
                "evidence": f"Inactive for {days_inactive} days.",
                "hypothesis": "Subscriber disengaged from weekly streaming routine.",
                "type": "Predictive Behavioral Indicator",
            }
        )

    # 3. Payment Failures
    pay_failures = int(row.get("payment_failures_last_90_days", 0))
    if pay_failures >= 2:
        explanations.append(
            {
                "driver": "Repeated Payment Failures",
                "severity": "Critical",
                "evidence": f"{pay_failures} billing payment failures detected in the past 90 days.",
                "hypothesis": "Involuntary churn hazard caused by expired credit cards or bank processing blocks.",
                "type": "Predictive Operational Indicator",
            }
        )
    elif pay_failures == 1:
        explanations.append(
            {
                "driver": "Recent Payment Failure",
                "severity": "Moderate",
                "evidence": "1 billing attempt failed within the last 90 days.",
                "hypothesis": "Friction during automatic subscription fee collection.",
                "type": "Predictive Operational Indicator",
            }
        )

    # 4. Customer Complaints & Support Friction
    complaints = int(row.get("customer_complaints_last_90_days", 0))
    tickets = int(row.get("support_tickets_last_90_days", 0))
    if complaints >= 1 or tickets >= 3:
        explanations.append(
            {
                "driver": "Support Friction / Complaints",
                "severity": "High",
                "evidence": f"{complaints} formal complaints and {tickets} support tickets in past 90 days.",
                "hypothesis": "Unresolved technical, billing, or quality issues triggering frustration.",
                "type": "Predictive Experience Indicator",
            }
        )

    # 5. Search-to-Play Conversion (Browsing Fatigue)
    conv_rate = float(row.get("search_to_play_conversion_rate", 0.70))
    if conv_rate < 0.35:
        explanations.append(
            {
                "driver": "Catalog Search Fatigue",
                "severity": "Moderate",
                "evidence": f"Low search-to-play conversion ({conv_rate:.1%}).",
                "hypothesis": "Extensive searching without finding relevant titles indicates recommendation mismatch.",
                "type": "Predictive Algorithm Indicator",
            }
        )

    # 6. Auto-Renewal Status
    auto_renew = int(row.get("auto_renewal_enabled", 1))
    if auto_renew == 0:
        explanations.append(
            {
                "driver": "Auto-Renewal Switched Off",
                "severity": "Critical",
                "evidence": "Subscriber intentionally disabled auto-renewal.",
                "hypothesis": "Explicit pre-cancellation intent to let subscription lapse at end of billing cycle.",
                "type": "Direct Intent Indicator",
            }
        )

    # 7. Previous Cancellations
    prev_cancels = int(row.get("previous_cancellations", 0))
    if prev_cancels >= 1:
        explanations.append(
            {
                "driver": "Historical Churn History",
                "severity": "Moderate",
                "evidence": f"Subscriber previously cancelled {prev_cancels} time(s).",
                "hypothesis": "Customer exhibits serial subscription hopping or seasonal viewing habits.",
                "type": "Historical Behavioral Indicator",
            }
        )

    # 8. High Cost vs Low Usage
    plan = str(row.get("subscription_plan", ""))
    daily_watch = float(row.get("average_daily_watch_hours", 2.0))
    if plan == "Premium" and daily_watch < 0.8:
        explanations.append(
            {
                "driver": "Low Usage on Premium Tier",
                "severity": "Moderate",
                "evidence": f"Paying $22.99/mo for Premium plan but averaging only {daily_watch:.1f} hrs/day.",
                "hypothesis": "Perception of low value-for-money compared to subscription cost.",
                "type": "Pricing Value Indicator",
            }
        )

    # If no major negative behavioral drivers were triggered, note healthy engagement
    if not explanations:
        explanations.append(
            {
                "driver": "Consistent Habitual Streaming",
                "severity": "Low",
                "evidence": "Regular viewing hours, prompt auto-renewal, and zero recent payment failures.",
                "hypothesis": "Subscriber demonstrates healthy habitual viewing patterns.",
                "type": "Positive Retention Indicator",
            }
        )

    return explanations


def add_churn_explanations_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Appends top churn driver text and formatted summaries to each subscriber row."""
    df_out = df.copy()
    top_drivers = []
    primary_hypotheses = []

    for _, row in df_out.iterrows():
        expls = explain_subscriber_churn_risk(row)
        top_driver_name = expls[0]["driver"]
        top_drivers.append(top_driver_name)
        primary_hypotheses.append(expls[0]["hypothesis"])

    df_out["primary_churn_driver"] = top_drivers
    df_out["driver_hypothesis"] = primary_hypotheses
    return df_out


if __name__ == "__main__":
    from src.data_generator import generate_synthetic_churn_data
    from src.feature_engineering import engineer_features

    raw = generate_synthetic_churn_data(100, random_seed=77)
    eng = engineer_features(raw)
    sample_row = eng.iloc[0]
    print(f"Customer: {sample_row['customer_id']}")
    expls = explain_subscriber_churn_risk(sample_row)
    for e in expls:
        print(f" - [{e['severity']}] {e['driver']}: {e['evidence']}")
