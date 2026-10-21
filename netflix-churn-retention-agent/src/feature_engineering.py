"""Feature engineering module for calculating behavioral signals, trends, and risk indicators.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes behavioral domain features without target leakage or division-by-zero risks.

    Args:
        df: Raw subscriber DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered behavioral columns appended.
    """
    df_feat = df.copy()

    # 1. Watch-Time Decline Percentage (Positive = viewing dropped, Negative = viewing grew)
    prev_watch = np.maximum(df_feat["watch_hours_previous_30_days"].to_numpy(dtype=float), 0.0)
    last_watch = np.maximum(df_feat["watch_hours_last_30_days"].to_numpy(dtype=float), 0.0)
    # Safe denominator with 0.1 hour epsilon
    watch_decline_pct = ((prev_watch - last_watch) / (prev_watch + 0.1)) * 100.0
    df_feat["watch_time_decline_pct"] = np.clip(np.round(watch_decline_pct, 2), -100.0, 100.0)

    # 2. Engagement Trend Categorization
    conditions = [
        df_feat["watch_time_decline_pct"] >= 40.0,
        (df_feat["watch_time_decline_pct"] >= 15.0) & (df_feat["watch_time_decline_pct"] < 40.0),
        (df_feat["watch_time_decline_pct"] >= -15.0) & (df_feat["watch_time_decline_pct"] < 15.0),
        df_feat["watch_time_decline_pct"] < -15.0,
    ]
    trend_labels = ["Severe Drop", "Moderate Drop", "Stable", "Growing"]
    df_feat["engagement_trend"] = np.select(conditions, trend_labels, default="Stable")

    # 3. Inactivity Indicator (binary flag: inactive for 14+ days)
    df_feat["inactivity_indicator"] = (df_feat["days_since_last_activity"] >= 14).astype(int)

    # 4. Payment Risk Indicator (binary flag: 1+ payment failures in past 90 days)
    df_feat["payment_risk_indicator"] = (df_feat["payment_failures_last_90_days"] >= 1).astype(int)

    # 5. Complaint Risk Indicator (binary flag: 1+ complaints or 3+ tickets)
    df_feat["complaint_risk_indicator"] = (
        (df_feat["customer_complaints_last_90_days"] >= 1)
        | (df_feat["support_tickets_last_90_days"] >= 3)
    ).astype(int)

    # 6. Marketing Engagement Rate (opened / received, bounded [0, 1])
    opened = np.maximum(df_feat["marketing_emails_opened"].to_numpy(dtype=float), 0.0)
    received = np.maximum(df_feat["marketing_emails_received"].to_numpy(dtype=float), 1.0)
    df_feat["marketing_engagement_rate"] = np.clip(np.round(opened / received, 3), 0.0, 1.0)

    # 7. Subscriber Tenure Cohort Group
    tenure_months = df_feat["subscription_tenure_months"].to_numpy(dtype=int)
    tenure_conditions = [
        tenure_months <= 6,
        (tenure_months > 6) & (tenure_months <= 18),
        (tenure_months > 18) & (tenure_months <= 36),
        tenure_months > 36,
    ]
    tenure_cohorts = ["New (1-6m)", "Established (7-18m)", "Mature (19-36m)", "Loyal (36m+)"]
    df_feat["subscriber_tenure_group"] = np.select(
        tenure_conditions, tenure_cohorts, default="Established (7-18m)"
    )

    # 8. Daily Viewing Intensity (watch hours per active day)
    active_days = np.maximum(df_feat["active_days_last_30_days"].to_numpy(dtype=float), 1.0)
    df_feat["viewing_intensity"] = np.clip(np.round(last_watch / active_days, 2), 0.0, 24.0)

    # 9. Baseline monthly revenue at risk
    df_feat["monthly_subscription_revenue"] = df_feat["monthly_subscription_fee"].copy()

    return df_feat


if __name__ == "__main__":
    from src.data_generator import generate_synthetic_churn_data

    sample = generate_synthetic_churn_data(100, random_seed=99)
    engineered = engineer_features(sample)
    print("Engineered columns sample:")
    print(
        engineered[
            [
                "customer_id",
                "watch_time_decline_pct",
                "engagement_trend",
                "inactivity_indicator",
                "payment_risk_indicator",
                "subscriber_tenure_group",
            ]
        ].head()
    )
