"""Synthetic data generator for Netflix-style customer subscription, viewing, and churn records.
"""
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Add project root to path for direct execution and module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AGE_GROUPS, GENRES, PLANS_CATALOG, REGIONS, REQUIRED_COLUMNS


def generate_synthetic_churn_data(
    num_records: int = 4000,
    random_seed: int = 42,
    snapshot_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generates a realistic, reproducible synthetic dataset of Netflix-inspired subscribers.

    Models behavioral patterns including viewing drop-offs, inactivity, payment failures,
    complaints, and search fatigue without creating a single deterministic predictor.

    Args:
        num_records: Number of fictional subscriber records to generate (minimum 3000).
        random_seed: Seed for reproducible pseudorandom generation.
        snapshot_date: Reference evaluation date for tenure and activity calculations.

    Returns:
        pd.DataFrame: Fictional subscriber dataset matching the exact schema requirements.
    """
    if num_records < 100:
        raise ValueError("num_records must be at least 100.")

    rng = np.random.default_rng(random_seed)

    if snapshot_date is None:
        snapshot_date = datetime(2026, 3, 1)

    customer_ids = [f"NFLX-{100000 + i}" for i in range(num_records)]

    # Plan distribution: Standard is most common, followed by Standard with Ads, Premium, Basic
    plan_names = list(PLANS_CATALOG.keys())
    plan_probs = [0.28, 0.17, 0.35, 0.20]  # Standard with Ads, Basic, Standard, Premium
    subscription_plans = rng.choice(plan_names, size=num_records, p=plan_probs)
    monthly_fees = np.array([PLANS_CATALOG[p] for p in subscription_plans])

    # Demographics and geography
    age_groups = rng.choice(AGE_GROUPS, size=num_records, p=[0.22, 0.35, 0.25, 0.12, 0.06])
    regions = rng.choice(REGIONS, size=num_records, p=[0.38, 0.28, 0.20, 0.14])
    preferred_genres = rng.choice(GENRES, size=num_records)

    # Tenure in months (gamma/exponential distribution: many new/recent, with loyal tail up to 48 months)
    tenure_raw = rng.gamma(shape=2.2, scale=7.5, size=num_records)
    subscription_tenure_months = np.clip(np.round(tenure_raw), 1, 60).astype(int)

    # Subscription start dates derived from tenure
    start_dates = [
        (snapshot_date - timedelta(days=int(m * 30.4 + rng.integers(0, 20)))).strftime("%Y-%m-%d")
        for m in subscription_tenure_months
    ]

    # Baseline viewing behavior
    # Average daily watch hours (0.3 to 7.0 hours)
    base_daily_watch = np.clip(rng.normal(loc=2.8, scale=1.3, size=num_records), 0.2, 7.5)
    average_daily_watch_hours = np.round(base_daily_watch, 2)

    # Watch hours previous 30 days
    watch_hours_previous_30_days = np.round(
        np.clip(base_daily_watch * 30 * rng.uniform(0.85, 1.15, size=num_records), 4.0, 220.0), 1
    )

    # Latent churn risk factors to generate realistic correlations
    # Subgroup with behavioral disengagement
    disengaged_latent = rng.random(num_records) < 0.26
    decline_factor = np.where(
        disengaged_latent,
        rng.uniform(0.15, 0.65, size=num_records),  # Sharp drop in viewing
        rng.uniform(0.80, 1.25, size=num_records),  # Stable or slightly growing
    )

    watch_hours_last_30_days = np.round(
        np.clip(watch_hours_previous_30_days * decline_factor, 0.0, 230.0), 1
    )

    # Active days in last 30 days (strongly correlated with watch hours)
    active_days_est = np.round((watch_hours_last_30_days / (average_daily_watch_hours + 0.1)) * rng.uniform(0.7, 1.1, size=num_records))
    active_days_last_30_days = np.clip(active_days_est, 0, 30).astype(int)

    # Days since last activity (inactive users have high numbers)
    days_since_active_base = np.where(
        active_days_last_30_days == 0,
        rng.integers(31, 65, size=num_records),
        np.where(
            active_days_last_30_days < 7,
            rng.integers(8, 30, size=num_records),
            rng.integers(0, 7, size=num_records),
        ),
    )
    days_since_last_activity = np.clip(days_since_active_base, 0, 90).astype(int)

    # Titles watched (correlated with watch hours)
    avg_title_duration_hours = rng.uniform(1.2, 2.2, size=num_records)
    titles_watched = np.round(watch_hours_last_30_days / avg_title_duration_hours).astype(int)
    number_of_titles_watched = np.clip(titles_watched, 0, 95)

    # Search to play conversion rate (0.1 to 0.95; lower conversion indicates difficulty finding content)
    search_to_play_conversion = np.clip(
        rng.normal(loc=0.68, scale=0.18, size=num_records), 0.08, 0.98
    )
    search_to_play_conversion_rate = np.round(search_to_play_conversion, 3)

    # Payment failures in last 90 days (Poisson distribution, heavily skewed towards 0)
    payment_failures_raw = rng.poisson(lam=0.22, size=num_records)
    # Extra failure probability if high tenure with card expirations or disengaged
    payment_failures_last_90_days = np.clip(payment_failures_raw, 0, 5).astype(int)

    # Customer complaints and support tickets
    complaints_raw = rng.poisson(lam=0.18, size=num_records)
    customer_complaints_last_90_days = np.clip(complaints_raw, 0, 4).astype(int)

    support_tickets_raw = customer_complaints_last_90_days + rng.poisson(lam=0.45, size=num_records)
    support_tickets_last_90_days = np.clip(support_tickets_raw, 0, 8).astype(int)

    # Marketing emails
    marketing_emails_received = rng.integers(2, 16, size=num_records)
    # Open rate: disengaged users open far fewer
    email_open_prob = np.where(disengaged_latent, rng.uniform(0.05, 0.25, size=num_records), rng.uniform(0.20, 0.65, size=num_records))
    marketing_emails_opened = np.clip(
        np.round(marketing_emails_received * email_open_prob).astype(int), 0, marketing_emails_received
    )

    # Previous cancellations (0, 1, 2)
    prev_cancels_raw = rng.choice([0, 1, 2], size=num_records, p=[0.78, 0.17, 0.05])
    previous_cancellations = prev_cancels_raw.astype(int)

    # Auto renewal enabled (most keep it enabled, disabling is a strong churn intent)
    auto_renew_prob = np.where(disengaged_latent, 0.72, 0.96)
    auto_renewal_enabled = (rng.random(num_records) < auto_renew_prob).astype(int)

    # Monthly subscription revenue mirrors the fee
    monthly_subscription_revenue = monthly_fees.copy()

    # --- Realistic Churn Label Generation via Multi-Factor Logit ---
    # Log-odds of churn influenced by realistic behavioral factors:
    # 1. Watch drop: (prev - last) / (prev + 1)
    watch_drop_ratio = np.clip(
        (watch_hours_previous_30_days - watch_hours_last_30_days) / (watch_hours_previous_30_days + 1.0),
        -1.0, 1.0
    )
    # 2. Prolonged inactivity
    inactivity_effect = np.where(days_since_last_activity > 21, 1.6, np.where(days_since_last_activity > 10, 0.8, -0.3))
    # 3. Payment failures
    payment_effect = payment_failures_last_90_days * 0.95
    # 4. Complaints
    complaints_effect = customer_complaints_last_90_days * 0.75
    # 5. Search frustration (low search to play conversion)
    search_frustration_effect = (0.55 - search_to_play_conversion_rate) * 1.5
    # 6. Auto renewal turned off
    no_autorenew_effect = (1 - auto_renewal_enabled) * 1.8
    # 7. Previous cancellations
    past_cancels_effect = previous_cancellations * 0.50
    # 8. Tenure protective effect (logarithmic loyalty curve)
    tenure_protective_effect = -np.log1p(subscription_tenure_months) * 0.45
    # 9. Low marketing engagement
    mktg_rate = marketing_emails_opened / (marketing_emails_received + 0.1)
    low_mktg_effect = np.where(mktg_rate < 0.15, 0.40, -0.20)
    # 10. Random realistic noise (unobserved life events, budget changes, competing streaming services)
    unobserved_noise = rng.normal(loc=0.0, scale=0.85, size=num_records)

    # Combined logit with baseline intercept targeting ~19-21% overall historical churn
    base_intercept = -1.35
    logit = (
        base_intercept
        + 1.85 * watch_drop_ratio
        + inactivity_effect
        + payment_effect
        + complaints_effect
        + search_frustration_effect
        + no_autorenew_effect
        + past_cancels_effect
        + tenure_protective_effect
        + low_mktg_effect
        + unobserved_noise
    )

    # Sigmoid function for probability
    churn_probabilities = 1.0 / (1.0 + np.exp(-logit))

    # Bernoulli trial for observed churn label
    churned = (rng.random(num_records) < churn_probabilities).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "subscription_plan": subscription_plans,
            "monthly_subscription_fee": monthly_fees,
            "subscription_start_date": start_dates,
            "subscription_tenure_months": subscription_tenure_months,
            "age_group": age_groups,
            "country_or_region": regions,
            "average_daily_watch_hours": average_daily_watch_hours,
            "watch_hours_last_30_days": watch_hours_last_30_days,
            "watch_hours_previous_30_days": watch_hours_previous_30_days,
            "days_since_last_activity": days_since_last_activity,
            "active_days_last_30_days": active_days_last_30_days,
            "number_of_titles_watched": number_of_titles_watched,
            "preferred_content_genre": preferred_genres,
            "search_to_play_conversion_rate": search_to_play_conversion_rate,
            "payment_failures_last_90_days": payment_failures_last_90_days,
            "customer_complaints_last_90_days": customer_complaints_last_90_days,
            "support_tickets_last_90_days": support_tickets_last_90_days,
            "marketing_emails_received": marketing_emails_received,
            "marketing_emails_opened": marketing_emails_opened,
            "previous_cancellations": previous_cancellations,
            "auto_renewal_enabled": auto_renewal_enabled,
            "monthly_subscription_revenue": monthly_subscription_revenue,
            "churned": churned,
        }
    )

    # Reorder columns explicitly to match schema definition
    df = df[REQUIRED_COLUMNS]
    return df


if __name__ == "__main__":
    from config import DEFAULT_DATASET_PATH

    print("Generating default synthetic dataset...")
    data = generate_synthetic_churn_data(num_records=4000, random_seed=42)
    data.to_csv(DEFAULT_DATASET_PATH, index=False)
    print(f"Saved {len(data)} records to {DEFAULT_DATASET_PATH}")
    print(f"Overall churn rate: {data['churned'].mean():.2%}")
