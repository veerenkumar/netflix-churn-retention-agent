"""Configuration settings and business constants for Netflix Customer Churn & Retention Intelligence AI Agent.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Default Artifact Paths
DEFAULT_DATASET_PATH = DATA_DIR / "netflix_subscribers_synthetic.csv"
MODEL_PATH = MODELS_DIR / "churn_classifier.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
INTERVENTIONS_LOG_PATH = DATA_DIR / "retention_interventions_log.csv"

# Prediction Window Definition
PREDICTION_WINDOW_DESCRIPTION = (
    "Churn prediction target represents subscriber cancellation within the subsequent 30-day billing cycle."
)

# Risk Threshold Defaults
DEFAULT_HIGH_RISK_THRESHOLD = 0.70
DEFAULT_MEDIUM_RISK_THRESHOLD = 0.40

# Subscription Plans and Standard Pricing
PLANS_CATALOG = {
    "Standard with Ads": 6.99,
    "Basic": 9.99,
    "Standard": 15.49,
    "Premium": 22.99,
}

AGE_GROUPS = ["18-25", "26-35", "36-50", "51-65", "65+"]
REGIONS = ["North America", "EMEA", "LATAM", "APAC"]
GENRES = [
    "Sci-Fi",
    "Crime Drama",
    "Comedy",
    "Action & Adventure",
    "Documentary",
    "Romance",
    "Anime",
    "Horror & Thriller",
]

# Primary Identifier and Target
ID_COLUMN = "customer_id"
TARGET_COLUMN = "churned"
DATE_COLUMN = "subscription_start_date"

# Raw Dataset Schema
REQUIRED_COLUMNS = [
    "customer_id",
    "subscription_plan",
    "monthly_subscription_fee",
    "subscription_start_date",
    "subscription_tenure_months",
    "age_group",
    "country_or_region",
    "average_daily_watch_hours",
    "watch_hours_last_30_days",
    "watch_hours_previous_30_days",
    "days_since_last_activity",
    "active_days_last_30_days",
    "number_of_titles_watched",
    "preferred_content_genre",
    "search_to_play_conversion_rate",
    "payment_failures_last_90_days",
    "customer_complaints_last_90_days",
    "support_tickets_last_90_days",
    "marketing_emails_received",
    "marketing_emails_opened",
    "previous_cancellations",
    "auto_renewal_enabled",
    "monthly_subscription_revenue",
    "churned",
]

# Features for Model Inputs (Leakage-free, pre-event only)
RAW_NUMERICAL_FEATURES = [
    "monthly_subscription_fee",
    "subscription_tenure_months",
    "average_daily_watch_hours",
    "watch_hours_last_30_days",
    "watch_hours_previous_30_days",
    "days_since_last_activity",
    "active_days_last_30_days",
    "number_of_titles_watched",
    "search_to_play_conversion_rate",
    "payment_failures_last_90_days",
    "customer_complaints_last_90_days",
    "support_tickets_last_90_days",
    "marketing_emails_received",
    "marketing_emails_opened",
    "previous_cancellations",
    "auto_renewal_enabled",
]

RAW_CATEGORICAL_FEATURES = [
    "subscription_plan",
    "age_group",
    "country_or_region",
    "preferred_content_genre",
]

# Business Value Assumptions (Management Configurable)
DEFAULT_CONTRIBUTION_MARGIN = 0.75  # 75% contribution margin after variable streaming/licensing costs
DEFAULT_LIFETIME_HORIZON_MONTHS = 6  # 6-month evaluation horizon for retained subscribers
DEFAULT_EXPECTED_UPLIFT = 0.20       # 20% relative retention improvement scenario

# Strategy Catalogs & Estimated Costs ($/user)
STRATEGY_CATALOG = {
    "Personalized Genre Re-engagement": {
        "cost": 0.85,
        "priority": "High",
        "description": "Dynamic algorithm-driven watchlist highlighting new releases in preferred genre.",
        "follow_up_metric": "30-day watch hours recovery (+25% target)",
    },
    "Targeted Win-Back & Trending Highlight": {
        "cost": 1.40,
        "priority": "Urgent",
        "description": "Multi-channel push and email showcase of platform top-10 trending hits.",
        "follow_up_metric": "Days since last activity reduced to < 7 days",
    },
    "Frictionless Payment Resolution": {
        "cost": 2.50,
        "priority": "Urgent",
        "description": "In-app billing update prompt, alternative payment options, and temporary grace period.",
        "follow_up_metric": "Successful billing retry within 5 days",
    },
    "VIP Priority Care Outreach": {
        "cost": 8.00,
        "priority": "Urgent",
        "description": "Proactive dedicated senior customer care ticket resolution and direct outreach.",
        "follow_up_metric": "Customer satisfaction score (CSAT) >= 4.5/5",
    },
    "Plan Right-Sizing Advisory": {
        "cost": 1.25,
        "priority": "Medium",
        "description": "Transparent recommendation to migrate to Standard or Ad-supported tier without friction.",
        "follow_up_metric": "Account retention across any tier over 90 days",
    },
    "High-Touch Loyalty Onboarding": {
        "cost": 3.50,
        "priority": "High",
        "description": "Curated feature tour, personalized preferences setup, and exclusive premiere invitations.",
        "follow_up_metric": "Retention past 6 months tenure milestone",
    },
    "Standard Loyalty & Perks": {
        "cost": 0.35,
        "priority": "Standard",
        "description": "Loyalty appreciation badges, trailer sneak-peeks, and community watch-party invites.",
        "follow_up_metric": "Sustained high NPS score",
    },
}
