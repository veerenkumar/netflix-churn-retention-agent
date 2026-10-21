"""Retention intervention tracking, execution logging, and empirical A/B lift analysis.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import INTERVENTIONS_LOG_PATH


class RetentionOutcomeTracker:
    """Manages intervention logs, human approval workflows, and empirical holdout lift analysis."""

    def __init__(self, log_path: Path = INTERVENTIONS_LOG_PATH):
        self.log_path = Path(log_path)
        self.df: pd.DataFrame = self._load_or_initialize()

    def _load_or_initialize(self) -> pd.DataFrame:
        """Loads existing intervention log or creates an empty DataFrame with proper schema."""
        if self.log_path.exists():
            try:
                return pd.read_csv(self.log_path)
            except Exception:
                pass

        schema_cols = [
            "customer_id",
            "recommended_action",
            "intervention_date",
            "is_approved",
            "approved_by",
            "is_executed",
            "is_control_group",
            "follow_up_outcome",
            "evaluation_period_days",
            "intervention_cost",
        ]
        return pd.DataFrame(columns=schema_cols)

    def log_interventions_batch(
        self,
        recommendations_df: pd.DataFrame,
        approved_by: str = "Retention Manager",
        control_group_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> int:
        """Logs a batch of recommended interventions with randomized control holdout assignment.

        Args:
            recommendations_df: Scored DataFrame with recommended actions.
            approved_by: Name or role of approving manager.
            control_group_ratio: Fraction reserved as uncontacted holdout control (default: 15%).
            random_seed: Seed for reproducible control split.

        Returns:
            int: Number of new records added to the log.
        """
        if recommendations_df.empty:
            return 0

        rng = np.random.default_rng(random_seed)
        n = len(recommendations_df)
        today_str = datetime.now().strftime("%Y-%m-%d")

        # Assign control group status
        is_control = rng.random(n) < control_group_ratio

        records: List[Dict[str, Any]] = []
        for i, (_, row) in enumerate(recommendations_df.iterrows()):
            records.append(
                {
                    "customer_id": row["customer_id"],
                    "recommended_action": row.get(
                        "recommended_action", "Personalized Re-engagement"
                    ),
                    "intervention_date": today_str,
                    "is_approved": True,
                    "approved_by": approved_by,
                    "is_executed": not bool(is_control[i]),
                    "is_control_group": bool(is_control[i]),
                    "follow_up_outcome": "Pending",
                    "evaluation_period_days": 30,
                    "intervention_cost": 0.0
                    if is_control[i]
                    else float(row.get("estimated_intervention_cost", 1.0)),
                }
            )

        new_df = pd.DataFrame(records)
        self.df = pd.concat([self.df, new_df], ignore_index=True).drop_duplicates(
            subset=["customer_id", "intervention_date", "recommended_action"], keep="last"
        )
        self.save()
        return len(new_df)

    def populate_demo_historical_outcomes(self, n_records: int = 300, random_seed: int = 42) -> None:
        """Populates realistic historical trial campaigns with completed 30-day outcomes

        to enable empirical lift evaluation demonstrations.
        """
        rng = np.random.default_rng(random_seed)
        actions = [
            "Frictionless Payment Resolution",
            "Personalized Genre Re-engagement",
            "Targeted Win-Back & Trending Highlight",
            "VIP Priority Care Outreach",
            "Plan Right-Sizing Advisory",
        ]
        base_retention_control = {
            "Frictionless Payment Resolution": 0.32,
            "Personalized Genre Re-engagement": 0.54,
            "Targeted Win-Back & Trending Highlight": 0.40,
            "VIP Priority Care Outreach": 0.48,
            "Plan Right-Sizing Advisory": 0.58,
        }
        treatment_lift = {
            "Frictionless Payment Resolution": 0.28,  # +28% retention lift
            "Personalized Genre Re-engagement": 0.16,
            "Targeted Win-Back & Trending Highlight": 0.19,
            "VIP Priority Care Outreach": 0.24,
            "Plan Right-Sizing Advisory": 0.18,
        }
        costs = {
            "Frictionless Payment Resolution": 2.50,
            "Personalized Genre Re-engagement": 0.85,
            "Targeted Win-Back & Trending Highlight": 1.40,
            "VIP Priority Care Outreach": 8.00,
            "Plan Right-Sizing Advisory": 1.25,
        }

        records = []
        for i in range(n_records):
            cid = f"NFLX-EXP-{1000 + i}"
            action = rng.choice(actions)
            is_ctrl = rng.random() < 0.20
            p_control = base_retention_control[action]
            p_retain = p_control if is_ctrl else min(p_control + treatment_lift[action], 0.95)
            retained = rng.random() < p_retain
            outcome = "Active" if retained else "Churned"
            date_logged = (datetime.now() - timedelta(days=int(rng.integers(35, 75)))).strftime(
                "%Y-%m-%d"
            )

            records.append(
                {
                    "customer_id": cid,
                    "recommended_action": action,
                    "intervention_date": date_logged,
                    "is_approved": True,
                    "approved_by": "Retention Lead (Trial Run)",
                    "is_executed": not is_ctrl,
                    "is_control_group": is_ctrl,
                    "follow_up_outcome": outcome,
                    "evaluation_period_days": 30,
                    "intervention_cost": 0.0 if is_ctrl else costs[action],
                }
            )

        self.df = pd.DataFrame(records)
        self.save()

    def compute_lift_analysis(self) -> pd.DataFrame:
        """Calculates observed empirical retention rates between treatment and control groups."""
        if self.df.empty:
            return pd.DataFrame()

        completed = self.df[self.df["follow_up_outcome"].isin(["Active", "Churned"])].copy()
        if completed.empty:
            return pd.DataFrame()

        completed["is_retained"] = (completed["follow_up_outcome"] == "Active").astype(int)

        summary_rows = []
        for action, group in completed.groupby("recommended_action"):
            treatment = group[~group["is_control_group"]]
            control = group[group["is_control_group"]]

            t_count = len(treatment)
            c_count = len(control)

            t_rate = treatment["is_retained"].mean() if t_count > 0 else 0.0
            c_rate = control["is_retained"].mean() if c_count > 0 else 0.0
            abs_lift = t_rate - c_rate
            rel_uplift = (abs_lift / c_rate * 100) if c_rate > 0 else 0.0

            summary_rows.append(
                {
                    "recommended_action": action,
                    "treatment_subscribers": t_count,
                    "control_subscribers": c_count,
                    "treatment_retention_rate": round(t_rate * 100, 1),
                    "control_retention_rate": round(c_rate * 100, 1),
                    "observed_absolute_lift_pct": round(abs_lift * 100, 1),
                    "relative_uplift_pct": round(rel_uplift, 1),
                    "total_campaign_cost": round(treatment["intervention_cost"].sum(), 2),
                }
            )

        return pd.DataFrame(summary_rows).sort_values(
            by="observed_absolute_lift_pct", ascending=False
        )

    def save(self) -> None:
        """Persists intervention tracking logs to disk."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.log_path, index=False)

    def export_csv_bytes(self) -> bytes:
        """Exports DataFrame to raw UTF-8 CSV bytes for Streamlit download."""
        return self.df.to_csv(index=False).encode("utf-8")


if __name__ == "__main__":
    tracker = RetentionOutcomeTracker()
    tracker.populate_demo_historical_outcomes(250)
    lift_df = tracker.compute_lift_analysis()
    print("Empirical Holdout Lift Analysis:")
    print(lift_df)
