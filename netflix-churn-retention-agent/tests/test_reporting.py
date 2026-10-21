"""Automated unit tests for ReportLab PDF report generation.
"""
from pathlib import Path
import sys
import tempfile
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.reporting import build_pdf_report


def test_pdf_report_generation():
    """Tests generating a complete PDF report from valid model and portfolio data."""
    summary_kpis = {}
    model_metadata = {
        "best_model_name": "Logistic Regression",
        "total_records": 1000,
        "train_churn_rate": 0.20,
        "evaluation_results": {
            "Logistic Regression": {
                "accuracy": 0.78,
                "precision": 0.65,
                "recall": 0.72,
                "f1_score": 0.68,
                "roc_auc": 0.84,
            }
        },
    }
    risk_summary = {
        "total_subscribers": 800,
        "high_risk_count": 120,
        "medium_risk_count": 280,
        "low_risk_count": 400,
        "high_risk_pct": 15.0,
        "medium_risk_pct": 35.0,
        "low_risk_pct": 50.0,
        "total_mrr": 12000.0,
        "total_mrr_at_risk": 3200.0,
        "high_risk_mrr_at_risk": 1800.0,
    }
    drivers = pd.DataFrame(
        [
            {"domain_driver": "Watch Hours Drop", "importance_pct": 35.0},
            {"domain_driver": "Inactivity", "importance_pct": 25.0},
        ]
    )
    high_risk_subs = pd.DataFrame(
        [
            {
                "customer_id": "NFLX-100",
                "churn_probability": 0.85,
                "subscription_plan": "Premium",
                "watch_time_decline_pct": 50.0,
                "days_since_last_activity": 18,
                "recommended_action": "Targeted Win-Back",
            }
        ]
    )
    strategy_priorities = pd.DataFrame(
        [
            {
                "strategic_rank": 1,
                "recommended_action": "Frictionless Payment Resolution",
                "eligible_subscribers": 45,
                "total_intervention_cost": 112.5,
                "total_expected_net_benefit": 450.0,
                "portfolio_roi_pct": 400.0,
            }
        ]
    )
    biz_kpis = {
        "total_campaign_cost": 500.0,
        "total_expected_net_benefit": 2400.0,
        "portfolio_roi_pct": 480.0,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_pdf_name = "test_audit_report.pdf"
        success, msg, pdf_path = build_pdf_report(
            summary_kpis=summary_kpis,
            model_metadata=model_metadata,
            risk_summary=risk_summary,
            top_churn_drivers=drivers,
            high_risk_subscribers=high_risk_subs,
            strategy_priorities=strategy_priorities,
            business_kpis=biz_kpis,
            output_filename=test_pdf_name,
        )

        assert success, f"Report generation failed: {msg}"
        assert pdf_path is not None
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000, "PDF file is too small or empty"


def test_pdf_report_handles_empty_tables_gracefully():
    """Tests generating a PDF report even with empty high-risk tables without crashing."""
    success, msg, pdf_path = build_pdf_report(
        summary_kpis={},
        model_metadata={},
        risk_summary={"total_subscribers": 0},
        top_churn_drivers=pd.DataFrame(),
        high_risk_subscribers=pd.DataFrame(),
        strategy_priorities=pd.DataFrame(),
        business_kpis={"total_campaign_cost": 0, "total_expected_net_benefit": 0, "portfolio_roi_pct": 0},
        output_filename="test_empty_report.pdf",
    )
    assert success
    assert pdf_path.exists()
