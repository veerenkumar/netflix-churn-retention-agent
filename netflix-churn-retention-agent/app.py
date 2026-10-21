"""Netflix Customer Churn & Retention Intelligence AI Agent

Interactive Management Dashboard built with Streamlit and Plotly.
"""
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DEFAULT_CONTRIBUTION_MARGIN,
    DEFAULT_DATASET_PATH,
    DEFAULT_EXPECTED_UPLIFT,
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_LIFETIME_HORIZON_MONTHS,
    DEFAULT_MEDIUM_RISK_THRESHOLD,
    MODEL_PATH,
    PREDICTION_WINDOW_DESCRIPTION,
)
from src.business_value import (
    aggregate_strategy_prioritization,
    calculate_business_value,
    get_business_value_kpis,
)
from src.churn_explanations import (
    CAUSAL_DISCLAIMER,
    explain_subscriber_churn_risk,
    get_global_feature_importance,
)
from src.churn_model import ChurnModelPipeline
from src.data_generator import generate_synthetic_churn_data
from src.data_loader import load_dataset, validate_dataset
from src.feature_engineering import engineer_features
from src.outcome_tracking import RetentionOutcomeTracker
from src.reporting import build_pdf_report
from src.retention_strategy import generate_retention_recommendations
from src.risk_segmentation import (
    filter_subscribers,
    get_segment_summary,
    segment_customers,
)

# Page Setup
st.set_page_config(
    page_title="Netflix Churn & Retention Intelligence Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Netflix Entertainment Theme)
st.markdown(
    """
    <style>
    .main {
        background-color: #141414;
        color: #FFFFFF;
    }
    .stMetric {
        background: #1f1f1f;
        border: 1px solid #2a2a2a;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #E50914;
    }
    .stMetric label {
        color: #a3a3a3 !important;
        font-size: 0.85rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    .badge-urgent {
        background-color: #E50914;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    .badge-high {
        background-color: #ff8c00;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    .badge-low {
        background-color: #22c55e;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    .disclaimer-box {
        background-color: #1a1a1a;
        border-left: 3px solid #ff9800;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #cccccc;
        margin-bottom: 12px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Helper Cache Functions
@st.cache_data(show_spinner=False)
def get_or_create_default_dataset() -> pd.DataFrame:
    """Loads existing default synthetic dataset or generates one if missing."""
    if DEFAULT_DATASET_PATH.exists():
        try:
            return pd.read_csv(DEFAULT_DATASET_PATH)
        except Exception:
            pass
    df = generate_synthetic_churn_data(4000, random_seed=42)
    df.to_csv(DEFAULT_DATASET_PATH, index=False)
    return df


@st.cache_resource(show_spinner=False)
def get_or_train_default_model(df: pd.DataFrame) -> ChurnModelPipeline:
    """Loads trained model artifact or fits a pipeline if none exists."""
    if MODEL_PATH.exists():
        try:
            return ChurnModelPipeline.load(MODEL_PATH)
        except Exception:
            pass
    pipeline = ChurnModelPipeline()
    pipeline.train_and_evaluate(df)
    pipeline.save()
    return pipeline


# Initialize Session State
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = get_or_create_default_dataset()

if "model_pipeline" not in st.session_state:
    st.session_state["model_pipeline"] = get_or_train_default_model(st.session_state["raw_df"])

if "outcome_tracker" not in st.session_state:
    st.session_state["outcome_tracker"] = RetentionOutcomeTracker()
    if st.session_state["outcome_tracker"].df.empty:
        st.session_state["outcome_tracker"].populate_demo_historical_outcomes(250)

# Sidebar Controls & Navigation
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
        width=150,
    )
    st.markdown("### **Retention Intelligence AI**")
    st.caption("AI-Powered Churn Analytics & Strategy Engine")

    nav_selection = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "📈 Churn Analytics",
            "🔍 Subscriber Risk Explorer",
            "🎯 Retention Strategy Center",
            "🤖 Model Performance & Evaluation",
            "📁 Data Management",
            "🧪 Retention Outcome Tracking",
            "📄 PDF Executive Report",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### **Risk Thresholds**")
    high_threshold = st.slider(
        "High Risk Threshold",
        min_value=0.50,
        max_value=0.90,
        value=DEFAULT_HIGH_RISK_THRESHOLD,
        step=0.05,
        help="Subscribers with churn probability above this are classified as High Risk.",
    )
    med_threshold = st.slider(
        "Medium Risk Threshold",
        min_value=0.20,
        max_value=high_threshold - 0.05,
        value=DEFAULT_MEDIUM_RISK_THRESHOLD,
        step=0.05,
        help="Subscribers between this and High Risk are classified as Medium Risk.",
    )

    st.markdown("---")
    st.markdown("#### **Unit Economics Assumptions**")
    horizon_months = st.slider(
        "Lifetime Horizon (Months)",
        min_value=1,
        max_value=18,
        value=DEFAULT_LIFETIME_HORIZON_MONTHS,
        help="Months of subscription value evaluated for retained accounts.",
    )
    margin_ratio = (
        st.slider(
            "Contribution Margin %",
            min_value=40,
            max_value=95,
            value=int(DEFAULT_CONTRIBUTION_MARGIN * 100),
            step=5,
        )
        / 100.0
    )
    uplift_rate = (
        st.slider(
            "Expected Intervention Uplift %",
            min_value=5,
            max_value=50,
            value=int(DEFAULT_EXPECTED_UPLIFT * 100),
            step=5,
            help="Relative reduction in churn achieved by successful retention action.",
        )
        / 100.0
    )

    st.markdown("---")
    st.caption("⚡ Independent Educational Prototype\nNot affiliated with Netflix Inc.")


# Shared Data Pipeline Processing
@st.cache_data(show_spinner=False)
def compute_scored_portfolio(
    df: pd.DataFrame,
    high_t: float,
    med_t: float,
    margin_r: float,
    horizon_m: int,
    uplift_r: float,
    _model_pipe: ChurnModelPipeline,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    # 1. Engineer behavioral features (watch drop %, trends, inactivity, tenure cohorts)
    df_engineered = engineer_features(df)
    # 2. Predict churn probabilities
    probs = _model_pipe.predict_proba(df_engineered)
    # 3. Classify risk tiers and revenue at risk
    segmented = segment_customers(df_engineered, probs, threshold_high=high_t, threshold_low=med_t)
    # 4. Generate explainable retention recommendations
    recommended = generate_retention_recommendations(segmented)
    # 5. Quantify business value and expected net benefit
    valued = calculate_business_value(
        recommended,
        contribution_margin_ratio=margin_r,
        horizon_months=horizon_m,
        incremental_retention_uplift=uplift_r,
    )
    seg_summary = get_segment_summary(valued, active_only=True)
    strat_agg = aggregate_strategy_prioritization(valued[valued["is_active_subscriber"]])
    biz_kpis = get_business_value_kpis(valued[valued["is_active_subscriber"]])
    return valued, seg_summary, strat_agg, biz_kpis


pipeline = st.session_state["model_pipeline"]
df_raw = st.session_state["raw_df"]

if pipeline.is_trained:
    scored_portfolio, segment_summary, strategy_agg, portfolio_kpis = compute_scored_portfolio(
        df_raw,
        high_threshold,
        med_threshold,
        margin_ratio,
        horizon_months,
        uplift_rate,
        pipeline,
    )
    active_subscribers = scored_portfolio[scored_portfolio["is_active_subscriber"]]
else:
    st.error("⚠️ Machine learning model is not yet trained. Please visit the Data Management tab.")
    st.stop()


# -------------------------------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------------------------------------------
if nav_selection == "📊 Executive Overview":
    st.title("🎬 Netflix Retention Intelligence Executive Overview")
    st.caption(
        f"Prediction Window: **{PREDICTION_WINDOW_DESCRIPTION}** | Target: Prospective Active Subscribers"
    )

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Eligible Active Subscribers",
            value=f"{segment_summary['total_subscribers']:,}",
            help="Active subscribers evaluated for proactive retention intervention.",
        )
    with col2:
        st.metric(
            label="Monthly Revenue at Risk",
            value=f"${segment_summary['total_mrr_at_risk']:,.2f}",
            delta=f"${segment_summary['high_risk_mrr_at_risk']:,.2f} Critical (High Risk)",
            delta_color="inverse",
            help="Total expected monthly revenue loss if predicted churn occurs.",
        )
    with col3:
        st.metric(
            label="High-Risk Subscribers",
            value=f"{segment_summary['high_risk_count']:,}",
            delta=f"{segment_summary['high_risk_pct']:.1f}% of active base",
            delta_color="inverse",
            help=f"Subscribers with estimated churn probability >= {high_threshold:.0%}",
        )
    with col4:
        st.metric(
            label="Expected Net ROI",
            value=f"{portfolio_kpis['portfolio_roi_pct']:.0f}%",
            delta=f"+${portfolio_kpis['total_expected_net_benefit']:,.2f} Net Gain",
            help="Modeled net incremental benefit from executing targeted retention campaigns.",
        )

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.subheader("Subscribers by Risk Segment")
        donut_data = pd.DataFrame(
            {
                "Tier": ["High Risk", "Medium Risk", "Low Risk"],
                "Count": [
                    segment_summary["high_risk_count"],
                    segment_summary["medium_risk_count"],
                    segment_summary["low_risk_count"],
                ],
            }
        )
        fig_donut = px.pie(
            donut_data,
            names="Tier",
            values="Count",
            hole=0.55,
            color="Tier",
            color_discrete_map={
                "High Risk": "#E50914",
                "Medium Risk": "#FFA000",
                "Low Risk": "#2E7D32",
            },
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.subheader("Monthly Revenue Exposure by Tier ($ MRR)")
        mrr_bar_data = pd.DataFrame(
            {
                "Segment": ["High Risk", "Medium Risk", "Low Risk"],
                "MRR_At_Risk": [
                    segment_summary["high_risk_mrr_at_risk"],
                    segment_summary["total_mrr_at_risk"]
                    - segment_summary["high_risk_mrr_at_risk"],
                    0.0,
                ],
            }
        )
        fig_mrr = px.bar(
            mrr_bar_data,
            x="Segment",
            y="MRR_At_Risk",
            color="Segment",
            text="MRR_At_Risk",
            color_discrete_map={
                "High Risk": "#E50914",
                "Medium Risk": "#FFA000",
                "Low Risk": "#2E7D32",
            },
        )
        fig_mrr.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_mrr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            xaxis_title="",
            yaxis_title="Monthly Revenue ($)",
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_mrr, use_container_width=True)

    # Strategy Highlights Alert
    st.subheader("⚡ Top Priority Campaign Recommendations")
    if not strategy_agg.empty:
        st.dataframe(
            strategy_agg[
                [
                    "strategic_rank",
                    "recommended_action",
                    "priority_level",
                    "eligible_subscribers",
                    "total_intervention_cost",
                    "total_expected_net_benefit",
                    "portfolio_roi_pct",
                ]
            ].rename(
                columns={
                    "strategic_rank": "Rank",
                    "recommended_action": "Intervention Campaign",
                    "priority_level": "Priority",
                    "eligible_subscribers": "Eligible Subscribers",
                    "total_intervention_cost": "Est. Cost ($)",
                    "total_expected_net_benefit": "Net Benefit ($)",
                    "portfolio_roi_pct": "ROI (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# -------------------------------------------------------------------------------------------------
# PAGE 2: CHURN ANALYTICS
# -------------------------------------------------------------------------------------------------
elif nav_selection == "📈 Churn Analytics":
    st.title("📈 Behavioral Churn Analytics")
    st.caption(
        "Exploration of behavioral disengagement, payment friction, and subscription demographics."
    )

    tab1, tab2, tab3 = st.tabs(
        ["📊 Churn by Plan & Tenure", "📉 Viewing Drop & Activity", "💳 Payment & Complaints"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Observed Churn Rate by Subscription Plan")
            plan_churn = (
                df_raw.groupby("subscription_plan")["churned"]
                .agg(churn_rate="mean", total="count")
                .reset_index()
            )
            plan_churn["churn_pct"] = plan_churn["churn_rate"] * 100
            fig_plan = px.bar(
                plan_churn,
                x="subscription_plan",
                y="churn_pct",
                color="subscription_plan",
                text="churn_pct",
                color_discrete_sequence=["#E50914", "#D81F26", "#B81D24", "#831010"],
            )
            fig_plan.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_plan.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#FFFFFF",
                yaxis_title="Historical Churn Rate (%)",
                xaxis_title="",
                showlegend=False,
            )
            st.plotly_chart(fig_plan, use_container_width=True)

        with c2:
            st.markdown("#### Churn Rate by Tenure Cohort")
            if "subscriber_tenure_group" in scored_portfolio.columns:
                tenure_churn = (
                    scored_portfolio.groupby("subscriber_tenure_group")["churned"]
                    .agg(churn_rate="mean")
                    .reset_index()
                )
                tenure_churn["churn_pct"] = tenure_churn["churn_rate"] * 100
                fig_tenure = px.bar(
                    tenure_churn,
                    x="subscriber_tenure_group",
                    y="churn_pct",
                    color="churn_pct",
                    color_continuous_scale="Reds",
                    text="churn_pct",
                )
                fig_tenure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_tenure.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#FFFFFF",
                    yaxis_title="Churn Rate (%)",
                    xaxis_title="",
                )
                st.plotly_chart(fig_tenure, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 30-Day Watch Time Decline Distribution")
            fig_watch = px.histogram(
                scored_portfolio,
                x="watch_time_decline_pct",
                color="risk_category",
                nbins=40,
                barmode="overlay",
                color_discrete_map={
                    "High Risk": "#E50914",
                    "Medium Risk": "#FFA000",
                    "Low Risk": "#2E7D32",
                },
            )
            fig_watch.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#FFFFFF",
                xaxis_title="Watch Time Decline (%)",
                yaxis_title="Subscriber Count",
            )
            st.plotly_chart(fig_watch, use_container_width=True)

        with c2:
            st.markdown("#### Inactivity Recency (Days Since Last Activity)")
            fig_box = px.box(
                scored_portfolio,
                x="risk_category",
                y="days_since_last_activity",
                color="risk_category",
                color_discrete_map={
                    "High Risk": "#E50914",
                    "Medium Risk": "#FFA000",
                    "Low Risk": "#2E7D32",
                },
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#FFFFFF",
                yaxis_title="Days Inactive",
                xaxis_title="",
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Payment Failures in Past 90 Days vs Churn")
            pay_churn = (
                df_raw.groupby("payment_failures_last_90_days")["churned"]
                .agg(churn_rate="mean")
                .reset_index()
            )
            pay_churn["churn_pct"] = pay_churn["churn_rate"] * 100
            fig_pay = px.bar(
                pay_churn,
                x="payment_failures_last_90_days",
                y="churn_pct",
                text="churn_pct",
                color_discrete_sequence=["#E50914"],
            )
            fig_pay.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_pay.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#FFFFFF",
                xaxis_title="Payment Failures",
                yaxis_title="Historical Churn Rate (%)",
            )
            st.plotly_chart(fig_pay, use_container_width=True)

        with c2:
            st.markdown("#### Complaints in Past 90 Days vs Churn")
            comp_churn = (
                df_raw.groupby("customer_complaints_last_90_days")["churned"]
                .agg(churn_rate="mean")
                .reset_index()
            )
            comp_churn["churn_pct"] = comp_churn["churn_rate"] * 100
            fig_comp = px.bar(
                comp_churn,
                x="customer_complaints_last_90_days",
                y="churn_pct",
                text="churn_pct",
                color_discrete_sequence=["#FFA000"],
            )
            fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_comp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#FFFFFF",
                xaxis_title="Complaints Count",
                yaxis_title="Historical Churn Rate (%)",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Global Model Feature Importance (Aggregated Domain Concepts)")
    global_drivers = get_global_feature_importance(pipeline)
    fig_imp = px.bar(
        global_drivers,
        x="importance_pct",
        y="domain_driver",
        orientation="h",
        text="importance_pct",
        color="importance_pct",
        color_continuous_scale="Reds",
    )
    fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF",
        xaxis_title="Relative Importance (%)",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown(f"<div class='disclaimer-box'>{CAUSAL_DISCLAIMER}</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------------------------------------
# PAGE 3: SUBSCRIBER RISK EXPLORER
# -------------------------------------------------------------------------------------------------
elif nav_selection == "🔍 Subscriber Risk Explorer":
    st.title("🔍 Subscriber Risk Explorer & Individual Profiling")
    st.caption("Inspect individual subscriber health, behavioral signals, and tailored interventions.")

    # Filter Controls
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        active_only = st.checkbox("Active Subscribers Only", value=True)
    with fc2:
        risk_filter = st.multiselect(
            "Filter Risk Tier",
            options=["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk"],
        )
    with fc3:
        plan_filter = st.multiselect(
            "Filter Subscription Plan",
            options=list(scored_portfolio["subscription_plan"].unique()),
            default=[],
        )
    with fc4:
        search_query = st.text_input("Search Customer ID", placeholder="e.g. NFLX-100012")

    filtered_df = filter_subscribers(
        scored_portfolio,
        active_only=active_only,
        risk_categories=risk_filter if risk_filter else None,
        plans=plan_filter if plan_filter else None,
        search_id=search_query if search_query else None,
    )

    st.markdown(f"**Showing {len(filtered_df):,} matching subscribers**")

    # Display Table
    display_cols = [
        "customer_id",
        "churn_probability",
        "risk_category",
        "subscription_plan",
        "subscription_tenure_months",
        "watch_time_decline_pct",
        "days_since_last_activity",
        "payment_failures_last_90_days",
        "recommended_action",
        "priority_level",
        "monthly_subscription_revenue",
        "revenue_at_risk",
    ]
    st.dataframe(
        filtered_df[display_cols]
        .rename(
            columns={
                "customer_id": "Subscriber ID",
                "churn_probability": "Churn Prob",
                "risk_category": "Risk Tier",
                "subscription_plan": "Plan",
                "subscription_tenure_months": "Tenure (mo)",
                "watch_time_decline_pct": "Watch Drop %",
                "days_since_last_activity": "Days Inactive",
                "payment_failures_last_90_days": "Pay Fails",
                "recommended_action": "Recommended Action",
                "priority_level": "Priority",
                "monthly_subscription_revenue": "MRR ($)",
                "revenue_at_risk": "Risk MRR ($)",
            }
        )
        .style.format(
            {
                "Churn Prob": "{:.1%}",
                "Watch Drop %": "{:.1f}%",
                "MRR ($)": "${:.2f}",
                "Risk MRR ($)": "${:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # CSV Download Button
    csv_bytes = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Predictions as CSV",
        data=csv_bytes,
        file_name=f"netflix_churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("🔬 Deep-Dive Subscriber Risk Explanation")

    selected_cid = st.selectbox(
        "Select Subscriber to Inspect Detailed Indicators & Recommended Strategy:",
        options=filtered_df["customer_id"].head(200).tolist()
        if not filtered_df.empty
        else ["None"],
    )

    if selected_cid and selected_cid != "None":
        row_match = filtered_df[filtered_df["customer_id"] == selected_cid]
        if not row_match.empty:
            sub_row = row_match.iloc[0]
            expls = explain_subscriber_churn_risk(sub_row)

            sc1, sc2 = st.columns([1, 1])
            with sc1:
                st.markdown(f"#### Subscriber Profile: `{selected_cid}`")
                st.write(f"**Subscription Plan:** {sub_row['subscription_plan']} (${sub_row['monthly_subscription_revenue']:.2f}/mo)")
                st.write(f"**Estimated Churn Probability:** `{sub_row['churn_probability']:.1%}` ({sub_row['risk_category']})")
                st.write(f"**Tenure:** {sub_row['subscription_tenure_months']} months ({sub_row.get('subscriber_tenure_group', 'Established')})")
                st.write(f"**Preferred Genre:** {sub_row['preferred_content_genre']}")
                st.write(f"**Recent 30-Day Watch Hours:** {sub_row['watch_hours_last_30_days']} hrs (Prior: {sub_row['watch_hours_previous_30_days']} hrs)")
                st.write(f"**Days Since Last Streamed:** {sub_row['days_since_last_activity']} days")

            with sc2:
                st.markdown("#### Primary Churn Drivers (Factual Indicators)")
                for exp in expls:
                    st.markdown(
                        f"- **[{exp['severity']}] {exp['driver']}**: {exp['evidence']}<br/>"
                        f"  *Hypothesis:* _{exp['hypothesis']}_",
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Advisory Retention Recommendation")
                st.info(
                    f"**Action:** {sub_row['recommended_action']} ({sub_row['priority_level']} Priority)\n\n"
                    f"**Rationale:** {sub_row['business_rationale']}\n\n"
                    f"**Suggested Campaign:** {sub_row['suggested_campaign']}\n\n"
                    f"**Intervention Cost:** ${sub_row['estimated_intervention_cost']:.2f} | **Metric:** {sub_row['follow_up_metric']}"
                )


# -------------------------------------------------------------------------------------------------
# PAGE 4: RETENTION STRATEGY CENTER
# -------------------------------------------------------------------------------------------------
elif nav_selection == "🎯 Retention Strategy Center":
    st.title("🎯 Personalized Retention Strategy Center")
    st.caption("Strategic budgeting, portfolio ROI estimation, and campaign prioritization.")

    st.markdown(
        f"<div class='disclaimer-box'><b>Hypothetical Business Scenario:</b> Calculations below model expected contribution margin "
        f"over a <b>{horizon_months}-month horizon</b> at <b>{margin_ratio:.0%} margin</b> with an assumed <b>{uplift_rate:.0%} relative retention uplift</b>. "
        f"Interventions require human operational review before execution.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Aggregated Campaign Prioritization Table")
    st.dataframe(
        strategy_agg.rename(
            columns={
                "strategic_rank": "Priority Rank",
                "recommended_action": "Intervention Campaign",
                "priority_level": "Urgency",
                "eligible_subscribers": "Eligible Accounts",
                "total_mrr": "Current MRR ($)",
                "total_mrr_at_risk": "MRR at Risk ($)",
                "total_intervention_cost": "Campaign Cost ($)",
                "total_expected_net_benefit": "Expected Net Benefit ($)",
                "portfolio_roi_pct": "Expected ROI (%)",
            }
        ).style.format(
            {
                "Current MRR ($)": "${:,.2f}",
                "MRR at Risk ($)": "${:,.2f}",
                "Campaign Cost ($)": "${:,.2f}",
                "Expected Net Benefit ($)": "${:,.2f}",
                "Expected ROI (%)": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    bc1, bc2 = st.columns(2)
    with bc1:
        st.subheader("Expected Net Benefit by Strategy ($)")
        fig_net = px.bar(
            strategy_agg,
            x="total_expected_net_benefit",
            y="recommended_action",
            orientation="h",
            color="total_expected_net_benefit",
            color_continuous_scale="Viridis",
            text="total_expected_net_benefit",
        )
        fig_net.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_net.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            xaxis_title="Net Incremental Benefit ($)",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_net, use_container_width=True)

    with bc2:
        st.subheader("Intervention Cost vs Return Matrix")
        fig_scatter = px.scatter(
            strategy_agg,
            x="total_intervention_cost",
            y="total_expected_net_benefit",
            size="eligible_subscribers",
            color="recommended_action",
            hover_name="recommended_action",
            text="recommended_action",
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            xaxis_title="Total Campaign Cost ($)",
            yaxis_title="Expected Net Benefit ($)",
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Batch Approval to Tracking Workflow
    st.markdown("---")
    st.subheader("📝 Batch Approve & Enqueue for Retention Campaign Tracking")
    st.write(
        "Authorize current recommended interventions for prospective active subscribers to track outcomes in the A/B testing engine."
    )
    if st.button("🚀 Approve & Enqueue Recommended Interventions (with 15% Holdout Control)"):
        count_logged = st.session_state["outcome_tracker"].log_interventions_batch(
            active_subscribers, approved_by="Marketing & Retention Lead", control_group_ratio=0.15
        )
        st.success(
            f"Successfully queued {count_logged:,} retention interventions into the tracking log (15% reserved as holdout control)!"
        )


# -------------------------------------------------------------------------------------------------
# PAGE 5: MODEL PERFORMANCE & EVALUATION
# -------------------------------------------------------------------------------------------------
elif nav_selection == "🤖 Model Performance & Evaluation":
    st.title("🤖 Machine Learning Model Performance & Evaluation")
    st.caption(
        "Empirical validation results on held-out test data (80/20 stratified split). No fabricated metrics."
    )

    eval_results = pipeline.evaluation_results
    best_name = pipeline.best_model_name

    st.markdown(f"**Primary Model Selected:** `{best_name}` (Optimized for Balanced F1/ROC-AUC)")

    # Model Comparison Table
    comp_rows = []
    for m_name, res in eval_results.items():
        comp_rows.append(
            {
                "Candidate Model": m_name,
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall (Sensitivity)": res["recall"],
                "F1-Score": res["f1_score"],
                "ROC-AUC": res["roc_auc"],
                "PR-AUC": res["pr_auc"],
                "Brier Score (Calibration)": res["brier_score"],
            }
        )

    st.dataframe(
        pd.DataFrame(comp_rows).style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall (Sensitivity)": "{:.3f}",
                "F1-Score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
                "Brier Score (Calibration)": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Trade-off Analysis Callout
    st.info(
        "💡 **Business Metric Trade-off Note:** In customer churn prevention, **False Negatives** (failing to identify a churning subscriber) "
        "cost up to 6 months of lost recurring subscription revenue (~$90-$138). **False Positives** (sending an advisory email or curated watchlist "
        "to a loyal customer) carry negligible marginal cost ($0.85). Hence, models with higher **Recall** and **F1-Score** are strategically prioritized."
    )

    c_cm, c_roc = st.columns(2)

    with c_cm:
        st.subheader("Confusion Matrix (Held-out Test Data)")
        best_cm = eval_results.get(best_name, {}).get("confusion_matrix", {})
        tn = best_cm.get("true_negatives", 0)
        fp = best_cm.get("false_positives", 0)
        fn = best_cm.get("false_negatives", 0)
        tp = best_cm.get("true_positives", 0)

        cm_matrix = np.array([[tn, fp], [fn, tp]])
        fig_cm = px.imshow(
            cm_matrix,
            text_auto=True,
            x=["Predicted: Retained", "Predicted: Churn"],
            y=["Actual: Retained", "Actual: Churn"],
            color_continuous_scale="Reds",
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_roc:
        st.subheader("ROC Curve (True Positive vs False Positive Rate)")
        roc_data = eval_results.get(best_name, {}).get("roc_curve", {})
        fpr_list = roc_data.get("fpr", [0, 1])
        tpr_list = roc_data.get("tpr", [0, 1])

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr_list,
                y=tpr_list,
                mode="lines",
                name=f"{best_name} (AUC = {eval_results.get(best_name, {}).get('roc_auc', 0.5):.3f})",
                line=dict(color="#E50914", width=3),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (AUC = 0.50)",
                line=dict(color="grey", dash="dash"),
            )
        )
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
        )
        st.plotly_chart(fig_roc, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# PAGE 6: DATA MANAGEMENT
# -------------------------------------------------------------------------------------------------
elif nav_selection == "📁 Data Management":
    st.title("📁 Data Management & Model Training Center")
    st.caption("Generate synthetic cohorts, upload production CSVs, validate schemas, and retrain ML pipelines.")

    dm_tab1, dm_tab2 = st.tabs(["⚡ Synthetic Generator", "📤 CSV Upload & Validation"])

    with dm_tab1:
        st.subheader("Generate Reproducible Netflix Cohort")
        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            n_gen = st.slider("Record Count", min_value=1000, max_value=8000, value=4000, step=500)
        with gen_col2:
            seed_gen = st.number_input("Random Seed", value=42, step=1)

        if st.button("Generate & Load Synthetic Cohort"):
            with st.spinner("Generating realistic subscriber behavioral dataset..."):
                new_data = generate_synthetic_churn_data(
                    num_records=n_gen, random_seed=int(seed_gen)
                )
                new_data.to_csv(DEFAULT_DATASET_PATH, index=False)
                st.session_state["raw_df"] = new_data
                st.cache_data.clear()
            st.success(
                f"Generated {len(new_data):,} fictional records! Historical churn rate: {new_data['churned'].mean():.2%}"
            )

    with dm_tab2:
        st.subheader("Upload Custom Subscriber CSV")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Ensure file contains required behavioral and subscription columns.",
        )
        if uploaded_file is not None:
            try:
                uploaded_df = load_dataset(uploaded_file)
                is_valid, errs, warns, stats = validate_dataset(uploaded_df)

                if is_valid:
                    st.success(f"✅ Dataset validated successfully! ({len(uploaded_df):,} rows)")
                    if warns:
                        for w in warns:
                            st.warning(f"⚠️ {w}")
                    if st.button("Apply Uploaded Dataset"):
                        st.session_state["raw_df"] = uploaded_df
                        st.cache_data.clear()
                        st.success("Custom dataset applied to workspace!")
                else:
                    st.error("❌ Dataset failed schema validation:")
                    for e in errs:
                        st.error(f"- {e}")
            except Exception as e:
                st.error(f"Error parsing CSV file: {str(e)}")

    st.markdown("---")
    st.subheader("🔄 Model Training & Retraining")
    st.write(
        f"Current Dataset in Workspace: **{len(st.session_state['raw_df']):,} records** "
        f"(Churn Rate: **{st.session_state['raw_df']['churned'].mean():.2%}**)"
    )

    if st.button("⚡ Retrain ML Classification Pipeline"):
        with st.spinner("Training Logistic Regression and Random Forest models..."):
            new_pipeline = ChurnModelPipeline()
            new_meta = new_pipeline.train_and_evaluate(st.session_state["raw_df"])
            new_pipeline.save()
            st.session_state["model_pipeline"] = new_pipeline
            st.cache_data.clear()
            st.cache_resource.clear()
        st.success(
            f"Model retraining complete! Best Model: '{new_meta['best_model_name']}' "
            f"(F1: {new_meta['evaluation_results'][new_meta['best_model_name']]['f1_score']:.3f})"
        )


# -------------------------------------------------------------------------------------------------
# PAGE 7: RETENTION OUTCOME TRACKING
# -------------------------------------------------------------------------------------------------
elif nav_selection == "🧪 Retention Outcome Tracking":
    st.title("🧪 Retention Outcome Tracking & Empirical Lift Analysis")
    st.caption("Measure actual retention uplift against uncontacted randomized holdout controls.")

    tracker: RetentionOutcomeTracker = st.session_state["outcome_tracker"]
    lift_results = tracker.compute_lift_analysis()

    st.markdown(
        "<div class='disclaimer-box'><b>Empirical Evidence Principle:</b> Never claim an intervention saved a customer without "
        "comparing against an uncontacted holdout control group. The table below compares treatment retention rates against holdout controls.</div>",
        unsafe_allow_html=True,
    )

    if not lift_results.empty:
        st.subheader("Observed Empirical Holdout Lift Analysis")
        st.dataframe(
            lift_results.rename(
                columns={
                    "recommended_action": "Intervention Strategy",
                    "treatment_subscribers": "Intervened (Treatment)",
                    "control_subscribers": "Holdout (Control)",
                    "treatment_retention_rate": "Treatment Retained (%)",
                    "control_retention_rate": "Control Retained (%)",
                    "observed_absolute_lift_pct": "Absolute Lift (%)",
                    "relative_uplift_pct": "Relative Uplift (%)",
                    "total_campaign_cost": "Campaign Cost ($)",
                }
            ).style.format(
                {
                    "Treatment Retained (%)": "{:.1f}%",
                    "Control Retained (%)": "{:.1f}%",
                    "Absolute Lift (%)": "+{:.1f}%",
                    "Relative Uplift (%)": "+{:.1f}%",
                    "Campaign Cost ($)": "${:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig_lift = px.bar(
            lift_results,
            x="recommended_action",
            y=["treatment_retention_rate", "control_retention_rate"],
            barmode="group",
            color_discrete_map={
                "treatment_retention_rate": "#E50914",
                "control_retention_rate": "#555555",
            },
        )
        fig_lift.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            xaxis_title="",
            yaxis_title="Observed Retention Rate (%)",
            legend_title="Cohort Group",
        )
        st.plotly_chart(fig_lift, use_container_width=True)
    else:
        st.info("No completed trial outcome records found in the log yet.")

    c_dl, c_pop = st.columns(2)
    with c_dl:
        csv_tracker_bytes = tracker.export_csv_bytes()
        st.download_button(
            label="📥 Download Interventions Log as CSV",
            data=csv_tracker_bytes,
            file_name=f"retention_interventions_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with c_pop:
        if st.button("🔄 Populate Demo A/B Trial Records (300 Accounts)"):
            tracker.populate_demo_historical_outcomes(300)
            st.success("Populated 300 completed trial records with holdouts!")
            st.rerun()


# -------------------------------------------------------------------------------------------------
# PAGE 8: PDF EXECUTIVE REPORT
# -------------------------------------------------------------------------------------------------
elif nav_selection == "📄 PDF Executive Report":
    st.title("📄 Executive PDF Report Generator")
    st.caption("Generate an audit-ready, executive PDF summary using ReportLab.")

    st.markdown(
        "Generate a downloadable professional PDF report summarizing dataset health, model evaluation, "
        "risk segments, behavioral churn drivers, high-risk subscribers, and expected net benefit scenarios."
    )

    if st.button("📑 Generate PDF Report", type="primary"):
        with st.spinner("Compiling executive PDF report with live data and model metrics..."):
            global_drivers = get_global_feature_importance(pipeline)
            high_risk_subs = active_subscribers[active_subscribers["risk_category"] == "High Risk"]

            success, msg, pdf_path = build_pdf_report(
                summary_kpis={},
                model_metadata=pipeline.training_metadata,
                risk_summary=segment_summary,
                top_churn_drivers=global_drivers,
                high_risk_subscribers=high_risk_subs,
                strategy_priorities=strategy_agg,
                business_kpis=portfolio_kpis,
            )

        if success and pdf_path and pdf_path.exists():
            st.success("✅ PDF report successfully compiled!")
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()

            st.download_button(
                label="⬇️ Download Executive PDF Report",
                data=pdf_data,
                file_name=pdf_path.name,
                mime="application/pdf",
            )
        else:
            st.error(f"Error generating PDF report: {msg}")

    st.markdown("---")
    st.subheader("Executive Report Preview")
    st.markdown(
        f"""
        - **Document Title:** Netflix Customer Churn & Retention Intelligence Executive Report
        - **Active Subscribers Audited:** {segment_summary['total_subscribers']:,}
        - **High-Risk Segment Size:** {segment_summary['high_risk_count']:,} accounts ({segment_summary['high_risk_pct']:.1f}%)
        - **Total MRR at Risk:** ${segment_summary['total_mrr_at_risk']:,.2f}
        - **Expected Campaign Investment:** ${portfolio_kpis['total_campaign_cost']:,.2f}
        - **Projected Net Contribution Benefit:** ${portfolio_kpis['total_expected_net_benefit']:,.2f}
        - **Selected ML Engine:** {pipeline.best_model_name}
    """
    )
