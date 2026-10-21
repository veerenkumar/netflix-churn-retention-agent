# 🎬 Netflix Customer Churn & Retention Intelligence AI Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-29%20Passed-brightgreen.svg)]()

> **Notice:** This project is an independent educational and research prototype developed for business intelligence and machine learning demonstration. It is not affiliated with, endorsed by, or utilizing internal proprietary data or systems of Netflix, Inc.

---

## 1. Executive Summary & Business Problem

Subscription video-on-demand (SVOD) streaming platforms face rising subscriber acquisition costs alongside elevated customer churn. In voluntary subscription models, cancellations are frequently preceded by identifiable behavioral patterns—including steep watch-time declines, viewing inactivity, recurring billing failures, and catalog search fatigue.

The **Netflix Customer Churn & Retention Intelligence AI Agent** is an end-to-end, production-grade machine learning system designed to:
1. **Predict Cancellations Early:** Estimate the precise 30-day churn probability for every active subscriber.
2. **Segment Customer Risk:** Categorize subscribers into configurable risk tiers (**High Risk &ge; 70%**, **Medium Risk 40-70%**, **Low Risk < 40%**).
3. **Isolate Active Accounts:** Strictly segregate active prospective accounts from historical churners to prevent wasteful interventions.
4. **Identify Churn Drivers:** Provide factual, subscriber-level behavioral explanations while strictly distinguishing **predictive indicators** from **proven causal effects**.
5. **Recommend Tailored Interventions:** Match at-risk accounts with advisory retention campaigns (e.g. watchlist re-engagement, frictionless billing recovery, concierge care).
6. **Quantify Financial Impact:** Calculate expected contribution margin horizon, expected incremental net benefit, and portfolio ROI.
7. **Empirically Measure Lift:** Log interventions with randomized 15% uncontacted holdout control groups to measure true incremental retention lift.
8. **Export Audit Reports:** Generate multi-page executive PDF reports via ReportLab and CSV exports.

---

## 2. System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │  Raw Subscriber Telemetry / Production CSV   │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Data Validation & Schema Hygiene Checks    │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │  Behavioral Feature Engineering Pipeline     │
                       │   • Watch decline %    • Inactivity flag     │
                       │   • Viewing intensity  • Payment risk flag   │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Leakage-Free Preprocessing Transformer     │
                       │    (StandardScaler + OHE fitted on Train)    │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │     Supervised Model Training & Evaluation   │
                       │  • Balanced Logistic Regression (Sensitivity)│
                       │  • Balanced Random Forest (Nonlinear trees)  │
                       └──────────────────────┬───────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌───────────────────────────────┐
     │  Risk Segmentation Engine     │                 │ Interpretable Explanations    │
     │  • High, Med, Low Tiers       │                 │ • Predictive Behavioral Flags │
     │  • Revenue at Risk ($ MRR)    │                 │ • Domain Business Hypotheses  │
     └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                     └────────────────────────┬────────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Personalized Retention Strategy Engine     │
                       │   • Deterministic, explainable decision tree │
                       │   • Advisory status (requires human approval)│
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Unit Economics & ROI Prioritization        │
                       │   Expected Net Benefit = (ΔP × Margin) - Cost│
                       └──────────────────────┬───────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌───────────────────────────────┐
     │  Interactive Streamlit App    │                 │   Executive PDF Report &      │
     │  (Plotly Charts, Filters, CSV)│                 │   A/B Holdout Tracking Engine │
     └───────────────────────────────┘                 └───────────────────────────────┘
```

---

## 3. Technology Stack

- **Core Runtime:** Python 3.11, 3.12, or 3.13 (macOS, Linux, Windows)
- **Data Manipulation:** Pandas (`>=2.0.0`), NumPy (`>=1.24.0`)
- **Machine Learning & Pipeline:** Scikit-Learn (`>=1.3.0`), Joblib (`>=1.3.0`)
- **Visualizations:** Plotly (`>=5.15.0`), Matplotlib (`>=3.7.0`)
- **Web Dashboard:** Streamlit (`>=1.28.0`) with custom Netflix-inspired charcoal/red dark theme
- **PDF Generation:** ReportLab (`>=4.0.0`) with dynamic two-pass `NumberedCanvas`
- **Testing & QA:** Pytest (`>=7.4.0`) with 29 automated test cases

---

## 4. Data Dictionary

The application works with a standard schema of 24 subscription and telemetry attributes:

| Column Name | Data Type | Description |
|---|---|---|
| `customer_id` | String | Unique identifier (e.g. `NFLX-100042`). |
| `subscription_plan` | String | Tier (`Standard with Ads`, `Basic`, `Standard`, `Premium`). |
| `monthly_subscription_fee` | Float | Recurring plan fee (`$6.99`, `$9.99`, `$15.49`, `$22.99`). |
| `subscription_start_date` | String | ISO date string (`YYYY-MM-DD`). |
| `subscription_tenure_months` | Integer | Active account longevity in months (1 to 60). |
| `age_group` | String | Demographic cohort (`18-25`, `26-35`, `36-50`, `51-65`, `65+`). |
| `country_or_region` | String | Geographic market (`North America`, `EMEA`, `LATAM`, `APAC`). |
| `average_daily_watch_hours` | Float | Typical daily streaming volume (0.2 to 7.5 hrs). |
| `watch_hours_last_30_days` | Float | Cumulative streaming hours in the most recent 30-day window. |
| `watch_hours_previous_30_days` | Float | Cumulative streaming hours in the preceding 30-day window. |
| `days_since_last_activity` | Integer | Days elapsed since the subscriber last streamed. |
| `active_days_last_30_days` | Integer | Days with at least 1 streaming session in past month (0 to 30). |
| `number_of_titles_watched` | Integer | Unique titles viewed in past 30 days. |
| `preferred_content_genre` | String | Top genre affinity (`Sci-Fi`, `Crime Drama`, `Comedy`, etc.). |
| `search_to_play_conversion_rate`| Float | Conversion from catalog search to play (low = search frustration). |
| `payment_failures_last_90_days` | Integer | Count of failed automated billing attempts. |
| `customer_complaints_last_90_days`| Integer | Count of formal customer complaints logged. |
| `support_tickets_last_90_days` | Integer | Total customer care inquiries submitted. |
| `marketing_emails_received` | Integer | Promotional emails delivered in past 90 days. |
| `marketing_emails_opened` | Integer | Promotional emails opened by subscriber. |
| `previous_cancellations` | Integer | Historical times subscriber previously cancelled and reactivated. |
| `auto_renewal_enabled` | Binary | 1 = enabled, 0 = disabled (strong pre-cancellation intent). |
| `monthly_subscription_revenue` | Float | Baseline monthly recurring revenue. |
| `churned` | Binary | Historical target: 1 = cancelled in next 30 days, 0 = continued. |

---

## 5. Machine Learning Methodology & Evaluation

Models are trained on an 80/20 stratified train/test split. Preprocessing parameters (medians, scalers, one-hot categories) are fitted **strictly on training data** to eliminate data leakage.

### Model Performance Comparison on Held-Out Test Data

| Candidate Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|
| **Logistic Regression (Balanced)** | **0.760** | **0.421** | **0.889** | **0.571** | **0.870** | **0.165** |
| **Random Forest Classifier (Balanced)** | **0.830** | **0.533** | **0.444** | **0.485** | **0.815** | **0.134** |

### The False Negative vs. False Positive Trade-Off
In subscription retention management:
- **False Negative (Missed Churner):** The system fails to flag a cancelling customer. Business loss = 6 months of lost subscription revenue ($\$90 - \$138$).
- **False Positive (False Alarm):** The system reaches out with a personalized genre watchlist or feature highlight to a subscriber who would have stayed anyway. Marginal cost = $\sim \$0.85$.

Because false negatives carry $\sim 100\times$ higher cost than false positives, **Balanced Logistic Regression** is selected as the primary engine for high-recall risk screening.

---

## 6. Retention Strategy Logic & Economics

The strategy engine applies explainable, deterministic business logic:

| Risk Signal | Strategy Key | Campaign Intervention | Unit Cost | Follow-Up Metric |
|---|---|---|---|---|
| Payment failures &ge; 1 | **Frictionless Payment Resolution** | In-app billing update modal + card expiration notice with grace period | $2.50 | Billing retry success within 5 days |
| Complaints &ge; 1 or Tickets &ge; 3 | **VIP Priority Care Outreach** | Proactive senior customer care outreach & technical ticket resolution | $8.00 | CSAT &ge; 4.5/5 |
| Inactive &ge; 14 days | **Targeted Win-Back & Trending Highlight** | Push/email showcase of top 3 trending hits in preferred genre | $1.40 | Days inactive reduced to < 7 days |
| Previous Cancels &ge; 1 & High Risk | **High-Touch Loyalty Onboarding** | Behind-the-scenes content & upcoming series roadmap | $3.50 | Active retention past 90-day milestone |
| Watch drop &ge; 20% | **Personalized Genre Re-engagement** | Algorithmic watchlist with 5 top titles tailored to viewer taste | $0.85 | 30-day watch hours recovery (+25%) |
| Premium Plan + Low Usage | **Plan Right-Sizing Advisory** | Advisory suggestion to migrate to Standard ($15.49) or Ad-supported ($6.99) | $1.25 | 90-day account retention on any tier |
| Healthy Engagement | **Standard Loyalty & Perks** | Loyalty appreciation badges, sneak-peek trailers, 4K reminders | $0.35 | Sustained high NPS |

### Net Economic Benefit Formula
$$\text{Expected Incremental Net Benefit} = (\text{Incremental Retention Prob} \times \text{Expected Margin Horizon}) - \text{Intervention Cost}$$
Where:
- $\text{Expected Margin Horizon} = \text{Monthly Revenue} \times 0.75 \text{ (Margin)} \times 6 \text{ (Months)}$
- $\text{Incremental Retention Prob} = P(\text{churn}) \times 0.20 \text{ (Assumed Uplift)}$

---

## 7. Installation & Quick Start (macOS)

### Step 1: Navigate to Project Directory
```bash
cd /Users/veerenkumarkonchada/.gemini/antigravity-ide/scratch/netflix-churn-retention-agent
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Automated Test Suite
```bash
python3 -m pytest -v
```
*(All 29 tests will execute and pass in ~6 seconds.)*

### Step 5: Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. Dashboard Walk-Through

1. **📊 Executive Overview:** Real-time KPI cards for active subscribers, revenue exposure, high/medium/low risk distribution, and prioritized campaign portfolio.
2. **📈 Churn Analytics:** Interactive Plotly charts examining churn by subscription tier, tenure cohorts, watch-time decay, billing friction, and global feature importance.
3. **🔍 Subscriber Risk Explorer:** Multi-filter searchable subscriber table with an expandable deep-dive drawer detailing individual churn indicators, evidence, and custom advisory retention plans.
4. **🎯 Retention Strategy Center:** Strategic ROI calculator with live parameter sliders (contribution margin %, lifetime evaluation horizon, expected uplift %) and batch approval to the campaign queue.
5. **🤖 Model Performance & Evaluation:** Verification of actual test metrics, confusion matrix heatmap, and ROC curve.
6. **📁 Data Management:** Synthetic cohort generator, custom CSV file uploader with automated schema validation, and 1-click model retraining.
7. **🧪 Retention Outcome Tracking:** Campaign execution logging with randomized 15% holdouts to measure observed absolute retention lift and relative uplift.
8. **📄 PDF Executive Report:** 1-click ReportLab compilation yielding a multi-page PDF report with running headers and page numbers.

---

## 9. GitHub Repository Setup Instructions

To push this repository to GitHub:

```bash
cd /Users/veerenkumarkonchada/.gemini/antigravity-ide/scratch/netflix-churn-retention-agent

# Initialize git repository
git init

# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: Netflix Customer Churn & Retention Intelligence AI Agent v1.0"

# Link to your remote GitHub repository
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/netflix-churn-retention-agent.git

# Set default branch and push
git branch -M main
git push -u origin main
```

---

## 10. Limitations & Future Roadmap

### Limitations
1. **Correlation vs. Causation:** Behavioral indicators reflect statistical associations rather than proven causal mechanisms. Interventions should always be validated via holdouts.
2. **Synthetic Data:** The default dataset is a realistically calibrated synthetic cohort designed for educational exploration.
3. **Fixed Billing Window:** Modeled specifically on a monthly recurring billing cycle.

### Future Roadmap
- Integration with live streaming event streams (Kafka/Flink) for real-time drop-off alerts.
- Uplift modeling (Causal Trees / Two-Model Approach) to predict treatment effect per individual subscriber.
- Multi-armed bandit (Thompson Sampling) for automated campaign allocation.

---

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
