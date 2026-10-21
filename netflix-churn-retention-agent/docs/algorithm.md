# Algorithmic Specification & Mathematical Formulation

## Netflix Customer Churn & Retention Intelligence AI Agent

---

### 1. Overview and System Architecture Flow

The intelligence agent executes an end-to-end processing pipeline transforming raw subscription telemetry into prioritized, economically quantified retention decisions:

```
[ Raw Subscription Telemetry ]
               │
               ▼
   [ Data Validation & Hygiene ]
               │
               ▼
[ Behavioral Feature Engineering ] 
   (Decline %, Inactivity Flag, Payment & Complaint Risk)
               │
               ▼
   [ Leakage-Free Preprocessing Pipeline ]
   (Imputation + Scaler + One-Hot Encoding fitted on Train only)
               │
               ▼
 [ ML Classifier Training & Calibration ]
   (Logistic Regression vs. Random Forest)
               │
               ▼
 [ Active Prospective Filtering & Risk Segmentation ]
   (High >= 70%, Med 40-70%, Low < 40%)
               │
               ▼
[ Explainable Churn Driver Extraction ]
   (Predictive Indicators vs. Business Hypotheses)
               │
               ▼
[ Deterministic Retention Strategy Engine ]
   (Advisory Recommendations + Priority + Costs)
               │
               ▼
[ Unit Economic Prioritization & Expected Net Benefit ]
   (Expected Margin, Incremental Uplift, Horizon)
               │
               ▼
[ Executive PDF Report & A/B Holdout Tracking ]
```

---

### 2. Algorithmic Modules and Pseudocode

#### Algorithm 1: Synthetic Telemetry Generation via Multi-Factor Logit
```python
INPUT: num_records N, random_seed S, snapshot_date T0
OUTPUT: DataFrame D of realistic subscriber records

1. Initialize pseudo-random number generator RNG(seed=S)
2. Generate base attributes:
     customer_id: "NFLX-100000" ... "NFLX-(100000 + N)"
     plan ~ Categorical(Standard: 35%, Ads: 28%, Premium: 20%, Basic: 17%)
     fee = CatalogPrice(plan)
     tenure_months ~ Gamma(shape=2.2, scale=7.5) clipped to [1, 60]
     start_date = T0 - (tenure_months * 30.4 days + uniform(0, 20))
3. Simulate latent engagement state:
     disengaged_latent = Bernoulli(p=0.26)
4. Simulate viewing telemetry:
     base_daily_watch ~ TruncatedNormal(mu=2.8, sigma=1.3, min=0.2, max=7.5)
     watch_prev_30d = base_daily_watch * 30 * Uniform(0.85, 1.15)
     decline_factor = Uniform(0.15, 0.65) if disengaged_latent else Uniform(0.80, 1.25)
     watch_last_30d = watch_prev_30d * decline_factor
     active_days = clip(round(watch_last_30d / (base_daily_watch + 0.1)), 0, 30)
     days_inactive = 0..7 if active_days >= 7 else (8..30 if active_days > 0 else 31..65)
5. Simulate operational friction:
     payment_failures ~ Poisson(lambda=0.22)
     complaints ~ Poisson(lambda=0.18)
     support_tickets = complaints + Poisson(lambda=0.45)
     auto_renewal_enabled = Bernoulli(p=0.72 if disengaged_latent else 0.96)
6. Compute log-odds of cancellation (logit):
     watch_drop_ratio = (watch_prev_30d - watch_last_30d) / (watch_prev_30d + 1.0)
     inactivity_effect = +1.6 if days_inactive > 21 else (+0.8 if days_inactive > 10 else -0.3)
     tenure_decay = -0.45 * log(1 + tenure_months)
     search_frustration = (0.55 - search_conversion) * 1.5
     
     logit = -1.35 
           + 1.85 * watch_drop_ratio 
           + inactivity_effect 
           + 0.95 * payment_failures 
           + 0.75 * complaints 
           + search_frustration 
           + 1.80 * (1 - auto_renewal_enabled) 
           + 0.50 * previous_cancels 
           + tenure_decay 
           + noise ~ N(0, 0.85)

7. Probability P(churn) = 1 / (1 + exp(-logit))
8. Label churned ~ Bernoulli(p=P(churn))
9. RETURN D
```

---

#### Algorithm 2: Feature Engineering & Preprocessing Pipeline
```python
INPUT: Raw DataFrame D
OUTPUT: Encoded Matrix X, Feature Names F

1. Compute Watch-Time Decline Percentage:
     watch_decline_pct = clip(((watch_prev_30d - watch_last_30d) / (watch_prev_30d + 0.1)) * 100, -100.0, 100.0)
2. Categorize Engagement Trend:
     if watch_decline_pct >= 40.0: "Severe Drop"
     else if watch_decline_pct >= 15.0: "Moderate Drop"
     else if watch_decline_pct >= -15.0: "Stable"
     else: "Growing"
3. Compute Operational Binary Indicators:
     inactivity_indicator = 1 if days_inactive >= 14 else 0
     payment_risk_indicator = 1 if payment_failures >= 1 else 0
     complaint_risk_indicator = 1 if (complaints >= 1 or support_tickets >= 3) else 0
4. Compute Marketing Responsiveness:
     marketing_engagement_rate = opened_emails / max(received_emails, 1)
5. Group Subscriber Tenure Cohorts:
     tenure_group = "New (1-6m)" | "Established (7-18m)" | "Mature (19-36m)" | "Loyal (36m+)"
6. Split Stratified Train / Test (80 / 20) on target 'churned'.
7. Fit ColumnTransformer on Train only:
     Numerical Pipeline: SimpleImputer(median) -> StandardScaler()
     Categorical Pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(ignore_unknown)
8. Transform Train and Test data.
```

---

#### Algorithm 3: Model Comparison & Probability Calibration
```python
INPUT: X_train, y_train, X_test, y_test
OUTPUT: Best Model M*, Evaluation Metadata

1. Fit Candidate 1: LogisticRegression(class_weight='balanced', max_iter=1000)
2. Fit Candidate 2: RandomForestClassifier(n_estimators=180, max_depth=8, class_weight='balanced')
3. For each candidate model m:
     y_pred = m.predict(X_test)
     y_prob = m.predict_proba(X_test)[:, 1]
     Compute: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Brier Score
4. Select M* = argmax_{m} [ F1_Score(m) ]
5. Return M* and evaluation dictionaries.
```

---

#### Algorithm 4: Active Subscriber Isolation & Risk Segmentation
```python
INPUT: Scored DataFrame D, Thresholds tau_high = 0.70, tau_low = 0.40
OUTPUT: Segmented DataFrame with Risk Tiers

1. Validate: 0.0 < tau_low < tau_high < 1.0
2. Assign Risk Categories:
     if churn_probability >= tau_high:
         risk_category = "High Risk"
     else if churn_probability >= tau_low:
         risk_category = "Medium Risk"
     else:
         risk_category = "Low Risk"
3. Filter Active Subscribers for Prospective Retention:
     is_active_subscriber = (churned == 0) if 'churned' in D else True
     (Historical churned accounts are excluded from active prospective interventions)
4. Compute Revenue Exposure:
     revenue_at_risk = monthly_subscription_revenue * churn_probability
```

---

#### Algorithm 5: Deterministic Advisory Retention Strategy Engine
```python
INPUT: Active Subscriber record row r
OUTPUT: Recommendation Dict Rec

1. Involuntary Billing Failure Rule:
     IF payment_failures >= 1:
         Rec.action = "Frictionless Payment Resolution"
         Rec.priority = "Urgent" if (risk == "High Risk" or payment_failures >= 2) else "High"
         Rec.cost = $2.50
         Rec.metric = "Successful billing retry within 5 days"
2. Customer Care & Dissatisfaction Rule:
     ELSE IF complaints >= 1 or support_tickets >= 3:
         Rec.action = "VIP Priority Care Outreach"
         Rec.priority = "Urgent"
         Rec.cost = $8.00
         Rec.metric = "CSAT score >= 4.5/5"
3. Prolonged Dormancy Rule:
     ELSE IF days_inactive >= 14:
         Rec.action = "Targeted Win-Back & Trending Highlight"
         Rec.priority = "Urgent" if days_inactive >= 21 else "High"
         Rec.cost = $1.40
         Rec.metric = "Days since last activity reduced to < 7 days"
4. Serial Churner Rule:
     ELSE IF previous_cancellations >= 1 and risk == "High Risk":
         Rec.action = "High-Touch Loyalty Onboarding"
         Rec.priority = "Urgent"
         Rec.cost = $3.50
         Rec.metric = "Active retention past 90-day milestone"
5. Content Disengagement Rule:
     ELSE IF watch_decline_pct >= 20.0:
         Rec.action = "Personalized Genre Re-engagement"
         Rec.priority = "High" if watch_decline_pct >= 40.0 else "Medium"
         Rec.cost = $0.85
         Rec.metric = "30-day watch hours recovery (+25% target)"
6. Premium Tier Underutilization Rule:
     ELSE IF plan == "Premium" and average_daily_watch_hours < 0.8:
         Rec.action = "Plan Right-Sizing Advisory"
         Rec.priority = "Medium"
         Rec.cost = $1.25
         Rec.metric = "Account retention across any tier over 90 days"
7. Healthy Baseline / Low Risk Rule:
     ELSE:
         Rec.action = "Standard Loyalty & Perks"
         Rec.priority = "Standard"
         Rec.cost = $0.35
         Rec.metric = "Sustained high NPS score"

8. Tag Rec.advisory_status = "Advisory (Requires Human Approval)"
```

---

#### Algorithm 6: Unit Economics & Expected Incremental Net Benefit
```python
INPUT: Scored subscriber row r, margin_ratio gamma = 0.75, horizon_months H = 6, relative_uplift lambda = 0.20
OUTPUT: Economic Projections

1. Expected Contribution Margin over Horizon:
     expected_margin = monthly_revenue * gamma * H
2. Incremental Retention Probability (NOT equal to churn probability!):
     incremental_saved_prob = clip(churn_probability * lambda, 0.0, 1.0)
3. Expected Incremental Gross Benefit:
     gross_benefit = incremental_saved_prob * expected_margin
4. Expected Incremental Net Benefit:
     net_benefit = gross_benefit - intervention_cost
5. Strategic Prioritization Score:
     urgency_weight = {"Urgent": 1.4, "High": 1.2, "Medium": 1.0, "Standard": 0.5}[priority]
     score = net_benefit * urgency_weight
```

---

#### Algorithm 7: Empirical Randomized Holdout Lift Tracking
```python
INPUT: Batch of approved recommendations, holdout_ratio = 0.15
OUTPUT: Interventions log with treatment / control assignment

1. For each approved subscriber i:
     is_control_group ~ Bernoulli(p=holdout_ratio)
     if is_control_group:
         is_executed = False
         intervention_cost = $0.00
     else:
         is_executed = True
         intervention_cost = strategy_cost
2. Log to data/retention_interventions_log.csv
3. At post-intervention audit (30/60 days):
     treatment_retention_rate = mean(is_retained | is_executed == True)
     control_retention_rate   = mean(is_retained | is_control == True)
     observed_absolute_lift   = treatment_retention_rate - control_retention_rate
     relative_uplift          = observed_absolute_lift / control_retention_rate
```
