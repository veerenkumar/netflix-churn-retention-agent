# Methodology & Modeling Decisions

## Netflix Customer Churn & Retention Intelligence AI Agent

---

### 1. Business Problem & Prediction Window

Subscription video-on-demand (SVOD) streaming platforms operate on recurring monthly billing cycles. Customer churn occurs when a subscriber cancels their auto-renewal or fails billing settlement and lapses at the end of their billing period.

#### Prediction Window Definition
- **Target Label (`churned = 1`):** Represents whether an active subscriber will cancel or lapse within the **subsequent 30-day billing window**.
- **Observation Horizon:** Telemetry is gathered up to day $T_0$ (snapshot date). The target evaluates outcomes between $T_0$ and $T_0 + 30$ days.
- **Target Leakage Prevention:** Any post-$T_0$ cancellation signals (e.g. cancellation feedback forms, refund requests, post-lapse surveys) are strictly excluded from input feature matrices.

---

### 2. Modeling Decisions & Candidate Selection

#### Candidate 1: Balanced Logistic Regression (Baseline & Risk Scoring)
- **Rationale:** Highly interpretable log-odds model. Provides smooth, continuous probability estimates across the entire risk spectrum.
- **Class Imbalance Strategy:** Employs `class_weight='balanced'` which inversely weights training samples by class frequency:
  $$w_j = \frac{N}{2 \cdot N_j}$$
- **Advantage in SVOD Retention:** Penalizes false negatives heavily, resulting in high Recall ($\sim 88\%$). In subscription retention, failing to identify an at-risk subscriber (False Negative) costs up to 6 months of lost recurring subscription revenue ($\$90 - \$138$), whereas reaching out with an advisory content watchlist (False Positive) carries negligible marginal cost ($\$0.85$).

#### Candidate 2: Balanced Random Forest Classifier (Nonlinear Benchmark)
- **Rationale:** Captures complex nonlinear interaction thresholds (e.g. high tenure combined with sudden drop in viewing hours).
- **Hyperparameter Regularization:** `max_depth=8`, `min_samples_split=6`, `min_samples_leaf=2` to prevent overfitting on noisy behavioral signals.
- **Advantage:** Provides higher Precision and natural Gini impurity feature importances.

#### Metric-Driven Model Selection
Rather than arbitrarily picking a model, the system evaluates both candidates on a held-out test split and selects the candidate that maximizes validation performance (primary default: **F1-Score** or **ROC-AUC**).

---

### 3. Evaluation Metric Trade-offs

| Metric | Business Implication in Streaming Retention | Priority Level |
|---|---|---|
| **Recall (Sensitivity)** | Percentage of actual churners successfully identified. High recall ensures the business captures at-risk revenue. | **Critical (Highest)** |
| **Precision** | Percentage of identified accounts that actually churned. Higher precision reduces wasted retention outreach. | **High** |
| **F1-Score** | Harmonic mean of Precision and Recall. Balances capturing churners while preventing excessive false alarms. | **Primary Selection Metric** |
| **ROC-AUC** | Evaluates ranking discrimination across all possible decision thresholds. | **High** |
| **Brier Score** | Mean squared difference between predicted probabilities and observed binary outcomes. Evaluates probability calibration. | **Audit & Validation** |
| **Accuracy** | Overall correct classifications. In an imbalanced dataset (e.g. 17% churn), predicting all 0s gives 83% accuracy but 0% business value. | **Secondary / Contextual** |

---

### 4. Active Prospective Segmentation vs. Historical Inactive Accounts

A common pitfall in commercial retention analytics is scoring historical churned subscribers as eligible for prospective retention actions.

#### Methodological Segregation
1. **Historical Training Cohort:** Contains both historical retained ($0$) and churned ($1$) subscribers to train and calibrate supervised classifiers.
2. **Prospective Retention Queue:** Strictly filters for **Active Subscribers** (`is_active_subscriber = True`). Historical churners are routed to win-back / re-acquisition workflows rather than regular retention campaigns.

---

### 5. Interpretable Explanations: Predictive Association vs. Proven Causality

A statistical correlation identified by a tree or regression model is **not** evidence of direct physical causation.

#### Clarifying Distinctions
- **Predictive Behavioral Indicator:** An observed statistical correlation (e.g. subscribers with $>14$ days inactivity churn at $4.2\times$ the baseline rate).
- **Potential Business Hypothesis:** A domain-informed explanation of why this correlation exists (e.g. content fatigue, lack of compelling new releases, or forgetting subscription value).
- **Proven Causal Effect:** An empirically demonstrated effect verified through randomized controlled trials (A/B testing).

The system explicitly tags individual subscriber explanations as *Predictive Behavioral Indicators* accompanied by an unambiguous scientific disclaimer to prevent misleading management into confusing correlation with causation.

---

### 6. Retention Economics: Incremental Uplift vs. Churn Probability

A subscriber having an $80\%$ probability of churn does **not** mean an email or discount will save them with $80\%$ certainty. Equating churn probability with retention probability grossly inflates projected financial returns.

#### Rigorous Business Logic
$$\text{Expected Incremental Net Benefit} = (\Delta P_{\text{retention}} \times \text{Expected Margin Horizon}) - \text{Intervention Cost}$$

Where:
- $\text{Expected Margin Horizon} = \text{Monthly Revenue} \times \text{Gross Margin \% (default 75\%)} \times \text{Horizon Months (default 6)}$
- $\Delta P_{\text{retention}} = P(\text{churn}) \times \text{Incremental Uplift \%}$ (e.g. $0.70 \times 0.20 = 0.14$)
- $\text{Intervention Cost}$ = Cost of executing campaign ($\$0.35 - \$8.00$)

This ensures that:
1. Gross revenue is never presented as profit.
2. Expensive interventions applied to low-risk customers produce negative ROI, discouraging wasteful discounting.
3. High-margin, high-risk accounts receive prioritized intervention.

---

### 7. Empirical A/B Holdout Testing

To transition from hypothetical business scenarios to verified commercial outcomes, the system incorporates an **Outcome Tracking Module** with randomized control holdouts:
- **Treatment Cohort (85%):** Receives the personalized retention intervention.
- **Control Cohort (15%):** Uncontacted randomized holdout.
- **Empirical Lift Calculation:**
  $$\text{Observed Absolute Lift} = \text{Treatment Retention Rate} - \text{Control Retention Rate}$$
  $$\text{Relative Uplift} = \frac{\text{Observed Absolute Lift}}{\text{Control Retention Rate}}$$

---

### 8. Operational Governance & Human-in-the-Loop

The retention strategy engine operates on **deterministic, transparent business rules** rather than unconstrained black-box generative agents. Recommendations are explicitly tagged as:
$$\text{Status: Advisory (Requires Human Operational Approval)}$$
No emails, SMS messages, billing alterations, or subscription discounts are automatically dispatched without authorization by retention management.
