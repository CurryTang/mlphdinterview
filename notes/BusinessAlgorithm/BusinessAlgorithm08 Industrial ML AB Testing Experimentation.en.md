# Industrial ML & A/B Testing: Principles, Causal Inference & Full-Factorial Sign-Up Funnel Experiment Design

In production machine learning and web systems engineering, offline validation metrics (such as AUC, NDCG, RMSE, Accuracy) serve merely as coarse gating filters. **Online A/B Testing (Randomized Controlled Trials, RCTs) is the ultimate empirical ground truth for validating whether an algorithmic or UI change delivers genuine causal business uplift**.

This note systematically constructs the industrial experimentation framework:
1. **Core Foundations: Why Offline Gains Often Fail to Translate into Online Business Metrics**
2. **The Production A/B Testing Lifecycle (Design, Stratified Routing, CUPED Variance Reduction, SRM Validation & Decision Rules)**
3. **End-to-End Case Study: Full-Factorial Sign-Up Funnel Experiment on Button Color and Position**

---

## Module 1: What is Industrial A/B Testing? Why Offline Metric Gains Do Not Guarantee Online Value

```text
Industrial Algorithm Iteration & Causal Verification Lifecycle:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Offline ML Pipeline                                                 │
│ • Feature Engineering ➔ Offline Training ➔ Validation (AUC +0.5%)      │
│ • Limitation: Static historical log data; fails to capture user drift  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Online Randomized Controlled Trial (A/B Testing / RCT)              │
│ • Randomized Traffic Allocation ➔ Eliminates Confounders               │
│ • Core Goal: Measure Average Treatment Effect (ATE) on North Star      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Business Decision & Full Rollout                                    │
│ • Primary Metric Significance + Positive ROI ➔ Guardrails Stable       │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Fundamental Differences Between Academic ML and Industrial ML

- **Academic / Offline ML**: Focuses on optimizing loss functions and ranking metrics over static, pre-collected datasets;
- **Industrial / Online ML**: Operates within a dynamic, real-time feedback loop, optimizing for **causal incremental uplift (ATE)**, long-term ecosystem health (DAU, GMV, Retention), and system computational ROI.

### 2. The 4 Root Causes of "Offline Win, Online Neutral or Regression"

1. **Feedback Loops & Selection Bias**:
   Offline datasets are logged by the incumbent production policy. Novel items recommended by a new model may have zero historical interactions in the test set and are incorrectly penalized as false negatives.
2. **Training-Serving Skew & Feature Leakage**:
   Offline features may accidentally incorporate future information or lack strict timestamp synchronization, whereas production inference experiences millisecond-level propagation delays and missing real-time signals.
3. **System Latency & P99 SLA Degradation**:
   A heavier model with slightly higher offline AUC may inflate production P99 latency by 50ms, triggering timeout fallback fallbacks and increasing user bounce rates.
4. **Multi-Objective Conflict & Metric Gaming**:
   Single-metric optimization (e.g., maximizing CTR) often induces clickbait patterns that inflate clicks while degrading downstream conversion rates (CVR) and long-term retention.

---

## Module 2: The Production A/B Testing Lifecycle & Methodologies

### 1. Statistical Foundations & Hypothesis Testing

- **Null Hypothesis ($H_0$) vs. Alternative Hypothesis ($H_1$)**:
  - $H_0: \mu_{\text{treatment}} - \mu_{\text{control}} \le 0$ (No positive treatment effect)
  - $H_1: \mu_{\text{treatment}} - \mu_{\text{control}} > 0$ (Statistically significant positive uplift)
- **Type I Error Rate ($\alpha$, Significance Level)**: Typically set to $0.05$ ($5\%$ chance of falsely rejecting a true null hypothesis).
- **Type II Error Rate ($\beta$) & Statistical Power ($1 - \beta$)**: Typically targeted at $\ge 80\%$ (probability of correctly detecting a true positive effect).
- **Minimum Detectable Effect (MDE)**: The smallest relative/absolute effect size that an experiment can reliably detect given $\alpha$, $\beta$, and metric variance.

### 2. Layered & Orthogonal Experimentation Architecture

```text
Production Multi-Layer Experimentation Topology:
100% Platform Traffic Pool
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Retrieval & Search Layer ➔ Hash(user_id, salt_1)              │
│ [ Exp A: Vector Retrieval (50%) ]      [ Exp B: Graph Retrieval (50%) ]│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Orthogonal Penetration
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Ranking & Multi-Task Layer ➔ Hash(user_id, salt_2)            │
│ [ Control: Legacy Ranker (50%) ]       [ Treatment: Transformer (50%) ]│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Orthogonal Penetration
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 3: UI/UX & Interaction Layer ➔ Hash(user_id, salt_3)             │
│ [ Control: Red Top Button (50%) ]      [ Treatment: Blue Bottom (50%) ]│
└────────────────────────────────────────────────────────────────────────┘
```

- **Intra-Layer Exclusivity**: Competing alternatives within the same subsystem (e.g., two ranking models) must be mutually exclusive to avoid interference.
- **Inter-Layer Orthogonality**: Distinct functional layers use different cryptographic hash salts (`hash(id, salt)`), distributing traffic uniformly and orthogonally across layers so hundreds of experiments run concurrently without confounding.

---

## Module 3: Case Study: Full-Factorial Sign-Up Funnel Experiment Design

> 📌 **Case Scenario**:
> A user acquisition sign-up funnel landing page currently features a **red button at the top of the page**. The product team proposes:
> 1. Changing the button color from **red to blue** (enhancing brand trust and readability);
> 2. Moving the button position from **top to bottom** (matching the natural reading and decision flow after scanning content);
> 3. Applying **both changes simultaneously**.
> 
> Design an end-to-end industrial A/B experiment to evaluate these modifications on click-through rate (CTR) and downstream business health.

```text
2x2 Full Factorial Experiment Matrix:
                       Button Color
                 Red (Baseline)     Blue (Modified)
              ┌─────────────────┬─────────────────┐
  Top (Base)  │   Control (T0)  │  Treatment (T1) │
              │    Red + Top    │    Blue + Top   │
Button        ├─────────────────┼─────────────────┤
Position      │  Treatment (T2) │  Treatment (T3) │
Bottom (Mod)  │   Red + Bottom  │  Blue + Bottom  │
              └─────────────────┴─────────────────┘
```

---

### 1. Hypotheses, Conditions, Randomization Unit & Primary Outcome

#### (1) Hypotheses
- **Null Hypothesis ($H_0$)**: Changing button color, button position, or both yields no positive causal uplift on sign-up button CTR ($\mu_T \le \mu_C$).
- **Alternative Hypotheses ($H_1$)**:
  - **Color Main Effect ($H_{1,\text{color}}$)**: Blue buttons significantly increase CTR over red buttons due to enhanced trust cues;
  - **Position Main Effect ($H_{1,\text{position}}$)**: Bottom buttons significantly increase CTR over top buttons by aligning with the natural F-shaped reading decision flow;
  - **Interaction Effect ($H_{1,\text{interaction}}$)**: Combining color and position modifications induces a non-zero synergistic or antagonistic effect.

#### (2) Control and Treatment Conditions
A **$2 \times 2$ Full Factorial Design** equally splits eligible traffic into 4 arms:
- **Control ($T_0$)**: Red + Top (Status quo baseline);
- **Treatment 1 ($T_1$)**: Blue + Top (Color modified only);
- **Treatment 2 ($T_2$)**: Red + Bottom (Position modified only);
- **Treatment 3 ($T_3$)**: Blue + Bottom (Both color and position modified).

#### (3) Randomization Unit
- **Recommended Selection**: **Persistent Anonymous Device ID / Cookie ID**.
- **Rationale**: Visitors entering the sign-up funnel are primarily unregistered guests who lack a `user_id`.
- **Integrity Requirement**: Deterministic hash partitioning (`hash(device_id, exp_salt) % 4`) ensures cross-session consistency. **Session ID or Request ID must never be used**: changing button color/position across page refreshes degrades UX and violates the Stable Unit Treatment Value Assumption (SUTVA).

#### (4) Primary Outcome Metric
- **Button Click-Through Rate (CTR)**:
  $$\text{CTR} = \frac{\text{Unique Eligible Devices Clicking the Sign-Up Button}}{\text{Unique Eligible Devices with Triggered Exposure on the Funnel Page}}$$
- **Best Practice**: Computed strictly as a **User/Device-Level Proportion**, avoiding raw click/impression ratios that could be distorted by heavy-clicking outlier sessions.

---

### 2. Tradeoffs: Separate Sequential Tests vs. Multi-Arm Factorial Experiment

| Dimension | Sequential Single-Variable Tests (Two Tests) | Multi-Arm Factorial Experiment ($2 \times 2$ Design) |
|---|---|---|
| **Timeline & Velocity** | **Slow** (2 sequential 2-week runs = 4–6 weeks total). | **Fast** (Single 2-week run for all 4 conditions). |
| **Environmental Stability** | Subject to seasonal drift, marketing campaigns, and macroeconomic shifts between runs. | **Guaranteed temporal balance** across all variants. |
| **Statistical Efficiency** | Samples are used only for single-variable comparison. | **Factorial Power Efficiency**: Main effects leverage all samples (e.g., $(T_1+T_3)$ vs. $(T_0+T_2)$ for color). |
| **Interaction Detection** | **Impossible** (assumes effects are purely additive). | **Fully mathematically evaluable**. |
| **Recommendation** | Not recommended for UI/UX elements. | **Industry Standard (Strongly Recommended)**. |

---

### 3. Assessing Interaction Between Button Color and Position

#### (1) Regression & ANOVA Formulation
We fit an Ordinary Least Squares (Linear Probability Model) or Logistic Regression:

$$Y_i = \beta_0 + \beta_1 \cdot \text{Blue}_i + \beta_2 \cdot \text{Bottom}_i + \beta_3 \cdot (\text{Blue}_i \times \text{Bottom}_i) + \epsilon_i$$

Where:
- $\text{Blue}_i \in \{0, 1\}$ and $\text{Bottom}_i \in \{0, 1\}$;
- $\beta_1$: Marginal main effect of Blue at Top position;
- $\beta_2$: Marginal main effect of Bottom position with Red color;
- $\beta_3$: **Interaction Coefficient**.

#### (2) Interpretation of $\beta_3$

$$\hat{\beta}_3 = (\bar{Y}_{T_3} - \bar{Y}_{T_1}) - (\bar{Y}_{T_2} - \bar{Y}_{T_0}) = \bar{Y}_{T_3} - \bar{Y}_{T_2} - \bar{Y}_{T_1} + \bar{Y}_{T_0}$$

We test $H_0: \beta_3 = 0$ via a two-sided Wald / t-test:
- **$\beta_3 > 0$ ($p < 0.05$)**: **Synergistic / Super-additive Interaction**. Blue buttons perform disproportionately better at the bottom, creating a $1 + 1 > 2$ compounding uplift;
- **$\beta_3 < 0$ ($p < 0.05$)**: **Antagonistic / Sub-additive Interaction**. Both improvements work individually, but combined they saturate or introduce visual clutter ($1 + 1 < 2$);
- **$\beta_3 \approx 0$ (Not Significant)**: Additive independence ($T_3$ uplift equals the sum of $T_1$ and $T_2$ uplifts).

---

### 4. Eligibility, Exposure Definition, Monitoring Period, Power & Decision Rules

#### (1) Eligibility Criteria
- **Included**: All organic and paid new visitor devices arriving at the sign-up funnel landing page;
- **Excluded**: Internal employee IPs, automated bot/crawler traffic, and legacy unsupported browsers.

#### (2) Triggered Exposure Definition (Critical Dilution Defense)
- **Exposure Event Trigger**:
  - For Top buttons: Logged upon initial DOM render;
  - **For Bottom buttons**: Logged **only when the button enters the browser viewport** (via Viewport Intersection Observer).
- **Why this matters**: Including visitors who bounce before scrolling down into the denominator artificially dilutes the bottom variant’s true conversion power (Dilution Bias).

#### (3) Sample Size & Power Analysis
Assuming baseline CTR $p = 5.0\%$, target relative $\text{MDE} = +10\%$ (absolute $\Delta = 0.5\%$), $\alpha = 0.05$, and $\text{Power} = 0.80$:
$$n_{\text{per\_arm}} = \frac{2 \cdot (Z_{\alpha/2} + Z_\beta)^2 \cdot p(1-p)}{\text{MDE}^2} = \frac{2 \cdot (1.96 + 0.84)^2 \cdot 0.05 \times 0.95}{(0.005)^2} \approx 29,800 \text{ exposed users / arm}$$
Total required sample size: **$\approx 120,000$ unique exposed visitors**.

#### (4) Monitoring Period
- **Fixed 14-Day Duration (2 Complete Weekly Cycles)**: Eliminates day-of-week seasonality (weekday vs. weekend conversion divergence).
- **Strict Prohibition of Early Stopping (Peeking Problem)**: Halting early upon observing early $p < 0.01$ inflates false positive rates from 5% to over 30%.

#### (5) Decision Criteria
- **Rule 1**: Primary CTR improvement is statistically significant ($p < 0.05$) with confidence interval lower bound above the business ROI threshold;
- **Rule 2**: Guardrail metrics show no statistically significant degradation;
- **Rule 3**: If $T_3$ is dominant and $\beta_3 \ge 0$ $\implies$ **Roll out $T_3$ (Blue + Bottom)**; if $T_1$ wins while Bottom position causes bounce regressions $\implies$ **Roll out $T_1$ only**.

---

### 5. Downstream Funnel Metrics, Guardrails & Cannibalization

```text
Sign-Up Funnel Metrics Hierarchy:
[Pre-Condition] Viewport Triggered Exposure
      │
[Primary Metric] Sign-Up Button CTR
      │
[Downstream Funnel]
      ├─► Form Start Rate
      ├─► Sign-Up Completion Rate ★ True North Star
      └─► D1 / D7 Active Retention
      │
[Guardrail Metrics]
      ├─► P99 Page Render Latency
      ├─► Landing Page Bounce Rate
      └─► Alternate CTA Cannibalization Rate (e.g. Enterprise Login)
```

1. **Downstream Funnel Metrics**:
   - **Sign-Up Completion Rate**: $\frac{\text{Completed Registrations}}{\text{Exposed Visitors}}$. Protects against deceptive button designs that drive superficial clicks without actual account completions;
   - **D1 / D7 Retention & Activation**: Confirms that newly acquired users represent genuine, high-quality intent.
2. **Guardrail Metrics**:
   - **Technical Performance**: Page Load Time, P99 Time-to-Interactive (TTI), JavaScript Client Error Rate;
   - **User Experience**: Page Bounce Rate, Session Dwell Time.
3. **Cannibalization Analysis**:
   - Monitored CTAs: `"Existing User Log In"`, `"Enterprise Demo Request"`, `"Help / FAQ"`;
   - Verify that prominent sign-up buttons do not severely cannibalize high-LTV enterprise inquiries or existing user login flows.

---

### 6. Risks of Misleading Results & Diagnostic Analyses

```text
Common A/B Testing Misleading Failure Modes & Diagnostic Protocols:
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Threat / Misleading Risk│ Mechanism & Mandatory Diagnostic Analysis                              │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Novelty / Priming    │ Returning visitors click out of curiosity; effect decays rapidly over  │
│    Effect               │ time. ➔ Diagnostic: Cohort analysis tracking Week 1 vs. Week 2 uplift; │
│                         │ filter for pure first-time visitors.                                   │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Sample Ratio Mismatch│ Actual allocated sample counts deviate from 1:1:1:1 ($p < 0.001$).     │
│    (SRM)                │ ➔ Diagnostic: Chi-Square (\chi^2) Goodness-of-Fit test. If SRM occurs,│
│                         │ invalidate experiment and patch routing/logging telemetry.             │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. Heterogeneous Effects│ Overall effect appears neutral due to offsetting sub-populations.      │
│    (HTE)                │ ➔ Diagnostic: Segment by device (Mobile vs Desktop) & acquisition      │
│                         │ channel (Organic vs Paid).                                             │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. Simpson's Paradox   │ Confounding mix shifts across marketing campaigns distort aggregates.  │
│                         │ ➔ Diagnostic: Propensity stratification and weighted average checks.   │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## Module 4: Key Industrial Experimentation Interview FAQs

### Q1: Why must we never perform continuous early stopping ("Peeking") without sequential correction?
> **Answer**:
> Standard fixed-horizon hypothesis testing assumes data is evaluated only once at predetermined sample size $N$. Repeatedly evaluating $p$-values daily and stopping immediately upon crossing $p < 0.05$ creates multiple comparison opportunities over time, inflating the cumulative Type I Error rate from $5\%$ to $30\%\sim40\%$. If dynamic early stopping is required, teams must use rigorous **Sequential Testing frameworks (such as mSPRT or Alpha-Spending Functions)**.

### Q2: How does CUPED (Controlled-experiment Using Pre-Experiment Data) reduce metric variance mathematically?
> **Answer**:
> CUPED leverages a pre-experiment covariate $X$ (which is correlated with outcome $Y$ and strictly independent of treatment assignment) to construct an adjusted estimator:
> $$\hat{Y}_{\text{CUPED}} = Y - \theta (X - \mathbb{E}[X]), \quad \text{where } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
> The adjusted metric variance scales down to $\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - \rho^2)$. If pre- and post-experiment correlation $\rho = 0.8$, variance drops by $64\%$, reducing required sample size or experiment duration by nearly $3\times$ without extra traffic!
