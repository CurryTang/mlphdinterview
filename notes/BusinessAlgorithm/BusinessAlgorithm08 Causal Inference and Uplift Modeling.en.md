# 21 · Causal Inference and Uplift Modeling in RecSys

In industrial search, recommendation, and marketing systems, machine learning models are fundamentally transitioning from **correlation prediction** to **causal incremental decision-making (Causation & Uplift)**.

Standard ranking models (e.g. CTR / CVR estimation) optimize observational conditional probabilities: $P(Y=1 \mid X, T)$. They answer: *"If this user is presented with a recommendation item or issued a discount coupon, what is the probability that they complete a transaction?"* Among loyal, high-intent users, this baseline probability often naturally exceeds 90%—they would purchase regardless of any intervention. Allocating marketing budgets based purely on CVR rankings systematically subsidizes "Free Riders" (natural converters), causing massive capital inefficiency.

**Causal Inference and Uplift Modeling** adopt a counterfactual perspective to explicitly quantify the **Conditional Average Treatment Effect (CATE)** driven strictly by the intervention itself:

$$
\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]
$$

This chapter systematically examines the mathematical foundations of causal inference, experimental compliance pitfalls, the five core Uplift Meta-Learners and causal forest architectures, offline evaluation metrics, and ROI-constrained decision optimization.

---

## Module 1: Causal Inference Fundamentals

### 1. The Neyman-Rubin Potential Outcomes Framework

For any individual unit $i$ in a population:
- **Treatment Indicator**: $T_i \in \{0, 1\}$. $T_i = 1$ denotes an active intervention (e.g., promotional banner, discount coupon, push re-engagement); $T_i = 0$ denotes control (business-as-usual).
- **Potential Outcomes**: $Y_i(1)$ is the potential outcome if treated; $Y_i(0)$ is the potential outcome if untreated.

#### The Fundamental Problem of Causal Inference

In the physical world, treatment assignments are mutually exclusive. We can only observe one of the potential outcomes for any given individual. The unobserved outcome is the **counterfactual**, permanently missing from observational data:

$$
Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)
$$

> **📊 The Fundamental Problem: Counterfactual Missing Data Matrix**
>
> 📌 **Core Takeaway**: Individual Treatment Effect (ITE) is strictly unobservable. Causal inference leverages randomized experiments or conditional unconfoundedness assumptions to robustly impute counterfactual expectations at cohort scale.

| Unit $i$ | Features $X_i$ | Assignment $T_i$ | Observed $Y_i$ | $Y_i(1)$ (Treated) | $Y_i(0)$ (Control) | True Lift $\tau_i$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| User 1 | High Activity / Tier 1 | **1 (Treatment)** | **1** | 1 (Observed) | ? (Counterfactual) | Unobservable |
| User 2 | Low Activity / Tier 3 | **0 (Control)** | **0** | ? (Counterfactual) | 0 (Observed) | Unobservable |
| User 3 | Mid Activity / Tier 2 | **1 (Treatment)** | **1** | 1 (Observed) | ? (Counterfactual) | Unobservable |

#### Three Levels of Treatment Effects (ATE vs. ATT vs. CATE)

In production environments, causal inference is never an abstract mathematical formula—it directly maps to data warehouse columns (Hive / ClickHouse) and real-time decision microservices:

---

##### 1. Production Entity Mapping (What are Feature $X$, Treatment $T$, and Outcome $Y$ in Practice?)

Consider two quintessential industrial scenarios: **"E-Commerce / Food Delivery Promotion Coupon Subsidy"** and **"App Push Notification Retention"**:

> **📋 Production Data Warehouse Schema (Hive / Feature Store Slice)**

| Variable Class | Warehouse Column Name | Data Type &amp; Meaning | Role in Causal Modeling |
| :--- | :--- | :--- | :--- |
| **Covariates $X$**<br>(Feature Vector) | `user_age_group`<br>`city_tier`<br>`device_brand_price`<br>`active_days_30d`<br>`pay_gmv_30d`<br>`cart_unpaid_cnt_7d`<br>`hist_coupon_use_rate`<br>`cur_browse_cate_l1` | • User static demographics (age, city tier, device price bracket)<br>• RFM consumption history (30-day active days, GMV spend)<br>• Strong unconverted intent (cart adds unpaid in past 7 days)<br>• Price sensitivity signals (historical coupon redemption rate)<br>• Real-time context (current browsing category: high-margin 3C vs low-margin groceries) | Model inputs. Causal estimators (e.g. Causal Forest / X-Learner) use $X$ to isolate effect heterogeneity—distinguishing price-sensitive high-intent buyers from loyal users who purchase regardless. |
| **Treatment $T$**<br>(Intervention) | `is_coupon_issued`<br>`treatment_type` | • Binary indicator: `1` (Issue \$20 discount coupon), `0` (Control: natural organic feed without discount)<br>• Multi-dose treatment: `T ∈ {0: None, 1: \$5 off \$50, 2: \$20 off \$100}` | The physical intervention signal dispatched by the system. In randomized A/B tests, determined by server-side hash bucketing. |
| **Outcome $Y$**<br>(Target Response) | `is_order_paid_24h`<br>`pay_gmv_24h`<br>`net_profit_24h` | • Binary classification target: paid within 24 hours (`0 or 1`)<br>• Continuous regression target: Total GMV transacted within 24h<br>• Net Profit: `GMV * TakeRate - (CouponCost * IsRedeemed)` | The business metric being stimulated. Optimizing gross GMV alone frequently burns marketing budgets; mature systems model Net Commercial Profit or cost-constrained conversion. |

---

---

##### 2. ATE, ATT, and CATE across the Business Lifecycle

| Metric | Mathematical Definition | Intuitive Meaning | How to Compute in Production / Internships | Key Application &amp; Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **ATE**<br>(Average Treatment Effect) | $$\mathbb{E}[Y(1) - Y(0)]$$ | **"If we universalize subsidies to all users, do we make money overall?"**<br><br>Expected incremental effect per capita if the intervention is rolled out to the entire population. | **Randomized A/B Test Difference in Means**:<br>Computed across randomized sample buckets:<br>$$\widehat{\text{ATE}} = \bar{Y}_{T=1} - \bar{Y}_{T=0}$$<br>p-value verified via two-sample Welch t-test. | **Business Proposal &amp; Global Viability Review**:<br>• Executive reporting: "Does this strategy generate net positive value globally?"<br>• **Pitfall**: If ATE = +0.01 (+1% conversion), issuing \$10 coupons to 10M users costs \$100M. If 1% extra conversions yield only \$2M commission, universal rollout yields a **\$98M net loss!** ATE must never be used for resource-constrained budget allocation. |
| **ATT**<br>(Average Treatment Effect on the Treated) | $$\mathbb{E}[Y(1) - Y(0) \mid T = 1]$$ | **"Did the cohort selected by operations rules actually benefit?"**<br><br>Net incremental gain strictly within the subpopulation targeted by current rules or algorithms. | **Targeted Cohort Holdout Test**:<br>For an operational rule (e.g. "inactive churn risk users in past 30 days"), withhold a 5%-10% holdout control (qualified but untreated):<br>$$\widehat{\text{ATT}} = \bar{Y}_{T=1, \text{rule}} - \bar{Y}_{T=0, \text{rule}}$$ | **Auditing Legacy Operations Rules &amp; Free-Riders**:<br>• Standard task for interns/engineers: audit rule efficacy.<br>• If ATT ≈ 0 or negative, users in that cohort would have converted organically (giving free money to bargain hunters) or are unresponsive. Provides data to decommission ineffective manual rules. |
| **CATE**<br>(Conditional Average Treatment Effect) | $$\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$$ | **"For a user with covariate vector $X$, what is the net conversion gain from treatment?"**<br><br>Quantifies individual effect heterogeneity; **the foundational target of Uplift Modeling**. | **Pointwise Causal ML Model Inference**:<br>Train X-Learner, Causal Forest, or DR-Learner. Score arriving user feature vector $X_i$ to output scalar prediction:<br>$$\hat{\tau}(X_i) \in (-\infty, +\infty)$$ | **Real-Time Policy Optimization &amp; Budget Knapsack**:<br>1. **Classify 4-quadrant cohorts**: suppress natural buyers &amp; sleeping dogs;<br>2. **Greedy Marginal ROI Thresholding**: maximize total incremental return under budget cap. |

---

##### 3. End-to-End CATE Production Pipeline (From Scoring to Real-Time Serving)

How do ML engineers turn trained CATE estimators into real revenue? The end-to-end production workflow follows four steps:

1. **Step 1: Pointwise Scoring &amp; 4-Quadrant User Classification**
   When a user session begins, the feature service extracts $X_i$, and the CATE model outputs $\hat{\tau}(X_i)$. Combined with baseline organic probability $P(Y=1 \mid X_i, T=0)$, users are segmented into four archetypes:
   - **Persuadables ($\hat{\tau}(X_i) \gg 0$)**: On the fence; treatment triggers purchase $\implies$ **Primary target cohort for subsidies**.
   - **Sure Things ($\hat{\tau}(X_i) \approx 0, Y_i(0)=1$)**: Loyal high-intent buyers who convert regardless $\implies$ **Never subsidize; capture organic margin**.
   - **Lost Causes ($\hat{\tau}(X_i) \approx 0, Y_i(0)=0$)**: Inactive accounts that will not convert $\implies$ **Suppress intervention to save costs**.
   - **Sleeping Dogs ($\hat{\tau}(X_i) < 0$)**: Disturbed by notifications/popups, leading to uninstalls/mutes $\implies$ **Hard blacklist from outreach**.

2. **Step 2: Offline Decile Monotonicity Verification**
   On the test holdout set, sort users by $\hat{\tau}(X_i)$ in descending order into 10 equal deciles (Decile 1 to 10). Measure empirical test-set uplift $\bar{Y}_{T=1} - \bar{Y}_{T=0}$ per bucket.
   A deployment-ready model must show **strict monotonic descent**: Decile 1 achieves the highest empirical uplift (+8%), while Decile 10 approaches zero or negative.

3. **Step 3: Constrained Budget Knapsack Thresholding (Marginal ROI)**
   With campaign budget $B$ (\$500k), treatment cost $c(X_i)$ (\$20 coupon), and gross margin per order $v$:
   
   $$
   \text{Marginal\_ROI}_i = \frac{\hat{\tau}(X_i) \times v}{c(X_i)}
   $$
   
   Sort candidates descending by $\text{Marginal\_ROI}_i$, greedily allocating treatment until cumulative cost $\sum c(X_i) = B$. The marginal score of the last funded user defines threshold $\theta$.

4. **Step 4: Online Gateway Serving &amp; Interception**
   When the live request hits the recommendation or subsidy service:
   
   $$
   \text{Decision}(X_i) = 
   \begin{cases} 
   \text{Dispatch \$20 Voucher (Treatment)}, & \text{if } \frac{\hat{\tau}(X_i) \times v}{c(X_i)} \ge \theta \\
   \text{Serve Organic Feed (Control)}, & \text{if } \frac{\hat{\tau}(X_i) \times v}{c(X_i)} < \theta
   \end{cases}
   $$

---

---

### 2. The Three Fundamental Identification Assumptions

To identify $\tau(X)$ from observational or experimental data, the data generation process must satisfy three axiomatic assumptions:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 20px 0;">
  <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: #ffffff; border-top: 4px solid #3b82f6;">
    <div style="font-weight: bold; font-size: 14px; color: #1e293b; margin-bottom: 8px;">1. Unconfoundedness (Ignorability)</div>
    <div style="font-family: monospace; font-size: 13px; color: #2563eb; background: #eff6ff; padding: 6px; border-radius: 4px; margin-bottom: 8px;">
      (Y(1), Y(0)) ⟂ T | X
    </div>
    <div style="font-size: 13px; color: #475569; line-height: 1.5;">
      Conditioned on observable covariates X, treatment assignment T is statistically independent of potential outcomes. There are no unobserved confounders driving both assignment and conversion.
    </div>
  </div>
  <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: #ffffff; border-top: 4px solid #10b981;">
    <div style="font-weight: bold; font-size: 14px; color: #1e293b; margin-bottom: 8px;">2. Positivity / Common Support</div>
    <div style="font-family: monospace; font-size: 13px; color: #059669; background: #ecfdf5; padding: 6px; border-radius: 4px; margin-bottom: 8px;">
      0 &lt; P(T=1 | X=x) &lt; 1, ∀x
    </div>
    <div style="font-size: 13px; color: #475569; line-height: 1.5;">
      Every subpopulation in feature space has non-zero probability of being assigned either condition. If P(T=1|X)=0 for a segment, counterfactuals cannot be evaluated without parametric extrapolation.
    </div>
  </div>
  <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: #ffffff; border-top: 4px solid #8b5cf6;">
    <div style="font-weight: bold; font-size: 14px; color: #1e293b; margin-bottom: 8px;">3. SUTVA</div>
    <div style="font-family: monospace; font-size: 13px; color: #7c3aed; background: #f5f3ff; padding: 6px; border-radius: 4px; margin-bottom: 8px;">
      Y_i(T_1,...,T_n) = Y_i(T_i)
    </div>
    <div style="font-size: 13px; color: #475569; line-height: 1.5;">
      ① <b>No Interference</b>: Individual i's outcome is unaffected by treatment assignments of unit j;<br>
      ② <b>No Hidden Variations</b>: Treatment dosage and delivery mechanism are uniform across all treated units.
    </div>
  </div>
</div>

---

## Module 2: Experimental Design & Causal Pitfalls in Production (RCT & Pitfalls)

### 1. Compliance & Cohort Definitions (Compliance: ITT vs. As-Treated / CACE)

In recommendation and marketing A/B tests, systems control assignment decisions, but users decide whether to interact. This causes a split between **Randomized Assignment ($Z$)** and **Actual Treatment Received ($T$)**:

<div style="margin: 24px 0; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; background: #ffffff;">
  <div style="font-weight: bold; font-size: 14px; color: #1e293b; margin-bottom: 12px;">
    📐 Causal DAG: Compliance Gap and Endogenous Selection Bias
  </div>
  <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-around; gap: 12px; padding: 12px 0;">
    <div style="border: 2px solid #3b82f6; border-radius: 6px; padding: 12px 16px; text-align: center; background: #eff6ff;">
      <div style="font-weight: bold; color: #1e40af;">Assignment Z ∈ {0, 1}</div>
      <div style="font-size: 12px; color: #3b82f6;">Randomized Server Hash</div>
    </div>
    <div style="font-size: 20px; font-weight: bold; color: #94a3b8;">➔</div>
    <div style="border: 2px solid #f59e0b; border-radius: 6px; padding: 12px 16px; text-align: center; background: #fffbeb;">
      <div style="font-weight: bold; color: #92400e;">Actual Receipt T ∈ {0, 1}</div>
      <div style="font-size: 12px; color: #d97706;">User opens / redeems coupon</div>
    </div>
    <div style="font-size: 20px; font-weight: bold; color: #94a3b8;">➔</div>
    <div style="border: 2px solid #10b981; border-radius: 6px; padding: 12px 16px; text-align: center; background: #ecfdf5;">
      <div style="font-weight: bold; color: #065f46;">Outcome Y ∈ {0, 1}</div>
      <div style="font-size: 12px; color: #059669;">Final conversion / GMV</div>
    </div>
  </div>
  <div style="margin-top: 12px; border-top: 1px dashed #cbd5e1; padding-top: 12px; font-size: 13px; color: #475569; display: flex; align-items: center; gap: 8px;">
    <span style="color: #dc2626; font-weight: bold;">⚠️ Unobserved Confounder U (Inherent Purchase Intent)</span>
    <span>Simultaneously influences actual redemption T and purchase outcome Y, opening an unblocked backdoor path!</span>
  </div>
</div>

#### Three Analytic Cohorts

1. **Intention-to-Treat (ITT)**:
   $$\text{ITT} = \mathbb{E}[Y \mid Z=1] - \mathbb{E}[Y \mid Z=0]$$
   - **Mechanism**: Groups users strictly by initial algorithmic assignment $Z$, preserving pure randomization.
   - **Production Standard**: Standard policy metric. Because financial costs scale with eligibility and delivery attempts, ITT truthfully measures net portfolio ROI.
2. **As-Treated (AT / Naive Comparison) and Selection Bias**:
   $$\text{AT} = \mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]$$
   - **Severe Flaw**: Compares users who actually redeemed coupons against non-redeemers. Because high-intent users actively redeem, $T=1$ users are naturally predisposed to purchase. This creates massive **endogenous selection bias**, artificially inflating measured treatment effects.
3. **Complier Average Causal Effect (CACE / LATE)**:
   Using random assignment $Z$ as an **Instrumental Variable (IV)** for actual uptake $T$:
   $$\text{CACE} = \frac{\text{ITT}_Y}{\text{ITT}_T} = \frac{\mathbb{E}[Y \mid Z=1] - \mathbb{E}[Y \mid Z=0]}{\mathbb{E}[T \mid Z=1] - \mathbb{E}[T \mid Z=0]}$$

---

### 2. Treatment Propensity & Inverse Probability Weighting (IPW)

In targeted experiments, the intervention probability varies with features $X$:

$$
e(X) = P(T=1 \mid X)
$$

The standard **Inverse Probability Weighting (IPW)** estimator is:

$$
\hat{\tau}_{\text{IPW}} = \frac{1}{N} \sum_{i=1}^N \left[ \frac{T_i Y_i}{e(X_i)} - \frac{(1 - T_i) Y_i}{1 - e(X_i)} \right]
$$

#### Variance Explosion & Propensity Clipping
When propensity approaches extreme boundaries (e.g., $e(X_i) = 0.001$), weight $\frac{1}{e(X_i)} = 1000$. A few positive conversions destabilize estimator variance.
Production systems enforce symmetric trimming:

$$
\tilde{e}(X) = \max(\alpha, \min(1 - \alpha, e(X))), \quad \alpha \in [0.01, 0.05]
$$

---

### 3. SUTVA Violation & Network Interference

In shared two-sided marketplaces and social networks, unit independence frequently breaks:
- **Two-Sided Supply Cannibalization (Negative Spillover)**: In ride-hailing or delivery promotions, subsidized riders absorb nearby drivers, increasing wait times and drop-offs for control riders. A/B test lift contains cannibalization gains that vanish upon full release.
- **Social Network Contagion (Positive Spillover)**: Treated users share videos or referral links with control peers, attenuating measured treatment lift.

<div style="border-left: 4px solid #0284c7; background: #f0f9ff; padding: 12px 16px; border-radius: 4px; margin: 16px 0; font-size: 13px; color: #0369a1;">
  <b>🚀 Production Mitigation Architectures:</b><br>
  1. <b>Cluster Randomization</b>: Partition users by geospatial boundaries (Uber H3 / Hexagonal partitions) or Louvain social graph clusters to encapsulate local spillovers.<br>
  2. <b>Switchback Experiments</b>: Alternate treatment and control states across the entire market in synchronized time slices (e.g. 30-minute intervals), decoupling spatial supply contention.
</div>

---

## Module 3: Uplift Modeling Algorithm Family (Meta-Learners & Tree Approaches)

The goal is to estimate $\hat{\tau}(X)$ from finite empirical data. Industrial practice is anchored on Meta-Learners and Causal Forests.

> **🧩 Comparison Matrix of Five Core Uplift Architectures**

| Architecture | Model Topology &amp; Estimator | Formulation of $\tau(X)$ | Core Production Strength | Production Pitfall &amp; Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| **S-Learner**<br>(Single Learner) | Single model $\mu(X, T)$; $\hat{\tau} = \mu(X, 1) - \mu(X, 0)$ | $\mu(X, T) = \mathbb{E}[Y \mid X, T]$<br>$\hat{\tau}(X) = \hat{\mu}(X, 1) - \hat{\mu}(X, 0)$ | Lowest development barrier; reuses existing ranking pipelines | **Regularization Bias**: 1D treatment $T$ is submerged by high-dimensional $X$; predictions shrink to 0 |
| **T-Learner**<br>(Two Learners) | Separate models $\mu_1(X), \mu_0(X)$; $\hat{\tau} = \mu_1 - \mu_0$ | $\mu_1(X) = \mathbb{E}[Y \mid X, T=1]$<br>$\mu_0(X) = \mathbb{E}[Y \mid X, T=0]$<br>$\hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$ | Preserves treatment signal without feature interference | **Variance Explosion on Class Imbalance**: If treated ratio is 5%, $\mu_1(X)$ suffers large estimation error |
| **X-Learner**<br>(Crossover Learner) | Two-stage counterfactual imputation + propensity weighting | Stage 1: Impute counterfactual residuals<br>$D_1 = Y_1 - \hat{\mu}_0(X_1)$<br>$D_0 = \hat{\mu}_1(X_0) - Y_0$<br>Stage 2: Weighted lift combination<br>$\hat{\tau}(X) = e(X)\hat{\tau}_0(X) + (1-e(X))\hat{\tau}_1(X)$ | **Optimal for severe class imbalance** ($P(T=1) \ll P(T=0)$); minimal variance | Requires 4 base models + 1 propensity model; highest offline orchestration overhead |
| **DR-Learner**<br>(Doubly Robust) | Constructs AIPW pseudo-outcome $Y^{\text{DR}}$; regresses directly | $Y^{\text{DR}} = \hat{\mu}_1(X) - \hat{\mu}_0(X) + \frac{T(Y - \hat{\mu}_1(X))}{e(X)} - \frac{(1-T)(Y - \hat{\mu}_0(X))}{1-e(X)}$<br>$\hat{\tau} = \arg\min_f \sum (Y_i^{\text{DR}} - f(X_i))^2$ | **Double Robustness &amp; Neyman Orthogonality**: Unbiased if either outcome or propensity model is correct | Extreme propensity scores $e(X) \to 0$ cause high variance; requires propensity clipping |
| **Causal Forest**<br>(Honest Forest) | Honest Splitting on heterogeneity variance + adaptive weighting | $\hat{\tau}(X) = \sum_{i=1}^n \alpha_i(X) Y_i$<br>Weights $\alpha_i(X)$ determined by leaf co-occurrence | Non-parametric point estimates with rigorous asymptotic normality &amp; valid confidence intervals | Splitting degrades on ultra-high-dimensional sparse ID features; higher serving latency |

---

---

### 1. S-Learner & Regularization Bias

Treats treatment $T$ as a standard scalar feature concatenated with $X$:

$$\mu(X, T) = \mathbb{E}[Y \mid X, T], \qquad \hat{\tau}_{\text{S}}(X) = \mu(X, 1) - \mu(X, 0)$$

#### Failure Mode: Regularization Bias
When $X$ spans hundreds of features, tree splitters select variables that minimize aggregate MSE across the full dataset. The single dimension $T$ is easily starved of split selections. Regularization penalties shrink its weight towards zero, causing $\hat{\tau}_{\text{S}}(X) \approx 0$ everywhere.

---

### 2. T-Learner & Imbalance Variance

Trains two isolated models without parameter sharing:
- Control model: $\mu_0(X) = \mathbb{E}[Y \mid X, T=0]$;
- Treatment model: $\mu_1(X) = \mathbb{E}[Y \mid X, T=1]$.

$$\hat{\tau}_{\text{T}}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$$

#### Core Limitation:
When only 5% of users receive subsidies (common in marketing), $\mu_1(X)$ suffers severe sample starvation, producing high variance that pollutes the difference $\hat{\mu}_1 - \hat{\mu}_0$.

---

### 3. X-Learner: Two-Stage Imputation for Imbalanced Treatments

Künzel et al. (2019) engineered the X-Learner to resolve sample imbalance by cross-imputing unobserved counterfactuals.

```text
Stage 1 (Base Models):
  Train μ₀(X) on control, and μ₁(X) on treatment.

Stage 2 (Counterfactual Residual Imputation):
  Treated unit counterfactual residual:  D₁ = Y₁ - μ₀(X₁)
  Control unit counterfactual residual:  D₀ = μ₁(X₀) - Y₀

Stage 3 (Residual Estimation):
  Train τ₁(X) on X₁ targeting D₁
  Train τ₀(X) on X₀ targeting D₀

Stage 4 (Propensity Fusion):
  τ(X) = e(X) · τ₀(X) + (1 - e(X)) · τ₁(X)
```

#### Why Does This Stabilize Variance?
When treatment units are scarce ($e(X) \to 0$):
- Weight $1 - e(X) \approx 1$, so the final prediction relies primarily on $\hat{\tau}_1(X)$;
- $\hat{\tau}_1(X)$ targets $D_1 = Y_1 - \hat{\mu}_0(X_1)$, where the baseline $\hat{\mu}_0$ was trained on the **massive control group (95%+ of data)**;
- This transfers the statistical power of the large control group to anchor the small treatment group's residuals.

---

### 4. DR-Learner (Doubly Robust / AIPW)

Constructs an augmented pseudo-outcome $Y_i^{\text{DR}}$ that satisfies **Neyman Orthogonality**:

$$
Y_i^{\text{DR}} = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_i (Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1 - T_i) (Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)}
$$

Then directly minimizes:

$$
\min_{\tau} \sum_{i=1}^N \left( Y_i^{\text{DR}} - \tau(X_i) \right)^2
$$

#### Mathematical Proof of Double Robustness

Taking conditional expectations $\mathbb{E}[Y^{\text{DR}} \mid X]$:

$$
\mathbb{E}[Y^{\text{DR}} \mid X] - \tau(X) = \left( \frac{e(X) - \hat{e}(X)}{\hat{e}(X)} \right) (\mu_1(X) - \hat{\mu}_1(X)) - \left( \frac{\hat{e}(X) - e(X)}{1 - \hat{e}(X)} \right) (\mu_0(X) - \hat{\mu}_0(X))
$$

- If propensity $\hat{e}(X) = e(X)$ is consistent: Error is zero regardless of outcome model quality.
- If outcome models $\hat{\mu}_0 = \mu_0, \hat{\mu}_1 = \mu_1$ are consistent: Error is zero regardless of propensity model quality.
- Achieves semi-parametric efficiency bound $O(n^{-1/2})$ convergence even if base learners converge at non-parametric $O(n^{-1/4})$ rates.

---

### 5. Causal Forest & Honest Splitting (Athey & Imbens)

Extends random forests to maximize **Treatment Effect Heterogeneity**:

$$
\Delta(L, R) = \frac{N_L \cdot N_R}{N_P^2} (\hat{\tau}_L - \hat{\tau}_R)^2
$$

> **🌲 Causal Forest: Honest Splitting Workflow**
>
> * **Sub-sample A: $S_{\text{split}}$ (Tree Architecture)**:
>   Used solely to evaluate splitting criterion $\Delta(L, R)$ and establish leaf boundaries. Discarded once tree topology is fixed.
> * **Sub-sample B: $S_{\text{est}}$ (Leaf Point Estimation)**:
>   Fresh samples dropped into pre-constructed leaves to evaluate clean local treatment effects:
>   $$\hat{\tau}_{\text{leaf}} = \bar{Y}_{1, \text{leaf}} - \bar{Y}_{0, \text{leaf}}$$
>
> 📌 **Statistical Guarantee**: Honest Splitting eliminates adaptive selection bias. Estimators are asymptotically Gaussian, delivering exact confidence intervals and hypothesis tests.

---

## Module 4: Offline Evaluation & Business Policy Optimization

Because ground-truth $\tau_i$ is unobservable, classification metrics (AUC, LogLoss, RMSE) are unusable for Uplift.

### 1. Uplift Ranking Metrics: AUUC & Qini Curves

Sort test set instances descending by predicted uplift $\hat{\tau}(X)$.

#### 1. Cumulative Uplift Curve & AUUC
At cutoff $k$:

$$
\text{Gain}(k) = \left( \frac{\sum_{i=1}^k Y_i \cdot T_i}{N_T(k)} - \frac{\sum_{i=1}^k Y_i \cdot (1 - T_i)}{N_C(k)} \right) \cdot (N_T(k) + N_C(k))
$$

#### 2. Qini Curve & Qini Score (Imbalance Adjusted)
Adjusts for localized sample size fluctuations between treatment and control:

$$
\text{Qini}(k) = \sum_{i=1}^k Y_i \cdot T_i - \left( \sum_{i=1}^k Y_i \cdot (1 - T_i) \right) \cdot \frac{N_T(k)}{N_C(k)}
$$

---

### 2. Decile Uplift Monotonicity Validation

Divide test instances into 10 deciles ranked by predicted uplift:

$$
\Delta \bar{Y}_k = \bar{Y}_{1, k} - \bar{Y}_{0, k}, \quad k \in \{1, 2, \dots, 10\}
$$

<div style="margin: 24px 0; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; background: #ffffff;">
  <div style="background: #f8fafc; padding: 12px 16px; border-bottom: 1px solid #cbd5e1; font-weight: bold; font-size: 14px;">
    📈 Production Uplift Decile Distribution
  </div>
  <div style="padding: 20px; display: flex; align-items: flex-end; justify-content: space-between; height: 180px; gap: 8px; border-bottom: 1px solid #e2e8f0;">
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+18.5%</div>
      <div style="width: 100%; height: 130px; background: #22c55e; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D1</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+14.2%</div>
      <div style="width: 100%; height: 100px; background: #4ade80; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D2</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+10.1%</div>
      <div style="width: 100%; height: 75px; background: #86efac; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D3</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+7.4%</div>
      <div style="width: 100%; height: 55px; background: #bbf7d0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D4</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+4.8%</div>
      <div style="width: 100%; height: 38px; background: #bbf7d0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D5</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #64748b; margin-bottom: 4px;">+2.3%</div>
      <div style="width: 100%; height: 22px; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D6</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #64748b; margin-bottom: 4px;">+0.8%</div>
      <div style="width: 100%; height: 12px; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D7</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #64748b; margin-bottom: 4px;">+0.1%</div>
      <div style="width: 100%; height: 5px; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D8</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #ef4444; margin-bottom: 4px;">-1.2%</div>
      <div style="width: 100%; height: 15px; background: #fca5a5; border-radius: 0 0 4px 4px; margin-top: 15px;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D9</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #b91c1c; margin-bottom: 4px;">-4.6%</div>
      <div style="width: 100%; height: 35px; background: #ef4444; border-radius: 0 0 4px 4px; margin-top: 35px;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D10</div>
    </div>
  </div>
  <div style="padding: 10px 16px; font-size: 12px; color: #475569; background: #f8fafc;">
    🔍 Two Acceptance Criteria: ① Strict monotonic downward slope from D1 to D10; ② Negative lift in D9/D10 isolating negative responders ("Sleeping Dogs").
  </div>
</div>

---

### 3. Four-Quadrant Uplift Matrix & Knapsack Budget Optimization

Units fall into four behavioral quadrants based on potential outcomes $(Y(1), Y(0))$:

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0;">
  <div style="border: 2px solid #22c55e; border-radius: 8px; padding: 14px; background: #f0fdf4;">
    <div style="font-weight: bold; color: #15803d; font-size: 14px;">🎯 1. Persuadables</div>
    <div style="font-family: monospace; font-size: 12px; color: #166534; margin: 4px 0;">Y(1) = 1,  Y(0) = 0  ➔  τ(X) &gt; 0</div>
    <div style="font-size: 13px; color: #374151; line-height: 1.4;">
      Convert only if treated.<br>
      <b>Policy Decision</b>: <b>Primary targeting audience; allocate budget here.</b>
    </div>
  </div>
  <div style="border: 2px solid #94a3b8; border-radius: 8px; padding: 14px; background: #f8fafc;">
    <div style="font-weight: bold; color: #475569; font-size: 14px;">☕ 2. Sure Things</div>
    <div style="font-family: monospace; font-size: 12px; color: #475569; margin: 4px 0;">Y(1) = 1,  Y(0) = 1  ➔  τ(X) = 0</div>
    <div style="font-size: 13px; color: #374151; line-height: 1.4;">
      Convert regardless of intervention.<br>
      <b>Policy Decision</b>: <b>Do NOT treat; protect gross margins.</b>
    </div>
  </div>
  <div style="border: 2px solid #cbd5e1; border-radius: 8px; padding: 14px; background: #f8fafc;">
    <div style="font-weight: bold; color: #64748b; font-size: 14px;">🧱 3. Lost Causes</div>
    <div style="font-family: monospace; font-size: 12px; color: #64748b; margin: 4px 0;">Y(1) = 0,  Y(0) = 0  ➔  τ(X) = 0</div>
    <div style="font-size: 13px; color: #374151; line-height: 1.4;">
      Never convert regardless of incentives.<br>
      <b>Policy Decision</b>: <b>Do NOT treat; avoid wasteful spend.</b>
    </div>
  </div>
  <div style="border: 2px solid #ef4444; border-radius: 8px; padding: 14px; background: #fef2f2;">
    <div style="font-weight: bold; color: #b91c1c; font-size: 14px;">⛔ 4. Sleeping Dogs</div>
    <div style="font-family: monospace; font-size: 12px; color: #991b1b; margin: 4px 0;">Y(1) = 0,  Y(0) = 1  ➔  τ(X) &lt; 0</div>
    <div style="font-size: 13px; color: #374151; line-height: 1.4;">
      Active organically; notifications cause churn or unsubscription.<br>
      <b>Policy Decision</b>: <b>Strict negative blacklist; never intervene.</b>
    </div>
  </div>
</div>

#### Budget-Constrained Knapsack ROI Optimization

With per-user intervention cost $c(X)$, conversion gross margin value $V$, and budget ceiling $B$:

$$
\max_{\{a_i \in \{0, 1\}\}} \sum_{i=1}^N a_i \Big[ V \cdot \hat{\tau}(X_i) - c(X_i) \Big] \quad \text{s.t.} \quad \sum_{i=1}^N a_i \cdot c(X_i) \le B
$$

At industrial scale ($N \ge 10^7$), solve via the greedy fractional knapsack approximation:

$$
\text{Priority}(X_i) = \frac{\hat{\tau}(X_i)}{c(X_i)}
$$

Intervene strictly down the ranked list until budget $B$ is exhausted or marginal net benefit $V \cdot \hat{\tau}(X) - c(X) < 0$.

---

## Module 5: Production Python Blueprint

```python
import numpy as np


class XLearner:
    """
    Production-ready X-Learner:
    Two-stage counterfactual imputation with propensity score weighting.
    """
    def __init__(self, base_model_cls, **model_params):
        self.m0 = base_model_cls(**model_params)
        self.m1 = base_model_cls(**model_params)
        self.tau0 = base_model_cls(**model_params)
        self.tau1 = base_model_cls(**model_params)
        self.fixed_p = 0.5

    def fit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray, p: np.ndarray = None):
        idx_0 = np.where(T == 0)[0]
        idx_1 = np.where(T == 1)[0]

        X0, y0 = X[idx_0], y[idx_0]
        X1, y1 = X[idx_1], y[idx_1]

        # Stage 1: Base outcome models
        self.m0.fit(X0, y0)
        self.m1.fit(X1, y1)

        # Stage 2: Impute counterfactuals & compute residuals
        D1 = y1 - self.m0.predict(X1)
        D0 = self.m1.predict(X0) - y0

        # Stage 3: Fit imputed effect models
        self.tau1.fit(X1, D1)
        self.tau0.fit(X0, D0)

        self.fixed_p = p if p is not None else float(len(idx_1)) / len(T)

    def predict(self, X: np.ndarray, p: np.ndarray = None) -> np.ndarray:
        # Stage 4: Propensity fusion
        tau0_pred = self.tau0.predict(X)
        tau1_pred = self.tau1.predict(X)
        prop = p if p is not None else self.fixed_p
        return prop * tau0_pred + (1.0 - prop) * tau1_pred
```

---

## Module 6: Interview Takeaways & Architecture Decision Matrix

| Dimension | Canonical Interview Question | Core Mechanism & Answer Strategy |
| :--- | :--- | :--- |
| **Foundations** | Why is supervised training on coupon redeemers invalid? | **Endogenous Selection Bias**: Redemption is driven by latent user intent $U$. Comparing redeemers to non-redeemers confounds intent with coupon impact. Must preserve ITT random assignment or use assignment as an IV. |
| **Model Selection** | In a 5% coupon vs 95% control setup, T-Learner or X-Learner? | **Choose X-Learner**: T-Learner's treatment model suffers severe variance; X-Learner anchors treatment residuals against the high-precision 95% control baseline model. |
| **Causal Forests** | How does Honest Splitting produce valid confidence intervals? | Splitting set $S_{\text{split}}$ and estimation set $S_{\text{est}}$ are disjoint. Leaf points did not inform leaf boundaries, eliminating adaptive bias and unlocking asymptotic Gaussianity via CLT. |
| **Production Failures** | Why does high offline Qini fail to translate into online lift? | Check: ① **SUTVA cannibalization** (treatment crowds out control supply); ② **Propensity drift** (inference distribution shifts from training); ③ **Compliance drop** (low delivery/exposure rate diluting ITT). |
| **Evaluation** | Why must decile curves strictly dip negative at the tail? | To verify identification of **Sleeping Dogs**. If the tail is negative, the model successfully isolated users disturbed by push messages, enabling production blacklisting. |
