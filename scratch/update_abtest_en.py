import os

with open('notes/BusinessAlgorithm/BusinessAlgorithm08 Industrial ML AB Testing Experimentation.en.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace header overview
old_overview = """This note systematically covers the following key areas:
1. **Foundations of Industrial ML & A/B Testing: Why offline metrics diverge from online business impact**
2. **End-to-End A/B Testing Lifecycle (Design, Layered Routing, CUPED Variance Reduction, SRM Checks & Decision Rules)**
3. **Flagship Case Study: Sign-Up Funnel Button Color & Position Full Factorial Experiment Design**"""

new_overview = """This note systematically covers the following key areas:
1. **Foundations of Industrial ML & A/B Testing: Why offline metrics diverge from online business impact**
2. **End-to-End A/B Testing Lifecycle (Design, Layered Routing, CUPED Variance Reduction, SRM Checks & Decision Rules)**
3. **Flagship Case Study 1: Sign-Up Funnel Button Color & Position Full Factorial Experiment Design**
4. **Flagship Case Study 2: E-Commerce Product Detail Page (PDP) Conversion Attribution (CTR Up, CVR Flat) & Small-Sample Tier Inference**"""

content = content.replace(old_overview, new_overview)

# Replace Module 4
target_module4 = """## Module 4: High-Yield A/B Testing Interview Rapid-Fire

### Q1: Why is continuous peeking and early stopping strictly prohibited even if p-value reaches 0.001 on Day 3?
> **Answer**:
> 1. **Alpha Inflation via Multi-Testing**: Standard hypothesis tests assume fixed sample sizes. Repeatedly evaluating data introduces severe False Positive Rate inflation (jumping from $5\%$ up to $20\%\sim40\%$);
> 2. **Day-of-Week Seasonality**: User behavior on weekends differs fundamentally from weekdays. Evaluating uncompleted weekly cycles creates severe temporal selection bias;
> 3. **Unresolved Novelty Effects**: Early spikes frequently represent novelty curiosity that rapidly decays over time. If dynamic stopping is required, one must employ formal **Sequential Testing (e.g., mSPRT / Alpha-spending functions)**.

### Q2: What is the underlying mathematical principle of CUPED variance reduction?
> **Answer**:
> 1. **Core Mechanism**: Leverages pre-experiment user baseline covariates $X$ (which correlate with outcome $Y$ but are orthogonal to the treatment) to subtract user-level invariant variance;
> 2. **Adjusted Metric**:
>    $$\hat{Y}_{\text{CUPED}} = Y - \theta(X - \mathbb{E}[X]), \quad \text{where } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
> 3. **Variance Scaling Factor**:
>    $$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - \rho^2)$$
>    With correlation $\rho = 0.8$, metric variance drops by $64\%$ (retaining only $36\%$), reducing required sample size by nearly $3\times$ without allocating extra traffic!"""

new_module4_and_5 = """## Module 4: Flagship Case Study 2: E-Commerce PDP UI Redesign (CTR Up, CVR Flat) & Small-Sample Tier Inference

> 📌 **Business Scenario**: An e-commerce platform rolls out a UI overhaul of the Product Detail Page (PDP) header and CTA. A/B testing results reveal:
> - **Search/Feed List to PDP Click-Through Rate (CTR) rises significantly: $+12\%$ ($p < 0.01$)**;
> - **In-PDP Purchase Conversion Rate (CVR) remains flat: $\Delta \text{CVR} \approx 0\%$ ($p = 0.65$)**.

---

### 1. Root-Cause Attribution & Statistical Funnel Decomposition

```text
Funnel Decomposition & The Traffic Dilution Mechanism:
[Total List Impressions] (100,000)
       │
       ▼  CTR (Control: 5.0% ➔ Treatment: 5.6%, Relative +12%)
[PDP Pageviews] (Control: 5,000 ➔ Treatment: 5,600)
       │
       ▼  CVR (Control: 10.0% ➔ Treatment: 10.0%, Relative +0%)
[Total Orders Placed] (Control: 500 ➔ Treatment: 560, Net +60 Orders! +12% Lift!)
```

#### (1) Why "CTR Up, CVR Flat" is a Decisive Commercial Win
By unconditional funnel decomposition:

$$\text{CTCVR} = \frac{\text{Total Purchases}}{\text{Total Impressions}} = \text{CTR} \times \text{CVR}$$

- If $\text{CTR}$ increases by $+12\%$ while conditional $\text{CVR}$ remains flat ($10\% \to 10\%$), the **overall net purchase order volume generated per impression increased by $+12\%$**!

#### (2) Why Conditional CVR Did Not Increase (Traffic Dilution & Marginal Selection Bias)
In causal inference, this reflects **Selection Bias & Traffic Dilution**:
- The improved UI lowered friction, attracting not only the core high-intent buyers, but also **marginal, casual visitors who previously would not have clicked**;
- Marginal visitors naturally carry lower baseline purchase intent. **Maintaining a constant CVR despite an influx of lower-intent visitors demonstrates that the new PDP converted both core and marginal users more effectively**.

---

### 2. Statistical Diagnostic & Experimental Verification Protocol

```text
The 4-Step PDP Diagnostic Matrix:
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Diagnostic Dimension    │ Metric & Actionable Threshold                                          │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Dwell Time & Bounce  │ Run Kolmogorov-Smirnov test on PDP dwell time. If median dwell time    │
│ (Quality of Click)      │ collapses and bounce rate surges, indicates clickbait / misleading UI. │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Intermediate Funnel  │ Evaluate Add-to-Cart (ATC) rate vs Buy-Now. If ATC increases but final │
│ (ATC & Cart Checkout)   │ checkout drops, the bottleneck resides in payment/shipping steps.      │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. Post-Purchase Health │ Monitor 7-day return rates and dispute rates to prevent buyer regret.  │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. Novelty Cohort Decay │ Track Week 1 vs. Week 2 cohort slopes to ensure durability.            │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 3. User Tier Heterogeneity (HTE) & The Small-Sample Tier Challenge

When slicing treatment effects across user segments (New Users, Casual, Core VIP, Ultra-High-Net-Worth / Enterprise Accounts), the enterprise/VIP tier often has an extremely small sample ($N_{\text{treatment}} = 120, N_{\text{control}} = 115$).

- **The Statistical Dilemma**: Standard two-sample t-tests suffer from severe power deficiency ($\text{Power} < 20\%$), producing wide confidence intervals $[-18\%, +22\%]$ with $p = 0.45$.
- **Pitfall**: Inexperienced teams confuse "absence of evidence ($p > 0.05$)" with "evidence of absence".

---

### 4. 3 Rigorous Statistical Approaches for Small-Sample Tier Inference

#### Approach 1: Empirical Bayes & Hierarchical Shrinkage Modeling ★Recommended
Rather than analyzing the small tier in isolation, fit a **Hierarchical Bayesian Model** to borrow statistical strength across all tiers:

$$\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \cdot \mu_{\text{grand}} + (1 - B) \cdot \bar{Y}_{\text{small}}, \quad \text{where } B = \frac{\sigma_{\text{small}}^2}{\sigma_{\text{small}}^2 + \tau^2}$$

- For small, high-variance tiers ($\sigma_{\text{small}}^2 \gg \tau^2$), $B \to 1$, shrinking noisy estimates toward the reliable global grand mean $\mu_{\text{grand}}$ to prevent over-reaction to small-sample noise.

#### Approach 2: CUPED with Pre-Experiment Historical Spending Covariates
Leverage 30-day pre-experiment order volume and frequency as baseline covariate $X$. High-tier VIP users exhibit high behavioral consistency ($\rho \approx 0.85 \sim 0.90$):
- $\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y)(1 - 0.85^2) \approx 0.28 \cdot \text{Var}(Y)$;
- **Effectively expands effective sample size by $3.5\times$**, pulling previously under-powered signals into actionable statistical confidence.

#### Approach 3: Non-Parametric Exact Permutation Test & Bayesian Posterior Superiority
- Exact Fisher-Pitman permutation test evaluates empirical distribution without relying on Central Limit Theorem normality;
- Report **Bayesian Posterior Probability of Superiority**: $P(\theta_{\text{Treatment}} > \theta_{\text{Control}} \mid \text{Data}) > 0.90$ for clear risk-managed executive decisions.

---

## Module 5: High-Yield A/B Testing Interview Rapid-Fire

### Q1: Why is continuous peeking and early stopping strictly prohibited even if p-value reaches 0.001 on Day 3?
> **Answer**:
> 1. **Alpha Inflation via Multi-Testing**: Standard hypothesis tests assume fixed sample sizes. Repeatedly evaluating data introduces severe False Positive Rate inflation (jumping from $5\%$ up to $20\%\sim40\%$);
> 2. **Day-of-Week Seasonality**: User behavior on weekends differs fundamentally from weekdays. Evaluating uncompleted weekly cycles creates severe temporal selection bias;
> 3. **Unresolved Novelty Effects**: Early spikes frequently represent novelty curiosity that rapidly decays over time. If dynamic stopping is required, one must employ formal **Sequential Testing (e.g., mSPRT / Alpha-spending functions)**.

### Q2: What is the underlying mathematical principle of CUPED variance reduction?
> **Answer**:
> 1. **Core Mechanism**: Leverages pre-experiment user baseline covariates $X$ (which correlate with outcome $Y$ but are orthogonal to the treatment) to subtract user-level invariant variance;
> 2. **Adjusted Metric**:
>    $$\hat{Y}_{\text{CUPED}} = Y - \theta(X - \mathbb{E}[X]), \quad \text{where } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
> 3. **Variance Scaling Factor**:
>    $$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - \rho^2)$$
>    With correlation $\rho = 0.8$, metric variance drops by $64\%$ (retaining only $36\%$), reducing required sample size by nearly $3\times$ without allocating extra traffic!

### Q3: If a PDP UI redesign increases feed CTR by $+15\%$ but decreases within-PDP CVR by $-3\%$, should you roll it out?
> **Answer**:
> 1. **Net Funnel Impact**: $\text{New CTCVR} = 1.15 \times 0.97 = 1.1155 \implies \mathbf{+11.55\%}$ **net lift in total orders per impression**!
> 2. **Verify Dilution vs Quality**: Ensure dwell time and return rates are healthy;
> 3. **Rollout Decision**: If net revenue and guardrails are positive, **roll out to 100%**.
"""

content = content.replace(target_module4, new_module4_and_5)

with open('notes/BusinessAlgorithm/BusinessAlgorithm08 Industrial ML AB Testing Experimentation.en.md', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully updated BusinessAlgorithm08 English note")
