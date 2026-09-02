# -*- coding: utf-8 -*-

with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.en.md", "r", encoding="utf-8") as f:
    en = f.read()

module_6_7_8_en = r"""## Module 6: Industrial Online Experimentation & A/B Testing Lifecycle

```text
Industrial A/B Experimentation & Causal Inference Lifecycle:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Pre-Experiment Design                                               │
│ • Formulate causal hypotheses (H0: Null vs H1: Lift), define Guardrails│
│ • Sample size & statistical power estimation (Power ≥ 80%, α = 0.05)   │
│ • Layered Orthogonal Sharding + Salt Isolation for multi-layer routing │
├────────────────────────────────────────────────────────────────────────┤
│ 2. In-Flight Monitoring & Execution                                    │
│ • SRM (Sample Ratio Mismatch) Chi-Square Goodness-of-Fit (p < 0.001)   │
│ • Prevent Peeking Bias: mSPRT / Always-Valid p-value Sequential Testing│
│ • CUPED Variance Reduction (slashing Var by 50%~80% with pre-covariates│
├────────────────────────────────────────────────────────────────────────┤
│ 3. Decision & Release Standards                                        │
│ • Run full 7/14-day cycles (eliminating weekend/diurnal seasonality)   │
│ • Multiple testing correction (Bonferroni / Benjamini-Hochberg FDR)    │
│ • Launch Criteria: ① Confident Primary Lift ② Zero Guardrail Degradation│
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Why "Offline Metric Lifts, but Online A/B Test is Flat or Drops"? (6 Root Causes)

A notorious bottleneck in industry is observing an offline AUC lift of +0.01 while online revenue drops -2%. The causal disconnect stems from six core architectural mechanisms:

```text
The 6 Core Disconnects between Offline Training and Online A/B Testing:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Selection Bias & Feedback Loop                                      │
│ • Offline logs are conditioned on what the "legacy production policy"  │
│   exposed. Novel high-potential items are unexposed / marked negative. │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Training-Serving Skew & Feature Lag                                 │
│ • Offline features inadvertently leak future information (Time Leakage)│
│ • Online real-time feature streaming (Kafka/Flink) has millisecond lag │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Inference Latency & Fallback Degradation (P99 SLAs)                 │
│ • Model parameter bloat increases P99 latency from 15ms to 45ms        │
│ • Triggers 20ms timeout circuit breakers, forcing fallback to heuristics│
├────────────────────────────────────────────────────────────────────────┤
│ 4. Multi-Objective Mismatch & Clickbait Traps                          │
│ • Offline models optimizing pure CTR over-index on sensationalist titles│
│ • Users bounce quickly, destroying long-term CVR, Dwell Time, and D7/30│
├────────────────────────────────────────────────────────────────────────┤
│ 5. Position Bias & Attentional Confounding                             │
│ • Offline models without Inverse Propensity Scoring (IPS) confuse high │
│   slot exposure with intrinsic item quality.                           │
├────────────────────────────────────────────────────────────────────────┤
│ 6. Marketplace Spillover & Resource Cannibalization                    │
│ • Treatment policy hoards limited shared courier/driver capacity in a  │
│   region, artificially degrading Control performance (Violating SUTVA). │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Hypothesis Testing Foundations & Minimum Sample Size Estimation (Power Analysis)

Before launching an A/B test, required sample size per variant must be rigorously calculated based on **Type I Error ($\alpha$)**, **Statistical Power ($1 - \beta$)**, and **Minimum Detectable Effect (MDE)**.

#### Statistical Foundations
- **Significance Level ($\alpha$)**: Typically $\alpha = 0.05$ (two-tailed $z_{1 - \alpha/2} = 1.96$), bounding false positive risk at $5\%$;
- **Statistical Power ($1 - \beta$)**: Standard $1 - \beta \ge 80\%$ (corresponding to $\beta = 0.20$, $z_{1 - \beta} = 0.84$), ensuring an $80\%$ chance of detecting true effects;
- **Minimum Detectable Effect (MDE, $\Delta$)**: The smallest meaningful effect size $\Delta = |\mu_T - \mu_C|$.

#### Continuous Metrics (e.g. Revenue per User, Dwell Time per User)
For a 50/50 split, minimum sample size $n^*$ per variant:

$$n^* = \frac{2 \left( z_{1 - \alpha/2} + z_{1 - \beta} \right)^2 \cdot \sigma^2}{\text{MDE}^2} = \frac{2 \cdot (1.96 + 0.84)^2 \cdot \sigma^2}{\Delta^2} \approx \frac{15.68 \cdot \sigma^2}{\Delta^2}$$

#### Binomial Conversion Metrics (e.g. CTR, CVR, Sign-Up Rate)
For baseline conversion rate $p$ and absolute lift $\Delta$:

$$n^* = \frac{\left( z_{1 - \alpha/2} \sqrt{2 p (1 - p)} + z_{1 - \beta} \sqrt{p(1 - p) + (p + \Delta)(1 - (p + \Delta))} \right)^2}{\Delta^2}$$

---

### 3. CUPED Variance Reduction Mathematical Derivation & Production Protocols

#### Motivation & Principles
Metrics like Revenue per User exhibit heavy-tailed, highly skewed distributions. Standard $t$-tests require massive sample sizes. **CUPED (Controlled-experiment Using Pre-Experiment Data, Microsoft 2013)** leverages pre-experiment user history to slash variance without increasing traffic.

#### Formal Mathematical Derivation
Let $Y$ be the observed metric during the experiment, and let $X$ be the user's identical metric measured in a pre-experiment window.
Define the CUPED estimator:

$$\hat{Y}_{\text{CUPED}} = Y - \theta (X - \mathbb{E}[X])$$

##### 1. Proof of Unbiasedness:
Because $X$ was generated before the experiment started, the treatment cannot retroactively alter pre-experiment data ($\mathbb{E}[X_{\text{Treatment}}] = \mathbb{E}[X_{\text{Control}}] = \mathbb{E}[X]$):
$$\mathbb{E}[\hat{Y}_{\text{CUPED}}] = \mathbb{E}[Y] - \theta (\mathbb{E}[X] - \mathbb{E}[X]) = \mathbb{E}[Y]$$

##### 2. Optimal Parameter $\theta^*$ for Variance Minimization:
$$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) + \theta^2 \text{Var}(X) - 2\theta \text{Cov}(Y, X)$$
Taking the derivative with respect to $\theta$ and setting to 0:
$$\frac{d}{d\theta} \text{Var}(\hat{Y}_{\text{CUPED}}) = 2\theta \text{Var}(X) - 2\text{Cov}(Y, X) = 0 \implies \theta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

##### 3. Variance Reduction Ratio:
Substituting $\theta^*$ back with Pearson correlation $\rho = \frac{\text{Cov}(Y, X)}{\sqrt{\text{Var}(Y)\text{Var}(X)}}$:

$$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - \rho^2)$$

```text
CUPED Variance Reduction Scaling Table:
┌────────────────────────┬────────────────────────┬────────────────────────┐
│ Pre/Post Correlation ρ │ Variance Ratio (1 - ρ²)│ Sample Size Needed     │
├────────────────────────┼────────────────────────┼────────────────────────┤
│ ρ = 0.0 (No signal)    │ 100% (No reduction)    │ 1.00x (Baseline)       │
│ ρ = 0.5 (Moderate)     │ 75% (25% reduction)    │ 0.75x (25% duration)   │
│ ρ = 0.8 (User spending)│ 36% (64% reduction!)   │ 0.36x (Nearly 3x fast!)│
│ ρ = 0.9 (Daily DAU/Act)│ 19% (81% reduction!)   │ 0.19x (Over 5x fast!)  │
└────────────────────────┴────────────────────────┴────────────────────────┘
```

#### Production Covariate Selection Rules
1. **Strictly Pre-Period**: $X$ must be sourced from 7~14 days prior to experiment launch;
2. **Cold-Start Handling**: Newly registered users in the experiment period receive $X = 0$, preserving unbiasedness;
3. **Multi-Covariate CUPED**: Multiple linear regression $\theta = (X^T X)^{-1} X^T Y$ over pre-period clicks, revenue, and sessions.

---

### 4. Sample Ratio Mismatch (SRM) Detection & Troubleshooting SOP

#### Why SRM Invalidates All Statistical Conclusions
If an experiment is configured for a $50\% : 50\%$ split, but after 7 days records:
- Control: $N_C = 1,000,000$
- Treatment: $N_T = 980,000$ ($50.5\% : 49.5\%$)

**SRM indicates that randomization has failed!** The user cohorts in Control and Treatment possess fundamentally biased demographic and behavioral attributes. Any observed lift is confounded, and the experiment must be **100% aborted**.

#### Chi-Square Goodness-of-Fit Test
$$\chi^2 = \frac{(O_C - E_C)^2}{E_C} + \frac{(O_T - E_T)^2}{E_T} \sim \chi^2(df = 1)$$
- **P0 Alert Threshold**: If $\chi^2 > 10.83$ ($p\text{-value} < 0.001$), immediately trigger an SRM P0 incident.

```text
SRM Root Cause Checklist:
1. Hash Bucketing Bias: Salt collisions or modulo arithmetic flaws;
2. Client Crash Imbalance: Treatment code triggers null-pointer crashes on Android;
3. Post-Assignment Trigger Leaks: Filtering data on "saw new UI element";
4. Redirect Latency Drops: Extra 300ms redirect in Treatment causing user bounce;
5. Asymmetric Bot Filtering: Fraud systems aggressively cleaning Treatment bots.
```

---

### 5. Layered Orthogonal Sharding Architecture

To run hundreds of concurrent experiments without interference, modern tech platforms (Google, Meta) deploy **Overlapping Layered Orthogonal Architectures**:

```text
Layered Sharding Hierarchy (1000 Buckets via Hash(UID + Layer_Salt) % 1000):
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 1: UI / Layout (Salt = "UI_Layer_2026")                          │
│ [ Buckets 000~499: Control UI ]    [ Buckets 500~999: Experimental UI ]│
├────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Retrieval & Candidate Gen (Salt = "Retrieval_2026")           │
│ [ 0~249: Vector ANN ] [ 250~499: Graph ] [ 500~749: Two-Tower ] [ Base]│
├────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Precision Ranking MTL (Salt = "Ranking_MTL_2026")             │
│ [ Buckets 000~499: PLE Model ]     [ Buckets 500~999: MMoE Baseline ]  │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Generative Reranking (Salt = "Rerank_2026")                   │
│ [ 0~333: Beam Search ] [ 334~666: Plackett-Luce ] [ 667~999: Baseline]│
└────────────────────────────────────────────────────────────────────────┘
```
- Because salts are independent, a user in Bucket 120 of Layer 1 is uniformly and orthogonally distributed across all buckets of Layer 3, guaranteeing zero cross-layer confounding.

---

### 6. Multiple Testing Corrections & False Discovery Rate (FDR)

#### Family-Wise Error Rate (FWER) Explosion
When monitoring 20 metrics simultaneously at $\alpha = 0.05$:
$$\text{FWER} = 1 - (1 - 0.05)^{20} = 64.15\%$$
Over $64\%$ of null experiments will show at least one "statistically significant" false positive!

#### Benjamini-Hochberg (BH) Procedure for FDR Control ($Q = 0.05$):
1. Compute and sort $m$ $p$-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$;
2. Find largest index $k$ satisfying: $p_{(k)} \le \frac{k}{m} Q$;
3. Reject null hypotheses for all tests $1 \dots k$.

---

### 7. Peeking Bias & Sequential Testing (mSPRT)

- **The Peeking Trap**: Checking $p$-values daily and stopping the test as soon as $p < 0.05$ inflates true false positive rates to $30\% \sim 50\%$;
- **Solution**: Use **mSPRT (Mixture Sequential Probability Ratio Test)** to generate **Always-Valid p-values**, enabling continuous real-time monitoring and early stopping while strictly bounding Type I error at $\alpha = 0.05$.

---

### 8. Release Decision Matrix & Guardrail Policies

```text
Launch Criteria Checklist:
1. Duration: Must run full 7 or 14 days (neutralizing weekend seasonality);
2. Integrity: Chi-square SRM test p ≥ 0.001 (zero sample ratio imbalance);
3. Primary Metric: Statistically significant positive lift (p < 0.05);
4. Guardrails: No statistically significant degradation in D7 Retention, Unsubscribe Rate, or P99 Latency (≤ 20ms);
5. Infrastructure ROI: Incremental revenue must exceed GPU/inference hosting costs.
```

---

## Module 7: Flagship Case Studies & Small-Sample Causal Inference

### Case 1: Sign-Up Funnel $2 \times 2$ Full Factorial Experiment

```text
2x2 Full Factorial Matrix:
                      Button Color (Factor A)
                 Red (A0 = 0)        Blue (A1 = 1)
              ┌──────────────────┬──────────────────┐
  Top (B0 = 0)│   Group 0 (T0)   │   Group 1 (T1)   │
Button Position│    Red + Top     │    Blue + Top    │
(Factor B)    ├──────────────────┼──────────────────┤
Bottom (B1 = 1│   Group 2 (T2)   │   Group 3 (T3)   │
              │   Red + Bottom   │   Blue + Bottom  │
              └──────────────────┴──────────────────┘
```

- **Linear Model with Interaction**:
  $$Y_i = \beta_0 + \beta_1 \text{Color}_i + \beta_2 \text{Pos}_i + \beta_3 (\text{Color}_i \times \text{Pos}_i) + \epsilon_i$$
  - $\beta_3 = \bar{Y}_{T3} - \bar{Y}_{T2} - \bar{Y}_{T1} + \bar{Y}_{T0}$ measures **Synergy / Super-Additivity**;
- **Triggered Exposure**: Use `IntersectionObserver` to record exposure only when the bottom button enters the user's viewport ($\ge 50\%$ visible for $\ge 1\text{s}$), eliminating inactive-user dilution.

---

### Case 2: PDP UI Redesign (CTR Up, CVR Flat) & Small-Sample VIP Inference

```text
E-commerce Funnel Decomposition:
【Total Impressions】 (N = 100,000)
       │
       ▼  CTR (Control: 5.0% ➔ Treatment: 5.6%, +12% Lift)
【PDP Product Pageviews】 (Control: 5,000 ➔ Treatment: 5,600)
       │
       ▼  CVR (Control: 10.0% ➔ Treatment: 10.0%, Flat)
【Total Completed Purchases】 (Control: 500 ➔ Treatment: 560, +12% Net Orders!)
```

1. **Net Funnel Lift**: $\text{CTCVR} = \text{CTR} \times \text{CVR} = 1.12 \times 1.0 = +12\%$ **net order volume growth**;
2. **Traffic Dilution Effect**: Lower click friction brings in marginal lower-intent users. Preserving baseline conversion rate on a diluted denominator proves strong landing page efficacy;
3. **Small-Sample VIP Tier Inference ($N \approx 100$, Power $< 20\%$)**:
   - **Empirical Bayes Partial Pooling / Shrinkage**:
     $$\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \cdot \mu_{\text{grand}} + (1 - B) \cdot \bar{Y}_{\text{small}}, \quad B = \frac{\sigma_{\text{small}}^2 / N}{\sigma_{\text{small}}^2 / N + \tau^2}$$
   - **CUPED with 30-Day Pre-Period Spending**: Slashes variance by $72.2\%$, expanding effective sample size by $3.6\times$;
   - **Exact Permutation Tests**: 100,000 re-shuffles for exact distribution-free $p$-values.

---

### Case 3: Marketplace Interference & Team Draft Interleaving

1. **Marketplace Spillover (SUTVA Violation)**: In on-demand delivery (DoorDash) or ridesharing (Uber), Treatment orders capture shared courier supply, degrading Control fulfillment;
   - **Solution**: **Switchback Experiments** (alternating entire cities between Control/Treatment in 2-hour windows) or **Spatial Cluster Randomization** (H3 hexagons).
2. **Team Draft Interleaving (Search & Ranking)**: Alternate item selections between Algorithm A and Algorithm B on a single search query. Single users serve as their own control, boosting statistical sensitivity by $100\times$!

---

## Module 8: High-Yield Industrial ML Multiple-Choice Quizzes

<details class="exercise">
<summary><span class="q-label">Q1 · Negative Downsampling & Probability Recovery</span> <span class="q-text">A feed ranking model operates with natural CTR $p = 1\%$. Negative samples are uniformly downsampled with retention rate $w = 10\%$ ($w=0.1$). If the model outputs $\hat{p} = 0.0917$ ($9.17\%$), how should downstream auction bidding restore natural probability $p$?</span></summary>

- [ ] **A.** Approximate via linear scaling: $\hat{p} \times w = 0.00917$.
- [ ] **B.** Rescale via $p = \hat{p} / w = 0.917$.
- [x] **C.** Apply exact closed-form probability recovery: $p = \frac{\hat{p}}{\hat{p} + \frac{1-\hat{p}}{w}}$, producing $p = 0.01$ ($1.0\%$).
- [ ] **D.** Negative downsampling only affects LogLoss, so no recovery is required for predicted probabilities.

> 💡 **Explanation**:
> - **Correct Answer: C**.
>   1. **Prior Distortion**: Under negative downsampling rate $w$, model predictions are distorted to $\hat{p} = \frac{p}{p + w(1-p)}$.
>   2. **Exact Inverse**: Solving for $p$ yields $p = \frac{\hat{p}}{\hat{p} + \frac{1-\hat{p}}{w}}$. Substituting $\hat{p}=0.0917$ and $w=0.1$ exactly recovers natural probability $p = 0.01$ ($1.0\%$).
</details>

<details class="exercise">
<summary><span class="q-label">Q2 · Multi-Objective Learning & Seesaw Mitigation</span> <span class="q-text">In a multi-objective ranking model (joint CTR & CVR prediction), which architecture and optimization protocol <strong>most effectively isolates task-specific representations from negative gradient conflicts ($\cos(\mathbf{g}_i, \mathbf{g}_j) < 0$)</strong>?</span></summary>

- [ ] **A.** Shared-Bottom architecture: all shared representations average gradients across tasks.
- [ ] **B.** Static loss weight grid search (e.g. $\mathcal{L} = 0.8\mathcal{L}_{\text{CTR}} + 0.2\mathcal{L}_{\text{CVR}}$).
- [x] **C.** PLE (Progressive Layered Extraction) with PCGrad orthogonal projection: physical decoupling of task-specific and shared experts with orthogonal gradient projection.
- [ ] **D.** Independent single-task training combined with offline probability multiplication ($p\text{CTCVR} = p\text{CTR} \times p\text{CVR}$).

> 💡 **Explanation**:
> - **Correct Answer: C**.
>   1. **Shared-Bottom Flaw**: Shared parameter layers suffer direct destructive interference from opposing gradient vectors.
>   2. **PLE Advantage**: PLE enforces structural physical isolation between task-specific and shared expert modules.
>   3. **PCGrad**: Projects conflicting gradient components onto the normal plane, eliminating gradient destruction.
</details>

<details class="exercise">
<summary><span class="q-label">Q3 · Grouped AUC (GAUC) Invariance</span> <span class="q-text">Why is <strong>Request-Grouped GAUC</strong> favored over Global ROC-AUC as the primary offline ranking benchmark in precision rankers?</span></summary>

- [ ] **A.** Request-GAUC is mathematically larger, providing inflated metrics for management presentations.
- [ ] **B.** Global AUC is sensitive to negative sampling, whereas GAUC is not.
- [x] **C.** Global AUC conflates cross-user activity baselines (Simpson's paradox); Request-GAUC measures within-slate discrimination on a single refresh, directly matching real user decisions.
- [ ] **D.** Request-GAUC can substitute for PCOC and ECE in evaluating absolute probability calibration.

> 💡 **Explanation**:
> - **Correct Answer: C**.
>   1. **Global AUC Bias**: A model that simply assigns higher scores to active users achieves high Global AUC while failing at within-session ranking.
>   2. **Request-GAUC Alignment**: Evaluates pairwise rankings strictly within the 6~10 items presented on a single screen refresh, eliminating diurnal drift.
</details>

<details class="exercise">
<summary><span class="q-label">Q4 · A/B Testing Attribution & Small-Sample Inference</span> <span class="q-text">An e-commerce PDP UI test increases list CTR by $+12\%$ ($p < 0.01$) while PDP CVR is flat ($\Delta \text{CVR} \approx 0\%$, $p = 0.65$). For an enterprise VIP tier with $N \approx 100$, what is the <strong>most rigorous statistical approach</strong>?</span></summary>

- [ ] **A.** Since $p > 0.05$ in the VIP slice, conclude the UI harms high-value users and abort the rollout.
- [ ] **B.** Conclude the experiment failed because within-PDP conversion rate did not increase.
- [x] **C.** Total orders per impression $\text{CTCVR} = \text{CTR} \times \text{CVR}$ increased by $+12\%$; flat CVR is driven by traffic dilution from marginal users; use <strong>Empirical Bayes Partial Pooling / Shrinkage</strong> and CUPED baseline covariates for VIP tier inference.
- [ ] **D.** Extend the test for 6 months until sample size in the VIP tier reaches hundreds of thousands.

> 💡 **Explanation**:
> - **Correct Answer: C**.
>   1. **Net Funnel Lift**: $\text{CTCVR} = 1.12 \times 1.0 = 1.12$ ($+12\%$ net order volume per impression).
>   2. **Traffic Dilution**: Lower click friction brings in marginal lower-intent visitors. Maintaining flat conversion confirms strong page performance.
>   3. **Small-Sample Inference**: $p > 0.05$ reflects low power ($\text{Power} < 20\%$), not evidence of absence. Empirical Bayes shrinkage borrows statistical strength across tiers.
</details>

<details class="exercise">
<summary><span class="q-label">Q5 · SRM Imbalance & Diagnostics</span> <span class="q-text">An experiment configured for a 50/50 split yields $N_C = 1,000,000$ and $N_T = 980,000$ ($p < 10^{-40}$). What is the <strong>only statistically valid response</strong>?</span></summary>

- [ ] **A.** Proceed with analysis since per-user metrics normalize by cohort size.
- [ ] **B.** Randomly drop 20,000 users from Control to balance sample sizes.
- [x] **C.** Trigger an SRM P0 incident and declare results invalid; investigate client crashes, redirect drop-offs, and post-assignment trigger leaks.
- [ ] **D.** Extend the test for 14 more days hoping sample sizes naturally equalize.

> 💡 **Explanation**:
> - **Correct Answer: C**. SRM proves randomization failed and user demographics are fundamentally confounded.
</details>

<details class="exercise">
<summary><span class="q-label">Q6 · CUPED Covariate Selection Protocol</span> <span class="q-text">When applying CUPED to variance-reduce user spending $Y$, which covariate strategy is <strong>strictly causally valid</strong>?</span></summary>

- [ ] **A.** Use user click counts during the first 3 days of the experiment period.
- [ ] **B.** Use any post-launch metric with correlation $\rho > 0.9$.
- [x] **C.** Use total user spending during the 14 days prior to experiment launch; impute 0 for newly registered users.
- [ ] **D.** Compute separate historical expectations $\mathbb{E}[X_T]$ and $\mathbb{E}[X_C]$ for each group.

> 💡 **Explanation**:
> - **Correct Answer: C**. Covariates must strictly predate the experiment launch.
</details>

<details class="exercise">
<summary><span class="q-label">Q7 · Lifelong Behavior Sequence Architecture</span> <span class="q-text">A platform has user histories of length $L = 50,000+$. Under a strict $P99 \le 12\text{ms}$ latency SLA, which architecture is <strong>most feasible for production deployment</strong>?</span></summary>

- [ ] **A.** 6-layer SASRec Transformer performing Full Self-Attention over all 50,000 items.
- [ ] **B.** Full DIN Target Attention across 50,000 items.
- [x] **C.** SIM Two-Stage Architecture: Hard Search filtering down to Top-50 candidates, followed by fine-grained Target Attention.
- [ ] **D.** Global Mean Pooling over all 50,000 embeddings.

> 💡 **Explanation**:
> - **Correct Answer: C**. SIM reduces network payload by 99.9% while activating lifelong relevant interest sub-sequences within 8ms.
</details>

<details class="exercise">
<summary><span class="q-label">Q8 · Generative Reranking Constrained Decoding</span> <span class="q-text">In generative reranking, what is the standard method to <strong>strictly prevent duplicate items and hallucinations</strong> during autoregressive decoding?</span></summary>

- [ ] **A.** Rerun beam search if duplicates are detected after generation.
- [ ] **B.** Use Greedy Search instead of Beam Search.
- [x] **C.** Dynamic Masked Softmax: set logits of already-selected items and illegal token IDs to $-\infty$ before computing softmax at each step.
- [ ] **D.** Apply L2 penalty in training loss for duplicate predictions.

> 💡 **Explanation**:
> - **Correct Answer: C**. Dynamic masking provides a 100% deterministic mathematical guarantee against duplicate generation.
</details>

<details class="exercise">
<summary><span class="q-label">Q9 · Peeking Bias & Sequential Testing</span> <span class="q-text">Checking A/B test $p$-values daily and launching as soon as $p < 0.05$ on Day 3 results in:</span></summary>

- [ ] **A.** Increased statistical power without inflating Type I error.
- [ ] **B.** Larger variance but valid significance.
- [x] **C.** Severe Peeking Bias: true false positive rate inflates from $5\%$ to $30\% \sim 50\%$.
- [ ] **D.** Complete elimination of novelty bias.

> 💡 **Explanation**:
> - **Correct Answer: C**. Peeking turns a single test into multiple sequential draws. Use mSPRT for continuous monitoring.
</details>

<details class="exercise">
<summary><span class="q-label">Q10 · Marketplace Interference & Switchbacks</span> <span class="q-text">In on-demand delivery platforms, why is 50/50 user-randomized A/B testing invalid for courier dispatching algorithms?</span></summary>

- [ ] **A.** User IDs cannot be hashed deterministically.
- [ ] **B.** Sample sizes cannot be balanced.
- [x] **C.** SUTVA Violation: Treatment orders cannibalize shared courier capacity, degrading Control delivery times and creating artificial lifts.
- [ ] **D.** User ID splits cause feature leakage.

> 💡 **Explanation**:
> - **Correct Answer: C**. Physical capacity constraints create spillover interference. Use Switchback time-slice randomization.
</details>

<details class="exercise">
<summary><span class="q-label">Q11 · Multiple Testing & FDR Control</span> <span class="q-text">When tracking 50 metrics simultaneously without correction, the theoretical probability of observing at least one false positive ($p < 0.05$) under the null hypothesis is:</span></summary>

- [ ] **A.** $5.0\%$
- [ ] **B.** $25.0\%$
- [ ] **C.** $50.0\%$
- [x] **D.** Over $92\%$ ($1 - 0.95^{50} \approx 92.3\%$)

> 💡 **Explanation**:
> - **Correct Answer: D**. $\text{FWER} = 1 - 0.95^{50} = 92.3\%$. Use Benjamini-Hochberg (BH) procedure to control FDR.
</details>

<details class="exercise">
<summary><span class="q-label">Q12 · Probability Calibration (PCOC / ECE) in Bidding Auctions</span> <span class="q-text">A new model achieves $+0.5\%$ GAUC lift, but advertiser ROI collapses and budgets are exhausted prematurely in production bidding ($eCPM = pCTR \times \text{Bid}$). What is the primary cause?</span></summary>

- [ ] **A.** GAUC lift is statistical noise.
- [ ] **B.** Feature interaction layers were omitted.
- [x] **C.** Discrimination (GAUC) improved, but **probability calibration failed (PCOC ≫ 1.0)**, systematically over-predicting CTR and causing auction bid inflation.
- [ ] **D.** Time-delta decay was unconfigured.

> 💡 **Explanation**:
> - **Correct Answer: C**. Real-time auctions depend on absolute probability scale. Severe PCOC overestimation multiplies bids, destroying advertiser budget efficiency.
</details>
"""

pos_m6_en = en.find("## Module 6: Industrial Online Experimentation & A/B Testing Lifecycle")
if pos_m6_en != -1:
    en = en[:pos_m6_en] + module_6_7_8_en
    with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.en.md", "w", encoding="utf-8") as f:
        f.write(en)
    print("Successfully expanded English Modules 6, 7, and 8!")

