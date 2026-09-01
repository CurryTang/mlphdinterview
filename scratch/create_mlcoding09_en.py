import os

en_content = """# ML Coding 09 · Industrial Online Experimentation & A/B Testing: Causal Inference, Funnel Attribution & Small-Sample Tier Inference

In industrial machine learning and Internet algorithm systems, offline evaluation metrics are merely admission gates, whereas **Online Controlled Experiments (A/B Testing / Randomized Controlled Trials, RCT) serve as the ultimate gold standard for measuring causal increment and business uplift**.

This note systematically establishes the 4 foundational pillars of industrial online experimentation:
1. **End-to-End A/B Testing Lifecycle (Design, Layered Routing, CUPED Variance Reduction, SRM Checks & Decision Rules)**
2. **Flagship Case Study 1: Sign-Up Funnel Button Color & Position 2x2 Full Factorial Experiment Design**
3. **Flagship Case Study 2: E-Commerce Product Detail Page (PDP) Conversion Attribution (CTR Up, CVR Flat) & Traffic Dilution Analysis**
4. **Small-Sample High-Value Tier Causal Inference (Empirical Bayes Partial Pooling / Shrinkage, CUPED & Exact Permutation Tests)**

---

## Module 1: Industrial A/B Testing Lifecycle & CUPED Variance Reduction

```text
End-to-End A/B Testing Lifecycle:
1. Design: Hypotheses (H0, H1), Sample size estimation (Power ≥ 80%), Orthogonal layers
2. Execution: Chi-square SRM goodness-of-fit check, CUPED variance reduction (Var·(1-ρ²)), Canary ramp
3. Rollout Decision: 14-day full cycle execution, No peeking, Guardrail metric verification
```

### Mathematical Principle of CUPED
$$\hat{Y}_{\text{CUPED}} = Y - \theta(X - \mathbb{E}[X]), \quad \text{where } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
$$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - \rho^2)$$

---

## Module 2: Flagship Case Study 1: Sign-Up Funnel 2x2 Full Factorial Experiment

```text
2x2 Full Factorial Design Matrix:
                       Button Color
                 Red (T0/T2)        Blue (T1/T3)
              ┌─────────────────┬─────────────────┐
  Top (T0/T1) │  Control (T0)   │  Treatment (T1) │
              │   Red + Top     │   Blue + Top    │
Button        ├─────────────────┼─────────────────┤
Position      │  Treatment (T2) │  Treatment (T3) │
  Bottom(T2/T3│  Red + Bottom   │   Blue + Bottom │
              └─────────────────┴─────────────────┘
```

- **Interaction Effect Estimator**:
  $$\hat{\beta}_3 = (T_3 - T_1) - (T_2 - T_0) = T_3 - T_2 - T_1 + T_0$$
- **Triggered Exposure Definition (Anti-Dilution Guardrail)**: Bottom buttons record exposure only when entering the visible viewport via an Intersection Observer.

---

## Module 3: Flagship Case Study 2: PDP UI Redesign (CTR Up, CVR Flat)

```text
Funnel Decomposition & Traffic Dilution:
[Total List Impressions] (100,000)
       │
       ▼  CTR (Control: 5.0% ➔ Treatment: 5.6%, Relative +12%)
[PDP Pageviews] (Control: 5,000 ➔ Treatment: 5,600)
       │
       ▼  CVR (Control: 10.0% ➔ Treatment: 10.0%, Relative +0%)
[Total Purchases Placed] (Control: 500 ➔ Treatment: 560, Net +60 Orders! +12% Lift!)
```

1. **Net Funnel Lift**: $\text{CTCVR} = \text{CTR} \times \text{CVR} \implies +12\%$ **net lift in total orders per impression**!
2. **Traffic Dilution**: An influx of lower-intent marginal users dilutes the denominator. Maintaining a flat CVR confirms robust landing page conversion power.

---

## Module 4: Statistical Inference for Small-Sample High-Value User Tiers

For small VIP tiers ($N \approx 100$), standard t-tests suffer from power deficiency ($\text{Power} < 20\%$).

1. **Empirical Bayes Partial Pooling / Shrinkage**:
   $$\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \cdot \mu_{\text{grand}} + (1 - B) \cdot \bar{Y}_{\text{small}}, \quad \text{where } B = \frac{\sigma_{\text{small}}^2}{\sigma_{\text{small}}^2 + \tau^2}$$
2. **CUPED with 30-Day Historical Baseline Spending**: Compresses variance to $28\%$ ($\rho \approx 0.85$), expanding effective sample size by $3.5\times$;
3. **Non-Parametric Exact Permutation Tests & Bayesian Posterior Superiority**: $P(\theta_T > \theta_C \mid \mathcal{D}) > 0.90$.
"""

with open("notes/MLCoding/MLCoding09 Industrial Experimentation AB Testing Causal Inference.en.md", "w", encoding="utf-8") as f:
    f.write(en_content)
print("Successfully created MLCoding09 English note")
