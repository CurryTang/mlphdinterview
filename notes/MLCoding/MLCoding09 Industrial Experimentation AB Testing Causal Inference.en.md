# ML Coding 09 · Industrial Online Experimentation & A/B Testing: Causal Inference, Funnel Attribution & Small-Sample Tier Inference

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
$$\hat{Y}_{	ext{CUPED}} = Y - 	heta(X - \mathbb{E}[X]), \quad 	ext{where } 	heta = rac{	ext{Cov}(Y, X)}{	ext{Var}(X)}$$
$$	ext{Var}(\hat{Y}_{	ext{CUPED}}) = 	ext{Var}(Y) \cdot (1 - ho^2)$$

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
  $$\hat{eta}_3 = (T_3 - T_1) - (T_2 - T_0) = T_3 - T_2 - T_1 + T_0$$
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

1. **Net Funnel Lift**: $	ext{CTCVR} = 	ext{CTR} 	imes 	ext{CVR} \implies +12\%$ **net lift in total orders per impression**!
2. **Traffic Dilution**: An influx of lower-intent marginal users dilutes the denominator. Maintaining a flat CVR confirms robust landing page conversion power.

---

## Module 4: Statistical Inference for Small-Sample High-Value User Tiers

For small VIP tiers ($N pprox 100$), standard t-tests suffer from power deficiency ($	ext{Power} < 20\%$).

1. **Empirical Bayes Partial Pooling / Shrinkage**:
   $$\hat{	heta}_{	ext{small}}^{	ext{shrunk}} = B \cdot \mu_{	ext{grand}} + (1 - B) \cdot ar{Y}_{	ext{small}}, \quad 	ext{where } B = rac{\sigma_{	ext{small}}^2}{\sigma_{	ext{small}}^2 + 	au^2}$$
2. **CUPED with 30-Day Historical Baseline Spending**: Compresses variance to $28\%$ ($ho pprox 0.85$), expanding effective sample size by $3.5	imes$;
3. **Non-Parametric Exact Permutation Tests & Bayesian Posterior Superiority**: $P(	heta_T > 	heta_C \mid \mathcal{D}) > 0.90$.
