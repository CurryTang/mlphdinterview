# ML Coding 07 · Industrial Recommendation Ranking: Base Model Family, Sampling, GAUC & Multi-Objective Seesaw Mitigation

In industrial recommender systems and computational advertising, building a high-throughput, multi-objective, and well-calibrated ranking stack is a core engineering and algorithmic foundation. In senior ML system interviews and architecture reviews, **how you structure the presentation of your base model family, training sample scale, negative sampling scheme, metric hierarchy (ROC-AUC vs. GAUC by User/Request, Calibration Plots, Slice Metrics), and multi-objective negative transfer mitigation (PLE, PCGrad, Constrained Pareto Fusion)** directly distinguishes senior practitioners.

This note systematically constructs the 5 foundational pillars of industrial recommendation ranking:
1. **The Base Model Family (Feature Interaction, Long-Term User Behavior Sequences & Multi-Task Learning)**
2. **Training Sample Construction, Negative Sampling Strategies & Mathematical Probability Recovery**
3. **The Multi-Tier Metric Hierarchy (ROC-AUC, GAUC User vs. Request Grouping, Unbiased Estimation with Downsampling & Calibration PCOC/ECE)**
4. **Multi-Objective Representation Conflicts & The Seesaw Effect (Shared-Bottom, MMoE, PLE, Uncertainty Weighting, PCGrad & Constrained Pareto Fusion)**
5. **Production Training Dashboard Instrumentation & Anti-Confusion Protocol**

---

## Module 1: The Ranking Base Model Family

Industrial recommendation rankers handle **billions of sparse categorical ID features** alongside **dense continuous numerical features**, evolving along three major architectural axes:

```text
The 3-Dimensional Evolution of Industrial Ranking Backbones:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Explicit & Implicit Feature Interactions                             │
│ • Linear/FM ➔ Wide&Deep ➔ DeepFM ➔ DCN-v2 (Low-Rank Cross Networks)   │
│ • Goal: Capture bounded high-order non-linear feature interactions     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Long & Short-Term User Behavior Sequence Modeling                   │
│ • Pooling (Sum/Mean) ➔ DIN (Target Attention) ➔                        │
│ • SIM (Two-Stage Search-based Sequence: Hard Search + Soft Attention)   │
│ • Goal: Dynamically extract candidate-relevant interests from 10k items│
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Multi-Task Learning & Gradient Routing (MTL)                        │
│ • Shared-Bottom ➔ MMoE (Multi-gate MoE) ➔ PLE (Progressive Extraction) │
│ • Goal: Resolve gradient conflicts and negative transfer across tasks  │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. The 3 Architectural Pillars
1. **Feature Interaction Backbone**: **DCN-v2** (explicit low-rank cross networks) & **DLRM** (decoupled sparse embeddings + bottom MLP dot-products);
2. **User Sequence Modeling Backbone**: **DIN** (candidate target-attention) & **SIM** (two-stage sub-sequence hard search + soft attention);
3. **Multi-Task Learning Backbone**: **MMoE** (dynamic gating over shared experts) & **PLE** (task-specific vs shared expert isolation).

---

## Module 2: Training Sample Scale, Negative Sampling & Probability Recovery

```text
Fundamental Differences: Retrieval vs. Ranking Datasets:
┌───────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Stage                 │ Candidate Universe & Negative Sampling Scheme                          │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Retrieval (Candidate) │ • Candidate Universe: Entire corpus of $10^7 \sim 10^9$ items.         │
│                       │ • Negatives: Uniform corpus random sampling + non-clicked impressions  │
│                       │   + hard negatives (coarse-rank drops) + in-batch cross negatives.     │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Ranking (Scoring)     │ • Candidate Universe: Filtered top $1,000 \sim 3,000$ candidates.      │
│                       │ • Positives: Observed user engagements (clicks, purchases, likes).     │
│                       │ • Negatives: Strictly real **Impression-Unclicked** events.            │
│                       │ • Strict Rule: Never inject random unexposed negatives into rankers!   │
└───────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Probability Recovery Formula Under Negative Downsampling ($w$)
$$\hat{p} = rac{p}{p + w(1-p)} \implies p = rac{\hat{p}}{\hat{p} + rac{1 - \hat{p}}{w}}$$

---

## Module 3: ROC-AUC, GAUC (User vs. Request Grouping) & Deep Metric Breakdown

### 1. Wilcoxon-Mann-Whitney Definition of Global ROC-AUC
$$	ext{AUC} = rac{1}{|\mathcal{D}^+| \cdot |\mathcal{D}^-|} \sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \left( \mathbb{I}(s_i > s_j) + rac{1}{2} \mathbb{I}(s_i = s_j) ight)$$

### 2. Grouped AUC (GAUC) Formulation
$$	ext{GAUC} = rac{\sum_{g \in \mathcal{G}, \, n_g^+ > 0, \, n_g^- > 0} w_g \cdot 	ext{AUC}_g}{\sum_{g \in \mathcal{G}, \, n_g^+ > 0, \, n_g^- > 0} w_g}$$

- **User-Grouped GAUC ($	ext{GAUC}_{	ext{user}}$)**: Evaluates user long-term profile alignment across visits, but is subject to diurnal intent drift;
- **Request-Grouped GAUC ($	ext{GAUC}_{	ext{request}}$)**: Evaluates within-slate item ranking on a single refresh, **the gold standard for precision ranking**.

### 3. Metric Invariance & Importance Weighting
- **AUC / GAUC**: Asymptotically unbiased under uniform random negative downsampling;
- **LogLoss**: Requires importance weighting ($rac{1}{w}$ for negative samples) for unbiased evaluation.

### 4. Ranking vs. Calibration
- **Ranking (Discrimination)**: Invariant to strictly monotonic transformations;
- **Calibration (Reliability)**: Evaluated via **PCOC** ($rac{\sum \hat{p}_i}{\sum y_i} = 1.0$), **ECE**, and **Brier Score**.

---

## Module 4: Multi-Objective Ranking & Seesaw Effect Mitigation

```text
The 3 Root Causes of Representation Conflicts:
1. Directional Gradient Conflicts: cos(g_A, g_B) < 0
2. Magnitude & Frequency Domination: High-frequency CTR dominates sparse CVR
3. Sample Space Mismatch: D_imp vs D_click
```

### Architectural & Optimization Solutions
- **PLE (Progressive Layered Extraction)**: Physical isolation of task-specific and shared experts;
- **Uncertainty Weighting**: Dynamic loss weighting $\mathcal{L} = \sum rac{1}{2\sigma_k^2}\mathcal{L}_k + \ln \sigma_k$;
- **PCGrad**: Orthogonal projection of conflicting gradients $\mathbf{g}_i \leftarrow \mathbf{g}_i - rac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2}\mathbf{g}_j$;
- **Constrained Pareto Score Fusion**: Real-time PID control tracking $\max 	ext{GMV} 	ext{ s.t. } 	ext{CTR} \ge 	ext{CTR}_0$.

---

## Module 5: Production Training Dashboard Instrumentation & Anti-Confusion Protocol

```text
Production Training Dashboard Layout (4-Tier Panel Structure):
┌────────────────────────────────────────────────────────────────────────┐
│ Panel 1: Optimization & Loss Dynamics (Weighted Loss, Grad Norm, LR)   │
├────────────────────────────────────────────────────────────────────────┤
│ Panel 2: Discrimination & Ranking (Request-GAUC, User-GAUC, NDCG@5)    │
├────────────────────────────────────────────────────────────────────────┤
│ Panel 3: Probability Calibration (PCOC Ratio, ECE, Reliability Plots)  │
├────────────────────────────────────────────────────────────────────────┤
│ Panel 4: Cohort Slices & Guardrail Proxies (Cold-Start GAUC, P99 SLA)  │
└────────────────────────────────────────────────────────────────────────┘
```
