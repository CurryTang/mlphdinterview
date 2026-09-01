# Industrial Recommendation Ranking: Base Model Family, Sampling Strategies & Multi-Tier Metrics

In industrial recommender systems and computational advertising, building a high-throughput, multi-objective, and well-calibrated ranking stack is a core engineering and algorithmic foundation. In senior ML system interviews and architecture reviews, **how you structure the presentation of your base model family, training sample scale, negative sampling scheme, and metric hierarchy (Ranking vs. Business Metrics, Calibration Plots, Slice Metrics)** directly distinguishes senior practitioners.

This note systematically constructs the 4 foundational pillars of industrial recommendation ranking:
1. **The Base Model Family (Feature Interaction, Long-Term User Behavior Sequences & Multi-Task Learning)**
2. **Training Sample Construction, Negative Sampling Strategies & Mathematical Probability Recovery**
3. **The Multi-Tier Metric Hierarchy (Ranking vs. Business Metrics, Calibration PCOC & Subgroup Slices)**
4. **Senior Interview Framework & Methodological Response Guide**

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

1. **Feature Interaction Backbone**:
   - **DCN-v2 (Deep & Cross Network v2)**: Leverages explicit cross layers combined with low-rank matrix decomposition to approximate arbitrary high-degree polynomial feature combinations with low latency;
   - **DLRM (Deep Learning Recommendation Model)**: Decouples sparse categorical embeddings from dense bottom MLPs, feeding explicit dot-product feature interactions into top MLP layers.
2. **User Sequence Modeling Backbone**:
   - **DIN (Deep Interest Network)**: Computes candidate-aware **Target Attention** over past historical user interactions, avoiding information collapse from static mean-pooling;
   - **SIM (Search-based Interest Model)**: Scales historical sequence capacity up to $10^4$ interactions via a two-stage paradigm: sub-sequence **Hard Search** (category/tag inverted index) followed by fine-grained **Soft Attention**.
3. **Multi-Task Learning Backbone**:
   - **MMoE (Multi-gate Mixture-of-Experts)**: Assigns dedicated softmax gating routing for each objective over shared expert sub-networks;
   - **PLE (Progressive Layered Extraction)**: Explicitly separates **Task-Specific Experts** from **Shared Experts**, eliminating negative transfer between weakly correlated objectives (e.g., Click vs. Conversion).

---

## Module 2: Training Sample Scale, Negative Sampling & Probability Recovery

In production systems, sample collection and negative construction differ fundamentally between Retrieval and Ranking:

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

### 1. Training Sample Scale & Streaming Pipeline

- **Rolling Horizon**: Trained on **14 to 30 days** of impression/click logs (spanning billions to tens of billions of rows);
- **Streaming Pipeline**:
  - Daily offline batch checkpoints update core base models;
  - Real-time streaming pipelines (e.g., Kafka/Flink) join impressions with clicks within minutes to perform streaming mini-batch gradient updates on embeddings and top layers.

---

### 2. Negative Downsampling & Probability Recovery Derivation

#### Why Production Rankers Downsample Negatives
In feed and e-commerce ranking, positive CTR is naturally sparse ($p pprox 1\% \sim 5\%$), causing $1:20 \sim 1:100$ class imbalance.
1. **Computational & Storage Efficiency**: Downsampling negatives by $90\%$ cuts training data volume by up to $80\%$, drastically accelerating training throughput;
2. **Gradient Stability**: Prevents gradients from being overwhelmed by uninformative negative instances.

#### Probability Distortion Under Negative Downsampling
Let negative samples be randomly retained with downsampling probability $w \in (0, 1]$, while retaining $100\%$ of positive samples.
The observed conditional probability $\hat{p} = P(Y=1 \mid X, 	ext{sampled})$ learned by the model is artificially inflated:

$$\hat{p} = rac{P(Y=1 \mid X)}{P(Y=1 \mid X) + w \cdot P(Y=0 \mid X)} = rac{p}{p + w(1-p)}$$

#### Mathematical Probability Recovery Formula
For auction bidding ($eCPM = pCTR 	imes pCVR 	imes 	ext{Bid}$) or score blending, raw model output $\hat{p}$ must be inverted back to true physical probability $p$:

$$p = rac{\hat{p}}{\hat{p} + rac{1 - \hat{p}}{w}}$$

```text
Numerical Verification Example:
Given true CTR p = 0.01 (1%), negative downsampling retention rate w = 0.1 (10%):
1. Observed CTR on sampled data: p_hat = 0.01 / (0.01 + 0.1 * 0.99) = 0.0917 (9.17%)
2. Real-time inference recalibration: p = 0.0917 / (0.0917 + (1 - 0.0917) / 0.1) = 0.01 (Exact 1% restored!)
```

---

## Module 3: The Multi-Tier Metric Hierarchy

```text
The Recommendation Metric Pyramid:
                     ▲
                    / \     [Tier 3: Business & North Star Metrics]
                   /   \    • DAU/MAU, Dwell Time, GMV, D7/D30 Retention, Creator Diversity
                  /─────                 /       \   [Tier 2: Slice & Guardrail Metrics]
                /         \  • New vs. Old Users, Long-Tail Items, P99 Latency
               /───────────              /             \ [Tier 1: Offline Ranking & Calibration Metrics]
             /               \• GAUC (User-Grouped), Global AUC, LogLoss, PCOC, NDCG@K
            └─────────────────┘
```

### 1. Offline Ranking Metrics vs. Online Business Metrics

| Metric Tier | Core Metric | Mathematical Definition | Industrial Insight |
|---|---|---|---|
| **Global Ranking** | **Global AUC** | Probability that a randomly chosen positive sample scores higher than a random negative across the entire dataset. | **Vulnerable to user activity bias**: A model that simply assigns higher scores to high-activity users achieves high Global AUC while failing to rank correctly within any single user's list. |
| **Personalized Ranking** | **Group AUC (GAUC)** | Computes $	ext{AUC}_u$ for each individual user $u$, weighted by impression count $w_u$:<br>$$	ext{GAUC} = rac{\sum_u w_u \cdot 	ext{AUC}_u}{\sum_u w_u}$$ | **The Core Offline North Star for Ranking**! Strips away inter-user activity bias and purely evaluates within-list personalized ranking quality. |
| **Probability Quality** | **LogLoss (Cross-Entropy)** | $-rac{1}{N}\sum [y \log \hat{p} + (1-y)\log(1-\hat{p})]$. | Highly sensitive to absolute predicted probabilities; directly impacted by negative sampling. |
| **Position Sensitivity**| **NDCG@K / MRR** | Evaluates top-ranked items using logarithmic position discounts $rac{1}{\log_2(	ext{rank}+1)}$. | Aligns with user viewing patterns restricted to top viewports (Top 3 / Top 5). |
| **Online Business** | **CTR / CVR / GMV / Retention** | Measured via online A/B testing over multi-week horizons. | Evaluates true end-to-end user satisfaction and ecosystem dynamics. |

---

### 2. Probability Calibration: PCOC & Reliability Curves

In computational advertising and blended multi-task ranking, **having a high AUC is insufficient—the predicted scores must match true empirical conversion rates**.

#### PCOC (Predictive-over-Observed Calibration Ratio)
$$	ext{PCOC} = rac{\sum_{i=1}^N \hat{p}_i}{\sum_{i=1}^N y_i} = rac{	ext{Average Model Predicted Probability}}{	ext{Average Observed Empirical CTR}}$$

- **$	ext{PCOC} = 1.00$**: Perfectly calibrated;
- **$	ext{PCOC} > 1.00$**: Model **over-estimates** (causes advertisers to exhaust budgets prematurely or over-bid);
- **$	ext{PCOC} < 1.00$**: Model **under-estimates** (causes under-delivery of high-quality items and lost platform revenue).

#### Calibration Curves (Reliability Diagrams)
Group predicted probabilities into 10–20 equal-frequency bins. Plot the average predicted score against the observed empirical conversion rate per bin. A well-calibrated model strictly adheres to the $y = x$ diagonal line.

---

### 3. Slice Metrics & Simpson's Paradox

An aggregate Global AUC gain of $+0.5\%$ can conceal catastrophic regressions in key cohorts. Production systems require **subgroup slicing**:

1. **User Cohort Slices**: New/Cold-Start Users vs. Casual Users vs. Power Users;
2. **Item Slices**: Viral Hit Items vs. Long-Tail Niche Items;
3. **Platform Slices**: iOS vs. Android, High-speed Wi-Fi vs. Weak Network.

> ⚠️ **Guarding Against Simpson's Paradox**:
> Dominant power users account for the majority of impressions. If a new model improves power-user AUC slightly while hurting cold-start users severely, the aggregate metric may still show a net positive. Slice metrics protect long-term platform health.

---

## Module 4: How to Present and Answer This Question in an Interview

```text
Structured 4-Step Presentation Framework:
┌────────────────────────────────────────────────────────────────────────┐
│ Step 1: Base Model Family                                              │
│ • Architecture decoupling: Sparse Embeddings + DCN-v2/DLRM + SIM + PLE │
├────────────────────────────────────────────────────────────────────────┤
│ Step 2: Training Sample Scale & Negative Sampling Scheme               │
│ • 14-30 day rolling logs + streaming Flink real-time updates           │
│ • Strict unclicked impression negatives + 10% downsampling + recovery  │
├────────────────────────────────────────────────────────────────────────┤
│ Step 3: Multi-Tier Metric Stack                                        │
│ • Offline GAUC/NDCG ➔ Calibration PCOC ➔ Slices ➔ Online A/B Retention │
├────────────────────────────────────────────────────────────────────────┤
│ Step 4: How Sampling Alters Metric Interpretation                      │
│ • Negative downsampling shifts prior distribution, distorts LogLoss    │
│ • Shows why GAUC is rank-invariant while PCOC requires mathematical inversion│
└────────────────────────────────────────────────────────────────────────┘
```

### Verbal Response Script (Verbatim Reference)

1. **Base Model Architecture**:
   "Our ranking stack employs a decoupled modular design: sparse categorical features map to embedding tables, fed into **DCN-v2 and DLRM dot-product layers** for explicit non-linear feature interaction. To model long-term user interests across thousands of past interactions, we utilize **SIM (Search-based Interest Model)** with two-stage hard search and soft attention. The top layer deploys **PLE (Progressive Layered Extraction)** to simultaneously predict pCTR, pLongView, and pConversion while eliminating task interference."
2. **Sample Scale & Sampling Scheme**:
   "The offline training corpus spans rolling 30-day window logs containing billions of impressions, augmented with streaming real-time hourly updates. For sample construction, **ranking uses strictly real, unclicked impressions as negative instances**. To handle severe 1:50 class imbalance, we apply negative downsampling with retention rate $w = 10\%$, dynamically inverting predictions via $p = \frac{\hat{p}}{\hat{p} + (1-\hat{p})/w}$ during online inference."
3. **Metric Hierarchy**:
   - "**Offline Evaluation**: Our primary metric is **GAUC (Group AUC by User)** to eliminate activity bias, paired with **PCOC** and reliability diagrams to verify calibration. We track dedicated **Slice GAUC** across new users and long-tail items."
   - "**Online Evaluation**: Evaluated through orthogonal A/B testing on North Star business metrics (Dwell Time, DAU, D7 Retention) alongside P99 latency SLAs."
4. **How Sampling Changes Metric Meanings**:
   "Negative downsampling shifts the baseline data distribution. Consequently, **raw LogLoss cannot be compared across different sampling ratios, and uncalibrated probabilities will distort multi-objective score fusion and $eCPM$ auction bidding**. However, because GAUC evaluates within-user relative rankings, it exhibits rank-order invariance under uniform negative downsampling."
