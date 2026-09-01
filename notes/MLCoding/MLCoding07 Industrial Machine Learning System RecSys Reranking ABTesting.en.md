# ML Coding 07 · Industrial ML Systems: RecSys Ranking, Long-Sequence Modeling, Generative Reranking & A/B Testing Causal Inference

In large-scale industrial machine learning and recommender systems, senior ML engineers must master not only deep model architectures and loss optimization, but also **the entire production lifecycle spanning high-throughput sparse sample streams, lifelong user behavior sequences, full-slate generative reranking, multi-tier metric hierarchies, and online A/B testing causal inference**.

This guide consolidates the end-to-end industrial ML engineering stack into 8 structured logical modules:
1. **Precision Ranking Backbones & Multi-Objective Seesaw Effect (Negative Transfer) Mitigation**
2. **Training Sample Scale, Negative Sampling Paradigms & Mathematical Probability Recovery**
3. **Industrial Lifelong Sequence Modeling (5 Paradigms, Latency Tradeoffs & Noise Suppression)**
4. **E-Commerce Generative Reranking Pipeline (Candidate Serialization, Constrained Beam Search & P99 ≤ 20ms SLAs)**
5. **Multi-Tier Metric Pyramid (ROC-AUC vs. GAUC by User/Request, Unbiased Downsampling & Calibration PCOC/ECE)**
6. **End-to-End Online Experimentation & A/B Testing (CUPED Variance Reduction, SRM & Decision Guardrails)**
7. **Flagship Case Studies: Sign-Up Funnel 2x2 Factorial Design, PDP Conversion Attribution & Small-Sample Tier Inference**
8. **Senior Technical Interview Pitch Framework & Rapid-Fire Questions**

---

## Module 1: Ranking Base Models & Multi-Objective Seesaw Mitigation

```text
The 3-Dimensional Evolution of Industrial Ranking Backbones:
1. Feature Interaction: DCN-v2 (Low-Rank Cross Networks) & DLRM (Explicit Embedding Dot-Products)
2. User Behavior Sequence: DIN (Target Attention) & SIM (Two-Stage Search: Hard Search + Soft Attention)
3. Multi-Task Learning: MMoE (Dynamic Gating) & PLE (Task-Specific vs. Shared Expert Physical Isolation)
```

### Multi-Objective Representation Conflicts & The Seesaw Effect
1. **Directional Gradient Conflicts**: $\cos(\mathbf{g}_A, \mathbf{g}_B) < 0$;
2. **Frequency & Magnitude Domination**: High-frequency CTR dominates sparse conversion tasks;
3. **Sample Space Mismatch**: CTR on $\mathcal{D}_{	ext{imp}}$ vs. CVR on $\mathcal{D}_{	ext{click}}$.

#### Solution Matrix
- **PLE (Progressive Layered Extraction)**: Physical isolation of task-specific and shared experts;
- **Uncertainty Weighting**: $\mathcal{L} = \sum rac{1}{2\sigma_k^2}\mathcal{L}_k + \ln \sigma_k$;
- **PCGrad**: Orthogonal projection of conflicting gradients $\mathbf{g}_i \leftarrow \mathbf{g}_i - rac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2}\mathbf{g}_j$;
- **Constrained Pareto Fusion**: Real-time PID control tracking $\max 	ext{GMV} 	ext{ s.t. } 	ext{CTR} \ge 	ext{CTR}_0$.

---

## Module 2: Training Sample Scale, Negative Sampling & Probability Recovery

```text
Fundamental Differences: Retrieval vs. Ranking Datasets:
• Retrieval: Entire corpus ($10^7 \sim 10^9$ items), uniform random negatives + in-batch negatives.
• Ranking: Top $1,000 \sim 3,000$ candidates, strictly real Impression-Unclicked negative events.
```

### Mathematical Probability Recovery Formula Under Negative Downsampling ($w$)
$$\hat{p} = rac{p}{p + w(1-p)} \implies p = rac{\hat{p}}{\hat{p} + rac{1 - \hat{p}}{w}}$$

---

## Module 3: Industrial Long-Sequence Modeling Paradigms

| Architecture | FLOPs Complexity / Query | Memory Bandwidth Bound | Online Latency (P99) | Max Sequence Length $L$ | Typical Industrial Placement |
|---|---|---|---|---|---|
| **Truncated Transformer<br>(SASRec/BST)** | $\mathcal{O}(K \cdot N^2 \cdot d)$ | **High** (Self-attention memory matrix) | Moderate ($10 \sim 20	ext{ms}$) | $N \le 100$ | Pre-ranking / Short-term intent |
| **Compressive Memory<br>(MIMN)** | $\mathcal{O}(K \cdot C \cdot d)$ | **Ultra-Low** (Slot matrix $C \ll L$) | **Ultra-Fast** ($< 3	ext{ms}$) | $L \ge 10,000$ | High-QPS Retrieval / Coarse Ranking |
| **Lifelong Target-Attention<br>(DIN)** | $\mathcal{O}(K \cdot L \cdot d)$ | **Excessive** (Fetches $L$ embeddings) | **Excessive** ($> 50	ext{ms}$) | $L \le 200$ | Short-to-medium sequence ranking |
| **Retrieval-Augmented<br>(SIM Hard Search)** | $\mathcal{O}(K \cdot M \cdot d)$ ($M \ll L$) | **Ultra-Low** (Fetches only Top-50 IDs) | **Ultra-Fast** ($5 \sim 8	ext{ms}$) | **$L \ge 50,000+$** | **Production Precision Ranking SOTA** |
| **Retrieval-Augmented<br>(SIM Soft / ETA)** | $\mathcal{O}(K \cdot M \cdot d + 	ext{LSH})$ | **Moderate** (Maintains vector index) | Excellent ($8 \sim 12	ext{ms}$) | **$L \ge 10,000+$** | Cross-category long-sequence ranking |

---

## Module 4: E-Commerce Generative Reranking Pipeline

```text
End-to-End Generative Reranking Pipeline:
[Precision Ranking Top-50 Output]
       │
       ▼
[1. Candidate Serialization: Slot Tokens <C_01> ~ <C_50> + User Prefix]
       │
       ▼
[2. Constrained Beam Search Generator (6-Layer Decoder-Only, Masked Softmax)]
       │
       ▼
[3. Final Display Slate: π = [π₁, π₂, ..., π_K] (K = 6~10)]
```

- **Plackett-Luce Loss**: $\mathcal{L}_{	ext{List}} = -\sum_{k=1}^K \log \left( rac{\exp(s_{\pi_k})}{\sum_{j=k}^K \exp(s_{\pi_j})} ight)$
- **Slate Reward RL**: $R(\pi) = \sum \gamma^{k-1}(	ext{Click}_k \cdot 	ext{Margin}_k + 	ext{GMV}_k) - \lambda \cdot 	ext{Redundancy}(\pi)$
- **P99 ≤ 20ms SLA**: Prefix KV Cache sharing across beam steps + 18ms hard timeout fallback to precision ranking order.

---

## Module 5: Multi-Tier Metric Pyramid & Training Dashboard Instrumentation

### 1. Global ROC-AUC vs Grouped AUC (GAUC)
- **Global AUC**: Vulnerable to inter-user activity bias;
- **GAUC**: Weighted within-group average $	ext{GAUC} = rac{\sum w_g 	ext{AUC}_g}{\sum w_g}$:
  - **User-GAUC**: Evaluates cross-session profile matching (confounded by diurnal drift);
  - **Request-GAUC**: Evaluates within-slate item ranking on a single refresh (**the gold standard for ranking**).

### 2. Probability Calibration (PCOC & ECE)
- **PCOC**: $rac{\sum \hat{p}_i}{\sum y_i} = 1.0$ (Critical for $eCPM$ auction bidding);
- **ECE**: $\sum rac{|B_m|}{N} |	ext{acc}(B_m) - 	ext{conf}(B_m)|$.

### 3. Production Training Dashboard Layout (4-Tier Instrumentation)
- Panel 1: Optimization Dynamics (Loss, Grad Norm, LR);
- Panel 2: Ranking Discrimination (Request-GAUC, User-GAUC, NDCG@5);
- Panel 3: Calibration Reliability (PCOC, ECE, Reliability Plots);
- Panel 4: Cohort Slices & Guardrails (Cold-Start GAUC, P99 Latency SLA).

---

## Module 6: Industrial Online Experimentation & A/B Testing Lifecycle

### Mathematical Principle of CUPED Variance Reduction
$$\hat{Y}_{	ext{CUPED}} = Y - 	heta(X - \mathbb{E}[X]), \quad 	ext{where } 	heta = rac{	ext{Cov}(Y, X)}{	ext{Var}(X)}$$
$$	ext{Var}(\hat{Y}_{	ext{CUPED}}) = 	ext{Var}(Y) \cdot (1 - ho^2)$$

---

## Module 7: Flagship Case Studies & Small-Sample Inference

### Case 1: Sign-Up Funnel $2 	imes 2$ Full Factorial Experiment
- Interaction effect estimator: $\hat{eta}_3 = T_3 - T_2 - T_1 + T_0$;
- Viewport Intersection Observer triggered exposure to prevent dilution bias.

### Case 2: PDP UI Redesign (CTR Up, CVR Flat) & Small-Sample Tier Inference
1. **Net Lift**: $	ext{CTCVR} = 	ext{CTR} 	imes 	ext{CVR} \implies +12\%$ **net gain in total purchase orders per impression**;
2. **Traffic Dilution**: Influx of lower-intent marginal users dilutes denominator while preserving baseline conversion rate;
3. **Small-Sample VIP Tier Statistical Inference**:
   - **Empirical Bayes Partial Pooling / Shrinkage**: $\hat{	heta}_{	ext{small}}^{	ext{shrunk}} = B \cdot \mu_{	ext{grand}} + (1 - B) \cdot ar{Y}_{	ext{small}}$;
   - **CUPED with 30-Day Historical Baseline Spending**: Compresses variance to $28\%$, expanding effective sample size by $3.5	imes$;
   - **Non-Parametric Exact Permutation Tests & Bayesian Posterior Superiority**: $P(	heta_T > 	heta_C \mid \mathcal{D}) > 0.90$.

---

## Module 8: Senior Technical Interview Pitch Framework

### Verbal Pitch Architecture
1. **Architecture & Multi-Objective**: "Decoupled DCN-v2 + DLRM feature interactions, SIM lifelong sequence modeling, and PLE with PCGrad & PID constrained Pareto fusion to eliminate seesaw negative transfer."
2. **Sampling & Unbiased Evaluation**: "Strict unclicked impression negatives, negative downsampling with exact probability recovery, evaluated on Request-GAUC and PCOC calibration."
3. **Generative Reranking**: "Slot token serialization, prefix KV cache sharing, and constrained beam search optimizing joint slate utility under P99 $\le 15	ext{ms}$."
4. **Online A/B Testing & Attribution**: "14-day orthogonal A/B testing with CUPED variance reduction, CTCVR funnel decomposition for traffic dilution, and Empirical Bayes shrinkage for small-sample VIP tiers."
