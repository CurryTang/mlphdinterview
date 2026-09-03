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
8. **High-Yield Industrial ML Multiple-Choice Quizzes**

---

## Module 1: Ranking Base Models & Multi-Objective Seesaw Mitigation

```text
The 3-Dimensional Evolution of Industrial Ranking Backbones:
1. Feature Interaction: DCN-v2 (Low-Rank Cross Networks) & DLRM (Explicit Embedding Dot-Products)
2. User Behavior Sequence: DIN (Target Attention) & SIM (Two-Stage Search: Hard Search + Soft Attention)
3. Multi-Task Learning: MMoE (Dynamic Gating) & PLE (Task-Specific vs. Shared Expert Physical Isolation)
```

### 1. The 3 Architectural Pillars (with Links to Business Algorithm Notes & Pseudocode)

Industrial precision rankers must execute bounded high-order feature interactions and activate tens of thousands of user behavior events under strict $P99 \le 15\text{ms}$ SLAs. The detailed architectural designs and PyTorch pseudocode implementations are documented across the Business Algorithm curriculum:

#### Pillar 1: Feature Interaction Backbone
> 📘 **Detailed Guide**: [[BusinessAlgorithm02C Feature Interaction.md|Chapter 11 · Feature Interaction, Coarse Ranking, and Personalization]] (with FM, DCN-v2 & DLRM code)
- **DCN-v2 (Deep & Cross Network v2)**: Employs low-rank decomposition ($\mathbf{W}_l = \mathbf{U}_l \mathbf{V}_l^T$) with MoE subspace gating to compute explicit bounded high-order polynomial cross terms with $\mathcal{O}(d \cdot r)$ efficiency:
  $$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l$$
- **DLRM (Deep Learning Recommendation Model)**: Meta's open-source architecture that explicitly decouples dense continuous features (Bottom MLP) from sparse categorical IDs (Embedding Tables), extracting upper-triangular dot-product interactions (`torch.bmm(E, E.T)`) into a Top MLP.

#### Pillar 2: User Behavior Sequence Modeling Backbone
> 📘 **Detailed Guide**: [[BusinessAlgorithm02D User Sequences.md|Chapter 12 · User Behavior Sequences]] (with DIN Target-Attention & SIM Two-Stage Retrieval code)
- **DIN (Deep Interest Network)**: Employs candidate query $\mathbf{q}$ over historical actions $[\mathbf{h}_1, \dots, \mathbf{h}_L]$ to compute **Target Attention**: $\mathbf{u}(\mathbf{q}) = \sum \alpha_j \mathbf{h}_j$ where $\alpha_j = \text{MLP}([\mathbf{q}, \mathbf{h}_j, \mathbf{q}-\mathbf{h}_j, \mathbf{q}\odot\mathbf{h}_j])$;
- **SIM (Search-based Interest Model)**: Two-stage decoupling—**Hard Search** filters Top-50 category-matched actions, followed by fine-grained soft attention with time-delta embeddings $\Delta t$, scaling to 50,000+ lifelong actions.

#### Pillar 3: Multi-Task Learning (MTL) Backbone
> 📘 **Detailed Guide**: [[BusinessAlgorithm02B Multi-Objective Ranking.md|Chapter 10 · Multi-Objective Learning and Score Fusion]] (with MMoE & PLE decoupled routing code)
- **MMoE (Multi-gate Mixture-of-Experts)**: Shared expert pool with task-specific Softmax gating $\mathbf{h}_t = \sum g_{t,e} f_e(\mathbf{x})$;
- **PLE (Progressive Layered Extraction)**: Physical isolation of **Task-Specific Experts** and **Shared Experts**, eliminating negative transfer and seesaw degradation between CTR and CVR.

---

### 2. Multi-Objective Representation Conflicts & The Seesaw Effect
1. **Directional Gradient Conflicts**: $\cos(\mathbf{g}_A, \mathbf{g}_B) < 0$;
2. **Frequency & Magnitude Domination**: High-frequency CTR dominates sparse conversion tasks;
3. **Sample Space Mismatch**: CTR on $\mathcal{D}_{\text{imp}}$ vs. CVR on $\mathcal{D}_{\text{click}}$.

#### Solution Matrix
- **PLE (Progressive Layered Extraction)**: Physical isolation of task-specific and shared experts;
- **Uncertainty Weighting**: $\mathcal{L} = \sum \left( \frac{1}{2\sigma_k^2}\mathcal{L}_k + \ln \sigma_k \right)$;
- **PCGrad**: Orthogonal projection of conflicting gradients $\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2}\mathbf{g}_j$;
- **Constrained Pareto Fusion**: Real-time PID control tracking $\max \text{GMV} \text{ s.t. } \text{CTR} \ge \text{CTR}_0$.

---

## Module 2: Training Sample Scale, Negative Sampling & Probability Recovery

```text
Fundamental Differences: Retrieval vs. Ranking Datasets:
• Retrieval: Entire corpus ($10^7 \sim 10^9$ items), uniform random negatives + in-batch negatives.
• Ranking: Top $1,000 \sim 3,000$ candidates, strictly real Impression-Unclicked negative events.
```

### Mathematical Probability Recovery Formula Under Negative Downsampling ($w$)
$$\hat{p} = \frac{p}{p + w(1-p)} \implies p = \frac{\hat{p}}{\hat{p} + \frac{1 - \hat{p}}{w}}$$

---

## Module 3: Industrial Long-Sequence Modeling Paradigms

| Architecture | FLOPs Complexity / Query | Memory Bandwidth Bound | Online Latency (P99) | Max Sequence Length $L$ | Typical Industrial Placement |
|---|---|---|---|---|---|
| **Truncated Transformer<br>(SASRec/BST)** | $\mathcal{O}(K \cdot N^2 \cdot d)$ | **High** (Self-attention memory matrix) | Moderate ($10 \sim 20\text{ms}$) | $N \le 100$ | Pre-ranking / Short-term intent |
| **Compressive Memory<br>(MIMN)** | $\mathcal{O}(K \cdot C \cdot d)$ | **Ultra-Low** (Slot matrix $C \ll L$) | **Ultra-Fast** ($< 3\text{ms}$) | $L \ge 10,000$ | High-QPS Retrieval / Coarse Ranking |
| **Lifelong Target-Attention<br>(DIN)** | $\mathcal{O}(K \cdot L \cdot d)$ | **Excessive** (Fetches $L$ embeddings) | **Excessive** ($> 50\text{ms}$) | $L \le 200$ | Short-to-medium sequence ranking |
| **Retrieval-Augmented<br>(SIM Hard Search)** | $\mathcal{O}(K \cdot M \cdot d)$ ($M \ll L$) | **Ultra-Low** (Fetches only Top-50 IDs) | **Ultra-Fast** ($5 \sim 8\text{ms}$) | **$L \ge 50,000+$** | **Production Precision Ranking SOTA** |
| **Retrieval-Augmented<br>(SIM Soft / ETA)** | $\mathcal{O}(K \cdot M \cdot d + \text{LSH})$ | **Moderate** (Maintains vector index) | Excellent ($8 \sim 12\text{ms}$) | **$L \ge 10,000+$** | Cross-category long-sequence ranking |

---

### 2. Five Long-Sequence Architectures PyTorch Pseudocode (Collapsible)

<details class="exercise" open>
<summary><span class="q-label">Arch 1 · Pseudocode</span> <span class="q-text">Truncated Self-Attention (SASRec / BST Truncated Transformer)</span></summary>

```python
import torch
import torch.nn as nn

class SASRecTruncatedTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, num_layers=2, max_len=50, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, seq_embeddings, mask=None):
        B, N, d = seq_embeddings.shape
        positions = torch.arange(N, device=seq_embeddings.device).unsqueeze(0).expand(B, N)
        x = self.layer_norm(seq_embeddings + self.pos_emb(positions))
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        padding_mask = (~mask) if mask is not None else None
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return out[:, -1, :] # context representation of latest action [B, d]
```
</details>

<details class="exercise">
<summary><span class="q-label">Arch 2 · Pseudocode</span> <span class="q-text">Compressive Memory Networks (MIMN Neural Slot Memory Matrix)</span></summary>

```python
class MIMNMemoryNetwork(nn.Module):
    def __init__(self, embed_dim=64, num_slots=8):
        super().__init__()
        self.num_slots = num_slots # compressed into C=8 slots
        self.write_gate = nn.Linear(embed_dim * 2, num_slots)
        self.erase_gate = nn.Linear(embed_dim, embed_dim)
        self.add_gate = nn.Linear(embed_dim, embed_dim)

    def forward(self, user_history_stream, initial_memory=None):
        B, L, d = user_history_stream.shape
        M = initial_memory if initial_memory is not None else torch.zeros(B, self.num_slots, d, device=user_history_stream.device)
        for t in range(L):
            xt = user_history_stream[:, t, :]
            w = torch.softmax(self.write_gate(torch.cat([xt, M.mean(dim=1)], dim=-1)), dim=-1) # [B, C]
            erase = torch.sigmoid(self.erase_gate(xt)).unsqueeze(1)
            add = torch.tanh(self.add_gate(xt)).unsqueeze(1)
            M = M * (1.0 - w.unsqueeze(-1) * erase) + (w.unsqueeze(-1) * add)
        return M # Slot matrix [B, C, d], read online with O(1) complexity
```
</details>

<details class="exercise">
<summary><span class="q-label">Arch 3 · Pseudocode</span> <span class="q-text">Lifelong Target-Attention (DIN Full Sequence Target Attention)</span></summary>

```python
class DINFullTargetAttention(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.activation_unit = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, candidate_query, lifelong_history, mask=None):
        B, L, d = lifelong_history.shape
        q_exp = candidate_query.unsqueeze(1).expand(B, L, d)
        interaction = torch.cat([q_exp, lifelong_history, q_exp - lifelong_history, q_exp * lifelong_history], dim=-1)
        scores = self.activation_unit(interaction).squeeze(-1) # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask, 0.0)
        return torch.bmm(scores.unsqueeze(1), lifelong_history).squeeze(1) # [B, d]
```
</details>

<details class="exercise">
<summary><span class="q-label">Arch 4 · Pseudocode</span> <span class="q-text">Hierarchical Multi-Resolution Pooling (HPMN Multi-Scale Decay)</span></summary>

```python
class HPMNHierarchicalPooling(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.decay_lambda = nn.Parameter(torch.tensor([0.05]))

    def forward(self, session_seq, daily_seq, monthly_seq, time_deltas_daily, time_deltas_monthly):
        h_session = session_seq.mean(dim=1)
        decay_daily = torch.exp(-torch.clamp(self.decay_lambda, min=1e-4) * time_deltas_daily).unsqueeze(-1)
        h_daily = (daily_seq * decay_daily).sum(dim=1) / (decay_daily.sum(dim=1) + 1e-6)
        decay_monthly = torch.exp(-torch.clamp(self.decay_lambda, min=1e-4) * time_deltas_monthly).unsqueeze(-1)
        h_monthly = (monthly_seq * decay_monthly).sum(dim=1) / (decay_monthly.sum(dim=1) + 1e-6)
        return torch.cat([h_session, h_daily, h_monthly], dim=-1) # [B, 3*d]
```
</details>

<details class="exercise">
<summary><span class="q-label">Arch 5 · Pseudocode</span> <span class="q-text">Retrieval-Augmented Lifelong History (SIM Hard/Soft Search)</span></summary>

```python
class SIMRetrievalAugmentedModel(nn.Module):
    def __init__(self, embed_dim=64, top_m=50):
        super().__init__()
        self.top_m = top_m
        self.time_delta_emb = nn.Embedding(100, embed_dim)
        self.attention = DINFullTargetAttention(embed_dim * 2, hidden_dim=64)

    def forward(self, cand_item_id, cand_category_id, user_lifelong_ids, user_lifelong_cats, user_lifelong_times, item_embed_table):
        # Stage 1: Hard Search category filtering
        match_mask = (user_lifelong_cats == cand_category_id.unsqueeze(1))
        # Stage 2: Target-Attention with time-delta embeddings
        cand_vec = item_embed_table(cand_item_id)
        hist_vec = item_embed_table(user_lifelong_ids[:, :self.top_m])
        time_vec = self.time_delta_emb(user_lifelong_times[:, :self.top_m])
        combined_hist = torch.cat([hist_vec, time_vec], dim=-1)
        combined_cand = torch.cat([cand_vec, torch.zeros_like(cand_vec)], dim=-1)
        return self.attention(combined_cand, combined_hist)
```
</details>


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

- **Plackett-Luce Loss**:
  $$\mathcal{L}_{\text{Plackett-Luce}} = -\sum_{k=1}^K \log \left( \frac{\exp(s_{\pi_k})}{\sum_{j=k}^K \exp(s_{\pi_j})} \right)$$
- **Slate Reward RL**:
  $$R(\pi) = \sum_{k=1}^K \gamma^{k-1}(\text{Click}_k \cdot \text{Margin}_k + \text{GMV}_k) - \lambda \cdot \text{Redundancy}(\pi)$$
- **P99 ≤ 20ms SLA**: Prefix KV Cache sharing across beam steps + 18ms hard timeout fallback to precision ranking order.

---

## Module 5: Multi-Tier Metric Pyramid & Training Dashboard Instrumentation

### 1. Global ROC-AUC vs Grouped AUC (GAUC)
- **Global AUC**:
  $$\text{AUC} = \frac{1}{|\mathcal{D}^+| \cdot |\mathcal{D}^-|} \sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \left( \mathbb{I}(s_i > s_j) + \frac{1}{2} \mathbb{I}(s_i = s_j) \right)$$
  (Vulnerable to inter-user activity bias and Simpson's paradox);
- **GAUC**: Weighted within-group average:
  $$\text{GAUC} = \frac{\sum_{g \in \mathcal{G}, \, n_g^+ > 0, \, n_g^- > 0} w_g \cdot \text{AUC}_g}{\sum_{g \in \mathcal{G}, \, n_g^+ > 0, \, n_g^- > 0} w_g}, \quad \text{where } w_g = n_g \text{ (Impressions)}$$
  - **User-GAUC**: Evaluates cross-session profile matching (confounded by diurnal drift);
  - **Request-GAUC**: Evaluates within-slate item ranking on a single refresh (**the gold standard for ranking**).

### 2. Probability Calibration (PCOC & ECE)
- **PCOC**:
  $$\text{PCOC} = \frac{\sum \hat{p}_i}{\sum y_i} \quad (=1.0 \text{ Perfect Calibration})$$
- **ECE**:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 3. Production Training Dashboard Layout (4-Tier Instrumentation)
- Panel 1: Optimization Dynamics (Loss, Grad Norm, LR);
- Panel 2: Ranking Discrimination (Request-GAUC, User-GAUC, NDCG@5);
- Panel 3: Calibration Reliability (PCOC, ECE, Reliability Plots);
- Panel 4: Cohort Slices & Guardrails (Cold-Start GAUC, P99 Latency SLA).

---

## Module 6: Industrial Online Experimentation & A/B Testing Lifecycle

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
