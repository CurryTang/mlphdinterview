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
> 📘 **Detailed Guide**: [Chapter 11 · Feature Interaction, Coarse Ranking, and Personalization](notes/BusinessAlgorithm/BusinessAlgorithm02C%20Feature%20Interaction.en.md) (with FM, DCN-v2 & DLRM code)
- **DCN-v2 (Deep & Cross Network v2)**: Employs low-rank decomposition ($\mathbf{W}_l = \mathbf{U}_l \mathbf{V}_l^T$) with MoE subspace gating to compute explicit bounded high-order polynomial cross terms with $\mathcal{O}(d \cdot r)$ efficiency:
  $$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l$$
- **DLRM (Deep Learning Recommendation Model)**: Meta's open-source architecture that explicitly decouples dense continuous features (Bottom MLP) from sparse categorical IDs (Embedding Tables), extracting upper-triangular dot-product interactions (`torch.bmm(E, E.T)`) into a Top MLP.

#### Pillar 2: User Behavior Sequence Modeling Backbone
> 📘 **Detailed Guide**: [Chapter 12 · User Behavior Sequences](notes/BusinessAlgorithm/BusinessAlgorithm02D%20User%20Sequences.en.md) (with DIN Target-Attention & SIM Two-Stage Retrieval code)
- **DIN (Deep Interest Network)**: Employs candidate query $\mathbf{q}$ over historical actions $[\mathbf{h}_1, \dots, \mathbf{h}_L]$ to compute **Target Attention**: $\mathbf{u}(\mathbf{q}) = \sum \alpha_j \mathbf{h}_j$ where $\alpha_j = \text{MLP}([\mathbf{q}, \mathbf{h}_j, \mathbf{q}-\mathbf{h}_j, \mathbf{q}\odot\mathbf{h}_j])$;
- **SIM (Search-based Interest Model)**: Two-stage decoupling—**Hard Search** filters Top-50 category-matched actions, followed by fine-grained soft attention with time-delta embeddings $\Delta t$, scaling to 50,000+ lifelong actions.

#### Pillar 3: Multi-Task Learning (MTL) Backbone
> 📘 **Detailed Guide**: [Chapter 10 · Multi-Objective Learning and Score Fusion](notes/BusinessAlgorithm/BusinessAlgorithm02B%20Multi-Objective%20Ranking.en.md) (with MMoE & PLE decoupled routing code)
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

### Mathematical Principle of CUPED Variance Reduction
$$\hat{Y}_{\text{CUPED}} = Y - \theta(X - \mathbb{E}[X]), \quad \text{where } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
$$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - ho^2)$$

When pre/post correlation is $ho = 0.8$, metric variance is slashed by $64\%$, reducing sample size requirements by nearly $3\times$!

---

## Module 7: Flagship Case Studies & Small-Sample Inference

### Case 1: Sign-Up Funnel $2 \times 2$ Full Factorial Experiment
- Interaction effect estimator: $\hat{\beta}_3 = T_3 - T_2 - T_1 + T_0$;
- Viewport Intersection Observer triggered exposure to prevent dilution bias.

### Case 2: PDP UI Redesign (CTR Up, CVR Flat) & Small-Sample Tier Inference
1. **Net Lift**: $\text{CTCVR} = \text{CTR} \times \text{CVR} \implies +12\%$ **net gain in total purchase orders per impression**;
2. **Traffic Dilution**: Influx of lower-intent marginal users dilutes denominator while preserving baseline conversion rate;
3. **Small-Sample VIP Tier Statistical Inference**:
   - **Empirical Bayes Partial Pooling / Shrinkage**: $\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \cdot \mu_{\text{grand}} + (1 - B) \cdot \bar{Y}_{\text{small}}$;
   - **CUPED with 30-Day Historical Baseline Spending**: Compresses variance to $28\%$, expanding effective sample size by $3.5\times$;
   - **Non-Parametric Exact Permutation Tests & Bayesian Posterior Superiority**: $P(\theta_T > \theta_C \mid \mathcal{D}) > 0.90$.

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
>   3. **Small-Sample Inference**: $p > 0.05$ reflects low power ($\text{Power} < 20\%$), not evidence of absence. Empirical Bayes shrinkage $\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \mu_{\text{grand}} + (1-B) \bar{Y}_{\text{small}}$ borrows statistical strength across tiers.
</details>
