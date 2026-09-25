# Multi-Objective Learning and Score Fusion

## Chapter 10: Multi-Objective Learning and Score Fusion

### 10.1 Why Multi-Objective?

Short-video platforms may simultaneously care about clicks, watch time, completion, likes, follows, and negative feedback. E-commerce cares about clicks, add-to-cart, orders, and GMV. Search must also consider relevance, quality, timeliness, geography, and personalization.

Roughly summing all objectives into one label loses structure. Training multiple models separately leads to redundant computation and causes low-frequency tasks to lack data. Multi-task learning finds a balance between these two extremes.

Two different problems are often conflated here:

```text
Multi-task learning: how to share representations while predicting CTR, duration, CVR, and other targets
Score fusion: how those predictions become the final online ranking score
```

MMoE, Shared-Bottom, and ESMM mainly address the first problem. Additive, multiplicative, rank-based, or learned fusion addresses the second. More accurate heads do not guarantee a better list. If the fusion weights let a frequent objective dominate quality and negative feedback, a sophisticated network will still optimize the wrong behavior.

Fusion cannot repair poorly defined labels either. Clicks, duration, and purchases have different observation windows, sampling rates, and calibration requirements. Each head needs an interpretable target before its weight is tuned. Online weighting is a product tradeoff among defined objectives, not a substitute for defining the training data.

### 10.2 Shared-Bottom

The simplest structure shares the bottom layers:

```text
features -> shared network -> task A tower
                           -> task B tower
                           -> task C tower
```

Total loss:

```math
\mathcal L=\sum_t \lambda_t\mathcal L_t.
```

The problem is that task gradients may conflict. Click preference favors title attractiveness, while long watch time favors sustained content value; they do not always update shared parameters in the same direction.

### 10.3 MMoE (Multi-gate Mixture-of-Experts)

MMoE deploys dedicated Softmax gating networks per task over shared expert layers:

$$\mathbf{h}_t(\mathbf{x}) = \sum_{e=1}^E g_{t,e}(\mathbf{x}) f_e(\mathbf{x}), \quad \mathbf{g}_t(\mathbf{x}) = \text{Softmax}(\mathbf{W}_t \mathbf{x})$$

#### Pseudocode: MMoE Architecture Forward Pass
```python
import torch
import torch.nn as nn

class MMoE(nn.Module):
    def __init__(self, in_features, num_experts=4, expert_dim=64, num_tasks=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, expert_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.task_gates = nn.ModuleList([
            nn.Linear(in_features, num_experts) for _ in range(num_tasks)
        ])
        self.task_towers = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1) # [B, E, d]
        task_predictions = []
        for t in range(self.num_tasks):
            gate_weights = torch.softmax(self.task_gates[t](x), dim=-1).unsqueeze(-1)
            task_rep = (expert_outputs * gate_weights).sum(dim=1)
            task_pred = self.task_towers[t](task_rep)
            task_predictions.append(task_pred)
        return task_predictions
```

---

### 10.4 PLE (Progressive Layered Extraction)

PLE physically decouples **Task-Specific Experts** from **Shared Experts**, eliminating negative transfer across weakly correlated objectives:

#### Pseudocode: PLE Extraction Layer Forward Pass
```python
class PLECustomExtractionLayer(nn.Module):
    def __init__(self, in_features, num_task_experts=2, num_shared_experts=2, expert_dim=64, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_experts = nn.ModuleList([
            nn.ModuleList([nn.Linear(in_features, expert_dim) for _ in range(num_task_experts)])
            for _ in range(num_tasks)
        ])
        self.shared_experts = nn.ModuleList([
            nn.Linear(in_features, expert_dim) for _ in range(num_shared_experts)
        ])
        total_task_experts = num_task_experts + num_shared_experts
        self.task_gates = nn.ModuleList([
            nn.Linear(in_features, total_task_experts) for _ in range(num_tasks)
        ])
        total_all_experts = num_task_experts * num_tasks + num_shared_experts
        self.shared_gate = nn.Linear(in_features, total_all_experts)

    def forward(self, task_inputs, shared_input):
        task_exp_outs = [[exp(task_inputs[t]) for exp in self.task_experts[t]] for t in range(self.num_tasks)]
        shared_exp_outs = [exp(shared_input) for exp in self.shared_experts]
        
        # 1. Task-Specific Routing
        task_next_inputs = []
        for t in range(self.num_tasks):
            pool = torch.stack(task_exp_outs[t] + shared_exp_outs, dim=1)
            gate = torch.softmax(self.task_gates[t](task_inputs[t]), dim=-1).unsqueeze(-1)
            task_next_inputs.append((pool * gate).sum(dim=1))
            
        # 2. Shared Global Routing
        all_pool = torch.stack([item for sublist in task_exp_outs for item in sublist] + shared_exp_outs, dim=1)
        shared_gate = torch.softmax(self.shared_gate(shared_input), dim=-1).unsqueeze(-1)
        shared_next_input = (all_pool * shared_gate).sum(dim=1)
        
        return task_next_inputs, shared_next_input
```


### 10.5 ESMM and Conversion Funnels
 and the Conversion Funnel

E-commerce CVR is only observable after a click. If CVR is trained using only click samples, the training distribution differs from the full exposure distribution.

ESMM utilizes:

```math
P(\text{click and conversion})
=P(\text{click})P(\text{conversion}\mid\text{click}),
```

It jointly learns CTR and CTCVR in the full exposure space, then constrains CVR through their relationship. It alleviates sample selection bias and conversion sparsity, but still relies on model assumptions and data definitions, and does not mean the counterfactual problem is completely solved.

### 10.5 Duration Modeling

Watch time is zero-inflated and influenced by video length. Directly regressing seconds will bias toward long videos.

Optional approaches:

### 10.5 Watch Time Modeling & Eliminating Duration Bias

Watch time exhibits zero inflation (swipes and bounces) and is fundamentally bounded by physical video length.

#### 1. Mathematical Derivation of YouTube (2016) Weighted Logistic Regression
The YouTube 2016 ranking model proposed a clean formulation: **use weighted classification loss to directly model continuous expected watch time without bias**, avoiding the extreme sensitivity of MSE on heavy tails while retaining negative impression signals.

**Weighted Sample Formulation:**
- Positive samples (clicked impressions with watch activity): assigned weight equal to observed watch seconds $T_i$;
- Negative samples (unclicked impressions or instant skips): assigned weight equal to 1.

**Mathematical Derivation:**
Let $N$ be total impressions, with $N_+$ positives and $N_- = N - N_+$ negatives.
Under weighted cross-entropy optimization, the odds learned by logistic regression equals the ratio of positive and negative sample weights:

```math
\operatorname{Odds} = \frac{p}{1-p} = \frac{\sum_{i \in \text{pos}} T_i}{\sum_{j \in \text{neg}} 1} = \frac{\sum_{i \in \text{pos}} T_i}{N - N_+} = \frac{\frac{1}{N}\sum_{i \in \text{pos}} T_i}{1 - \frac{N_+}{N}} = \frac{\mathbb{E}[T]}{1 - p_{\text{click}}}
```

Because online baseline click rates are low ($p_{\text{click}} \ll 1$, e.g., $0.02 \sim 0.05$), the denominator $1 - p_{\text{click}} \approx 1$:

```math
\operatorname{Odds} \approx \mathbb{E}[T]
```

Since logistic regression predicts $p = \sigma(z) = \frac{1}{1 + e^{-z}}$, the odds ratio is identically the exponential of the final linear logit: $\operatorname{Odds} = e^z$.

**Key Systems Takeaway:**
- **Training Stage**: Optimize standard binary cross-entropy with sample weights (`BCEWithLogitsLoss(weight=...)`);
- **Inference Stage**: **Directly compute the exponential of the final linear activation $\hat{T} = e^z$ as the unbiased estimate of expected watch time!** No negative sample filtering, no log-transform distortion—seamlessly bridging binary classification and continuous duration prediction.

#### 2. The Duration Bias Paradox in Short-Video Feeds
In short-video feeds (TikTok, Reels, Kuaishou), directly optimizing completion rate or raw duration causes pathological ecosystem distortion:
- **Completion Rate Trap**: A 15s video watched for 12s has an 80% completion rate (VCR); a 180s deep-dive video watched for 45s has only a 25% VCR. Optimizing purely for completion collapses the feed into homogeneous 5s trivial clips.
- **Duration Trap**: Optimizing raw duration heavily favors 45s over 12s, driving the algorithm to promote bloated, slow-paced videos, spiking immediate swipe-away rates and destroying retention.

**Production Solution: Bucket-wise Z-Score Normalization**
Segment videos into length buckets based on physical duration $L$ (e.g., `[0, 15s)`, `[15, 30s)`, `[30, 60s)`, `[60, 180s)`, `[180s, +inf)`).
Compute the empirical mean $\mu_L$ and standard deviation $\sigma_L$ of watch times within each bucket, normalizing the training label via Z-score:

```math
\tilde{T} = \frac{T - \mu_L}{\sigma_L}
```

By stripping away physical duration scale advantages, candidates compete on **relative outperformance within their peer length class**, maintaining healthy diversity across both snappy short clips and immersive long-form content.

### 10.6 Score Fusion

Model outputs usually cannot be added linearly. CTR might be in `[0, 0.2]`, duration prediction is in seconds, and CVR is even sparser. Calibrate first, then discuss fusion.

Common form:

```math
S
=w_1f_1(\hat p_{\text{click}})
+w_2f_2(\hat t)
+w_3f_3(\hat p_{\text{conversion}})
-w_4\hat p_{\text{negative}}.
```

`f_t` can be a log, power function, piecewise function, or quantile mapping. Weights do not rely solely on offline search; they ultimately require online experimentation.

Several common fusion forms behave differently:

```math
S_{\text{add}}
=p_{\text{click}}+w_1p_{\text{like}}+w_2p_{\text{share}}+\cdots,
```

```math
S_{\text{rank}}
=\sum_j\frac{w_j}{r_j+\beta_j},
```

```math
S_{\text{commerce}}
=p_{\text{click}}^{\alpha}
\times p_{\text{cart}}^{\beta}
\times p_{\text{pay}}^{\gamma}
\times \operatorname{price}^{\delta}.
```

The additive form depends on calibrated scales. Rank fusion is more scale-robust but discards score gaps. The multiplicative e-commerce form follows the exposure-to-payment funnel and strongly suppresses an item when any stage is near zero.

Another path is to learn a fusion model, taking scores from each objective and context as input. However, it still requires training labels and is harder to interpret regarding objective trade-offs. Strong business constraints are best kept in the re-ranking or rule layer.

### 10.7 Probability Calibration & Negative Downsampling Correction

If a model predicts 0.2, and 20% of samples in that predicted bin truly click, the score is fully calibrated.

#### 1. Why Calibration is Mission-Critical
- **Multi-Objective Score Fusion**: Linear blending $w_1 p_{\text{click}} + w_2 p_{\text{like}}$ assumes probabilities reflect an identical physical scale. If an architecture update inflates $p_{\text{click}}$ by 50%, the fusion balance is silently destroyed;
- **Ad Bidding (eCPM)**: Ads rank and charge by $\text{eCPM} = 1000 \times \text{pCTR} \times \text{pCVR} \times \text{bid}$. If predicted probabilities are inflated by 20%, advertiser budgets drain prematurely, triggering platform over-cost liabilities.

#### 2. Calibration Metric: E/O (Expected over Observed / COP)
```math
\operatorname{E/O} = \frac{\sum_{i=1}^N \hat{p}_i}{\sum_{i=1}^N y_i}
```
$\operatorname{E/O} > 1$ denotes systematic over-prediction; $\operatorname{E/O} < 1$ denotes under-prediction. Production systems enforce overall and slice-level $\operatorname{E/O} \in [0.98, 1.02]$. Standard calibration techniques include **Platt Scaling (logistic calibration)**, **Isotonic Regression**, and **Piecewise Binning**.

#### 3. Mathematical Derivation of Negative Downsampling Correction
To economize training storage and compute, industrial pipelines downsample unclicked negative impressions. If all positives are retained and negatives are retained with probability $\alpha \in (0, 1)$:
- True data odds: $\text{Odds} = \frac{N_+}{N_-}$;
- Sampled dataset negatives drop to $\alpha N_-$, yielding sampled odds:
  $$\text{Odds}_s = \frac{N_+}{\alpha N_-} = \frac{1}{\alpha} \text{Odds} \implies \text{Odds} = \alpha \cdot \text{Odds}_s$$
- With model prediction $p_s$ on the sampled data, $\text{Odds}_s = \frac{p_s}{1 - p_s}$. Substituting into true probability $p = \frac{\text{Odds}}{1 + \text{Odds}}$:

```math
p = \frac{\alpha \frac{p_s}{1 - p_s}}{1 + \alpha \frac{p_s}{1 - p_s}} = \frac{\alpha p_s}{1 - p_s + \alpha p_s}
```

Downsampling without this mathematical restoration inflates probabilities by an order of magnitude, destroying multi-objective blending and bidding integrity.

### 10.8 From Ranking Loss to Preference Optimization

BCE judges a single pair, BPR compares a pair of items, and InfoNCE makes one positive example compete with a set of candidates. All three utilize positive/negative feedback, but differ in comparison granularity and negative sample sources.

Generative recommendation extends the comparison unit to tokens or complete sequences. Next-token CE competes with the entire vocabulary, DPO compares chosen/rejected sequences, and policy gradient uses advantage to weight rollouts. Low-advantage RL rollouts cannot simply be treated as fixed negative samples because candidates are generated by the current policy, and sample weights change with training. For details, see [[BusinessAlgorithm05 Generative Recommendation.md#18.10 From Positive/Negative Samples to RL|Preference Optimization in Generative Recommendation]].

### 10.9 Chapter Self-Test

1. Where does negative transfer in Shared-Bottom come from?
2. How can MMoE gates be diagnosed?
3. Which two problems does ESMM solve for CVR?
4. Why does directly predicting watch seconds bias toward long videos?
5. When AUC is unchanged, why might calibration still improve online fusion?
6. How do the comparison granularities of BCE, BPR, and InfoNCE differ?

<details>
<summary>Reference answers</summary>

1. Shared parameters receive gradients from multiple tasks; gradients may conflict, and a high-volume task may dominate updates.
2. Inspect gate distributions, entropy, expert load, and specialization by task and slice. Constant or identical gates are warning signs.
3. ESMM models exposure-to-click and click-to-conversion jointly, reducing CVR sample-selection bias and sparsity from clicked-only training.
4. Watch-time scale grows with video length, so the model may learn length instead of satisfaction. Use completion rate, buckets, or normalization.
5. AUC only measures order. Fusion needs comparable probability scales, and calibration stops one head from dominating through numerical scale alone.
6. BCE classifies one sample, BPR compares one positive-negative pair, and InfoNCE compares the positive against a set of candidates.

</details>
