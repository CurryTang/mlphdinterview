# Ranking Objectives and Offline Evaluation

## Chapter 9: Ranking Objectives and Offline Evaluation

A ranker does not see the full corpus. It sees a small candidate set selected by upstream retrieval, filtering, and exposure policies. It learns "which item should come first among candidates produced by this system," not a global preference function independent of the candidate distribution. When retrieval changes, negative difficulty, category mix, and score distributions change with it; gains on the old test set may not survive.

This is why ranking comparisons should hold the candidate set fixed. On the same candidates, the difference isolates reordering quality. If a new experiment changes both retrieval and ranking and reports one NDCG number, the source of the gain is unknown. Both changes matter online, but they must be separated during diagnosis.

The meaning of the score also needs to be explicit. A CTR score may be calibrated as a click probability. A relevance score may preserve only grades or order. A final multi-objective score may be an uncalibrated business utility. Without that distinction, thresholds, fusion, and experiment conclusions become ambiguous.

### 9.1 Defining "Relevance" for Ranking

Search relevance is typically categorized into graded labels:

- Highly Relevant: Directly satisfies the primary intent of the query;
- Relevant: Solves the problem, but is incomplete or imprecise;
- Weakly Relevant: Covers only a portion of the intent;
- Irrelevant.

Real-world labeling is more complex. Queries may be polysemous, documents may be topically relevant but of poor quality, and timeliness can change the answer. Labeling guidelines should separate "topical relevance" from "whether it is ultimately worth displaying," otherwise, the model will conflate quality, authority, and relevance.

Recommendation labels are derived from behavior. Clicks, effective views, likes, follows, and purchases represent different intensities and have different latencies. Model objectives must align with product objectives; one should not train solely on CTR just because click samples are abundant.

### 9.2 Pointwise

Pointwise treats each candidate as an independent classification or regression sample:

```math
\mathcal L_{\text{point}}
=-\left[ y\log p+(1-y)\log(1-p) \right].
```

The advantage is that both sampling and training are simple, and predicted probabilities can be calibrated. The disadvantage is that it does not directly express the order between candidates under the same query/user.

CTR, CVR, and duration estimation often start with pointwise. If search relevance has graded labels, it can also be treated as multi-class classification or regression.

### 9.3 Pairwise

Pairwise constructs positive-negative candidate pairs, aiming for a higher score for the positive example:

```math
\mathcal L_{\text{pair}}
=-\log \sigma(s^+-s^-).
```

It is closer to "which item is ranked ahead of which," but the number of pairs can explode. How negative examples are selected significantly changes training: random negatives are simple, while hard negatives—those the current model ranks incorrectly—are more informative, though they are also more likely to contain labeling noise.

RankNet is a pairwise approach. LambdaRank/LambdaMART adjust gradient weights based on the impact of swapping two candidates on NDCG, ensuring that errors at the top of the list are prioritized.

### 9.4 Listwise

Listwise treats the entire candidate list as the training object. A simple form is to apply softmax to the list:

```math
P(i\mid q)=\frac{e^{s_i}}{\sum_j e^{s_j}},
```

Then use cross-entropy with the target distribution. One can also directly optimize an approximation of NDCG or generate candidate permutations.

Listwise is closer to the final task but requires candidates for the same query to be grouped for training, making memory usage, sampling, and implementation more complex. LLM ranking often uses listwise prompts, but long candidate lists encounter context window limits and position bias, which will be expanded upon in Chapter 18.

### 9.5 Search Relevance Models

Traditional text scores include:

- BM25;
- Query term coverage;
- Term proximity and order;
- Field matching (title, body, anchor text, etc.);
- Click and bounce statistics.

Cross-BERT encodes the query and document together, enabling the recognition of deep semantics and negation relationships. Its serving cost is high, so it is often placed in a later stage, processing only the top-N after retrieval.

Two-tower BERT can pre-calculate document representations, making it suitable for retrieval; Cross-BERT is suitable for re-ranking. When comparing the two, one must account for the number of calls: top-200 re-ranking means executing 200 query-document forward passes per query.

Search performs fusion at retrieval truncation, pre-ranking, and fine ranking; only the candidate count and model budget change. All three may consume relevance, CTR or engagement, content quality, freshness, geography, and aggregate features. Retrieval truncation favors two-tower and linear models, pre-ranking uses smaller cross-encoders or trees, and fine ranking can afford the heaviest interactions.

Early search systems often start with rules: bucket documents by relevance, then fuse click, quality, and freshness signals within a bucket. A clickbait failure or a sudden change in video share can be corrected directly. A learned fusion model becomes useful after labels accumulate, but the system should still monitor the relevance-grade mix in top results so that click gains cannot hide weaker relevance.

Fusion supervision can combine human overall-satisfaction labels with behavior. Annotators judge the query, document, time, and location, producing a grade that includes relevance, quality, and freshness. Clicks and engagement add personalized evidence. If labels are scarce, train a small teacher on non-personalized features, score a large log set, then combine its satisfaction estimate with behavior to train the online model that includes user features. The teacher must use point-in-time features, including document age at request time.

### 9.6 Industrial End-to-End Evaluation Framework for Recommendation & Search

In production recommendation systems (RecSys) and search engines, offline evaluation cannot rely on a single isolated metric (such as Global AUC or NDCG) to judge model performance. A production pipeline exhibits a hierarchical funnel structure: **Retrieval ➔ Pre-ranking (Coarse) ➔ Heavy Ranking (Fine) ➔ Re-ranking / Slate Optimization ➔ Calibration & Auction Bidding ➔ Online A/B Testing & Long-Term Ecosystem**. Each stage operates under distinct candidate set cardinalities, score distributions, strict latency SLAs (P99), and optimization targets, demanding stage-specific evaluation metrics and guardrail criteria.

#### 9.6.1 Comprehensive Pipeline Metric Matrix Across the RecSys Workflow

| Stage | Metric | Mathematical Formulation | What It Measures & Typical Use Cases | How to Read & Baseline Benchmarks | Key Focus & Pitfalls (Guardrails) |
|---|---|---|---|---|---|
| **① Retrieval**<br>(Two-Tower / Vector / Graph) | **Recall@K** | $\frac{|\mathcal{R}_K \cap \mathcal{G}^+|}{|\mathcal{G}^+|}$ | Proportion of true positive items retrieved as the candidate pool is pruned from millions to thousands. | $0 \sim 1$. $K$ typically set to 200~1000. Higher is better on identical candidate sets. | **Funnel Ceiling**: Positives missed by retrieval can never be recovered downstream; high Recall@K does not guarantee top positions. |
| | **HitRate@K (HR@K)** | $\mathbb{I}(|\mathcal{R}_K \cap \mathcal{G}^+| \ge 1)$ | Whether at least one relevant positive item is captured in top-K candidates. Suitable when a single hit defines success. | $0 \sim 1$. Crucial for full-screen immersive video feeds or single-card placements. | Binary indicator: ignores the total count or diversity of recovered positives; insensitive to multi-interest coverage. |
| | **MRR@K** | $\frac{1}{|Q|}\sum_{q}\frac{1}{\text{rank}_q^{(1)}}$ | Speed and rank of surfacing the **very first relevant item** in candidate generation. | $(0, 1]$. Rank 1 yields 1.0; rank 10 yields only 0.1. | Heavily punishes missing the top slot; completely indifferent to the rank of the 2nd and 3rd positives (ideal for QA, poor for feeds). |
| | **Catalog Coverage** | $\frac{|\bigcup_{u} \mathcal{R}_K(u)|}{|\mathcal{I}_{\text{total}}|}$ | Proportion of total catalog items activated across all users. Measures marketplace and long-tail health. | Percentage. Higher indicates cold-start and niche content receive exposure opportunities. | Must be evaluated alongside relevance/CTR; blindly maximizing coverage pollutes the candidate pool with low-quality noise. |
| **② Pre-ranking**<br>(Lightweight / Initial Filter) | **Top-M Rank Correlation** | Kendall's $\tau$ or Spearman's $\rho$ | Relative rank consistency between lightweight pre-ranker scores and full heavy ranker outputs. | $[-1, 1]$. Pre-ranking must maintain strong positive correlation with fine ranking ($\tau > 0.6$). | Constrained by extreme SLAs (P99 $\le 5\sim 10\text{ms}$); pre-ranking's primary role is **not discarding candidates that the heavy ranker values most**. |
| | **Recall@Top-M against Ranker** | $\frac{|\mathcal{C}_{\text{coarse\_M}} \cap \mathcal{C}_{\text{fine\_topN}}|}{N}$ | Fraction of the heavy ranker's Top-N items successfully preserved in the pre-ranker's Top-M output. | Quantifies **heavy ranker upper-bound truncation loss**. | Treats heavy ranker predictions as pseudo ground truth; if the heavy ranker is biased, the pre-ranker inherits and reinforces the bias. |
| **③ Heavy Ranking**<br>(CTR / CVR / Multi-Task) | **Request-GAUC** | $\frac{\sum_{r} w_r \cdot \operatorname{AUC}_r}{\sum_{r} w_r}$ | **The #1 Golden Offline Ranking Metric**: Probability that positives outrank negatives within the same request/viewport. | Computed per RequestID and weighted by impression count. $\Delta\text{GAUC} \ge +0.003$ (+0.3%) reliably drives online A/B gains. | **Eliminates cross-request/cross-user confounding and Simpson's Paradox**; requests with zero clicks (all negatives) are excluded. |
| | **User-GAUC** | $\frac{\sum_{u} w_u \cdot \operatorname{AUC}_u}{\sum_{u} w_u}$ | Evaluates whether positives outrank negatives across historical impressions for the same user. | Computed per UserID and weighted by impressions. Typically higher than Request-GAUC. | Vulnerable to diurnal intent shifts (workday vs. evening relaxation), where negatives from different sessions are conflated. |
| | **Global ROC-AUC** | $\frac{\sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \mathbb{I}(p_i > p_j)}{|\mathcal{D}^+| \cdot |\mathcal{D}^-|}$ | Broad discrimination power across all pooled positive and negative instances. | 0.5 is random guessing; 1.0 is perfect. Useful as a baseline sanity check. | **False Prosperity Trap**: Easily inflated by heavy power users or viral items; cross-user comparisons introduce Simpson's Paradox. |
| | **PR-AUC (Average Precision)** | $\sum_{k} (R_k - R_{k-1})P_k$ | **Mandatory gold standard for extremely sparse conversions (CVR, high-value purchases, fraud/abuse triage)**. | Area under Precision-Recall curve. **Must be benchmarked against positive prevalence $\pi = P(Y=1)$**, not 0.5. | Immune to vast True Negatives diluting FPR; however, absolute values are not directly comparable across datasets with different priors. |
| | **LogLoss (BCE)** | $-\frac{1}{N}\sum [y\log p + (1-y)\log(1-p)]$ | Evaluates absolute probability calibration; heavily penalizes overconfident errors. | Lower is better. Compared as relative reduction over baseline models. | Sensitive to seasonal swings in baseline CTR; under negative subsampling, importance weighting is mandatory to prevent distortion. |
| | **NE (Normalized Cross Entropy / RIG)** | $\frac{\operatorname{LogLoss}(p, y)}{H(y_{\text{base}})}$, $\text{RIG}=1-\text{NE}$ | Industrial CTR benchmark (Meta, TikTok). Measures entropy reduction relative to baseline constant CTR entropy. | Lower NE is better; higher RIG is better (a 1% RIG lift represents a major breakthrough). | **Robust against background CTR shifts**: unaffected by promotional campaigns or diurnal fluctuations; enables fair comparison across splits. |
| **④ Re-ranking**<br>(Whole-Slate Optimization) | **NDCG@K** | $\frac{\operatorname{DCG@K}}{\operatorname{IDCG@K}}$ | Evaluates multi-grade relevance and top-heavy list quality with **logarithmic position discounting**. | $0 \sim 1$. $K$ must strictly match the physical device viewport (e.g., $K=4 \sim 6$ for mobile double-column feeds). | Insensitive to misrankings at the tail of the slate; assumes independent item utilities without modeling item substitutability. |
| | **ILD (Intra-List Diversity)** | $\frac{2}{K(K-1)}\sum_{i < j} \operatorname{dist}(i, j)$ | Semantic and topical diversity across items within the same recommended slate; combats filter bubbles. | Mean pairwise embedding or category distance. | Diversity trades off with immediate CTR (Pareto frontier); excessive diversity disrupts user immersion. |
| | **Novelty / Serendipity** | $\frac{1}{K}\sum_{i=1}^K -\log_2 P(item_i)$ | Self-information of recommendations: penalizes trivial popularity bias; rewards discovering relevant niche content. | Higher self-information indicates greater novelty. | Must be bounded by user relevance; recommending bizarre or low-quality obscure items damages engagement. |
| **⑤ Calibration & Bidding**<br>(Ad Tech / Monetization) | **PCOC** | $\frac{\sum_{i} \hat{p}_i}{\sum_{i} y_i}$ | Ratio of sum of predicted probabilities to sum of observed conversions (Predictive-over-Observed Click/Conversion Ratio). | **Perfect calibration is 1.000**. Values like 1.10 indicate 10% global overestimation; 0.90 indicates 10% underestimation. | **The lifeblood of auction bidding**: Overestimation burns advertiser budgets prematurely; underestimation loses winnable auctions. High AUC does not imply PCOC $\approx 1$. |
| | **ECE (Expected Calibration Error)** | $\sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ | Weighted mean absolute difference between predicted confidence and observed empirical frequency across binned intervals. | Lower is better (0.0 is ideal). | Monotonic scaling (e.g., temperature scaling) dramatically improves ECE without affecting AUC or NDCG. |
| | **Reliability Diagram** | Binned calibration curve | Visualizes empirical event frequency vs. mean predicted probability across equal-width or quantile bins. | Perfect alignment follows the $y=x$ diagonal; curves above denote underestimation; curves below denote overestimation. | High-confidence bins often have tiny sample sizes, causing visual variance; evaluate alongside bin histogram. |
| **⑥ Online A/B & Ecosystem**<br>(Ground Truth Value) | **North Star Business Metrics** | DAU, MAU, Dwell Time per User, D7/D30 Retention, GMV, eCPM. | Ultimate causal impact of algorithmic modifications on platform monetization and user retention. | Statistical significance ($p < 0.05$ with Power $\ge 80\%$). Must run 7~14 full days to cancel novelty and day-of-week effects. | Long feedback loop and high variance; requires CUPED variance reduction and holdout validation. |
| | **Engagement & Conversion Proxies** | CTR, Long-Play Rate (VTR > 5s), Completion Rate, Social Interactions (like, share, save). | Fast-moving leading indicators of immediate user satisfaction. | Monitored across conversion funnels. | **Clickbait Trap**: Surging CTR accompanied by plunging dwell time or 7-day retention indicates deceptive clickbait recommendation. |
| | **Guardrails** | Negative feedback rate (dislike, report, hide), P99 inference latency, timeout fallback rate. | Non-negotiable protective thresholds for system reliability and ecosystem safety. | **Zero Regression Rule**: Guardrail metrics must not deteriorate with statistical confidence. | Algorithmic gains achieved by degrading latency SLAs or increasing user annoyance are rejected. |
| | **SRM (Sample Ratio Mismatch)** | $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ | Validates whether experimental traffic allocation strictly obeys configured assignment ratios. | Chi-squared test $p < 0.001$ triggers an SRM alert. | **Foundational Validity Check**: When SRM occurs, traffic is contaminated by drops, timeouts, or leakage; **all experiment conclusions are null and void**. |

#### 9.6.2 Four In-Depth Interview Themes in Offline Evaluation

##### 1. Why Does Offline Global ROC-AUC Increase While Online CTR Declines? (Simpson's Paradox & Request-GAUC)
- **Confounding Mechanism**: Global ROC-AUC pools positive/negative comparisons across disparate users.
  - Active User A has an organic click rate of 40%; the model assigns scores in $[0.6, 0.9]$.
  - Inactive User B has an organic click rate of 2%; the model assigns scores in $[0.1, 0.3]$.
  - When pooling all samples, User A's positive interactions dominate User B's negative impressions. A model that simply learns "User A's features get high scores, User B's get low scores" will attain a **stellar Global ROC-AUC of 0.85+ without learning any intra-user item preferences**.
- **Serving Reality**: When User A opens the app, the ranker must order 10 candidates for that specific viewport. If intra-user ranking is random ($\text{AUC}_A = 0.50$), User A sees irrelevant items at rank 1 and bounces immediately.
- **Interview Key Takeaway**:
  > Ranking models must be evaluated with **Request-GAUC** (or User-GAUC), because items compete strictly within a single impression slate. Request-GAUC conditions on the request context, eliminating cross-user confounding.

##### 2. Why Is PR-AUC Mandatory for CVR and Fraud Detection While ROC-AUC Fails?
- **FPR Denominator Dilution**:
  $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}, \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  Under extreme skew (e.g., $1:10000$ purchase conversion or financial fraud):
  - True Negatives ($\text{TN} \approx 10^6$) dominate the denominator. Even if the model produces 5,000 false alarms for only 100 true conversions ($\text{FP} = 5000, \text{TP} = 100$), $\text{FPR} \approx \frac{5000}{1000000} = 0.005$ remains near zero!
  - The ROC curve hugs the top-left corner, displaying an **inflated ROC-AUC of 0.98+**.
  - However, operational precision is $\text{Precision} = \frac{100}{5100} \approx 1.96\%$. Ninety-eight percent of recommendations or fraud alerts are incorrect, overwhelming operational capacity.
- **The Baseline Difference**:
  - The random baseline for ROC-AUC is unconditionally **0.5**.
  - The random baseline for PR-AUC is positive prevalence $\pi = P(Y=1)$. For a $0.1\%$ conversion rate, the random PR-AUC is **0.001**. A model reaching PR-AUC = 0.15 delivers a 150x signal-to-noise amplification over random guessing.

##### 3. Why Do Meta and TikTok Prioritize NE (Normalized Cross Entropy / RIG) over Raw LogLoss?
- **Formula**:
  $$\text{NE} = \frac{\operatorname{LogLoss}(p, y)}{- \bar{p}\log\bar{p} - (1-\bar{p})\log(1-\bar{p})}, \quad \text{RIG} = 1 - \text{NE}$$
  Where the denominator is the **background Shannon entropy $H(\bar{p})$** of the empirical conversion rate $\bar{p}$.
- **Why Eliminate Background Entropy?**
  - Traffic surges during promotional campaigns shift baseline CTR from 3% to 7%. The intrinsic entropy of sample labels changes, causing absolute LogLoss to fluctuate wildly even if model parameters remain identical.
  - Engineers cannot distinguish whether a LogLoss change is caused by model quality or seasonal traffic shifts.
  - **NE measures the proportion of uncertainty compressed beyond a naive constant prior estimator**. NE is invariant to baseline conversion shifts, enabling fair comparisons across time and slices.

##### 4. Decoupling Ranking Discrimination (AUC) from Probability Calibration (PCOC / ECE)
- **Impact of Monotonic Transformations**:
  - Applying any strictly increasing transformation to predictions (e.g., $g(s) = s^2$ or $g(s) = \sigma(10s)$) leaves relative order unchanged; **AUC and NDCG@K remain 100% invariant**.
  - However, physical probability values become distorted: **PCOC deviates sharply (e.g., surging to 2.5), and ECE deteriorates**.
- **Catastrophic Impact on Auction Bidding**:
  - In advertising, bids are computed as $\text{Bid}_{\text{actual}} = p\text{CTR} \times p\text{CVR} \times \text{Bid}_{\text{target}}$.
  - If a model's GAUC rises from 0.75 to 0.78 while PCOC degrades to 1.30 (30% overestimation), the auction engine submits inflated bids. Advertiser budgets burn out within minutes without delivering target conversions, triggering customer complaints and budget refunds.
- **Production Decoupling Architecture**:
  - Keep the heavy ranking backbone focused on maximizing GAUC;
  - Feed raw ranking logits into an independent, monotonic **post-calibration layer** (e.g., 100-bin Isotonic Regression, Platt Scaling, or piecewise polynomial calibrators) to align predictions with physical click probabilities.

##### 5. Invariance of AUC vs. Odds Ratio Correction under Negative Subsampling
Industrial rankers frequently subsample unclicked negative impressions at rate $w \in (0, 1)$ (e.g., $w=0.1$):
- **AUC Invariance**: AUC reflects the probability that a positive outranks a random negative. Uniform random subsampling preserves the empirical rank distribution of negatives; the expected pairwise rank order remains unchanged. Hence, **AUC under negative subsampling is an asymptotically unbiased estimator of true AUC**.
- **Probability Distortion and Odds Ratio Inversion**:
  - Subsampling inflates the effective positive ratio by $\frac{1}{w}$. Raw model outputs $p_{\text{sampled}}$ are systematically overconfident.
  - To recover true physical probabilities $p_{\text{real}}$ for auction bidding, apply the **Odds Ratio correction**:
    $$\text{Odds}_{\text{real}} = \text{Odds}_{\text{sampled}} \cdot w \implies \frac{p_{\text{real}}}{1 - p_{\text{real}}} = \frac{p_{\text{sampled}}}{1 - p_{\text{sampled}}} \cdot w$$
    Yielding:
    $$p_{\text{real}} = \frac{p_{\text{sampled}}}{p_{\text{sampled}} + \frac{1 - p_{\text{sampled}}}{w}}$$
  - Alternatively, weight retained negatives by $\frac{1}{w}$ in the loss function during training.

### 9.7 Misalignment Between Training and Evaluation

Common misalignments:

- Training CTR with BCE but evaluating ranking with NDCG;
- Training on exposure logs but testing on manually constructed candidate sets;
- Training candidates coming from an old model, while online candidates come from new retrieval;
- Search labeling only looking at relevance, while online ranking mixes in quality and timeliness;
- Offline recommendation keeping only one positive example, while the system actually needs to generate a multi-interest list.

Proxy objectives can differ from final metrics, but experimental records must account for this gap. When candidate sets, label definitions, or traffic distributions change, offline gains cannot be directly extrapolated to online performance.

### Quick Coding: NDCG@K

Input graded relevance sorted by predicted order to calculate DCG, IDCG, and NDCG. Boundary conditions include `k <= 0`, lists shorter than `k`, and cases where all relevance values are zero.

Implementation:

```python
def ndcg_at_k(relevances, k):
    ...
```

`relevances` are already sorted by the model's predicted order, where each value is a non-negative relevance grade. Use:

```math
DCG@K=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}.
```

Return `0.0` if there are no relevant results, or if `k <= 0`.

<details>
<summary>Reference answer</summary>

```python
from math import log2


def ndcg_at_k(relevances, k):
    if k <= 0:
        return 0.0

    all_values = list(relevances)

    def dcg(items):
        return sum(
            (2 ** rel - 1) / log2(rank + 1)
            for rank, rel in enumerate(items, start=1)
        )

    actual = dcg(all_values[:k])
    ideal = dcg(sorted(all_values, reverse=True)[:k])
    return 0.0 if ideal == 0 else actual / ideal
```

Sorting the ideal list costs `O(n log n)`. A small bounded relevance scale allows a linear-time counting approach.

</details>

### Quick Coding: Computing GAUC (Group AUC)

Calculate impression-weighted Group AUC from offline logs, handling boundary users with all-positive or all-negative labels.

```python
def calculate_gauc(user_ids, labels, preds):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import defaultdict


def calculate_gauc(user_ids, labels, preds):
    # 1. Group (label, pred) by user
    user_data = defaultdict(lambda: ([], []))
    for u, y, p in zip(user_ids, labels, preds):
        user_data[u][0].append(y)
        user_data[u][1].append(p)

    total_weight = 0
    weighted_auc_sum = 0.0

    for u, (u_labels, u_preds) in user_data.items():
        n_pos = sum(u_labels)
        n_neg = len(u_labels) - n_pos

        # Boundary condition: if user has only positives or only negatives, AUC is undefined
        if n_pos == 0 or n_neg == 0:
            continue

        # Fast intra-user AUC via Wilcoxon-Mann-Whitney rank sum
        ranked = sorted(zip(u_preds, u_labels), key=lambda x: x[0])
        rank_sum = 0
        for rank, (_, y) in enumerate(ranked, start=1):
            if y == 1:
                rank_sum += rank

        auc_u = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

        weight = len(u_labels)  # Weighted by impression volume
        weighted_auc_sum += auc_u * weight
        total_weight += weight

    return weighted_auc_sum / total_weight if total_weight > 0 else 0.5
```

Time complexity is $\sum O(N_u \log N_u) \le O(N \log N)$; auxiliary space is $O(N)$.

</details>

### 9.8 Chapter Self-Test

1. Pointwise scores can be calibrated; why might ranking still be poor?
2. Why is GAUC mandatory in industrial recsys instead of relying solely on Global AUC?
3. How should hard negatives for Pairwise be generated?
4. Why is NDCG more sensitive to the top of the list?
5. Where should Cross-BERT and two-tower BERT be placed?
6. How should relevance, content quality, and final ranking scores be decoupled?
7. Why does search fusion often start with relevance buckets and hand-written rules before a learned fusion model?
8. Why do industry rankers evaluate CTR models with NE (Normalized Cross Entropy / RIG) rather than raw LogLoss?
9. In extremely sparse CVR (e.g., e-commerce orders) or rare fraud triage, why does ROC-AUC exhibit "false prosperity", and what should be used instead?
10. If offline training applies a 10% negative subsampling rate ($w=0.1$), what mathematical transformation is mandatory before sending predicted probabilities to the downstream eCPM bidding engine?

<details>
<summary>Reference answers</summary>

1. Calibration aligns probabilities with empirical frequency; it does not guarantee the relative order of nearby candidates.
2. Global AUC pools positive/negative comparisons across disparate users. Highly active users have high baseline CTRs, so a model memorizing user priors yields high global AUC while intra-user rankings remain random. GAUC computes AUC strictly within each user's impressions, isolating true personalization quality.
3. Use exposed candidates from the same request that an old model ranked highly but received negative feedback, or sample top ANN/BM25 results. Filter false negatives and immature labels.
4. Logarithmic discount gives top positions more weight, so the same displacement costs more near rank one than near the tail.
5. Two-tower BERT fits retrieval or coarse ranking. Cross-BERT jointly encodes query and document and belongs in fine ranking over a small set.
6. Produce separate relevance, quality, and business-objective scores, calibrate them, combine them by scenario, and keep interpretable hard guardrails.
7. Rules are easier to inspect and repair while data and the pipeline are still changing. A learned model needs trustworthy satisfaction and behavior labels, plus relevance-grade monitoring so that it cannot trade relevance for clicks.
8. Raw LogLoss fluctuates with overall natural CTR drift (promotions, holidays, diurnal shifts), making it impossible to separate model degeneration from traffic composition changes. NE divides LogLoss by the background Shannon entropy, measuring the proportion of uncertainty compressed relative to a naive constant prior, ensuring robust comparability across traffic splits and time windows.
9. Under extreme class imbalance, the vast True Negative count ($\text{TN}$) inflates the denominator of $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$. Even when thousands of false positives overwhelm true conversions, FPR remains near zero and ROC-AUC is artificially inflated (0.98+). Practitioners must use **PR-AUC (Average Precision)** and evaluate it against the baseline positive prevalence $\pi = P(Y=1)$.
10. Uniform negative subsampling preserves AUC ranking order, but shatters physical probability calibration, causing raw scores $p_{\text{sampled}}$ to be severely inflated. Before feeding scores to auction bidding, calibrate back to true CTR via the odds-ratio inversion formula: $p_{\text{real}} = \frac{p_{\text{sampled}}}{p_{\text{sampled}} + \frac{1 - p_{\text{sampled}}}{w}}$, or train with weighted cross-entropy using $\frac{1}{w}$ on negative samples.

</details>
