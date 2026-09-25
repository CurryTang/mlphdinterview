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

$$
\mathcal{L}_{\text{point}} = -\left[ y\log p + (1-y)\log(1-p) \right]
$$

The advantage is that both sampling and training are simple, and predicted probabilities can be calibrated. The disadvantage is that it does not directly express the order between candidates under the same query/user.

CTR, CVR, and duration estimation often start with pointwise. If search relevance has graded labels, it can also be treated as multi-class classification or regression.

### 9.3 Pairwise

Pairwise constructs positive-negative candidate pairs, aiming for a higher score for the positive example:

$$
\mathcal{L}_{\text{pair}} = -\log \sigma(s^+ - s^-)
$$

It is closer to "which item is ranked ahead of which," but the number of pairs can explode. How negative examples are selected significantly changes training: random negatives are simple, while hard negatives—those the current model ranks incorrectly—are more informative, though they are also more likely to contain labeling noise.

RankNet is a pairwise approach. LambdaRank/LambdaMART adjust gradient weights based on the impact of swapping two candidates on NDCG, ensuring that errors at the top of the list are prioritized.

### 9.4 Listwise

Listwise treats the entire candidate list as the training object. A simple form is to apply softmax to the list:

$$
P(i \mid q) = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

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

| Stage | Core Metric | Primary Optimization Goal & Scenario | Expected Baseline & Reading Convention | Key Focus & Guardrail Pitfalls |
|---|---|---|---|---|
| **① Retrieval**<br>(Candidate Generation/Vector Two-Tower) | **Recall@K** | Proportion of true positive items retrieved from catalog pruning. | $0 \sim 1$, $K \in [200, 1000]$. Higher is better on identical sets. | **Funnel Ceiling**: Positives missed here are lost forever; does not ensure top slots. |
| | **HitRate@K** | Whether at least one positive is captured in Top-$K$. Suitable when 1 hit defines success. | $0 \sim 1$. Critical for single-card placements and full-screen video feeds. | Binary metric; insensitive to the quantity or diversity of captured positives. |
| | **MRR@K** | Rank position and speed of retrieving the **very first relevant item**. | $(0, 1]$. Rank 1 yields 1.0; rank 10 yields only 0.1. | Heavily penalizes missing top slot; ignores the ranks of 2nd and 3rd positives. |
| | **Catalog Coverage** | Proportion of catalog items activated across all user recommendations. | Percentage. Higher indicates strong long-tail and niche discovery. | Must evaluate with CTR; blindly maximizing coverage injects low-quality noise. |
| **② Pre-ranking**<br>(Lightweight / Filtering) | **Kendall's $\tau$ / Spearman's $\rho$** | Relative rank consistency between lightweight pre-ranker and heavy ranker. | $[-1, 1]$. Must maintain high positive correlation ($\tau > 0.6$). | Constrained by P99 $\le 5\text{ms}$; core goal is **not pruning heavy ranker top items**. |
| | **Recall@Top-M against Ranker** | Proportion of heavy ranker's Top-$N$ items retained in pre-ranker Top-$M$. | Measures **heavy ranker upper-bound truncation loss**. | Treats heavy ranker as pseudo-truth; inherits heavy ranker biases. |
| **③ Heavy Ranking**<br>(CTR / CVR / Multi-Task) | **Request-GAUC** | **The #1 Golden Ranking Metric**: Probability positives outrank negatives per request. | Computed per RequestID and impression-weighted. $\Delta\text{GAUC} \ge +0.003$ significant. | **Eliminates cross-request prior bias & Simpson's Paradox**; excludes zero-click slates. |
| | **User-GAUC** | Ability to rank positives above negatives across historical user impressions. | Computed per UserID and impression-weighted. Typically higher than Request-GAUC. | Vulnerable to diurnal intent shifts (work vs. leisure), conflating different contexts. |
| | **Global ROC-AUC** | Coarse baseline measuring global discrimination across pooled pairs. | 0.5 is random; 1.0 is perfect. Used as a baseline defense guardrail. | **False Prosperity Trap**: Easily inflated by power users or viral items; Simpson's paradox. |
| | **PR-AUC (AP)** | **Mandatory gold standard for extremely sparse events (CVR, fraud detection)**. | Area under PR curve. **Must benchmark against positive prevalence $\pi = P(Y=1)$**. | Immune to vast True Negatives diluting FPR; absolute values not comparable across priors. |
| | **LogLoss** | Evaluates absolute probability calibration; heavily penalizes confident errors. | Lower is better. Compared as relative reduction over baseline models. | Sensitive to seasonal swings in baseline CTR; requires importance weighting under sampling. |
| | **NE / RIG** | Industrial CTR benchmark (Meta, TikTok). Measures entropy reduction over baseline CTR. | Lower NE is better; higher RIG is better (a 1% RIG lift is a major breakthrough). | **Robust against background CTR shifts**: unaffected by promotional campaigns or seasonality. |
| **④ Re-ranking**<br>(Whole-Slate / Presentation) | **NDCG@K** | Multi-grade relevance and top-heavy list quality with **logarithmic discounting**. | $0 \sim 1$. $K$ must strictly match physical device viewport (e.g., $K=4 \sim 6$). | Insensitive to tail misrankings; assumes independent utilities without substitutability. |
| | **ILD (Intra-List Diversity)** | Semantic and topical diversity across items within the same recommended slate. | Mean pairwise embedding or category distance. | Diversity trades off with immediate CTR; excessive diversity disrupts immersion. |
| | **Novelty / Serendipity** | Self-information of recommendations: penalizes trivial popularity bias. | Higher self-information indicates greater novelty. | Must be bounded by user relevance; recommending bizarre niche items damages retention. |
| **⑤ Calibration & Bidding**<br>(Ad Tech / Monetization) | **PCOC** | Ratio of sum of predicted probabilities to observed conversions (over/under estimation). | **Perfect calibration is 1.000**. 1.10 = 10% overestimation; 0.90 = 10% underestimation. | **Lifeblood of auction bidding**: Overestimation burns budgets prematurely; underestimation loses bids. |
| | **ECE** | Weighted mean absolute difference between confidence and frequency across bins. | Lower is better (0.0 is ideal). | Monotonic scaling (temperature scaling) improves ECE without changing AUC. |
| | **Brier Score** | Mean squared error between predicted probabilities and binary labels. | Lower is better. Decomposes orthogonally into Reliability, Resolution, Uncertainty. | Reflects ranking and calibration simultaneously, but less intuitive than PCOC. |
| **⑥ Online A/B & Ecosystem**<br>(Ground Truth Value) | **North Star Business Metrics** | DAU, MAU, Dwell Time per User, D7/D30 Retention, GMV, eCPM. | Ultimate causal impact of algorithmic modifications on business and user retention. | Requires 7~14 full days to cancel day-of-week seasonality; paired with CUPED. |
| | **Engagement & Guardrails** | CTR, Completion Rate, **Negative feedback rate (dislike/report, strictly zero regression)**. | Leading indicators and system defensive red lines. | **Clickbait Trap**: High CTR with plunging dwell time indicates deceptive content. |
| | **SRM (Sample Ratio Mismatch)** | Validates whether experimental traffic allocation strictly obeys configured ratios. | Chi-squared test $p < 0.001$ triggers an SRM alert. | **Foundational Validity Check**: If SRM triggers, traffic is contaminated; all metrics void. |

---

#### 9.6.2 Mathematical Formulations Across the Full RecSys Lifecycle

##### 1. Candidate Retrieval Formulations

- **Recall@K**:
  $$
  \operatorname{Recall@K} = \frac{\left| \mathcal{R}_K \cap \mathcal{G}^+ \right|}{\left| \mathcal{G}^+ \right|}
  $$
  Where $\mathcal{R}_K$ denotes the set of Top-$K$ items retrieved, and $\mathcal{G}^+$ is the ground-truth set of positive feedback items for the user.
- **HitRate@K (HR@K)**:
  $$
  \operatorname{HR@K} = \mathbb{I}\left( \left| \mathcal{R}_K \cap \mathcal{G}^+ \right| \ge 1 \right)
  $$
  Where $\mathbb{I}(\cdot)$ is the indicator function, returning 1 if at least one positive item is present in Top-$K$, and 0 otherwise.
- **MRR@K (Mean Reciprocal Rank)**:
  $$
  \operatorname{MRR@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\operatorname{rank}_q^{(1)}}
  $$
  Where $\operatorname{rank}_q^{(1)}$ is the position of the first hit positive candidate for query/request $q$; if no positive item appears in Top-$K$, the reciprocal rank is 0.
- **Catalog Coverage**:
  $$
  \operatorname{Coverage} = \frac{\left| \bigcup_{u \in \mathcal{U}} \mathcal{R}_K(u) \right|}{\left| \mathcal{I}_{\text{total}} \right|}
  $$
  Measuring the union of distinct items recommended across all users $\mathcal{U}$ as a fraction of the total catalog $\mathcal{I}_{\text{total}}$.

##### 2. Pre-ranking & Filtering Formulations

- **Kendall's $\tau$ Rank Correlation (Pre-ranking & Fine-ranking Consistency)**:
  $$
  \tau = \frac{P - Q}{\frac{1}{2} M (M - 1)}
  $$
  Where $M$ is the candidate volume scored by the pre-ranker, $P$ is the number of concordant candidate pairs, and $Q$ is the number of discordant pairs between coarse and fine model scores.
- **Truncation Recall (Recall@Top-M against Heavy Ranker)**:
  $$
  \operatorname{Recall@Top\text{-}M}_{\text{coarse}} = \frac{\left| \mathcal{C}_{\text{coarse\_M}} \cap \mathcal{C}_{\text{fine\_topN}} \right|}{N}
  $$
  Measuring the fraction of the heavy ranker's Top-$N$ preferred candidates successfully retained within the pre-ranker's Top-$M$ output.

##### 3. Heavy Ranking & Scoring Formulations

- **Request-GAUC (Within-Request Impression Grouped AUC)**:
  $$
  \operatorname{Request-GAUC} = \frac{\sum_{r \in \mathcal{R}, \, n_r^+ > 0, \, n_r^- > 0} w_r \cdot \operatorname{AUC}_r}{\sum_{r \in \mathcal{R}, \, n_r^+ > 0, \, n_r^- > 0} w_r}
  $$
  Where $\operatorname{AUC}_r$ is computed strictly over candidates exposed in request $r$, weighted by impression count $w_r = n_r$. Slates with all clicks or zero clicks have no positive-negative pairs and are filtered out.
- **User-GAUC (User Grouped AUC)**:
  $$
  \operatorname{User-GAUC} = \frac{\sum_{u \in \mathcal{U}, \, n_u^+ > 0, \, n_u^- > 0} w_u \cdot \operatorname{AUC}_u}{\sum_{u \in \mathcal{U}, \, n_u^+ > 0, \, n_u^- > 0} w_u}
  $$
  Where $\operatorname{AUC}_u$ is computed over historical impressions served to user $u$, weighted by $w_u = n_u$.
- **Global ROC-AUC**:
  $$
  \operatorname{AUC}_{\text{global}} = \frac{1}{\left| \mathcal{D}^+ \right| \cdot \left| \mathcal{D}^- \right|} \sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \left( \mathbb{I}(p_i > p_j) + \frac{1}{2}\mathbb{I}(p_i = p_j) \right)
  $$
  Evaluating pairwise discrimination pooled across the entire test set.
- **PR-AUC (Precision-Recall AUC / Average Precision)**:
  $$
  \operatorname{PR-AUC} = \sum_{k=1}^N \left( \operatorname{Recall}_k - \operatorname{Recall}_{k-1} \right) \cdot \operatorname{Precision}_k
  $$
  Integrating precision over recall step-increments, focusing exclusively on rare positive detection quality.
- **LogLoss (Binary Cross-Entropy Loss)**:
  $$
  \operatorname{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
  $$
- **NE (Normalized Cross Entropy) & RIG (Relative Information Gain)**:
  $$
  \operatorname{NE} = \frac{\operatorname{LogLoss}(p, y)}{H(\bar{p})} = \frac{-\frac{1}{N}\sum_{i=1}^N \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right]}{-\bar{p}\log \bar{p} - (1-\bar{p})\log(1-\bar{p})}
  $$
  $$
  \operatorname{RIG} = 1 - \operatorname{NE}
  $$
  Where $\bar{p} = \frac{1}{N}\sum y_i$ is the empirical background positive prevalence and $H(\bar{p})$ is the background Shannon entropy.

##### 4. Re-ranking & Slate Optimization Formulations

- **NDCG@K (Normalized Discounted Cumulative Gain)**:
  $$
  \operatorname{DCG@K} = \sum_{i=1}^K \frac{2^{\operatorname{rel}_i} - 1}{\log_2(i + 1)}, \quad \operatorname{NDCG@K} = \frac{\operatorname{DCG@K}}{\operatorname{IDCG@K}}
  $$
  Where $\operatorname{rel}_i$ is the relevance grade at rank $i$, and $\operatorname{IDCG@K}$ is the ideal DCG obtained by sorting items in perfect descending relevance order.
- **ILD (Intra-List Diversity)**:
  $$
  \operatorname{ILD} = \frac{2}{K(K - 1)} \sum_{i=1}^K \sum_{j=i+1}^K \operatorname{dist}(\text{item}_i, \text{item}_j)
  $$
  Where $\operatorname{dist}(i, j) = 1 - \cos(\mathbf{e}_i, \mathbf{e}_j)$ is the cosine distance between item embeddings or category indicators.
- **Novelty (Self-Information)**:
  $$
  \operatorname{Novelty} = \frac{1}{K} \sum_{i=1}^K -\log_2 P(\text{item}_i)
  $$
  Where $P(\text{item}_i)$ is the item's historical marginal exposure probability across the platform.

##### 5. Calibration & Auction Bidding Formulations

- **PCOC (Predictive-over-Observed Ratio)**:
  $$
  \operatorname{PCOC} = \frac{\sum_{i=1}^N \hat{p}_i}{\sum_{i=1}^N y_i} \quad (\text{Perfect Calibration } = 1.000)
  $$
- **ECE (Expected Calibration Error)**:
  $$
  \operatorname{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right|
  $$
  Partitioning predictions into $M$ bins $B_m$, where $\operatorname{conf}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m}\hat{p}_i$ is the mean predicted confidence, and $\operatorname{acc}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m} y_i$ is empirical event accuracy.
- **Brier Score Decomposition**:
  $$
  \operatorname{BS} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2 = \operatorname{Reliability} - \operatorname{Resolution} + \operatorname{Uncertainty}
  $$
  Where $\operatorname{Reliability} = \sum_m \frac{|B_m|}{N}(\operatorname{conf}(B_m) - \operatorname{acc}(B_m))^2$, $\operatorname{Resolution} = \sum_m \frac{|B_m|}{N}(\operatorname{acc}(B_m) - \bar{y})^2$, and $\operatorname{Uncertainty} = \bar{y}(1 - \bar{y})$.
- **Odds Ratio Inversion Formula under Negative Subsampling**:
  When unclicked impressions are downsampled at rate $w \in (0, 1)$, raw model scores $p_{\text{sampled}}$ are mapped back to physical probabilities via:
  $$
  \operatorname{Odds}_{\text{real}} = \operatorname{Odds}_{\text{sampled}} \cdot w \implies \frac{p_{\text{real}}}{1 - p_{\text{real}}} = \frac{p_{\text{sampled}}}{1 - p_{\text{sampled}}} \cdot w
  $$
  $$
  p_{\text{real}} = \frac{p_{\text{sampled}}}{p_{\text{sampled}} + \frac{1 - p_{\text{sampled}}}{w}}
  $$

##### 6. Online Experimentation & Causal Inference Formulations

- **SRM Chi-Square Goodness-of-Fit Test**:
  $$
  \chi^2 = \sum_{k=1}^C \frac{(O_k - E_k)^2}{E_k}, \quad \text{Degrees of Freedom } df = C - 1
  $$
  Where $O_k$ is the observed traffic allocation count in bucket $k$, and $E_k$ is the expected allocation based on hash configuration.
- **CUPED Variance Reduction Estimator**:
  $$
  \hat{Y}_{\text{CUPED}} = \bar{Y} - \theta (\bar{X} - \mathbb{E}[X]), \quad \text{Optimal Coefficient } \theta^* = \frac{\operatorname{Cov}(Y, X)}{\operatorname{Var}(X)}
  $$
  $$
  \operatorname{Var}\left( \hat{Y}_{\text{CUPED}} \right) = \operatorname{Var}(\bar{Y}) \cdot \left( 1 - \rho_{XY}^2 \right)
  $$
  Using pre-experiment user history $X$ (orthogonal to assignment) to absorb variance and shrink required sample sizes by $50\%\sim 80\%$.

---

#### 9.6.3 Five In-Depth Interview Themes in Offline Evaluation

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
  $$
  \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}, \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$
  Under extreme skew (e.g., $1:10000$ purchase conversion or financial fraud):
  - True Negatives ($\text{TN} \approx 10^6$) dominate the denominator. Even if the model produces 5,000 false alarms for only 100 true conversions ($\text{FP} = 5000, \text{TP} = 100$), $\text{FPR} \approx \frac{5000}{1000000} = 0.005$ remains near zero!
  - The ROC curve hugs the top-left corner, displaying an **inflated ROC-AUC of 0.98+**.
  - However, operational precision is $\text{Precision} = \frac{100}{5100} \approx 1.96\%$. Ninety-eight percent of recommendations or fraud alerts are incorrect, overwhelming operational capacity.
- **The Baseline Difference**:
  - The random baseline for ROC-AUC is unconditionally **0.5**.
  - The random baseline for PR-AUC is positive prevalence $\pi = P(Y=1)$. For a $0.1\%$ conversion rate, the random PR-AUC is **0.001**. A model reaching PR-AUC = 0.15 delivers a 150x signal-to-noise amplification over random guessing.

##### 3. Why Do Meta and TikTok Prioritize NE (Normalized Cross Entropy / RIG) over Raw LogLoss?
- **Formula**:
  $$
  \text{NE} = \frac{\operatorname{LogLoss}(p, y)}{- \bar{p}\log\bar{p} - (1-\bar{p})\log(1-\bar{p})}, \quad \text{RIG} = 1 - \text{NE}
  $$
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
    $$
    \text{Odds}_{\text{real}} = \text{Odds}_{\text{sampled}} \cdot w \implies \frac{p_{\text{real}}}{1 - p_{\text{real}}} = \frac{p_{\text{sampled}}}{1 - p_{\text{sampled}}} \cdot w
    $$
    Yielding:
    $$
    p_{\text{real}} = \frac{p_{\text{sampled}}}{p_{\text{sampled}} + \frac{1 - p_{\text{sampled}}}{w}}
    $$
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

$$
\operatorname{DCG@K} = \sum_{i=1}^{K}\frac{2^{\operatorname{rel}_i}-1}{\log_2(i+1)}
$$

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
