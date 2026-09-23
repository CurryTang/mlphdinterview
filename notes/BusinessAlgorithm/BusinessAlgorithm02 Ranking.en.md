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

### 9.6 Measuring Ranking Performance

For pointwise estimations like CTR and CVR, first look at LogLoss:

```math
\operatorname{LogLoss}
=-\frac{1}{N}\sum_i
\left[ y_i\log p_i+(1-y_i)\log(1-p_i) \right].
```

It focuses on the probability itself and absolute calibration quality.

#### Global AUC and Its Pitfalls in Recommendation
AUC measures the probability that a randomly chosen positive instance is scored higher than a randomly chosen negative instance:
$$\operatorname{AUC} = \frac{\sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \left[ \mathbb{I}(p_i > p_j) + 0.5 \times \mathbb{I}(p_i = p_j) \right]}{|\mathcal{D}^+| \times |\mathcal{D}^-|}$$

**Why can Global AUC mislead recommendation systems?**
Global AUC evaluates positive and negative pairs across different users. If active User A has a high baseline click rate (30%) while inactive User B rarely clicks (1%), a model that merely memorizes demographic priors will predict uniformly high scores for User A and low scores for User B:
- Across users, User A's clicks easily outrank User B's impressions, driving **Global AUC as high as 0.85+**;
- However, in production serving, **the system ranks candidate items for a single user within a single session**—items for different users never compete on the same viewport;
- If intra-user ranking is completely random ($\text{AUC}_A = 0.5, \text{AUC}_B = 0.5$), the user experience degrades into noise despite the stellar offline Global AUC.

#### Group AUC (GAUC) — The Industrial Gold Standard
To eliminate cross-user prior biases and isolate intra-user personalized ranking performance, modern recommendation systems (e.g., ByteDance, Meta, Alibaba) standardize on **GAUC (Group AUC)**:

```math
\operatorname{GAUC}
=\frac{\sum_{u \in \mathcal{U}} w_u \times \operatorname{AUC}_u}{\sum_{u \in \mathcal{U}} w_u}
```

Where:
- $\operatorname{AUC}_u$ is the local AUC computed strictly over impressions served to user $u$;
- $w_u$ is the user's weight, typically set to impression count ($\text{impressions}_u$) or click volume;
- **Boundary condition**: If user $u$ has **only positive samples (all clicks)** or **only negative samples (zero clicks)**, positive-negative pairs count is zero and $\operatorname{AUC}_u$ is undefined; such users must be filtered out during accumulation.

**Rule of Thumb in Industry**: Holding retrieval and coarse ranking fixed, a fine-ranking gain of $\Delta\operatorname{GAUC} \ge +0.003$ (+0.3%) reliably translates into statistically significant online A/B gains in CTR, watch time, or retention.

Search and recommendation lists often use DCG/NDCG:

```math
\operatorname{DCG@K}
=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)},
```

```math
\operatorname{NDCG@K}
=\frac{\operatorname{DCG@K}}{\operatorname{IDCG@K}}.
```

The higher the position, the lower the discount, and the cost of misranking a highly relevant candidate is greater. Relevance grades for search can come from human labeling; recommendations often map clicks, purchases, or behavioral intensity into grades, and it must be explained what such mappings represent.

If the task primarily cares about the first correct result, MRR can be used:

```math
\operatorname{MRR}
=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{rank_q}.
```

These metrics must be compared on the same candidate set. If the retrieval set changes, changes in NDCG may stem from better candidates or the ranking model itself; these two must be experimented with separately.

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

<details>
<summary>Reference answers</summary>

1. Calibration aligns probabilities with empirical frequency; it does not guarantee the relative order of nearby candidates.
2. Global AUC pools positive/negative comparisons across disparate users. Highly active users have high baseline CTRs, so a model memorizing user priors yields high global AUC while intra-user rankings remain random. GAUC computes AUC strictly within each user's impressions, isolating true personalization quality.
3. Use exposed candidates from the same request that an old model ranked highly but received negative feedback, or sample top ANN/BM25 results. Filter false negatives and immature labels.
4. Logarithmic discount gives top positions more weight, so the same displacement costs more near rank one than near the tail.
5. Two-tower BERT fits retrieval or coarse ranking. Cross-BERT jointly encodes query and document and belongs in fine ranking over a small set.
6. Produce separate relevance, quality, and business-objective scores, calibrate them, combine them by scenario, and keep interpretable hard guardrails.
7. Rules are easier to inspect and repair while data and the pipeline are still changing. A learned model needs trustworthy satisfaction and behavior labels, plus relevance-grade monitoring so that it cannot trade relevance for clicks.

</details>
