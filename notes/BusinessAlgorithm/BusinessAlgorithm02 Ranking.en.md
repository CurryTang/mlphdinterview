# Ranking Objectives and Offline Evaluation

## Chapter 9: Ranking Objectives and Offline Evaluation

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
=-\left[y\log p+(1-y)\log(1-p)\right].
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
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
```

It focuses on the probability itself. AUC measures the probability that a random positive example is ranked higher than a random negative example, which is suitable for observing overall discriminative ability, but it does not specifically focus on the top of the list, nor does it indicate whether scores are calibrated.

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

Problem: [[BusinessAlgorithm09 Quick Coding.md#QC05 NDCG@K|QC05 NDCG]].

### 9.8 Chapter Self-Test

1. Pointwise scores can be calibrated; why might ranking still be poor?
2. How should hard negatives for Pairwise be generated?
3. Why is NDCG more sensitive to the top of the list?
4. Where should Cross-BERT and two-tower BERT be placed?
5. How should relevance, content quality, and final ranking scores be decoupled?
6. Why does search fusion often start with relevance buckets and hand-written rules before a learned fusion model?

<details>
<summary>Reference answers</summary>

1. Calibration aligns probabilities with empirical frequency; it does not guarantee the relative order of nearby candidates.
2. Use exposed candidates from the same request that an old model ranked highly but received negative feedback, or sample top ANN/BM25 results. Filter false negatives and immature labels.
3. Logarithmic discount gives top positions more weight, so the same displacement costs more near rank one than near the tail.
4. Two-tower BERT fits retrieval or coarse ranking. Cross-BERT jointly encodes query and document and belongs in fine ranking over a small set.
5. Produce separate relevance, quality, and business-objective scores, calibrate them, combine them by scenario, and keep interpretable hard guardrails.
6. Rules are easier to inspect and repair while data and the pipeline are still changing. A learned model needs trustworthy satisfaction and behavior labels, plus relevance-grade monitoring so that it cannot trade relevance for clicks.

</details>
