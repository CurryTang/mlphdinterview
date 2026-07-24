# Two-Tower Models, Negatives, and Vector Retrieval

## Chapter 4 Two-Tower, Negative Samples, and Vector Retrieval

### 4.1 Two-Tower Turns Matching into Nearest Neighbor

The two towers encode the query/user and document/item separately:

```math
z_q=f_\theta(x_q), \qquad z_i=g_\phi(x_i),
```

```math
s(q,i)
=\frac{z_q^\top z_i}{\|z_q\|_2\|z_i\|_2}.
```

On the recommendation side, `x_q` consists of user, history, and context, while `x_i` consists of item features. On the search side, `x_q` is the query, and `x_i` is the document.

The greatest benefit of independent encoding on both sides is that item/document vectors can be calculated offline. Online, only the query vector is calculated, followed by ANN. The cost is that the query and candidates cannot perform fine-grained token/feature interaction during the encoding stage.

### 4.2 Training Objective

A softmax contrastive objective is commonly used on a set of positive and negative samples:

```math
\mathcal L
=-\log
\frac{\exp(s(q,i^+)/\tau)}
{\exp(s(q,i^+)/\tau)+
\sum_{j\in\mathcal N_q}\exp(s(q,j)/\tau)}.
```

The temperature `τ` controls the sharpness of the distribution. The negative sample set `N_q` often determines performance more than the network architecture itself.

Two-tower models also support pointwise, pairwise, and listwise training. Pointwise classifies one user-item pair; pairwise compares one positive with one negative; the sampled softmax above is a common listwise form in which one positive competes with many negatives. Large retrieval systems often favor in-batch sampled softmax because one batch of encodings creates many comparisons.

### 4.3 Negative Samples Should Not Be Sampled Randomly

Random negative samples are easily distinguished by the model, leading to fast convergence during training, but the model fails to distinguish truly similar candidates online.

Common sources:

- Full-corpus random negatives: Cheap, usually too simple;
- In-batch negatives: Using positive examples of other samples as negatives for the current query, high throughput;
- Exposed but unclicked items: difficult examples with position bias and many false negatives;
- Hard negatives: Candidates recalled by old models or BM25 that are semantically similar but irrelevant;
- Mixed negatives: Balancing coverage, difficulty, and stability.

In-batch negatives contain false negatives. Two users might both like the same item, or a negative example for one query might actually be relevant. This can be mitigated through deduplication, same-query masking, popularity correction, and soft labels.

For a retrieval model, exposed-but-unclicked items are usually poor default negatives. The old retriever and ranker already considered them plausible, and a missing click may come from position, timing, or chance. A safer mix starts with corpus or in-batch negatives and adds items rejected by pre-ranking or ranking as hard negatives. Treat exposed misses separately and verify that they help.

Sampling changes the prior distribution. If item `j` enters the negative set with probability `p_j`, an in-batch softmax can correct its logit with:

```math
s'(q,j)=s(q,j)-\log p_j.
```

Without this correction, a popular item can be penalized simply because it appears in more batches. Aggressive popularity downsampling creates the opposite mismatch. Record the sampling probability, loss correction, and serving score together.

### 4.4 ANN and Vector Databases

Performing exact top-k inner products across the entire corpus remains expensive. Approximate Nearest Neighbor (ANN) trades a small amount of recall loss for speed.

Common approaches:

- IVF: Partition vectors into buckets, query only scans the most relevant buckets;
- PQ: Segment and quantize vectors, compressing storage and approximating distance calculation;
- HNSW: Construct a multi-layer adjacency graph, approaching the query through graph search.

Index tuning must simultaneously consider:

- Recall vs. latency;
- Memory vs. quantization error;
- Index build time vs. incremental updates;
- Top-k size vs. subsequent ranking costs.

Whether the index supports incremental writes after item vector updates is also critical. Rebuilding the entire index once a day may not keep up with news, product inventory, or short video trends.

When validating ANN, fix the same set of queries and plot the `Recall@K - P95/P99 Latency - Memory` curve, rather than just reporting a single top-K Recall. This curve reveals how much loss the approximate retrieval itself incurs when the model embedding remains constant but index parameters change.

### 4.5 Online Service

Typical two-tower pipeline:

```text
Offline:
item features -> item tower -> item embedding -> ANN index

Online:
user/query features -> query tower -> query embedding
                               -> ANN top-k
                               -> filtering and subsequent ranking
```

Key concerns:

- P99 latency of the query tower;
- Consistency between embedding versions and index versions;
- Latency of new item vector generation and indexing;
- Default values for missing features;
- Vector norms, quantization, and similarity definitions;
- Fallback channels in case of index failure.

Production updates usually run at two cadences. A daily full job shuffles the previous day's data, trains both towers, and republishes all item vectors. Intraday jobs consume recent logs and update user ID embeddings so that new interests reach the user tower sooner. Incremental data arrive in time order, have biased coverage, and contain less mature labels, so they do not replace full training. Serving should track the full checkpoint, incremental offset, and item-index version independently and be able to roll back to the last complete release.

### 4.6 Two-Tower and Cross-Encoder

A cross-encoder concatenates the query and candidate:

```text
[CLS] query [SEP] document [SEP] -> relevance score
```

It enables fine-grained interaction and is usually more accurate, but every query-candidate pair must pass through the model, making it impossible to precompute the document side. The most common combination is two-tower retrieval followed by cross-encoder re-ranking.

This is also the foundation for understanding LLM rankers. LLM rankers do not eliminate computational costs; they simply amplify the semantic capabilities of the cross-encoder.

### 4.7 Discrete features in the model

User IDs, item IDs, categories, and cities are mapped to integers and then looked up in embedding tables. One-hot encoding works for tiny vocabularies; hundred-million-scale users and items require dense embeddings.

The fragile part is mapping and version control:

- default IDs for logged-out users and OOV values;
- when new items receive IDs, vectors, and index entries;
- whether training and serving use the same vocabulary;
- dedicated IDs for frequent values versus hashing the tail;
- collision and table-capacity monitoring.

ID embeddings capture collaborative information, while content features help tail and new items. An ID-only model often fits active items well and fails cold start completely.

### 4.8 Self-supervised two-tower training

Head items have abundant click supervision; tail embeddings are weak. Self-supervision creates two views of one item by:

- randomly masking fields;
- dropping values from multi-valued categories or keywords;
- splitting fields into complementary views;
- masking groups of fields with high mutual information.

The two views of one item should be close and different items should be separated:

```math
\mathcal L_{\text{ssl}}(i)
=-\log
\frac{\exp(\operatorname{sim}(z_i^{(1)},z_i^{(2)})/\tau)}
{\sum_j\exp(\operatorname{sim}(z_i^{(1)},z_j^{(2)})/\tau)}.
```

Training combines click and self-supervised losses:

```math
\mathcal L
=\mathcal L_{\text{click}}
+\alpha\mathcal L_{\text{ssl}}.
```

Augmentation must preserve item identity. Masking brand and model together may make two products indistinguishable and teach the encoder to ignore essential fields.

### 4.9 Deep Retrieval

Deep Retrieval associates each item with one or more discrete paths, such as `(2,4,1)`, and maintains:

```text
item -> paths
path -> items
```

Given user features `x`, the model autoregressively predicts a path:

```math
p(a,b,c\mid x)
=p_1(a\mid x)
\cdot p_2(b\mid a,x)
\cdot p_3(c\mid a,b,x).
```

The number of paths grows exponentially with depth, so serving uses beam search, then reads each selected path's posting list:

```text
user -> paths -> items
```

Training alternates between two relationships:

1. a user click raises the probability of paths assigned to that item;
2. user-path affinity updates the item's assigned paths.

A load regularizer prevents popular items from collapsing onto a few paths. Compared with ANN, the retrievable structure itself is learned, at the cost of more complex path versioning, beam search, and bidirectional indexes.

### 4.10 Chapter Self-Test

1. Why is the two-tower model suitable for retrieval but not for directly replacing all fine-ranking?
2. Why are in-batch negative samples efficient, and where do false negatives come from?
3. Are hard negatives always better the harder they are?
4. What structures do HNSW, IVF, and PQ utilize, respectively?
5. After item embeddings are updated, what other versions need to be synchronized online?
6. Why do tail items benefit more from self-supervision?
7. Why limit the number of items assigned to one Deep Retrieval path?

<details>
<summary>Reference answers</summary>

1. Item embeddings can be precomputed and searched with ANN. Because the towers do not perform fine-grained interaction before scoring, they do not replace cross-encoders or rich rankers.
2. Other positives in the batch become negatives without extra encoding. False negatives appear when another user's positive is also valid for the current user.
3. No. An extremely hard sample may be mislabeled, a false negative, or genuinely indistinguishable. It must be difficult and reliably labeled.
4. HNSW navigates a hierarchical neighbor graph; IVF restricts search with coarse centroids; PQ splits and quantizes vectors to approximate distance compactly.
5. Synchronize the model, embeddings, ANN index, feature schema, and availability versions, keeping the query tower compatible with indexed item vectors.
6. Tail items have little click supervision. Multiple content views provide extra training signals and make their representations stable under missing or perturbed fields.
7. If many items collapse onto a few paths, posting lists and serving costs explode, candidate congestion rises, and other paths fail to specialize.

</details>
