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

### 4.3 Negative Samples Should Not Be Sampled Randomly

Random negative samples are easily distinguished by the model, leading to fast convergence during training, but the model fails to distinguish truly similar candidates online.

Common sources:

- Full-corpus random negatives: Cheap, usually too simple;
- In-batch negatives: Using positive examples of other samples as negatives for the current query, high throughput;
- Exposure without clicks: Closer to the online distribution, but contains position bias;
- Hard negatives: Candidates recalled by old models or BM25 that are semantically similar but irrelevant;
- Mixed negatives: Balancing coverage, difficulty, and stability.

In-batch negatives contain false negatives. Two users might both like the same item, or a negative example for one query might actually be relevant. This can be mitigated through deduplication, same-query masking, popularity correction, and soft labels.

Sampling also changes the prior distribution. If popular items are more likely to enter a batch, the model will be biased toward popularity; if popular items are over-downsampled, online probabilities may be distorted. The training objective, sampling distribution, and online scoring must be designed together.

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

### 4.6 Two-Tower and Cross-Encoder

A cross-encoder concatenates the query and candidate:

```text
[CLS] query [SEP] document [SEP] -> relevance score
```

It enables fine-grained interaction and is usually more accurate, but every query-candidate pair must pass through the model, making it impossible to precompute the document side. The most common combination is two-tower retrieval followed by cross-encoder re-ranking.

This is also the foundation for understanding LLM rankers. LLM rankers do not eliminate computational costs; they simply amplify the semantic capabilities of the cross-encoder.

### 4.7 Chapter Self-Test

1. Why is the two-tower model suitable for retrieval but not for directly replacing all fine-ranking?
2. Why are in-batch negative samples efficient, and where do false negatives come from?
3. Are hard negatives always better the harder they are?
4. What structures do HNSW, IVF, and PQ utilize, respectively?
5. After item embeddings are updated, what other versions need to be synchronized online?

<details>
<summary>Reference answers</summary>

1. Item embeddings can be precomputed and searched with ANN. Because the towers do not perform fine-grained interaction before scoring, they do not replace cross-encoders or rich rankers.
2. Other positives in the batch become negatives without extra encoding. False negatives appear when another user's positive is also valid for the current user.
3. No. An extremely hard sample may be mislabeled, a false negative, or genuinely indistinguishable. It must be difficult and reliably labeled.
4. HNSW navigates a hierarchical neighbor graph; IVF restricts search with coarse centroids; PQ splits and quantizes vectors to approximate distance compactly.
5. Synchronize the model, embeddings, ANN index, feature schema, and availability versions, keeping the query tower compatible with indexed item vectors.

</details>
