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

#### Industrial Archetype: YouTube DNN (2016) Candidate Generation & Example Age
YouTube 2016 established the structural baseline for industrial two-tower retrieval:
- **User Tower Inputs**: Average-pooled embeddings of user watch history IDs + average-pooled embeddings of search query tokens + static demographic and geographic features;
- **Continuous Freshness Feature: Example Age**:
  - **The Problem**: Recommendation systems exhibit severe recency bias. Older videos have accumulated massive watch counts and dominate positive training signals; newly uploaded high-quality videos lack historical impressions and suffer from low prior scores.
  - **Training-Time Modeling**: Feed the elapsed time between video upload and sample generation as a continuous feature: $x_{\text{age}} = t_{\text{event}} - t_{\text{upload}}$. The network explicitly learns how video engagement naturally decays with age.
  - **Serving-Time Trick**: At online inference, clamp `example_age` to 0 (or a small negative value, simulating "just uploaded now") for all candidate videos. This strips historical Matthew effects and recency bias, allowing fresh uploads to compete fairly based purely on relevance and content quality.

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

Hard negatives also expire. Once the model fixes a class of errors, an old mined set may become too easy; training on it indefinitely overfits a few failure patterns. A common loop periodically mines again with the current checkpoint, keeps a stable fraction of random negatives, and manually audits false negatives among the hardest examples. Rejection by an old model means only that the old model did not choose the item, not that the item is a reliable negative label.

#### Popularity Bias in In-Batch Negatives and Google logQ Correction
While in-batch negative sampling is computationally efficient ($B$ samples in a batch compute a $B \times B$ score matrix without extra negative embeddings), it introduces severe **sampling distribution bias**:
- In a batch, item $j$'s marginal probability of being selected as a negative is proportional to its global impression frequency: $p_j \propto \text{frequency}_j$;
- **Pathological consequence**: Highly popular head items are penalized as negative samples far more frequently than cold tail items! To minimize training loss, the optimizer collapses the embedding norms and inner-product scores of head items. At serving time, this suppresses high-quality popular items and induces erratic tail hallucinations.

**Mathematical Derivation of Google logQ Correction:**
The true full-vocabulary cross-entropy loss is:
$$\mathcal{L} = -\sum_{i=1}^B \log \frac{\exp(s(u_i, y_i))}{\sum_{j \in \mathcal{V}} \exp(s(u_i, j))}$$

Under Importance Sampling, when negatives are sampled from proposal distribution $P$ with probability $p_j$, the denominator partition function is estimated unbiasedly by:
$$\sum_{j \in \mathcal{V}} \exp(s(u_i, j)) = \mathbb{E}_{j \sim P}\left[ \frac{\exp(s(u_i, j))}{p_j} \right] \approx \sum_{j \in \mathcal{B}} \exp\left(s(u_i, j) - \log p_j\right)$$

Therefore, explicitly subtracting $\log p_j$ from candidate logits:

```math
s'(u_i, j) = s(u_i, j) - \log p_j
```

Transforms the training objective into:

```math
\mathcal{L}_{\text{in-batch}} = -\sum_{i=1}^B \log \frac{\exp\left(s(u_i, y_i) - \log p_{y_i}\right)}{\exp\left(s(u_i, y_i) - \log p_{y_i}\right) + \sum_{j \in \mathcal{B}, j \ne y_i} \exp\left(s(u_i, j) - \log p_j\right)}
```

- **Decoupling Training and Serving**: At training time, $-\log p_j$ neutralizes the gradient penalty caused by high sample frequency; **at serving time, the $-\log p_j$ term is completely removed**, scoring candidates by pure vector inner product $\langle \mathbf{u}, \mathbf{v}_j \rangle$ inside the ANN index, ensuring theoretically unbiased similarity retrieval.

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

Filtering order changes effective recall. Retrieving ANN top-100 and then applying inventory, location, and category filters may leave nothing. Blindly over-fetching top-1000 increases both index and downstream cost. Options include sharded indexes for strong filters, ANN systems with metadata filtering, coarse filtering before vector search, or dynamic over-fetch based on historical filter rates. High offline ANN recall paired with frequent online trending backfill usually points to recall after filtering, not the embedding benchmark.

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

Incremental updates must preserve vector-space compatibility. If an intraday job changes the item tower, shared bottom layers, or any parameter that moves both sides' coordinate system, new query embeddings no longer match the old item index. Updating user-only parameters can reuse the old index. Changing the shared space requires recomputing item vectors and switching indexes together, or keeping the old query tower until the new index is ready.

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

### 4.11 Core Technical Q&A: Two-Tower Computational Overhead and Physical ANN Mechanics

<details>
<summary>Q1: What are the primary end-to-end latency components in a two-tower online retrieval system? What are the key bottlenecks and industrial engineering optimizations at each stage?</summary>

End-to-end online retrieval latency in a two-tower system is governed by three primary pillars: **Query tower online inference**, **ANN vector search**, and **system, network, and post-filtering overhead**. The overarching architecture is fundamentally an optimization balance between **Recall**, **Memory Bandwidth**, and **Tail Latency (P99)**.

#### 1. Query Tower Online Inference

1. **Computational Bottlenecks**
   - Inference latency depends directly on model depth and operator complexity (e.g., deep MLPs, user sequence Transformer/Self-Attention blocks, cross-feature networks).
   - Unlike item embeddings that are precomputed offline, user features (real-time interaction sequences, dynamic context, statistical profiles) are highly dynamic and must be computed via live forward passes upon request arrival.

2. **Industrial Engineering Optimizations**
   - **Embedding Caching for Frequent Users**:
     - Maintain an in-memory LRU cache or distributed cache (e.g., Redis Cluster) storing query embeddings for active users. Within a short validity window (e.g., 5–15 minutes) with no new interactions, cache hits bypass model inference completely, reducing P50 inference latency to 0ms.
   - **Dynamic Batching at the Serving Layer**:
     - Configure microsecond queuing windows (`max_queue_delay_microseconds = 1000 ~ 2000`) and maximum batch sizes (`max_batch_size = 32 ~ 64`) inside inference gateways (e.g., NVIDIA Triton, TorchServe). Leveraging GPU parallel matrix compute amortizes memory access overhead, multiplying throughput (QPS) at the cost of a modest 1–2ms per-request latency overhead.

---

#### 2. ANN Vector Search Bottlenecks

The ANN stage represents the primary compute and memory bottleneck, dictated by three physical constraints: **vector dimensionality $d$**, **candidate pool scale $N$**, and **ANN index topology/hyperparameters**.

1. **Memory Bandwidth Bound**
   - Although vector search involves floating-point inner products, at massive candidate scale the true hardware bottleneck is typically **memory and memory bus bandwidth**, not arithmetic compute (ALU).
   - For a catalog of $N = 10^7$ (10 million) items with dimension $d = 128$ (FP32 consuming 512 bytes), the raw index occupies $5.12\text{ GB}$. Scanning large candidate subsets under high concurrency instantly saturates memory bus bandwidth.

2. **Vector Quantization (INT8 / PQ) for Bandwidth Relief**
   - **Scalar Quantization (SQ8 / INT8)**: Maps 4-byte Float32 values to 1-byte Int8 integers, reducing storage and memory bandwidth by $75\%$ while enabling vectorized CPU AVX-512 or GPU DP4A integer dot products.
   - **Product Quantization (PQ)**: Decomposes a 128-dim vector into 8 sub-vectors of dimension 16, quantizing each into a 1-byte centroid index. A single vector compresses to just 8 bytes ($64\times$ compression ratio), dramatically mitigating memory bus traffic.

3. **Hyperparameter Tuning: Recall vs. Latency**
   - **HNSW Index**: Key hyperparameters include search exploration budget `efSearch` and construction connectivity degree `M`. Increasing `efSearch` improves greedy routing accuracy across graph layers, boosting Recall@K, but linearly increases dot-product evaluations and degrades P99 latency.
   - **IVF Index**: Key hyperparameters are probe cluster count `nprobe` and total centroid count `nlist`. Increasing `nprobe` inspects more inverted lists, capturing boundary items, but multiplies the total number of scanned postings.

4. **Index Sharding and RPC Fan-Out (Scatter-Gather)**
   - When candidate catalogs scale to tens or hundreds of millions, single-node capacity is exceeded, necessitating horizontal sharding across $S$ search nodes.
   - **Scatter-Gather Bottleneck**: Queries are broadcast (scattered) concurrently across all $S$ shards, and top candidate lists are merged (gathered) at the aggregator gateway.
   - **Tail Latency Amplification**: By the principle of *The Tail at Scale*, end-to-end latency is bound by the slowest responding shard:
     $$P99_{\text{overall}} \approx 1 - (1 - P99_{\text{single}})^S$$
     As shard count $S$ grows, network jitter, thread scheduling variations, and garbage collection (GC) pauses are exponentially amplified, driving up P99 and P999 tail latency.

---

#### 3. Filtering, Merging, and Rescoring Overhead

1. **Hard Filtering Dilemmas**
   - Production recommendation requires strict filtering rules (out-of-stock items, geofencing, read items, blocked categories).
   - **Pre-Filtering**: Applying boolean attribute filters before ANN traversal fractures the spatial graph connectivity of HNSW, causing searches to terminate prematurely in local dead ends.
   - **Post-Filtering**: Filtering after ANN requires aggressive over-fetching (e.g., retrieving $K' = 10 \times K$ candidates) to prevent under-filling the final $K$ slots, heavily inflating ANN traversal and network payload costs.
   - **Hybrid Approaches**: Practical systems adopt **attribute-constrained graph search (AC-HNSW)** or partition indexes into independent sub-indexes by high-cardinality categories.

2. **Shard Merging and Exact Rescoring (FP32 Rerank)**
   - **Heap Merging**: Aggregator nodes collect $S \times K'$ candidates and execute a multi-way min-heap merge to distill the global top-$K'$ candidate set.
   - **Exact FP32 Rescoring**: Quantization (PQ/SQ8) introduces geometric distortion (quantization loss). After initial coarse pruning, systems fetch uncompressed FP32 dense vectors from RAM/SSD for the top-$K'$ candidates and recompute exact dot products. This rescoring step recovers lost recall at the cost of additional memory lookups.

</details>

<details>
<summary>Q2: What does a vector index physically look like in memory? Why can a system execute "direct Top-K" retrieval immediately upon obtaining the Query Embedding?</summary>

In production vector search engines (such as Faiss, ScaNN, and Milvus), an Index is not an abstract black box—it is a **concrete in-memory data structure (inverted postings lists, adjacency graphs, or quantized code tables)**. The ability to execute "direct Top-K" retrieval stems directly from the mathematical formulation of two-tower scoring combined with low-level C++ pruning mechanics.

#### 1. Physical In-Memory Layout of Mainstream Indexes

---

##### 1. Inverted File with Product Quantization (IVF-PQ)

IVF-PQ is the standard lightweight, low-memory inverted index, organized like a two-tier indexed dictionary:

```text
[Centroids Table]
  Cluster 0: [ 0.12, -0.45,  0.88, ... ] (d-dimensional Float32 vector)
  Cluster 1: [ 0.81,  0.03, -0.21, ... ]
  ...
  Cluster K: [ ... ]

[Inverted Lists / Postings]
  Key: Cluster 0 ──► List: [ 
                             (Item_102, [PQ Compressed Code: 0x1A, 0x3F, 0x09, 0xB2, ...]),
                             (Item_589, [PQ Compressed Code: 0x02, 0x8C, 0xF1, 0x4D, ...]),
                             ...
                           ]
  Key: Cluster 1 ──► List: [ (Item_12, ...), (Item_441, ...), ... ]
  ...
```

* **Offline Indexing Pipeline**:
  1. **Coarse Clustering**: Run K-Means over all catalog item vectors to establish $K$ coarse cluster centroids (e.g., $K = 4096$).
  2. **Inverted Posting Assignment**: Assign each item to its nearest centroid posting list.
  3. **Residual Product Quantization**: Compute residual vectors $\mathbf{r} = \mathbf{v} - \mathbf{c}_k$ against the assigned centroid. Split the residual into $M$ sub-vectors (e.g., $M = 8$) and quantize each to a 1-byte codebook index. A 128-dimensional Float32 vector (512 bytes) is compressed into an 8-byte code.
* **Memory Footprint**: Tens of millions of items require only hundreds of megabytes to a few gigabytes, residing entirely in RAM.

---

##### 2. Hierarchical Navigable Small World (HNSW)

HNSW delivers state-of-the-art recall by generalizing the **multi-layer Skip-List concept to multi-dimensional graph topologies**:

```text
[Layer 2 (Highway / Sparse Long Hops)]     Node_A ───────────────────────────► Node_K
                                           │                                  │
                                           ▼                                  ▼
[Layer 1 (Expressway / Mid-level)]         Node_A ────────► Node_D ─────────► Node_K ────► Node_M
                                           │                │                 │            │
                                           ▼                ▼                 ▼            ▼
[Layer 0 (Base Layer / Full Adjacency)]    Complete Item Adjacency Graph (16~64 nearest neighbor pointers per node)
```

* **Underlying Physical Structures**:
  1. **Dense Vector Array**: A contiguous array storing `Item_ID -> Vector` entries (FP32 or SQ8 quantized).
  2. **Hierarchical Adjacency Lists**: Each node on each layer maintains a variable-length array `neighbors: List[int]` containing the item IDs of its $M$ closest bidirectional neighbors in geometric space. Top layers are sparsely populated, while Layer 0 contains the complete catalog graph.

---

#### 2. Why Does Query Embedding Enable "Direct Top-K"?

A frequent point of confusion is: *"After generating the user vector, why don't we need another model forward pass over all candidates to compute scores?"*

**No forward passes are needed.** This is an intentional design achieved through mathematical constraints and systems engineering:

##### 1. Mathematical Formulation: Pure Geometric Operations

Complex ranking models (e.g., DCN, DIN) cannot perform direct Top-K because their scoring functions involve non-linear feature interactions:
$$\text{Score} = \text{MLP}\Big(\text{Concat}(\mathbf{u}, \mathbf{v}, \mathbf{u} \times \mathbf{v})\Big)$$
This requires concatenating $\mathbf{u}$ with millions of candidate $\mathbf{v}$ vectors and executing millions of deep network forward passes on live traffic, which is computationally intractable.

In contrast, two-tower models **strictly constrain the scoring function to a linear dot product or cosine similarity**:
$$\text{Score}(u, v) = \mathbf{u}^\top \mathbf{v} \quad \left(\text{with } L_2 \text{ normalization, } \cos(\mathbf{u}, \mathbf{v}) = 1 - \frac{1}{2}\Vert \mathbf{u} - \mathbf{v} \Vert_2^2\right)$$

- **Mathematical Equivalence**:
  - All complex historical behaviors, demographics, and long-term user interests are compressed into a single $d$-dimensional vector $\mathbf{u}$ by the query tower.
  - All multimodal, taxonomic, and pricing item attributes are precomputed into a static vector $\mathbf{v}$ by the item tower.
  - Finding candidate items with the highest predicted scores is mathematically identical to **Maximum Inner Product Search (MIPS)**: finding the Top-$K$ points in multi-dimensional space with minimum angular distance (maximum dot product) to vector $\mathbf{u}$.

---

##### 2. Engineering Execution: Sub-5ms Top-K Pruning

Because Top-K is reduced to geometric nearest neighbor search, neural network inference is bypassed in favor of hardware-level C++ pruning:

- **IVF-PQ Retrieval Path**:
  1. **Centroid Coarse Pruning (Microsecond scale)**: Compute dot products between $\mathbf{u}$ and $K = 4096$ coarse centroids, selecting only the top $n_{\text{probe}}$ closest centroids (e.g., 8 clusters).
  2. **Geometric Pruning (Millisecond scale)**: Over $99\%$ of the catalog is instantly discarded, leaving only candidates within the 8 selected posting lists (several tens of thousands of items).
  3. **Asymmetric Distance Computation (ADC)**: The query vector $\mathbf{u}$ remains unquantized in FP32. Precompute a small lookup table ($M \times 256$) containing inner products between query sub-vectors and PQ codebook sub-centroids. Scanning posting lists **requires zero vector dequantization**; distances are computed by table lookups indexed by the item's 8-byte code. Accelerated by CPU AVX-512 SIMD instructions, tens of thousands of inner products are evaluated in 1–3ms, with top scores tracked in an in-memory min-heap of size $K$.

- **HNSW Retrieval Path**:
  1. **Top-Layer Navigation**: Enter at the topmost entry point (Layer 2), evaluate dot products with immediate neighbors, and greedily hop along the path of maximum inner product.
  2. **Layer Descent**: When no neighbor yields a higher score on the current layer, descend to the corresponding node on the layer below (Layer 1) and resume greedy routing on denser connectivity.
  3. **Convergence on Layer 0**: Arrive at Layer 0 and perform local beam exploration. Evaluating only several hundred to a few thousand vector dot products reliably discovers the true Top-$K$ nearest neighbors with $95\%+$ Recall.

---

#### 3. Summary

By reducing complex non-linear feature interactions to **linear inner products**, the two-tower model converts probabilistic inference into geometric nearest neighbor search. Once the query embedding is generated, candidate retrieval relies entirely on optimized C++ dot products, SIMD lookup tables, and min-heap merges, completing Top-$K$ candidate generation over tens of millions of items in under 5ms.

</details>

