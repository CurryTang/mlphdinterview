# Retrieval: From Full Corpus to Candidate Set

## Chapter 3 Sparse Retrieval and Collaborative Retrieval

### 3.1 Two Types of Sparse Evidence

Term co-occurrence in search and behavioral co-occurrence in recommendation look different, but their computational habits are quite similar.

Search asks: "Which documents contain the important terms from the query?"

Recommendation asks: "Which items are consumed by similar groups of people who also interacted with the user's items?"

Both establish inverted relationships in advance, finding a batch of candidates online from a small number of keys. The difference lies in the semantics of the key: search uses terms, ItemCF uses items, and UserCF uses similar users.

### 3.2 Inverted Index

A forward index is:

```text
doc_id -> document content
```

An inverted index is:

```text
term -> [(doc_id, tf, positions), ...]
```

`tf` is the frequency of the term in the document, and `positions` records the locations, which can be used for word distance and phrase matching. Online queries first tokenize the input, then retrieve multiple posting lists to perform intersection, union, or more complex traversals.

The strengths of an inverted index are:

- Precise term matching is stable and interpretable;
- Index compression and skip list technologies are mature;
- Incremental document updates are relatively straightforward;
- It is highly reliable for precise queries such as entity names, model numbers, and codes.

Its weaknesses are also straightforward: it is difficult to retrieve items when there is no literal overlap. "LV bag" and "LOUIS VUITTON handbag" are semantically close, but pure term-based retrieval may not know this.

### Quick Coding: Inverted Index

Given a set of tokenized documents, construct a `term -> posting list`. Each posting must store the `doc_id`, term frequency, and positions, and be sorted by document ID. This exercise checks whether you truly understand what data is required for TF, word distance, and phrase matching.

Problem: [[BusinessAlgorithm09 Quick Coding.md#QC02 Build Inverted Index|QC02 Build Inverted Index]].

### 3.3 TF-IDF and BM25

TF-IDF considers both the frequency of a term in the current document and its rarity across the entire corpus:

```math
\operatorname{tfidf}(t,d)
=\operatorname{tf}(t,d)
\log\frac{N}{df(t)}.
```

Common terms have low discriminative power, so IDF assigns them a smaller weight. BM25 builds on this by adding term frequency saturation and document length normalization:

```math
\operatorname{BM25}(q,d)
=\sum_{t\in q}
\operatorname{IDF}(t)
\frac{tf(t,d)(k_1+1)}
{tf(t,d)+k_1(1-b+b|d|/\operatorname{avgdl})}.
```

A term appearing 20 times should not be twice as important as one appearing 10 times; this is the purpose of the saturation term. Long documents are naturally more likely to hit terms, and length normalization corrects for this.

BM25 is fast, stable, and interpretable. When evaluating dense retrieval or generative retrieval, compare against it first; comparing only against weaker neural baselines makes it difficult to judge whether the added complexity is worthwhile.

### 3.4 ItemCF

ItemCF first calculates item similarity based on user-item interactions, then expands candidates from the user's history.

The simplest co-occurrence similarity is:

```math
\operatorname{sim}(i,j)
=\frac{|U_i\cap U_j|}
{\sqrt{|U_i||U_j|}}.
```

`U_i` is the set of users who have interacted with item `i`. The denominator suppresses popular items.

The score for user `u` on candidate `i` can be written as:

```math
\operatorname{score}(u,i)
=\sum_{j\in H_u}
w(u,j)\operatorname{sim}(j,i),
```

`H_u` is the user's history, and `w(u,j)` can incorporate behavior type, time decay, and viewing depth.

The complete engineering workflow:

1. Offline statistics of the user-to-item behavior table;
2. Generate item co-occurrence;
3. Calculate and truncate the top similar items for each item;
4. Read user history online;
5. Query the similarity table, aggregate, and filter out already exposed content;
6. Output top candidates.

The benefits of ItemCF are ease of interpretation and low service cost. The downsides are that new items have no co-occurrence, popular items easily dominate, and it may fail to expand when interests cross categories.

### Quick Coding: ItemCF

Starting from a small dataset of `user -> interacted items`, calculate co-occurrence cosine similarity and recall un-interacted items for a specified user. You are required to filter history, handle tied candidates stably, and explain why this interview code cannot be directly used to serve full-scale traffic.

Problem: [[BusinessAlgorithm09 Quick Coding.md#QC03 ItemCF Recommendation|QC03 ItemCF Recommendation]].

### 3.5 UserCF and Swing

UserCF first finds similar users, then recalls items they liked that the current user has not seen. It is useful when user interests are relatively stable and the number of users is not so large that the similarity table becomes unmaintainable. Large-scale content platforms more commonly use ItemCF because item similarity relationships are generally more stable than user similarity relationships.

Swing addresses "small-circle noise" in co-occurrence. If two items are only interacted with by a small group of highly overlapping users, standard co-occurrence might judge them as overly similar. Swing penalizes the number of common items for user pairs; the gist is:

```math
\operatorname{sim}(i,j)
=\sum_{u,v\in U_i\cap U_j}
\frac{1}{\alpha+|I_u\cap I_v|}.
```

Contributions from user pairs with excessive common interests are reduced. The formula is not the point; the point is which statistical bias it corrects.

### 3.6 Matrix Factorization

Approximate the user-item matrix `R` as two low-rank matrices:

```math
R\approx UV^\top,
\qquad
\hat r_{ui}=u_u^\top v_i.
```

Matrix factorization compresses discrete co-occurrence into continuous vectors. It is easier to express latent interests than simple ItemCF, but it still relies primarily on interactions, and the cold-start problem does not disappear into thin air.

Implicit feedback usually uses targets with confidence levels, rather than treating all non-clicks as explicit negatives. Training also requires careful negative sampling; otherwise, the massive number of zero entries will drown out the positive samples.

### 3.7 When to Use Which

| Method | Strengths | Significant Weaknesses |
| --- | --- | --- |
| Inverted Index/BM25 | Precise text, mature, efficient, easy to update | No semantic overlap |
| ItemCF | Cheap, interpretable, strong behavioral correlation | Cold start, popularity bias |
| UserCF | Discovers candidates via similar groups | Unstable user relationships, large scale |
| Swing | Suppresses small circles and overly strong co-occurrence | More complex calculation and parameters |
| Matrix Factorization | Continuous latent space, better generalization | Still relies on behavior, limited feature utilization |

Online systems usually retain multiple channels. Even if a traditional method's single-path metric is not superior, it may find candidates missed by deep models.

### 3.8 Chapter Self-Test

1. Why does an inverted index need to store term positions?
2. What do the saturation term and length normalization in BM25 solve, respectively?
3. What precomputed tables are needed for ItemCF online service?
4. What kind of problem does Swing aim to correct in standard co-occurrence?
5. What is the connection and difference between matrix factorization and two-tower models?

---

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

---

## Chapter 5 Query, Content, and Multi-Path Retrieval

### 5.1 Query Processing Determines Retrieval Boundaries

Search queries are short and ambiguous; input quality directly determines the upper limit of retrieval.

Tokenization breaks continuous strings into searchable terms. Term weighting determines which word in "Apple mobile wallpaper" dictates the intent. Category recognition decides whether to go to the product, video, or encyclopedia index. Intent recognition also controls downstream modules such as timeliness, geography, and adult safety.

These modules are not necessarily independent models. Large systems share encoders while retaining dictionaries and rules to handle model numbers, spelling, sensitive words, and strong business intents.

### 5.2 Query Rewriting

The goal of rewriting is practical: to allow relevant content that would otherwise not be found to enter the candidate set.

Common types:

- Spelling correction;
- Synonym and alias expansion;
- Tokenization correction;
- Entity normalization;
- Ellipsis completion;
- Intent expansion;
- Historical query association.

The original query and the rewritten query are usually retrieved separately, and then candidates are merged. Excessive rewriting introduces semantic drift, so it requires the original query as a fallback, as well as rewriting confidence and relevance constraints.

In generative search, query rewriting adds another layer: complex questions are broken into multiple sub-questions, and the search agent continues to rewrite based on intermediate evidence. We will return to this in Chapter 14.

### 5.3 Content Retrieval on the Recommendation Side

Recommendation has no query, but candidates can still be generated from content:

- Categories, tags, and authors;
- Text, image, and audio embeddings;
- Geography, language, and publication time;
- Similar items to content the user just interacted with;
- Operational configurations and hot pools.

Content retrieval is especially important for new items and the long tail. When collaborative signals have not yet formed, content is one of the few available pieces of information. However, "looking similar" does not equal "the same user will like it," so online systems usually run content retrieval and collaborative retrieval in parallel.

### 5.4 How to Merge Multi-Path Retrieval

Suppose there are five paths: ItemCF, two-tower, popular, followed authors, and geographic location. Direct concatenation leads to:

- Different scales of scores across channels;
- Duplicate items appearing;
- Strong channels consuming the entire quota;
- Incremental candidates from small channels being drowned out;
- Certain channels being nearly empty after filtering.

Online systems usually combine fixed or dynamic quotas, intra-channel truncation, score normalization, learned fusion, and deduplication/backfilling. Fusion logs must retain the full retrieval source of each candidate; otherwise, you only see the final adopted channel and cannot calculate single-path coverage, overlap rates, or incremental conversion.

A channel with low independent Recall may still be valuable. If the positive examples it retrieves are long-tail content that other channels cannot find, its incremental value is high.

### Quick Coding: RRF Fusion

Without calibrating the raw scores of different retrieval channels, implement Reciprocal Rank Fusion based on the rank of each channel. Remember to handle duplicate candidates within a single list and stable tie-breaking.

Problem: [[BusinessAlgorithm09 Quick Coding.md#QC04 Reciprocal Rank Fusion|QC04 Reciprocal Rank Fusion]].

### 5.5 Exposure Filtering and Cache Retrieval

Recommendation usually filters out content the user has recently seen or explicitly disliked. If the filtering window is too short, repetition occurs; if it is too long, high-quality content that could be re-watched is excluded forever. Exposure filtering is best handled by retaining behavior type, time, and scenario, rather than just maintaining a permanent blacklist.

Search cache/KV retrieval utilizes stable results for high-frequency queries. It has low latency but must handle timeliness, personalization, and cache pollution. Hot news queries cannot reuse old results for long, and personalized queries should not directly share caches across all users.

### 5.6 How to Validate Retrieval Performance

In the retrieval stage, first check whether the target candidates have entered the top-K:

```math
\operatorname{Recall@K}
=\frac{\lvert \text{Relevant items in top-K}\rvert}
{\lvert \text{Relevant items in test window}\rvert}.
```

The denominator in recommendation is usually the clicks, effective views, or purchases that occurred in the test window, not all of the user's potential interests. Search can use human relevance annotations, but they must be bucketed by head, long-tail, entity, and complex queries. A single total Recall is easily masked by high-frequency samples.

Multi-path retrieval also requires looking at:

- Coverage and independent retrieval volume of each path;
- Candidate overlap between channels;
- Coverage of new items, long-tail, and cold queries;
- Retention rate after merging and filtering;
- How many candidates can be returned under a fixed P99 latency.

When metrics drop, replay along the candidate funnel:

1. Did the ground-truth enter any channel?
2. If it entered, was the intra-channel ranking too low?
3. Was it truncated by the merged quota?
4. Was it filtered out by exposure, inventory, or safety rules?
5. Was it mistakenly killed after entering ranking?

Locate where the candidate disappears first, then decide whether to retrain the two-tower model. Changing the model will not solve the problem if the filtering window or channel quotas are incorrect.

### 5.7 Chapter Self-Test

1. How can query rewriting avoid semantic drift?
2. Why is content retrieval suitable for cold starts, yet cannot handle recommendation alone?
3. How to evaluate whether a low-traffic retrieval channel has independent value?
4. Why can't scores from multi-path retrieval be compared directly?
5. Why might the recall of long-tail queries still worsen when overall Recall increases?
