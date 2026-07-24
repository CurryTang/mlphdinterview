# Query, Content, and Multi-Channel Retrieval

## Chapter 6 Query, Content, and Multi-Path Retrieval

### 6.1 Query Processing Determines Retrieval Boundaries

Search queries are short and ambiguous; input quality directly determines the upper limit of retrieval.

Tokenization breaks continuous strings into searchable terms. Term weighting determines which word in "Apple mobile wallpaper" dictates the intent. Category recognition decides whether to go to the product, video, or encyclopedia index. Intent recognition also controls downstream modules such as timeliness, geography, and adult safety.

These modules are not necessarily independent models. Large systems share encoders while retaining dictionaries and rules to handle model numbers, spelling, sensitive words, and strong business intents.

### 6.2 Query Rewriting

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

In generative search, query rewriting adds another layer: complex questions are broken into multiple sub-questions, and the search agent continues to rewrite based on intermediate evidence. We will return to this in Chapter 19.

### 6.3 Content Retrieval on the Recommendation Side

Recommendation has no query, but candidates can still be generated from content:

- Categories, tags, and authors;
- Text, image, and audio embeddings;
- Geography, language, and publication time;
- Similar items to content the user just interacted with;
- Operational configurations and hot pools.

Content retrieval is especially important for new items and the long tail. When collaborative signals have not yet formed, content is one of the few available pieces of information. However, "looking similar" does not equal "the same user will like it," so online systems usually run content retrieval and collaborative retrieval in parallel.

Several useful side channels can be written directly as index paths:

```text
GeoHash / city -> recent high-quality content
user -> followed or recently engaged author -> newest items
user -> preferred author -> similar authors -> newest items
```

The first path is nearly non-personalized and covers local supply. The two U2A2I paths use explicit follows for precision and implicit engagement or similar authors for coverage.

### 6.4 How to Merge Multi-Path Retrieval

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

### 6.5 Exposure Filtering and Cache Retrieval

Recommendation usually filters out content the user has recently seen or explicitly disliked. If the filtering window is too short, repetition occurs; if it is too long, high-quality content that could be re-watched is excluded forever. Exposure filtering is best handled by retaining behavior type, time, and scenario, rather than just maintaining a permanent blacklist.

Large exposure sets often use a Bloom filter for approximate membership tests. It answers "definitely unseen" or "possibly seen." The second answer may be a false positive and suppress an unseen item. For `n` inserted items, `m` bits, and `k` hash functions, the false-positive rate is approximately:

```math
\left(1-e^{-kn/m}\right)^k.
```

A standard Bloom filter cannot delete individual entries. A rolling exposure window can keep one filter per day or week and discard whole expired filters. Counting Bloom filters or exact storage are alternatives when individual deletion is required. Choose `m` and `k` from an acceptable false-drop rate, not memory alone.

Recommendation systems may also cache candidates that ranked well but were not exposed after reranking. They return as a small retrieval path on the next request and leave the cache after exposure, too many retries, TTL expiry, or capacity eviction. Inventory, safety, and freshness checks still run again.

Search cache/KV retrieval utilizes stable results for high-frequency queries. It has low latency but must handle timeliness, personalization, and cache pollution. Hot news queries cannot reuse old results for long, and personalized queries should not directly share caches across all users.

### 6.6 Offline and reverse retrieval

Results for frequent queries are relatively stable, so a heavier search pipeline can run offline:

```text
frequent query
  -> expanded retrieval
  -> large relevance and quality models
  -> human/rule filtering
  -> query -> top documents cache
```

Serving reads the posting directly. Cache entries need normalization, geography, freshness, and version conditions, plus active invalidation when trends change or documents disappear.

A search log cache is more than one saved result page. For each head query, aggregate top documents from recent logs and combine appearance, click and engagement frequencies with relevance, content quality, and document age:

```text
q -> [(d, score), ...]
```

The index may be non-personalized or weakly personalized as `(q, group) -> documents`. Too many groups dilute statistics and multiply memory. Daily jobs decay and merge old and new scores. Retrieval experiments must also be excluded from cache construction; otherwise, documents discovered only in treatment leak into control.

An offline search pipeline can expand retrieval and scoring for head queries during low-traffic hours and store fine-rank relevance scores in the KV entry. Serving then skips part of non-personalized retrieval and relevance inference. Breaking or strongly fresh queries, local intent, new documents, and personalized paths must remain online.

Reverse retrieval starts from a document. At publication time, a model can generate Anchor Queries or use the document vector to search a query inventory, then filter with Cross-BERT:

```text
document -> candidate queries -> relevance filter
         -> query -> document inverted postings
```

This helps tail and new documents. Even without historical query-document co-occurrence, content models can attach the document to appropriate queries.

D2Q training pairs can come from head-query logs: keep documents that ranked high, received strong clicks, and passed the fine-rank relevance threshold. Train first on the large mined set and fine-tune on human relevance pairs. During offline generation, reject short or low-quality documents, sample several queries per document, and filter them with the relevance model. Useful diagnostics are relevance pass rate and the number of distinct high-relevance queries generated per document.

Rewriting can also feed cached retrieval:

```text
tail query q -> head rewrite q' -> cached documents d
```

This path needs both a reliable `(q,q')` rewrite and a good `(q',d)` cache entry.

Offline retrieval does not replace online retrieval. Inventory, freshness, user context, and new queries still require live processing. A typical union is:

```text
original-query online retrieval
+ rewrite online retrieval
+ offline query postings
+ cache and trending channels
```

### 6.7 How to Validate Retrieval Performance

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

### 6.8 Chapter Self-Test

1. How can query rewriting avoid semantic drift?
2. Why is content retrieval suitable for cold starts, yet cannot handle recommendation alone?
3. How to evaluate whether a low-traffic retrieval channel has independent value?
4. Why can't scores from multi-path retrieval be compared directly?
5. Why might the recall of long-tail queries still worsen when overall Recall increases?
6. How does reverse retrieval index a new document without historical exposure?
7. Why can a Bloom filter suppress an unseen item, and how can it represent a rolling window?
8. How can a new retrieval experiment contaminate a search cache?

<details>
<summary>Reference answers</summary>

1. Record rewrite source and confidence, restrict allowed rewrite types, and retain the original-query channel. Evaluate by intent slice and monitor bad cases.
2. Content can encode an item without behavior, but it lacks collaborative preference and audience feedback. Alone, it can retrieve related yet personally unsuitable items.
3. Under a fixed total candidate budget, disable the channel and measure exclusive positives, tail coverage, and downstream lift.
4. BM25, cosine similarity, co-occurrence, and rule scores have different scales. Calibrate them, fuse ranks, or let a downstream model combine them.
5. Head queries dominate traffic, so an average can hide tail regressions. Report Recall by frequency, intent, language, and other slices.
6. At publication time, generate or vector-retrieve related queries, filter them with a relevance model, and add the document to those query postings.
7. Hash collisions can leave every checked bit set for an unseen item. Keep daily or weekly filters and drop expired ones, or use a counting Bloom filter when individual deletion is required.
8. Documents found only by treatment can enter the shared cache through treatment logs and later appear in control. Exclude experimental candidates during cache construction or isolate caches by experiment.

</details>
