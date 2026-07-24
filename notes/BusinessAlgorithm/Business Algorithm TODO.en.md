# System Overview

> **Acknowledgment**: This section builds on Professor Wang Shusen's public courses *Industrial Recommender Systems* and *Search Engine Technology*, along with their lecture notes.

## Chapter 1: The Multi-Stage Pipeline of Recommendation and Search

```business-algorithm-map
```

The diagram follows one request from input to response. Switch between the cascade and generative paths to compare candidate counts, inputs, outputs, and latency. Both paths feed the same logging and training loop at the bottom.

### 1.1 Starting with a Single Request

The online pipeline begins with a request carrying a unique `request_id`. A recommendation request contains at least the user, scenario, time, device, and recent behavior; search requests also include the query, filtering conditions, and location. A simplified input set is:

```text
request_id
user_id / session_id
query or current entry point
recent behavior sequence
time, location, device, experiment bucket
```

Taking a short-video homepage as an example, the system might return 20 items from a pool of hundreds of millions within 150 ms:

```text
Hundreds of millions of items
  -> Recall 3,000 items via ItemCF / Two-Tower / Followed / Trending channels
  -> Deduplication, filter viewed, safety and location filtering, leaving 1,800 items
  -> Pre-ranking retains 300 items
  -> Ranking calculates scores for clicks, duration, negative feedback, etc., retaining 80 items
  -> Re-ranking controls author repetition, exploration, and ad placement
  -> Return 20 items and log complete exposure
```

The numbers depend on the product and its latency budget. The tradeoff is stable: early stages touch more candidates with cheap computation; later stages spend more work on fewer candidates.

Search uses a similar funnel, but the query is a strong input. Recall must handle exact keywords, semantics, and attribute conditions simultaneously, while ranking must maintain query-item relevance. Recommendation primarily infers intent from history and context, allowing for a small number of exploration slots.

### 1.2 Why Traditional Architectures Adopt Multi-Stage Pipelines

Traditional systems split retrieval, pre-ranking, ranking, and reranking because they cannot afford the same computation on every item. Inverted indexes, similarity tables, and ANN can retrieve thousands of candidates quickly. Real-time features and expensive feature interactions are reserved for the final few hundred.

The cost of errors also differs across layers. If recall misses an item, subsequent models have no chance to recover, so this layer prioritizes coverage. Ranking deals with pre-selected candidates, where the goal is to distinguish subtle preferences.

Inventory, location, safety, ad frequency control, and author deduplication are usually deterministic. They remain outside the model for practical reasons: rules require auditing, hot-fixing, and independent rollback.

Cascading architectures also have costs. Different stages are trained separately, potentially leading to conflicting objectives; errors in early stages cannot be recovered; and multiple sets of features, models, and RPCs increase maintenance and communication costs. End-to-end generative systems address these issues, though the scope of "end-to-end" varies across representative works.

- [HSTU / Generative Recommenders](https://proceedings.mlr.press/v235/zhai24a.html) reformulates recommendation as sequence transduction, using a scalable model to predict subsequent actions or items based on user action sequences. It unifies the recommendation modeling approach and training expansion path. The paper does not claim that inventory, safety rules, or all online candidate services have disappeared as a result.
- [OneRec](https://arxiv.org/abs/2502.18965) explicitly replaces the retrieve-and-rank cascade with a single encoder-decoder, generating session-level item lists directly from user history and using DPO for preference alignment. It unifies recall, pre-ranking, and ranking objectives, but still requires Semantic IDs, a reward model, constrained decoding, and online rule integration.
- [OneSearch](https://openreview.net/forum?id=JKGgHY9FKa), aimed at e-commerce search, feeds queries, user behavior, and item Semantic IDs into a unified generative framework, replacing the traditional recall-pre-ranking-ranking funnel. Search must still handle query-item strong relevance, inventory, and product attributes; the paper's system also retains item encoding and additional reward model selection.

When a paper says "end-to-end," check its boundary: which objectives and online stages moved into the model, and which indexes, rules, and failover paths stayed outside. The phrase is otherwise too ambiguous to compare systems.

### 1.3 Decomposing the System by Request Order

The remaining chapters follow the online sequence in Figure 1-1:

| Stage | Input | Output | Future Coverage |
| --- | --- | --- | --- |
| Data & Request | User, query, item, context, logs | Reproducible samples & features | Chapter 2 |
| Query & Retrieval | Full corpus, inverted index, similarity table, vector index | Thousands of candidates & channel attribution | Chapters 3-6 |
| Experience & Ranking | Candidates, real-time features, cross-features | Candidate scores & truncated lists | Chapters 7-12 |
| List & Cold Start | Ranking results, rules, and exploration budget | Final list and exploration traffic | Chapters 13-14 |
| Experiments & Query Recommendation | Experiment traffic and user/query/item signals | Launch conclusions and recommended queries | Chapters 15-16 |
| Generative Path | Query/history and item/doc identifiers | Generated items, SIDs, or answers | Chapters 17-19 |
| System Design | Full-chain logs, versions, capacity, and fallbacks | End-to-end design and diagnosis | Chapter 20 |

For any paper or design, ask four questions: What is the input? Which stage does it replace? Who consumes its output? What happens when it fails? The network architecture answers only part of the design.

### 1.4 Offline and Online Boundaries

The lower part of Figure 1-1 represents the offline and near-real-time pipelines. ItemCF similarity tables, item vectors, Semantic IDs, and inverted indices are built in advance; user short-term behavior, current queries, inventory, and device status are read at request time. Both sides are aligned via version numbers:

```text
model_version
feature_schema_version
index_version
rule_version
experiment_id
```

If an exposure lacks these fields, it is difficult to determine whether a change in metrics is caused by updates to the model, index, features, or rules.

An architectural diagram suitable for troubleshooting must also indicate:

1. Offline logging and sample generation;
2. Model training and index construction;
3. Online feature services;
4. Recall, ranking, and re-ranking;
5. Exposure and interaction log feedback.

If exposure and interaction logs are not fed back, training samples, online metrics, and version releases cannot be aligned.

### 1.5 Chapter Self-Test

1. Given a recommendation request, how do you map the process of candidate scale reduction layer by layer?
2. Why do traditional systems decouple recall, pre-ranking, ranking, and re-ranking?
3. What are the differences in request inputs between search and recommendation?
4. Why must exposure logs record the recall channel, position, and version?
5. What does "end-to-end" unify in the context of HSTU, OneRec, and OneSearch, respectively?

<details>
<summary>Reference answers</summary>

1. Start with corpus size and the latency budget, then annotate candidate counts after retrieval, filtering, coarse ranking, fine ranking, and reranking, for example `10^8 → 3000 → 300 → 80 → 20`.
2. Complex interaction models cannot run over billions of items. Retrieval favors coverage, ranking separates candidates, and reranking enforces list constraints; separate stages can also scale and roll back independently.
3. Search has an explicit query, filters, and a strong relevance constraint. Recommendation infers intent from history and context and usually reserves some exploration capacity.
4. These fields explain where a candidate came from, why it occupied a position, and which model and index produced it. Without them, channel attribution and request replay are impossible.
5. HSTU unifies recommendation sequence modeling; OneRec jointly generates retrieved items and a session list; OneSearch unifies search retrieval, pre-ranking, and ranking. Inventory, safety, indexing, and failover remain system concerns.

</details>
