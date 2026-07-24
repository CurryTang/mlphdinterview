# System Overview

> **Acknowledgments**: The foundational framework and several conceptual explanations in this section are inspired by Professor Wang Shusen's Recommender Systems course. We are grateful to Professor Wang for his long-term commitment to sharing high-quality educational content publicly.

## Chapter 1: The Multi-Stage Pipeline of Recommendation and Search

```business-algorithm-map
```

This diagram organizes the subsequent content according to the execution sequence of a single request. By toggling between traditional cascading and generative paths, one can observe the candidate scale, inputs/outputs, and latency at each layer. At the bottom lies the shared data feedback loop.

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

These numbers vary with the product and latency budget, but the layering logic remains constant: the earlier the stage, the larger the candidate pool and the less computation available per candidate; the later the stage, the smaller the pool and the more granular the model interactions and list constraints.

Search uses a similar funnel, but the query is a strong input. Recall must handle exact keywords, semantics, and attribute conditions simultaneously, while ranking must maintain query-item relevance. Recommendation primarily infers intent from history and context, allowing for a small number of exploration slots.

### 1.2 Why Traditional Architectures Adopt Multi-Stage Pipelines

The most direct reason traditional systems decouple recall, pre-ranking, ranking, and re-ranking is the inability to distribute computational power evenly. Inverted indices, similarity tables, and ANN can quickly retrieve thousands of items from a massive library; real-time features and complex cross-features must be reserved for the final few hundred candidates.

The cost of errors also differs across layers. If recall misses an item, subsequent models have no chance to recover, so this layer prioritizes coverage. Ranking deals with pre-selected candidates, where the goal is to distinguish subtle preferences.

Inventory, location, safety, ad frequency control, and author deduplication are usually deterministic. They remain outside the model for practical reasons: rules require auditing, hot-fixing, and independent rollback.

Cascading architectures also have costs. Different stages are trained separately, potentially leading to conflicting objectives; errors in early stages cannot be recovered; and multiple sets of features, models, and RPCs increase maintenance and communication costs. End-to-end generative systems address these issues, though the scope of "end-to-end" varies across representative works.

- [HSTU / Generative Recommenders](https://proceedings.mlr.press/v235/zhai24a.html) reformulates recommendation as sequence transduction, using a scalable model to predict subsequent actions or items based on user action sequences. It unifies the recommendation modeling approach and training expansion path. The paper does not claim that inventory, safety rules, or all online candidate services have disappeared as a result.
- [OneRec](https://arxiv.org/abs/2502.18965) explicitly replaces the retrieve-and-rank cascade with a single encoder-decoder, generating session-level item lists directly from user history and using DPO for preference alignment. It unifies recall, pre-ranking, and ranking objectives, but still requires Semantic IDs, a reward model, constrained decoding, and online rule integration.
- [OneSearch](https://openreview.net/forum?id=JKGgHY9FKa), aimed at e-commerce search, feeds queries, user behavior, and item Semantic IDs into a unified generative framework, replacing the traditional recall-pre-ranking-ranking funnel. Search must still handle query-item strong relevance, inventory, and product attributes; the paper's system also retains item encoding and additional reward model selection.

When discussing "end-to-end," it is essential to define the boundaries: which training objectives and online stages are unified by the model, and which indices, rules, and disaster recovery paths remain external. Otherwise, the same term may refer to entirely different systems.

### 1.3 Decomposing the System by Request Order

Subsequent chapters unfold according to the online sequence in Figure 1-1:

| Stage | Input | Output | Future Coverage |
| --- | --- | --- | --- |
| Data & Request | User, query, item, context, logs | Reproducible samples & features | Chapter 2 |
| Recall | Full library, inverted index, similarity table, vector index | Thousands of candidates & channel attribution | Chapters 3-5 |
| Pre-ranking & Ranking | Candidates, real-time features, cross-features | Candidate scores & truncated lists | Chapters 6-9 |
| List Decision | Ranking results, rules, and exploration budget | Final display list | Chapters 10-11 |
| Generative Path | Query/history and item/doc identifiers | Generated items, SIDs, or answers | Chapters 12-14 |
| System Validation | Full-chain logs, versions, and experiment traffic | Launch conclusions & troubleshooting | Chapter 15 |

When reading papers or designing solutions, ask four questions along the table: What is the input, which layer is being replaced, to whom is the output passed, and how to downgrade upon failure? Network architecture is only one of these items.

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
