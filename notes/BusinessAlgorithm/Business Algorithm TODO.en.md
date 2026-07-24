# Business Algorithms: System Overview and Data Foundations

> **Acknowledgments**: The foundational framework and several conceptual explanations in this section are inspired by Professor Wang Shusen's Recommender Systems course. We are grateful to Professor Wang for his long-term commitment to sharing high-quality educational content publicly. The authors of this site are responsible for the reorganization, expansion, and any potential omissions in these notes.

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

---

## Chapter 2: Data, Samples, and Feature Streams

Training tables are derived from three types of raw records: online requests, candidate trajectories at each layer, and user feedback arriving with latency. Only after aligning these by time and version can one discuss recall pairs, exposure samples, or list preferences.

### 2.1 Requests are the Primary Key for All Data

A single request passes through multiple services. The `request_id` must persist through query understanding, recall, ranking, re-ranking, and exposure to reconstruct at which layer an item entered and where it was dropped.

Common fields for recommendation requests:

```text
request_id, user_id, session_id
scene, timestamp, device, region
recent_actions
experiment_ids
```

Search also requires:

```text
raw_query, normalized_query
query_intent, filters
rewrite_source
```

User profiles and historical sequences should not be copied in full into every log entry. Logs should record snapshot versions or feature keys, and be reconstructed via point-in-time joins when needed.

### 2.2 Candidate Trajectories Must Be Replayable Layer by Layer

Recording only the final exposure is insufficient. Recall tuning requires knowing which channel each candidate came from; pre-ranking requires knowing which candidates were prematurely discarded; rule troubleshooting requires saving the reason for filtering.

A candidate trajectory contains at least:

| Field | Purpose |
| --- | --- |
| `request_id, item_id` | Connects the same candidate within a single request |
| `stage, channel` | Marks recall, pre-ranking, ranking, re-ranking, and recall source |
| `raw_score, calibrated_score` | Replays model scores and fusion process |
| `rank_before, rank_after` | Observes how each layer changes positions |
| `filter_reason` | Distinguishes reasons for deletion (inventory, safety, viewed, quota, etc.) |
| `model/index/rule_version` | Locates releases and version mismatches |
| `event_time` | Ensures time-slicing and feature replay |

Saving all candidates at every layer may be too expensive; intermediate trajectories can be sampled, while final exposures, critical filtering, and experiment traffic should retain complete records.

### 2.3 Feedback Must Mature Over Time

Clicks can be observed shortly after exposure, but purchases, refunds, and next-day retention take longer:

```text
t0 Request and exposure
t1 Click / Skip
t2 Add to cart / Play completion
t3 Order placement
t4 Refund or retention window ends
```

Training samples must declare a label cutoff time. For example, when predicting 7-day purchases, today's exposures cannot be treated as negative examples tomorrow. A common practice is to use only samples that have passed the maturity window or to model immature labels separately.

Behavioral intensity should not be conflated into a single `label`. Clicks, effective plays, likes, purchases, and reports have different meanings; subsequent multi-objective models will consume these fields separately.

### 2.4 Non-Clicks Are Not Automatically Negative Samples

Users can only provide feedback on content displayed by the system. Non-recalled items have no observed labels, and exposure without a click may occur because the position was too low, the page was not scrolled, or the network was interrupted.

Several types of bias need to be addressed:

- Selection bias: Old recall and ranking strategies determine who enters the logs;
- Position bias: Top candidates naturally receive more interaction;
- Popularity bias: High exposure continues to accumulate more positive feedback;
- Survivorship bias: Content filtered out by rules disappears from the training set;
- Delayed feedback: Labels are misrecorded as negative before they mature.

Random exposure, exploration buckets, propensity weighting, and delay correction can mitigate some of these issues. Negative sampling only changes the training distribution and cannot conjure real feedback for candidates that were never displayed.

### 2.5 Different Stages Require Different Samples

| Module | Training Sample | Common Positive Examples | Candidate or Negative Examples |
| --- | --- | --- | --- |
| Collaborative Recall | User history and next item | Clicks, effective plays, purchases | Full-library sampling, in-batch, or hard negatives |
| Search Recall | Query and document/item | Human-labeled relevance or high-quality interaction | Hard examples from BM25/old models |
| Ranking | User/query-item in an exposure | Clicks, duration, conversion | Uninteracted candidates in the same exposure |
| List Model | Request and complete slate | List-level feedback | Other permutations or strategy-generated lists |
| Generative Rec | History sequence and item/SID sequence | Next item or preference list | Vocabulary competitors, rejected, rollout |

The same user behavior can enter multiple tasks, but the sample units differ. Mixing recall pairs, ranking exposure samples, and list preferences into a single table usually loses the candidate source and list context.

### 2.6 Features Are Joined Based on Availability Time

Features are roughly divided into four groups:

- User: Long-term profile, recent behavior, activity level;
- Query or scenario: Terms, intent, time, location, entry point;
- Item/doc: Content, category, author, quality, inventory;
- Cross: User's historical preference for categories, query-item text matching, real-time statistics in context.

During training, only values that existed before the event occurred can be read. If you use the final daily sales to predict clicks from that morning, time travel has occurred. A reliable feature platform must support point-in-time joins and record feature generation time, schema, and default values.

Handling of missing values online must also be consistent with training. If missing samples are deleted during training but filled with zeros during service, the model will encounter a new input distribution after deployment.

### 2.7 Datasets Are Split by Time

Recommendation and search logs have a clear chronological order. Random shuffling will leak future popular items, subsequent user behavior, or new feature versions into the training set.

```text
Training Window      Validation Window     Test Window
──────────────|───────────|───────────>
              Freeze feature and sample construction versions
```

Each experiment should be able to answer: Which log segment was used, when labels matured, how negative samples were collected, which version of feature snapshots was used, and which strategy generated the candidates. Without this information, offline results are difficult to reproduce.

### 2.8 Chapter Self-Test

1. Why must `request_id` persist throughout the entire online pipeline?
2. What recall and pre-ranking issues are missed if only the final exposure is recorded?
3. Why do 7-day conversion labels require a maturity window?
4. Under what conditions is an exposure without a click suitable as a negative example?
5. What problem does a point-in-time join solve?
