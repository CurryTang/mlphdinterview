# Data, Samples, and Feature Streams

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

### 2.7 Large-Scale Sparse Embedding Systems & Infrastructure

Industrial recommendation models exhibit extreme architectural heterogeneity:
- **Sparse Embedding Side (Memory-Bound)**: Categorical ID cardinalities (user IDs, item IDs, queries, historical sequence IDs) reach hundreds of billions, scaling parameters from hundreds of gigabytes to tens of terabytes. Operations are computationally trivial (table lookups and pooling) but far exceed GPU VRAM capacities;
- **Dense MLP Side (Compute-Bound)**: Dense feed-forward and multi-head attention components contain only tens of megabytes to a few gigabytes of parameters. Compute is heavily matrix-multiplication intensive, ideally suited for GPU Tensor Cores.

#### 1. Hybrid Parallelism: Parameter Server (PS) vs GPU Table Sharding
To train terabyte-scale recommendation models (e.g., Meta DLRM, ByteDance Monolith, Alibaba WDL), modern distributed training couples **Model Parallelism** with **Data Parallelism**:
- **Model Parallelism (Table Sharding)**: Enormous embedding tables are partitioned across workers (via consistent hashing or domain-based sharding) across distributed CPU host memory (Parameter Server) or high-memory GPU clusters;
- **Data Parallelism (Dense Layers)**: Every worker replica maintains an identical copy of the dense neural network, processing independent micro-batches of training instances;
- **All-to-All Collective Bottleneck**: After workers look up local embedding shards, an `All-to-All` collective communication pass (over NVLink or RoCE) must transpose and route embedding slices to the respective dense GPU worker responsible for forward-pass feature concatenation.

#### 2. Terabyte-Scale Dynamic Feature Admission and Eviction
Unbounded feature growth inevitably causes Out-Of-Memory (OOM) crashes and severe overfitting on tail noise:
- **Admission Control**:
  - The overwhelming majority of tail IDs (one-time visitors or transient items) represent random noise; allocating parameters to them wastes capacity and overfits;
  - Systems maintain probabilistic sketches (**Bloom Filters** or **Count-Min Sketches**) to track ID occurrence frequencies in sliding windows;
  - An embedding vector is formally allocated in persistent memory only after an ID's observation count crosses a threshold (e.g., $k \ge 5$).
- **Eviction Policy**:
  - Combined with **$L_2$ Weight Decay**: Inactive or cold IDs experience continuous norm shrinkage toward zero over training epochs;
  - Asynchronous background sweeps run **LRU / LFU cache eviction**, reclaiming memory from dormant embeddings whose norms fall below an $\epsilon$ threshold (e.g., collision-free dynamic hash tables in Monolith), enforcing fixed memory bounds.

### 2.8 Datasets Are Split by Time

Recommendation and search logs have a clear chronological order. Random shuffling will leak future popular items, subsequent user behavior, or new feature versions into the training set.

```text
Training Window      Validation Window     Test Window
──────────────|───────────|───────────>
              Freeze feature and sample construction versions
```

Each experiment should be able to answer: Which log segment was used, when labels matured, how negative samples were collected, which version of feature snapshots was used, and which strategy generated the candidates. Without this information, offline results are difficult to reproduce.

### 2.9 Chapter Self-Test

1. Why must `request_id` persist throughout the entire online pipeline?
2. What recall and pre-ranking issues are missed if only the final exposure is recorded?
3. Why do 7-day conversion labels require a maturity window?
4. Why do large-scale recommendation systems require a hybrid "Embedding Model Parallelism + MLP Data Parallelism" architecture?
5. What core engineering challenges do dynamic feature admission and eviction solve?
6. What problem does a point-in-time join solve?

<details>
<summary>Reference answers</summary>

1. It joins the request, stage-by-stage candidates, final exposure, and delayed feedback into one replayable event.
2. It loses retrieval channel, coarse-ranking false negatives, filter reasons, and rank transitions, leaving no way to localize funnel loss.
3. A user who has not purchased on day one may still convert during days two through seven. Labeling early creates systematic false negatives.
4. Terabyte-scale sparse embedding tables are memory-bound and exceed GPU VRAM, requiring model parallelism via distributed table sharding. Dense MLPs are compute-bound and require data parallelism across GPUs. An All-to-All collective bridges the two stages.
5. They prevent memory OOM and curb overfitting on low-frequency noise IDs. Bloom filters prevent transient tail IDs from allocating parameters, while L2 decay and LRU/LFU evict obsolete cold features.
6. It ensures a historical sample reads only feature values available at event time, preventing future behavior or newer profiles from leaking backward.

</details>
