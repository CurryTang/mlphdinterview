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

<details>
<summary>Reference answers</summary>

1. It joins the request, stage-by-stage candidates, final exposure, and delayed feedback into one replayable event.
2. It loses retrieval channel, coarse-ranking false negatives, filter reasons, and rank transitions, leaving no way to localize funnel loss.
3. A user who has not purchased on day one may still convert during days two through seven. Labeling early creates systematic false negatives.
4. The user must have had a real chance to see the position, with position, scroll depth, and interruptions accounted for. An unexposed item is not an observed negative.
5. It ensures a historical sample reads only feature values available at event time, preventing future behavior or newer profiles from leaking backward.

</details>
