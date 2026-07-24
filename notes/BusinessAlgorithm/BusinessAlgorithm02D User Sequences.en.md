# User Behavior Sequences

## Chapter 12: User Behavior Sequences

### 12.1 What Does Average Pooling Lose?

A user has viewed basketball, cooking, music, and travel content. Averaging all item embeddings yields a fuzzy "overall interest," but it doesn't know which part of the history is relevant to the current candidate, nor does it account for temporal order.

Sequence models primarily solve three things:

- Different behaviors have different weights;
- The current candidate needs to read different parts of the history;
- Interests evolve over time.

### 12.2 Last-N

The simplest approach takes the last N behaviors. It is inexpensive and often stronger than complex models suggest.

One can add:

- Behavioral type weights;
- Time decay;
- Deduplication and continuous playback compression;
- Effective view thresholds;
- Category or author grouping.

For Last-N, bigger is not always better. Long histories introduce noise, storage, and service costs, and may re-amplify old interests.

### 12.3 DIN

DIN uses the candidate item `q` to query historical behaviors `h_j`:

```math
\alpha_j
=\operatorname{MLP}
(h_j,q,h_j-q,h_j\odot q),
```

```math
u(q)=\sum_j\alpha_j h_j.
```

The same user will get different interest representations when facing basketball shoes versus a wok. When remembering DIN, grasp that "user representation depends on the candidate"; attention is merely the implementation means.

The cost is also here: every candidate must interact with the history. When there are many candidates and long sequences, the computational load rises rapidly.

### 12.4 SIM

SIM processes long sequences in two steps:

1. Coarsely select sub-sequences related to the candidate from a very long history;
2. Perform fine-grained attention modeling on the sub-sequences.

Hard Search keeps behaviors in the candidate's category. Soft Search uses the candidate vector to retrieve top-k neighbors from the user's history. Hard search is cheap and stable; soft search is semantically flexible but needs a per-user sequence index.

Long-term behavior also needs time-gap embeddings. Two clicks in the same category should not receive the same weight when one happened yesterday and the other two years ago.

The logic is the same as the retrieval-ranking funnel: filter cheaply first, then interact expensively. Without efficient retrieval, long-sequence models are difficult to deploy online.

### 12.5 Temporal Issues in Training

The most dangerous bug in sequence models is time leakage.

If a sample occurs at time `t`, the history can only use behaviors visible before `t`. Statistical features, profiles, and item popularity must also be cut off at `t`. If one reads the full history of the current user offline, the metrics will be unrealistically good.

One must also handle:

- Label leakage within the same session;
- Repeated exposure;
- Out-of-order behavior logs;
- Delayed arrival;
- Negative feedback and ineffective views;
- Inconsistencies between training truncation and online truncation.

### 12.6 Real-Time Updates

The value of sequence models often comes from the most recent behaviors. If a user just finished watching a skiing video, and the feature service only updates five minutes later, no matter how complex the model is, it cannot react.

Common practices include:

- Offline storage for long-term sequences;
- Streaming updates for short-term behaviors;
- Online concatenation and deduplication;
- Degradation for missing or late behaviors;
- Recording feature versions for replay.

Model parameters can also update incrementally: train a full model on a complete window overnight, then consume fresh logs for small hourly updates. This shortens response time to interest shifts but introduces delayed labels, catastrophic forgetting, and rapid propagation of bad data. Serving must be able to fall back to the latest full checkpoint and track full and incremental data versions separately.

### 12.7 Chapter Self-Test

1. What information does Last-N average pooling lose?
2. Why does DIN's user representation depend on the candidate?
3. Why does SIM use a two-stage process for long sequences?
4. How can one check if sequence features have time leakage?
5. How to degrade when online short-term behavior updates fail?

<details>
<summary>Reference answers</summary>

1. Mean pooling loses order, time gaps, repetition, and interest transitions.
2. DIN uses the candidate as the attention query, so the same user receives a different representation for each candidate.
3. SIM cheaply retrieves a candidate-relevant subsequence from long history, then models that shorter sequence in detail.
4. Verify every event precedes the request and replay features with a point-in-time join; aggregation windows must not include future events.
5. Fall back to an older versioned sequence or long-term features, record the degradation rate, and do not disguise missing data as a genuine empty history.

</details>
