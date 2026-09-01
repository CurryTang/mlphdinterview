# Online Experimentation and Metric Growth

## Chapter 15: Online Experimentation and Metric Growth

Offline evaluation asks whether a model is better on fixed data. Online evaluation asks whether real users interacting with the full system are better off. Candidate changes, rules, latency, feedback loops, and product behavior sit between those questions.

### 15.1 Write the experiment question first

Start the recommendation metric tree with the funnel: `impression -> click -> read or play -> like, save, share, comment`. CTR divides by impressions; like, save, and share rates usually divide by clicks. North-star metrics sit above the funnel: DAU or MAU, consumption time or items per user, and creator-side publishing penetration and output per DAU. Offline evaluation only screens a candidate; release still moves through a small A/B test and then full rollout.

A usable design specifies:

```text
hypothesis: a GNN retrieval path adds tail positives missed elsewhere
randomization unit: user_id
primary metric: valid consumption time per user
diagnostics: unique recall, CTR, valid views
guardrails: P99, complaints, creator concentration
MDE: +0.3%
window: 14 days, covering full weekly cycles
```

Choose a small set of primary metrics before launch. Inspecting dozens and reporting only winners creates multiple-testing and reporting bias.

### 15.2 Stable bucketing

Typical user bucketing is:

```text
bucket = hash(user_id, experiment_id, salt) mod B
```

It should be stable across requests, nearly uniform, independently salted, consistent across online and offline code, and versioned.

Python's built-in `hash()` is process-randomized and unsuitable across machines. Use a fixed algorithm such as SHA-256 or MurmurHash.

The unit need not be a user. Network treatments may require clusters; seller strategies may randomize sellers; creator treatments may randomize authors. Match the unit to where treatment and interference occur.

### 15.3 Layers: exclusion within, orthogonality across

Experiment platforms often separate retrieval, pre-ranking, ranking, reranking, UI, and ads.

- Experiments in one layer are mutually exclusive when their policies replace one another.
- Experiments in different layers randomize independently and intentionally overlap.

Orthogonality is an assumption, not a law. A new retrieval path may interact strongly with a ranker that has never seen its candidates. Important combinations need factorial cells or explicit interaction tests.

### 15.4 Run A/A first

A/A tests the infrastructure without changing policy:

- sample-ratio mismatch;
- pre-experiment balance;
- missing exposure or attribution logs;
- cross-bucket users;
- unexpected metric variance;
- inconsistent bot and internal-traffic filtering.

SRM can use:

```math
\chi^2
=\sum_b \frac{(O_b-E_b)^2}{E_b}.
```

If SRM is significant, repair allocation or logging before reading business metrics. Broken randomization invalidates downstream p-values.

### 15.5 Effects, intervals, and sample size

The mean difference is:

```math
\Delta=\bar X_T-\bar X_C.
```

Report absolute and relative effects, confidence intervals, sample sizes, duration, predefined primary metrics, and guardrails.

`p < 0.05` does not guarantee business value. A non-significant result also does not prove no effect; if the interval still includes a meaningful gain, the data are inconclusive.

Estimate sample size from baseline mean and variance, significance level, power, and MDE. CUPED reduces variance with a pre-experiment covariate:

```math
Y_{\text{cuped}}
=Y-\theta(X-\mathbb E[X]),
\qquad
\theta=\frac{\operatorname{Cov}(Y,X)}
{\operatorname{Var}(X)}.
```

`X` must precede treatment. A post-treatment covariate can remove real effects or introduce bias.

### 15.6 Holdout, rollout, and reversal

Experiment layers estimate one release. A long-term holdout estimates the cumulative value of many releases by keeping users on the older policy.

After rollout, a small reversal bucket can retain the old policy:

```text
most users: new policy
reversal bucket: old policy
```

Clicks respond quickly; retention and ecosystem effects lag. Reversal frees ordinary buckets while preserving long-term comparison.

Holdouts age too. Old policies may become incompatible with the current system, so define reset cadence, version boundaries, and safety floors.

### 15.7 Why cold-start experiments are hard

New-item policies affect both consumers and creators:

- consumer metrics: new-item CTR, engagement, time, negative feedback;
- creator metrics: publishing, output, and time to first useful feedback.

User bucketing lets the same new items compete for traffic across groups. Author bucketing still makes new items compete with all mature items, and the experiment's market differs from full rollout.

Every design should ask:

1. Do treatment and control items compete for a fixed exploration budget?
2. How do new and mature items compete?
3. Does isolating users and authors shrink the content pool too much?
4. Which candidates lose exposure when quotas are guaranteed?
5. Which unit identifies consumer and creator effects?

With network interference, no split is free of tradeoffs. State whether the estimand is a local direct effect or an approximation to a full-rollout total effect.

### 15.8 Finding growth opportunities systematically

Start with the funnel.

Retrieval checks include missing positives, weak multi-behavior supervision, truncated long-term interests, duplicated channels, and accidental exposure filtering.

Ranking checks include feature coverage, pre-rank false drops, biased or immature labels, multi-objective calibration, long sequences, real-time features, and incremental training.

List checks include repetition, exploration budget, underserved cohorts, and rules that cancel model gains.

### 15.9 Special populations

New, low-activity, and minority users may differ from the population that dominates training. Three intervention levels are useful:

- content pools built for cohort quality and retention;
- policies with lighter ads, lower-risk exploration, or different fusion weights;
- a shared model with residual adapters, expert gating, or cohort calibration.

Maintaining a full large model per cohort is brittle. The main model keeps improving while neglected cohort models fall behind. A shared base with light adaptation stays current more easily.

### 15.10 Follows, shares, and comments

These interactions also change relationships, creator behavior, and off-platform return traffic.

Follows can strengthen a user's relationship with the platform and give new creators feedback. Systems can use follow-oriented pools, explicit or implicit U2A2I retrieval, and weights that decrease with existing follows or followers.

A fusion term can be written as `w(f_u) p_follow(u,i)`: users who follow few creators receive a larger `w(f_u)`. The creator-side version decreases with the author's follower count and helps new creators gain their first audience.

Shares matter when they cause off-platform return traffic. Raising share rate alone may hurt clicks, so treatment should condition on users and content that actually create return traffic.

The KOL signal in the course is return traffic caused by a user's historical shares, not that user's follower count on the current platform. Apply a share-oriented score or retrieval path only after identifying that off-platform value.

Comments can motivate creators and community discussion, but count and quality differ. Comment-oriented strategies need abuse, report, and low-quality-interaction guardrails.

New items with few comments can receive a larger comment-oriented weight. Users who consistently write highly liked comments can receive a discussion-oriented content pool. The first helps creators get feedback; the second targets community quality.

### 15.11 Chapter self-test

1. Why do experiments need different salts?
2. What do exclusion within layers and orthogonality across layers solve?
3. Why stop interpretation after significant SRM?
4. Does non-significance prove a policy is ineffective?
5. Why must CUPED covariates precede treatment?
6. Why do cold-start experiments suffer interference?
7. Why avoid a full model per special cohort?
8. Why are follows, shares, and comments not ordinary positive labels?

<details>
<summary>Reference answers</summary>

1. Salts create independent, reproducible splits instead of assigning the same users to treatment everywhere.
2. Exclusion prevents conflicting policies from co-applying; orthogonality lets distinct modules reuse the full traffic pool.
3. SRM shows that allocation, exposure, or attribution violated the randomized design, so group comparisons are unreliable.
4. No. Check whether the confidence interval excludes meaningful gains; a wide interval means insufficient evidence.
5. Treatment may change in-experiment variables. Adjusting for them can erase effects or create bias.
6. New items share traffic budgets and compete with mature content, so one group's policy changes opportunities available to another.
7. Cohort models are expensive and quickly miss main-model upgrades. A shared base plus residuals, experts, or calibration is easier to maintain.
8. Their value depends on relationships, external return traffic, creator activity, and community health; maximizing counts can create harmful interaction.

</details>
