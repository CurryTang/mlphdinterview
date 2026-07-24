# Search Experience and Evaluation

## Chapter 7: Search Experience and Evaluation

Document CTR alone cannot determine search quality. A click may lead to an immediate abandonment, while a direct-answer card may satisfy the user without a click. Evaluation starts by separating the factors behind satisfaction.

### 7.1 Five sources of satisfaction

Search ranking commonly combines:

1. relevance: whether the document answers the query;
2. content quality: whether the source and content are useful and trustworthy;
3. freshness: whether the user needs recent information;
4. personalization: preferences among relevant results;
5. geography: compatibility with the requested or current location.

These signals should not disappear into an ambiguous `quality_score`. Their labels, update rates, and failure modes differ, so they should be produced separately and fused later.

### 7.2 Content quality

The course uses EAT: expertise, authoritativeness, and trustworthiness. Google's later public terminology often uses E-E-A-T, adding experience. The engineering point is the same: source and author quality matter more for medical, financial, and legal YMYL queries than for casual entertainment.

Quality models often produce several scores:

- author or site reputation;
- factuality and expertise;
- originality and information density;
- clickbait, keyword stuffing, and advertising risk;
- image or video quality;
- safety, fraud, and harmful-content risk.

Most can be computed offline or nearline when a document is published and stored in the document profile. The ranker reads those scores instead of rerunning expensive models for every query.

### 7.3 Freshness

Freshness does not mean that newer is always better.

- breaking freshness: query or publication volume spikes after an event;
- general freshness: the text itself indicates a preference for recent information;
- periodic freshness: recurring events become fresh in a known window;
- no freshness need: stable knowledge and historical material.

Breaking freshness relies mainly on data mining. Query text alone cannot tell whether a match or policy announcement happened minutes ago. General freshness can use semantic models.

All downstream stages must support the signal:

```text
freshness intent
  -> new-document or trending retrieval
  -> quality and relevance filtering
  -> age decay or freshness boost
  -> expiration detection
```

A publication-time feature in the final ranker cannot recover breaking documents omitted by retrieval.

### 7.4 Geography

Geographic search distinguishes locations in the query from the user's current location:

```text
"coffee in Shanghai Xuhui" -> explicit location
"coffee nearby"            -> device location
"hotels near Disneyland"   -> POI with implicit local intent
```

The path includes POI recognition, place normalization, geographic retrieval, distance features, and ranking. Distance is not always dominant: it matters greatly for `"gas station nearby"` but less for a travel guide.

A geographic model needs at least two outputs: the strength of local intent and sensitivity to distance. `"Food nearby"` is close to a hard local constraint, while `"hot pot"` may need only some local results. Walking to a restaurant and driving on a day trip also imply different distance ranges. These outputs control local-result quotas and distance decay separately.

Experiments should slice by location permission, city tier, and explicit versus implicit local intent. Overall averages hide effects because many users do not grant location access.

### 7.5 Personalization

Broad queries such as `"avatar"` have many relevant documents and benefit from personalization. Precise queries already state the need, so long-term profiles should not override them.

A useful rule is:

> Personalization orders relevant candidates; it does not make irrelevant content relevant.

CTR and engagement models also have non-personalized value because relevant, high-quality documents tend to earn stronger feedback. They complement semantic models, but position bias and clickbait must be controlled.

### 7.6 North-star, process, and human metrics

Search evaluation has three layers:

| Layer | Metrics | Question |
| --- | --- | --- |
| North star | Search DAU, penetration, LT7/LT30 | Do users keep using search? |
| Process | Query CTR, first-screen CTR, first-click rank, reformulation, engagement | Which part of one search improved? |
| Human | GSB, relevance grades, DCG/NDCG | How does quality change without reliable behavior labels? |

Search penetration is `Search DAU / DAU`. A single algorithm experiment rarely moves retention in a short window, so process metrics diagnose quickly while a long-term holdout measures cumulative value.

Retention needs an explicit window. "Return within `n` days" asks whether today's search users return at least once during the next `n` days; "day-`n` retention" checks only day `n`. The first is non-decreasing with `n`, the second usually falls, and neither is final until its window matures. A higher rate can also come from losing low-activity users, so read it with Search DAU and cohort size.

### 7.7 Document CTR and query CTR

Document CTR uses document impressions:

```math
\operatorname{DocCTR}
=\frac{\#\text{document clicks}}
{\#\text{document impressions}}.
```

Query CTR measures whether a search had at least one click:

```math
\operatorname{QueryCTR}
=\frac{\#\text{searches with at least one click}}
{\#\text{searches}}.
```

Adding many similar documents may increase total clicks without helping users find an answer. For tool-like search, query CTR and first-screen query CTR are often closer to success than per-document CTR.

Average first-click rank must assign a capped rank to no-click searches. Dropping them leaves an average computed only over successful requests.

An effective click can require either dwell time above a threshold or a later action such as a save or follow. This removes some accidental clicks and clickbait, but thresholds need calibration by content type.

Browse depth is the last result position exposed during a search. It is often better when smaller because users found an answer near the top. Exploratory queries naturally go deeper, so compare it within intent slices rather than as one global average.

### 7.8 Reformulation and engagement

Rapidly changing `"mechanical keyboard"` to `"mechanical keyboard for women"` may indicate weak personalization. Active reformulation excludes system-suggested rewrites and uses a time window plus query similarity.

Likes, saves, follows, and purchases are stronger than clicks but sparser. An experiment can combine them into a stable metric based on their relationship with retention, while keeping individual guardrails so one frequent action does not hide regressions elsewhere.

The denominator changes the interpretation. Likes per search measure how much engagement one query ultimately creates; likes per click are closer to post-click content quality.

Queries per active search user are ambiguous by themselves. Better results can reduce reformulation and lower the count, while stronger search habits can raise it. Interpret the metric with active reformulation, Search DAU, and session depth.

### 7.9 Side-by-Side and GSB

Side-by-Side evaluation fixes the query, user profile, and context, then shows old and new result pages. Annotators label the new strategy:

- Good;
- Same;
- Bad.

A common aggregate is:

```math
\operatorname{GSB}
=\frac{G-B}{G+S+B}.
```

Sampling should cover head, torso, tail, fresh, geographic, and high-risk queries. Pure log sampling is dominated by head traffic.

Human evaluation finds relevance and quality failures but cannot replace A/B testing. It lacks real user choice, latency effects, personalization feedback, and long-term behavior.

It also has operational limits. A study may contain only a few hundred queries, depends on a precise rubric and experienced raters, and becomes harder when results are strongly personalized. Use blinded presentation, repeated labels, and agreement checks; one positive GSB run is weak launch evidence.

### 7.10 Chapter self-test

1. Why model relevance, quality, and freshness separately?
2. Why can BERT not determine breaking freshness from text alone?
3. How do the denominators of query CTR and document CTR differ?
4. How should average first-click rank handle no-click searches?
5. Why must the query dominate personalization?
6. Can GSB replace an online A/B test?
7. How does "return within seven days" differ from day-7 retention?
8. Why is lower browse depth often better, and why must it be sliced by intent?

<details>
<summary>Reference answers</summary>

1. The labels, update rates, and failure modes differ. Separate signals support scenario-specific fusion, debugging, and guardrails.
2. Breaking news depends on current world state, which requires query volume, publication volume, or external trend signals.
3. Document CTR divides by document impressions; query CTR divides by search requests and asks whether each request had any click.
4. Assign no-click and beyond-threshold requests the same capped rank so failures remain in the metric.
5. The query states the current need. Profiles may reorder relevant results but should not promote irrelevant documents.
6. No. GSB is a controlled human comparison; it cannot measure actual behavior, service cost, or long-term effects.
7. The first asks whether today's users return at least once during the next seven days; the second checks only the seventh day. The first is at least as large, and both require a mature window.
8. Finding an answer near the top usually ends browsing, but exploratory queries naturally expose more results. A global average mixes those behaviors.

</details>
