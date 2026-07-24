# Query Recommendation

## Chapter 16: Query Recommendation

Search ranking decides what users see after searching. Query recommendation influences what they search for and runs its own retrieval, ranking, and evaluation loop.

### 16.1 Four scenarios

| Scenario | Available context | Main purpose |
| --- | --- | --- |
| Pre-search recommendation | User history, trends, context | Trigger a search that would not otherwise happen |
| SUG | Typed prefix, user, context | Reduce input cost, complete, or correct |
| SERP recommendation | Current query and user | Deepen or broaden the need |
| In-document recommendation | Current document and user | Move from content consumption into related search |

Pre-search and in-document recommendations can increase search penetration. SUG serves users who already decided to search and mainly improves efficiency. SERP recommendations increase search depth and interest breadth after a search.

### 16.2 Query inventory

Candidate queries come from:

- curated high-quality inventories;
- frequent internal queries;
- titles and OCR text from high-quality documents;
- external trends and entity dictionaries;
- generative models.

Before indexing, check safety, quality, freshness, and result supply. A highly clickable query with no relevant documents only moves the failure to the SERP.

### 16.3 SUG retrieval

SUG has a tight latency budget, so deterministic indexes remain central:

- prefix completion;
- phonetic prefix and abbreviation;
- token containment and first-character matching;
- core-term retrieval;
- homophone and input-method correction;
- generative expansion when basic channels return too little.

Aggressive channels should activate only when reliable channels are insufficient. Generated completions need co-occurrence, relevance, and supply filtering.

SUG indexes need frequent updates and fast trend insertion. Daily offline refreshes miss events that became popular minutes ago.

### 16.4 Q2Q

Q2Q retrieves related queries from a current query.

Collaborative methods treat queries as items. If many users search both `q_1` and `q_2`, the queries may share an interest:

```math
\operatorname{sim}(q_1,q_2)
=\frac{|U(q_1)\cap U(q_2)|}
{\sqrt{|U(q_1)|\,|U(q_2)|}}.
```

Swing reduces the contribution of small, tightly connected user groups, limiting fake associations caused by coordinated searches.

A dual-encoder Q2Q model uses text, category, and core terms. Positive pairs can come from clicked SERP query recommendations; negatives combine random, in-batch, and hard examples. Text models find semantic neighbors, while collaborative methods find interest relations with weak lexical overlap. Both are useful.

### 16.5 D2Q

D2Q builds:

```text
document_id -> related queries
```

Three signal families are common:

- generate queries from the document, then filter with relevance;
- reverse-retrieve a query inventory with a dual encoder;
- mine queries searched shortly after document consumption.

Generation and Cross-BERT filtering can run in a nearline queue rather than blocking publication. The index needs document and model versions so edits and expiration trigger recomputation.

Behavioral D2Q needs a time window and a cap on subsequent queries. Arbitrary searches days later were not necessarily caused by the document.

### 16.6 Combining channels by scenario

Pre-search recommendation can use:

```text
U2Q2Q: user query history -> Q2Q -> query candidates
U2D2Q: user document history -> D2Q -> query candidates
trending and editorial pools
```

Seeds should not be only the most recent N events. Keep recent events and sample balanced long-term history to retain both current interest and breadth.

SERP recommendation uses the current query for Q2Q. In-document recommendation starts with D2Q and may add one D2Q2Q hop, but each extra hop increases drift.

### 16.7 Ranking: clicks are only the first conversion

Query ranking should model at least two stages:

```text
recommended query is clicked
  -> its SERP produces a useful click or interaction
```

A tempting rare query may have high recommendation CTR and no content supply. A score may combine:

```math
s(q)
=w_1 P(\text{query click})
+w_2 P(\text{SERP success}\mid \text{query click})
+w_3 \operatorname{quality}(q)
+w_4 \operatorname{freshness}(q).
```

SUG also values click position and typed length because it is an efficiency tool. Pre-search and SERP recommendations behave more like recommendation systems and need personalization, diversity, seen-query filtering, and frequency caps.

### 16.8 Diversity

Repeated synonyms may maximize immediate click probability without broadening the user's need. MMR or rules can diversify categories, entities, and embeddings:

- keep distinct attributes of the same entity;
- keep one of pure synonyms;
- cap high-risk and strongly time-sensitive queries;
- downweight recently searched queries;
- prevent one long-term interest from occupying every pre-search slot.

Apply diversity after safety, relevance, quality, and supply thresholds. Diversity should not rescue an invalid query.

### 16.9 Evaluation

| Metric | Scenario | Meaning |
| --- | --- | --- |
| Query-list CTR | All | Whether a display receives any query click |
| Click position | SUG | How far users scan to find the target |
| Typed length | SUG | Input saved |
| SERP query CTR / first click | All | Whether search fulfills the recommendation |
| Search penetration | Pre-search, in-document | Whether more users start searching |
| Search depth | SERP | Distinct queries in a session |
| Interest breadth | Pre-search, SERP | Category expansion |

Every query-recommendation experiment should monitor downstream SERP conversion. Optimizing recommendation CTR alone produces clickbait queries.

### 16.10 Chapter self-test

1. Why do SUG and pre-search recommendation need different metrics?
2. What relationships can collaborative Q2Q find beyond text similarity?
3. Why does behavioral D2Q need a time window?
4. Why predict SERP success in query ranking?
5. Why not add unlimited D2Q2Q hops?
6. After which gates should diversity run?

<details>
<summary>Reference answers</summary>

1. SUG is an input tool, so click position and typed length measure efficiency. Pre-search recommendation aims to trigger search and is judged by CTR and penetration.
2. It finds co-interest relationships from user behavior even when the query strings and embeddings are not close.
3. Distant or numerous later searches may arise naturally rather than being triggered by the document.
4. A clicked query may still have no relevant documents. SERP success penalizes attractive queries with weak supply.
5. Every hop expands candidates and noise, gradually moving away from the source document.
6. After safety, relevance, quality, and supply gates; diversity should only reorder valid candidates.

</details>
