# Sparse and Collaborative Retrieval

## Chapter 3 Sparse Retrieval and Collaborative Retrieval

Retrieval is not responsible for choosing the final first result. Its job is to collect a set of items that may matter downstream, subject to a latency and candidate budget. A missed positive is more costly here than an imperfect order: a ranker can reorder retrieved items, but it cannot score an item that never entered the candidate set.

A retrieval result should therefore carry more than an `item_id`. It should include its channel, raw score, and triggering seed so that later stages can fuse it and offline analysis can explain why it appeared. The value of a channel is not that it has the highest standalone accuracy. It is how many positives it adds that the other channels miss while the total candidate budget stays fixed.

This chapter starts with two forms of evidence that avoid expensive online interaction: lexical relations in search and behavioral co-occurrence in recommendation. Both are cheap and stable, and both have a hard boundary. If a term has no posting or no co-occurrence edge exists, the system cannot follow that relation. Vector retrieval in Chapter 4 fills part of this gap with continuous representations.

### 3.1 Two Types of Sparse Evidence

Term co-occurrence in search and behavioral co-occurrence in recommendation look different, but their computational habits are quite similar.

Search asks: "Which documents contain the important terms from the query?"

Recommendation asks: "Which items are consumed by similar groups of people who also interacted with the user's items?"

Both establish inverted relationships in advance, finding a batch of candidates online from a small number of keys. The difference lies in the semantics of the key: search uses terms, ItemCF uses items, and UserCF uses similar users.

### 3.2 Inverted Index

A forward index is:

```text
doc_id -> document content
```

An inverted index is:

```text
term -> [(doc_id, tf, positions), ...]
```

`tf` is the frequency of the term in the document, and `positions` records the locations, which can be used for word distance and phrase matching. Online queries first tokenize the input, then retrieve multiple posting lists to perform intersection, union, or more complex traversals.

The strengths of an inverted index are:

- Precise term matching is stable and interpretable;
- Index compression and skip list technologies are mature;
- Incremental document updates are relatively straightforward;
- It is highly reliable for precise queries such as entity names, model numbers, and codes.

Its weaknesses are also straightforward: it is difficult to retrieve items when there is no literal overlap. "LV bag" and "LOUIS VUITTON handbag" are semantically close, but pure term-based retrieval may not know this.

The most common production failures are not in the formula. The tokenizer, dictionary, and index must use compatible versions. If serving tokenizes `"iPhone15"` as one term while the old index stored `"iPhone" + "15"`, recall may drop to zero. Postings for frequent terms can be too long to traverse fully, so systems use skip structures, WAND/Block-Max WAND, or stricter term combinations to prune work. Deletions often write tombstones first and reclaim space during segment merges; removing a document from forward storage without invalidating its postings keeps returning an unavailable result.

AND versus OR is not merely an implementation choice either. AND is precise but can empty the result set when one term is missing. OR improves coverage but may retrieve documents that match only a weak modifier. A mature system varies the query plan with core terms, entities, and observed recall volume, and records the chosen plan in the request trace.

### Quick Coding: Inverted Index

Given a set of tokenized documents, construct a `term -> posting list`. Each posting must store the `doc_id`, term frequency, and positions, and be sorted by document ID. This exercise checks whether you truly understand what data is required for TF, word distance, and phrase matching.

The input consists of tokenized documents:

```python
documents = {
    1: ["deep", "learning", "deep"],
    2: ["learning", "system"],
}
```

Implementation:

```python
def build_inverted_index(documents):
    ...
```

Output the posting list for each term, where each posting is:

```text
(doc_id, term_frequency, zero_based_positions)
```

For example, `deep` corresponds to `[(1, 2, [0, 2])]`. Postings should be sorted by `doc_id` in ascending order.

<details>
<summary>Reference answer</summary>

```python
from collections import defaultdict


def build_inverted_index(documents):
    index = defaultdict(list)

    for doc_id in sorted(documents):
        positions = defaultdict(list)
        for position, term in enumerate(documents[doc_id]):
            positions[term].append(position)

        for term, term_positions in positions.items():
            index[term].append(
                (doc_id, len(term_positions), term_positions)
            )

    return dict(index)
```

For `T` total tokens and `D` documents, time is `O(T + D log D)` and index space is `O(T)`.

</details>

### 3.3 TF-IDF and BM25

TF-IDF considers both the frequency of a term in the current document and its rarity across the entire corpus:

```math
\operatorname{tfidf}(t,d)
=\operatorname{tf}(t,d)
\log\frac{N}{df(t)}.
```

Common terms have low discriminative power, so IDF assigns them a smaller we\right. BM25 builds on this by adding term frequency saturation and document length normalization:

```math
\operatorname{BM25}(q,d)
=\sum_{t\in q}
\operatorname{IDF}(t)
\frac{tf(t,d)(k_1+1)}
{tf(t,d)+k_1(1-b+b|d|/\operatorname{avgdl})}.
```

A term appearing 20 times should not be twice as important as one appearing 10 times; this is the purpose of the saturation term. Long documents are naturally more likely to hit terms, and length normalization corrects for this.

BM25 is fast, stable, and interpretable. When evaluating dense retrieval or generative retrieval, compare against it first; comparing only against weaker neural baselines makes it difficult to judge whether the added complexity is worthwhile.

Document fields are rarely weighted equally. Title, body, anchor, and structured-attribute matches can receive separate BM25 scores and then be combined with BM25F or a linear model. Overweighting the title rewards keyword stuffing; overweighting the body favors long documents. Tune field weights on relevance slices rather than clicks alone.

BM25 scores are also not directly comparable across queries. A query with several rare terms naturally produces a different score range. Thresholding, caching, or multi-channel fusion that needs a cross-request scale should use within-query rank, bucketed normalization, or calibration.

### 3.4 ItemCF

ItemCF first calculates item similarity based on user-item interactions, then expands candidates from the user's history.

The simplest co-occurrence similarity is:

```math
\operatorname{sim}(i,j)
=\frac{|U_i\cap U_j|}
{\sqrt{|U_i||U_j|}}.
```

`U_i` is the set of users who have interacted with item `i`. The denominator suppresses popular items.

The score for user `u` on candidate `i` can be written as:

```math
\operatorname{score}(u,i)
=\sum_{j\in H_u}
w(u,j)\operatorname{sim}(j,i),
```

`H_u` is the user's history, and `w(u,j)` can incorporate behavior type, time decay, and viewing depth.

The co-occurrence table first depends on which events enter it. Effective views, explicit saves, and purchases can carry different weights; autoplay, accidental taps, and bot traffic should be downweighted or removed. Two items consumed in one short session usually provide stronger local evidence than two items touched six months apart. Construction also caps user activity and session length so that a power user does not create an edge between almost every pair of items.

The complete engineering workflow:

1. Offline statistics of the user-to-item behavior table;
2. Generate item co-occurrence;
3. Calculate and truncate the top similar items for each item;
4. Read user history online;
5. Query the similarity table, aggregate, and filter out already exposed content;
6. Output top candidates.

The benefits of ItemCF are ease of interpretation and low service cost. The downsides are that new items have no co-occurrence, popular items easily dominate, and it may fail to expand when interests cross categories.

### Quick Coding: ItemCF

Starting from a small dataset of `user -> interacted items`, calculate co-occurrence cosine similarity and recall un-interacted items for a specified user. You are required to filter history, handle tied candidates stably, and explain why this interview code cannot be directly used to serve full-scale traffic.

Implementation:

```python
def item_cf_recommend(user_items, target_user, k):
    ...
```

Use item co-occurrence cosine similarity:

```math
sim(i,j)=\frac{|U_i\cap U_j|}{\sqrt{|U_i||U_j|}}.
```

The candidate score is the sum of similarities between the target user's interacted items and the candidate items. Filter out items already interacted with, and return the top `k` `(item_id, score)` pairs sorted by `score` in descending order and `item_id` in ascending order.

<details>
<summary>Reference answer</summary>

```python
from collections import defaultdict
from math import sqrt


def item_cf_recommend(user_items, target_user, k):
    item_users = defaultdict(set)
    for user, items in user_items.items():
        for item in set(items):
            item_users[item].add(user)

    seen = set(user_items.get(target_user, []))
    scores = []

    for candidate, candidate_users in item_users.items():
        if candidate in seen:
            continue

        score = 0.0
        for item in seen:
            users = item_users.get(item, set())
            if users and candidate_users:
                score += len(users & candidate_users) / sqrt(
                    len(users) * len(candidate_users)
                )
        if score > 0:
            scores.append((candidate, score))

    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return scores[:k]
```

This is a direct interview implementation. Production systems precompute item-to-item top similarities instead of scanning the catalog per request.

</details>

### 3.5 UserCF and Swing

UserCF first finds similar users, then recalls items they liked that the current user has not seen. It is useful when user interests are relatively stable and the number of users is not so large that the similarity table becomes unmaintainable. Large-scale content platforms more commonly use ItemCF because item similarity relationships are generally more stable than user similarity relationships.

Swing addresses "small-circle noise" in co-occurrence. If two items are only interacted with by a small group of highly overlapping users, standard co-occurrence might judge them as overly similar. Swing penalizes the number of common items for user pairs; the gist is:

```math
\operatorname{sim}(i,j)
=\sum_{u,v\in U_i\cap U_j}
\frac{1}{\alpha+|I_u\cap I_v|}.
```

Contributions from user pairs with excessive common interests are reduced. The formula is not the point; the point is which statistical bias it corrects.

### 3.6 Matrix Factorization

Approximate the user-item matrix `R` as two low-rank matrices:

```math
R\approx UV^\top,
\qquad
\hat r_{ui}=u_u^\top v_i.
```

Matrix factorization compresses discrete co-occurrence into continuous vectors. It is easier to express latent interests than simple ItemCF, but it still relies primarily on interactions, and the cold-start problem does not disappear into thin air.

Implicit feedback usually uses targets with confidence levels, rather than treating all non-clicks as explicit negatives. Training also requires careful negative sampling; otherwise, the massive number of zero entries will drown out the positive samples.

One weighted matrix factorization objective is:

```math
\min_{U,V}
\sum_{u,i} c_{ui}
\left( p_{ui}-u_u^\top v_i \r\right)^2
\lambda(\|U\|_F^2+\|V\|_F^2),
```

where `p_ui` marks whether a preference was observed and `c_ui` expresses confidence. Unobserved entries still affect the objective, but with less weight than explicit positive feedback; "not observed" is not an equally strong dislike.

Online serving can run ANN from a user vector to item vectors or precompute top items. Matrix factorization and two-tower retrieval both use low-dimensional inner products, but their inputs differ. Classical matrix factorization mainly learns IDs from interactions and does not naturally consume text, images, or context. A two-tower model generates vectors from features, which makes new-item and cross-domain transfer easier.

### 3.7 When to Use Which

| Method | Strengths | Significant Weaknesses |
| --- | --- | --- |
| Inverted Index/BM25 | Precise text, mature, efficient, easy to update | No semantic overlap |
| ItemCF | Cheap, interpretable, strong behavioral correlation | Cold start, popularity bias |
| UserCF | Discovers candidates via similar groups | Unstable user relationships, large scale |
| Swing | Suppresses small circles and overly strong co-occurrence | More complex calculation and parameters |
| Matrix Factorization | Continuous latent space, better generalization | Still relies on behavior, limited feature utilization |

Online systems usually retain multiple channels. Even if a traditional method's single-path metric is not superior, it may find candidates missed by deep models.

### 3.8 Chapter Self-Test

1. Why does an inverted index need to store term positions?
2. What do the saturation term and length normalization in BM25 solve, respectively?
3. What precomputed tables are needed for ItemCF online service?
4. What kind of problem does Swing aim to correct in standard co-occurrence?
5. What is the connection and difference between matrix factorization and two-tower models?

<details>
<summary>Reference answers</summary>

1. Positions support phrase queries, proximity matching, and highlighting. Document IDs and term frequencies alone cannot recover term order.
2. Saturation prevents term frequency from growing without bound; length normalization prevents long documents from winning merely because they contain more terms.
3. Serving needs an item-to-item top-similarity table, the user's recent items, and item availability. Similarities are normally built offline or nearline.
4. Ordinary co-occurrence can be inflated by highly active users and accidental overlap among popular items. Swing downweights these common connections.
5. Both learn low-dimensional user/item representations and often use a dot product. Matrix factorization mainly learns IDs from interactions; a two-tower model can consume content and context and encode new items.

</details>
