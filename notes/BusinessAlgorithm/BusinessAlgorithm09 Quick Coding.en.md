# Quick Coding: 8 Short Problems

These problems only test the core logic and use the Python standard library. Each answer sits directly below its problem and is collapsed by default.

## QC01 Stable A/B Bucketing

Implementation:

```python
def assign_bucket(user_id: str, salt: str, num_buckets: int) -> int:
    ...
```

Requirements:

- The same `user_id + salt` must always result in the same bucket.
- The return value must fall within `[0, num_buckets)`.
- Changing the `salt` should result in independent experimental bucketing.
- Do not use Python's built-in `hash()`, as it does not guarantee stability across processes by default.
- Raise a `ValueError` if `num_buckets <= 0`.

<details>
<summary>Reference answer</summary>

```python
import hashlib


def assign_bucket(user_id: str, salt: str, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")

    payload = f"{salt}\0{user_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % num_buckets
```

Time complexity is `O(len(user_id) + len(salt))`; auxiliary space is `O(1)`.

</details>

## QC02 Build Inverted Index

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

## QC03 ItemCF Recommendation

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

## QC04 Reciprocal Rank Fusion

Implementation:

```python
def reciprocal_rank_fusion(rankings, rrf_k=60, top_n=None):
    ...
```

Each ranking is a list of candidate IDs. Fusion score:

```math
score(d)=\sum_m\frac{1}{rrf\_k+rank_m(d)}.
```

Ranks start at 1. For duplicate candidates within the same ranking, only the first occurrence is counted. Return results sorted by score in descending order and candidate ID in ascending order.

<details>
<summary>Reference answer</summary>

```python
from collections import defaultdict


def reciprocal_rank_fusion(rankings, rrf_k=60, top_n=None):
    scores = defaultdict(float)

    for ranking in rankings:
        seen = set()
        for rank, candidate in enumerate(ranking, start=1):
            if candidate in seen:
                continue
            seen.add(candidate)
            scores[candidate] += 1.0 / (rrf_k + rank)

    result = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return result if top_n is None else result[:top_n]
```

For `N` total ranking entries and `M` unique candidates, scoring is `O(N)` and final sorting is `O(M log M)`.

</details>

## QC05 NDCG@K

Implementation:

```python
def ndcg_at_k(relevances, k):
    ...
```

`relevances` are already sorted by the model's predicted order, where each value is a non-negative relevance grade. Use:

```math
DCG@K=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}.
```

Return `0.0` if there are no relevant results, or if `k <= 0`.

<details>
<summary>Reference answer</summary>

```python
from math import log2


def ndcg_at_k(relevances, k):
    if k <= 0:
        return 0.0

    all_values = list(relevances)

    def dcg(items):
        return sum(
            (2 ** rel - 1) / log2(rank + 1)
            for rank, rel in enumerate(items, start=1)
        )

    actual = dcg(all_values[:k])
    ideal = dcg(sorted(all_values, reverse=True)[:k])
    return 0.0 if ideal == 0 else actual / ideal
```

Sorting the ideal list costs `O(n log n)`. A small bounded relevance scale allows a linear-time counting approach.

</details>

## QC06 FM Forward Pass

Implementation:

```python
def fm_predict(x, bias, linear_weights, factors):
    ...
```

Where `factors[i]` is the `latent_dim`-dimensional vector for the `i`-th feature. You are required to use the following equivalent formula to reduce the second-order interaction complexity from `O(d²k)` to `O(dk)`:

```math
\frac{1}{2}\sum_f
\left[
\left(\sum_i v_{i,f}x_i\right)^2
-\sum_i(v_{i,f}x_i)^2
\right].
```

Raise a `ValueError` if input dimensions are inconsistent.

<details>
<summary>Reference answer</summary>

```python
def fm_predict(x, bias, linear_weights, factors):
    if len(x) != len(linear_weights) or len(x) != len(factors):
        raise ValueError("feature dimensions do not match")
    if not factors:
        return float(bias)

    latent_dim = len(factors[0])
    if any(len(vector) != latent_dim for vector in factors):
        raise ValueError("factor dimensions do not match")

    linear = sum(weight * value for weight, value in zip(linear_weights, x))
    interaction = 0.0

    for latent in range(latent_dim):
        summed = sum(factors[i][latent] * x[i] for i in range(len(x)))
        squared = sum(
            (factors[i][latent] * x[i]) ** 2
            for i in range(len(x))
        )
        interaction += 0.5 * (summed ** 2 - squared)

    return bias + linear + interaction
```

For `d` features and latent dimension `m`, time is `O(dm)` and auxiliary space is `O(1)`.

</details>

## QC07 MMR Reranking

Implementation:

```python
def mmr_rerank(candidates, relevance, similarities, k, theta):
    ...
```

Selection at each round:

```math
\arg\max_i
\left[
\theta r_i-(1-\theta)\max_{j\in S}sim(i,j)
\right].
```

`similarities` uses a dictionary of `(item_a, item_b) -> score`; providing only one direction is acceptable. In case of ties in MMR scores, sort by candidate ID in ascending order.

<details>
<summary>Reference answer</summary>

```python
def mmr_rerank(candidates, relevance, similarities, k, theta):
    remaining = set(candidates)
    selected = []

    def similarity(a, b):
        return similarities.get(
            (a, b),
            similarities.get((b, a), 0.0),
        )

    while remaining and len(selected) < k:
        def score(item):
            redundancy = max(
                (similarity(item, chosen) for chosen in selected),
                default=0.0,
            )
            return theta * relevance[item] - (1 - theta) * redundancy

        chosen = min(remaining, key=lambda item: (-score(item), item))
        selected.append(chosen)
        remaining.remove(chosen)

    return selected
```

The direct implementation costs `O(k²N)`. Caching each candidate's maximum similarity to the selected set reduces updates to `O(kN)`.

</details>

## QC08 Residual Quantization for Semantic IDs

Implementation:

```python
def residual_quantize(vector, codebooks):
    ...
```

For each layer's codebook, select the codeword with the smallest Euclidean distance to the current residual, then subtract that codeword. Return:

```text
([indices of selected codewords per layer], final residual)
```

All codewords must have the same dimensionality as the input vector; raise a `ValueError` if a codebook is empty or dimensions are mismatched.

<details>
<summary>Reference answer</summary>

```python
def residual_quantize(vector, codebooks):
    residual = list(vector)
    codes = []

    for codebook in codebooks:
        if not codebook:
            raise ValueError("codebook must not be empty")
        if any(len(codeword) != len(residual) for codeword in codebook):
            raise ValueError("codeword dimensions do not match")

        best_index = min(
            range(len(codebook)),
            key=lambda index: sum(
                (residual[d] - codebook[index][d]) ** 2
                for d in range(len(residual))
            ),
        )
        codes.append(best_index)
        residual = [
            value - code
            for value, code in zip(residual, codebook[best_index])
        ]

    return codes, residual
```

For `L` layers, `K` codewords per layer, and vector dimension `d`, time complexity is `O(LKd)`.

</details>
