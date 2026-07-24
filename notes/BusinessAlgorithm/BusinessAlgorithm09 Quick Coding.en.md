# Quick Coding: 8 Short Problems

Limit each problem to 10-20 minutes. Write a correct version first, then add edge cases and complexity analysis. The webpage only contains the problems to avoid seeing the answers immediately upon opening.

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

Time target: 10 minutes.

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

Time target: 15 minutes.

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

Time target: 20 minutes.

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

Time target: 10 minutes.

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

Time target: 10 minutes.

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

Time target: 15 minutes.

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

Time target: 15 minutes.

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

Time target: 15 minutes.

## Suggested Review Method

First, write a correct version within the time limit. After passing the test cases, add edge cases and complexity analysis. Finally, explain orally at which step of the online pipeline this runs, and what implementation should be used if the data volume scales up.

For example, the ItemCF problem is only suitable as a small-scale interview question. In a real system, co-occurrence and top-similarity tables are constructed offline; the system would not iterate through all users and items online.
