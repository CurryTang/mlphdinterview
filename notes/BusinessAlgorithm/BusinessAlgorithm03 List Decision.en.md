# Reranking, Diversity, and Rules

## Chapter 13: Reranking, Diversity, and Rules

### 13.1 Adding an Item-Item Constraint Layer to Lists

Fine-ranking predicts the utility of items individually. If the top ten candidates are all from the same author discussing the same topic, each might have a high score, but the list experience will be poor.

The input to reranking is a set of candidates and their fine-ranking scores; the output is a constrained, ordered list. It must simultaneously consider:

- Individual item utility;
- Similarity between candidates;
- Coverage of categories, authors, and content formats;
- Advertising, operational, and safety rules;
- Dependencies between adjacent positions;
- Content the user has already consumed.

### 13.2 How to Quantify Diversity

The simplest form is category coverage:

```math
\operatorname{Coverage@K}
=|\{category(i): i\in topK\}|.
```

One can also calculate the average intra-list similarity:

```math
\operatorname{ILS}
=\frac{2}{K(K-1)}
\sum_{i<j}\operatorname{sim}(i,j).
```

A lower ILS usually implies higher diversity, but "dissimilarity" is not always good. Randomly inserting irrelevant content can lower ILS but will destroy the user experience. Diversity must be evaluated alongside relevance or estimated utility.

There are also multiple types of similarity. Identical categories, same authors, similar text, or close embeddings represent different types of redundancy. In practice, these often need to be combined.

### 13.3 MMR

MMR selects one candidate at a time from the unselected set:

```math
i^*
=\arg\max_{i\in R}
\left[
\theta\,r_i
-(1-\theta)\max_{j\in S}\operatorname{sim}(i,j)
\right].
```

`r_i` is the fine-ranking utility, `S` is the set of already selected items, and `θ` controls the trade-off between utility and diversity.

MMR is intuitive, easy to tune, and supports hard filtering. Its limitation is that it is greedy and only considers the similarity between a candidate and the most similar selected item, failing to fully capture the structure of the entire set.

### Quick Coding: MMR Reranking

Given candidate relevance and pairwise similarity, greedily select the top-k. Support scenarios where only one-way similarity is provided, and ensure a deterministic order when scores are tied.

Problem: [[BusinessAlgorithm09 Quick Coding.md#QC07 MMR Reranking|QC07 MMR Reranking]].

### 13.4 DPP

DPP uses a positive semi-definite kernel matrix `L` to describe quality and similarity:

```math
P(S)\propto\det(L_S).
```

If two candidates are very similar, their corresponding vectors are nearly linearly dependent, causing the determinant to shrink; when candidates are high-quality and diverse in direction, the determinant is larger.

Typically:

```math
L_{ij}=q_i\,S_{ij}\,q_j
```

where `q_i` is quality and `S_{ij}` is similarity. Finding the exact optimal subset is expensive; online systems usually employ greedy approaches and incremental matrix updates.

The mathematical form of DPP is elegant, but services must also face numerical stability, candidate scale, and hard rules. Many teams ultimately use DPP-inspired business approximations rather than the full sampling method found in textbooks.

### 13.5 Sliding Window

Long lists do not necessarily require global deduplication. Users are most sensitive to repetition in adjacent content; one can simply penalize the last `W` selected candidates:

```text
When selecting the t-th candidate, compare similarity only with positions [t-W, t-1]
```

This is computationally cheaper and allows the same topic to reappear after a certain interval. `W` should be tuned based on screen layout and consumption rhythm rather than being copied blindly.

### 13.6 How Rules Enter Reranking

Rules can be divided into three categories.

Hard constraints must be satisfied, such as safety, inventory, legal requirements, and ad caps. These should be filtered out first or strictly checked during list construction.

Soft constraints allow for trade-offs, such as author diversification, category coverage, and long-tail support. These can be implemented as penalties or rewards.

Quota constraints require a certain number of items of a specific type to appear within an interval, such as at least 1 cold-start item or no more than 2 ads. These can be handled via bucket selection, integer programming approximations, or constrained greedy algorithms.

Make the interval explicit. Examples include no more than five consecutive image or video posts, at most one promoted item in any nine positions, or at most one commerce card in the first four. At each greedy step, first remove candidates that violate the active window rules, then run MMR on the feasible set. Logs can then distinguish a similarity penalty from a hard rule rejection.

Stuffing all rules into a long if-else chain quickly leads to chaos. A more robust approach is to define rules as configurable constraints, log which rule removed each candidate, and support offline replay.

### 13.7 Diversity in Search vs. Recommendation

Search queries usually have specific answers. If a user searches for "iPhone 15 specs," the top results must be relevant; you cannot insert Android phones just for the sake of diversity.

Recommendations can expand interests, but they cannot explore infinitely. Both require diversity, but the priority of constraints differs:

```text
Search: Relevance baseline > Quality/Recency > Diversity
Recommendation: Estimated interest and long-term experience jointly constrain diversity
```

### 13.8 Chapter Self-Test

1. Why is lower intra-list similarity not always better?
2. What happens when the `θ` in MMR is increased?
3. Why can the determinant in DPP represent diversity?
4. What consumption scenarios are suitable for sliding window reranking?
5. Which rules should be hard constraints, and which are suitable for soft scoring?

<details>
<summary>Reference answers</summary>

1. Minimizing similarity too aggressively sacrifices relevance and coherence. Diversity should reduce repetition within relevant candidates.
2. A larger `θ` emphasizes relevance and stays closer to the original order; a smaller value emphasizes diversity and increases relevance risk.
3. Similar items create similar kernel rows and a small determinant; complementary high-quality sets span a larger volume.
4. Short-video, music, and feed sessions are suitable because users mainly perceive repetition among nearby positions.
5. Safety, inventory, regional permission, and legal rules should be hard. Author repetition, category balance, and novelty are usually soft or budgeted constraints.

</details>
