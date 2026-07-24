# List Decision-Making: Reranking, Exploration, and Cold Start

## Chapter 10: Reranking, Diversity, and Rules

### 10.1 Adding an Item-Item Constraint Layer to Lists

Fine-ranking predicts the utility of items individually. If the top ten candidates are all from the same author discussing the same topic, each might have a high score, but the list experience will be poor.

The input to reranking is a set of candidates and their fine-ranking scores; the output is a constrained, ordered list. It must simultaneously consider:

- Individual item utility;
- Similarity between candidates;
- Coverage of categories, authors, and content formats;
- Advertising, operational, and safety rules;
- Dependencies between adjacent positions;
- Content the user has already consumed.

### 10.2 How to Quantify Diversity

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

### 10.3 MMR

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

### 10.4 DPP

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

### 10.5 Sliding Window

Long lists do not necessarily require global deduplication. Users are most sensitive to repetition in adjacent content; one can simply penalize the last `W` selected candidates:

```text
When selecting the t-th candidate, compare similarity only with positions [t-W, t-1]
```

This is computationally cheaper and allows the same topic to reappear after a certain interval. `W` should be tuned based on screen layout and consumption rhythm rather than being copied blindly.

### 10.6 How Rules Enter Reranking

Rules can be divided into three categories.

Hard constraints must be satisfied, such as safety, inventory, legal requirements, and ad caps. These should be filtered out first or strictly checked during list construction.

Soft constraints allow for trade-offs, such as author diversification, category coverage, and long-tail support. These can be implemented as penalties or rewards.

Quota constraints require a certain number of items of a specific type to appear within an interval, such as at least 1 cold-start item or no more than 2 ads. These can be handled via bucket selection, integer programming approximations, or constrained greedy algorithms.

Stuffing all rules into a long if-else chain quickly leads to chaos. A more robust approach is to define rules as configurable constraints, log which rule removed each candidate, and support offline replay.

### 10.7 Diversity in Search vs. Recommendation

Search queries usually have specific answers. If a user searches for "iPhone 15 specs," the top results must be relevant; you cannot insert Android phones just for the sake of diversity.

Recommendations can expand interests, but they cannot explore infinitely. Both require diversity, but the priority of constraints differs:

```text
Search: Relevance baseline > Quality/Recency > Diversity
Recommendation: Estimated interest and long-term experience jointly constrain diversity
```

### 10.8 Chapter Self-Test

1. Why is lower intra-list similarity not always better?
2. What happens when the `θ` in MMR is increased?
3. Why can the determinant in DPP represent diversity?
4. What consumption scenarios are suitable for sliding window reranking?
5. Which rules should be hard constraints, and which are suitable for soft scoring?

---

## Chapter 11: Cold Start, Exploration, and Long-term Feedback

### 11.1 Embeddings Only Provide Initial Estimates

New items have no interactions, so collaborative filtering cannot find them; new users have no history, so personalized models have no basis for judgment. Even if content encoders are used to generate embeddings, the system still lacks real consumption feedback.

Cold start requires solving two problems:

1. Providing initial estimates using existing information;
2. Obtaining real feedback under controlled risk.

Doing only the first without exploration makes it difficult for the model to correct initial misjudgments.

### 11.2 New Item Candidates

Common channels include:

- Category, tags, author, and region;
- Text/image/multimodal similarity;
- Clustering and placement into matching interest clusters;
- Look-alike audience expansion;
- New item pools after operational or quality review.

Look-alike starts from a small number of seed users to find populations with similar profiles or behaviors. It depends on seed quality and can easily amplify early biases.

### 11.3 New Users

For new users, one can utilize:

- Region, language, device, and entry point;
- First-screen interactions;
- Explicit interest selection;
- Contextual popularity;
- Similar anonymous sessions.

The most valuable data for a new user is their first few actions. The system should quickly update short-term interests and avoid showing default popular content for too long.

### 11.4 Where to Inject Traffic

Cold-start candidates can be injected at different layers:

- Recall injection: Low risk, but likely to be eliminated by coarse and fine ranking;
- Post-coarse-ranking quota: Easier to gain exposure, but requires higher quality control;
- Post-fine-ranking insertion: Stable exposure, but may significantly harm the main list;
- Reranking quota: Most controllable, but requires clear budgets and guardrails.

The later the injection, the more guaranteed the exposure, but the higher the risk of damaging the main list. High-quality review pools can be guaranteed in reranking; new content of unknown origin is better suited for small-scale traffic testing at the recall layer.

### 11.5 Exploration vs. Exploitation

Exploitation selects the content currently estimated to be the best, while exploration selects content that is uncertain but potentially promising.

A simple UCB form:

```math
\operatorname{UCB}_i
=\hat\mu_i
+c\sqrt{\frac{\log t}{n_i}}.
```

`μ_i` is the current utility estimate, and `n_i` is the number of exposures. Items with fewer exposures receive an uncertainty bonus.

Utility in recommendations depends on user context, items expire, and feedback has delays and position effects, so the UCB formula only indicates the direction of exploration. Even without bandits, one must provide a small amount of controlled traffic to high-uncertainty candidates; otherwise, the system cannot collect the data needed to correct initial estimates.

### 11.6 Cold-Start Evaluation

Overall CTR will be drowned out by mature items. At least look at the following separately:

- Proportion of new items entering recall;
- Waiting time for first exposure;
- Traffic required to reach minimum feedback volume;
- Stratified CTR/duration/negative feedback for new items;
- Damage to the main list caused by exploration;
- Proportion of high-quality new items retained after exploration.

Also, stratify by author, category, and quality priors. Mixing all new items together and averaging them will make the strategy appear erratic.

### 11.7 Feedback Loops

Recommendation systems are trained on their own exposures. Models favor popular items, popular items get more exposure, and they become even more popular. The long-term result can be a narrowing of content and difficulty for new authors to enter.

Mitigation methods include:

- Random or uncertainty-based exploration;
- Popularity debiasing and propensity weighting;
- New item quotas;
- Long-term holdouts;
- Creator-side ecosystem metrics;
- Exposure correction in training samples.

These methods sacrifice some short-term clicks. Whether this is worth it is a product and ecosystem decision, not purely a model problem.

### 11.8 Chapter Self-Test

1. Why does the cold-start problem persist even after having content embeddings?
2. What are the risks of injecting cold-start candidates after recall versus after fine-ranking?
3. Why does the UCB uncertainty bonus decrease as exposure increases?
4. Which metrics reflect that new items have truly been given an opportunity?
5. How can one determine if the system is forming a popularity feedback loop?
