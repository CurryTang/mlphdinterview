# Ranking: Objectives, Features, and User Sequences

## Chapter 6: Ranking Objectives and Offline Evaluation

### 6.1 Defining "Relevance" for Ranking

Search relevance is typically categorized into graded labels:

- Highly Relevant: Directly satisfies the primary intent of the query;
- Relevant: Solves the problem, but is incomplete or imprecise;
- Weakly Relevant: Covers only a portion of the intent;
- Irrelevant.

Real-world labeling is more complex. Queries may be polysemous, documents may be topically relevant but of poor quality, and timeliness can change the answer. Labeling guidelines should separate "topical relevance" from "whether it is ultimately worth displaying," otherwise, the model will conflate quality, authority, and relevance.

Recommendation labels are derived from behavior. Clicks, effective views, likes, follows, and purchases represent different intensities and have different latencies. Model objectives must align with product objectives; one should not train solely on CTR just because click samples are abundant.

### 6.2 Pointwise

Pointwise treats each candidate as an independent classification or regression sample:

```math
\mathcal L_{\text{point}}
=-\left[y\log p+(1-y)\log(1-p)\right].
```

The advantage is that both sampling and training are simple, and predicted probabilities can be calibrated. The disadvantage is that it does not directly express the order between candidates under the same query/user.

CTR, CVR, and duration estimation often start with pointwise. If search relevance has graded labels, it can also be treated as multi-class classification or regression.

### 6.3 Pairwise

Pairwise constructs positive-negative candidate pairs, aiming for a higher score for the positive example:

```math
\mathcal L_{\text{pair}}
=-\log \sigma(s^+-s^-).
```

It is closer to "which item is ranked ahead of which," but the number of pairs can explode. How negative examples are selected significantly changes training: random negatives are simple, while hard negatives—those the current model ranks incorrectly—are more informative, though they are also more likely to contain labeling noise.

RankNet is a pairwise approach. LambdaRank/LambdaMART adjust gradient weights based on the impact of swapping two candidates on NDCG, ensuring that errors at the top of the list are prioritized.

### 6.4 Listwise

Listwise treats the entire candidate list as the training object. A simple form is to apply softmax to the list:

```math
P(i\mid q)=\frac{e^{s_i}}{\sum_j e^{s_j}},
```

Then use cross-entropy with the target distribution. One can also directly optimize an approximation of NDCG or generate candidate permutations.

Listwise is closer to the final task but requires candidates for the same query to be grouped for training, making memory usage, sampling, and implementation more complex. LLM ranking often uses listwise prompts, but long candidate lists encounter context window limits and position bias, which will be expanded upon in Chapter 13.

### 6.5 Search Relevance Models

Traditional text scores include:

- BM25;
- Query term coverage;
- Term proximity and order;
- Field matching (title, body, anchor text, etc.);
- Click and bounce statistics.

Cross-BERT encodes the query and document together, enabling the recognition of deep semantics and negation relationships. Its serving cost is high, so it is often placed in a later stage, processing only the top-N after retrieval.

Two-tower BERT can pre-calculate document representations, making it suitable for retrieval; Cross-BERT is suitable for re-ranking. When comparing the two, one must account for the number of calls: top-200 re-ranking means executing 200 query-document forward passes per query.

### 6.6 Measuring Ranking Performance

For pointwise estimations like CTR and CVR, first look at LogLoss:

```math
\operatorname{LogLoss}
=-\frac{1}{N}\sum_i
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
```

It focuses on the probability itself. AUC measures the probability that a random positive example is ranked higher than a random negative example, which is suitable for observing overall discriminative ability, but it does not specifically focus on the top of the list, nor does it indicate whether scores are calibrated.

Search and recommendation lists often use DCG/NDCG:

```math
\operatorname{DCG@K}
=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)},
```

```math
\operatorname{NDCG@K}
=\frac{\operatorname{DCG@K}}{\operatorname{IDCG@K}}.
```

The higher the position, the lower the discount, and the cost of misranking a highly relevant candidate is greater. Relevance grades for search can come from human labeling; recommendations often map clicks, purchases, or behavioral intensity into grades, and it must be explained what such mappings represent.

If the task primarily cares about the first correct result, MRR can be used:

```math
\operatorname{MRR}
=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{rank_q}.
```

These metrics must be compared on the same candidate set. If the retrieval set changes, changes in NDCG may stem from better candidates or the ranking model itself; these two must be experimented with separately.

### 6.7 Misalignment Between Training and Evaluation

Common misalignments:

- Training CTR with BCE but evaluating ranking with NDCG;
- Training on exposure logs but testing on manually constructed candidate sets;
- Training candidates coming from an old model, while online candidates come from new retrieval;
- Search labeling only looking at relevance, while online ranking mixes in quality and timeliness;
- Offline recommendation keeping only one positive example, while the system actually needs to generate a multi-interest list.

Proxy objectives can differ from final metrics, but experimental records must account for this gap. When candidate sets, label definitions, or traffic distributions change, offline gains cannot be directly extrapolated to online performance.

### Quick Coding: NDCG@K

Input graded relevance sorted by predicted order to calculate DCG, IDCG, and NDCG. Boundary conditions include `k <= 0`, lists shorter than `k`, and cases where all relevance values are zero.

Problem: [[BusinessAlgorithm09 Quick Coding.md#QC05 NDCG@K|QC05 NDCG]].

### 6.8 Chapter Self-Test

1. Pointwise scores can be calibrated; why might ranking still be poor?
2. How should hard negatives for Pairwise be generated?
3. Why is NDCG more sensitive to the top of the list?
4. Where should Cross-BERT and two-tower BERT be placed?
5. How should relevance, content quality, and final ranking scores be decoupled?

---

## Chapter 7: Multi-Objective Learning and Score Fusion

### 7.1 Why Multi-Objective?

Short-video platforms may simultaneously care about clicks, watch time, completion, likes, follows, and negative feedback. E-commerce cares about clicks, add-to-cart, orders, and GMV. Search must also consider relevance, quality, timeliness, geography, and personalization.

Roughly summing all objectives into one label loses structure. Training multiple models separately leads to redundant computation and causes low-frequency tasks to lack data. Multi-task learning finds a balance between these two extremes.

### 7.2 Shared-Bottom

The simplest structure shares the bottom layers:

```text
features -> shared network -> task A tower
                           -> task B tower
                           -> task C tower
```

Total loss:

```math
\mathcal L=\sum_t \lambda_t\mathcal L_t.
```

The problem is that task gradients may conflict. Click preference favors title attractiveness, while long watch time favors sustained content value; they do not always update shared parameters in the same direction.

### 7.3 MMoE

MMoE uses multiple experts to generate representations, with each task having its own gate:

```math
h_t(x)
=\sum_{e=1}^{E}g_{t,e}(x)f_e(x),
```

```math
g_t(x)=\operatorname{softmax}(W_t x).
```

Task `t` dynamically combines experts based on the sample. It is more flexible than Shared-Bottom, but do not interpret the gate as a stable business division of labor. A specific expert does not necessarily represent "clicks" permanently, and the gate may collapse into a few experts.

When diagnosing MMoE, one can look at:

- The entropy of each task's gate;
- Whether expert usage is balanced;
- Task gradient cosine similarity;
- Gains from single-task vs. multi-task grouping;
- Whether low-frequency tasks are suppressed by high-frequency tasks.

### 7.4 ESMM and the Conversion Funnel

E-commerce CVR is only observable after a click. If CVR is trained using only click samples, the training distribution differs from the full exposure distribution.

ESMM utilizes:

```math
P(\text{click and conversion})
=P(\text{click})P(\text{conversion}\mid\text{click}),
```

It jointly learns CTR and CTCVR in the full exposure space, then constrains CVR through their relationship. It alleviates sample selection bias and conversion sparsity, but still relies on model assumptions and data definitions, and does not mean the counterfactual problem is completely solved.

### 7.5 Duration Modeling

Watch time is zero-inflated and influenced by video length. Directly regressing seconds will bias toward long videos.

Optional approaches:

- Predict effective views and conditional duration;
- Log-transform or bucketize duration;
- Predict the watch ratio;
- Calibrate by video length;
- Model exit using survival/hazard analysis.

When evaluating, bucket by content length, user activity, and scenario. An increase in average duration might just mean the system pushed more long videos.

### 7.6 Score Fusion

Model outputs usually cannot be added linearly. CTR might be in `[0, 0.2]`, duration prediction is in seconds, and CVR is even sparser. Calibrate first, then discuss fusion.

Common form:

```math
S
=w_1f_1(\hat p_{\text{click}})
+w_2f_2(\hat t)
+w_3f_3(\hat p_{\text{conversion}})
-w_4\hat p_{\text{negative}}.
```

`f_t` can be a log, power function, piecewise function, or quantile mapping. Weights do not rely solely on offline search; they ultimately require online experimentation.

Another path is to learn a fusion model, taking scores from each objective and context as input. However, it still requires training labels and is harder to interpret regarding objective trade-offs. Strong business constraints are best kept in the re-ranking or rule layer.

### 7.7 Calibration

If a model says 0.2, and the samples have approximately 20% actual clicks, the score is calibrated. Common methods:

- Platt scaling;
- Isotonic regression;
- Temperature scaling;
- Scenario-based or population-based calibration.

Ranking only requires relative order, but fusion often requires comparable probabilities. Calibration changes do not necessarily change AUC, but they can significantly change the results of multi-objective fusion.

### 7.8 From Ranking Loss to Preference Optimization

BCE judges a single pair, BPR compares a pair of items, and InfoNCE makes one positive example compete with a set of candidates. All three utilize positive/negative feedback, but differ in comparison granularity and negative sample sources.

Generative recommendation extends the comparison unit to tokens or complete sequences. Next-token CE competes with the entire vocabulary, DPO compares chosen/rejected sequences, and policy gradient uses advantage to weight rollouts. Low-advantage RL rollouts cannot simply be treated as fixed negative samples because candidates are generated by the current policy, and sample weights change with training. For details, see [[BusinessAlgorithm05 Generative Recommendation.md#13.9 From Positive/Negative Samples to RL|Preference Optimization in Generative Recommendation]].

### 7.9 Chapter Self-Test

1. Where does negative transfer in Shared-Bottom come from?
2. How can MMoE gates be diagnosed?
3. Which two problems does ESMM solve for CVR?
4. Why does directly predicting watch seconds bias toward long videos?
5. When AUC is unchanged, why might calibration still improve online fusion?
6. How do the comparison granularities of BCE, BPR, and InfoNCE differ?

---

## Chapter 8: Feature Interaction, Coarse Ranking, and Personalization

### 8.1 Business Patterns Often Lie in Interaction Terms

Looking at "user age" or "content category" alone is insufficient; the model needs to capture "the preference of a certain age group for a certain type of content at a certain time." The number of sparse feature combinations is enormous, and manual enumeration quickly spirals out of control.

### 8.2 FM

Factorization Machine uses low-rank vectors for second-order interactions:

```math
\hat y
=w_0+\sum_i w_ix_i
+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j.
```

If `x` is one-hot, most dimensions are zero, and calculation only involves non-zero features. Inner products share statistical strength; even if a pair of features rarely appears together, reasonable interactions can be learned through their respective embeddings.

FM expresses second-order interactions. More complex relationships require deep networks or explicit high-order structures.

### Quick Coding: FM Forward Pass

Implement an FM forward pass without using nested feature loops. Use the sum-of-squares identity to reduce second-order interaction from `O(d²k)` to `O(dk)`, and verify the results using naive pairwise calculation.

Problem: [[BusinessAlgorithm09 Quick Coding.md#QC06 FM Forward Pass|QC06 FM Forward Pass]].

### 8.3 DCN

The cross layer of DCN is often written as:

```math
x_{l+1}
=x_0(x_l^\top w_l)+b_l+x_l.
```

Each layer retains the original input `x_0`, constructing higher-order explicit interactions layer by layer. Parallel deep networks learn implicit relationships, which are then concatenated.

DCN adds structured multiplicative interactions to the model, with fewer parameters than brute-force high-order combinations. Remembering it as "one layer more than an MLP" misses this point.

### 8.4 LHUC, SENet, and FiBiNET

LHUC performs conditional scaling on hidden units:

```math
h' = a(u, c)\odot h,
```

where `a` is generated by the user or scenario. It allows the same base to use different capacities for different users, domains, or scenarios.

SENet first generates weights based on the entire set of features to recalibrate each field. FiBiNET then performs bilinear interaction on field embeddings. They are suitable for CTR scenarios with many fields where importance varies by sample.

Do not treat these models as a checklist of abbreviations to memorize. Remembering three questions is more practical:

1. What interaction does it explicitly model?
2. How are parameters shared across features or populations?
3. How much online computation and feature dependency is added?

### 8.5 Coarse Ranking

Coarse ranking sits between retrieval and re-ranking. Candidates are still numerous, so the model must be inexpensive; however, using only retrieval scores leads to false negatives.

Common strategies:

- Small MLP/three-tower models;
- Re-ranking distillation;
- Using only batch-readable features;
- Feature selection and low-precision inference;
- Layered top-k or early exit.

When evaluating coarse ranking, AUC alone is insufficient. The better question is: "Given the computational budget, how many candidates that would have been selected by re-ranking were preserved?" This can be answered using top-k consistency, re-ranking top-N recall, and online false-negative analysis.

### 8.6 How Personalized Features Enter the Model

`user_id` embeddings are strong for active users but powerless for new users and cross-domain users. Robust personalization typically combines:

- ID and statistical features;
- Long-term and short-term behavior;
- Population and scenario;
- Content semantics;
- Real-time intent.

The stronger the personalization, the more one must be mindful of privacy, filter bubbles, and feedback loops. In search, explicit queries must be placed before user history; long-term profiles should not override current needs.

### 8.7 Chapter Self-Test

1. Why can FM handle second-order interactions of sparse features?
2. How does the DCN cross layer differ from a standard MLP?
3. For which scenarios is LHUC's conditional scaling suitable?
4. Why shouldn't coarse ranking be evaluated solely by its own AUC?
5. Why should queries remain dominant in search personalization?

---

## Chapter 9: User Behavior Sequences

### 9.1 What Does Average Pooling Lose?

A user has viewed basketball, cooking, music, and travel content. Averaging all item embeddings yields a fuzzy "overall interest," but it doesn't know which part of the history is relevant to the current candidate, nor does it account for temporal order.

Sequence models primarily solve three things:

- Different behaviors have different weights;
- The current candidate needs to read different parts of the history;
- Interests evolve over time.

### 9.2 Last-N

The simplest approach takes the last N behaviors. It is inexpensive and often stronger than complex models suggest.

One can add:

- Behavioral type weights;
- Time decay;
- Deduplication and continuous playback compression;
- Effective view thresholds;
- Category or author grouping.

For Last-N, bigger is not always better. Long histories introduce noise, storage, and service costs, and may re-amplify old interests.

### 9.3 DIN

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

### 9.4 SIM

SIM processes long sequences in two steps:

1. Coarsely select sub-sequences related to the candidate from a very long history;
2. Perform fine-grained attention modeling on the sub-sequences.

The logic is the same as the retrieval-ranking funnel: filter cheaply first, then interact expensively. Without efficient retrieval, long-sequence models are difficult to deploy online.

### 9.5 Temporal Issues in Training

The most dangerous bug in sequence models is time leakage.

If a sample occurs at time `t`, the history can only use behaviors visible before `t`. Statistical features, profiles, and item popularity must also be cut off at `t`. If one reads the full history of the current user offline, the metrics will be unrealistically good.

One must also handle:

- Label leakage within the same session;
- Repeated exposure;
- Out-of-order behavior logs;
- Delayed arrival;
- Negative feedback and ineffective views;
- Inconsistencies between training truncation and online truncation.

### 9.6 Real-Time Updates

The value of sequence models often comes from the most recent behaviors. If a user just finished watching a skiing video, and the feature service only updates five minutes later, no matter how complex the model is, it cannot react.

Common practices include:

- Offline storage for long-term sequences;
- Streaming updates for short-term behaviors;
- Online concatenation and deduplication;
- Degradation for missing or late behaviors;
- Recording feature versions for replay.

### 9.7 Chapter Self-Test

1. What information does Last-N average pooling lose?
2. Why does DIN's user representation depend on the candidate?
3. Why does SIM use a two-stage process for long sequences?
4. How can one check if sequence features have time leakage?
5. How to degrade when online short-term behavior updates fail?
