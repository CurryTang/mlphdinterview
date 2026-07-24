# Search Relevance and BERT

## Chapter 8: Search Relevance and BERT

Relevance answers whether a document satisfies a query. Content quality, freshness, personalization, and business goals affect final ranking, but they should not be hidden inside relevance labels.

### 8.1 Relevance grades

A practical 0–3 scale is:

| Grade | Meaning |
| --- | --- |
| 3 | Directly and completely satisfies a major intent |
| 2 | Relevant and useful, but incomplete |
| 1 | Covers only a secondary intent or fragment |
| 0 | Semantically unrelated or does not answer the need |

Annotation guidelines must list major intents for ambiguous queries. Whether a document about the fruit or the company is highly relevant to `"apple"` depends on traffic intent and product context, not an annotator's guess.

### 8.2 Three offline views

Pointwise metrics measure the prediction for one `(q,d)` pair: MSE, LogLoss, AUC, or grade accuracy.

Pairwise metrics compare documents under the same query. If `y_i > y_j`, the model should produce `s_i > s_j`. Pair accuracy is:

```math
\frac{\#\{(i,j):y_i>y_j,\ s_i>s_j\}}
{\#\{(i,j):y_i>y_j\}}.
```

Listwise metrics evaluate the result list, commonly with DCG/NDCG. These views measure value preservation, ordering, and head-of-list quality; none subsumes the others.

### 8.3 Lexical matching

Inverted-index retrieval still relies on:

- TF-IDF and BM25;
- query-term coverage;
- field matches in title, body, and anchor text;
- term order and proximity.

BM25 saturates term frequency and corrects for document length:

```math
\operatorname{BM25}(q,d)
=\sum_{t\in q}
\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1(1-b+b|d|/\operatorname{avgdl})}.
```

Bag-of-words models miss context. A document containing a look-alike brand can have strong lexical overlap but fail the brand intent. Proximity features help with word distance, not full semantics.

Lexical signals remain cheap, interpretable retrieval and pre-ranking features. Neural models did not remove the cost of generating candidates from a large corpus.

### 8.4 Cross-BERT and dual-encoder BERT

Cross-BERT jointly encodes:

```text
[CLS] query [SEP] title [SEP] body [SEP] -> relevance
```

Tokens interact directly, which improves accuracy but requires inference for each `(q,d)` pair.

A dual encoder computes:

```math
z_q=f(q),\qquad z_d=g(d),\qquad s(q,d)=z_q^\top z_d.
```

Document vectors are precomputed. Dual encoders fit retrieval and coarse ranking; Cross-BERT fits later stages with fewer candidates. Production systems usually increase interaction depth stage by stage.

### 8.5 Token granularity and long documents

Chinese systems may use characters or a mixed character-word vocabulary. Mixed granularity shortens sequences, reduces attention cost, and retains more information under a fixed token limit.

Long-document options include:

- prioritizing title and opening passages;
- query-aware passage selection;
- offline extractive summaries;
- segment scoring and aggregation;
- Anchor Queries as short document labels.

An Anchor Query is generated from a document and filtered for relevance. It especially helps dual encoders by narrowing the structural gap between short queries and long documents.

### 8.6 Reducing inference cost

Cross-BERT optimizations include:

- caching `(query_id, document_id, model_version) -> score`;
- INT8 PTQ or QAT;
- teacher-student distillation;
- shorter token limits;
- passage preselection;
- batching and dynamic padding.

Cache keys need model, tokenizer, and document versions. Otherwise a rollout reads scores produced by an older model or document.

Dual-encoder cost lies in query encoding, vector storage, and ANN. Quantization and approximate indexes lose recall, so evaluation should separate model loss from ANN approximation loss.

### 8.7 Four-stage training

The course organizes relevance-BERT training as:

```text
pretraining -> post-pretraining -> supervised fine-tuning -> distillation
```

Pretraining uses MLM and related tasks.

Post-pretraining uses search logs. Each record provides `(q,d,x)`, where `x` contains click and engagement statistics. A small clean human-labeled set trains a mapping:

```math
\tilde y=t(x).
```

The mapping creates weak labels for a much larger `(q,d)` corpus. BERT continues training on those labels while retaining MLM to reduce forgetting.

The mapping `t(x)` must not use the old relevance-model score. The old model already affects exposure; feeding its score back into label generation creates a loop that reproduces its blind spots.

### 8.8 Fine-tuning for value and order

Regression or soft-label CE keeps predictions close to grades:

```math
\mathcal L_{\text{CE}}
=-y\log p-(1-y)\log(1-p).
```

Pairwise logistic encourages the right order:

```math
\mathcal L_{\text{pair}}
=\log\left(1+\exp[-\gamma(s_i-s_j)]\right),
\qquad y_i>y_j.
```

Production objectives often mix pointwise, pairwise, and MLM losses. Plain multiclass classification ignores that relevance grades are ordered.

### 8.9 Distillation

A large teacher provides accurate soft labels; a smaller student meets latency limits:

1. fully train the teacher;
2. score a large `(q,d)` corpus;
3. warm up the student with the normal training pipeline;
4. fit teacher scores and pair ordering;
5. evaluate on human labels to detect copied teacher bias.

Matching every hidden layer is optional. The target is relevance and ranking quality; more high-quality distillation pairs are often more useful than layer-wise imitation.

### 8.10 Chapter self-test

1. Why keep content quality out of relevance labels?
2. What do pointwise, pairwise, and listwise metrics measure?
3. Why is BM25 useful in a BERT-based system?
4. Why does a Cross-BERT cache key need a model version?
5. How does post-pretraining derive weak labels from behavior?
6. Why exclude old relevance scores from the weak-label mapper?
7. Why warm up a student before distillation?

<details>
<summary>Reference answers</summary>

1. Their meanings and update schedules differ. Separate signals distinguish relevant-but-low-quality from high-quality-but-irrelevant documents.
2. Pointwise checks individual values, pairwise checks relative order, and listwise checks the full list, especially its head.
3. It is cheap, interpretable, effective for retrieval, and useful as a neural-ranking feature.
4. Model, tokenizer, or document updates can change the correct score for the same pair; unversioned caches return stale values.
5. A small labeled set learns a mapping from click and engagement statistics to relevance, then labels a much larger log corpus.
6. Old scores already influence exposure. Reusing them as supervision creates a self-reinforcing feedback loop.
7. Warm-up gives the student language and relevance knowledge before it learns the teacher's finer distribution, improving stability.

</details>
