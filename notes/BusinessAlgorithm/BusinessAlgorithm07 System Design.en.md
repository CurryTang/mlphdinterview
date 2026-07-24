# Business Algorithm System Design

## Chapter 20: Three System Design Case Studies

First, fix the business objectives, traffic scale, and latency, then map out the flow along the path of a real request:

```text
Goals and Constraints
  -> Data and Labels
  -> Candidate Generation
  -> Ranking and List Decisioning
  -> Online Serving
  -> Evaluation and Experimentation
  -> Risk and Degradation
```

Every arrow must answer three questions: What is the data volume, what is the allowed latency, and where does it fall back to upon failure? Model names should only be placed after these constraints are defined.

### Case 1: Short Video Recommendation

#### Goals

Assume the product cares about effective watch time and next-day retention. CTR is merely a process metric; rapid swipes, reports, and creator concentration must be monitored as guardrails.

Primary metrics can be effective watch time or next-day retention. Intermediate metrics include clicks, completion, likes, and follows. Guardrails include exits, reports, excessive nighttime consumption, creator concentration, and P99 latency.

#### Data

Positive feedback is distinguished by intensity: effective play, completion, like, follow, and share. Rapid swipes, "not interested" clicks, and reports are negative feedback. Exposure logs must include position, recall channel, model version, and rule-based reasons.

Sequence samples are sliced by time; it is strictly forbidden to read behaviors that occur after the label. For labels with slow conversion, a maturity window must be set.

#### Recall

The following can be used in parallel:

- ItemCF/Swing;
- User-Item Two-Tower;
- Followed creators;
- Content similarity;
- Popularity and geography;
- Cold-start pool;
- Generative recall experimental channel.

Record the source for each path and perform deduplication. If using Semantic ID for generative recall, use it as a supplementary channel initially to facilitate measuring independent incremental gains and invalid decoding rates.

#### Ranking

Use a small model for coarse ranking to preserve top candidates for fine ranking. Fine ranking performs multi-objective estimation; MMoE can be used to share tasks like clicks, watch time, and likes. Watch time must be calibrated by video length, and negative feedback should be modeled separately.

For behavior sequences, start with Last-N + DIN. If sequences are very long, consider SIM or HSTU-like structures, but first confirm real-time behavior updates and latency budgets.

#### Reranking

Use MMR or rule-based greedy algorithms to control creator, category, and content similarity; incorporate cold-start quotas, ad frequency capping, and safety filtering. Exploration content must have a budget and a quality floor.

#### Serving

Item embeddings and most static features are calculated offline, while short-term user sequences are updated via streaming. ANN, feature services, and ranking models must all be versioned. The degradation order can be:

```text
Complex fine ranking failure -> Small fine ranking
Vector recall failure -> ItemCF/Popularity
Real-time feature failure -> Recent snapshot
Reranking failure -> Safety rules + Fine ranking order
```

#### Experimentation

First, look at recall coverage, fine-ranking NDCG/calibration, and reranking list metrics, then proceed to A/B testing. New product exploration must be layered separately to avoid conflating exploration gains with model gains in the main experiment. Use long-term holdouts to observe retention and content ecosystem health.

### Case 2: E-commerce Search

#### Goals

The goal is for users to find purchasable items as quickly as possible after entering a query. Relevance is the bottom line; GMV is not the sole objective. If high-margin but irrelevant items are pushed to the top, short-term revenue might rise, but search trust will collapse.

Primary metrics can be search success rate, purchase conversion, or GMV. Process metrics include zero-result rate, query rewriting rate, clicks, and add-to-cart. Guardrails include human-rated relevance, returns, complaints, latency, and inventory errors.

#### Query Understanding

Handle tokenization, brand/model entities, category, price and attribute constraints, spell checking, and rewriting.

Example:

```text
"Apple 15 256 black"
 -> brand=Apple
 -> model=iPhone 15
 -> storage=256GB
 -> color=black
```

Strong structural attributes should enter filtering or structured recall and should not rely solely on embeddings.

#### Recall

- Inverted index/BM25 to ensure exact terms;
- Attribute filtering to ensure brand, model, and inventory;
- Query-item two-tower for semantic supplementation;
- Multi-path recall via query rewriting;
- Cached results for high-frequency queries;
- Commercial or new product channels included within defined quotas.

Hybrid results can be fused using RRF or learning-to-rank, while retaining attribution for each path.

#### Ranking

The front end handles relevance first; Cross-BERT or lightweight cross-encoders process the top-N. Subsequently, integrate quality, sales, price, inventory, recency, personalization, and business goals.

A scoring order that is easy to control is:

```text
Relevance floor
  -> Quality/Purchasability
  -> Personalization and business goals
  -> List rules
```

Do not let GMV estimation directly push weakly relevant items to the top position.

#### Where to Place Generative Capabilities

LLMs can be used for complex query parsing, attribute extraction, rewriting, top-20 reranking, or product Q&A. Conventional navigational queries do not require agentic search.

If the page provides a "direct answer," such as comparing two cameras, use structured product data, details, and trusted reviews as evidence. Prices and inventory in the answer must be queried in real-time and cannot rely on model memory.

#### Evaluation

Offline, evaluate query understanding, recall, relevance NDCG, and attribute constraint accuracy separately. Online, monitor search success, transactions, and long-term return visits. Bucket by head/tail queries, brand words, category words, and complex attribute words.

### Case 3: Generative Search with Citations

#### Goals and Boundaries

The goal is to provide answers that are evidenced and traceable. If evidence is insufficient, return search results or explicitly refuse to answer; do not rely on fluent phrasing to compensate for missing facts.

First, define the data scope: public web pages, internal corporate documents, or specific knowledge bases. Permissions, recency, and source quality will directly alter the architecture.

#### Indexing

1. Crawl or receive documents;
2. Clean the body text, retaining title, time, author, URL, and permissions;
3. Chunk by structure, establishing parent-child relationships;
4. Build BM25 and vector indices;
5. Record versions and deletion flags.

High-risk scenarios such as finance, law, and medicine require stricter source whitelists and update SLAs.

#### Online

```text
Question classification
  -> Is search needed?
  -> Query rewrite / decomposition
  -> Hybrid retrieval
  -> Rerank
  -> Context deduplication and compression
  -> Evidence-based generation
  -> Claim-citation alignment
  -> Fact-checking and refusal
```

Simple facts can be handled in a single round; multi-hop questions enter a search loop with maximum step and cost budgets. For each search, retain the query, returned documents, selection reasoning, and time.

#### Generation Constraints

Prompts can only declare constraints; execution relies on:

- Binding citations to specific chunks;
- Decomposing the answer into claims;
- Checking if claims are entailed by the cited passages;
- Not forcing the merging of conflicting sources;
- Degrading to a list of search results or refusing to answer when evidence is insufficient.

#### Evaluation

When building a question set, annotate both answers and evidence. Diagnose using the following funnel:

```text
Is the evidence in the database?
  -> Is it recalled?
  -> Is it included in the final context?
  -> Does the model use it?
  -> Does the citation actually support the claim?
```

Offline, also test for prompt injection, malicious web pages, unauthorized documents, and expired sources. Online, monitor user follow-ups, citation clicks, corrections, and abandonment rates; do not rely solely on an LLM-as-a-judge total score.

### 20.1 How to Deconstruct "Metric Growth"

Locate the funnel first; do not start by reporting model names.

Recall issues:

- Positive examples not in candidates;
- Insufficient coverage of new interests or long-tail;
- Filtering/quota false negatives;
- Slow index updates.

Ranking issues:

- Misalignment between labels and objectives;
- Sample bias;
- Missing or leaked features;
- Multi-objective fusion and calibration errors;
- Coarse-ranking false negatives.

List issues:

- Repetition and fatigue;
- Overly restrictive rules;
- No opportunity for cold starts;
- Short-term gains harming long-term experience.

Engineering issues:

- P99 latency;
- Feature versioning;
- Stale cache;
- Inconsistent model or index canary deployment;
- Increased degradation rate.

Choose data, model, strategy, or engineering optimizations only after diagnosing the layer. Model upgrades are just one type of action.

### 20.2 How to Validate Before and After Launch

Discuss metrics only after the component is built, and first clarify which layer it corresponds to:

| Layer | Example | Purpose |
| --- | --- | --- |
| Component Metrics | Recall, NDCG, LogLoss, Calibration Error, P99 | Determine if recall, ranking, or serving works as expected |
| Process Metrics | CTR, Effective Play, Add-to-Cart, Zero-result Rate | Explain at which step user behavior changes |
| Business Results | Retention, Total Time, Transactions, Search Success | Decide if the product gains real value |
| Guardrails | Complaints, Safety, Latency, Creator Concentration, Commercial Load | Prevent harmful methods from boosting primary metrics |

Offline results only determine if it is worth proceeding to online experiments. Common discrepancies between offline and online include: different candidate sets, misalignment between labels and business goals, feature leakage, rules canceling out model gains, and increased latency.

A/B testing uses stable bucketing:

```text
bucket = hash(user_id, experiment_salt) mod B
```

The same user should consistently fall into the same bucket. Conflicting experiments are mutually exclusive in the same layer, while parallelizable experiments are placed in orthogonal layers. Perform AA checks before the experiment and pre-determine the randomization unit, primary metrics, guardrails, minimum detectable effect, and observation window.

Do not judge results solely by `p < 0.05`. Report the difference, confidence interval, and sample size; if the confidence interval still contains business-meaningful gains, the conclusion should be "insufficient evidence" rather than "strategy ineffective."

Long-term holdouts are used to observe the cumulative effects of multiple releases. After a full rollout, a small-traffic reversal bucket can be retained to confirm that gains were not misled by traffic fluctuations, novelty effects, or concurrent activities.

#### Quick Coding: Stable A/B Bucketing

Use SHA-256 to implement stable bucketing; do not use Python's built-in `hash()`. Beyond determinism and value range, handle invalid bucket counts and explain why different experiments require different salts.

Problem: [[BusinessAlgorithm09 Quick Coding.md#QC01 Stable A/B Bucketing|QC01 Stable A/B Bucketing]]

### 20.3 Chapter Self-Test

1. Why should a business-algorithm system design not begin with a model diagram?
2. What must a candidate trace record for stage-by-stage debugging?
3. If offline metrics improve but online metrics do not, which breaks should be checked?
4. Why does A/B bucketing need a stable hash and an experiment salt?
5. How can generative search distinguish "evidence was not retrieved" from "the model did not use the evidence"?

<details>
<summary>Reference answers</summary>

1. Request shape, scale, latency, objectives, guardrails, and fallback determine the stages and models. Starting from a network diagram hides serving constraints.
2. Record `request_id`, `item_id`, stage, channel, before/after rank, raw and calibrated scores, filter reason, and model, index, and rule versions.
3. Check candidate-set parity, objective alignment, temporal leakage, missing online features, rules that cancel gains, and changes in latency or degradation rate.
4. A stable hash keeps the same unit in the same bucket. A salt gives each experiment a distinct, reproducible assignment.
5. Label required evidence, then test whether it exists, was retrieved, and entered final context before checking whether generated claims used and cited it.

</details>
