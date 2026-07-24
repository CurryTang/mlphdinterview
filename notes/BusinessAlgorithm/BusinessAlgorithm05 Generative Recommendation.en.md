# Generative Recommendation and LLM Ranking

## Chapter 13: LLM Ranking and Generative Recommendation

### 13.1 Same Transformer, Different Stages

| Method | Input | Output | What it does not handle |
| --- | --- | --- | --- |
| LLM reranker | query + existing candidates | Score, preference, or ranking | No full-corpus retrieval |
| Generative retrieval | query / user history | item or doc ID | Usually does not produce the final list directly |
| Generative list recommendation | user history and context | Ordered sequence of items | Still requires handling inventory, safety, and service degradation |

When comparing papers, first identify which layer they replace. Using a decoder-only Transformer does not imply the same system boundaries.

### 13.2 Pointwise LLM Ranking

Ask a question for each query-document pair independently:

```text
Query: ...
Document: ...
Is the document relevant? yes/no
```

One can read the logit of the `yes` token as a score, rather than parsing the natural language output.

The advantage is simplicity and parallelizability; the disadvantage is that candidates are not compared against each other, and scores are influenced by prompts and token biases. Calling a large model for every candidate is also very costly.

### 13.3 Pairwise Ranking Prompting

Provide the model with two candidates and ask it to judge which is more relevant:

```text
For query q, which is more relevant, A or B?
```

[PRP](https://aclanthology.org/2024.findings-naacl.97/) (Qin et al., Findings of NAACL 2024) demonstrates that medium-sized open-source models can achieve strong results in a pairwise format. Pairwise comparison is easier than understanding an entire ranking table at once.

Naive pairwise requires comparing $O(n^2)$ pairs. Costs can be reduced using bubble sort passes, tournaments, or local comparisons. One should also swap A/B positions and repeat the query to mitigate position bias.

### 13.4 Listwise and RankGPT

Listwise prompts provide the model with a set of candidates and require the following output:

```text
[4] > [1] > [3] > [2]
```

[RankGPT](https://aclanthology.org/2023.emnlp-main.923/) (Sun et al., EMNLP 2023) uses a sliding window to process long lists and explores distilling large model rankings into a 440M parameter model.

Listwise can compare multiple documents directly, but issues include:

- Candidate order affects results;
- Long documents consume context;
- The model may miss IDs, duplicate IDs, or output illegal formats;
- Sliding windows only see local information;
- Autoregressive generation of complete rankings is slow.

During testing, candidate order should be randomly shuffled to evaluate ranking stability. High NDCG in a single prompt does not prove the absence of position bias.

### 13.5 FIRST

[FIRST](https://aclanthology.org/2024.emnlp-main.491/) (Gangi Reddy et al., EMNLP 2024) does not generate a complete ID ranking; instead, it reads the logits of candidate IDs at the first generation position and uses them to obtain a ranking directly. The paper also incorporates a learning-to-rank loss to penalize errors on highly relevant candidates more heavily.

FIRST uses the representations of generative models without generating the full sequence. The paper reports an inference speedup of approximately 50%; actual gains still depend on candidate length, model size, and deployment method.

### 13.6 Where to Place LLM Ranking

Current common deployment positions include:

- Final-stage reranking of the top 20/50;
- Routing difficult queries to LLMs, while using small models for standard queries;
- Serving as a teacher to generate soft labels or candidate rankings;
- Generating hard negatives for cross-encoders;
- Offline annotation of complex relevance;
- Filtering final context in RAG.

Having a large model rank the top 100 for full-volume, high-QPS search is usually not cost-effective. Model quantization, KV cache, batching, and distillation can reduce costs, but cascading remains essential.

### 13.7 HSTU and Generative Recommenders

[HSTU](https://proceedings.mlr.press/v235/zhai24a.html) (Zhai et al., ICML 2024) frames recommendation as sequence transduction: inputting a sequence of user actions to predict subsequent actions/content. It is designed for the high cardinality, non-stationarity, and ultra-long sequences of recommendation data, rather than simply copying a standard Transformer.

The paper reports:

- Up to 65.8% relative NDCG improvement on public and synthetic data;
- 5.3x to 15.2x speed advantage over FlashAttention2 Transformers at length 8192;
- Deployment of a 1.5 trillion parameter industrial model across multiple scenarios, reporting a 12.4% improvement in online metrics.

These figures come from the data and platforms described in the paper and cannot be directly extrapolated to other businesses. The primary change in HSTU is organizing heterogeneous recommendation features and behavioral sequences into a scalable sequence model and observing its scaling behavior.

The "generative" aspect of HSTU does not equate to generating natural language. It generates target events or items within a recommendation sequence.

### 13.8 OneRec

[OneRec](https://arxiv.org/abs/2502.18965) (Deng et al., 2025 preprint) uses an encoder-decoder to jointly perform retrieval and ranking:

- The encoder reads user history;
- The decoder gradually generates a list of videos for a session;
- Hierarchical discrete codes represent items;
- Sparse MoE expands capacity;
- Reward models and iterative DPO perform preference alignment.

Traditional systems estimate item-by-item and then use rules to assemble a list. OneRec learns directly:

```math
P(i_1,\ldots,i_m\mid H_u)
=\prod_{t=1}^{m}
P(i_t\mid i_{<t},H_u).
```

The subsequent item is conditioned on the already generated list, allowing the model to learn complementarity and repetition within the list. The preprint reports a 1.6% increase in watch-time on Kuaishou's main scenario.

These are industrial results reported by the authors; the paper was treated as a preprint during the compilation of this manual. Unified models must still address issues such as rollbacks, rules, long-tail coverage, invalid IDs, and online decoding costs.

### 13.9 From Positive/Negative Samples to RL

The following methods all increase the probability of high-value actions, but their supervision signals, competitors, and sample weights differ.

| Method | Good/Bad Signal | Competitor | Update Granularity |
| --- | --- | --- | --- |
| BCE | Click, purchase, etc. labels | Single user-item pair | Single item probability |
| BPR | Positive feedback item beats sampled item | Pair of items | Score difference |
| InfoNCE / sampled softmax | Matching item | Batch or sampled candidates | Softmax in representation space |
| next-item CE / SFT | Gold item or SID token | Implicit competitors in full vocabulary | Token or sequence likelihood |
| DPO | Chosen beats rejected | Two complete sequences | Sequence likelihood relative to reference |
| PPO / GRPO methods | Rollout reward and advantage | Sampling from current or near-current policy | Expected reward |

#### DPO Can Be Viewed as Sequence-Level BPR

BPR optimization:

```math
\mathcal L_{\mathrm{BPR}}
=-\log\sigma\left(s(u,i^+)-s(u,i^-)\right).
```

For [DPO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html), first define the sequence score relative to the reference:

```math
g_\theta(x,y)
=\log\pi_\theta(y\mid x)
-\log\pi_{\mathrm{ref}}(y\mid x).
```

The objective becomes:

```math
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\left(
\beta[g_\theta(x,y^+)-g_\theta(x,y^-)]
\right).
```

The two are formally isomorphic: both require the score of the positive example to be higher than that of the negative example. BPR scores are typically user-item scores, while DPO scores are the log probability of the entire generated sequence relative to the reference. The reference limits policy drift but does not solve the data problem for us. Recommendation logs usually only display one list; chosen/rejected pairs must still be constructed via logs, sampling, old policies, or reward models.

#### RL is Advantage-Weighted Dynamic Feedback

The core term of policy gradient is:

```math
\nabla_\theta J
\approx
\mathbb E\left[
A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
```

When `A_t > 0`, the probability of the action is increased; when `A_t < 0`, it is decreased. Thus, RL can be intuitively understood as reward-weighted positive/negative sample learning, though this is merely an analogy. Whether a rollout is positive or negative depends on its advantage relative to the baseline, not just the absolute value of the reward; weights also change with the policy, sampling batch, and baseline.

Compared to fixed negative sampling, policy optimization differs in that:

- Rollouts come from the current or near-current policy, so training follows the errors the model is currently prone to making;
- Rewards can be continuous values, allowing multiple reasonable items or lists for the same request;
- Non-differentiable links such as inventory filtering, posting materialization, rankers, GMV, and diversity can be synthesized into a scalar reward at the end;
- If an action changes subsequent user states, long-term returns across requests can also be optimized.

Only when an action changes subsequent user states and the objective includes cross-request returns is one optimizing a long-term recommendation MDP. Even if SID or slates are generated token-by-token, as long as the reward is given at the end of a single request, it remains closer to sequence-level policy optimization or contextual bandits.

#### A Practical Training Sequence

A common practice is to first learn representations and valid IDs using contrastive learning or CE, then learn stable generation using SFT, and only then consider DPO or online policy optimization. RL is suitable for complete slates, multiple reasonable answers, non-differentiable business metrics, or long-term states; if the goal is simply next-item Recall/NDCG and data is sufficient, CE, BPR, or InfoNCE are often more stable and significantly cheaper.

### 13.10 Checklist for Preference Optimization and RL

Before introducing preference optimization, answer the following:

1. Where do rejected samples come from, and are false negatives mixed in?
2. Can the reward be gamed, and are guardrails for inventory, safety, or diversity missing?
3. How far is the rollout from the online policy, and is the offline data outdated?
4. Are you optimizing a single request, an entire slate, or long-term value across requests?
5. Has the CE/SFT baseline already plateaued, and what has the added complexity gained?

If these questions cannot be answered clearly, switching to DPO, PPO, or GRPO will only hide label bias within a longer training pipeline.

### 13.11 Will Traditional Cascading Disappear?

In the short term, a hybrid system is more likely:

```text
Classic sparse/vector retrieval
        +
Generative retrieval supplement
        ↓
Small model ranking
        ↓
LLM or generative list model processing small candidate sets
        ↓
Hard rules and safety layers
```

Generative models excel at complex intents, long sequences, and joint list modeling; classic systems are easier to satisfy high throughput, incremental updates, explainable debugging, and deterministic constraints. A hybrid architecture allows both types of modules to handle the parts they excel at.

### 13.12 Chapter Self-Test

1. What is the main cost of pointwise, pairwise, and listwise LLM ranking?
2. Why is FIRST faster than generating a complete ranking?
3. Does HSTU's "generative recommendation" mean natural-language generation?
4. Why can DPO be viewed as sequence-level BPR?
5. Under what conditions is recommendation policy optimization truly long-term sequential RL?

<details>
<summary>Reference answers</summary>

1. Pointwise lacks direct candidate comparison and invokes the model per pair; naive pairwise costs `O(n²)`; listwise suffers from context limits, position bias, and malformed output.
2. It reads candidate-ID logits at the first generation position and ranks directly, avoiding autoregressive generation of the complete sequence.
3. No. HSTU formulates user behavior and items as sequence transduction and predicts subsequent events or items.
4. Both optimize a chosen score above a rejected score. BPR uses a user-item score difference; DPO uses a sequence log-probability difference relative to a reference policy.
5. The action must change future user state and the objective must include return across requests. A one-request slate reward is closer to a contextual bandit or sequence-level policy optimization.

</details>
