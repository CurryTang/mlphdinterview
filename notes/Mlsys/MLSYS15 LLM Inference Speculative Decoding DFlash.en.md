# MLSYS17 · Inference: Parallel Decoding and Speculative Verification

Parallel decoding and speculative verification aim to answer four system-level questions:

```text
Why is LLM decoding slow?
How does speculative decoding provide lossless acceleration?
What are the actual differences between Medusa, EAGLE, and DFlash?
In a real serving system, when should it be enabled, and when should it be avoided?
```

The core conclusion first:

> Speculative decoding is not "letting a small model answer for the large model." It is letting a cheap drafter propose several tokens, then having the expensive target model perform a single forward pass to verify these tokens in parallel. As long as the rejection sampling is implemented correctly, the final output distribution is identical to sampling directly from the target model.

---

## Table of Contents

1. [[#I. Why is decoding the bottleneck in inference systems]]
2. [[#II. The simplest speculative decoding]]
3. [[#III. Why it maintains the target model distribution]]
4. [[#IV. Where does the acceleration come from: A performance model you can memorize]]
5. [[#V. From Medusa to EAGLE-3: How drafters get stronger]]
6. [[#VI. MTP: Building the draft model next to the target]]
7. [[#VII. DFlash: Block Diffusion Speculative Decoding]]
8. [[#VIII. Kernel and system scheduling layers: Inference is not just one forward pass]]
9. [[#IX. How Spec Decode is implemented in serving runtimes]]
10. [[#X. How to enable DFlash in SGLang / vLLM]]
11. [[#XI. Engineering judgment: When to enable, when to avoid]]
12. [[#XII. Exercises]]
13. [[#References]]

---

## I. Why is decoding the bottleneck in inference systems

LLM inference can be broken down into two stages:

| Stage | Input | Output | System Characteristics |
|---|---|---|---|
| Prefill | All tokens of the prompt | First KV cache and logits | Large matrix multiplication, large batch/sequence dimensions, compute-heavy |
| Decode | 1 new token at a time | Next token | Strong serial dependency, growing KV cache, memory-bandwidth-bound |

The root cause of slow decoding is not that "the model cannot be parallelized," but the autoregressive dependency:

```text
token_t depends on token_0 ... token_{t-1}
token_{t+1} depends on token_t
```

Therefore, a standard decode loop looks like this:

```python
tokens = prompt_ids
kv_cache = None

for step in range(max_new_tokens):
    logits, kv_cache = target_model.forward(tokens[-1:], kv_cache)
    next_token = sample(logits)
    tokens.append(next_token)
```

For every token generated, the target model must run a forward pass. Even if the input is only one token, the model still must:

- Read large-scale weights
- Read/write the KV cache
- Perform attention / MLP for all layers
- Wait for the token to be sampled before continuing

Consequently, common metrics in inference systems are:

| Metric | Meaning | Most Influenced By |
|---|---|---|
| TTFT | Time to first token | Prefill, queuing, scheduling |
| ITL | Inter-token latency | Decode latency per step |
| TPOT | Time per output token | Average decode cost |
| Throughput | tokens/s or requests/s | Batching, KV, scheduling, spec decode |

Speculative decoding primarily targets the ITL/TPOT of decoding.

### 1.1 Looking at TTFT from the system entry point

Online chat requests typically follow this path:

```text
Client
  -> API Gateway
  -> Session Manager
  -> Message Queue
  -> Inference Service
  -> Model
  -> Streaming Response
```

A slow first token is not necessarily due to a slow model forward pass; it could also be slow due to:

| Layer | Impact on TTFT |
|---|---|
| API Gateway | Authentication, logging, protocol conversion, rate limiting |
| Session Manager | Reading history, system prompt, tool schema, RAG context |
| Message Queue | Queuing during peak hours, providing space for the backend scheduler to form batches |
| Inference Service | Tokenizer, prefill, KV allocation, batching, GPU scheduling |

Rate limiting should not just look at requests per minute. What truly strains LLM services is GPU time and KV cache capacity, so the industry standard metrics are:

```text
requests/minute
tokens/minute
concurrent running sequences
max batched tokens
KV pages in use
```

Session state is not "the model remembering the conversation." Models are typically stateless; multi-turn memory is reorganized by the application layer into the prompt. Recent state can be stored in Redis, while full messages and model replies must be asynchronously persisted. Strict systems place a WAL (Write-Ahead Log) in the request path to prevent session state loss if the process crashes mid-generation.

---

## II. The simplest speculative decoding

Assume two models:

```text
target model q: Large, slow, must maintain its distribution
draft model p: Small, fast, guesses subsequent tokens
```

A speculative step performs three things:

```text
1. The draft model autoregressively generates k candidate tokens
2. The target model performs one forward pass to calculate logits for these k positions in parallel
3. Verify from left to right, accepting the longest valid prefix; if a mismatch occurs, resample based on the corrected distribution
```

Visualization:

```text
context:  A B C

draft proposes:
          d1 d2 d3 d4

target verifies in one pass:
          q1 q2 q3 q4 q5

accepted prefix:
          d1 d2 | reject at d3

next context:
          A B C d1 d2 x
```

The key is that one forward pass of the target model can obtain logits for multiple positions simultaneously:

```text
Input:  [A, B, C, d1, d2, d3, d4]
Output:        q(d1), q(d2), q(d3), q(d4), q(next)
```

A common misunderstanding: acceleration is not because target verification is cheaper for a "single token." Looking at a single target forward pass, verification handles multiple draft tokens, so the workload might be higher than a single-step decode. The real gain comes from **the target collapsing multiple autoregressive steps into a single parallel verification**.

Target-only decoding must proceed like this:

```text
target step 1: context A B C -> sample y1
target step 2: context A B C y1 -> sample y2
target step 3: context A B C y1 y2 -> sample y3
target step 4: context A B C y1 y2 y3 -> sample y4
```

Each step depends on the token sampled in the previous step, so the GPU cannot calculate future logits in advance. Speculative decoding lets the draft provide `d1 d2 d3 d4` first, allowing the target to treat these tokens as known inputs and use a causal mask to calculate the probability for each position under the target distribution in one pass:

```text
q1 = q(d1 | A B C)
q2 = q(d2 | A B C d1)
q3 = q(d3 | A B C d1 d2)
q4 = q(d4 | A B C d1 d2 d3)
q5 = q(next | A B C d1 d2 d3 d4)
```

This step is more like a very short prefill than running four consecutive decodes. It still respects the causal mask; the difference is that the input tokens have already been written by the draft, so the target can generate logits for these positions in parallel in one forward pass.

At the system level, verification speedup comes from several dilution effects:

| Cost | Target-only decode | Speculative verify |
|---|---|---|
| Autoregressive dependency | Each token must wait for the previous one | Draft tokens are known, multiple positions verified together |
| Kernel launch / scheduling | One round per token | Multiple tokens combined into one round |
| Weight reading and GEMM | Small batch, small matrix, poor GPU utilization | Sequence dimension increases, matrix multiplication is fuller |
| KV cache | Read history KV every step | Reuse the same history KV within one forward pass |

Therefore, a more accurate statement is:

```text
Draft proposing is very cheap but still serial;
Target verification is not necessarily cheaper than single-token decoding;
The value of verification is using one target call to advance multiple tokens.
```

Back to the example:

```text
context: A B C
draft:   d1 d2 d3 d4

target calculates q1..q5 in one pass
verification proceeds left to right:
  d1 accepted
  d2 accepted
  d3 rejected

output only commits:
  A B C d1 d2 x
```

Although `d4` after `d3` was also calculated by the target, because the prefix broke at `d3`, the context for `d4` is no longer valid and must be discarded. The `x` here is the token resampled at the rejection position according to the target's corrected distribution. The runtime must also synchronize this: only commit the accepted prefix and the replacement token's KV, discarding the draft state of the rejected suffix.

A simplified code snippet:

```python
def speculative_decode(prompt, target, draft, k):
    tokens = list(prompt)

    while not finished(tokens):
        proposed = []
        draft_ctx = tokens[:]

        for _ in range(k):
            p_logits = draft.forward_next(draft_ctx)
            d = sample(p_logits)
            proposed.append(d)
            draft_ctx.append(d)

        q_logits = target.forward_verify(tokens, proposed)

        accepted, replacement = verify_prefix(proposed, q_logits, draft)
        tokens.extend(accepted)

        if replacement is not None:
            tokens.append(replacement)

    return tokens
```

Real systems don't write it this way because they must handle KV cache, batching, tree attention, routers, streaming, CUDA graphs, logprob returns, etc. But the logic remains: **cheap propose, expensive verify**.

---

## III. Why it maintains the target model distribution

The most common misunderstanding:

> If the draft model guesses wrong and we discard it, does that change the sampling distribution?

Yes, it does. Therefore, classic speculative sampling is not simply "accept if correct, sample from target if wrong," but rejection sampling with correction.

For a draft token `x`:

```text
p(x) = draft model probability
q(x) = target model probability
```

Acceptance probability:

$$
\alpha(x) = \min\left(1, \frac{q(x)}{p(x)}\right)
$$

If accepted, put `x` into the output. If rejected, sample from the corrected distribution:

$$
q'(x) = \frac{\max(q(x) - p(x), 0)}{\sum_y \max(q(y) - p(y), 0)}
$$

Intuition:

```text
Draft gives a token too high a probability:
  p(x) > q(x)
  -> Cannot accept all, accept only at the ratio of q/p

Draft gives a token too low a probability:
  p(x) < q(x)
  -> Accept whenever the draft mentions it, fill the remaining probability mass with q'
```

Pseudocode:

```python
def accept_or_resample(draft_token, p_probs, q_probs, rng):
    p = p_probs[draft_token]
    q = q_probs[draft_token]

    accept_prob = min(1.0, q / max(p, 1e-12))
    if rng.random() < accept_prob:
        return draft_token, True

    residual = (q_probs - p_probs).clip(min=0)
    residual = residual / residual.sum()
    return sample_from_probs(residual, rng), False
```

This is the meaning of "lossless acceleration": it's not that the output text is necessarily the same, but that the sampling distribution is identical to target-only decoding. For greedy / temperature=0 cases, the verification logic degrades into a simpler token match.

---

## IV. Where does the acceleration come from: A performance model you can memorize

Let:

```text
k = number of candidate tokens per draft round
a = average number of accepted tokens
C_t = cost of one target model decode forward pass
C_d = cost of draft generating k tokens
C_v = cost of one target verify pass, usually close to a longer decode/prefill-like forward
```

The cost of generating `a + 1` tokens with standard decoding is approximately:

```text
(a + 1) * C_t
```

The cost of one speculative decode round is approximately:

```text
C_d + C_v
```

So, the rough speedup is:

$$
\text{speedup} \approx \frac{(a + 1) C_t}{C_d + C_v}
$$

This explains all engineering phenomena:

| Phenomenon | Reason |
|---|---|
| More accurate draft is faster | `a` increases, target can advance more tokens per verify |
| Draft too large is slower | `C_d` increases, offsetting verification gains |
| Gains decrease with large batches | Target verification is more easily filled by batches, extra draft cost is more apparent |
| Gains decrease with high-temperature sampling | Draft/target distribution divergence increases, acceptance rate drops |
| Long output tasks benefit more | Decoding accounts for a higher proportion, saving more time |

A one-sentence definition of Speculative decoding:

```text
Use the extra small cost of the draft to trade for multi-token parallel verification by the target model.
Its gains are determined by acceptance length, draft cost, and target verify efficiency.
```

---

## V. From Medusa to EAGLE-3: How drafters get stronger

The earliest draft models were independent small models. But independent small models have two problems:

```text
1. Need to train/deploy/store an extra model
2. Insufficient alignment with target distribution, unstable acceptance rate
```

Many subsequent methods answer the same question:

> How to construct a cheap but more accurate drafter?

### 5.1 Medusa: Adding multiple prediction heads to the target

Medusa adds multiple decoding heads after the target model:

```text
hidden state h_t
  -> head_1 predicts token t+1
  -> head_2 predicts token t+2
  -> head_3 predicts token t+3
```

Its advantages:

- No need to run a separate full draft model
- Heads share the target backbone's hidden state
- Can use tree attention to verify multiple candidate paths at once

The cost:

- Requires modifying the target model or training extra heads
- The further the head, the harder the prediction, candidate quality drops
- Tree attention / candidate packing increases serving complexity

### 5.2 EAGLE: Autoregressive drafting at the feature level

The intuition of the EAGLE series is: predicting tokens directly is too hard; predict the next hidden feature first, then use the target's LM head to get token probabilities.

Simplified diagram:

```text
target hidden feature h_t
  -> lightweight autoregressive drafter predicts h_{t+1}
  -> LM head maps feature to token logits
  -> target verifies
```

EAGLE-3 further uses multi-layer feature fusion, allowing the drafter to see multi-layer target information rather than just one hidden state. This makes candidates more accurate and acceptance rates higher.

The core tradeoff of EAGLE-style methods:

```text
Closer to target than independent small models
Cheaper than direct target decoding
But the draft process is still autoregressive
```

This is important because it leads directly to DFlash.

### 5.3 Drafter taxonomy: From small models to block diffusion

| Route | Where proposals come from | Is draft serial | Target modification | Runtime difficulty | Suitable judgment |
|---|---|---|---|---|---|
| Independent draft model | Separate small model | Yes | No | Dual-model KV, tokenizer/logprob alignment | Easy to integrate, but acceptance rate often unstable |
| Medusa | Multiple heads after target hidden | No, heads parallelizable | Requires adding heads | Tree candidates, tree attention, head training | Target modifiable, hope to maintain fewer models |
| EAGLE / EAGLE-3 | Predict next feature from target feature | Yes | Requires feature plumbing | Feature extraction, drafter autoregressive loop | High acceptance rate, still limited by draft seriality |
| MTP | Target-adjacent multi-token module | Common impl. still sequential | Added during training | MTP state, KVShare, IndexShare, acceptance metadata | Pretraining/inference integration route |
| DFlash | Hidden-state conditioned block diffusion | Parallel within block | Requires draft module | KV injection, anchor, non-causal mask, verify batch | When draft serial cost becomes the bottleneck |

The system implication of this table is: the closer the drafter is to the target, the higher the acceptance rate, but the more complex the hidden states, KV lifecycles, and metadata the runtime must handle. The value of DFlash is not "training another small model," but pushing the draft loop from token-by-token to block-level parallel proposal.

---

## VI. MTP: Building the draft model next to the target

MTP stands for Multi-Token Prediction. It can be a training objective or a built-in drafter for speculative decoding.

The simplest multi-token prediction is:

```text
shared trunk hidden h_t
  -> head_1 predicts token t+1
  -> head_2 predicts token t+2
  -> head_3 predicts token t+3
```

The MTP paper by Gloeckle et al. emphasizes training gains: predicting multiple future tokens at the same position provides denser training signals, especially helpful for coding tasks. During inference, these extra heads can also provide candidates for speculative decoding.

The MTP modules in large models like DeepSeek-V3 / GLM-5 are closer to a sequential MTP module rather than independent heads. It preserves the full causal chain:

```text
target backbone gives h_t

MTP step 1:
  input: h_t, token_t
  output: draft token t+1

MTP step 2:
  input: updated MTP hidden, draft token t+1
  output: draft token t+2

MTP step 3:
  input: updated MTP hidden, draft token t+2
  output: draft token t+3
```

This is slightly more expensive than independent heads but acts more like a small target-adjacent drafter, usually yielding better candidate quality.

### 6.1 Difference between MTP and Medusa

| Method | Draft source | Depends on target hidden | Is draft serial | Main cost |
|---|---|---|---|---|
| Medusa | Multiple decoding heads | Yes | Multi-head parallel | Head training, tree attention |
| EAGLE | Feature-level drafter | Yes | Still autoregressive | Drafter forward and feature plumbing |
| MTP | Target-adjacent MTP module | Yes | Common impl. is sequential | MTP layer, KV sharing, verify metadata |
| DFlash | Block diffusion drafter | Yes | Parallel within block | Diffusion drafter and KV injection |

A one-sentence difference between the four drafters:

```text
Medusa is more like adding heads.
EAGLE is more like predicting target features.
MTP is more like the target having its own small future-token drafter.
DFlash attempts to parallelize the draft block.
```

### 6.2 MTP training objective

The standard next-token loss is:

$$
\mathcal{L}_{NTP} = -\log p(x_{t+1} \mid x_{\le t})
$$

MTP adds multiple future positions:

$$
\mathcal{L}_{MTP} =
\sum_{d=1}^{D}
-\log p_d(x_{t+d} \mid x_{\le t})
$$

In a sequential MTP module, the state of step `d` depends on the input and hidden state of the previous MTP step, rather than future heads being completely independent. The motivation is to ensure the `d`-th prediction respects the causal chain:

```text
predict t+3 should know what was predicted for t+1 and t+2
```

MTP is not just for inference acceleration during training. It also forces the backbone to form representations better at predicting future tokens. The DeepSeek-V3 technical report uses MTP as both a training objective and the foundation for speculative decoding.

### 6.3 How MTP connects to speculative decoding during inference

When MTP acts as a drafter, a round roughly looks like:

```text
1. Target backbone decodes to the current position, obtaining the hidden state
2. MTP module generates k candidate tokens
3. Target model verifies these k tokens
4. Rejection sampling accepts the longest prefix and handles rejected positions
```

Pseudocode:

```python
def mtp_spec_step(context, target, mtp, k):
    hidden, target_kv = target.decode_last(context)

    draft_tokens = []
    mtp_state = mtp.init_from_target_hidden(hidden)
    for _ in range(k):
        logits, mtp_state = mtp.forward_one(mtp_state, draft_tokens[-1:])
        token = sample(logits)
        draft_tokens.append(token)

    q_logits = target.verify(context, draft_tokens, kv_cache=target_kv)
    accepted = rejection_sample(draft_tokens, q_logits)
    return accepted
```

Real implementations don't separate `target.decode_last` and `target.verify` so cleanly. The runtime reuses existing KV cache, constructs a short prefill-like batch with `Q length = k + 1` during verification, and only commits accepted tokens.

### 6.4 GLM-5.2: MTP with IndexShare and KVShare

The MTP path in GLM-5.2 has three key implementation details:

```text
1. The MTP layer also uses IndexShare.
2. In multi-step MTP, the first step runs the indexer, and subsequent steps reuse the top-k indices from the first step.
3. The MTP KV cache only contains KV from target model hidden states, not mixing in subsequent MTP hidden states.
```

This is what KVShare solves: the KV source in MTP training and inference must be consistent.

Without this, the context for the second MTP step would become:

```text
kv1:4 from target model
kv5   from MTP layer
```

If the source mix differs between training and inference, the acceptance length will be affected.

GLM-5.2's approach:

```text
for MTP step 1:
  use target hidden h1:4
  compute indexer top-k

for MTP step 2..D:
  reuse step 1 KV cache
  reuse step 1 top-k indices
  MTP parameters are shared across steps
```

In official ablation studies, combining IndexShare + KVShare, rejection sampling, and end-to-end TV loss increased the MTP acceptance length in coding scenarios from 4.56 to 5.47, an improvement of about 20%.

### 6.5 What you will see in the code

In Transformers' assisted generation, the candidate generator first takes candidate tokens, then lets the target verify `candidate_length + 1` logits. Speculative sampling is used in sampling mode:

```python
probability_ratio = p_target(candidate_token) / q_draft(candidate_token)
accept = random() <= probability_ratio
```

If rejected, sample from the residual distribution:

```python
p_prime = clamp(p_target - q_draft, min=0)
```

This is the form of rejection sampling from theory in engineering code.

The GLM sparse path in ATOM/vLLM also handles sparse attention metadata for MTP. `_should_skip_index_topk` has a dedicated branch for the MTP layer:

```python
if layer_id >= num_hidden_layers and index_share_for_mtp_iteration:
    return True
```

When the metadata builder sees multi-token decoding, it flattens multiple decode tokens of a request into multiple single-token batch entries, extending `seq_lens` and `block_table`. This allows paged MQA logits, top-k, and sparse MLA kernels to reuse the same interface.

### 6.6 System bottlenecks of MTP

MTP sounds like just predicting a few more tokens, but it affects four areas when deployed:

| Module | Change |
|---|---|
| KV cache | Candidate tokens might need temporary caching, only committed after acceptance |
| Scheduler | One verification round consumes multiple token budgets |
| Attention metadata | Decode is not always query length 1 |
| Sampler | Requires alignment between draft logprob and target logprob |

Therefore, MTP gains are not fixed. They depend on:

```text
acceptance length
MTP module cost
target verify efficiency
KV cache headroom
batch occupancy
```

If the batch is already full, or the MTP module occupies too much VRAM, reducing KV cache capacity, end-to-end throughput might decrease rather than increase.

---

## VII. DFlash: Block Diffusion Speculative Decoding

The breakthrough of DFlash is:

> Although traditional EAGLE / MTP methods made the drafter smaller, draft tokens were still generated one by one. DFlash uses a block diffusion drafter to generate an entire block of candidate tokens in parallel.

### 7.1 From autoregressive draft to block draft

Traditional draft:

```text
d1 = f(context)
d2 = f(context, d1)
d3 = f(context, d1, d2)
d4 = f(context, d1, d2, d3)
```

DFlash draft:

```text
[d1, d2, d3, d4, ..., dB] = block_denoise(context, mask_tokens)
```

In other words, the draft stage itself changes from serial to more parallel. This change is critical for latency because `C_d` is a component of the total cost of spec decoding. If the draft itself is serial, `C_d` increases when `k` is large; block diffusion can put multiple tokens into a single denoising pass.

### 7.2 KV injection: Why DFlash is not just "another small model"

DFlash is not blind guessing. It injects the target model's hidden states into the draft model's KV cache, allowing the drafter to be conditioned on target features.

Mental model:

```text
target prefix hidden states
        │
        ▼
draft KV cache receives target features
        │
        ▼
block diffusion drafter predicts token block
```

This design solves two problems simultaneously:

| Problem | DFlash handling |
|---|---|
| Draft too weak, low acceptance rate | Use target hidden features to constrain the drafter |
| Draft too slow, high cost | Use block diffusion to generate token blocks in parallel |

### 7.3 Non-causal attention mask

The attention mask for a standard autoregressive decoder is causal:

```text
token i can only see tokens <= i
```

The DFlash block drafter needs to perform denoising within a block, so the mask is more like:

```text
prefix tokens:     causal / already known
draft block tokens: can attend to prefix and mask-token embeddings
target features:   injected as conditioning context
```

This is why DFlash belongs to "speculative decoding algorithms," but its implementation is close to a specialized diffusion-style draft module.

### 7.4 Anchor mechanism

The vLLM speculators documentation explains the DFlash verification process as anchor-based speculative decoding. Think of it as:

```text
1. Select anchor positions from the context
2. DFlash predicts multiple candidate blocks in parallel from anchors
3. Target model verifies these blocks
4. Accept the longest valid prefix
```

Diagram:

```text
context:  [........ A ........ B ........ C]
anchors:            A          B          C

DFlash proposes blocks:
          A -> a1 a2 a3 a4
          B -> b1 b2 b3 b4
          C -> c1 c2 c3 c4

target verifies:
          accept longest valid prefix, reject/resample when mismatch
```

System gains come from two levels of parallelism:

```text
parallel draft of tokens within a block
parallel verification of multiple anchors / blocks
```

### 7.5 How DFlash fits into the performance model

Back to the formula:

$$
\text{speedup} \approx \frac{(a + 1) C_t}{C_d + C_v}
$$

DFlash does two main things:

```text
Reduces C_d:
  Block diffusion generates multiple candidate tokens in one pass

Increases a:
  KV injection makes the draft distribution closer to the target
```

DFlash demonstrates higher throughput for high-concurrency decoding on models like gpt-oss-120b, Llama 3.1 8B, Qwen3, and Gemma; DFlash papers and vLLM/SGLang documentation also position it as a stronger lossless speculative decoding route than EAGLE-3.

---

## VIII. Kernel and system scheduling layers: Inference is not just one forward pass

What was discussed earlier is the speculative decoding algorithm. In a real inference system, one decode step has at least two layers of scheduling:

```text
request scheduler:
  Which requests enter the current batch?
  Who does prefill, who does decode, who gets preempted?
  Where do KV cache slots come from?

kernel scheduler:
  How to cut tiles for attention/GEMM/MoE kernels in this round?
  How to map variable-length KV to blocks/pages?
  CUDAGraph requires static shapes, but request lengths are dynamic; how to reconcile?
```

This is why the same speculative algorithm that accelerates in a notebook might require a mountain of runtime work to fit into vLLM/SGLang.

### 8.1 Real data flow of a decode step

Standard decoding is not a simple call to `model(input_id)`. A serving runtime typically does the following:

```python
def decode_iteration(waiting, running, kv_allocator):
    # 1. admission control: select requests that can run this round
    batch = scheduler.pick(
        waiting=waiting,
        running=running,
        free_kv_pages=kv_allocator.free_pages(),
        max_num_batched_tokens=MAX_TOKENS,
        max_num_seqs=MAX_SEQS,
    )

    # 2. Allocate KV pages for new tokens
    for req in batch:
        kv_allocator.append_slot(req.request_id)

    # 3. Construct attention metadata
    metadata = build_decode_metadata(batch)

    # 4. launch kernels
    logits = model.decode_one_token(
        input_ids=batch.last_tokens,
        positions=batch.positions,
        kv_cache=kv_allocator.kv_cache,
        metadata=metadata,
    )

    # 5. sample / stop / stream / release finished KV
    next_tokens = sampler(logits, batch.sampling_params)
    scheduler.commit(batch, next_tokens)
```

Key judgment: **The scheduler determines batch shape, and batch shape determines kernel efficiency**. Speculative decoding changes the step granularity: one target verification might advance multiple tokens, so the scheduler can no longer assume "each running request adds only 1 KV slot per round."

The core of continuous batching is not "waiting for a batch of requests to fill up," but re-deciding who enters the batch every decode iteration. A request can release its position immediately after generating EOS; as long as token and KV page budgets allow, new requests can be inserted into the next round. This strategy improves GPU utilization but introduces a common conflict: when long prefills are inserted into a decode batch, they slow down the next token for existing users.

Therefore, the system significance of chunked prefill is to break long prompts into small, schedulable fragments:

```text
long prefill:
  [4096 prompt tokens] blocks decode for a long time

chunked prefill:
  [512] -> decode steps -> [512] -> decode steps -> ...
```

This is very similar to the speculative verification problem: verification is not pure decoding; it carries `k + 1` query tokens, making it more like a short prefill. The scheduler needs to put standard decoding, chunked prefill, and spec verification into the same token budget, rather than queuing only by request count.

### 8.2 Paged KV Cache: Why decoding needs a memory manager

KV cache size is approximately:

```text
num_layers * 2(K,V) * num_tokens * num_kv_heads * head_dim * bytes
```

With long context and high concurrency, the KV cache is more likely to become a service bottleneck than weights. The core of PagedAttention / paged KV is cutting the KV of each sequence into fixed-size pages:

```text
request A logical tokens:
  [0..15] [16..31] [32..47]

physical KV pages:
  page 7 -> A[0..15]
  page 2 -> A[16..31]
  page 9 -> A[32..47]
```

Benefits:

- Variable-length sequences do not require a single contiguous block of VRAM
- Completed requests can release pages immediately
- Scheduler can perform admission control based on free pages

Costs:

- Attention kernels require indirect addressing via page tables
- Page size affects fragmentation and memory access continuity
- Prefix cache / speculative accept / reject all change page lifecycles

The extra problem with speculative decoding is: draft tokens are proposed first, but only the accepted prefix becomes the official context. Therefore, common runtime strategies are:

```text
Before verify:
  Draft tokens can be placed in temporary buffers or speculative slots

After verify:
  Accepted tokens are committed to the official KV cache
  Temporary KV / metadata corresponding to the rejected suffix are discarded
```

If all draft tokens were written directly into the official KV and then rolled back, it would complicate the allocator, streaming, and prefix cache.

### 8.3 CUDA-level: What decode attention kernels do

Below is a highly simplified CUDA perspective. Real implementations have tensor cores, warp-level reduction, vectorized loads, FlashAttention-style online softmax, but the mental model is:

```cuda
// one query token attends to paged KV blocks
__global__ void paged_decode_attention(
    half* q, half* k_cache, half* v_cache,
    int* block_table, int* seq_lens,
    half* out
) {
    int seq = blockIdx.x;       // request in batch
    int head = blockIdx.y;      // attention head
    int lane = threadIdx.x;

    float m = -INFINITY;        // online softmax max
    float l = 0.0f;             // online softmax normalizer
    float acc[HEAD_DIM];        // output accumulator

    for (int logical_block = 0; logical_block < num_blocks(seq); ++logical_block) {
        int physical_block = block_table[seq, logical_block];

        // load K/V tile from physical KV page
        half* k_tile = k_cache + physical_block_offset(physical_block, head);
        half* v_tile = v_cache + physical_block_offset(physical_block, head);

        // q @ k, update online softmax, accumulate p * v
        update_attention_tile(q, k_tile, v_tile, &m, &l, acc);
    }

    store_normalized(out, seq, head, acc, l);
}
```

There are three kernel-level difficulties:

| Difficulty | Why it affects latency |
|---|---|
| Variable-length sequences | Different requests have different `seq_len` and block counts, making SM load balancing difficult |
| Indirect addressing via page table | KV is non-contiguous, making memory coalescing harder |
| Small batch decoding | Each token query is small, compute might not be saturated, HBM access is more of a bottleneck |

The value of kernel libraries like FlashInfer lies here: they don't just provide one attention kernel, but a set of templates and runtime schedulers that adapt to different KV layouts, batch shapes, and attention variants.

### 8.4 FlashInfer's plan/run mode

An important system point in the FlashInfer paper is: request lengths change dynamically, but CUDAGraph prefers static launch configurations. The solution can be understood as two stages:

```text
plan():
  Read seq_lens / page table / qo_indptr of the current batch
  Generate load-balanced scheduling metadata
  Try to make SM workload uniform

run():
  All layers reuse plan metadata
  Launch attention kernel
  Compatible with CUDAGraph / JIT templates
```

This is more stable than "re-allocating work dynamically based on shape for every layer":

```python
plan = flashinfer.plan(
    qo_indptr=batch.qo_indptr,
    paged_kv_indptr=batch.kv_indptr,
    paged_kv_indices=batch.kv_indices,
    num_heads=num_heads,
    head_dim=head_dim,
)

for layer in layers:
    hidden = layer.attn.run(hidden, kv_cache[layer], plan)
    hidden = layer.mlp(hidden)
```

This design explains a common phenomenon: LLM inference optimization is not just "writing a faster CUDA kernel," but designing the **kernel API, metadata format, scheduler, and CUDAGraph replay** together.

### 8.5 Prefill / Decode disaggregation

Prefill and decode have different hardware behaviors:

| Stage | Kernel form | System goal |
|---|---|---|
| Prefill | Large matrix, large sequence, compute-heavy | Get first token ASAP, consume long prompts |
| Decode | 1 or few tokens per step, memory-heavy | Stable ITL, sustained high concurrency |

Mixing prefill and decode on the same GPU batch causes interference:

```text
long prefill fills the batch token budget
  -> decode request waits
  -> p95/p99 ITL degrades
```

Modern serving systems perform chunked prefill, prefill/decode separation, or dedicate different instances to different phases. Speculative decoding exacerbates this: target verification is somewhat like a "short prefill," inputting `context + draft block`, making the decode step more prefill-like. The scheduler must decide:

```text
Is this round for letting more standard decodes in?
Or for performing speculative verification for certain requests?
Will a verify block that is too long slow down others' ITL?
```

This is why DFlash / EAGLE are truly difficult in serving: the algorithm only talks about acceptance length, but the system must also consider batch interference.

### 8.6 Admission control: When not to enter the GPU at all

Admission control must be performed before the expensive GPU stage, rather than letting all requests queue into the GPU. This principle applies equally to LLM serving, just with the granularity shifting from "a long video job" to "a request's prefill / decode / tool-call episode."

Admission can be split into three layers:

| Layer | Decision |
|---|---|
| quota / rate limit | Does this user have quota; limit by request, token, and concurrency |
| resource check | Are there enough KV pages, batch token budget, and target model replicas |
| traffic shaping | free / paid / enterprise priority; whether to downgrade, queue, or transfer to another pool |

For multi-stage tasks, DAGs and state machines are clearer than a single queue:

```text
ready -> scheduling -> running -> completed
running -> retry -> ready
running -> failed
```

If a stage requires multiple GPUs, such as large model TP/PP inference, MoE expert parallel, or video DiT's CP/TP combinations, gang scheduling must be performed: do not start if the full GPU set is not available. Otherwise, partially started jobs will occupy resources without making progress.

Preemption must also be staged. Cheap stages, such as tokenizer, safety check, prompt rewrite, and RAG retrieval, can be paused and retried; once the model forward pass begins, saving activation state is often not worth it. A more practical approach is to preempt before the GPU forward pass begins, or let running jobs drain before inserting new high-priority requests into the next round.

---

## IX. How Spec Decode is implemented in serving runtimes

### 9.1 Runtime state machine

A request supporting speculative decoding is usually not in a single `RUNNING` state, but more like:

```text
PREFILL
  -> DRAFTING
  -> VERIFYING
  -> COMMIT_ACCEPTED
  -> STREAM_OUTPUT
  -> DRAFTING ...
```

Pseudocode:

```python
while not req.done:
    if req.state == "DRAFTING":
        req.draft_ids = drafter.propose(req.context, k=req.k)
        req.state = "VERIFYING"

    elif req.state == "VERIFYING":
        q_logits = target.verify(req.context, req.draft_ids)
        accepted, replacement = rejection_sample(req.draft_ids, q_logits)
        req.pending_commit = accepted + maybe(replacement)
        req.state = "COMMIT_ACCEPTED"

    elif req.state == "COMMIT_ACCEPTED":
        kv_cache.commit(req.id, req.pending_commit)
        streamer.emit(req.pending_commit)
        req.state = "DRAFTING"
```

This state machine must coexist with continuous batching. That is, the same batch might contain:

```text
Standard decode requests
DFlash draft requests
Target verify requests
Prefill requests
Requests that just finished and need to release KV
```

The essence of the scheduler is to pack these different types of work into the same GPU batch, balancing throughput, TTFT, ITL, and VRAM.

### 9.2 Why verify kernels are close to prefill

When verifying `k` draft tokens, the target does not run token by token, but inputs:

```text
[context, d1, d2, ..., dk]
```

Calculating logits for each position of the draft suffix in parallel:

```text
logits for d1: target(context)
logits for d2: target(context, d1)
...
logits for dk: target(context, d1, ..., d{k-1})
```

At the kernel level, this is more like a short sequence prefill:

```text
Q length = k + 1
KV length = context_len + k
attention mask = causal
```

Therefore, it consumes more `max_num_batched_tokens` and introduces more temporary buffers for activations/logits. If you only look at tokens/s online, it's easy to ignore the squeeze verification puts on the ITL of other requests.

### 9.3 DFlash specifically needs hidden-state plumbing

DFlash is not just "one more draft model." It requires target hidden features as a condition:

```text
target model:
  context -> hidden features

DFlash drafter:
  hidden features + masked block -> draft block
```

Therefore, the serving runtime must additionally support:

| Requirement | System impact |
|---|---|
| Expose target hidden states | Target forward API cannot return only logits |
| Hidden feature cache | Avoid redundant calculation, but control VRAM |
| Drafter KV injection | Draft model's KV cache must align with target features |
| Online speculator training | Drafter must keep up when policy / target changes |

This explains why DFlash/EAGLE productionization usually relies on runtimes like vLLM/SGLang, rather than adding a few lines of Python to a Transformers loop.

### 9.4 Correctness boundary: Lossless does not mean simple implementation

The theoretical correctness of speculative decoding depends on three conditions:

```text
1. The proposal probability p of draft tokens can be correctly obtained
2. The target distribution q is calculated on the same context
3. Sample from the residual distribution q' after rejection
```

Common points of failure in system implementation:

| Failure point | Consequence |
|---|---|
| Inconsistent tokenizer / chat template | Draft tokens and target tokens don't match |
| Target verify context missing an accepted token | q is not the same conditional distribution |
| top-p/top-k mask not synchronized | Residual distribution is wrong |
| FP8/INT8 quantized draft logprob not calibrated | Acceptance ratio is biased |
| Streaming unverified tokens | User sees content that is later rejected |

Therefore, production systems typically only stream accepted tokens that have already been committed, rather than draft tokens.

---

## X. How to enable DFlash in SGLang / vLLM

In real deployment, you won't hand-write `verify_prefix`. You will enable the corresponding spec decode backend in the serving runtime.

### 10.1 SGLang example

Taking the public Qwen3-8B DFlash checkpoint as an example, the SGLang launch command is roughly:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --tp-size 1 \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --mem-fraction-static 0.75 \
  --trust-remote-code
```

What to focus on is not memorizing the command, but what these parameters correspond to:

| Parameter | Meaning |
|---|---|
| `--model-path` | Target model |
| `--speculative-algorithm DFLASH` | Enable DFlash verifier/drafter logic |
| `--speculative-draft-model-path` | DFlash drafter checkpoint |
| `--attention-backend` | Whether to use efficient attention backends like FA3 |
| `--mem-fraction-static` | Reserved memory for KV / graph / runtime |

### 10.2 vLLM example

vLLM speculator configuration is usually placed in JSON parameters:

```bash
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3-8B-DFlash-b16", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

`num_speculative_tokens` is not "the bigger the better":

```text
Too small:
  Target verify advances few tokens, limited gains

Moderate:
  High acceptance rate, draft cost controllable

Too large:
  Acceptance rate drops in the second half, extra draft/verify costs increase
```

Online, one usually profiles based on model, sampling parameters, request length, and batch state to find the global optimum, rather than fixing a value.

---

---

## XI. Engineering judgment: When to enable, when to avoid

### 11.1 Scenarios suitable for speculative decoding

| Scenario | Reason |
|---|---|
| Long output | Many decode tokens, large space savings |
| Low temperature / greedy / deterministic workload | High draft acceptance |
| Target model is very large | Target forward is expensive, multi-token verify is more valuable |
| Batch is not always full | Spec decode can pull down single-request latency |
| Drafter is very close to target | Stable acceptance length |

### 11.2 Scenarios that might not be suitable

| Scenario | Risk |
|---|---|
| High-temperature creative writing | Draft/target distribution divergence, low acceptance rate |
| Extremely short output | Draft initialization and verify overhead not worth it |
| Serving batch is already huge | Target throughput is fully diluted, spec decode gains decrease |
| Drafter occupies too much VRAM | KV cache available space decreases, reducing concurrency |
| Target/draft tokenizer mismatch | Correctness and implementation complexity are dangerous |

### 11.3 Metrics you must watch online

```text
spec/acceptance_length_p50
spec/acceptance_length_p90
spec/rejection_rate
spec/draft_latency_ms
spec/verify_latency_ms
serving/itl_ms
serving/tpot_ms
serving/gpu_memory_used
serving/max_running_requests
```

A common failure mode:

```text
Benchmark tokens/s increases
But p95 latency degrades
```

The reason might be that the drafter occupied VRAM, KV cache capacity decreased, and the scheduler queued more frequently. Speculative decoding is a system optimization, not a single-kernel optimization; it must be viewed together with the scheduler / memory manager.

---

### 11.4 Reading papers with system questions in mind

System relationships of recent lines:

| Paper/Implementation | What you should focus on |
|---|---|
| FlashInfer | How dynamic request shapes coexist with CUDAGraph, JIT kernels, and load-balanced scheduling |
| vLLM / SGLang | How request scheduler, paged KV, prefix cache, chunked prefill, and spec decode combine |
| EAGLE-3 / EAGLE 3.1 | After drafter quality improves, serving bottleneck shifts from target to draft / verify plumbing |
| DFlash | Uses block diffusion to reduce autoregressive draft cost, but requires runtime support for hidden-state extraction and KV injection |
| MiniMax-M2 Series | Prefill/decode disaggregation, MTP spec decode, and global KV cache pool appear together in agent RL serving |

Answer structure:

```text
DFlash reduces draft serial cost and increases acceptance length;
But the serving runtime must handle hidden-state plumbing, verify batch scheduling,
KV commit/rollback, streaming correctness, and p95 ITL protection.
```

---

## XII. Exercises

<details class="exercise">
<summary><span class="q-label">Q1</span> <span class="q-text">Why is speculative decoding not distillation?</span></summary>

Distillation is letting a small model learn from a large model, potentially using the small model for output directly. In speculative decoding, the final distribution is still determined by the target model; the draft only proposes candidates. Candidates are verified by the target and corrected using rejection sampling when necessary.

</details>

<details class="exercise">
<summary><span class="q-label">Q2</span> <span class="q-text">Why can the target verify multiple tokens in one forward pass?</span></summary>

Because after inputting `context + draft_tokens` to the target, the transformer calculates logits for each position in parallel. Whether the `i`-th draft token is reasonable can be verified using the target logits of its preceding position.

</details>

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">What is the difference between acceptance rate and acceptance length?</span></summary>

Acceptance rate usually looks at the token-level acceptance ratio; acceptance length looks at how many tokens are advanced on average per verification round. System acceleration depends more directly on acceptance length because it determines how many output tokens the target forward pass is amortized over.

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">One-sentence difference between DFlash and EAGLE?</span></summary>

EAGLE-3 is still a lightweight autoregressive drafter, just using multi-layer target features to improve candidate quality. DFlash uses a block diffusion drafter to generate token blocks in parallel and conditions them on target hidden states via KV injection, simultaneously reducing draft latency and increasing acceptance rate.

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">Why is DFlash lossless?</span></summary>

As long as verification and rejection sampling follow the target distribution, the output distribution is identical to target-only decoding. DFlash changes the candidate generation method, not the principle that the target verifies and corrects the final output.

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">What is the first step to enabling spec decode online?</span></summary>

Record a baseline on the real workload first:

```text
prompt length distribution
output length distribution
temperature / top_p
batch occupancy
ITL / TTFT / throughput
GPU memory headroom
```

Then enable spec decode gradually, comparing acceptance length, latency, memory, and p95/p99, rather than just looking at single-request demos.

</details>

<details class="exercise">
<summary><span class="q-label">Q7</span> <span class="q-text">Why might speculative decoding be less cost-effective when the batch is full?</span></summary>

When the target decode batch can already saturate the GPU, the extra drafter forward, verify packing, temporary KV slots, and rollback management of spec decode consume the gains. It is most suitable for scenarios where target single-token decoding is limited by launches / HBM / small batches.

</details>

<details class="exercise">
<summary><span class="q-label">Q8</span> <span class="q-text">What is the difficulty of KV commit / rollback in spec decode?</span></summary>

Only the accepted prefix of draft tokens can enter the official context. The runtime must distinguish between speculative slots and committed KV: KV for accepted tokens can be committed, while KV for the rejected suffix must be released or ignored. If written directly to the official block table, prefix cache, streaming output, and allocator all become complex.

</details>

<details class="exercise">
<summary><span class="q-label">Q9</span> <span class="q-text">Why does MTP with KVShare emphasize consistent KV sources?</span></summary>

When multi-step MTP generates future tokens, if subsequent steps write MTP hidden states into the KV, it becomes inconsistent with the context definition of training/target verification. KVShare allows MTP to use KV from target hidden states, avoiding semantic drift between draft context and target context.

</details>

<details class="exercise">
<summary><span class="q-label">Q10</span> <span class="q-text">Why does DFlash need hidden-state plumbing?</span></summary>

The DFlash drafter doesn't just look at token IDs; it uses target hidden features as a condition via KV injection. The runtime must expose hidden states from the target, write them into the drafter KV, and maintain these extra states during verification, streaming, and scheduling.

</details>

---

## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [GLM-5.2 official blog](https://huggingface.co/blog/zai-org/glm-52-blog)
- [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)
- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)
- [vLLM: Inside vLLM, Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [vLLM EAGLE 3.1 Blog](https://vllm.ai/blog/2026-05-26-eagle-3-1)
- [NVIDIA: Boost Inference Performance up to 15x on Blackwell Using DFlash Speculative Decoding](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/)
- [LMSYS: Next-Generation Speculative Decoding with DFlash and Spec V2](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- [vLLM Speculators DFlash Documentation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)
- [z-lab/dflash GitHub](https://github.com/z-lab/dflash)
- [Qwen3-8B-DFlash-b16 Hugging Face model card](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16)
- [MiniMax-M2 Series Technical Report](https://arxiv.org/abs/2605.26494)
