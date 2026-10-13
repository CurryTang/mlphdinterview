# 05 · Inference Decoding: Sampling & Speculative Decoding

MLCoding01's generation loop already ships the simplest version of top-p sampling. This note fills in the rest of the set that interviewers actually ask about: combining top-k with top-p, understanding beam search's length-bias trap, and the correctness argument behind speculative decoding's "lossless speedup" claim. All three problems operate on logits/probability vectors rather than model internals. The code is written entirely in PyTorch (`torch`), designed to be clean, intuitive, step-by-step, and easy to memorize for whiteboard interviews without skipping steps or relying on obscure syntax tricks.

## Module 9: Decoding Strategies

### Exercise 1: Combined Top-k / Top-p Sampling

Top-k alone cuts off too many reasonable options when the distribution is flat (open-ended continuation). Top-p alone keeps a pile of near-impossible noise tokens when the distribution is peaked (code completion, where only one or two tokens make sense). In practice the two are chained: top-k does a coarse pre-filter (keeps long-tail noise out of the candidate pool), then top-p adaptively truncates what's left (keeps only 1-2 tokens when the distribution is sharp, more when it's flat).

Two edge cases that are easy to miss in an interview:

- `temperature = 0` needs its own argmax branch, because `logits / 0` produces `inf`/`nan`, so you cannot rely on the division degenerating into greedy decoding on its own.
- After top-p truncation the surviving probabilities no longer sum to 1, so they must be renormalized, or `torch.multinomial` will either error out or sample with the wrong bias.

#### Quick Coding: `sample_top_k_top_p`

```python
def sample_top_k_top_p(logits, k=None, p=None, temperature=1.0, generator=None):
    ...
```

<details>
<summary>Reference solution</summary>

```python
import torch

def sample_top_k_top_p(logits, k=None, p=None, temperature=1.0, generator=None):
    logits = torch.as_tensor(logits, dtype=torch.float32)

    # 1. temperature = 0: greedy selection (argmax)
    if temperature == 0:
        idx = torch.argmax(logits).item()
        probs = torch.zeros_like(logits)
        probs[idx] = 1.0
        return idx, probs

    # 2. Temperature scaling and softmax
    probs = torch.softmax(logits / temperature, dim=-1)

    # 3. Top-k filtering: retain only top-k candidates
    if k is not None and k < len(probs):
        topk_probs, topk_indices = torch.topk(probs, k)
        new_probs = torch.zeros_like(probs)
        new_probs[topk_indices] = topk_probs
        probs = new_probs / new_probs.sum()

    # 4. Top-p filtering (Nucleus Sampling): cumulative probability cutoff
    if p is not None and p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=0)

        # Cutoff index where cumulative probability reaches or exceeds p (keep >= 1 token)
        cutoff = torch.searchsorted(cdf, torch.tensor(p)).item() + 1
        cutoff = min(max(cutoff, 1), len(probs))

        keep_indices = sorted_indices[:cutoff]
        keep_probs = sorted_probs[:cutoff]

        new_probs = torch.zeros_like(probs)
        new_probs[keep_indices] = keep_probs
        probs = new_probs / new_probs.sum()

    # 5. Multinomial sampling from final distribution
    idx = torch.multinomial(probs, num_samples=1, generator=generator).item()
    return idx, probs
```

```python
gen = torch.Generator().manual_seed(0)
logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.1, -1.0, -2.0])

_, probs_k = sample_top_k_top_p(logits, k=3, generator=gen)
assert torch.count_nonzero(probs_k) <= 3
assert abs(probs_k.sum().item() - 1.0) < 1e-6

full_probs = torch.softmax(logits, dim=-1)
_, probs_p = sample_top_k_top_p(logits, p=0.8, generator=gen)
kept = probs_p > 0
assert full_probs[kept].sum().item() >= 0.8 - 1e-6  # the surviving set really does clear p under the original distribution

idx0, onehot = sample_top_k_top_p(logits, temperature=0)
assert idx0 == int(torch.argmax(logits).item())
```

Measured (`torch.Generator().manual_seed(0)`, a 7-token toy distribution): `k=3` leaves exactly 3 nonzero entries; `p=0.8` leaves a surviving set whose cumulative probability under the original distribution is about 0.817 (at least 0.8, since the cumulative sum can only cross the threshold, never land on it exactly); `temperature=0` reduces exactly to `argmax`.

</details>

### Exercise 2: Beam Search and Length Bias

Beam search keeps `beam_width` hypotheses alive. At each step every hypothesis is expanded by every vocabulary token, and the global top `beam_width` candidates by **cumulative log-probability** survive. It is not an exact search: an insufficient beam width can miss the globally best sequence, and a surprising number of people can state this fact without being able to point to a concrete step where it happens.

Two classic gotchas:

- **Length bias**: every extra generated token multiplies the cumulative probability by another factor `<1` (equivalently, adds another negative number in log space), so an unnormalized cumulative score is inherently biased toward shorter sequences. The standard fix is length normalization: divide the score by `len(seq) ** alpha` (`alpha` is commonly 0.6-1.0), which is equivalent to comparing average log-probability instead of total log-probability.
- **EOS handling**: a hypothesis that emits EOS is "finished" and should stop expanding, but its final score still has to be compared against the still-active hypotheses in the same candidate pool for the remaining steps. Pulling finished hypotheses out of the comparison early can make the search converge prematurely on a suboptimal short sequence.

#### Quick Coding: `beam_search`

```python
def beam_search(log_prob_fn, start_token, beam_width, max_len, eos_id):
    ...
```

<details>
<summary>Reference solution</summary>

```python
import torch

def length_normalized_score(seq, cum_logprob, alpha=0.7):
    return cum_logprob / (len(seq) ** alpha)

def beam_search(log_prob_fn, start_token, beam_width=3, max_len=20, eos_id=None, alpha=0.0):
    initial_seq = [start_token] if isinstance(start_token, int) else list(start_token)
    beams = [(initial_seq, 0.0, False)]

    for _ in range(max_len):
        candidates = []
        for seq, score, done in beams:
            if done:
                candidates.append((seq, score, True))
                continue

            next_logp = log_prob_fn(seq)
            if not isinstance(next_logp, torch.Tensor):
                next_logp = torch.tensor(next_logp, dtype=torch.float32)

            for tok_id in range(len(next_logp)):
                tok_score = next_logp[tok_id].item()
                is_eos = (eos_id is not None and tok_id == eos_id)
                candidates.append((seq + [tok_id], score + tok_score, is_eos))

        candidates.sort(
            key=lambda x: length_normalized_score(x[0], x[1], alpha) if alpha > 0 else x[1],
            reverse=True
        )
        beams = candidates[:beam_width]

        if all(done for _, _, done in beams):
            break

    beams.sort(
        key=lambda x: length_normalized_score(x[0], x[1], alpha) if alpha > 0 else x[1],
        reverse=True
    )
    return beams[0][0], beams[0][1]
```

Using a toy transition table small enough to brute-force (vocabulary size 2: 0 for X, 1 for Y; fixed depth 2):

```python
import itertools
import torch

# Vocabulary: 0 for X, 1 for Y; -1 for start token <s>
LOG_TABLE = {
    -1: torch.log(torch.tensor([0.6, 0.4])),
    0:  torch.log(torch.tensor([0.5, 0.5])),
    1:  torch.log(torch.tensor([0.9, 0.1])),
}

def toy_log_prob_fn(seq):
    return LOG_TABLE[seq[-1]]

def score(seq):
    prev, s = -1, 0.0
    for tok in seq:
        s += LOG_TABLE[prev][tok].item()
        prev = tok
    return s

def brute_force_best(depth):
    return max(
        (list(c) for c in itertools.product([0, 1], repeat=depth)),
        key=score,
    )

best_seq = brute_force_best(depth=2)                                   # [1, 0] i.e. [Y, X], joint probability 0.36
wide_seq, _ = beam_search(toy_log_prob_fn, start_token=-1, beam_width=2, max_len=2)
narrow_seq, _ = beam_search(toy_log_prob_fn, start_token=-1, beam_width=1, max_len=2)  # beam_width=1 is greedy

assert wide_seq[1:] == best_seq
assert narrow_seq[1:] != best_seq  # greedy picks 0 (X) at step 1 since 0.6 > 0.4, and can never recover
```

Measured result: the single-step probability of `<s> -> X` (0.6) is higher than `<s> -> Y` (0.4), but `X`'s best continuation is only 0.5, while `Y`'s best continuation is 0.9. The globally optimal sequence is `[Y, X]` (joint probability 0.36), and beam width 2 finds it. Beam width 1, equivalent to greedy decoding, locks onto `X` at the very first step and can never recover, ending up with `[X, X]` (joint probability 0.30). This is the minimal counterexample to "greedy decoding is optimal decoding."

</details>

### Exercise 3: Speculative Decoding

The speedup in speculative decoding is direct: a small, cheap draft model proposes `k` candidate tokens autoregressively, and a large, expensive target model needs only **one parallel forward pass**, not `k` sequential ones, to score all `k` positions at once. The hard part isn't the parallelism; it's guaranteeing that, even though sampling used the draft model, the final emitted-token distribution is *exactly* the same as sampling from the target model alone. This is a lossless speedup, not an approximate one, and interviewers routinely ask you to justify that claim.

Acceptance rule (modified rejection sampling): for a draft-proposed token `x`, accept it with probability `min(1, p_target(x) / p_draft(x))`. On rejection, you cannot simply fall back to resampling from `p_target`; you must resample from the corrected residual distribution `max(0, p_target - p_draft)` (renormalized).

Proof that the resulting marginal distribution equals `p_target` exactly (discrete case, summing over the vocabulary `x`):

```text
P(accept x) = p_draft(x) . min(1, p_target(x)/p_draft(x)) = min(p_draft(x), p_target(x))
P(overall rejection) = 1 - sum_x min(p_draft(x), p_target(x))
P(resample yields x | rejected) = P(overall rejection) . residual(x), where
    residual(x) = max(0, p_target(x)-p_draft(x)) / sum_x max(0, p_target(x)-p_draft(x))

Note sum_x max(0, p_target(x)-p_draft(x)) equals P(overall rejection) exactly, because for
every x, p_target(x) - min(p_draft(x), p_target(x)) is either 0 or p_target(x)-p_draft(x),
and summing either side over x gives 1 - sum_x min(...).

So P(resample yields x | rejected) . P(overall rejection) = max(0, p_target(x) - p_draft(x))

Final P(output x) = P(accept x) + max(0, p_target(x) - p_draft(x))
                   = min(p_draft(x), p_target(x)) + max(0, p_target(x) - p_draft(x))
                   = p_target(x)   for every x (check both cases directly:
                     when p_draft(x) <= p_target(x): p_draft(x) + (p_target(x)-p_draft(x)) = p_target(x);
                     when p_draft(x) >  p_target(x): p_target(x) + 0 = p_target(x))
```

In other words: no matter how inaccurate the draft model is, as long as the accept/reject rule is applied correctly, the final output distribution equals the target model's distribution exactly. Draft model quality only affects the acceptance rate, and therefore the speedup, never correctness.

#### Quick Coding: `speculative_accept_step`

```python
def speculative_accept_step(p_target, p_draft, generator=None):
    ...
```

<details>
<summary>Reference solution</summary>

```python
import torch

def speculative_accept_step(p_target, p_draft, generator=None):
    p_target = torch.as_tensor(p_target, dtype=torch.float32)
    p_draft = torch.as_tensor(p_draft, dtype=torch.float32)

    # 1. Draft model proposes a token based on its distribution
    x = torch.multinomial(p_draft, num_samples=1, generator=generator).item()

    # 2. Acceptance probability min(1, p_target(x) / p_draft(x))
    accept_prob = min(1.0, (p_target[x] / p_draft[x]).item())

    # 3. Accept/reject check
    u = torch.rand(1, generator=generator).item()
    if u < accept_prob:
        return x, True

    # 4. On rejection, resample from residual distribution max(0, p_target - p_draft)
    residual = torch.clamp(p_target - p_draft, min=0.0)
    residual = residual / residual.sum()
    resampled_x = torch.multinomial(residual, num_samples=1, generator=generator).item()
    return resampled_x, False

def speculative_decode(target_prob_fn, draft_prob_fn, prompt, num_speculative_tokens, generator=None):
    tokens = list(prompt)
    proposals = []
    draft_probs_list = []

    # 1. Draft model autoregressively proposes K tokens
    for _ in range(num_speculative_tokens):
        p_draft = draft_prob_fn(tokens + proposals)
        if not isinstance(p_draft, torch.Tensor):
            p_draft = torch.tensor(p_draft, dtype=torch.float32)
        tok = torch.multinomial(p_draft, num_samples=1, generator=generator).item()
        proposals.append(tok)
        draft_probs_list.append(p_draft)

    # 2. Target model parallel verification (simulated by per-position checks)
    accepted = []
    for i, x in enumerate(proposals):
        p_target = target_prob_fn(tokens + proposals[:i])
        if not isinstance(p_target, torch.Tensor):
            p_target = torch.tensor(p_target, dtype=torch.float32)
        p_draft_i = draft_probs_list[i]

        accept_prob = min(1.0, (p_target[x] / p_draft_i[x]).item())
        u = torch.rand(1, generator=generator).item()
        if u < accept_prob:
            accepted.append(x)
            continue

        # Reject: resample from residual and discard remaining draft proposals
        residual = torch.clamp(p_target - p_draft_i, min=0.0)
        residual = residual / residual.sum()
        resampled_tok = torch.multinomial(residual, num_samples=1, generator=generator).item()
        accepted.append(resampled_tok)
        break

    return tokens + accepted
```

Monte Carlo verification (5-token toy vocabulary, `torch.Generator().manual_seed(0)`, 200,000 trials):

```python
p_target = torch.tensor([0.05, 0.35, 0.30, 0.20, 0.10])
p_draft  = torch.tensor([0.30, 0.10, 0.10, 0.40, 0.10])

gen = torch.Generator().manual_seed(0)
N = 200_000
samples = torch.tensor([speculative_accept_step(p_target, p_draft, generator=gen)[0] for _ in range(N)])
empirical = torch.bincount(samples, minlength=5).float() / N

assert (empirical - p_target).abs().max().item() < 0.01  # converges to target distribution
assert (empirical - p_draft).abs().max().item() > 0.05   # clearly not draft distribution
```

The measured empirical distribution is `[0.0501, 0.3499, 0.3007, 0.1992, 0.1001]`, a maximum error of about `0.0008` against `p_target = [0.05, 0.35, 0.30, 0.20, 0.10]` (within Monte Carlo noise at 200,000 samples), versus a maximum error of about `0.25` against `p_draft` (clearly not the draft distribution). The acceptance rate equals `sum(min(p_target, p_draft))`, about `0.55` for these numbers, so on average roughly one in every two draft tokens is accepted outright.

</details>
