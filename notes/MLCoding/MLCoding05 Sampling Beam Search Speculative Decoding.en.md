# ML Coding 05 · Decoding Strategies: Top-k/Top-p Sampling, Beam Search, and Speculative Decoding

MLCoding01's generation loop already ships the simplest version of top-p sampling. This note fills in the rest of the set that interviewers actually ask about: combining top-k with top-p, understanding beam search's length-bias trap, and the correctness argument behind speculative decoding's "lossless speedup" claim. All three problems operate on logits/probability vectors rather than model internals, so the code is plain NumPy, the same shape a whiteboard answer would take.

## Module 9: Decoding Strategies

### Exercise 1: Combined Top-k / Top-p Sampling

Top-k alone cuts off too many reasonable options when the distribution is flat (open-ended continuation). Top-p alone keeps a pile of near-impossible noise tokens when the distribution is peaked (code completion, where only one or two tokens make sense). In practice the two are chained: top-k does a coarse pre-filter (keeps long-tail noise out of the candidate pool), then top-p adaptively truncates what's left (keeps only 1-2 tokens when the distribution is sharp, more when it's flat).

Two edge cases that are easy to miss in an interview:

- `temperature = 0` needs its own argmax branch, because `logits / 0` produces `inf`/`nan`, so you cannot rely on the division degenerating into greedy decoding on its own.
- After top-p truncation the surviving probabilities no longer sum to 1, so they must be renormalized, or `np.random.choice` / `torch.multinomial` will either error out or sample with the wrong bias.

#### Quick Coding: `sample_top_k_top_p`

```python
def sample_top_k_top_p(logits, k=None, p=None, temperature=1.0, rng=None):
    ...
```

<details>
<summary>Reference solution</summary>

```python
import numpy as np

def sample_top_k_top_p(logits, k=None, p=None, temperature=1.0, rng=None):
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64)

    if temperature == 0:
        idx = int(np.argmax(logits))
        probs = np.zeros_like(logits)
        probs[idx] = 1.0
        return idx, probs

    scaled = (logits / temperature) - (logits / temperature).max()
    probs = np.exp(scaled)
    probs /= probs.sum()

    if k is not None and k < len(probs):
        keep = np.argsort(-probs)[:k]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()

    if p is not None and p < 1.0:
        order = np.argsort(-probs)
        cdf = np.cumsum(probs[order])
        cutoff = max(int(np.searchsorted(cdf, p)) + 1, 1)
        keep = order[:cutoff]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()

    idx = rng.choice(len(probs), p=probs)
    return idx, probs
```

```python
rng = np.random.default_rng(0)
logits = np.array([2.0, 1.5, 1.0, 0.5, 0.1, -1.0, -2.0])

_, probs_k = sample_top_k_top_p(logits, k=3, rng=rng)
assert np.count_nonzero(probs_k) <= 3
assert abs(probs_k.sum() - 1.0) < 1e-9

full_probs = np.exp(logits - logits.max())
full_probs /= full_probs.sum()
_, probs_p = sample_top_k_top_p(logits, p=0.8, rng=rng)
kept = probs_p > 0
assert full_probs[kept].sum() >= 0.8 - 1e-9  # the surviving set really does clear p under the original distribution

idx0, onehot = sample_top_k_top_p(logits, temperature=0)
assert idx0 == int(np.argmax(logits))
```

Measured (`np.random.default_rng(0)`, a 7-token toy distribution): `k=3` leaves exactly 3 nonzero entries; `p=0.8` leaves a surviving set whose cumulative probability under the original distribution is about 0.817 (at least 0.8, since the cumulative sum can only cross the threshold, never land on it exactly); `temperature=0` reduces exactly to `argmax`.

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
def beam_search(log_prob_fn, vocab, start_token, beam_width, max_len, eos_id):
    # log_prob_fn(prev_token) -> {token: log_prob}; the toy version only conditions on the previous token
    beams = [([start_token], 0.0, False)]
    for _ in range(max_len):
        candidates = []
        for seq, score, done in beams:
            if done:
                candidates.append((seq, score, True))
                continue
            next_logp = log_prob_fn(seq[-1])
            for tok in vocab:
                candidates.append((seq + [tok], score + next_logp[tok], tok == eos_id))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
        if all(done for _, _, done in beams):
            break
    beams.sort(key=lambda x: x[1], reverse=True)
    return beams[0][0], beams[0][1]

def length_normalized_score(seq, cum_logprob, alpha=0.7):
    return cum_logprob / (len(seq) ** alpha)
```

Using a toy transition table small enough to brute-force (vocabulary `{X, Y}`, fixed depth 2):

```python
import itertools
import numpy as np

VOCAB = ["X", "Y"]
TABLE = {
    "<s>": {"X": 0.6, "Y": 0.4},
    "X":   {"X": 0.5, "Y": 0.5},
    "Y":   {"X": 0.9, "Y": 0.1},
}
LOGTABLE = {prev: {t: np.log(p) for t, p in d.items()} for prev, d in TABLE.items()}

def score(seq):
    prev, s = "<s>", 0.0
    for tok in seq:
        s += LOGTABLE[prev][tok]
        prev = tok
    return s

def brute_force_best(depth):
    return max(
        (list(c) for c in itertools.product(VOCAB, repeat=depth)),
        key=score,
    )

def toy_beam_search(beam_width, depth):
    beams = [([], 0.0)]
    for _ in range(depth):
        cand = []
        for seq, sc in beams:
            prev = seq[-1] if seq else "<s>"
            for tok in VOCAB:
                cand.append((seq + [tok], sc + LOGTABLE[prev][tok]))
        cand.sort(key=lambda x: x[1], reverse=True)
        beams = cand[:beam_width]
    return max(beams, key=lambda x: x[1])

best_seq = brute_force_best(depth=2)                      # ['Y', 'X'], probability 0.36
wide_seq, _ = toy_beam_search(beam_width=2, depth=2)
narrow_seq, _ = toy_beam_search(beam_width=1, depth=2)     # greedy = beam_width 1

assert wide_seq == best_seq
assert narrow_seq != best_seq   # greedy only sees 0.6 > 0.4 at step 1, picks X, and can never recover
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
def speculative_accept_step(p_target, p_draft, rng):
    ...
```

<details>
<summary>Reference solution</summary>

```python
import numpy as np

def speculative_accept_step(p_target, p_draft, rng):
    vocab_size = len(p_target)
    x = rng.choice(vocab_size, p=p_draft)          # draft proposes a token
    accept_prob = min(1.0, p_target[x] / p_draft[x])
    if rng.random() < accept_prob:
        return x, True
    residual = np.maximum(0.0, p_target - p_draft)
    residual /= residual.sum()
    return rng.choice(vocab_size, p=residual), False

def speculative_decode(target_logits_fn, draft_logits_fn, prompt, num_speculative_tokens, rng):
    # target_logits_fn / draft_logits_fn: given the current context, return the full next-token distribution
    tokens = list(prompt)
    proposals = []
    for _ in range(num_speculative_tokens):
        p_draft = draft_logits_fn(tokens + proposals)
        proposals.append(int(rng.choice(len(p_draft), p=p_draft)))

    # one parallel target-model forward pass scores every proposed position at once
    accepted = []
    for i, x in enumerate(proposals):
        p_target = target_logits_fn(tokens + proposals[:i])
        p_draft_i = draft_logits_fn(tokens + proposals[:i])
        accept_prob = min(1.0, p_target[x] / p_draft_i[x])
        if rng.random() < accept_prob:
            accepted.append(x)
            continue
        residual = np.maximum(0.0, p_target - p_draft_i)
        residual /= residual.sum()
        accepted.append(int(rng.choice(len(p_target), p=residual)))
        break  # after the first rejection, all remaining draft tokens are discarded; the resampled token ends this round
    return tokens + accepted
```

Monte Carlo verification (5-token toy vocabulary, `np.random.default_rng(0)`, 200,000 trials):

```python
p_target = np.array([0.05, 0.35, 0.30, 0.20, 0.10])
p_draft  = np.array([0.30, 0.10, 0.10, 0.40, 0.10])

rng = np.random.default_rng(0)
samples = np.array([speculative_accept_step(p_target, p_draft, rng)[0] for _ in range(200_000)])
empirical = np.bincount(samples, minlength=5) / 200_000

assert np.max(np.abs(empirical - p_target)) < 0.01       # converges to the target distribution
assert np.max(np.abs(empirical - p_draft)) > 0.05         # clearly not the draft distribution
```

The measured empirical distribution is `[0.0499, 0.3497, 0.3018, 0.1990, 0.0996]`, a maximum error of about `0.0018` against `p_target = [0.05, 0.35, 0.30, 0.20, 0.10]` (within Monte Carlo noise at 200,000 samples), versus a maximum error of about `0.25` against `p_draft` (clearly not the draft distribution). The acceptance rate equals `sum(min(p_target, p_draft))`, about `0.55` for these numbers, so on average roughly one in every two draft tokens is accepted outr\right.

</details>
