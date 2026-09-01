# ML Coding 06 · Quantization & Alignment: INT8 Quantization and DPO / GRPO / PPO / OPD Losses

This note covers two high-frequency ML coding interview clusters: inference efficiency and post-training alignment. What they share is that every one of them can be written as a pure numerical function that doesn't touch autograd internals: interviewers usually just want the forward formula and the reasoning behind it, not a full training run.

## Module 9: Quantization and Alignment Losses

### Exercise 1 · INT8 Quantization

`Int8Linear` isn't really testing whether you can turn a float into an int. It's testing three things: how per-channel scale is defined, why `scale` can't be an `nn.Parameter`, and how to derive the theoretical error bound.

Symmetric quantization is applied per output channel (i.e., per row) of the weight matrix `W`:

```text
scale[o]      = max(|W[o, :]|) / 127
W_int8[o, i]  = clamp(round(W[o, i] / scale[o]), -127, 127)
```

This is "symmetric" quantization because the zero point is pinned at 0, so there's no need to store a separate `zero_point` (asymmetric quantization needs one, and is typically used for activations, which have skewed non-negative distributions after something like ReLU). The reason for per-channel rather than per-tensor is direct: if the whole weight matrix shared a single scale, one unusually large row would drag down the quantization precision of every other row. Per-channel decouples that.

`scale` (and `zero_point`, in the asymmetric case) must be registered as a `buffer`, not an `nn.Parameter`: it's a deterministic function of the weight statistics, it doesn't participate in backprop, and the optimizer shouldn't touch it. Registering it as a `Parameter` would let something that gradient descent should never update leak into `.parameters()`, and it muddies the semantics of the state_dict.

The error bound falls out in one line: `round()` error lies within `[-0.5, 0.5]` quantization steps, so after dequantizing the error is `|round(x) - x| * scale <= scale / 2`. This is the follow-up question interviewers ask most: "the error scales with `scale`, with a hard `scale/2` ceiling" matters more than reciting the code.

#### Quick Coding: `Int8Linear`

```python
class Int8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

<details>
<summary>Reference solution</summary>

```python
import torch
from torch import nn

class Int8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        # weight: (out_features, in_features), a trained, frozen float weight before quantization
        scale = we\right.abs().amax(dim=1, keepdim=True) / 127.0
        w_int8 = torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8)

        self.register_buffer("w_int8", w_int8)
        self.register_buffer("scale", scale.squeeze(-1))  # (out_features,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features); only the weight is quantized here, activations stay float,
        # the common "weight-only INT8" variant (saves memory at inference, not compute)
        y = torch.einsum("...i,oi->...o", x, self.w_int8.float())
        return y * self.scale
```

```python
# Numeric self-check (reproduced in NumPy so it doesn't depend on a working local PyTorch install)
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(size=(6, 17)) * 3.7
scale = np.abs(W).max(axis=1, keepdims=True) / 127.0
W_int8 = np.clip(np.round(W / scale), -127, 127)
W_dq = W_int8 * scale

max_err_per_row = np.abs(W - W_dq).max(axis=1)
bound = scale.squeeze(-1) / 2.0 + 1e-9
assert np.all(max_err_per_row <= bound)  # every row's max error stays under scale/2
```

</details>

### Exercise 2 · DPO Loss

DPO (Direct Preference Optimization) solves this problem: given a "chosen" and a "rejected" response to the same prompt, write a supervised loss that collapses the entire "train a reward model, then run PPO rollouts" RLHF pipeline into one term.

The derivation starts from the KL-regularized RL objective:

```text
max_pi  E_{y~pi(.|x)}[r(x,y)] - beta * KL(pi(.|x) || pi_ref(.|x))
```

This objective has a closed-form optimum:

```text
pi*(y|x) = pi_ref(y|x) * exp(r(x,y) / beta) / Z(x)
```

Solving for the implicit reward:

```text
r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

`log Z(x)` depends only on the prompt, not on the specific response `y`. Substituting this implicit reward into the Bradley-Terry preference model `P(chosen > rejected) = sigmoid(r(chosen) - r(rejected))`, the `log Z(x)` term cancels exactly when taking the difference, leaving the DPO loss:

```text
loss = -log sigmoid(
    beta * [(logp_chosen - logp_ref_chosen) - (logp_rejected - logp_ref_rejected)]
)
```

That cancellation is exactly why DPO needs no explicit reward model: the reward is already parameterized as "the log-ratio between the policy and the reference," so training that ratio directly is training an implicit reward. The two reference-model terms also act as a per-prompt baseline: if `pi_ref` itself already assigns different probabilities to the chosen and rejected responses because of length or difficulty, subtracting `logp_ref` cancels that difference out, which stops the model from learning shortcuts (e.g. "shorter is better") that have nothing to do with the actual preference.

#### Quick Coding: `dpo_loss`

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float):
    ...
```

<details>
<summary>Reference solution</summary>

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -torch.nn.functional.logsigmoid(logits).mean()
```

```python
# Numeric self-check: a larger margin should strictly lower the loss (monotonicity)
import numpy as np

def dpo_loss_np(pol_c, pol_r, ref_c, ref_r, beta):
    margin = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    return -np.log(1 / (1 + np.exp(-margin)))

beta = 0.1
ref_c, ref_r = -12.0, -13.0
base   = dpo_loss_np(np.array([-10.0]), np.array([-13.0]), np.array([ref_c]), np.array([ref_r]), beta)
better = dpo_loss_np(np.array([-9.0]),  np.array([-13.0]), np.array([ref_c]), np.array([ref_r]), beta)
assert better < base  # chosen's log-prob relative to ref is larger -> bigger margin -> lower loss

worse_reject = dpo_loss_np(np.array([-10.0]), np.array([-11.0]), np.array([ref_c]), np.array([ref_r]), beta)
assert worse_reject > base  # rejected's log-prob relative to ref grows (margin shrinks) -> higher loss
```

</details>

### Exercise 3 · GRPO Loss

GRPO (Group Relative Policy Optimization) targets the most expensive part of PPO: the value function (critic) network. PPO needs a critic roughly as large as the policy to estimate a baseline, and training and running that network costs real compute.

GRPO's approach: sample a group of responses (e.g. K=8) for the same prompt, and replace the value function's job with statistics computed within that group:

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
```

The group mean is an empirical baseline for that batch of samples, and the group standard deviation also normalizes away the fact that reward scale can differ from prompt to prompt. Once `A_i` is computed, the rest of the objective is identical to PPO's clipped surrogate in Exercise 4. That's the one-line summary interviewers are usually fishing for: **GRPO isn't a new loss shape. It's a cheaper way to estimate the advantage, and the outer clip mechanism is lifted from PPO unchanged.**

#### Quick Coding: `grpo_loss`

```python
def grpo_loss(logps, old_logps, rewards, group_ids, eps: float, clip_ratio: float):
    ...
```

<details>
<summary>Reference solution</summary>

```python
def grpo_loss(logps, old_logps, rewards, group_ids, eps: float, clip_ratio: float):
    advantages = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        mask = group_ids == g
        r = rewards[mask]
        advantages[mask] = (r - r.mean()) / (r.std() + eps)

    ratio = torch.exp(logps - old_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()
```

```python
# Numeric self-check: within-group advantages should have mean ~0, std ~1
import numpy as np

rewards = np.array([1.0, 3.0, 2.0, 5.0, 0.0])
eps = 1e-6
A = (rewards - rewards.mean()) / (rewards.std() + eps)
manual = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)
assert np.allclose(A, manual)
assert abs(A.mean()) < 1e-6
assert abs(A.std() - 1.0) < 1e-3
```

</details>

### Exercise 4 · PPO Loss

The core question behind PPO's clipped surrogate objective is: why is plain policy gradient (`-mean(ratio * A)`) unstable, and how exactly does clipping keep it in check?

The importance ratio `ratio = exp(new_logp - old_logp)` measures how much the new policy's preference for an action has shifted relative to the old one. Updating unconstrained on `ratio * A` means that once some action's `ratio` swings large from a single update, the next update extrapolates further from an importance weight that's already far from the true distribution, and one bad step compounds into the next (the exact problem trust-region methods are built to prevent).

```text
loss = -mean( min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) )
```

There's a classic interview trap here: **which side of the clip actually binds depends on the sign of `A`.**

- `A > 0` (this action is better than average, you want to raise its probability): the `min()` only picks the suppressed `clip` branch once `ratio` exceeds `1+eps`. The clip is stopping you from pushing an already-improving action's probability up without limit.
- `A < 0` (this action is worse than average, you want to lower its probability): the `min()` only picks the `clip` branch once `ratio` drops below `1-eps`. This time the clip is stopping you from crushing the probability down too fast.

One-line rule: **the clip always stops an update that's already moving in the correct direction from accelerating further. It never stops the movement itself.** That's why the two conditions are mirror images of each other.

#### Quick Coding: `ppo_loss`

```python
def ppo_loss(new_logps, old_logps, advantages, clip_ratio: float):
    ...
```

<details>
<summary>Reference solution</summary>

```python
def ppo_loss(new_logps, old_logps, advantages, clip_ratio: float):
    ratio = torch.exp(new_logps - old_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()
```

```python
# Numeric self-check: verify the asymmetric clip's activation condition
import numpy as np

clip_ratio = 0.2

def which_bound_binds(r, A):
    unclipped = r * A
    clipped = np.clip(r, 1 - clip_ratio, 1 + clip_ratio) * A
    return min(unclipped, clipped), clipped < unclipped

# A > 0: the clip branch is only selected by min() when r > 1+eps
for r in np.linspace(0.5, 2.0, 7):
    m, clip_active = which_bound_binds(r, A=1.0)
    assert clip_active == (r > 1 + clip_ratio)

# A < 0: the clip branch is only selected by min() when r < 1-eps
for r in np.linspace(0.5, 2.0, 7):
    m, clip_active = which_bound_binds(r, A=-1.0)
    assert clip_active == (r < 1 - clip_ratio)
```

</details>

### Exercise 5 · OPD Loss

The distinction between OPD (On-Policy Distillation) and standard knowledge distillation (KD) is the entire point of this question. Standard KD computes the forward KL `KL(teacher || student)` over the teacher's offline training data (or data sampled from the teacher). OPD flips it: sample sequences from **the student's own current policy**, then compute the reverse KL `KL(student || teacher)` on those student-generated tokens.

```text
KL(student || teacher) = E_{token ~ student}[ log p_student(token) - log p_teacher(token) ]
```

That expectation is taken under the student's own sampling distribution, so the Monte Carlo estimate of the loss is simply "the mean of `log p_student - log p_teacher` over tokens the student itself generated."

Two questions determine how deep this goes:

1. **Why reverse KL instead of forward KL?** Reverse KL is mode-seeking: minimizing `KL(student||teacher)` pushes the student to concentrate its probability mass on regions the teacher considers plausible, without forcing the student to cover every high-probability region of the teacher. The student only has to answer for what it actually generates. Forward KL is mode-covering: it forces the student to keep some probability mass on every mode of the teacher's distribution, even paths the student would never take on its own.
2. **Why does it have to be "on-policy" (sampled from the student)?** If the reverse KL were still computed on the teacher's offline data, the token sequences the student sees during training wouldn't match the distribution it produces at inference (exposure bias): during training the student is always fed a correct prefix, but at inference it has to keep generating on top of prefixes it may have already gotten wrong. Sampling from the student's own rollouts targets the distillation signal at the distribution the student actually encounters, rather than one it never uses at inference.

The asymmetry of KL is itself a common follow-up: `KL(P||Q) != KL(Q||P)`, and the numeric check below computes both directions directly to confirm it.

#### Quick Coding: `opd_loss`

```python
def opd_loss(student_logits, teacher_logits, sampled_tokens):
    ...
```

<details>
<summary>Reference solution</summary>

```python
def opd_loss(student_logits, teacher_logits, sampled_tokens):
    # student_logits, teacher_logits: (batch, seq_len, vocab)
    # sampled_tokens: (batch, seq_len), sampled from the student's own distribution
    log_p_student = torch.log_softmax(student_logits, dim=-1)
    log_p_teacher = torch.log_softmax(teacher_logits, dim=-1)

    log_p_s = torch.gather(log_p_student, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    log_p_t = torch.gather(log_p_teacher, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

    return (log_p_s - log_p_t).mean()  # Monte Carlo estimate of KL(student || teacher)
```

```python
# Numeric self-check: KL asymmetry, and whether opd_loss's Monte Carlo estimate converges to the true value
import numpy as np

outcomes = np.arange(5)
p_student = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
p_teacher = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

kl_fwd = np.sum(p_teacher * (np.log(p_teacher) - np.log(p_student)))  # KL(teacher||student)
kl_rev = np.sum(p_student * (np.log(p_student) - np.log(p_teacher)))  # KL(student||teacher)
assert not np.isclose(kl_fwd, kl_rev)  # the two directions genuinely differ

rng = np.random.default_rng(0)
for n in [100, 10_000, 1_000_000]:
    samples = rng.choice(len(p_student), size=n, p=p_student)
    est = np.mean(np.log(p_student[samples]) - np.log(p_teacher[samples]))
    # est should get closer to kl_rev as n grows
assert abs(est - kl_rev) < 0.01  # check convergence using the last round's estimate (n=1_000_000)
```

</details>

## Summary: what these five questions are really testing

| Problem | Surface-level ask | What's actually being tested |
| --- | --- | --- |
| INT8 Quantization | The quantization formula | Why per-channel is needed, `buffer` vs `Parameter`, deriving the error bound |
| DPO | The loss formula | Solving the closed-form implicit reward from the KL-regularized RL objective, and how `log Z(x)` cancels |
| GRPO | Advantage normalization | The motivation for replacing the value function with within-group statistics |
| PPO | Clip syntax | Which side of the clip protects which case, depending on the sign of `A` |
| OPD | The reverse-KL formula | Mode-seeking vs. mode-covering, and why on-policy sampling matters |

If you only remember one thing: all five questions are asking "what component did this method eliminate, and what mathematical substitution let it do that," not just "can you recite the final formula."
