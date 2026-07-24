# ML Coding · Training Components

Corresponds to CS336 Assignment 1: Section 4.

Usage: For each problem, first review the objectives and acceptance criteria, then complete the TODOs using the "Problem Template." Finally, expand the reference solution to check against boundary conditions, sanity checks, and implementation details.

## Exercise 1 · Cross-Entropy

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `cross_entropy`

Interface:

```text
inputs: (..., vocab_size)
targets: (...)
return: scalar mean loss
```

Key constraints:

- Use the log-sum-exp form.
- Subtract the maximum logit to ensure numerical stability.
- Support arbitrary leading batch dimensions.
- Return the batch mean.

Tests:

```bash
uv run pytest -k test_cross_entropy
```

Problem Template:

```python
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Input:
        inputs: (..., vocab_size) logits
        targets: (...) integer class ids
    Output:
        scalar mean cross entropy
    """
    logits = ...
    y = ...
    max_logits = ...
    log_sum_exp = ...
    correct_logits = ...
    return ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The input to cross entropy is logits, not probabilities. For a stable implementation, use log-sum-exp:

```python
def cross_entropy(inputs, targets):
    # inputs: (..., vocab), targets: (...)
    logits = inputs.reshape(-1, inputs.shape[-1])
    y = targets.reshape(-1)

    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + max_logits.squeeze(-1)
    correct = logits[torch.arange(logits.shape[0], device=logits.device), y]
    loss = log_sum_exp - correct
    return loss.mean()
```

Equivalent understanding:

```text
CE = -log softmax(logits)[target]
   = logsumexp(logits) - logits[target]
```

Checkpoints:

- The shape of targets must match the shape of inputs after removing the final vocab dimension.
- Return a scalar.
- Do not apply softmax before log; this results in worse numerical stability.

</details>

## Lab · SGD LR Toy Sweep

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `learning_rate_tuning`

Objective: Observe the impact of learning rate on a toy quadratic loss.

Experiment:

```text
lr = 1e1
lr = 1e2
lr = 1e3
steps = 10
```

Record:

```text
loss curve
final loss
converged / oscillated / diverged
```

Experiment Template:

```python
def run_sgd_lr(lr: float, steps: int = 10) -> list[float]:
    """
    Input:
        lr and number of update steps
    Output:
        loss values over time
    """
    x = ...
    losses = []
    for step in range(steps):
        loss = ...
        grad = ...
        x = ...
        losses.append(...)
    return losses

def sweep_lrs(lrs):
    return {lr: run_sgd_lr(lr) for lr in lrs}
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The purpose of the toy sweep is not to train a model, but to observe the relationship between learning rate and curvature. You can use a 1D quadratic:

```python
def run_sgd_lr(lr, steps=10):
    x = torch.tensor([10.0])
    values = []
    for _ in range(steps):
        loss = x.pow(2).sum()
        grad = 2 * x
        x = x - lr * grad
        values.append(float(loss))
    return values
```

For `f(x)=x^2`, the gradient descent update is:

```text
x_{t+1} = (1 - 2lr) x_t
```

Therefore:

| lr | Behavior |
|---|---|
| Very small | Converges slowly |
| Close to 0.5 | Approaches 0 rapidly |
| Greater than 1 | Oscillates and diverges |

If the assignment specifies `1e1/1e2/1e3`, these learning rates will clearly diverge; when recording the loss curve, note that this is to demonstrate instability, not to suggest these as recommended values.

</details>

## Exercise 2 · AdamW

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `adamw`

Interface:

```text
AdamW(params, lr, betas, eps, weight_decay)
step()
```

State:

```text
t
m: first moment
v: second moment
```

Key constraints:

- Decoupled weight decay.
- Bias correction uses a timestep starting from 1.
- Skip if `p.grad is None`.
- Store state per parameter.

Tests:

```bash
uv run pytest -k test_adamw
```

Problem Template:

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = ...
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                ...  # initialize step, m, v
                ...  # update biased moments
                ...  # bias correction
                ...  # decoupled weight decay
                ...  # parameter update
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The key to AdamW is decoupled weight decay: perform the Adam update based on the gradient first, then apply weight decay separately, or equivalently, add `weight_decay * p` into the update, but do not mix weight decay into the gradient moments.

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                update = m_hat / (torch.sqrt(v_hat) + eps)

                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss
```

Common mistakes:

- Starting bias correction with a timestep of 0.
- Updating state even when `p.grad is None`.
- Adding weight decay into the gradient, resulting in Adam + L2, rather than AdamW.

</details>

## Exercise 3 · AdamW Accounting

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `adamw_accounting`

Output:

```text
parameter memory
activation memory
gradient memory
optimizer state memory
total memory as function of batch_size
AdamW FLOPs per step
training time under MFU assumption
```

Checkpoints:

- AdamW state includes at least `m` and `v`.
- Gradients and parameters are of the same order of magnitude.
- Backward pass is approximately 2x forward FLOPs.

Problem Template:

```python
def adamw_memory_accounting(num_params, param_bytes=2, grad_bytes=2, state_bytes=4):
    """
    Output:
        memory bytes for params, grads, Adam moments, and total
    """
    return {
        "parameters": ...,
        "gradients": ...,
        "adam_m": ...,
        "adam_v": ...,
        "total": ...,
    }

def estimate_training_time(total_tokens, flops_per_token, peak_flops, mfu):
    total_flops = ...
    return ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

Write the memory ledger based on parameter count `P`:

```text
parameters:       P * param_bytes
gradients:        P * grad_bytes
AdamW m:          P * state_bytes
AdamW v:          P * state_bytes
master weights:   Optional; mixed-precision training sometimes stores an extra FP32 copy
activations:      depends on batch_size * context_length * d_model * layers
```

Common estimates for BF16 parameters + FP32 AdamW:

```text
param      2 bytes/param
grad       2 or 4 bytes/param
Adam m     4 bytes/param
Adam v     4 bytes/param
```

Therefore, optimizer state is often larger than the parameters themselves. Rough estimate for training FLOPs:

```text
forward FLOPs = F
backward FLOPs ≈ 2F
optimizer step FLOPs ≈ O(P), usually smaller than forward/backward
one train step ≈ 3F + optimizer
```

Under the MFU assumption:

```python
def training_time_seconds(total_flops, peak_flops, mfu):
    return total_flops / (peak_flops * mfu)
```

When answering these types of questions, discuss parameters, gradients, optimizer state, and activations separately; do not simply say "a 7B model uses 14GB."

</details>

## Exercise 4 · Cosine LR with Warmup

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `learning_rate_schedule`

Interface:

```text
get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
```

Segments:

```text
it < warmup_iters: linear warmup
warmup_iters <= it <= cosine_cycle_iters: cosine decay
it > cosine_cycle_iters: min_lr
```

Tests:

```bash
uv run pytest -k test_get_lr_cosine_schedule
```

Problem Template:

```python
def get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    """
    Piecewise schedule:
        warmup -> cosine decay -> min_lr
    """
    if ...:
        return ...
    if ...:
        return ...
    progress = ...
    coeff = ...
    return ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The cosine schedule consists of three segments:

```python
import math

def get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_lr

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)
```

Checkpoints:

```text
it=0 -> 0 or close to 0
it=warmup_iters -> max_lr
it=cosine_cycle_iters -> min_lr
it>cosine_cycle_iters -> min_lr
```

If `warmup_iters == 0`, handle it separately to avoid division by zero.

</details>

## Exercise 5 · Gradient Clipping

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `gradient_clipping`

Interface:

```text
clip_grad_norm(parameters, max_l2_norm)
```

Key constraints:

- Calculate the global L2 norm for all parameter gradients.
- Do not change if the norm is less than the max.
- Scale all gradients proportionally if the norm exceeds the max.
- Modify `.grad` in-place.

Tests:

```bash
uv run pytest -k test_gradient_clipping
```

Problem Template:

```python
def clip_grad_norm(parameters, max_l2_norm, eps=1e-6):
    """
    Input:
        iterable of parameters with .grad
    Output:
        original global grad norm
    Side effect:
        scale gradients in-place if norm is too large
    """
    grads = ...
    total_norm = ...
    scale = ...
    for p in parameters:
        ...
    return total_norm
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The global grad norm is calculated by treating all parameter gradients as a single long vector to compute the L2 norm:

```python
def clip_grad_norm(parameters, max_l2_norm, eps=1e-6):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)

    total = torch.zeros((), device=params[0].grad.device)
    for p in params:
        total += torch.sum(p.grad.detach() ** 2)
    norm = torch.sqrt(total)

    scale = torch.clamp(max_l2_norm / (norm + eps), max=1.0)
    for p in params:
        p.grad.mul_(scale)
    return norm
```

Key points:

- Use the same scale for all gradients; do not clip each tensor individually.
- Clamp the scale to 1 when the norm is below the threshold, leaving gradients unchanged.
- Modify `.grad` in-place, but return the norm before clipping to facilitate logging.

</details>
