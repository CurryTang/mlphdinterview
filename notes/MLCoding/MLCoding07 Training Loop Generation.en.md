# ML Coding · Training Loop & Generation

Corresponds to CS336 Assignment 1: Sections 5-6.

Usage: For each exercise, first review the objectives and acceptance criteria, then complete the TODOs using the "Solution Template." Finally, expand the reference solution to check against edge cases, sanity checks, and implementation details.

## Exercise 1 · Next-Token Batch Sampler

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `data_loading`

Interface:

```text
get_batch(dataset, batch_size, context_length, device) -> (x, y)
```

Output:

```text
x: (batch_size, context_length)
y: (batch_size, context_length)
```

Key constraints:

- `y` is the target, which is `x` shifted right by one position.
- The sample start index must ensure the target is valid.
- Output tensors must be moved to the specified device.
- For large datasets, use `np.memmap` or `np.load(..., mmap_mode='r')`.

Testing:

```bash
uv run pytest -k test_get_batch
```

Solution Template:

```python
def get_batch(dataset, batch_size: int, context_length: int, device):
    """
    Input:
        dataset: 1D token id array
    Output:
        x, y: both (batch_size, context_length)
        y is x shifted right by one token
    """
    starts = ...
    xs, ys = [], []
    for s in starts:
        chunk = ...  # length context_length + 1
        xs.append(...)
        ys.append(...)
    return ..., ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The next-token batch sampler takes a 1D token array as input and outputs `x` and `y` (shifted right by one).

```python
import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    # dataset can be np.ndarray / np.memmap / torch tensor
    n = len(dataset)
    starts = torch.randint(0, n - context_length, (batch_size,))

    xs, ys = [], []
    for s in starts.tolist():
        chunk = dataset[s : s + context_length + 1]
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk.astype(np.int64))
        else:
            chunk = torch.as_tensor(chunk, dtype=torch.long)
        xs.append(chunk[:-1])
        ys.append(chunk[1:])

    x = torch.stack(xs).to(device=device, dtype=torch.long)
    y = torch.stack(ys).to(device=device, dtype=torch.long)
    return x, y
```

Why the maximum start index is `n - context_length - 1`: Each sample requires `context_length + 1` tokens, where the last token serves as the target. Since the `high` parameter in `torch.randint(0, n - context_length)` is exclusive, the maximum start index is `n - context_length - 1`.

Checkpoint:

```python
x, y = get_batch(np.arange(1000), 4, 8, "cpu")
assert x.shape == y.shape == (4, 8)
assert torch.all(y[:, :-1] == x[:, 1:])
```

</details>

## Exercise 2 · Checkpoint Save / Load

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `checkpointing`

Interface:

```text
save_checkpoint(model, optimizer, iteration, out)
load_checkpoint(src, model, optimizer) -> iteration
```

Must save:

```text
model.state_dict()
optimizer.state_dict()
iteration
```

Sanity checks:

```text
save -> mutate -> load -> parameters restored
optimizer moments restored
iteration restored
path and file-like object both work
```

Testing:

```bash
uv run pytest -k test_checkpointing
```

Solution Template:

```python
def save_checkpoint(model, optimizer, iteration: int, out):
    """
    Save model state, optimizer state, and iteration.
    """
    payload = {
        "model": ...,
        "optimizer": ...,
        "iteration": ...,
    }
    ...

def load_checkpoint(src, model, optimizer) -> int:
    """
    Restore states and return saved iteration.
    """
    payload = ...
    ...
    return ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

A checkpoint must save at least three things: model parameters, optimizer state, and the current iteration.

```python
def save_checkpoint(model, optimizer, iteration, out):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(payload, out)

def load_checkpoint(src, model, optimizer):
    payload = torch.load(src, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload["iteration"]
```

To support file-like objects, `torch.save` / `torch.load` already handle this natively. Key sanity test:

```python
save_checkpoint(model, opt, 123, path)
old = {k: v.clone() for k, v in model.state_dict().items()}

for p in model.parameters():
    p.data.normal_()

it = load_checkpoint(path, model, opt)
assert it == 123
for k, v in model.state_dict().items():
    assert torch.equal(v, old[k])
```

Optimizer state is crucial when resuming training; restoring only the model will cause AdamW moments to be lost, which may lead to significant jumps in the loss curve.

</details>

## Exercise 3 · Full Training Script

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `training_together`

Configuration items:

```text
model hyperparameters
optimizer hyperparameters
batch_size
context_length
train / val data path
checkpoint path
eval interval
log interval
device
dtype / matmul precision
```

Training loop:

```text
sample batch
forward
compute cross entropy
backward
clip gradients
optimizer step
zero gradients
update LR
periodic eval
periodic checkpoint
periodic logging
```

Debug milestones:

| milestone | expected signal |
|---|---|
| overfit one minibatch | train loss approaches near-zero |
| tiny model on tiny data | loss decreases smoothly |
| validation eval | no gradient graph retained |
| checkpoint resume | curve continues without reset |

Solution Template:

```python
def train(config):
    model = ...
    optimizer = ...
    train_tokens, val_tokens = ...
    start_iter = ...

    for it in range(start_iter, config.max_iters):
        lr = ...
        ...  # set optimizer LR

        x, y = ...
        logits = ...
        loss = ...

        optimizer.zero_grad(set_to_none=True)
        ...
        ...  # gradient clipping
        optimizer.step()

        if it % config.eval_interval == 0:
            val_loss = ...
        if it % config.ckpt_interval == 0:
            ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The goal of a full training script is not to write a massive file, but to integrate data, model, optimizer, scheduler, evaluation, and checkpointing. Minimal loop:

```python
for it in range(start_iter, max_iters):
    lr = get_lr(it, max_lr, min_lr, warmup_iters, cosine_iters)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_tokens, batch_size, context_length, device)
    logits = model(x)
    loss = cross_entropy(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm(model.parameters(), max_grad_norm)
    optimizer.step()

    if it % log_interval == 0:
        print({"iter": it, "loss": float(loss), "lr": lr})

    if it % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_iters):
                vx, vy = get_batch(val_tokens, batch_size, context_length, device)
                val_losses.append(cross_entropy(model(vx), vy).item())
        model.train()
        print({"iter": it, "val_loss": sum(val_losses) / len(val_losses)})

    if it % ckpt_interval == 0:
        save_checkpoint(model, optimizer, it, ckpt_path)
```

Debugging sequence:

1. Overfit a fixed minibatch first to confirm the model and loss can decrease.
2. Connect the real dataloader to confirm batch shifting is correct.
3. Enable evaluation, ensuring `torch.no_grad()` is used.
4. Finally, test checkpoint resumption to ensure the iteration and loss curve do not reset.

Common bugs:

- Forgetting `optimizer.zero_grad()` after `backward()` but before `step()`.
- Forgetting `model.eval()` during evaluation or failing to switch back to `model.train()`.
- Logging a loss tensor without calling `.item()`, which keeps the computation graph in memory.

</details>

## Exercise 4 · Autoregressive Decoder

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `decoding`

Functionality:

```text
prompt -> token ids
loop:
  model forward
  take last-position logits
  apply temperature
  apply top-p filter
  sample token
  append token
  stop on EOS or max_new_tokens
decode ids -> text
```

Parameters:

```text
max_new_tokens
temperature
top_p
eos_token_id
context_length
```

Key constraints:

- Use only the logits from the last position.
- Retain the most recent window when exceeding context length.
- Re-normalize after top-p filtering.

Output:

```text
generated text
token count
stop reason
sampling parameters
```

Solution Template:

```python
@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature=1.0, top_p=1.0, eos_token_id=None):
    """
    Input:
        prompt text and sampling parameters
    Output:
        generated text
    """
    ids = ...
    tokens = ...
    for _ in range(max_new_tokens):
        idx_cond = ...
        logits = ...
        if temperature == 0:
            next_id = ...
        else:
            probs = ...
            probs = ...
            next_id = ...
        tokens = ...
        if ...:
            break
    return ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

Generation uses only the logits from the last position in each iteration:

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature=1.0, top_p=1.0, eos_token_id=None):
    model.eval()
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)

    for _ in range(max_new_tokens):
        idx_cond = tokens[:, -model.context_length:]
        logits = model(idx_cond)[:, -1, :]

        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            probs = top_p_filter(probs, top_p)
            next_id = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_id], dim=1)
        if eos_token_id is not None and int(next_id.item()) == eos_token_id:
            break

    return tokenizer.decode(tokens[0].tolist())
```

Top-p filter:

```python
def top_p_filter(probs, top_p):
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= top_p
    keep[..., 0] = True
    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
    return filtered / filtered.sum(dim=-1, keepdim=True)
```

Note:

- When the context is too long, use the most recent `context_length` tokens.
- `temperature=0` typically implies greedy decoding; avoid division by zero.
- Re-normalization is mandatory after top-p filtering.

</details>
