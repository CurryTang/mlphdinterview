# ML Coding · Training Loop & Generation

对应 CS336 Assignment 1：Section 5-6。

使用方式：每题先看目标和验收标准，再按“解题模板”把 TODO 补完整；最后展开参考答案，对照边界条件、sanity checks 和实现细节。

## Exercise 1 · Next-Token Batch Sampler

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`data_loading`

接口：

```text
get_batch(dataset, batch_size, context_length, device) -> (x, y)
```

输出：

```text
x: (batch_size, context_length)
y: (batch_size, context_length)
```

关键约束：

- `y` 是 `x` 向右偏移一位的 target。
- sample start index 要保证 target 合法。
- 输出 tensor 放到指定 device。
- 大数据用 `np.memmap` 或 `np.load(..., mmap_mode='r')`。

测试：

```bash
uv run pytest -k test_get_batch
```

解题模板：

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
<summary>参考答案</summary>

next-token batch sampler 的输入是一维 token array，输出 `x` 和右移一位的 `y`。

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

为什么 start 最大是 `n - context_length - 1` 的等价形式：每个样本需要 `context_length + 1` 个 token，最后一个 token 是 target。上面 `torch.randint(0, n - context_length)` 的 high 是 exclusive，所以最大 start 是 `n - context_length - 1`。

检查点：

```python
x, y = get_batch(np.arange(1000), 4, 8, "cpu")
assert x.shape == y.shape == (4, 8)
assert torch.all(y[:, :-1] == x[:, 1:])
```

</details>

## Exercise 2 · Checkpoint Save / Load

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`checkpointing`

接口：

```text
save_checkpoint(model, optimizer, iteration, out)
load_checkpoint(src, model, optimizer) -> iteration
```

必须保存：

```text
model.state_dict()
optimizer.state_dict()
iteration
```

Sanity checks：

```text
save -> mutate -> load -> parameters restored
optimizer moments restored
iteration restored
path and file-like object both work
```

测试：

```bash
uv run pytest -k test_checkpointing
```

解题模板：

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
<summary>参考答案</summary>

checkpoint 至少要保存三件事：模型参数、optimizer state、当前 iteration。

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

如果要支持 file-like object，`torch.save` / `torch.load` 本身已经支持。关键 sanity test：

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

训练 resume 时 optimizer state 很重要；只恢复 model 会让 AdamW moments 丢失，loss curve 可能出现明显跳变。

</details>

## Exercise 3 · Full Training Script

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`training_together`

配置项：

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

训练 loop：

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

Debug milestones：

| milestone | expected signal |
|---|---|
| overfit one minibatch | train loss approaches near-zero |
| tiny model on tiny data | loss decreases smoothly |
| validation eval | no gradient graph retained |
| checkpoint resume | curve continues without reset |

解题模板：

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
<summary>参考答案</summary>

完整训练脚本的目标不是写一个很长的文件，而是把数据、模型、optimizer、scheduler、eval、checkpoint 接起来。最小 loop：

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

调试顺序：

1. 先 overfit 一个固定 minibatch，确认模型和 loss 能下降。
2. 再接真实 dataloader，确认 batch shift 正确。
3. 再打开 eval，确认用了 `torch.no_grad()`。
4. 最后测试 checkpoint resume，确认 iteration 和 loss curve 不重置。

常见 bug：

- `optimizer.zero_grad()` 放在 `backward()` 后但 step 前遗漏。
- eval 时忘记 `model.eval()` 或忘记切回 `model.train()`。
- logging loss tensor 没 `.item()`，长期持有 computation graph。

</details>

## Exercise 4 · Autoregressive Decoder

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`decoding`

功能：

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

参数：

```text
max_new_tokens
temperature
top_p
eos_token_id
context_length
```

关键约束：

- 只用最后一个 position 的 logits。
- 超出 context length 时保留最近窗口。
- top-p 后重新 normalize。

输出：

```text
generated text
token count
stop reason
sampling parameters
```

解题模板：

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
<summary>参考答案</summary>

generation 每轮只用最后一个位置的 logits：

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

top-p filter：

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

注意：

- context 超长时用最近 `context_length` 个 token。
- `temperature=0` 通常走 greedy，不要除以 0。
- top-p 后必须重新 normalize。

</details>
