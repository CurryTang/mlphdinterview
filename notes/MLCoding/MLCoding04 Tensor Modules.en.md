# ML Coding · Tensor Modules

Corresponds to CS336 Assignment 1: Sections 3.2-3.4.3.

Usage: For each problem, first review the objectives and acceptance criteria, then complete the TODOs using the "Solution Template." Finally, expand the reference answer to check against boundary conditions, sanity checks, and implementation details.

## Warmup · Tensor Shape Gym

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Objective: Practice batch-like dimensions, einsum, and rearrange.

Exercises:

```text
batched linear projection
split d_model -> num_heads * d_head
merge heads -> d_model
broadcast scalar/vector over leading dims
pixel mixing / arbitrary tensor contraction
```

Record for each problem:

```text
input shape
output shape
einsum / rearrange pattern
toy sanity check
```

Solution Template:

```python
def shape_gym():
    x = ...      # e.g. (batch, seq, d_model)
    W = ...      # e.g. (out_features, in_features)

    y_linear = ...       # (..., in) x (out, in) -> (..., out)
    heads = ...          # (B, T, D) -> (B, H, T, Dh)
    merged = ...         # (B, H, T, Dh) -> (B, T, D)

    assert ...
    return {
        "linear": y_linear.shape,
        "heads": heads.shape,
        "merged": merged.shape,
    }
```

</details>

<details class="solution">
<summary>Reference Answer</summary>

The goal of this section is to become comfortable with the convention that "the last dimension is the feature, and all preceding dimensions are batch-like." Several common patterns:

```python
import torch
from einops import rearrange

x = torch.randn(2, 3, 12)      # batch=2, seq=3, d_model=12
W = torch.randn(16, 12)        # out, in

# batched linear: (..., in), (out, in) -> (..., out)
y = torch.einsum("...i,oi->...o", x, W)
assert y.shape == (2, 3, 16)

# split heads
h = rearrange(x, "b s (nh dh) -> b nh s dh", nh=3)
assert h.shape == (2, 3, 3, 4)

# merge heads
x2 = rearrange(h, "b nh s dh -> b s (nh dh)")
assert x2.shape == x.shape
```

It is recommended to test with these toy shapes before writing modules. Most bugs in Transformers are caused by misaligning head, sequence, or batch dimensions.

A practical checklist:

| Operation | Input | Output |
|---|---|---|
| Linear | `(..., d_in)` | `(..., d_out)` |
| MHA split | `(B, T, D)` | `(B, H, T, Dh)` |
| MHA merge | `(B, H, T, Dh)` | `(B, T, D)` |
| RMSNorm | `(..., D)` | `(..., D)` |
| RoPE | `(..., T, Dh)` | `(..., T, Dh)` |

</details>

## Exercise 1 · Linear Module

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `linear`

Interface:

```text
Linear(in_features, out_features, device=None, dtype=None)
forward(x) -> (..., out_features)
```

Key constraints:

- Weight shape is `(out_features, in_features)`.
- Support arbitrary leading dimensions.
- Do not use `nn.Linear` or `nn.functional.linear`.
- Initialize using truncated normal.

Testing:

```bash
uv run pytest -k test_linear
```

Solution Template:

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(...)
        ...  # truncated normal init

    def forward(self, x):
        """
        Input:
            x: (..., in_features)
        Output:
            (..., out_features)
        """
        return ...  # no nn.Linear / F.linear
```

</details>

<details class="solution">
<summary>Reference Answer</summary>

`Linear` should implement the core behavior of PyTorch's `nn.Linear`, but without directly calling `nn.Linear` or `F.linear`.

```python
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=dtype
        ))
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return torch.einsum("...i,oi->...o", x, self.weight)
```

Why the weight is `(out_features, in_features)`: This ensures each output channel corresponds to a row of weights. The formula is:

```text
y[..., o] = sum_i x[..., i] * weight[o, i]
```

Checkpoint:

```python
m = Linear(4, 7)
x = torch.randn(2, 3, 4)
assert m(x).shape == (2, 3, 7)
```

</details>

## Exercise 2 · Embedding Module

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `embedding`

Interface:

```text
Embedding(num_embeddings, embedding_dim, device=None, dtype=None)
forward(token_ids) -> token_ids.shape + (embedding_dim,)
```

Key constraints:

- Embedding table shape is `(vocab_size, d_model)`.
- Token IDs are integer tensors.
- Do not use `nn.Embedding`.

Testing:

```bash
uv run pytest -k test_embedding
```

Solution Template:

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(...)  # (num_embeddings, embedding_dim)
        ...

    def forward(self, token_ids):
        """
        Input:
            integer tensor with arbitrary shape
        Output:
            token_ids.shape + (embedding_dim,)
        """
        return ...
```

</details>

<details class="solution">
<summary>Reference Answer</summary>

Embedding is a lookup table module: token IDs are integer indices, and the output is the corresponding row vector.

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        ))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return self.weight[token_ids]
```

If `token_ids.shape == (B, T)`, the output is `(B, T, D)`. PyTorch's advanced indexing preserves the shape of the input indices and appends the embedding dimension at the end.

Common mistakes:

- Using float tensors for token IDs.
- Initializing weights as `(embedding_dim, vocab_size)`.
- Forgetting to support arbitrary shapes for token IDs (e.g., `(T,)` and `(B,T)` should both work).

</details>

## Exercise 3 · RMSNorm

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `rmsnorm`

Interface:

```text
RMSNorm(d_model, eps=1e-5)
forward(x) -> same shape
```

Key constraints:

- Normalize the last dimension.
- Gain shape is `(d_model,)`.
- Upcast to float32 before square/mean.
- Cast output back to original dtype.

Sanity checks:

```text
all-zero input remains finite
shape preserved
scaling input mostly cancels after normalization
```

Solution Template:

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(...)

    def forward(self, x):
        """
        Normalize over the last dimension.
        """
        in_dtype = x.dtype
        x_float = ...
        rms = ...
        y = ...
        return ...
```

</details>

<details class="solution">
<summary>Reference Answer</summary>

RMSNorm performs root-mean-square normalization on the last dimension without subtracting the mean:

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        y = x_float / rms
        return (y * self.weight).to(in_dtype)
```

Why upcast: BF16/FP16 can suffer from precision issues when calculating the sum of squares and mean, especially when the hidden dimension is large. Calculating normalization statistics in FP32 is more stable, followed by casting back to the input dtype.

Checkpoint:

```python
x = torch.randn(2, 3, 8, dtype=torch.float16, device="cuda")
norm = RMSNorm(8, device="cuda", dtype=torch.float16)
y = norm(x)
assert y.shape == x.shape
assert y.dtype == x.dtype
```

</details>

## Exercise 4 · SwiGLU Feed-Forward

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `positionwise_feedforward`

Structure:

```text
SiLU(W1 x) * W3 x
then W2 down projection
```

Key constraints:

- `d_ff` is approximately `8/3 * d_model`.
- `d_ff` adjusted to a multiple of 64.
- Support arbitrary leading dims.

Testing:

```bash
uv run pytest -k test_swiglu
```

Solution Template:

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        d_ff = ...  # around 8/3 * d_model, rounded to multiple of 64
        self.w1 = ...  # gate projection
        self.w3 = ...  # up projection
        self.w2 = ...  # down projection

    def forward(self, x):
        gate = ...
        up = ...
        return ...
```

</details>

<details class="solution">
<summary>Reference Answer</summary>

SwiGLU FFN performs gated feed-forward. Common LLaMA style:

```text
FFN(x) = W_down( SiLU(W_gate x) * W_up x )
```

Implementation:

```python
import math

def round_up_to_multiple(x, multiple):
    return multiple * math.ceil(x / multiple)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = round_up_to_multiple(int(8 * d_model / 3), 64)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # up
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # down

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
```

Why `8/3 * d_model`: SwiGLU has three sets of matrices, whereas a standard FFN typically has two sets with a hidden dimension of approximately `4 * d_model`. To keep the parameter count similar, the hidden dimension of SwiGLU is scaled down to approximately `8/3 * d_model`.

</details>

## Exercise 5 · RoPE

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `rope`

Interface:

```text
RotaryPositionalEmbedding(theta, d_k, max_seq_len)
forward(x, token_positions) -> same shape
```

Key constraints:

- `x` shape is `(..., seq_len, d_k)`.
- `token_positions` shape is `(..., seq_len)`.
- cos/sin can be precomputed as non-persistent buffers.
- Rotation preserves the L2 norm of each pair of dimensions.

Testing:

```bash
uv run pytest -k test_rope
```

Solution Template:

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        ...  # precompute cos/sin buffers for positions and dim pairs

    def forward(self, x, token_positions):
        """
        Input:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len)
        Output:
            same shape as x
        """
        x_even = ...
        x_odd = ...
        cos = ...
        sin = ...
        return ...
```

</details>

<details class="solution">
<summary>Reference Answer</summary>

RoPE treats every two hidden dimensions as a 2D plane and rotates them based on position. Core formula:

```text
[x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
```

Implementation skeleton:

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        assert d_k % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x, token_positions):
        # x: (..., seq, d_k)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)
```

Checkpoint:

```python
y = rope(x, positions)
assert y.shape == x.shape
assert torch.allclose(
    (x[..., 0::2] ** 2 + x[..., 1::2] ** 2),
    (y[..., 0::2] ** 2 + y[..., 1::2] ** 2),
    atol=1e-4,
)
```

RoPE is applied only to Q/K in attention, not to V.

</details>

## Debug Checklist

<details class="exercise">
<summary><span class="q-label">Debug</span> <span class="q-text">Expand checklist</span></summary>

- All modules must support `device` / `dtype`.
- Explicitly label sequence dimensions and head dimensions, and use assertions to check shapes.
- Printing intermediate tensors with small shapes is faster than running the full model.

</details>
