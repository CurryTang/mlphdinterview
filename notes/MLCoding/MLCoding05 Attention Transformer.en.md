# ML Coding · Attention & Transformer LM

Corresponds to CS336 Assignment 1: Sections 3.4.4-3.5.

Usage: For each exercise, first review the objectives and acceptance criteria, then complete the TODOs using the "Solution Template." Finally, expand the reference solution to check against boundary conditions, sanity checks, and implementation details.

## Exercise 1 · Stable Softmax

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `softmax`

Interface:

```text
softmax(x, dim) -> same shape
```

Key Constraints:

- Subtract the maximum value along `dim`.
- Output must sum to 1 along `dim`.
- Support arbitrary shapes.

Tests:

```bash
uv run pytest -k test_softmax_matches_pytorch
```

Solution Template:

```python
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Input:
        x: arbitrary shape tensor
        dim: dimension to normalize
    Output:
        same shape, probabilities summing to 1 along dim
    """
    x_max = ...
    shifted = ...
    exp = ...
    return ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The core of stable softmax is to subtract the maximum value along the target dimension first to prevent `exp` overflow.

```python
import torch

def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    shifted = x - x_max
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)
```

Checkpoints:

```python
x = torch.tensor([1000.0, 1001.0, 1002.0])
y = softmax(x, dim=0)
assert torch.isfinite(y).all()
assert torch.allclose(y.sum(), torch.tensor(1.0))
```

Do not perform `exp(x)` before normalizing; large logits will overflow immediately.

</details>

## Exercise 2 · Scaled Dot-Product Attention

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `scaled_dot_product_attention`

Interface:

```text
Q: (..., queries, d_k)
K: (..., keys, d_k)
V: (..., keys, d_v)
mask: (..., queries, keys)
output: (..., queries, d_v)
```

Key Constraints:

- Divide logits by `sqrt(d_k)`.
- Softmax probability for positions where mask is False must be 0.
- Probabilities for positions where mask is True must sum to 1.
- Support 3D / 4D / arbitrary leading dims.

Tests:

```bash
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
```

Solution Template:

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Input:
        Q: (..., queries, d_k)
        K: (..., keys, d_k)
        V: (..., keys, d_v)
        mask: (..., queries, keys), True means visible
    Output:
        (..., queries, d_v)
    """
    scores = ...
    if mask is not None:
        scores = ...
    probs = ...
    return ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

Scaled dot-product attention performs three steps: calculating query-key similarity, masking out illegal positions, and weighting values using softmax probabilities.

```python
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)
    return torch.einsum("...qk,...kd->...qd", attn, V)
```

Mask semantics must be consistent: here `True` means visible, `False` means forbidden. A causal mask can be written as:

```python
def causal_mask(q_len, k_len, device=None):
    q = torch.arange(q_len, device=device)[:, None]
    k = torch.arange(k_len, device=device)[None, :]
    return k <= q
```

If an entire row is masked, softmax will result in NaN; standard causal self-attention will not encounter this, as each position can at least attend to itself.

</details>

## Exercise 3 · Causal MHA

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `multihead_self_attention`

Structure:

```text
x -> Q/K/V projections
reshape into heads
optional RoPE on Q/K
causal SDPA
merge heads
output projection
```

Key Constraints:

- Causal mask only allows `j <= i`.
- Treat head dimension as a batch-like dimension.
- Q/K/V projection can be three batched matmuls.
- Output shape must match input shape.

Tests:

```bash
uv run pytest -k test_multihead_self_attention
uv run pytest -k test_multihead_self_attention_with_rope
```

Solution Template:

```python
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = ...
        self.d_head = ...
        self.q_proj = ...
        self.k_proj = ...
        self.v_proj = ...
        self.o_proj = ...
        self.rope = rope

    def forward(self, x, token_positions=None):
        """
        Input:
            x: (batch, seq, d_model)
        Output:
            same shape
        """
        q = ...  # project and split heads
        k = ...
        v = ...
        if self.rope is not None:
            q = ...
            k = ...
        mask = ...
        out = ...
        return ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The key to implementing Causal MHA is shape discipline. Given input `(B,T,D)`, split it into `(B,H,T,Dh)` and treat the head dimension as a batch-like dimension.

```python
from einops import rearrange

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, x, token_positions=None):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope is not None:
            q = self.rope(q, token_positions[:, None, :])
            k = self.rope(k, token_positions[:, None, :])

        mask = causal_mask(T, T, device=x.device)[None, None, :, :]
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

RoPE is applied only to Q/K because attention scores require relative positional information; V is the content being read and does not participate in matching.

</details>

## Exercise 4 · Transformer Block

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `transformer_block`

Pre-norm block:

```text
y = x + MHA(RMSNorm(x))
out = y + FFN(RMSNorm(y))
```

Key Constraints:

- Residual stream shape is always `(batch, seq, d_model)`.
- Two RMSNorm parameters are independent.
- MHA uses causal mask and RoPE.

Tests:

```bash
uv run pytest -k test_transformer_block
```

Solution Template:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = ...
        self.attn = ...
        self.ln2 = ...
        self.ffn = ...

    def forward(self, x, token_positions=None):
        """
        Pre-norm residual block.
        """
        x = x + ...
        x = x + ...
        return x
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

CS336 recommends implementing a pre-norm Transformer block because it is more stable during training:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
```

Key invariants:

```text
Input shape:  (B, T, D)
attn output:  (B, T, D)
ffn output:   (B, T, D)
Residual stream is always (B, T, D)
```

If the residual shape changes, subsequent blocks will inevitably fail.

</details>

## Exercise 5 · Transformer LM

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `transformer_lm`

Structure:

```text
token ids
-> token embedding
-> N transformer blocks
-> final RMSNorm
-> LM head
-> logits
```

Output:

```text
logits: (batch_size, sequence_length, vocab_size)
```

Tests:

```bash
uv run pytest -k test_transformer_lm
```

Solution Template:

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, ...):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = ...
        self.rope = ...
        self.layers = ...
        self.ln_final = ...
        self.lm_head = ...

    def forward(self, token_ids):
        """
        Input:
            token_ids: (batch, seq)
        Output:
            logits: (batch, seq, vocab_size)
        """
        positions = ...
        x = ...
        for layer in self.layers:
            x = ...
        return ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The Transformer LM maps token IDs to logits; it does not perform softmax inside the model:

```python
class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size, context_length, d_model, num_layers,
        num_heads, d_ff, rope_theta=10000, device=None, dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length, device=device
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        B, T = token_ids.shape
        assert T <= self.context_length
        positions = torch.arange(T, device=token_ids.device).expand(B, T)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions=positions)
        x = self.ln_final(x)
        return self.lm_head(x)
```

During training, cross entropy consumes logits directly; during generation, only the logits of the last position are used:

```python
logits = model(tokens)
next_logits = logits[:, -1, :]
```

</details>

## Exercise 6 · Resource Accounting

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `transformer_accounting`

Write a parameter count / FLOPs accounting script.

Input:

```text
vocab_size
context_length
num_layers
d_model
num_heads
d_ff
```

Output:

```text
parameter count by component
memory to load parameters
forward FLOPs by component
attention FLOPs vs MLP FLOPs
context length sensitivity
```

Comparison:

```text
GPT-2 small
GPT-2 medium
GPT-2 large
GPT-2 XL
GPT-2 XL with context_length = 16,384
```

Solution Template:

```python
def transformer_accounting(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    """
    Output:
        parameter count and rough forward FLOPs by component
    """
    params = {
        "embedding": ...,
        "attention_per_layer": ...,
        "ffn_per_layer": ...,
        "norms_per_layer": ...,
        "lm_head": ...,
    }
    flops = {
        "qkv_o": ...,
        "attention_qk_pv": ...,
        "ffn": ...,
    }
    return {"params": params, "flops": flops}
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The goal of resource accounting is to train you to estimate bottlenecks, rather than precisely replicating a profiler. A practical parameter ledger:

```python
def transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff):
    token_emb = vocab_size * d_model
    final_norm = d_model

    # per layer
    attn = 4 * d_model * d_model          # Q,K,V,O
    ffn = 3 * d_model * d_ff              # SwiGLU gate/up/down
    norms = 2 * d_model
    per_layer = attn + ffn + norms

    lm_head = vocab_size * d_model
    return {
        "token_embedding": token_emb,
        "layers": num_layers * per_layer,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": token_emb + num_layers * per_layer + final_norm + lm_head,
    }
```

Rough FLOPs:

```text
Linear forward FLOPs ≈ 2 * tokens * in_dim * out_dim
Attention QK FLOPs ≈ 2 * B * H * T * T * Dh
Attention PV FLOPs ≈ 2 * B * H * T * T * Dh
MLP FLOPs ≈ 2 * B*T*D*Dff * 3
```

When context length increases from 1024 to 16384, the `T^2` term in attention scales significantly; MLP still grows primarily linearly with `T`. This comparison explains why long-context training and inference focus heavily on attention kernels and KV cache.

</details>

## Debug Checklist

<details class="exercise">
<summary><span class="q-label">Debug</span> <span class="q-text">Expand checklist</span></summary>

- Causal mask True/False semantics must be consistent with the SDPA implementation.
- RoPE is applied only to Q/K, not to V.
- LM head output is logits, not softmax probabilities.

</details>
