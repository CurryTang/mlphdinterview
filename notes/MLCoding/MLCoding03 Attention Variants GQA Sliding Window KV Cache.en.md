# ML Coding 03 · Attention Zoo: From MHA to GQA, Sliding Window, Linear Attention, KV Cache, and Flash Attention

MLCoding01 already builds stable softmax, scaled dot-product attention, and causal multi-head self-attention with RoPE: the minimum set needed to run a GPT-style model. But interviews ask about far more attention variants than that: encoders use bidirectional MHA, translation models use cross-attention, LLaMA 2/3 use GQA to shrink the KV cache, Mistral uses sliding-window attention to cut complexity, and any inference engine needs KV caching and flash attention. This note only covers the mechanics and pitfalls of these variants; it does not repeat the softmax, mask semantics, or RoPE derivation already covered in MLCoding01.

The topic selection here follows the open-source practice set [TorchCode](https://github.com/duoan/TorchCode) (an auto-graded PyTorch interview practice repo); the explanations and code below are independently written.

## Prerequisites reused from MLCoding01

Every exercise below assumes you already have these pieces from MLCoding01. Only the signatures are shown here as a reminder; see MLCoding01 for the implementations:

```python
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor: ...
def scaled_dot_product_attention(Q, K, V, mask=None) -> torch.Tensor: ...
class Linear(nn.Module): ...  # y = x @ weight.T, no bias
```

## Module 9: Attention Variants

### Exercise 1 · MultiHeadAttention (bidirectional, non-causal)

Compared with MLCoding01's `CausalMultiHeadSelfAttention`, the only real difference is the mask: encoder architectures like BERT and ViT let every position see every other position, so the mask is either `None` or used purely to block padding, never a lower-triangular matrix. Everything else (splitting heads, concatenating heads, the projections) is identical.

A common interview trap is thinking of "multi-head" as "run single-head attention H times and concatenate": conceptually correct, but the real implementation reshapes `(B, T, D)` into `(B, H, T, Dh)` and does one batched matmul, not a Python for-loop. The order of the reshape matters too: the `d_model` dimension must first split into `(H, Dh)` and then transpose into `(B, H, T, Dh)`. Transpose first and reshape second, and each head ends up mixing channels from the wrong group. Training still runs, but what it learns is wrong.

#### Quick Coding: `MultiHeadAttention`

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        ...

    def forward(self, x, key_padding_mask=None):
        ...
```

<details>
<summary>Reference solution</summary>

```python
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.num_heads)

        mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, T), True means this position is a real token
            mask = key_padding_mask[:, None, None, :]  # broadcast to (B, 1, 1, T)

        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

</details>

### Exercise 2 · MultiHeadCrossAttention

Cross-attention separates where Q comes from and where K/V come from: Q comes from the current sequence (e.g. decoder hidden states), K/V come from a different sequence (e.g. encoder outputs, or image features in a vision-language model). The two sequences can have different lengths: `T_q != T_kv` is the normal case, not an edge case.

The most common trap is assuming the attention matrix is square. Its shape is `(..., T_q, T_kv)`, and any causal mask or padding mask must be shaped to match this non-square shape. Copying the `(T, T)` mask construction from self-attention leads straight to a shape mismatch.

#### Quick Coding: `MultiHeadCrossAttention`

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        ...

    def forward(self, x_q, x_kv, key_padding_mask=None):
        ...
```

<details>
<summary>Reference solution</summary>

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x_q, x_kv, key_padding_mask=None):
        # x_q:  (B, T_q,  D)  from the decoder
        # x_kv: (B, T_kv, D)  from the encoder, T_kv need not equal T_q
        q = rearrange(self.q_proj(x_q), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x_kv), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.v_proj(x_kv), "b t (h d) -> b h t d", h=self.num_heads)

        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, T_kv)

        out = scaled_dot_product_attention(q, k, v, mask)  # (B, H, T_q, Dh)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

`scores` has shape `(B, H, T_q, T_kv)`, and `attn @ V` returns to `(B, H, T_q, Dh)`. The output sequence length always follows Q, never K/V. That's the first thing worth checking when debugging a shape mismatch here.

</details>

### Exercise 3 · Grouped Query Attention (GQA)

Standard MHA gives Q, K, and V each `num_heads` heads, so at inference time you need one cached K/V per head, and KV cache memory grows linearly with the head count. GQA (used in LLaMA 2/3) cuts the number of K/V heads down to `n_kv_heads < n_q_heads`; every `group_size = n_q_heads // n_kv_heads` Q heads share one K/V head. KV cache memory shrinks by a factor of `n_q_heads / n_kv_heads` while Q's representational capacity is unchanged. The extreme case `n_kv_heads = 1` is Multi-Query Attention (MQA).

The real trap is in how the heads get aligned. K/V only produce `n_kv_heads` heads and need to be broadcast up to `n_q_heads` before they can be batched-matmul'd against Q. This must use `repeat_interleave` (equivalent to NumPy's `repeat`: `[0, 1]` becomes `[0, 0, 0, 0, 1, 1, 1, 1]`), not `tensor.repeat` (equivalent to NumPy's `tile`: `[0, 1]` becomes `[0, 1, 0, 1, 0, 1, 0, 1]`). Both produce the same output shape, the code runs, loss still goes down, but `tile` incorrectly assigns Q heads 0, 2, 4, 6 to kv head 0 and Q heads 1, 3, 5, 7 to kv head 1. The grouping is scrambled end to end: a classic bug that runs fine but learns the wrong thing.

#### Quick Coding: `GroupedQueryAttention`

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, device=None, dtype=None):
        ...

    def forward(self, x, mask=None):
        ...
```

<details>
<summary>Reference solution</summary>

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.num_kv_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.num_kv_heads)

        # the key step: repeat_interleave, so q head i maps to kv head i // group_size
        k = torch.repeat_interleave(k, self.group_size, dim=1)  # (B, H, T, Dh)
        v = torch.repeat_interleave(v, self.group_size, dim=1)

        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

KV cache memory comparison (per layer, float16, sequence length `T`):

| Scheme | K/V heads | KV cache size per layer |
| --- | --- | --- |
| MHA | `num_heads` | `2 * num_heads * T * d_head * 2 bytes` |
| GQA | `num_kv_heads` | `2 * num_kv_heads * T * d_head * 2 bytes` |
| MQA | `1` | `2 * T * d_head * 2 bytes` |

That `repeat_interleave` groups by `i // group_size` rather than interleaving like `tile` has been verified with NumPy (`np.repeat` matches `torch.repeat_interleave`'s semantics).

</details>

### Exercise 4 · Sliding Window Attention

The local attention used in Mistral: each query only looks at the most recent `window` keys, including itself, never anything further back. The mask goes from a lower triangle to a diagonal band: position `i` may attend to `j` only if `i - window < j <= i`. Per-layer complexity drops from `O(T^2)` to `O(T * window)`.

This is not an approximation that "can't see long-range information"; it delegates long-range dependencies to depth. This layer can't see beyond `window` tokens back, but after stacking `L` layers, a position at layer `L` can indirectly depend on inputs up to `L * window` away, the same idea as stacking small convolution kernels to grow a CNN's receptive field.

#### Quick Coding: `sliding_window_attention`

```python
def sliding_window_attention(Q, K, V, window):
    ...
```

<details>
<summary>Reference solution</summary>

```python
def sliding_window_mask(T, window, device=None):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (j > i - window)  # lower triangle AND inside the window

def sliding_window_attention(Q, K, V, window):
    T = Q.shape[-2]
    mask = sliding_window_mask(T, window, device=Q.device)
    mask = mask[None, None, :, :]  # broadcast to (B, H, T, T)
    return scaled_dot_product_attention(Q, K, V, mask)
```

An implementation that actually saves compute doesn't build the full `(T, T)` mask and run dense attention (that only saves bandwidth, not FLOPs); it tiles by `window` and only matmuls the K/V inside each window. This definitional version is written to match `scaled_dot_product_attention`'s interface; the performance-oriented tiling idea is the same one used in flash attention below.

</details>

### Exercise 5 · Linear Attention

Softmax attention's complexity bottleneck is that it must materialize the full `(T, T)` score matrix before it can normalize. Linear attention replaces `softmax(QK^T)` with a positive feature map `φ` (e.g. `elu(x) + 1`), giving a kernel approximation whose associativity can be reordered:

```text
softmax(QK^T) V            grouped as (Q K^T) V, O(T^2 d)
φ(Q) (φ(K)^T V)             grouped as Q (K^T V), O(T d^2)
```

When `d << T` (long sequences, moderate head_dim), the second form is significantly faster and never needs an explicit `(T, T)` matrix. The normalizer works the same way: `sum_j exp(...)` becomes `φ(Q) · sum_j φ(K_j)`.

The point worth stating clearly: this is not the same kind of speedup as flash attention, which is mathematically identical to the dense computation and just reorders arithmetic. Linear attention swaps in a different similarity kernel: `φ(Q)φ(K)^T` is not equal term-by-term to `softmax(QK^T)`. It's an approximation to softmax attention via a kernel-method reparameterization, and it trades away some expressiveness and accuracy, which is why it hasn't fully replaced softmax attention.

#### Quick Coding: `linear_attention`

```python
def linear_attention(Q, K, V, causal=False):
    ...
```

<details>
<summary>Reference solution</summary>

```python
def feature_map(x):
    return F.elu(x) + 1  # stays non-negative, doesn't zero out negative inputs like relu would

def linear_attention(Q, K, V, causal=False):
    Qp, Kp = feature_map(Q), feature_map(K)  # (..., T, d)

    if not causal:
        kv = torch.einsum("...kd,...ke->...de", Kp, V)      # (..., d, dv)
        k_sum = Kp.sum(dim=-2)                                # (..., d)
        num = torch.einsum("...qd,...de->...qe", Qp, kv)     # (..., T, dv)
        den = torch.einsum("...qd,...d->...q", Qp, k_sum).unsqueeze(-1)
        return num / den

    # causal variant: replace the full matmul with cumulative sums,
    # updating an O(d * dv) running state one token at a time
    outer = torch.einsum("...tk,...tv->...tkv", Kp, V)  # (..., T, d, dv)
    S_cum = torch.cumsum(outer, dim=-3)                   # running sum_{j<=t} phi(K_j) V_j^T
    z_cum = torch.cumsum(Kp, dim=-2)                      # running sum_{j<=t} phi(K_j)
    num = torch.einsum("...td,...tdv->...tv", Qp, S_cum)
    den = torch.einsum("...td,...td->...t", Qp, z_cum).unsqueeze(-1)
    return num / den
```

The associativity identity `φ(Q)(φ(K)^T V) == (φ(Q)φ(K)^T) V`, and the equivalence between the causal variant's "loop, accumulate step by step" and "batch cumsum" forms, have both been checked numerically with NumPy, along with confirming that linear attention's output genuinely differs from softmax attention's output. They're two different similarity kernels, not two implementations of the same function.

</details>

### Exercise 6 · KV Cache Attention

During autoregressive generation, step `t` only needs the Q for the new token, but it must attend over the K/V of every position `0..t`. Recomputing all historical K/V at every step costs `O(T^2)` in total; caching the K/V already computed and only adding one new position per step brings the total down to `O(T)`. This is also why inference serving splits into a prefill phase (process the whole prompt at once, filling the cache) and a decode phase (process one new token per step, growing the cache by one).

Two details are easy to miss. First, if the model uses RoPE, what gets cached must be the K "already rotated at its own absolute position," not the raw pre-rotation K. RoPE's relative-position property comes from rotating each K individually at its absolute position; cache the unrotated K and rotate everything uniformly later, and the position information is wrong across the board. Second, the decode phase must compute the new token's position starting from `cache_len`, not from 0, or the new token collides with a position that's already in the cache.

#### Quick Coding: `KVCacheAttention`

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        ...

    def forward(self, x, start_pos, use_cache=True):
        ...

    def reset_cache(self):
        ...
```

<details>
<summary>Reference solution</summary>

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope
        self.cache_k = None  # (B, H, cache_len, Dh)
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, start_pos, use_cache=True):
        B, T, D = x.shape  # prefill: T = prompt_len; decode: T = 1
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope is not None:
            positions = torch.arange(start_pos, start_pos + T, device=x.device)
            positions = positions[None, None, :].expand(B, 1, T)
            q = self.rope(q, positions)
            k = self.rope(k, positions)  # must rotate before caching

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v

        # the new query only needs to see keys 0..(start_pos+T-1), which is
        # already causal by construction; decode with T=1 needs no explicit mask.
        Tk = k.shape[2]
        if T > 1:
            q_idx = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
            k_idx = torch.arange(Tk, device=x.device)[None, :]
            mask = (k_idx <= q_idx)[None, None, :, :]
        else:
            mask = None

        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

A typical call sequence: `forward(prompt_embeds, start_pos=0)` for prefill, then `forward(next_token_embed, start_pos=cache_len)` per decode step, with `cache_len` incrementing by 1 each time.

</details>

### Exercise 7 · Flash Attention (tiling + online softmax)

Flash attention isn't about whether attention is computed correctly. It is about whether computing it requires holding the entire `(T, T)` score matrix in memory. The standard implementation computes the full `scores` matrix and then softmaxes it all at once; flash attention tiles K/V by `block_size`, runs local attention against Q one block at a time, and maintains a running max and running sum that update as each block arrives, rescaling each block's contribution correctly as it accumulates into the final result. The whole computation only needs `O(T)` intermediate state instead of `O(T^2)`, and it is mathematically identical to the one-shot result, not an approximation.

The core of online softmax is this step: when a new block's `block_max` exceeds the currently maintained `running_max`, both the accumulated output and the accumulated normalizer must be rescaled by `exp(running_max - new_max)` before the new block's contribution is added. Getting this rescaling factor wrong is the most common, and most subtle, bug in a flash attention implementation. It won't throw an error; it will just silently produce output that's numerically off from standard attention.

#### Quick Coding: `flash_attention`

```python
def flash_attention(Q, K, V, block_size, causal=False):
    ...
```

<details>
<summary>Reference solution</summary>

```python
def flash_attention(Q, K, V, block_size, causal=False):
    *lead, T, d = Q.shape
    Tk = K.shape[-2]
    dv = V.shape[-1]

    out = torch.zeros(*lead, T, dv, device=Q.device, dtype=Q.dtype)
    running_max = torch.full((*lead, T, 1), float("-inf"), device=Q.device)
    running_sum = torch.zeros((*lead, T, 1), device=Q.device)

    for start in range(0, Tk, block_size):
        end = min(start + block_size, Tk)
        Kb, Vb = K[..., start:end, :], V[..., start:end, :]
        scores = torch.einsum("...qd,...kd->...qk", Q, Kb) / math.sqrt(d)

        if causal:
            q_idx = torch.arange(T, device=Q.device)[:, None]
            k_idx = torch.arange(start, end, device=Q.device)[None, :]
            scores = scores.masked_fill(q_idx < k_idx, float("-inf"))

        block_max = scores.amax(dim=-1, keepdim=True)
        block_max = torch.where(torch.isneginf(block_max), running_max, block_max)
        new_max = torch.maximum(running_max, block_max)

        # rescale the previously accumulated result by exp(old max - new max)
        alpha = torch.exp(torch.where(torch.isneginf(running_max),
                                       torch.full_like(running_max, float("-inf")),
                                       running_max - new_max))
        alpha = torch.nan_to_num(alpha, nan=0.0, neginf=0.0)
        out = out * alpha
        running_sum = running_sum * alpha

        p = torch.exp(scores - new_max)
        p = torch.nan_to_num(p, nan=0.0)  # a fully-masked row has scores == new_max == -inf
        out = out + torch.einsum("...qk,...kd->...qd", p, Vb)
        running_sum = running_sum + p.sum(dim=-1, keepdim=True)
        running_max = new_max

    return out / running_sum
```

This implementation has been checked against the naive "compute the full `(T, T)` matrix, then softmax" version with NumPy `allclose`, in both causal and non-causal settings, including the edge case where an entire row is fully masked out within some block (`block_max = -inf`).

</details>

## Common mistakes in this module

- MHA and causal MHA differ only in the mask, never in how heads are split or concatenated.
- Cross-attention's attention matrix is `(T_q, T_kv)`; masks must follow K/V's sequence length, and should never be assumed square.
- Broadcasting K/V heads for GQA requires `repeat_interleave` (contiguous groups), never `repeat`/`tile` (interleaved groups): same output shape, completely different semantics.
- Linear attention is an approximation using a different similarity kernel and is not equivalent to softmax attention; flash attention is a pure computation-order optimization and is numerically exact.
- KV cache must store the "already rotated" K; decode-phase position indices must start at `cache_len`, never restart from 0.
- In flash attention's online-softmax update, the rescaling factor for the old accumulator is `exp(running_max - new_max)`. Getting the sign backward, or forgetting to rescale the old `running_sum`, are the two most common bugs.
