# Transformer Basics · Attention Zoo: From MHA to GQA, Sliding Window, Linear Attention, KV Cache, and Flash Attention

Interviews and production systems frequently explore attention variants:
- **Encoder Architectures** (BERT / ViT): Bidirectional, full-visibility `MultiHeadAttention` with padding masking;
- **Cross-Modal & Translation** (Encoder-Decoder / VLM): Asymmetric-length sequence `MultiHeadCrossAttention`;
- **Modern Foundation LLMs** (LLaMA 2/3, Mistral, Qwen): `GroupedQueryAttention (GQA)` and `Sliding Window Attention`;
- **Autoregressive Inference Engines** (vLLM / TensorRT-LLM): `KVCacheAttention` and `Flash Attention` (on-chip SRAM tiling & online softmax).

This note **does not use any opaque high-level blackbox helper** (such as `torch.nn.MultiheadAttention` or `F.scaled_dot_product_attention`). Everything is built from scratch using pure PyTorch and `einops.rearrange` to explicitly unroll linear projections, tensor rearranging, scaled dot products, key padding masks, and softmax aggregation. Theory, gotchas, and implementations are cleanly organized into collapsible blocks.

The exercises follow the topics in the open-source [TorchCode](https://github.com/duoan/TorchCode) benchmark, with all derivations, code, and pitfall analyses independently rewritten.

---

## Core Architectures & Hands-On Attention Variants

### Exercise 1 · MultiHeadAttention (Bidirectional, Non-Causal)

In encoder models like BERT and ViT, tokens attend bidirectionally across the full sequence. There is no lower-triangular causal mask; the mask is either `None` or used to ignore padding positions.

<details>
<summary><b>Deep Dive: MHA Tensor Shapes, Lifecycle & Common Interview Traps</b></summary>

#### 1. End-to-End Tensor Shape Trace
- Input: $X \in \mathbb{R}^{B \times T \times D}$
- Explicit Linear Projections: $Q = X W_Q^T, K = X W_K^T, V = X W_V^T$, shape $(B, T, D)$
- Head Split via `rearrange`: `rearrange(x, "b t (h d) -> b h t d", h=num_heads)`, shape $(B, H, T, D_h)$ where $D_h = D / H$
- Batched Score Matmul: $S = \frac{Q K^T}{\sqrt{D_h}} \in \mathbb{R}^{B \times H \times T \times T}$
- Key Padding Masking: If `key_padding_mask` $(B, T)$ is given, unsqueeze to $(B, 1, 1, T)$ and fill `True` positions with $-\infty$
- Softmax & Value Aggregation: $A = \operatorname{softmax}(S, \dim=-1)$, $O_{\text{head}} = A V \in \mathbb{R}^{B \times H \times T \times D_h}$
- Merge Heads & Output Projection: `rearrange(out, "b h t d -> b t (h d)")` back to $(B, T, D)$, followed by $W_O$ projection

#### 2. Critical Pitfalls
1. **Reshape vs Transpose Order**:
   Split $D$ into $(H, D_h)$ before transposing to $(B, H, T, D_h)$. If transposed first and reshaped second, heads interleave incorrect channels. The code runs without error and loss may even descend, but attention topology is scrambled.
2. **Padding Mask Broadcasting**:
   `key_padding_mask` has shape $(B, T)$. Because scores are $(B, H, T, T)$, the mask must be expanded to `[:, None, None, :]` to broadcast over heads and queries while masking out keys.
3. **Convention Semantics**:
   Under PyTorch / HuggingFace convention, `True` denotes "ignore/pad this key token (fill with $-\infty$)", and `False` denotes a valid token.

</details>

#### Quick Coding: `MultiHeadAttention`

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        ...

    def forward(self, x, key_padding_mask=None):
        ...
```

<details open>
<summary><b>Reference Implementation: Scratch MultiHeadAttention (Explicit Projections & einops)</b></summary>

```python
import torch
import torch.nn as nn
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = nn.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, D)
        q = rearrange(
            self.W_Q(x), "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        k = rearrange(
            self.W_K(x), "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        v = rearrange(
            self.W_V(x), "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        # q, k, v: (B, H, T, Dh)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        # scores: (B, H, T, T)

        if key_padding_mask is not None:
            # Convention: True means "ignore/mask this key position"
            mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, T)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        # out: (B, H, T, Dh)

        out = rearrange(out, "b h t d -> b t (h d)")
        return self.W_O(out)  # (B, T, D)
```

</details>

---

### Exercise 2 · MultiHeadCrossAttention

Cross-attention decouples Query from Key/Value sources: Q originates from the decoder sequence, while K and V come from an external sequence (e.g. encoder representations, multimodal vision embeddings).

<details>
<summary><b>Deep Dive: Non-Square Attention Maps & Mask Alignment</b></summary>

#### 1. Non-Square Geometry
- Decoder query length $T_q$ and encoder key length $T_{kv}$ are **rarely equal** ($T_q \ne T_{kv}$);
- Score matrix shape: $(B, H, T_q, T_{kv})$;
- After softmax and multiplying by $V \in (B, H, T_{kv}, D_h)$, shape returns to $(B, H, T_q, D_h)$;
- **Golden Rule**: Output sequence length is strictly dictated by $Q$ ($T_q$), regardless of $T_{kv}$.

#### 2. Key Padding Masking
- The mask aligns with the encoder keys: shape $(B, T_{kv})$;
- Broadcast to $(B, 1, 1, T_{kv})$ so it replicates along the query dimension $T_q$.

</details>

#### Quick Coding: `MultiHeadCrossAttention`

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        ...

    def forward(self, x_q, x_kv, key_padding_mask=None):
        ...
```

<details open>
<summary><b>Reference Implementation: Scratch MultiHeadCrossAttention (Decoupled Q & KV)</b></summary>

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = nn.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x_q, x_kv, key_padding_mask=None):
        # x_q:  (B, T_q,  D)  from decoder
        # x_kv: (B, T_kv, D)  from encoder / vision features
        q = rearrange(
            self.W_Q(x_q), "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        k = rearrange(
            self.W_K(x_kv), "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        v = rearrange(
            self.W_V(x_kv), "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        # q: (B, H, T_q, Dh), k/v: (B, H, T_kv, Dh)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        # scores: (B, H, T_q, T_kv)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, T_kv)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        # out: (B, H, T_q, Dh)

        out = rearrange(out, "b h t d -> b t (h d)")
        return self.W_O(out)  # (B, T_q, D)
```

</details>

---

### Exercise 3 · GroupedQueryAttention (GQA)

Standard MHA maintains separate K/V heads for every Q head, leading to linear KV cache expansion during inference. GQA (LLaMA 2/3, Mistral, Qwen) compresses K/V head count to $n_{\text{kv\_heads}} < n_{\text{heads}}$, sharing each K/V head across a group of query heads. Setting $n_{\text{kv\_heads}} = 1$ yields Multi-Query Attention (MQA).

<details>
<summary><b>Deep Dive: KV Cache Footprint Derivation & repeat_interleave vs tile Pitfall</b></summary>

#### 1. Per-Layer KV Cache Footprint (FP16, Sequence Length $T$)
| Attention Paradigm | K/V Heads | Cache Bytes per Layer | Relative Footprint |
|---|---|---|---|
| **MHA** | $H_q$ | $2 \times H_q \times T \times D_h \times 2$ bytes | Baseline (100%) |
| **GQA** | $H_{kv}$ | $2 \times H_{kv} \times T \times D_h \times 2$ bytes | Scaled to $\frac{H_{kv}}{H_q}$ (e.g. 1/8 in LLaMA-70B) |
| **MQA** | $1$ | $2 \times 1 \times T \times D_h \times 2$ bytes | Scaled to $\frac{1}{H_q}$ |

#### 2. Head Broadcasting Trap: `repeat_interleave` vs `tile`
K and V produce $H_{kv}$ heads, which must expand to $H_q$ heads to compute dot products with Q:
- **Correct**: `torch.repeat_interleave(k, group_size, dim=1)`
  - Index map: $[0, 1] \to [0, 0, 0, 0, 1, 1, 1, 1]$
  - Query heads $0 \sim 3$ map to KV head 0; heads $4 \sim 7$ map to KV head 1.
- **Flawed**: `k.repeat(1, group_size, 1, 1)` (NumPy `tile`)
  - Index map: $[0, 1] \to [0, 1, 0, 1, 0, 1, 0, 1]$
  - Interleaves even/odd heads to mismatched KV slices. The tensor shapes match identically, no error is thrown, but training semantics are ruined.

</details>

#### Quick Coding: `GroupedQueryAttention`

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, device=None, dtype=None):
        ...

    def forward(self, x, mask=None):
        ...
```

<details open>
<summary><b>Reference Implementation: Scratch GroupedQueryAttention (Explicit repeat_interleave)</b></summary>

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.group_size = num_heads // num_kv_heads

        self.W_Q = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.W_O = nn.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        q = rearrange(
            self.W_Q(x), "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        k = rearrange(
            self.W_K(x), "b t (h d) -> b h t d",
            h=self.num_kv_heads,
        )
        v = rearrange(
            self.W_V(x), "b t (h d) -> b h t d",
            h=self.num_kv_heads,
        )

        # Broadcast KV heads to match Q groups
        k = torch.repeat_interleave(k, self.group_size, dim=1)  # (B, H, T, Dh)
        v = torch.repeat_interleave(v, self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        # scores: (B, H, T, T)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        # out: (B, H, T, Dh)

        out = rearrange(out, "b h t d -> b t (h d)")
        return self.W_O(out)  # (B, T, D)
```

</details>

---

### Exercise 4 · Sliding Window Attention

Mistral's localized sliding window attention constrains each query to attend only to the most recent `window` keys (including itself). Single-layer computational complexity is cut from $\mathcal{O}(T^2)$ to $\mathcal{O}(T \cdot \text{window})$.

<details>
<summary><b>Deep Dive: Receptive Field Diffusion Across Layers</b></summary>

#### Receptive Field Cascade
While a single layer observes only `window` positions, stacking $L$ layers yields an effective receptive field of $L \times \text{window}$. This mirrors CNN designs where stacks of small $3 \times 3$ kernels cover wide input fields.

</details>

#### Quick Coding: `sliding_window_attention`

```python
def sliding_window_attention(Q, K, V, window):
    ...
```

<details open>
<summary><b>Reference Implementation: Scratch sliding_window_attention</b></summary>

```python
def sliding_window_mask(T, window, device=None):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    # Lower triangular causal constraint AND within window distance
    return (j <= i) & (j > i - window)

def sliding_window_attention(Q, K, V, window):
    # Q, K, V: (B, H, T, Dh)
    d_k = Q.shape[-1]
    T = Q.shape[-2]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    valid_mask = sliding_window_mask(T, window, device=Q.device)  # (T, T)
    scores = scores.masked_fill(~valid_mask[None, None, :, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)  # (B, H, T, Dh)
```

</details>

---

### Exercise 5 · Linear Attention

Softmax attention is bottlenecked by materializing the full $(T, T)$ score matrix. Linear attention replaces the exponential similarity kernel with a non-negative feature map $\phi(x)$ (such as $\operatorname{ELU}(x) + 1$), reordering the associative product:

$$\operatorname{Softmax}(Q K^T) V \quad \Longrightarrow \quad \phi(Q) \big(\phi(K)^T V\big)$$

- Associative reordering: Shifts from $(Q K^T) V$ at $\mathcal{O}(T^2 d)$ to $Q (K^T V)$ at $\mathcal{O}(T d^2)$. When $T \gg d$, compute and memory scale linearly.

<details>
<summary><b>Deep Dive: Associativity & Causal Prefix Sums</b></summary>

#### Streaming Causal Recurrence
In autoregressive mode, future steps cannot be viewed. Cumulative sums (`torch.cumsum`) over outer products $S_t = \sum_{j \le t} \phi(K_j) V_j^T$ allow token-by-token recurrence in $\mathcal{O}(d \cdot d_v)$ state without ever forming a dense $(T, T)$ matrix.

</details>

#### Quick Coding: `linear_attention`

```python
def linear_attention(Q, K, V, causal=False):
    ...
```

<details open>
<summary><b>Reference Implementation: Scratch linear_attention</b></summary>

```python
import torch.nn.functional as F

def feature_map(x):
    return F.elu(x) + 1.0  # Guarantees strictly positive values

def linear_attention(Q, K, V, causal=False):
    Qp, Kp = feature_map(Q), feature_map(K)  # (..., T, d)

    if not causal:
        kv = torch.einsum("...kd,...ke->...de", Kp, V)      # (..., d, dv)
        k_sum = Kp.sum(dim=-2)                                # (..., d)
        num = torch.einsum("...qd,...de->...qe", Qp, kv)     # (..., T, dv)
        den = torch.einsum("...qd,...d->...q", Qp, k_sum).unsqueeze(-1)
        return num / den

    # Causal mode: replace full matrix product with cumsum for O(d * dv) recurrence
    outer = torch.einsum("...tk,...tv->...tkv", Kp, V)  # (..., T, d, dv)
    S_cum = torch.cumsum(outer, dim=-3)
    z_cum = torch.cumsum(Kp, dim=-2)
    num = torch.einsum("...td,...tdv->...tv", Qp, S_cum)
    den = torch.einsum("...td,...td->...t", Qp, z_cum).unsqueeze(-1)
    return num / den
```

</details>

---

### Exercise 6 · KV Cache Attention

Autoregressive inference operates in two phases:
1. **Prefill Phase**: Ingest the entire input prompt at once, generate the first token, and save all keys and values to cache;
2. **Decode Phase**: Generate one token at a time ($T=1$), query attending over cached historical keys/values. Cumulative compute scales as $\mathcal{O}(T)$ rather than $\mathcal{O}(T^2)$.

<details>
<summary><b>Deep Dive: RoPE Timing & Absolute Position Offsets</b></summary>

#### 1. RoPE Rotation Timing
Cached keys **must be stored with their absolute position rotations already applied**. Rotating after concatenation destroys relative position properties.

#### 2. Decode Position Offsets
During decode, position indices for the newly generated token must start at `start_pos = cache_len`, not reset to 0.

</details>

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

<details open>
<summary><b>Reference Implementation: Scratch KVCacheAttention (Explicit Causal Masking)</b></summary>

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = nn.Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = rope
        self.cache_k = None  # (B, H, cache_len, Dh)
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, start_pos, use_cache=True):
        B, T, D = x.shape  # prefill T = prompt_len; decode T = 1
        q = rearrange(self.W_Q(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.W_K(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.W_V(x), "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope is not None:
            positions = torch.arange(start_pos, start_pos + T, device=x.device)
            positions = positions[None, None, :].expand(B, 1, T)
            q = self.rope(q, positions)
            k = self.rope(k, positions)  # Must rotate before appending to cache

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v

        Tk = k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        if T > 1:
            q_idx = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
            k_idx = torch.arange(Tk, device=x.device)[None, :]
            mask = k_idx > q_idx
            scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.W_O(out)
```

</details>

---

### Exercise 7 · Flash Attention (Tiling + Online Softmax)

Flash attention isn't about whether attention is computed correctly. It is about whether computing it requires holding the entire $(T, T)$ score matrix in memory. The standard implementation computes the full `scores` matrix and then softmaxes it all at once; flash attention tiles $K, V$ by `BLOCK_N`, runs local attention against the current $Q$ block, and maintains a running max and running sum inside on-chip registers that update as each block arrives, rescaling accumulated state by $\exp(m_{\text{old}} - m_{\text{new}})$. It is mathematically identical to the one-shot result, not an approximation.

> **Key Complexity Distinctions**:
> - **Activation Memory Footprint**: Drops from $\mathcal{O}(T^2)$ to $\mathcal{O}(T \cdot D)$ (recomputation in backward discards intermediate score maps, completely eliminating training OOM);
> - **HBM IO Traffic**: Reduces from $\mathcal{O}(T^2 + TD)$ to $\mathcal{O}(T D)$ (intermediates stay resident inside fast on-chip SRAM registers);
> - **Compute FLOPs**: Strictly remains $\mathcal{O}(T^2 D)$ (backward pass even increases by ~16.7% FLOPs due to on-the-fly SRAM recomputation). Speedups stem entirely from transitioning from Memory-Bound to Compute-Bound execution regimes.

This exercise is structured into two progressive stages:
1. **Part A · PyTorch Algorithmic Simulation (CPU / Prototyping)**: Verify tiling and Online Softmax rescaling logic;
2. **Part B · Production Triton GPU Kernel Implementation (`_flash_attn_fwd_kernel`)**: Full GPU kernel programming with SRAM tiling, Tensor Cores, and causal early termination.

---

#### Part A · PyTorch Algorithmic Implementation: `flash_attention`

##### Quick Coding: `flash_attention`

```python
def flash_attention(Q, K, V, block_size, causal=False):
    ...
```

<details>
<summary>Reference solution (PyTorch Algorithm)</summary>

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

---

#### Part B · Triton GPU Kernel Implementation: `flash_attn_triton`

##### Kernel Systems Architecture & Hardware Mapping
1. **Grid Partitioning & CTA Scheduling**:
   - 2D execution grid: `grid = (triton.cdiv(N_CTX, BLOCK_M), Batch * NumHeads)`;
   - Each Program instance (CTA / Thread Block) independently computes one Query tile (size `BLOCK_M = 64`) for a specific Batch index and Head;
2. **On-Chip SRAM Dataflow**:
   - **Outer Loop**: Current Query block (`[BLOCK_M, HEAD_DIM]`) is loaded once into on-chip SRAM registers and **remains persistent** across all inner iterations;
   - **Inner Loop**: Streams through Key and Value tiles along the sequence dimension in blocks of `BLOCK_N = 64`;
   - **Register Accumulators**:
     - `m_i`: `[BLOCK_M]` running row maxima;
     - `l_i`: `[BLOCK_M]` running row normalizer denominators;
     - `acc_o`: `[BLOCK_M, HEAD_DIM]` running unnormalized attention output matrix;
3. **Causal Early Termination**:
   - If causal masking is enabled, when Key tile start index `start_n > (start_m + 1) * BLOCK_M`, all future tiles are strictly unobservable. The kernel immediately executes `break`, halving global FLOPs;
   - Overlapping diagonal blocks apply row-wise boolean masking via `offs_m[:, None] >= offs_n[None, :]`.

##### Quick Coding: `_flash_attn_fwd_kernel`

```python
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    ...

def flash_attention_triton(q, k, v, causal=True, sm_scale=None):
    ...
```

<details>
<summary>Reference solution (Complete Triton Kernel Implementation)</summary>

```python
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    # Relative offsets inside current block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    # Base pointers
    q_offset = off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = off_z * stride_kz + off_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_offset = off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # 1. Load Query tile into SRAM registers (outer loop invariant)
    q = tl.load(Q + q_offset, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 2. Initialize Online Softmax accumulators in registers
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 3. Causal boundary upper limit
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    # 4. Inner loop over streaming Key and Value blocks
    for start_n in range(lo, hi, BLOCK_N):
        curr_offs_n = start_n + offs_n
        
        # Load K and V tiles into SRAM
        k = tl.load(K + k_offset + start_n * stride_kn, mask=curr_offs_n[None, :] < N_CTX, other=0.0)
        v = tl.load(V + v_offset + start_n * stride_vn, mask=curr_offs_n[:, None] < N_CTX, other=0.0)

        # Tensor Core GEMM: Q @ K.T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # Causal masking
        if IS_CAUSAL:
            mask = offs_m[:, None] >= curr_offs_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        # Local block maximum and exponentials
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Dynamic rescaling by exp(m_old - m_new)
        alpha = tl.exp(m_i - m_ij)
        acc_o = acc_o * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Accumulate Value contribution
        acc_o += tl.dot(p.to(v.dtype), v)

    # 5. Final normalization and write-back to global HBM
    acc_o = acc_o / l_i[:, None]
    out_offset = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(Out + out_offset, acc_o.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    
    # Store logsumexp for backward pass
    if L is not None:
        l_ptrs = L + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def flash_attention_triton(q, k, v, causal=True, sm_scale=None):
    """
    Input tensor shapes: (Batch, Heads, SeqLen, HeadDim)
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        
    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)
    L = torch.empty((Z * H, N_CTX), device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64

    # Grid: (Query chunk count, Batch * Heads)
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        L, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=D,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return out
```

##### Unit Test & Numerical Validation
```python
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        B, H, S, D = 2, 8, 1024, 64
        q = torch.randn((B, H, S, D), device=device, dtype=torch.float16)
        k = torch.randn((B, H, S, D), device=device, dtype=torch.float16)
        v = torch.randn((B, H, S, D), device=device, dtype=torch.float16)

        # 1. Run Triton FlashAttention kernel
        out_triton = flash_attention_triton(q, k, v, causal=True)

        # 2. Run PyTorch eager reference attention
        mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D) + mask
        attn = torch.softmax(scores, dim=-1)
        out_ref = torch.matmul(attn, v)

        # 3. Validate numerical precision
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
        print("Triton FlashAttention validation passed!")
```

</details>

## Common mistakes in this module

- MHA and causal MHA differ only in the mask, never in how heads are split or concatenated.
- Cross-attention's attention matrix is `(T_q, T_kv)`; masks must follow K/V's sequence length, and should never be assumed square.
- Broadcasting K/V heads for GQA requires `repeat_interleave` (contiguous groups), never `repeat`/`tile` (interleaved groups): same output shape, completely different semantics.
- Linear attention is an approximation using a different similarity kernel and is not equivalent to softmax attention; flash attention is a pure computation-order optimization and is numerically exact.
- KV cache must store the "already rotated" K; decode-phase position indices must start at `cache_len`, never restart from 0.
- In flash attention's online-softmax update, the rescaling factor for the old accumulator is `exp(running_max - new_max)`. Getting the sign backward, or forgetting to rescale the old `running_sum`, are the two most common bugs.
