# ML Coding · Attention & Transformer LM

对应 CS336 Assignment 1：Section 3.4.4-3.5。

使用方式：每题先看目标和验收标准，确认自己知道要实现什么；再展开参考答案，对照代码骨架、边界条件和 sanity checks。

## Exercise 1 · Stable Softmax

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`softmax`

接口：

```text
softmax(x, dim) -> same shape
```

关键约束：

- 沿 `dim` 减最大值。
- 输出沿 `dim` 求和为 1。
- 支持 arbitrary shape。

测试：

```bash
uv run pytest -k test_softmax_matches_pytorch
```

</details>

<details class="solution">
<summary>参考答案</summary>

stable softmax 的核心是先减去目标维度上的最大值，避免 `exp` overflow。

```python
import torch

def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    shifted = x - x_max
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)
```

检查点：

```python
x = torch.tensor([1000.0, 1001.0, 1002.0])
y = softmax(x, dim=0)
assert torch.isfinite(y).all()
assert torch.allclose(y.sum(), torch.tensor(1.0))
```

不要先 `exp(x)` 再 normalize；大 logit 会直接 overflow。

</details>

## Exercise 2 · Scaled Dot-Product Attention

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`scaled_dot_product_attention`

接口：

```text
Q: (..., queries, d_k)
K: (..., keys, d_k)
V: (..., keys, d_v)
mask: (..., queries, keys)
output: (..., queries, d_v)
```

关键约束：

- logits 除以 `sqrt(d_k)`。
- mask False 的位置 softmax probability 为 0。
- mask True 的位置概率和为 1。
- 支持 3D / 4D / arbitrary leading dims。

测试：

```bash
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
```

</details>

<details class="solution">
<summary>参考答案</summary>

Scaled dot-product attention 做三件事：算 query-key 相似度、mask 掉非法位置、用 softmax 权重加权 value。

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

mask 语义要固定：这里 `True` 表示可以看，`False` 表示禁止看。causal mask 可以写成：

```python
def causal_mask(q_len, k_len, device=None):
    q = torch.arange(q_len, device=device)[:, None]
    k = torch.arange(k_len, device=device)[None, :]
    return k <= q
```

如果某一行全部被 mask，softmax 会得到 NaN；标准 causal self-attention 不会出现这种情况，因为每个位置至少能看自己。

</details>

## Exercise 3 · Causal MHA

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`multihead_self_attention`

结构：

```text
x -> Q/K/V projections
reshape into heads
optional RoPE on Q/K
causal SDPA
merge heads
output projection
```

关键约束：

- causal mask 只允许 `j <= i`。
- head dimension 当作 batch-like dimension。
- Q/K/V projection 可以是三次 batched matmul。
- output shape 与 input shape 一致。

测试：

```bash
uv run pytest -k test_multihead_self_attention
uv run pytest -k test_multihead_self_attention_with_rope
```

</details>

<details class="solution">
<summary>参考答案</summary>

Causal MHA 的实现重点是 shape discipline。输入 `(B,T,D)`，拆成 `(B,H,T,Dh)` 后把 head 维当作 batch-like dim。

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

RoPE 只给 Q/K 加位置旋转，因为 attention score 需要相对位置信息；V 是被读取的内容，不参与匹配。

</details>

## Exercise 4 · Transformer Block

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`transformer_block`

pre-norm block：

```text
y = x + MHA(RMSNorm(x))
out = y + FFN(RMSNorm(y))
```

关键约束：

- residual stream shape 始终是 `(batch, seq, d_model)`。
- 两个 RMSNorm 参数独立。
- MHA 使用 causal mask 和 RoPE。

测试：

```bash
uv run pytest -k test_transformer_block
```

</details>

<details class="solution">
<summary>参考答案</summary>

CS336 里推荐实现 pre-norm Transformer block，因为训练更稳定：

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

关键不变量：

```text
输入 shape:  (B, T, D)
attn 输出:   (B, T, D)
ffn 输出:    (B, T, D)
residual stream 一直是 (B, T, D)
```

如果 residual shape 改了，后续 block 一定会出错。

</details>

## Exercise 5 · Transformer LM

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`transformer_lm`

结构：

```text
token ids
-> token embedding
-> N transformer blocks
-> final RMSNorm
-> LM head
-> logits
```

输出：

```text
logits: (batch_size, sequence_length, vocab_size)
```

测试：

```bash
uv run pytest -k test_transformer_lm
```

</details>

<details class="solution">
<summary>参考答案</summary>

Transformer LM 把 token ids 映射到 logits，不在模型内部做 softmax：

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

训练时 cross entropy 直接吃 logits；generation 时只取最后一个位置的 logits：

```python
logits = model(tokens)
next_logits = logits[:, -1, :]
```

</details>

## Exercise 6 · Resource Accounting

<details class="exercise">
<summary><span class="q-label">参考</span> <span class="q-text">展开目标、接口与验收标准</span></summary>

对应 PDF：`transformer_accounting`

写一个参数量 / FLOPs accounting script。

输入：

```text
vocab_size
context_length
num_layers
d_model
num_heads
d_ff
```

输出：

```text
parameter count by component
memory to load parameters
forward FLOPs by component
attention FLOPs vs MLP FLOPs
context length sensitivity
```

比较：

```text
GPT-2 small
GPT-2 medium
GPT-2 large
GPT-2 XL
GPT-2 XL with context_length = 16,384
```

</details>

<details class="solution">
<summary>参考答案</summary>

resource accounting 的目标是训练你估算瓶颈，而不是精确复刻 profiler。一个实用参数账本：

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

粗略 FLOPs：

```text
Linear forward FLOPs ≈ 2 * tokens * in_dim * out_dim
Attention QK FLOPs ≈ 2 * B * H * T * T * Dh
Attention PV FLOPs ≈ 2 * B * H * T * T * Dh
MLP FLOPs ≈ 2 * B*T*D*Dff * 3
```

context length 从 1024 增到 16384 时，attention 的 `T^2` 项会明显放大；MLP 仍主要随 `T` 线性增长。这个对比就是为什么长上下文训练和推理会特别关注 attention kernel 与 KV cache。

</details>

## Debug Checklist

<details class="exercise">
<summary><span class="q-label">Debug</span> <span class="q-text">展开检查项</span></summary>

- causal mask 的 True/False 语义要和 SDPA 实现一致。
- RoPE 只作用于 Q/K，不作用于 V。
- LM head 输出是 logits，不是 softmax probabilities。

</details>
