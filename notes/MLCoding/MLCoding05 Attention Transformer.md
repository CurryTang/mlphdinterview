# ML Coding · Attention & Transformer LM

对应 CS336 Assignment 1：Section 3.4.4-3.5。

## Exercise 1 · Stable Softmax

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

## Exercise 2 · Scaled Dot-Product Attention

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

## Exercise 3 · Causal MHA

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

## Exercise 4 · Transformer Block

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

## Exercise 5 · Transformer LM

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

## Exercise 6 · Resource Accounting

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

## Debug Checklist

- causal mask 的 True/False 语义要和 SDPA 实现一致。
- RoPE 只作用于 Q/K，不作用于 V。
- LM head 输出是 logits，不是 softmax probabilities。
