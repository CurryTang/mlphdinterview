# 03 · 注意力机制全家桶

面试与生产实践中被问到的注意力机制涵盖了各种变体：
- **Encoder 架构**（BERT / ViT）：双向全可见 `MultiHeadAttention`（带 Padding 屏蔽）；
- **跨模态 / 翻译架构**（Encoder-Decoder / VLM）：非对称长度序列的 `MultiHeadCrossAttention`；
- **现代大模型主流**（LLaMA 2/3、Mistral、Qwen）：`GroupedQueryAttention (GQA)` 与 `Sliding Window Attention`；
- **自回归推理引擎**（vLLM / TensorRT-LLM）：`KVCacheAttention` 与 `Flash Attention`（片上 SRAM Tiling 与 Online Softmax）。

本篇笔记**不使用任何封装好的高层黑盒算子**（如 `torch.nn.MultiheadAttention` 或 `F.scaled_dot_product_attention`），全部基于原生 PyTorch 与 `einops.rearrange` 显式展开线性投影、张量重排、缩放点积、掩码填充与矩阵乘法，将基础知识与手写代码通过折叠块清晰分离。

题目选取参考了开源题库 [TorchCode](https://github.com/duoan/TorchCode)，所有理论推导、代码实现与易错陷阱剖析均为独立重构。

---

## 核心架构与注意力变体实战

### Exercise 1 · MultiHeadAttention（双向，非因果）

在 BERT、ViT 等 Encoder 结构中，每个位置都能双向看到全部 Token，因此不需要因果下三角 Mask。Mask 要么为 `None`，要么用来屏蔽 Padding Token。

<details>
<summary><b>展开深入：MHA 张量流动、核心机制与面试常考陷阱</b></summary>

#### 1. 张量维度演进全生命周期
- 输入张量：$X \in \mathbb{R}^{B \times T \times D}$
- 显式线性投影：$Q = X W_Q^T, K = X W_K^T, V = X W_V^T$，投影后尺寸仍为 $(B, T, D)$
- 头拆分重排：通过 `einops.rearrange(x, "b t (h d) -> b h t d", h=num_heads)`，将通道维解耦为多头，张量尺寸变为 $(B, H, T, D_h)$，其中 $D_h = D / H$
- 批次矩阵乘得分：$S = \frac{Q K^T}{\sqrt{D_h}} \in \mathbb{R}^{B \times H \times T \times T}$
- 掩码填充：若给定 `key_padding_mask` $(B, T)$，需扩维广播至 $(B, 1, 1, T)$，将 Mask 为 True 的位置置为 $-\infty$
- 概率归一化与聚合：$A = \operatorname{softmax}(S, \dim=-1)$，输出 $O_{\text{head}} = A V \in \mathbb{R}^{B \times H \times T \times D_h}$
- 多头合并与最终投影：`rearrange(out, "b h t d -> b t (h d)")` 恢复至 $(B, T, D)$，经过 $W_O$ 线性变换输出

#### 2. 面试致命踩坑点
1. **多头切分顺序错误**：
   必须先按 $(H, D_h)$ 拆解隐藏维，再 transpose 到 $(B, H, T, D_h)$。如果先 transpose 再 reshape，各个 Head 会交织混合错误的通道特征，代码能跑且不会报错，但学出的注意力拓扑完全混乱。
2. **Padding Mask 广播规则**：
   `key_padding_mask` 维度为 $(B, T)$，而注意力分数为 $(B, H, T, T)$。必须在维度 1 和 2 插入单例维度 `[:, None, None, :]`，使得 Mask 能同时沿 Head 维与 Query 序列维广播，精准作用于 Key 序列。
3. **Convention 语义统一**：
   工业标准（PyTorch / HuggingFace）中，Padding Mask 中 `True` 代表“该位置是 Padding，应被忽略（填充 $-\infty$）”，`False` 代表“有效 Token”。

</details>

#### Quick Coding：`MultiHeadAttention`

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        ...

    def forward(self, x, key_padding_mask=None):
        ...
```

<details open>
<summary><b>参考代码：从零手写 MultiHeadAttention（显式投影与 einops 重排）</b></summary>

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
            # 约定：True 表示“忽略/屏蔽该 Key 位置”
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

Cross-attention 将 Query 与 Key/Value 的来源解耦：Query 来自解码器（如 Decoder 当前隐状态），Key/Value 来自另一模态或编码器输出（如 Encoder 文本上下文、ViT 视觉图像 Patch 特征）。

<details>
<summary><b>展开深入：交叉注意力非对称序列长度与 Mask 形状对齐解析</b></summary>

#### 1. 非方阵注意力图核心机理
- Query 长度 $T_q$ 与 Key/Value 长度 $T_{kv}$ 通常**严格不相等**（例如：解码生成第 5 个 Token，而 Encoder 文本包含 512 个 Token）；
- 点积矩阵形状为 $(B, H, T_q, T_{kv})$；
- 经过 Softmax 后与 $V \in (B, H, T_{kv}, D_h)$ 相乘，结果形状回到 $(B, H, T_q, D_h)$；
- **黄金定律**：Cross-Attention 的输出序列长度永远由 Query 决定（$T_q$），与 Key/Value 的长度 $T_{kv}$ 无关。

#### 2. Key Padding Mask 的非方阵广播
- `key_padding_mask` 的形状对应于 Encoder 序列，即 $(B, T_{kv})$；
- 广播形状必须为 $(B, 1, 1, T_{kv})$，直接沿 $T_q$ 维度复制；若误写为 $(B, 1, T_q, 1)$ 或 $(B, T_q)$ 则会触发静默逻辑错误或运行时 Shape Mismatch。

</details>

#### Quick Coding：`MultiHeadCrossAttention`

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        ...

    def forward(self, x_q, x_kv, key_padding_mask=None):
        ...
```

<details open>
<summary><b>参考代码：从零手写 MultiHeadCrossAttention（解耦 Q 与 KV 输入）</b></summary>

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
        # x_q:  (B, T_q,  D)  来自 decoder
        # x_kv: (B, T_kv, D)  来自 encoder / 图像特征，T_kv 可以不等于 T_q
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
            # key_padding_mask: (B, T_kv)，True 表示对应 key 位置需被屏蔽
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

### Exercise 3 · GroupedQueryAttention（GQA）

标准 MHA 中 Q/K/V 头数完全相同，自回归推理时每个头均需保留完整的 KV Cache，显存带宽与容量消耗随着并发并发度线性爆炸。GQA（LLaMA 2/3、Mistral、Qwen 标配）将 K/V 的头数压减到 $n_{\text{kv\_heads}} < n_{\text{heads}}$，每组若干个 Q 头共享同一组 K/V 头。极端情况 $n_{\text{kv\_heads}} = 1$ 即为 Multi-Query Attention (MQA)。

<details>
<summary><b>展开深入：GQA 核心设计、显存推导与 repeat_interleave vs tile 致命陷阱</b></summary>

#### 1. 每层 KV Cache 显存容量理论推导（FP16，序列长 $T$）
| 架构方案 | K/V 头数 | 单层 KV Cache 字节数 | 相对 MHA 显存降幅 |
|---|---|---|---|
| **MHA** | $H_q$ | $2 \times H_q \times T \times D_h \times 2$ bytes | 基线 (100%) |
| **GQA** | $H_{kv}$ | $2 \times H_{kv} \times T \times D_h \times 2$ bytes | 降低至 $\frac{H_{kv}}{H_q}$（如 LLaMA-70B 降为 1/8） |
| **MQA** | $1$ | $2 \times 1 \times T \times D_h \times 2$ bytes | 降低至 $\frac{1}{H_q}$ |

#### 2. 头对齐机制的致命陷阱：`repeat_interleave` vs `tile`
K/V 投影产出 $H_{kv}$ 个头，需广播扩展为 $H_q$ 个头才能与 Q 计算点积：
- **正确做法**：`torch.repeat_interleave(k, group_size, dim=1)`
  - 索引变换：$[0, 1] \to [0, 0, 0, 0, 1, 1, 1, 1]$
  - 保证 Q 的头 $0 \sim 3$ 归属第 0 组 KV，头 $4 \sim 7$ 归属第 1 组 KV；
- **错误写法**：`k.repeat(1, group_size, 1, 1)`（等价于 NumPy 的 `tile`）
  - 索引变换：$[0, 1] \to [0, 1, 0, 1, 0, 1, 0, 1]$
  - 导致偶数编号的 Q 头匹配 KV 头 0，奇数编号匹配 KV 头 1，语义拓扑彻底割裂！
  - **危险性**：张量 Shape 完全一致，程序不报任何 Exception，Loss 依然能够缓慢下降，但模型能力产生永久性损伤。

</details>

#### Quick Coding：`GroupedQueryAttention`

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, device=None, dtype=None):
        ...

    def forward(self, x, mask=None):
        ...
```

<details open>
<summary><b>参考代码：从零手写 GroupedQueryAttention（显式投影与 repeat_interleave）</b></summary>

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

        # 核心步骤：使用 repeat_interleave 按组广播 KV 头
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

Mistral 采用的局部滑动窗口注意力：每个 Query 只关注最近 `window` 个历史 Key（含自身），更早期的 Token 被强行屏蔽。单层注意力计算复杂度由 $\mathcal{O}(T^2)$ 直降为 $\mathcal{O}(T \cdot \text{window})$。

<details>
<summary><b>展开深入：滑动窗口感受野传播定理与计算量对比</b></summary>

#### 感受野级联扩散定理
单层虽然仅能看到 `window` 跨度的 Token，但当网络堆叠 $L$ 层时，第 $L$ 层的有效感受野达到 $L \times \text{window}$。这与卷积神经网络通过堆叠多层 $3 \times 3$ 小卷积核构建广阔全局感受野的原理完全同构。

</details>

#### Quick Coding：`sliding_window_attention`

```python
def sliding_window_attention(Q, K, V, window):
    ...
```

<details open>
<summary><b>参考代码：从零手写 sliding_window_attention</b></summary>

```python
def sliding_window_mask(T, window, device=None):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    # 下三角因果约束 (j <= i) 且窗口距离限制 (j > i - window)
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

Softmax Attention 的计算瓶颈在于必须物化完整的 $(T, T)$ 分数矩阵。线性注意力采用非负特征映射 $\phi(x)$（如 $\operatorname{ELU}(x) + 1$）替换指数相似度核，借助矩阵乘法的结合律重构计算图：

$$\operatorname{Softmax}(Q K^T) V \quad \Longrightarrow \quad \phi(Q) \big(\phi(K)^T V\big)$$

- 关联顺序改变：从 $(Q K^T) V$ 的 $\mathcal{O}(T^2 d)$ 跃迁为 $Q (K^T V)$ 的 $\mathcal{O}(T d^2)$。当 $T \gg d$ 时，计算与显存开销均降为严格线性。

<details>
<summary><b>展开深入：核技巧结合律重构与非自回归/因果前缀和推导</b></summary>

#### 因果版本的流式递推
在自回归因果模式下，未来时间步不可见。通过对外积项 $S_t = \sum_{j \le t} \phi(K_j) V_j^T$ 做累加求和（`torch.cumsum`），实现无显式注意力图的因果递推。

</details>

#### Quick Coding：`linear_attention`

```python
def linear_attention(Q, K, V, causal=False):
    ...
```

<details open>
<summary><b>参考代码：从零手写 linear_attention（因果与非因果）</b></summary>

```python
import torch.nn.functional as F

def feature_map(x):
    return F.elu(x) + 1.0  # 保证严格非负，避免 ReLU 零梯度死区

def linear_attention(Q, K, V, causal=False):
    Qp, Kp = feature_map(Q), feature_map(K)  # (..., T, d)

    if not causal:
        kv = torch.einsum("...kd,...ke->...de", Kp, V)      # (..., d, dv)
        k_sum = Kp.sum(dim=-2)                                # (..., d)
        num = torch.einsum("...qd,...de->...qe", Qp, kv)     # (..., T, dv)
        den = torch.einsum("...qd,...d->...q", Qp, k_sum).unsqueeze(-1)
        return num / den

    # 因果模式：利用累加和 cumsum 替代全量矩阵乘法，单 Token 递推复杂度仅 O(d * dv)
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

自回归文本生成阶段分为两步：
1. **Prefill 阶段**：一次性吞吐整段输入 Prompt，生成首个 Token，将所有历史位置的 $K, V$ 写入 Cache；
2. **Decode 阶段**：每次生成单 Token（$T=1$），$Q$ 仅对应当前新位置，与已缓存的历史全部 $K, V$ 拼接计算 Attention。整体生成时间复杂度由 $\mathcal{O}(T^2)$ 骤降至 $\mathcal{O}(T)$。

<details>
<summary><b>展开深入：Prefill 与 Decode 显存/访存带宽深度辨析（RoPE 旋转时机与绝对位置偏移）</b></summary>

#### 1. RoPE 旋转时机
缓存的 Key **必须是在各自绝对位置上已经旋转过的状态**。如果在缓存之后再整体旋转，相对位置语义将被彻底颠覆。

#### 2. Decode 绝对位置偏移
Decode 阶段为新 Token 分配位置索引时，必须从 `start_pos = cache_len` 开始计数，不可回退至 0。

</details>

#### Quick Coding：`KVCacheAttention`

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
<summary><b>参考代码：从零手写 KVCacheAttention（显式拼接与下三角因果 Mask）</b></summary>

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
        B, T, D = x.shape  # prefill 阶段 T = prompt_len；decode 阶段 T = 1
        q = rearrange(self.W_Q(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.W_K(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.W_V(x), "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope is not None:
            positions = torch.arange(start_pos, start_pos + T, device=x.device)
            positions = positions[None, None, :].expand(B, 1, T)
            q = self.rope(q, positions)
            k = self.rope(k, positions)  # 必须在存入缓存前完成绝对位置编码旋转

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
            # Prefill 阶段需对历史 Key 施加因果屏蔽
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

### Exercise 7 · Flash Attention（分块 + online softmax）

Flash attention 要解决的不是“attention 算得对不对”，而是“算的时候要不要把整个 $(T, T)$ 分数矩阵摆在显存里”。标准实现要先算出完整 `scores`，再整体做 softmax；FlashAttention 把 $K, V$ 按 `BLOCK_N` 分块，依次和当前 $Q$ 块做局部注意力，同时在片上寄存器维护流式更新的 running max 和 running sum，将先前累加量按新旧最大值差的指数项 $\exp(m_{\text{old}} - m_{\text{new}})$ 重新缩放。数学上和一次性算完的结果完全一致，不是近似。

> **核心复杂度辨析**：
> - **显存容量复杂度**：从 $\mathcal{O}(T^2)$ 骤降至 $\mathcal{O}(T \cdot D)$（反向丢弃中间激活图而在片上极速重算，彻底杜绝长文本训练 OOM）；
> - **HBM 访存 IO 量**：从 $\mathcal{O}(T^2 + TD)$ 降至 $\mathcal{O}(TD)$（数据驻留片上高速 SRAM 寄存器，打破显存带宽墙）；
> - **计算复杂度 (FLOPs)**：依然严格为 $\mathcal{O}(T^2 D)$（反向传播甚至因 SRAM 现场重算略微增加了 ~16.7% 浮点运算量）。性能提升纯粹源于摆脱内存带宽受限（Memory-Bound $\to$ Compute-Bound）。

本练习分为两个递进层次：
1. **Part A · PyTorch 在线 Softmax 算法逻辑模拟（CPU/原型验证）**：跑通纯 Python 分块与重缩放逻辑；
2. **Part B · Triton GPU 算子级实现（片上 SRAM Tiling + Tensor Core + 因果剪枝）**：工程生产级 GPU Kernel 编程实战。

---

#### Part A · PyTorch 算法级实现：`flash_attention`

##### Quick Coding：`flash_attention`

```python
def flash_attention(Q, K, V, block_size, causal=False):
    ...
```

<details>
<summary>参考答案（PyTorch 算法实现）</summary>

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

        # 之前累积的结果要按 exp(旧 max - 新 max) 重新缩放
        alpha = torch.exp(torch.where(torch.isneginf(running_max),
                                       torch.full_like(running_max, float("-inf")),
                                       running_max - new_max))
        alpha = torch.nan_to_num(alpha, nan=0.0, neginf=0.0)
        out = out * alpha
        running_sum = running_sum * alpha

        p = torch.exp(scores - new_max)
        p = torch.nan_to_num(p, nan=0.0)  # 整行被 mask 掉时 scores 和 new_max 都是 -inf
        out = out + torch.einsum("...qk,...kd->...qd", p, Vb)
        running_sum = running_sum + p.sum(dim=-1, keepdim=True)
        running_max = new_max

    return out / running_sum
```

这份实现和一次性算完整 `(T, T)` 矩阵再 softmax 的朴素版本，在因果和非因果两种设置下都用 NumPy 数值验证过 `allclose`，包括某一整行在某个分块里被完全 mask 掉（`block_max = -inf`）这种边界情况。

</details>

---

#### Part B · Triton GPU 算子级实战：`flash_attn_triton`

##### 算子工程设计与硬件映射
1. **Grid 划分与 CTA 映射**：
   - 启动网格为 2D：`grid = (triton.cdiv(N_CTX, BLOCK_M), Batch * NumHeads)`；
   - 每个 Program 实例（CTA / Thread Block）独占处理某一个 Batch、某一个 Head 下的一个 Query Tile（大小为 `BLOCK_M = 64`）；
2. **SRAM 片上数据流**：
   - **外层**：将当前 Query 块（`[BLOCK_M, HEAD_DIM]`）一次性装载至片上 SRAM 寄存器，在整个内层循环中**常驻不变**；
   - **内层**：沿 Key/Value 序列维度按 `BLOCK_N = 64` 流式循环加载；
   - **寄存器累加器**：
     - `m_i`: `[BLOCK_M]` 行最大值；
     - `l_i`: `[BLOCK_M]` 归一化分母；
     - `acc_o`: `[BLOCK_M, HEAD_DIM]` 输出累加值；
3. **因果边界剪枝（Causal Early Termination）**：
   - 若开启因果遮罩，当 Key 块起始位置 `start_n > (start_m + 1) * BLOCK_M` 时，后续所有块均为严格未来时间步，直接截断 `break` 循环，将计算量直接减半；
   - 处于对角线重叠的块，使用 `offs_m[:, None] >= offs_n[None, :]` 进行行级掩码。

##### Quick Coding：`_flash_attn_fwd_kernel`

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
<summary>参考答案（Triton Kernel 算子级完整实现）</summary>

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

    # 行与列在当前 Block 内的相对偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    # 内存基地址指针
    q_offset = off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = off_z * stride_kz + off_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_offset = off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # 1. 载入 Query 块至 SRAM 寄存器（外层常驻）
    q = tl.load(Q + q_offset, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 2. 初始化 Online Softmax 寄存器累加器
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 3. 因果注意力的分块循环上界计算
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    # 4. 内层循环：沿 Key/Value 流式载入分块
    for start_n in range(lo, hi, BLOCK_N):
        curr_offs_n = start_n + offs_n
        
        # 载入 K 块与 V 块到片上 SRAM
        k = tl.load(K + k_offset + start_n * stride_kn, mask=curr_offs_n[None, :] < N_CTX, other=0.0)
        v = tl.load(V + v_offset + start_n * stride_vn, mask=curr_offs_n[:, None] < N_CTX, other=0.0)

        # Tensor Core 点积：Q @ K.T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # 因果下三角屏蔽
        if IS_CAUSAL:
            mask = offs_m[:, None] >= curr_offs_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        # 局部分块的最大值与指数和
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # 核心：根据 (旧 max - 新 max) 重新缩放先前的累加和
        alpha = tl.exp(m_i - m_ij)
        acc_o = acc_o * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # 累加当前块的 V 特征贡献
        acc_o += tl.dot(p.to(v.dtype), v)

    # 5. 最终除以归一化分母，写回全局内存 HBM
    acc_o = acc_o / l_i[:, None]
    out_offset = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(Out + out_offset, acc_o.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    
    # 存储 logsumexp 便于反向传播
    if L is not None:
        l_ptrs = L + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def flash_attention_triton(q, k, v, causal=True, sm_scale=None):
    """
    输入张量 shape 均为：(Batch, Heads, SeqLen, HeadDim)
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        
    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)
    L = torch.empty((Z * H, N_CTX), device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64

    # 网格维度：(Query 分块数, Batch * Heads)
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

##### 单元测试与数值对齐验证代码
```python
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        B, H, S, D = 2, 8, 1024, 64
        q = torch.randn((B, H, S, D), device=device, dtype=torch.float16)
        k = torch.randn((B, H, S, D), device=device, dtype=torch.float16)
        v = torch.randn((B, H, S, D), device=device, dtype=torch.float16)

        # 1. 运行 Triton FlashAttention 算子
        out_triton = flash_attention_triton(q, k, v, causal=True)

        # 2. 运行 PyTorch 原生 Eager Attention
        mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D) + mask
        attn = torch.softmax(scores, dim=-1)
        out_ref = torch.matmul(attn, v)

        # 3. 验证精度完全对齐
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
        print("Triton FlashAttention 数值精度验证通过！")
```

</details>

## 本模块易错点

- MHA 和 causal MHA 的区别只在 mask，不在切 head、拼 head 的流程。
- Cross-attention 的 attention 矩阵是 `(T_q, T_kv)`，mask 形状要跟着 K/V 的序列长度走，不能默认方阵。
- GQA 广播 K/V 头必须用 `repeat_interleave`（分组连续），不能用 `repeat`/`tile`（分组交错），两者 shape 一样但语义完全不同。
- 线性注意力是换了一个相似度核的近似，不等价于 softmax attention；flash attention 是纯计算顺序优化，数值上完全等价。
- KV cache 要缓存"旋转后"的 K；decode 阶段的位置编号必须从 `cache_len` 开始，不能从 0 重新数。
- flash attention 的 online softmax 更新里，旧累积量的缩放系数是 `exp(running_max - new_max)`，写反符号或者忘记缩放旧的 `running_sum` 是最常见的两处 bug。
