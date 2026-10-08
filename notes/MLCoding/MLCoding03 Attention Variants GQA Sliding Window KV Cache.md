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
- **维度对应关系**：Cross-Attention 的输出序列长度永远由 Query 决定（$T_q$），与 Key/Value 的长度 $T_{kv}$ 无关。

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

标准 Softmax Attention 的核心计算与显存瓶颈在于必须物化完整的 $(B, H, T, T)$ 注意力得分矩阵。线性注意力采用严格非负特征映射 $\phi(x)$（如 $\operatorname{ELU}(x) + 1$）替代指数相似度核，借助矩阵乘法的结合律重构计算图：

$$\operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{D_h}}\right) V \quad \Longrightarrow \quad \frac{\phi(Q) \big(\phi(K)^T V\big)}{\phi(Q) \big(\sum_{t=1}^T \phi(K)_{t}\big)^T}$$

- **张量维度与关联顺序改变**：
  - **标准 Softmax 路径**：$(Q K^T) V \to (B, H, T, T) \times (B, H, T, D_v)$，计算复杂度为 $\mathcal{O}(B \cdot H \cdot T^2 \cdot D_h)$，显存占用为 $\mathcal{O}(B \cdot H \cdot T^2)$；
  - **线性注意力重构路径**：$\phi(Q) \big(\phi(K)^T V\big) \to (B, H, T, D_h) \times (B, H, D_h, D_v)$，计算复杂度为 $\mathcal{O}(B \cdot H \cdot T \cdot D_h \cdot D_v)$，显存占用为 $\mathcal{O}(B \cdot H \cdot D_h \cdot D_v)$。
- **长序列计算飞跃**：当序列长度 $T \gg D_h$ 时（例如长文本 $T=128\text{k}, D_h=128$），计算量与显存开销由二次方 $\mathcal{O}(T^2)$ 骤降为严格线性 $\mathcal{O}(T)$，解码推理显存降为 $\mathcal{O}(1)$ 常数。

<details>
<summary><b>展开深入：张量维度演进全生命周期、因果递推与理论边界</b></summary>

#### 1. 张量维度演进全生命周期对照表

| 计算阶段 / 张量 | 标准 Softmax Attention 形状 | 线性注意力 (Linear Attention) 形状 | 物理语义与计算特点 |
|---|---|---|---|
| **Query 输入 $Q$** | $(B, H, T, D_h)$ | $(B, H, T, D_h) \xrightarrow{\phi} (B, H, T, D_h)$ | 经特征映射 $\phi(Q) = \operatorname{elu}(Q) + 1$ 保障非负 |
| **Key 输入 $K$** | $(B, H, T, D_h)$ | $(B, H, T, D_h) \xrightarrow{\phi} (B, H, T, D_h)$ | 经特征映射 $\phi(K) = \operatorname{elu}(K) + 1$ 保障非负 |
| **Value 输入 $V$** | $(B, H, T, D_v)$ | $(B, H, T, D_v)$ | 内容特征（通常 $D_v = D_h$） |
| **中间聚合矩阵** | 注意力分数图 $S \in (B, H, T, T)$ | 键值联想记忆 $M = \phi(K)^T V \in (B, H, D_h, D_v)$ | **核心分水岭**：$M$ 与序列长 $T$ 完全无关，尺寸恒定 |
| **未归一化分子** | $A V \in (B, H, T, D_v)$ | $\text{Num} = \phi(Q) M \in (B, H, T, D_v)$ | 结合律先缩减 $T$ 维，再与 Query 点乘恢复序列长度 |
| **分母归一化项** | Softmax 沿最后一维天然和为 1 | $\text{Den} = \phi(Q) \big(\sum_{t} \phi(K)_t\big)^T \in (B, H, T, 1)$ | 补偿核映射缺少的分母归一化因子，防止长序列发散 |
| **最终输出 $O$** | $(B, H, T, D_v)$ | $\text{Num} / \text{Den} \in (B, H, T, D_v)$ | 形状完全一致，即插即用替换标准 Attention |

#### 2. 自回归因果递推形式与状态空间对偶性 (Recurrent Formulation & State-Space Duality)

在因果自回归生成模式下，线性注意力拥有**矩阵并行扫描（Training Parallel Scan）**与**流式递推（Inference Recurrent Step）**的双重等价形式：

##### (1) 逐时间步递推数学形式 (Step-by-Step Recurrence)
- **隐状态定义**：
  - 键值联想记忆矩阵：$S_t \in \mathbb{R}^{B \times H \times D_h \times D_v}$（初始状态 $S_0 = \mathbf{0}$）；
  - 键特征累加向量：$z_t \in \mathbb{R}^{B \times H \times D_h}$（初始状态 $z_0 = \mathbf{0}$，用于标量分母归一化）。
- **状态转移步 (State Update Step)**：
  $$S_t = S_{t-1} + \phi(k_t)^T v_t \in \mathbb{R}^{B \times H \times D_h \times D_v}$$
  $$z_t = z_{t-1} + \phi(k_t) \in \mathbb{R}^{B \times H \times D_h}$$
- **输出发射步 (Emission Step)**：
  $$o_t = \frac{\phi(q_t) S_t}{\phi(q_t) z_t^T} \in \mathbb{R}^{B \times H \times 1 \times D_v}$$

##### (2) 与线性状态空间模型 (Linear SSM) 的结构统一
上述递推形式与连续/离散线性状态空间模型（State-Space Model）严格同构：
$$S_t = \mathbf{A}_t S_{t-1} + \mathbf{B}_t x_t, \qquad y_t = \mathbf{C}_t S_t$$
在朴素线性注意力中：
- 状态转移矩阵 $\mathbf{A}_t = \mathbf{I}$（恒等矩阵，表示无衰减的累加记忆）；
- 写入投影 $\mathbf{B}_t = \phi(k_t)^T, x_t = v_t$（输入信号的外积投影）；
- 发射矩阵 $\mathbf{C}_t = \frac{\phi(q_t)}{\phi(q_t) z_t^T}$（基于当前 Query 的归一化读取探针）。

##### (3) 双重计算图对偶性 (Parallel Scan vs Recurrent Step)
- **训练阶段（并行关联扫描 Parallel Scan）**：
  由于外积加法满足结合律 $((a + b) + c = a + (b + c))$，全序列计算可转化为前缀和（`torch.cumsum`），单次并行 Forward 即可算完全部时间步，打满 GPU Tensor Core 矩阵乘法吞吐。
- **推理阶段（流式单步递推 Streaming Step）**：
  自回归 Decode 时退化为单步 RNN。每生成一个 Token，仅需将当前 Key 与 Value 的外积加进固定尺寸的状态矩阵 $S_{t-1}$ 中：
  - **显存占用**：恒定 $\mathcal{O}(D_h \cdot D_v) = \mathcal{O}(1)$（彻底消灭随序列长度线性膨胀的 KV Cache！）；
  - **单步耗时**：恒定 $\mathcal{O}(D_h \cdot D_v) = \mathcal{O}(1)$（生成第 1 个 Token 与生成第 100,000 个 Token 的延迟完全一致）。

---

#### 3. 表达能力差距：线性注意力 vs Full Softmax Attention (Expressive Capacity Gap)

尽管线性注意力在长文本上具备 $\mathcal{O}(T)$ 的渐进复杂度与 $\mathcal{O}(1)$ 的推理显存优势，但在学术与工业实践中，其多项能力指标（语言建模困惑度、少样本 In-Context Learning、长文本检索）始终与标准 Full Softmax Attention 存在显著鸿沟。这种表达能力差距源于四个根本性理论瓶颈：

##### 差距 1 · 锐度与极值聚焦能力缺失 (Lack of argmax Sharpness)
- **Softmax 的指数放大效应**：
  $$\operatorname{Softmax}(S)_{ij} = \frac{\exp(q_i k_j^T / \sqrt{D_h})}{\sum_{m=1}^T \exp(q_i k_m^T / \sqrt{D_h})}$$
  指数函数 $\exp(\cdot)$ 具有极端的非线性放大特性。当某一个 Key 与 Query 的匹配度比其他候选略微高出几个标准差时，Softmax 会将该位置的权重急剧推向 1，而将其余全部候选指数级压制为 0（逼近 $\operatorname{argmax}$）。这种“硬聚焦（Hard Attention）”能力使 Transformer 能够充当精确的**内容寻址指针（Pointer Network / Exact Addressing）**。
- **线性注意力的平滑弥散缺陷**：
  线性核 $\operatorname{sim}(q, k) = \phi(q) \phi(k)^T$ 仅为特征空间的普通内积，缺乏非线性极值拉大机制。其注意力权重在序列各位置间表现得平缓而弥散（Flat & Diffuse），无法形成尖锐的概率峰值，导致其难以在大量候选中精准“锁定”单个离散 Token。

##### 差距 2 · 有限状态容量与固定秩上界瓶颈 (Finite Memory Capacity & Rank Bottleneck)
- **Full Attention 的无限外置记忆**：
  Softmax Attention 的“记忆载体”是显式保存的历史 KV Cache 矩阵序列，有效表示空间维度随序列长度动态增长为 $\mathcal{O}(T \times D_h)$。其注意力分布图 $A = \operatorname{softmax}(Q K^T) \in \mathbb{R}^{T \times T}$ 理论上是**全秩（Full-Rank）**或极高秩的，能够独立保存并区分序列中所有 $T$ 个历史 Token 的正交语义。
- **线性注意力的秩坍缩限制**：
  线性注意力的因果递推将任意长历史强制压缩进单一矩阵 $S_t \in \mathbb{R}^{D_h \times D_v}$。根据线性代数基本定理，该状态矩阵的代数秩受限于维度瓶颈：
  $$\operatorname{rank}(S_t) \le \min(D_h, D_v)$$
  当序列长度 $T \gg D_h$（例如长文本 $T = 100,000, D_h = 128$）时，将十万个外积向量叠加在一个秩至多为 128 的有限子空间内，根据鸽巢原理（Pigeonhole Principle），历史特征必然发生剧烈的重叠混叠与信息坍缩。

##### 差距 3 · 注意力稀释与无门控遗忘 (Attention Dilution & SNR Collapse)
- **恒等累加导致信噪比崩溃**：
  标准线性注意力的状态转移是无门控累加：$S_t = S_{t-1} + \phi(k_t)^T v_t$。随着序列推进，$S_t$ 的整体范数随着 $t$ 线性膨胀。
- **早期重要信息的冲刷淹没**：
  若在 $t=10$ 处输入了关键实体（如密码或人名），在到达 $t=100,000$ 时，该实体的外积特征被后续 99,990 个普通背景 Token 的外积累加完全冲淡。由于缺少“遗忘门（Forget Gate）”与“擦除机制”，系统无法主动丢弃无用噪声，早期信号的信噪比（Signal-to-Noise Ratio, SNR）降至噪声基底以下。
- **Full Attention 的免疫性**：
  Full Attention 在每个时间步均由当前 Query 重新对全量历史做一次全景比较，完全不存在累加导致的信噪比稀释。

##### 差距 4 · 算法基准与上下文学习能力失效场景 (Algorithmic Failure Modes)
| 关键评估任务 | Full Softmax Attention 表现 | 纯线性注意力 (Plain Linear Attention) 表现 | 根本失效原因 |
|---|---|---|---|
| **大海捞针 (Needle in a Haystack)** | 接近 100% 检索准确率（绿屏） | 长度超过几千后准确率崩塌至接近随机（红屏） | 缺乏 Softmax 极值放大；有限秩导致单点事实被背景噪声稀释。 |
| **归纳头联想复制 (Induction Heads: $[A][B] \dots [A] \to [B]$)** | 完美实现高阶二阶寻址复制，支撑强大 ICL 能力 | 易发生严重键混淆（Key Collision），多步联想复制失败 | 键值对在外积空间中混合，无法在多查询联想回忆（MQAR）中精准解耦。 |
| **计数与形式语言 (Formal Language / Dyck 语言)** | 强电路表达力，精确追踪括号嵌套与状态跳转 | 无法精确识别深层嵌套结构 | 无门控线性循环系统计算复杂性被限制在低阶电路类（$TC^0$），表达能力弱于通用图灵机状态追踪。 |

##### 差距 5 · 现代改进架构演进脉络 (How Modern Models Bridge the Gap)
为克服线性注意力的上述理论缺陷，近年前沿架构演变出三条主流破局路径：
1. **引入时间/数据依赖衰减门控 (Decay Gates)**：
   如 **RetNet**（引入静态衰减 $\gamma^t$）与 **Mamba / RWKV**（引入输入自适应遗忘门 $g_t$）：$S_t = \operatorname{diag}(\alpha_t) S_{t-1} + \phi(k_t)^T v_t$，主动衰减过期历史，消除范数发散与无脑稀释；
2. **引入 Delta 规则在线擦除机制 (Delta Rule / Fast Weight Programmers)**：
   如 **DeltaNet** 与 **Gated DeltaNet**：在写入新记忆前先计算预测误差并正交擦除冲突旧记忆：
   $$S_t = S_{t-1} + \beta_t \big(v_t - S_{t-1} \phi(k_t)\big) \phi(k_t)^T$$
   使得有限容量的状态矩阵 $S_t$ 始终保持最优正交存储效率，MQAR 与长文本检索能力逼近 Softmax；
3. **混合架构 (Hybrid Architectures)**：
   如 **Jamba**、**Nemotron-4**：底层 80% 堆叠线性注意力/SSM 层实现低显存长上下文吞吐，顶层保留 20% 的 Full Softmax Attention 层负责高锐度精确检索与复杂逻辑推理，达成帕累托最优解。

---

#### 4. 特征映射 $\phi(x)$ 的数学约束
1. **为何不用恒等映射 (Identity，无激活)？**
   若允许负值，核函数退化为低秩线性分解，分母可能抵消为 0 导致数值发散，且破坏注意力权重的非负测度概率解释；
2. **为何不用 ReLU？**
   负值区间梯度完全归零（死区），造成大量 Key 向量被硬性抹去；
3. **业界通用标准采用 $\phi(x) = \operatorname{ELU}(x) + 1.0$**：
   严格满足 $\phi(x) > 0$，处处连续可微，保留微弱负向梯度。

</details>

#### Quick Coding：`linear_attention` 与流式单步递推

```python
def linear_attention(Q, K, V, causal=False):
    ...

def linear_attention_step(q_t, k_t, v_t, prev_S=None, prev_z=None):
    ...
```

<details open>
<summary><b>参考代码：从零手写 linear_attention（批量并行扫描与流式单步递推）</b></summary>

```python
import torch
import torch.nn.functional as F

def feature_map(x):
    # 严格非负特征映射，避免分母为 0 与梯度死区
    return F.elu(x) + 1.0

def linear_attention(Q, K, V, causal=False):
    """批量模式：支持训练期并行关联扫描 (Parallel Associative Scan)

    Q, K: (B, H, T, Dh) 或 (..., T, Dh)
    V:    (B, H, T, Dv) 或 (..., T, Dv)，通常 Dv = Dh
    """
    Qp, Kp = feature_map(Q), feature_map(K)

    if not causal:
        # 1. 键值联想记忆状态 M = Kp^T @ V
        # (..., T, Dh)^T @ (..., T, Dv) -> (..., Dh, Dv)
        kv_state = torch.einsum("... t d, ... t v -> ... d v", Kp, V)

        # 2. Key 特征沿序列维累加和 (用于分母归一化): (..., Dh)
        k_sum = Kp.sum(dim=-2)

        # 3. 分子: Qp @ M -> (..., T, Dv)
        num = torch.einsum("... t d, ... d v -> ... t v", Qp, kv_state)

        # 4. 分母: Qp @ k_sum^T -> (..., T, 1)
        den = torch.einsum("... t d, ... d -> ... t", Qp, k_sum).unsqueeze(-1)
        return num / den  # (..., T, Dv)

    # 因果模式 (Causal Prefix Scan)：
    # 1. 每个时间步的外积增量: (..., T, Dh, Dv)
    outer = torch.einsum("... t d, ... t v -> ... t d v", Kp, V)

    # 2. 沿序列维度的因果前缀累加和 (Prefix Sum)
    # S_cum: (..., T, Dh, Dv), z_cum: (..., T, Dh)
    S_cum = torch.cumsum(outer, dim=-3)
    z_cum = torch.cumsum(Kp, dim=-2)

    # 3. 分子: (..., T, Dh) @ (..., T, Dh, Dv) -> (..., T, Dv)
    num = torch.einsum("... t d, ... t d v -> ... t v", Qp, S_cum)

    # 4. 分母: (..., T, Dh) * (..., T, Dh) -> (..., T, 1)
    den = torch.einsum("... t d, ... t d -> ... t", Qp, z_cum).unsqueeze(-1)
    return num / den  # (..., T, Dv)


def linear_attention_step(q_t, k_t, v_t, prev_S=None, prev_z=None):
    """流式单步递推模式：用于自回归 Decode 阶段，单步显存与时间均为严格 O(1) 常数

    q_t, k_t: (B, H, Dh) 当前步单个 Query 与 Key
    v_t:      (B, H, Dv) 当前步单个 Value
    prev_S:   (B, H, Dh, Dv) 上一步的键值记忆状态矩阵 S_{t-1}，首步传 None
    prev_z:   (B, H, Dh)     上一步的 Key 累加向量 z_{t-1}，首步传 None
    返回:
        out_t:  (B, H, Dv) 当前时间步的输出
        curr_S: (B, H, Dh, Dv) 更新后的记忆状态 S_t
        curr_z: (B, H, Dh)     更新后的累加向量 z_t
    """
    qp_t, kp_t = feature_map(q_t), feature_map(k_t)

    # 状态初始化 (首步 S_0 = 0, z_0 = 0)
    if prev_S is None:
        prev_S = torch.zeros(
            *q_t.shape[:-1], q_t.shape[-1], v_t.shape[-1],
            device=q_t.device, dtype=q_t.dtype
        )
        prev_z = torch.zeros_like(q_t)

    # 1. 状态转移步：S_t = S_{t-1} + kp_t^T @ v_t
    delta_S = torch.einsum("... d, ... v -> ... d v", kp_t, v_t)
    curr_S = prev_S + delta_S
    curr_z = prev_z + kp_t

    # 2. 输出发射步：o_t = (qp_t @ S_t) / (qp_t @ z_t^T)
    num = torch.einsum("... d, ... d v -> ... v", qp_t, curr_S)
    den = torch.einsum("... d, ... d -> ...", qp_t, curr_z).unsqueeze(-1)
    out_t = num / den

    return out_t, curr_S, curr_z
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
