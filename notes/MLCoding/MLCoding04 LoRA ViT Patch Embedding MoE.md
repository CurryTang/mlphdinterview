# ML Coding 04 · 架构组件：LoRA 低秩微调、ViT Patch Embedding 与 Mixture of Experts

MLCoding01 已经拼出了一个能跑的 Transformer block。这一节不再从零搭主干，而是往主干上装三个真实场景会用到的架构扩展：怎么低成本微调、怎么把图像喂进同一套 attention 主干、怎么把 FFN 换成一堆专家来扩容而不等比例扩算力。

## 模块九：架构扩展

### Exercise 1 · LoRA Linear

全量微调一个大模型的成本主要来自优化器状态（Adam 的一阶、二阶矩和权重本身同量级），而不是前向计算本身。LoRA 的做法是冻结原始权重 `W`，只在旁边加一条低秩通路：

```text
h = x @ W^T + (alpha / r) * x @ (B @ A)^T
```

其中 `A` 形状 `(r, in_features)`，`B` 形状 `(out_features, r)`，`r` 远小于 `in_features` 和 `out_features`。可训练参数从 `in_features * out_features` 降到 `r * (in_features + out_features)`。

有一个初始化细节比公式本身更容易在面试里被追问：`B` 必须初始化成全零，`A` 用正态分布或 Kaiming 初始化。原因是训练刚开始时你不希望这条新增通路扰动一个已经预训练好的模型。如果 `B` 和 `A` 都是随机的，`B @ A` 一开始就是个随机噪声矩阵，直接加到 `W` 上会把模型的初始行为搅乱。把 `B` 设成零，`B @ A` 在第 0 步恒为零矩阵，`h` 在初始化时精确等于冻结基座的输出，微调是从"完全不变"开始，再让梯度慢慢把 `A`、`B` 推向有效的更新方向。

第二个常被追问的点是"能不能推理时零开销"。因为 `W' = W + (alpha / r) * B @ A` 和 `W` 形状完全一样，训练结束后可以直接把 `B @ A` 加回 `W` 里，之后推理就是普通的单个矩阵乘法，不会比原模型多任何计算或显存。但这个合并操作是一次性的：如果你需要在同一个基座上按请求切换不同任务的 LoRA adapter（比如一个 serving 系统同时服务几十个客户，每个客户一个 adapter），合并就意味着每次切换都要重新计算一次 `W + delta`，这在多租户场景下不划算，所以这类系统通常保持 `A`、`B` 独立不合并，前向时单独算这条低秩分支再加到基座输出上。

#### Quick Coding：`LoRALinear`

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, device=None, dtype=None):
        ...

    def forward(self, x):
        ...

    def merge(self):
        ...
```

<details>
<summary>参考答案</summary>

```python
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, device=None, dtype=None):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        # 冻结的基座权重，不参与梯度更新
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 低秩通路：A 正常初始化，B 初始化为全零
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = torch.einsum("...i,oi->...o", x, self.weight)
        delta = torch.einsum("...i,ri->...r", x, self.lora_A)
        delta = torch.einsum("...r,or->...o", delta, self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merge(self):
        # 训练结束后一次性合并，推理时退化为普通 Linear，零额外开销
        self.weight += self.scaling * (self.lora_B @ self.lora_A)
        self.lora_A.zero_()
        self.lora_B.zero_()
```

用 NumPy 验证过两件事：`lora_B` 全零时 `forward` 和纯 `x @ W^T` 逐元素相等（初始化即恒等映射），以及 `merge()` 前后对同一个 `x` 的输出完全一致（误差量级 `1e-15`，只是浮点舍入）。

</details>

### Exercise 2 · ViT Patch Embedding

Vision Transformer 想直接复用 NLP 那套 Transformer block，第一步就要把图像 `(B, C, H, W)` 变成一串 token。做法是把图像切成互不重叠的 `P × P` patch，每个 patch 展平成一个向量，再线性投影到 `d_model`：

```text
image (B, C, H, W)
  -> patches (B, num_patches, C*P*P)   # num_patches = (H/P) * (W/P)
  -> linear projection
  -> tokens (B, num_patches, d_model)
```

面试里常被要求写出两种等价实现,并解释为什么等价：

1. **显式 unfold + reshape + linear**：把图像切块、展平、过一个 `Linear(C*P*P, d_model)`。
2. **单个 `Conv2d(C, d_model, kernel_size=P, stride=P)`**：卷积核大小和步长都等于 patch size。

两者等价的关键在于 `kernel_size == stride == P`：卷积的每个输出位置对应输入里一个完全不重叠的 `P × P` 窗口，每个窗口内做的运算就是"把窗口展平后和卷积核展平后的向量做内积"，这正好是线性投影在做的事。如果 `stride < kernel_size`（有重叠）或者 `stride > kernel_size`（有间隙跳过的像素），这个等价关系就不成立了。ViT patch embedding 之所以能用卷积实现，本质上是因为它根本没有用到卷积的"滑动窗口共享感受野"特性,只是借用卷积算子在做一次分块矩阵乘法。

拼完 patch token 之后，通常还要做两件事：在序列最前面拼一个可学习的 `[CLS]` token 用于分类任务的全局表示，再给每个位置加一个可学习（或正弦）的位置编码，因为 patch 展平之后已经丢失了它们在图像里的二维空间位置信息。

#### Quick Coding：`PatchEmbedding`

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model, device=None, dtype=None):
        ...

    def forward(self, img):
        ...
```

<details>
<summary>参考答案</summary>

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model, device=None, dtype=None):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        # 用一个 stride == kernel_size 的卷积实现"分块 + 线性投影"
        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size,
            device=device, dtype=dtype,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model, device=device, dtype=dtype))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, d_model, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, img):
        B = img.shape[0]
        x = self.proj(img)                      # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)         # (B, num_patches, d_model)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)           # (B, num_patches + 1, d_model)
        return x + self.pos_embed
```

`proj` 换成"先手动 unfold 再过 `Linear`"是完全等价的写法，只是矩阵乘法被卷积算子代劳了。用 NumPy 把两条路径都实现了一遍，在同一份随机图像和权重上前向，输出逐元素相等（误差量级 `1e-15`）。

</details>

### Exercise 3 · Mixture of Experts

MoE 想解决的问题是：模型参数量和每个 token 的计算量能不能解耦。传统 dense FFN 里，参数越多，每个 token 过这层的计算量也线性增长。MoE 的做法是把一个 FFN 换成 `num_experts` 个独立的 FFN（每个都和 MLCoding01 的 `SwiGLU` 结构一样），但每个 token 只路由到其中 `k` 个（Mixtral 用 `k=2`），不激活的专家完全不参与这个 token 的前向。

```text
router_logits = Linear(d_model, num_experts)(x)      # (B, T, num_experts)
router_probs  = softmax(router_logits, dim=-1)
top_k_probs, top_k_idx = router_probs.topk(k, dim=-1)
top_k_probs  = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 重新归一化

output = sum_{i in top_k_idx} top_k_probs[i] * expert_i(x)
```

这样总参数量是 `num_experts` 倍于单个专家，但单个 token 的计算量只有大约 `k` 倍。如果 `num_experts=8, k=2`，就是用 `2/8 = 25%` 的计算量换来了 `8` 倍的参数容量,这也是"参数量很大但推理很便宜"这类 MoE 大模型的核心卖点。

一个容易被面试官追问的实现细节是重新归一化：选出 top-k 个专家后,它们的 router 概率之和通常小于 1（因为分母里还有没被选中的专家），如果直接拿原始概率做加权和，输出的尺度会系统性偏小，且这个偏移量随 token 而变化,难以在训练里通过学习率或初始化补偿；标准做法是把选中的 `k` 个概率重新归一化到和为 1 再做加权，用 NumPy 验证过，归一化前后对同一组专家输出算出的加权和不相等,归一化确实改变了输出的整体尺度。

最后一个常见追问是"如果某几个专家一直被选中,其余专家几乎没有梯度怎么办"，也就是 expert collapse。标准做法是加一个 load-balancing 辅助损失，鼓励 router 把 token 大致均匀地分配到所有专家上（常见形式是让"每个专家被选中的频率"和"每个专家收到的平均 router 概率"这两个分布的点积最小化），这个辅助损失以很小的权重加到主 loss 上，只起到"别偏科"的调节作用。

#### Quick Coding：`MixtureOfExperts`

```python
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, device=None, dtype=None):
        ...

    def forward(self, x):
        ...
```

<details>
<summary>参考答案</summary>

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, device=None, dtype=None):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False, device=device, dtype=dtype)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, device=device, dtype=dtype) for _ in range(num_experts)]
        )

    def forward(self, x):
        B, T, D = x.shape
        flat_x = x.reshape(-1, D)                              # (N, D), N = B*T

        router_logits = self.router(flat_x)                    # (N, num_experts)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_probs, top_k_idx = router_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 重新归一化

        out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            # 找出这个专家在 top_k_idx 里出现的 (token, slot) 位置
            token_idx, slot_idx = (top_k_idx == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            weight = top_k_probs[token_idx, slot_idx].unsqueeze(-1)
            out[token_idx] += weight * expert(flat_x[token_idx])

        return out.reshape(B, T, D)
```

用 NumPy 复现了 router 部分的数值逻辑：对随机 logits 做 softmax、取 top-2、重新归一化后概率和为 1（未归一化前只有约 0.68），并确认归一化前后加权求和的结果确实不同。

</details>

#### 本模块易错点

- LoRA 的 `B` 必须零初始化，`A` 才是承担"探索方向"的那一侧；反过来初始化会让训练一开始就扰动基座权重。
- Patch embedding 的卷积等价性只在 `stride == kernel_size`（无重叠、无跳过）时成立，别把它和普通卷积的感受野共享混为一谈。
- MoE 的 top-k 概率必须重新归一化，否则输出尺度会随路由结果系统性漂移。
