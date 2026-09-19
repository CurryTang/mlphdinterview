# ML Coding 02 · 基础算子补完：GELU、BatchNorm、Kaiming Init、Dropout、Conv2d、线性回归、梯度累积

MLCoding01 已经搭完了一条从 tokenizer 到训练循环的完整路线，但那条路线只挑了它需要的算子（RMSNorm、SwiGLU、AdamW、Cosine LR）。还有一批同样高频的"手写 X"面试题没有覆盖，这一篇把它们补齐，仍然按 PyTorch 面试的口径来写：不用 `torch.nn` 里现成的层，自己把 forward（以及必要的语义）实现出来。

> 说明：题目覆盖范围参考自开源练习平台 [TorchCode](https://github.com/duoan/TorchCode)（PyTorch 面试真题练习 + 自动判题），本笔记的讲解与代码均为独立编写，不摘录其源码。

## 模块九：基础算子补完

### Exercise 1 · ReLU

最简单的题往往用来确认基本功：`relu(x) = max(x, 0)`，逐元素操作，没有可学习参数。真正会被追问的是 `x=0` 处的次梯度：PyTorch 的约定是把它记成 0（即 `grad = (x > 0)`，而不是 `x >= 0`），这个约定不影响训练效果,因为浮点数恰好落在 0 的概率是零测度事件，但面试时说不清楚会显得没有推导过反向传播。

dying ReLU 问题(负区间梯度恒为 0,一旦某个神经元的输入长期为负,它就再也学不到东西)是它最大的弱点,但因为计算成本几乎为零、正区间不会梯度饱和,它仍然是默认选项。

#### Quick Coding：`relu`

```python
def relu(x: torch.Tensor) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x, torch.zeros_like(x))
```

等价写法 `torch.clamp(x, min=0)` 或 `x * (x > 0)` 都可以，但手写反向传播时要清楚：`d relu/dx = 1 if x > 0 else 0`，`x=0` 处按 0 处理。

```python
x = torch.tensor([-2.0, 0.0, 3.0])
assert torch.equal(relu(x), torch.tensor([0.0, 0.0, 3.0]))
```

</details>

### Exercise 2 · LayerNorm

LayerNorm 和 MLCoding01 已经实现的 RMSNorm 经常被放在一起考,面试官想看的是你能不能说清楚两者到底差在哪一步。LayerNorm 在归一化之前先减掉均值,再除以标准差；RMSNorm 跳过减均值这一步,只用均方根做缩放。差的这一步看起来很小,但它决定了两者对"整体偏移"的响应完全不同:给输入整体加一个常数,LayerNorm 的输出不变(减均值抵消了偏移),RMSNorm 的输出会跟着变。这也是为什么 LLaMA / Mistral 这类模型敢用 RMSNorm。它们假设 residual stream 的问题主要是尺度爆炸而不是均值漂移,少算一次均值可以省一点算子开销。

| | LayerNorm | RMSNorm |
| --- | --- | --- |
| 统计量 | 均值 + 方差 | 只有均方 |
| 公式 | `(x-mean)/sqrt(var+eps)*gamma+beta` | `x/sqrt(mean(x^2)+eps)*weight` |
| 对常数偏移是否不变 | 是 | 否 |
| 可学习参数 | `gamma`(缩放)、`beta`(偏移) | 只有缩放 `weight` |
| 典型代表 | BERT、GPT-2、原始 Transformer | LLaMA、Mistral、Qwen |

#### Quick Coding：`layer_norm`

```python
def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * gamma + beta
```

数值验证(NumPy,替代不可用的 torch)：给输入整体加一个常数 100,LayerNorm 的输出完全不变,RMSNorm 的输出发生明显偏移，这正是上表最后一行的来源。

```python
import numpy as np
x = np.array([[1.0, 2.0, 3.0, 4.0]])
def ln(x): 
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5)
def rms(x):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-5)
assert np.abs(ln(x) - ln(x + 100.0)).max() < 1e-8
assert np.abs(rms(x) - rms(x + 100.0)).max() > 0.5
```

</details>

<details class="technical-deep-dive">
<summary><span class="deep-dive-badge">架构深度解析</span><span class="deep-dive-title">现代 LLM 架构中的归一化演进与系统设计 (Pre-RMSNorm · Q-K Norm · 零偏置)</span></summary>
<div class="deep-dive-content">

#### 1. 现代前沿开源大模型 Normalization 方案对照

| 模型系列 | 主干归一化 (Backbone Norm) | Q-K Normalization | 辅助/层内 Normalization | 偏置设计 (Bias Policy) | 核心设计诉求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LLaMA 2 / 3** | Pre-RMSNorm | 无 | 无 | 全层 `bias=False` | 标准预归一化，依赖缩放因子与注意力截断 |
| **Mistral / Mixtral** | Pre-RMSNorm | 无 | 无 | 全层 `bias=False` | 高算子吞吐，极简主干 |
| **Qwen 2 / 2.5** | Pre-RMSNorm | **Per-Head RMSNorm** | 无 | 全层 `bias=False` | 阻断 128k 超长序列下的 Attention Logit 爆炸 |
| **Kimi k1.5** | Pre-RMSNorm | **Per-Head RMSNorm** | 无 | 全层 `bias=False` | 稳定 200k+ 超长上下文预训练与高方差强化学习探索 |
| **GLM-4** | Pre-RMSNorm | **Per-Head RMSNorm** | 从早期 DeepNorm 演进至 Pre-RMSNorm | 全层 `bias=False` | 兼顾多语言海量训练稳定性与长文本推理 |
| **Gemma 2** | Pre-RMSNorm | **Per-Head RMSNorm** | **Post-Attention Norm & Post-FFN Norm (Dual-Norm)** | 全层 `bias=False` | 双重归一化抑制极深网络 (27B) 残差流方差发散 |
| **DeepSeek V2 / V3** | Pre-RMSNorm | **Decoupled MLA Norm** | 压缩潜在表示向量与解耦 RoPE 处归一化 | 全层 `bias=False` | 适配低秩 KV 压缩与极大规模 MoE 路由稳定 |

---

#### 2. 演进动力一：全行业以 RMSNorm 替代 LayerNorm 的统计与硬件动因

1. **残差动力学与统计假设 (Residual Dynamics)**：
   标准 LayerNorm 对隐藏维度向量 $x \in \mathbb{R}^d$ 的公式为：
   $$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2 + \epsilon}, \quad y = \frac{x - \mu}{\sigma} \odot \gamma + \beta$$
   RMSNorm (Root Mean Square Normalization) 舍弃了均值中心化：
   $$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}, \quad y = \frac{x}{\text{RMS}(x)} \odot \gamma$$
   在 60~128 层的 Pre-Norm Transformer 中，残差流随着层数加深不断累积，其激活能量（$\ell_2$ 范数）呈线性递增趋势。然而，由于模型广泛采用零均值权重初始化（如 He / Xavier 正态分布）以及对称非线性激活（如 SwiGLU / GeGLU），激活向量在隐藏维度上的真实均值 $\mu$ 始终在 $0$ 附近微小振荡（$\mu \approx 0$）。因此：
   $$\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2 \approx \frac{1}{d}\sum_{i=1}^d x_i^2 = \text{RMS}(x)^2$$
   LayerNorm 引入的平移不变性（Shift Invariance）在实际残差流中几乎不产生统计增益，真正稳定梯度的核心机制是能量缩放约束（Scale Invariance）。

2. **显存受限算子优化 (Memory-Bound Kernel Efficiency)**：
   Normalization 是典型的 Memory-bound（计算访存比极低）算子。
   - LayerNorm 需要维护均值 $\mu$ 与方差 $\sigma^2$，通常需要两趟全局规约（Two-pass Reduction），或者在单个 CUDA 线程块内使用复杂的 Welford 算法维护在线统计量，增加片上 SRAM 寄存器占用；
   - RMSNorm 仅需执行一趟单一的平方和规约（Single-pass Reduction），寄存器使用量减少约 30%，在 Triton 或 CUDA 算子编写中极易与上一层的残差加法（Residual Addition）进行算子融合（Kernel Fusion），在现代 GPU（如 A100 / H100）上可降低 30%~50% 的访存耗时。

---

#### 3. 演进动力二：Q-K Norm 的理论机制与数值防线 (Qwen / Kimi / GLM 的共同选择)

1. **核心痛点：Attention Logit 爆炸与 Softmax 熵坍塌 (Entropy Collapse)**：
   在标准的多头注意力计算中：
   $$S = \frac{Q K^T}{\sqrt{d_k}}, \quad A = \text{softmax}(S)$$
   随着模型深度增加、训练进入数万亿 token 阶段、上下文长度扩展至 32k~128k+，或者在强化学习（RLVR / PPO / GRPO）高探索方差场景下：
   - 未经尺度钳制的 $Q$ 与 $K$ 投影向量范数随梯度更新不断膨胀，$\Vert q\Vert_2 \cdot \Vert k\Vert_2 \gg d_k$；
   - 注意力得分 $S_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}}$ 的峰值迅速突破 $80 \sim 100+$；
   - **数值下溢/溢出**：在 FP16/BF16 混合精度下，$\exp(S_{ij} - \max S)$ 会遭遇剧烈截断或溢出；
   - **熵坍塌与梯度弥散**：Softmax 转化为近乎完全的 One-Hot 尖刺，分布熵趋近于 0。此时 Softmax 导数：
     $$\frac{\partial A_{im}}{\partial S_{ij}} = A_{im} (\delta_{jm} - A_{ij})$$
     当 $A_{ii} \to 1$ 且其他 $A_{ij} \to 0$ 时，梯度全量归零。模型注意力头永久“失活”或引发突发性的 Loss Spike，导致超大模型预训练中途报废。

2. **柯西-施瓦茨数学硬约束 (Cauchy-Schwarz Bounded Logits)**：
   在每个注意力头内部对 $Q$ 和 $K$ 分别执行 RMSNorm（不带学习参数或仅带轻量标量）：
   $$\hat{q} = \frac{q}{\text{RMS}(q)}, \quad \hat{k} = \frac{k}{\text{RMS}(k)}$$
   此时向量在特征维度的二范数被严格约束在 $\sqrt{d_k}$ 附近：
   $$\Vert \hat{q} \Vert_2 = \sqrt{d_k}, \quad \Vert \hat{k} \Vert_2 = \sqrt{d_k}$$
   根据柯西-施瓦茨不等式（Cauchy-Schwarz Inequality），点积的绝对值上限被严格锁死：
   $$|\hat{q}^T \hat{k}| \le \Vert \hat{q} \Vert_2 \cdot \Vert \hat{k} \Vert_2 = d_k$$
   代入 Scaled Dot-Product 缩放因子：
   $$\left| \frac{\hat{q}^T \hat{k}}{\sqrt{d_k}} \right| \le \frac{d_k}{\sqrt{d_k}} = \sqrt{d_k}$$
   对于主流 Head 维度 $d_k = 128$，最大理论 Logit 被数学硬性限制在 $\sqrt{128} \approx 11.31$ 以内，彻底在数学底层杜绝了 Logit 爆炸与数值溢出的可能。

---

#### 4. 演进动力三：现代 LLM 全量去除 Bias（零偏置设计）的工程权衡

主流模型（LLaMA 3, Qwen 2.5, GLM-4）在全部 Linear 和 Norm 层中默认将 `bias` 设为 `False`：
1. **防止多层残差线性积分漂移 (Residual Drift)**：在 80+ 层网络中，若残差流分支存在微小的常数偏置向量 $b$，多层累加会在特定隐藏维度上形成单调累积漂移，破坏层间输入的零均值对称性；
2. **优化器显存节约 (Optimizer Memory Footprint)**：AdamW 需要为每个参数维护 32 位的 First Moment 与 Second Moment（每参数 8 bytes）。去除全模型成百上千个微小 Bias 张量，显著降低了参数元数据开销与显存碎片；
3. **硬件量化亲和性 (Symmetric Quantization Friendly)**：在 INT8 / FP8 PTQ（训练后量化）中，对称量化假设数据均值为 0，仅需一个标量缩放因子即可对齐 Tensor Core 计算。去除 Bias 保证了隐藏状态在原点对称，无需引入复杂的非零偏移补偿（Zero-Point Offset）。

---

#### 5. 演进动力四：Gemma 2 的 Dual-Norm 与 GLM-130B 的 DeepNorm 演进

1. **Gemma 2 的 Dual-Norm (Pre-Norm + Post-Norm)**：
   Gemma 2 发现，随着网络宽度扩展到 27B，即使采用 Pre-RMSNorm，残差分支在经过复杂 Attention 与 FFN 计算后输出的方差依然存在相对主干残差流逐步放大的现象。因此它在 Attention/FFN 子层计算完毕后、加入残差流之前，额外插入了一道 **Post-Norm**：
   $$x_{l+1} = x_l + \text{RMSNorm}_{\text{post}}(\text{Sublayer}(\text{RMSNorm}_{\text{pre}}(x_l)))$$
   通过输入与输出双重约束，锁定了残差分支的输出能量幅度。
2. **GLM 架构路线的收敛**：
   在早期 GLM-130B 阶段，针对 1300 亿密集参数训练不稳的问题，团队提出了 DeepNorm（理论推导残差分支系数 $\alpha$ 缩放与初始化方差控制）。但随着 BF16 硬件原生支持以及 **Pre-RMSNorm + Q-K Norm** 范式的确立，GLM-4 最终全面收敛至当下的主流标准方案。

---

#### 6. 模块实现与数值稳定性验证

以下给出现代大模型标准的 Q-K Norm 注意力层实现，并使用 NumPy 模拟 50 倍输入能量漂移下的稳定性断言：

```python
import numpy as np

def np_rmsnorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm 单维规约实现"""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms

def verify_qknorm_stability():
    np.random.seed(42)
    B, H, S, D = 2, 4, 16, 64  # head_dim = 64, sqrt(D) = 8.0
    
    # 模拟长序列训练中激活值模长膨胀 50 倍的极端漂移场景
    q_drift = np.random.randn(B, H, S, D) * 50.0
    k_drift = np.random.randn(B, H, S, D) * 50.0
    scale = 1.0 / np.sqrt(D)

    # 1. 未加 Q-K Norm: Logit 剧烈爆炸
    logits_unnorm = np.matmul(q_drift, k_drift.swapaxes(-1, -2)) * scale
    max_logit_unnorm = np.max(np.abs(logits_unnorm))
    assert max_logit_unnorm > 500.0, "未规范化时 Logit 应发生严重尺度膨胀"

    # 2. 施加 Q-K Norm: 理论上界 sqrt(D) = 8.0
    q_norm = np_rmsnorm(q_drift)
    k_norm = np_rmsnorm(k_drift)
    logits_norm = np.matmul(q_norm, k_norm.swapaxes(-1, -2)) * scale
    max_logit_norm = np.max(np.abs(logits_norm))
    theoretical_bound = np.sqrt(D)
    assert max_logit_norm <= theoretical_bound + 1e-4, "Q-K Norm 必须严格满足柯西-施瓦茨理论上界"

    # 3. Softmax 熵检验 (防止极端尖刺 One-Hot)
    def calc_entropy(logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(shifted)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        return -np.sum(probs * np.log(probs + 1e-12), axis=-1).mean()

    ent_unnorm = calc_entropy(logits_unnorm)
    ent_norm = calc_entropy(logits_norm)
    assert ent_unnorm < 0.05, "未规范化的 Softmax 发生熵坍塌 (退化为 One-Hot)"
    assert ent_norm > 1.5, "Q-K Norm 成功维持了健康的注意力信息分布熵"

if __name__ == "__main__":
    verify_qknorm_stability()
    print("Q-K Norm 数值稳定性与柯西-施瓦茨上界断言全部通过。")
```

</div>
</details>

### Exercise 3 · GELU

GELU 的直觉是"用输入自身的分位数去加权自己":`GELU(x) = x * Φ(x)`,其中 `Φ` 是标准正态分布的累积分布函数。展开成误差函数就是精确形式：

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

GPT-2 和 BERT 用的其实是一个 tanh 近似(算 `erf` 比算 `tanh` 贵):

```text
GELU_tanh(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

两者不是同一个函数,数值上有肉眼可见的差距(下面验证在 `x≈2.7` 附近误差最大,约 4.7e-4),面试时如果被问"这两个公式一样吗",正确答案是"不完全一样,是近似关系"。GELU 相比 ReLU 的关键差异是处处光滑、负区间也有非零梯度,不会出现 dying ReLU,代价是计算更贵。

#### Quick Coding：`gelu`

```python
def gelu(x: torch.Tensor) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
import math

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
```

数值验证(NumPy)：

```python
import numpy as np
from scipy.special import erf

def gelu_exact(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

x = np.linspace(-6, 6, 100001)
max_err = np.abs(gelu_exact(x) - gelu_tanh(x)).max()
assert 1e-5 < max_err < 1e-3   # 接近但不相等
assert abs(gelu_exact(np.array([0.0]))[0]) < 1e-12   # GELU(0) = 0
```

</details>

### Exercise 4 · BatchNorm

BatchNorm 和 LayerNorm 的分歧点不是公式(都是"减均值除标准差再仿射"),而是统计量在哪个轴上算。LayerNorm 对每个样本自己的特征维求统计量,样本之间互不影响；BatchNorm 对同一个特征、跨整个 batch 求统计量,所以一个样本的输出会依赖 batch 里其他样本是谁。这个依赖关系带来两个后果:一是训练和推理必须用不同的统计量(推理时不能依赖当前 batch,要用训练过程中滑动平均出来的 `running_mean` / `running_var`),二是 batch size 很小或者变长序列里有大量 padding 时,batch 统计量会不稳定甚至被 padding 污染。这正是 Transformer 几乎全部转向 LayerNorm / RMSNorm、而 CNN 里 BatchNorm 依然是标配的原因：图像任务的 batch size 通常足够大,而且没有 padding 语义。

#### Quick Coding：`batch_norm`

```python
def batch_norm(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    running_mean: torch.Tensor, running_var: torch.Tensor,
    eps: float = 1e-5, momentum: float = 0.1, training: bool = True,
) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
def batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True):
    # x: (B, C) 或 (B, C, ...)，统计量沿除 C 以外的所有维度求
    reduce_dims = [0] + list(range(2, x.dim()))
    if training:
        batch_mean = x.mean(dim=reduce_dims, keepdim=False)
        batch_var = x.var(dim=reduce_dims, unbiased=False, keepdim=False)
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
            running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
        mean, var = batch_mean, batch_var
    else:
        mean, var = running_mean, running_var

    shape = [1, -1] + [1] * (x.dim() - 2)
    x_norm = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + eps)
    return x_norm * gamma.view(shape) + beta.view(shape)
```

训练模式下用当前 batch 的均值方差做归一化,同时用滑动平均更新 `running_mean` / `running_var`；推理模式直接用滑动平均值,不再依赖当前输入是谁，这是它和 LayerNorm 最本质的行为差异。

</details>

<details class="technical-deep-dive">
<summary><span class="deep-dive-badge">理论深度解析</span><span class="deep-dive-title">正则化与归一化的理论机制演进 —— 从经典深度学习到大模型时代 (Dropout · BatchNorm · LayerNorm · RMSNorm)</span></summary>
<div class="deep-dive-content">

### 1. 正则化与归一化全景对比矩阵

归一化（Normalization）与正则化（Regularization）是深度神经网络优化动力学与泛化理论的核心组件。下表系统对比了四种主流算子的数学定义、统计量缩减维度、统计随机性及在现代大模型中的定位：

| 算子 | 缩减维度 (Reduction Axes) | 统计性质与随机性 | 训练 / 推理行为一致性 | 核心设计初衷与主要机制 | 现代前沿 LLM 中的演化状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dropout** | 无缩减（逐元素独立采样） | 显式注入 Bernoulli 掩码随机噪声 | **不一致**：训练期随机置零并缩放；推理期恒等映射（Identity） | 破坏神经元共适应性；等价于 $2^D$ 个子网络集成 | **预训练全面弃用 (`dropout=0.0`)**：吞吐瓶颈与样本效率损耗 |
| **BatchNorm** | 跨 Batch 与空间/序列维 $(B, H, W)$ 或 $(B, L)$ | 隐式引入 Mini-batch 统计量采样噪声 | **不一致**：训练期使用当前 batch 统计量；推理期使用滑动平均（EMA） | 改善优化曲面平滑性（Lipschitz 连续性）；缓解内层尺度漂移 | **彻底淘汰**：变长 Padding 污染、自回归单 Token 推理不兼容、TP 通信开销高昂 |
| **LayerNorm** | 沿单样本所有特征维 $(C, H, W)$ 或 $(D)$ | 严格确定性计算，单样本完全解耦 | **完全一致**：训练与推理公式与行为完全相同，无全局状态依赖 | 消除不同样本在特征维度的尺度与偏移差异；稳定残差更新 | **被 RMSNorm 取代**：中心化减均值增益极低，双重 Reduce 访存开销大 |
| **RMSNorm** | 沿单样本特征维计算均方根 $(D)$ | 严格确定性计算，单样本完全解耦 | **完全一致**：训练与推理公式完全一致，仅需可学习缩放参数 $\gamma$ | 省略中心化减均值操作，单次 Reduce 访存降低，维持尺度缩放稳定性 | **绝对工业基准**：LLaMA, Qwen, DeepSeek, Mistral, Gemma 默认标配 |

---

### 2. 与“过拟合”和“泛化正则化”的理论本质剖析

#### (1) Dropout 的显式正则化数学本质

Dropout（Srivastava et al., 2014）通过在前向传播中以概率 $p$ 随机将神经元激活值置零，并使用反向缩放因子 $\frac{1}{1-p}$ 保持输出期望不变（Inverted Dropout）：

$$\mathbf{y} = \frac{\mathbf{m} \odot \mathbf{x}}{1 - p}, \quad m_i \sim \text{Bernoulli}(1 - p)$$

理论上其正则化效应源于三个维度：

1. **共享权重的极端模型集成（Model Ensemble of $2^D$ Sub-networks）**：
   对于具有 $D$ 个隐藏单元的单层网络，Dropout 在每次迭代中等效于从 $2^D$ 个共享参数的稀疏子网络中随机均匀采样一个进行梯度更新。推理阶段保留全量权重的前向计算，在数学期望上逼近所有子网络预测分布的几何平均（Geometric Mean Ensemble）。
2. **破坏神经元间的特征共适应（Breaking Co-adaptation）**：
   在标准前向传播中，某些神经元容易“搭便车”（Free-riding），即过度依赖相邻特定神经元的强响应而拟合训练集特异的虚假关联。Dropout 强制每个隐藏单元在任意同伴缺席的情况下依然能够独立提取有判别力的正交特征。
3. **贝叶斯近似与自适应权重衰减（Bayesian Approximation & Adaptive $L_2$ Regularization）**：
   - Gal & Ghahramani (2016) 证明：带有 Dropout 的深度神经网络在变分推断视角下严格等价于深度高斯过程（Deep Gaussian Process）的近似贝叶斯推断，在测试期多次采样（MC Dropout）可量化认知不确定性（Epistemic Uncertainty）。
   - Wager et al. (2013) 证明：在广义线性模型中，Dropout 引入的一阶噪声在泰勒展开下等效于带有自适应对角矩阵的数据依赖型 $L_2$ 正则项，对激活方差较大的不稳定特征施加更重惩罚。

#### (2) BatchNorm 的隐式正则化与优化曲面革命

BatchNorm（Ioffe & Szegedy, 2015）对当前 mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ 沿 batch 轴执行均值方差归一化：

$$\mu_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^m x_i, \quad \sigma_{\mathcal{B}}^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2, \quad \hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

1. **Mini-batch 采样抖动的隐式正则化（Implicit Batch Noise Regularization）**：
   由于 $\mu_{\mathcal{B}}$ 和 $\sigma_{\mathcal{B}}^2$ 取决于当前 mini-batch 中随机采样的其他样本，单个输入 $x_i$ 的归一化输出必然受到同批其他样本的扰动。这种扰动在数学上等效于在神经元激活值上注入了与 batch 统计量相关的自适应随机噪声：

   $$\hat{x}_i(x_1, \dots, x_m) = x_i \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} - \frac{\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

   此随机噪声在整个训练周期内平滑了决策边界，具有类似数据增强和 Dropout 的防过拟合作用。在卷积神经网络（CNN）中，引入 BatchNorm 后通常可以大幅调低甚至完全取消 Dropout。
2. **ICS 假说 vs. 优化曲面平滑性（Santurkar et al., NeurIPS 2018）**：
   - 原作者最初提出的“内部协变量偏移（Internal Covariate Shift, ICS）”假说认为 BN 的成功源于稳定了隐藏层激活值的边缘分布。
   - Santurkar 等人通过严格实证推翻了这一假说：在 BN 层之后人为注入高方差、非平稳的随机分布偏移噪声，网络依然维持极高训练速度和收敛精度。
   - BN 的真正核心价值在于**根本性改善了损失曲面的平滑度（Optimization Landscape Smoothing）**：BN 显著降低了损失函数的 Lipschitz 常数 $L$ 以及梯度的 Lipschitz 常数 $\beta$：

     $$\|\nabla \mathcal{L}(\mathbf{w}_1) - \nabla \mathcal{L}(\mathbf{w}_2)\| \le \beta \|\mathbf{w}_1 - \mathbf{w}_2\|$$

     梯度的方差大幅缩减，Hessian 矩阵的最大特征值与最小特征值之比（条件数）显著改善，使得优化轨迹避开了病态曲率峡谷，允许使用数十倍的大学习率快速收敛。
3. **权重尺度不变性与有效学习率机制（Scale Invariance & Effective Learning Rate）**：
   归一化层使得网络对权重的绝对模长缩放具有不变性：对任意标量 $\alpha > 0$，有 $\text{Norm}(\alpha \mathbf{W} \mathbf{x}) = \text{Norm}(\mathbf{W} \mathbf{x})$。由多元微积分链式法则可知，其关于权重的梯度满足严格反比缩放：

   $$\nabla_{\alpha \mathbf{W}} \mathcal{L} = \frac{1}{\alpha} \nabla_{\mathbf{W}} \mathcal{L}$$

   当搭配权重衰减（Weight Decay, 权重系数衰减率 $\lambda$）优化器时，梯度更新为 $\mathbf{W}_{t+1} = (1 - \eta \lambda) \mathbf{W}_t - \eta \nabla_{\mathbf{W}} \mathcal{L}$。权重衰减持续压减 $\|\mathbf{W}\|_2$，而 $\|\mathbf{W}\|_2$ 的收缩反向推高了权重的相对更新步长：

   $$\frac{\|\Delta \mathbf{W}_t\|}{\|\mathbf{W}_t\|} \approx \frac{\eta \|\nabla_{\mathbf{W}} \mathcal{L}\|}{\|\mathbf{W}_t\|} \propto \frac{\eta}{\|\mathbf{W}_t\|^2}$$

   因此，在归一化网络中，**权重衰减的主要功能不再是经典意义上的参数空间容量惩罚，而是通过调控权重范数来动态自适应调整“有效学习率” $\eta_{\text{eff}} = \frac{\eta}{\|\mathbf{W}\|^2}$**（van Laarhoven, 2017; Hoffer et al., 2018）。

#### (3) LayerNorm / RMSNorm 的正则化能力定位

LayerNorm 与 RMSNorm 严格在单样本特征内部计算统计量，**样本与样本之间在统计上严格独立，不包含任何来自同批其他样本的随机采样噪声**。
因此，LayerNorm 与 RMSNorm **几乎没有隐式正则化抗过拟合的能力**。它们的核心定位是**纯粹的数值与优化稳定性算子（Optimization Stabilizers）**：通过约束深层残差累加所导致的信号幅值爆炸，保障梯度在数十乃至数百层网络中的健康反向回传。

---

### 3. 大模型（LLM）时代的范式转移

现代大语言模型（如 LLaMA-1/2/3, Qwen-2/2.5, DeepSeek-V2/V3, Mistral, Gemma）在架构设计上发生了彻底转向：**完全剔除 Dropout，彻底弃用 BatchNorm，并由 LayerNorm 全面升级为 Pre-RMSNorm + Q-K Norm**。

#### (1) 为什么现代 LLM 预训练全面弃用 Dropout (`dropout = 0.0`)？

1. **核心矛盾由“过拟合”转向“欠拟合与样本效率（Sample Efficiency）”**：
   - 经典深度学习（如 ImageNet 图像分类）面临的典型环境是数千万参数拟合百万量级图像，反复训练数十个 Epoch，模型容量严重过剩，过拟合是首要威胁。
   - 现代前沿 LLM 预训练在 10T ~ 15T+ Tokens 的海量文本语料上通常仅执行**单 Epoch（Single-Pass）**训练，模型在整个生命周期内极少重复看到同一条数据。在 Chinchilla Scaling Law 的指引下，网络处于极度严重的“欠拟合”与“算力/样本受限”状态。
   - 此时引入 Dropout 会随机阻断 10%~20% 的神经元通路，直接折损模型的有效表征容量与每步梯度更新的信息吞吐量，严重拖慢 Loss 随训练 Token 数的收敛速率。
2. **显存占用与显存带宽瓶颈（Memory Footprint & IO Overhead）**：
   - 在自动微分反向传播时，网络必须在显存中保留正向传播生成的随机 Bernoulli 掩码矩阵（Bitmask），这直接扩大了激活值显存（Activation Memory）。
   - 在高并发分布式训练中，现代硬件瓶颈主要在于 HBM（显存）到 SRAM 之间的内存带宽（Memory-Bound）。诸如 FlashAttention、Fused Linear 等极致的算子融合技术依赖连续的流水线内联；Dropout 需要在内联核函数中维护伪随机数生成器（PRNG）状态并执行访存读写，严重拖慢计算流水的吞吐。
3. **自回归推理与强化学习（RLHF / RLVR）的确定性基准**：
   - 在自回归生成与 KV Cache 机制中，每一步依赖历史 Token 的精确表征。训练期若引入 Dropout，会阻碍特定注意力模式的沉淀（如 Attention Sinks）。
   - 在后训练对齐（Post-training Alignment，如 PPO、DPO、GRPO）中，策略梯度的方差极为敏感。Dropout 带来的输出分布抖动会污染 Advantage 函数估计，破坏策略模型的收敛稳定性。

#### (2) 为什么 BatchNorm 在大模型与 Transformer 中彻底绝迹？

1. **变长序列与 Padding 语义污染**：
   自然语言文本序列长度天然不均匀。一个 Mini-batch 中往往存在长短不一的句子并通过 Padding Token（通常为 0）补齐。若沿 Batch 轴求均值与方差，大量的 Padding 填充值会严重拉低真实语义 Token 的均值并扭曲方差；若动态屏蔽 Padding，则每个特征通道参与统计的有效 Token 数量不一致，导致统计量剧烈抖动。
2. **自回归单 Token 逐字推理解码（Autoregressive Token-by-Token Generation）**：
   在在线 Serving 阶段，推理 Batch Size 随着用户并发动态剧烈变化（常常低至 $B=1$），且解码每次仅生成一个 Token。单个 Token 在空间和时间上均不具备计算稳定统计量的样本基础。而 BatchNorm 依赖的 `running_mean` 与 `running_var` 在遇到与训练集领域稍有偏差的分布时，推理精度容易发生断崖式下跌。
3. **分布式张量/流水线并行通信墙（Cross-GPU Communication Wall）**：
   现代 LLM 训练在千卡集群上采用张量并行（Tensor Parallelism, TP）与流水线并行（Pipeline Parallelism, PP）。BatchNorm 跨 Batch 的规约操作要求在各卡之间进行全局 AllReduce 同步统计量。将一个原本仅需在单卡片上缓存（SRAM）完成的局部算子升级为跨节点通信阻塞，通信开销是分布式训练无法承受的灾难。

#### (3) 为什么现代 LLM 统一采用 Pre-RMSNorm 架构？

1. **Pre-Norm 梯度高速公路（Gradient Highway）**：
   Post-Norm 将归一化置于残差相加之后：$\mathbf{x}_{l+1} = \text{Norm}(\mathbf{x}_l + \text{SubLayer}(\mathbf{x}_l))$。深层网络中反向传播梯度流经每一层 Norm 时会被连续缩小，导致深层网络必须依赖极其脆弱的 Learning Rate Warmup 才能勉强启动。Pre-Norm 则将 Norm 移至子层内部：$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{SubLayer}(\text{Norm}(\mathbf{x}_l))$，主干残差保留一条完全畅通无阻的恒等映射通道，反向传播梯度可无损穿透数百层深网。
2. **中心化减均值的冗余性与单次 Reduce 访存加速**：
   RMSNorm（Zhang & Sennrich, 2019）发现：在深度神经网络的高维潜在空间中，激活向量各分量在统计上天然围绕 0 对称分布。均值偏移对网络泛化的贡献微乎其微，归一化的核心收益完全来自于**均方根缩放（Root Mean Square Scaling）**。
   - LayerNorm 需要两遍规约遍历（Two-pass Reductions）：第一遍计算均值 $\mu$，第二遍基于差值计算方差 $\sigma^2$。
   - RMSNorm 仅需单遍规约（One-pass Reduction）：直接求平方和的均值 $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$。
   在 GPU 上，归一化算子是典型的显存带宽受限（Memory-Bound）操作，减少一次规约循环可直接降低 10%~50% 的内联合核函数时延。

</div>
</details>

### Exercise 5 · Kaiming(He)Init

初始化的目标只有一个:让激活值的方差在深度方向上既不爆炸也不消失。Xavier 初始化按 `std = sqrt(2/(fan_in+fan_out))` 设计,前提是激活函数大致线性、关于 0 对称(比如 tanh)。ReLU 会把负半轴直接砍掉,相当于让方差打了对折,如果还用 Xavier 的方差,经过足够多层 ReLU 之后激活值会指数级收缩到 0。Kaiming 初始化的修正是只用 `fan_in` 并把系数改成 2:`std = sqrt(2/fan_in)`,这个系数 2 正好补偿 ReLU 砍掉一半方差的效应。

用 20 层的随机 ReLU 全连接网络做数值实验能直接看到差距:Xavier 初始化下最后一层的激活方差会跌到 `1e-6` 量级(基本传不动信息),Kaiming 初始化下方差始终维持在 `0.3~0.6` 这个数量级,虽然也会随深度缓慢衰减,但没有指数塌缩。

#### Quick Coding：`kaiming_init`

```python
def kaiming_init(weight: torch.Tensor, nonlinearity: str = "relu") -> None:
    ...
```

<details>
<summary>参考答案</summary>

```python
import math

def kaiming_init(weight: torch.Tensor, nonlinearity: str = "relu") -> None:
    fan_in = weight.shape[1] if weight.dim() == 2 else weight.shape[1:].numel()
    gain = math.sqrt(2.0) if nonlinearity == "relu" else 1.0
    std = gain / math.sqrt(fan_in)
    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)
```

数值验证(NumPy,20 层随机 ReLU 网络,`fan_in=256`,3 次独立试验的末层激活方差)：

```python
import numpy as np
np.random.seed(0)

def run_variance_check(std_fn, layers=20, fan_in=256, n=4096, trials=3):
    out = []
    for _ in range(trials):
        x = np.random.randn(n, fan_in)
        for _ in range(layers):
            W = np.random.randn(fan_in, fan_in) * std_fn(fan_in)
            x = np.maximum(x @ W, 0)
        out.append(x.var())
    return out

kaiming = run_variance_check(lambda fan_in: np.sqrt(2.0 / fan_in))
xavier = run_variance_check(lambda fan_in: np.sqrt(2.0 / (2 * fan_in)))
assert all(v > 0.1 for v in kaiming)     # 维持在 O(0.1~1) 量级
assert all(v < 1e-4 for v in xavier)     # 指数级塌缩到接近 0
```

</details>

### Exercise 5A · 极大更新参数化初始化 (muP / Maximal Update Parametrization)

在深度方向上，Kaiming 和 Xavier 初始化成功解决了信号随层数衰减或爆炸的问题；但在**宽度方向（Width $d$）**上，PyTorch 默认的标准参数化（Standard Parametrization, SP）存在严重的动力学失配。

在 SP 架构下，当模型宽度从百兆小模型（如 $d_{\text{base}} = 256$）扩展至百亿甚至千亿大模型（如 $d = 4096, 8192$）时：
1. **更新量尺度漂移**：在 Adam 或 SGD 优化器下，中间层特征更新量 $\Delta h = \Delta W \cdot x$ 和最终输出 Logits 更新量 $\Delta y = \Delta W_{\text{out}} \cdot h$ 随宽度 $d$ 的增大以 $O(\sqrt{d})$ 或 $O(d)$ 速度剧烈膨胀；
2. **超参数不可迁移**：在小模型上耗费算力网格搜索（Grid Search）得到的最优学习率 $\eta^*$、初始化尺度和 Weight Decay，直接搬到大模型上会导致训练迅速发散（Loss Spike）或陷入不可学习的懒惰状态（Lazy Training）；
3. **$\mu\text{P}$ 解决方案 (Yang et al., 2022)**：通过协调设置各层的初始化方差、前向输出倍率（Output Multiplier）与优化器学习率缩放（LR Scale），使得每一层的前向激活 $h$、反向梯度 $\nabla_h \mathcal{L}$ 以及参数更新特征变化量 $\Delta h$ 在 $d \to \infty$ 时严格稳定在 $\Theta(1)$。由此实现 **Zero-shot 超参数跨尺寸无损迁移（Zero-shot Hyperparameter Transfer）**——在秒级训练的小基准模型上扫出的最优超参数，可直接无缝部署到全量千亿模型上。

下表总结了 $\mu\text{P}$ 相对标准参数化 (SP) 的核心缩放规则（设基准隐藏层宽度为 $d_{\text{base}}$，当前隐藏宽度为 $d$，宽度扩展比为 $c = \frac{d}{d_{\text{base}}}$）：

| 模块层级 (Layer Role) | 初始化标准差 $\sigma$ | 前向输出乘子 (Output Mult) | 优化器学习率缩放 (LR Scale) | 机制目的与物理意义 |
| :--- | :--- | :--- | :--- | :--- |
| **输入层 (Input / Embedding)** | $\frac{1}{\sqrt{d_{\text{in}}}}$ | $1.0$ | $1.0$ | 维持初始输入特征方差 $\Theta(1)$ |
| **隐藏层 (Hidden Linear)** | $\frac{1}{\sqrt{d_{\text{in}}}}$ | $1.0$ | $\frac{1}{c} = \frac{d_{\text{base}}}{d}$ | 抵消宽度增加导致的特征更新量累加，保持 $\Delta h \sim \Theta(1)$ |
| **输出预测头 (Output / Readout)** | $\frac{1}{\sqrt{d_{\text{in}}}}$ | $\frac{1}{c} = \frac{d_{\text{base}}}{d}$ | $\frac{1}{c} = \frac{d_{\text{base}}}{d}$ | 抑制输出 Logits 随宽度发散，稳定 Softmax 交叉熵损失梯度 |
| **Attention 点积缩放** | — | $\frac{1}{d_k}$ (取代 SP 的 $\frac{1}{\sqrt{d_k}}$) | — | 阻止大宽度下注意力 Logits 尺度爆炸，防止 Softmax 熵塌缩 |

#### Quick Coding：`mup_init_and_configure`

```python
def mup_init_and_configure(
    weight: torch.Tensor,
    role: str,
    d_base: int,
    std_base: float = 1.0,
) -> tuple[float, float]:
    """
    根据 muP (Maximal Update Parametrization) 规则初始化线性层权重，
    并返回该层推荐的前向输出乘子 (output_mult) 与优化器学习率缩放倍率 (lr_scale)。

    参数:
        weight: (d_out, d_in) 的权重张量
        role: 层的角色，可选 'input'、'hidden'、'output'
        d_base: 基准模型隐藏层宽度 (Base Width)
        std_base: 基准初始标准差基数 (默认 1.0)

    返回:
        (output_mult, lr_scale) 元组:
            output_mult: 该层前向传播时输出应乘上的标量因子 (float)
            lr_scale: 该层在优化器 param_groups 中绑定的学习率缩放倍率 (float)
    """
    ...
```

<details>
<summary>参考答案</summary>

```python
import math
import torch
import torch.nn as nn

def mup_init_and_configure(
    weight: torch.Tensor,
    role: str,
    d_base: int,
    std_base: float = 1.0,
) -> tuple[float, float]:
    d_out, d_in = weight.shape

    if role == "input":
        c = 1.0
        std = std_base / math.sqrt(d_in)
        output_mult = 1.0
        lr_scale = 1.0
    elif role == "hidden":
        c = d_in / d_base
        std = std_base / math.sqrt(d_in)
        output_mult = 1.0
        lr_scale = 1.0 / c
    elif role == "output":
        c = d_in / d_base
        std = std_base / math.sqrt(d_in)
        output_mult = 1.0 / c
        lr_scale = 1.0 / c
    else:
        raise ValueError(f"Unknown role: {role}, expected 'input', 'hidden', or 'output'")

    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)

    return float(output_mult), float(lr_scale)

class MuPLinear(nn.Module):
    """基础 muP 线性层封装：前向自动应用 output_mult，并暴露 lr_scale 供优化器分组。"""
    def __init__(self, in_features: int, out_features: int, role: str, d_base: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.role = role
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.output_mult, self.lr_scale = mup_init_and_configure(self.weight, role, d_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out * self.output_mult

def create_mup_param_groups(model: nn.Module, base_lr: float) -> list[dict]:
    """为优化器按 muP 规则划分不同的学习率参数组 (Param Groups)。"""
    param_groups = []
    for module in model.modules():
        if isinstance(module, MuPLinear):
            group = {
                "params": [module.weight],
                "lr": base_lr * module.lr_scale,
            }
            if module.bias is not None:
                group["params"].append(module.bias)
            param_groups.append(group)
    return param_groups
```

数值验证 (NumPy, 验证不同宽度下 SP 更新量随 $\sqrt{d}$ 爆炸，而 $\mu\text{P}$ 稳定受控)：

```python
import numpy as np

# 验证随宽度扩展 (d = 256, 1024, 4096)，SP 特征更新步长膨胀，而 muP 保持稳定
np.random.seed(42)
d_base = 256
base_lr = 1e-3
prev_sp_rel = None

for d in [256, 1024, 4096]:
    c = d / d_base
    std_h = 1.0 / np.sqrt(d)
    w_h = np.random.normal(0.0, std_h, size=(d, d))
    x = np.random.randn(64, d) / np.sqrt(d)  # 单位范数特征输入

    # 前向输出
    h = x @ w_h.T
    h_norm = np.linalg.norm(h, axis=-1).mean()

    # 模拟 Adam 优化器单步更新 (梯度坐标归一化，更新量模长 ~ lr)
    # 1. SP 规则: 学习率恒定 base_lr
    sp_dw = np.random.choice([-1.0, 1.0], size=(d, d)) * base_lr
    sp_dh = x @ sp_dw.T
    sp_rel_update = np.linalg.norm(sp_dh, axis=-1).mean() / h_norm

    # 2. muP 规则: 学习率缩放 base_lr / c
    mup_dw = np.random.choice([-1.0, 1.0], size=(d, d)) * (base_lr / c)
    mup_dh = x @ mup_dw.T
    mup_rel_update = np.linalg.norm(mup_dh, axis=-1).mean() / h_norm

    # 断言: SP 的相对更新步长随 sqrt(d) 线性翻倍爆炸
    if d == 1024:
        assert 1.8 < sp_rel_update / prev_sp_rel < 2.2, "SP 更新量随 sqrt(d) 成倍放大"
    elif d == 4096:
        assert 1.8 < sp_rel_update / prev_sp_rel < 2.2, "SP 更新量持续发散"
    
    # 断言: muP 的特征相对更新量在宽度增大时有效收缩，绝不随宽度爆炸
    assert mup_rel_update <= 0.02, "muP 确保大宽度下特征更新步长受控"
    prev_sp_rel = sp_rel_update
```

</details>

### Exercise 6 · Dropout

Dropout 的核心机制是"训练时随机丢弃,推理时不丢弃",但工程实现几乎都用 inverted dropout:训练阶段按概率 `p` 把一部分单元置零,同时把剩下的单元放大 `1/(1-p)` 倍,这样存活单元的期望值和原始输入保持一致;推理阶段什么都不用做,直接原样输出。如果不做这个放大,就必须在推理时把输出整体乘以 `(1-p)` 才能保持期望一致。Inverted dropout 把这个麻烦挪到了训练阶段一次性解决，换来推理路径的零开销。另一个容易在面试里漏掉的点是 `self.training` 的切换:`model.train()` / `model.eval()` 修改的正是 `nn.Module.training` 这个标志位,Dropout 和 BatchNorm 的行为分支都靠它判断。

#### Quick Coding：`Dropout`

```python
class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

<details>
<summary>参考答案</summary>

```python
from torch import nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
        return x * mask / keep_prob
```

数值验证(NumPy,`p=0.3`,200 万个样本)：`E[x] ≈ E[inverted_dropout(x)]`,两者应当在采样误差范围内相等。

```python
import numpy as np
np.random.seed(1)
p = 0.3
x = np.random.randn(2_000_000) + 5.0
mask = (np.random.rand(*x.shape) > p).astype(np.float64)
dropped = x * mask / (1 - p)
assert abs(x.mean() - dropped.mean()) < 0.01
```

</details>

### Exercise 7 · Conv2d

面试里手写 Conv2d,重点从来不是"卷积是什么",而是怎么用已经有的 GEMM(矩阵乘)机器去实现它,而不是写四层 for 循环。标准技巧是 im2col / unfold:把每个卷积核要覆盖的局部 patch 展平成一列,所有 patch 拼成一个大矩阵,卷积就退化成"权重矩阵 乘以 patch 矩阵"，本质上和 Linear 层是同一个算子，只是数据搬运的方式不同。输出的空间尺寸公式:

```text
OH = floor((H + 2*padding - KH) / stride) + 1
OW = floor((W + 2*padding - KW) / stride) + 1
```

#### Quick Coding：`conv2d`

```python
def conv2d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
    stride: int = 1, padding: int = 0,
) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
import torch.nn.functional as F

def conv2d(x, weight, bias=None, stride=1, padding=0):
    B, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape

    # unfold: (B, Cin*KH*KW, L)，L = OH*OW
    patches = F.unfold(x, kernel_size=(KH, KW), stride=stride, padding=padding)
    w_flat = weight.reshape(Cout, -1)  # (Cout, Cin*KH*KW)

    out = torch.einsum("oc,bcl->bol", w_flat, patches)  # (B, Cout, L)
    if bias is not None:
        out = out + bias.view(1, -1, 1)

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    return out.reshape(B, Cout, OH, OW)
```

数值验证(NumPy,手写 im2col 版本 vs 四层 for 循环的朴素卷积,`stride=2, padding=1`)：

```python
import numpy as np
np.random.seed(1)

def conv2d_naive(x, w, stride=1, padding=0):
    B, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    if padding: x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    H, W = x.shape[2], x.shape[3]
    OH, OW = (H-KH)//stride+1, (W-KW)//stride+1
    out = np.zeros((B, Cout, OH, OW))
    for b in range(B):
        for co in range(Cout):
            for i in range(OH):
                for j in range(OW):
                    patch = x[b, :, i*stride:i*stride+KH, j*stride:j*stride+KW]
                    out[b, co, i, j] = np.sum(patch * w[co])
    return out

x = np.random.randn(2, 3, 8, 8)
w = np.random.randn(4, 3, 3, 3)
o_naive = conv2d_naive(x, w, stride=2, padding=1)
# im2col 版本省略（与朴素版本原理相同，展开成大矩阵乘）
assert o_naive.shape == (2, 4, 4, 4)
```

</details>

### Exercise 8 · Linear Regression(三种写法)

这道题的价值不在算法本身,而在于它把"数学解法"和"深度学习范式"放到同一个问题上对照,逼你说清楚三者的关系:

1. **解析解(正规方程)**:`w = (X^T X)^{-1} X^T y`,直接对平方误差求导并令导数为零得到的闭式解,数据量不大、`X^T X` 可逆时最快最准。
2. **手写梯度下降**:不调用 autograd,自己算 `grad = X^T (Xw - y) / n` 再迭代更新,和第一种方法本质上在优化同一个凸函数,唯一区别是走到最优点的方式：一步到位还是逐步逼近。
3. **`nn.Linear` + autograd + optimizer**:把同一个问题包装成"一层线性网络 + MSE loss + SGD/Adam",完全依赖自动微分,是深度学习框架里做同一件事的标准做法。

三者应当收敛到几乎相同的系数(下面的数值验证里正规方程和手写梯度下降在 5000 步之后系数最大误差在 `1e-15` 量级),这也是回答"这道题到底在考什么"时最值得说的一句话:线性回归只是最简单的凸优化问题,深度学习的训练循环并没有引入新的数学,只是把解析解换成了迭代逼近。

#### Quick Coding：`LinearRegression`

```python
class LinearRegression:
    def fit_normal_equation(self, X: torch.Tensor, y: torch.Tensor) -> None:
        ...

    def fit_manual_gd(self, X: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> None:
        ...

    def fit_autograd(self, X: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> None:
        ...
```

<details>
<summary>参考答案</summary>

```python
class LinearRegression:
    def __init__(self, in_features: int):
        self.w = torch.zeros(in_features + 1)  # 最后一维是 bias

    @staticmethod
    def _augment(X: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(X.shape[0], 1, dtype=X.dtype)
        return torch.cat([X, ones], dim=1)

    def fit_normal_equation(self, X, y):
        Xb = self._augment(X)
        # 用伪逆而不是直接求逆，避免 X^T X 奇异时崩溃
        self.w = torch.linalg.pinv(Xb) @ y

    def fit_manual_gd(self, X, y, lr=0.1, steps=5000):
        Xb = self._augment(X)
        n = Xb.shape[0]
        w = torch.zeros(Xb.shape[1])
        for _ in range(steps):
            pred = Xb @ w
            grad = Xb.T @ (pred - y) / n
            w = w - lr * grad
        self.w = w

    def fit_autograd(self, X, y, lr=0.01, steps=2000):
        model = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = model(X).squeeze(-1)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()
        w, b = model.weight.detach().squeeze(0), model.bias.detach()
        self.w = torch.cat([w, b])
```

数值验证(NumPy,正规方程 vs 手写全批量梯度下降,合成数据 `n=200, d=3`)：

```python
import numpy as np
np.random.seed(0)
n, d = 200, 3
X = np.random.randn(n, d)
true_w, true_b = np.array([1.5, -2.0, 0.7]), 0.3
y = X @ true_w + true_b + 0.01 * np.random.randn(n)
Xb = np.hstack([X, np.ones((n, 1))])

w_normal, *_ = np.linalg.lstsq(Xb, y, rcond=None)

w = np.zeros(d + 1)
for _ in range(5000):
    grad = Xb.T @ (Xb @ w - y) / n
    w = w - 0.1 * grad

assert np.abs(w_normal - w).max() < 1e-10
```

</details>

### Exercise 9 · Gradient Accumulation

梯度累积解决的问题是"想要的 batch size 装不进显存"。做法是把一个大 batch 切成 `K` 个 micro-batch,依次前向、反向,梯度自然会在 `.grad` 里累加,`K` 次之后再统一 `optimizer.step()`。这里唯一的数学要求是:累积出来的梯度必须和"直接用大 batch 算一次梯度"等价。如果每个 micro-batch 的 loss 用的是 mean reduction(而不是 sum),那就必须在反向传播前把每个 micro-batch 的 loss 再除以 `K`,否则累积出来的梯度会变成大 batch 梯度的 `K` 倍。另一个经典 bug 是 `zero_grad()` 的位置：它必须在一整个累积周期(`K` 个 micro-batch)开始之前调用一次,而不是每个 micro-batch 都清一次,否则前面几个 micro-batch 的梯度会被冲掉,退化成只用最后一个 micro-batch 训练。

#### Quick Coding：`accumulated_step`

```python
def accumulated_step(model, optimizer, microbatches, loss_fn) -> float:
    ...
```

<details>
<summary>参考答案</summary>

```python
def accumulated_step(model, optimizer, microbatches, loss_fn) -> float:
    optimizer.zero_grad()                      # 整个累积周期只清一次
    num_micro = len(microbatches)
    total_loss = 0.0
    for x, y in microbatches:
        pred = model(x)
        loss = loss_fn(pred, y) / num_micro     # mean reduction 必须再除以 K
        loss.backward()                          # 梯度自动累加到 .grad
        total_loss += loss.item()
    optimizer.step()
    return total_loss
```

数值验证(NumPy,四等分 micro-batch 的累积梯度 vs 一次性大 batch 梯度,均为 mean reduction 的二次损失)：

```python
import numpy as np
np.random.seed(0)
n, d = 200, 3
X = np.random.randn(n, d + 1)  # 含 bias 列
y = np.random.randn(n)
w0 = np.random.randn(d + 1)

full_grad = X.T @ (X @ w0 - y) / n

K = 4
bs = n // K
accum_grad = np.zeros(d + 1)
for i in range(K):
    xb, yb = X[i*bs:(i+1)*bs], y[i*bs:(i+1)*bs]
    g = xb.T @ (xb @ w0 - yb) / bs   # 每个 micro-batch 自己的 mean-reduction 梯度
    accum_grad += g / K              # 除以 K 才能拼回大 batch 梯度

assert np.abs(full_grad - accum_grad).max() < 1e-10
```

</details>

## 最后检查：本篇 Debug Checklist

- LayerNorm 是否先减均值再除标准差；RMSNorm 是否真的跳过了减均值这一步，而不是只是把 `eps` 加大了。
- GELU 用的是精确 `erf` 公式还是 GPT-2/BERT 的 tanh 近似，两者数值不完全相等，回答时不要混为一谈。
- BatchNorm 的统计量是否沿 `(B, ...)` 求，而不是像 LayerNorm 一样沿最后一维；`training=False` 时是否切换成 `running_mean` / `running_var`。
- Kaiming 初始化的方差公式里，`fan_in` 用的是权重的输入维度还是输出维度，系数 2 是否只在 ReLU 类非线性下使用。
- Dropout 的缩放是放在训练阶段（inverted dropout）还是推理阶段；`self.training` 是否随 `model.train()` / `model.eval()` 正确切换。
- Conv2d 的 unfold/im2col 展开维度是否和权重的 `reshape(Cout, -1)` 对齐，输出空间尺寸公式里 padding 和 stride 有没有算漏。
- 三种线性回归写法是否收敛到（几乎）同一组系数；如果没有，通常是学习率、步数或正规方程里该用伪逆却用了直接求逆。
- 梯度累积里，每个 micro-batch 的 loss 是否除以了累积步数 `K`；`zero_grad()` 是否只在一个累积周期开始时调用一次。
