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

<details>
<summary>深度解析：现代 LLM 架构中的归一化演进与系统设计 (Pre-RMSNorm · Q-K Norm · 零偏置)</summary>

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
