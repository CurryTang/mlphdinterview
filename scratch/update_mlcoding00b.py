import os

with open('notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace header overview
old_overview = """本篇系统梳理大模型基础两大技术支柱：
1. **Decoder-Only 架构深度剖析与文本向量表征：为什么 Decoder-Only 成为绝对主流？现代 Decoder 如何攻克各向异性（Anisotropy）与表征退化？**
2. **硬件底层数值精度格式（FP32 / FP16 / BF16 / FP8 / NVFP4）与混合精度训练体系（Master Weights、Loss Scaling 与随机舍入）**"""

new_overview = """本篇系统梳理大模型基础三大技术支柱：
1. **Decoder-Only 架构深度剖析与文本向量表征：为什么 Decoder-Only 成为绝对主流？现代 Decoder 如何攻克各向异性（Anisotropy）与表征退化？**
2. **硬件底层数值精度格式（FP32 / FP16 / BF16 / FP8 / NVFP4）与混合精度训练体系（Master Weights、Loss Scaling 与随机舍入）**
3. **自回归语言模型初始 Loss 数学推导与第 0 步健全性检查（ln(V) 严密推导、温度系数、Label Smoothing、Tokenizer 词表效应与千卡预训练防翻车排查规范）**"""

content = content.replace(old_overview, new_overview)

# Replace Module 3 & 4
target_tail = """## 模块三：高频面试精选问答（Interview Rapid-Fire）

### Q1：为什么现在几乎没有团队使用 BERT 架构来做通用大模型了？
> **答**：
> 1. BERT 的双向自注意力机制虽然利于上下文特征提取，但其预训练目标是 Masked Language Modeling（MLM），破坏了自回归生成的因果链条，无法自然、连续地生成任意长度的文本；
> 2. BERT 没有自回归因果 KV Cache 概念，每次推断新词必须对全序列重走一次前向传播，推理时间复杂度高达 $O(S^2)$，在工程上完全无法支撑开放式长文本生成。

### Q2：在训练大模型时，遇到 Loss 变为 NaN，排查梯度的第一步是什么？
> **答**：
> 1. 检查当前步的梯度是否发生溢出（Overflow），确认是否使用了 FP16 且 Loss Scale 过大；
> 2. 检查 Attention 矩阵的缩放因子 $\\frac{1}{\\sqrt{d_k}}$ 是否丢失导致 Softmax 溢出；
> 3. 检查 LayerNorm / RMSNorm 的 $\\epsilon$ 是否过小（推荐 $10^{-5} \\sim 10^{-6}$）导致除以零；
> 4. 检查数据集中是否存在超长离群样本或未清洗的空文本/全零输入。"""

new_tail = """## 模块三：自回归语言模型初始 Loss 数学推导与第 0 步健全性检查（Initial Loss, ln(V) Derivation & Step-0 Sanity Check）

在启动百亿/千亿参数大模型的千卡分布式预训练（耗费数十万 GPU 时）之前，**第一步迭代（Step 0 / First-Step Loss）的数值表现是排查模型代码实现、因果掩码、数据加载与数值溢出最关键的试金石（Sanity Check）**。

```text
第 0 步初始 Loss 诊断决策树:
┌────────────────────────────────────────────────────────────────────────┐
│ 计算 Step 0 前向传播损失 L_0                                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│ L_0 ≈ ln(V)  │             │  L_0 ≫ ln(V) │             │  L_0 ≪ ln(V) │
│  (± 0.05)    │             │  (如 20.0+)  │             │  (如 3.5)    │
├──────────────┤             ├──────────────┤             ├──────────────┤
│  ✅ 完美正常  │             │ ❌ 权重发散/ │             │ ❌ 数据穿越/ │
│  进入训练    │             │ 残差累加爆炸 │             │ 未掩码 Padding│
└──────────────┘             └──────────────┘             └──────────────┘
```

---

### 1. 标准 Next-Token Prediction 初始 Loss 数学推导

设语言模型的词表大小为 $V$（Vocabulary Size），对于长度为 $S$ 的序列，模型在位置 $t$ 输出的未归一化对数概率（Logits）为 $z^{(t)} = [z_1, z_2, \dots, z_V] \in \mathbb{R}^V$。真实标签为单个 Token $c \in \{1, 2, \dots, V\}$。

标准因果语言建模采用交叉熵损失（Cross-Entropy Loss）：

$$\mathcal{L} = -\ln P(y = c \mid x_{<t}) = -\ln \left( \frac{e^{z_c}}{\sum_{v=1}^V e^{z_v}} \right) = -z_c + \ln \left( \sum_{v=1}^V e^{z_v} \right)$$

#### (1) 随机初始化下的均匀分布假设
在标准的模型参数初始化（如高斯分布 $\mathcal{N}(0, \sigma^2)$，其中 $\sigma = 0.02$ 或 Xavier/Kaiming 初始化）下：
- 所有输出 Logits $z_v$ 均值接近于 0（$\mathbb{E}[z_v] \approx 0$），且方差极小；
- 此时各 Token 的预测概率接近均匀分布（Uniform Distribution）：
  $$P(y = v) = \frac{e^{z_v}}{\sum_{j=1}^V e^{z_j}} \approx \frac{e^0}{V \cdot e^0} = \frac{1}{V}$$
- 交叉熵损失严格简化为自然对数：
  $$\mathcal{L}_{\text{init}} = -\ln \left(\frac{1}{V}\right) = \ln V$$

#### (2) 考虑 Logits 初始方差的二阶泰勒展开推导
若初始 Logits $z_v \sim \text{i.i.d. } \mathcal{N}(0, \sigma_z^2)$，令 $S = \sum_{v=1}^V e^{z_v}$。通过对 Log-Sum-Exp 进行二阶泰勒展开：

$$\mathbb{E}[\mathcal{L}_{\text{init}}] = -\mathbb{E}[z_c] + \mathbb{E}[\ln S] \approx \ln V + \frac{\sigma_z^2}{2} + \mathcal{O}(\sigma_z^4)$$

在合理的权重初始化尺度下（$\sigma_z \approx 0.02 \sim 0.1$），$\frac{\sigma_z^2}{2} < 0.005$ 极小，因此**理论期望初始 Loss 极其精确地等于 $\ln V$**。

---

### 2. 业界主流基础大模型理论初始 Loss 基准对照表

| 基础模型家族 | 词表大小 $V$ | 理论初始 Loss：$\ln V$ (nats/token) |
|---|---|---|
| **GPT-2 / GPT-3** | $50,257$ | $\ln(50257) \approx \mathbf{10.825}$ |
| **LLaMA-1 / LLaMA-2 / Mistral-7B** | $32,000$ | $\ln(32000) \approx \mathbf{10.373}$ |
| **LLaMA-3 / LLaMA-3.1 / LLaMA-3.3** | $128,256$ | $\ln(128256) \approx \mathbf{11.762}$ |
| **DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1** | $129,280$ | $\ln(129280) \approx \mathbf{11.770}$ |
| **Qwen-2 / Qwen-2.5** | $152,064$ | $\ln(152064) \approx \mathbf{11.932}$ |
| **Gemma / Gemma-2** | $256,000$ | $\ln(256000) \approx \mathbf{12.453}$ |

---

### 3. 温度系数（Temperature）与标签平滑（Label Smoothing）的理论影响

#### (1) 温度系数 $T$ 的影响
若在前向计算 Loss 时引入了温度系数 $T$（即计算 Softmax($z / T$)）：

$$\mathcal{L}(T) = -\frac{z_c}{T} + \ln \left( \sum_{v=1}^V e^{z_v / T} \right)$$

- 当 $T \to \infty$ 时：Logits 方差被极致压缩，输出绝对均匀，$\mathcal{L} \to \ln V$；
- 当 $T < 1.0$ 时：放大了初始 Logits 的微小随机扰动，初始 Loss 的方差与均值会略微偏大（$\mathbb{E}[\mathcal{L}] \approx \ln V + \frac{\sigma_z^2}{2 T^2}$）。

#### (2) 标签平滑（Label Smoothing $\epsilon$）的影响
若采用标签平滑，真实目标分布变为 $q(v) = (1-\epsilon)\mathbb{I}(v=c) + \frac{\epsilon}{V}$：

$$\mathcal{L}_{\text{smooth}} = -(1-\epsilon)\ln P(c) - \frac{\epsilon}{V}\sum_{v=1}^V \ln P(v)$$

在随机初始化（$P(v) \approx \frac{1}{V}$）时代入：

$$\mathcal{L}_{\text{smooth}} = -(1-\epsilon)\ln \left(\frac{1}{V}\right) - \frac{\epsilon}{V} \cdot V \ln \left(\frac{1}{V}\right) = \ln V$$

> 💡 **核心结论**：**无论标签平滑系数 $\epsilon$ 设为多少，在均匀随机初始化下，理论期望初始 Loss 严格恒等于 $\ln V$**！

---

### 4. Tokenizer 词表规模效应与对齐填充

1. **词表扩大对 Token 级 Loss vs. 文本级压缩率的影响**：
   - 词表从 32K（LLaMA-2）扩展到 128K（LLaMA-3），Token 级别的初始 Loss 从 10.37 上升到 11.76（上升了约 1.39 nats）；
   - **但是**：大词表具有更高的文本压缩率（相同的 1000 英文单词，32K Tokenizer 需 1300 个 Token，而 128K Tokenizer 仅需 950 个 Token）。因此**整篇文档的总负对数似然（Sequence-level NLL）反而显著下降**！
2. **词表对齐填充（Vocabulary Padding for GPU Tensor Core Alignment）**：
   - 工业界常将词表大小向 64 或 128 的整数倍向上取整（例如实际有效词表为 32,000，但 Embedding Head 维度填充为 $V_{\text{embed}} = 32,256$ 以最大化 Tensor Core 矩阵乘法吞吐）；
   - **注意**：此时第 0 步计算的理论 Loss 对应为 $\ln(V_{\text{embed}})$，而非未填充的有效词表。

---

### 5. 工业级千卡预训练启动前“第 0 步 Loss 排查协议”（Step-0 Sanity Check Protocol）

在点击集群批量提交大规模训练任务前，必须在单机单卡上运行 Step 0 进行严格断言：

```text
第 0 步异常排查对照矩阵：
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 观测现象                │ 根本原因 (Root Causes) 与紧急修复措施                                  │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. L_0 ≈ ln(V) ± 0.05   │ ✅ 完美通过。参数初始化、因果掩码、Embedding 投影与 Loss 计算完全正确。 │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. L_0 ≫ ln(V)          │ ❌ 权重方差过大；残差流缺少 1/√(2*N_layers) 缩放导致深度层 Logits 爆炸；│
│    (如 L_0 = 25.0)      │ ❌ 输出线性层权重未归一化；LayerNorm 权重被错误初始化。                │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. L_0 ≪ ln(V)          │ ❌ 致命数据穿越：因果下三角 Mask 缺失（变成双向注意力，模型直接抄后文）；│
│    (如 L_0 = 3.5)       │ ❌ Target Padding 泄露：未设置 ignore_index=-100，模型无脑预测 [PAD]； │
│                         │ ❌ 标签偏移错误：labels 未错开一位 (shift_labels = input_ids[..., 1:])。│
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. L_0 = NaN / Inf      │ ❌ Attention 缩放因子 1/√d_k 丢失导致 Softmax 溢出；                   │
│                         │ ❌ FP16 动态 Loss Scale 初值过大，或混合精度未开启 FP32 Master Weights。 │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 模块四：高频面试精选问答（Interview Rapid-Fire）

### Q1：为什么现在几乎没有团队使用 BERT 架构来做通用大模型了？
> **答**：
> 1. BERT 的双向自注意力机制虽然利于上下文特征提取，但其预训练目标是 Masked Language Modeling（MLM），破坏了自回归生成的因果链条，无法自然、连续地生成任意长度的文本；
> 2. BERT 没有自回归因果 KV Cache 概念，每次推断新词必须对全序列重走一次前向传播，推理时间复杂度高达 $O(S^2)$，在工程上完全无法支撑开放式长文本生成。

### Q2：在训练大模型时，遇到 Loss 变为 NaN，排查梯度的第一步是什么？
> **答**：
> 1. 检查当前步的梯度是否发生溢出（Overflow），确认是否使用了 FP16 且 Loss Scale 过大；
> 2. 检查 Attention 矩阵的缩放因子 $\\frac{1}{\\sqrt{d_k}}$ 是否丢失导致 Softmax 溢出；
> 3. 检查 LayerNorm / RMSNorm 的 $\\epsilon$ 是否过小（推荐 $10^{-5} \\sim 10^{-6}$）导致除以零；
> 4. 检查数据集中是否存在超长离群样本或未清洗的空文本/全零输入。

### Q3：为什么自回归语言模型的第 0 步 Loss 必须等于 $\\ln V$？如果等于 2.0 说明了什么？
> **答**：
> 1. **原理解释**：在随机初始化状态下，模型未学到任何语言知识，对词表中所有 $V$ 个 Token 给出接近均匀的概率分布 $P(y=c) = \\frac{1}{V}$。交叉熵损失 $\\mathcal{L} = -\\ln(1/V) = \\ln V$；
> 2. **Loss = 2.0 诊断**：说明发生了**极其严重的实现 Bug 或数据穿越（Data Leakage）**：
>    - **原因 1（因果 Mask 丢失）**：注意力掩码未正确配置为因果下三角矩阵，模型利用双向注意力直接在同一行看到了“未来的目标词”；
>    - **原因 2（Padding 未屏蔽）**：数据批次中的大量 `[PAD]` 填充 Token 没有在 Loss 计算中被屏蔽（未设置 `ignore_index=-100`），模型只需在绝大多数位置无脑预测 `[PAD]` 即可获得极低的虚假损失；
>    - **原因 3（自回归标签未对齐）**：输入序列与目标序列没有错开 1 位（`labels = input_ids`），导致模型直接预测当前输入词本身。"""

content = content.replace(target_tail, new_tail)

with open('notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.md', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully updated MLCoding00B Chinese note")
