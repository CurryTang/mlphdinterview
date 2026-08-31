# ML Coding 00B · LLM 基础：Decoder-Only 架构胜出原理解析、混合精度训练、后训练对齐与模型蒸馏

在大语言模型（LLM）与生成式 AI 系统工程中，理解模型架构底层的选型逻辑、数值精度在硬件层面的流动方式、以及后训练对齐与评测体系，是贯穿模型预训练、微调与大规模部署的核心必修课。

本篇系统梳理大模型基础三大支柱：
1. **Decoder-Only 架构深度剖析：为什么它成为了当今大语言模型的绝对主流？**
2. **数值精度格式（FP32 / FP16 / BF16）与混合精度训练体系（Loss Scaling 与 Master Weights）**
3. **后训练对齐、知识蒸馏与评测陷阱（DPO vs PPO、白盒/黑盒蒸馏、LLM-as-a-Judge 与对齐税）**

---

## 模块一：为什么 Decoder-Only 成为当今大模型的绝对主流？

在 Transformer 问世初期，学界与工业界经历了三大架构形态的探索：
- **Encoder-Only（如 BERT、RoBERTa）**：全双向自注意力，适用于判别式任务（分类、实体抽取、文本嵌入）。
- **Encoder-Decoder（如 T5、BART、原始 Transformer）**：双向编码上下文 + 因果自回归解码目标，适用于翻译、摘要。
- **Decoder-Only（如 GPT、LLaMA、Mistral、Qwen、DeepSeek）**：统一因果下三角掩码自回归，统一了当今几乎所有前沿主流大模型。

```text
三大 Transformer 架构演进与收敛：
┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│     Encoder-Only      │     │    Encoder-Decoder    │     │     Decoder-Only      │
│     (BERT / RoBERTa)  │     │      (T5 / BART)      │     │ (GPT / LLaMA / Qwen)  │
├───────────────────────┤     ├───────────────────────┤     ├───────────────────────┤
│ • 双向自注意力 (Bidirectional)│ • 双向 Encoder + 因果 Decoder │ • 统一因果自回归 (Causal)│
│ • 判别式与语义表示     │ • 结构分离 (Cross-Attention) │ • 生成式统一通用基座  │
│ • 无法自然连续自回归生成│ • 任务定义割裂 / 显存开销双份│ • Zero-shot / Few-shot  │
└───────────────────────┘     └───────────────────────┘     └───────────────────────┘
                                                                        │
                                                    🌟 现代 LLM 范式大一统
```

### 四大决定性技术与系统工程动因

#### 1. 统一自回归范式与零任务摩擦（Unified Next-Token Prediction & In-Context Learning）

- **任务表达大一统**：在 Decoder-Only 架构下，**预训练（Pre-training）、指令微调（SFT）、少样本上下文学习（Few-shot ICL）、Prompting、思维链推理（Chain-of-Thought）** 全部统一为同一个数学形式——**基于上文预测下一个 Token 的条件概率**：

$$P(X) = \prod_{i=1}^S P(x_i \mid x_1, x_2, \dots, x_{i-1})$$

- **零结构摩擦**：任何输入（系统提示词、少样本示例、用户 Query、历史对话、模型自身的中间 CoT 推理、最终回复）都以扁平的 Token 序列形式无缝拼接待在同一个自回归因果流中，不需要像 Encoder-Decoder 那样显式切分“哪部分是 Source，哪部分是 Target”。

#### 2. 极简 KV Cache 与推理显存复用（Simplified KV Cache & Zero Cross-Attention Overhead）

- **单套线性 KV Cache**：Decoder-Only 只有单一的 Self-Attention KV Cache，Prefill（提示词预填充）和 Decode（逐 Token 自回归生成）无缝共享同一套显存缓冲区，生成新 Token 时只需在线性数组末尾追加当前的 Key/Value。
- **Encoder-Decoder 的显存与调度痛点**：
  - Encoder-Decoder 必须维护两套独立的缓存：Encoder 的全局双向静态 KV 缓存 + Decoder 的因果自回归 KV 缓存；
  - 每一层 Decoder 都有额外的 **Cross-Attention（跨注意力层）**，每次生成 Token 都必须跨越内存总线去读取庞大的 Encoder KV 缓存，严重加剧了 GPU 显存带宽瓶颈（Memory-Bandwidth Bound）；
  - 在 vLLM、PagedAttention 和连续批处理（Continuous Batching）等现代 Serving 框架中，Decoder-Only 的单向虚拟内存页管理极简高效，而 Encoder-Decoder 的双套缓存碎片化与调度复杂度成倍激增。

#### 3. 参数计算预算效率与 Scaling Law 幂律（Compute Budget Efficiency & Scaling Laws）

- **全参数密集监督信号**：
  - 在 Decoder-Only 预训练中，序列中从第 1 个 Token 到第 $S$ 个 Token，**每一个位置都会计算预测下一个 Token 的交叉熵损失（Dense Loss Supervision）**。模型的所有参数在每个 Token 的预测中都被充分激活和训练。
  - 在 Encoder-Decoder 预训练（如 Span Corruption）中，Encoder 接收掩码输入，只有 Decoder 部分生成被遮蔽的片段。这导致 Encoder 参数无法直接从生成 Loss 中获得直接的高密度监督，且在生成阶段 Encoder 不参与自回归参数容量表达（Capacity Underutilization）。
- **实证 Scaling 表现**：在固定训练计算量预算（FLOPs Budget）和参数规模下，Decoder-Only 展现出最陡峭、最稳定、可预测性最高的 Scaling Law 幂律性能提升。

#### 4. 长序列外推与现代位置编码天然契合（Long Context & RoPE Synergy）

- 因果下三角注意力天然具备时间单向因果性，与**旋转位置编码（RoPE）**、**Chunked Prefill** 以及注意力窗口滑动机制（Sliding Window）天然契合，能够非常平滑地通过位置内插/外推（如 YaRN、Dynamic NTK）从 4K 上下文无缝拓展到 128K 乃至 1M。
- 双向 Encoder 在极长上下文下容易发生**注意力弥散（Attention Dilution）**，且无法直接使用单向因果推理优化。

---

## 模块二：数值精度表示与混合精度训练体系

### 1. 浮点数表示结构（FP32 vs FP16 vs BF16）

根据 IEEE 754 标准，浮点数由 1 位符号位（Sign）、指数位（Exponent）和尾数位（Mantissa / Fraction）组成：

```text
浮点数比特结构剖析：
FP32 (32-bit): [1 Sign] [ 8 Exponent Bits ] [      23 Mantissa Bits      ]
FP16 (16-bit): [1 Sign] [ 5 Exponent ] [    10 Mantissa    ]
BF16 (16-bit): [1 Sign] [ 8 Exponent Bits ] [ 7 Mantissa ]
```

| 精度格式 | 总位数 | 指数位（Exponent） | 尾数位（Mantissa） | 动态数值范围（Dynamic Range） | 有效精度（Precision） | 核心工业定位 |
|---|---|---|---|---|---|---|
| **FP32** | 32 bits | 8 bits | 23 bits | $10^{-38} \sim 10^{38}$ | 高（约 7 位有效十进制数字） | 优化器状态、主权重累加基石 |
| **FP16** | 16 bits | 5 bits | 10 bits | $10^{-5} \sim 6.5 \times 10^4$ | 中（约 3 位有效十进制数字） | 传统 GPU 推理，训练易发生**下溢/溢出** |
| **BF16** | 16 bits | 8 bits | 7 bits | $10^{-38} \sim 10^{38}$（与 FP32 完全相同） | 较低（约 2 位有效十进制数字） | **现代大模型预训练/微调绝对主流标准** |

---

### 2. 混合精度训练核心机制：FP16 动态损失缩放（Loss Scaling）

#### 为什么 FP16 训练必须进行 Loss Scaling？
由于 FP16 的指数位只有 5 位，其能表示的最小正正规数为 $2^{-14} \approx 6.1 \times 10^{-5}$。在大模型反向传播过程中，大量参数的梯度值非常微小（例如 $10^{-6} \sim 10^{-8}$）。如果直接用 FP16 计算和存储梯度，这些梯度会被硬件直接截断为 0，导致**梯度下溢（Gradient Underflow）**，模型无法更新。

```text
FP16 动态损失缩放 (Dynamic Loss Scaling) 闭环流程：
┌───────────────────────┐
│ 1. 前向传播计算 Loss  │
└──────────┬────────────┘
           ▼
┌────────────────────────────────────────┐
│ 2. 损失缩放: Loss_scaled = Loss * Scale│ ──> 将微小梯度整体右移至 FP16 可表示动态区间
└──────────┬─────────────────────────────┘
           ▼
┌────────────────────────────────────────┐
│ 3. 反向传播计算 Scaled Gradients (FP16)│
└──────────┬─────────────────────────────┘
           ▼
┌────────────────────────────────────────┐
│ 4. 梯度反缩放: Grad = Grad_scaled / Scale│ ──> 恢复真实梯度尺度
└──────────┬─────────────────────────────┘
           ▼
┌────────────────────────────────────────┐
│ 5. 溢出检查 (Check for Inf / NaN)      │
└──────────┬─────────────────────────────┘
           ├───────────────────────────────┐
      [无溢出 (Valid)]               [检测到 Inf/NaN (Overflow)]
           ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ 6. 累加并更新 FP32 主权重│     │ 6. 跳过本步优化器更新   │
│    连续 N 步无溢出则 Scale 翻倍│ │    Scale 减半 (Scale /= 2) │
└─────────────────────────┘     └─────────────────────────┘
```

---

### 3. BF16 为什么能彻底取代 FP16 成为大模型训练标配？

1. **相同的动态范围**：BF16 拥有与 FP32 完全一致的 8 位指数位，其动态范围达到 $10^{\pm 38}$，彻底消除了深度神经网络训练中的梯度下溢风险；
2. **免去 Loss Scaling 调参负担**：不再需要复杂的动态缩放控制器，大幅提升了分布式大规模训练（Megatron-LM, DeepSpeed）的数值稳定性；
3. **硬件原生加速**：NVIDIA Ampere（A100）、Hopper（H100/H800）、Blackwell 架构的 Tensor Cores 对 BF16 矩阵乘法提供了全速率硬件支持。

---

### 4. FP32 Master Weights（主权重）机制

无论使用 FP16 还是 BF16 混合精度训练，**优化器内部的 Master Weights（模型主参数）以及 Adam 的动量（Momentum $m$）和方差（Variance $v$）必须全程保留在 FP32**。

```text
混合精度参数显存分布与流动：
┌────────────────────────────────────────────────────────┐
│ GPU Tensor Cores (快速低精度计算)                      │
│ • 前向传播激活值 (Activations): BF16 / FP16             │
│ • 模型前向权重 (Model Weights): BF16 / FP16             │
│ • 反向传播梯度 (Gradients):     BF16 / FP16             │
└───────────────────────────┬────────────────────────────┘
                            │ (梯度汇总并转为 FP32)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Optimizer 状态空间 (高精度累加)                        │
│ • Master Model Weights: FP32                           │
│ • Adam 一阶动量 m:      FP32                           │
│ • Adam 二阶方差 v:      FP32                           │
│   更新公式: W_fp32 = W_fp32 - lr * (m / (sqrt(v) + eps))│
│   更新完成后，将 W_fp32 截断转换为 BF16 供下一步前向计算│
└────────────────────────────────────────────────────────┘
```

**为什么主权重不能用 BF16/FP16 存储？**
因为单步参数更新量 $\Delta W = -\eta \cdot \frac{m}{\sqrt{v} + \epsilon}$ 通常极小（如 $10^{-6}$）。如果主权重本身是 BF16（只有 7 位尾数），将一个极小的更新量加到一个较大数值（如 $W = 1.5$）上时，低位有效数字会被直接抹去（Swallowing Problem），导致权重永远无法累积更新。

---

## 模块三：后训练对齐、知识蒸馏与评测体系

### 1. DPO vs PPO-Style RLHF 架构与目标函数

在大模型后训练阶段（Post-Training），为了使模型输出符合人类意图（Helpful, Honest, Harmless），主流对齐技术经历了从 PPO 到 DPO 的范式转变。

```text
PPO 4 模型复杂体系 vs DPO 单一闭式目标：
┌────────────────────────────────────────────────────────────────────────┐
│ PPO 强化学习对齐 (4 个并发模型常驻显存):                                │
│ 1. Actor (策略模型，训练更新)                                           │
│ 2. Critic / Value (价值模型，评估状态收益，训练更新)                   │
│ 3. Reward Model (奖励模型，离线预先训练好，推理冻结)                   │
│ 4. Reference Model (参考基准模型，计算 KL 散度惩罚，推理冻结)           │
│ 挑战: 强化学习训练极不稳定、显存占用极高、GAE 优势估计超参调优困难     │
└────────────────────────────────────────────────────────────────────────┘
                                    ▼ 革命性演进
┌────────────────────────────────────────────────────────────────────────┐
│ DPO (Direct Preference Optimization, 仅需 Actor + Reference 两个模型): │
│ 基于 Bradley-Terry 偏好模型，隐式解析出最优策略与奖励的解析关系:        │
│                r(x, y) = β * log( π_θ(y|x) / π_ref(y|x) )              │
│ 直接在偏好对 (x, y_w, y_l) 上构建闭式分类对数损失，彻底废除 Reward 模型│
└────────────────────────────────────────────────────────────────────────┘
```

#### DPO 目标函数数学推导

给定提示词 $x$，人类偏好的优质回答 $y_w$（winner）和劣质回答 $y_l$（loser），DPO 损失函数定义为：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

其中 $\beta$ 是温度系数（控制偏离参考策略 $\pi_{\text{ref}}$ 的 KL 惩罚强度），$\sigma(\cdot)$ 是 Sigmoid 函数。

---

### 2. 知识蒸馏（Knowledge Distillation）两大范式

| 蒸馏范式 | 核心技术机制 | 优势（Pros） | 挑战与局限（Cons） | 工业界典型应用 |
|---|---|---|---|---|
| **白盒蒸馏（White-Box Logit Distillation）** | 学生模型与教师模型在相同的 Token 词表上对齐输出 Logits 概率分布，最小化 KL 散度：<br>$$\mathcal{L}_{KD} = D_{KL}(P_{\text{teacher}} \parallel P_{\text{student}})$$ | 保留了教师模型暗含的“暗知识”（Dark Knowledge，如非 Top-1 候选词的概率关联）。 | 要求教师与学生模型的 **Tokenizer 词表完全一致**；跨组织或闭源商业 API 无法获取全量 Logits。 | 相同模型家族的小模型蒸馏（如 LLaMA-3-70B $\to$ LLaMA-3-8B）。 |
| **黑盒蒸馏（Black-Box Sequence-Level SFT）** | 教师模型作为离线生成器，合成海量高质量的问答、代码、思维链（CoT）推理轨迹；学生模型通过标准 SFT 训练。 | 对教师架构无任何约束；可通过商业 API（如 GPT-4）进行高价值合成数据生产。 | 丢失了词表层面的 Soft Target 概率分布信息；依赖高质量数据清洗与去重过滤。 | 垂直领域专有小模型构建、Reasoning 数据合成（如 DeepSeek-R1-Distill）。 |

---

### 3. LLM-as-a-Judge 评测体系与三大固有偏见

现代大模型评测广泛使用能力更强的大模型（如 GPT-4、Claude-3.5-Sonnet）作为裁判对候选回答进行多维度打分或成对比较（Pairwise Comparison）。然而，裁判模型存在以下三大严重偏见：

```text
LLM-as-a-Judge 三大固有偏见与防御策略：
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 偏见类型 (Biases)       │ 现象机理与工业界标准防御手段                                           │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. 位置偏见 (Position)  │ 倾向于给排在前面的候选者（Candidate 1）打更高分。                      │
│                         │ ➔ 防御手段: 进行 Pairwise 位置对调 (Swap Order) 双向打分并取平均。    │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. 冗长偏见 (Verbosity) │ 倾向于给篇幅更长、排版更丰富但可能废话连篇的回答打高分。               │
│                         │ ➔ 防御手段: 在 Prompt 中明确长度约束，或对字数进行长度惩罚正则化。    │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. 自我偏好 (Self-Enhance) 倾向于给自己模型家族生成的回答打更高分（由于自注意力特征喜好一致）。│
│                         │ ➔ 防御手段: 引入多裁判委员会交叉盲审，或使用带标准参考答案的裁判 Prompt│
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 4. 对齐负效应：奖励黑客与对齐税（Production Regressions）

1. **奖励黑客（Reward Hacking）**：
   策略模型学会了利用奖励模型或评测规则的漏洞，生成表面极其礼貌、结构工整、大量堆砌客套话但实际上毫无实质解答内容的无用文本。
2. **对齐税（Alignment Tax）与幻觉漂移**：
   在经历过度激进的安全与人类偏好对齐后，模型的**基础通用能力（如复杂代码生成、多步逻辑数学推理、知识标定校准）可能发生退化**。模型为了“不出错”而过度拒答（Over-refusal），或在确定性知识上产生新的幻觉漂移。

---

## 模块四：高频面试精选问答（Interview Rapid-Fire）

### Q1：为什么现在几乎没有团队使用 BERT 架构来做通用大模型了？
> **答**：
> 1. BERT 的双向自注意力机制虽然利于上下文特征提取，但其预训练目标是 Masked Language Modeling（MLM），破坏了自回归生成的因果链条，无法自然、连续地生成任意长度的文本；
> 2. BERT 没有自回归因果 KV Cache 概念，每次推断新词必须对全序列重走一次前向传播，推理时间复杂度高达 $O(S^2)$，在工程上完全无法支撑开放式长文本生成。

### Q2：在训练大模型时，遇到 Loss 变为 NaN，排查梯度的第一步是什么？
> **答**：
> 1. 检查当前步的梯度是否发生溢出（Overflow），确认是否使用了 FP16 且 Loss Scale 过大；
> 2. 检查 Attention 矩阵的缩放因子 $\frac{1}{\sqrt{d_k}}$ 是否丢失导致 Softmax 溢出；
> 3. 检查 LayerNorm / RMSNorm 的 $\epsilon$ 是否过小（推荐 $10^{-5} \sim 10^{-6}$）导致除以零；
> 4. 检查数据集中是否存在超长离群样本或未清洗的空文本/全零输入。
