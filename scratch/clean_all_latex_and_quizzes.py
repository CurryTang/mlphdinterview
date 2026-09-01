import re

# ----------------------------------------------------
# 1. Update MLCoding00B.md
# ----------------------------------------------------
with open("notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.md", "r", encoding="utf-8") as f:
    content_00b = f.read()

# Fix module 3 & 4 text
target_m3_zh = r"""## 模块三：自回归语言模型初始 Loss 数学推导与第 0 步健全性检查（Initial Loss, ln(V) Derivation & Step-0 Sanity Check）

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
  $$\mathcal{L}_{\text{init}} = -\ln \left( \frac{1}{V} \right) = \ln V$$

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
若在前向计算 Loss 时引入了温度系数 $T$（即计算 $\text{Softmax}(z / T)$）：

$$\mathcal{L}(T) = -\frac{z_c}{T} + \ln \left( \sum_{v=1}^V e^{z_v / T} \right)$$

- 当 $T \to \infty$ 时：Logits 方差被极致压缩，输出绝对均匀，$\mathcal{L} \to \ln V$；
- 当 $T < 1.0$ 时：放大了初始 Logits 的微小随机扰动，初始 Loss 的方差与均值会略微偏大（$\mathbb{E}[\mathcal{L}] \approx \ln V + \frac{\sigma_z^2}{2 T^2}$）。

#### (2) 标签平滑（Label Smoothing $\epsilon$）的影响
若采用标签平滑，真实目标分布变为 $q(v) = (1-\epsilon)\mathbb{I}(v=c) + \frac{\epsilon}{V}$：

$$\mathcal{L}_{\text{smooth}} = -(1-\epsilon)\ln P(c) - \frac{\epsilon}{V}\sum_{v=1}^V \ln P(v)$$

在随机初始化（$P(v) \approx \frac{1}{V}$）时代入：

$$\mathcal{L}_{\text{smooth}} = -(1-\epsilon)\ln \left( \frac{1}{V} \right) - \frac{\epsilon}{V} \cdot V \ln \left( \frac{1}{V} \right) = \ln V$$

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
│    (如 L_0 = 3.5)       │ ❌ 未对 Padding Token 设置 ignore_index=-100（导致对填充符虚假拟合）。 │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. L_0 = NaN / Inf      │ ❌ Attention 分数缺少 1/√d_k 缩放导致 Softmax 上溢；未开启 FP16 缩放。 │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 6. 高频大模型面试自测单选题（Multiple-Choice Question）

<details class="exercise">
<summary><span class="q-label">Q1 · 单选题</span> <span class="q-text">使用 LLaMA-3（词表大小 $V = 128,256$）在千卡集群启动千亿 Token 预训练，在 Step 0 随机初始化未经过任何梯度更新时，下列关于训练损失（Training Loss）与排查诊断的说法中，<strong>哪一项是完全正确的</strong>？</span></summary>

- [ ] **A.** 初始 Loss 理论期望值应接近 0，若大于 1.0 说明参数初始化方差过大发生数值发散。
- [ ] **B.** 理论期望初始 Loss 为 $\ln(128256) \approx 11.76$；若实测 $L_0 = 3.2$，说明模型预训练能力极强收敛极快。
- [x] **C.** 理论期望初始 Loss 为 $\ln(128256) \approx 11.76$；若实测 $L_0 = 3.2$，极大概率存在因果下三角掩码（Causal Mask）丢失或 Padding Token 未设置 `ignore_index=-100` 的严重数据泄漏 bug。
- [ ] **D.** 若开启了标签平滑（Label Smoothing $\epsilon = 0.1$），第 0 步理论期望 Loss 将显著下降为 $(1-\epsilon)\ln V \approx 10.58$。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **理论初值**：在随机均匀初始化下，每个 Token 的预测概率 $P(y=v) \approx \frac{1}{V}$，标准交叉熵损失 $\mathcal{L}_{\text{init}} = -\ln(1/V) = \ln V$。对于 LLaMA-3，$V=128,256$，$\ln(128256) \approx 11.762$；
>   2. **$L_0 \ll \ln V$ 的严重缺陷**：如果第 0 步 Loss 远低于 11.76（例如 3.2 甚至接近 0），绝非模型天生收敛快，而是模型发生**向未来偷看（Information Leakage）**：Causal Mask 失效变成了双向注意力（直接抄写后一个 Token 的 Embedding），或者 Padding 填充位置参与了 Loss 计算；
>   3. **D 选项错误**：数学上已严密证明，无论标签平滑系数 $\epsilon$ 设为多少，在均匀随机初始化下 $\mathcal{L}_{\text{smooth}} = -(1-\epsilon)\ln(1/V) - \frac{\epsilon}{V} \cdot V \ln(1/V) = \ln V$，保持完全恒定！
</details>
"""

# Replace in MLCoding00B.md from ## 模块三 to end
m3_pos = content_00b.find("## 模块三：自回归语言模型初始")
if m3_pos != -1:
    content_00b = content_00b[:m3_pos] + target_m3_zh
else:
    content_00b += "\n\n" + target_m3_zh

with open("notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.md", "w", encoding="utf-8") as f:
    f.write(content_00b)

# ----------------------------------------------------
# 2. Update MLCoding00B.en.md
# ----------------------------------------------------
with open("notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.en.md", "r", encoding="utf-8") as f:
    content_00b_en = f.read()

target_m3_en = r"""## Module 3: Next-Token Initial Loss ln(V) Derivation & Step-0 Sanity Check

Before launching distributed training on thousands of GPUs (costing hundreds of thousands of GPU hours), **the numerical behavior at Step 0 (First-Step Loss) is the single most critical sanity check for verifying causal masking, data loaders, and numerical stability**.

```text
Step 0 Loss Diagnostic Decision Tree:
┌────────────────────────────────────────────────────────────────────────┐
│ Compute Step 0 Forward Loss L_0                                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│ L_0 ≈ ln(V)  │             │  L_0 ≫ ln(V) │             │  L_0 ≪ ln(V) │
│  (± 0.05)    │             │  (e.g. 20.0+)│             │  (e.g. 3.5)  │
├──────────────┤             ├──────────────┤             ├──────────────┤
│  ✅ PASS     │             │ ❌ Exploding │             │ ❌ Causal    │
│  Proceed     │             │ Residuals/LR │             │ Leakage/Pad  │
└──────────────┘             └──────────────┘             └──────────────┘
```

---

### 1. Mathematical Derivation of Standard Next-Token Initial Loss

For vocabulary size $V$ and sequence length $S$, the unnormalized log-probabilities (Logits) at position $t$ are $z^{(t)} = [z_1, z_2, \dots, z_V] \in \mathbb{R}^V$ for target token $c \in \{1, 2, \dots, V\}$.

Standard causal cross-entropy loss:

$$\mathcal{L} = -\ln P(y = c \mid x_{<t}) = -\ln \left( \frac{e^{z_c}}{\sum_{v=1}^V e^{z_v}} \right) = -z_c + \ln \left( \sum_{v=1}^V e^{z_v} \right)$$

#### (1) Uniform Distribution Under Random Initialization
Under standard parameter initialization ($\mathcal{N}(0, \sigma^2)$ with $\sigma \approx 0.02$ or Xavier/Kaiming):
- Output logits $z_v$ have mean near 0 ($\mathbb{E}[z_v] \approx 0$) with near-zero variance;
- Output probabilities approach a uniform distribution:
  $$P(y = v) = \frac{e^{z_v}}{\sum_{j=1}^V e^{z_j}} \approx \frac{e^0}{V \cdot e^0} = \frac{1}{V}$$
- Cross-entropy strictly simplifies to the natural logarithm:
  $$\mathcal{L}_{\text{init}} = -\ln \left( \frac{1}{V} \right) = \ln V$$

#### (2) Second-Order Taylor Expansion with Initial Logit Variance
If $z_v \sim \text{i.i.d. } \mathcal{N}(0, \sigma_z^2)$, second-order Taylor expansion over Log-Sum-Exp yields:

$$\mathbb{E}[\mathcal{L}_{\text{init}}] = -\mathbb{E}[z_c] + \mathbb{E}[\ln S] \approx \ln V + \frac{\sigma_z^2}{2} + \mathcal{O}(\sigma_z^4)$$

With $\sigma_z \approx 0.02 \sim 0.1$, $\frac{\sigma_z^2}{2} < 0.005$, proving **$\mathbb{E}[\mathcal{L}_{\text{init}}] \approx \ln V$ with extreme precision**.

---

### 2. Standard Foundation LLM Theoretical Initial Loss Reference

| Model Family | Vocab Size $V$ | Theoretical Initial Loss: $\ln V$ (nats/token) |
|---|---|---|
| **GPT-2 / GPT-3** | $50,257$ | $\ln(50257) \approx \mathbf{10.825}$ |
| **LLaMA-1 / LLaMA-2 / Mistral-7B** | $32,000$ | $\ln(32000) \approx \mathbf{10.373}$ |
| **LLaMA-3 / LLaMA-3.1 / LLaMA-3.3** | $128,256$ | $\ln(128256) \approx \mathbf{11.762}$ |
| **DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1** | $129,280$ | $\ln(129280) \approx \mathbf{11.770}$ |
| **Qwen-2 / Qwen-2.5** | $152,064$ | $\ln(152064) \approx \mathbf{11.932}$ |
| **Gemma / Gemma-2** | $256,000$ | $\ln(256000) \approx \mathbf{12.453}$ |

---

### 3. Theoretical Impact of Temperature ($T$) & Label Smoothing ($\epsilon$)

1. **Temperature $T$**: $\mathcal{L}(T) = -\frac{z_c}{T} + \ln \left( \sum_{v=1}^V e^{z_v / T} \right)$. As $T \to \infty$, variance is compressed and $\mathcal{L} \to \ln V$.
2. **Label Smoothing $\epsilon$**: With smoothed targets $q(v) = (1-\epsilon)\mathbb{I}(v=c) + \frac{\epsilon}{V}$:
   $$\mathcal{L}_{\text{smooth}} = -(1-\epsilon)\ln \left(\frac{1}{V}\right) - \frac{\epsilon}{V} \cdot V \ln \left(\frac{1}{V}\right) = \ln V$$
   > 💡 **Core Invariance**: **Regardless of the label smoothing factor $\epsilon$, theoretical expected initial loss at random initialization is strictly invariant at $\ln V$**.

---

### 4. High-Yield Interview Multiple-Choice Question

<details class="exercise">
<summary><span class="q-label">Q1 · Multiple Choice</span> <span class="q-text">When starting pre-training for LLaMA-3 ($V = 128,256$) on a 1000-GPU cluster at Step 0 before any optimizer steps, which of the following statements is <strong>strictly correct</strong>?</span></summary>

- [ ] **A.** Step 0 loss should be close to 0.0; any value above 1.0 indicates severe initialization divergence.
- [ ] **B.** Expected initial loss is $\ln(128256) \approx 11.76$; an observed $L_0 = 3.2$ indicates rapid and superior model convergence.
- [x] **C.** Expected initial loss is $\ln(128256) \approx 11.76$; an observed $L_0 = 3.2$ strongly indicates a critical bug such as missing causal attention masks (future token leakage) or unmasked target padding tokens (`ignore_index=-100`).
- [ ] **D.** If label smoothing $\epsilon = 0.1$ is applied, initial loss drops significantly to $(1-\epsilon)\ln V \approx 10.58$.

> 💡 **Explanation**:
> - **Correct Answer: C**.
>   1. Under uniform initialization, $P(y=v) \approx \frac{1}{V} \implies \mathcal{L}_{\text{init}} = \ln V = \ln(128256) \approx 11.762$.
>   2. If $L_0 \ll \ln V$, the model has causal mask leakage (acting as bidirectional attention) or padding tokens are contributing zero loss artificially.
>   3. Option D is mathematically false because label smoothing under uniform probability distribution preserves the expectation $\ln V$ exactly.
</details>
"""

m3_pos_en = content_00b_en.find("## Module 3")
if m3_pos_en != -1:
    content_00b_en = content_00b_en[:m3_pos_en] + target_m3_en
else:
    content_00b_en += "\n\n" + target_m3_en

with open("notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.en.md", "w", encoding="utf-8") as f:
    f.write(content_00b_en)

print("Updated MLCoding00B zh and en")
