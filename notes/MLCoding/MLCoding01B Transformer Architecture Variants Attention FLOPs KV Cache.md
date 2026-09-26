# 01B · Transformer 架构变体与算力分析

系统解析 Transformer 算子底层的五大核心模块：架构分类与掩码矩阵、MHA 张量流动与数学推导、FLOPs 严密分解与计算瓶颈体制转移（Regime Shift）、自回归推理与 KV Cache 显存模型、以及长序列硬件感知优化全景（FlashAttention、三大效率路线、头数压缩与工程速查清单）。

---

## 模块一：Transformer 架构分类学与注意力掩码模式

```text
三大架构注意力掩码模式对比：
Encoder-Only (BERT):           Decoder-Only (GPT / LLaMA):     Encoder-Decoder (T5 / BART):
┌───┬───┬───┬───┐             ┌───┬───┬───┬───┐               ┌───┬───┬───┬───┐
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ -∞│ -∞│ -∞│               │ 0 │ 0 │ 0 │ 0 │  (Encoder: 全双向)
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               ├───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ -∞│ -∞│               │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               └───┴───┴───┴───┘
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ 0 │ -∞│               ┌───┬───┬───┬───┐
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               │ 0 │ -∞│ -∞│ -∞│  (Decoder: 因果掩码)
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ 0 │ 0 │               └───┴───┴───┴───┘
└───┴───┴───┴───┘             └───┴───┴───┴───┘               + Cross-Attention: Q_dec × K_enc^T
[全双向无掩码 M_ij = 0]       [因果下三角掩码 j > i 时 -∞]   [双向编码 + 因果解码 + 跨注意力]
```

### 三大架构形态高密度对比

| 架构形态 | 注意力掩码矩阵 $M_{ij}$ | 输入与解码范式 | KV Cache 状态 | 典型代表与适用场景 |
|---|---|---|---|---|
| **Encoder-Only** | 全双向（$M_{ij} = 0$） | 非自回归，单次前向并行处理全部 $S$ 个 Token | **无**（单次前向直接输出） | BERT, RoBERTa（文本分类、实体识别、向量表征） |
| **Decoder-Only** | 因果下三角（$j > i$ 时 $M_{ij} = -\infty$） | 自回归生成，逐 Token 依赖历史上下文 | **必须维护**（缓存历史 Key/Value 避免重复计算） | GPT-4, LLaMA-3, Qwen, DeepSeek（通用大模型、代码生成、推理） |
| **Encoder-Decoder** | Encoder 双向 + Decoder 因果 + **Cross-Attention** | 双向编码源端，自回归生成目标端 | **双份缓存**（Encoder 静态缓存 + Decoder 动态缓存） | T5, BART, Whisper（机器翻译、文本摘要、ASR） |

#### Cross-Attention（跨注意力）计算本质
- **Query（$Q$）**：来源于 Decoder 上一层的隐层状态 $Q_{\text{dec}} = X_{\text{dec}} W_Q \in \mathbb{R}^{B \times S_{\text{dec}} \times D}$；
- **Key（$K$）与 Value（$V$）**：来源于 Encoder 顶层输出 $K_{\text{enc}} = X_{\text{enc}} W_K, \ V_{\text{enc}} = X_{\text{enc}} W_V \in \mathbb{R}^{B \times S_{\text{enc}} \times D}$；
- **执行特性**：Encoder 的 $K_{\text{enc}}, V_{\text{enc}}$ 在 Prefill 阶段仅计算一次并缓存，解码全程被所有解码步反复共享读取。

---

## 模块二：多头注意力（MHA）数学推导、张量形状演化与执行流水线

设批量大小为 $B$，序列长度为 $S$，模型隐藏维度为 $D$，注意力头数为 $H$，每个头的维度为 $d_k = D / H$。

```text
多头注意力 (MHA) 张量流动全景图：
输入 X (B, S, D)
  ├──> W_Q (D, D) ──> Q (B, S, D) ──> Reshape & Transpose ──> (B, H, S, d_k) ┐
  ├──> W_K (D, D) ──> K (B, S, D) ──> Reshape & Transpose ──> (B, H, S, d_k) ┼──> Scaled Dot-Product & Softmax
  └──> W_V (D, D) ──> V (B, S, D) ──> Reshape & Transpose ──> (B, H, S, d_k) ┘     │
                                                                                    ▼
                                                                           Score A (B, H, S, S)
                                                                                    │ × V (B, H, S, d_k)
                                                                                    ▼
                                                                           Context (B, H, S, d_k)
                                                                                    │
                                                                           Transpose & Concat (B, S, D)
                                                                                    │ × W_O (D, D)
                                                                                    ▼
                                                                           Output (B, S, D)
```

### 详细六步张量变换流程

1. **输入与线性投影（Linear Projections）**：
   输入张量 $\mathbf{X} \in \mathbb{R}^{B \times S \times D}$，权重矩阵 $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$：

$$Q = \mathbf{X}W_Q, \quad K = \mathbf{X}W_K, \quad V = \mathbf{X}W_V \quad \in \mathbb{R}^{B \times S \times D}$$

2. **多头拆分与轴转置（Head Reshape & Transposition）**：
   将隐藏维度 $D$ 拆分为 $H$ 个头，每个头维度为 $d_k$：

$$\text{Reshape: } (B, S, D) \to (B, S, H, d_k) \xrightarrow{\text{Transpose (1, 2)}} (B, H, S, d_k)$$

3. **缩放点积注意力得分（Scaled Dot-Product Attention Scores）**：

$$A = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times S \times S}$$

   > **为什么必须除以 $\sqrt{d_k}$？**  
   > 假设 $Q$ 和 $K$ 的各个分量是均值为 0、方差为 1 的独立随机变量，则点积 $\sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，**方差为 $d_k$**。如果不进行缩放，在高维情况下点积数值会变得极大，导致 Softmax 函数进入**梯度饱和区（极度平坦）**，反向传播时梯度几乎消失。除以 $\sqrt{d_k}$ 将方差重新拉回 1，保持 Softmax 的灵敏度。

4. **因果掩码与归一化（Causal Masking & Softmax）**：

$$\tilde{A} = \text{softmax}(A + M), \quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

5. **Value 聚合与头拼接（Value Aggregation & Concatenation）**：

$$\text{Head}_h = \tilde{A}_h V_h \in \mathbb{R}^{B \times H \times S \times d_k} \xrightarrow{\text{Transpose \& Reshape}} \text{MultiHead} \in \mathbb{R}^{B \times S \times D}$$

6. **输出投影（Output Projection）**：

$$\text{Output} = \text{MultiHead} \cdot W_O \in \mathbb{R}^{B \times S \times D}, \quad W_O \in \mathbb{R}^{D \times D}$$

---

## 模块三：MHA 计算复杂度 FLOPs 严密分解与体制转移

在算法面试与系统设计中，精确估算单层 Attention 的浮点运算次数（FLOPs，乘加各算 1 次，一次乘加 = 2 FLOPs）至关重要。

### 1. FLOPs 严密分解（以单样本 $B=1$ 为例）

1. **四次线性投影（$Q, K, V, W_O$）**：
   每个投影为 $(S \times D) \times (D \times D)$ 的矩阵乘法：
   $$\text{FLOPs}_{\text{proj}} = 4 \times (2 \times S \times D \times D) = \mathbf{8 S D^2} \implies \mathcal{O}(S D^2)$$
2. **计算注意力得分矩阵（$Q K^T$）**：
   $H$ 个头，每个头做 $(S \times d_k) \times (d_k \times S)$ 的矩阵乘法：
   $$\text{FLOPs}_{QK^T} = H \times (2 \times S \times d_k \times S) = 2 S^2 (H \cdot d_k) = \mathbf{2 S^2 D} \implies \mathcal{O}(S^2 D)$$
3. **Value 加权聚合（$\tilde{A} V$）**：
   $H$ 个头，每个头做 $(S \times S) \times (S \times d_k)$ 的矩阵乘法：
   $$\text{FLOPs}_{AV} = H \times (2 \times S \times S \times d_k) = \mathbf{2 S^2 D} \implies \mathcal{O}(S^2 D)$$
4. **单层 MHA 总计算量**：

$$\text{Total FLOPs}_{\text{MHA}} = 8 S D^2 + 4 S^2 D$$

---

### 2. 计算瓶颈体制转移（Regime Shift Analysis）

```text
MHA 计算量主导项随序列长度 S 的变化：
FLOPs
  ▲
  │                                    /  O(S² D) Attention 矩阵乘法
  │                                   /   (长文本场景，二次方爆炸)
  │                                  /
  │            O(S D²) 线性投影     /
  │           (短文本场景，占主导) /
  │         ─────────────────────/
  │                             /
  └────────────────────────────┴─────────────► 序列长度 S
                             S ≈ 2D (临界交叉点)
```

- **短序列常规体制（$S < 2D$，如 $S=2048, D=4096$）**：
  $8 S D^2 > 4 S^2 D$，**线性投影 $O(S D^2)$ 占据绝大部分计算量**（占比 $>80\%$）。此时优化重点是 GEMM 矩阵乘法效率。
- **长序列长文本体制（$S \gg D$，如 $S=32K \sim 128K, D=4096$）**：
  $4 S^2 D \gg 8 S D^2$，**注意力矩阵计算 $O(S^2 D)$ 呈二次方爆炸并成为绝对算力瓶颈**。此时必须依赖 FlashAttention、稀疏注意力或线性注意力进行优化。

---

## 模块四：自回归推理机制与 KV Cache 显存模型

### 1. Prefill 阶段 vs. Decode 阶段

大模型推理在计算特征上分为两个截然不同的阶段：

```text
推理双阶段特征对比：
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 推理阶段                │ 硬件行为与瓶颈特征                                                     │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Prefill 阶段         │ • 输入所有 Prompt Token（长序列），全并行计算 Q, K, V                  │
│    (Prompt 预填充)      │ • 填充并生成初始 KV Cache                                              │
│                         │ • 算术强度高，属于**算力受限（Compute-Bound）**                        │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Decode 阶段          │ • 每次只输入上一步生成的 1 个 Token ($x_t \in \mathbb{R}^{1 \times D}$)│
│    (Token 逐字生成)     │ • 生成当前步的 $q_t, k_t, v_t$，将 $k_t, v_t$ 追加到 KV Cache 末尾     │
│                         │ • 每次生成 1 个 Token 都需从显存搬运整个历史 KV Cache 与全部权重       │
│                         │ • 算术强度极低（≈ 1 FLOP/Byte），属于**显存带宽受限（Memory-Bound）** │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

#### 为什么不需要 Cache Query（$Q$）？
- 当前时间步 $t$ 生成的查询向量 $q_t \in \mathbb{R}^{1 \times D}$，只需要与历史所有 Key 向量 $K_{\le t}$ 计算注意力得分；
- 在下一个时间步 $t+1$，新生成的 Token 会产生全新的查询向量 $q_{t+1}$；
- **历史上的旧查询向量 $q_1, q_2, \dots, q_t$ 永远不会再被未来任何步骤使用**，因此 $Q$ 的生命周期仅在当前时间步内，随用随弃，无需占用显存缓存。

---

### 2. KV Cache 显存占用精确数学公式

对于批量大小 $B$，当前上下文长度 $S$，模型总层数 $L$，KV 头数 $H_{KV}$，每个头的维度 $d_k$，每个参数占用字节数 $b$（例如 FP16/BF16 占用 $b=2$ 字节）：

$$\text{Memory}_{\text{KVCache}} = 2 \times B \times S \times L \times H_{KV} \times d_k \times b \quad \text{Bytes}$$

> *公式前面的系数 $2$ 代表 Key 和 Value 两个张量。*

#### 工业级实例测算（LLaMA-3-70B）
- 参数配置：$L=80, D=8192, H_Q=64, H_{KV}=8 \text{ (GQA)}, d_k=128, b=2 \text{ (BF16)}$
- 单 Token 的 KV Cache 显存消耗：
  $$\text{Per-Token Memory} = 2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ Bytes} \approx \mathbf{320 \text{ KB / Token}}$$
- 当并发批次 $B=64$，上下文长度 $S=8192$ 时：
  $$\text{Total KV Cache} = 64 \times 8192 \times 320 \text{ KB} \approx \mathbf{167.77 \text{ GB}}$$
  **KV Cache 显存甚至直接超过了 70B 模型本身的权重显存（140 GB）！**

---

## 模块五：长序列与硬件感知注意力优化体系

### 1. 长序列 Attention 核心矛盾与三大物理墙

当序列长度进入长文本体制（$S \gg 2D$）时，注意力矩阵交互的二次方计算与显存占用超越线性投影，模型在训练与 Serving 阶段同时撞上三大物理墙：

```text
长序列三大物理瓶颈特征：
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ 1. 算力二次方 (Compute) │ 2. 训练激活显存 (Memory)│ 3. 推理 KV Cache (IO带宽│
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • S 从 4K 拓展到 128K:  │ • 物化 S×S 的 Logits 与 │ • 显存容量 O(B·S·L·d)   │
│   长度扩大 32 倍         │   Attention Score 矩阵   │   呈线性持续膨胀        │
│ • Attn FLOPs 暴增 1024倍│ • 经典实现 O(S²) 导致   │ • Decode 每生成 1 Token │
│ • Prefill 耗时呈二次方  │   GPU 发生显存溢出(OOM) │   需读取全量历史 KV     │
│   剧烈爆炸              │ • 反向传播需保留得分梯度│ • 算术强度极低，带宽受限│
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

为了从根本上化解这一矛盾，业界形成了**系统补丁 $\to$ 三大算法路线（改计算复杂度 / 改系统切分）$\to$ 混合收敛形态（Hybrid）**的演进脉络：

![[assets/attention-efficiency-landscape.png|长序列 Attention 效率演进脉络]]

---

### 2. 系统/算子层补丁：FlashAttention（Exact，不改变渐近复杂度）

FlashAttention（Dao et al.）是现代大模型基础设施级系统优化，核心特征为**数学结果完全等价（Exact Attention，零精度损失）**。

#### (1) 三大核心机制
- **Tiling（SRAM 分块平铺）**：将输入 $Q, K, V$ 划分为适合 GPU 片上高速 SRAM（通常为 100KB~228KB/SM）大小的子块，矩阵乘法与归一化计算均在 SRAM 内部完成；
- **Online Softmax（增量动态归一化）**：维护流式局部最大值 $m_i$ 与归一化因子 $l_i$，在流式加载子块时动态更新局部 Attention 输出，**彻底消除在慢速高带宽显存（HBM）中显式读写 $S \times S$ 激活值矩阵的过程**；
- **Recomputation in Backward（反向重算）**：反向传播时不保存前向的 $S \times S$ 激活图，而在 SRAM 中极速重算，将训练激活显存从 $\mathcal{O}(S^2)$ 压低至 $\mathcal{O}(S D)$。

#### (2) 物理边界与权衡
- **解决的问题**：消除训练期 $\mathcal{O}(S^2)$ 激活值显存 OOM 危机；将 HBM 访存复杂度从 $\mathcal{O}(S^2)$ 降为 $\mathcal{O}(S)$，MFU 提升 2~4 倍；
- **无法解决的问题**：
  1. **总计算量依然严格为 $\mathcal{O}(S^2 D)$**：在超长上下文（如 1M+ tokens）的 Prefill 阶段，二次方的矩阵乘算力开销依然会导致严重的延迟爆炸；
  2. **自回归 Decode 访存墙依然存在**：每一步 Decode 生成仍需遍历全量历史 KV Cache，吞吐受限于 KV 搬运带宽；
  3. **无法降低推理 KV Cache 显存**：KV Cache 占用依然为 $\mathcal{O}(B \cdot S \cdot L \cdot D)$，随序列长度线性膨胀。

> [!TIP]
> **配套实战编程演练**：
> 想亲手编写 FlashAttention 的核心逻辑与 GPU 算子？前往 **[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.md#Exercise 7 · Flash Attention（分块 + online softmax）|ML Coding 03 · Exercise 7：Flash Attention 从 PyTorch 在线 Softmax 到 Triton GPU Kernel 算子级实现]]** 进行实战编码与数值验证。

### 3. 三大效率路线：改算法复杂度与改系统切分

为了彻底突破 FlashAttention 留下的“二次方计算量”与“自回归 Decode 阶段 KV 访存墙”，学术界与工业界衍生出三大效率路线。下表在相同的多维物理复杂度度量框架下，对三大路线与标准 Attention、FlashAttention 进行了统一度量与机制对比：

#### (0) 全景复杂度与硬件瓶颈多维对比矩阵

| 机制 / 效率路线 | 训练激活显存<br>(Activation Memory) | HBM 访存 IO 量<br>(Memory Traffic) | 前向计算量<br>(Forward FLOPs) | 反向计算量<br>(Backward FLOPs) | 自回归推理单步<br>(Decode Step 开销) | 硬件运行瓶颈<br>(Hardware Regime) | 核心物理代价与工程约束 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **标准 Attention<br>(Standard / Eager)**<br>🔗 *[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.md#Exercise 1 · MultiHeadAttention（双向，非因果）|练习 1 · MHA 原生实现]]* | $\mathcal{O}(S^2)$<br>物化保存完整 $S \times S$ 矩阵 | $\mathcal{O}(S^2 + S D)$<br>频繁往返 HBM 倾倒矩阵 | $4 S^2 D$<br>(因果掩码 $2 S^2 D$) | $8 S^2 D$<br>(因果掩码 $4 S^2 D$) | 显存：$\mathcal{O}(S D)$ 线性膨胀<br>计算：$2 S D$ 遍历历史 | **严重 Memory-Bound**<br>算术强度 $< 1$，带宽极度饥饿 | 训练长序列瞬间显存 OOM 崩溃；无任何片上数据复用。 |
| **FlashAttention<br>(Dao et al. 系统补丁)**<br>🔗 *[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.md#Exercise 7 · Flash Attention（分块 + online softmax）|练习 7 · FlashAttn 算子]]* | $\mathcal{O}(S \cdot D)$<br>丢弃中间激活，仅存 $O$ 与 $L_i$ | $\mathcal{O}\left(\frac{S^2 D^2}{M}\right) \approx \mathcal{O}(S D)$<br>SRAM 分块流水线融合 | $4 S^2 D$<br>(因果掩码 $2 S^2 D$) | $10 S^2 D$<br>(因果掩码 $5 S^2 D$)<br>重算浮点量增 ~25% | 显存：$\mathcal{O}(S D)$ 仍线性膨胀<br>计算：$2 S D$ 仍需遍历历史 | **Compute-Bound**<br>充分跑满 Tensor Core 脉动阵列 | **未降低总 FLOPs 二次方**；Prefill 耗时依然剧烈爆炸；自回归无法摆脱 KV 访存墙。 |
| **路线 A：原生稀疏注意力<br>(DeepSeek NSA / Sparse)**<br>🔗 *[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.md#Exercise 4 · Sliding Window Attention|练习 4 · 局部窗口]]*<br>⚡ *[[#DeepSeek NSA 原生稀疏注意力的硬件对齐设计与 Triton 算子实现|解析：NSA Triton 算子]]* | $\mathcal{O}(S \cdot D)$<br>仅维护粗筛与 Top-$k$ 活跃块 | $\mathcal{O}(S \cdot k_{\text{eff}} \cdot D)$<br>TMA 块级对齐连续加载 | $\approx 4 S \cdot k_{\text{eff}} \cdot D$<br>(降为与序列近似线性) | $\approx 8 S \cdot k_{\text{eff}} \cdot D$<br>(计算量压低数倍至数十倍) | 显存：$\mathcal{O}(k_{\text{eff}} \cdot D)$<br>计算：$2 k_{\text{eff}} D$ 只算活跃块 | **Compute-Bound**<br>(依赖 64-token 块硬件对齐) | 必须保证硬件块对齐（64-token）；离散 Token 剪枝会导致非合并访存灾难。 |
| **路线 B：线性与状态空间<br>(DeltaNet / RetNet / SSM)**<br>🔗 *[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.md#Exercise 5 · Linear Attention|练习 5 · 线性注意]]*<br>⚡ *[[#DeltaNet 联想记忆更新规则、Chunkwise 块级并行扫描与 Triton 算子实现|解析：DeltaNet 算子]]* | $\mathcal{O}(S \cdot D)$<br>Chunkwise 块级传递 $D \times D$ 状态 | $\mathcal{O}(S \cdot D)$<br>单次流式线性扫描，极低 IO | $\approx 4 S D^2$<br>(长文本下 $S \gg D$，计算降千倍) | $\approx 8 S D^2$<br>(严格线性复杂度) | 显存：$\mathcal{O}(D^2)$ **严格恒定 $\mathcal{O}(1)$**<br>计算：$\mathcal{O}(D^2)$ **严格恒定 $\mathcal{O}(1)$** | **Compute-Bound** (训练)<br>**Throughput-Flat** (推理) | 纯核化易产生注意力稀释与容量饱和；需引入 Delta 规则在线擦除投影，少样本检索略弱于 Softmax。 |
| **路线 C：分布式上下文并行<br>(RingAttention / CP)**<br>⚡ *[[#RingAttention 环形 P2P 双缓冲异步重叠、因果块跳过与流式 Softmax 融合实现|解析：RingAttn 分布式]]* | 单卡 $\mathcal{O}\left(\frac{S}{P} \cdot D\right)$<br>随 GPU 卡数 $P$ 严格线性均摊 | 单卡片上 SRAM 极速流转；跨卡走 NVLink/RDMA 环 | 单卡 $\approx \frac{2 S^2 D}{P}$<br>集群总算力严格等价 | 单卡 $\approx \frac{5 S^2 D}{P}$<br>集群总算力严格等价 | 单卡切片持有 $\mathcal{O}\left(\frac{S}{P} \cdot D\right)$<br>环形异步流动检索 | **Overlap Compute-Bound**<br>(双缓冲重叠通信耗时) | 强依赖高速网络互联带宽；切片过小会导致通信无法被计算完全掩盖（退化为 Comm-Bound）。 |

---

#### 路线 A：稀疏注意力（Sparse Attention，剪枝图）
- **核心思想与“Top-$k$ 悖论”**：
  - **核心痛点**：若“先算全量 $S \times S$ 的 Attention 再做 Top-$k$”，计算量依然是严格的 $\mathcal{O}(S^2 D)$，显存峰值仍是 $\mathcal{O}(S^2)$，稀疏不仅无法降低开销，还会额外引入巨大的全量排序延迟；
  - **破解逻辑**：绝不能直接算全量 Attention！业界衍生出三大稀疏化选拔机制：
    1. **静态拓扑规则（Static Heuristics，如 Longformer / BigBird）**：完全不做动态 Top-$k$，硬编码局部窗口（Local Window）+ 跨步空洞（Dilated Window）+ 固定全局锚点（Global Tokens），算力降至 $\mathcal{O}(S \cdot w)$，但对复杂长程语义完全盲目；
    2. **哈希/聚类分桶（Clustering/LSH，如 Reformer）**：通过随机超平面将向量哈希进桶内，只在同桶内算注意力，但因桶大小严重不均（Bucket Imbalance）导致 GPU 线程分化与 Padding 浪费，已被工业界淘汰；
    3. **分层金字塔粗筛（Hierarchical Coarse-to-Fine，以 DeepSeek NSA 为代表）**：现代大模型的最优解，先用超轻量的“块级压缩（AvgPool 压缩 64 倍）”以 $<1.5\%$ 的算力打出粗筛分数，选出得分最高的 Top-$k$ 个**连续整块（Block-level Top-$k$）**，再仅将这几个块载入片上 SRAM 执行高精度 FlashAttention，实现严格线性的计算与显存。
- **优缺点**：保留 Softmax 指数放大与注意力锐度；但必须做块级硬件对齐，否则离散访存开销会抵消算力节约。

> [!TIP]
> **配套实战演练与底层算子**：
> - 局部稀疏基石算法：前往 **[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.md#Exercise 4 · Sliding Window Attention|ML Coding 03 · Exercise 4：Sliding Window Attention]]** 进行滑动窗口因果注意力手写；
> - 硬件对齐 Triton 算子：点击下方展开折叠块查看 **DeepSeek NSA 原生稀疏注意力的 Triton GPU 算子完整实现**。

<details class="technical-deep-dive">
<summary><span class="deep-dive-badge">Kernel 深度解析</span><span class="deep-dive-title">DeepSeek NSA 原生稀疏注意力的硬件对齐设计与 Triton 算子实现</span></summary>
<div class="deep-dive-content">

##### 1. 传统 Token 级稀疏在现代 GPU 上的硬件失配困境
在早期稀疏算法（如动态 Top-$k$ Token 剪枝）中，算法虽然消减了理论 FLOPs，但在现代 GPU（如 Hopper H100、Blackwell B200）上吞吐往往**不升反降**：
- **非连续访存与合并缺失（Memory Coalescing Breakdown）**：GPU HBM3/HBM3e 的峰值带宽依赖于 Warp（32 线程）发起连续 128 字节的 Cache Line 请求。逐 Token 动态稀疏索引会触发离散 Gather/Scatter 寻址，导致实际有效带宽跌落至峰值的 $10\%$ 以下；
- **Tensor Core 脉动阵列失配**：现代 GPU 的 Tensor Core（如 Hopper `wgmma` 架构）以固定尺寸的密集矩阵分块（$64 \times 64$ 或 $128 \times 128$）直接在片上共享内存（SRAM）中运作。细粒度离散 Token 无法填充张量单元，硬件被迫回退到低吞吐的通用 CUDA Cores；
- **Warp 线程分化（Branch Divergence）**：同 Warp 内不同线程如果执行不同长度或位置的稀疏索引分支，会导致严重流水线气泡。

##### 2. DeepSeek NSA 的三路硬件对齐设计：Top-$k$ 是如何无开销计算的？
DeepSeek NSA（Native Sparse Attention）彻底抛弃细粒度逐 Token 寻址，强制以**连续块（Block-Aligned，通常为 $L_b = 64$ 个连续 Token）**作为硬件流转基元，其 Top-$k$ 完整运算流程如下：

1. **第一步：Key 序列块级均值压缩（Spatial Pooling）**：
   - 原始 Key: $K \in \mathbb{R}^{S \times D}$；
   - 将每连续 $L_c = 64$ 个 Token 压缩为单向量：$K^{\text{cmp}} = \text{AvgPool}_{L_c}(K) \in \mathbb{R}^{\frac{S}{L_c} \times D}$；
   - 序列长度瞬间压缩 64 倍（例如 128K 长度仅余 2048 个块代表向量）。
2. **第二步：轻量粗筛扫描（Coarse-Grained Attention Scan）**：
   - Query 仅与压缩后的键向量做密集点积：$\text{Scores}_{\text{coarse}} = Q (K^{\text{cmp}})^T \in \mathbb{R}^{S \times \frac{S}{L_c}}$；
   - **算力开销仅为全量的 $1/L_c = 1/64 \approx 1.5\%$**，张量连续规整，直接由 Tensor Core 脉动阵列极速扫过。
3. **第三步：块级 Top-$k$ 索引截取（Block-level Top-$k$ Selection）**：
   - 对每个 Query 块的粗筛打分，选出得分最高的 $n_s$ 个**整块索引**（例如 $n_s = 4$ 或 $8$ 个 Block）：$\mathcal{I}_{\text{sel}} = \text{TopK}(\text{Scores}_{\text{coarse}}, n_s)$；
   - 绝不逐 Token 离散挑选，保证每次选中的都是连续 $L_b$ 个 Token 的整块内存。
4. **第四步：细粒度片上精算（Fine-Grained Selected Blocks in SRAM）**：
   - 依据 $\mathcal{I}_{\text{sel}}$，仅将选中的 $n_s$ 个块（总计 $n_s \times L_b = 256 \sim 512$ 个 Token）的**原始高精 Key/Value** 从 HBM 通过 Hopper TMA 异步加载至片上 SRAM；
   - 在 SRAM 内部调用 Tensor Core 执行高精度密集 FlashAttention 点积与 Softmax。
5. **第五步：局部滑动窗口保底（Sliding Window）与统一 Online Softmax**：
   - 维持最近 $W$ 个 Token（如 512 个，即 8 个连续块）的密集因果注意力，保全局部语法词法；
   - 粗筛得分仅用于选块，不参与 Softmax 归一化；细粒度选中块与滑动窗口块在 SRAM 寄存器中共享同一组动态累加器 $(m_i, l_i, \text{acc}_o)$，流式融合写回。

##### 3. NSA 前向 Triton Kernel 核心骨架

```python
import triton
import triton.language as tl

@triton.jit
def _nsa_fwd_kernel(
    Q, K, V, SelectedIndices, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_sb, stride_sh, stride_sm, stride_sk,
    sm_scale,
    Q_LEN: tl.constexpr,
    NUM_SELECTED_BLOCKS: tl.constexpr,  # 例如粗筛选出的 Top-4 块
    BLOCK_SIZE: tl.constexpr,           # 硬件对齐块大小 (64 tokens)
    HEAD_DIM: tl.constexpr,             # 头隐层维度 (64 或 128)
    BLOCK_M: tl.constexpr,              # Query Tiling 大小 (64)
):
    # 网格配置：(cdiv(Q_LEN, BLOCK_M), NUM_HEADS, BATCH)
    pid_m = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    # 1. 片上 SRAM 载入 Query 块 [BLOCK_M, HEAD_DIM]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN, other=0.0)

    # 2. 寄存器初始化 Online Softmax 累加状态
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. 遍历选中的整块连续索引（硬件合并读入）
    sel_base = SelectedIndices + batch_idx * stride_sb + head_idx * stride_sh + pid_m * stride_sm
    
    for k_idx in range(NUM_SELECTED_BLOCKS):
        block_id = tl.load(sel_base + k_idx * stride_sk)
        
        # 内存连续块偏移计算（完全消除逐 Token 散列寻址）
        offs_n = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        
        # 利用 Tensor Core 块级对齐载入 SRAM
        k_block = tl.load(k_ptrs) # [HEAD_DIM, BLOCK_SIZE]
        v_block = tl.load(v_ptrs) # [BLOCK_SIZE, HEAD_DIM]

        # 计算块内密集点积 [BLOCK_M, BLOCK_SIZE]
        s_ij = tl.dot(q, k_block) * sm_scale

        # 流式更新局部最大值与归一化分母
        m_curr = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_curr)
        p = tl.exp(s_ij - m_curr[:, None])

        # 累加 Attention 加权和与 Softmax 因子
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_curr

    # 4. 局域滑动窗口块类似流程（连续遍历并在 SRAM 融合更新 m_i, l_i, acc_o）...

    # 5. 归一化并写回 HBM
    out_ptrs = Out + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, (acc_o / l_i[:, None]).to(Out.dtype.element_ty), mask=offs_m[:, None] < Q_LEN)
```

</div>
</details>

#### 路线 B：低秩与核化线性注意力（Linear Attention & Delta Rule，改写结合律）
- **核心思想**：利用非线性映射 $\phi(\cdot)$ 解耦 Softmax 为内积 $\phi(Q)\phi(K)^T$，借由乘法结合律调整计算顺序：
  $$\text{Standard: } (Q K^T) V \in \mathcal{O}(S^2 D) \implies \text{Linear: } \phi(Q) \left(\phi(K)^T V\right) \in \mathcal{O}(S \cdot D^2)$$
- **自回归推理 RNN 恒定态**：
  $$S_t = S_{t-1} + \phi(k_t) v_t^T \in \mathbb{R}^{d \times d}, \quad o_t = \phi(q_t) S_t$$
  单步推理维护固定维度状态 $S_t$，**显存与计算复杂度均为 $\mathcal{O}(1)$**，无需线性膨胀的 KV Cache。
- **突破“容量饱和”：Delta 学习规则（DeltaNet / RetNet）**：
  纯累加缺乏擦除机制会导致历史无关信息填满状态（Attention Dilution）。引入经典关联记忆擦除机制：
  $$W_t = W_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
  新 Key 写入前先从记忆矩阵中扣除旧值投影，辅以 Chunkwise 并行扫描算子，召回率逼近标准 Softmax。
- **工业落地形态**：**Hybrid 混合架构**（如 Jamba、Nemotron-4），周期性交替堆叠 SSM/线性层与因果 Attention 层，兼顾恒定吞吐与复杂检索精度。

> [!TIP]
> **配套实战演练与底层算子**：
> - 线性注意力基石算法：前往 **[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.md#Exercise 5 · Linear Attention|ML Coding 03 · Exercise 5：Linear Attention]]** 编写 $\phi(Q)(\phi(K)^T V)$ 特征映射与因果前缀和；
> - Chunkwise 并行 Triton 算子：点击下方展开折叠块查看 **DeltaNet 联想记忆更新规则与 Chunkwise Triton 算子完整实现**。

<details class="technical-deep-dive">
<summary><span class="deep-dive-badge">Kernel 深度解析</span><span class="deep-dive-title">DeltaNet 联想记忆更新规则、Chunkwise 块级并行扫描与 Triton 算子实现</span></summary>
<div class="deep-dive-content">

##### 1. 经典线性注意力容量饱和与 Delta 擦除数学本质
在朴素核化线性注意力中，递推式为纯累加：$S_t = S_{t-1} + k_t v_t^T$。
- **容量饱和（Capacity Saturation）**：记忆状态 $S_t \in \mathbb{R}^{d \times d}$ 是外积的线性叠加。当序列长度 $t \gg d$ 时，状态矩阵秩饱和，早期噪声无法被主动遗忘，导致模型在少样本（In-Context Learning）与检索任务上发生严重“注意力稀释（Attention Dilution）”；
- **Delta 学习规则（Online Associative Gradient Descent）**：
  DeltaNet 将单步写入建模为对目标记忆的在线纠错。定义预测损失：
  $$\mathcal{L}_t = \frac{1}{2} \| S_{t-1} k_t - v_t \|_2^2$$
  对记忆状态求梯度并进行单步步长为 $\beta_t \in [0, 1]$ 的梯度更新：
  $$S_t = S_{t-1} - \beta_t \nabla_S \mathcal{L}_t = S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
  其中矩阵 $(I - \beta_t k_t k_t^T)$ 类似 Householder 投影变换，其在写入新特征 $v_t$ 之前，**在几何上将旧记忆沿 $k_t$ 空间的分量进行主动投影擦除**。

##### 2. 训练并行化悖论与 Chunkwise 并行扫描算法
- **并行化悖论**：在推理阶段，单步状态更新是纯粹的 $\mathcal{O}(1)$；但在预训练阶段，$S_t$ 强依赖于 $S_{t-1}$ 的序列因果递推。若在 GPU 上顺序循环执行 $S=4096$ 步，将导致 GPU SM 算力被严重串行化阻塞；
- **Chunkwise 分块并行解耦**：将序列划分为大小为 $C = 64$（对齐硬件 Tile）的子块。
  1. **块内矩阵化求解（Intra-Chunk Fast Solve）**：
     块内部的因果级联通过下三角矩阵方程在 SRAM 内部一次性解出：
     $$V_{\text{new}} = (I + \text{tril}(\beta K K^T, -1))^{-1} (\beta \odot V)$$
     由于 $C=64$ 极小，逆矩阵 $(I + L)^{-1}$ 可直接在片上 SRAM 中利用纽曼级数展开（$I - L + L^2$）或精确三角代换用 Tensor Core 极速展开；
     块内自注意力输出为：$O_{\text{intra}} = \text{tril}(Q K^T) V_{\text{new}}$；
  2. **块间状态转移（Inter-Chunk State Transfer）**：
     块之间的隐状态传递等价于宏观 RNN：
     $$S_c = S_{c-1} A_c + B_c$$
     其中 $A_c = \prod_{t \in c} (I - \beta_t k_t k_t^T) \in \mathbb{R}^{d \times d}$ 为块衰减累积矩阵，$B_c = K_c^T V_{\text{new}}$。
     宏观状态在每个 Chunk 只需更新一次，总步骤缩减为 $S / C$（例如 $4096 / 64 = 64$ 步），使得跨块递推开销可以被忽略，兼得 RNN 的线性复杂度与 Transformer 的高 Tensor Core 硬件利用率。

##### 3. DeltaNet Chunkwise Triton Kernel 实现骨架

```python
import triton
import triton.language as tl

@triton.jit
def _deltanet_chunk_fwd_kernel(
    Q, K, V, Beta, Out,
    stride_b, stride_h, stride_s, stride_d,
    CHUNK_SIZE: tl.constexpr, # 硬件对齐块大小 (64)
    DIM: tl.constexpr,        # 隐状态维度 (64 或 128)
    NUM_CHUNKS: tl.constexpr  # S // CHUNK_SIZE
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    base_ptr = batch_idx * stride_b + head_idx * stride_h

    # 1. 初始化片上寄存器记忆状态 S [DIM, DIM]
    offs_d1 = tl.arange(0, DIM)
    offs_d2 = tl.arange(0, DIM)
    s_state = tl.zeros([DIM, DIM], dtype=tl.float32)

    # 2. 宏观跨块循环 (S / CHUNK_SIZE 次)
    for c_idx in range(NUM_CHUNKS):
        offs_c = c_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
        
        # 加载本 Chunk 的 Q, K, V, Beta 到 SRAM
        q = tl.load(Q + base_ptr + offs_c[:, None] * stride_s + offs_d1[None, :] * stride_d)
        k = tl.load(K + base_ptr + offs_c[:, None] * stride_s + offs_d1[None, :] * stride_d)
        v = tl.load(V + base_ptr + offs_c[:, None] * stride_s + offs_d1[None, :] * stride_d)
        beta = tl.load(Beta + base_ptr + offs_c[:, None] * stride_s) # [CHUNK_SIZE, 1]

        # 3. 计算来自历史记忆的输出贡献：O_inter = Q @ S_{c-1}
        o_inter = tl.dot(q, s_state) # [CHUNK_SIZE, DIM]

        # 4. 块内因果修正（Intra-Chunk Delta Inversion）
        # 构建因果 Gram 矩阵: G = tril(beta * K @ K.T)
        gram = tl.dot(k, tl.trans(k)) * beta
        mask_tril = offs_c[:, None] > offs_c[None, :]
        gram_tril = tl.where(mask_tril, gram, 0.0)

        # 纽曼级数一阶展开在 SRAM 快速求解有效值向量
        v_eff = beta * v
        v_eff = v_eff - tl.dot(gram_tril, v_eff)

        # 计算块内自注意力交互贡献：O_intra = tril(Q @ K.T) @ v_eff
        qk = tl.dot(q, tl.trans(k))
        qk_causal = tl.where(mask_tril, qk, 0.0)
        o_intra = tl.dot(qk_causal, v_eff)

        # 5. 写回本 Chunk 完整输出：Out = O_inter + O_intra
        tl.store(Out + base_ptr + offs_c[:, None] * stride_s + offs_d1[None, :] * stride_d, o_inter + o_intra)

        # 6. 更新全局隐状态 S_c = S_{c-1} (I - beta K^T K) + K^T v_eff
        decay = tl.dot(tl.trans(k), tl.dot(k, s_state)) * tl.mean(beta)
        s_state = s_state - decay + tl.dot(tl.trans(k), v_eff)
```

</div>
</details>

#### 路线 C：分块与系统级并行（Chunking & RingAttention，改系统切分）
- **核心思想**：算法上固定局部注意力块 $B \ll S$，或系统架构上将长序列切分分散至多张 GPU 流转。
- **RingAttention（Liu et al.）**：
  - 将序列切分为 $P$ 块分布于 $P$ 张 GPU，每卡持有本地 $Q$；
  - 核心流转：$K, V$ 块通过 GPU 环形拓扑（Ring P2P）跨卡流动；
  - **计算通信完全重叠（Compute-Comm Overlap）**：计算当前块注意力时，底层异步流水线同步发送与接收下一块 $K, V$，网络传输耗时被计算隐藏；
  - **Chunked Prefill**：将超长 Prompt 切片打散分批调度，防止单次大 Prefill 独占计算资源导致 Decode 任务出现排队毛刺（Head-of-Line Blocking）。

<details class="technical-deep-dive">
<summary><span class="deep-dive-badge">分布式系统深度解析</span><span class="deep-dive-title">RingAttention 环形 P2P 双缓冲异步重叠、因果块跳过与流式 Softmax 融合实现</span></summary>
<div class="deep-dive-content">

##### 1. 上下文并行（Context Parallelism）的核心痛点与 Ring 拓扑化解
面对百万级上下文（$S = 1\text{M} \sim 10\text{M}$），单张 GPU 的 80GB HBM 甚至无法容纳长序列的 KV Cache 与激活值。
- **AllGather 的显存爆炸**：若采用常规并行将全量 $K, V$ 广播至各卡，单卡显存开销立刻退化回 $\mathcal{O}(S)$，完全抵消多卡并行的意义；
- **Ring 拓扑流转（Ring P2P）**：将世界大小为 $P$ 的 GPU 排列为环形逻辑拓扑（$0 \to 1 \to \dots \to P-1 \to 0$）。
  - 每张 GPU 仅持有局部序列切片 $S_{\text{local}} = S / P$ 的 $Q, K, V$；
  - 整个流转过程中，本地 Query 切片 $Q_{\text{local}}$ 永驻本地 SRAM/HBM；
  - $K, V$ 切片以块为单位沿环异步流转，各卡仅在计算时借用，用完即换，**单卡显存占用恒定维持在严格的 $\mathcal{O}(S / P)$**。

##### 2. 通信计算完全重叠（Double-Buffering Overlap）条件
通过维护 Ping-Pong 接收双缓冲：
- **流水线步骤 $s$**：
  1. **异步通信引擎**：在后台通过 NCCL P2P（`isend` / `irecv`）将当前的 $K^{(s)}, V^{(s)}$ 发往下一节点 $(r+1)\%P$，同时从上一节点 $(r-1)\%P$ 接收 $K^{(s+1)}, V^{(s+1)}$；
  2. **计算核心**：在前台通过 Tensor Core 计算本地 $Q$ 与当前 $K^{(s)}, V^{(s)}$ 的 FlashAttention 分块；
- **零通信开销物理准则**：
  $$T_{\text{comm}} = \frac{4 \times (S/P) \times D \times b}{\text{Bandwidth}_{\text{ring}}}, \quad T_{\text{comp}} = \frac{4 \times (S/P)^2 \times D}{\text{TFLOPS}_{\text{GPU}}}$$
  只要切片块大小满足 $S/P \ge \frac{\text{TFLOPS}_{\text{GPU}}}{\text{Bandwidth}_{\text{ring}}} \cdot b$，计算耗时必然大于网络传输耗时，网络通信被**完全隐藏（100% Compute-Bound）**。

##### 3. 因果掩码裁剪与跨步流式 Softmax 数学融合
- **因果块跳过（Causal Pruning）**：
  对第 $i$ 号 GPU，其仅需计算源节点 $j \le i$ 的历史键值块：
  - 若 $j > i$（未来块）：完全跳过计算与通信等待，直接省去 $50\%$ 的全局 FLOPs；
  - 若 $j == i$（对角块）：应用因果下三角 Mask；
  - 若 $j < i$（历史块）：全矩阵无 Mask 计算。
- **跨环步 Online Softmax 递推更新**：
  在各环步 $s$ 完成分块 Attention 后，本地维护状态 $(m_{\text{run}}, l_{\text{run}}, O_{\text{run}})$ 按如下公式平滑更新：
  $$m_{\text{new}} = \max(m_{\text{run}}, m_{\text{block}})$$
  $$\alpha = \exp(m_{\text{run}} - m_{\text{new}}), \quad \beta = \exp(m_{\text{block}} - m_{\text{new}})$$
  $$l_{\text{run}} = \alpha \cdot l_{\text{run}} + \beta \cdot l_{\text{block}}$$
  $$O_{\text{run}} = \alpha \cdot O_{\text{run}} + \beta \cdot O_{\text{block}}$$
  全部 $P$ 步完成后执行最终归一化 $O_{\text{final}} = O_{\text{run}} / l_{\text{run}}$，其结果与单卡全量 Attention **在数学上严格等价（Bit-Exact）**。

##### 4. PyTorch + 分布式 P2P 双缓冲核心实现骨架

```python
import torch
import torch.distributed as dist

def ring_flash_attention_forward(q_local, k_local, v_local, group=None):
    """
    q_local, k_local, v_local: [Batch, S_local, Heads, Dim]，其中 S_local = Total_Seq / P
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    # 1. 分配 Ping-Pong 传输双缓冲，避免通信覆写冲突
    k_curr, v_curr = k_local.clone(), v_local.clone()
    k_next = torch.empty_like(k_local)
    v_next = torch.empty_like(v_local)

    # 2. 初始化跨环步流式 Softmax 累加状态
    m_running = torch.full((q_local.shape[0], q_local.shape[2], q_local.shape[1]), -float('inf'), device=q_local.device)
    l_running = torch.zeros_like(m_running)
    o_running = torch.zeros_like(q_local)

    # 3. 沿环推进 world_size 步
    for step in range(world_size):
        # 发起非阻塞 P2P 发送与接收
        work_handles = []
        if step < world_size - 1:
            reqs = [
                dist.P2POp(dist.isend, k_curr, next_rank, group),
                dist.P2POp(dist.isend, v_curr, next_rank, group),
                dist.P2POp(dist.irecv, k_next, prev_rank, group),
                dist.P2POp(dist.irecv, v_next, prev_rank, group),
            ]
            work_handles = dist.batch_isend_irecv(reqs)

        # 确定当前 K, V 块在原始序列中的全局位置
        source_rank = (rank - step + world_size) % world_size

        # 因果掩码裁剪：只计算历史与当前块 (source_rank <= rank)
        if source_rank <= rank:
            is_causal = (source_rank == rank)
            # 调用底层 FlashAttention Kernel 计算局部块注意力
            out_block, m_block, l_block = flash_attn_chunk(q_local, k_curr, v_curr, causal=is_causal)
            
            # 流式跨步 Online Softmax 动态重新缩放与累加
            m_new = torch.maximum(m_running, m_block)
            alpha = torch.exp(m_running - m_new)
            beta = torch.exp(m_block - m_new)
            
            l_running = alpha * l_running + beta * l_block
            o_running = alpha.unsqueeze(-1) * o_running + beta.unsqueeze(-1) * out_block
            m_running = m_new

        # 等待后台通信完成，交换 Ping-Pong 缓冲区指针
        if step < world_size - 1:
            for req in work_handles:
                req.wait()
            k_curr, k_next = k_next, k_curr
            v_curr, v_next = v_next, v_curr

    return o_running / l_running.unsqueeze(-1)
```

</div>
</details>

---

### 4. 架构级头数压缩与 Serving 显存优化

```text
MHA vs MQA vs GQA 架构对比：
MHA (Multi-Head Attention):        MQA (Multi-Query Attention):       GQA (Grouped-Query Attention):
Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1][2] [3][4] [5][6] [7][8]
K/V Heads: [1] [2] [3] [4] [5] [6] [7] [8]  K/V Heads: [         1 (共享)          ]  K/V Heads:  [ 1 ]  [ 2 ]  [ 3 ]  [ 4 ]
(KV 缓存 100%, 显存占用最大)                (KV 缓存压缩至 1/H, 表达力略损)             (LLaMA-3 标配: 兼顾容量与吞吐)
```

- **MHA**：$H_Q = H_{KV}$。每个 Query 独享一组 Key/Value，表达力最强，但 KV Cache 显存开销最大；
- **MQA**：$H_Q = H, H_{KV} = 1$。所有 Query 头共享 1 组 Key/Value，KV Cache 骤降 $H$ 倍，但长程复杂推理表征容量有所损耗；
- **GQA**：$H_Q = H, H_{KV} = G$（$1 < G < H$）。Query 头分为 $G$ 组（如 LLaMA-3 的 64:8），实证表明 GQA 能以接近 MHA 99% 的性能达成接近 MQA 的吞吐与带宽压缩比；
- **PagedAttention（vLLM）**：借鉴虚拟内存分页，将逻辑连续 KV 张量映射到离散物理内存页（如 16 Tokens/页），显存碎片从 $60\% \sim 80\%$ 压至 $<4\%$；
- **KV Cache 量化（FP8 / INT4）**：将缓存数值从 16-bit 压缩至 8-bit 或 4-bit，带宽与容量需求减半至四分之一。

> [!TIP]
> **配套实战编程演练**：
> - 组查询注意力：前往 **[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.md#Exercise 3 · GroupedQueryAttention（GQA）|ML Coding 03 · Exercise 3：Grouped Query Attention (GQA)]]** 实战编写 Query 分组与 Key/Value 广播映射；
> - 自回归 KV 缓存维护：前往 **[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.md#Exercise 6 · KV Cache Attention|ML Coding 03 · Exercise 6：KV Cache Attention]]** 实现 Prefill 与逐 Token Decode 缓存拼接。

---

### 5. 工业落地收敛形态与混合架构演进

> [!NOTE]
> 各机制在显存容量、HBM 访存 IO、前向/反向 FLOPs 及推理单步开销的精确数值对比，已汇总于 **[[#3. 三大效率路线：改算法复杂度与改系统切分|(0) 全景复杂度与硬件瓶颈多维对比矩阵]]**。

在千亿参数与百万 Token 上下文的生产实践中，单一机制无法独立解决全部物理瓶颈，工业界已形成高度协同的**分层立体收敛体系**：

#### 1. 分层技术栈协同（Systems Stack Synergy）
- **单卡硬件层（Micro-Architecture）**：统一标配 **FlashAttention**（如 FlashAttention-2 / FlashAttention-3），利用 Tensor Core 脉动阵列与 SRAM Tiling 将单卡访存推至算力极限；
- **多卡集群层（Context Parallelism）**：在预训练长文本与超长 Prompt 检索时，标配 **RingAttention / USP（Unified Sequence Parallelism）**，通过环形 P2P 双缓冲将单卡显存开销平摊 $P$ 倍，通信被前台计算完全隐藏；
- **推理 Serving 层（Inference Memory Optimization）**：
  - 架构上标配 **GQA（8:1 压缩比）** 减少 KV 产生量；
  - 显存管理标配 **PagedAttention（vLLM）** 消除内存碎片；
  - 数据精度采用 **FP8 / INT4 KV Cache 量化**，降低访存带宽需求至 50%~25%。

#### 2. 模型架构层收敛：混合注意力形态（Hybrid Architecture）
单纯的线性注意力和状态空间模型（SSM）在少样本上下文学习（ICL）与精确多跳检索任务上存在天然容量上限，而密集 Softmax 注意力在超长序列下的二次方计算与 Decode 显存开销不可承受。当前工业界最先进的长文本模型普遍收敛于**混合架构（Hybrid）**：

- **周期交替堆叠（Periodic Hybrid Stacking，如 Jamba, Nemotron-4）**：
  - 每 $N$ 层（如 7 层或 3 层）线性注意力 / Mamba SSM 层之后，插入 1 层密集标准 Softmax 注意力层；
  - 线性/SSM 层以 $\mathcal{O}(1)$ 恒定开销高速压缩与流转全局语境，密集层以全注意力捕获跨文档精细依赖；
- **原生稀疏+密集混合（NSA + Sliding Window，如 DeepSeek-V3 / R1 系列）**：
  - 底层采用基于 64-token 硬件对齐的 NSA，粗筛与细筛分离，在保留 Softmax 检索锐度的同时，将计算与显存开销压低一个数量级。

---

## 模块六：核心工程公式与推导速查清单

| 核心工程指标 | 精确计算式 | 典型工业规模基准 (LLaMA-3-70B, $S=4096$) |
|---|---|---|
| **单层 MHA 投影 FLOPs** | $\text{FLOPs}_{\text{proj}} = 8 S D^2$ | $8 \times 4096 \times 8192^2 \approx \mathbf{2.20 \text{ TFLOPs}}$ |
| **单层 MHA 注意力 FLOPs** | $\text{FLOPs}_{\text{attn}} = 4 S^2 D$ | $4 \times 4096^2 \times 8192 \approx \mathbf{0.55 \text{ TFLOPs}}$ |
| **单 Token KV Cache 容量** | $\text{Memory}_{\text{token}} = 2 L H_{KV} d_k b$ | $2 \times 80 \times 8 \times 128 \times 2 = \mathbf{320 \text{ KB / Token}}$ |
| **自回归单步算术强度** | $\text{Operational Intensity} \approx \frac{2 \times \text{Params}}{\text{Params} \times b + \text{KVCache}} \approx 1$ | 处于绝对 Memory-Bound 状态，吞吐上限受限于显存带宽 |
| **FlashAttention 访存优化比** | HBM 读写从 $\mathcal{O}(S^2)$ 降低到 $\mathcal{O}(S)$ | 激活显存开销从 $\mathcal{O}(S^2)$ 压至 $\mathcal{O}(S D)$，MFU 提升 2~4 倍 |
| **线性注意力单步推理复杂度** | 状态更新 $S_t = S_{t-1} + k_t v_t^T \in \mathbb{R}^{d \times d}$ | 时间 $\mathcal{O}(1)$，隐状态显存 $\mathcal{O}(D^2)$，无线性膨胀 Cache |

### 四条架构与系统工程法则
1. **FlashAttention 不改 FLOPs**：其本质是硬件层访存优化，解决了激活显存 OOM 与 Memory-Bound 算子等待问题，长文本的二次方计算瓶颈必须依靠稀疏或线性注意力化解；
2. **稀疏注意力的成败在于内存对齐**：必须采用如 DeepSeek NSA 的块级对齐结构，离散非连续寻址会抵消算法 FLOPs 节约；
3. **线性注意力必须具备记忆擦除**：单纯累加会导致信息弥散饱和，引入 Delta 规则（DeltaNet）动态擦除旧记忆是逼近 Softmax 表达力的关键；
4. **RingAttention 是系统切分而非近似**：通过环形通信隐藏传输时延，在集群上保持数学结果完全等价（Exact Attention）。
