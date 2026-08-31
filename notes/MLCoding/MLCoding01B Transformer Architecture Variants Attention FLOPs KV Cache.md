# ML Coding 01B · Transformer 架构变体：MHA 张量维度推导、FLOPs 分解与 KV Cache 硬件优化

在大语言模型（LLM）与 Transformer 架构工程中，深入理解多头注意力（MHA）的精确张量流动、计算复杂度（FLOPs）的算力瓶颈转移、以及自回归推理阶段 KV Cache 的显存瓶颈与硬件感知加速（FlashAttention、GQA、PagedAttention），是模型架构设计、性能调优与大规模 Serving 部署的核心基本功。

本篇系统梳理 Transformer 算子底层的 5 大核心体系：
1. **Transformer 架构分类学（Encoder-Only vs Decoder-Only vs Encoder-Decoder）**
2. **多头注意力（MHA）数学推导、张量形状演化与执行流水线**
3. **MHA 计算复杂度 FLOPs 严密分解与长短序列体制转移（Regime Shift）**
4. **自回归推理动态与 KV Cache 显存容量精确数学模型**
5. **硬件感知注意力优化体系（MQA / GQA、FlashAttention SRAM 分块平铺、PagedAttention 与 KV 量化）**

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

### 三大架构形态全景对比

| 架构形态 | 注意力机制与掩码模式 | 输入与生成范式 | KV Cache 需求 | 工业界典型代表 | 最佳适用场景 |
|---|---|---|---|---|---|
| **Encoder-Only** | 全双向自注意力（$M_{ij} = 0$） | 非自回归，单次前向并行处理全部 $S$ 个 Token | **无需 KV Cache**（单次前向完成） | BERT, RoBERTa, DeBERTa | 文本分类、命名实体识别（NER）、向量表征（Embedding） |
| **Decoder-Only** | 因果下三角自注意力（$j > i$ 时 $M_{ij} = -\infty$） | 自回归生成，逐 Token 依次依赖历史上下文 | **必须维护 KV Cache**（避免重复计算历史 Key/Value） | GPT-4, LLaMA-3, Mistral, Qwen, DeepSeek | 通用大语言模型、指令遵循、代码生成、多步推理 |
| **Encoder-Decoder** | Encoder 双向 + Decoder 因果自注意力 + **Cross-Attention（跨注意力）** | 双向编码输入上下文，自回归生成目标序列 | **需双份 KV Cache**（Encoder 静态缓存 + Decoder 动态缓存） | T5, BART, Whisper, 原始 Transformer | 机器翻译、文本摘要、语音识别（ASR） |

#### Cross-Attention（跨注意力机制）工作原理
在 Encoder-Decoder 架构中：
- **Query（$Q$）**：来源于 Decoder 上一层的隐层状态（表示“当前解码器需要关注什么”）；
- **Key（$K$）与 Value（$V$）**：来源于 Encoder 顶层的输出表征（表示“输入上下文提供了哪些全局信息”）；
- **执行逻辑**：Encoder 的 $K, V$ 在 Prefill 阶段只需计算一次，随后在整个自回归解码过程中被所有解码步骤反复共享读取。

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

## 模块五：硬件感知注意力优化体系

### 1. 注意力架构演化：MHA vs MQA vs GQA

```text
MHA vs MQA vs GQA 架构对比：
MHA (Multi-Head Attention):        MQA (Multi-Query Attention):       GQA (Grouped-Query Attention):
Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1][2] [3][4] [5][6] [7][8]
K/V Heads: [1] [2] [3] [4] [5] [6] [7] [8]  K/V Heads: [         1 (共享)          ]  K/V Heads:  [ 1 ]  [ 2 ]  [ 3 ]  [ 4 ]
(KV 缓存 100%, 显存占用最大)                (KV 缓存压缩至 1/H, 表达力略损)             (LLaMA-3 标配: 兼顾容量与吞吐)
```

- **MHA**：$H_Q = H_{KV}$。每个 Query 头对应独立的 Key/Value 头，表达能力最强，但 KV Cache 显存最大；
- **MQA**：$H_Q = H, H_{KV} = 1$。所有 Query 头共享单一组 Key/Value 头，KV Cache 显存骤降 $H$ 倍，但大幅降低了模型在长文本和复杂多轮推理下的多头表达容量；
- **GQA**：$H_Q = H, H_{KV} = G$（$1 < G < H$）。将 Query 头划分为 $G$ 组，每组共享一组 Key/Value 头（如 LLaMA-3 的 64:8 分组）。实证表明，**GQA 能够以接近 MHA 99% 的模型效果，获得接近 MQA 的推理显存带宽与吞吐提升**。

---

### 2. FlashAttention：IO 感知分块平铺与在线 Softmax

```text
标准 Attention 与 FlashAttention 显存交互对比：
标准 Attention (HBM 访存受限):
[GPU SRAM] ──(写入 S×S 矩阵)──> [GPU HBM 慢速显存] ──(读取 S×S 矩阵)──> [GPU SRAM] ──(写入 S×D)──> [HBM]
• 产生 O(S²) 巨大的 HBM 读写流量！

FlashAttention (SRAM 分块计算 + Kernel 融合):
[GPU SRAM] ──(Tiling 分块加载 Q,K,V，在片上高速 SRAM 增量计算 Online Softmax，绝不回写 S×S 矩阵)──> 最终直接输出 (S×D) 写入 HBM
• HBM 访存降为 O(S)！IO 速度提升 2~4 倍！
```

- **传统 Attention 性能瓶颈**：标准 Attention 计算 $QK^T$ 后，需要将庞大的 $S \times S$ 中间得分矩阵写回低速 GPU HBM，再从 HBM 读出计算 Softmax，然后再写回 HBM，产生极大的内存带宽读写开销（IO-Bound）；
- **FlashAttention 创新核心**：
  1. **Tiling（分块计算）**：将 $Q, K, V$ 划分为适合 GPU 快速片上 SRAM（如 A100 的 192KB/SM）大小的子块；
  2. **Online Softmax 增量归一化**：在不物化完整 $S \times S$ 矩阵的前提下，通过记录局部的最大值 $m(x)$ 和指数和 $l(x)$，在 SRAM 内动态修正注意力输出；
  3. **Recomputation in Backward**：反向传播时不保存 $S \times S$ 激活值矩阵，而是在 SRAM 中直接快速重算，大幅压缩训练显存。

---

### 3. PagedAttention（vLLM 核心引擎）

- **传统显存痛点**：传统推理引擎要求 KV Cache 在物理显存中保持严格连续分配。由于生成长度不可预知，系统必须提前预留最大长度（如 4K/8K），导致**内部碎片（已预留未填充）**与**外部碎片（内存无法凑齐整块）**高达 $60\% \sim 80\%$；
- **PagedAttention 方案**：借鉴操作系统虚拟内存分页机制，将 KV Cache 划分为固定大小的**物理块（Block / Page，如 16 个 Token 一页）**。通过逻辑块到物理块的页表映射，实现非连续显存动态分配，将显存浪费降低至 $<4\%$，同等硬件下的 Serving 并发吞吐提升 2~4 倍。

---

### 4. KV Cache 量化（FP8 / INT4）

在超长上下文（32K ~ 128K）与高并发场景中，将 KV Cache 从 BF16（16 bits）量化为 FP8（8 bits）或 INT4（4 bits）：
- 显存占用直接降低 $50\% \sim 75\%$；
- 显存带宽搬运需求减半，在 Memory-Bound 的 Decode 阶段直接实现推理生成速度翻倍。

---

## 模块六：面试高频必背公式与速答清单

### Q1：计算一个序列长度为 $S=4096$、隐藏维度 $D=4096$ 的单层 MHA 线性投影和注意力计算的 FLOPs 分别是多少？
> **答**：
> 1. 线性投影（4 次矩阵乘法）：
>    $$\text{FLOPs}_{\text{proj}} = 8 S D^2 = 8 \times 4096 \times (4096)^2 = 8 \times 4096 \times 1.678 \times 10^7 \approx \mathbf{5.498 \times 10^{11} \text{ FLOPs} \ (550 \text{ GFLOPs})}$$
> 2. 注意力矩阵计算（$QK^T$ 与 $\tilde{A}V$）：
>    $$\text{FLOPs}_{\text{attn}} = 4 S^2 D = 4 \times (4096)^2 \times 4096 \approx \mathbf{2.749 \times 10^{11} \text{ FLOPs} \ (275 \text{ GFLOPs})}$$
> 3. 在此序列长度下（$S=D$），线性投影计算量约是注意力矩阵计算量的 2 倍。

### Q2：为什么 FlashAttention 能够在数学结果完全等价（Exact Attention）的前提下，实现 2~4 倍的速度提升？
> **答**：
> 因为 GPU 计算单元（Tensor Cores）的速度远远快于显存带宽（HBM）。标准 Attention 的瓶颈不在于算力，而在于反复向慢速 HBM 读写 $S \times S$ 的中间激活值矩阵。FlashAttention 通过 Tiling 分块将所有计算锁在片上极速 SRAM 中完成，彻底消除了中间矩阵的 HBM 访存往返，将 HBM 访存复杂度从 $O(S^2)$ 降低到 $O(S)$。
