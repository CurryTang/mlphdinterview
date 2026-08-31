# ML Coding 01B · Transformer Architecture Variants: MHA Tensor Shapes, FLOPs Breakdown & KV Cache Hardware Optimizations

In large language model (LLM) system design and generative AI engineering, a rigorous grasp of Multi-Head Attention (MHA) tensor transformations, computational complexity (FLOPs) regime shifts, and autoregressive Key-Value (KV) cache memory scaling—alongside hardware-aware accelerations like FlashAttention, GQA, and PagedAttention—is fundamental for modern foundation model architecture, optimization, and large-scale serving.

This note systematically covers 5 foundational pillars of Transformer mechanisms:
1. **Transformer Architectural Taxonomies (Encoder-Only vs. Decoder-Only vs. Encoder-Decoder)**
2. **Multi-Head Attention (MHA) Mathematical Derivation, Tensor Shapes & Execution Pipeline**
3. **MHA FLOPs Complexity Decomposition & Context Regime Shifts**
4. **Autoregressive Inference Dynamics & Exact KV Cache Memory Footprint Modeling**
5. **Hardware-Aware Attention Optimizations (MQA / GQA, FlashAttention SRAM Tiling, PagedAttention & KV Quantization)**

---

## Module 1: Architectural Taxonomies, Masking Patterns & Cross-Attention

```text
Comparison of Attention Masking Patterns:
Encoder-Only (BERT):           Decoder-Only (GPT / LLaMA):     Encoder-Decoder (T5 / BART):
┌───┬───┬───┬───┐             ┌───┬───┬───┬───┐               ┌───┬───┬───┬───┐
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ -∞│ -∞│ -∞│               │ 0 │ 0 │ 0 │ 0 │  (Encoder: Fully Bidirectional)
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               ├───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ -∞│ -∞│               │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               └───┴───┴───┴───┘
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ 0 │ -∞│               ┌───┬───┬───┬───┐
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               │ 0 │ -∞│ -∞│ -∞│  (Decoder: Causal Lower-Triangular)
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ 0 │ 0 │               └───┴───┴───┴───┘
└───┴───┴───┴───┘             └───┴───┴───┴───┘               + Cross-Attention: Q_dec × K_enc^T
[Bidirectional M_ij = 0]      [Causal Lower-Triangular]       [Bidirectional Enc + Causal Dec + Cross-Attn]
```

### Comprehensive Comparison of Transformer Archetypes

| Archetype | Attention Masking Pattern | Processing / Generation Paradigm | KV Cache Requirement | Canonical Models | Primary Use Cases |
|---|---|---|---|---|---|
| **Encoder-Only** | Fully bidirectional ($M_{ij} = 0$) | Non-autoregressive; processes all $S$ tokens in a single parallel forward pass | **No KV Cache required** | BERT, RoBERTa, DeBERTa | Text classification, NER, dense retrieval embeddings |
| **Decoder-Only** | Causal lower-triangular ($M_{ij} = -\infty$ for $j > i$) | Autoregressive; generates tokens sequentially conditioned on historical context | **KV Cache is mandatory** | GPT-4, LLaMA-3, Mistral, Qwen, DeepSeek | Generative foundation LLMs, instruction following, reasoning |
| **Encoder-Decoder** | Encoder bidirectional + Decoder causal + **Cross-Attention** | Bidirectional encoding of prompt; autoregressive generation of target | **Dual KV Cache required** (Static encoder + Dynamic decoder) | T5, BART, Whisper, Original Transformer | Machine translation, abstractive summarization, ASR |

#### Cross-Attention Mechanics
In an Encoder-Decoder model:
- **Queries ($Q$)**: Generated from the decoder's preceding hidden representations;
- **Keys ($K$) and Values ($V$)**: Generated from the final encoder representations;
- **Execution**: Encoder $K, V$ are computed once during prompt processing and reused across all subsequent decoding steps.

---

## Module 2: Multi-Head Attention (MHA) Math, Tensor Shapes & Execution Pipeline

Let batch size be $B$, sequence length $S$, hidden dimension $D$, number of attention heads $H$, and per-head dimension $d_k = D / H$.

```text
MHA Tensor Flow Lifecycle:
Input X (B, S, D)
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

### Detailed 6-Step Tensor Transformation

1. **Linear Projections**:
   Input tensor $\mathbf{X} \in \mathbb{R}^{B \times S \times D}$ with projection weights $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$:

$$Q = \mathbf{X}W_Q, \quad K = \mathbf{X}W_K, \quad V = \mathbf{X}W_V \quad \in \mathbb{R}^{B \times S \times D}$$

2. **Head Reshaping and Transposition**:

$$\text{Reshape: } (B, S, D) \to (B, S, H, d_k) \xrightarrow{\text{Transpose (1, 2)}} (B, H, S, d_k)$$

3. **Scaled Dot-Product Attention Scores**:

$$A = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times S \times S}$$

   > **Why divide by $\sqrt{d_k}$?**  
   > Assuming $Q$ and $K$ components are independent random variables with zero mean and unit variance, the dot product $\sum_{i=1}^{d_k} q_i k_i$ has mean 0 and **variance $d_k$**. Scaling by $\frac{1}{\sqrt{d_k}}$ preserves unit variance, preventing dot products from exploding into extreme values that saturate softmax gradients.

4. **Masking & Softmax**:

$$\tilde{A} = \text{softmax}(A + M), \quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

5. **Value Aggregation & Concatenation**:

$$\text{Head}_h = \tilde{A}_h V_h \in \mathbb{R}^{B \times H \times S \times d_k} \xrightarrow{\text{Transpose \& Reshape}} \text{MultiHead} \in \mathbb{R}^{B \times S \times D}$$

6. **Output Projection**:

$$\text{Output} = \text{MultiHead} \cdot W_O \in \mathbb{R}^{B \times S \times D}, \quad W_O \in \mathbb{R}^{D \times D}$$

---

## Module 3: MHA Computational Complexity & Regime Shift Analysis

Each multiply-accumulate operation corresponds to 2 FLOPs.

### 1. FLOPs Breakdown (for batch size $B=1$)

1. **Linear Projections ($Q, K, V, W_O$)**:
   Four matrix multiplications of shape $(S \times D) \times (D \times D)$:
   $$\text{FLOPs}_{\text{proj}} = 4 \times (2 \times S \times D \times D) = \mathbf{8 S D^2} \implies \mathcal{O}(S D^2)$$
2. **Attention Score Matrix ($Q K^T$)**:
   $H$ heads performing $(S \times d_k) \times (d_k \times S)$ multiplication:
   $$\text{FLOPs}_{QK^T} = H \times (2 \times S \times d_k \times S) = \mathbf{2 S^2 D} \implies \mathcal{O}(S^2 D)$$
3. **Value Aggregation ($\tilde{A} V$)**:
   $H$ heads performing $(S \times S) \times (S \times d_k)$ multiplication:
   $$\text{FLOPs}_{AV} = H \times (2 \times S \times S \times d_k) = \mathbf{2 S^2 D} \implies \mathcal{O}(S^2 D)$$
4. **Total MHA Layer FLOPs**:

$$\text{Total FLOPs}_{\text{MHA}} = 8 S D^2 + 4 S^2 D$$

---

### 2. Context Length Regime Shifts

```text
MHA FLOPs Component Crossover:
FLOPs
  ▲
  │                                    /  O(S² D) Attention MatMul
  │                                   /   (Dominates in long-context)
  │                                  /
  │            O(S D²) Projections  /
  │           (Dominates in short) /
  │         ─────────────────────/
  │                             /
  └────────────────────────────┴─────────────► Sequence Length S
                             S ≈ 2D (Crossover Point)
```

- **Short-Context Regime ($S < 2D$, e.g., $S=2048, D=4096$)**:
  $8 S D^2 > 4 S^2 D$. Projection matrix multiplications dominate compute ($>80\%$). GEMM throughput on Tensor Cores is the primary optimization objective.
- **Long-Context Regime ($S \gg D$, e.g., $S=32K \sim 128K, D=4096$)**:
  $4 S^2 D \gg 8 S D^2$. The quadratic attention matrix calculation explodes, dominating runtime and memory. Hardware-aware tiling (FlashAttention) and sparse/linear variants become essential.

---

## Module 4: Autoregressive Inference & KV Cache Memory Growth

### 1. Prefill Phase vs. Decode Phase

```text
Inference Phase Characteristics:
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Phase                   │ Hardware Dynamics & Bottlenecks                                        │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Prefill Phase        │ • Parallel processing of entire prompt sequence ($S_{\text{prompt}}$)  │
│    (Prompt Processing)  │ • Computes initial KV cache                                            │
│                         │ • High arithmetic intensity $\implies$ **Compute-Bound**               │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Decode Phase         │ • Step-by-step single token input ($x_t \in \mathbb{R}^{1 \times D}$)  │
│    (Token Generation)   │ • Appends new $k_t, v_t$ to cache; computes $q_t K_{\le t}^T V_{\le t}$│
│                         │ • Must stream full model weights & KV cache per generated token        │
│                         │ • Arithmetic intensity $\approx 1 \text{ FLOP/Byte} \implies$ **Memory-Bound**│
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

#### Why Is Query ($Q$) Never Cached?
- Query vector $q_t \in \mathbb{R}^{1 \times D}$ is only needed to compute attention against past keys $K_{\le t}$ for the current token;
- At step $t+1$, the newly generated token produces its own query $q_{t+1}$;
- **Past queries $q_1, \dots, q_t$ are never reused**, giving $Q$ an ephemeral lifespan.

---

### 2. KV Cache Memory Footprint Formula

For batch size $B$, sequence length $S$, number of layers $L$, number of key-value heads $H_{KV}$, per-head dimension $d_k$, and byte precision $b$ (e.g., $b=2$ for FP16/BF16):

$$\text{Memory}_{\text{KVCache}} = 2 \times B \times S \times L \times H_{KV} \times d_k \times b \quad \text{Bytes}$$

> *The factor of 2 accounts for Key and Value tensors.*

#### Case Study: LLaMA-3-70B
- Specifications: $L=80, D=8192, H_Q=64, H_{KV}=8 \text{ (GQA)}, d_k=128, b=2$
- Per-token memory footprint:
  $$\text{Per-Token Memory} = 2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ Bytes} \approx \mathbf{320 \text{ KB / Token}}$$
- For batch $B=64$ and context $S=8192$:
  $$\text{Total Memory} = 64 \times 8192 \times 320 \text{ KB} \approx \mathbf{167.77 \text{ GB}}$$
  *(Exceeds the 140 GB static model weights footprint!)*

---

## Module 5: Hardware-Aware Attention Optimizations

### 1. Architecture Variants: MHA vs. MQA vs. GQA

```text
MHA vs. MQA vs. GQA Comparison:
MHA (Multi-Head Attention):        MQA (Multi-Query Attention):       GQA (Grouped-Query Attention):
Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1][2] [3][4] [5][6] [7][8]
K/V Heads: [1] [2] [3] [4] [5] [6] [7] [8]  K/V Heads: [         1 (Shared)        ]  K/V Heads:  [ 1 ]  [ 2 ]  [ 3 ]  [ 4 ]
(KV Cache 100%, highest VRAM)               (KV Cache 1/H, quality trade-off)           (LLaMA-3 Standard: Balanced)
```

- **Multi-Head Attention (MHA)**: $H_Q = H_{KV}$. Maximum expressivity, highest KV memory footprint;
- **Multi-Query Attention (MQA)**: $H_Q = H, H_{KV} = 1$. Single key/value head shared across all queries; drops KV cache by $H\times$, but can degrade multi-turn reasoning capacity;
- **Grouped-Query Attention (GQA)**: $H_Q = H, H_{KV} = G$ ($1 < G < H$). Groups query heads into $G$ key/value pairs (e.g., 8:1 ratio in LLaMA-3), retaining $\approx 99\%$ of MHA performance while drastically cutting KV memory traffic.

---

### 2. FlashAttention: IO-Aware Tiling & Online Softmax

```text
Standard Attention vs. FlashAttention GPU Memory Traffic:
Standard Attention (HBM Memory-Bound):
[GPU SRAM] ──(Write S×S Matrix)──> [GPU HBM (Slow)] ──(Read S×S)──> [GPU SRAM] ──(Write S×D)──> [HBM]
• Generates massive O(S²) HBM read/write traffic!

FlashAttention (On-Chip SRAM Tiling & Kernel Fusion):
[GPU SRAM] ──(Loads Q,K,V blocks into SRAM, computes Online Softmax incrementally, never materializes S×S in HBM)──> Writes only final (S×D) to HBM
• Reduces HBM memory traffic to O(S)! 2-4x speedup!
```

- **Key Innovations**:
  1. **Tiling**: Partitions $Q, K, V$ into blocks sized for fast on-chip SRAM (192 KB/SM on A100);
  2. **Online Softmax**: Computes normalized softmax dynamically without materializing the full $S \times S$ matrix in HBM;
  3. **Recomputation in Backward**: Recomputes attention activations on the fly during backpropagation instead of saving $S \times S$ matrices in HBM.

---

### 3. PagedAttention (vLLM) & KV Quantization

- **PagedAttention**: Partitions continuous KV tensors into non-contiguous virtual memory blocks (pages), reducing memory fragmentation from $>60\%$ to $<4\%$.
- **KV Quantization (FP8/INT4)**: Quantizes cached keys and values to 8-bit or 4-bit precision, halving memory bandwidth demands during decoding.

---

## Module 6: High-Frequency Interview Essentials

### Q1: Calculate the FLOPs for a single MHA layer with sequence length $S=4096$ and hidden dimension $D=4096$.
> **Answer**:
> 1. Linear projections ($4$ matmuls): $\text{FLOPs}_{\text{proj}} = 8 S D^2 = 8 \times 4096 \times (4096)^2 \approx \mathbf{5.498 \times 10^{11} \text{ FLOPs} \ (550 \text{ GFLOPs})}$.
> 2. Attention matrix math ($QK^T$ and $\tilde{A}V$): $\text{FLOPs}_{\text{attn}} = 4 S^2 D = 4 \times (4096)^2 \times 4096 \approx \mathbf{2.749 \times 10^{11} \text{ FLOPs} \ (275 \text{ GFLOPs})}$.
> 3. Total MHA FLOPs: $\approx \mathbf{825 \text{ GFLOPs}}$.

### Q2: Why does FlashAttention achieve a 2–4× speedup while producing mathematically exact attention outputs?
> **Answer**:
> GPUs compute much faster than they transfer data between HBM and compute units. Standard attention is bottlenecked by repeated $O(S^2)$ memory round-trips to HBM for intermediate attention score matrices. FlashAttention fuses operations inside fast on-chip SRAM using tiling and online softmax, reducing HBM IO complexity from $O(S^2)$ to $O(S)$.
