# ML Coding 00B · LLM Basics: Why Decoder-Only Won, Dense Text Embeddings & Mixed-Precision Training Systems

In large language model (LLM) system design and generative AI engineering, mastering the architectural selection rationale, dense text embedding bottlenecks, and hardware-level numerical precision dynamics is essential for pre-training, fine-tuning, and production serving.

This note systematically covers 3 foundational pillars of modern LLM systems:
1. **Why Decoder-Only Won & Dense Text Embeddings: Architectural convergence analysis and how modern research overcame Anisotropy and representation collapse**
2. **Hardware Numerical Precision Formats (FP32 / FP16 / BF16 / FP8 / NVFP4) & Mixed-Precision Training (Master Weights, Loss Scaling, and Stochastic Rounding)**
3. **Mathematical Derivation of Initial Loss & Step-0 Sanity Check (ln(V) Proof, Temperature Scaling, Label Smoothing, Tokenizer Effects & Pre-training Diagnostics)**

---

## Module 1: Why Has Decoder-Only Become the Dominant LLM Architecture?

During the early evolution of Transformers, three architectural archetypes were extensively explored:
- **Encoder-Only (e.g., BERT, RoBERTa)**: Fully bidirectional self-attention; optimal for discriminative tasks (classification, NER, sentence embeddings).
- **Encoder-Decoder (e.g., T5, BART, original Transformer)**: Bidirectional context encoder + causal autoregressive decoder with cross-attention; specialized for translation and summarization.
- **Decoder-Only (e.g., GPT, LLaMA, Mistral, Qwen, DeepSeek)**: Unified causal lower-triangular autoregression; has unified virtually all frontier foundation models.

```text
Transformer Architectural Evolution & Convergence:
┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│     Encoder-Only      │     │    Encoder-Decoder    │     │     Decoder-Only      │
│     (BERT / RoBERTa)  │     │      (T5 / BART)      │     │ (GPT / LLaMA / Qwen)  │
├───────────────────────┤     ├───────────────────────┤     ├───────────────────────┤
│ • Bidirectional self-  │     │ • Bidirectional Enc + │     │ • Unified Causal      │
│   attention ($M_{ij}=0$)│     │   Causal Dec          │     │   Autoregression      │
│ • Discriminative only │     │ • Structural split    │     │ • Foundation model    │
│ • Cannot naturally    │     │ • Cross-Attention     │     │   standard            │
│   generate sequences  │     │   memory overhead     │     │ • Zero-shot / Few-shot│
└───────────────────────┘     └───────────────────────┘     └───────────────────────┘
                                                                        │
                                                    🌟 Modern LLM Convergence
```

### The 4 Decisive Technical and Systems Drivers

#### 1. Unified Next-Token Prediction & Zero Task-Friction (In-Context Learning)

- **Unification of Learning Paradigms**: Under the Decoder-Only framework, **Pre-training, Supervised Fine-Tuning (SFT), In-Context Learning (Few-shot ICL), Prompting, and Chain-of-Thought (CoT) reasoning** are all mathematically unified into the exact same objective—**conditional autoregressive next-token prediction**:

$$P(X) = \prod_{i=1}^S P(x_i \mid x_1, x_2, \dots, x_{i-1})$$

- **Zero Structural Friction**: Any arbitrary input (system prompt, few-shot demonstrations, user query, dialogue history, intermediate reasoning tokens, and final response) exists as a flat sequence of tokens in a single causal stream. There is no artificial architectural boundary dictating what constitutes "source" vs. "target".

#### 2. Simplified KV Cache & Zero Cross-Attention Memory Bloat

- **Single Contiguous KV Cache**: A Decoder-Only model maintains a single Self-Attention Key-Value cache. The Prefill (prompt processing) and Decode (token generation) phases seamlessly share the exact same memory buffers. Each new token simply appends its Key and Value vectors to the end of the existing cache.
- **Encoder-Decoder Serving Bottlenecks**:
  - Encoder-Decoder architectures require maintaining two separate cache systems: a static bidirectional KV cache for the full input sequence in the encoder, plus an autoregressive KV cache for the decoder;
  - Every decoder layer contains an extra **Cross-Attention layer**, forcing the memory bus to fetch the entire encoder KV cache on every single generated token, severely exacerbating memory bandwidth saturation;
  - In modern serving engines (e.g., vLLM, PagedAttention), managing non-contiguous virtual memory pages for a single causal KV cache is drastically simpler than scheduling dual-cache allocations under continuous batching.

#### 3. Parameter Compute Budget Efficiency & Scaling Laws

- **Dense Autoregressive Supervision**:
  - In Decoder-Only pre-training, **every single token position from $1$ to $S$ calculates cross-entropy loss against the next token**. All model parameters receive dense gradients and participate actively in sequence generation.
  - In Encoder-Decoder pre-training (e.g., span corruption in T5), only masked spans in the decoder produce training loss. The encoder parameters do not receive direct generative supervision, and during autoregressive decoding, the encoder parameters remain dormant after prompt encoding (capacity underutilization).
- **Empirical Scaling Law Superiority**: Given a fixed compute budget (FLOPs) and parameter count, Decoder-Only models exhibit the steepest, most predictable power-law scaling trajectories in open-ended text comprehension and generation.

#### 4. Long Context Scaling & Modern Position Embeddings (RoPE Synergy)

- Causal attention naturally respects the temporal arrow of causality, pairing seamlessly with **Rotary Position Embedding (RoPE)**, **Chunked Prefill**, and sliding-window KV eviction policies. This allows models to interpolate and extrapolate context lengths from 4K to 128K and 1M tokens smoothly.
- Bidirectional encoders suffer from **attention dilution** across extreme context lengths and incur strict $O(S^2)$ memory bottlenecks.

---

### 5. Text Representation & Embeddings: Why Were Encoders Dominant, and How Did Modern Research Solve Decoder Embedding Bottlenecks?

Beyond autoregressive text generation, **Dense Text Embeddings** serve as the backbone for Retrieval-Augmented Generation (RAG), semantic vector search, clustering, and recommendation systems.

```text
Evolution of Text Embeddings:
The Classical Era (2018-2022)             Bottlenecks (Why Vanilla Decoders Failed at Embedding)
┌────────────────────────────────┐        ┌────────────────────────────────────────────────────────┐
│ Encoder-Only (BERT / RoBERTa)  │        │ • Unidirectional Blindspots: Tokens cannot see future  │
│ Encoder-Decoder (T5 / Contriever)│ ───> │ • Anisotropy Crisis: Vectors collapse into narrow cone │
│ Mechanism: Bidirectional + CLS │        │ • High compute cost & pre-training objective mismatch  │
└────────────────────────────────┘        └────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼ Modern Research Breakthroughs (2023-2026)
                                          ┌────────────────────────────────────────────────────────┐
                                          │ Modern Decoder-Only Embeddings (E5-Mistral, NV-Embed)  │
                                          │ 1. Unmasking: Remove causal mask for bidirectional SFT │
                                          │ 2. Task-Instructed Contrastive InfoNCE + Hard Negatives│
                                          │ 3. Latent Attention Pooling (Perceiver Cross-Attn)     │
                                          │ 4. Unified Representation & Generation (GritLM)        │
                                          │ 5. Matryoshka Representation Learning (MRL)            │
                                          └────────────────────────────────────────────────────────┘
```

#### Why Did Dense Retrieval Historically Prefer Encoders & Encoder-Decoders?

1. **Global Context Compression via Bidirectional Attention**:
   - BERT, RoBERTa, and T5 utilize a fully bidirectional attention matrix ($M_{ij} = 0$). Every token interacts with both preceding and following tokens across all layers without restriction.
   - The prepended `[CLS]` token acts as a natural information bottleneck, condensing sentence-wide semantics into a high-quality global representation.
   - In contrast, vanilla Decoder-Only models suffer from **unidirectional blindspots**:
     - Under **Last-Token Pooling**, earlier tokens cannot incorporate information from late-occurring semantic shifts or qualifiers (e.g., in *"The ambiance was fantastic but the food was terrible"*, early tokens cannot see the crucial negative turn);
     - Under **Mean Pooling**, because of the lower-triangular causal mask, token $1$ has a receptive field of $1$ while token $S$ has a receptive field of $S$, causing an extreme imbalance in feature abstraction depths.
2. **The Anisotropy Crisis & The Cone Effect: Mathematical Derivation**:

#### (1) Formal Definitions of Isotropy vs. Anisotropy

Let text representation vectors be $\mathbf{h} \in \mathbb{R}^d$ ($L_2$-normalized such that $\|\mathbf{h}\| = 1$).
- **Ideal Isotropy**: Representation vectors are **uniformly distributed** in all directions across the unit hypersphere $\mathcal{S}^{d-1}$.
  The covariance matrix satisfies the isotropic identity:

$$\mathbb{E}_{\mathbf{h}}\left[ \mathbf{h} \mathbf{h}^T \right] = \frac{1}{d} \mathbf{I}_d$$

  All eigenvalues of the covariance matrix are identical ($\lambda_1 = \dots = \lambda_d = \frac{1}{d}$), achieving maximal effective rank. The expected cosine similarity between two independent, semantically unrelated vectors $\mathbf{h}_i, \mathbf{h}_j$ is zero:

$$\mathbb{E}_{i \neq j} \left[ \cos(\mathbf{h}_i, \mathbf{h}_j) \right] \approx 0$$

- **Representation Degeneration & Anisotropy (The Cone Effect)**:
  In uncalibrated autoregressive decoders, the representation covariance matrix exhibits **severe spectral decay**:
  $$\lambda_1 \gg \lambda_2 \gg \dots \gg \lambda_d$$
  All token representations share a dominant common mean bias vector $\mathbf{\mu} = \mathbb{E}[\mathbf{h}]$:

$$\mathbf{h}_i = \mathbf{\mu} + \tilde{\mathbf{h}}_i, \quad \text{where } \|\mathbf{\mu}\| \gg \|\tilde{\mathbf{h}}_i\|$$

#### (2) Proof of Cosine Similarity Collapse

Computing the cosine similarity between two **semantically unrelated** sentences $\mathbf{h}_i$ and $\mathbf{h}_j$:

$$\cos(\mathbf{h}_i, \mathbf{h}_j) = \frac{\mathbf{h}_i^T \mathbf{h}_j}{\|\mathbf{h}_i\| \|\mathbf{h}_j\|} = \frac{(\mathbf{\mu} + \tilde{\mathbf{h}}_i)^T (\mathbf{\mu} + \tilde{\mathbf{h}}_j)}{\|\mathbf{\mu} + \tilde{\mathbf{h}}_i\| \|\mathbf{\mu} + \tilde{\mathbf{h}}_j\|}$$

Expanding the numerator and denominator:

$$\cos(\mathbf{h}_i, \mathbf{h}_j) = \frac{\|\mathbf{\mu}\|^2 + \mathbf{\mu}^T (\tilde{\mathbf{h}}_i + \tilde{\mathbf{h}}_j) + \tilde{\mathbf{h}}_i^T \tilde{\mathbf{h}}_j}{\sqrt{\|\mathbf{\mu}\|^2 + 2\mathbf{\mu}^T\tilde{\mathbf{h}}_i + \|\tilde{\mathbf{h}}_i\|^2} \sqrt{\|\mathbf{\mu}\|^2 + 2\mathbf{\mu}^T\tilde{\mathbf{h}}_j + \|\tilde{\mathbf{h}}_j\|^2}}$$

When the dominant bias norm $\|\mathbf{\mu}\|$ vastly exceeds the residual variation $\|\tilde{\mathbf{h}}\|$, the cross-terms asymptotically vanish:

$$\cos(\mathbf{h}_i, \mathbf{h}_j) \approx \frac{\|\mathbf{\mu}\|^2}{\|\mathbf{\mu}\|^2 + \sigma^2} \approx 1.0$$

> **Theorem**: The high-dimensional embedding space collapses into a **narrow cone**. Unrelated sentences subtend angles of only $5^\circ \sim 15^\circ$, crushing cosine similarities into $0.95 \sim 0.99$ and completely wiping out semantic discriminative capacity!

#### (3) Why Does Autoregressive Pre-training Cause the Cone Effect? (Gao et al., 2019)

1. **Zipf's Law & Softmax Gradient Pull**:
   Next-token cross-entropy forces hidden states $\mathbf{h}$ to predict word probabilities $P(w \mid \mathbf{h}) \propto \exp(\mathbf{w}_w^T \mathbf{h})$. High-frequency tokens (punctuation, stopwords) appear billions of times, continuously pulling all representations along a shared directional vector;
2. **Convex Hull Excludes Origin**:
   As proven by Gao et al. (2019), the convex hull of word embeddings fails to enclose the origin $\mathbf{0}$, forcing hidden representations into a positive semi-definite cone;
3. **Deep Residual Accumulation**:
   Residual streams $\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \text{Attn}(\mathbf{h}^{(l)})$ compound low-frequency components layer by layer, causing deep rank collapse.

#### (4) Theoretical Proof: Breaking the Cone via Contrastive InfoNCE (Wang & Isola, 2020)

Contrastive InfoNCE loss expands the collapsed cone onto the entire hypersphere:

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[ \log \frac{e^{\cos(\mathbf{h}_i, \mathbf{h}_i^+) / \tau}}{e^{\cos(\mathbf{h}_i, \mathbf{h}_i^+) / \tau} + \sum_{j \in \mathcal{N}} e^{\cos(\mathbf{h}_i, \mathbf{h}_j^-) / \tau}} \right]$$

Wang & Isola (ICML 2020) proved that as $N \to \infty$, minimizing $\mathcal{L}_{\text{InfoNCE}}$ asymptotically decomposes into two orthogonal geometric imperatives:

$$\mathcal{L}_{\text{InfoNCE}} \iff \underbrace{\mathbb{E}_{(\mathbf{x}, \mathbf{x}^+)} [\|\mathbf{h} - \mathbf{h}^+\|^2]}_{\mathcal{L}_{\text{align}} \text{ (Alignment: Pulls true positives together)}} + \underbrace{\log \mathbb{E}_{\mathbf{x}, \mathbf{y} \sim p_{\text{data}}} \left[ \exp\left( -2 \|\mathbf{h}_x - \mathbf{h}_y\|^2 \right) \right]}_{\mathcal{L}_{\text{uniform}} \text{ (Uniformity: Maximizes entropy across the hypersphere, eliminating the cone)}}$$

```anisotropy-cone-demo
```

3. **Serving Efficiency & QPS Economics**:
   - Production vector retrieval requires thousands of queries per second (QPS) with sub-10ms latency. Lightweight models (BERT-base 110M, BGE-large 330M) easily deliver high throughput on a single GPU, whereas early 7B/13B decoders were computationally prohibitive.

---

#### How Modern Research Solved Decoder-Only Embedding Deficiencies

Modern foundation models (7B–70B) possess vast world knowledge, multilingual proficiency, and complex reasoning traces far superior to 300M-scale encoders. Recent breakthroughs have enabled Decoder-Only models to dominate global benchmarks like MTEB:

##### 1. Unmasking Causal Attention: Bidirectional Fine-Tuning
- **Pioneering Work**: `E5-Mistral-7B`, `SFR-Embedding`, `BGE-en-ICL`
- **Mechanism**: During embedding fine-tuning, **the causal lower-triangular mask is completely replaced with a fully bidirectional attention matrix**.
- **Outcome**: The decoder retains its pre-trained parametric knowledge while gaining true BERT-like bidirectional contextual reasoning, completely eliminating unidirectional receptive field blindspots.

##### 2. Instruction-Aware Contrastive Learning & Task Prompts
- **Pioneering Work**: `Instructor`, `E5-Mistral`, `Qwen2-Embed`
- **Mechanism**: Prepends task-specific instruction prompts to the query:
  ```text
  Instruct: Given a financial question, retrieve relevant SEC filing passages.
  Query: What were the Q3 capital expenditures?
  ```
- **Resolving Anisotropy**: Multi-stage contrastive training (InfoNCE with in-batch negatives and mined hard negatives) pulls positive pairs together and pushes negatives apart, restoring **isotropic uniformity** on the unit hypersphere.

##### 3. Architectural Pooling Innovation: Latent Attention Pooling
- **Pioneering Work**: `NV-Embed-v1/v2` (ranked #1 on MTEB)
- **Mechanism**: Replaces naive mean/last-token pooling with a **Latent Attention Pooling layer** (inspired by the Perceiver Resampler):
  - A set of learnable latent query vectors $\mathbf{Q}_{\text{latent}}$ attends over the entire sequence of decoder token states via cross-attention;
  - Dynamically extracts multi-scale semantic salience into a compact, expressive embedding vector.

##### 4. Unified Representation & Generation: GritLM
- **Pioneering Work**: `GritLM-7B / 8B` (Generative Representational Instruction-Tuning)
- **Mechanism**: Trains a single model to act as both a **dense embedding retriever** and a **generative autoregressive LLM**.
- **Implementation**: Uses a dual-mask schedule during training (bidirectional for embedding tasks, causal for generation tasks). A single deployed model can perform both retrieval and answer generation in RAG pipelines, cutting serving VRAM in half.

##### 5. Matryoshka Representation Learning (MRL)
- **Pioneering Work**: `OpenAI text-embedding-3`, `Nomic-Embed`, `BGE-M3`
- **Mechanism**: Optimizes InfoNCE loss simultaneously across nested vector sub-dimensions $d \in \{64, 128, 256, 512, 1024, 4096\}$.
- **Production Value**: Allows embeddings to be truncated without retraining—enabling ultra-fast coarse retrieval at 64 dimensions, followed by fine re-ranking at 4096 dimensions to minimize vector database index size.

---

## Module 2: Precision Formats & Mixed-Precision Training

### 1. Floating-Point Bit Layouts (FP32 vs. FP16 vs. BF16)

Under the IEEE 754 standard, floating-point representations consist of a Sign bit, Exponent bits (determining dynamic range), and Mantissa/Fraction bits (determining precision):

```text
Floating-Point Bit Anatomy:
FP32 (32-bit): [1 Sign] [ 8 Exponent Bits ] [      23 Mantissa Bits      ]
FP16 (16-bit): [1 Sign] [ 5 Exponent ] [    10 Mantissa    ]
BF16 (16-bit): [1 Sign] [ 8 Exponent Bits ] [ 7 Mantissa ]
```

| Precision Format | Total Bits | Exponent Bits | Mantissa (Fraction) Bits | Dynamic Range | Precision | Industrial Role |
|---|---|---|---|---|---|---|
| **FP32** | 32 bits | 8 bits | 23 bits | $10^{-38} \sim 10^{38}$ | High ($\approx 7$ decimal digits) | Master weights, optimizer states |
| **FP16** | 16 bits | 5 bits | 10 bits | $10^{-5} \sim 6.5 \times 10^4$ | Medium ($\approx 3$ decimal digits) | Legacy GPU inference; prone to **underflow/overflow** |
| **BF16** | 16 bits | 8 bits | 7 bits | $10^{-38} \sim 10^{38}$ (Identical to FP32) | Lower ($\approx 2$ decimal digits) | **De facto standard for modern LLM pre-training & SFT** |

---

### 2. FP16 Mixed-Precision & Dynamic Loss Scaling

#### Why is Dynamic Loss Scaling Necessary for FP16?
Because FP16 allocates only 5 bits to the exponent, its smallest positive normalized number is $2^{-14} \approx 6.1 \times 10^{-5}$. During backpropagation in deep transformers, small gradient values (e.g., $10^{-6} \sim 10^{-8}$) underflow to zero, stalling model training.

```text
FP16 Dynamic Loss Scaling Execution Loop:
┌─────────────────────────┐
│ 1. Forward Pass -> Loss │
└───────────┬─────────────┘
            ▼
┌─────────────────────────────────────────┐
│ 2. Scale Loss: Loss_scaled = Loss * S   │ ──> Shifts gradients into FP16 representable range
└───────────┬─────────────────────────────┘
            ▼
┌─────────────────────────────────────────┐
│ 3. Backward Pass -> Scaled Gradients    │
└───────────┬─────────────────────────────┘
            ▼
┌─────────────────────────────────────────┐
│ 4. Unscale Gradients: G = G_scaled / S  │ ──> Restores true mathematical scale
└───────────┬─────────────────────────────┘
            ▼
┌─────────────────────────────────────────┐
│ 5. Overflow Check (Scan for Inf / NaN)  │
└───────────┬─────────────────────────────┘
            ├───────────────────────────────┐
       [No Overflow (Valid)]          [Inf/NaN Detected (Overflow)]
            ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ 6. Update FP32 Master Wts │   │ 6. Skip Optimizer Step    │
│    Double S after N steps │   │    Halve S (S = S / 2)    │
└───────────────────────────┘   └───────────────────────────┘
```

---

### 3. Why BF16 Replaced FP16 as the Universal Standard

1. **Identical Dynamic Range to FP32**: Matching FP32's 8-bit exponent gives BF16 a dynamic range of $10^{\pm 38}$, eliminating gradient underflow;
2. **Zero Loss-Scaling Hyperparameter Tuning**: Removes the complexity of dynamic scale controllers, stabilizing large-scale distributed training (Megatron-LM, DeepSpeed);
3. **Native Tensor Core Acceleration**: Supported at full hardware throughput on NVIDIA Ampere (A100), Hopper (H100), and Blackwell architectures.

---

### 4. Classical Master Weights & Modern Low-Precision Paradigms (FP8 / NVFP4 / Stochastic Rounding)

#### (1) Classical Automatic Mixed Precision (AMP) Rationale

In conventional 16-bit mixed-precision training (e.g., PyTorch AMP, Megatron-LM, DeepSpeed ZeRO), forward and backward matrix multiplications execute in BF16/FP16, while **the optimizer workspace maintains an FP32 copy of the Master Model Weights alongside FP32 Adam momentum ($m$) and variance ($v$) states**.

```text
Classical Mixed-Precision Parameter Flow:
┌────────────────────────────────────────────────────────┐
│ GPU Tensor Cores (Fast Low-Precision GEMM Operations)  │
│ • Forward Activations: BF16 / FP16                     │
│ • Compute Weights:     BF16 / FP16                     │
│ • Backward Gradients:  BF16 / FP16                     │
└───────────────────────────┬────────────────────────────┘
                            │ (Accumulate gradients -> cast to FP32)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Optimizer Workspace (High-Precision Accumulation)      │
│ • Master Model Weights: FP32                           │
│ • Adam Momentum m:      FP32                           │
│ • Adam Variance v:      FP32                           │
│   Update: W_fp32 = W_fp32 - lr * (m / (sqrt(v) + eps)) │
│   Quantize/Cast W_fp32 -> BF16 for next forward step   │
└────────────────────────────────────────────────────────┘
```

**Why Does Deterministic Round-to-Nearest Require FP32 Master Weights?**
- **Machine Epsilon ($\epsilon_{\text{mach}}$) Limitations**:
  - BF16 has only 7 mantissa bits, yielding $\epsilon_{\text{mach}} = 2^{-7} \approx 7.8 \times 10^{-3}$;
  - FP16 has 10 mantissa bits, yielding $\epsilon_{\text{mach}} = 2^{-10} \approx 9.7 \times 10^{-4}$.
- **The "Swallowing" / Cancellation Problem**:
  In deep learning optimization, single-step parameter deltas $\Delta W = -\eta \frac{m}{\sqrt{v} + \epsilon}$ are frequently tiny ($10^{-5} \sim 10^{-6}$). Under standard hardware **Round-to-Nearest-Even**, when $|\Delta W| < \frac{\epsilon_{\text{mach}}}{2} \cdot |W|$:
  $$\text{Round}(W + \Delta W) = W$$
  Adding a microscopic delta to a unit-scale weight ($W = 1.0$) causes the lower bits to be completely truncated. Direct accumulation on BF16/FP16 weights without high-precision accumulators results in stalled convergence.

---

#### (2) Is FP32 Strictly Mandatory? SOTA Low-Precision Innovations

**The requirement that "master weights and optimizer states must unconditionally remain FP32" applies strictly to classical baseline AMP. Modern hardware and research have introduced major low-precision alternatives:**

```text
Precision Hierarchy & Optimizer Evolution:
┌───────────────────────┬───────────────────────┬───────────────────────┬───────────────────────┐
│ 1. Classical AMP(2017)│ 2. 8-Bit Optimizers   │ 3. FP8 Training(Hopper│ 4. Native NVFP4 (B200)│
├───────────────────────┼───────────────────────┼───────────────────────┼───────────────────────┤
│ • GEMM: BF16 / FP16   │ • GEMM: BF16 / FP16   │ • GEMM: FP8 (E4M3/E5M2│ • GEMM: NVFP4 (E2M1)  │
│ • Master: FP32        │ • Master: FP32 / BF16 │ • Master: FP32 / BF16 │ • Microscaling Vectors│
│ • Adam (m,v): FP32    │ • Adam (m,v): 8-Bit   │ • Adam: FP8 / BF16    │ • Scaled Accumulators │
└───────────────────────┴───────────────────────┴───────────────────────┴───────────────────────┘
```

##### 1. Stochastic Rounding (SR): Eliminating FP32 Master Weights
- **Unbiased Expectation**: Replaces deterministic rounding with probabilistic rounding:
  $$\text{SR}(x) = \begin{cases} \lfloor x \rfloor & \text{with prob } 1 - \frac{x - \lfloor x \rfloor}{\delta} \\ \lceil x \rceil & \text{with prob } \frac{x - \lfloor x \rfloor}{\delta} \end{cases}$$
- **Mathematical Theorem**: $\mathbb{E}[\text{SR}(x)] = x$. Even if $\Delta W = 10^{-6}$ is far below the machine epsilon of BF16, it maintains a non-zero probability ($10^{-4}$) of flipping the least significant bit. Over millions of iterations, **the expected accumulation matches full precision**, enabling pure 16-bit weight updates without allocating extra FP32 master weight memory.

##### 2. 8-Bit Optimizers (bitsandbytes / Block-wise Dynamic Quantization)
- Tim Dettmers et al. proved that Adam momentum $m$ and variance $v$ can be quantized into 8-bit non-linear (quantile/FP8) formats with dynamic per-block scales (e.g., 2048 elements/block), cutting optimizer memory by 75% without performance loss.

##### 3. FP8 Mixed-Precision Training (Hopper H100 / Transformer Engine / MS-AMP)
- **E4M3 (1 sign + 4 exponent + 3 mantissa)**: Used for forward GEMMs and compute weights (higher numerical precision);
- **E5M2 (1 sign + 5 exponent + 2 mantissa)**: Used for backward gradients (wider dynamic range to prevent underflow);
- **Delayed Scaling**: Tracks running maximum absolute values across recent steps to calibrate dynamic FP8 scale factors.

##### 4. Native NVFP4 (E2M1) Microscaling Training on NVIDIA Blackwell
NVIDIA Blackwell (B200 / GB200) introduces native **NVFP4 Tensor Cores** and **Microscaling Formats (MXFP4 / OCP)**:
- **NVFP4 Bit Layout (E2M1)**: Consists of 1 sign bit, 2 exponent bits, and 1 mantissa bit, scaled by an 8-bit (E8M0) microscaling factor shared across 16 or 32 elements;
- **How Does Native NVFP4 Training Execute?**
  1. **Compute Phase (GEMMs)**: Forward and backward matrix multiplications ($QK^T$, FFNs) run on 5th-gen Tensor Cores in **NVFP4 (E2M1)**, achieving up to 8x higher FLOP throughput compared to BF16;
  2. **Accumulation & Weight Re-quantization**: Because NVFP4 has only 16 discrete representable values, direct delta addition would cause immediate quantization collapse. Thus:
     - **Master Accumulators** are maintained in high precision (FP32, BF16 with Stochastic Rounding, or high-precision micro-blocks);
     - After each optimizer update, weights are dynamically re-quantized into NVFP4 blocks with updated scale vectors for the subsequent training step.

---

## Module 3: Next-Token Loss at Random Initialization & Step-0 Sanity Check (Initial Loss & ln(V) Derivation)

Before committing thousands of GPU hours to pre-train large language models, **the exact numerical value of the Step-0 loss is the single most critical empirical sanity check** to verify parameter initialization, causal attention masking, label alignment, and numerical stability.

```text
Step-0 Initial Loss Diagnostic Decision Tree:
┌────────────────────────────────────────────────────────────────────────┐
│ Compute Step-0 Forward Pass Loss L_0                                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│ L_0 ≈ ln(V)  │             │  L_0 ≫ ln(V) │             │  L_0 ≪ ln(V) │
│  (± 0.05)    │             │ (e.g. 20.0+) │             │  (e.g. 3.5)  │
├──────────────┤             ├──────────────┤             ├──────────────┤
│  ✅ PASS     │             │ ❌ Exploding │             │ ❌ Data Leak │
│  Proceed run │             │ Init / Scale │             │ Unmasked Pad │
└──────────────┘             └──────────────┘             └──────────────┘
```

---

### 1. Mathematical Derivation of Initial Cross-Entropy Loss ($\ln V$)

Consider an autoregressive language model with vocabulary size $V$. At sequence position $t$, the output linear projection produces unnormalized logits $z = [z_1, z_2, \dots, z_V] \in \mathbb{R}^V$. For target token $c \in \{1, 2, \dots, V\}$, standard cross-entropy loss is:

$$\mathcal{L} = -\ln P(y = c \mid x_{<t}) = -\ln \left( rac{e^{z_c}}{\sum_{v=1}^V e^{z_v}} ight) = -z_c + \ln \left( \sum_{v=1}^V e^{z_v} ight)$$

#### (1) Uniform Probability Distribution Under Random Initialization
Under standard zero-mean weight initializations (e.g. Gaussian $\mathcal{N}(0, \sigma^2)$ with $\sigma pprox 0.02$ or Xavier/He initialization):
- All logits $z_v$ have expectation $\mathbb{E}[z_v] pprox 0$ with small variance;
- The softmax probabilities approximate a uniform categorical distribution:
  $$P(y = v) = rac{e^{z_v}}{\sum_{j=1}^V e^{z_j}} pprox rac{1}{V}$$
- The cross-entropy loss simplifies strictly to the natural log of vocabulary size:
  $$\mathcal{L}_{	ext{init}} = -\ln \left(rac{1}{V}ight) = \ln V$$

#### (2) Second-Order Taylor Expansion with Non-Zero Logit Variance
If initial logits $z_v \sim 	ext{i.i.d. } \mathcal{N}(0, \sigma_z^2)$, expanding the Log-Sum-Exp function yields:

$$\mathbb{E}[\mathcal{L}_{	ext{init}}] = -\mathbb{E}[z_c] + \mathbb{E}\left[\ln \sum_{v=1}^V e^{z_v}ight] pprox \ln V + rac{\sigma_z^2}{2} + \mathcal{O}(\sigma_z^4)$$

For well-calibrated initializations ($\sigma_z \le 0.1$), $rac{\sigma_z^2}{2} < 0.005$, making $\mathcal{L}_{	ext{init}} pprox \ln V$ extremely accurate.

---

### 2. Theoretical Initial Loss Across Frontier Foundation Models

| Foundation Model | Vocabulary Size $V$ | Theoretical Step-0 Loss: $\ln V$ (nats/token) |
|---|---|---|
| **GPT-2 / GPT-3** | $50,257$ | $\ln(50257) pprox \mathbf{10.825}$ |
| **LLaMA-1 / LLaMA-2 / Mistral-7B** | $32,000$ | $\ln(32000) pprox \mathbf{10.373}$ |
| **LLaMA-3 / LLaMA-3.1 / LLaMA-3.3** | $128,256$ | $\ln(128256) pprox \mathbf{11.762}$ |
| **DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1** | $129,280$ | $\ln(129280) pprox \mathbf{11.770}$ |
| **Qwen-2 / Qwen-2.5** | $152,064$ | $\ln(152064) pprox \mathbf{11.932}$ |
| **Gemma / Gemma-2** | $256,000$ | $\ln(256000) pprox \mathbf{12.453}$ |

---

### 3. Theoretical Impact of Temperature and Label Smoothing

#### (1) Temperature Scaling $T$
When temperature $T$ is applied to logits ($z / T$):
$$\mathcal{L}(T) = -rac{z_c}{T} + \ln \left( \sum_{v=1}^V e^{z_v / T} ight)$$
- As $T 	o \infty$: Logit variance vanishes, driving probabilities strictly uniform and $\mathcal{L} 	o \ln V$.
- For $T < 1.0$: Amplifies logit fluctuations, slightly increasing initial loss variance ($\mathbb{E}[\mathcal{L}] pprox \ln V + rac{\sigma_z^2}{2 T^2}$).

#### (2) Label Smoothing $\epsilon$
With label smoothing target distribution $q(v) = (1-\epsilon)\mathbb{I}(v=c) + rac{\epsilon}{V}$:
$$\mathcal{L}_{	ext{smooth}} = -(1-\epsilon)\ln P(c) - rac{\epsilon}{V}\sum_{v=1}^V \ln P(v)$$
Substituting uniform random initialization ($P(v) pprox rac{1}{V}$):
$$\mathcal{L}_{	ext{smooth}} = -(1-\epsilon)\ln \left(rac{1}{V}ight) - rac{\epsilon}{V} \cdot V \ln \left(rac{1}{V}ight) = \ln V$$
> 💡 **Core Insight**: **Label smoothing does NOT alter the expected initial loss at random initialization—it remains strictly equal to $\ln V$!**

---

### 4. Tokenizer Effects & Vocabulary Padding

1. **Per-Token Loss vs. Sequence-Level Compression**:
   - Expanding vocabulary from 32K (LLaMA-2) to 128K (LLaMA-3) increases per-token initial loss from 10.37 to 11.76 ($\Delta pprox +1.39$ nats/token);
   - **However**: Larger vocabularies compress text more efficiently (requiring $\sim 30\%$ fewer tokens for identical text), so total document-level cross-entropy loss is actually lower.
2. **Vocabulary Alignment Padding (GPU Tensor Core Alignment)**:
   - In production frameworks (Megatron-LM / HuggingFace), vocabularies are often padded to multiples of 64 or 128 (e.g., $V_{	ext{embed}} = 32,256$ for $V_{	ext{actual}} = 32,000$ to optimize memory bus alignment);
   - **Note**: The expected initial loss will evaluate to $\ln(V_{	ext{embed}})$, matching the actual projection head dimension.

---

### 5. The Step-0 Pre-Training Sanity Check Diagnostic Matrix

```text
Step-0 Diagnostic Protocol:
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Observed Loss           │ Root Causes & Immediate Fixes                                          │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. L_0 ≈ ln(V) ± 0.05   │ ✅ PASS. Initialization, causal mask, projections, and loss are valid. │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. L_0 ≫ ln(V)          │ ❌ Initialization variance too large; missing 1/√(2*N_layers) residual │
│    (e.g., L_0 = 25.0)   │    scaling causing logit explosion; LayerNorm weights misconfigured.   │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. L_0 ≪ ln(V)          │ ❌ Fatal Data Leakage: Causal attention mask missing (model looks ahead);│
│    (e.g., L_0 = 3.5)    │ ❌ Target Padding Leakage: Missing `ignore_index=-100` on [PAD] tokens;│
│                         │ ❌ Off-by-one label alignment missing (`shift_labels = input_ids[1:]`).│
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. L_0 = NaN / Inf      │ ❌ Missing 1/√d_k attention scale causing softmax exponent overflow;   │
│                         │ ❌ FP16 dynamic loss scale initial value too aggressive.               │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## Module 4: High-Yield Interview Rapid-Fire

### Q1: Why is BERT's masked bidirectional architecture unsuitable for modern foundation LLMs?
> **Answer**:
> 1. Masked Language Modeling (MLM) breaks the causal autoregressive chain, preventing the model from naturally generating variable-length text sequences;
> 2. BERT lacks a causal KV cache. Generating each new token requires a full $O(S^2)$ forward pass over the entire sequence, making open-ended generation computationally prohibitive.

### Q2: When an LLM training run encounters NaN loss, what is the systematic debugging sequence?
> **Answer**:
> 1. Check for gradient overflow under FP16 and verify dynamic loss scale values;
> 2. Ensure the $\frac{1}{\sqrt{d_k}}$ attention scale factor is active to prevent softmax exponent overflow;
> 3. Verify that normalization layer $\epsilon$ values (LayerNorm/RMSNorm) are sufficient ($10^{-5} \sim 10^{-6}$) to avoid division by zero;
> 4. Audit dataset batches for corrupted all-zero inputs or extreme length anomalies.

### Q3: Why must the Step-0 loss of an autoregressive language model equal $\ln V$? What does it mean if $\mathcal{L}_0 = 2.0$?
> **Answer**:
> 1. **Mechanism**: At random initialization with zero-mean weights, the model assigns an approximately uniform probability $P(y=c) = \frac{1}{V}$ across all $V$ candidate tokens. The cross-entropy loss is therefore $-\ln(1/V) = \ln V$.
> 2. **Loss = 2.0 Diagnosis**: Indicates a **severe implementation bug or bidirectional data leakage**:
>    - **Cause 1 (Broken Causal Mask)**: The attention mask failed to enforce causal lower-triangular masking, allowing self-attention to attend directly to future target tokens.
>    - **Cause 2 (Unmasked Padding)**: Target batch padding tokens lacked `ignore_index=-100`, allowing the model to trivially predict constant `[PAD]` tokens.
>    - **Cause 3 (Unshifted Labels)**: The target labels were not shifted by one position relative to inputs (`labels = input_ids`), causing the model to simply copy the current input token.
