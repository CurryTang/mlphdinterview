# 10 · Classic Tech Report Reading: MAI-Thinking-1 Architecture & Slide Notes

The frontier of foundation models has entered an era of deep co-design between physical systems hardware and multi-thousand-step reinforcement learning (RL). In June 2026, the Microsoft AI (MAI) team released their landmark 109-page technical report: **"MAI-Thinking-1: Building a Hill-Climbing Machine"**. The report offers an end-to-end disclosure of a trillion-parameter Mixture-of-Experts (MoE) system trained from 29.2T unique human-written tokens, tilted through mid-training, and pushed through thousands of asynchronous RL steps powered by 4,096 inference GPUs and 768 learner GPUs.

This chapter systematically dissects this classic technical report alongside its accompanying 80-slide deep-dive presentation deck, covering:
1. **Model Architecture & Interconnect Co-Design**: Dense/MoE alternation, LatentMoE projection, and a $23\%$ reduction in All-to-All communication traffic.
2. **Pre-training Ablations & Scaling Laws**: The Ladder matrix, Efficiency Gain (EG) metric, and ranking inversion pitfalls across compute scales.
3. **Real-World Production Regimes**: Why production $TPP \gg 20$, data repetition alignment in small ablations, and Muennighoff's 4-epoch exponential value decay law.
4. **Data Engineering & Mixing Laws**: Governance over 29.2T human-only tokens, convex Mixing Laws formulation, and narrow domain saturation dynamics.
5. **Mid-Training & Training Recipes**: Sampling threshold theory for RL, zero-initialized attention to prevent router collapse, and the geometric trade-offs between AdamW and Muon on spectral balls.
6. **YOLO Training Infrastructure**: $90.0\%$ production Goodput across 8,192 GB200 GPUs, bit-exact deterministic restarts, and MFU bottleneck migration.
7. **Key RL Technologies**: Fully expanded token-level GRPO objective, wide clipping bands ($\epsilon=0.6$) for asynchronous staleness, dynamic entropy steering, and self-distillation after collapse.
8. **Distributed RL Engine Rocket**: Decoupled physical pools (4,096 samplers + 768 learners), exact numerics replay (routing & Top-P mask replay), and compiled point-to-point RDMA weight transfer plans.

---

## Slide Deck PDF Viewer & Resource Index

<div style="margin: 20px 0; padding: 18px; border: 1px solid var(--border-color, #e2e8f0); border-radius: 8px; background: var(--bg-subtle, #f8fafc);">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
    <div>
      <strong style="font-size: 16px;">📄 Full Slide Deck PDF (80 Slides) · MAI-Thinking-1: Building a Hill-Climbing Machine</strong>
      <div style="font-size: 13px; color: var(--text-muted, #64748b); margin-top: 4px;">Author: Zhikai Chen · Accompanying slides for the Microsoft AI technical report deep dive</div>
    </div>
    <a href="/slides/mai_thinking_1_v2.pdf" target="_blank" rel="noopener noreferrer" download="mai_thinking_1_v2.pdf" style="padding: 8px 16px; background: #2563eb; color: #ffffff; border-radius: 6px; text-decoration: none; font-weight: 500; font-size: 14px; display: inline-flex; align-items: center; gap: 6px;">
      ⬇️ Download / Open Fullscreen PDF
    </a>
  </div>
  <iframe src="/slides/mai_thinking_1_v2.pdf" width="100%" height="560px" style="border: 1px solid #cbd5e1; border-radius: 6px; background: #ffffff;" title="MAI-Thinking-1 Slides PDF Viewer"></iframe>
</div>

---

## Executive Overview & Three Core Systems Theses

### 1. Model Specifications & MoE Comparison Matrix

Built on top of the MAI-Base-1 checkpoint, MAI-Thinking-1 uses a 78-layer Transformer architecture with a native 256K context window. It packs 962B total parameters, activating 34.7B parameters per token (an activation ratio of only $3.6\%$). Pre-training consumed 30T tokens, followed by 3.55T mid-training tokens on 8,192 NVIDIA GB200 GPUs. Post-training asynchronous RL peaked at 4,096 inference GPUs + 768 learner GPUs (GB300).

| Model | Release | Total Params | Active Params | Active Ratio | Routed + Shared | Expert Sparsity | MoE Layers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MAI-Thinking-1** | 2026.06 | **962B** | **34.7B** | **3.6%** | **8 / 512 + 0** | **64×** | **39 / 78 (50%)** |
| Kimi K2 | 2025.07 | 1T | 32B | 3.2% | 8 / 384 + 1 | 48× | 60 / 61 (98%) |
| DeepSeek-V3.2-Exp | 2025.09 | 671B | 37B | 5.5% | 8 / 256 + 1 | 32× | 58 / 61 (95%) |
| DeepSeek-V4-Pro | 2026.04 | 1.6T | 49B | 3.1% | 6 / 384 + 1 | 64× | ~ / 61 |
| Llama 4 Maverick | 2025.04 | 400B | 17B | 4.3% | 1 / 128 + 1 | 128× | 24 / 48 (50%) |
| Qwen3-235B-A22B | 2025.04 | 235B | 22B | 9.4% | 8 / 128 + 0 | 16× | 94 / 94 (100%) |

On benchmark evaluations (Pass@1, mean of 4 independent samples), MAI-Thinking-1 demonstrates competitive reasoning capabilities:
- **AIME 2025**: $97.0$ (surpassing Claude 3.7 Sonnet 4.6 at $95.6$)
- **GPQA Diamond**: $84.2$ (Sonnet 4.6 at $89.9$)
- **LiveCodeBench v6 (LCB v6)**: $87.7$
- **SWE-bench Verified**: $73.5$ (Sonnet 4.6 at $79.6$)
- **SWE-bench Pro**: $52.8$
- **Terminal-Bench**: $46.0$ (Sonnet 4.6 at $59.1$)

![MAI-Thinking-1 Overview](/slides/mai-thinking-1/slide-02.png)

### 2. Three Core Systems Claims

The central narrative of the MAI report hinges on three foundational engineering claims:

![Three Claims](/slides/mai-thinking-1/slide-05.png)

1. **Decide small, check big, with the right yardstick**
   - Every pre-training architecture decision, data mix, and optimizer adjustment is priced in compute saved via an Efficiency Gain (EG) metric across an 8-size model ladder.
   - Engineers must guard against false signals: ideas that look like clear winners on small models or on a per-FLOP basis frequently invert when evaluated on 1T scales, in physical wall-clock hours, or after thousands of RL steps.
2. **Design for the machine, not for FLOPs**
   - Conventional MoE designs optimize for arithmetic FLOPs while disregarding physical All-to-All network bottlenecks across GPUs.
   - By alternating dense and MoE layers and projecting tokens down before dispatch, MAI compresses inter-GPU traffic to $23\%$ of baseline MoEs. Bit-exact determinism guarantees checkable, rapid restarts across 8K-GPU clusters.
3. **Keep asynchronous RL alive for thousands of steps without collapse**
   - Starting from a cold-start base model devoid of synthetic reasoning traces, MAI dispenses with KL penalties and entropy bonuses. Instead, it relies on a wide clipping band ($\epsilon=0.6$), an asymmetric upper-bound parameter $k$, hard ratio caps, exact numerics replay, and self-distillation to prevent premature entropy collapse.

---

## Part 1: Model Architecture & Communication Co-Design

### 1. MoE Execution Dynamics & The All-to-All Bottleneck

At 1T parameters, expert weights cannot fit on a single GPU. Under Expert Parallelism (EP), experts are partitioned across GPUs. Following attention computation, the router selects Top-K experts for each token. Because chosen experts reside on different nodes, tokens must be routed through an All-to-All collective network during dispatch and aggregated back during combine.

![EP Toy Example](/slides/mai-thinking-1/slide-08.png)

Consider a minimal toy setup with 4 GPUs, 8 experts, and Top-1 routing:
1. **Before Dispatch**: Each GPU holds tokens from its local batch. The router computes target expert IDs and destination GPU IDs.
2. **Dispatch Phase (All-to-All)**: Each GPU transmits distinct token subsets across NVLink/InfiniBand to host GPUs.
3. **Expert Compute**: Local experts process incoming tokens concurrently.
4. **Combine Phase (All-to-All)**: Expert outputs are returned to their origin GPUs and weighted by gating coefficients.

**The Physical Cost**: Token dispatch consumes physical wall-clock time rather than arithmetic FLOPs. Arithmetic FLOP accounting ignores network latency, yet subsequent Transformer layers stall until combine finishes. With deep MoE stacks, All-to-All communication latency becomes the dominant bottleneck throttling Model FLOPs Utilization (MFU).

![Communication Bottleneck](/slides/mai-thinking-1/slide-09.png)

### 2. Two Levers for Communication Reduction: Dense-MoE Alternation & LatentMoE

To eliminate interconnect saturation, MAI implemented two co-designed architectural levers:

![Two Levers for Communication Reduction](/slides/mai-thinking-1/slide-10.png)

1. **Alternating Dense FFN and MoE FFN Layers**
   - Prior models (e.g., DeepSeek-V3, Kimi K2) deploy MoE across virtually all layers (58~60 of 61).
   - MAI structures its 78-layer backbone into 39 repeating blocks, each consisting of one Dense FFN layer followed by one MoE FFN layer (only 39 MoE layers total). Dense FFNs execute entirely on local GPUs without emitting a single inter-GPU packet, instantly slicing global All-to-All traffic by $50\%$.
   - To retain 962B parameters despite having half the MoE layers, MAI scales each MoE layer to 512 routed experts, selecting 8 per token.
2. **Pre-Dispatch Dimensionality Compression (LatentMoE)**
   - The primary hidden dimension is $d_{\text{model}} = 6656$. Transmitting 6656-wide vectors across networks incurs immense serialization overhead.
   - MAI introduces a local linear projection that contracts token embeddings down to $d_{\text{latent}} = 3072$ immediately prior to All-to-All dispatch. Tokens traverse the network as 3072-wide vectors; remote experts execute intermediate projections and return 3072-wide vectors; local GPUs subsequently project them back up to 6656.
   - Crucially, the router continues to score all 512 experts using the uncompressed 6656-dimensional representation.

Quantitative traffic reduction per token:
- **Baseline (78 MoE layers, 6656-dim transmission)**: $100\%$ baseline traffic ($78 \times 6656$).
- **Lever 1 (Dense Alternation)**: $39 \times 6656 = 50\%$ traffic.
- **Lever 2 (LatentMoE Compression)**: $39 \times 3072 = 23.08\%$ traffic.

Global communication volume drops to less than one-fourth ($23\%$) of conventional designs. Furthermore, all All-to-All communications are strictly confined within a single 64-GPU NVLink rack, bypassing inter-rack optical links.

### 3. Empirical Ablations: Discarding the Shared Expert

Leading open architectures (DeepSeek-V3, Kimi K2) incorporate shared experts that every token must activate. MAI evaluated this design directly (Table 2):

![Ablation on MoE Layout](/slides/mai-thinking-1/slide-11.png)

- **Layout A (MoE Everywhere, 8/384)**: Lost across both per-FLOP efficiency and wall-clock training hours.
- **Layout B (MoE Everywhere + Shared Expert, 7/384 + 1)**: Matched MAI on per-FLOP efficiency ($1.03$ vs $1.0$), but collapsed to $0.82$ in throughput per wall-clock hour ($18\%$ slowdown) due to doubling the number of MoE dispatch layers.
- **Conclusion**: MAI's alternating dense layers already provide an always-on shared path. Adding an explicit shared expert delivered negligible gain while doubling network contention. MAI solidified an 8/512 purely routed configuration with zero shared experts.

---

## Part 2: Pre-Training Ablations & Scaling Laws

### 1. The Scaling Ladder & Metric Selection

Ablating architectural alternatives on trillion-parameter models is computationally prohibitive. MAI introduced **The Ladder**, a family of 8 models spanning 0.37B to 35.6B active parameters (L12, L18, L24, L30, L36, L42, L66, L78).

![The Ladder](/slides/mai-thinking-1/slide-13.png)

1. **Geometric Scaling**: Layer counts $L$ are multiples of 6 (matching the 5 local + 1 global attention cycle). Model width is constrained by:
   $$\text{Width} = \frac{256}{3} \times L \approx 85 L$$
   and maintained as a multiple of 512. All sizes train at an identical Tokens-Per-Parameter (TPP) ratio, isolating scale as the sole variable along the loss frontier.
2. **Why Academic Benchmarks Fail**: At sub-billion and few-billion scales, downstream benchmark scores hover near chance-level noise (Schaeffer et al. 2023 demonstrated that supposed emergent leaps frequently stem from non-linear threshold metrics).
3. **Smooth Held-Out Loss Aggregation**: Evaluation relies on held-out cross-entropy loss across ~40 diverse suites (vendor STEM/math, private code repositories, previous model interaction logs), with Code assigned a $50\%$ weight and STEM assigned $20\%$.

### 2. Efficiency Gain (EG) as the Universal Yardstick

Absolute cross-entropy margins (e.g., $2.31 \to 2.30$) are difficult to interpret and inherently compress as scale grows. MAI normalizes every architectural modification into equivalent compute saved:

![Efficiency Gain](/slides/mai-thinking-1/slide-15.png)

$$\text{EG} = \frac{C_{\text{baseline}}(L)}{C_{\text{candidate}}(L)}$$

If a candidate architecture matches the loss of a baseline consuming 119 compute units using only 100 units, its $\text{EG} = 1.19$ (a $19\%$ compute dividend). **Hard Rule: Only modifications demonstrating $\text{EG} > 1$ across all sizes of the ladder are approved for the production run.**

### 3. What Transferred vs. What Inverted

![Sparsity at Scale](/slides/mai-thinking-1/slide-16.png)

1. **Robustly Transferred Features**:
   - **Persistent Returns on Sparsity**: Scaling from 256 to 512 and 1024 experts drove consistent EG upward slopes as compute expanded (particularly on code and weighted averages). 512 experts was selected as the sweet spot balancing NVLink bandwidth and inference serving footprints.
   - **Global Batch Load Balancing**: Calculating auxiliary router loss $\mathcal{L}_{\text{aux}} = N \sum_{i=1}^N f_i P_i$ over global batches rather than micro-batches permitted local per-GPU routing specialization while ensuring global balance.
   - **Dropless Dynamic Routing**: Accommodating local expert skew by allocating dynamic execution buffers eliminated token dropping, preserving dense context semantics.
2. **Inversion Pitfalls (Did NOT Transfer)**:
   - **Data Mixture Inversion**: Data mixes crowned as optimal via small-scale grid searches saw their scaling curves intersect and invert at scale. Over-indexing on code yielded rapid initial convergence on small models but suffered sharp diminishing returns on large models.
   - **Cheap vs. Real Ruler Divergence**: Configurations that triumphed on per-FLOP loss (e.g., decaying learning rates to $0.01\times$ peak) failed during downstream RL because the resulting base model lacked the entropy needed for policy exploration.

---

## Part 3: From Ladder to Real-World Production Training

### 1. Why Production $TPP \gg 20$

Chinchilla scaling dictates that for a fixed FLOP budget, the compute-optimal split between tokens and active parameters is $TPP \approx 20$. However, production training across the industry operates at dramatically higher ratios:

![Why Real TPP >> 20](/slides/mai-thinking-1/slide-23.png)

- Llama 3 8B: $TPP = 1875$ (15T / 8B)
- Qwen3-235B: $TPP = 1636$ (36T / 22B active)
- **MAI-Thinking-1**: $TPP = 865$ (30T / 34.7B active)
- DeepSeek-V3: $TPP = 400$ (14.8T / 37B active)

Key system drivers include:
1. **Serving Amortization**: Chinchilla minimizes one-time training FLOPs, completely ignoring post-deployment serving economics. Serving cost scales with active parameter footprint over months or years of user queries and RL rollouts. Training a smaller model far past its Chinchilla optimum (500~1000 TPP) incurs a minor loss penalty during training while delivering persistent, multi-million dollar savings during inference (Sardana et al. 2024).
2. **Flat Optimization Valley**: Moving away from the theoretical training optimum incurs minimal loss penalty, while downstream utility continues to climb.
3. **MoE Active vs. Stored Parameter Asymmetry**: In MoE networks, total parameters vastly exceed active parameters per token, shifting the empirical optimum toward 100~200 TPP even under training-only metrics.

### 2. Repetition Asymmetry in Small-Scale Ablations

A critical hidden flaw in small-model ablations is **epoch distortion**:

![Data Repetition in Ablations](/slides/mai-thinking-1/slide-24.png)

- Suppose a high-quality math dataset contains 1T unique tokens, allocated a $5\%$ share in the overall training mix.
- In a 30T production run, $30\text{T} \times 5\% = 1.5\text{T}$ tokens are trained, consuming the math corpus for **1.5 full epochs**.
- If ablated naively on a 4B ladder model running 150 TPP (600B total tokens), the math slice consumes only $600\text{B} \times 5\% = 30\text{B}$ tokens, representing a mere **0.03 epochs** (the model sees only fresh, non-repeating data).
- **The Bias**: Because fresh tokens provide significantly stronger learning gradients than repeated ones, naive ablations massively overrate scarce high-quality corpora, assigning them unrealistically high mixture weights.
- **The Remedy**: In ladder ablations, MAI deliberately downsamples scarce datasets so that their repetition count in the ablation matches their exact repetition count in the full-scale 30T run.

### 3. Repeated Data Value Decay (Muennighoff et al., NeurIPS 2023)

When web-scale unique text is exhausted, how many epochs can a model revisit before memorization takes over? Muennighoff et al. analyzed over 400 training runs, establishing an exponential decay relationship:

![Repeated Data Laws](/slides/mai-thinking-1/slide-25.png)

The effective value of a token on its $R$-th repetition decays as:
$$\text{Value}(R) \approx e^{-R / 15}$$
- Within **$\le 4$ epochs**, loss reduction is virtually identical to training on fresh tokens ($>93\%$ efficiency).
- Beyond 16 epochs, marginal returns deteriorate sharply; by 40 epochs, repetitions yield near-zero generalization, prompting severe verbatim memorization.
- **Production Guardrails**: In MAI's 30T pre-training, the highest repetition frequency was math at **5.28 epochs**, with a strict cluster-wide hard cap of **8 epochs** across all data sources.

---

## Part 4: Pre-Training Data Engineering & Mixing Laws

### 1. Corpus Governance: Human-Only & Multi-Tier Deduplication

MAI explicitly avoided synthetic text and third-party model distillation, assembling a curated 29.2T-token human-written dataset:

![Data Sources and Cleaning](/slides/mai-thinking-1/slide-27.png)

1. **Multi-Stage Filtering**:
   - 1.2T raw web pages were pruned to 794B via policy and safety blocklists;
   - Exact hash deduplication trimmed volume to 423B;
   - MinHash locality-sensitive hashing grouped pages with $\ge 80\%$ similarity, retaining 190B pages (one in four filtered pages);
   - Code repositories underwent embedding-based clustering to strip redundant syntactic templates.
2. **Benchmark Decontamination**: All crawls from huggingface.co and mirrors were purged; 20-gram fuzzy matches against public benchmarks (SWE-bench Verified, HumanEval, GSM8k) were rigorously removed.

### 2. The 30T Compositional Breakdown

Dataset weighting was governed by marginal held-out utility rather than raw web availability:

![30T Data Mix](/slides/mai-thinking-1/slide-28.png)

- **Code**: Read 2.22 epochs, commanding **$55\%$** of all training tokens (compared to $\sim 17\%$ in Llama 3).
- **STEM Data**: Read 2.17 epochs.
- **Mathematics**: Read 5.28 epochs (highest repetition intensity).
- **Books & Academic Journals**: Read 1.65 epochs.
- **General Web Text**: Read 0.55 epochs (the bulk of noisy web scrapes were left untouched).
- **Multilingual Corpora**: Read 0.06 epochs (strictly gated for linguistic quality).

### 3. Formalization of Mixing Laws

Rather than performing unguided heuristic searches, recent theory (Ye et al., ICLR 2025; RegMix) formulates data mixture optimization as a convex regression problem:

![Mixing Laws](/slides/mai-thinking-1/slide-30.png)

$$\mathcal{L}_i(r) = c_i + k_i \exp\left( \sum_{j} t_{ij} r_j \right)$$

- $r_j$: Proportion of domain $j$ in the mixture ($\sum r_j = 1$).
- $c_i$: Irreducible baseline cross-entropy on evaluation domain $i$.
- $t_{ij}$: Cross-domain transfer coefficient. If $t_{ij} < 0$, training on corpus $j$ directly reduces loss on domain $i$ (positive transfer); if $t_{ij} > 0$, domain $j$ crowds out domain $i$'s representational capacity.

Empirical validations highlight distinct cross-domain dynamics: mathematical PDFs aggressively drive down physics evaluation loss ($t < 0$), whereas disproportionate code allocations eventually depress general linguistic reasoning.

**Narrow Domain Saturation**: As overall compute scales, highly specialized domains saturate their potential loss reduction much faster than broad general corpora, requiring engineers to dynamically down-weight niche domains at higher parameter scales.

---

## Part 5: Mid-Training Architectural Philosophy

### 1. Mid-Training Purpose & Theoretical Boundaries

Following the 30T pre-training run, MAI executed a 3.55T-token mid-training phase.

![What Mid-Training Is](/slides/mai-thinking-1/slide-34.png)

1. **Tilting the Mixture Without Injecting Novel Data**: Mid-training did not introduce unseen corpora. Instead, it rebalanced the sampling distribution, drastically amplifying the weights of code, mathematics, and complex reasoning traces.
2. **Foundational Principle: RL Only Sharpens Existing Sample Spaces**:
   $$\text{RL sharpens what the base can already sample.}$$
   Policy gradient updates require at least one generated candidate trajectory in a batch to achieve a positive advantage ($A_i > 0$). If a base model possesses a true zero probability of generating a viable line of reasoning, the advantage is zero, yielding zero policy gradient. Mid-training raises the base generation distribution above the critical sampling threshold.
3. **Context Window Staging**: Sequence lengths were held at 16K throughout pre-training to maximize compute throughput, and smoothly extended to the full 256K context during the final tail of mid-training.

---

## Part 6: Training Recipes & Optimizer Geometry

### 1. AdamW Tuning & Learning Rate Floors

While emerging models have explored alternative optimizers, MAI stayed with a production-hardened AdamW recipe:

![Optimizer Recipe](/slides/mai-thinking-1/slide-40.png)

- **Peak Learning Rate**: $2 \times 10^{-4}$ governed by cosine decay.
- **Learning Rate Floor**: Terminated at **$0.1\times$ peak** ($2 \times 10^{-5}$), diverging from the standard $0.01\times$ practice. While a higher floor yielded slightly higher final pre-training loss, it preserved vital policy entropy that proved indispensable during subsequent RL exploration.
- **Momentum Coefficients**: $\beta_1 = 0.95, \beta_2 = 0.925$ (smoothing variance estimation). The RL learner tuned this further to $\beta_1 = 0.95, \beta_2 = 0.95$.
- **Residual Dropout**: Injected $p = 0.15$ dropout on transformer block outputs prior to the residual addition. This mitigated MoE expert memorization on math data (repeated 5.28 epochs) and provided entropy stabilization during self-distillation.

### 2. Zero-Initialized Attention (Zero-Init Attention)

At step 0 of large-scale MoE training, initial random weights triggered severe router collapse: the busiest expert experienced an initial load skew of **$+298\%$**.

![Zero-Init Attention](/slides/mai-thinking-1/slide-41.png)

1. **Root Cause Analysis**: At step 0, random projection matrices produce near-uniform softmax distributions, causing attention heads to collapse into causal mean pooling. Every token absorbs an identical average of preceding tokens, homogenizing the representations fed into the MoE router. Identical vectors predictably activate identical expert subsets, destabilizing early training.
2. **The Fix**: Initialize the learnable scale gain of the attention output RMSNorm to **0**. At step 0, the attention branch emits zero, causing each block to behave strictly as an independent per-token MLP where token diversity is preserved. Attention contributions smoothly fade in as the gain parameters learn.
3. **Outcome**: Peak expert load skew plummeted from $+298\%$ to **$+59\%$**, delivering immediate EG gains across the ladder.

### 3. Geometric Unification of Optimizers: SGD, Adam, and Muon

The technical report contextualizes optimizer mechanics through first-order constrained optimization:

![Geometry of Optimizers](/slides/mai-thinking-1/slide-42.png)

$$\Delta W = \arg\min_{\Delta} \left\langle G, \Delta \right\rangle + \frac{\lambda}{2} \|\Delta\|^2$$

Different optimizers emerge from selecting different geometric norm balls:
1. **SGD (Euclidean $\ell_2$ Ball)**:
   $$\Delta W = -G$$
   Steps follow the steepest descent vector; dimensions with large gradients move aggressively, while small gradients remain static.
2. **Adam (Hypercube Box Constraint, $\ell_\infty$ Ball)**:
   Disregarding momentum smoothing, Adam updates scale as $-1 / \sqrt{v} \cdot g \approx -\text{sign}(g)$. Updates land on hypercube vertices, moving every parameter by an equal step size. Adam treats weight tensors as a "bag of independent numbers".
3. **Muon (Spectral Ball)**:
   $$\Delta W = -U V^T$$
   where $G = U \Sigma V^T$ represents the SVD of the gradient matrix. Updates are constrained within the orthogonal matrix manifold, granting equal energy across all singular directions. Muon treats weight tensors as holistic "linear mapping operators".

**Why MAI Maintained AdamW in Production**:
- **Diminishing Returns with Scale**: Stanford empirical studies (Wen et al. 2025) demonstrated that once AdamW baselines are tuned with equivalent rigor, Muon's effective compute speedup compresses from $2.0\times$ at 0.1B to $1.1\times$ at 1.2B.
- **ZeRO Optimizer Sharding**: MAI's YOLO infrastructure shards optimizer states across nodes. AdamW's per-element state shards trivially; Muon requires full-matrix Newton-Schulz orthogonalization iterations, introducing heavy inter-node communication overheads.
- **Stage Continuity**: Retaining AdamW into the RL phase avoided optimizer transition shocks.

---

## Part 7: YOLO Training Infrastructure & High Goodput Systems

Sustaining continuous training across 8,192 GB200 GPUs requires robust fault-tolerance. MAI engineered the YOLO infrastructure stack, securing a production-grade **$90.0\%$ Goodput** (Effective Compute Time / Total Wall-Clock Time).

![Reliability and Goodput](/slides/mai-thinking-1/slide-45.png)

### 1. Bit-Exact Deterministic Restarts

Standard distributed training allows non-deterministic floating-point reduction orders induced by GPU hardware atomic adds and network switch in-fabric aggregation (NVIDIA SHARP). This introduces run-to-run drift.

MAI disabled non-deterministic atomic operations and in-switch aggregation, enforcing fixed batch loading orders and deterministic Top-K routing:
- **Bit-Level Verification**: Following a hardware interruption, hot-standby workers reload checkpoints and replay failed steps. The resulting replayed loss values must match previously logged checkpoints down to the **exact bit representation**.
- **Diagnostic Power**: Any deviation exposes silent data corruption, faulty memory modules, or driver bugs immediately, preventing silently corrupted workers from poisoning model weights.

### 2. Architecture-System Co-Design Bottlenecks

Across iterations v2 through v5, architectural Efficiency Gain (EG) expanded from $1.0\times$ to $1.69\times$, driving continuous system refactoring:

![Architecture System Co-Design](/slides/mai-thinking-1/slide-46.png)

1. **v2 (Initial GB200 Launch)**: Enabled GPU-Direct RDMA, optimized attention kernels, and tuned ZeRO-2, lifting MFU from $18\%$ to $22\%$.
2. **v3 (Dropless Routing, EG $1.40\times$)**: Dynamic token distributions induced run-time skew. Overlapping communication with compute preserved $22\%$ MFU.
3. **v4 (512 Experts, Top-8, LatentMoE, 8K GPUs, EG $1.69\times$)**: Finer expert granularity reduced individual expert GEMM matrix dimensions, leaving GPUs idle waiting on CPU driver dispatch (MFU dropped to $16\%$). Upgrading to FlashAttention-4 and optimizing CPU kernel queues restored MFU to $20\%$.
4. **v5 (35B Active, 1T Total Params)**: ZeRO-3 weight gather collectives saturated interconnect bandwidth in backward passes, dropping MFU to $18\%$. Offloading activations allowed the cluster to revert to ZeRO-2, stabilizing long-run MFU at $20\%$.

---

## Part 8: Key RL Technologies & Collapse Prevention

MAI-Thinking-1 executed asynchronous RL directly from a human-data-only base model, without relying on external distilled reasoning traces.

### 1. The Token-Level GRPO Formulation

The training objective is formulated as token-level Group Relative Policy Optimization (GRPO):

![The RL Objective](/slides/mai-thinking-1/slide-48.png)

$$\mathcal{J}(\theta) = \frac{1}{\sum_{i=1}^G |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \min \left( \hat{r}_{i,t} A_i, \; \text{clip}\left(\hat{r}_{i,t}, \, 1 - \epsilon, \, (1 - \epsilon)^{-1} + k\right) A_i \right)$$

Components:
1. **Global Token Normalization**: The denominator divides by the aggregate token count $\sum |y_i|$ across all $G$ candidate rollouts (following DAPO), preventing long-winded answers from dominating policy gradients.
2. **Critic-Free Advantage Estimation**: For each prompt, $G$ candidate outputs are evaluated against verifiable answer keys:
   $$A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$$
3. **Importance Sampling Ratio**:
   $$r_{i,t} = \frac{\pi_\theta(y_{i,t} \mid q, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid q, y_{i,<t})}$$
4. **Hard Ratio Clamping**:
   $$\hat{r}_{i,t} = \text{clip}(r_{i,t}, \, 0, \, 50)$$
   Standard PPO leaves negative advantage cases ($A_i < 0, r > 1$) unbounded, precipitating catastrophic gradient spikes during exploration. Hard clamping prevents numerical divergence.
5. **Wide Clipping Band**: Setting $\epsilon = 0.6$ establishes a lower bound of $1 - \epsilon = 0.4$ and a baseline upper bound of $(1 - \epsilon)^{-1} = 2.5$. This wide band ensures that rollout trajectories generated up to 40 learner steps prior continue to deliver productive gradients.
6. **Elimination of KL & Entropy Bonuses**: Rigid KL penalties suppress complex multi-step reasoning exploration in asynchronous settings, while naive entropy bonuses encourage token-level gibberish.

### 2. Asynchronous Staleness Management

Across 4,096 inference and 768 training GPUs, generation and training overlap asynchronously:

![Staleness](/slides/mai-thinking-1/slide-49.png)

- Complex rollouts requiring extended reasoning chains take longer to generate, arriving at the learner 5~8 weight versions behind the current parameters.
- Rollouts exceeding an 8-version lag are dropped; rollouts within 1~8 versions are incorporated via wide-band importance weights.

### 3. Entropy Collapse Dynamics & Asymmetric Steering via $k$

During iterative RL, policies naturally collapse toward safe, low-entropy sequences. The policy entropy $H(\pi_\theta)$ can decay exponentially, resulting in severe mode collapse.

![Entropy Dynamics](/slides/mai-thinking-1/slide-50.png)

MAI modulated exploration dynamically via the asymmetric clipping parameter $k \in [0, 2.5]$:
- When entropy decay accelerates, expanding $k$ elevates the upper clipping ceiling $(1 - \epsilon)^{-1} + k$, permitting larger gradient rewards for high-probability novel exploration actions;
- Once policy entropy stabilizes, $k$ is contracted back toward zero.

![Entropy Control via k](/slides/mai-thinking-1/slide-51.png)

### 4. Self-Distillation for Policy Recovery

When localized policy collapse occurs, rolling back to days-old checkpoints incurs massive compute loss. MAI adopted **Self-Distillation**:

![Self-Distillation](/slides/mai-thinking-1/slide-53.png)

- High-advantage trajectories generated prior to the collapse are harvested into a multi-task dataset spanning code, mathematics, and agentic traces;
- The pre-collapse checkpoint acts as a teacher model, performing supervised fine-tuning with $p = 0.15$ residual dropout;
- This resets policy entropy and restores representational diversity, allowing the model to resume RL hill-climbing without restarting the run.

---

## Part 9: Three Specialist RL Climbs

MAI decoupled post-training into three parallel specialist pipelines that climb independently before being distilled into the unified foundation:

![Three Climbs Pipeline](/slides/mai-thinking-1/slide-55.png)

1. **STEM Climb**:
   - Relies exclusively on rigid, programmatic ground-truth verifiers (Answer Key Verifiers), discarding error-prone LLM-as-a-Judge evaluations.
2. **Agentic Climb**:
   - An exercise in environment engineering: isolated Docker containers, Linux shells, and live compilers supply ground-truth execution feedback and stack traces, driving self-correction.
3. **Helpfulness & Safety**:
   - **Safety as a Gate, Not a Weight**: Incorporating safety into a unified reward function encourages reward hacking along safety margins. MAI deployed safety as an independent pass/fail gate; only trajectories satisfying strict safety criteria qualify for reasoning reward allocation.

---

## Part 10: Distributed RL Infrastructure: Rocket

To eliminate serving-training bottlenecks, MAI developed the **Rocket** distributed RL engine.

![Rocket Architecture](/slides/mai-thinking-1/slide-60.png)

### 1. Decoupled Physical Compute Pools

- **Inference Pool (Samplers)**: 4,096 GPUs dedicated to high-throughput streaming autoregressive rollout generation.
- **Training Pool (Learners)**: 768 GPUs handling large-batch gradient backpropagation.
- Physical isolation avoids memory fragmentation and CUDA kernel recompilation churn inherent in colocated, time-sliced RL systems.

### 2. High-Fidelity Numerics Replay

Open-source RL stacks (verl, slime, SkyRL) typically recompute $\pi_{\text{old}}$ on learner GPUs. However, divergence in CUDA kernels, KV-cache precision, or reduction orders between inference engines (vLLM/SGLang) and training engines (Megatron-LM) causes subtle logit shifts that compound across long sequences, destabilizing policy ratios.

![Numerics Replay](/slides/mai-thinking-1/slide-61.png)

Rocket adheres to a strict "replay, don't correct" paradigm:
1. **Direct Sampling Probability Reuse**: Learners reuse the exact token probabilities logged by samplers during rollout generation.
2. **End-to-End BF16 Parity**: Enforces matched numerical precision across all rollout and learner kernels.
3. **MoE Routing Replay**: Expert selections recorded during sampling are strictly locked; learners are prohibited from re-routing tokens during backward passes.
4. **Tail Mask Replay**: Top-P (e.g., $97\%$) sampling masks are replayed during learner softmax calculations, preventing unselected long-tail logits from injecting noisy gradients.

### 3. Compiled Point-to-Point Weight Transfer Plans

Synchronizing hundreds of gigabytes of weights to 4,096 inference GPUs every 5 steps requires optimized networking:

![Weight Transfer Plan](/slides/mai-thinking-1/slide-62.png)

- **The Naive Approach**: Gathering weights on learners and broadcasting via NCCL causes massive network congestion.
- **Rocket's Compiled Schedule**:
  - At job startup, Rocket analyzes the tensor sharding layouts across learner ZeRO partitions and sampler TP/PP partitions;
  - It compiles a static point-to-point memory intersection map;
  - During synchronization, GPU-Direct RDMA routes only the exact intersecting tensor slices, bypassing global broadcast overhead. With 4 data-parallel groups feeding 36 serving clusters, synchronization latency drops dramatically.

### 4. Distributed RL Framework Comparison

| Dimension | Open RL Stacks (verl / slime / AReaL / SkyRL) | Rocket (Microsoft AI) |
| :--- | :--- | :--- |
| **Compute Topology** | Often shared pool (time-sliced); emerging split pools | **Strict physical split** (4,096 inference + 768 learner) |
| **Staleness Handling** | Synchronous barriers or capped version lag (2~8) | **Free-running streaming**, drops $>8$ version lag |
| **$\pi_{\text{old}}$ Computation** | Recomputed on learner via extra forward pass | **Reuses sampler-logged raw probabilities** |
| **Kernel Discrepancies** | Corrected via truncated importance sampling | **Strict replay** (locks routing and Top-P 97% tail mask) |
| **Long Context Footprint** | Heavy activation memory for forward pass on 1T/128K | **Zero extra forward pass overhead** on learner |
| **Weight Synchronization** | Gather + NCCL broadcast / CUDA IPC | **Compiled point-to-point RDMA intersection plan** |

---

## Critical Evaluation & Open Questions

While MAI-Thinking-1 establishes impressive engineering milestones, several methodological questions remain open:

![Open Questions](/slides/mai-thinking-1/slide-64.png)

1. **Ablation Transparency in RL**: While pre-training choices are backed by rigorous Efficiency Gain curves, RL decisions (such as $\epsilon=0.6$, ratio clamp 50, and dynamic $k$) are largely justified narratively ("we found") rather than via systematic ablations.
2. **The True Cost of Self-Distillation**: The report notes ten self-distillation interventions during RL collapses, but omits the SFT compute and lost training steps required to re-establish the RL climb.
3. **Mixing Law Extrapolation Limits**: Finalist data mixtures were verified at only $2.8\times$ search compute, despite theoretical indications that optimal mixture frontiers continue shifting across scale.
4. **Unexplored Alternatives**: The ladder investigated depth scaling exclusively, leaving $\mu\text{P}$ width transfer unaddressed, while spectral optimizers like Muon were bypassed without direct empirical head-to-head testing at the 1T scale.

---

## Appendix: 80-Slide Core Content Index

| Slide Range | Section | Key Concepts & Engineering Takeaways |
| :--- | :--- | :--- |
| **01 - 05** | Overview & Thesis | 962B/34.7B MoE specifications, benchmark scorecard, "Learned-not-inherited" philosophy, Three core claims |
| **06 - 11** | Part 1: Architecture | Alternating Dense/MoE layers, EP All-to-All bottleneck, LatentMoE 6656→3072 compression, $23\%$ traffic reduction, Shared expert ablation |
| **12 - 20** | Part 2: Ablations | The Ladder (8 sizes), held-out loss metric, Efficiency Gain (EG) formulation, Global batch balancing, Dropless routing, Metric inversion |
| **21 - 25** | Part 3: Real Run | Production $TPP \gg 20$, repetition alignment in small ablations, Muennighoff exponential decay $e^{-R/15}$, 8-epoch cap |
| **26 - 32** | Part 4: Pre-training Data | 29.2T human-only tokens, MinHash/clustering dedup, 55% code mix, Mixing Laws equation, Narrow domain saturation |
| **33 - 38** | Part 5: Mid-Training | Mix tilting without novel data, RL sampling feasibility hypothesis, 16K to 256K context extension |
| **39 - 43** | Part 6: Recipe & Optim | AdamW 0.1x LR floor, Zero-initialized attention to prevent router collapse, Geometric comparison of SGD/Adam/Muon on spectral balls |
| **44 - 46** | Part 7: YOLO Infra | 90.0% Goodput, Bit-exact deterministic training, MFU bottleneck migration across 8K GB200 GPUs |
| **47 - 54** | Part 8: RL Tech | Token-level GRPO objective, Wide clip ($\epsilon=0.6$) for staleness, Asymmetric $k$ entropy steering, Self-distillation recovery |
| **55 - 58** | Part 9: Three Climbs | STEM programmatic verifiers, Agentic sandbox execution feedback, Safety as an independent gate, Multi-specialist distillation |
| **59 - 63** | Part 10: Rocket RL | 4,096 inference + 768 learner split pool, Numerics replay (exact routing & Top-P mask replay), Compiled RDMA weight transfer plan |
| **64 - 65** | Closing & Takeaways | Open questions and critique, Summary of highest-ROI systems engineering paradigms |
| **66 - 80** | Backup Slides | Parallelism taxonomy, Spectral norm mathematical derivations, Industrial hurdles for Muon, FP8 GEMM vs FP32 precision boundaries |
