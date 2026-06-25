> For a better reading experience, please visit GitHub → Currytang → hitchhikers-guide-to-ml-phd-job-hunting
> This lecture will be especially detailed, since I’ll be doing related research :)

# Core Methods in Low-bit Quantization

## Background and Historical Development

Research on model quantization predates the rise of deep learning. In signal processing and communications, quantization has long been the basic tool for discretizing continuous signals (e.g., PCM encoding), and Shannon’s rate-distortion theory established its information-theoretic foundation. In the deep learning era, works such as BinaryConnect, BNN, and XNOR-Net from 2015-2016 were the first to explore extremely low-bit (1-bit) representations for weights and activations, attempting to replace floating-point multiplication with XNOR + popcount. However, the accuracy loss was too severe, and on modern GPU Tensor Cores they offered no meaningful speed advantage. Later, in 2018-2019, PACT and LSQ established the QAT (quantization-aware training) paradigm: by treating the clipping range and quantization step size as learnable parameters, they made 4-bit and even 2-bit training feasible under the STE framework. Meanwhile, mixed-precision training (FP16/BF16 + FP32 master weights) became standard practice, and the work of Micikevicius et al. established the core paradigm of "low-precision computation + high-precision accumulation."
From 2020 to 2022, post-training quantization (PTQ) gradually matured for CNNs: AdaRound modeled rounding direction as an optimizable variable, BRECQ introduced block-wise second-order reconstruction, and QDrop flattened the loss landscape by randomly skipping activation quantization, pushing low-bit PTQ for CNNs to a practically usable level. These methods provided the methodological foundation for later LLM quantization.
Quantization research for LLMs took off in 2022, driven primarily by the memory and inference cost pressure created by rapidly scaling model sizes. Dettmers et al. were the first to identify the emergent outlier phenomenon in large-model activations, where a few channels are more than 100× larger than the rest, and proposed LLM.int8() to address it with vector-wise mixed precision. SmoothQuant then used an equivalent mathematical transformation to shift the quantization difficulty of activations onto the weights in advance, enabling true W8A8 deployment. In the more aggressive 4-bit regime, GPTQ extended the second-order OBQ framework to tens-of-billions-scale models, while AWQ showed that protecting the weight columns corresponding to salient activations is the key. In 2024, QuaRot and SpinQuant used orthogonal rotations to eliminate outliers at the root, making end-to-end W4A4+KV4 possible; KIVI focused specifically on 2-bit compression of KV cache for long-context scenarios. In parallel, FP8 (E4M3/E5M2), natively supported on Hopper/Blackwell, is becoming the new baseline precision for training and inference.

---

## 1 Quantization Fundamentals

### 1.1 Uniform Quantization Formula

Map a floating-point tensor $x$ to a $b$-bit integer:

$$x_q = \text{clamp}\!\Big(\!\left\lfloor \frac{x}{s} \right\rceil + z,\; 0,\; 2^b - 1\Big)$$

Dequantization:

$$\hat{x} = s \cdot (x_q - z)$$

There are two key parameters:

- **Scale $s$**: controls the quantization step size, i.e., the real spacing between adjacent quantized values. A larger $s$ covers a wider floating-point range, but also increases the gap between adjacent quantization levels and lowers precision; a smaller $s$ gives higher precision, but narrows the representable range, so out-of-range values get clipped by `clamp`. In essence, $s$ determines "how large a numeric interval we cover with a finite set of quantization bins."
- **Zero-point $z$**: the integer-domain value corresponding to floating-point zero. When $z = 0$, floating-point zero maps exactly to integer zero (symmetric quantization); when $z \neq 0$, the integer range can be shifted, so the quantization interval need not be symmetric around zero and can more compactly cover one-sided skewed distributions (e.g., all-positive activations after ReLU).

Two common configurations:

- **Symmetric quantization**: $z = 0$, $s = \max(|x|)\;/\;(2^{b-1} - 1)$. The integer range is symmetric around zero, which is simple to implement, but wastes quantization levels if the distribution is heavily skewed.
- **Asymmetric quantization**: $z \neq 0$, $s = (x_{\max} - x_{\min})\;/\;(2^b - 1)$, $z = \lfloor -x_{\min}/s \rceil$. The zero-point $z$ aligns integer zero with the lower bound of the actual distribution, making more compact use of all $2^b$ quantization levels.

### 1.2 Terminology and Notation

**Quantization targets**: the main tensors in a model that can be quantized fall into the following categories, each with different difficulty and benefit trade-offs:

- **Weight**: model parameters, fixed during inference. Their distributions are relatively stable and easy to analyze offline, making them the easiest targets to quantize. Weight-only quantization is the most common starting point; GPTQ and AWQ both belong here. The main gains are smaller model size and reduced memory-bandwidth bottlenecks when loading weights.
- **Activation**: the inputs/outputs of each layer, which vary dynamically with the input data. Their distributions are unstable and prone to outliers, making them much harder to quantize than weights. However, if both weights and activations can be quantized, GEMM can run entirely in low-bit integers (e.g., INT8 × INT8), providing real compute acceleration rather than bandwidth savings alone.
- **KV Cache**: the cached Key and Value tensors during Transformer inference. In long-context settings, its memory footprint can far exceed that of the model weights themselves (e.g., LLaMA-2-7B can use tens of GB of KV cache at 128K context). Methods such as KIVI target this component and can compress it to 2-bit.
- **Gradient**: backpropagated gradients during training. In FP8 training, gradients are typically stored in E5M2 format (larger dynamic range). Their distributions are symmetric but contain sparse large values, so their quantization strategy differs from that of weights and activations.
- **Optimizer State**: Adam’s first moment $m$ and second moment $v$, each stored in FP32 and each occupying model-sized memory, making them a major part of training memory. The 8-bit optimizer in bitsandbytes compresses them to INT8, saving about 75% of memory.

**WxAy notation**: based on the categories above, quantization configurations are commonly described with abbreviations where W stands for Weight, A for Activation, and the number for bit width. For example:

- **W8A8**: 8-bit weights and 8-bit activations. The typical SmoothQuant configuration, where both weights and activations use INT8 for GEMM.
- **W4A16**: 4-bit weights while activations remain FP16. The typical GPTQ/AWQ configuration, where weights are dequantized back to FP16 before matrix multiplication during inference (weight-only quantization).
- **W4A4**: both weights and activations are 4-bit. The end-to-end low-bit configuration targeted by QuaRot/SpinQuant, which places the highest demands on kernel support.

You may also see notations such as **KV4** (4-bit KV cache) and **W1.58** (the ternary weights in BitNet b1.58).

**Why does quantization hurt model accuracy?** The root cause is information loss. Compressing an FP32 number (roughly $4.2 \times 10^9$ possible values) into INT8 (256 values) or INT4 (16 values) maps many original values into the same quantization bin, producing irreversible rounding error. Concretely:

- **Accumulated rounding error**: the rounding error of any single weight may be small, but matrix multiplication propagates and amplifies errors through the computation graph. The deeper the network and the larger the model, the stronger this cumulative effect.
- **Outlier problem**: if a tensor contains a few extremely large values (outliers), the scale becomes large, forcing the many normal values into a very narrow quantization interval and sharply degrading precision. This is exactly the core challenge in LLM quantization: emergent outliers in Transformer activations cause naive quantization to fail outright.
- **Dynamic-range mismatch**: low-bit integers have limited representable range; values outside that range are clipped by `clamp`, causing permanent information loss.

The central goal of quantization research is therefore to keep these accuracy losses within an acceptable range at the lowest possible bit width, using various techniques such as better scale selection, error compensation, and outlier handling.

### 1.3 Quantization Granularity

| Granularity | Meaning | Typical Use Case |
|---|---|---|
| Per-tensor | One shared $(s, z)$ for the entire tensor | INT8 inference |
| Per-channel | One $(s, z)$ per output channel for weights | CNN / Transformer weights |
| Per-group | One group per $g$ consecutive elements ($g$ is often 128) | 4-bit weights in GPTQ / AWQ |
| Per-token | One group per row of the activation matrix | SmoothQuant W8A8 |
| Per-block | One group per fixed-size tile (e.g., 32) | FP8 microscaling / MXFP8 |

Finer granularity gives higher precision, but also increases the storage and compute overhead of scale/zero-point metadata; whether hardware can access that metadata efficiently is critical.

### 1.4 Straight-Through Estimator (STE)

**Why do we need STE?** The core quantization operation is rounding $\lfloor \cdot \rceil$, a staircase function: the output jumps at integer points and is completely flat elsewhere. Mathematically, its gradient is almost everywhere zero. This means that if we introduce quantization during training (i.e., QAT), gradients stop at the quantization node during backpropagation, so weights cannot be updated by gradient descent and training fails completely. STE is the standard way around this contradiction: quantize honestly in the forward pass, but in the backward pass pretend the quantization operator does not exist and let the gradient pass through directly.

In QAT, STE is used to bypass the issue:

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \hat{x}} \cdot \mathbf{1}\{x \in [\text{lo},\, \text{hi}]\}$$

Within the clipping range, the gradient passes through unchanged; outside the range, it is zero. **All QAT methods** (PACT, LSQ, BitNet, ...) are built on STE.

### 1.5 Mixed-Precision Training

> Micikevicius et al., 2018 · ICLR

**Why can’t we train directly in pure FP16?** Naively replacing all FP32 tensors with FP16 runs into two fatal problems: (1) **gradient underflow** — the smallest positive subnormal in FP16 is about $6 \times 10^{-8}$, and many layer gradients have smaller magnitude, so they become zero and weight updates stop; (2) **vanishing weight updates** — even if the gradient does not underflow, when `learning_rate × gradient` is much smaller than the weight itself, FP16’s limited precision (10-bit mantissa) can cause the `weight + update` addition to round the update away entirely. Mixed-precision training is designed precisely to solve these two problems.

**Core idea**: "**low-precision computation + high-precision accumulation**" — use FP16/BF16 for forward and backward GEMMs to get 2× memory compression and compute acceleration, while retaining FP32 master weights to preserve training stability.

**Three key tricks**:

1. **FP16 storage and computation**: weights / activations / gradients are stored in FP16, and forward/backward GEMMs run on FP16 Tensor Cores (A100 FP16 throughput is 312 TFLOPS vs 19.5 TFLOPS for FP32, a 16× gap). Memory use is cut by 2×.
2. **FP32 master weights**: the optimizer maintains an FP32 copy of the weights. In each training step: FP32 master weights → cast to FP16 for forward/backward → FP16 gradients cast back to FP32 → optimizer update is performed in FP32. This ensures small weight updates are not rounded away.
3. **Loss scaling**: multiply the loss by a constant $S$ (e.g., 1024), equivalently shifting all gradients left by 10 bits, pulling small gradients that would otherwise underflow into the FP16 representable range. Before the optimizer update, divide by $S$ to restore the original scale. In practice, **dynamic loss scaling** is commonly used: start from a large $S$, reduce it if inf/NaN gradients are detected, otherwise gradually increase it.

```
Data flow of one mixed-precision training step:

FP32 master weights ──cast──→ FP16 weights
                                    │
                              Forward pass (FP16 Tensor Core GEMM)
                                    │
                                    ▼
                              loss × S  (loss scaling)
                                    │
                              Backward pass (FP16)
                                    │
                                    ▼
                          FP16 gradients / S ──cast──→ FP32 gradients
                                                          │
                                                   Optimizer update
                                                   (FP32 accumulation)
                                                          │
                                                          ▼
                                                FP32 master weights (after update)
```

**Memory analysis**: take a 1B-parameter model as an example (4 bytes per parameter in FP32):

| Component | Pure FP32 | Mixed Precision |
|---|---|---|
| Model weights | 4 GB (FP32) | 2 GB (FP16) + 4 GB (FP32 master) = 6 GB |
| Activations | 4× depending on batch | 2× depending on batch (FP16, halved) |
| Optimizer (Adam) | 8 GB (m + v, FP32) | 8 GB (m + v, still FP32) |

At first glance, mixed precision seems to add an extra FP16 copy of the weights. But activations dominate — activation memory grows linearly with batch size and sequence length, and the memory saved by FP16 activations far exceeds the cost of storing an extra FP16 weight copy. Moreover, the 2× compute speedup is the main benefit.

**BF16 vs FP16**:

| | FP16 | BF16 |
|---|---|---|
| Exponent bits | 5 bit (max 65504) | 8 bit (max $3.4 \times 10^{38}$, same as FP32) |
| Mantissa bits | 10 bit | 7 bit |
| Need loss scaling? | Yes | Almost never (dynamic range is large enough; gradients rarely underflow) |
| Precision | Higher | Slightly lower (but in practice with little impact on training convergence) |
| Hardware support | All modern GPUs | A100+ (V100 does not support it) |

Because BF16 eliminates the complexity of loss scaling and is more stable in training, it has become the de facto standard for large-model training. In PyTorch, it can be enabled simply with `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.

### 1.6 Overview of Low-Precision Numeric Formats

Quantization and low-precision training involve a variety of floating-point and integer formats. The table below lists the structure and representable range of common formats to give an intuitive sense of the difference in "information capacity" across precisions:

| Format | Structure (sign + exponent + mantissa) | Max Value | Smallest Positive Subnormal | Precision | Typical Use |
|---|---|---|---|---|---|
| **FP32** | 1 + 8 + 23 | $3.4 \times 10^{38}$ | $\sim 1.4 \times 10^{-45}$ | ~7 decimal digits | Master weights, optimizer state |
| **FP16** | 1 + 5 + 10 | 65504 | $6.0 \times 10^{-8}$ | ~3 decimal digits | Mixed-precision training (needs loss scaling) |
| **BF16** | 1 + 8 + 7 | $3.4 \times 10^{38}$ | $\sim 9.2 \times 10^{-41}$ | ~2 decimal digits | Standard large-model training format (same dynamic range as FP32) |
| **FP8 E4M3** | 1 + 4 + 3 | 448 | $\sim 0.002$ | 3-bit mantissa | Forward: weights / activations |
| **FP8 E5M2** | 1 + 5 + 2 | 57344 | $\sim 6.1 \times 10^{-8}$ | 2-bit mantissa | Backward: gradients (needs larger dynamic range) |
| **FP4 E2M1** | 1 + 2 + 1 | 6.0 | 0.5 | 1-bit mantissa | Experimental, MXFP4 |
| **INT8** | 8-bit integer | 127 (signed) | 1 | Uniform spacing | PTQ inference (W8A8) |
| **INT4** | 4-bit integer | 7 (signed) | 1 | Only 16 levels | Weight compression (W4A16) |

**Key comparisons**:
- **FP16 vs BF16**: both are 16-bit, but FP16 has higher precision (10-bit mantissa) and a narrow dynamic range (max 65504), making large values prone to overflow and requiring loss scaling. BF16 has the same dynamic range as FP32, is more stable in training, and is therefore the current default for large models.
- **E4M3 vs E5M2**: both are FP8. E4M3 has higher precision but smaller range, making it suitable for weights/activations; E5M2 has larger range but lower precision, making it suitable for gradients and other tensors that require a wide dynamic range.
- **Floating point vs integer**: floating-point formats have non-uniform quantization intervals (denser near small values), while integer quantization uses uniform spacing. For approximately normal-distributed weights, non-uniform quantization (e.g., the NF4 codebook in §9.2) is theoretically better.

---

## 2 Overview of Quantization Methods: QAT vs PTQ

**Why distinguish QAT from PTQ?** The quantization formula in §1.1 looks simple — compute a scale, round, and you’re done. But the key question is: **how do we choose the optimal scale?** If we set the scale using $\max(|x|)$, rare extreme values enlarge the scale and waste most quantization levels; if we clip those extremes to tighten the scale, we may truncate important information. More fundamentally, quantization changes the output distribution of every layer, and this error propagates and accumulates through the network — optimizing the scale of each layer independently does not guarantee global optimality.

There are two ways to address this contradiction:

- **QAT**: since quantization error propagates, let the model *see* that error during training, and let end-to-end gradient descent automatically adjust the weight distribution to adapt to quantization. The cost is a full training pipeline.
- **PTQ**: the model has already been trained, so we leave the weights themselves unchanged and instead use calibration data to analyze each layer’s distribution and choose optimal quantization parameters (scale, rounding direction, etc.) to minimize output error. The cost is that there is no training-time error correction, so the accuracy ceiling is lower than QAT.

Model quantization is divided into these two paradigms according to **whether training is required**. This section provides an overview; specific LLM quantization methods are introduced progressively in §3-§6.

### 2.1 QAT (Quantization-Aware Training) — Quantization During Training

> Classic QAT methods (PACT, LSQ, etc.) emerged in the CNN era (2018-2019), targeting vision models such as ResNet and MobileNet. In the LLM setting, the representative QAT line is BitNet (§7), which follows the route of "native low-bit training from scratch."

**Core idea**: insert **fake quantization nodes** during training — quantize → dequantize weights and activations in the forward pass (the data type remains FP32, but the values now include quantization error), allowing the model to learn quantization-friendly distributions through gradient descent. In the backward pass, use STE (§1.4) so gradients pass through the quantization nodes and update FP32 master weights normally. After training, export the true low-bit weights.

```
┌─────────────────────────── Forward pass ───────────────────────────┐
│                                                                │
│  x (FP32)                                                      │
│    │                                                           │
│    ▼                                                           │
│  ┌──────────────────┐                                          │
│  │  Fake quantization (Fake Q) │  quantize → dequantize         │
│  │  x → x_q → x̂    │  Data type remains FP32, but values now contain quantization error │
│  └────────┬─────────┘                                          │
│           ▼                                                    │
│  ┌──────────────────┐     w (FP32 master weight)               │
│  │  Linear(x̂, ŵ)   │ ◄── ŵ = dequant(quant(w))  same fake quantization │
│  └────────┬─────────┘                                          │
│           ▼                                                    │
│         loss                                                   │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────────── Backward pass ───────────────────────────┐
│                                                                │
│  ∂L/∂loss                                                      │
│    │                                                           │
│    ▼                                                           │
│  ∂L/∂ŵ ──→ encountering a fake-quant node ──→ STE passthrough (§1.4) │
│    │                                                           │
│    ▼                                                           │
│  Update master weights normally with FP32 gradients               │
└────────────────────────────────────────────────────────────────┘
```

**Key learnable parameters**: the core of QAT is turning the hyperparameters in the quantization formula into learnable quantities:

| Learnable Target | Idea | Representative Work |
|---|---|---|
| Clipping range $\alpha$ | Learn the optimal clipping range per layer to avoid either truncating information or wasting bits | PACT (2018) |
| Quantization step size $s$ | Treat step size as a learnable parameter + use gradient scaling for stability | LSQ (2019) |
| Both together | Learn clipping range and step size jointly | LSQ+ |

### 2.2 PTQ (Post-Training Quantization) — Quantization After Training

> **The mainstream route for LLM quantization**. After the model is trained, weights are no longer updated; instead, a small amount of calibration data (a few hundred unlabeled samples) is used to determine quantization parameters. LLM quantization methods such as GPTQ, AWQ, and SmoothQuant all belong to PTQ.

**Basic workflow**: load a pretrained model → run forward passes on calibration data to collect statistics → determine scale/zero-point for each layer → quantize weights (and activations for some methods) → evaluate accuracy.

**Three technical levels of PTQ**:

| Level | Idea | Accuracy |
|---|---|---|
| **Round-to-Nearest (RTN)** | Direct nearest-integer rounding, with no optimization | Usable at 8-bit; usually collapses at 4-bit |
| **Layer/block reconstruction** | Minimize output reconstruction error $\min \|\|Wx - \hat{W}x\|\|_F^2$, with optimizable rounding direction | Usable at 4-bit |
| **Second-order error compensation** | Use Hessian information to quantize column by column and compensate residuals into unquantized columns | Usable at 4-bit and even 3-bit |

**Core idea of reconstruction-based optimization**: quantization does not have to be round-to-nearest — for some weights, rounding up gives smaller output error than rounding down. Treat rounding direction as an optimizable variable and minimize $\min_{\hat{W}} \|Wx - \hat{W}x\|_F^2$ layer by layer (where $x$ comes from calibration data). In practice, block-wise reconstruction offers the best cost-performance trade-off. CNN-era works such as AdaRound, BRECQ, and QDrop established this methodology, which was directly inherited by LLM methods such as GPTQ and AWQ.

### 2.3 QAT vs PTQ Comparison

| | QAT | PTQ |
|---|---|---|
| Need training data? | Yes (full training set) | No (a few hundred unlabeled calibration samples) |
| Compute cost | Full training cycle | Minutes to hours |
| Best suited for | Extremely low bit (2-3 bit), training from scratch | Deployment compression, 4-bit+, **LLM mainstream** |
| Accuracy | Higher (especially at low bit) | Slightly lower but usually sufficient |
| LLM representative | BitNet (training from scratch) | GPTQ / AWQ / SmoothQuant |

For LLMs, full-parameter QAT costs nearly as much as training from scratch, so in practice almost all work on "compressing an existing model" follows the PTQ route. QAT is used only in directions such as BitNet, where the model is natively trained in low bit from scratch.

---

## 3 The Core Challenge of LLM Quantization: Outliers

In §1-§2 we built the basic toolbox for quantization. Now we move into LLM quantization in practice — but before examining specific methods, we must first understand **why LLM quantization is much harder than CNN quantization**.

**The Emergent Outlier phenomenon** (Dettmers et al., 2022): once Transformer parameter counts exceed roughly 6.7B, activation tensors begin to exhibit **a small number of fixed channels whose values are more than 100× larger than the rest**. These outliers are *emergent* — they do not exist in small models, only in large ones — and they concentrate in specific hidden dimensions while remaining stable across different input tokens.

**Why do outliers emerge?** The question is not fully resolved, but several influential explanations have been proposed:

- **Interaction between LayerNorm and residual connections**: Bondarenko et al. (2023, "Quantizable Transformers") propose that outliers are essentially **"no-op" signals** for attention heads. When an attention head has no meaningful information to extract for the current token, it still needs a way to "do nothing" — but softmax forces attention weights to normalize and cannot output all zeros. The learned strategy is to concentrate attention on large values in certain fixed dimensions, whose contribution to the final output can still be controlled after subsequent processing. The presence of LayerNorm allows the model to safely use extremely large values in a few channels without destabilizing the rest.
- **Massive Activations**: Sun et al. (2024, "Massive Activations in Large Language Models") systematically study large-activation phenomena and find that they occur at **fixed positions** (e.g., tokens near the beginning of the sequence) and in **fixed dimensions**, while also serving functional roles — removing them severely hurts model performance. This suggests that outliers are not a training "bug" but rather a learned **information encoding mechanism**.
- **Scale-emergence hypothesis**: similar to other emergent abilities in large models (in-context learning, chain-of-thought, etc.), one view is that outliers are a phase transition that only appears once the model becomes large enough. Small models lack the representational capacity to "afford" an encoding strategy that concentrates information in a few channels, while large models can exploit redundant dimensions for more efficient information routing.
![Pasted image 20260306201502.png](./assets/Pasted_image_20260306201502.png)
> Here we reproduced this phenomenon on two Qwen2.5 scales: 0.5B and 7B.

**Why do outliers make naive quantization collapse?** Recall the quantization formula in §1.1: the scale $s$ is determined by the maximum absolute value in the tensor. Suppose most values in an activation vector lie in $[-1, 1]$, but one channel takes value 100:

- $s = 100 / 127 \approx 0.79$ (INT8 symmetric quantization)
- A normal value $0.5$ is quantized to $\lfloor 0.5 / 0.79 \rceil = 1$, then dequantized back to $0.79$, giving 58% error
- Without the outlier: $s = 1 / 127 \approx 0.008$, $0.5$ is quantized to $63$, dequantized back to $0.50$, with error < 1%

**A single outlier enlarges the scale and destroys the quantization precision of the remaining 99.9% of normal values.**

This is the central contradiction in LLM quantization: outliers cannot simply be discarded (they are crucial to model outputs), but their presence makes it difficult to quantize the whole tensor with a single scale. All methods in §4-§6 are, in essence, different strategies for resolving this contradiction.

```
Progressive roadmap for LLM quantization:

§4  First solve whether quantization is feasible ──→  W8A8: 8-bit inference
    Strategy: handle outliers      LLM.int8() (separate outliers at runtime)
                            SmoothQuant (remove outliers by preprocessing)
                                │
                                ▼
§5  Then pursue higher compression ──→  W4A16: 4-bit weight quantization
    Strategy: smarter rounding     GPTQ (second-order error compensation)
                            AWQ (activation-aware scaling)
                                │
                                ▼
§6  Final goal: end-to-end low-bit ──→  W4A4 + KV4
    Strategy: eliminate outliers at the source  QuaRot (orthogonal rotation disperses outliers)
                              SpinQuant (learn the optimal rotation)
                              KIVI (specialized for KV cache)
```

---

## 4 W8A8: 8-bit Inference Quantization

> The first goal: make LLMs run GEMM in INT8 to obtain 2× bandwidth savings plus compute acceleration. The central challenge is how to handle outliers in activations.

### 4.1 LLM.int8() — Runtime Outlier Separation

> Dettmers et al., 2022 · NeurIPS

**Idea**: since outliers occupy only a few channels, separate them at runtime and process them in FP16, while the rest uses INT8.

```
Input X ∈ ℝ^{n×d}, weights W ∈ ℝ^{d×m}

1. Detect outlier dimensions  O = { j : max_i |X_ij| > τ },  τ = 6.0
2. Separate:
     X_out = X[:, O],  W_out = W[O, :]   →  FP16 GEMM
     X_reg = X[:, Ō],  W_reg = W[Ō, :]   →  INT8 absmax GEMM
3. Y = Y_out + Y_reg
```

- Outlier dimensions typically account for only 0.1%–1%, so the vast majority of computation runs in INT8.
- No calibration and no training. The PPL increase is < 0.1 even for a 175B model.
- Limitation: outliers are detected at runtime, so the kernel must support split-and-merge execution; the FP16 path limits overall bandwidth gains.

### 4.2 SmoothQuant — Eliminating Outliers by Preprocessing

> Xiao et al., 2022 · ICML 2023

**The problem with LLM.int8()**: although its accuracy is almost lossless, the runtime separation mechanism introduces major engineering and performance costs:

1. **Two GEMMs per layer**: every linear layer requires one INT8 GEMM + one FP16 GEMM + result fusion, and kernel launch / synchronization overhead becomes significant in small-batch inference. In practice, LLM.int8() is often **slower than pure FP16**, especially for single-sample inference, because the saved computation is offset by extra kernel overhead.
2. **Unfriendly dynamic branching**: the number and locations of outlier dimensions must be detected layer by layer at runtime. This kind of data-dependent dynamic branching is unfriendly to GPU parallel execution and compiler optimization, making it hard to optimize with frameworks such as `torch.compile`.
3. **Cannot use standard INT8 kernels**: standard INT8 GEMM kernels (e.g., cuBLAS INT8) require uniform precision across the entire matrix. The mixed-precision decomposition in LLM.int8() requires custom kernels, limiting portability across hardware platforms.

**The idea of SmoothQuant**: can we **preprocess away outliers before deployment**, so that inference runs entirely in INT8 and standard kernels can provide acceleration directly?

**Core insight**: weights are smooth (easy to quantize), while activations contain outliers (hard to quantize). Use an **equivalent mathematical transformation** to move the difficulty from activations to weights.

For a linear layer $Y = XW$, introduce per-channel scaling $\mathbf{s} \in \mathbb{R}^d$:

$$Y = \underbrace{(X\,\text{diag}(\mathbf{s})^{-1})}_{\hat{X}} \;\cdot\; \underbrace{(\text{diag}(\mathbf{s})\,W)}_{\hat{W}}$$

Choose $\mathbf{s}$ to balance the quantization difficulty of $\hat{X}$ and $\hat{W}$:

$$s_j = \frac{\big(\max_i |X_{ij}|\big)^{\alpha}}{\big(\max_k |W_{jk}|\big)^{1-\alpha}}, \qquad \alpha \in [0.5, 0.75]$$

**Why does this formula suppress outliers?** Consider a concrete example. Suppose the $j$-th channel of activation $X$ is an outlier channel with $\max |X_{:,j}| = 100$, while the corresponding weight column has $\max |W_{j,:}| = 0.5$. Take $\alpha = 0.5$:

$$s_j = \frac{100^{0.5}}{0.5^{0.5}} = \frac{10}{0.707} \approx 14.1$$

After the transformation:
- Activation channel $j$: $\hat{X}_{:,j} = X_{:,j} / s_j = X_{:,j} / 14.1$, so the original 100 becomes ~7.1, meaning **the outlier is compressed**
- Weight row $j$: $\hat{W}_{j,:} = s_j \cdot W_{j,:} = 14.1 \cdot W_{j,:}$, so the original 0.5 becomes ~7.1, meaning **the weights are enlarged**

The key is that **the outlier activation channels correspond precisely to weight rows with relatively small values** (a commonly observed phenomenon in LLMs — the model encodes information by multiplying large activations by small weights). So when $s_j$ is large, dividing activations by a large number flattens them, while multiplying the weights by the same number enlarges them — but since the weights were small to begin with, they still stay within a reasonable range after scaling. After the transformation, the numeric ranges of activations and weights become much closer, so both sides become "easy to quantize."

Mathematically, the formula for $s_j$ performs a **geometric balancing**: $\alpha$ controls how much quantization difficulty is shifted from activations to weights. When $\alpha = 1$, scaling is determined entirely by activations (activations become fully smoothed, but weights may blow up); when $\alpha = 0$, activations are untouched. In practice, $\alpha \in [0.5, 0.75]$ gives the best trade-off — because weight distributions are much smoother than activation distributions and can "absorb" more of the difficulty transfer.

**Deployment**: fuse $s^{-1}$ into the previous LayerNorm/bias and absorb $s$ into $W$. Inference then incurs zero extra computation and runs directly as **W8A8 INT8 GEMM**.

- Difference from LLM.int8(): SmoothQuant **eliminates** outliers through preprocessing rather than separating them at runtime.
- Limitation: it mainly targets INT8; when pushing to 4-bit, the smoothing is no longer sufficient.

**Detailed calibration process**: SmoothQuant uses calibration data to determine the scaling factor $s_j$ for each channel. The workflow is:

1. **Prepare a calibration set**: randomly sample a few hundred unlabeled text sequences (typically 128–512) from the training set or public corpora such as Pile or C4, and truncate them to a fixed length (e.g., 2048 tokens). These data do not need labels; they only need to represent the input distribution seen in practice.
2. **Collect statistics with forward passes**: run the calibration data through the model using forward inference only (no backpropagation), and record the **per-channel maximum absolute value** $\max_i |X_{ij}|$ of the activation tensor at the input of each linear layer. Because the locations of outlier channels are highly consistent across inputs (the emergent outlier property discussed in §3), the statistics become very stable after only a few hundred samples.
3. **Compute scaling factors**: combine the activation statistics with the per-channel maxima of the weights themselves, and compute $s_j$ for each channel in each layer using the formula $s_j = (\max_i |X_{ij}|)^\alpha / (\max_k |W_{jk}|)^{1-\alpha}$. The only hyperparameter is $\alpha$, which is usually tuned by grid search over $[0.5, 0.75]$ using quantized output error on the calibration set as the criterion.
4. **Absorb scaling into weights**: multiply $\text{diag}(\mathbf{s})$ into the weight matrix $W$, and fuse $\text{diag}(\mathbf{s})^{-1}$ into the LayerNorm parameters of the previous layer ($\gamma \leftarrow \gamma / s$). This step is done offline, and the modified model is saved as a new weight file.
5. **Determine quantization parameters**: compute the INT8 scale/zero-point for the smoothed weights and activations separately (per-channel for weights, per-token for activations), and complete quantization.

The entire process requires **forward passes only**, with no weight updates, and usually takes just a few minutes to a dozen minutes. On OPT-175B it is nearly lossless.

This calibration paradigm is also inherited by methods such as GPTQ and AWQ — they likewise use a small amount of calibration data and forward passes to collect statistics (GPTQ collects the Hessian $H = 2X^TX$, while AWQ collects activation magnitudes $\text{mean}_i |X_{ij}|$); the difference lies only in what optimization they perform with those statistics.

### 4.3 W8A8 Summary: A Problem That Is Essentially Solved

W8A8 can now be regarded as **basically solved** in current LLM quantization. On mainstream models (OPT, LLaMA, Mistral, etc.), SmoothQuant usually incurs a PPL increase of < 0.1, and the accuracy drop on downstream tasks is negligible. Remaining edge issues include:

- **Tuning $\alpha$**: the optimal $\alpha$ may differ across models and layers. The original SmoothQuant uses a global $\alpha$, and later work (e.g., OS+) tries layer-wise adaptive $\alpha$, but with limited gains, suggesting that a global $\alpha$ is already good enough.
- **Adapting to special architectures**: in newer architectures such as GQA (Grouped Query Attention), KV projections have different channel counts, so the propagation path of smoothing factors needs to be adapted. But this is an engineering issue, not a methodological bottleneck.
- **Replacement by FP8**: H100+ hardware natively supports FP8 Tensor Cores (§8.1). In many scenarios, FP8 inference is more convenient than INT8 (no smoothing transform required — just cast directly), and its accuracy is sufficient. FP8 is gradually replacing INT8 as the default 8-bit inference choice.

**Conclusion**: the accuracy problem of W8A8 has been effectively solved by SmoothQuant, and the research frontier has fully shifted toward more aggressive 4-bit and even lower-bit quantization. What remains for W8A8 is mainly engineering deployment and adaptation to new hardware.

---

## 5 W4A16: 4-bit Weight Quantization

> W8A8 solves the question of "can we quantize at all?" The natural next step is: **can we compress even harder?** 4-bit weights cut model size in half again. Here only weights are quantized (weight-only), while activations remain FP16, and weights are dequantized back to FP16 for GEMM during inference. The main challenge shifts from "outliers" to "how to minimize weight quantization error when only 16 quantization levels are available."

**W8A8 vs W4A16: how should we choose?** The two have different design goals. There is no absolute winner; the choice depends on the deployment scenario:

| | W8A8 (SmoothQuant) | W4A16 (GPTQ/AWQ) |
|---|---|---|
| **Accuracy** | Nearly lossless (PPL increase < 0.1) | Some degradation (PPL increase 0.3-0.5) |
| **Model size** | ~50% of original (weights + activations both 8-bit) | ~25% of original (4-bit weights) |
| **GEMM compute** | INT8 × INT8 → real compute acceleration | INT4→dequant→FP16 × FP16 → compute remains FP16 |
| **Source of speedup** | 2× bandwidth + 2× compute | Mainly 4× bandwidth savings, compute unchanged |

**The key to speed comparison: it depends on batch size and the system bottleneck**:

- **Small-batch decode (memory-bound)**: the bottleneck is loading weights from HBM. W4A16 weights are half the size of W8A8 weights, so **they load 2× faster**. Although W4A16 GEMM still runs in FP16, compute is not the bottleneck at small batch sizes, so **W4A16 is faster**. This is the most common LLM inference setting (autoregressive decoding with small batch size).
- **Large-batch prefill (compute-bound)**: the bottleneck is GEMM computation. W8A8 INT8 × INT8 Tensor Cores deliver 2× the throughput of FP16, while W4A16 still performs FP16 compute, so **W8A8 is faster**.
- **Memory-constrained scenarios**: W4A16 has a smaller model footprint (4-bit vs 8-bit weights), so the same GPU can run larger models or longer contexts.

**Current usage trends**:
- **Open-source community / local deployment**: W4A16 (GPTQ/AWQ) dominates. Most users run medium-sized models on a single GPU, where the typical setting is small-batch, memory-bound decode. The 4× bandwidth savings of W4A16 translate directly into speedup, and the smaller model size enables deployment on consumer GPUs.
- **Production serving**: increasingly shifting to **FP8** (natively supported on H100+). Its accuracy is close to FP16, it requires no calibration or smoothing transform, and operationally it is the simplest choice. In large-batch serving, the compute-acceleration advantage of W8A8/FP8 is more pronounced.
- **Edge / mobile deployment**: W4A16 and even lower bit widths (e.g., `Q4_K_M` in `llama.cpp`) are used to maximize compression ratio.

**A simple decision rule**: if you care most about accuracy → W8A8/FP8; if you care most about minimum size → W4A16; if you have H100+ → FP8 is the easiest option.

### 5.1 GPTQ — Approximate Second-Order Weight Quantization

> Frantar et al., 2023 · ICLR

**Why does RTN collapse at 4-bit?** Round-to-Nearest, i.e., directly rounding each value to the nearest integer, works well at 8-bit but degrades sharply at 4-bit. The root cause is that INT4 has only 16 quantization levels, so the relative rounding error of an individual weight can be large (in the worst case, the error equals half a quantization step, and the step size is `range/15`). More importantly, each output element in matrix multiplication $Y = WX$ is a weighted sum of thousands of weights, and independent rounding errors do not cancel perfectly — they accumulate on the order of $\sqrt{d}$. RTN rounds each weight independently and completely ignores error correlation between weights.

**Core idea of GPTQ**: quantization should not be treated independently — after quantizing one weight, the resulting error can be compensated by **adjusting the weights that have not yet been quantized**. This idea comes from the OBS (Optimal Brain Surgeon) framework: the Hessian matrix $H$ describes how each weight affects the output, and the direction and magnitude of error compensation are given by $H^{-1}$.

**From OBQ to GPTQ**:

OBQ (Optimal Brain Quantizer, Frantar & Alistarh 2022) is the precursor to GPTQ and follows exactly the same idea — quantize one weight at a time + compensate the error using the Hessian. But OBQ must **greedily choose** which weight to quantize at every step (the one with the smallest error), resulting in $O(d^3)$ complexity; quantizing even a BERT model takes hours, making it impossible to scale to LLMs. GPTQ’s key contribution is the observation that **fixing the quantization order (left to right) incurs almost no accuracy loss**, while reducing complexity to $O(d^2)$ and enabling large-scale parallelization.

**Core algorithm**: for a weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, process each row independently. Within each row, quantize columns from left to right. When quantizing column $i$:

$$\hat{w}_i = Q(w_i)$$

$$\delta_i = w_i - \hat{w}_i$$

$$w_{j>i} \;\leftarrow\; w_{j>i} - \delta_i \cdot \frac{[H^{-1}]_{ij}}{[H^{-1}]_{ii}}$$

**Intuition**: $\delta_i$ is the quantization error of column $i$. The ratio $[H^{-1}]_{ij} / [H^{-1}]_{ii}$ describes "how much the error in column $i$ should optimally adjust column $j$" — essentially the optimal projection on the quadratic error surface defined by the Hessian. Here $H = 2X^TX$ is the Hessian corresponding to the row weights, where $X$ is the activation matrix from calibration data; it can be computed from a single forward pass.

**Why is error compensation effective?** Consider a simplified example with two weights $w_1 = 0.3, w_2 = 0.7$, corresponding activations $x_1, x_2$, and output $y = w_1 x_1 + w_2 x_2$. If after quantization $w_1$ becomes $\hat{w}_1 = 0$ (error 0.3), then under RTN, $w_2$ remains 1 (quantized to the nearest integer), and the output error is $0.3 x_1$. But if we compensate the error of $w_1$ into $w_2$ — enlarging $w_2$ appropriately based on the correlation between $x_1$ and $x_2$ — we can make $\hat{w}_1 x_1 + \hat{w}_2 x_2 \approx w_1 x_1 + w_2 x_2$. The Hessian matrix encodes exactly this information about "which weights can compensate for each other."

**Engineering optimizations**:

| Optimization | Effect |
|---|---|
| Fixed column order (0→d-1) | Avoids OBQ’s stepwise greedy sorting cost of $O(d^3)$, with almost no loss in measured accuracy |
| Cholesky factorization of $H^{-1}$ | Factorize $H^{-1}$ once with Cholesky; when quantizing each column, directly read the corresponding row instead of repeatedly inverting |
| Block quantization (128 columns per batch) | Quantize one group of 128 columns at a time: column-wise quantization + compensation within the group, one-shot residual update across groups. Lowers peak memory and improves GPU parallelism |
| Group quantization ($g=128$) | Each 128-column group shares one scale/zero-point, much more accurate than per-tensor with only ~0.5 bit/weight extra storage |
| Damping term $H \leftarrow H + \lambda I$ | Prevents ill-conditioned $H$ (e.g., tiny diagonal entries causing compensation blow-up), with $\lambda$ typically set to $0.01 \cdot \text{mean}(\text{diag}(H))$ |

**Numbers**: with 128 calibration samples, a single GPU can quantize a 175B model in a few hours. At 4-bit group quantization ($g=128$), the PPL increase is < 0.5 on LLaMA-65B. 3-bit also works, but with more noticeable degradation.

**Limitations**:
- **Weight-only**: only weights are quantized, activations remain FP16, and the actual GEMM is mixed INT4×FP16. End-to-end speedup depends on specialized kernels such as MARLIN (§9.1).
- **Large-batch bottleneck**: small-batch inference is memory-bound, so 4-bit weight compression directly speeds things up. But large-batch inference is compute-bound, and the cost of dequantizing back to FP16 becomes non-negligible.
- **Sensitivity to calibration data**: the quality of the Hessian depends on how representative the calibration data are. If the calibration distribution differs significantly from the real deployment distribution, quantization quality degrades.

### 5.2 AWQ — Activation-Aware Weight Quantization

> Lin et al., 2024 · MLSys (Best Paper)

**Progression of the idea**: GPTQ uses second-order Hessian information for error compensation, which works well but is computationally heavy (matrix inversion, column-wise compensation). AWQ takes a different angle — rather than optimizing the rounding rule, it first **protects important weights** before quantizing them.
**Core insight**: not all weights are equally important. Consider $Y = XW$, where the output error is $\Delta Y = X \cdot \Delta W$. The effect of the quantization error $\Delta w_j$ in column $j$ on the output is proportional to the magnitude of the corresponding activation $X_{:,j}$. If activation values in channel $j$ are generally large (a salient channel), then even a small $\Delta w_j$ can produce a large $X_{:,j} \cdot \Delta w_j$ — the weights in those channels are therefore "important," and their quantization errors are **amplified by activation magnitude**. **A natural idea is: why not simply skip quantizing those important weights?** The most intuitive approach would be to find the top 1% salient channels and keep their weights in FP16 while quantizing the rest. But this breaks matrix-multiplication efficiency in hardware — mixed-precision columns require special handling, similar to the issue in LLM.int8().
**AWQ’s solution — equivalent scaling**: inspired by SmoothQuant, use a mathematical identity to protect important weights without literally skipping them.
**Algorithm**:
```
1. Use calibration data to collect activation magnitude statistics:  s_j = mean_i |X_{ij}|   (measures the importance of each channel)
2. For each channel, apply equivalent scaling according to importance:
      Ŵ_{:,j} = W_{:,j} · α_j       (enlarge important weights → smaller relative quantization error)
      X̂_{:,j} = X_{:,j} / α_j       (shrink corresponding activations → preserve equivalence)
   Mathematical identity: X̂Ŵ = XW
3. Grid-search the optimal α_j:
      objective: min_{α_j} ‖Q(Ŵ_{:,j}) · X̂_{:,j} − W_{:,j} · X_{:,j}‖
      search space: α_j ∈ [1, max_scale], searched independently per channel
```
**Why does enlarging the weight protect it?** The upper bound of absolute quantization error is about half a quantization step, $s/2$, where $s = \max|w| / (2^{b-1}-1)$. After multiplying the weight by $\alpha > 1$, $s$ also grows, but the weight value itself grows by a factor of $\alpha$, so the **relative error** $|\Delta w| / |w|$ decreases. For salient channels, activation magnitude amplifies absolute error, so reducing relative error is especially important.
**Connection to and difference from SmoothQuant**:

|        | SmoothQuant            | AWQ                          |
| ------ | ---------------------- | ---------------------------- |
| Goal     | Smooth activation outliers to make W8A8 feasible      | Protect important weights to improve W4A16 accuracy           |
| Transform     | Shrink outlier activation channels, enlarge corresponding weights        | Enlarge important weights, shrink corresponding activations                |
| Source of scaling factors | Geometric balancing of activation max and weight max   | Activation mean (importance) + grid search |
| Where it is absorbed   | $s^{-1}$ fused into LayerNorm | $\alpha^{-1}$ fused into the previous layer         |

At a fundamental level, both methods use **per-channel equivalent scaling** to reshape where quantization difficulty lies, but their optimization goals differ: SmoothQuant seeks to make "activations and weights equally easy to quantize," while AWQ seeks to "minimize the quantization error of important weights."

**Numbers**:
- Calibration takes only minutes, with no backpropagation, and is much faster than GPTQ.
- With 4-bit group quantization ($g=128$), AWQ **outperforms GPTQ** on the LLaMA family (lower PPL).
- Paired with the TinyChat runtime, it enables end-to-end 4-bit inference with 3×+ speedup on edge devices such as Jetson Orin.

**Limitations**:
- **Weight-only**: same as GPTQ, activations remain FP16 and the method does not accelerate the compute itself.
- **Hardware-dependent optimal settings**: the best choices of $\alpha$ / group size depend on hardware memory bandwidth and compute capability.
- **Complementary to GPTQ**: AWQ protects important channels, while GPTQ performs error compensation. In principle they can be combined (apply AWQ scaling first, then GPTQ compensation), and some open-source implementations already support this.

### 5.3 MR-GPTQ — Quantization Specialized for FP4 Microscaling Formats

> Egiazarian, Panferov et al., 2026 · ICLR 2026

**Why do we need FP4-specific quantization?** The next-generation GPUs from NVIDIA Blackwell and AMD natively support two microscaling floating-point formats, MXFP4 and NVFP4, with theoretically doubled throughput. But directly applying INT4-era quantization methods (GPTQ, QuaRot, etc.) to FP4 works poorly — MXFP4’s E8M0 power-of-two scales are too coarse, causing around 10% accuracy loss, while NVFP4’s very small group size (16) makes traditional outlier-mitigation techniques ineffective. MR-GPTQ is the first quantization algorithm specifically tailored to the properties of FP4 microscaling formats.

**Two microscaling formats: MXFP4 vs NVFP4**:

| | MXFP4 | NVFP4 |
|---|---|---|
| Element format | FP4 E2M1 | FP4 E2M1 |
| Block size | 32 | 16 |
| Scale format | E8M0 (pure exponent, no mantissa) | E4M3 (standard FP8) |
| Extra scale | None | Per-tensor FP32 |
| Average bits per element | 4.25 | 4.5 |
| Advantage | More space-efficient, simpler hardware multiplication | Higher scale precision, better outlier preservation |
| Disadvantage | Large error from power-of-two scale | Slightly more storage |

**Core finding — rotations behave differently for FP4 than for INT4**:

MR-GPTQ first analyzes theoretically how Hadamard rotation affects the two FP4 formats:

- **MXFP4**: rotation is beneficial. The bottleneck of E8M0 scales is the coarseness of power-of-two quantization; once rotation makes the weight distribution more uniform, the impact of coarse scales is reduced.
- **NVFP4**: rotation **can be harmful**. NVFP4’s small group size (16) naturally preserves outliers well (the top element is effectively "promoted" to E4M3 precision). Rotation destroys this advantage by spreading the error of the top element across the whole group.
- **Key conclusion**: there is a "crossover point" — when group size is small, the MSE of the Laplace (original) distribution is lower than that of the Normal (post-rotation) distribution; when group size is large, the opposite holds.

**Three core improvements in MR-GPTQ**:

1. **Block-wise Hadamard rotation**: instead of global Hadamard, use a block-diagonal Hadamard matrix whose size matches the quantization group size (e.g., $H_{32}$), and rotate within each group. On the weight side, this can be fused into the weights offline; on the activation side, it is computed online. Because block size ≤ 128, the transform is memory-bound, so any rotation matrix — not just Hadamard — has the same runtime cost.

2. **MSE-optimized scale search**: for NVFP4’s two-level scaling structure (per-group + per-tensor), use alternating optimization to search for the scale combination with minimum MSE rather than simply taking absmax. For MXFP4, because the distribution is uniform after rotation, a fixed static scale is sufficient.

3. **Static activation reordering**: the original GPTQ `act-order` trick (sorting columns by Hessian diagonal) improves accuracy, but requires runtime column shuffling, introducing 10-20% inference overhead. MR-GPTQ instead fixes scales and the grid first, shuffles columns for GPTQ compensation, and then shuffles them back — obtaining the same accuracy gain with **zero runtime overhead**.

**GPU kernel support (QuTLASS)**: MR-GPTQ also releases QuTLASS, a kernel library optimized for the Blackwell architecture:
- A lightweight fused kernel performs online rotation + quantization + scale computation with negligible latency
- Supports two compute capabilities: SM100 (B200) and SM120 (RTX5090)
- MXFP4 kernel throughput even **exceeds the idealized NVFP4** matrix multiplication throughput

**Experimental results** (Llama-3.1-8B-Instruct, W4A4):

| Method | Format | Avg. Recovery |
|---|---|---|
| RTN | NVFP4 | 94.7% |
| SmoothQuant | NVFP4 | 96.5% |
| GPTQ | NVFP4 | **97.2%** |
| MR-GPTQ | NVFP4 | **97.0%** |
| RTN | MXFP4 | 88.4% |
| QuaRot | MXFP4 | 91.3% |
| MR-GPTQ | MXFP4 | **95.2%** |

- On MXFP4, MR-GPTQ raises accuracy recovery from 88% to 95%, approaching NVFP4-level quality
- End-to-end inference speedup: **2.2×** on B200 and **4×** on RTX5090 (vs FP16 baseline)
- Layer-wise speedup: **3.6×** on B200 and **6×** on RTX5090

**Relation to GPTQ/AWQ**: MR-GPTQ is the natural extension of GPTQ to FP4 microscaling formats. Traditional GPTQ is designed for uniform INT formats, whereas MR-GPTQ adapts to FP4’s non-uniform quantization properties through block-wise rotations and format-aware optimization.

---

## 6 W4A4 + KV4: End-to-End Low-Bit Quantization

> W4A16 quantizes only the weights, while activations remain FP16 — so GEMM is still mixed INT4×FP16 computation, and the speedup is limited. The ultimate goal is to make **weights, activations, and KV cache all 4-bit or lower**, so GEMM can run entirely in low-bit integers. But as analyzed in §3, activation outliers are the biggest obstacle to end-to-end quantization. SmoothQuant-style smoothing is no longer sufficient at 4-bit — stronger tools are needed.

### 6.1 QuaRot — Orthogonal Rotation to Disperse Outliers

> Ashkboos et al., 2024

**Why is SmoothQuant no longer sufficient at 4-bit?** SmoothQuant uses per-channel scaling to *move* activation outliers into the weights, and works well at 8-bit. But 4-bit provides only 16 quantization levels, so even after smoothing the activation dynamic range is still too large. Scaling can change the magnitude of each channel, but it cannot change how energy is distributed across dimensions. A stronger transformation is needed.

**Core idea**: outliers are essentially **energy concentrated in a few dimensions**. If we apply an **orthogonal transformation to rotate the coordinate system**, distributing energy evenly across all dimensions, the outliers disappear — and because orthogonal transformations preserve vector norms and inner products, the model output remains unchanged.

**Mathematical principle**: for any orthogonal matrix $R$ satisfying $R R^T = I$,

$$Y = XW = (XR^T)(RW)$$

Let $\tilde{X} = XR^T$ and $\tilde{W} = RW$; then $Y = \tilde{X}\tilde{W}$, so the output is exactly unchanged. The key property is that **random orthogonal transforms** (such as Hadamard matrices) have a "democratizing" effect — they spread energy concentrated in a few dimensions uniformly over all dimensions. Intuitively, every row of a Hadamard matrix is an equal-amplitude combination of $\pm 1/\sqrt{d}$ terms, so after rotation each new dimension becomes an equal-weight mixture of all original dimensions, and any extreme value in a single original dimension is diluted.

**Why use a Hadamard matrix rather than an arbitrary orthogonal matrix?** A general orthogonal matrix multiplication costs $O(d^2)$, the same order as GEMM, making it too expensive as preprocessing. Hadamard matrices admit a fast transform (analogous to FFT) with only $O(d \log d)$ complexity, far cheaper than GEMM’s $O(d^2)$.

**Where to insert it in a Transformer**: we cannot simply rotate the whole model once and be done with it — nonlinear operations such as LayerNorm and softmax in Transformers break the ability to absorb rotations. QuaRot’s key engineering contribution is identifying which rotations can be absorbed offline into weights and which must be computed online:

```
Rotation insertion in a Transformer block:

1. Between linear layers: R can be absorbed into weights (offline)
   W_q, W_k, W_v ← R · W_q, R · W_k, R · W_v
   W_o ← W_o · Rᵀ
   → zero overhead at inference

2. After LayerNorm: RMSNorm(x) · W = RMSNorm(x) · Rᵀ · (R · W)
   RMSNorm is not commutative with rotation → need to compute xRᵀ online
   → use fast Hadamard transform, O(d log d)

3. Inside attention: Q and K must be rotated together to keep the inner product unchanged
   Q' = QRᵀ_head,  K' = KRᵀ_head  (per-head rotation)
   → Q'K'ᵀ = QRᵀR Kᵀ = QKᵀ  ✓
   → the KV cache is also rotated → outliers are dispersed → the KV cache can also be quantized to low bit
```

**Effect**: after rotation, the variance across activation channels is greatly reduced, and outliers that were previously concentrated in 0.1% of channels are dispersed across all dimensions. On top of this, simple RTN or GPTQ is enough to achieve end-to-end W4A4+KV4 quantization.

**Numbers**: on LLaMA-2-70B with W4A4 quantization, the PPL increase is around 0.5-1.0, far better than directly quantizing without rotation (which collapses).

**Limitations**:
- **Runtime cost of online Hadamard transforms**: although $O(d \log d)$ is far cheaper than GEMM, it requires custom CUDA kernels and is invasive to existing inference frameworks.
- **Sensitivity to random seeds**: the choice of random Hadamard matrix affects final accuracy; different seeds can vary by 0.2-0.3 PPL, so in practice multiple trials may be needed.
- **Attention kernel modifications required**: once the KV cache is rotated, the attention computation’s memory layout and kernels must be adapted accordingly.

### 6.2 SpinQuant — Learning the Optimal Rotation

> Liu et al., 2025 · ICLR

**Progression of the idea**: QuaRot uses a random Hadamard matrix, so its quality depends partly on luck in the random seed. The natural question is: can we **learn an optimal rotation matrix** instead of drawing one at random?

**Core challenge**: the rotation matrix $R$ must remain orthogonal ($RR^T = I$), making this a constrained optimization problem. If we update $R$ directly with gradient descent, it ceases to be orthogonal after the update, and the model output changes. SpinQuant solves this elegantly with a **Cayley parameterization**:

$$R = (I + A)^{-1}(I - A), \qquad A = -A^T \text{ (skew-symmetric matrix)}$$

When $A$ is skew-symmetric, the Cayley transform automatically produces an orthogonal matrix $R$. This means we only need to optimize $A$ without constraints (the upper-triangular entries of $A$ are free parameters), and gradient descent naturally guarantees that $R$ remains orthogonal at all times.

**Optimization objective**: note that SpinQuant **optimizes only the rotation matrix $R$, not the model weights themselves** — it is not end-to-end QAT, but a search for the best preprocessing rotation before PTQ. Given calibration data, it minimizes the output reconstruction error after "rotation + quantization":

$$\min_A \sum_{\text{layers}} \|\text{Quant}(\tilde{W}_l) \cdot \tilde{X}_l - W_l \cdot X_l\|_F^2$$

where $\tilde{W}_l = R_l W_l$ and $\tilde{X}_l = X_l R_l^T$ are the rotated weights and activations, and $\text{Quant}(\cdot)$ is simulated quantization. The only optimization variable is $A$ (which determines $R$); the model weights $W$ remain completely frozen. After optimization, we obtain the best rotation $R^*$ and then apply standard PTQ methods such as GPTQ, AWQ, or RTN to the rotated weights $R^* W$. One can also optimize end-to-end loss directly (more expensive but more accurate), but fundamentally only $R$ is being learned.

**Training cost**: tens to hundreds of gradient-descent steps (optimizing only the parameters of $R$ through $A$, with model weights frozen), much cheaper than QAT (no full retraining required) but more expensive than pure PTQ (because backpropagation is needed to compute $\partial \mathcal{L}/\partial A$). One can learn rotations only for a subset of sensitive layers and use random Hadamard elsewhere to reduce the cost further.

**Comparison with QuaRot**:

| | QuaRot | SpinQuant |
|---|---|---|
| Rotation matrix | Random Hadamard | Learned (Cayley parameterization) |
| Optimization cost | Zero (use as-is) | Tens to hundreds of gradient steps |
| Accuracy | Good, but high variance | Better, lower variance |
| W4A4 setting | Usable but somewhat unstable | Consistently better than QuaRot by 0.2-0.5 PPL |

**Limitations**:
- **Costs lie between PTQ and QAT**: backpropagation is needed to optimize $A$, making it slower than pure PTQ (e.g., GPTQ/AWQ), but still much faster than full-parameter QAT.
- **Rotation structure still needs kernel support**: as in QuaRot, online Hadamard transforms require custom kernels.
- **Must be combined with existing quantizers**: SpinQuant solves the problem of "removing outliers"; once they are removed, actual weight quantization still needs GPTQ/AWQ/RTN or similar methods.

### 6.3 KIVI — 2-bit Quantization for KV Cache

> Liu et al., 2024 · ICML

**The problem shifts**: we have addressed end-to-end quantization for weights and activations. But in long-context scenarios, there is another major memory consumer — the **KV cache**.

**Why is KV cache the bottleneck in long-context inference?** During LLM inference, the attention mechanism in every layer must cache the Key and Value vectors of all previous tokens. For LLaMA-2-7B, for example:

- Per-layer KV cache: $2 \times \text{seq\_len} \times d_{\text{head}} \times n_{\text{heads}} \times 2\text{B}$ (FP16)
- 32 layers × 128K context: $2 \times 128000 \times 128 \times 32 \times 32 \times 2 \approx 32\text{GB}$

This is far larger than the model weights themselves (~14GB in FP16), and **KV cache grows linearly with sequence length**. The longer the context, the larger the KV cache, until it becomes the dominant memory bottleneck limiting maximum batch size and context length.

**Why can’t we directly apply one unified quantization method to KV cache?** KIVI’s key contribution is to show that **the outlier structures of Key and Value are fundamentally different**, so they require different quantization strategies:

| Tensor | Outlier Pattern | Reason | Suitable Granularity |
|---|---|---|---|
| **Key** | Outliers concentrate in **fixed channels** (stable across tokens) | Certain output channels of the Key projection naturally produce large values regardless of the input token | **Per-channel** |
| **Value** | Outliers vary along the **token dimension** | Value magnitude depends on token semantic importance; e.g., BOS often has unusually large Values | **Per-token** |

If Key were quantized per-token, fixed-channel outliers would enlarge each token’s scale and waste precision in the other channels; if Value were quantized per-channel, abnormal tokens would enlarge that channel’s scale. KIVI’s approach is to **choose granularity according to outlier structure**.

**Method details**:

**Step 1: observe the different outlier patterns of Key and Value**

KIVI first systematically analyzes the numerical distributions of KV cache across multiple models (LLaMA-2-7B/13B, Mistral-7B, etc.):

- **Key matrix** $K \in \mathbb{R}^{T \times d}$: for each channel $j$, examine values $K_{:,j}$ across all tokens. KIVI finds that **a small number of fixed channels are consistently 10-100× larger than the rest**, and these "outlier channels" remain nearly unchanged across different input tokens. This is consistent with the emergent outlier phenomenon discussed in §3, and is fundamentally caused by certain output channels of the Key projection weight $W_K$ naturally producing large values.
- **Value matrix** $V \in \mathbb{R}^{T \times d}$: in contrast, Value outliers are **not fixed to certain channels, but to specific tokens**. Some tokens (e.g., BOS, punctuation, high-frequency function words) have Value vectors whose overall magnitude is far larger than that of ordinary tokens, while values within the same channel vary greatly across tokens.

This asymmetry means that the two cannot be handled with the same quantization granularity.

**Step 2: choose the matching quantization granularity**

```
Key cache K ∈ ℝ^{T×d} (outliers lie in the channel dimension and are stable across tokens):
  → Per-channel quantization: for each channel j, compute scale and zero-point
    s_j = (max(K_{:,j}) - min(K_{:,j})) / (2^b - 1)
    z_j = round(-min(K_{:,j}) / s_j)
  → A large value in a fixed channel is covered by that channel's own s_j and does not affect other channels

Value cache V ∈ ℝ^{T×d} (outliers lie in the token dimension and vary across channels):
  → Per-token quantization: for each token t, compute scale and zero-point
    s_t = (max(V_{t,:}) - min(V_{t,:})) / (2^b - 1)
    z_t = round(-min(V_{t,:}) / s_t)
  → A large value in an anomalous token is covered by that token's own s_t and does not affect other tokens

Both use asymmetric quantization (z ≠ 0), 2-bit precision, and adjustable group size (32/64/128)
```

**Step 3: the residual token mechanism for incremental updates**

In autoregressive inference, each generated step appends a new token, whose Key/Value vectors must be added to the cache. This introduces a problem: **the values of the new token may exceed the quantization range of the existing cache**, especially for per-channel quantization of Key. A new token may have a channel value larger than all historical tokens, requiring an update to that channel’s $s_j$, which would in turn require requantizing the entire Key cache — far too costly.

KIVI solves this with a **residual token buffer**:

```
KV cache structure:
┌─────────────────────────┐  ┌─────────────┐
│  Quantized cache (2-bit) │  │ Residual buffer │
│  token 1 ~ token T-R     │  │ (FP16, most recent R) |
│  Already quantized, no longer modified │  │  token T-R+1  │
│                          │  │  ...          │
│                          │  │  token T      │
└─────────────────────────┘  └─────────────┘

Whenever the residual buffer fills up (after accumulating R tokens) → batch-quantize these R tokens
→ append to the quantized cache → clear the buffer
```

- The most recent $R$ tokens remain in FP16 and are not quantized ($R$ is typically 128) — these are usually the tokens with the largest attention weights due to locality, so it is most important to keep them in high precision
- Only the "historical" tokens are quantized, and once quantized they are never modified again — avoiding repeated scale updates
- For Key, per-channel scales are computed based on the current buffer of $R$ tokens at each batch quantization step, rather than globally updated

**Step 4: attention computation after quantization**

After quantization, attention must be computed in two parts — the quantized portion and the FP16 residual portion:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q \cdot [K_{\text{quant}}; K_{\text{res}}]^T}{\sqrt{d}}\right) \cdot [V_{\text{quant}}; V_{\text{res}}]$$

Here $K_{\text{quant}}, V_{\text{quant}}$ are the 2-bit quantized portion, and $K_{\text{res}}, V_{\text{res}}$ are the FP16 residual buffer. In practice:

1. Compute $Q \cdot K_{\text{quant}}^T$: this requires dequantizing the 2-bit Key (multiply by scale + subtract zero-point), then multiplying by Q
2. Compute $Q \cdot K_{\text{res}}^T$: standard FP16 matrix multiplication
3. Concatenate and pass through softmax
4. Compute $\text{softmax\_weights} \cdot [V_{\text{quant}}; V_{\text{res}}]$: again split into dequant + multiply for the quantized part and standard FP16 for the residual part

This segmented computation requires custom attention kernel support; standard FlashAttention does not support mixed-precision KV cache.

**Why can it go down to 2-bit?** The key is that KV-cache quantization has much more tolerance than weight quantization:

1. **On the Key side**: the attention score is $QK^T / \sqrt{d}$, and after softmax its dynamic range is heavily compressed. Small Key quantization errors shift attention scores only slightly, but the winner-take-all nature of softmax leaves rankings almost unchanged — the truly important tokens still receive the largest attention weights.
2. **On the Value side**: $\text{softmax}(QK^T) \cdot V$ is a weighted average, and attention weights are themselves sparse (most tokens have near-zero weights), so only a few attended tokens materially affect the output. These important tokens are often in the residual buffer (the most recent tokens), where FP16 precision is preserved.
3. **Extra benefit of asymmetric quantization**: 2-bit symmetric quantization offers only four values such as {-1, 0, +1, 2}, whereas asymmetric quantization, with a shifted zero-point, can cover arbitrary intervals more flexibly. This is especially effective for skewed KV-cache distributions, such as Value vectors for certain tokens being entirely positive.

**Memory analysis** (LLaMA-2-7B, 4K context, per head):

| Precision | Per-layer KV cache | Total for 32 layers | 128K context |
|---|---|---|---|
| FP16 | $2 \times 4096 \times 128 \times 2\text{B} = 2\text{MB}$ | 64 MB | ~2 GB |
| KIVI 2-bit | $2 \times 4096 \times 128 \times 0.25\text{B} + \text{scales} \approx 0.28\text{MB}$ | ~9 MB | ~0.28 GB |
| Savings | **~7×** | | |

Even after accounting for the residual buffer’s fixed FP16 overhead (about 2MB/layer for $R=128$ tokens), the overall savings remain substantial. In practice, peak memory at 128K context drops from >32GB to ~12GB.

**Numbers**:
- Tuning-free, with no calibration data required — scale/zero-point are computed fully online
- On LLaMA-2-7B/13B and Mistral-7B, 2-bit KV cache typically increases PPL by < 0.2
- Peak memory drops by **2.6×+**, and maximum batch size increases substantially (especially in long-context scenarios)
- Orthogonal to weight quantization (GPTQ/AWQ): one can use 4-bit weights and 2-bit KV cache together for compounded compression

**Limitations**:
- **Attention-kernel adaptation**: 2-bit KV cache requires custom packing/unpacking logic and memory layouts. Standard FlashAttention cannot be used directly; a mixed-precision attention kernel must be implemented.
- **Per-channel Key statistics rely on batch quantization**: when batching $R$ tokens for quantization, per-channel scales are computed from those $R$ tokens only. If that batch is unrepresentative (e.g., the first 128 tokens differ strongly from later ones), scale estimation may be inaccurate. In practice, $R=128$ is usually robust enough.
- **Interaction with GQA**: in GQA, multiple query heads share the same KV head set, meaning the KV cache is already compressed (e.g., LLaMA-3 uses 8 KV heads vs 32 query heads). Further quantizing to 2-bit requires more careful evaluation, since each KV head serves more query heads and its errors may be amplified more often.
- **Choosing the residual buffer size**: if $R$ is too small, even recent tokens are quantized, hurting local attention precision; if $R$ is too large, the FP16 portion consumes too much memory and reduces the benefit of quantization. In practice, $R = 128$ is an empirical sweet spot.

### 6.4 MambaQuant — Quantization Challenges Beyond Transformers

> Xu, Yue et al., 2025 · ICLR 2025

**Why care about Mamba quantization?** All methods in §4-§6 assume the target architecture is a Transformer. But Mamba, based on Selective State Space Models, is becoming a major competitor to Transformers thanks to its linear complexity on long-sequence tasks. When we try to apply Transformer-successful quantization methods such as QuaRot directly to Mamba, we find that **accuracy collapses** — for example, QuaRot reduces W8A8 accuracy on Vim-T by 21%. MambaQuant is the first work to systematically study quantization for Mamba and propose a solution.

**Three major challenges in Mamba quantization**:

1. **Outliers in gate/output projection layers**: the gate projection weights and the input activations of the output projection in Mamba blocks exhibit significant outliers, similar to Transformers but with a different distribution pattern.

2. **Parallel Scan (PScan) amplifies outliers**: the core Mamba operation is PScan — repeatedly multiplying by a fixed parameter matrix $A$: $h(t) = A h(t-1) + B x(t)$. High-value channels are repeatedly amplified by PScan, while low-value channels are suppressed, causing the variance gap across output channels to become **much larger than in Transformers**.

3. **Hadamard transforms no longer suffice**: in Transformers, Hadamard rotation effectively equalizes maximum values (the core of QuaRot). But MambaQuant proves theoretically that a Hadamard transform **cannot guarantee equalized channel variance**. Specifically, after a Hadamard transform, the variance of channel $l$ becomes:

$$(\mathbf{C}_{\mathbf{X}\mathbf{H}})_{ll} = \frac{1}{n-1}\sum_{j=1}^{m}\left(\sum_{i=1}^{m}H_{il}K_{ij}\right)^2\lambda_j$$

Since $H$ is fixed while $K$ (eigenvectors) and $\lambda$ (eigenvalues) vary with the input, Hadamard cannot adapt to different channel distributions, so variances remain unequal after rotation.

**MambaQuant’s solution**:

**Offline mode — KLT-enhanced rotation**: combine the Karhunen-Loève Transform (KLT) with a Hadamard matrix. KLT identifies principal directions via eigendecomposition, and the combined matrix $H_K = KH$ (KLT first, then Hadamard) yields:

$$(\mathbf{C}_{\mathbf{X}\mathbf{H}_K})_{ll} = \frac{1}{(n-1)m}\sum_{j=1}^{m}\lambda_j$$

so that **all channel variances become exactly equal**, equal to the mean eigenvalue. KLT is computed offline from calibration data and adds no inference overhead. It is applied to LoRA modules and block-to-block connections (output/gate/state projection).

**Online mode — Smooth-Fused rotation**: for locations that cannot be absorbed offline (e.g., PScan output), first apply SmoothQuant-style channel scaling before Hadamard to equalize variance, then use Hadamard to equalize maximum values. The key innovation is that the smoothing parameters are carefully fused into the Mamba structure:
- **Output projection**: extend SiLU into Smooth-SiLU (S-SiLU), with smoothing factors absorbed into gate and output weights
- **Matrix multiplication**: absorb smoothing factors separately into the B and C projection weights, and use `addcmul` handling for the exponential operation on $\Delta$

**Experimental results**:

| Model | Method | W8A8 | W4A8 |
|---|---|---|---|
| Vim-T (76.1%) | QuaRot | 59.3% | 52.7% |
| Vim-T (76.1%) | **MambaQuant** | **75.6%** | **72.1%** |
| Vim-S (80.5%) | QuaRot | 73.8% | 72.0% |
| Vim-S (80.5%) | **MambaQuant** | **80.3%** | **79.4%** |
| Mamba-LLM | QuaRot | Significant degradation | — |
| Mamba-LLM | **MambaQuant** | <1% accuracy loss | ~1% accuracy loss |

**Key takeaway**: **quantization methods cannot be transferred blindly across architectures**. Best practices for Transformers may fail completely on SSM/Mamba architectures — one must understand the architecture-specific numerical properties (e.g., PScan amplification) and design accordingly. This is also instructive for future hybrid architectures such as Jamba (Mamba + Attention).

---

## 7 The Extreme Route: 1-bit / Ternary Networks

> The methods in §4-§6 all compress existing models (PTQ). A completely different route is to **train from scratch directly in extremely low bit**, so the model is inherently adapted to quantization. This is the aggressive application of QAT (§2.1) in the LLM era.

**Why take this route?** PTQ methods (GPTQ, AWQ, etc.) are essentially trying to "salvage" a model trained in FP16/BF16 — during training, the weight distribution is optimized for high precision, and forcing it into 4-bit inevitably loses information. A natural idea is: if the model is **optimized in low bit from the very beginning of training**, the weight distribution will naturally adapt to low-precision representation, and in principle can achieve better accuracy than PTQ. The cost is a full training pipeline.

**Important clarification: the "low-bit training" here is simulated quantization, not true low-precision computation.** The QAT framework introduced in §2.1 applies here as well — fake-quantization nodes (quantize → dequantize) in the forward pass let the model *see* quantization error, but **the actual GEMMs are still executed on FP16/FP32 Tensor Cores**, and gradients are still accumulated on FP32 latent weights. Low-bit weights are only exported for inference. In other words, BitNet’s training cost is comparable to that of FP16 models — what it saves is **inference**, not training-time compute. Methods that truly reduce training precision are discussed in §8 (FP8/FP4 training).

### 7.1 The Classic Route (CNN Era)

| Method | Year | Weights | Activations | Core Operation | Key Idea |
|---|---|---|---|---|---|
| BinaryConnect | 2015 | {-1, +1} | Real-valued | Multiplication → sign flip | First to show that binary weights can be trained |
| BNN | 2016 | {-1, +1} | {-1, +1} | XNOR + popcount | Binarize both weights and activations |
| TWN | 2016 | {-α, 0, +α} | Real-valued | Threshold-to-zero + scaling | Introduce zeros and sparsity |
| XNOR-Net | 2016 | {-1, +1} | {-1, +1} | XNOR + popcount + channel-wise scale | Recover accuracy with real-valued scale factors |

**The common framework of these methods**: maintain FP32 latent weights during training (the truly trainable parameters). In the forward pass, latent weights are quantized to binary/ternary, and then **immediately dequantized back to FP16/FP32** for standard GEMM — this is exactly the fake quantization introduced in §2.1. The model sees quantization error, but the actual computation still runs in floating point, so **there is no training-time speed benefit**. In backpropagation, STE (§1.4) lets gradients pass through the sign function and updates the FP32 latent weights. Only at inference time are the latent weights discarded and true binary/ternary weights exported, at which point specialized kernels (e.g., XNOR+popcount or `bitnet.cpp`) can provide acceleration.

**Why did this line not succeed in CNNs?** In theory, it gives 32× compression (FP32 → 1-bit), but in practice there are many problems:
- **Accuracy gap**: binarizing ResNet-18 on ImageNet reduces top-1 accuracy by 10-15%, and the drop is even larger for bigger networks (ResNet-50+).
- **No real hardware speedup**: XNOR + popcount is indeed fast on CPUs, but modern GPUs are optimized for FP16/INT8 matrix multiplication on Tensor Cores, and binary ops can actually be slower due to lack of hardware support.
- **Training instability**: STE is a very crude gradient approximation (the true gradient of `sign` is 0, while STE pretends it is 1), making training susceptible to oscillating gradients and convergence difficulties.

### 7.2 BitNet — 1-bit Networks in the LLM Era

> Wang et al., 2023 (Microsoft Research)

BitNet is the first work to extend extreme low-bit training to LLM scale. Its key architectural modification is to replace every `nn.Linear` with **BitLinear**:

**Weight binarization**:
$$\hat{W} = \text{Sign}(W - \mathbb{E}[W])$$

First subtract the mean (centering), then take the sign. The mean subtraction is crucial — if the weight distribution is asymmetric (mean ≠ 0), applying `Sign` directly leads to a severe imbalance between +1 and -1, wasting representational capacity.

**Activation quantization**: activations are quantized to $b$ bits (8-bit in the paper), using absmax symmetric quantization:
$$\hat{X} = \text{Quant}(X) = \text{Clip}\!\left(\left\lfloor \frac{X}{\max|X|} \cdot (2^{b-1} - 1) \right\rceil, -2^{b-1}+1, 2^{b-1}-1\right)$$

**Full BitLinear forward pass**:
```
BitLinear forward during training (fake quantization):
1. LayerNorm(x)                         ← stabilize the input distribution
2. W_bin = Sign(W - mean(W))            ← binarize the weights (values are ±1, but data type remains FP16)
3. X_q = absmax_quant(x, 8bit)          ← quantize activations then dequantize (values contain quantization error, type remains FP16)
4. Y = X_q @ W_bin                      ← standard FP16 GEMM (no acceleration during training)
5. Y = Y · (β · γ / Q_max)             ← rescale to restore the numeric range
   where β = mean(|W|), γ = max(|X|)

BitLinear at inference (true low bit):
  the exported W_bin is a true 1-bit packed format
  → a specialized kernel uses adds/subtracts instead of multiplies → achieves real speedup
```

**Training**: STE + FP32 latent weights. As in the classic methods, the forward pass uses {-1, +1} weights, while the backward pass propagates gradients to FP32 latent weights and updates them.

**Key finding**: BitNet shows that **scaling laws still hold for 1-bit models** — as model size grows, the accuracy gap between 1-bit and FP16 shrinks. This suggests that sufficiently large 1-bit models may match the performance of smaller FP16 models.

### 7.3 BitNet b1.58 — From {-1, +1} to {-1, 0, +1}

> Ma et al., 2024 (Microsoft Research)

BitNet b1.58 is the key improvement over BitNet: the weight space is extended from binary {-1, +1} to ternary {-1, 0, +1}, so each weight carries $\log_2 3 \approx 1.58$ bits of information.

**Weight quantization**:

$$\hat{W} = \text{RoundClip}\!\left(\frac{W}{\gamma},\,-1,\,1\right), \qquad \gamma = \frac{\|W\|_1}{nm}$$

$\gamma$ is the mean absolute value of the weights (an L1 normalization factor). After normalization, most weights fall in $[-1, 1]$ and can be rounded to {-1, 0, +1}.

**Why is introducing 0 such a key improvement?**

1. **Matrix multiplication becomes pure addition/subtraction**: in $y_i = \sum_j w_j x_j$, when $w_j \in \{-1, 0, +1\}$, the term $w_j x_j$ is either $+x_j$, $-x_j$, or 0 (skip it), eliminating multiplication completely. Compared with BitNet’s {-1, +1} (addition/subtraction only), 0 introduces **sparsity** — some weights are skipped entirely.
2. **Better feature selection**: 0 lets the model *ignore* certain input dimensions, effectively performing implicit feature selection. {-1, +1} forces every input to participate in the computation, even if some inputs are unimportant for the current output.
3. **Substantially better accuracy**: at the same model size, b1.58 significantly outperforms BitNet (1-bit), and at 3B scale can already **match FP16 LLaMA performance**.

**Fundamental difference from PTQ methods**:

| | PTQ (GPTQ/AWQ) | BitNet b1.58 |
|---|---|---|
| Starting point | A model trained in FP16 | Trained from scratch |
| Weight distribution | Optimized for FP16 and forcibly compressed into low bit | Naturally adapts to ternary values during training |
| Accuracy | Good at 4-bit; collapses below 3-bit | 1.58-bit can match FP16 |
| Cost | Minutes to hours | Full training (same cost as FP16 training) |
| Inference hardware | Existing GPUs + custom kernels | Requires entirely new hardware/kernel support |

**Challenges and open questions**:
- **Requires specialized hardware/kernels**: current GPU Tensor Cores do not support ternary matrix multiplication. Although addition/subtraction is theoretically much faster than multiplication, the speedup cannot be realized without native hardware support. Microsoft has released the `bitnet.cpp` inference engine.
- **Training infrastructure**: training still requires FP32/FP16 latent weights + STE, so the training cost is comparable to FP16 models. The advantage exists only at inference time.
- **Compatibility with alignment/RLHF**: after pretraining, large models still need alignment steps such as SFT, RLHF, and DPO. The stability and effectiveness of these stages under ternary weight constraints remain insufficiently validated.
- **Scaling to larger models**: the largest public experiments so far are at 3B scale; evidence beyond tens of billions is still lacking.

---

## 8 Low Precision on the Training Side

> §4-§6 focus on PTQ inference quantization, while BitNet in §7 follows the route of "training in low bit from scratch" — but BitNet’s goal is still to produce a low-bit **inference** model, and its training process itself still uses FP16/FP32. This section focuses on a different direction: **making the training computation itself low precision** — not to obtain a low-bit model, but to **make training faster and more memory-efficient**. The foundation of mixed-precision training was introduced in §1.5; here we focus on more aggressive FP8/FP4 training and low-precision optimizers.

### 8.1 FP8 Training and Scaling

> Micikevicius et al., 2022

**Why go beyond FP16/BF16 down to FP8?** Mixed-precision training has already reduced GEMM from FP32 to FP16/BF16 (§1.5), but training scale continues to grow exponentially. FP8 Tensor Cores offer **2×** the throughput of FP16 (H100: FP8 1979 TFLOPS vs FP16 990 TFLOPS), while also halving memory-bandwidth usage. If training GEMMs can be pushed further down to FP8, another theoretical 2× speedup becomes possible.

**Why is FP8 training harder than FP16 training?** FP8 has only 8 bits, so both representable range and precision are very limited (§1.6). The key difficulty is that different tensors in training have very different numerical characteristics:

- **Weights**: relatively concentrated distributions, usually with magnitudes in roughly $[0, \text{a few}]$, so precision matters → use **E4M3** (3-bit mantissa, range ±448)
- **Activations**: distributions similar to weights but may contain outliers → also use **E4M3**
- **Gradients**: extreme distributions — most values near zero, with a few very large values (especially early in training and in deeper layers), so they need a large dynamic range → use **E5M2** (2-bit mantissa, range ±57344)

| Format | Exponent | Mantissa | Dynamic Range | Typical Use |
|---|---|---|---|---|
| **E4M3** | 4 | 3 | ±448 | Forward: weights / activations |
| **E5M2** | 5 | 2 | ±57344 | Backward: gradients |

**Scaling is the core technology of FP8 training**. Even with the right E4M3/E5M2 format, FP8’s dynamic range remains limited. If actual tensor values are far smaller than the FP8 maximum, most quantization levels are wasted; if they exceed the FP8 maximum, they overflow directly to NaN/Inf. Scaling means rescaling the tensor before casting to FP8:

$$x_{\text{fp8}} = \texttt{cast\_fp8}(x \,/\, s), \qquad s = \frac{\max|x|}{f_{\max}}$$

where $f_{\max}$ is the maximum representable value of the FP8 format (E4M3: 448, E5M2: 57344). During dequantization, multiply by $s$ to recover the original scale.

**Comparison of scaling strategies** — finer granularity improves precision but increases overhead:

| Scaling Strategy | Description | Accuracy | Overhead | Representative |
|---|---|---|---|---|
| **Per-tensor** | One $s$ for the whole tensor | Low (outliers enlarge $s$) | Lowest | Transformer Engine default |
| **Per-token × per-channel** | One $s$ per token for activations, one per output channel for weights | Medium (dimension-wise adaptation) | Medium | Some academic work |
| **Per-block (MXFP8)** | One $s$ per fixed tile (e.g., 32 elements), with $s$ itself stored in 8-bit | High (good local precision) | Higher | Blackwell MXFP8 |
| **Delayed scaling** | Use the `amax` from previous steps instead of the current step’s $\max\lvert x \rvert$ (which would require an extra pass) | Medium | Low (but introduces temporal dependence) | NVIDIA recipe |

**Delayed scaling in detail**: one practical issue with per-tensor scaling is that to compute $s = \max|x| / f_{\max}$, one must first run a forward pass to obtain $x$, then compute $s$, and then rerun the forward pass with quantization. To avoid this extra pass, delayed scaling estimates the current step’s $s$ using historical $\max|x|$ values from previous steps. Assuming that neighboring steps have similar statistics (which is usually true), this one-step delay is accurate enough.

**The full end-to-end FP8 training recipe**:

```
Forward pass:
  weights cast to FP8 E4M3 (per-tensor scaling)
  activations cast to FP8 E4M3 (per-tensor scaling)
  GEMM: FP8 × FP8 → FP16/FP32 accumulation (Tensor Cores use high-precision accumulation internally)

Backward pass:
  gradients cast to FP8 E5M2 (per-tensor scaling)
  weight-gradient GEMM: FP8 × FP8 → FP16/FP32 accumulation

Parameter update:
  FP32 master weights + FP32 optimizer states (same as FP16 mixed precision)
```

**Current hardware support**:
- **H100 (Hopper)**: native FP8 Tensor Cores supporting both E4M3 and E5M2, with per-tensor scaling. NVIDIA’s Transformer Engine provides an out-of-the-box FP8 linear layer.
- **B200 (Blackwell)**: further supports MXFP8 (per-block scaling, block size 32), with higher accuracy; the scale $s$ is stored with an 8-bit shared exponent so metadata overhead remains controlled.
- **AMD MI300X**: supports FP8, but the software ecosystem and optimization maturity lag behind NVIDIA.

**Bottleneck**: FP8 currently accelerates GEMM operations primarily. But training also contains many non-GEMM operators (LayerNorm, softmax, residual connections, activation functions, etc.) that still run in FP16/BF16. If these operators account for a substantial portion of runtime — especially in small-batch or attention-heavy settings — the speedup from FP8 GEMMs will be diluted by the non-GEMM part, per Amdahl’s law.

### 8.2 Low-Precision Optimizers

> Dettmers et al., 2021 (bitsandbytes 8-bit Adam)

**Problem**: when training an LLM, the main memory consumer is not the model weights, but the **optimizer state**. For example, with Adam on a 7B model:

| Component | Precision | Memory |
|---|---|---|
| Model weights | BF16 | 14 GB |
| Gradients | BF16 | 14 GB |
| FP32 master weights | FP32 | 28 GB |
| Adam $m$ (first moment) | FP32 | 28 GB |
| Adam $v$ (second moment) | FP32 | 28 GB |
| **Total** | | **~112 GB** |

Optimizer states ($m + v$) occupy 56 GB, more than 4× the weight memory itself. If $m$ and $v$ can be compressed from FP32 to INT8, that saves 42 GB (75%).

**Why not just store $m, v$ in FP16?** The distributions of $m$ and $v$ differ from those of weights: $m$ is the exponential moving average of gradients, and can be positive or negative with a relatively concentrated distribution; $v$ is the moving average of squared gradients, always non-negative, and its scale can vary by orders of magnitude across parameters. FP16’s 5-bit exponent provides limited dynamic range, so some $v$ values may underflow to zero, causing Adam’s update step to explode ($\text{update} \propto m / \sqrt{v}$; as $v \to 0$, the denominator approaches zero).

**The bitsandbytes approach — block-wise dynamic INT8 quantization**:

```
Adam update at each step:

1. Dequantize: m_fp32 = dequant_int8(m_int8),  v_fp32 = dequant_int8(v_int8)
2. Standard Adam update (FP32):
     m_fp32 = β1 · m_fp32 + (1-β1) · grad
     v_fp32 = β2 · v_fp32 + (1-β2) · grad²
     weight -= lr · m_fp32 / (√v_fp32 + ε)
3. Requantize: m_int8 = quant_int8(m_fp32),  v_int8 = quant_int8(v_fp32)
```

**Key design — why block-wise?** If one uses a single scale to quantize the entire $m$ or $v$ vector into INT8, a few large values enlarge the scale and squeeze the precision of all other parameters — exactly the same outlier problem discussed in §3. Block-wise quantization (block size = 2048) gives each block an independent scale, so local large values affect only local precision rather than contaminating the whole tensor.

**Stable Embedding**: the embedding layer is special because its gradients are extremely sparse — in each batch, only embeddings corresponding to sampled tokens have nonzero gradients, while the rest are zero. This makes updates to $m, v$ highly irregular (staying at zero for long periods and then suddenly jumping), so INT8 quantization can easily lose small updates. The solution is simple: **keep FP32 optimizer state for embedding layers**, and use INT8 for the rest. Embeddings account for relatively few parameters, so the extra FP32 memory is acceptable.

**Numbers**:
- About 75% memory savings (optimizer state from FP32 → INT8 + a small amount of scale metadata)
- On GPT-2, RoBERTa, BLOOM, and similar models, convergence curves are nearly unchanged, and final accuracy matches FP32 Adam closely
- Minimal usage change: simply replace `torch.optim.Adam` with `bnb.optim.Adam8bit`; the interface is fully compatible

**Subsequent developments**:
- **4-bit optimizers** (Dettmers et al., 2024): further compress to 4-bit using non-uniform quantization (similar to NF4), giving more memory savings but requiring more careful hyperparameter tuning
- **Galore** (Zhao et al., 2024): compress from a different angle by projecting gradients into a low-rank subspace, reducing optimizer-state dimensionality instead of precision. This is orthogonal to 8-bit optimizers and can be combined with them.

### 8.3 Quartet — FP4 Pretraining

> Panferov, Chen et al., 2025 · ICML 2025

**Why go lower than FP8?** §8.1 introduced FP8 training, which is already natively supported on Blackwell. But Blackwell also supports **MXFP4** Tensor Core computation, which is theoretically another 2× faster than FP8. The question is whether such a coarse representation — FP4 has only seven nonzero representable values in E2M1 — can still converge in training. Quartet shows that it can, and gives a concrete method.

**Recap of MXFP4**: every 32 FP4 elements share one E8M0 scale (power-of-two). Each element uses 1 sign bit + 2 exponent bits + 1 mantissa bit. The representable positive values are only {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}; adding 0 and corresponding negatives gives 15 values in total.

**Core method — asymmetric quantization for forward vs backward**:

Quartet’s key insight is that **forward and backward propagation have completely different requirements for quantization**.

- **Forward pass** (weights × activations): requires high precision because it directly determines output quality. Quartet uses **QuEST** (Quantization with Error-aware Stochastic Training) — after Hadamard rotation, it searches for the clipping ratio that minimizes MSE and then quantizes deterministically. This is accurate but somewhat heavier.
- **Backward pass** (gradients × activations/weights): can tolerate much more noise, since gradients are already stochastic estimates. Quartet therefore uses **stochastic rounding (SR)**, turning quantization noise into a zero-mean random variable so the gradient remains unbiased in expectation. SR is cheap, though any single sample is less precise.

```
One training step of Quartet:

Forward pass:
  X̃ = Hadamard(X)    # rotate to disperse outliers
  W̃ = Hadamard(W)
  Y = QuEST_FP4(X̃) × QuEST_FP4(W̃)    # deterministic optimal quantization

Backward pass:
  ∂L/∂X = SR_FP4(∂L/∂Y) × W̃ᵀ          # stochastic rounding, unbiased
  ∂L/∂W = X̃ᵀ × SR_FP4(∂L/∂Y)

Optimizer update: FP32 master weights (same as mixed-precision training)
```

**Low-precision scaling law**: Quartet proposes a precision-aware scaling law:

$$L(N, D, P_{fwd}, P_{bwd}) = \frac{A}{(\text{effN})^\alpha} + \frac{B}{(\text{effD})^\beta} + E$$

where $\text{effN} = N \cdot \rho(P_{fwd})$ is the "effective parameter count" (lower forward precision reduces parameter information capacity), and $\text{effD} = D \cdot \eta(P_{bwd})$ is the "effective data amount" (lower-precision gradients reduce how much each token contributes to learning). The core findings are:

- **Forward precision affects parameter efficiency**: FP4 forward has $\rho \approx 0.69-0.78$, meaning an FP4 model needs about 1.3-1.45× more parameters than BF16 to match the same accuracy
- **Backward precision affects data efficiency**: FP4 backward has $\eta \approx 0.85$, meaning about 1.18× more data is needed as compensation
- **Forward is more sensitive than backward**: this explains why Quartet uses more accurate QuEST in the forward pass and simpler SR in the backward pass

**Experimental results**:
- On Llama-style architectures at 60M-1.3B scale, FP4 training trails the FP8 baseline by only **0.5-1.0 PPL**
- Compared with a BF16 baseline, FP4 needs about 1.3× more parameters to reach the same accuracy
- **Hardware acceleration**: on Blackwell GPUs, FP4 GEMM throughput is **2×** that of FP8, yielding about **2×** end-to-end training speedup

**Limitation**: so far this has been validated only for pretraining; FP4’s extremely coarse granularity may be more challenging in fine-tuning, where learning rates are small and gradient signals are weak.

### 8.4 HALO — Hadamard-Assisted Low-Precision Fine-Tuning

> Ashkboos et al., 2025

**Motivation**: §8.1-§8.3 discussed low precision for pretraining. But before deployment, large models usually still need fine-tuning (SFT/RLHF), and for many users this is the primary training scenario. Fine-tuning has two special characteristics: (1) its learning rate is 10-100× lower than pretraining, so gradient signals are weaker and more precision-sensitive; (2) it is often done on consumer GPUs (e.g., RTX 4090), where memory is tighter. HALO is designed specifically for low-precision fine-tuning.

**Core idea — a two-level quantization scheme**:

HALO proposes two levels so users can choose based on hardware constraints:

**HALO-1 (FP6 level)**: quantize weights and activations to FP6, and run forward GEMM in FP6. This is the conservative option, with very small accuracy loss and primarily a 1.5× memory reduction.

**HALO-2 (INT8 level)**: quantize weights and activations to INT8, and run both forward and part of backward GEMMs in INT8. This is more aggressive, giving 2× memory savings and compute acceleration.

**Key technique — right-hand-side vs left-hand-side Hadamard rotation**:

Methods such as QuaRot rotate only on the "right-hand side": $Y = (XR^T)(RW) = \tilde{X}\tilde{W}$, dispersing outliers in activations and weights. But HALO finds that in fine-tuning, the **error gradients** ($\partial L / \partial Y$) also exhibit severe outliers. If only forward-side $X$ and $W$ are rotated while backward-side $\partial L / \partial Y$ is left untreated, quantization error in backpropagation becomes large.

HALO’s innovation is to also rotate on the "left-hand side":

$$Y = (Q \cdot X \cdot R^T) \cdot (R \cdot W \cdot P^T)$$

- $R$: right-side rotation, dispersing activation and weight outliers
- $Q$: left-side rotation, dispersing outliers in the error gradient $\partial L / \partial Y$
- $P$: output-side rotation, handling outliers along the output direction

In backpropagation: $\partial L / \partial X = Q^T \cdot \text{quant}(\partial L / \partial Y_{\text{rotated}}) \cdot R \cdot W$. In other words, $\partial L / \partial Y$ is first rotated by $Q$ to equalize it, then quantized, leading to much higher precision.

**HQ-FSDP: quantized distributed communication**:

Fine-tuning large models typically uses FSDP (Fully Sharded Data Parallelism), where communication is a bottleneck. HALO also quantizes the tensors transmitted during FSDP `all-gather` (weight distribution) and `reduce-scatter` (gradient aggregation):
- All-gather: transmit quantized weights and dequantize them on the receiving side
- Reduce-scatter: quantize gradients before transmission and dequantize after aggregation
- Communication volume drops by 2-4×, providing a significant speedup in multi-GPU training

**Experimental results**:
- On Llama-2-7B fine-tuning, HALO-2 incurs <0.5% accuracy loss and speeds training up by **1.41×** on a single RTX 4090
- HQ-FSDP gives an additional 15-20% speedup on 4× RTX 4090
- Compared with QLoRA: HALO performs full-parameter low-precision fine-tuning, while QLoRA uses low-rank adaptation + quantization. They are complementary and can be combined.

### 8.5 Hidden Risks of Low-Precision Training with Flash Attention

> Qiu & Yao, 2025 — "Why Low-Precision Transformer Training Fails"

**Problem**: BF16 Flash Attention training can suddenly experience loss explosion in some settings — for example, GPT-2 diverges after roughly 6600 steps. This behavior has been repeatedly reported in issues for projects such as nanoGPT and flash-attention, but previously lacked a mechanistic explanation. This section dissects the root cause and presents a fix.

**Tracing the failure chain**:

Through systematic debugging, the authors locate the issue to a key intermediate quantity in Flash Attention backpropagation:

$$\delta[T] = \text{rowsum}(dO \odot O)[T]$$

where $O = \text{softmax}(QK^\top)V$ is the attention output, and $dO$ is the output gradient. Under low precision, a **systematic bias** (not random noise) appears between low-precision $\delta_{\text{lp}}$ and high-precision $\delta_{\text{hp}}$, causing the weight gradient to accumulate a low-rank error matrix whose spectral norm grows until training blows up.

**Two root causes acting together**:

| Root Cause | Mechanism | Consequence |
|---|---|---|
| **Similar low-rank representations** | During training, the attention matrix $P$, projection matrix $K$, and hidden state $X$ develop similar low-rank structures $R$ across tokens | Gradient error $dW_{\text{hp}} - dW_{\text{lp}} \approx \alpha \sum (\delta_{\text{lp}} - \delta_{\text{hp}})[T] \cdot R$, which does not vanish when averaged over tokens |
| **Biased BF16 rounding** | In safe softmax, if multiple values in a score row share the same maximum, exact 1’s appear in $\bar{P}$. The subsequent BF16 accumulation in $\bar{P}V$ then incurs a **systematic negative bias** due to significand overflow | $O$ becomes negatively biased → $(\delta_{\text{lp}} - \delta_{\text{hp}})[T]$ becomes systematically positive → the spectral norm of the weights grows monotonically |

**Why is it dangerous when $\bar{P}$ contains exact 1’s?**

Safe softmax computes $\bar{P} = \exp(S - m)$, where $m = \text{rowmax}(S)$. If multiple tokens in a row share the maximum score, then $S[T,t] - m = 0$ and $\exp(0) = 1.0$. In this case, the BF16 fused multiply-add in $\bar{P}[T,t] \times V[t,i]$ produces a biased error due to significand alignment and rounding — completely different from the symmetric error statistics observed when $\bar{P} < 1$.

**Connection to attention sinks**: the attention sink phenomenon — where a few tokens attract extremely high attention scores — makes the case $\bar{P} = 1$ occur more often, providing a direct numerical explanation for why attention sinks exacerbate training instability. This also echoes the outlier issue discussed in §3: outliers affect not only inference quantization, but also training stability through concentrated attention.

**Fix: dynamic maximum adjustment (Stabilized Flash Attention)**

The core idea is to exploit the shift-invariance of softmax, $\text{softmax}(z) = \text{softmax}(z - c)$. When repeated maxima are detected within a row, dynamically increase the normalization constant $m$ so that all entries of $\bar{P}$ are strictly less than 1:

$$r_m = \text{rowmax}(S), \quad r_s = \text{rowsum}(S \equiv r_m)$$
$$m = \begin{cases} \beta \cdot r_m & \text{if } r_m > 0 \wedge r_s > 1 \\ 0 & \text{if } r_m < 0 \wedge r_s > 1 \\ r_m & \text{otherwise} \end{cases}$$

where $\beta \in [2, 8]$. Then $\max(S - m) < 0$, hence $\max(\bar{P}) < 1$, eliminating the trigger for biased rounding.

**Design considerations**:
- **Why not use a fixed offset?** Subtracting a fixed small constant introduces a new systematic rounding bias when converting $\bar{P}$ to BF16, so it does not solve the problem fundamentally
- **Why trigger conditionally?** Unconditionally adjusting $m$ can cause $\exp(S - \beta r_m)$ to underflow to 0 when $r_m$ is very large, leading to division-by-zero errors
- **Only the forward pass is modified**: the change is just one extra conditional in the softmax tiling algorithm (Algorithm 1, lines 8-9); the backward pass does not need to be changed

**Experimental validation**:
- GPT-2 BF16 training: original Flash Attention explodes at around 6600 steps, while Stabilized Flash Attention ($\beta=7$) remains stable throughout training
- The fix works on NVIDIA A100, RTX 4090, and Huawei Ascend 910B
- It is mathematically exactly equivalent to standard attention (it merely chooses a different shift constant $c$), so model accuracy is unchanged

**Takeaway**: this work reveals an easily overlooked fact — **instability in low-precision training does not necessarily come from explicit quantization operations; it may also hide in the numerical implementation details of basic kernels such as Flash Attention**. The BF16 rounding issue in Flash Attention also applies to the low-precision training methods discussed in §8.1-§8.4 and should be addressed with stabilization in tandem.

### 8.6 How Does the Optimizer Choice Affect Quantization?

> Vlassis, Ashkboos et al., 2025 · "Beyond Outliers"

In §3 we introduced outliers as the core challenge in LLM quantization, and in §4-§6 we covered various PTQ methods for dealing with them. One overlooked question is: **all quantization methods assume a fixed pretrained model — but are models trained with different optimizers equally quantization-friendly?** Beyond Outliers is the first work to systematically study the interaction between optimizers (AdamW, Muon, PSGD, Shampoo, SOAP, Scion) and quantization (PTQ + QAT), at scales from 50M to 1.5B parameters, and it arrives at several counterintuitive conclusions.

**Core finding 1: outlier metrics do not predict PTQ accuracy**

§3 introduced outliers as the central difficulty of LLM quantization. Intuitively, larger outliers should mean harder quantization — traditional work uses metrics such as MMR (max/median ratio) or kurtosis to measure outlier severity, and designs quantization methods accordingly (e.g., SmoothQuant in §4.2 explicitly aims to reduce activation outliers). But Beyond Outliers finds that **when comparing across optimizers, MMR and kurtosis are almost uncorrelated with post-PTQ accuracy** ($\rho = 0.62$ and $\rho = -0.89$ for a 760M model):

- **Muon** has the lowest MMR (fewest outliers), but suffers the **largest drop** after PTQ (760M: 64.63% → 50.00%)
- **Shampoo** has the highest MMR (largest outliers), but preserves accuracy **best** after PTQ (760M: 63.05% → 59.26%)

**Why does MMR fail?** The paper gives a theoretical explanation via the ABC decomposition framework. MMR is a **single-layer** metric, but the total effect of quantization error depends on how that error **propagates and amplifies layer by layer**. Specifically, the quantization error at layer $\ell$, $\Delta h_\ell = h_\ell^q - h_\ell$, can be decomposed as:

$$R_\ell = A_\ell + B_\ell + C_\ell$$

- $A_\ell$: accumulated errors from previous layers propagated through the current layer (**the dominant term** — similar to the propagation phenomenon discussed in §6.1 QuaRot)
- $B_\ell$: new quantization error introduced at the current layer (the part MMR can partly predict)
- $C_\ell$: the interaction term between the two

The key finding is that $R_\ell$ is dominated almost entirely by $A_\ell$. Even if MMR can predict the single-layer error $B_\ell$, the total error is governed by the **gain** $G_\ell = A_\ell / R_{\ell-1}$ — the amplification factor of quantization error through each layer:

$$G_\ell = G_{1,\ell} \cdot G_{2,\ell}$$

where $G_{1,\ell}$ is the spectral-norm ratio (change in weight spectral norm before vs after quantization, close to 1 across optimizers), and $G_{2,\ell}$ is the alignment ratio (how well the quantization-error direction aligns with the dominant singular direction of the weight matrix). **Muon has the largest $G_\ell$ in linear layers** — its quantization error points exactly in the direction most amplified by the weights, so errors accumulate rapidly; Shampoo and AdamW have the smallest $G_\ell$.

The paper’s proposed metric $R_L$ (the accumulated quantization error in the final layer) correlates strongly with PTQ accuracy ($\rho = 0.70$).

**Core finding 2: the best optimizer for QAT is not the best optimizer for full precision**

In full-precision training, Muon performs best, but under 4-bit QAT (using the QuEST scheme mentioned in §8.3), the optimizer ranking changes completely. **Shampoo suffers the least degradation in QAT** — for example, on 760M, Shampoo drops by only -0.46%, while Muon drops by -3.57%.

**Core finding 3: a scaling law for QAT**

Similar to the low-precision scaling law proposed by Quartet in §8.3 (using effN to describe how precision affects parameter efficiency), Beyond Outliers derives an optimizer-aware scaling law for QAT: $L = A' / (N \cdot \rho)^\alpha + E$, where $\rho$ is "parameter efficiency" — the equivalent parameter count of a 4-bit QAT model is $\rho \cdot N$. The reported $\rho_{4bit}$ for different optimizers is:

| Optimizer | $\rho_{4bit}$ | Meaning |
|---|---|---|
| **Shampoo** | **0.879** | 4-bit retains 87.9% parameter efficiency |
| AdamW | 0.863 | |
| Scion | 0.856 | |
| Muon | 0.852 | |
| SOAP | 0.822 | |

**Practical takeaway**: if the final goal is to deploy a quantized model, optimizer choice during training matters too — models trained with Shampoo are more robust to quantization. For PTQ, traditional outlier metrics (MMR, kurtosis) can mislead optimization, and one should focus on error propagation rather than single-layer statistics. **Quantization friendliness must be evaluated from a global perspective, not just local metrics.**

---

## 9 Engineering Implementation of Quantization

> Quantization algorithms ultimately need engineering implementations before they can be deployed. This section uses bitsandbytes as an example to explain the implementation details of quantization libraries, and then discusses the architectural differences between quantization libraries and inference frameworks.

### 9.1 bitsandbytes Quantization Implementation

bitsandbytes is a typical engineering realization of the uniform quantization formula in §1.1, with two layers: a Python API and CUDA kernels.

**Python-side invocation**:

```python
import bitsandbytes.functional as F

x_q, state = F.quantize_blockwise(x_fp32)   # quantize
x_deq = F.dequantize_blockwise(x_q, state)  # dequantize
```

`state` (a `QuantState` object) stores `absmax` (the maximum absolute value per block = scale), `code` (the codebook), `blocksize`, and related metadata.

**8-bit symmetric quantization (blockwise)** — corresponding to the symmetric formula in §1.1, but computing the scale independently per block (default 2048 elements):

$$s_{\text{block}} = \frac{\text{absmax}_{\text{block}}}{127}, \qquad x_q = \left\lfloor \frac{x}{s_{\text{block}}} \right\rceil, \qquad \hat{x} = s_{\text{block}} \cdot x_q$$

The benefit of block-wise quantization is that it prevents a few large global values from inflating the scale and squeezing the precision of the remaining normal values.

**4-bit NF4 quantization** (used by QLoRA) — assuming weights are approximately normally distributed, precompute 16 optimal quantization points as a codebook:

```python
# NF4 codebook (hard-coded, information-theoretically optimal for N(0,1))
code = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0]

# Quantization: normalize then find the nearest codebook index
x_norm = x / absmax_per_block
x_q = argmin_i |x_norm - code[i]|     # 4-bit index (0~15)

# Dequantization
x_hat = absmax_per_block * code[x_q]
```

Unlike uniform quantization, NF4’s quantization points are non-uniformly distributed: denser near the center and sparser in the tails, matching the probability density of a normal distribution.

**CUDA kernel layer** (`csrc/kernels.cu`) — key implementation snippets:

**1. absmax reduction within a block** — uses CUB’s `BlockReduce` primitive, which is more efficient than a handwritten shared-memory reduction:

```cuda
// Each thread processes `NUM_PER_TH` elements, first computing a local abs max, then using CUB for block-level reduction
local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, BNB_MAX_OP, valid_items);

if (threadIdx.x <mark> 0) {
    smem_absmax_value[0] = 1.0f / local_abs_max;  // Store the reciprocal so multiplication can replace division later
    absmax[i / BLOCK_SIZE] = local_abs_max;
}
```

**2. NF4 quantization — decision tree instead of linear search**: nearest-neighbor lookup over 16 non-uniform quantization points is compiled into a binary decision tree ($O(\log 16) = 4$ comparisons), avoiding a full codebook scan:

```cuda
__device__ unsigned char dQuantizeNF4(float x) {
    // Expanded at compile time into 4 levels of if-else; each leaf returns a 4-bit index
    if (x > 0.03979014977812767f)
        if (x > 0.3893125355243683f)
            if (x > 0.6427869200706482f)
                if (x > 0.8614784181118011f) return 0b1111;
                else return 0b1110;
            else if (x > 0.5016634166240692f) return 0b1101;
            else return 0b1100;
        // ... 16 leaf nodes total; thresholds are the midpoints of adjacent codebook values
}
```

**3. 4-bit packing — two values packed into one byte**:

```cuda
// NF4: each byte stores two 4-bit quantized values, high 4 bits + low 4 bits
for (int j = 0; j < NUM_PER_TH / 2; j++) {
    qvals[j]  = dQuantizeNF4(((float)vals[2*j])   * local_abs_max) << 4;
    qvals[j] |= dQuantizeNF4(((float)vals[2*j+1]) * local_abs_max);
}
```

**4. Dequantization — lookup table + bit operations**:

```cuda
__device__ __forceinline__ float dDequantizeNF4(unsigned char val) {
    return nf4_dequantization_lut[val & 0x0F];  // 16-entry LUT with precomputed codebook values
}
```

**Main engineering challenges**:
- **Reduction efficiency**: each block (2048 elements) needs an absmax reduction. A naive shared-memory reduction suffers from bank conflicts and synchronization overhead. bitsandbytes relies on CUB’s `BlockReduce`, which internally uses warp-shuffle instructions to avoid shared-memory traffic.
- **Alignment issues in 4-bit packing/unpacking**: two 4-bit values share one byte, so reads/writes require shifts and masks. Special handling is needed when block boundaries are misaligned (`valid_items`).
- **Warp utilization for small blocks**: when blocksize is small (e.g., 64), standard kernels waste threads because the thread block is too large. bitsandbytes therefore implements `kQuantizeBlockwiseSmall`, replacing `BlockReduce` with `WarpReduce` so one thread block can process multiple blocks.
- **Memory access pattern in dequantization**: the LUT used by `dDequantizeNF4` contains only 16 floats and stays resident in L1 cache, but random access patterns of quantized values can still cause cache misses. The implementation mitigates this with `__ldg()` (read-only cache) and contiguous tile loading.

**Overall architecture**:

```
Python API (functional.py)
  ├── quantize_blockwise()   →  symmetric INT8, blocksize=2048
  ├── dequantize_blockwise()
  ├── quantize_nf4() / quantize_fp4()  →  4-bit non-uniform codebook
  └── QuantState  ← stores absmax, code, blocksize, dtype
        │
CUDA kernels (csrc/kernels.cu)
  ├── kQuantizeBlockwise / kDequantizeBlockwise   ← 8-bit
  └── kQuantize4bit / kDequantize4bit             ← NF4/FP4
```

In summary, bitsandbytes adds two key engineering optimizations on top of the basic formula: **Blockwise** quantization (compute a separate scale for each block for higher precision) and the **NF4 codebook** (for 4-bit, replace uniform quantization with quantization points optimized for a normal distribution, matching weight distributions better).

### 9.2 Quantization Libraries vs Inference Frameworks: Two Deployment Paths

bitsandbytes represents the **quantization toolkit** route — it provides quantize/dequantize primitives that users call from PyTorch training or inference pipelines. But in production deployment, the dominant approach is to use **inference frameworks** such as vLLM and TensorRT-LLM, which deeply fuse quantization into inference kernels. The fundamental difference between the two is:

| Dimension | Quantization Toolkit (bitsandbytes, TorchAO) | Inference Framework (vLLM, TensorRT-LLM, SGLang) |
|---|---|---|
| **Where quantization happens** | Explicit quantize/dequantize calls at the Python level | Quantization logic is fused directly into CUDA kernels |
| **Compute flow** | Quantized storage → dequantize back to FP16 → standard GEMM | Quantized weights participate in computation directly; dequantization is pipelined with GEMM (e.g., the MARLIN kernel hides INT4 unpacking behind memory latency) |
| **Performance bottleneck** | Dequantization is a separate step with extra overhead | Dequantization overhead is fully hidden by compute, approaching the theoretical bandwidth limit |
| **Typical speedup** | Mainly saves memory; speed gains are limited | W4A16 can reach ~3.9× speedup (vs FP16), close to the theoretical 4× |
| **Use case** | Research experiments, QLoRA fine-tuning, rapid prototyping | Production deployment, high-throughput online serving |

**Why can inference frameworks approach theoretical speedup?** Consider the MARLIN kernel (Frantar et al., 2024), used by default in vLLM. Small-batch LLM inference is memory-bound, where the bottleneck is loading weights from HBM, and 4-bit weights should theoretically cut bandwidth by 4×. MARLIN uses an asynchronous dequantization + compute pipeline: while the $i$-th batch of weights is copied asynchronously from global memory to shared memory, Tensor Cores execute the FP16 GEMM for batch $i-1$. This hides dequantization completely behind memory latency, and on A100 reaches more than 95% of the theoretical bandwidth ceiling.

**Practical choice**: use bitsandbytes/TorchAO for fast iteration and validation during research → once the quantization scheme is finalized, deploy it using fused kernels in an inference framework. These are not competing alternatives, but different stages in the quantization workflow.

### 9.3 Reading the AWQ Triton Kernel

> The source code comes from vLLM’s AWQ Triton implementation (`quantization/test_awq_triton.py`). By analyzing this real inference kernel line by line, we can understand concretely what it means in §9.2 to "fuse quantization into the kernel."

§5.2 introduced the AWQ algorithm (activation-aware weight quantization); here we focus on its kernel implementation. AWQ quantizes FP16 weights into a 4-bit asymmetric format, packing eight 4-bit values into one `int32`:

```
Original weights:  W (K, N) float16     ← occupy K×N×2 bytes
After quantization:
  qweight: (K, N/8)   int32     ← each int32 = 8 4-bit weights, 4× compression
  zeros:   (K/G, N/8) int32     ← per-group zero-points, packed the same way
  scales:  (K/G, N)   float16   ← per-group scaling factors
```

Dequantization formula (asymmetric quantization): $W_{\text{fp16}} = (W_{\text{int4}} - Z_{\text{int4}}) \times S_{\text{fp16}}$

#### 9.3.1 Kernel 1: `awq_dequantize_kernel`

This kernel unpacks the packed `int32` weights of shape `(K, N/8)` into an FP16 matrix of shape `(K, N)`.

**AWQ’s special packing order**:

```python
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]           # when packing, the i-th weight goes into the AWQ_ORDER[i]-th nibble
AWQ_REVERSE_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]   # inverse mapping used during unpacking
```

This order is not arbitrary — it is chosen to work with Triton’s `tl.interleave` hardware instruction so that unpacking has effectively zero extra cost.

**Core unpacking procedure**:

```python
# 1. Load (BY, BX) int32 values
iweights = tl.load(qweight_ptr + offsets)

# 2. Apply interleave three times to replicate each int32 eight times → (BY, BX*8)
iweights = tl.interleave(iweights, iweights)   # [a,b] → [a,a,b,b]         ×2
iweights = tl.interleave(iweights, iweights)   # → [a,a,a,a,b,b,b,b]       ×4
iweights = tl.interleave(iweights, iweights)   # → [a×8, b×8, ...]          ×8

# 3. Construct shift values: [0, 16, 4, 20, 8, 24, 12, 28] — bit offset of each nibble
reverse_awq_order_tensor = (
    (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
).reshape(8)                                    # = [0, 4, 1, 5, 2, 6, 3, 7]
shifts = reverse_awq_order_tensor * 4           # = [0, 16, 4, 20, 8, 24, 12, 28]

# 4. Right-shift to extract each nibble
iweights = (iweights >> shifts) & 0xF
```

Each of the 8 identical copies of the same `int32` is right-shifted by a different amount, then masked with `& 0xF` to extract the low 4 bits:

```
int32 value: 0x76543210

Position 0: >> 0  → nibble 0 → weight 0    (AWQ_ORDER[0] = 0)
Position 1: >> 16 → nibble 4 → weight 1    (AWQ_ORDER[1] = 4)
Position 2: >> 4  → nibble 1 → weight 2    (AWQ_ORDER[2] = 1)
...
Result: 8 weights arranged in the correct order
```

**Why does AWQ use this odd order?** The repetition pattern produced by three `interleave` calls, combined with the shift sequence `[0,16,4,20,8,24,12,28]`, makes the eight nibbles come out in exactly the desired order. AWQ’s packing order is the **inverse** of the unpacking shift pattern. Packing happens only once offline, but unpacking happens at every inference step — so the complexity is pushed onto the packing side.

**Zero-points and scaling factors**:

```python
# zeros (K/G, N/8) are unpacked in exactly the same way as the weights
zeros = tl.load(zeros_ptr + zero_offsets)
zeros = tl.interleave(zeros, zeros)   # ×3 → unpack
zeros = (zeros >> shifts) & 0xF

# scales (K/G, N) are already elementwise and do not need unpacking
scales = tl.load(scales_ptr + scale_offsets)

# Dequantization
iweights = (iweights - zeros) * scales    # int32 → float → float16
```

Zero-points share one row every `group_size` rows, accessed through `pid_y * BLOCK_SIZE_Y // group_size` to index the corresponding group.

#### 9.3.2 Kernel 2: `awq_gemm_kernel` — Fused Dequantization + GEMM

This is the concrete realization of the "fused kernel" idea from §9.2: $C_{M \times N} = A_{M \times K} \times \text{dequant}(B_{K \times N})$.

**Grid design**:

```python
pid = tl.program_id(axis=0)         # linear CTA id over the M×N dimensions
pid_z = tl.program_id(1)            # Split-K dimension

pid_m = pid // num_pid_n            # row-direction tile index
pid_n = pid % num_pid_n             # column-direction tile index
```

```
Output matrix C (M × N):                K-dimension split:
     pid_n=0  pid_n=1  pid_n=2          pid_z=0 handles K[0:BK]
    ┌──────┬──────┬──────┐              pid_z=1 handles K[BK:2BK]
 m=0 │pid=0 │pid=1 │pid=2 │              ...
    ├──────┼──────┼──────┤              finally reduce with result.sum(0)
 m=1 │pid=3 │pid=4 │pid=5 │
    └──────┴──────┴──────┘
```

**Main loop — unpacking while computing**:

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
    # Load A: (BM, BK) float16
    a = tl.load(a_ptrs, mask=masks_a)

    # Load B: (BK, BN/8) int32 → interleave ×3 → (BK, BN) int32
    b = tl.load(b_ptrs, mask=masks_b)
    b = tl.interleave(b, b)
    b = tl.interleave(b, b)
    b = tl.interleave(b, b)

    # Load the zeros and scales corresponding to the current K position (indexed by group_size)
    zeros = tl.load(zeros_ptrs)     # (1, BN/8) → interleave ×3 → broadcast → (BK, BN)
    scales = tl.load(scales_ptrs)   # (1, BN) → broadcast → (BK, BN)

    # Fused unpacking + dequantization
    b = (b >> shifts) & 0xF
    zeros = (zeros >> shifts) & 0xF
    b = (b - zeros) * scales        # → float16

    # Matrix multiply-accumulate
    accumulator = tl.dot(a, b, accumulator)

    # Advance K pointers (stride = BK × SPLIT_K, skipping stripes handled by other pid_z)
    a_ptrs += BLOCK_SIZE_K * SPLIT_K
    b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)
```

Data flow: `b_int32 (BK, BN/8) → interleave ×3 → shift & mask → (b-zeros)*scales → dot(a, b) → accumulate`

**Role of Split-K**: when $M$ is very small (e.g., batch=1 in decode), there are too few tiles along the M dimension and many SMs go idle. Split-K parallelizes over the K dimension as well, assigning it to multiple CTAs in parallel, and the Python side later reduces with `result.sum(0)`. This adds a reduction step, but in small-batch settings the benefit of higher parallelism is much larger than the reduction overhead.

**Write-back**: each `pid_z` writes to the `pid_z`-th slice of the output tensor (`c_ptr + pid_z * N * M`).

#### 9.3.3 Performance Analysis

**Dequantize Kernel** (measured on A5000):

| K | N | Group Size | Latency | Bandwidth Utilization |
|---|---|---|---|---|
| 4096 | 4096 | 128 | 0.064 ms | 85% |
| 4096 | 11008 | 128 | 0.169 ms | 87% |
| 8192 | 8192 | 128 | 0.250 ms | 88% |

With a theoretical peak of 768 GB/s on the A5000, the kernel reaches ~88% utilization, close to the memory-bound limit.

**Fused GEMM vs staged pipeline** (A5000):

| M | K×N | Split-K | cuBLAS (FP16) | deq+mm | fused | fused vs deq+mm |
|---|---|---|---|---|---|---|
| 1 | 4096×4096 | 4 | 0.052 ms | 0.117 ms | 0.052 ms | **2.25×** |
| 16 | 4096×4096 | 1 | 0.058 ms | 0.123 ms | 0.063 ms | **1.97×** |
| 32 | 4096×11008 | 1 | 0.135 ms | 0.305 ms | 0.147 ms | **2.08×** |
| 64 | 4096×4096 | 1 | 0.056 ms | 0.123 ms | 0.110 ms | 1.12× |
| 128 | 4096×4096 | 1 | 0.065 ms | 0.129 ms | 0.208 ms | 0.62× |

**Key observations**:
- **M≤32 (decode stage)**: fused is **1.8-2.3×** faster than deq+mm. In memory-bound scenarios, reading packed 4-bit data saves 4× bandwidth compared with reading expanded 16-bit data, and it also eliminates intermediate tensor allocation.
- **M=1 + Split-K=4**: performance reaches **parity** with cuBLAS (using pre-decompressed FP16), while using only one quarter of the memory.
- **M≥64 (prefill stage)**: fused becomes slower — the workload turns compute-bound, and cuBLAS achieves much higher Tensor Core utilization than the overhead of unpacking on the fly.
- **Conclusion**: the sweet spot of AWQ fused GEMM is **small-batch decode (M≤32)**, which is exactly the most common setting for autoregressive LLM inference.

#### 9.3.4 Design Summary

| Design Choice | Reason |
|---|---|
| Pack 4-bit values into `int32` | Reduces memory footprint and bandwidth demand by 8× |
| Asymmetric quantization (with zero-point) | 4-bit has only 16 levels, and the zero-point avoids wasting precision |
| AWQ’s special nibble order | Works with the `tl.interleave` hardware instruction for zero-overhead unpacking |
| Per-group scales/zeros | Higher accuracy than per-tensor, smaller storage than per-channel |
| Fused dequant + GEMM | Avoids allocating and reading/writing an intermediate FP16 weight tensor |
| Split-K | Increases parallelism along K at small batch sizes and improves SM utilization |
| FP16 accumulator | Trades some precision for speed (vs standard FP32 accumulation) |

### 9.4 SageAttention — INT8 Quantized Attention

> The source code comes from the SageAttention project (`quant_per_block.py` + `attn_qk_int8_per_block.py`). While §9.3 showed kernel fusion for weight quantization, this section shows another fusion direction: **quantizing the QK^T matrix multiply inside attention to INT8**, accelerating it within the FlashAttention framework.

**Core idea**: quantize Q and K block-wise into INT8, use INT8 Tensor Cores (about 2× the throughput of FP16) to accelerate $QK^T$, and keep V in FP16 without quantization.

```
Overall flow:
Q, K (FP16) ──per_block_int8()──► Q_int8, K_int8, Q_scale, K_scale ──forward()──► O (FP16)
```

#### 9.4.1 Per-Block INT8 Quantization

**Quantization kernel `quant_per_block_int8_kernel`**: each CTA processes one `[BLK, head_dim]` submatrix.

```python
# Coordinates of each CTA
off_blk = tl.program_id(0)    # block index along the sequence dimension
off_h = tl.program_id(1)      # head index
off_b = tl.program_id(2)      # batch index

# Load data of shape [BLK, head_dim]
x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
x = x.to(tl.float32)
x *= sm_scale                  # multiply by 1/√d × log₂(e) on the Q side; on the K side sm_scale=1.0

# Symmetric quantization: the whole block shares one scale
scale = tl.max(tl.abs(x)) / 127.

# Manual rounding (Triton's to(int8) truncates)
x_int8 = x / scale
x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)   # +0.5 then truncate = round
x_int8 = x_int8.to(tl.int8)

tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
tl.store(scale_ptrs, scale)    # one FP32 scale per block
```

**The trick of folding `sm_scale` into quantization**: during Q quantization, pre-multiply by `sm_scale × log₂(e) = 1/√d × 1.44269504`. Then after dequantization, $Q_\text{int8} \cdot K_\text{int8}^T \times q_\text{scale} \times k_\text{scale}$ directly yields $QK^T / \sqrt{d} \times \log_2 e$, allowing the use of hardware-faster `exp2` instead of `exp`:

$$e^{x/\sqrt{d}} = 2^{x/\sqrt{d} \cdot \log_2 e}$$

**Comparison with AWQ quantization**:

| Dimension | SageAttention (Q/K quantization) | AWQ (weight quantization) |
|---|---|---|
| Quantized object | Runtime activations (Q, K) | Offline weights |
| Quantization scheme | Symmetric (no zero-point) | Asymmetric (with zero-point) |
| Granularity | Per-block (128/64 tokens × head_dim) | Per-group (128 weights) |
| Unquantized part | V (kept in FP16) | Activations (kept in FP16) |

After LayerNorm, Q/K distributions are approximately symmetric, so symmetric quantization is sufficient; AWQ weight distributions are not necessarily symmetric and require a zero-point.

#### 9.4.2 INT8 Attention Kernel

This kernel adopts FlashAttention’s Q-stationary tiling strategy: the outer loop fixes one Q block, while the inner loop traverses all K/V blocks.

**Core computation — INT8 matrix multiply + dequantization**:

```python
# q: [BLOCK_M, HEAD_DIM] int8     k: [HEAD_DIM, BLOCK_N] int8 (already transposed)
qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
```

This single line does the following:
1. `tl.dot(q, k)`: INT8 Tensor Core matrix multiplication, producing INT32 output
2. `.to(tl.float32)`: cast to FP32
3. `× (q_scale × k_scale)`: dequantize back to the true attention scores (already including $1/\sqrt{d} \times \log_2 e$)

**Online softmax (the FlashAttention algorithm)**:

```python
# Initial state: m_i = -inf, l_i = 1.0, acc = 0

for each K/V block:
    # 1. Load K block → INT8 matmul → dequantization
    qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

    # 2. Update row maxima
    m_ij = tl.maximum(m_i, tl.max(qk, 1))

    # 3. Rescale historical accumulation (needed when the maximum changes)
    alpha = tl.math.exp2(m_i - m_ij)       # correction factor
    l_i = l_i * alpha + tl.sum(tl.math.exp2(qk - m_ij[:, None]), 1)
    acc = acc * alpha[:, None]

    # 4. P @ V (FP16 Tensor Core, not INT8)
    p = tl.math.exp2(qk - m_ij[:, None])
    v = tl.load(V_ptrs)                     # V stays in FP16
    acc += tl.dot(p.to(tl.float16), v)

    m_i = m_ij

# Final normalization
output = acc / l_i[:, None]
```

The essence of online softmax is that when the running maximum changes from $m_\text{old}$ to $m_\text{new}$, the historical accumulation must be rescaled by the correction factor $\alpha = 2^{m_\text{old} - m_\text{new}}$. If the maximum does not change, $\alpha = 1$; if the maximum increases, then $\alpha < 1$, shrinking the historical values.

**GQA support**: the head index for K/V is computed as `off_h // num_kv_groups`, so multiple Q heads naturally share one KV head.

**Mask optimization**:

```python
if mask.dtype </mark> tl.int1:           # Bool mask
    if tl.max(mask_block) == 0:     # the entire block is masked
        skip = True                  # skip all computation!
    else:
        qk += tl.where(mask_block, 0, -1.0e6)
else:                                # Float mask (additive)
    qk += mask_block
```

Bool masks support block-level skipping — when an entire `[BLOCK_M, BLOCK_N]` tile is masked, all computation for that tile is skipped directly, giving substantial speedup for sparse masks.

#### 9.4.3 Design Summary

| Design Choice | Reason |
|---|---|
| Quantize Q/K to INT8 while keeping V in FP16 | $QK^T$ is smoothed by softmax and is less sensitive to quantization error; V error directly affects the output |
| Per-block quantization (BLKQ=128, BLKK=64) | Aligned with FlashAttention tile sizes, with an independent scale for each block |
| Symmetric quantization (no zero-point) | Q/K are approximately symmetric after LayerNorm, so a zero-point is not very useful |
| Fold `sm_scale` into Q quantization | Avoids extra floating-point multiplication inside the attention kernel |
| Use `exp2` instead of `exp` | Hardware natively supports `exp2` more efficiently, paired with pre-multiplication by `log₂(e)` |
| Bool-mask block-level skipping | If a whole block is masked, skip all computation; very effective for sparse masks |
| Use FP16 for P@V instead of INT8 | Balances accuracy and speed, applying INT8 acceleration only in the $QK^T$ phase |
