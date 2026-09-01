import os

with open('notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.en.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace header overview
old_overview = """This note systematically covers 2 foundational pillars of modern LLM systems:
1. **Why Decoder-Only Won & Dense Text Embeddings: Architectural convergence analysis and how modern research overcame Anisotropy and representation collapse**
2. **Hardware Numerical Precision Formats (FP32 / FP16 / BF16 / FP8 / NVFP4) & Mixed-Precision Training (Master Weights, Loss Scaling, and Stochastic Rounding)**"""

new_overview = """This note systematically covers 3 foundational pillars of modern LLM systems:
1. **Why Decoder-Only Won & Dense Text Embeddings: Architectural convergence analysis and how modern research overcame Anisotropy and representation collapse**
2. **Hardware Numerical Precision Formats (FP32 / FP16 / BF16 / FP8 / NVFP4) & Mixed-Precision Training (Master Weights, Loss Scaling, and Stochastic Rounding)**
3. **Mathematical Derivation of Initial Loss & Step-0 Sanity Check (ln(V) Proof, Temperature Scaling, Label Smoothing, Tokenizer Effects & Pre-training Diagnostics)**"""

content = content.replace(old_overview, new_overview)

# Replace Module 3 & 4
target_tail = """## Module 3: High-Yield Interview Rapid-Fire

### Q1: Why is BERT's masked bidirectional architecture unsuitable for modern foundation LLMs?
> **Answer**:
> 1. Masked Language Modeling (MLM) breaks the causal autoregressive chain, preventing the model from naturally generating variable-length text sequences;
> 2. BERT lacks a causal KV cache. Generating each new token requires a full $O(S^2)$ forward pass over the entire sequence, making open-ended generation computationally prohibitive.

### Q2: When an LLM training run encounters NaN loss, what is the systematic debugging sequence?
> **Answer**:
> 1. Check for gradient overflow under FP16 and verify dynamic loss scale values;
> 2. Ensure the $\\frac{1}{\\sqrt{d_k}}$ attention scale factor is active to prevent softmax exponent overflow;
> 3. Verify that normalization layer $\\epsilon$ values (LayerNorm/RMSNorm) are sufficient ($10^{-5} \\sim 10^{-6}$) to avoid division by zero;
> 4. Audit dataset batches for corrupted all-zero inputs or extreme length anomalies."""

new_tail = """## Module 3: Next-Token Loss at Random Initialization & Step-0 Sanity Check (Initial Loss & ln(V) Derivation)

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

$$\mathcal{L} = -\ln P(y = c \mid x_{<t}) = -\ln \left( \frac{e^{z_c}}{\sum_{v=1}^V e^{z_v}} \right) = -z_c + \ln \left( \sum_{v=1}^V e^{z_v} \right)$$

#### (1) Uniform Probability Distribution Under Random Initialization
Under standard zero-mean weight initializations (e.g. Gaussian $\mathcal{N}(0, \sigma^2)$ with $\sigma \approx 0.02$ or Xavier/He initialization):
- All logits $z_v$ have expectation $\mathbb{E}[z_v] \approx 0$ with small variance;
- The softmax probabilities approximate a uniform categorical distribution:
  $$P(y = v) = \frac{e^{z_v}}{\sum_{j=1}^V e^{z_j}} \approx \frac{1}{V}$$
- The cross-entropy loss simplifies strictly to the natural log of vocabulary size:
  $$\mathcal{L}_{\text{init}} = -\ln \left(\frac{1}{V}\right) = \ln V$$

#### (2) Second-Order Taylor Expansion with Non-Zero Logit Variance
If initial logits $z_v \sim \text{i.i.d. } \mathcal{N}(0, \sigma_z^2)$, expanding the Log-Sum-Exp function yields:

$$\mathbb{E}[\mathcal{L}_{\text{init}}] = -\mathbb{E}[z_c] + \mathbb{E}\left[\ln \sum_{v=1}^V e^{z_v}\right] \approx \ln V + \frac{\sigma_z^2}{2} + \mathcal{O}(\sigma_z^4)$$

For well-calibrated initializations ($\sigma_z \le 0.1$), $\frac{\sigma_z^2}{2} < 0.005$, making $\mathcal{L}_{\text{init}} \approx \ln V$ extremely accurate.

---

### 2. Theoretical Initial Loss Across Frontier Foundation Models

| Foundation Model | Vocabulary Size $V$ | Theoretical Step-0 Loss: $\ln V$ (nats/token) |
|---|---|---|
| **GPT-2 / GPT-3** | $50,257$ | $\ln(50257) \approx \mathbf{10.825}$ |
| **LLaMA-1 / LLaMA-2 / Mistral-7B** | $32,000$ | $\ln(32000) \approx \mathbf{10.373}$ |
| **LLaMA-3 / LLaMA-3.1 / LLaMA-3.3** | $128,256$ | $\ln(128256) \approx \mathbf{11.762}$ |
| **DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1** | $129,280$ | $\ln(129280) \approx \mathbf{11.770}$ |
| **Qwen-2 / Qwen-2.5** | $152,064$ | $\ln(152064) \approx \mathbf{11.932}$ |
| **Gemma / Gemma-2** | $256,000$ | $\ln(256000) \approx \mathbf{12.453}$ |

---

### 3. Theoretical Impact of Temperature and Label Smoothing

#### (1) Temperature Scaling $T$
When temperature $T$ is applied to logits ($z / T$):
$$\mathcal{L}(T) = -\frac{z_c}{T} + \ln \left( \sum_{v=1}^V e^{z_v / T} \right)$$
- As $T \to \infty$: Logit variance vanishes, driving probabilities strictly uniform and $\mathcal{L} \to \ln V$.
- For $T < 1.0$: Amplifies logit fluctuations, slightly increasing initial loss variance ($\mathbb{E}[\mathcal{L}] \approx \ln V + \frac{\sigma_z^2}{2 T^2}$).

#### (2) Label Smoothing $\epsilon$
With label smoothing target distribution $q(v) = (1-\epsilon)\mathbb{I}(v=c) + \frac{\epsilon}{V}$:
$$\mathcal{L}_{\text{smooth}} = -(1-\epsilon)\ln P(c) - \frac{\epsilon}{V}\sum_{v=1}^V \ln P(v)$$
Substituting uniform random initialization ($P(v) \approx \frac{1}{V}$):
$$\mathcal{L}_{\text{smooth}} = -(1-\epsilon)\ln \left(\frac{1}{V}\right) - \frac{\epsilon}{V} \cdot V \ln \left(\frac{1}{V}\right) = \ln V$$
> 💡 **Core Insight**: **Label smoothing does NOT alter the expected initial loss at random initialization—it remains strictly equal to $\ln V$!**

---

### 4. Tokenizer Effects & Vocabulary Padding

1. **Per-Token Loss vs. Sequence-Level Compression**:
   - Expanding vocabulary from 32K (LLaMA-2) to 128K (LLaMA-3) increases per-token initial loss from 10.37 to 11.76 ($\Delta \approx +1.39$ nats/token);
   - **However**: Larger vocabularies compress text more efficiently (requiring $\sim 30\%$ fewer tokens for identical text), so total document-level cross-entropy loss is actually lower.
2. **Vocabulary Alignment Padding (GPU Tensor Core Alignment)**:
   - In production frameworks (Megatron-LM / HuggingFace), vocabularies are often padded to multiples of 64 or 128 (e.g., $V_{\text{embed}} = 32,256$ for $V_{\text{actual}} = 32,000$ to optimize memory bus alignment);
   - **Note**: The expected initial loss will evaluate to $\ln(V_{\text{embed}})$, matching the actual projection head dimension.

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
> 2. Ensure the $\\frac{1}{\\sqrt{d_k}}$ attention scale factor is active to prevent softmax exponent overflow;
> 3. Verify that normalization layer $\\epsilon$ values (LayerNorm/RMSNorm) are sufficient ($10^{-5} \\sim 10^{-6}$) to avoid division by zero;
> 4. Audit dataset batches for corrupted all-zero inputs or extreme length anomalies.

### Q3: Why must the Step-0 loss of an autoregressive language model equal $\\ln V$? What does it mean if $\\mathcal{L}_0 = 2.0$?
> **Answer**:
> 1. **Mechanism**: At random initialization with zero-mean weights, the model assigns an approximately uniform probability $P(y=c) = \\frac{1}{V}$ across all $V$ candidate tokens. The cross-entropy loss is therefore $-\\ln(1/V) = \\ln V$.
> 2. **Loss = 2.0 Diagnosis**: Indicates a **severe implementation bug or bidirectional data leakage**:
>    - **Cause 1 (Broken Causal Mask)**: The attention mask failed to enforce causal lower-triangular masking, allowing self-attention to attend directly to future target tokens.
>    - **Cause 2 (Unmasked Padding)**: Target batch padding tokens lacked `ignore_index=-100`, allowing the model to trivially predict constant `[PAD]` tokens.
>    - **Cause 3 (Unshifted Labels)**: The target labels were not shifted by one position relative to inputs (`labels = input_ids`), causing the model to simply copy the current input token."""

content = content.replace(target_tail, new_tail)

with open('notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.en.md', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully updated MLCoding00B English note")
