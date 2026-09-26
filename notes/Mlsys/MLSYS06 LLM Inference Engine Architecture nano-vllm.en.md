# 06 · LLM Inference Engine Architecture: Deep Dive into nano-vllm

As an industry benchmark for modern large language model serving, vLLM redefined high-concurrency, high-throughput LLM serving paradigms. This chapter conducts a comprehensive line-by-line deconstruction of `nano-vllm`, connecting the model architecture layer (GQA, RoPE, KV Projection) with the core serving engine (LLMEngine, Sequence state machine, preemptive Scheduler, Paged KV Cache BlockManager, ModelRunner, and Worker execution loops).

---

## Part 1: Model Architecture & Operator View (GQA, KV Projections, and RoPE)

## nano-vllm part 1

**nano-vllm** is a minimal LLM inference engine implemented in ~1000 lines of pure Python/PyTorch. The project goal is to restore the core design of vLLM with the least code to facilitate learning and research.

| Features | Description |
|------|------|
| Model support | Qwen3 series (expandable) |
| Inference Optimization | Paged KV Cache, Continuous Batching |
| Parallel strategy | Tensor Parallelism (column parallelism + row parallelism) |
| Code size | ~1000 lines, no complex dependencies |

### Revisit transformers

Transformer is an architecture proposed by Google in the paper *"Attention Is All You Need"* in 2017, and has become the basis of modern large language models (such as GPT, LLaMA, Claude). Core innovations: **Self-Attention** (direct attention to any position in the sequence), **parallel computing** (without the sequential dependence of RNN), **position coding**.

#### Transformer Decoder layer structure

```
Input [B,T,D]
  │
  ├─ RMSNorm → MHA (Q/K/V/O proj, Params: 4D²) ─┐
  └────────────────────────────── (+) residual ←┘
  │
  ├─ RMSNorm → FFN (Up→Act→Down, Params: 3DF) ──┐
  └────────────────────────────── (+) residual ←┘
  │
Output [B,T,D]
```


MHA allows the model to learn information from different "representation subspaces" simultaneously, three core vectors: **Q (what am I looking for)**, **K (what labels do I have)**, **V (what is my content)**.

#### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

Calculation: ① $QK^T$ gets the similarity matrix → ② Divide by $\sqrt{d_k}$ to prevent gradient disappearance → ③ softmax → ④ Find V with weighting. MHA divides the input into multiple heads and learns different modes in parallel: $\text{MultiHead} = \text{Concat}(\text{head}_i)W^O$, where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

Three variants: **MHA** ($N=K$, standard), **MQA** ($K=1$, extremely economical KV cache), **GQA** ($K$ is evenly divided by $N$, compromise, used by Qwen3).



### Analyze and understand the roofline properties of transformer

Notation: $x, y \in [P]$ (vector), $A \in [N, P]$, $B \in [P, M]$ (matrix).

#### FLOPs calculation rules

| Operations | FLOPs | Data Transfer | Description |
|------|-------|----------|------|
| Vector dot product $x \cdot y$ | $2P$ | $2P$ | P multiplications + P additions |
| Matrix-vector multiplication $Ax$ | $2NP$ | $NP + P$ | N dot products |
| Matrix-matrix multiplication $AB$ | $2NPM$ | $NP + PM$ | M matrix-vector multiplication |

Matrix multiplication: calculation $O(N^3)$, data transmission $O(N^2)$. The larger the matrix, the easier it is to achieve compute-bound. This is why deep learning uses matrix multiplication extensively.

#### General Einsum Rules

For high-dimensional tensor shrinkage, we need to distinguish three types of dimensions:
- ==Red== **Shrink Dimension**: appears in both tensors and disappears in the output (summed away)
- ==Blue== **Batch Dimensions**: present in both tensors and outputs (computed independently in parallel)
- **Normal dimensions**: only appear in one input tensor and output

**Example analysis**:
$$
C[\textcolor{blue}{GH}IJ\textcolor{red}{KL}] \cdot D[\textcolor{blue}{GH}MN\textcolor{red}{KL}]  ightarrow E[\textcolor{blue}{GH}IJMN]
$$

Use einsum to express: `einsum('ghijkl,ghmnkl->ghijmn', C, D)`

```
Tensor C: [G, H, I, J, K, L]
        ↑  ↑  ↑  ↑  ↑  ↑
        B  B  N  N  R  R
        A  A  O  O  E  E

Tensor D: [G, H, M, N, K, L]
        ↑  ↑  ↑  ↑  ↑  ↑
        B  B  N  N  R  R
        A  A  O  O  E  E

Output E: [G, H, I, J, M, N]  ← K, L are summed away
```

**Understanding of calculation process**:
- For each position of the output `E[g,h,i,j,m,n]`, it is necessary to calculate:
  $$E[g,h,i,j,m,n] = \sum_{k,l} C[g,h,i,j,k,l] \times D[g,h,m,n,k,l]$$
- This is a $K \times L$ multiplication and addition

**FLOPs calculation**:
$$
\text{FLOPs} = 2 \times \textcolor{blue}{G} \times \textcolor{blue}{H} \times I \times J \times M \times N \times \textcolor{red}{K} \times \textcolor{red}{L}
$$

- Factor 2: 1 multiplication + 1 addition required per element
- all dimensions are multiplied, but each dimension is counted only once (regardless of how many tensors it appears in)

**Memory**: FLOPs = 2 × product of all dimensions (including shrinking dimensions)

---

### FLOPs of forward and backward propagation

Let $A[N,P]$, $B[P,M]$, $C=AB$:

| Phases | Operations | FLOPs |
|------|------|-------|
| Forward propagation | $C = AB$ | $2NPM$ |
| Backpropagation (weight gradient) | $\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$ | $2NPM$ |
| Backpropagation (input gradient) | $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T$ | $2NPM$ |
| **Total** | - | ==$6NPM$== |

**Inference** = $2NPM$ (forward only); **Training** = $6NPM$ (forward + two reverses), training is 3 times faster than inference.

### Symbol definition

$B$=batch, $T/S$=sequence length (Q/KV), $D$=model dim, $F$=FFN dim, $N$=Q heads, $K$=KV heads, $H$=head dim, $L$=number of layers, $V$=vocabulary size.

---

### Calculation of MLP layer

Modern Transformer uses **Gated MLP** (SwiGLU): $W_{out} \cdot [\sigma(W_{in1} x) \odot W_{in2} x]$, which has 1 more matrix (parameters +50%) than the traditional 2 matrices, and has stronger expressive power. Common variants: GLU (sigmoid), GEGLU (GELU), SwiGLU (SiLU, used by LLaMA/Qwen3).

| Operations | Training FLOPs | Number of parameters |
|------|------------|--------|
| $A[B,T,\textcolor{red}{D}] \cdot W_{in1}[\textcolor{red}{D}, F]$ | $6BTDF$ | $DF$ |
| $A[B,T,\textcolor{red}{D}] \cdot W_{in2}[\textcolor{red}{D}, F]$ | $6BTDF$ | $DF$ |
| $\sigma(A_{in1}) \odot A_{in2}$ (activation+gating) | $O(BTF)$ can be ignored | - |
| $A[B,T,\textcolor{red}{F}] \cdot W_{out}[\textcolor{red}{F}, D]$ | $6BTDF$ | $DF$ |
| **MLP TOTAL** | ==$\approx 18BTDF$== | ==$3DF$== |

*Traditional MLP without Gating: 2 matrices, parameter size $2DF$. Modern models (LLaMA, DeepSeek, etc.) all use Gating variants. *

---

### Calculation of Attention layer

#### QKVO projection matrix

| Operations | Training FLOPs | Number of parameters |
|------|------------|--------|
| $A[B,T,\textcolor{red}{D}] \cdot W_Q[\textcolor{red}{D}, N, H]$ | $6BTDNH$ | $DNH$ |
| $A[B,T,\textcolor{red}{D}] \cdot W_K[\textcolor{red}{D}, K, H]$ | $6BTDKH$ | $DKH$ |
| $A[B,T,\textcolor{red}{D}] \cdot W_V[\textcolor{red}{D}, K, H]$ | $6BTDKH$ | $DKH$ |
| $A[B,T,\textcolor{red}{N},\textcolor{red}{H}] \cdot W_O[\textcolor{red}{N},\textcolor{red}{H}, D]$ | $6BTDNH$ | $DNH$ |
| **QKVO TOTAL** | ==$12BTD(N+K)H$== | ==$2D(N+K)H$== |

#### Dot-Product Attention

| Operations | Training FLOPs |
|------|------------|
| $Q[\textcolor{blue}{B}, T, \textcolor{blue}{K}, G, \textcolor{red}{H}] \cdot K[\textcolor{blue}{B}, S, \textcolor{blue}{K}, \textcolor{red}{H}]^T  ightarrow S[B,T,S,N]$ | $6BTSNH$ |
| $\text{softmax}_S(S)  ightarrow P$ | $O(BTSN)$ can be ignored |
| $P[\textcolor{blue}{B}, T, \textcolor{red}{S}, \textcolor{blue}{K}, G] \cdot V[\textcolor{blue}{B}, \textcolor{red}{S}, \textcolor{blue}{K}, H]  ightarrow O[B,T,N,H]$ | $6BTSNH$ |
| **Attention total** (self-attention: S=T) | ==$\approx 12BT^2NH$== |

*Note: Decoder-only causal attention only counts the lower triangle, and the actual FLOPs are halved, but it requires dedicated kernels such as Flash Attention to utilize it. *

---

### 6ND Rule

Ignoring dot-product attention (reasonable for short context), FLOPs of the entire model:

$$
\boxed{\text{Total FLOPs} = 6 \times \text{num\_tokens} \times \text{num\_parameters}}
$$

The FLOPs projected by MLP and QKVO are both **6 × BT × parameter amount** (MLP: $18BTDF = 6BT \cdot 3DF$; QKVO: $12BTD(N+K)H = 6BT \cdot 2D(N+K)H$), so after ignoring attention:

$$
\text{FLOPs} = 6 \times BT \times \underbrace{(3DF + 2D(N+K)H) \times L}_{\text{Total number of parameters}} = 6 \times N_{\text{tokens}} \times N_{\text{params}}
$$

Coefficient 6 = 2 (multiply and add) × 3 (forward + weight gradient + input gradient), each parameter is "used" 6 times during training.

> [!example]Use the 6ND rule
> This rule makes estimating training costs very simple:
> 
> ```
> Training FLOPs ≈ 6 × number of parameters × number of training tokens
> ```
> 
> **Example**: Train a 70B parameter model, using 2T tokens:
> ```
> FLOPs = 6 × 70×10⁹ × 2×10¹² = 8.4×10²³ FLOPs
> ```

#### Attention vs MLP: When does attention start to dominate?

Typical configuration: $F = 4D$, $D = NH$, $N = K$

$$
\frac{\text{Attention FLOPs}}{\text{Matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{T}{8D}
$$

> [!important]Key conclusions
> FLOPs of Dot-product attention only start to dominate when ==$T > 8D$==.
> 
> For a model with $D = 8192$, this means **~65K tokens**.
> 
> ==For large models, the quadratic complexity of attention is actually not that terrible! ==



### Advanced topics

#### Mixture of Experts (MoE)

MoE replaces a single dense MLP with multiple independent "expert" MLPs, and dynamically selects which experts to activate through the router.

**Core idea**: Increase the model capacity (amount of parameters) without increasing the amount of calculation disproportionately.

```
Dense:  x → MLP → y

MoE:    x → Router (select top-k) → E_1, E_2, ..., E_k → weighted sum → y
```

**Vanilla MoE formula (the specific design in LLM is different)**:

1. **Router calculation** (select which experts):
$$
G(x) = \text{softmax}(\text{TopK}(x \cdot W_r))
$$
Among them, $W_r \in \mathbb{R}^{D \times E}$ is the router weight, TopK retains the top k maximum values, and the rest is $-\infty$.

2. **MoE output** (weighted combination):
$$
\text{MoE}(x) = \sum_{i=1}^{E} G(x)_i \cdot \text{Expert}_i(x)
$$
Due to TopK, only k experts are actually activated (the rest $G(x)_i = 0$).

3. **Load balancing loss** (to prevent uneven use of experts):
$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i
$$
Where $f_i$ is the proportion of tokens assigned to expert $i$, and $p_i$ is the average probability of router to expert $i$.

#### MoE Parameters

| Parameters | Description | Example (DeepSeek v3) |
|------|------|-------------------|
| **E** | Expert quantity | 256 |
| **k** | Number of experts activated for each token | 8 |
| **Sparsity** | $E / k$ | 32 |

#### Computing features

- The total number of parameters increases $O(E)$ times
- The activation parameters are only increased by $k$ times per token
- Introduce AllToAll communication overhead

To achieve compute-bound, $B > 120E/k$ is required (about 3840 when DeepSeek E=256, k=8), which is a quite large batch size during inference.

---

### Gradient Checkpointing

Saving intermediate activations avoids $O(L^2)$ recalculation, but the memory overhead is huge:

> [!warning]Activate memory example
> For $BT = 4M$ tokens, $L = 64$ layer, $D = 8192$:
> ```
> Activate memory ≈ 2 × 20 × BT × D × L = 84 TB (bf16)!
> ```

#### Checkpoint strategy

| Strategy | Saving Content | FLOPs Overhead |
|------|----------|------------|
| **Block Remat** | Input per layer (1 checkpoint/layer) | Increased from $6ND$ to $8ND$ |
| **Big Matmuls Only** | Output of big matrix multiplication (7/layer) | Avoid recomputing big matrix multiplication |

---

### KV Cache

Two stages of LLM inference:

1. **Prefill**: Process prompt and save K/V to cache
2. **Decode**: Generate token by token and reuse KV cache

$$
\text{KV Cache Size} = 2 \times S \times L \times K \times H
$$

> [!example]KV Cache size example
> 8K context, 64 layers, $KH = D = 8192$:
> ```
> KV Cache = 2 × 8192 × 64 × 8192 = 8 GiB (int8)
> ```
> ==This is why GQA ($K \ll N$) is so important! ==


### Typical model configuration reference

| Parameters | 7B | 13B | 70B |
|------|-----|------|------|
| D (model dim) | 4096 | 5120 | 8192 |
| L (layers) | 32 | 40 | 80 |
| N (heads) | 32 | 40 | 64 |
| F (FFN dim) | 11008 | 13824 | 28672 |
| KV Cache/token (int8) | ~256KB | ~400KB | ~1.3MB |

## Back to nano-vllm

> This part mainly talks about the design of each **module** (submodules under `models/qwen3.py`, `layers/`), including the implementation details of attention, linear, embedding, sampler, etc. The next article (Part 2) will talk about **system design**: KV cache management, scheduler, continuous batching and other inference engine cores.

Next, go back to nano-vllm and start with models/qwen3.py

The core of this document is the following Attention design:

### 1. Grouped Query Attention (GQA)

`num_heads (Q) >> num_kv_heads (K/V)`: The number of Q heads is much more than the number of K/V heads (for example, 32Q heads vs 8KV heads). The K/V weight is shared among multiple Q heads, greatly reducing the KV Cache memory.

```python
# qwen3.py:30-38
self.num_heads    = self.total_num_heads    // tp_size  # Q heads per GPU
self.num_kv_heads = self.total_num_kv_heads // tp_size  # KV heads per GPU
self.q_size  = self.num_heads    * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim        # kv_size << q_size
```

### 2. QK Norm (RMSNorm on Q and K)

The unique design of Qwen3: do RMSNorm (per head_dim) on Q and K once before RoPE:

```python
# qwen3.py:81-83
if not self.qkv_bias:
    q = self.q_norm(q)  # RMSNorm on each head's q
    k = self.k_norm(k)  # RMSNorm on each head's k
```

**Why do you need QK Norm? **

- Prevent attention logit ($QK^T / \sqrt{d}$) value from exploding, making training more stable
- Replaced the traditional `qkv_bias` (enabled when `qkv_bias=False`)
- Differences from models such as Llama: Qwen3 performs norm on each head separately, with finer granularity

![[assets/Pasted image 20260227222651.png]]

### 3. SwiGLU activation function

**SwiGLU** is a variant of GLU (Gated Linear Unit), the formula is:

$$\text{SwiGLU}(x_1, x_2) = \text{SiLU}(x_1) \times x_2$$

Two advantages: **Gating mechanism** ($x_2$ dynamically controls the information flow in each dimension, with stronger expressive ability); **SiLU smoothing** ($x \cdot \sigma(x)$, more stable than ReLU gradient). Google experimentally proved that SwiGLU works best on language models in *GLU Variants Improve Transformer (2020)*, and has since become mainstream.

**`layers/activation.py` forward line-by-line interpretation**:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)  # split the last dimension into two halves
    return F.silu(x) * y    # apply SiLU to the first half, then multiply elementwise with the second half
```

| step | operation | result shape |
|------|------|-----------|
| Input | Linear projects hidden to 2× width | `[B, L, 2H]` |
| `x.chunk(2, -1)` | Cut the last dimension in half to get x and y | Each `[B, L, H]` |
| `F.silu(x)` | Activation: $x \cdot \sigma(x)$ | `[B, L, H]` |
| `* y` | Element-wise multiplication of gated signal y | `[B, L, H]` |

> [!note]Why does Linear project to 2H?
> SwiGLU requires two inputs. The first Linear output of FFN is set to `intermediate_size * 2`, and the chunk in forward is divided into two parts, one for activation and one for gate. This is exactly what `layers/activation.py:13` does.

### 4. LayerNorm vs RMSNorm (`layers/layernorm.py`)

| | Formula | Features |
|---|---|---|
| **LayerNorm** | $(x - \mu) / \sqrt{\sigma^2 + \varepsilon} \cdot \gamma + \beta$ | Center first and then normalize |
| **RMSNorm** | $x / \sqrt{\text{mean}(x^2) + \varepsilon} \cdot \gamma$ | Remove the mean subtraction step, faster |

RMSNorm eliminates the need for mean calculation, and experiments show that the effect is equivalent to LayerNorm, so it has become the default choice for modern LLM (LLaMA, Qwen3).

### 5. RoPE position encoding (`layers/rope.py`)

**Core idea**: Rotate each pair of dimensions $(x_1, x_2)$ in head, frequency $\theta_i = 1 / \text{base}^{2i/d}$ (base is usually 10000, Qwen uses a larger value):

$$y_1 = x_1\cos\theta - x_2\sin\theta, \quad y_2 = x_2\cos\theta + x_1\sin\theta$$

`apply_rotary_emb` directly implements the above formula without additional transformation.

**Three engineering optimizations**:

**① Precomputation cache** (in `__init__`):

```python
inv_freq = 1.0 / (base ** (arange(0, d, 2) / d))     # frequency vector
freqs    = einsum("i,j->ij", positions, inv_freq)      # outer product [max_pos, d/2]
cache    = cat(cos, sin, dim=-1).unsqueeze_(1)         # [max_pos, 1, d]
```

Calculate the cos/sin of all positions in advance and store them in the table, and gather them directly during inference to avoid repeated calculations.

**② Index by position** (in `forward`):

```python
cos_sin = self.cos_sin_cache[positions]   # direct gather, supports non-consecutive positions
cos, sin = cos_sin.chunk(2, dim=-1)
```

Support non-consecutive token positions in prefill/decode mixed scenarios.

**③ `lru_cache(1)` singleton**:

```python
@lru_cache(1)
def get_rope(...):
    assert rope_scaling is None   # only standard RoPE is supported
```

Only one RoPE instance is created globally to save video memory; `assert` explicitly does not support extensions such as YaRN/linear interpolation.

### 6. Sampling strategy (`layers/sampler.py`)

The original nano-vllm uses ordinary greedy sampling. Here we can briefly expand and discuss top-p sampling.
Sampling is divided into three steps: temperature scaling → Top-p filtering → Gumbel-max sampling.

#### Step 1 — Temperature Scaling (line 13)

```python
logits = logits.div_(temperatures.unsqueeze(1))
```

The essence of temperature is "stretching or compressing the gap between logits":

| Temperature | Effect | Usage |
|------|------|------|
| T < 1 (such as 0.3) | The gap is enlarged, and the probability of high-scoring tokens is more concentrated | The output is more certain and conservative |
| T = 1 | unchanged | model original distribution |
| T > 1 (such as 2.0) | The gap narrows and the probability tends to be uniform | The output is more diverse and creative |

*Example: logits = [10, 5, 1], T=0.5 → [20, 10, 2], the probability of the maximum value is higher after softmax. *

#### Step 2 — Top-p filtering (lines 15–20)

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = softmax(sorted_logits).cumsum(-1)
# mask tokens after cumulative probability exceeds p as -inf
to_remove = (cumulative_probs - softmax(sorted_logits)) >= top_ps
logits = logits.masked_fill(to_remove, float('-inf'))
```

```
token:   A     B     C     D     E
prob:   0.5   0.3   0.1   0.06  0.04
cumsum: 0.5   0.8   0.9   0.96  1.0
                    ↑ top_p=0.9 cuts off exactly here; D/E are masked
```

`cumsum - prob` (the cumulative probability *before* adding the current token) ensures that the token that happens to touch p is retained.

#### Step 3 — Gumbel-max sampling (lines 22–23)

```python
probs.div_(torch.empty_like(probs).exponential_(1)).argmax()
```

Randomly sample from the filtered distribution. The randomness comes from $\text{Exp}(1)$ random variables sampled independently for each token, and then $p_i / E_i$ is used to take argmax.

**Why is it equivalent to sampling by probability? **(Gumbel-Max Trick)

Sampling from a categorical distribution $p$ is equivalent to:

$$\text{argmax}\bigl(\log p_i + G_i\bigr), \quad G_i \sim \text{Gumbel}(0,1)$$

Using $G_i = -\log E_i,\ E_i \sim \text{Exp}(1)$, substitute:

$$\text{argmax}\bigl(\log p_i - \log E_i\bigr) = \text{argmax}\bigl(\log(p_i / E_i)\bigr) = \text{argmax}(p_i / E_i)$$

That is, the way it is written in the code is strictly mathematically equivalent to sampling from the original distribution.

### 7. Column parallelism and row parallelism Linear (`layers/linear.py`, `embed_head.py`)

![[assets/Pasted image 20260302111018.png]]

For linear layer $Y = XW$, TP has two cutting methods:

**Column Parallel (cut by output dimension W)**: Each card counts an output channel $Y_i = XW_i$, and the output of each card naturally does not overlap.
- Forward: No communication (each card is calculated independently, subsequent concat/all-gather requires complete Y)
- Backward: All-reduce is required when calculating $dX = \sum_i dY_i W_i^\top$

**Row Parallel (press input dimension W)**: Each card gets a piece of input $X_i$, calculates the partial sum $Y_i = X_i W_i$, and finally $Y = \sum_i Y_i$.
- Forward: requires all-reduce (add up the parts of each card)
- Backward: $dX_i = dY W_i^\top$ Each card is calculated separately, no all-reduce is required

**One sentence**: For forward communication, column parallelism is gather and row parallelism is reduce; but column parallel backward still requires all-reduce.

> [!note]vLLM is an inference system without backward
> Only run forward during inference: column parallelism has no communication at all, and row parallelism still requires all-reduce (summing the partial logit of each card). Therefore, the gate/up matrix of FFN in vLLM uses column parallelism (free), and the down matrix uses row parallelism (one all-reduce per layer).

**All Linear classes inherit from `LinearBase`**. The core difference is how the weight is divided (`weight_loader`) and whether forward communication is required:

| class | weight shape (single card) | communication | purpose |
|---|---|---|---|
| `ReplicatedLinear` | `[O, I]` (complete copy) | None | Small parameters such as RMSNorm weight |
| `ColumnParallelLinear` | `[O/tp, I]` (cut by output dimension) | None | FFN up/gate (followed by Row) |
| `MergedColumnParallelLinear` | `[(O1+O2)/tp, I]` | None | FFN gate+up merge once kernel |
| `QKVParallelLinear` | `[(Nq+2Nkv)·H/tp, I]` | None | Q/K/V merge, K/V shard is smaller in GQA |
| `RowParallelLinear` | `[O, I/tp]` (press input dimension) | all_reduce | FFN down, sum the partial sum of each card |

**Key implementation details**:

`weight_loader` is mounted on `Parameter` as an attribute, and checkpoint is called uniformly when loading. Each class decides how to split it - no need to modify the loading logic.

`RowParallelLinear` bias is only added at rank 0: `bias if tp_rank == 0 else None` to avoid bias being accumulated tp_size times after all_reduce.

**Mathematical equivalence** (Column + Row pairing):
```
Full: y = x @ W.T
Sharded: x is full, W is split along output dim → each GPU computes y_i = x @ Wi.T, concat gives y   ← ColumnParallel
         x is sharded, W is split along input dim → each GPU computes y_i = xi @ Wi.T, all_reduce sum ← RowParallel
```

### 8. VocabParallelEmbedding and ParallelLMHead (`embed_head.py`)

#### VocabParallelEmbedding

**`__init__`**: The vocabulary list is divided equally by TP, and each card holds `vocab/tp` vectors:

```python
self.vocab_start_idx = num_embeddings_per_partition * tp_rank
self.vocab_end_idx   = vocab_start_idx + num_embeddings_per_partition
self.weight = nn.Parameter(torch.empty(num_embeddings_per_partition, embedding_dim))
```

**`forward`**: lookup table + all_reduce

```python
mask = (x >= vocab_start_idx) & (x < vocab_end_idx)
x    = mask * (x - vocab_start_idx)  # zero out out-of-range values (avoid out-of-bounds crashes)
y    = F.embedding(x, self.weight)   # safe lookup; out-of-range results are meaningless
y    = mask.unsqueeze(1) * y         # mask out meaningless results
dist.all_reduce(y)                   # for each token only one GPU is nonzero, so sum = correct vector
```

Why not just skip out-of-range tokens? If/else is very expensive on GPU, use the trick of "safety check first, then clear mask" to avoid conditional branches.

Numerical example (tp=2, vocab=10):
```
x = [2, 7, 0, 5]
GPU 0 (id 0~4): y = embed([2,0,0,0]) * mask → [vec2, 0, vec0, 0]
GPU 1 (id 5~9): y = embed([0,2,0,0]) * mask → [0, vec7, 0, vec5]
all_reduce sum → [vec2, vec7, vec0, vec5] ✓
```

#### ParallelLMHead

Inherits `VocabParallelEmbedding` **Shared weights**, but forward is completely different:

```python
def forward(self, x):
    # In prefill, only take the last token of each segment (only need to predict the next token)
    if context.is_prefill:
        x = x[context.cu_seqlens_q[1:] - 1].contiguous()

    logits = F.linear(x, self.weight)   # [batch, vocab/tp]

    # gather to rank 0 and concatenate full logits (only rank 0 performs sampling)
    if tp_size > 1:
        all_logits = [torch.empty_like(logits) for _ in range(tp_size)] if rank == 0 else None
        dist.gather(logits, all_logits, dst=0)
        logits = torch.cat(all_logits, dim=-1) if rank == 0 else None
    return logits
```

| | VocabParallelEmbedding | ParallelLMHead |
|---|---|---|
| communication | all_reduce (all cards require results) | gather → rank 0 (only rank 0 samples) |
| Reason | The next layer of all cards must use embedding | Sampling is only done at rank 0 |

**Prefill only counts the last token**: the input sequence [tok1...tokN] only needs the logits of tokN, and the calculation amount of LM Head is reduced from `seq_len × vocab` to `batch × vocab`.



---

## GQA implementation key points (interview focus)

**Core idea**: Multiple Q heads share the same set of K/V heads, and the KV cache size is ∝ `num_kv_heads` (not `num_heads`).

```
MHA (28Q, 28KV): Q0↔KV0, Q1↔KV1, ..., Q27↔KV27
GQA (28Q,  4KV): Q0~Q6↔KV0, Q7~Q13↔KV1, Q14~Q20↔KV2, Q21~Q27↔KV3
Savings factor = num_heads / num_kv_heads = 28 / 4 = 7×
```

**5-step implementation** (taking Qwen3 + nano-vllm as an example):

**① Declare asymmetric head number**
```python
self.num_heads    = total_num_heads    // tp_size   # Q heads / GPU (large)
self.num_kv_heads = total_num_kv_heads // tp_size   # KV heads / GPU (small)
self.q_size  = self.num_heads    * head_dim
self.kv_size = self.num_kv_heads * head_dim
```

**② QKV projection output asymmetric dimension**
```python
# output = (28 + 2×4) × head_dim = 36 × head_dim
output_size = (total_num_heads + 2 * total_num_kv_heads) * head_dim
```

**③ split split into different sizes**
```python
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
q = q.view(-1, num_heads,    head_dim)   # [N, 28/tp, d]
k = k.view(-1, num_kv_heads, head_dim)   # [N,  4/tp, d]
v = v.view(-1, num_kv_heads, head_dim)   # [N,  4/tp, d]
```

**④ Flash Attention natively supports GQA**: `flash_attn_varlen_func` automatically broadcasts K/V when `num_heads != num_kv_heads`, without manual `repeat_kv`.

**⑤ KV cache is allocated according to kv_heads**
```python
# kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
#                                                          ↑ stores only 4 copies, not 28
```

**Frequently Asked Interview Questions**:

> **Q: What kind of memory does GQA save? How much is saved? **
> KV cache (the main memory bottleneck during inference). Savings multiplier = `num_heads / num_kv_heads`. Qwen3-8B: 28Q/8KV = 3.5×; LLaMA-3-70B: 64Q/8KV = 8×. The model weights themselves (W_K, W_V) also decrease proportionally.

> **Q: What is the difference between GQA and MQA? **
> MQA (Multi-Query Attention) is the extreme case of GQA (`num_kv_heads=1`), all Q heads share the same K/V. GQA is a compromise solution, the effect is close to MHA, and KV cache is close to MQA.

> **Q: How is GQA split under TP? **
> Q/K/V are all divided according to `// tp_size`, but it must be ensured that `num_kv_heads` can be evenly divided by `tp_size`. If `num_kv_heads < tp_size`, some cards do not have KV heads, requiring additional broadcast logic (uncommon).

> **Q: Why is there no need to manually repeat K/V? **
> Flash Attention kernel internally performs broadcast directly based on `num_heads / num_kv_heads` (group size), which is completed at the register level, saving video memory and more efficient than explicit repeat.

---

## Part 2: Inference Engine Systems Design (Scheduler, BlockManager & Batch Execution)

## 1 From the user perspective: LLM entrance

```python
# nanovllm/llm.py
class LLM(LLMEngine):
    pass
```

`LLM` directly inherits `LLMEngine`, and the core API exposed to the outside world is `generate()`:

```python
llm = LLM("path/to/qwen3")
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.7, max_tokens=128))
```

Behind this line, nano-vllm completes in sequence: tokenize → scheduling → model forward → sampling → check termination conditions → loop until all is completed.

---

## 2 Global configuration

```python
@dataclass
class Config:
    model: str                          # Model path (local directory)
    max_num_batched_tokens: int = 16384 # Maximum prefill tokens per batch
    max_num_seqs: int = 512             # Maximum concurrent sequence count
    max_model_len: int = 4096           # Maximum length of a single sequence
    gpu_memory_utilization: float = 0.9 # Upper bound on GPU memory utilization
    tensor_parallel_size: int = 1       # TP parallelism degree
    enforce_eager: bool = False         # Force eager mode (disable CUDA Graph)
    kvcache_block_size: int = 256       # KV Cache block size (token count)
```

Key constraints:
- `kvcache_block_size` must be a multiple of 256 - aligns vectorization of Triton kernel
- `max_num_batched_tokens >= max_model_len`——Guaranteed to prefill at least one complete sequence
- Automatically read model parameters from HuggingFace config in `__post_init__`

---

## 3 Prefill and Decode: two stages of LLM reasoning

Understanding these two stages is the key to understanding the entire engine architecture.

### 3.1 The essence of Autoregressive generation

The generation of LLM is **autoregressive**: only one token is generated at a time, spliced ​​into the sequence and then continues forward. Key observation: **The K and V vectors of the previous token will not change with the new token** (causal mask guarantee), so they can be cached - this is **KV Cache**.

With KV Cache, the generation process is naturally divided into two stages with distinct characteristics.

### 3.2 Prefill: Compute-Bound

Processing the entire prompt (length $L$) at once, the core overhead is the $O(L^2 d)$ matrix multiplication of attention.

```
Prefill Attention arithmetic intensity:
  FLOPs  = 2 · L² · d
  Bytes  = 2 · L · d · sizeof
  AI     = L / sizeof   (grows linearly with sequence length)

  L=2048, bf16 → AI = 1024 FLOP/Byte
  A100 compute/bandwidth ≈ 156 FLOP/Byte
  → AI >> 156 → Compute-Bound ✓
```

Optimization strategies: Flash Attention (reduce HBM access), variable length Batching.

### 3.3 Decode: Memory-Bound

Each step only processes **1 token**, but reads the entire KV Cache.

```
Decode Attention arithmetic intensity:
  FLOPs  = 2 · S · d
  Bytes  = 2 · S · d · sizeof
  AI     = 1 / sizeof = 0.5   (independent of sequence length!)

  A100's 156 FLOP/Byte >> 0.5 → purely Memory-Bound ✓
  99.7% of GPU compute is wasted; the bottleneck is HBM read bandwidth
```

Optimization strategy: large batch size (read one weight to serve multiple tokens), CUDA Graph (eliminate kernel launch overhead), Paged KV Cache (maximize the number of concurrent sequences).

### 3.4 Why must we separate the two stages?

| Dimensions | Prefill | Decode |
|------|---------|--------|
| Bound type | Compute-Bound | Memory-Bound |
| Number of tokens per step | Hundreds to thousands | 1 |
| Attention calculation | GEMM ($L \times L$) | GEMV ($1 \times S$) |
| Flash Attention API | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| CUDA Graph | Not applicable (input shape dynamic) | Very applicable (shape fixed) |

Nano-vllm's choice: Each round of `step()` will either do all prefill or all decode without mixing.

> [!note]About Chunked Prefill
> vLLM supports mixing long prompt slices and decode tokens in the same batch to improve overall GPU utilization. nano-vllm does not implement this feature to keep the code simple.

### 3.5 KV Cache memory consumption

```
KV Cache per token = 2 × num_layers × num_kv_heads × head_dim × sizeof(dtype)

Example: Qwen3-8B (32 layers, 8 KV heads, head_dim=128, bf16):
  = 2 × 32 × 8 × 128 × 2 = 128 KB / token

512 concurrent sequences × 2048 context = 128 GB ← far beyond a single GPU!
```

This is why **Paged Attention** is required.

### 3.6 Paged Attention: From operating system to GPU memory management

#### Problems with traditional solutions

Pre-allocate `max_model_len` of continuous space for each sequence, three fatal problems:

1. **Internal fragmentation**: allocated according to maximum length, actual usage is usually < 50%
2. **External fragmentation**: Space is released after the sequence is completed, leaving holes of varying sizes.
3. **Unable to share**: KV Cache with the same prompt prefix is ​​stored independently.

#### Core idea: virtual memory paging

| Operating system concepts | Paged Attention correspondence |
|-------------|---------------------|
| Virtual address space | Sequence of logical KV Cache (token 0, 1, 2, ...) |
| Physical Page Frame | Pre-allocated fixed size KV Cache Block on GPU |
| Page table | `block_table`: logical block index → ​​physical block ID |
| Page size | `block_size` (256 tokens in nano-vllm) |
| Allocation on demand | New blocks are allocated only when the sequence grows, non-reserved max_model_len |
| Sharing page | Prefix Caching: Multiple sequences share the same physical block of prefix |

#### Memory layout after paging

```
GPU KV Cache physical space (preallocated block pool):
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ Blk0 │ Blk1 │ Blk2 │ Blk3 │ Blk4 │ Blk5 │ Blk6 │ Blk7 │
│(free)│ SeqA  │ SeqB │(free)│ SeqA │ SeqB │ (free)│shared │
│      │0-255 │0-255 │      │256-  │256-  │      │prefix│
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

SeqA (len=500): block_table=[1,4]  ← 2 blocks, utilization 500/512=98%
SeqB (len=456): block_table=[2,5]
SeqC (shared prefix): block_table=[7,3]  ← Block7 is shared with another sequence
```

Compared with tradition: SeqA is allocated according to max=4096, and the utilization rate is 500/4096=12%.

#### Three key mechanisms

**Premise definition**: Each sequence maintains a `block_table: list[int]`, mapping the **logical block index** (0, 1, 2, ...) to the **physical block ID** on the GPU. All physical blocks are pre-allocated at one time when the engine starts, and are stored in the global KV Cache tensor `[2, layers, num_blocks, block_size, heads, d]`. The physical block ID is the index of the third dimension of this tensor. `block_size=256` means that each block stores 256 token K/V vectors.

---

**Mechanism 1: Allocation on demand**

Sequences do not reserve space of maximum length and only allocate new blocks when needed:

```
Let `block_size=256`, and a sequence grow from 0:

len=1   → logical block 0 is just created, assign physical block #5 → block_table=[5]
len=256 → block #5 becomes exactly full (stores token 0..255)
len=257 → enter logical block 1, assign physical block #3 → block_table=[5,3]
len=512 → block #3 becomes full (stores token 256..511)
len=513 → enter logical block 2, allocate another new block ...
```

Implemented by `BlockManager.can_append()` and `may_append()`, the condition is `len(seq) % block_size == 1` (the first token that just entered the new block).

---

**Mechanism 2: block_table indirect addressing**

The sequence of KV Cache is physically **discontinuous**, and a layer of indirect mapping is done through `block_table`:

```
Sequence total length 500 tokens, `block_size=256`, `block_table=[5, 3]`

Logical view (contiguous):   token 0..255       | token 256..499
                            logical block 0      logical block 1
                         ↓ block_table[0]=5    ↓ block_table[1]=3

Physical view (non-contiguous):  Block #5 stores K/V[0..255]   Block #3 stores K/V[256..499]
```

The `block_table` parameter of Flash Attention directly receives this mapping and completes indirect addressing within the GPU kernel, **without the need to copy scattered blocks into continuous memory**.

---

**Mechanism 3: Reference counting and prefix sharing**

If two sequences have the same prompt prefix, they can share the same set of physical blocks instead of storing one copy each:

```
SeqA: "system prompt (256 tokens) + question A"  block_table=[5, 2]
SeqB: "system prompt (256 tokens) + question B"  block_table=[5, 8]
                                              ↑ shared Block #5
                                              ref_count = 2
```

`block.ref_count` records how many sequences refer to the block, and is only truly recycled when it reaches zero. Sharing is matched by **content hash**: the hash of each block is calculated in a chain (xxhash64) by `(hash of the previous block, current block token_ids)`. Sequences with the same prefix will hit the same hash when allocating, and the physical block will be automatically reused.

#### slot_mapping: precise mapping of Token to physical location

```
physical slot = block_table[logical block] × block_size + intra-block offset

Example: `block_table=[5,3]`, `block_size=256`
  token 0   → slot = 5×256+0   = 1280
  token 256 → slot = 3×256+0   = 768    ← Note: slot is not monotonically increasing!
  token 400 → slot = 3×256+144 = 912
```

Triton kernel `store_kvcache` receives the `slot_mapping` array, and scatter writes the K/V of each token.

#### The three-layer implementation architecture of nano-vllm

```
┌──────────────────────────────────────────────────────┐
│ BlockManager (CPU, Python)                           │
│   Manages block allocation/deallocation/sharing/hash │
│   Outputs: block_table, slot_mapping                 │
├──────────────────────────────────────────────────────┤
│ ModelRunner.prepare_prefill / prepare_decode         │
│   Converts block_table → GPU tensor, transfers async │
├──────────────────────────────────────────────────────┤
│ Attention Layer (GPU)                                │
│   store_kvcache (Triton): scatter-writes KV Cache    │
│   flash_attn (CUDA): indirect reads via block_table  │
│   Physical storage: [2, layers, blocks, block_size, heads, d] │
└──────────────────────────────────────────────────────┘
```

> [!important]Why block_size = 256?
> Too small (such as 16): block_table becomes long and management overhead is high; too large (such as 4096): Internal fragmentation increases and degenerates into continuous allocation. 256 is friendly to the GPU memory access mode of Flash Attention, with an average waste of 128 tokens ≈ 16 KB, which is far less than the MB-level waste of traditional solutions.

---

## 4 LLMEngine: Reasoning main loop

### 4.1 Initialization: Multi-process and component creation

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config = Config(model, **config_kwargs)

        # ── 1. Multi-process startup (Tensor Parallelism) ──
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()

        # ── 2. Main-process ModelRunner (rank 0) ──
        self.model_runner = ModelRunner(config, 0, self.events)

        # ── 3. Tokenizer and Scheduler ──
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.scheduler = Scheduler(config)
```

**Why does TP use multi-process instead of multi-thread? ** Superposition of two constraints:

1. **GIL**: Only one thread in Python can execute bytecode at the same time. When doing TP with multiple threads, the matrix multiplication of two GPUs is serialized at the Python level and cannot be truly parallelized.
2. **CUDA context**: Each GPU requires an independent CUDA context to manage the video memory and kernel queue. Sharing the context of the same process among multiple threads will cause competition; multiple processes are naturally isolated, and each process exclusively occupies the context of a card.
3. **NCCL**: Collection communication such as `all_reduce` is designed for inter-process and does not support inter-thread mode.

Therefore, the correct posture of TP is: **N GPUs = N processes, each process occupies one card**, and tensor synchronization is performed between processes through NCCL.

On this basis, nano-vllm has been lightweighted: **control instructions** (which sequences are scheduled and what operations are performed) use SharedMemory + Event to avoid socket/pipe overhead; NCCL is only used where merger activation is really needed.

**`ctx = mp.get_context("spawn")`**: Python multiprocessing has three startup modes:

| Mode | Behavior | Applicable Scenarios |
|------|------|---------|
| `fork` | The child process inherits the full memory copy (COW) of the parent process | Linux default, but not CUDA compatible |
| `spawn` | Child process starts a new Python interpreter from scratch | **CUDA safe**, used by nano-vllm |
| `forkserver` | forked by the server process | somewhere in between |

**`ctx.Event()`**: Cross-process synchronization primitive, similar to Boolean flag:

```python
event.wait()    # Worker: block and wait
event.set()     # Rank 0: wake worker
event.clear()   # Worker: reset, prepare for next round
```

**Worker life cycle**: `self.loop()` is called at the end of `ModelRunner.__init__` (when rank > 0), entering an infinite loop - the worker process **never returns** from `__init__`.

The entire system is divided into two stages after startup: **initialization** (single) and **inference loop** (repeatedly).

**Initialization phase**: Rank 0. First use `process.start()` to kick up the worker, and then initialize `ModelRunner(rank=0)` yourself. Both sides load model weights, allocate KV Cache, and capture CUDA Graph in parallel. After completion, the worker enters `loop()` to block and wait.

**Inference loop phase**: Each time `step()`, Rank 0 writes instructions to SharedMemory, `event.set()` wakes up the worker; both parties execute the same model forward at the same time, and when encountering `RowParallelLinear` and other places where the results need to be merged, the tensors are synchronized through NCCL `all_reduce`; after the forward ends, the worker blocks again, and Rank 0 gets the logits to continue scheduling.

```
[Initialization phase (once)]

  Rank 0 (main process)        Rank 1 (Worker)
  ─────────────────────        ─────────────────────
  process.start()      ──→     __init__() starts
  ModelRunner(rank=0)  starts     load model weights
    load model weights            allocate KV Cache
    allocate KV Cache             capture CUDA Graph
    capture CUDA Graph            loop() → event.wait() ← blocks here
  return, enter `step()` loop


[Inference loop phase (each `step()`)]

  Rank 0 (main process)        Rank 1 (Worker)
  ─────────────────────        ─────────────────────
  write_shm("run", seqs)
  event.set()          ──→     event.wait() returns
                                read_shm() reads command
  model.forward() starts        model.forward() starts
      │                             │
      └──── NCCL all_reduce ────────┘   ← synchronize at RowParallel
      │                             │
  model.forward() ends          model.forward() ends
  get logits, continue scheduling   event.wait() ← blocks again
```

### 4.2 Step loop: Scheduling → Execution → Post-processing

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

LLMEngine is a **single-threaded event loop**. Each `step()` completes a complete round of "scheduling → execution → post-processing". `generate()` calls `step()` repeatedly inside `while not self.is_finished()`.

> [!important]Prefill takes priority
> `schedule()` always processes prefill first. Only enter decode when the waiting queue is empty. Each round of steps is either all prefill or all decode - there is no mixed scheduling.

---

## 5 Sequence: The life cycle of the sequence

**The position of Sequence in the entire system**: It is the carrier of data flow. Every request from the user will be encapsulated into a Sequence object, flowing from `generate()` to `postprocess()`. Scheduler operates the queue of Sequence, BlockManager allocates physical blocks to Sequence, ModelRunner converts Sequence into GPU tensors, and the sampling results are written back to Sequence - **all components revolve around Sequence**.

What is stored in Sequence:

| Properties | Description |
|------|------|
| `token_ids` | Complete token sequence (prompt + generated completion) |
| `num_prompt_tokens` | prompt length (unchanged, used to distinguish prompt and completion) |
| `num_cached_tokens` | Prefix Cache hits the number of tokens, skip this part of the calculation when prefilling |
| `block_table` | The list of physical block IDs occupied by this sequence, maintained by BlockManager |
| `status` | The current stage (see state machine) |
| `last_token` | The last token, only this one is needed for model input during decoding |

### State machine

Each Sequence undergoes an explicit state transition in the system:

```
              user request arrives
                  ↓
              WAITING         ← waiting in the `waiting` queue to be scheduled
                  │
     (`scheduler.schedule()` selects it, `BlockManager` allocates blocks)
                  ↓
              RUNNING         ← in the `running` queue, processed every decode round
                  │
     (generate EOS token, or reach `max_tokens`)
                  ↓
              FINISHED        ← removed from `running`, blocks are released

              [Special case]
              RUNNING ──(insufficient KV Cache space, preempted)──→ WAITING
```

The preemption path of RUNNING → WAITING is the only way nano-vllm can cope with the memory pressure: the content generated by the preempted sequence will not be lost (the token_ids are still there), but the KV Cache will be cleared, and the entire sequence needs to be re-prefilled the next time you re-enter RUNNING.

### Serialization optimization

Sequence needs to be passed to TP worker through SharedMemory. The less content transmitted, the better:

```python
def __getstate__(self):
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)
```

Key design: **prefill passes complete `token_ids`** (worker needs to know all tokens to construct input), **decode only passes `last_token`** (only the newly generated token in the previous step will be used as model input, and worker does not need history). For long sequences, this can compress the transfer volume from thousands of tokens to 1 token.

---

## 6 Scheduler: The core logic of scheduling

**Scheduler's position in the entire system**: It is the first step of `step()` and the entrance to each reasoning loop. The Scheduler's task is to answer a question: **In this round, which sequences should be processed, prefill or decode? ** It does not touch the GPU, it is pure CPU logic, outputs a sequence list and an `is_prefill` flag, and gives it to ModelRunner for execution.

Scheduler maintains two queues, corresponding to two waiting states of the sequence:

```python
self.waiting: deque[Sequence]  # Newly arrived requests, not prefilling yet
self.running: deque[Sequence]  # Prefill already completed, now decoding token by token
```

### 6.1 Schedule: Decision-making logic for each round

The decision-making of `schedule()` has strict priority: **process the waiting queue first (prefill), and only process the running queue (decode) when waiting is empty**. The two will never be mixed in the same batch.

The reason is that the computing characteristics are different (analyzed in §3.4): prefill is compute-bound, decode is memory-bound, and mixed scheduling will make both sides less than optimal.

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # ── Phase 1: pack as many waiting sequences as possible for prefill ──
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens \
           or not self.block_manager.can_allocate(seq):
            break   # Total tokens exceed the limit, or not enough KV Cache blocks; stop loading
        self.block_manager.allocate(seq)      # Allocate physical blocks for this sequence
        num_batched_tokens += len(seq) - seq.num_cached_tokens  # Prefix-cache hits do not count
        seq.status = RUNNING
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True   # There are prefill tasks, return immediately

    # ── Phase 2: waiting is empty; process the running queue for decode ──
    while self.running:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            # Decode needs to write the KV for a new token, but there are not enough free blocks
            # Solution: preempt the last-added sequence in the running queue (LIFO) to free blocks
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq); break      # Even this sequence must be preempted; skip this decode round
        else:
            self.block_manager.may_append(seq)   # Confirm/allocate the blocks needed by this token
            scheduled_seqs.append(seq)
    return scheduled_seqs, False
```

`num_batched_tokens += len(seq) - seq.num_cached_tokens` This line is the key to the prefix cache: tokens that hit the cache do not need to be calculated by the model, so they do not occupy the token quota of the batch. This allows requests with common system prompt words to load more sequences.

### 6.2 Preemption preemption mechanism

**Preemption only occurs during the decode phase**. Prefill and decode handle insufficient space in completely different ways:

| Phase | Behavior when insufficient space | Whether existing RUNNING sequences are affected |
|------|----------------|--------------------------|
| Prefill (Phase 1) | `break`, stop loading more sequences, less prefill in this round | **not affected** |
| Decode (Phase 2) | Active `preempt()`, preempting the sequence at the end of the running queue | **Affected, kicked back to WAITING** |

The reasons are also different: Insufficient space in the prefill stage means that the current number of concurrent sequences is enough, and new requests can be rejected; insufficient space in the decode stage is because the existing sequences continue to grow, and space must be made to continue advancing.

**The sequence that is preempted is always the RUNNING state**. Sequences in WAITING are not preempted - at most they are "not scheduled this round".

```python
def preempt(self, seq: Sequence):
    seq.status = WAITING
    self.block_manager.deallocate(seq)   # Release all physical blocks used by this sequence
    self.waiting.appendleft(seq)         # Put it back at the head of the waiting queue (not the tail!)
```

Two design details:

- **LIFO selection**: Preempt the **tail** of the running queue (the one recently added), because it decodes the least tokens and has the smallest re-prefill cost.
- **Replace the head**: `appendleft` instead of `append`, to ensure that the preempted sequence will be restored first in the next round and will not be queued by new requests.

> [!note]Differences from vLLM
> vLLM supports two preemption strategies: **recompute** (clear the KV Cache and re-prefill during recovery) and **swap** (swap out the KV Cache to CPU memory and replace it with the GPU during recovery). Swap avoids the computational overhead of re-prefill, but the implementation complexity is much higher. nano-vllm only implements recompute, exchanging simplicity for engineering costs.

### 6.3 Postprocess: Finishing after model execution

`postprocess()` is called at the end of each `step()`, receives the token sampled by the model, writes the result back to the sequence, and determines whether it is finished.

It does three things:

**①Append new token**: Append the sampling result to `seq.token_ids`, and the sequence length +1. This new token will become the `last_token` for the next round of decoding.

**② Judgment termination**: It ends when one of the two conditions is met - EOS token is generated (and `ignore_eos` is not set), or the generated length reaches the upper limit of `max_tokens`.

**③ Clean up resources**: The terminated sequence immediately `deallocates` and returns the physical blocks to the BlockManager. These blocks can be used by new requests immediately.

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)                          # ① append
        if (not seq.ignore_eos and token_id == self.eos) \
           or seq.num_completion_tokens == seq.max_tokens:  # ② check
            seq.status = FINISHED
            self.block_manager.deallocate(seq)              # ③ cleanup
            self.running.remove(seq)
```

---

## 7 BlockManager: The essence of Paged KV Cache

**BlockManager's position in the entire system**: It is the "operating system kernel" of KV Cache memory. The physical KV Cache space on the GPU is allocated once by ModelRunner at startup. BlockManager is responsible for the **logical management** of this space - deciding which tokens occupy which physical blocks, which blocks can be shared, and when to release them.

Interaction between BlockManager and other components:

```
Scheduler.schedule()
  └── can_allocate(seq) → allocate(seq)     # Before prefill: allocate blocks for a new sequence
  └── can_append(seq)  → may_append(seq)    # Before decode: confirm/allocate blocks for the new token
  └── deallocate(seq)                       # When a sequence ends or is preempted: release blocks

ModelRunner.prepare_prefill/decode()
  └── read `seq.block_table`                # Compute `slot_mapping` and `block_tables` tensors
                                            # Pass them to the GPU kernels
```

BlockManager itself is **pure CPU logic** and does not touch the GPU. It only manages logical mapping (block_table, reference counting, hash table), the real KV vector writing is done by the Triton kernel, and the reading is done by Flash Attention.

### 7.1 Block data structure

```python
class Block:
    block_id: int
    ref_count: int = 0    # Reference count (supports sharing)
    hash: int = -1        # Content hash (used for prefix caching)
    token_ids: list = []  # Token content stored in this block
```

BlockManager’s core data structure:

```python
self.blocks: list[Block]
self.free_block_ids: deque[int]
self.used_block_ids: set[int]
self.hash_to_block_id: dict[int, int]    # hash → block ID (prefix cache)
```

> [!important]physical correspondence
> KV Cache actual storage shape: `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`. The `block_id` of Block directly corresponds to the index of the third dimension. A block stores the K and V vectors of `block_size` tokens.

### 7.2 Allocate: First allocation and prefix caching

```python
def allocate(self, seq: Sequence):
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # Only compute a hash for full blocks (the last partial block gets hash -1)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)

        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True   # Once there is a miss, all subsequent blocks are misses

        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size   # Prefix cache hit!
            if block_id in self.used_block_ids:
                self.blocks[block_id].ref_count += 1   # Share an existing block
            else:
                block = self._allocate_block(block_id)

        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

Step by step analysis:
1. **Chained Hash**: The hash of each block is calculated by `(prefix_hash, token_ids)`, ensuring the same prefix + the same content → the same hash
2. **Cache Miss Propagation**: Once a block is missed, subsequent misses are inevitable (the hash chain is broken)
3. **Reference Count**: If the hit block has been used by other sequences, ref_count +1 will be used to achieve sharing.
4. **The last incomplete block**: the hash is -1 and does not participate in the prefix cache (the content will also change)

### 7.3 Deallocate and May_Append: Dynamic scaling

**Deallocate** (Release in reverse order, cooperate with free_block_ids order, try to retain the earlier prefix cache):

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)   # Truly free only when the reference count reaches zero
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**Can_Append / May_Append** (expanded during decode, the core conditions are extremely delicate):

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
    # len % block_size == 1 means we just crossed into a new block and need exactly 1 new block
    # Otherwise no new block is needed (we can write into the remaining space of the current block)

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        # Allocate a new block
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        seq.block_table.append(block_id)
    elif len(seq) % self.block_size == 0:
        # Block just became full → compute hash and register it in the prefix cache
        token_ids = seq.block(seq.num_blocks - 1)
        prefix_hash = self.blocks[seq.block_table[-2]].hash if len(seq.block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix_hash)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
```

> [!important]Delayed registration of Prefix Cache
> The hash is calculated and registered only when the block is filled. **Prefix cache is gradually established during the decode process** instead of being completed all at once during allocate. If subsequent sequences have the same prefix, these blocks can be hit when allocating.

### 7.4 Prefix Caching complete process example

```
Request A: "system prompt + user question A"
Request B: "system prompt + user question B"  ← share the first 256 tokens
```

1. **Request A allocate**: Block0 stores "system prompt word" (256 token), `hash_to_block_id[hash_0]=5`
2. **Request B allocate**: Block0 hit → `seq.num_cached_tokens += 256`, `blocks[5].ref_count=2`
3. **prefill request B**: `input_ids` skips the first 256 tokens, `cu_seqlens_q < cu_seqlens_k`, flash_attn reads the cache KV through `block_table`

---

## 8 ModelRunner: from scheduling to execution

**ModelRunner's position in the entire system**: It is the **bridge** between CPU scheduling logic and GPU computing. What Scheduler outputs is a decision on "which sequences should be processed." ModelRunner translates this decision into tensors that the GPU can execute, drives the model forward, and then returns the sampling results to Scheduler.

The responsibility chain of ModelRunner in a `step()`:

```
Receive `seqs + is_prefill`
  │
  ├── `prepare_prefill(seqs)`  or  `prepare_decode(seqs)`
  │     ├── concatenate `input_ids`, `positions`
  │     ├── compute `cu_seqlens_q/k`, `slot_mapping`, `block_tables`
  │     └── `set_context(...)`   ← put these tensors into the global Context
  │
  ├── `call("run", ...)`  →  notify TP workers via SharedMemory + Event
  │
  ├── `run_model(input_ids, positions, is_prefill)`
  │     ├── Prefill: eager mode, `flash_attn_varlen_func`
  │     └── Decode: CUDA Graph mode, `flash_attn_with_kvcache`
  │
  └── return `token_ids` (sampling results) to `Scheduler.postprocess()`
```

ModelRunner is also responsible for two one-time initialization tasks: **KV Cache allocation** (calculating how many blocks can be allocated based on the remaining video memory, one-time `torch.empty`) and **CUDA Graph capture** (pre-recording the decode calculation graph under each batch size).

### 8.1 Inter-process communication of Tensor Parallelism

```
Rank 0 (main process)                  Rank 1..N (Worker)
  ├── write_shm(method, args)
  └── event.set() ────────────────────→ event.wait() → read_shm()
                                          getattr(self, method)(*args)
  ←──────────── NCCL all_reduce ──────────────────────────────────┘
```

- **SharedMemory**: 2MB shared memory, pickle serialization method name and parameters
- **Event**: multiprocessing.Event as synchronization signal
- **NCCL**: Actual tensor synchronization (all_reduce) happens inside `RowParallelLinear`

### 8.2 KV Cache allocation

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

    # Bytes per block
    block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
    #             ↑K+V

    # Allocate all KV Cache at once
    num_blocks = (total * gpu_utilization - used) // block_bytes
    self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)

    # Bind each layer's k_cache/v_cache pointers to Attention modules
    for module in self.model.modules():
        if hasattr(module, "k_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
```

> [!important]Pre-allocation strategy
> KV Cache is allocated once when it is started. Subsequent BlockManager only performs logical block allocation (modifying block_table) and does not involve malloc/free of GPU memory. Avoid CUDA memory fragmentation.

### 8.3 Prepare Prefill: variable length splicing and slot_mapping

```python
def prepare_prefill(self, seqs):
    for seq in seqs:
        # Skip tokens that hit the prefix cache
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(range(seq.num_cached_tokens, len(seq)))

        seqlen_q = len(seq) - seq.num_cached_tokens   # Actual Q length to compute
        seqlen_k = len(seq)                            # Full K length (including cache)
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

        # slot_mapping: only map tokens that need to be written into cache
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * block_size
            end = start + (block_size if not last_block else seq.last_block_num_tokens)
            slot_mapping.extend(range(start, end))

    # block_tables are needed when a prefix cache exists (K length > Q length)
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)

    set_context(is_prefill=True, cu_seqlens_q, cu_seqlens_k, slot_mapping, block_tables)
```

| Concept | Description |
|------|------|
| `cu_seqlens_q` | The cumulative sequence length of Q, only the newly calculated token is included in the prefix cache |
| `cu_seqlens_k` | The cumulative sequence length of K, always the complete sequence (including cache) |
| `slot_mapping` | token position in batch → KV Cache physical slot |
| `block_tables` | Required only for prefix cache, tells flash_attn which blocks to read KV from |

### 8.4 Prepare Decode: input structure for token-by-token decoding

```python
def prepare_decode(self, seqs):
    for seq in seqs:
        input_ids.append(seq.last_token)          # Only the last token is needed
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))             # Full context length
        slot_mapping.append(
            seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
        )
    block_tables = self.prepare_block_tables(seqs)   # Decode always requires block_tables
    set_context(is_prefill=False, slot_mapping, context_lens, block_tables)
```

Compare with prefill:

| Dimensions | Prefill | Decode |
|------|---------|--------|
| input_ids length | variable length (hundreds to thousands) | fixed 1 per seq |
| slot_mapping | continuous range | single value per seq |
| block_tables | Only required for prefix cache | Always required |
| context_lens | Not used | Tells flash_attn how long to read the KV |

### 8.5 Run Model: Eager vs CUDA Graph

#### What is CUDA Graph?

Under normal circumstances (Eager mode), every time PyTorch calls an operator (matrix multiplication, softmax, norm...), the CPU will issue a kernel launch instruction to the GPU. This launch process itself has overhead: the CPU needs to prepare parameters, call the driver, and write to the GPU's command queue.

```
Eager mode (this process repeats every decode round):

CPU                         GPU
 ├─ launch matmul ────────→  execute matmul
 ├─ launch softmax ───────→  execute softmax
 ├─ launch norm ──────────→  execute norm
 ├─ ... (hundreds of kernels) ──→  ...
 └─ launch final_proj ────→  execute final_proj

Every `step()` repeats the launch process above
```

For prefill, this is not a problem - each kernel has to process a large number of tokens, and the calculation time is much greater than the emission overhead. However, **decode only processes 1 token** per sequence, and the actual calculation time of each kernel is very short. At this time, the proportion of kernel launch overhead to the total time is very significant.

**CUDA Graph's solution**: "record" the entire forward calculation into a graph in advance. After that, each replay only requires the CPU to issue one instruction, and the GPU internally executes all kernels according to the graph:

```
CUDA Graph mode:

[Capture phase (one-time)]
CPU: with torch.cuda.graph(g): model.forward(...)  ← record the full kernel call sequence

[Execution phase (each decode)]
CPU                         GPU
 └─ graph.replay() ──────→  execute all kernels in recorded order (CPU no longer participates)
```

**Limitations**: CUDA Graph records a fixed calculation graph, which requires **the shape of input and output to be completely consistent during capture and playback**. This is why only decode can be used - each input in the decode stage is `[bs, 1]` (fixed shape), while the input length of prefill changes with prompt.

What nano-vllm does is to pre-capture a set of discrete batch sizes (`[1, 2, 4, 8, ..., max_bs]`). When actually decoding, select ≥ the minimum value of the actual bs. The insufficient positions are filled with padding (`slot_mapping=-1`).

---

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # CUDA Graph path
        bs = input_ids.size(0)
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["slot_mapping"].fill_(-1)       # -1 means padding; the Triton kernel will skip it
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

Applicable conditions for CUDA Graph: only used in decode stage (prefill input shape is not fixed), batch size ≤ 512, non-enforce_eager mode.

> [!important]Performance implications of CUDA Graph
> Decode only generates 1 token for each sequence, and the calculation amount is very small. At this time, **kernel launch overhead becomes the bottleneck**. CUDA Graph eliminates kernel-by-kernel launch overhead and can bring about 2-3 times speed improvement under small batch sizes.

**CUDA Graph capture** (capture from large to small, share `graph_pool`, large graph is allocated memory pool first, small graph is reused):

```python
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
    self.graphs[bs] = graph
```

---

## 9 Attention Layer: KV Cache writing and dual path calculation

**Attention Layer's position in the entire system**: It is the actual executor of KV Cache reading and writing, and is also the intersection of the two paths of Prefill and Decode. Every step of scheduling and allocation decisions of the entire inference engine is ultimately implemented at this layer as scatter writing and Flash Attention calculations on the GPU.

Understanding Attention Layer requires answering two questions first:

1. **What "external" information does it need**: In addition to the Q/K/V tensor itself, it also needs `slot_mapping` (where KV is written), `cu_seqlens_q/k` (sequence boundaries), `block_tables` (physical distribution of KV) - these all come from the scheduling layer and are not generated by the model calculation itself.

2. **How ​​does this information get here**: From the generation of ModelRunner to the consumption of Attention Layer, it must pass through model.forward and N DecoderLayer.forward. How to deliver it gracefully?


```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()                     # ← Read scheduling metadata from the global Context

        # Step 1: Write newly computed K and V into the KV Cache (every forward pass does this)
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        # Step 2: Run attention; Prefill and Decode take different paths
        if context.is_prefill:
            if context.block_tables is not None:    # With a prefix cache, K/V must be read from cache
                k, v = self.k_cache, self.v_cache
            o = flash_attn_varlen_func(q, k, v, ...)
        else:
            o = flash_attn_with_kvcache(q.unsqueeze(1), self.k_cache, self.v_cache, ...)
        return o
```

### 9.1 Context: Implicit channel for reasoning metadata

#### Background: Who needs this metadata and who doesn’t?

The scheduling metadata required by Attention Layer comes from ModelRunner's calculation in `prepare_prefill/decode`:

```
ModelRunner.prepare_prefill()
    ├── compute `cu_seqlens_q / cu_seqlens_k`   (sequence boundaries)
    ├── compute `slot_mapping`                   (KV write locations)
    └── compute `block_tables`                   (physical KV layout)
```

This data needs to eventually reach `Attention.forward()`, but the calling path is like this:

```
ModelRunner.run_model(input_ids, positions)
  └→ model.forward(input_ids, positions)
       └→ DecoderLayer.forward(hidden_states, positions)   × 32 layers
            └→ Attention.forward(q, k, v)                 ← consumed here
```

The middle `model.forward` and the 32 `DecoderLayer.forward` **do not need this metadata at all**, they only handle `hidden_states` and `positions`.

#### Problem: Parameter penetration (Prop Drilling)

If passed with function parameters,
```python
# DecoderLayer itself does not use these parameters, but it must receive and pass them through
def forward(self, hidden_states, positions,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            slot_mapping, context_lens, block_tables):
    ...
    # DecoderLayer itself uses none of these parameters; it only forwards them to Attention
    attn_out = self.attn(q, k, v, cu_seqlens_q, cu_seqlens_k,
                         max_seqlen_q, max_seqlen_k, slot_mapping, ...)
```

All 32 layers must be written like this, and every time nano-vllm adds or removes a metadata field, the signatures of all layers must be changed accordingly - this is a typical interface pollution.
#### Solution: Global Context (implicit channel)

```
ModelRunner                                  Attention
     │                                           │
     │  set_context(cu_seqlens_q, ...)           │  context = get_context()
     │         ↓                                 │
     │    [global Context singleton] ───────────→│
     │
     Intermediate layers remain completely unaware; the interface stays clean:
     DecoderLayer.forward(hidden_states, positions)  ← same as the original HuggingFace version
```

The cost is the introduction of implicit state (global variables), but the gain is that the model code remains clean and portable. The new model only needs to modify a few lines of `Attention.forward` and does not need to change all the middle layer interfaces.

> [!note]Same design in vLLM
> vLLM uses the `forward_context` global variable, and the principle is exactly the same. This is a common practice in the field of inference engines. The trade-off between clean interfaces and parameter penetration is to choose the former.

#### Context data structure

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None   # Prefill: sequence boundaries of Q
    cu_seqlens_k: torch.Tensor | None = None   # Prefill: sequence boundaries of K (including prefix cache)
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None   # Prefill + Decode: physical slots for KV writes
    context_lens: torch.Tensor | None = None   # Decode: KV context length of each sequence
    block_tables: torch.Tensor | None = None   # Prefill(prefix cache) + Decode: physical block layout
```

Life cycle and calling relationship:

```
prepare_prefill() / prepare_decode()
  └── set_context(...)       ← set: preparation phase of each `step()`

Attention.forward()          ← read: every layer of model forward
  └── get_context()

after `run()` ends
  └── reset_context()        ← cleanup: release tensor references to prevent VRAM leaks
```

### 9.2 store_kvcache: Triton Kernel writes to the paging cache

#### Why write first and then read?

The first step of `Attention.forward` is always `store_kvcache`, which writes the current newly calculated K/V into the KV Cache, and then calls Flash Attention for calculation.

Reason: The K/V of the current token is part of the KV sequence required for attention calculation (for decoding, the K/V of the current token also participates in the attention calculation, because `cache_seqlens` contains the current position). Write first, Flash Attention reads the complete sequence directly from the cache without additional splicing.

#### Why use Triton instead of PyTorch index?

Problem: KV Cache is paged (non-contiguous), and the newly calculated K/V needs to be spread to different physical locations according to `slot_mapping` - this is a scatter write:

```
Newly computed K (the `i`-th token in the batch)  →  physical slot[i] in the KV Cache
```

PyTorch's `index_put_` can do scatter, but each call itself is a kernel launch, which is expensive and cannot use vectorization. Triton kernel combines all token writes into one kernel launch and uses `tl.arange` for vectorized loading/storage:

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                          k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)                           # Each thread block handles one token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return                             # Padding token (CUDA Graph fill), skip
    key_offsets   = idx * key_stride + tl.arange(0, D)   # Source: token `idx` in the batch
    cache_offsets = slot * D + tl.arange(0, D)           # Destination: slot position in the KV Cache
    tl.store(k_cache_ptr + cache_offsets, tl.load(key_ptr + key_offsets))
    # Same for value
```

`D = num_heads * head_dim`: Flatten `(num_heads, head_dim)` of each token, continuous in memory, and can be vectorized. The skip logic of `slot == -1` corresponds to the padding of CUDA Graph: when decoding, the batch size will be padded to a discrete value, the extra position slot=-1, and the kernel will skip and not write.

#### Why is Triton faster than PyTorch?

This is essentially a **scatter write merge optimization**, and PyTorch's native implementation needs to be split into multiple steps:

```python
# PyTorch equivalent: each step is one independent CUDA kernel launch
mask = slot_mapping != -1                           # kernel 1: compare
valid_slots = slot_mapping[mask]                    # kernel 2: compact
k_cache[valid_slots] = key[mask].reshape(-1, D)     # kernel 3+4: gather + scatter
v_cache[valid_slots] = value[mask].reshape(-1, D)   # kernel 5+6: gather + scatter
```

The advantages of the Triton version are on three levels:

| Layers | PyTorch | Triton |
|------|---------|--------|
| Kernel launch times | ~6 times (mask/select/reshape/put × 2) | **1 times** |
| slot_mapping read | K and V are read once each | **Shared read**, one thread block writes K and V at the same time |
| Intermediate tensor | Temporary tensors such as mask and valid_slots need to be allocated | **All in registers**, zero additional allocation |

Each kernel launch has a fixed overhead of about 5–10 μs. For prefill (each token does a lot of calculations), this overhead can be ignored; however, in the decode stage, each sequence only writes the KV of 1 token, and the actual writing time is very short. At this time, the proportion of launch overhead is very significant - this is exactly the same as the motivation of CUDA Graph to eliminate launch overhead.

#### Actual measurement Benchmark (A6000)

A complete reasoning comparison of the two implementations on A6000:

| Implementation | Overall throughput | Decode real-time speed | Total time spent |
|------|---------|---------------|--------|
| Triton (nano-vllm implementation) | 3902 tok/s | ~310 tok/s | 34.33s |
| PyTorch (index_put_ equivalent implementation) | 3867 tok/s | ~284 tok/s | 34.64s |

The overall throughput difference is only ~0.9%, but the real-time speed difference during the decode phase is **~8.5%** (310 vs 284 tok/s). The gap is expected to be concentrated in decoding: the batch size during decoding is very small, with only 1 token per sequence. At this time, the fixed overhead of each kernel launch accounts for a higher proportion of the actual computing time.

The benchmark process also exposed an important hidden limitation of PyTorch implementation: **boolean indexing produces dynamic shapes that cannot be captured by CUDA Graph**. This means that the PyTorch version is incompatible with the CUDA Graph path, and the decode phase can only be in eager mode, further adding up to the loss. The conditional branch inside the Triton kernel (`if slot == -1: return`) does not affect the grid size, so it is naturally compatible with CUDA Graph - this is another hidden advantage of the Triton implementation.

**Source of `slot_mapping`**: Calculated by `ModelRunner.prepare_prefill/decode()`, the formula is `block_table[logical block] × block_size + intra-block offset` (§3.6 has detailed derivation). `slot_mapping` is not monotonic - different sequences of KVs are scattered throughout the physical space, which is what the Triton scatter write is for.

### 9.3 Prefill Attention: flash_attn_varlen_func

#### Background: attention calculation for variable length batches

When prefilling, there are multiple sequences of different lengths in a batch, and a unified tensor shape cannot be used like decoding. `flash_attn_varlen_func` is specially designed for variable length scenarios, receiving the spliced ​​Q/K/V (all sequence tokens are put together) and the `cu_seqlens` array to distinguish sequence boundaries:

```
cu_seqlens_q = [0, 5, 13]  →  sequence 0 Q range: [0,5), sequence 1: [5,13)
```

#### Two Prefill modes

**No prefix cache** (normal case):

- Q/K/V all come from the calculation results of the current prompt (passed in by `ModelRunner`)
- `cu_seqlens_q == cu_seqlens_k`, the sequence length is the same
- `block_table=None`, Flash Attention is calculated directly on continuous memory

**With prefix cache**:

- Q contains only **uncached** tokens (skipping the part that hits the prefix), `seqlen_q < seqlen_k`
- K/V needs to contain the **complete** context (cache KV with prefix), so read from KV Cache
- `block_table` non-None, tells Flash Attention which physical block the complete KV is in

```python
o = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=context.cu_seqlens_q,    # Q boundary of each sequence (shorter than K with prefix cache)
    cu_seqlens_k=context.cu_seqlens_k,    # K boundary of each sequence (always the full sequence length)
    max_seqlen_q=context.max_seqlen_q,
    max_seqlen_k=context.max_seqlen_k,
    softmax_scale=self.scale,
    causal=True,
    block_table=context.block_tables      # Non-None only with prefix cache; when None, use `k, v` directly
)
```

| condition | Q source | K/V source | block_table |
|------|--------|----------|-------------|
| None prefix cache | Newly calculated Q (full prompt) | Newly calculated K, V | `None` |
| With prefix cache | Newly calculated Q (skip prefix part) | Entire KV Cache (including cached) | Not `None` |

The difference between `cu_seqlens_q` and `cu_seqlens_k` is the amount of calculation saved by the prefix cache - the token that hits the cache does not participate in the calculation of the Q side, and K/V is directly read from the cache for attention.

### 9.4 Decode Attention: flash_attn_with_kvcache

#### Background: Memory-Bound + Pagination KV

Decode only processes **1 new token** per sequence, but must attend to the KV of all tokens in history. Historical KV is distributed in physically discontinuous blocks and must be indirectly addressed through `block_table`.

`flash_attn_with_kvcache` is an API specially designed for paged decode. It handles block_table addressing directly internally. There is no need to splice KV into continuous memory in advance:

```python
o = flash_attn_with_kvcache(
    q.unsqueeze(1),           # [batch, 1, num_heads, head_dim]  ← seqlen=1
    k_cache, v_cache,         # [num_blocks, block_size, num_kv_heads, head_dim]
    cache_seqlens=context.context_lens,   # Valid KV length of each sequence (avoid reading other sequences' data)
    block_table=context.block_tables,     # [batch, max_blocks_per_seq], each row is one sequence's block addresses
    softmax_scale=self.scale,
    causal=True,
)
```

**The reason for `q.unsqueeze(1)`**: Flash Attention expects the shape of Q to be `[batch, seqlen_q, num_heads, head_dim]`. There is only 1 new token for each sequence in the decode stage, so 1 is inserted in the seqlen dimension.

**The role of `cache_seqlens` (i.e. `context_lens`)**: The historical KV length of each sequence is different (different numbers of tokens are generated). This array tells Flash Attention to only read the KV of the first `cache_seqlens[i]` slots for each sequence to avoid reading data from other sequences out of bounds.

**Why decode always requires `block_table`**: The KV of decode is completely stored in the KV Cache (paged), and Flash Attention must perform indirect addressing through block_table to find the KV location of each token. The K/V of prefill (when there is no prefix cache) is passed directly in the function parameters, without block_table.

> [!tip]The performance cost of Paged indirect addressing
> Decode itself is memory-bound, and the bottleneck is HBM reading speed. block_table indirect addressing is equivalent to a layer of pointer jumping, and there is almost no performance loss for memory-bound calculations - because the memory access delay of the GPU is already a bottleneck, and one more address calculation is not the bottleneck.

---

## 10 End-to-end process concatenation

Take 2 prompts (5 tokens each), `max_tokens=3` as an example:

### Step 1: Prefill

```
scheduler.schedule():
  → allocate(A): block_table=[0]
  → allocate(B): block_table=[1]
  → return ([A, B], is_prefill=True)

prepare_prefill():
  input_ids    = [A0..A4, B0..B4]       (10 tokens)
  positions    = [0,1,2,3,4, 0,1,2,3,4]
  cu_seqlens_q = [0, 5, 10]
  cu_seqlens_k = [0, 5, 10]
  slot_mapping = [0,1,2,3,4, 256,257,258,259,260]

run_model() [eager, is_prefill]:
  At each Attention layer:
    1. store_kvcache → write the KV for 10 tokens
    2. flash_attn_varlen_func → compute attention output
  Sampler → [tokenA5, tokenB5]
```

### Step 2-4: Decode (3 rounds)

```
scheduler.schedule():
  waiting is empty → decode
  can_append(A): 6%256=6≠1, no new block needed → True

prepare_decode():
  input_ids    = [tokenA5, tokenB5]
  context_lens = [6, 6]
  slot_mapping = [5, 261]       (block0*256+5, block1*256+5)
  block_tables = [[0], [1]]

run_model() [CUDA Graph, bs=2]:
  graph.replay() → at each Attention layer:
    1. store_kvcache → write the KV for 2 tokens
    2. flash_attn_with_kvcache → read the full KV from cache
  Sampler → [tokenA6, tokenB6]
```

### Step 5: Complete

```
All sequences reach `max_tokens=3` → postprocess: `status=FINISHED`, deallocate blocks
is_finished() → True → exit loop → return results
```

---

## 11 Key design summary

### Three levels of abstraction of Paged Attention

```
Logical layer (Sequence)   Management layer (BlockManager)   Physical layer (KV Cache Tensor)
seq.block_table=[5,12,3]   blocks[5].ref_count=2        kv_cache[0, layer, 5, :, :, :]
                            hash_to_block_id[h]=5
```

### Simplified comparison with vLLM

| Features | vLLM | nano-vllm |
|------|------|----------|
| Preemption | Recompute + Swap | Recompute only |
| Scheduling | Prefill-Decode mixed (Chunked Prefill) | Prefill takes priority, not mixed |
| Prefix Cache | Radix Tree | Simple Hash Chain |
| CUDA Graph | Support | Support |
| Tensor Parallelism | Ray / Multi-process | SharedMemory + Event |
| Beam Search | Supported | Not supported |

### Core Design Principles

1. **Pre-allocation + paging management**: One-time allocation of GPU memory, block_table virtualization management, to avoid fragmentation
2. **Prefill/Decode separation**: The calculation characteristics of the two stages are different, and the optimal strategies are used respectively (Flash Attention vs CUDA Graph)
3. **Global Context**: Avoid passing metadata layer by layer and simplify model code
4. **Minimize communication**: Sequence serialization only transmits necessary information, SharedMemory avoids socket overhead
5. **Delayed Prefix Cache registration**: The hash is calculated when the block is full, and the cache is naturally established during the decoding process.

---

## Part 3: Practice Exercises & Review Self-Checks

### Module A: Inference Model Architecture & GQA

<details class="exercise">
<summary><span class="q-label">Exercise 1</span> <span class="q-text">What constitutes the minimal closed loop of nano-vllm?</span></summary>

The minimal loop comprises tokenization -> prompt queue insertion -> scheduler batch allocation -> physical block reservation in BlockManager -> ModelRunner forward pass (PagedAttention) -> sampler token generation -> sequence state update.

</details>

<details class="exercise">
<summary><span class="q-label">Exercise 2</span> <span class="q-text">Why does Grouped-Query Attention (GQA) reduce KV cache footprint?</span></summary>

Standard Multi-Head Attention maintains private Key and Value heads for each Query head. GQA shares a single Key/Value head across $G$ Query heads, reducing KV cache memory bandwidth and storage by a factor of $G$ while preserving model capacity.

</details>

### Module B: Engine Scheduling & Memory Systems

<details class="exercise">
<summary><span class="q-label">Exercise 1</span> <span class="q-text">How does BlockManager handle Copy-on-Write (CoW) during parallel sampling?</span></summary>

When generating multiple completions for the same prompt, sequences share physical token blocks with reference counting > 1. When a sequence appends a token, the block is duplicated only for the writing sequence, preventing unnecessary memory allocation.

</details>
