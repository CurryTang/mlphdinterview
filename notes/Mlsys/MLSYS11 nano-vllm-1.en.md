# MLSYS11 · nano-vllm Reading Notes (1)

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
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
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
C[\textcolor{blue}{GH}IJ\textcolor{red}{KL}] \cdot D[\textcolor{blue}{GH}MN\textcolor{red}{KL}] \rightarrow E[\textcolor{blue}{GH}IJMN]
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
| $Q[\textcolor{blue}{B}, T, \textcolor{blue}{K}, G, \textcolor{red}{H}] \cdot K[\textcolor{blue}{B}, S, \textcolor{blue}{K}, \textcolor{red}{H}]^T \rightarrow S[B,T,S,N]$ | $6BTSNH$ |
| $\text{softmax}_S(S) \rightarrow P$ | $O(BTSN)$ can be ignored |
| $P[\textcolor{blue}{B}, T, \textcolor{red}{S}, \textcolor{blue}{K}, G] \cdot V[\textcolor{blue}{B}, \textcolor{red}{S}, \textcolor{blue}{K}, H] \rightarrow O[B,T,N,H]$ | $6BTSNH$ |
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

## References

- [How To Scale Your Model - Part 4: Transformers](https://jax-ml.github.io/scaling-book/transformers/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- https://github.com/GeeeekExplorer/nano-vllm
