# MLSYS9 · Compute-Bound Kernel (3)

## 5. Softmax and Normalization Operations

Softmax and normalization operations involve **reduction** steps—each output element depends on the global statistics of all input elements in the same row, which inherently limits parallelism. This chapter explains the Online Softmax algorithm in depth (a key prerequisite for understanding Flash Attention), and then uses Triton to implement Softmax, LayerNorm, and RMSNorm.

---

### 5.1 Principles of the Online Softmax Algorithm

#### 5.1.1 Standard Softmax (3-Pass Algorithm)

Given an input vector $\mathbf{x} = [x_1, x_2, \dots, x_N]$:

**First pass (Pass 1) — compute the maximum:**
$$m = \max_{i=1}^{N} x_i$$
**Second pass (Pass 2) — compute the sum of exponentials:**
$$\ell = \sum_{i=1}^{N} \exp(x_i - m)$$
**Third pass (Pass 3) — normalize the output:**
$$y_i = \frac{\exp(x_i - m)}{\ell}$$
Subtracting the maximum value $m$ ensures numerical stability by preventing $\exp(x_i)$ from overflowing.

**Problem**: Each of the three passes must traverse the entire row of data, for a total of $3N$ elements loaded from HBM. On GPUs, global memory bandwidth is the most expensive resource.

#### 5.1.2 Online Softmax (1-Pass Statistics Computation)

The core idea of Online Softmax is: **maintain the running max and running exp-sum simultaneously in a single pass, and when a new maximum appears, rescale the previous partial sum with a correction factor.**

Initialization: $m_0 = -\infty$, $\ell_0 = 0$

When processing the $j$-th element $x_j$:
$$m_j = \max(m_{j-1}, x_j)$$
$$\ell_j = \ell_{j-1} \times \exp(m_{j-1} - m_j) + \exp(x_j - m_j)$$
**Correctness derivation**: After processing the first $j-1$ elements, $\ell_{j-1} = \sum_{i=1}^{j-1} \exp(x_i - m_{j-1})$. When the new element arrives, the new maximum becomes $m_j = \max(m_{j-1}, x_j)$, and we need:
$$\ell_j = \sum_{i=1}^{j} \exp(x_i - m_j)$$
Expanding this gives:
$$= \underbrace{\sum_{i=1}^{j-1} \exp(x_i - m_{j-1})}_{\ell_{j-1}} \cdot \underbrace{\exp(m_{j-1} - m_j)}_{\text{correction factor}} + \exp(x_j - m_j)$$
Note that when $m_j = m_{j-1}$, the correction factor is $1$, which degenerates to a simple accumulation.

After all elements have been processed, a second pass is still needed to compute $y_i = \exp(x_i - m_N) / \ell_N$. **In summary: 3-pass → 2-pass, and memory traffic drops from $3N$ to $2N$.**

#### 5.1.3 Block-wise Online Softmax (the Core of Flash Attention)

On GPUs, data is processed block by block. Suppose the current global statistics are $(m, \ell)$, and we process a new block $B_j$:

**Step 1: local statistics within the block**
$$m_j^{\text{local}} = \max_{x \in B_j} x, \quad \ell_j^{\text{local}} = \sum_{x \in B_j} \exp(x - m_j^{\text{local}})$$
**Step 2: update the global statistics**
$$m^{\text{new}} = \max(m, m_j^{\text{local}})$$
$$\ell^{\text{new}} = \ell \cdot \exp(m - m^{\text{new}}) + \ell_j^{\text{local}} \cdot \exp(m_j^{\text{local}} - m^{\text{new}})$$
**Step 3: correct the previous output** (the accumulated $O$ matrix in Flash Attention)
$$O^{\text{new}} = O \cdot \frac{\ell \cdot \exp(m - m^{\text{new}})}{\ell^{\text{new}}} + \frac{\exp(m_j^{\text{local}} - m^{\text{new}})}{\ell^{\text{new}}} \cdot P_j \cdot V_j$$
This formula is the core of Flash Attention: **there is no need to materialize the full attention matrix**. Instead, K/V blocks are processed incrementally in SRAM while normalization is maintained via online softmax.

---

### 5.2 Overview of Triton Grid Patterns

Different kinds of kernels correspond to different grid partitioning strategies. The core principle is: **the number of independent parallel dimensions determines the number of grid dimensions you need**. The common patterns are summarized below for reference in the kernel implementations that follow.

#### 1. Vector operations (elementwise) — 1D grid

There are $N$ elements, and each program processes `BLOCK_SIZE` elements. This is the simplest case.

```python
# Triton
grid = (triton.cdiv(N, BLOCK_SIZE),)

pid = tl.program_id(0)
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```
```c
// CUDA equivalent
dim3 grid(cdiv(N, BLOCK_SIZE));
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

#### 2. Matrix operations (GEMM: M×K @ K×N) — 2D grid

Each program is responsible for one `BLOCK_M × BLOCK_N` tile of the output matrix. Tiles along the M and N dimensions are independent of each other, while accumulation along K happens in the inner loop.

```python
# Option A: 2D grid
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

# Option B: 1D flattening (more common, convenient for swizzle optimization)
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

pid = tl.program_id(0)
grid_n = triton.cdiv(N, BLOCK_N)
pid_m = pid // grid_n     # output row block
pid_n = pid % grid_n      # output column block
```

```
Output matrix C (M × N):
         N direction →
    ┌────┬────┬────┐
M   │0,0 │0,1 │0,2 │
dim ├────┼────┼────┤
↓   │1,0 │1,1 │1,2 │
↓   ├────┼────┼────┤
    │2,0 │2,1 │2,2 │
    └────┴────┴────┘
Each cell = one program, size BLOCK_M × BLOCK_N
The inner loop accumulates along the K dimension
```

#### 3. Batched GEMM — add a batch dimension

```python
# Option A: 3D grid
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), batch)
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
pid_b = tl.program_id(2)

# Option B: flatten the M×N tile into axis=0, put batch on axis=1 (more common)
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), batch)
pid   = tl.program_id(0)
pid_b = tl.program_id(1)
pid_m = pid // grid_n
pid_n = pid % grid_n
```

#### 4. Flash Attention — flatten `batch × heads`

```python
grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

# axis 0: tiled index of Q (the inner loop traverses K/V blocks)
# axis 1: batch and head flattened into one dimension (they are fully independent)
```

Why flatten `batch × heads`? These dimensions are completely independent, so there is no need to dedicate two separate axes to them. Triton grids support at most 3 dimensions, so flattening preserves an axis for possible future extensions.

#### 5. 2D convolution — 3D grid

```python
grid = (
    triton.cdiv(out_H, BLOCK_H) * triton.cdiv(out_W, BLOCK_W),  # flatten spatial dimensions
    out_C,                                                         # output channel
    batch,                                                         # batch
)

pid_spatial = tl.program_id(0)
pid_oc      = tl.program_id(1)
pid_b       = tl.program_id(2)
pid_h = pid_spatial // grid_w
pid_w = pid_spatial % grid_w
```

#### 6. Reductions (softmax, layernorm) — 1D grid

Each program processes the reduction for an entire row, and rows are parallelized independently.

```python
# softmax: input (M, N), each row is normalized independently
grid = (M,)

pid = tl.program_id(0)    # row index
offs = tl.arange(0, BLOCK_N)
x = tl.load(X_ptr + pid * stride + offs, mask=offs < N)
# ... whole-row max → exp → sum → normalize
```

#### 7. Batched LayerNorm — 2D grid

```python
# input (batch, seq_len, hidden_dim), each (batch, position) is normalized independently
grid = (seq_len, batch)

pid_s = tl.program_id(0)  # sequence position
pid_b = tl.program_id(1)  # batch
# reduce along the hidden_dim dimension
```

**General rule**: independent output tiles map to grid dimensions; reduction/accumulation dimensions belong in the inner loop. When there are more than 3 independent dimensions, flatten fully independent dimensions (such as `batch × heads`) to fit Triton’s 3D grid limit.

#### Methodology for Designing Grid Dimensions

The number of grid dimensions is **decoupled** from the dimensionality of the problem itself—it is not “the problem has N dimensions, so the grid has N dimensions.” What matters is which dimensions require reduction.

**Step 1: list all problem dimensions and label them as parallel or reduction**

| Dimension role | Where it goes | Why |
|---|---|---|
| **Parallel dimension**: no communication is needed across blocks | → grid dimension | Can run independently in parallel |
| **Reduction dimension**: requires accumulation/comparison/aggregation | → inner loop | Must be handled serially or cooperatively |

Take several core kernels as examples:

```
GEMM: C[M,N] = A[M,K] @ B[K,N]
  M → parallel (each row is independent)
  N → parallel (each column is independent)
  K → reduction (must be accumulated)

Flash Attention: O[B,H,M,d] = softmax(Q @ K^T) @ V
  B → parallel
  H → parallel
  M → parallel (each query position is independent)
  N → reduction (must traverse all keys for softmax + accumulation)
  d → processed entirely within one block, not split

LayerNorm: y[B,S,D] = norm(x[B,S,D], dim=-1)
  B → parallel
  S → parallel
  D → reduction (must compute mean/var)

2D Conv: out[B,OC,OH,OW]
  B  → parallel
  OC → parallel
  OH → parallel
  OW → parallel
  IC, KH, KW → reduction
```

**Step 2: map parallel dimensions to the grid (up to 3 dimensions; flatten if there are more)**

```
Number of parallel dims     Strategy
─────────────────────────────────
  1           1D grid, map directly
  2           2D grid, map directly
  3           3D grid, map directly
  ≥4          Flatten: merge some dimensions into one
```

Flattening strategy—prefer flattening dimensions that are semantically related or relatively small:

```python
# Flash Attention: 4 parallel dims (B,H,M,d), but d is not split, so effectively 3
# B and H are semantically similar (both indicate "which group"), so flatten them
grid = (cdiv(M, BLOCK_M), B * H)            # 2D

# 2D Conv: 4 parallel dims (B,OC,OH,OW)
# OH and OW are spatial dimensions, so flatten them
grid = (cdiv(OH,BH) * cdiv(OW,BW), OC, B)   # 3D

# Batch GEMM: 3 parallel dims (B,M,N)
# Flatten M and N (convenient for swizzle optimization of L2 cache)
grid = (cdiv(M,BM) * cdiv(N,BN), B)          # 2D
```

**Step 3: place reduction dimensions in the inner loop**

```python
# GEMM: K is the reduction dimension
for k in range(0, K, BLOCK_K):
    a = load(A_block)   # streaming load
    b = load(B_block)
    acc += dot(a, b)     # accumulate

# Flash Attention: N (key sequence) is the reduction dimension
for start_n in range(0, N_CTX, BLOCK_N):
    k = load(K_block)   # streaming load
    v = load(V_block)
    # online softmax + accumulation
```

#### Practical Engineering Considerations

Beyond the basic parallel/reduction split, three additional factors affect the final design choice:

**1. SRAM capacity limits → determines which dimensions must be tiled**

```
SRAM capacity is limited (A100 ~192KB/SM), it must fit:
  Q block:  BLOCK_M × d     = 128 × 64 × 2B = 16KB
  K block:  BLOCK_N × d     = 64 × 64 × 2B  = 8KB
  V block:  BLOCK_N × d     = 64 × 64 × 2B  = 8KB
  acc:      BLOCK_M × d     = 128 × 64 × 4B = 32KB
  ──────────────────────────────────────────────
  Total ~64KB ✓ fits

If d=256, it may not fit → d must also become a reduction dimension, with an extra loop
```

**2. SM utilization → the total grid size must be large enough**

An A100 has 108 SMs, so the total number of programs in the grid should be much larger than 108 to fully utilize the chip.

```
Flash Attention example:
  grid = (4, 16) = 64 programs
  Only 64/108 ≈ 59% of the SMs are used → wasteful

If seq_len is very short, you can reduce BLOCK_M to create more work on axis 0
```

**3. L2 cache friendliness → affects flattening and traversal order**

```python
# Classic GEMM optimization: swizzled traversal order
# Not simple row-major, but a "grouped" order
# This makes neighboring programs access neighboring K/V data and improves L2 hit rate

pid = tl.program_id(0)
# Instead of pid_m = pid // grid_n; pid_n = pid % grid_n
# use swizzling:
GROUP_SIZE = 8
group_id = pid // (GROUP_SIZE * grid_n)
group_m  = group_id * GROUP_SIZE + (pid % GROUP_SIZE)
pid_n    = (pid % (GROUP_SIZE * grid_n)) // GROUP_SIZE
```

---

### 5.3 Triton Softmax

When an entire row fits inside a single `BLOCK_SIZE` (the most common case in practice), you can achieve a **true single pass**: 1 load + 1 store, which is the theoretical minimum memory traffic of $2N$.

**Why is a 1-pass implementation possible only when the whole row fits, and otherwise 2 passes are unavoidable?** The key is whether SRAM can simultaneously hold both the original data and the final statistics. If the row fits, the entire row $x$ is loaded into SRAM once, and max → exp → sum → normalization are all completed in registers, so when computing $y_i = \exp(x_i - m)/\ell$, all three quantities $x_i$, $m$, and $\ell$ are simultaneously available. If the row does not fit, SRAM can hold only one block at a time. By the time the final $m$ and $\ell$ have been computed after traversing all blocks, the original data $x$ from earlier blocks has already been overwritten by later blocks. But the normalization formula $y_i = \exp(x_i - m_{\text{final}})/\ell_{\text{final}}$ requires both the original $x_i$ and the global statistics. Since the two cannot coexist in SRAM at the same time, $x$ must be reloaded from HBM, forcing an extra pass.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-block Softmax: load the entire row into SRAM at once and complete all computation in registers.

    Memory traffic analysis:
      This implementation: N (load) + N (store) = 2N (theoretical optimum)
      Standard 3-pass: 3N (load) + N (store) = 4N
      Online 2-pass: 2N (load) + N (store) = 3N
    """
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    # Load the entire row into registers/SRAM at once
    # Why choose other=-inf:
    #   Does not affect tl.max: max(x_valid, -inf) = x_valid
    #   Does not affect tl.sum: exp(-inf) = 0
    #   Guarantees output 0: exp(-inf - m) / l = 0
    x = tl.load(
        input_ptr + row_idx * input_row_stride + col_offs,
        mask=mask, other=float('-inf')
    )

    # Complete the three softmax steps in registers
    # tl.max uses warp-shuffle reduction underneath (__shfl_xor_sync)
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)        # numerical stability: subtract max
    x_sum = tl.sum(x_exp, axis=0)     # warp-shuffle reduction for the sum
    result = x_exp / x_sum

    # Write back in one shot
    tl.store(
        output_ptr + row_idx * output_row_stride + col_offs,
        result, mask=mask
    )


@triton.jit
def softmax_online_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online Softmax: used when n_cols > BLOCK_SIZE.
    First pass computes (m, l) with the online algorithm, second pass writes the output. Total: 2 passes.
    """
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride

    # ========== First pass: online statistics ==========
    m_i = float('-inf')
    l_i = 0.0

    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(row_start + col_offs, mask=col_offs < n_cols, other=float('-inf'))

        m_ij = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        # Core: online softmax update
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    # ========== Second pass: normalized output ==========
    out_start = output_ptr + row_idx * output_row_stride
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        x = tl.load(row_start + col_offs, mask=mask, other=float('-inf'))
        result = tl.exp(x - m_i) / l_i
        tl.store(out_start + col_offs, result, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Unified entry point: automatically choose the optimal version based on row length."""
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(min(N, 65536))
    grid = (M,)

    if N <= 65536:
        softmax_kernel[grid](x, output, N, x.stride(0), output.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    else:
        BLOCK_SIZE = max(128, triton.next_power_of_2(min(N, 4096)))
        softmax_online_kernel[grid](x, output, N, x.stride(0), output.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return output
```

**Why doesn’t the Softmax kernel use 2D tiling like GEMM?** GEMM outputs an M×N matrix, and its output tiles are independent of each other, so it needs a mapping of `pid → (pid_m, pid_n) → (offs_m, offs_n)` from a 1D program ID to 2D tile coordinates. In contrast, Softmax has reduction dependencies within each row (max and sum must see the full row), so **the column dimension cannot be split into independent tiles**—one row must be handled by exactly one program. Therefore, the grid is naturally 1D, `(M,)` where M is the number of rows, and `program_id(0)` directly serves as the row index, with no need for tile-coordinate decomposition. Parallelism exists only across rows.

**How `tl.max` and `tl.sum` work**: when `BLOCK_SIZE = 1024`, one block contains 32 warps. `tl.max(x, axis=0)` first computes local maxima within each thread, then uses `__shfl_xor_sync` for warp-level reduction, and finally uses shared memory for reduction across warps. Triton’s compiler generates all of this code automatically.

**Precision strategy**: at the PTX level, `tl.exp` compiles to `ex2.approx.ftz.f32` (an approximate base-2 exponential), which is about 2× faster than precise `expf`. The approximation error is negligible for softmax.

---

### 5.4 Triton LayerNorm
$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
```python
@triton.jit
def layer_norm_kernel(
    input_ptr, output_ptr,
    gamma_ptr, beta_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm: each program processes one row.

    Precision-critical points:
    - mean/var must be computed in FP32. FP16 precision is 2^{-10} ≈ 0.001,
      and after 4096 accumulations the error can reach 4.096 (catastrophic cancellation).
      FP32 precision is 2^{-23}, and after 4096 accumulations the error is only ~5×10^{-4}.
    - Use the two-step method (subtract mean first, then compute var), rather than E[x²]-E[x]² (numerically unstable).
    """
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    # Load and convert to FP32
    x = tl.load(input_ptr + row_idx * input_row_stride + col_offs,
                mask=mask, other=0.0).to(tl.float32)

    # Mean and variance
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # Normalization + affine transform
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * inv_std
    gamma = tl.load(gamma_ptr + col_offs, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
    result = x_norm * gamma + beta

    tl.store(output_ptr + row_idx * output_row_stride + col_offs,
             result.to(tl.float16), mask=mask)
```

**The role of `col_offs` and Triton’s thread assignment mechanism**: `col_offs = tl.arange(0, BLOCK_SIZE)` generates the column index vector $[0, 1, \dots, \text{BLOCK\_SIZE}-1]$. One Triton program corresponds to one CUDA thread block (CTA). Suppose `BLOCK_SIZE=1024` and `num_warps=4`; then the CTA contains $4 \times 32 = 128$ threads, and each thread is responsible for $1024/128=8$ elements. `x`, `gamma`, and `beta` are all loaded using the same `col_offs` indices, so **the `x[i]`, `gamma[i]`, and `beta[i]` held by the same thread naturally correspond to the same column**. Elementwise operations (`x_norm * gamma + beta`, `x - mean`, `exp`) are computed independently by each thread on its own elements, with no thread-to-thread communication. Only reduction operations (`tl.sum`) require cooperation: each thread first computes a local partial sum, then warp shuffle reduces within each warp, then shared memory reduces across warps, and the final reduction result is broadcast back to all threads. Thus, once `mean` and `var` are computed, every thread holds the same scalar values, and the later `x - mean` returns to purely elementwise per-thread computation.

**What if `n_cols` exceeds a single `BLOCK_SIZE`?** The current implementation assumes `BLOCK_SIZE >= n_cols` (the whole row is loaded at once), which is sufficient for common hidden dimensions in practice (4096–8192, only 8–16 KB in FP16). If `n_cols` becomes very large, block-by-block looping is required, but **unlike softmax, LayerNorm accumulation does not need a correction factor**. The reason is that softmax uses $\ell = \sum \exp(x_i - m)$, which depends on the global max $m$; a new block may update the max, invalidating the previous exp-sum and requiring a correction by $\exp(m_{\text{old}} - m_{\text{new}})$. By contrast, LayerNorm uses $\sum x_i$, which is plain addition; when a new block arrives, you simply keep accumulating, and previous partial sums do not become “wrong.” However, multiple passes are still required—there is an ordering dependency between mean and variance (variance needs the mean first). A naive implementation requires 3 passes (first pass for mean, second for variance, third for output). It can be optimized to 2 passes by accumulating both $\sum x_i$ and $\sum x_i^2$ in the first pass, then using $\sigma^2 = E[x^2] - (E[x])^2$ to obtain mean and variance in one shot (at the cost of slightly worse numerical stability), followed by a second pass for normalization. In either case, the root cause of the extra passes is the same as in softmax: **the global statistics and the original data cannot reside in SRAM simultaneously.**

**LayerNorm is a purely memory-bound operation**: its arithmetic intensity is extremely low, so performance is determined entirely by HBM bandwidth. A single pass loads $N$ input values, $N$ `gamma` values, and $N$ `beta` values, and writes $N$ output values, for a total of $4N$ memory operations. The optimization directions are to reduce the number of passes and maximize effective memory bandwidth.

---

### 5.5 Triton RMSNorm
$$y = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2 + \epsilon}$$
Compared with LayerNorm, RMSNorm **removes mean-centering and bias**: one fewer sum reduction, one fewer elementwise subtraction, and no need to load the $N$ `beta` parameters, saving roughly 30% of the computation and 25% of the memory traffic overall.

```python
@triton.jit
def rms_norm_kernel(
    input_ptr, output_ptr,
    gamma_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    x = tl.load(input_ptr + row_idx * input_row_stride + col_offs,
                mask=mask, other=0.0).to(tl.float32)

    # RMS = sqrt(mean(x²) + eps)
    x_sq = tl.where(mask, x * x, 0.0)
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rms_inv = 1.0 / tl.sqrt(mean_sq + eps)

    gamma = tl.load(gamma_ptr + col_offs, mask=mask, other=1.0).to(tl.float32)
    result = x * rms_inv * gamma

    tl.store(output_ptr + row_idx * output_row_stride + col_offs,
             result.to(tl.float16), mask=mask)
```

Modern large models (LLaMA, Mistral, etc.) almost universally use RMSNorm. Studies show that mean-centering contributes very little to model quality while accounting for roughly one-third of LayerNorm’s compute overhead.

---

## 6. Flash Attention

### 6.1 Motivation

Standard self-attention, $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) \times V$, requires materializing an $N \times N$ attention matrix (with $Q, K, V \in \mathbb{R}^{N \times d}$), which costs $O(N^2)$ memory and enormous bandwidth.

The core ideas of Flash Attention are:

1. **Never materialize the $N \times N$ matrix**: compute it block by block in SRAM via tiling
2. **Process incrementally with block-wise online softmax** (the principle was detailed in Section 5)
3. **Keep intermediate results in SRAM**: GPU SRAM bandwidth is roughly 19 TB/s versus 2 TB/s for HBM
4. **Reduce memory from $O(N^2)$ to $O(N)$**: store only the output $O$ and a small set of softmax statistics $(m, \ell)$

---

### 6.2 The Flash Attention 2 Algorithm

**Input**: Q, K, V $\in \mathbb{R}^{N \times d}$, **output**: O = softmax(QK^T / √d) × V

**Blocking**: Q is split into $T_r$ blocks (each with $B_r$ rows), while K/V are split into $T_c$ blocks (each with $B_c$ rows).

```
Input: Q_i ∈ R^{B_r × d}
Initialize: O_i = 0, m_i = -∞, l_i = 0

For each K,V block j = 1, ..., T_c:
    S_ij = Q_i × K_j^T / √d              // computed in SRAM
    m_ij = rowmax(S_ij)                   // row-wise maximum of the current block
    m_new = max(m_i, m_ij)               // update the global maximum

    // Correct the previous results (block-wise online softmax, see 5.1.3)
    l_i = l_i × exp(m_i - m_new)
    O_i = O_i × exp(m_i - m_new)

    P_ij = exp(S_ij - m_new)             // softmax numerator
    l_i = l_i + rowsum(P_ij)
    O_i = O_i + P_ij × V_j              // GEMM-II

    m_i = m_new

O_i = O_i / l_i                          // final normalization
```

**Correctness**: The correction factor $\exp(m_{\text{old}} - m_{\text{new}})$ converts previously computed exponentials relative to $m_{\text{old}}$ into values relative to $m_{\text{new}}$: $\exp(x - m_{\text{old}}) \times \exp(m_{\text{old}} - m_{\text{new}}) = \exp(x - m_{\text{new}})$. Expanding the formula shows it is exactly equivalent to the standard definition.

**Memory**: HBM only needs to store Q, K, V ($3Nd$), O ($Nd$), and the statistics m, l ($2N$). SRAM temporarily stores $O(B_r d + B_c d + B_r B_c)$, which does not grow with $N$.

### 6.3 Triton Implementation of Flash Attention

**Stride and address computation**: Q has shape `(batch, heads, seq_len, d_model)`, and `stride` gives the number of elements to skip when moving by one step along each dimension. For a contiguous row-major tensor (`batch=2, heads=8, seq_len=512, d_model=64`):

| stride | Corresponding dimension | Value | Meaning |
|---|---|---|---|
| `stride_qb` | batch | $8 \times 512 \times 64 = 262144$ | jump to the next batch |
| `stride_qh` | heads | $512 \times 64 = 32768$ | jump to the next head |
| `stride_qm` | seq_len | $64$ | jump to the next row (token) |
| `stride_qk` | d_model | $1$ | jump to the next column (innermost contiguous dimension) |

The address of `Q[b, h, m, k]` is: `Q_ptr + b*stride_qb + h*stride_qh + m*stride_qm + k*stride_qk`.

In the code, `off_hz = tl.program_id(1)` is the flattened `batch × heads` index, and the base address uses `off_hz * stride_qh` instead of explicitly decomposing it. This works because under a contiguous layout, `stride_qb = heads × stride_qh`, so `b * stride_qb + h * stride_qh = (b * heads + h) * stride_qh = off_hz * stride_qh`. One multiplication is enough. If the tensor is non-contiguous (for example after a `transpose`), then it must be decomposed as `off_hz // heads * stride_qb + off_hz % heads * stride_qh`.

**Constructing the pointer matrix for a Q block**: `q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk` constructs a `[BLOCK_M, BLOCK_DMODEL]` pointer matrix. Conceptually, it manually flattens the multidimensional index `Q[b,h,m,d]` into linear addresses. It has three parts:

```
Q_ptr + q_offset                → locate the starting address of the (batch, head) block
+ offs_m[:, None] * stride_qm   → row offsets, [BLOCK_M, 1] column vector
+ offs_d[None, :] * stride_qk   → column offsets, [1, BLOCK_DMODEL] row vector
```

Broadcasting adds the two offsets to form the full 2D pointer matrix (assuming `start_m=1`, `stride_qm=64`, `stride_qk=1`):

```
Row offsets (×64)     Column offsets (×1)      Result (relative offsets)
┌──────┐              ┌──────────────┐         ┌─────────────────────┐
│ 8192 │              │ 0  1  ... 63 │         │ 8192  8193  ... 8255│ ← Q[128, 0:64]
│ 8256 │       +      │ 0  1  ... 63 │    =    │ 8256  8257  ... 8319│ ← Q[129, 0:64]
│  ... │              │     ...      │         │         ...         │
│16320 │              │ 0  1  ... 63 │         │16320 16321  ...16383│ ← Q[255, 0:64]
└──────┘              └──────────────┘         └─────────────────────┘
[128,1] broadcast     [1,64] broadcast         [128,64] pointer matrix
```

Each cell is a memory address, and `tl.load(q_ptrs)` loads the entire `[BLOCK_M, BLOCK_DMODEL]` Q block into SRAM in one shot. Since Triton does not have native multidimensional array support, 2D block loads must be simulated through pointer arithmetic plus broadcasting.

```python
import triton
import triton.language as tl
import torch


@triton.jit
def flash_attention_forward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX, sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Flash Attention 2 Forward — Triton implementation.
    Each program processes one Q block (BLOCK_M rows) for one (batch, head).
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Base address offsets
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load the Q block (resides in SRAM throughout the computation)
    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Initialize online softmax state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Main loop: traverse K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K block
        k_ptrs = K_ptr + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # S = Q @ K^T * scale
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))
        s *= sm_scale
        s = tl.where(offs_m[:, None] < N_CTX, s, float('-inf'))
        s = tl.where((start_n + offs_n)[None, :] < N_CTX, s, float('-inf'))

        # Online softmax update (core: block-wise version, see 5.1.3)
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)          # correction factor
        p = tl.exp(s - m_new[:, None])       # softmax numerator
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]            # correct the previous output

        # Load the V block and accumulate
        v_ptrs = V_ptr + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Write back
    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


def flash_attention_triton(q, k, v):
    """
    Flash Attention forward. q, k, v: (batch, heads, seq_len, d_model) fp16.
    """
    BLOCK_M, BLOCK_N = 128, 64
    batch, heads, seq_len, d_model = q.shape
    o = torch.empty_like(q)
    sm_scale = 1.0 / (d_model ** 0.5)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    flash_attention_forward_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=d_model,
    )
    return o
```



### 6.4 Backpropagation

Backpropagation for Flash Attention is more complex than the forward pass because the attention matrix must be recomputed (the forward pass stores only $O$, $m$, and $\ell$, not $P$).

#### 6.4.1 Backpropagation Mathematics

Let $P = \text{softmax}(S/\sqrt{d})$, $S = QK^T$, and $O = PV$. Given the upstream gradient $dO$:

**Step 1**: compute the auxiliary quantity $D_i = \sum_j dO_{ij} \cdot O_{ij}$ (row-wise dot product)

**Step 2**: for each pair of (Q block $i$, KV block $j$):
- Recompute $S_{ij} = Q_i K_j^T / \sqrt{d}$
- Recompute $P_{ij} = \exp(S_{ij} - m_i) / \ell_i$ (using the forward-saved $m$, $\ell$)
- $dV_j \mathrel{+}= P_{ij}^T \cdot dO_i$
- $dP_{ij} = dO_i \cdot V_j^T$
- $dS_{ij} = P_{ij} \odot (dP_{ij} - D_i)$ (elementwise, the softmax backward formula)
- $dQ_i \mathrel{+}= dS_{ij} \cdot K_j / \sqrt{d}$
- $dK_j \mathrel{+}= dS_{ij}^T \cdot Q_i / \sqrt{d}$

#### 6.4.2 Triton Backpropagation (Approximate Version)

The following is a conceptual Triton implementation of the backward pass. Production code typically splits it into two kernels (one for dQ and one for dK/dV); here they are combined to highlight the core logic. **Note: this version is a simplified teaching implementation and may have limitations in boundary handling and performance.**

```python
@triton.jit
def flash_attention_backward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    L_ptr, M_ptr,             # forward-saved log-sum-exp (l) and max (m)
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX, sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Flash Attention backward pass — simplified version.
    Each program processes one Q block and traverses all K/V blocks to compute dQ.
    dK and dV are accumulated via atomic_add (production code should use dedicated kernels to avoid atomics).
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q block, O block, and dO block
    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o = tl.load(o_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    do_ptrs = dO_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Load the softmax statistics saved during the forward pass
    m_ptrs = M_ptr + off_hz * N_CTX + offs_m
    l_ptrs = L_ptr + off_hz * N_CTX + offs_m
    m_i = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
    l_i = tl.load(l_ptrs, mask=offs_m < N_CTX, other=1.0)

    # Step 1: D_i = rowsum(dO * O)
    D_i = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)  # (BLOCK_M,)

    # Initialize the dQ accumulator
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)

    # Step 2: traverse K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K and V blocks
        k_ptrs = K_ptr + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V_ptr + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # Recompute S and P
        s = tl.dot(q, tl.trans(k)) * sm_scale                    # (BLOCK_M, BLOCK_N)
        p = tl.exp(s - m_i[:, None]) / l_i[:, None]              # recompute softmax
        p = tl.where((start_n + offs_n)[None, :] < N_CTX, p, 0.0)

        # dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))                              # (BLOCK_M, BLOCK_N)

        # dS = P * (dP - D_i)  — softmax backward formula
        ds = p * (dp - D_i[:, None]) * sm_scale                  # (BLOCK_M, BLOCK_N)

        # dQ += dS @ K
        dq += tl.dot(ds.to(tl.float16), k)

        # dV += P^T @ dO (simplified: atomic_add, production code should avoid this)
        # dK += dS^T @ Q (simplified: atomic_add)
        # The atomic accumulation for dV/dK is omitted here; a real implementation needs dedicated handling

    # Write back dQ
    dq_ptrs = dQ_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    tl.store(dq_ptrs, dq.to(tl.float16), mask=offs_m[:, None] < N_CTX)
```

**Known issues in the simplified version above**:
1. **Forward statistics were not saved**: the backward pass expects to load $m$ and $\ell$ from `M_ptr`/`L_ptr`, but the forward kernel does not write them back to HBM. A complete implementation must add `tl.store` before the final normalization in the forward pass to save `m_i` and `l_i`.
2. **dK/dV are completely missing**: the code computes only dQ; dK and dV accumulation are mentioned in comments but not implemented.
3. **Precision loss**: `ds.to(tl.float16)` truncates the gradient to FP16 before `tl.dot`. In backpropagation, gradients may be very small, and FP16 truncation can introduce significant numerical error. It should remain in FP32 or use BF16.
4. **Missing compiler hints**: the forward pass includes `tl.multiple_of(start_n, BLOCK_N)` to help optimization, but the backward pass omits it.

**Implementation challenges in the backward pass**:
1. **Accumulating dK/dV**: multiple Q blocks need to contribute gradients to the same K/V block. Production implementations usually use a dedicated kernel (outer loop over K/V blocks, inner loop over Q blocks) to avoid atomics.
2. **The cost of recomputation**: the backward pass must recompute $S$ and $P$, making it about 2.5× as expensive as the forward pass. But compared with storing the full $N \times N$ matrix $P$, recomputation is still the better trade-off.
3. **Numerical consistency**: the recomputed $\exp(S_{ij} - m_i) / \ell_i$ in the backward pass must exactly match the forward computation, including identical max and sum values.

---

### 6.5 Flash Attention 3: Hopper-Specific Optimizations

FA2 reaches about 70% utilization on A100, but only about 35% on H100. The reason is that Hopper introduces entirely new hardware capabilities (asynchronous Tensor Cores, TMA, FP8), and FA2’s synchronous design cannot exploit them. FA3 is a complete rewrite tailored for Hopper.

#### 6.5.1 Three Core Improvements

**1. Warp specialization**

In FA2, all warps perform homogeneous work—they both load data and compute. FA3 divides warps into two roles:

```
┌─────────────────── CTA (Thread Block) ───────────────────┐
│                                                           │
│  Producer Warps              Consumer Warps               │
│  ┌──────────────┐           ┌──────────────────────────┐  │
│  │ TMA load K_j │──SMEM──→ │ WGMMA: S = Q @ K^T       │  │
│  │ TMA load V_j │──SMEM──→ │ Softmax: P = softmax(S)  │  │
│  │ (ring buffer)   │           │ WGMMA: O += P @ V        │  │
│  └──────────────┘           └──────────────────────────┘  │
│  fewer registers ← setmaxnreg → more registers (GEMM accumulators) │
└───────────────────────────────────────────────────────────┘
```

- **Producer**: uses TMA (Tensor Memory Accelerator) exclusively to asynchronously load data from HBM into a shared-memory ring buffer
- **Consumer**: uses WGMMA (warpgroup-level matrix multiply) exclusively for computation
- **`setmaxnreg`**: a new Hopper instruction that dynamically reallocates registers—producer warps release registers to consumer warps, giving GEMM accumulators more register space

**2. Two-stage pipeline: overlap GEMM and softmax**

The bottleneck in FA2 is that the two GEMMs ($S = QK^T$ and $O = PV$) are separated by softmax and therefore execute strictly serially. FA3 breaks this dependency using WGMMA’s asynchronous semantics:

```
Time →
──────────────────────────────────────────────────
Iteration j:   [GEMM0: S_j=QK_j^T]  [softmax(S_j)]  [GEMM1: O+=P_j·V_j]
Iteration j+1:                       [GEMM0: S_{j+1}] [softmax(S_{j+1})]  [GEMM1]

↓ After overlap:

Iteration j:   [GEMM0_j] [softmax_j + GEMM0_{j+1}] [GEMM1_j]
                     ↑ softmax uses scalar units (MUFU)
                     ↑ GEMM0 uses Tensor Cores
                     → the two can execute simultaneously!
```

The key is: after launching `GEMM0_{j+1}`, FA3 does not wait for it to finish (`commit_group` but not `wait_group`), and immediately runs `softmax_j` on MUFU. The two use different hardware units, so they truly execute in parallel.

**3. Pingpong scheduling (alternating dual warpgroup execution)**

Even with the two-stage pipeline, MUFU throughput for softmax (~3.9 TFLOPS) is still about 256× lower than Tensor Core throughput for GEMM (~989 TFLOPS FP16), so softmax can still consume a noticeable number of cycles. Pingpong scheduling hides it using two warpgroups:

```
Time →
WarpGroup 0: [GEMM] [softmax] [GEMM] [softmax] ...
WarpGroup 1:        [GEMM] [softmax] [GEMM] [softmax] ...
                     ↑ while WG1 does GEMM, WG0 does softmax
                            → softmax is fully hidden
```

The two warpgroups alternate through the `bar.sync` hardware barrier, so the softmax latency is fully hidden under the other group’s GEMM execution time.

#### 6.5.2 FP8 Support: Incoherent Processing

The issue with FP8 is that Q/K in transformers often contain a small number of “outlier” dimensions with very large values, and FP8’s dynamic range cannot represent both large and small values well at the same time.

**Solution**: multiply by a random orthogonal matrix $M$ (Hadamard + random sign flips) to “spread out” the outliers:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(QM)(KM)^T}{\sqrt{d}}\right) V$$
Because $MM^T = I$, the math is unchanged: $(QM)(KM)^T = QMM^TK^T = QK^T$. But after multiplying by $M$, the magnitudes across dimensions become more uniform, reducing FP8 quantization error by 2.6×. Applying $M$ can be implemented with the Fast Walsh-Hadamard Transform at complexity $O(d \log d)$, and it can be fused with RoPE at effectively zero additional cost.

#### 6.5.3 Feasibility and Limitations in Triton

| FA3 technique | Feasibility in Triton | Notes |
|---|---|---|
| Warp specialization | **Not feasible** | Triton’s abstraction hides warp-level control, so producer/consumer roles cannot be assigned explicitly |
| `setmaxnreg` | **Not feasible** | Triton does not expose register allocation instructions |
| TMA | **Partially feasible** | Triton 3.0+ supports TMA primitives such as `tl.async_copy`, but with less flexibility than CUDA |
| WGMMA asynchronous semantics | **Not feasible** | Triton’s `tl.dot` is synchronous and cannot implement a commit-without-wait pipeline |
| Pingpong scheduling | **Not feasible** | It requires warpgroup-level barrier control, which Triton does not expose |
| Two-stage GEMM-softmax pipeline | **Limited** | The Triton compiler may perform some instruction reordering automatically, but explicit control is unavailable |
| FP8 + block quantization | **Feasible** | Triton supports FP8 types and FP8 operands for `tl.dot` |
| Incoherent processing | **Feasible** | The Hadamard transform is elementwise + butterfly structure, which Triton can implement |
| Basic online-softmax tiling | **Feasible** | This is the core of FA2, and Triton already supports it well (as in the implementation above) |

**Conclusion**: FA3’s main advantages (warp specialization, asynchronous pipelining, pingpong scheduling) depend deeply on Hopper’s low-level hardware primitives and therefore must be implemented in CUDA/CUTLASS. What Triton can deliver is FA2-style tiling + online softmax + FP8 quantization, corresponding to roughly 50–60% of FA3 performance. This also explains why FA3 is written in CUDA (based on CUTLASS 3.x) rather than Triton.

---

## 7. Activation Function Fusion Optimizations

### 7.1 Why Fuse?

Take the fully connected layer `D = GELU(A × B + bias)` as an example:

```
Unfused (3 separate kernels):
  HBM → A, B → GEMM → C written back to HBM     (2 reads + 1 write)
  HBM → C, bias → addition → C' written back to HBM  (2 reads + 1 write)
  HBM → C' → GELU → D written back to HBM         (1 read + 1 write)
  Total: 5 HBM reads + 3 HBM writes

Fused (1 kernel):
  HBM → A, B, bias → in registers: GELU(A×B + bias) → D written back to HBM
  Total: 3 HBM reads + 1 HBM write (eliminates 2 round trips of intermediate results to HBM)
```

The intermediate result C (for example, 4096×4096 FP16 = 32 MB) takes about 32 μs for one write + read round trip on an A100 at 2 TB/s—pure wasted bandwidth. After fusion, the GEMM result passes through Bias and GELU directly in registers, with zero additional HBM traffic.

---

### 7.2 CUTLASS Epilogue Fusion

CUTLASS GEMM kernels have three stages: Prologue (load A and B) → Mainloop (tiled matrix multiplication) → **Epilogue (post-processing + writeback)**.

In the epilogue, the GEMM result is still in registers (as fragments), so any elementwise operation can be fused at effectively zero cost. CUTLASS implements this through epilogue-functor templates:

```cpp
// D = GELU(alpha * A×B + beta * C)
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::GELU,     // activation function
    cutlass::half_t,                     // output type
    8,                                   // number of elements per vectorized writeback
    float,                               // accumulator type (FP32)
    float                                // compute type
>;

// Embed EpilogueOp into the full GEMM type definition
using FusedGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,      // A
    cutlass::half_t, cutlass::layout::ColumnMajor,    // B
    cutlass::half_t, cutlass::layout::RowMajor,       // C/D
    float,                                            // accumulator
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,           // CTA tile
    cutlass::gemm::GemmShape<64, 64, 32>,             // Warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,              // MMA instruction
    EpilogueOp,                                       // ← fused GELU
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3                                                 // number of pipeline stages
>;
```

**How the epilogue executes**: once the mainloop finishes computing an output tile (for example 128×128), the result is distributed as FP32 fragments across thread registers. The epilogue functor applies `output[i] = GELU(alpha * acc[i] + beta * source[i])` to the elements held by each thread, then writes them back to HBM in vectorized form. The entire process requires no additional SMEM or HBM accesses.

Bias can be passed through the C-matrix interface: `{bias, stride=0}` means broadcasting along the M dimension, and `beta=1.0` yields `D = GELU(A×B + bias)`.

---

### 7.3 Triton Fused Kernels

Fusion is even more direct in Triton—all operations are written inside a single kernel, with no special framework required.

#### 7.3.1 GEMM + Bias + GELU (full implementation)

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_gelu_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    C = GELU(A @ B + bias)

    All three fused steps are completed in registers:
    1. GEMM accumulates into FP32 registers (acc)
    2. bias is broadcast-added to acc (zero HBM overhead—bias has only N elements and is loaded once)
    3. GELU is computed directly on acc
    4. write back to HBM in one shot
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # ---- GEMM main loop ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_offset in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k_offset * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- Bias (broadcast addition in registers) ----
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]  # (1, BLOCK_N) broadcast to (BLOCK_M, BLOCK_N)

    # ---- GELU (computed in registers) ----
    # tanh approximation: GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    kAlpha = 0.7978845608028654   # √(2/π)
    kBeta = 0.044715
    inner = kAlpha * (acc + kBeta * acc * acc * acc)
    acc = 0.5 * acc * (1.0 + tl.math.tanh(inner))

    # ---- Write back (one store includes GEMM + Bias + GELU) ----
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def gemm_bias_gelu(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    """A: (M,K) fp16, B: (K,N) fp16, bias: (N,). Returns: (M,N) fp16."""
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_gelu_kernel[grid](
        A, B, bias, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
    )
    return C
```

#### 7.3.2 Brief Note on SwiGLU Fusion

SwiGLU (used in the LLaMA family): `output = Swish(X @ W_gate) ⊙ (X @ W_up)`

The key fusion insight is: **the two GEMMs share the same input X**. Inside the K-dimension iteration loop, the X tile only needs to be loaded once, then used in separate `tl.dot` operations with `W_gate` and `W_up`, saving 50% of the input bandwidth. Swish and the elementwise product are executed in registers after accumulation completes.

```python
# SwiGLU core loop (conceptual code)
gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

for k_start in range(0, D_in, BLOCK_K):
    x = tl.load(...)           # X tile — loaded only once
    wg = tl.load(...)          # W_gate tile
    wu = tl.load(...)          # W_up tile
    gate_acc += tl.dot(x, wg)  # share x
    up_acc += tl.dot(x, wu)    # share x

# Swish + elementwise multiply (done in registers)
result = (gate_acc * tl.sigmoid(gate_acc)) * up_acc
tl.store(out_ptrs, result.to(tl.float16), mask=mask)
```

**Principle for fusion strategy**: prioritize fusing memory-bound operators (activation functions, bias, residual connections). These operations have extremely low compute-to-memory ratios, so fusion delivers the greatest benefit. Fusion between two large GEMMs should be approached carefully, because it can create excessive register pressure and reduce occupancy.
