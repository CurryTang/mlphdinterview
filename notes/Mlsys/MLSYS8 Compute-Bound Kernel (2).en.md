# MLSYS8 · Compute-Bound Kernel (2)

## 4. Convolution Optimization

### 4.1 Core Idea: Map Convolution to Matrix Multiplication

#### 4.1.1 Mathematical Definition of Convolution

Standard 2D convolution (Conv2D):
$$\text{Output}[n, k, p, q] = \sum_{c=0}^{C-1} \sum_{r=0}^{R-1} \sum_{s=0}^{S-1} \text{Input}[n, c, p \cdot s_h + r - \text{pad}_h, q \cdot s_w + s - \text{pad}_w] \times \text{Weight}[k, c, r, s]$$

| Tensor | Shape | Meaning |
|------|------|------|
| Input | `[N, C, H, W]` | Batch size N, input channels C, height H, width W |
| Weight | `[K, C, R, S]` | Output channels K, input channels C, kernel height R, width S |
| Output | `[N, K, P, Q]` | Batch size N, output channels K, output height P, output width Q |

The pseudocode for the naive implementation makes the computational structure of convolution immediately clear:

```python
# Naive convolution: 7 nested loops
for n in range(N):              # Iterate over the batch
    for k in range(K):          # Iterate over output channels
        for p in range(P):      # Iterate over output height
            for q in range(Q):  # Iterate over output width
                acc = 0.0
                for c in range(C):      # Reduction: input channels
                    for r in range(R):  # Reduction: kernel height
                        for s in range(S):  # Reduction: kernel width
                            h = p * stride_h + r - pad_h
                            w = q * stride_w + s - pad_w
                            if 0 <= h < H and 0 <= w < W:
                                acc += Input[n, c, h, w] * Weight[k, c, r, s]
                Output[n, k, p, q] = acc
```

**Key observation**: the outer 4 loops `(n, k, p, q)` enumerate all output positions, while the inner 3 loops `(c, r, s)` perform the reduction sum. This structure is completely isomorphic to matrix multiplication $C_{ij} = \sum_k A_{ik} B_{kj}$—the outer loops enumerate output coordinates `(i, j)`, and the inner loop reduces over dimension `k`.

#### 4.1.2 Implicit GEMM: A Zero-Overhead Convolution → Matrix Multiplication Mapping

The traditional im2col method explicitly constructs an expanded matrix of shape `(N·P·Q) × (C·R·S)`, and its memory overhead can reach tens of times the size of the original input. **The core breakthrough of implicit GEMM is: it does not build the expanded matrix, but instead computes addresses dynamically at access time.**

> [!tip] im2col (Image to Column)
> im2col "flattens" each convolution window into one row of a matrix: for every output position $(p, q)$ in the input, it extracts $C \times R \times S$ elements and arranges them into one row; stacking all positions yields matrix A of shape $(N \cdot P \cdot Q) \times (C \cdot R \cdot S)$. The weights are reshaped into matrix B of shape $(K) \times (C \cdot R \cdot S)$. The convolution then becomes a standard GEMM: $\text{Output} = A \times B^T$.
> **Advantage**: directly reuses highly optimized cuBLAS GEMM, with a simple implementation.
> **Disadvantage**: the expanded matrix is highly redundant—adjacent windows share a fraction $(R-1)(S-1)/(RS)$ of their elements (about 55% for a 3×3 convolution), but im2col duplicates them in full. The memory expansion factor is $= R \times S$ (9× for 3×3). This is exactly the problem implicit GEMM solves.

Mapping rule:

$$\underbrace{(N \times P \times Q)}_{M} \times \underbrace{K_{\text{out}}}_{N_{\text{gemm}}} = \underbrace{(N \times P \times Q)}_{M} \times \underbrace{(C \times R \times S)}_{K_{\text{gemm}}} \cdot \underbrace{(C \times R \times S)}_{K_{\text{gemm}}} \times \underbrace{K_{\text{out}}}_{N_{\text{gemm}}}$$

- **M dimension** = N × P × Q (all output spatial positions flattened)
- **N_gemm dimension** = K (number of output channels)
- **K_gemm dimension** = C × R × S (reduction dimension)

**Coordinate decoding** is the core of the entire implementation. The `(m, k_gemm)` coordinates in the GEMM iteration must be decoded back into tensor coordinates:

```python
# M-dimension decoding: flattened output index → (n, p, q)
n = m // (P * Q)
p = (m % (P * Q)) // Q
q = m % Q

# K-dimension decoding: flattened reduction index → (c, r, s)
c = k // (R * S)
r = (k % (R * S)) // S
s = k % S

# Compute input-tensor coordinates from (n, p, q, c, r, s)
h = p * stride_h + r - pad_h
w = q * stride_w + s - pad_w
# If h or w is out of bounds, that location is zero-padded (via mask)
```

These coordinate calculations are performed entirely in registers, completely eliminating the global-memory overhead of the im2col buffer.

---

### 4.2 Triton Implementation (Full Version)

> CUTLASS and Triton share exactly the same core ideas—the coordinate mapping for implicit GEMM, tile blocking, and the reduction loop are fundamentally identical across the two frameworks. The difference is that achieving the same performance in CUTLASS requires manually configuring many low-level details such as TMA descriptors, warp-specialized pipelines, and SMEM swizzling, whereas Triton handles these automatically in the compiler. So we use the Triton version here to explain the algorithm clearly first.

The Triton implementation uses exactly the same mapping strategy, expressed in a Python DSL. The compiler automatically handles low-level details such as shared-memory allocation, data prefetching, and instruction scheduling.

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Config 1: large tile, suitable for large-scale convolutions
        # 128×128 output tile + 3-stage pipeline + 8 warps (256 threads)
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=3, num_warps=8
        ),
        # Config 2: medium tile, lower register pressure, allows deeper pipelining
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=4, num_warps=4
        ),
        # Config 3: asymmetric tile, suitable when output channels are moderate but spatial dimensions are large
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=3, num_warps=4
        ),
        # Config 4: small tile, suitable for small batches or small feature maps
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=5, num_warps=2
        ),
    ],
    key=['M', 'N_gemm', 'K_total'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    # ---- Tensor pointers ----
    input_ptr,    # Input tensor [N, C, H, W] (NCHW layout)
    weight_ptr,   # Weight tensor [K_out, C, R, S]
    output_ptr,   # Output tensor [N, K_out, P, Q]
    # ---- Tensor dimensions ----
    batch, C_in, H, W, K_out, R, S, P, Q,
    # ---- Convolution parameters ----
    pad_h, pad_w, stride_h, stride_w,
    # ---- Implicit GEMM dimensions (passed explicitly instead of derived inside the kernel) ----
    #
    # Why not compute M = batch*P*Q and K_total = C*R*S directly inside the kernel?
    #
    # 1. Decouple convolution semantics from GEMM semantics
    #    Implicit GEMM flattens (n,p,q) into a 1D m, and the kernel only tiles,
    #    applies masks, and handles boundaries over [0, M). Passing M explicitly
    #    means declaring: "the M-axis length of this GEMM is this value". The
    #    kernel does not need to care which convolution dimensions it came from.
    #
    # 2. Boundary masks need M, and the earlier the better
    #    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    #    mask_m = offs_m < M
    #    Using M directly checks the tail-tile boundary with a single comparison.
    #    If this became offs_m < batch*P*Q, each thread would perform an extra
    #    multiplication chain and would hard-wire the "boundary definition" to the
    #    specific relationship among batch, P, and Q, which blocks the common variants below.
    #
    # 3. Allow more general M values (split-K / padding / different layouts)
    #    - split-M pipeline: this kernel computes only a subrange of the M axis;
    #      M is the local length participating in this GEMM, not the global batch*P*Q.
    #    - Alignment/vectorization may require padding: logically the output is P×Q,
    #      but the effective M from the GEMM viewpoint may be the padded length.
    #    - Different layouts such as NHWC / blocked / fused epilogue: the flattened
    #      leading dimension can no longer be written simply as N*P*Q.
    #    Passing these explicitly lets the host control the meaning of M uniformly,
    #    while keeping the kernel generic.
    M,            # = N * P * Q (GEMM M dimension)
    N_gemm,       # = K_out (GEMM N dimension)
    K_total,      # = C * R * S (GEMM K dimension)
    # ---- Tensor strides ----
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_k, stride_wt_c, stride_wt_r, stride_wt_s,
    stride_out_n, stride_out_k, stride_out_p, stride_out_q,
    # ---- Compile-time constants ----
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implicit GEMM convolution kernel.

    GEMM mapping:
      M axis (rows): flatten (n, p, q) → enumerate all output spatial positions
      N axis (columns): enumerate output channel k
      K axis (reduction): flatten (c, r, s) → enumerate input channels × kernel spatial positions

    Each program instance computes one BLOCK_M × BLOCK_N output tile.
    """

    # ================================================================
    # 1. Program ID → tile coordinates
    # ================================================================
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    # Column-major traversal: neighboring pids access neighboring M tiles → share input data in L2 cache
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # ================================================================
    # 2. M-index decoding → (n, p, q) coordinates
    # ================================================================
    # This is the core of implicit GEMM: reverse-decode flattened indices into tensor coordinates
    n_idx = offs_m // (P * Q)
    residual = offs_m % (P * Q)
    p_idx = residual // Q
    q_idx = residual % Q

    # ================================================================
    # 3. K-dimension iteration (core reduction loop)
    # ================================================================
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # K-index decoding → (c, r, s) coordinates
        c_idx = offs_k // (R * S)
        rs_residual = offs_k % (R * S)
        r_idx = rs_residual // S
        s_idx = rs_residual % S

        # ---- Load input tile A: (BLOCK_M, BLOCK_K) ----
        # Compute input coordinates (n, c, h, w) from (n, p, q) and (c, r, s)
        # Note the broadcasting: p_idx is [BLOCK_M], r_idx is [BLOCK_K]
        h_in = p_idx[:, None] * stride_h + r_idx[None, :] - pad_h
        w_in = q_idx[:, None] * stride_w + s_idx[None, :] - pad_w

        # Boundary check: out-of-bounds positions correspond to zero padding
        valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
        valid = valid & (offs_m[:, None] < M) & (offs_k[None, :] < K_total)

        a_ptrs = (input_ptr
                  + n_idx[:, None] * stride_in_n
                  + c_idx[None, :] * stride_in_c
                  + h_in * stride_in_h
                  + w_in * stride_in_w)
        a = tl.load(a_ptrs, mask=valid, other=0.0)

        # ---- Load weight tile B: (BLOCK_N, BLOCK_K) ----
        b_ptrs = (weight_ptr
                  + offs_n[:, None] * stride_wt_k
                  + c_idx[None, :] * stride_wt_c
                  + r_idx[None, :] * stride_wt_r
                  + s_idx[None, :] * stride_wt_s)
        mask_b = (offs_n[:, None] < K_out) & (offs_k[None, :] < K_total)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # ---- Matrix multiply-accumulate ----
        # a: (BLOCK_M, BLOCK_K) × b^T: (BLOCK_K, BLOCK_N) → (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, tl.trans(b))

    # ================================================================
    # 4. Write back output
    # ================================================================
    out_ptrs = (output_ptr
                + n_idx[:, None] * stride_out_n
                + offs_n[None, :] * stride_out_k
                + p_idx[:, None] * stride_out_p
                + q_idx[:, None] * stride_out_q)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < K_out)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_out)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor,
                  padding: tuple = (0, 0), stride: tuple = (1, 1)) -> torch.Tensor:
    """
    Triton implicit GEMM convolution wrapper.
    input:  [N, C, H, W], FP16, NCHW
    weight: [K, C, R, S], FP16
    Returns: [N, K, P, Q], FP16
    """
    N, C, H, W = input.shape
    K, C_w, R, S = weight.shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    P = (H + 2 * pad_h - R) // stride_h + 1
    Q = (W + 2 * pad_w - S) // stride_w + 1

    output = torch.empty((N, K, P, Q), dtype=torch.float16, device=input.device)
    M = N * P * Q
    N_gemm = K
    K_total = C * R * S

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N_gemm, meta['BLOCK_N']),)

    conv2d_implicit_gemm_kernel[grid](
        input, weight, output,
        N, C, H, W, K, R, S, P, Q,
        pad_h, pad_w, stride_h, stride_w,
        M, N_gemm, K_total,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    return output
```

#### 4.2.1 Key Implementation Details

**Zero-overhead implementation of padding**: when `h_in` or `w_in` is out of bounds, `mask` sets the loaded value to 0. Mathematically this is equivalent to zero padding, but it does not require allocating a padded tensor.

**Design logic behind the autotune configurations**:
- **BLOCK_M × BLOCK_N** determines the compute-to-memory ratio. Large tiles (128×128) are compute-dense; small tiles (32×64) are better for small problems.
- **BLOCK_K = 32** corresponds to two Tensor Core operations' worth of work (K=16×2), balancing pipeline efficiency and register pressure.
- **num_stages** controls pipeline depth. More stages hide longer global-memory latency (~400-800 cycles), but increase SMEM usage.
- **num_warps** affects thread-level parallelism. Large tiles require more warps to fill the compute pipeline.

#### 4.2.2 The Essence of `kred` Alignment

`kred` is exactly `offs_k` in the code—the coordinate on the reduction axis `K_total = C * R * S`, corresponding to the flattened 1D index of `(c, r, s)`. Each advance of the K loop by one `BLOCK_K` gives `offs_k`, the coordinate vector for this tile along the reduction dimension.

A question that often trips up beginners is: **how do `A[m, kred]` and `B[kred, kout]` guarantee that `kred` points to the same `(c,r,s)` tuple?**

The answer becomes clear in three steps.

**Step 1: Write convolution as a dot product**

For a fixed output position $(n, oh, ow)$ and output channel $kout$, convolution is defined as:

$$\text{out}[n, kout, oh, ow] = \sum_{c,r,s} \text{in}[n, c,\; oh \cdot s_h + r - p_h,\; ow \cdot s_w + s - p_w] \cdot w[kout, c, r, s]$$

The right-hand side sums over $(c,r,s)$—**which is exactly the dot product of two vectors of length $C \cdot R \cdot S$**. You can think of it directly as:

> Each output point = "input patch vector" · "convolution kernel vector", both of length $C \cdot R \cdot S$

**Step 2: Flatten M / N / K**

Implicit GEMM flattens two sets of dimensions:

| GEMM axis | Meaning | Flattening rule |
|---------|------|---------|
| M (rows) | Output position $(n, oh, ow)$ | $m = n \cdot OH \cdot OW + oh \cdot OW + ow$ |
| N (columns) | Output channel $kout$ | Direct correspondence, $N_{gemm} = K_{out}$ |
| K (reduction) | Patch index $(c, r, s)$ | $kred = c \cdot R \cdot S + r \cdot S + s$ |

So the two matrices (A does not physically exist; B is the weight matrix) are:

$$A[m, kred] = \text{in}[n, c, h(oh,r), w(ow,s)], \quad B[kred, kout] = w[kout, c, r, s]$$

$$\text{out}[m, kout] = \sum_{kred} A[m, kred] \cdot B[kred, kout]$$

**Step 3: `offs_k` drives the indices for both A and B simultaneously**

Inside the K loop, one key operation happens:

```python
offs_k = k_start + tl.arange(0, BLOCK_K)

# Use the same inverse mapping to recover (c, r, s)
c_idx   = offs_k // (R * S)
r_idx   = offs_k % (R * S) // S
s_idx   = offs_k % S
```

This inverse mapping is the reverse of $kred = c \cdot RS + r \cdot S + s$. Then:

- **A loading** uses `c_idx[None,:]` (broadcast across columns) plus `r_idx/s_idx` to compute `h_in, w_in`, and reads `input[n, c, h, w]`
- **B loading** uses `c_idx[None,:]`, `r_idx[None,:]`, and `s_idx[None,:]` to read `weight[kout, c, r, s]`

Both use **the same `offs_k`, the same inverse mapping, and the same `(c,r,s)`**. The reduction axis of `tl.dot(a, tl.trans(b))` is this same $kred$, so each product term $a[i,j] \cdot b[t,j]$ corresponds exactly to the term with the same $(c_j, r_j, s_j)$ in the convolution formula:

$$\text{in}[n_i,\, c_j,\, h(oh_i, r_j),\, w(ow_i, s_j)] \cdot w[kout_t,\, c_j,\, r_j,\, s_j]$$

When the K loop runs from 0 to `K_total`, it accumulates over all $(c,r,s)$ and produces the complete convolution result.

> **Mental mnemonic**: rows are output positions $(m \leftrightarrow n,oh,ow)$, and K is the patch-vector index $(kred \leftrightarrow c,r,s)$. As long as both sides use the same flattening rule, `kred` is always aligned.

#### 4.2.3 The Triton Mental Model

Once you finish writing a convolution kernel, you can distill the core Triton programming mindset, and it differs from CUDA in one fundamental way:

**CUDA asks, "which element does each thread compute?" Triton asks, "for the tile covered by this program, what are the coordinates of each element?"**

Triton's threads are semi-transparent to the user—you do not manually bind `threadIdx`; the hardware is responsible for distributing the scalar values in the coordinate tensors across warps/lanes. You only need to reason about two things:

1. **What is the shape of each coordinate tensor?**
2. **What logical coordinate in the global tensor does each element represent?**

**Core abstraction: coordinate tensors → pointer tensors**

A Triton kernel is essentially constructing a map from "coordinates → addresses":

```
program_id selects a tile → arange builds coordinate tensors → coordinate tensors multiply by strides → pointer tensors → load/store
```

Every `_ptrs` variable is a concrete instance of this map—its shape exactly matches `acc`, element by element. `a_ptrs` is `[BM, BK]`, `b_ptrs` is `[BN, BK]`, and `out_ptrs` is `[BM, BN]`, matching the accumulator's shape.

**Generic kernel skeleton**

Most dense operators (GEMM, convolution, attention) fit the same overall framework:

```python
# 1. Tile coordinates
pid_m, pid_n = ...
offs_m = pid_m * BM + tl.arange(0, BM)   # [BM]
offs_n = pid_n * BN + tl.arange(0, BN)   # [BN]

# 2. Semantic decoding (plain GEMM skips this; conv/attn map m → (n,oh,ow), etc. here)
...

# 3. Reduction loop
acc = tl.zeros((BM, BN), tl.float32)
for k_start in range(0, K_total, BK):
    offs_k = k_start + tl.arange(0, BK)
    # Reduction-coordinate decoding (conv: kred → (c,r,s); GEMM: use offs_k directly)
    a_ptrs = base_a + ...   # [BM,BK] pointers
    b_ptrs = base_b + ...   # [BK,BN] pointers
    a = tl.load(a_ptrs, mask=..., other=0.)
    b = tl.load(b_ptrs, mask=..., other=0.)
    acc += tl.dot(a, b)

# 4. Write back
tl.store(out_ptrs, acc.to(out_dtype), mask=mask_out)
```

Across operators, the differences lie almost entirely in **the semantic decoding in step 2** and **the coordinate → pointer mapping inside step 3**. The main framework stays the same; only the decoding logic is swapped in and out. Convolution's `kred → (c,r,s)` is one canonical example; attention's causal mask is another.

---
