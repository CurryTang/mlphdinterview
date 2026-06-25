# MLSYS7 · Compute-Bound Kernel (1)

## 1. Introduction: What Is a Compute-Bound Kernel

A compute-bound kernel is one whose performance bottleneck is the throughput of the GPU's compute units (ALU/FPU/Tensor Core), rather than memory bandwidth. Such kernels typically have the following characteristics:

- **High arithmetic intensity**: each byte loaded from memory is used for many floating-point operations
- **Compute-unit utilization is critical**: the optimization goal is to maximize the compute throughput of each SM (Streaming Multiprocessor)
- **Typical examples**: matrix multiplication, convolution, attention, and similar operators

### 1.1 Criteria for Identifying Compute-Bound Kernels

```
Arithmetic intensity = FLOPs / Bytes_Accessed

If arithmetic intensity > GPU peak compute throughput / GPU peak bandwidth
then the kernel is compute-bound
```

Taking the NVIDIA A100 as an example:
- Peak compute throughput (FP32): 19.5 TFLOPS
- Peak bandwidth: 2039 GB/s
- Critical arithmetic intensity ≈ 9.6 FLOPs/Byte

### 1.2 Common Types of Compute-Bound Kernels

| Kernel type | Arithmetic intensity | Typical application scenarios |
|------------|---------|-------------|
| GEMM | O(N) | Fully connected layers, attention projections |
| Convolution | O(K²) | CNN, image processing |
| Self-Attention | O(N) | Transformer |
| Complex activation functions | O(1) but computationally intensive | GELU, Swish, SiLU |
| Fused kernels | Varies widely | Fusion of multiple operators |

---

## 2. Matrix Multiplication (GEMM) Optimization

Matrix multiplication `C = A × B` is the canonical compute-bound kernel and one of the most fundamental operations in deep learning.

### 2.1 Computational Characteristics of GEMM

For `C[M,N] = A[M,K] × B[K,N]`:
- **FLOPs**: 2 × M × N × K
- **Data access**: M×K + K×N + M×N (write-back)
- **Arithmetic intensity**: approximately 2×M×N×K / (M×K + K×N + M×N) ≈ O(min(M,N,K))

When the matrices are large enough, GEMM is almost always compute-bound.

### 2.2 Naive Implementation (Baseline)

```cuda
// Naive GEMM - each thread computes one element of C
__global__ void gemm_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Problems**:
1. Poor global memory access patterns (column-wise accesses to matrix B are not contiguous)
2. No data reuse
3. Each thread computes only one output element, leading to low efficiency

### 2.3 Shared-Memory Tiling Optimization

---

> 💡 **Core Technique: Shared-Memory Tiling**
>
> Tiling is one of the most important general techniques in CUDA optimization, and it appears in almost every high-performance kernel. The core idea is simple: **move data from slow global memory (HBM, ~1 TB/s) into fast shared memory (SRAM, ~19 TB/s), then reuse it many times from the fast memory**.

**Why do we need tiling?** Take GEMM as an example: to compute $C_{ij} = \sum_k A_{ik} B_{kj}$, the naive implementation has each thread independently load one row of A and one column of B from global memory. But neighboring threads often need the same row of A or the same column of B, so the same data is fetched repeatedly with no reuse. Tiling fixes this by having all threads in a block **cooperatively** load a small tile of A and B into shared memory, after which each thread reads from shared memory to do its computation. Each data item can then be shared by `TILE_SIZE` threads, reducing global-memory traffic by roughly a factor of `TILE_SIZE`.

**Tiling’s general pattern (three steps)**:

```
1. Collaborative load: all threads in the block each handle part of the work, moving the tile from global memory → shared memory
2. Synchronization barrier: `__syncthreads()` ensures all threads have finished loading
3. Local computation: each thread reads data from shared memory and computes on it (hitting SRAM, latency ~20 cycles vs. HBM ~400 cycles)
```

**How tiling is used in different scenarios** (see [MLSYS6.md](notes/Mlsys/MLSYS6.md) for a more detailed discussion):

| Scenario | Role of the tile | Key details |
|------|-----------|---------|
| **GEMM** (this section) | Reuse rows of A and columns of B | Each data item is reused by `TILE_SIZE` threads |
| **Matrix transpose** (MLSYS6 Pattern 2) | Intermediate buffer for orthogonal read/write directions | Coalesced read → tile → coalesced write; `tile[DIM][DIM+1]` padding removes bank conflicts |
| **Stencil/Convolution** (MLSYS6 Pattern 3) | Reuse neighborhood data plus halo regions | The tile must additionally load halo boundaries of ±R; reuse factor = 2R+1 (stencil width) |
| **Histogram** (MLSYS6 Principle D) | Private bins reduce atomic contention | Each block maintains a private histogram in shared memory, then merges it into global memory |

**Performance analysis**: Suppose the tile size is $T$, and each original element would otherwise be read $R$ times (the reuse factor). With tiling, global-memory traffic ideally drops to $\frac{1}{R}$ of the original. For GEMM, $R = T$ (`TILE_SIZE`); for stencil, $R = 2 \times \text{radius} + 1$. **Larger tiles give higher reuse, but they are limited by shared-memory capacity** (typically 48-228 KB per SM). This is the central trade-off in choosing a tile size: too small → insufficient reuse; too large → too much shared memory consumption or lower occupancy.

**Common advanced tiling techniques** (used in later sections):
- **Double buffering** (Section 2.5): alternate between two tile buffers, one being computed on and one being prefetched, to hide memory latency
- **Register Blocking** (Section 2.4): On top of shared memory tiling, load a small piece of the tile into the register to form a three-level hierarchy of global → shared → register
- **Padding** (section 2.5): `tile[DIM][DIM + padding]` avoids bank conflict at the cost of occupying a small amount of shared memory
- **Vectorized loading** (Section 2.4): Use `float4` to load 128 bits at a time, reducing the number of loading instructions and improving bandwidth utilization

---

```cuda
#define TILE_SIZE 32

__global__ void gemm_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // iterate over all tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // collaboratively load the A and B tiles into shared memory
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute the partial sum
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Optimization points**:
1. Use shared memory to achieve data reuse
2. Each data is reused TILE_SIZE times
3. Reduce the number of global memory accesses

### 2.4 Register blocking + vectorized access

```cuda
#define BM 128  // Block tile size in M dimension
#define BN 128  // Block tile size in N dimension
#define BK 8    // Block tile size in K dimension
#define TM 8    // Thread tile size in M dimension
#define TN 8    // Thread tile size in N dimension

__global__ void gemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // shared-memory declaration
    __shared__ float As[BK][BM];  // Stored transposed to avoid bank conflicts
    __shared__ float Bs[BK][BN];
    
    // Compute the block and thread positions
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // each block has (BM/TM) × (BN/TN) threads
    const int thread_x = tx;
    const int thread_y = ty;
    
    // Registers store the thread's computed results
    float thread_results[TM][TN] = {0.0f};
    
    // Registers store fragments of A and B
    float reg_a[TM];
    float reg_b[TN];
    
    // Starting positions of A and B
    const float* A_ptr = A + by * BM * K;
    const float* B_ptr = B + bx * BN;
    
    // Compute the load indices
    const int num_threads = (BM / TM) * (BN / TN);
    const int thread_id = ty * (BN / TN) + tx;
    
    // Main loop: iterate over the K dimension
    for (int k_base = 0; k_base < K; k_base += BK) {
        // ============ Collaboratively load the A tile into shared memory ============
        // use vectorized loads (float4)
        #pragma unroll
        for (int i = 0; i < BM * BK / (num_threads * 4); ++i) {
            int load_idx = thread_id + i * num_threads;
            int load_row = load_idx / (BK / 4);
            int load_col = (load_idx % (BK / 4)) * 4;
            
            if (by * BM + load_row < M && k_base + load_col < K) {
                float4 tmp = *reinterpret_cast<const float4*>(
                    &A_ptr[load_row * K + k_base + load_col]
                );
                As[load_col + 0][load_row] = tmp.x;
                As[load_col + 1][load_row] = tmp.y;
                As[load_col + 2][load_row] = tmp.z;
                As[load_col + 3][load_row] = tmp.w;
            }
        }
        
        // ============ Collaboratively load the B tile into shared memory ============
        #pragma unroll
        for (int i = 0; i < BK * BN / (num_threads * 4); ++i) {
            int load_idx = thread_id + i * num_threads;
            int load_row = load_idx / (BN / 4);
            int load_col = (load_idx % (BN / 4)) * 4;
            
            if (k_base + load_row < K && bx * BN + load_col < N) {
                float4 tmp = *reinterpret_cast<const float4*>(
                    &B_ptr[(k_base + load_row) * N + load_col]
                );
                reinterpret_cast<float4*>(&Bs[load_row][load_col])[0] = tmp;
            }
        }
        
        __syncthreads();
        
        // ============ Compute the thread's tile ============
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // load from shared memory into registers
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_a[m] = As[k][thread_y * TM + m];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                reg_b[n] = Bs[k][thread_x * TN + n];
            }
            
            // Outer-product computation
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    thread_results[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // ============ Write back the result ============
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int global_row = by * BM + thread_y * TM + m;
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int global_col = bx * BN + thread_x * TN + n;
            if (global_row < M && global_col < N) {
                float4 result;
                result.x = thread_results[m][n + 0];
                result.y = thread_results[m][n + 1];
                result.z = thread_results[m][n + 2];
                result.w = thread_results[m][n + 3];
                *reinterpret_cast<float4*>(&C[global_row * N + global_col]) = result;
            }
        }
    }
}
```

### 2.5 Eliminating Bank Conflicts

Shared memory is organized into 32 banks, with each consecutive 4-byte word (one `float`) mapped to the next bank. A bank conflict occurs when multiple threads in the same warp access different addresses in the same bank.

```cuda
// Access pattern with bank conflicts
__shared__ float smem[32][32];
// Thread `i` accesses `smem[i][k]`; there is a conflict when all threads access the same column

// Solution 1: padding
__shared__ float smem[32][33];  // add one padding column

// Solution 2: interleaved access (swizzle)
__shared__ float smem[32][32];
// when accessing：smem[row][col ^ (row % 32)]
```

## 3. CUTLASS and Triton programming paradigms

In the first two chapters, we built a CUDA GEMM kernel from scratch and progressively introduced optimization techniques such as shared-memory tiling, register blocking, vectorized memory access, and bank-conflict elimination. These low-level optimizations give us a deep understanding of the GPU execution model. In production, however, writing CUDA kernels entirely by hand comes with major challenges. This chapter introduces two mainstream high-level programming paradigms—NVIDIA CUTLASS and OpenAI Triton—which offer more efficient and maintainable ways to develop GPU kernels through C++ template metaprogramming and compiler-driven Python code, respectively.

---

### 3.1 Why high-level abstraction is needed

#### 3.1.1 The Challenge of Handwritten CUDA

Looking back on the GEMM optimization journey in the first two chapters, we had to manually manage all the following details:

- **Shared-memory management**: manually calculate the required size, declare `__shared__` arrays, handle pointer swapping for double buffering, and place `__syncthreads()` correctly. A single missing synchronization can lead to a race condition that is very hard to reproduce.
- **Register tiling**: manually decompose warp-level computation into per-thread register fragments, precisely determine which elements each thread owns, and keep register usage under hardware limits (otherwise register spilling can destroy performance).
- **Bank-conflict elimination**: shared memory is divided into 32 banks, and conflicts occur when multiple threads in the same warp access different addresses in the same bank. Avoiding this requires manual padding or access-pattern changes, which hurts readability.
- **Vectorized memory access**: to fully utilize memory bandwidth, developers must use vector types such as `float4` and `int4` for 128-bit accesses, while also handling pointer alignment and type conversion by hand.
- **Instruction-level optimization**: manually unroll loops (`#pragma unroll`), interleave compute and load instructions to hide latency, and choose the right `mma` instruction variants.

#### 3.1.2 Problems with cross-architecture migration

A more serious problem is that the optimal implementation strategy differs fundamentally between different GPU architectures:

| Features | Volta (SM70) | Ampere (SM80) | Hopper (SM90) |
|------|-------------|---------------|---------------|
| Tensor Core instructions | `wmma` | `mma` | `wgmma` |
| Optimal tile size | 128x128x32 | 128x256x64 | 256x256x64 |
| Asynchronous copy | Not supported | `cp.async` | `cp.async.bulk` (TMA) |
| Software-pipeline stages | 2 | 3-4 | 4-8 |
| Shared memory size | 96 KB | 164 KB | 228 KB |
| Cluster scheduling | Not supported | Not supported | Thread Block Cluster |

A carefully tuned Ampere kernel may fail to exploit Hopper-specific features such as TMA, `wgmma`, and thread-block clusters; its performance may even fall behind a general-purpose library. In practice, each new GPU generation often requires the kernel to be reworked almost from scratch, which makes maintenance extremely costly.

#### 3.1.3 Requirements for composable abstraction

What the industry needs is a set of composable abstractions that separate algorithmic logic from hardware mapping:

- **Algorithm level**: define mathematical semantics such as "GEMM = three nested loops + accumulation + epilogue"
- **Scheduling level**: define strategies such as tile size, pipeline depth, and warp allocation
- **Hardware level**: map the computation onto specific instructions (`mma` vs `wgmma`), memory levels (shared vs register), and synchronization mechanisms

CUTLASS and Triton represent two different philosophies: CUTLASS composes these abstraction layers at compile time through C++ templates, while Triton relies on the compiler to derive the low-level implementation automatically.

---

### 3.2 CUTLASS Design Philosophy and Core Abstractions

[CUTLASS](https://github.com/NVIDIA/cutlass) (CUDA Templates for Linear Algebra Subroutines) is NVIDIA’s open-source C++ template library. It provides a set of hierarchical, composable building blocks for GEMM and related operations. Its core idea is to decompose GEMM into layers using C++ template metaprogramming, so developers can customize a kernel by composing template parameters instead of writing everything from scratch.

#### 3.2.1 Hierarchical Decomposition

CUTLASS decomposes GEMM into four layers, each corresponding to a level of execution in GPU hardware:

**Device Level (Grid of Threadblocks)**

The full GEMM problem $D = \alpha \cdot A \times B + \beta \cdot C$ is partitioned into a 2D grid of thread blocks. Each thread block computes one tile of the output matrix $D$ (typically 128x128 or 256x128). Thread blocks are fully independent and require no inter-block synchronization.

CUTLASS provides **Swizzle** strategies to control how thread blocks map onto output tiles and thereby improve L2-cache locality. For example, `GemmIdentityThreadblockSwizzle<8>` maps groups of 8 neighboring thread blocks to neighboring output tiles, increasing their reuse of A or B matrix data.

**Threadblock Level (Shared Memory Tiling)**

Each thread block cooperatively loads A and B tiles from global memory into shared memory, then iterates over the K dimension one K-tile at a time. The key technique at this level is **software pipelining**: while computing on the current K-tile, the kernel asynchronously prefetches the next one. CUTLASS controls pipeline depth through the `num_stages` parameter.

**Warp Level (Tensor Core MMA)**

Tiles in shared memory are further partitioned across warps. Each warp uses Tensor Core MMA (Matrix Multiply-Accumulate) instructions for computation. On Ampere, for example, `mma.sync` computes a small matrix multiplication such as 16x8x16 per instruction. CUTLASS encapsulates this warp-level tiling and MMA invocation through the `MmaWarp` abstraction.

**Thread Level (Epilogue)**

After all K-tile iterations are complete, each thread holds part of the result in accumulators. The epilogue stage writes these accumulated values from registers back to global memory, while also performing post-processing such as scaling ($\alpha, \beta$), bias addition, and activation functions (ReLU, GELU).

#### 3.2.2 Key Abstractions

**1. Layout——Tensor Memory Layout**

Layout describes how a logical multi-dimensional tensor is mapped into one-dimensional physical memory. CUTLASS supports multiple layouts:

```cpp
// Row-major：logical coordinates (i, j) → physical offset i * stride + j
using LayoutA = cutlass::layout::RowMajor;

// Column-major：logical coordinates (i, j) → physical offset i + j * stride
using LayoutB = cutlass::layout::ColumnMajor;

// Convolution-specific layout
using LayoutNHWC = cutlass::layout::TensorNHWC;

// Interleaved layout (for INT8/INT4 quantized inference)
using LayoutInterleaved = cutlass::layout::ColumnMajorInterleaved<32>;
```

The choice of layout directly affects memory-access contiguity, which in turn affects the feasibility of vectorized loads and the bank-conflict pattern in shared memory. CUTLASS iterators automatically generate an optimized memory-access instruction sequence based on the selected layout.

**2. TileDescription——Tile Size Description**

Tile size is one of the most important tuning parameters for GEMM performance. In CUTLASS, it is specified at three levels using the `GemmShape` template:

```cpp
// Threadblock level: each threadblock computes a 128x128 output tile, with K=32 per iteration
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;

// Warp level: each warp computes a 64x64 sub-tile
// Therefore each threadblock has `(128/64) * (128/64) = 4` warps
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;

// Instruction level: the shape of the Tensor Core MMA instruction
// Ampere mma.sync.aligned.m16n8k16.f32.f16.f16.f32
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
```

There are divisibility constraints across these three levels:
- `ShapeMMAThreadBlock::kM` must be divisible by `ShapeMMAWarp::kM`
- `ShapeMMAWarp::kM` must be divisible by `ShapeMMAOp::kM`
- The K dimension follows the same rule

Violating these constraints causes a compile-time error (`static_assert` in CUTLASS).

**3. Epilogue——Post-Processing Operations**

One major strength of CUTLASS is its templated epilogue, which allows common post-processing to be fused directly into the GEMM kernel, avoiding extra kernel launches and additional global-memory round-trips:

```cpp
// Linear combination + ReLU activation：D = max(0, alpha * A@B + beta * C)
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    float,                                    // output element type
    128 / cutlass::sizeof_bits<float>::value,  // number of elements per vectorized access (=4)
    float,                                    // accumulator type
    float                                     // compute type
>;

// Linear combination + GELU activation
using EpilogueGelu = cutlass::epilogue::thread::LinearCombinationGELU<
    cutlass::half_t, 8, float, float
>;

// Pure linear combination (no activation)：D = alpha * A@B + beta * C
using EpilogueLinear = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 8, float, float
>;
```

Epilogue fusion is a major advantage of CUTLASS over bare cuBLAS: cuBLAS GEMM can only produce $D = \alpha AB + \beta C$, and any subsequent activation must run in a separate kernel. CUTLASS, by contrast, can fuse those activations into the GEMM write-back path and complete everything in a single pass.

**4. Iterator——Memory-Access Pattern Abstraction**

The iterator is one of the most complex yet most important abstractions in CUTLASS. It encapsulates data movement from global memory to shared memory and from shared memory to registers:

- **Global-memory iterator**: chooses the best vectorized load method (such as `LDG.128`) based on tensor layout and alignment, and handles boundary conditions automatically when a tile crosses matrix boundaries
- **Shared-memory iterator**: generates a bank-conflict-free shared-memory access pattern and works together with swizzle strategies to reorganize data layout
- **Asynchronous copy**: on Ampere and later, uses `cp.async` to copy directly from global memory to shared memory asynchronously, bypassing registers

Developers usually do not manipulate iterators directly; instead, they configure them indirectly by choosing predefined strategy classes.


#### 3.2.3 CUTLASS 3.x and CuTe

CUTLASS 3.x introduces a new abstraction called **CuTe** (CUDA Tensors), representing a fundamental redesign of CUTLASS 2.x. Although the abstraction system in CUTLASS 2.x (`GemmShape`, `Iterator`, `Policy`, etc.) is powerful, it has two fundamental issues:

1. **Concept fragmentation**: Data layout, thread mapping, and memory access patterns are scattered in different abstractions, and their relationships are hidden in complex template specializations.
2. **Combinatorial explosion**: each new MMA instruction, memory-access mode, or layout requires a large amount of new specialized code

CuTe's core insight is this: **nearly every problem in GPU programming—data layout, thread assignment, tiling strategy, memory-access pattern—is fundamentally the same problem: mapping integer coordinates to integer offsets. CuTe expresses all of this using a unified layout algebra.**

##### Layout = (Shape, Stride)——CuTe's First Principles

A layout is fully defined by two tuples: **Shape** and **Stride**. Given a logical coordinate, the layout maps it to a 1D physical offset:

```
offset = coord[0] * stride[0] + coord[1] * stride[1] + ...
```

```cpp
using namespace cute;

// 4x8 row-major matrix: coordinates `(i,j)` → offset `i*8 + j`
auto row_major = make_layout(make_shape(4, 8), make_stride(8, 1));
// row_major(0, 0) = 0,  row_major(0, 1) = 1,  row_major(1, 0) = 8

// 4x8 column-major matrix: coordinates `(i,j)` → offset `i + j*4`
auto col_major = make_layout(make_shape(4, 8), make_stride(1, 4));
// col_major(0, 0) = 0,  col_major(0, 1) = 4,  col_major(1, 0) = 1
```

Key point: Shape and Stride can themselves be hierarchical—the elements of a tuple can also be tuples. This lets CuTe use the same syntax to describe everything from simple row-major/column-major layouts to complex blocked layouts:

```cpp
// Hierarchical layout: a `(4,8)` matrix is divided into `2x4` small `(2,2)` blocks
// Shape:  ((2,2), (2,4))  —— the outer level is coordinates within a tile, the inner level is the number of tiles
// Stride: ((1,2), (4,16)) —— corresponding strides
auto tiled_layout = make_layout(
    make_shape(make_shape(2, 2), make_shape(2, 4)),
    make_stride(make_stride(1, 2), make_stride(4, 16))
);
// Equivalent to RowMajor for a 4x8 matrix, but it "knows" that the matrix is partitioned into 2x2 tiles
```

This is crucial: tiling is not a data-movement operation, but a reinterpretation of layout. Operations such as `local_tile` and `logical_divide` merely reorganize the nested structure of Shape and Stride without moving data, and therefore add zero runtime overhead.

##### Tensor = Pointer + Layout

In CuTe, a Tensor is simply a pointer plus a Layout:

```cpp
// Construct a tensor from a global-memory pointer
auto tensor_A = make_tensor(make_gmem_ptr(A_ptr),
    make_layout(make_shape(M, K), make_stride(K, Int<1>{})));

// Construct a tensor from shared memory
auto smem_A = make_tensor(make_smem_ptr(smem_ptr),
    make_layout(make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})));

// Construct a tensor from registers (a register is also a kind of "memory")
auto reg_C = make_tensor<float>(make_shape(Int<8>{}, Int<4>{}));  // allocate in registers
```

Note the compile-time constants such as `Int<128>{}`. CuTe relies heavily on compile-time shape/stride information, allowing the compiler to resolve address calculations ahead of time so that the generated PTX contains only the actual data-movement and compute instructions.

##### Layout Algebra—Core Operations

The power of CuTe lies in a set of **closed layout-transformation operations**. Each operation takes layouts as input and produces a layout as output, so they can be composed freely:

**1. `logical_divide`——Logical Division**

This splits a layout according to a specified tile shape. The result is a hierarchical layout whose outer index refers to coordinates within a tile and whose inner index refers to the tile number:

```cpp
auto layout = make_layout(make_shape(16, 32));  // 16x32 matrix

// Tile by `(4, 8)` → resulting shape: ((4,8), (4,4))
//   first dimension (4,8)：row/column coordinates within a tile
//   second dimension (4,4)：tile row/column indices in the matrix
auto tiled = logical_divide(layout, make_shape(4, 8));
```

**2. `local_tile`——Extracting a Specific Tile**

After `logical_divide`, you can use coordinates to extract a specific tile:

```cpp
// Take tile `(2, 1)`：extract the 4x8 submatrix covering rows 8~11 and columns 8~15 from the 16x32 matrix
auto tile_2_1 = local_tile(tensor_A,
    make_shape(Int<4>{}, Int<8>{}),   // tile shape
    make_coord(2, 1));                // tile coordinates
// `tile_2_1` has shape `(4, 8)` and points to the corresponding region of the original data
```

**3. `composition`——Layout Composition**

Two layouts can be combined into a new layout, mathematically corresponding to function composition $L_1 \circ L_2$:

```cpp
// L2 maps logical coordinates to intermediate indices, and L1 then maps those intermediate indices to physical offsets
auto composed = composition(L1, L2);
// composed(coord) = L1(L2(coord))
```

This is the basis for implementing **swizzle**: swizzle is just an XOR transformation composed on top of a base layout.

**4. `complement`——Complementary Layout**

Given a layout that maps thread IDs to data elements ("which elements each thread accesses"), `complement` computes the layout of the remaining uncovered elements ("which elements still need to be covered"). This is important when constructing thread-value (TV) decompositions.

##### Thread-Value (TV) Decomposition - A Unified Framework for Thread Mapping

One of the core questions in GPU programming is: **if 128 threads cooperate to process a 128x32 tile, which elements should each thread handle?**

CUTLASS 2.x uses separate mapping logic for each case (GMEM loads, SMEM writes, MMA operations). CuTe unifies them through the concept of a **TV Layout**.

A TV Layout is a mapping `(Thread, Value)` → `(M, K)`:
- **Thread dimension**: thread ID (0 ~ NumThreads-1)
- **Value dimension**: the indices of the multiple elements handled by each thread

```cpp
// A vectorized loading strategy for 128 threads: each thread uses `LDG.128` to load 8 FP16 elements
// In total this handles `128 * 8 = 1024` elements = one row of a `128x8` tile
auto copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{};

// Construct `TiledCopy`: 128 threads arranged as a `(32, 4)` thread grid
// Each thread handles `(1, 8)` elements per operation (`LDG.128` = 8 FP16 values)
auto tiled_copy = make_tiled_copy(
    copy_atom,
    Layout<Shape<_32, _4>>{},    // Thread layout: 32x4 thread arrangement
    Layout<Shape<_1, _8>>{}      // Value layout: 1x8 elements per thread
);
// Total covered area: (32*1, 4*8) = (32, 32) —— which is exactly one 32x32 SMEM tile
```

Likewise, MMA operations also have a TV decomposition:

```cpp
// Ampere mma.sync.m16n8k16 TV decomposition
auto mma_atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{};

// Expand into `TiledMma`: 4 warps (128 threads) covering a `(64, 64, 16)` tile
auto tiled_mma = make_tiled_mma(
    mma_atom,
    Layout<Shape<_2, _2, _1>>{},    // Atom layout: 2x2x1 atom arrangement
    Tile<_64, _64, _16>{}           // target tile size
);
// Each call computes (64, 64) @ (64, 16)^T a partial result of
```

The advantage of TV decomposition is **consistency**: whether the operation is `cp.async` from GMEM→SMEM, `ldmatrix` from SMEM→registers, or `mma.sync`, it can all be described as a `(Thread, Value) → (M, K/N)` layout. That makes it much easier to verify that data mappings line up across stages—just check whether the corresponding TV layouts are compatible.

##### Swizzle - Using Layout Algebra to Eliminate Bank Conflicts

Shared memory has 32 banks, with bank ID = `(address / 4 bytes) % 32`. A bank conflict occurs when multiple threads in the same warp access the same bank.

CuTe uses the **Swizzle** transformation to eliminate bank conflicts. At heart, this is just a bitwise transformation. In CuTe, Swizzle is parameterized as `Swizzle<B, M, S>`:

```
swizzled_offset = offset ^ ((offset >> S) & mask)
where mask = ((1 << B) - 1) << M
```

- `B`: number of bits participating in the XOR (the swizzle "width")
- `M`: starting bit position of the mask
- `S`: right-shift amount (which higher bits are XORed into the lower bits)

```cpp
// Common shared-memory swizzle configuration
// For FP16 (2 bytes), a 32x32 SMEM tile:
auto smem_layout = composition(
    Swizzle<3, 3, 3>{},                         // XOR transformation
    make_layout(make_shape(Int<32>{}, Int<32>{}),
                make_stride(Int<32>{}, Int<1>{}))  // Base row-major layout
);
// Swizzle<3,3,3> means: XOR the low 3 bits of the row with the high 3 bits of the column
// Effect: adjacent rows in the same column map to different banks → eliminating bank conflicts for column accesses
```

This is much cleaner than manually writing swizzle specializations in CUTLASS 2.x: swizzle is just another layout `composition`, and it works seamlessly with all other layout operations.

#### 3.2.4 Case Study: In-Depth Analysis of CuTe GEMM

We have introduced CuTe's core concepts—layout algebra, TV decomposition, and swizzle—but how do they come together in a complete high-performance GEMM kernel? This section walks through a production-grade CuTe GEMM implementation to show the engineering philosophy of CUTLASS 3.x, from end-to-end data flow down to the intent behind individual lines of code.
##### Complete code

```cpp
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

using namespace cute;

// ==================== Configuration parameters ====================
struct GemmConfig {
    using T = half_t;
    using AccT = float;

    // Three-level tiling parameters
    static constexpr int kBlockM  = 128;   // CTA tile M
    static constexpr int kBlockN  = 128;   // CTA tile N
    static constexpr int kBlockK  = 32;    // CTA tile K（K depth processed per main-loop iteration）
    static constexpr int kStages  = 3;     // pipeline depth

    // MMA atom: Ampere mma.sync.aligned.m16n8k16.f32.f16.f16.f32
    // TN means A enters MMA as row-major (T) and B as column-major (N)
    using MMAOp = SM80_16x8x16_F32F16F16F32_TN;

    // TiledMMA: Expand the MMA atom to cover the entire CTA tile
    //   Layout<Shape<_2, _2, _1>>: 4 warps: 2 along M and 2 along N
    //   Tile<_128, _128, _16>: targeting a 128×128×16 subproblem
    //   → each warp handles 64×64×16
    using TiledMMA = decltype(
        make_tiled_mma(MMAOp{},
                       Layout<Shape<_2, _2, _1>>{},
                       Tile<_128, _128, _16>{})
    );

    // G→S: cp.async 128-bit (CACHEGLOBAL policy, without polluting L1)
    using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>;
    // S→R: ldmatrix instruction (SMEM→register transfer designed specifically for Tensor Cores)
    using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using S2RCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

    // Thread count = the number of threads required by `TiledMMA` = 4 warps × 32 = 128
    static constexpr int kThreadNum = size(TiledMMA{});
};

// ==================== Shared-memory layout ====================
static constexpr int kBlockM  = GemmConfig::kBlockM;
static constexpr int kBlockN  = GemmConfig::kBlockN;
static constexpr int kBlockK  = GemmConfig::kBlockK;
static constexpr int kStages  = GemmConfig::kStages;

// SmemLayoutAtom: a swizzled row-major layout for 128×32
// Swizzle<3,3,3> eliminates bank conflicts for `ldmatrix` column accesses
using SmemLayoutAtom = decltype(
    composition(Swizzle<3, 3, 3>{},
                make_layout(make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                            make_stride(Int<kBlockK>{}, Int<1>{})))
);
//Dimension 0：kBlockM
//Size of the CTA A-tile along the M dimension (number of rows).

//Dimension 1：kBlockK
//Size of the K chunk consumed by the CTA in each main-loop iteration (number of columns / K-slice width).

//Dimension 2：kStages
//Pipeline-stage dimension: there are `kStages` mutually independent buffers (commonly 2 or 3), each storing one swizzled A tile. 
using SmemLayoutA = decltype(
    tile_to_shape(SmemLayoutAtom{},
                  make_shape(Int<kBlockM>{}, Int<kBlockK>{}, Int<kStages>{}))
);
using SmemLayoutB = decltype(
    tile_to_shape(SmemLayoutAtom{},
                  make_shape(Int<kBlockN>{}, Int<kBlockK>{}, Int<kStages>{}))
);

// ==================== Kernel implementation ====================
template <typename Config>
__global__ void cute_gemm_kernel(
    const void* __restrict__ Aptr,
    const void* __restrict__ Bptr,
    void* __restrict__ Cptr,
    int M, int N, int K
) {
    using T    = typename Config::T;
    using AccT = typename Config::AccT;

    constexpr int kBlockM = Config::kBlockM;
    constexpr int kBlockN = Config::kBlockN;
    constexpr int kBlockK = Config::kBlockK;
    constexpr int kStages = Config::kStages;

    int tid = threadIdx.x;

    // ---- Global-memory tensors ----
    // A: (M, K) row-major → stride = (K, 1)
    Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(Aptr)),
                            make_shape(M, K), make_stride(K, Int<1>{}));
    // B: (N, K) row-major → the host side applies `B.t().contiguous()`, matching the TN convention
    Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(Bptr)),
                            make_shape(N, K), make_stride(K, Int<1>{}));
    // C: (M, N) row-major → stride = (N, 1)
    Tensor mC = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(Cptr)),
                            make_shape(M, N), make_stride(N, Int<1>{}));

    // ---- CTA tiling: extract the submatrices owned by the current block from the global matrices ----
    // `cta_tiler` describes the tile sizes along all three dimensions
    auto cta_tiler = make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
    // blockIdx.y → M direction，blockIdx.x → N direction，_ → K direction (kept entirely and traversed by the main loop)
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
    // Step<_1, X, _1>: take the tile along the M and K dimensions from `A(M,K)`, skipping N
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    // Step<X, _1, _1>: take the tile along the N and K dimensions from `B(N,K)`, skipping M
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    // Step<_1, _1, X>: take the tile along the M and N dimensions from `C(M,N)`, skipping K
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    // ---- Shared-memory allocation ----
    extern __shared__ char smem_buf[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_buf)),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(
                            smem_buf + cosize(SmemLayoutA{}) * sizeof(T))),
                            SmemLayoutB{});

    // ---- G→S Copy: use 128-bit `cp.async` to move data from global memory to shared memory ----
    auto g2s_copy = make_tiled_copy(
        typename Config::G2SCopyAtom{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},   // 128 128 threads arranged as 32×4
        Layout<Shape< _1, _8>>{}                     // each thread moves 1×8 `half_t` values
    );
    auto g2s_thr_copy_A = g2s_copy.get_slice(tid);
    Tensor tAgA_g = g2s_thr_copy_A.partition_S(gA);  // this thread's GMEM view
    Tensor tAsA_s = g2s_thr_copy_A.partition_D(sA);  // this thread's SMEM view

    auto g2s_thr_copy_B = g2s_copy.get_slice(tid);
    Tensor tBgB_g = g2s_thr_copy_B.partition_S(gB);
    Tensor tBsB_s = g2s_thr_copy_B.partition_D(sB);

    // ---- TiledMMA + register fragments ----
    typename Config::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // partition_fragment_A/B: allocate MMA operand storage in registers
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));   // A register fragment
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));   // B register fragment
    Tensor tCrC = thr_mma.partition_fragment_C(gC);            // C FP32 accumulator
    clear(tCrC);  // initialize to 0

    // ---- S→R Copy: use `ldmatrix` to move data from shared memory into registers ----
    // make_tiled_copy_A: automatically construct an S→R copy compatible with `TiledMMA`
    auto s2r_copy_A = make_tiled_copy_A(typename Config::S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_slice(tid);
    Tensor tCsA = s2r_thr_copy_A.partition_S(sA);       // copy view in SMEM
    Tensor tCrA_view = s2r_thr_copy_A.retile_D(tCrA);   // copy view in registers
    // `tCrA` and `tCrA_view` are two "views" of the same registers

    auto s2r_copy_B = make_tiled_copy_B(typename Config::S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_slice(tid);
    Tensor tCsB = s2r_thr_copy_B.partition_S(sB);
    Tensor tCrB_view = s2r_thr_copy_B.retile_D(tCrB);

    // ==================== Software pipeline ====================
    int num_k_tiles = ceil_div(K, kBlockK);
    constexpr int num_mma_k = size<2>(tCrA);   // kBlockK / MMA_K = 32/16 = 2

    // ---- Pipeline Fill: preload the first `(kStages-1)` tiles ----
    CUTE_UNROLL
    for (int stage = 0; stage < kStages - 1; ++stage) {
        if (stage < num_k_tiles) {
            copy(g2s_copy, tAgA_g(_, _, _, stage), tAsA_s(_, _, _, stage));
            copy(g2s_copy, tBgB_g(_, _, _, stage), tBsB_s(_, _, _, stage));
        }
        cp_async_fence();   // treat each stage as an independent fence group
    }

    // ---- Main Loop: compute while loading ----
    int smem_pipe_read  = 0;              // stage currently being computed
    int smem_pipe_write = kStages - 1;    // stage to write next

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // Step 1: wait for the current stage data to be ready
        cp_async_wait<kStages - 2>();     // wait<1>: allow 1 group to remain in flight
        __syncthreads();

        // Step 2: asynchronously load the next tile (if any)
        int next_tile = k_tile + kStages - 1;
        if (next_tile < num_k_tiles) {
            copy(g2s_copy, tAgA_g(_, _, _, next_tile),
                           tAsA_s(_, _, _, smem_pipe_write));
            copy(g2s_copy, tBgB_g(_, _, _, next_tile),
                           tBsB_s(_, _, _, smem_pipe_write));
        }
        cp_async_fence();

        // Step 3: S→R (ldmatrix) + MMA compute
        CUTE_UNROLL
        for (int k_inner = 0; k_inner < num_mma_k; ++k_inner) {
            copy(s2r_copy_A, tCsA(_, _, k_inner, smem_pipe_read),
                             tCrA_view(_, _, k_inner));
            copy(s2r_copy_B, tCsB(_, _, k_inner, smem_pipe_read),
                             tCrB_view(_, _, k_inner));
        }
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);   // tCrC += tCrA @ tCrB

        // Step 4: advance the ring buffer
        smem_pipe_read  = (smem_pipe_read  + 1) % kStages;
        smem_pipe_write = (smem_pipe_write + 1) % kStages;
        __syncthreads();
    }

    // ==================== Epilogue: FP32 → FP16, write back to GMEM ====================
    Tensor tCgC = thr_mma.partition_C(gC);            // C the current thread's GMEM view
    Tensor tCrC_out = make_tensor_like<T>(tCrC);      // allocate FP16 registers
    copy(tCrC, tCrC_out);                             // FP32 → FP16 truncating conversion
    copy(AutoVectorizingCopy{}, tCrC_out, tCgC);      // automatically vectorized writeback
}

// ==================== Host-side PyTorch interface ====================
torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().type() == torch::kCUDA, "A must be on CUDA");
    TORCH_CHECK(B.device().type() == torch::kCUDA, "B must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // B: (K,N) row-major → Bt: (N,K) row-major，matches the TN convention
    auto Bt = B.t().contiguous();
    auto C = torch::empty({M, N}, A.options());

    using Config = GemmConfig;

    dim3 block(Config::kThreadNum);   // 128 threads
    dim3 grid(
        (N + Config::kBlockN - 1) / Config::kBlockN,   // N direction
        (M + Config::kBlockM - 1) / Config::kBlockM    // M direction
    );

    // Shared memory: A(24KB) + B(24KB) = 48KB
    constexpr int smem_size =
        cosize(SmemLayoutA{}) * sizeof(half_t) +
        cosize(SmemLayoutB{}) * sizeof(half_t);

    // If it exceeds 48KB, an explicit request is required
    if constexpr (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            cute_gemm_kernel<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    cute_gemm_kernel<Config><<<grid, block, smem_size>>>(
        A.data_ptr(), Bt.data_ptr(), C.data_ptr(), M, N, K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
```

```
[Kernel-1] Construct global views mA(M,K), mB(N,K), mC(M,N)（gmem_ptr + row-major layout）
[Kernel-2] CTA CTA tiling: use `blockIdx` to select the current CTA's global tile views gA(BM,BK,ktile), gB(BN,BK,ktile), gC(BM,BN)
[Kernel-3] Dynamic shared-memory partitioning：sA(BM,BK,stages), sB(BN,BK,stages)（smem_ptr + swizzled SmemLayout）
[Kernel-4] Construct `G2S tiled_copy` (`cp.async` 128b) and partition it per thread: `tAgA_g`/`tAsA_s` and `tBgB_g`/`tBsB_s` (thread-local GMEM/SMEM copy views)
[Kernel-5] Construct `tiled_mma` (`mma.sync`) and allocate register fragments per thread：tCrA,tCrB,tCrC；clear(tCrC)
[Kernel-6] Construct `S2R tiled_copy_A/B` (`ldmatrix`) and partition them per thread: `tCsA`/`tCsB` (SMEM read views) and `tCrA_view`/`tCrB_view = retile_D(tCrA/tCrB)` (register write views)
[Kernel-7] Pipeline warm-up: for `stage in [0..kStages-2]`, if `stage < num_k_tiles`, then issue `cp.async copy(gA→sA[stage])` and `copy(gB→sB[stage])`; `cp_async_fence()` commits one group
[Kernel-8] Main loop：for k_tile in [0..num_k_tiles-1] execute cp_async_wait(kStages-2)+__syncthreads() wait until the current read stage is ready
[Kernel-9] Main-loop parallel prefetch: compute `next_tile = k_tile + kStages - 1`; if it exists, then issue `cp.async copy(next_tile's gA/gB → sA/sB[smem_pipe_write])`; `cp_async_fence()`
[Kernel-10] Main-loop compute: for `k_inner in [0..BK/16-1]`, use `ldmatrix` to move `sA/sB[smem_pipe_read] → tCrA_view/tCrB_view`; then run `gemm(tiled_mma): tCrC += tCrA @ tCrB`
[Kernel-11] advance the ring-buffer stage：smem_pipe_read=(read+1)%kStages；smem_pipe_write=(write+1)%kStages；__syncthreads()
[Kernel-12] Epilogue：tCgC=partition_C(gC) obtain this thread's GMEM writeback view；tCrC_out(FP16) = convert(tCrC FP32→FP16)；vectorized copy(tCrC_out→tCgC)
```

##### CuTe Core API Cheat Sheet

| API | Function | Return |
|-----|------|------|
| `make_tensor(ptr, layout)` | Create tensor | Tensor |
| `make_layout(shape, stride)` | Create layout | Layout |
| `local_tile(tensor, tiler, coord, step)` | Extract sub-tile | Tensor |
| `make_tiled_copy(atom, thr_layout, val_layout)` | Create a collaborative copy | TiledCopy |
| `make_tiled_mma(atom, thr_layout, val_tile)` | Create collaborative MMA | TiledMMA |
| `partition_S(tensor)` / `partition_D(tensor)` | Partition by copy (Source/Dest) | Tensor |
| `partition_fragment_A/B/C(tensor)` | Partition by MMA (register fragment) | Tensor |
| `retile_D(tensor)` | Repartition existing registers (zero-overhead view conversion) | Tensor (alias) |
| `copy(copy_op, src, dst)` | Perform copy | void |
| `gemm(mma, D, A, B, C)` | Execute MMA: D = A × B + C | void |
| `cp_async_fence()` | Mark async group boundaries | void |
| `cp_async_wait<N>()` | Wait until ≤N groups are not completed | void |
| `composition(swizzle, layout)` | Apply the swizzle transformation | Layout |
| `tile_to_shape(layout, shape)` | Copy tile to target shape | Layout |
| `cosize(layout)` | The co-domain size of the layout (actual number of occupied elements) | int |

##### Global View: Data Flow and Bottlenecks

Computing `C = A × B` (FP16 compute, FP32 accumulation, FP16 output) requires solving three bottlenecks simultaneously on the GPU:

| Bottleneck | Root cause | CUTLASS solution |
|------|------|---------------|
| **Global memory bandwidth** | HBM latency ~400 cycles | Multi-level tiling + data reuse |
| **Shared-memory bank conflicts** | 32 banks × 4B, same-bank accesses serialize | Swizzled address remapping |
| **Instruction issue / compute throughput** | The MMA units must be fed continuously | Software pipelining with overlapped load and compute |

The data flow path of the entire kernel:

```
Global Memory (HBM)
    │  cp.async 128-bit   ← TiledCopy G→S
    ▼
Shared Memory (SMEM)
    │  ldmatrix            ← TiledCopy S→R
    ▼
Register File
    │  mma.sync            ← TiledMMA
    ▼
Accumulator (FP32 regs)
    │  convert + store     ← Epilogue
    ▼
Global Memory (HBM)
```

Each stage has a corresponding hardware instruction and CuTe abstraction. Understanding this data flow is the key to understanding the full kernel.

##### Three-Level Tiling Architecture

The core design pattern in CUTLASS is **three-level tiling**—decomposing a large matrix multiplication hierarchically:

```
Level 1: CTA Tile (Thread Block)
    Problem: the full `M×N` matrix C is too large
    Solution：Each thread block is responsible for a 128×128 sub-block of C
    The K dimension iterates with step size `kBlockK=32`

Level 2: Warp Tile
    Problem：128×128 is still too large for a single warp
    Solution: `TiledMMA` automatically partitions the block tile across multiple warps
    Layout<Shape<_2, _2, _1>>  → 4 warps: 2 along M and 2 along N

Level 3: MMA Instruction
    Problem: each warp must execute the actual matrix multiplication
    Solution：SM80_16x8x16 MMA atom，computes a 16×8 sub-block of C each time, consuming K depth 16
```

**Key parameter relationship**:

```
Block Tile:    128 × 128 × 32  (kBlockM × kBlockN × kBlockK)
MMA Atom:       16 ×   8 × 16  (MMA_M × MMA_N × MMA_K)
Thread Layout:   2 ×   2 ×  1  (warp replication counts along the M, N, and K directions)
Value Tile:     64 ×  64 × 16  (M, N, K extent covered by each warp)

Validate full coverage：
  M direction: 64 × 2 = 128 = kBlockM ✓
  N direction: 64 × 2 = 128 = kBlockN ✓
  K direction: 16 × 1 = 16 (`MMA_K`; the inner loop must iterate `32/16 = 2` times) ✓
```

`make_tiled_mma` encodes these three levels into a single object, after which `partition_fragment_A/B/C` can automatically derive the register fragments owned by each thread.

##### The TN Layout Convention

This is one of the most confusing parts of CUTLASS. The key is to distinguish **storage layout** from the **layout expected by the MMA instruction**.

The MMA instruction `SM80_16x8x16_F32F16F16F32_TN` means:
- **T (Transposed)**: the A operands are arranged in registers in row-major form
- **N (Normal)**: the B operands are arranged in registers in column-major form

Correspondence in code:

```cpp
// A Storage: (M, K) row-major，stride = (K, 1)
// → directly matches "T"，no additional transpose is needed

// B Storage: PyTorch provides (K, N) row-major
// → the kernel requires `(N, K)` row-major (each row of N corresponds to one output column)
// → so the host side performs B.t().contiguous()
// → and then declares it as make_shape(N, K), make_stride(K, 1)
```

Why `TN` instead of `NN` or `TT`? Because when A is row-major and B is also row-major after being transposed to `(N, K)`, the `TN` form matches exactly. This is the most common path in CUTLASS, and most optimizations target it.

##### Detailed Explanation of TiledCopy and TiledMMA

**TiledCopy: The Work Partition for Data Movement**

```cpp
auto g2s_copy = make_tiled_copy(
    G2SCopyAtom{},                              // hardware instruction：cp.async 128-bit
    Layout<Shape<_32, _4>, Stride<_4, _1>>{},   // Thread layout: 32 rows × 4 columns
    Layout<Shape<_1, _8>>{}                      // Each thread moves 1×8 elements
);
```

The physical meaning of this configuration is:
- 128 threads arranged in a 32×4 grid
- Each thread uses `cp.async` to move 128 bits = 8 `half_t` values at a time
- A single call covers 32×(4×8) = 32×32 = 1024 elements
- Our tile is 128×32, so `partition_S` splits it into multiple `CPY_M` blocks, and each `copy` call moves all blocks in the loop

If you draw a picture, this layout looks like this:
```
Shape = 32 rows × 4 cols

          c=0  c=1  c=2  c=3
r=0        0    1    2    3
r=1        4    5    6    7
r=2        8    9   10   11
r=3       12   13   14   15
r=4       16   17   18   19
r=5       20   21   22   23
r=6       24   25   26   27
r=7       28   29   30   31   ← warp0 (`tid 0..31`) exactly covers the full 8×4 block for `r=0..7`
r=8       32   33   34   35   ← warp1 starts
...

```

For comparison, `Layout<Shape<32,4>, Stride<1,32>>` looks like this:
```
          c=0  c=1  c=2   c=3
r=0        0   32   64    96
r=1        1   33   65    97
r=2        2   34   66    98
r=3        3   35   67    99
r=4        4   36   68   100
r=5        5   37   69   101
r=6        6   38   70   102
r=7        7   39   71   103
...
r=31      31   63   95   127

```

**partition_S and partition_D**

```cpp
Tensor tAgA_g = g2s_thr_copy_A.partition_S(gA);  // Source: global memory
Tensor tAsA_s = g2s_thr_copy_A.partition_D(sA);  // Dest:   shared memory
```

After partitioning, the tensor dimensions become `(CPY, CPY_M, CPY_K, ...)`:
- **CPY**: the number of elements moved by one copy instruction (8 `half_t` values)
- **CPY_M, CPY_K**: the number of iterations this thread must traverse along the M and K dimensions
- Final dimension: the K-tile index or stage index

When calling `copy(g2s_copy, src, dst)`, CuTe automatically unrolls the `CPY_M × CPY_K` loop.

**TiledMMA’s Double View Technique**

This is an elegant design:

```cpp
// MMA view (used by `gemm()`)
Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));

// Copy view (used by `copy()`)
Tensor tCrA_view = s2r_thr_copy_A.retile_D(tCrA);
```

**`tCrA`** and **`tCrA_view` point to the exact same registers**, but with different layouts:
- The layout of `tCrA` matches the operand format expected by the MMA instruction
- The layout of `tCrA_view` matches the output format of `ldmatrix`

`retile_D` performs this layout transformation without moving data; it only changes the logical view. As a result, once `copy(s2r, src, tCrA_view)` loads the data into registers, `gemm(mma, tCrC, tCrA, tCrB, tCrC)` can use it directly with no further register rearrangement.

##### Software Pipeline

This is the key to the performance of the entire kernel.

**Why is a pipeline needed?**

The latency of `cp.async` when moving data from global memory to shared memory is roughly **400-800 cycles**. In a naive implementation:

```
load A[0], B[0] to smem     ← wait 400 cycles
__syncthreads()
compute on smem data         ← compute ~100 cycles
```

The compute unit spends 80% of its time waiting!

**Core idea of pipelined overlap**:

```
Naive approach:
  [===LOAD===][COMPUTE][===LOAD===][COMPUTE][===LOAD===][COMPUTE]

Pipeline (3 stage):
  [===LOAD 0===]
       [===LOAD 1===]
            [===LOAD 2===][===LOAD 3===][===LOAD 4===]
  [          ][COMPUTE 0 ][COMPUTE 1   ][COMPUTE 2   ][COMPUTE 3]...

While the GPU is computing tile `i`, the load for tile `i+2` is already happening asynchronously!
```

**What `kStages = 3` means**: three tile copies coexist in shared memory at the same time:

```
SMEM layout:
  ┌──────────┬──────────┬──────────┐
  │ Stage 0  │ Stage 1  │ Stage 2  │
  │ 128×32   │ 128×32   │ 128×32   │
  └──────────┴──────────┴──────────┘
     ↑ read     ↑ read     ↑ write
     (currently computing)  (already loaded)   (currently loading)
```

Three stages are the classic trade-off between latency hiding and resource usage:
- **2 stages**: can hide only one tile's latency, which is often not enough
- **3 stages**: can hide the latency of two tiles, which is usually sufficient
- **4+ stages**: consume too much shared memory and reduce occupancy

**Key features of cp.async**:

`cp.async` is an asynchronous copy instruction introduced in SM80. It has two important properties:

1. **Non-blocking**: the thread continues execution immediately after issuing the copy, while the data transfer proceeds in the background
2. **Register bypass**: data is copied directly from global memory to shared memory without passing through the register file

With `cp_async_fence()` and `cp_async_wait<N>()`:

```cpp
cp_async_fence();       // insert a "fence", marking all `cp.async` operations issued so far as one group
cp_async_wait<N>();     // Wait until at most N groups remain unfinished
```

**Swizzle and multi-stage cooperation**:

```cpp
using SmemLayoutA = decltype(
    tile_to_shape(SmemLayoutAtom{},
                  make_shape(Int<128>{}, Int<32>{}, Int<3>{}))
);
```

`tile_to_shape` replicates the tile along a third dimension, which is equivalent to allocating three independent 128×32 buffers in shared memory, each preserving the swizzled layout. This is the hardware foundation of the pipeline.

##### Precise Structure of the Main Loop

This is the most intricate part of the kernel logically.

**Pipeline Fill**

```cpp
for (int stage = 0; stage < kStages - 1; ++stage) {    // stage = 0, 1
    if (stage < num_k_tiles) {
        copy(g2s_copy, tAgA_g(_, _, _, stage), tAsA_s(_, _, _, stage));
        copy(g2s_copy, tBgB_g(_, _, _, stage), tBsB_s(_, _, _, stage));
    }
    cp_async_fence();    // treat each stage's loads as an independent group
}
```

The fill phase preloads `kStages - 1 = 2` tiles. Each tile load is followed by a fence so that `cp_async_wait` can later wait for exactly the right group.

**Main Loop Four Steps**

Each iteration executes four strictly ordered steps:

```cpp
int smem_pipe_read  = 0;          // Stage currently being computed
int smem_pipe_write = kStages - 1; // stage to write next
```

**Step 1: Wait for the current stage data to be ready**

```cpp
cp_async_wait<kStages - 2>();   // wait<1>: at most 1 group may remain unfinished
__syncthreads();
```

The semantics of `cp_async_wait<1>` is "wait until the number of outstanding fenced groups is ≤ 1". Since we issue a new group on each iteration, this guarantees that the group corresponding to `smem_pipe_read` has completed.

Why `kStages - 2 = 1` instead of 0? Because we intentionally allow one set of loads to remain in flight—this is the essence of pipelining:

```
Current state (when `k_tile = 0`):
  group 0 (stage 0): ✅ complete (guaranteed by the wait)
  group 1 (stage 1): 🔄 may still be transferring (allowed)
  → we can safely compute using the data in stage 0
```

**Step 2: Load the next tile asynchronously**

```cpp
int next_tile = k_tile + kStages - 1;
if (next_tile < num_k_tiles) {
    copy(g2s_copy, tAgA_g(_, _, _, next_tile), tAsA_s(_, _, _, smem_pipe_write));
    copy(g2s_copy, tBgB_g(_, _, _, next_tile), tBsB_s(_, _, _, smem_pipe_write));
}
cp_async_fence();
```

The stage pointed to by `smem_pipe_write` has either already been consumed or has not yet been used, so it can safely be overwritten.

**Step 3: S→R copy + MMA calculation**

```cpp
for (int k_inner = 0; k_inner < num_mma_k; ++k_inner) {   // 0, 1 (32/16=2)
    // ldmatrix: move data from shared memory into registers
    copy(s2r_copy_A, tCsA(_, _, k_inner, smem_pipe_read), tCrA_view(_, _, k_inner));
    copy(s2r_copy_B, tCsB(_, _, k_inner, smem_pipe_read), tCrB_view(_, _, k_inner));
}
gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
```

`kBlockK = 32` is split into `num_mma_k = 2` chunks by the MMA K dimension (16). Interleaving `ldmatrix` and `mma` lets the compiler perform instruction-level pipelining: while the MMA unit executes a multiply-accumulate, `ldmatrix` can prepare the next chunk on the load path.

**Step 4: Advance the Ring Buffer**

```cpp
smem_pipe_read  = (smem_pipe_read  + 1) % kStages;
smem_pipe_write = (smem_pipe_write + 1) % kStages;
__syncthreads();
```

`% kStages` implements a ring buffer. `__syncthreads()` ensures that all threads have finished reading the current stage before it is overwritten.

**Full Timeline** (K=160, kBlockK=32, kStages=3, num_k_tiles=5):

```
Operation      │ k=0      │ k=1      │ k=2      │ k=3      │ k=4
─────────────┼──────────┼──────────┼──────────┼──────────┼─────────
Fill (prefill) │ G→S[0,1] │          │          │          │
Wait         │ wait≤1   │ wait≤1   │ wait≤1   │ wait≤1   │ wait≤1
Load (async) │ G→S[2]   │ G→S[3]   │ G→S[4]   │ (skip)   │ (skip)
Compute      │ stage 0  │ stage 1  │ stage 2  │ stage 0  │ stage 1
             │ S→R+MMA  │ S→R+MMA  │ S→R+MMA  │ S→R+MMA  │ S→R+MMA
read_idx     │ 0        │ 1        │ 2        │ 0        │ 1
write_idx    │ 2        │ 0        │ 1        │ 2        │ 0

Groups in flight:
k=0: group[0]✅ group[1]🔄 →issue→ group[2]🔄
k=1: group[1]✅ group[2]🔄 →issue→ group[3]🔄
k=2: group[2]✅ group[3]🔄 →issue→ group[4]🔄
k=3: group[3]✅ group[4]🔄  (no new issue)
k=4: group[4]✅             (no new issue)
```

##### Epilogue: Accumulator Write-Back

MMA accumulates into FP32 registers for accuracy, but the output matrix C is stored in FP16 for memory efficiency:

```cpp
Tensor tCgC = thr_mma.partition_C(gC);         // MMA partition view in global memory
Tensor tCrC_out = make_tensor_like<T>(tCrC);    // Allocate FP16 register space
copy(tCrC, tCrC_out);                           // FP32 → FP16 truncation
copy(AutoVectorizingCopy{}, tCrC_out, tCgC);    // Write back to global memory
```

The tensor returned by `thr_mma.partition_C(gC)` contains only the global-memory locations of the C elements owned by the current thread. Its shape matches `tCrC` one-to-one, so a direct copy is sufficient. `AutoVectorizingCopy` lets CuTe automatically choose the widest possible vectorized store instruction (such as `ST.128`).

> **💡 Tip: Additional considerations for production-grade Epilogue**
>
> This code is a simplified version. The complete CUTLASS epilogue also requires processing:
> 1. **Boundary condition**: When M or N is not an integer multiple of the block tile, predication is required to avoid out-of-bounds writes
> 2. **Epilogue Fusion**: bias addition, activation (ReLU/GELU), residual connection, etc. can be done together during writeback to avoid additional kernel launch (see Chapter 7 for details)
> 3. **Split-K**: When K is large and M×N is small, multiple blocks can be used to share the K dimension, and finally reduction is done

### 3.3 Triton Design Philosophy and Core Abstractions

[Triton](https://github.com/triton-lang/triton) is a Python-based GPU programming language and compiler developed by OpenAI. Unlike CUTLASS's C++ template approach, Triton follows the philosophy of "**write block-level pseudocode, and let the compiler generate efficient GPU code**".

#### 3.3.1 Block-Level Programming Model

Triton's core innovation lies in its **programming granularity**:

- In CUDA, developers write the behavior of a single **thread**, then launch thousands of threads
- In Triton, developers write the behavior of a single **program instance** (roughly corresponding to a thread block), and the basic unit of operation is a **block** (a small 2D tensor)

This means developers do not need to worry about:
- How threads cooperate within a warp (decided by the compiler)
- When shared memory is allocated and how data is laid out (decided by the compiler)
- How registers are allocated to each thread (decided by the compiler)
- How bank conflicts are avoided (handled by the compiler's swizzle pass)
- How instructions are scheduled to hide latency (handled by the compiler's scheduling pass)

Developers only need to describe the block-level algorithm, and the Triton compiler maps it efficiently to hardware.

> **💡 Tip: Triton Block vs CUDA Block - similar names, different meanings**
>
> The two have similar names, but refer to completely different things:
>
> **CUDA Block (Thread Block)**: a collection of threads (for example, 256 threads). You must manually manage what each thread does: thread indexing, shared-memory allocation, synchronization (`__syncthreads`), coalesced memory access, and so on. The hierarchy is Grid → Block → Thread. Threads within a block share shared memory and can synchronize.
>
> **Triton Block (Tile / Block of Data)**: a block of data (for example, a `(128, 32)` matrix slice). Triton does not expose threads directly to the user—you operate on the entire tile. `BLOCK_SIZE_M`, `BLOCK_SIZE_K`, and similar parameters describe data-block sizes, not thread counts. Under the hood, thread scheduling, shared memory, and coalesced accesses are all handled automatically by the Triton compiler.
> **Essential mapping**: a Triton program (an instance identified by `tl.program_id(0)`) is ultimately lowered to a CUDA thread block. Based on the declared `BLOCK_SIZE_*` values, the Triton compiler automatically decides how many threads are needed, how shared memory is allocated, and how memory accesses are organized.
>
> In short: **CUDA block = you manage threads and move data yourself; Triton block = you declare the shape of the data to process, and the compiler manages threads and data movement for you.** That is why Triton kernels are written using tile-level operations such as `tl.load`, `tl.dot`, and `tl.store`, without directly touching `threadIdx` or shared memory.

#### 3.3.2 A Blueprint for Writing Triton Kernels

Almost every Triton kernel follows the same fixed process. Once you understand this blueprint, writing vector addition, softmax, or FlashAttention mostly comes down to filling in the operation-specific logic for each step.

```
Step 1: Determine the `pid → tile` mapping ("which output tile am I responsible for?")
        │
Step 2: Construct input pointers from tile coordinates (`tl.arange` + pointer arithmetic)
        │
Step 3: Load data (`tl.load` + `mask` for boundary handling)
        │
Step 4: compute（scalar ops / tl.dot / tl.sum / ...）
        │
Step 5: Store the result (`tl.store` + `mask`)
```

**Step 1 is the most important step**: it determines the parallelization strategy and data-access pattern of the entire kernel. The remaining steps are largely mechanical transformations built on top of Step 1. The next two examples illustrate how Step 1 evolves as the problem becomes more complex.

---

**Example 1: Vector Add - the Simplest 1D Mapping**

The output is a 1D vector of length N. Each program handles `BLOCK_SIZE` consecutive elements:

```
Output vector: [ ---- BLOCK_SIZE ---- | ---- BLOCK_SIZE ---- | ... | -- remainder -- ]
             pid=0                  pid=1                       pid=G-1

Grid size: G = cdiv(N, BLOCK_SIZE)
```

The mapping logic fits in one line:

```python
pid = tl.program_id(0)                              # which program am I
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # indices of the elements I am responsible for
mask = offsets < N                                     # the last block may go out of bounds
```

There is no need for `pid_m` / `pid_n`: because the data is one-dimensional, a single `pid` is sufficient.

---

**Example 2: GEMM - 2D Mapping + Grouped Ordering (Swizzle)**

The output is a 2D matrix C (M×N). Each program is responsible for one `(BLOCK_M, BLOCK_N)` tile. Now we need to map a 1D `pid` onto 2D coordinates `(pid_m, pid_n)`:

```
Output matrix C:
         N direction →
        ┌──────────┬──────────┬──────────┬──────────┐
        │(0,0)     │(0,1)     │(0,2)     │(0,3)     │  ← each cell is one
   M    │ BLOCK_M  │          │          │          │    (BLOCK_M × BLOCK_N)
       │ × BLOCK_N│          │          │          │    `(BLOCK_M × BLOCK_N)` tile
       ├──────────┼──────────┼──────────┼──────────┤
   ↓    │(1,0)     │(1,1)     │(1,2)     │(1,3)     │
        ├──────────┼──────────┼──────────┼──────────┤
        │(2,0)     │(2,1)     │(2,2)     │(2,3)     │
        └──────────┴──────────┴──────────┴──────────┘

Grid size: G = cdiv(M, BLOCK_M) × cdiv(N, BLOCK_N)
```

**Naive mapping (row-major)**:

```python
pid = tl.program_id(0)       # 1D pid: 0, 1, 2, ..., G-1
pid_m = pid // num_pid_n     # row index
pid_n = pid  % num_pid_n     # column index
```

pid allocation result:

```
        col0  col1  col2  col3
row0  [  0     1     2     3  ]  ← `pid 0~3` run simultaneously and need the same row of A
row1  [  4     5     6     7  ]    but need 4 different columns of B → poor L2 cache utilization
row2  [  8     9    10    11  ]
```

**Grouped ordering (swizzle)**: group adjacent pids into a small rectangular group, and iterate over columns first within the group:

```python
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n     # total number of tiles per group
group_id = pid // num_pid_in_group              # which group it belongs to
first_pid_m = group_id * GROUP_SIZE_M           # starting row of the group
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  # column-major within the group
pid_n = (pid % num_pid_in_group) // group_size_m
```

pid allocation when `GROUP_SIZE_M=2`:

```
        col0  col1  col2  col3
row0  [  0     2     4     6  ]  ← pid 0,1 share col0 of B
row1  [  1     3     5     7  ]    pid 0,2 share row0 of A
row2  [  8    10    12    14  ]    → simultaneously running programs share more data
row3  [  9    11    13    15  ]    → L2 cache hit rate improves significantly
```

---

**Why the blueprint matters**: once Step 1 determines `pid_m, pid_n` (or, more simply, `pid + offsets`), the remaining steps become highly regular: derive input pointers, load data, compute, and write back. **The main differences across kernels are usually Step 1 (how tiles are partitioned) and Step 4 (how computation is performed).** All the code examples below follow this blueprint.

---

#### 3.3.3 Core abstraction and code examples

**1. Program ID and Grid - Work Assignment**

The first step in a Triton kernel is to determine which portion of the data the current program instance is responsible for:

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,  # compile-time constant
):
    # Get the ID of the current program instance (similar to `blockIdx.x`)
    pid = tl.program_id(axis=0)

    # Compute the element range handled by the current block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # shape: (BLOCK_SIZE,)

    # Generate a mask to handle boundary conditions（the last block may go out of bounds）
    mask = offsets < N

    # Block-level load：load the entire block of data at once
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Block-level compute
    result = x + y

    # Block-level Storage
    tl.store(out_ptr + offsets, result, mask=mask)
```

Launch the kernel:

```python
import torch

N = 100000
x = torch.randn(N, device='cuda')
y = torch.randn(N, device='cuda')
out = torch.empty(N, device='cuda')

# Compute the grid size
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

# Launch the kernel
vector_add_kernel[grid](x, y, out, N, BLOCK_SIZE=1024)
```

**2. Pointer Arithmetic - Block-Level Pointer Construction**

Triton's pointer model is one of the key differences from NumPy. In NumPy, we operate on array slices; in Triton, we explicitly construct pointer blocks:

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    # Each program instance processes one row
    row_idx = tl.program_id(0)

    # Construct the column-offset vector
    col_offsets = tl.arange(0, BLOCK_SIZE)  # [0, 1, 2, ..., BLOCK_SIZE-1]

    # Construct the pointer block to all elements in the current row
    input_ptrs = input_ptr + row_idx * n_cols + col_offsets
    mask = col_offsets < n_cols

    # Load the entire row
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Block-level softmax compute
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # write back
    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, result, mask=mask)
```

**3. `tl.dot()`——Block-Level Matrix Multiplication**

`tl.dot()` is Triton's core matrix-multiplication primitive. It automatically uses Tensor Cores to multiply and accumulate two 2D blocks:

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Compute the output-tile coordinates handled by the current block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Construct row/column offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    offs_k = tl.arange(0, BLOCK_K)                     # (BLOCK_K,)

    # Construct the pointer blocks for A and B
    # A: (BLOCK_M, BLOCK_K), B: (BLOCK_K, BLOCK_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Initialize the FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate along the K dimension
    for k in range(0, K, BLOCK_K):
        # Load the current tiles of A and B
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Block-level matrix multiplication — automatically uses Tensor Cores
        # a: (BLOCK_M, BLOCK_K), b: (BLOCK_K, BLOCK_N) → acc += a @ b
        acc = tl.dot(a, b, acc)

        # Move the pointers to the next K-tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    # Convert the FP32 accumulated result back to FP16 and write it back
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
```

Notice how concise this is: fewer than 40 lines of Python implement a Tensor Core GEMM kernel, whereas the equivalent CUDA code often takes more than 200 lines.

**4. `tl.constexpr`——Compile-Time Constants**

All parameters marked `tl.constexpr` are fixed at compile time, allowing the compiler to:

- Fully unroll the loop (`for k in range(0, K, BLOCK_K)` where `BLOCK_K` is known)
- Statically allocate shared memory and registers
- Optimize instruction selection (choose the best Tensor Core instruction for the block shape)

```python
BLOCK_SIZE: tl.constexpr  # the compiler fixes the concrete value during JIT compilation
# each distinct `BLOCK_SIZE` value produces a different kernel binary
```

**5. `@triton.autotune`——Autotuning**

Manually searching for the best tile-size configuration is tedious and easy to get wrong. Triton provides a built-in autotuning decorator:

```python
@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=3, num_warps=8
        ),
    ],
    key=['M', 'N', 'K'],  # re-search for the optimal configuration when these values change
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    # ... strides ...
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ... kernel implementation ...
    pass
```

Autotuning workflow:
1. On the first call, benchmark all candidate configurations
2. Choose the configuration that takes the shortest time
3. Subsequent calls reuse the best configuration directly (cached results are indexed by the `key` parameter)

#### 3.3.4 Triton Compiler Pipeline

Understanding the Triton compiler pipeline helps explain both its performance characteristics and its limitations:

```
Python source code (function decorated with `@triton.jit`)
        │
        ▼
    Python AST parsing
        │
        ▼
    Triton IR (MLIR Dialect)
    ├─ type inference
    ├─ Block-level operation semantics
    └─ thread mapping has not been decided yet
        │
        ▼
    Triton GPU IR
    ├─ automatically insert shared-memory operations
    ├─ determine the mapping from threads/warps to data
    ├─ generate software pipelining (according to `num_stages`)
    ├─ apply memory-coalescing optimizations
    └─ Bank conflict avoidance (swizzle)
        │
        ▼
    LLVM IR (NVPTX backend)
    ├─ Register allocation
    ├─ Instruction scheduling
    └─ Peephole optimizations
        │
        ▼
    PTX Assembly
        │
        ▼
    SASS (compiled by `ptxas`)——the final machine code
```

Key optimizations performed automatically by the compiler include:

- **Shared-memory allocation**: analyze `tl.load` / `tl.store` patterns and automatically decide which data should be cached in shared memory
- **Software pipelining**: overlap data loading and computation according to the `num_stages` parameter. For example, `num_stages=3` means that three K-tiles can be in flight simultaneously
- **Register tiling**: lower block-level operations into per-thread register operations
- **Vectorized loads**: generate 128-bit vector load instructions automatically when contiguous access patterns are detected
- **Tensor Core lowering**: lower `tl.dot` automatically into the appropriate `mma` instruction sequence

#### 3.3.5 Case Study: Triton High-Performance GEMM

With the blueprint and core abstractions in place, we can now look at a complete production-grade Triton GEMM implementation. Compared with the handwritten CUDA in Chapter 2 (Sections 2.3-2.6), which requires manually managing low-level details such as shared memory, register blocking, bank conflicts, and vectorized access (~200 lines), Triton needs only ~60 lines of core logic, and the compiler takes care of the rest.

```python
import triton
import triton.language as tl
import torch

@triton.autotune(
    # Automatically search for the best configuration (similar to CUTLASS tile-size selection)
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],  # re-search for the optimal configuration when matrix sizes change
)
@triton.jit
def gemm_kernel(
    # matrix pointers
    A_ptr, B_ptr, C_ptr,
    # matrix dimensions
    M, N, K,
    # matrix leading dimensions (supports non-contiguous layouts)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # scale factor
    alpha,
    # compile-time constants (chosen by autotune)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = alpha * A @ B
    A: (M, K), B: (K, N), C: (M, N)

    Each Triton program instance computes one `(BLOCK_M, BLOCK_N)` tile of C.
    """
    # ==================== Step 1: pid → tile mapping ====================
    # `pid` is the current program's 1D index and must be mapped to 2D tile coordinates in matrix C
    pid = tl.program_id(axis=0)

    # Grouped ordering (an L2-cache-friendly tile traversal order)
    # See the GEMM swizzle example in the blueprint in Section 3.3.2
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ==================== Step 2: Construct input pointers ====================
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # 2D pointer arrays: broadcast `(BLOCK_M,1)` + `(1,BLOCK_K)` → `(BLOCK_M, BLOCK_K)`
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ==================== Step 3+4: load + compute (blockwise accumulation along the K dimension) ====================
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # FP32 accumulation

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k_start * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        # tl.dot automatically uses Tensor Cores and automatically manages SMEM and register tiling
        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = (accumulator * alpha).to(tl.float16)

    # ==================== Step 5: write back the result ====================
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def gemm_triton(a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Triton GEMM wrapper: a (M,K) @ b (K,N) → c (M,N)"""
    assert a.shape[1] == b.shape[0], "matrix dimensions do not match"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 1D launch grid: a total of `ceil(M/BLOCK_M) * ceil(N/BLOCK_N)` programs
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        alpha,
    )
    return c
```

Notice that the comments in the code explicitly mark the five blueprint steps. Compared with the handwritten CUDA implementation in Chapter 2:

| Dimension | CUDA (Chapter 2, handwritten) | Triton (this section) |
| ---------------- | --------------------------- | ------------------------ |
| Code size | ~200 lines | ~60 lines of core logic |
| Abstraction level | Thread level (`threadIdx`, `warp_id`) | Block level (`tl.program_id`) |
| Shared memory | Manual declaration, loading, padding | Automatically managed by the compiler |
| Register blocking | Manual `reg_a/reg_b/reg_c` | Handled internally by `tl.dot()` |
| Bank conflict | Manual +4 padding | Automatic elimination by compiler |
| Double buffering | Manually alternate two buffers | `num_stages` parameter, the compiler generates a pipeline |
| Tensor Core | Manual WMMA API | Automatically used by `tl.dot()` |
| Tile-size tuning | Manual trial and error | Automatic search via `@triton.autotune` |
| Performance | Hand-tuned implementations can reach cuBLAS 90%+ | Typically reaches cuBLAS 80-90% |



