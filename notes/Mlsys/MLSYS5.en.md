# CUDA Parallel Primitives: Histogram & Scan

This lecture builds on the previous lecture’s Reduce kernel and introduces two important parallel primitives: Histogram and Scan (Prefix Sum).

## Part 1: Histogram Kernel

https://leetgpu.com/challenges/histogramming

### 1.1 From Reduce to Histogram

In the previous lecture, we studied Reduce: reducing $N$ elements to a single value. Histogram can be viewed as a **multi-target Reduce**:

```
Reduce:     N elements → 1 value      (all elements reduce to the same target)
Histogram:  N elements → K bins       (elements reduce to K different targets based on condition)
```

| Comparison Dimension | Reduce | Histogram |
|---------|--------|-----------|
| Output size | 1 | K (number of bins) |
| Write target | Fixed | Data-dependent |
| Parallel challenge | Reduction tree design | **Atomic operation contention** |
| Roofline | Memory-bound | Memory-bound + Atomic-bound |

The key difference is that the write target in Reduce is predetermined, whereas the write target in Histogram depends on the input value. As a result, **multiple threads may update the same bin simultaneously**.

### 1.2 Histogram’s Core Challenge: Atomic Contention

When the data distribution is concentrated—for example, when most elements fall into only a few bins—atomic operations become heavily serialized:

```
Thread 0 → bin[3] ─┐
Thread 1 → bin[3] ─┼─→ Serialized execution!
Thread 2 → bin[3] ─┤
Thread 3 → bin[5] ─┘
```

### 1.3 Solution: Hierarchical Privatization

The core idea is to **reduce the scope of contention**, moving from global contention → block-local contention → warp-local contention.

```
┌─────────────────────────────────────────────────────────┐
│  Version 1: Global Atomic                               │
│  All threads → Global Memory (most severe contention)   │
├─────────────────────────────────────────────────────────┤
│  Version 2: Shared Memory Privatization                 │
│  Threads within block → Shared Memory → Global Memory   │
│  Contention scope shrinks from whole GPU to one block   │
├─────────────────────────────────────────────────────────┤
│  Version 3: Warp-level + Local Accumulation             │
│  Further reduces the number of atomics                  │
└─────────────────────────────────────────────────────────┘
```

### 1.4 Implementation Versions

#### Version 1: Naive (Baseline)

```cpp
// __global__ qualifier: declares this as a GPU kernel function
//   - Called by the CPU (host), executed on the GPU (device)
//   - The return type must be void
__global__ void histogram_v1_naive(
    // __restrict__ keyword: tells the compiler this pointer is the only way to access this memory
    //   - Guarantees that data and hist point to non-overlapping memory regions (no pointer aliasing)
    //   - Allows more aggressive compiler optimizations (such as loop unrolling and instruction reordering)
    //   - Similar to C99's restrict, but uses double underscores in CUDA
    const int* __restrict__ data,  // input data array (read-only)
    int* __restrict__ hist,         // output histogram array (read/write)
    int n,                          // number of input elements
    int num_bins                    // number of histogram bins
) {
    // Compute the global index of the current thread
    // blockIdx.x: index of the current block in the grid
    // blockDim.x: number of threads per block
    // threadIdx.x: index of the current thread within the block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Stride for the grid-stride loop pattern
    // gridDim.x: total number of blocks in the grid
    // stride = total number of threads, used when the data size exceeds the thread count
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each thread processes multiple elements
    // Advantages of this pattern:
    //   1. Can handle input data of arbitrary size
    //   2. The thread count can be tuned independently of the data size
    //   3. Preserves good memory access patterns (adjacent threads access adjacent memory)
    for (int i = idx; i < n; i += stride) {
        int bin = data[i];
        // Bounds check: make sure the bin value is in the valid range
        if (bin >= 0 && bin < num_bins) {
            // atomicAdd: atomic addition operation
            //   - Ensures correctness when multiple threads update the same location concurrently
            //   - Drawback: serialization occurs when many threads contend for the same bin
            //   - This is the main performance bottleneck of the naive version
            atomicAdd(&hist[bin], 1);
        }
    }
}
```

**Memory Flow and Latency Analysis**

Let us analyze the memory flow in this kernel step by step:

1. **Index computation stage**: the calculations of `idx` and `stride` are performed entirely in registers. `blockIdx.x`, `blockDim.x`, `threadIdx.x`, and `gridDim.x` are CUDA built-in variables stored in special registers, so their access latency is extremely low (about 1 clock cycle).

2. **Data load stage**: `int bin = data[i]` is the first memory bottleneck in the kernel. The `data` array resides in global memory, whose access latency can be as high as 400–800 clock cycles. However, because we use a grid-stride loop, adjacent threads access adjacent memory addresses, which forms a coalesced access pattern. When a warp (32 threads) accesses 32 consecutive `int`s, these requests can be coalesced into a single 128-byte memory transaction, greatly improving bandwidth utilization. The loaded `bin` value is then stored in each thread’s register.

3. **Bounds check stage**: the comparisons in `if (bin >= 0 && bin < num_bins)` are executed in registers, so the latency is negligible. There may be branch divergence here, but we ignore it for now.

4. **Atomic update stage**: `atomicAdd(&hist[bin], 1)` is the core performance bottleneck. This operation involves:
   - First computing the address of the corresponding element in the `hist` array based on `bin` (a register operation)
   - Then issuing a global-memory atomic read-modify-write
   - The latency of the atomic operation itself is similar to a normal global-memory access (400–800 cycles), but the real issue is serialization: when multiple threads update the same bin simultaneously, these operations must queue up
   - If the data distribution is skewed (some bins are especially hot), contention becomes even more severe, and latency can accumulate into the thousands of cycles

**Latency Hiding and Performance Characteristics**

GPUs hide memory latency through massive thread-level parallelism. When one warp is waiting for memory access to complete, the scheduler switches to other ready warps. However, the problem with this naive version is that:

- Global-memory atomic operations cannot be hidden effectively, because updates to the same bin must serialize
- All threads contend for the same `hist` array, causing severe memory contention
- When the number of bins is small or the data distribution is concentrated, performance drops sharply

This is why later versions use Shared Memory for privatization: Shared Memory has an access latency of only about 20–30 clock cycles—roughly an order of magnitude faster than global memory—which can greatly reduce the cost of atomic operations.

#### Version 2: Shared Memory Privatization ⭐

This is the most practical optimized version:

```cpp
__global__ void histogram_v2_shared(
    const int* __restrict__ data,
    int* __restrict__ hist,
    int n, int num_bins
) {
    extern __shared__ int s_hist[];  // private histogram for each block
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Step 1: Initialize shared memory
    for (int i = tid; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Step 2: Accumulate in shared memory (contention is within a block, ~10x faster than global)
    for (int i = idx; i < n; i += stride) {
        int bin = data[i];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();
    
    // Step 3: Reduce to global memory (only 1 global atomic per bin)
    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}
```

**Why it works**:
- Shared-memory atomics are about 10× faster than global-memory atomics
- Contention is reduced from millions of threads to 256 threads per block
- The final number of global atomics is `num_bins × num_blocks` rather than `n`

**Limitation**: the number of bins is constrained by shared-memory capacity (48KB → about 12K `int` bins)

#### Version 3: Local Accumulation (Handling Data Locality)

When the data exhibits locality (consecutive elements tend to fall into the same bin):

```cpp
// Core idea: exploit temporal locality in the data by accumulating consecutive identical bins in registers
// Applicable scenario: when the input has locality (such as image pixels or sorted data), consecutive elements often fall into the same bin
// Optimization principle: merge multiple atomic operations into one to reduce the total number of atomics
__global__ void histogram_v3_local(
    const int* __restrict__ data,
    int* __restrict__ hist,
    int n, int num_bins
) {
    extern __shared__ int s_hist[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_bins; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // ========== Core optimization: local accumulator ==========
    // last_bin: records the index of the previously processed bin (stored in a register)
    // count: accumulator that records how many times the same bin appears consecutively (stored in a register)
    // Key insight: register access is essentially free (1 cycle), while atomicAdd is expensive
    int last_bin = -1;
    int count = 0;

    for (int i = idx; i < n; i += stride) {
        int bin = data[i];
        if (bin == last_bin) {
            // Encounter the same bin again: just increment the counter in the register
            // This is a pure register operation with zero memory access overhead
            count++;
        } else {
            // Encounter a different bin: need to "flush" the previously accumulated value to shared memory
            // atomicAdd is executed only when the bin changes, not for every element
            if (count > 0) atomicAdd(&s_hist[last_bin], count);
            last_bin = bin;  // update the tracked bin
            count = 1;       // reset the counter
        }
    }
    // After the loop, the last accumulated count is still in a register and needs a final flush
    if (count > 0) atomicAdd(&s_hist[last_bin], count);
    __syncthreads();

    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}
// Performance analysis:
// - Best case: data is perfectly ordered, so each thread needs only 1 atomicAdd (all elements fall into the same bin)
// - Worst case: data is completely random, degrading to Version 2 performance (every element triggers an atomicAdd)
// - Extra overhead: one extra comparison and branch per loop iteration, but that is far cheaper than atomicAdd
```

### 1.5 Performance Characteristics and Selection Guide

| Scenario | Recommended Version | Reason |
|------|---------|------|
| `bins < 12K` | V2 Shared | Best general-purpose choice |
| Data has locality | V3 Local | Reduces the number of atomics |
| `bins > 12K` | Multi-pass or CUB | Exceeds shared-memory capacity |
| Production environment | CUB library | Highly optimized |

**Roofline Analysis**

Let us use the Roofline model to analyze the performance characteristics of the Histogram kernel.

**Notation**:
- $\pi$: peak GPU compute throughput (FLOP/s)
- $\beta$: memory bandwidth (Byte/s)
- $N$: number of input elements
- $B$: number of bins
- $s$: bytes per data element (e.g., for `int32`, $s=4$)


**Arithmetic Intensity Analysis of Histogram**:

For the naive version, processing $N$ elements:

- **Memory traffic**:
  - Read input data: $N \cdot s$ bytes
  - Atomically update the histogram (read-modify-write): $N \cdot 2s$ bytes (worst case, each update hits a different bin)
  - Total: $M = 3Ns$ bytes

- **Compute**:
  - Per element: bounds check + addition ≈ $\alpha$ FLOPs ($\alpha \approx 2$)
  - Total: $F = \alpha N$ FLOPs

- **Arithmetic intensity**:
$$I_{hist} = \frac{F}{M} = \frac{\alpha N}{3Ns} = \frac{\alpha}{3s}$$

For `int32` ($s=4$), $I_{hist} = \frac{\alpha}{12} \ll 1$ FLOP/Byte.

**Roofline Conclusion**:

Since modern GPUs typically have $I_{ridge} \gg 1$ (usually $I_{ridge} > 100$), while $I_{hist} < 1$, we have:
$$I_{hist} \ll I_{ridge}$$

Histogram therefore lies deep in the **memory-bound** region, with attainable performance:
$$P_{attainable} = I_{hist} \cdot \beta = \frac{\alpha \beta}{3s}$$

Peak compute utilization:
$$\eta = \frac{P_{attainable}}{\pi} = \frac{\alpha \beta}{3s\pi} = \frac{\alpha}{3s \cdot I_{ridge}} \ll 1$$

**Accounting for Atomic Contention**:

Let $\gamma \in (0, 1]$ denote the effective bandwidth factor under atomic contention (the more severe the contention, the smaller $\gamma$). Then the actual performance is:
$$P_{real} = \gamma \cdot I_{hist} \cdot \beta = \frac{\gamma \alpha \beta}{3s}$$

**Roofline View of the Optimized Versions**:

| Version | Optimization Effect | Roofline Impact |
|------|---------|--------------|
| V2 Shared | Replaces Global Memory with Shared Memory | Effectively boosts $\beta$ by a factor of $\kappa$ ($\kappa \approx 10\text{-}20$) |
| V3 Local | Uses register accumulation to reduce the number of atomics | Pushes $\gamma$ closer to 1 |

**Conclusion**: Histogram has $I \ll I_{ridge}$ and is always memory-bound. The optimization strategies are:
1. **Increase effective bandwidth**: use a faster level in the memory hierarchy (Shared Memory), effectively increasing $\beta$
2. **Reduce contention**: privatize the histogram to increase $\gamma$
3. **Reduce memory traffic**: use local accumulation to reduce $M$

---

## Part 2: Scan (Prefix Sum) Kernel

https://leetgpu.com/challenges/prefix-sum

### 2.1 What Is Scan

Scan (prefix sum) is another core parallel primitive that computes cumulative operations over an array:

```
Input:  [3, 1, 7, 0, 4, 1, 6, 3]

Exclusive Scan (excluding current element):
Output:  [0, 3, 4, 11, 11, 15, 16, 22]
       ↑  ↑
       0  0+3

Inclusive Scan (including current element):  
Output:  [3, 4, 11, 11, 15, 16, 22, 25]
       ↑  ↑
       3  3+1
```

### 2.2 Why Scan Matters

Scan is a building block for many other parallel algorithms:

| Application | How Scan Is Used |
|------|-------------|
| Stream Compaction | Mark → Scan → Scatter |
| Radix Sort | Count → Scan → Assign positions |
| Sparse matrices | `row_ptr` in CSR format |
| Parallel allocation | Compute each thread’s output offset |

### 2.3 The Problem with Naive Parallel Scan

The most straightforward idea is to let each element independently compute its prefix sum:

```cpp
// Wrong! O(n²) complexity
__global__ void scan_naive(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i = 0; i <= idx; i++) {  // each thread iterates over [0, idx]
        sum += data[i];
    }
    data[idx] = sum;
}
```

The problem is that thread $i$ performs $i$ additions, so the total work is $O(n^2)$ and does not exploit parallelism at all.

### 2.4 Efficient Parallel Scan: The Blelloch Algorithm

The Blelloch algorithm has two phases, with total work $O(n)$ and span $O(\log n)$:

```
Phase 1: Up-sweep (Reduce)
Build the reduction tree and compute partial sums

Phase 2: Down-sweep
Propagate from the root downward to compute prefix sums
```

#### Pseudocode

```python
def blelloch_scan(x):
    """
    Blelloch parallel prefix-sum algorithm
    Input: x[0..n-1], where length n must be a power of 2
    Output: exclusive prefix sum
    """
    n = len(x)

    # ========== Phase 1: Up-sweep (Reduce) ==========
    # Build the reduction tree from leaves to root
    # At each step, store the sum of adjacent element pairs at the right element
    for d in range(log2(n)):           # d = 0, 1, ..., log2(n)-1
        stride = 2^(d+1)               # stride = 2, 4, 8, ...
        for i in parallel(0, n, stride):  # i = 0, stride, 2*stride, ...
            x[i + stride - 1] += x[i + stride/2 - 1]

    # At this point, x[n-1] contains the sum of all elements

    # ========== Phase 2: Down-sweep ==========
    # Compute prefix sums from root to leaves using the reduction tree
    x[n-1] = 0                         # Set the root to 0 (identity element)

    for d in range(log2(n)-1, -1, -1): # d = log2(n)-1, ..., 1, 0
        stride = 2^(d+1)               # stride = n, n/2, ..., 4, 2
        for i in parallel(0, n, stride):
            left = i + stride/2 - 1
            right = i + stride - 1

            temp = x[left]             # Save the value of the left child
            x[left] = x[\right]         # Left child = parent value (propagated from above)
            x[\right] += temp           # Right child = parent value + original left child value

    return x  # x now contains the exclusive prefix sum
```

**Complexity analysis**:
- Time complexity: $O(\log N)$ steps (with parallel work inside each step)
- Work complexity: $O(N)$ additions
- Space complexity: $O(1)$ extra space (in-place algorithm)

#### Illustration (8 elements)

```
Input: [3, 1, 7, 0, 4, 1, 6, 3]

=== Up-sweep (Reduce) ===
Step 1 (stride=1): 
[3, 4, 7, 7, 4, 5, 6, 9]
     ↑     ↑     ↑     ↑
    3+1   7+0   4+1   6+3

Step 2 (stride=2):
[3, 4, 7, 11, 4, 5, 6, 14]
           ↑            ↑
         4+7          5+9

Step 3 (stride=4):
[3, 4, 7, 11, 4, 5, 6, 25]
                        ↑
                     11+14

=== Down-sweep ===
Set root to 0: [3, 4, 7, 11, 4, 5, 6, 0]
                                  ↑

Step 1 (stride=4):
[3, 4, 7, 0, 4, 5, 6, 11]
           ↑            ↑
        swap and accumulate

Step 2 (stride=2):
[3, 4, 7, 0, 4, 5, 6, 11]
     ↓     ↓     ↓      ↓
[3, 0, 7, 4, 4, 11, 6, 16]

Step 3 (stride=1):
[0, 3, 4, 11, 11, 15, 16, 22]

Output (Exclusive): [0, 3, 4, 11, 11, 15, 16, 22] ✓
```


```
new_left  = right           // left child inherits the value passed down from the parent
new_right = left + right    // right child = inherited value + left sibling's subtree sum
Tree view

Up-sweep result (subtree sums):   Down-sweep propagation (sum on the left):
        25                              0
       /  \                           /    \
     11    14                        0       11
    / \   / \                       / \     /   \
   4   7  5  9                     0   4   11    16
  /\  /\ /\  /\                   /\  /\   /\    /\
 3 1 7 0 4 1 6 3                 0 3 4 11 11 15 16 22


**Rules**:
- **Left child**: inherits the parent's value (what is to my left = what is to my parent's left)
- **Right child**: parent value + left sibling's sum (what is to my left = what is to my parent's left + left sibling)


temp[n-1] = 0;  // before starting down-sweep
This is an **exclusive** scan, so the prefix sum of the first element is 0 (there is nothing to its left). This 0 propagates to position 0 during down-sweep.


Input:     [3, 1, 7, 0, 4, 1, 6, 3]

After up-sweep: [3, 4, 7, 11, 4, 5, 6, 25]
                                   ↓ set to 0
           [3, 4, 7, 11, 4, 5, 6, 0]

Down-sweep:
  stride=4: [3, 4, 7, 0,  4, 5,  6, 11]   // root level
  stride=2: [3, 0, 7, 4,  4, 11, 6, 16]   // second level
  stride=1: [0, 3, 4, 11, 11, 15, 16, 22] // leaf level ✓
```

### 2.5 CUDA Implementation

#### Version 1: Single-Block Scan (Blelloch)

```cpp
__global__ void scan_blelloch_single_block(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    
    // Load into shared memory
    temp[2*tid] = data[2*tid];
    temp[2*tid+1] = data[2*tid+1];
    
    int offset = 1;
    
    // === Up-sweep (Reduce) ===
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element (prepare for down-sweep)
    if (tid == 0) temp[n-1] = 0;
    
    // === Down-sweep ===
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // Write back
    data[2*tid] = temp[2*tid];
    data[2*tid+1] = temp[2*tid+1];
}
```

**Limitation**: can only handle data that fits in a single block (typically ≤ 2048 elements)

#### Version 2: Multi-Block Scan (Three Stages)

Handling an array of arbitrary size requires three stages:

```
Stage 1: Block-level Scan
Each block scans its own portion independently and saves the block sum

Stage 2: Scan Block Sums  
Perform scan over the sums of all blocks

Stage 3: Add Block Offsets
Each block adds the total sum of all preceding blocks
```

```cpp
// Stage 1: each block scans and saves its sum
__global__ void scan_blocks(int* data, int* block_sums, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * blockDim.x * 2;
    
    // Load data
    int ai = tid;
    int bi = tid + blockDim.x;
    temp[ai] = (block_offset + ai < n) ? data[block_offset + ai] : 0;
    temp[bi] = (block_offset + bi < n) ? data[block_offset + bi] : 0;
    
    // Blelloch scan (same as above)
    // ... up-sweep ...
    // ... down-sweep ...
    
    __syncthreads();
    
    // Save the block sum
    if (tid == 0) {
        block_sums[bid] = temp[blockDim.x * 2 - 1];
    }
    
    // Write back the scan result
    if (block_offset + ai < n) data[block_offset + ai] = temp[ai];
    if (block_offset + bi < n) data[block_offset + bi] = temp[bi];
}

// Stage 3: add the block offset
__global__ void add_block_sums(int* data, int* block_sums, int n) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (blockIdx.x > 0 && idx < n) {
        data[idx] += block_sums[blockIdx.x];
    }
    if (blockIdx.x > 0 && idx + blockDim.x < n) {
        data[idx + blockDim.x] += block_sums[blockIdx.x];
    }
}
```

#### Version 3: Work-Efficient with Bank-Conflict Avoidance

Shared memory has bank-conflict issues. We add padding to avoid them:

```cpp
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

__global__ void scan_optimized(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    
    // Padded indices to avoid bank conflicts
    int ai = tid;
    int bi = tid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
    temp[ai + bankOffsetA] = data[ai];
    temp[bi + bankOffsetB] = data[bi];
    
    int offset = 1;
    
    // Up-sweep with conflict-free addressing
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // ... handle down-sweep similarly ...
}
```


### 2.6 Roofline Characteristics of Scan

```
Memory access: read n + write n = 2n
Computation: O(n) additions
Arithmetic Intensity: ~0.25 ops/byte (int32)

Conclusion: memory-bound, but better than histogram (no atomic operations)
```


## Part 3: Scan in Mamba 1.0

### 3.1 Background: State Space Model (SSM)

An SSM is a sequence-modeling method that can be viewed as a discretized continuous-time system:

```
Continuous form:
  h'(t) = A·h(t) + B·x(t)     (state update)
  y(t)  = C·h(t) + D·x(t)     (output)

After discretization:
  h_t = Ā·h_{t-1} + B̄·x_t    (linear recurrence!)
  y_t = C·h_t + D·x_t
```

**This is a linear recurrence, which is exactly the form that Scan can parallelize!**

### 3.2 Two Computation Modes for SSMs

**Mode 1: Recurrent (Sequential Computation)**

```python
# O(L) time, O(1) space, but fully sequential
h = zeros(N)  # hidden state
for t in range(L):
    h = A @ h + B @ x[t]   # must wait for the previous step to finish
    y[t] = C @ h
```

- Efficient during inference: $O(1)$ per token
- Slow during training: cannot parallelize, so GPU utilization is low

**Mode 2: Convolution (Parallel Computation)**

For **time-invariant** SSMs (fixed `A`, `B`, and `C`), the recurrence can be expanded into a convolution:

```
y = x * K, where K = (CB̄, CĀB̄, CĀ²B̄, ...)
```

- Efficient during training: FFT convolution in $O(L \log L)$
- But requires `A`, `B`, and `C` to be constants (time-invariant)

### 3.3 Mamba’s Dilemma: Selective but Not Convolution-Friendly

Mamba’s key innovation is the **Selective SSM**: make `B`, `C`, and `Δ` depend on the input.

```python
# Selective SSM: parameters vary with the input
Δ_t = Linear(x_t)  # discretization step size
B_t = Linear(x_t)  # input matrix
C_t = Linear(x_t)  # output matrix

h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B_t·x_t
y_t = C_t·h_t
```

**Problem**: the parameters now vary with time → the system is no longer time-invariant → the convolution formulation breaks down.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mamba's Dilemma                                                            │
│                                                                             │
│  Want Selective behavior (good quality) → B, C, Δ must depend on input → time-varying system │
│  Want Fast Training                → need parallelization → convolution requires time-invariance │
│                                                                             │
│  Contradiction! Traditional methods force a trade-off                       │
│                                                                             │
│  Solution: Parallel Associative Scan                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Associative Scan: Parallelizing Linear Recurrences

**Key insight**: even though the parameters vary over time, the recurrence still satisfies **associativity**.

The SSM recurrence can be written as:
```
h_t = A_t · h_{t-1} + b_t

Define the pair: (A_t, b_t)
Define the associative operator ⊗: (A₂, b₂) ⊗ (A₁, b₁) = (A₂·A₁, A₂·b₁ + b₂)
```

**Verifying associativity**:

```
(A₃, b₃) ⊗ [(A₂, b₂) ⊗ (A₁, b₁)]
= (A₃, b₃) ⊗ (A₂·A₁, A₂·b₁ + b₂)
= (A₃·A₂·A₁, A₃·A₂·b₁ + A₃·b₂ + b₃)

[(A₃, b₃) ⊗ (A₂, b₂)] ⊗ (A₁, b₁)
= (A₃·A₂, A₃·b₂ + b₃) ⊗ (A₁, b₁)
= (A₃·A₂·A₁, A₃·A₂·b₁ + A₃·b₂ + b₃)

The two sides are equal! Associativity holds ✓
```

### 3.5 Applying Parallel Scan to SSMs

Once we have associativity, we can use Blelloch scan to compute in parallel:

```
Input: [(A₁,b₁), (A₂,b₂), (A₃,b₃), (A₄,b₄), ...]

Goal: compute all prefix products
  h₁ = (A₁,b₁)
  h₂ = (A₂,b₂) ⊗ (A₁,b₁)
  h₃ = (A₃,b₃) ⊗ (A₂,b₂) ⊗ (A₁,b₁)
  ...

Use Blelloch Scan:
  Step 1 (Up-sweep): build partial products
  Step 2 (Down-sweep): propagate prefix products
  
  Complexity: O(L) work, O(log L) span
  Can be completed in O(log L) steps!
```

**Illustration**:

```
Sequential (O(L) steps):
  h₁ → h₂ → h₃ → h₄ → h₅ → h₆ → h₇ → h₈
  
Parallel Scan (O(log L) steps):
  Step 1:  [1-2]   [3-4]   [5-6]   [7-8]     (4 pairs)
  Step 2:  [1-4]           [5-8]             (2 pairs)  
  Step 3:  [1-8]                             (1 pair)
  Down-sweep: distribute prefix products to each position
  
  A total of 2·log₂(8) = 6 steps, instead of 8
```


```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mamba's Three Main Optimization Strategies                                 │
│                                                                             │
│  1. Kernel Fusion                                                            │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Traditional: HBM → Discretize → HBM → Scan → HBM → Output → HBM │    │
│     │             (multiple HBM reads/writes, creating an I/O bottleneck) │ │
│     │                                                                 │    │
│     │ Fused: HBM → SRAM [Discretize + Scan + Output] → HBM            │    │
│     │        (one read in, one write out)                             │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. Parallel Scan in SRAM                                                   │
│     • Do not materialize intermediate states to HBM                        │
│     • Perform all scan operations entirely in SRAM                         │
│     • Write only the final output to HBM                                   │
│                                                                             │
│  3. Recomputation                                                           │
│     • Forward: do not save intermediate states                             │
│     • Backward: recompute the required states                              │
│     • Trade compute for memory                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.6 Core Mamba-1 Kernel Code Walkthrough

> Source code from: https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan/

#### Core File Structure

```
csrc/selective_scan/
├── selective_scan_common.h      # Associative scan operator definition
├── selective_scan_fwd_kernel.cuh  # Forward pass kernel
├── selective_scan_bwd_kernel.cuh  # Backward pass kernel
└── reverse_scan.cuh             # Reverse scan (used for gradient propagation)
```

#### 1. Associative Scan Operator (The Core!)

The SSM recurrence is: `h[t] = A * h[t-1] + B * x[t]`

**Key insight**: this recurrence can be represented as an associative operator over pairs `(a, b)`:
- Represent the state as `(decay, value)` = `(A, B*x)`
- Combine two states as `(a0, b0) ⊕ (a1, b1) = (a1*a0, a1*b0 + b1)`

```cuda
// selective_scan_common.h - core operator
// This is the mathematical foundation that enables Mamba to be parallelized!

template<>
struct SSMScanOp<float> {
    __device__ __forceinline__ float2 operator()(
        const float2 &ab0,  // (a0, b0) = previous state
        const float2 &ab1   // (a1, b1) = current state
    ) const {
        // Associative law: (a1*a0, a1*b0 + b1)
        // ab.x = decay factor (accumulated decay)
        // ab.y = value contribution (accumulated input)
        return make_float2(
            ab1.x * ab0.x,           // accumulated decay: a1 * a0
            ab1.x * ab0.y + ab1.y    // accumulated value: a1 * b0 + b1
        );
    }
};

// Complex-number version (used for some SSM variants)
template<>
struct SSMScanOp<complex_t> {
    __device__ __forceinline__ float4 operator()(
        const float4 &ab0, const float4 &ab1
    ) const {
        complex_t a0(ab0.x, ab0.y), b0(ab0.z, ab0.w);
        complex_t a1(ab1.x, ab1.y), b1(ab1.z, ab1.w);
        complex_t out_a = a1 * a0;
        complex_t out_b = a1 * b0 + b1;
        return make_float4(out_a.real_, out_a.imag_,
                          out_b.real_, out_b.imag_);
    }
};

// State propagation across chunk boundaries
template <typename scalar_t>
struct SSMScanPrefixCallbackOp {
    using scan_t = std::conditional_t<
        std::is_same_v<scalar_t, float>, float2, float4>;
    scan_t running_prefix;  // final state of the previous chunk

    __device__ scan_t operator()(scan_t block_aggregate) {
        scan_t old_prefix = running_prefix;
        // Combine the current block aggregate with the running prefix
        running_prefix = SSMScanOp<scalar_t>()(running_prefix, block_aggregate);
        return old_prefix;  // return it for use by the current block
    }
};
```

#### 2. Core Logic of the Forward Kernel

```cuda
// selective_scan_fwd_kernel.cuh (simplified version)

template<typename Ktraits>
__global__ void selective_scan_fwd_kernel(SSMParamsBase params) {
    // ========== Constants and configuration ==========
    constexpr int kNThreads = Ktraits::kNThreads;     // number of threads
    constexpr int kNItems = Ktraits::kNItems;         // number of elements processed per thread
    constexpr int kChunkSize = kNThreads * kNItems;   // 2048 (typical value)

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;   // each block handles one (batch, dim)

    // ========== Step 1: Load parameters ==========
    // A: [D, N] - state transition matrix (usually negative, indicating decay)
    // delta: [B, L, D] - time step size (input-dependent)
    // B: [B, L, N] or [B, N] - input matrix
    // C: [B, L, N] or [B, N] - output matrix
    // u: [B, L, D] - input

    float A_val = A[dim_id * N + state_idx];  // for each state dimension
    A_val *= LOG2E;  // pre-multiply by log2(e), since exp2f is faster

    // ========== Step 2: Process the sequence chunk by chunk ==========
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        int chunk_offset = chunk * kChunkSize;

        // 2a. Load this chunk's data into registers
        float delta_vals[kNItems], u_vals[kNItems];
        float B_vals[kNItems], C_vals[kNItems];

        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            int seq_idx = chunk_offset + threadIdx.x * kNItems + i;
            delta_vals[i] = delta[batch_id][seq_idx][dim_id];
            u_vals[i] = u[batch_id][seq_idx][dim_id];

            // Delta softplus (optional): delta = log(1 + exp(delta))
            if (kDeltaSoftplus) {
                delta_vals[i] = delta_vals[i] <= 20.f
                    ? log1pf(expf(delta_vals[i]))
                    : delta_vals[i];
            }
        }

        // 2b. Compute the scan inputs: (decay, value) pairs
        float2 thread_data[kNItems];

        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            // decay = exp(delta * A) = exp2(delta * A * log2(e))
            float decay = exp2f(delta_vals[i] * A_val);

            // value = delta * B * u
            float delta_u = delta_vals[i] * u_vals[i];
            float value = delta_u * B_vals[i];

            thread_data[i] = make_float2(decay, value);
        }

        // ========== Step 3: Parallel Associative Scan ==========
        // Use CUB's BlockScan together with a custom SSMScanOp

        using BlockScanT = cub::BlockScan<float2, kNThreads,
                                          cub::BLOCK_SCAN_WARP_SCANS>;

        SSMScanPrefixCallbackOp<float> prefix_op(running_prefix);

        BlockScanT(smem_scan).InclusiveScan(
            thread_data,        // input: (decay, value) pairs
            thread_data,        // output: scanned result
            SSMScanOp<float>(), // associative operator
            prefix_op           // handles state propagation across chunks
        );

        // Update running_prefix for the next chunk
        running_prefix = prefix_op.running_prefix;

        // ========== Step 4: Compute the output ==========
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            // thread_data[i].y is now h[t] (the hidden state)
            // output = C * h
            out_vals[i] += thread_data[i].y * C_vals[i];
        }

        // Write back to HBM (only after accumulation over all state dimensions is complete)
    }
}
```

#### 3. Diagram of the Mathematical Principle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mathematical Foundation of Associative Scan                                │
│                                                                             │
│  Recurrence: h[t] = a[t] * h[t-1] + b[t]                                    │
│                                                                             │
│  Represent as a pair: (a, b), where a=decay and b=input                     │
│                                                                             │
│  Combination operator ⊕:                                                    │
│    (a0, b0) ⊕ (a1, b1) = (a1*a0, a1*b0 + b1)                               │
│                                                                             │
│  Verify associativity:                                                      │
│    [(a0,b0) ⊕ (a1,b1)] ⊕ (a2,b2)                                           │
│    = (a1*a0, a1*b0+b1) ⊕ (a2,b2)                                           │
│    = (a2*a1*a0, a2*(a1*b0+b1)+b2)                                          │
│    = (a2*a1*a0, a2*a1*b0 + a2*b1 + b2)                                     │
│                                                                             │
│    (a0,b0) ⊕ [(a1,b1) ⊕ (a2,b2)]                                           │
│    = (a0,b0) ⊕ (a2*a1, a2*b1+b2)                                           │
│    = (a2*a1*a0, a2*a1*b0 + a2*b1 + b2)   ✓ Equal!                           │
│                                                                             │
│  Therefore, parallel prefix sum can be used!                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Parallel Scan Execution (8-element example)                                │
│                                                                             │
│  Input: (a0,b0) (a1,b1) (a2,b2) (a3,b3) (a4,b4) (a5,b5) (a6,b6) (a7,b7)    │
│                                                                             │
│  Up-sweep (reduce):                                                         │
│  Level 0:  [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]            │
│              \    /        \    /        \    /        \    /              │
│  Level 1:   [0:1]         [2:3]         [4:5]         [6:7]               │
│                 \          /                \          /                   │
│  Level 2:       [0:3]                       [4:7]                          │
│                      \                    /                                │
│  Level 3:            [0:7] (global aggregate)                              │
│                                                                             │
│  Down-sweep (distribute):                                                   │
│  Propagate partial sums back to obtain the inclusive scan result at each position │
│                                                                             │
│  Output: h[0]  h[1]  h[2]  h[3]  h[4]  h[5]  h[6]  h[7]                    │
│                                                                             │
│  Complexity: O(log L) depth, O(L) total work                               │
└─────────────────────────────────────────────────────────────────────────────┘
```


### 3.7 Mamba-2: From Scan to Matrix Multiplication

Mamba-1’s parallel scan has one issue: **it cannot exploit Tensor Cores**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mamba-1 vs Mamba-2                                                         │
│                                                                             │
│  Mamba-1:                                                                   │
│  • Uses parallel associative scan                                           │
│  • The scan operation is element-wise, so it cannot use Tensor Cores        │
│  • State dimension is limited to N=16 (larger values become slower)         │
│  • A100: only reaches 19 TFLOPS (FP32 arithmetic)                           │
│                                                                             │
│  Mamba-2:                                                                   │
│  • Observes that SSM can be written as structured matrix multiplication     │
│  • Replaces scan with matrix multiplication (which can use Tensor Cores!)   │
│  • State dimension can scale to N=64, 128                                   │
│  • A100: can reach 312 TFLOPS (BF16 matmul) - 16x improvement!              │
│                                                                             │
│                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```
