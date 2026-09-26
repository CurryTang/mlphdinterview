# 02 · Parallel Primitives & Memory-Bound Operator Optimization: Reduce, Scan, Histogram & Patterns

This chapter systematically covers the most critical parallel primitives (Reduction, Histogram, Prefix Scan) and memory-bound operator implementations in high-performance kernel engineering. From memory coalescing, bank conflict elimination, and warp shuffle register exchanges to Mamba-1/2 associative scans for modern State Space Models, this guide builds a complete mental framework for memory-bound acceleration.

---

## Part 1: Parallel Reduction Operators (7 Evolutionary Versions of CUDA Reduce)

## 1. What Is a Reduce Kernel

### 1.1 Definition

**Reduce (reduction)** is an operation that aggregates a set of data into a single result using some binary operator, such as addition, maximum, or minimum.

```
Input:   [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇]
Op:      sum (addition)
Output:  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇
```

### 1.2 Why Is Reduce Important in MLSys?

Reduce operations are everywhere in deep learning:

| Scenario | Reduce Type | Example |
|------|-------------|------|
| Loss computation | Sum/Mean | CrossEntropyLoss averaged over the batch |
| Softmax | Max + Sum | Numerical stability requires computing max first |
| LayerNorm/BatchNorm | Mean + Variance | Statistic computation |
| Attention | Sum | Weighted summation after Softmax |
| Gradient aggregation | Sum | AllReduce in distributed training |


## 2. Algorithmic Principles and Parallelization Ideas

### 2.1 Tree Reduction

The core idea behind parallel Reduce is **tree reduction**:

```
Step 0:  [a₀] [a₁] [a₂] [a₃] [a₄] [a₅] [a₆] [a₇]
              ↘↙      ↘↙      ↘↙      ↘↙
Step 1:    [a₀+a₁]  [a₂+a₃]  [a₄+a₅]  [a₆+a₇]
                 ↘  ↙            ↘  ↙
Step 2:      [a₀+a₁+a₂+a₃]  [a₄+a₅+a₆+a₇]
                      ↘    ↙
Step 3:        [a₀+a₁+a₂+a₃+a₄+a₅+a₆+a₇]
```

- **At each step**: the number of active threads is halved
- **Total number of steps**: log₂(n)
- **Work (total operations)**: n-1 (same as the serial case)
- **Span (critical path)**: log₂(n)

### 2.2 Two Indexing Schemes for Tree Reduction

There are two common ways to implement tree reduction on a GPU, and the choice directly affects performance:

#### Method 1: Interleaved Addressing (Increasing Stride)

```
Array: [0] [1] [2] [3] [4] [5] [6] [7]    (8 elements)

Step s=1: stride=1, threads 0,2,4,6 work
  Thread 0: arr[0] += arr[1]    →  [0+1] [ ] [2] [3] [4] [5] [6] [7]
  Thread 2: arr[2] += arr[3]    →  [0+1] [ ] [2+3] [ ] [4] [5] [6] [7]
  Thread 4: arr[4] += arr[5]
  Thread 6: arr[6] += arr[7]
  Result: [0+1] [ ] [2+3] [ ] [4+5] [ ] [6+7] [ ]

Step s=2: stride=2, threads 0,4 work
  Thread 0: arr[0] += arr[2]
  Thread 4: arr[4] += arr[6]
  Result: [0..3] [ ] [ ] [ ] [4..7] [ ] [ ] [ ]

Step s=4: stride=4, thread 0 works
  Thread 0: arr[0] += arr[4]
  Result: [0..7] ...

Index formula: if (tid % (2*s) == 0) arr[tid] += arr[tid + s]
```

**Problems**:
- Active threads are non-contiguous (0,2,4,6 → 0,4 → 0), causing **warp divergence**
- Later accesses use large strides, causing **bank conflicts**

#### Method 2: Sequential Addressing (Decreasing Stride) ✓ Recommended

```
Array: [0] [1] [2] [3] [4] [5] [6] [7]    (8 elements)

Step s=4: stride=4, threads 0,1,2,3 work (first half)
  Thread 0: arr[0] += arr[4]    →  [0+4] [1] [2] [3] | [4] [5] [6] [7]
  Thread 1: arr[1] += arr[5]    →  [0+4] [1+5] [2] [3] | ...
  Thread 2: arr[2] += arr[6]
  Thread 3: arr[3] += arr[7]
  Result: [0+4] [1+5] [2+6] [3+7] | (no longer needed)

Step s=2: stride=2, threads 0,1 work
  Thread 0: arr[0] += arr[2]
  Thread 1: arr[1] += arr[3]
  Result: [part of 0..3+4..7] [the other part] | ...

Step s=1: stride=1, thread 0 works
  Thread 0: arr[0] += arr[1]
  Result: [final sum] ...

Index formula: if (tid < s) arr[tid] += arr[tid + s]
```

**Advantages**:
- Active threads are always contiguous (0,1,2,3 → 0,1 → 0), with **no warp divergence**
- Contiguous threads access contiguous memory, with **no bank conflicts**


## 3. The Evolution of Reduce Kernels: 7 Versions

We will implement a kernel that sums `N = 2^24 = 16M` floats and optimize it step by step.

### Version 0: Interleaved Addressing with Divergent Branching

**The most naive implementation**

```cpp
__global__ void reduce_v0(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    // Each thread loads one element from global memory into shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // Tree reduction
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // ❌ Problem: thread divergence!
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Only thread 0 writes back the result
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

Understanding the memory layout of the variables

![[assets/Pasted image 20251229150638.png]]

Understanding the data flow of the variables

![[assets/Pasted image 20251229151159.png]]

![[assets/Pasted image 20251229151249.png]]


When is syncthreads needed?
![[assets/Pasted image 20251229151721.png]]

**Problem analysis:**

```
Step s=1:  threads 0,2,4,6... active; 1,3,5,7... idle
           → only 16 threads are active in one warp (32 threads)
           → 50% efficiency loss + branch divergence

Step s=2:  threads 0,4,8,12... active
           → 25% efficiency

...and so on
```

**Performance bottlenecks:**
- Warp divergence (threads within the same warp take different branches)
- A large number of idle threads
- The condition `tid % (2*s) == 0` is expensive

---

### Version 1: Interleaved Addressing with Bank Conflicts

**Eliminates branch divergence, but introduces bank conflicts**

```cpp
__global__ void reduce_v1(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // Improvement: contiguous threads perform the same operation
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // Compute the paired index
        int index = 2 * s * tid;
        
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**Improvement:**
- The first N/2 threads execute contiguously, eliminating warp divergence
- But... it introduces a new problem: **shared memory bank conflicts**

```
Initial data: sdata[0..7] = [a, b, c, d, e, f, g, h]

═══════════════════════════════════════════════════════════════════════
                         V0: tid % (2*s) == 0
═══════════════════════════════════════════════════════════════════════

s=1: Active threads satisfy tid % 2 == 0, i.e. tid = 0, 2, 4, 6
     
     tid:    0     1     2     3     4     5     6     7
           active idle  active idle  active idle  active idle
             │           │           │           │
             ▼           ▼           ▼           ▼
           [0]+[1]     [2]+[3]     [4]+[5]     [6]+[7]

     Problem: within one warp, odd threads are idle → Warp Divergence!

s=2: Active threads satisfy tid % 4 == 0, i.e. tid = 0, 4
     
     tid:    0     1     2     3     4     5     6     7
           active idle  idle  idle  active idle  idle  idle
             │                       │
             ▼                       ▼
           [0]+[2]                 [4]+[6]

     Problem: more threads are idle, so divergence gets worse!

═══════════════════════════════════════════════════════════════════════
                      V1: index = 2 * s * tid  
═══════════════════════════════════════════════════════════════════════

s=1: index = 2 * 1 * tid = 2*tid
     
     tid:    0     1     2     3     4     5     6     7
           active active active active idle   idle  idle   idle
             │     │     │     │
             ▼     ▼     ▼     ▼
     index:  0     2     4     6
             │     │     │     │
             ▼     ▼     ▼     ▼
           [0]+[1] [2]+[3] [4]+[5] [6]+[7]

     Improvement: the first 4 threads run contiguously, and the last 4 are contiguously idle → no divergence!

s=2: index = 2 * 2 * tid = 4*tid
     
     tid:    0     1     2     3     4     5     6     7
           active active idle  idle  idle   idle  idle   idle
             │     │
             ▼     ▼
     index:  0     4
             │     │
             ▼     ▼
           [0]+[2] [4]+[6]

     Improvement: the first 2 threads run contiguously → no divergence!
```

## Core idea
```
V0 idea: each thread decides "should I work?"
         tid=0 works, tid=1 does not, tid=2 works, tid=3 does not...
         → interleaved active/idle threads → divergence

V1 idea: each thread computes "which position should I operate on?"
         tid=0 operates on index=0, tid=1 on index=2, tid=2 on index=4...
         → the first N/2 threads are active contiguously → no divergence
```

## Why does V1 still have issues?

V1 eliminates divergence, but introduces **bank conflicts**:
```
s=1:
  Thread 0 accesses sdata[0] and sdata[1]
  Thread 1 accesses sdata[2] and sdata[3]
  → no problem

s=16: index = 32 * tid
  Thread 0 accesses sdata[0]  and sdata[16]   → Bank 0, Bank 16
  Thread 1 accesses sdata[32] and sdata[48]   → Bank 0, Bank 16  ← conflict!
  
  sdata[0]  is in Bank 0
  sdata[32] is in Bank 0  (32 % 32 = 0)
  → two threads access different addresses in the same bank → serialized!
```

**Explanation of bank conflicts:**

Shared memory is divided into 32 banks (one bank per 4 bytes). When multiple threads within the same warp access different addresses in the same bank, the accesses are **serialized**.

```
Step s=1:
Thread 0 accesses sdata[0] and sdata[1]  → Bank 0, Bank 1
Thread 1 accesses sdata[2] and sdata[3]  → Bank 2, Bank 3
...no problem

Step s=16:
Thread 0 accesses sdata[0] and sdata[16]  → Bank 0, Bank 16 ✓
Thread 1 accesses sdata[32] and sdata[48] → Bank 0, Bank 16 ✗ conflict!
...32-way bank conflict!
```

---

### Version 2: Sequential Addressing (Eliminating Bank Conflicts)

**Key improvement: change the reduction direction**

```cpp
__global__ void reduce_v2(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // Improvement: start from a large stride and halve it gradually
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**Why does this eliminate bank conflicts?**

```
blockDim.x = 256, s = 128:
Thread 0 accesses sdata[0] and sdata[128]   → Bank 0, Bank 0 (same bank, same address = broadcast)
Thread 1 accesses sdata[1] and sdata[129]   → Bank 1, Bank 1
...

s = 64:
Thread 0 accesses sdata[0] and sdata[64]    → Bank 0, Bank 0
...

Contiguous threads access contiguous memory, with no conflicts!
```

**Memory access pattern comparison:**

```
Version 1 (Interleaved):         Version 2 (Sequential):
Step 1: [0,1] [2,3] [4,5]...     Step 1: [0,128] [1,129] [2,130]...
Step 2: [0,2] [4,6] [8,10]...    Step 2: [0,64] [1,65] [2,66]...
→ stride keeps increasing, conflicts worsen   → contiguous access, no conflicts
```

---

### Version 3: First Add During Load (Reducing Global Memory Accesses)

```cpp
__global__ void reduce_v3(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Improvement: each thread performs one addition during load
    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();
    
    // Subsequent reduction is the same as v2
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**Effect analysis:**

```
Originally: N elements require N/blockDim.x blocks
Now:        N elements only require N/(blockDim.x*2) blocks

→ Number of blocks is halved
→ Each thread does more work
→ Better hides memory latency
```

**Extension: each thread can load even more elements**

```cpp
// Each thread loads 4 elements
unsigned int i = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
float mySum = 0;
if (i < n) mySum += g_idata[i];
if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
if (i + 2*blockDim.x < n) mySum += g_idata[i + 2*blockDim.x];
if (i + 3*blockDim.x < n) mySum += g_idata[i + 3*blockDim.x];
```

![[assets/Pasted image 20251229161226.png]]



How do we find the optimum? We will introduce the grid-stride loop later.

### Version 4: Unroll Last Warp (Leveraging Implicit Synchronization Within a Warp)

**Key insight**: when s <= 32, all active threads are in the same warp

In CUDA, **threads within the same warp execute in lockstep by default** (SIMT), so `__syncthreads()` is unnecessary!

```cpp
// Warp-level reduction helper (use volatile to prevent compiler optimization)
__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v4(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();
    
    // Only need to reduce while s > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Reduction within the last warp, no synchronization needed
    if (tid < 32) warpReduce(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**Why is `volatile` needed?**

Without `volatile`, the compiler may:
1. Cache `sdata[tid]` in a register
2. Write back to shared memory only after multiple operations
3. Cause other threads to read stale values

`volatile` forces every operation to access shared memory directly.

**Modern alternative: use `__shfl_down_sync`** (see Version 6)

---

### Version 5: Complete Unroll (Fully Unrolling the Loop)

**When blockDim.x is known at compile time, the loop can be fully unrolled**

```cpp
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce_v5(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    
    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockSize < n) mySum += g_idata[i + blockSize];
    sdata[tid] = mySum;
    __syncthreads();
    
    // Fully unrolled reduction loop
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Invocation:
// reduce_v5<256><<<gridSize, 256, 256*sizeof(float)>>>(d_in, d_out, n);
```

**Compiler optimizations:**

Because `blockSize` is a compile-time constant, the compiler will:
1. Eliminate all `if` branches whose conditions are false
2. Fully unroll the loop
3. Generate the leanest possible instruction sequence

---

### Version 6: Warp Shuffle (Best Practice on Modern GPUs)

**Using warp shuffle instructions: zero extra latency and no shared memory needed**

Starting from the Kepler architecture (CC 3.0), CUDA provides **warp shuffle** instructions:

```cpp
// T __shfl_down_sync(unsigned mask, T var, unsigned int delta);
// Let lane i get the value of var from lane i+delta
```

```cpp
__device__ float warpReduceSum(float val) {
    // 0xffffffff means all 32 lanes participate
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    // Each warp first reduces internally
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    // The first few threads of warp 0 collect the results from each warp
    __shared__ float shared[32];  // At most 32 warps
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Only warp 0 performs the final reduction
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__global__ void reduce_v6(float *g_idata, float *g_odata, int n) {
    float sum = 0;
    
    // Grid-stride loop: each thread processes multiple elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        sum += g_idata[i];
    }
    
    // Block-level reduction
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) g_odata[blockIdx.x] = sum;
}
```

**Advantages of warp shuffle:**

| Property | Shared Memory | Warp Shuffle |
|------|--------------|--------------|
| Latency | ~5 cycles | ~1 cycle |
| Synchronization needed | Yes | No (within a warp) |
| Bank conflict | Possible | None |
| Resource usage | Consumes shared memory | None |
![[assets/Pasted image 20260102223044.png]]

```
T __shfl_down_sync(unsigned mask, T var, unsigned int delta);

// mask: which lanes participate (0xffffffff = all 32)
// var:  the value to exchange (in registers)
// delta: get the value from lane+delta

// Return value: lane i gets the value of var from lane i+delta
//               if i+delta >= 32, it returns its own var

### Illustration of `__shfl_down_sync`
__shfl_down_sync(0xffffffff, val, 4):

Before:
Lane:    0    1    2    3    4    5    6    7   ...   28   29   30   31
val:    [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7] ... [a28][a29][a30][a31]

After (return value):
Lane:    0    1    2    3    4    5    6    7   ...   28   29   30   31
result: [a4] [a5] [a6] [a7] [a8] [a9][a10][a11] ... [a28][a29][a30][a31]
                                                      ↑    ↑    ↑    ↑
                                                    out of range, returns its own value

Lane 0 gets the value from Lane 4
Lane 1 gets the value from Lane 5
...
Lane 27 gets the value from Lane 31
Lane 28-31 keep their own values (because 28+4=32 >= 32)
```


Implementation of blockReduceSum
```
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // Results from at most 32 warps
    
    int lane = threadIdx.x % 32;  // Position within the warp
    int wid = threadIdx.x / 32;   // Warp ID
    
    // Level 1: reduce within each warp
    val = warpReduceSum(val);
    
    // Lane 0 of each warp writes to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Level 2: warp 0 reduces the results from all warps
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}
```

### Two-level reduction structure
```
Assume blockDim.x = 256 (8 warps)

┌─────────────────────────────────────────────────────────────────────────┐
│                     Level 1: Intra-warp reduction                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Warp 0 (Thread 0-31):    32 values ─warpReduce─► sum_0 (in Lane 0)    │
│  Warp 1 (Thread 32-63):   32 values ─warpReduce─► sum_1 (in Lane 0)    │
│  Warp 2 (Thread 64-95):   32 values ─warpReduce─► sum_2 (in Lane 0)    │
│  ...                                                                    │
│  Warp 7 (Thread 224-255): 32 values ─warpReduce─► sum_7 (in Lane 0)    │
│                                                                         │
│  Uses: warp shuffle (no shared memory, no synchronization)              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Middle: Write to Shared Memory                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  if (lane == 0) shared[wid] = val;                                     │
│                                                                         │
│  shared[0] = sum_0  (written by Thread 0)                              │
│  shared[1] = sum_1  (written by Thread 32)                             │
│  shared[2] = sum_2  (written by Thread 64)                             │
│  ...                                                                    │
│  shared[7] = sum_7  (written by Thread 224)                            │
│                                                                         │
│  __syncthreads();  // Ensure all warps have finished writing            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  Level 2: Warp 0 reduces warp results                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  // Only the first 8 threads of Warp 0 participate                      │
│  val = (threadIdx.x < 8) ? shared[lane] : 0;                           │
│                                                                         │
│  Warp 0, Lane 0: val = shared[0] = sum_0                               │
│  Warp 0, Lane 1: val = shared[1] = sum_1                               │
│  ...                                                                    │
│  Warp 0, Lane 7: val = shared[7] = sum_7                               │
│  Warp 0, Lane 8-31: val = 0  (padding)                                 │
│                                                                         │
│  if (wid == 0) val = warpReduceSum(val);                               │
│                                                                         │
│  → Lane 0 of Warp 0 holds the final result!                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why do we only need 32 shared-memory entries?
```
Maximum block size = 1024 threads
1024 / 32 = 32 warps
So at most 32 warp results need to be stored

Compared with V2-V5:
  shared[blockDim.x] = 256 or 1024 floats are required

V6:
  only shared[32] = 32 floats are needed!
  
Shared memory usage: 1024 bytes → 128 bytes (8x reduction!)
```

**Explanation of the grid-stride loop:**

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
     i < n; 
     i += blockDim.x * gridDim.x)
```

- Each thread processes more than one element, stepping by `gridSize * blockSize`
- Advantages:
  1. The same code works for inputs of arbitrary size
  2. Grid size can be tuned to optimize occupancy
  3. Better memory bandwidth utilization

```
// Three core elements
for (int i = blockIdx.x * blockDim.x + threadIdx.x;  // 1. starting index
     i < n;                                          // 2. boundary
     i += blockDim.x * gridDim.x)                    // 3. stride
gridDim.x = 2, blockDim.x = 4 (simplified example), n = 20

Total threads = 2 * 4 = 8
Stride = 8

Thread IDs and starting i:
  Block 0: Thread 0 → i=0, Thread 1 → i=1, Thread 2 → i=2, Thread 3 → i=3
  Block 1: Thread 0 → i=4, Thread 1 → i=5, Thread 2 → i=6, Thread 3 → i=7

Global memory indices:
  [ 0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 | 16 17 18 19 ]
    ─────────────────────   ───────────────────────   ───────────
           Round 1                  Round 2              Round 3
           (i)                   (i + 8)             (i + 16)

Thread 0 (Block 0): i = 0, 8, 16     → processes 3 elements
Thread 1 (Block 0): i = 1, 9, 17     → processes 3 elements  
Thread 2 (Block 0): i = 2, 10, 18    → processes 3 elements
Thread 3 (Block 0): i = 3, 11, 19    → processes 3 elements
Thread 0 (Block 1): i = 4, 12        → processes 2 elements (stops because 20 is not < 20)
Thread 1 (Block 1): i = 5, 13        → processes 2 elements
Thread 2 (Block 1): i = 6, 14        → processes 2 elements
Thread 3 (Block 1): i = 7, 15        → processes 2 elements

Total: 4*3 + 4*2 = 20 elements ✓
```


### Version 7: Cooperative Groups + atomicAdd

**The cleanest implementation (CUDA 9.0+)**

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void reduce_v7(float *g_idata, float *g_odata, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    float sum = 0;
    
    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        sum += g_idata[i];
    }
    
    // Warp reduce using cooperative groups
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum += warp.shfl_down(sum, offset);
    }
    
    // Lane 0 of each warp atomically adds into the result
    if (warp.thread_rank() == 0) {
        atomicAdd(g_odata, sum);
    }
}
```

**On the performance of atomicAdd:**

On older architectures, global-memory `atomicAdd` was very slow because it serialized execution. But on modern GPUs:
- Hardware optimizations have significantly improved atomic performance
- For scenarios with only a small number of atomics (one per warp), the overhead is acceptable
- The code is extremely concise and easy to maintain

```
// ═══════════════════════════════════════════════════════════════════════
//                        Traditional approach
// ═══════════════════════════════════════════════════════════════════════

// Compute position within the warp
int lane = threadIdx.x % 32;           // computed manually
int wid = threadIdx.x / 32;            // computed manually

// Warp shuffle
val += __shfl_down_sync(0xffffffff, val, offset);  // mask specified manually

// Block synchronization
__syncthreads();                        // global function


// ═══════════════════════════════════════════════════════════════════════
//                    Cooperative Groups approach
// ═══════════════════════════════════════════════════════════════════════

// Get thread groups
cg::thread_block block = cg::this_thread_block();
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

// Position within the warp
int lane = warp.thread_rank();          // clearer!
int wid = warp.meta_group_rank();       // clearer!

// Warp shuffle
val += warp.shfl_down(val, offset);     // no need to specify the mask manually!

// Block synchronization
block.sync();                           // object-oriented style
```

---

## Part 2: Histogram Operators (Hierarchical Privatization & Contention Minimization)

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

---

## Part 3: Prefix Sum Operators (Work-Efficient Parallel Blelloch Scan)

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

---

## Part 4: Parallel Scan in State Space Models (Mamba-1 / 2)

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

---

## Part 5: Memory-Bound Pattern Library & Kernel Implementation

## 3. Pattern Library: Six Classes of Memory-Bound Kernels

Below, memory-bound kernels are grouped into six categories based on their memory access patterns. When facing a new kernel, first determine which category it belongs to, then combine the corresponding principles for optimization.

---

### Pattern 1: Streaming (Linear Read/Write)

**Typical scenarios**: `out[i] = f(in[i])`, elementwise map, vector addition

**Memory access characteristics**: input is read sequentially, output is written sequentially, and there is no data reuse.

**Applicable principles**: B (coalescing) + E (latency hiding)

**Optimization techniques**: vectorized load/store (`float4`), memory alignment, grid-stride loop, loop unrolling

```cpp
// Vectorized elementwise kernel
// Use float4 for vectorized loads and stores; each memory transaction moves 16 bytes
__global__ void vector_add_v4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n4  // n / 4, i.e. the number of float4 elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n4; i += stride) {
        float4 va = a[i];  // 128-bit load
        float4 vb = b[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[i] = vc;  // 128-bit store
    }
}

//// Approximate definition of float4
// struct float4 {
//    float x, y, z, w;
//};

// Launch configuration
// n is the total number of float elements and must be a multiple of 4 (otherwise the tail needs extra handling)
// vector_add_v4<<<(n/4 + 255) / 256, 256>>>(a4, b4, c4, n/4);
```

> [!note] Why `float4` helps
> After switching to `float4`, each load/store instruction moves 16 bytes instead of 4. This reduces the total number of load/store instructions required, allowing the compiler to schedule the instruction pipeline more effectively and improve ILP. When the 32 threads in a warp execute `float4` loads simultaneously, they generate 4 memory transactions of 128B each, transferring 512 bytes in total.

---

### Pattern 2: Reorder / Permutation

**Typical scenario**: Matrix Transpose

**Core conflict**: the read direction and write direction are orthogonal. If the reads are coalesced, the writes are necessarily strided, and vice versa.

**Solution**: use shared memory as an intermediate buffer. Read along the row direction and coalesce into shared memory, then write from shared memory along the column direction with coalescing.

**Applicable principles**: B (both reads and writes must be coalesced) + C (shared memory as a reordering buffer) + D (avoid bank conflicts)

```cpp
// Shared-memory-tiled matrix transpose
// Input [M x N], output [N x M]
#define TILE_DIM 32
#define BLOCK_ROWS 8  // block dimensions: TILE_DIM x BLOCK_ROWS
                      // each block processes a TILE_DIM x TILE_DIM tile

__global__ void transpose_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    // Columns +1 as padding to eliminate bank conflicts (see explanation below)
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Step 1: load from global memory into shared memory along rows (coalesced read)
    // Each thread is responsible for loading TILE_DIM / BLOCK_ROWS = 4 rows
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }

    __syncthreads();

    // Step 2: after swapping coordinates, write from shared memory to global memory along rows (coalesced write)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < M && (y + j) < N) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Launch configuration
// dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
// dim3 block(TILE_DIM, BLOCK_ROWS);
// transpose_optimized<<<grid, block>>>(input, output, M, N);
```

> [!important] Padding to eliminate bank conflicts
> Shared memory consists of 32 banks, each 4 bytes wide. If the tile has exactly 32 columns, then all elements in the same column map to the same bank, causing a 32-way bank conflict during column-wise accesses. By setting the number of columns to `TILE_DIM + 1 = 33`, elements at the same column position in adjacent rows are shifted by one bank, eliminating the conflict. This trick is widely used in kernels that perform column-wise accesses in shared memory.

---

### Pattern 3: Stencil / Neighborhood

**Typical scenarios**: 1D/2D stencil, image convolution, image-processing filters

**Core characteristic**: each input element is reused by multiple neighboring output points. For a 1D 3-point stencil, for example, each input element is read once by the left, center, and right output points.

**Applicable principles**: C (tile + halo for deterministic reuse) + B (keep halo loading coalesced)

```cpp
// 1D Stencil: out[i] = c0*in[i-R] + c1*in[i-R+1] + ... + c2R*in[i+R]
// R = stencil radius; this example uses R = 4 (9-point stencil)

#define RADIUS 4
#define BLOCK_SIZE 256
// Each block computes BLOCK_SIZE output points
// It needs to load BLOCK_SIZE + 2*RADIUS input points (including halo regions on both sides)

__constant__ float coeff[2 * RADIUS + 1];  // stencil coefficients stored in constant memory

__global__ void stencil_1d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float smem[BLOCK_SIZE + 2 * RADIUS];

    int gidx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lidx = threadIdx.x + RADIUS;  // offset index inside shared memory

    // Step 1: load the center region
    smem[lidx] = (gidx < n) ? input[gidx] : 0.0f;

    // Step 2: load the left halo (handled by the first RADIUS threads)
    if (threadIdx.x < RADIUS) {
        int halo_idx = gidx - RADIUS;
        smem[threadIdx.x] = (halo_idx >= 0) ? input[halo_idx] : 0.0f;
    }

    // Step 3: load the right halo (handled by the last RADIUS threads)
    if (threadIdx.x >= BLOCK_SIZE - RADIUS) {
        int halo_idx = gidx + RADIUS;
        smem[lidx + RADIUS] = (halo_idx < n) ? input[halo_idx] : 0.0f;
    }

    __syncthreads();

    // Step 4: compute the stencil from shared memory, with no global memory access
    if (gidx < n) {
        float result = 0.0f;
        #pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++) {
            result += coeff[j + RADIUS] * smem[lidx + j];
        }
        output[gidx] = result;
    }
}
```

**Memory traffic comparison**:

| Version | Global memory read volume | Explanation |
|------|---------------------|------|
| Naive (no shared memory) | $N \times (2R+1)$ | Each output point reads $2R+1$ inputs from HBM |
| Tiled (using shared memory) | $N + 2R \times \text{num\_blocks}$ | Each input element is essentially loaded from HBM only once |

When the stencil radius $R$ becomes larger, data reuse increases and the benefit of tiling becomes more significant.

---

### Pattern 4: Indirection / Gather (Irregular Reads)

**Typical scenarios**: CSR-format SpMV, gather, embedding lookup

**Core difficulty**: indirect indexing `x[col_idx[j]]` makes access addresses depend on the data itself, making ideal coalescing hard to achieve.

**Applicable principles**: A (include index bytes in the accounting) + E (hide latency through warp-level mapping)

Sparse matrix-vector multiplication (SpMV)

```
CSR (Compressed Sparse Row) stores a sparse matrix using three arrays:
═══════════════════════════════════════════════════════════════════════════════

1. values[nnz]:   all nonzero values, stored row by row
2. col_idx[nnz]:  the column index corresponding to each nonzero value
3. row_ptr[M+1]:  the starting position of each row in values/col_idx

For the matrix above:

values[]:   [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
             ↑       ↑   ↑   ↑   ↑           ↑   ↑
            row0    row0 row1 row1 row2      row2 row3

col_idx[]:  [ 0,  2,  4,  1,  5,  0,  1,  2,  3,  5]
             column index for each nonzero element

row_ptr[]:  [ 0,  3,  5,  9, 10]
              ↑   ↑   ↑   ↑   ↑
             row0 row1 row2 row3 end
             start start start start

The indices between row_ptr[i] and row_ptr[i+1] are the nonzero elements of row i
```


warp-per-row strategy
```
Core idea: one warp (32 threads) cooperatively processes one matrix row
═══════════════════════════════════════════════════════════════════════════════

Why not use Thread-per-Row?
─────────────────────────────
If a row has many nonzero elements (for example, 1000), serial processing by a single thread is too slow

Advantages of Warp-per-Row:
─────────────────────────────
1. 32 threads process a row's nonzero elements in parallel
2. Accesses to values[] and col_idx[] are contiguous -> Coalesced Access
3. Uses Warp Shuffle for reduction, with no Shared Memory needed
```

```cpp
// CSR-format SpMV: y = A * x
// CSR storage: row_ptr[M+1], col_idx[nnz], values[nnz]
//
// Mapping strategy:
//   Row lengths are fairly uniform: thread-per-row
//   Row lengths vary widely: warp-per-row (this example uses this strategy)

// Warp-per-row: each warp (32 threads) processes one matrix row
// Advantages:
//   1. Threads within a warp access consecutive col_idx and values entries (coalesced)
//   2. Uses warp shuffle for reduction, with no shared memory or atomic needed
//   3. 32 threads issue loads simultaneously, effectively hiding latency
__global__ void spmv_csr_warp_per_row(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= num_rows) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    float sum = 0.0f;

    // The 32 threads in the warp iterate over this row's nonzeros with stride 32
    for (int j = row_start + lane; j < row_end; j += 32) {
        // col_idx[j], values[j]: contiguous accesses, coalesced
        // x[col_idx[j]]: random access, relies on L2 cache and latency hiding
        sum += values[j] * x[col_idx[j]];
    }

    // Warp-level reduction (warp shuffle), no shared memory or barrier needed
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[warp_id] = sum;
    }
}

// Launch configuration
// int threads_per_block = 256;  // each block contains 8 warps
// int num_blocks = (num_rows * 32 + threads_per_block - 1) / threads_per_block;
// spmv_csr_warp_per_row<<<num_blocks, threads_per_block>>>(...);
```

If all lanes need the result, you can broadcast with `__shfl_sync`, or use `__shfl_xor_sync` for a butterfly reduction. But here we only need to write a single `y[warp_id]`, so it is enough for lane 0 alone to hold the result.

> [!note] The bandwidth ceiling for sparse kernels
> For sparse kernels, the achievable bandwidth ceiling is often lower than the theoretical HBM peak. The access pattern of `x[col_idx[j]]` is determined by the sparsity structure of the matrix and cannot be fully controlled at the kernel level. As a result, the optimization goal shifts toward "reducing randomness in accesses," for example by reordering matrix columns to improve locality when accessing vector `x`.

---

### Pattern 5: Scatter / Atomic (Irregular Writes + Contention)

**Typical scenarios**: histogram, scatter-add, some graph algorithms

**Core difficulty**: the write address depends on the data values, and hot destinations (such as high-frequency bins) cause severe serialization of atomic operations.

**Applicable principle**: D (hierarchical privatization)

The optimization method for this class of kernels was already presented in detail in the three Histogram versions from the previous lecture. The core strategy is as follows:

```
Level 0: Global atomic — all threads contend for global memory
  ↓ Privatize
Level 1: Block-level shared memory atomic — contention shrinks to 256 threads
  ↓ Further privatize
Level 2: Warp-level / register-local accumulation — contention shrinks to 32 threads or is fully eliminated
  ↓ Final reduction
Write back to global memory — the number of atomic calls drops from N to num_bins × num_blocks
```

---

### Pattern 6: Filter / Compaction (Conditional Filtering)

**Typical scenarios**: stream compaction, remove-zero, predicate filter

#### What is Stream Compaction?

**Filter out elements that satisfy a condition and store them compactly**

```
Input:      [ 3, -1, 4, 0, -2, 5, 0, 1 ]
Condition:  element > 0
Output:     [ 3, 4, 5, 1 ]
```

#### Why is it hard to parallelize?

Each thread does not know its own output position — **solve it with an Exclusive Prefix Sum**:

```
flag:   [ 1,  0,  1,  0,  0,  1,  0,  1 ]   ← mark elements satisfying the condition
scan:   [ 0,  1,  1,  2,  2,  2,  3,  3 ]   ← exclusive scan

scan[i] = "how many satisfied the condition before me" = my output position
```

#### Three-step workflow

| Step | What it does | Code |
|-----|-------|------|
| **Flag** | Mark elements that satisfy the condition | `flag = (val > 0) ? 1 : 0` |
| **Scan** | Exclusive prefix sum (Blelloch algorithm) | Up-sweep + Down-sweep |
| **Scatter** | Write to the correct position | `output[offset + scan[tid]] = val` |


```cpp
// Stream Compaction: filter elements > 0 from input and write them compactly into output
// Fused implementation: complete flag, block-level scan, and scatter inside a single kernel

#define BLOCK_SIZE 256

__global__ void stream_compaction(
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ output_count,  // total number of output elements
    int n
) {
    __shared__ int scan[BLOCK_SIZE];
    __shared__ int block_output_offset;

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // Step 1: Flag — mark elements satisfying the condition
    int val = 0;
    int flag = 0;
    if (gid < n) {
        val = input[gid];
        flag = (val > 0) ? 1 : 0;
    }
    scan[tid] = flag;
    __syncthreads();

    // Step 2: Block-level exclusive scan (Blelloch algorithm)

    // Up-sweep
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < BLOCK_SIZE) {
            scan[ai] += scan[ai - offset];
        }
        __syncthreads();
    }

    // Extract the total number of qualifying elements in the block and allocate output space with one atomic
    if (tid == 0) {
        int block_total = scan[BLOCK_SIZE - 1];
        block_output_offset = atomicAdd(output_count, block_total);
        scan[BLOCK_SIZE - 1] = 0;
    }
    __syncthreads();

    // Down-sweep
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < BLOCK_SIZE) {
            int temp = scan[ai - offset];
            scan[ai - offset] = scan[ai];
            scan[ai] += temp;
        }
        __syncthreads();
    }

    // Step 3: Scatter — write qualifying elements to the correct positions in the output array
    if (gid < n && flag) {
        output[block_output_offset + scan[tid]] = val;
    }
}
```

> [!tip] Key optimization
> `atomicAdd(output_count, block_total)` is called `num_blocks` times rather than N times. This combines Principle D (reduce contention) with Principle A (reduce the memory cost of atomics): each block computes local offsets internally via scan, and only a single atomic operation is needed at the end to allocate output space.

---

## 4. Optimization Checklist

When optimizing a memory-bound kernel, it is recommended to check the following items in order:

| Step | Item | How to observe it |
|------|------|----------|
| 1 | Compute arithmetic intensity and bandwidth upper bound | Roofline model, manual calculation |
| 2 | Measure effective bandwidth | Bytes / kernel_time, compare against the theoretical peak |
| 3 | Check coalescing | `ncu`: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` |
| 4 | Check contention and synchronization overhead | `ncu`: atomic throughput, barrier wait time |
| 5 | Tune occupancy and ILP | grid-stride loop, loop unrolling, multiple elements per thread |

The importance of this order is that if coalescing is not satisfied, then tuning occupancy will yield only limited benefit. You should first ensure that the memory access pattern is correct, and only then move on to higher-level tuning.

---

## 5. Summary

| Principle | Optimization goal | Typical applicable kernels |
|------|----------|----------------|
| A Byte accounting | Accurately quantify memory traffic | All kernels |
| B Coalescing | Consecutive addresses within a warp, fewer memory transactions | streaming, transpose |
| C Explicit reuse | Reuse data multiple times after loading it from HBM into SRAM | stencil, some sparse kernels |
| D Reduce contention | Lower the serialization cost of barriers and atomics | histogram, scatter, compaction |
| E Latency hiding | Mask memory latency with parallelism and ILP | sparse, gather |

The optimizations for Reduce, Histogram, and Scan in the previous two lectures already covered all of these principles (`coalescing`, `first add during load`, `warp shuffle`, `ILP`, `grid-stride loop`). The purpose of this lecture is to abstract those techniques, previously tied to specific examples, into a general analytical framework.

When facing a new memory-bound kernel, first determine which Pattern (1-6) it belongs to, and then formulate an optimization strategy by combining the corresponding principles.

---

## Part 6: Practice Exercises & Review Self-Checks

### Module A: Reduce Kernel Evolution & Optimization

<details class="exercise">
<summary><span class="q-label">Exercise 1</span> <span class="q-text">Why is the naive Reduce kernel slow?</span></summary>

A naive reduce implementation suffers from severe branch divergence (`if (tid % (2*stride) == 0)` where only half the threads remain active each step) and shared memory bank conflicts (strided accesses mapping to the same bank). Sequential addressing, loop unrolling, and warp shuffle instructions eliminate these bottlenecks.

</details>

<details class="exercise">
<summary><span class="q-label">Exercise 2</span> <span class="q-text">What is the advantage of using warp shuffle primitives (__shfl_down_sync)?</span></summary>

Warp shuffle allows threads within the same warp to exchange register values directly without touching shared memory or requiring `__syncthreads()`. This reduces shared memory footprint to just 32 elements per block and cuts instruction latency drastically.

</details>

### Module B: Histogram & Scan Parallel Primitives

<details class="exercise">
<summary><span class="q-label">Exercise 1</span> <span class="q-text">Why is histogram more challenging than reduction?</span></summary>

In reduction, the target location for each element is statically deterministic and tree-structured. In histogram, data values determine which bin is incremented, leading to arbitrary atomic collision and serializing memory writes. Hierarchical privatization (global -> shared -> local) resolves this contention.

</details>

<details class="exercise">
<summary><span class="q-label">Exercise 2</span> <span class="q-text">What is the core invariant of the Blelloch work-efficient parallel scan?</span></summary>

The Blelloch scan computes prefix sums in $O(N)$ total operations via an Up-Sweep (reduction tree) followed by a Down-Sweep (distribution tree). By zeroing the root and passing values down, each node receives the sum of preceding elements, maintaining work-efficiency.

</details>

### Module C: Memory-Bound General Tuning Experience

<details class="exercise">
<summary><span class="q-label">Exercise 1</span> <span class="q-text">How does padding eliminate shared memory bank conflicts in matrix transpose?</span></summary>

When transposing a 32x32 tile, consecutive rows in shared memory start at bank index `(row * 32) % 32 = 0`, causing column-wise reads to generate 32-way bank conflicts. Declaring `__shared__ float tile[32][33]` adds a 1-element pad per row, skewing column accesses across distinct banks.

</details>
