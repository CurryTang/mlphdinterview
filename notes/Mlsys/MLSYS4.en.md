# Complete Guide to CUDA Reduce Kernels: From Principles to Optimization

## Contents
1. [What Is a Reduce Kernel](#1-what-is-a-reduce-kernel)
2. [Algorithmic Principles and Parallelization Ideas](#2-algorithmic-principles-and-parallelization-ideas)
3. [The Evolution of Reduce Kernels: 7 Versions](#3-the-evolution-of-reduce-kernels-7-versions)
4. [Roofline Analysis and Performance Modeling](#4-roofline-analysis-and-performance-modeling)
5. [Profiling](#5-profiling)

---

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

## 4. Roofline Analysis and Performance Modeling

### 4.1 Roofline Analysis of Reduce

**Computing the operational intensity:**

For sum reduction over N elements:
- **FLOPs**: N-1 additions ≈ N
- **Bytes**: reading N floats = 4N bytes, writing 1 float ≈ 4N bytes
- **OI = N / 4N = 0.25 FLOP/Byte**

This is an **extremely low operational intensity**!

**Compared with hardware (taking A100 as an example):**

```
A100 specs:
- Peak FP32 Performance: 19.5 TFLOPS
- Memory Bandwidth: 2039 GB/s

Ridge point (slope = bandwidth, roof = peak compute):
OI_ridge = 19.5 TFLOPS / 2.039 TB/s = 9.56 FLOP/Byte

Reduce OI = 0.25 << 9.56

→ Reduce is strongly memory-bound!
```

**Roofline illustration:**

```
Performance (TFLOPS)
    ^
19.5├─────────────────────────┬─────────
    │                        /│
    │                       / │
    │                      /  │
    │                     /   │
    │                    /    │
    │                   /     │
 0.5├──────────*───────/      │  ← Reduce (OI=0.25)
    │         /       /       │
    │        /       /        │
    │       /       /         │
    │      /       /          │
    │     /       /           │
    └─────┴───────┴───────────┴──────────→ OI
         0.25    9.56
         
* At OI=0.25, the theoretical maximum performance is:
  0.25 × 2039 GB/s = 509.75 GFLOPS ≈ 0.5 TFLOPS
```

Profiling (A6000)

You can see that the v3 first-add optimization is especially important.

| Kernel | Time(ms) | BW(GB/s) | BW Eff% | GFLOPS | vs V0 |
|--------|----------|----------|---------|--------|-------|
| V0-Naive | 0.3326 | 201.79 | 26.3 | 50.45 | 1.00x |
| V1-NoDiverg | 0.2422 | 277.08 | 36.1 | 69.27 | 1.37x |
| V2-Sequential | 0.2331 | 287.94 | 37.5 | 71.99 | 1.43x |
| V3-FirstAdd | 0.1253 | 535.56 | 69.7 | 133.89 | 2.65x |
| V4-UnrollWarp | 0.1081 | 620.63 | 80.8 | 155.16 | 3.08x |
| V5-FullUnroll | 0.1081 | 621.03 | 80.9 | 155.26 | 3.08x |
| V6-Shuffle | 0.1141 | 588.15 | 76.6 | 147.04 | 2.91x |
| V7-CoopGroups | 0.1110 | 604.48 | 78.7 | 151.12 | 3.00x |
| PyTorch-sum | 0.1016 | 660.73 | 86.0 | 165.18 | 3.27x |
### 4.3  Optimization Strategies for Memory-Bound Kernels

Since Reduce is memory-bound, the optimization goal is to **maximize memory-bandwidth utilization**:

| Strategy | Effect |
|------|------|
| Coalesced access | 32 threads access 128 contiguous bytes |
| Reduce the number of reads | First add during load |
| Reduce reliance on shared memory | Warp shuffle |
| Increase ILP | Loop unrolling |
| Grid-stride loop | Better occupancy |



## Ref

1. [NVIDIA Parallel Reduction (Mark Harris)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
2. [CUB Library Documentation](https://nvlabs.github.io/cub/)
