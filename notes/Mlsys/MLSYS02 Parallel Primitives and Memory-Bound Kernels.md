# 02 · 经典并行原语与 Memory-Bound 算子手撕指南

本章系统掌握高性能算子工程中最核心的经典并行原语（Reduction、Histogram、Prefix Scan）与 Memory-Bound 算子手撕代码库。从硬件访存合并、Bank Conflict 消除、Warp Shuffle 寄存器交换，到 Mamba-1/2 的前沿状态空间模型关联扫描，构建完整的访存受限算子调优心智。

---

## 第一部分：并行规约算子（CUDA Reduce Kernel 的 7 个演进版本）

## 1. 什么是 Reduce Kernel

### 1.1 定义

**Reduce（归约）** 是一种将一组数据通过某种二元运算（如加法、最大值、最小值）聚合成单个结果的操作。

```
输入:  [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇]
操作:  sum (加法)
输出:  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇
```

### 1.2 为什么 Reduce 在 MLSys 中重要？

Reduce 操作在深度学习中无处不在：

| 场景 | Reduce 类型 | 示例 |
|------|-------------|------|
| Loss 计算 | Sum/Mean | CrossEntropyLoss 对 batch 求平均 |
| Softmax | Max + Sum | 数值稳定性需要先求 max |
| LayerNorm/BatchNorm | Mean + Variance | 统计量计算 |
| Attention | Sum | Softmax 后的加权求和 |
| 梯度聚合 | Sum | 分布式训练 AllReduce |


## 2. 算法原理与并行化思想

### 2.1 树形归约（Tree Reduction）

并行 Reduce 的核心思想是**树形归约**：

```
Step 0:  [a₀] [a₁] [a₂] [a₃] [a₄] [a₅] [a₆] [a₇]
              ↘↙      ↘↙      ↘↙      ↘↙
Step 1:    [a₀+a₁]  [a₂+a₃]  [a₄+a₅]  [a₆+a₇]
                 ↘  ↙            ↘  ↙
Step 2:      [a₀+a₁+a₂+a₃]  [a₄+a₅+a₆+a₇]
                      ↘    ↙
Step 3:        [a₀+a₁+a₂+a₃+a₄+a₅+a₆+a₇]
```

- **每一步**：活跃线程数减半
- **总步数**：log₂(n)
- **Work（总操作数）**：n-1（与串行相同）
- **Span（关键路径）**：log₂(n)

### 2.2 树形归约的两种索引方式

在 GPU 上实现树形归约有两种常见的索引方式，选择不同会直接影响性能：

#### 方式一：Interleaved Addressing（步长递增）

```
数组: [0] [1] [2] [3] [4] [5] [6] [7]    (8个元素)

Step s=1: 步长=1，线程 0,2,4,6 工作
  Thread 0: arr[0] += arr[1]    →  [0+1] [ ] [2] [3] [4] [5] [6] [7]
  Thread 2: arr[2] += arr[3]    →  [0+1] [ ] [2+3] [ ] [4] [5] [6] [7]
  Thread 4: arr[4] += arr[5]
  Thread 6: arr[6] += arr[7]
  结果: [0+1] [ ] [2+3] [ ] [4+5] [ ] [6+7] [ ]

Step s=2: 步长=2，线程 0,4 工作
  Thread 0: arr[0] += arr[2]
  Thread 4: arr[4] += arr[6]
  结果: [0..3] [ ] [ ] [ ] [4..7] [ ] [ ] [ ]

Step s=4: 步长=4，线程 0 工作
  Thread 0: arr[0] += arr[4]
  结果: [0..7] ...

索引公式: if (tid % (2*s) == 0) arr[tid] += arr[tid + s]
```

**问题**：
- 活跃线程不连续（0,2,4,6 → 0,4 → 0），导致 **warp divergence**
- 后期访问步长大，导致 **bank conflict**

#### 方式二：Sequential Addressing（步长递减）✓ 推荐

```
数组: [0] [1] [2] [3] [4] [5] [6] [7]    (8个元素)

Step s=4: 步长=4，线程 0,1,2,3 工作（前半部分）
  Thread 0: arr[0] += arr[4]    →  [0+4] [1] [2] [3] | [4] [5] [6] [7]
  Thread 1: arr[1] += arr[5]    →  [0+4] [1+5] [2] [3] | ...
  Thread 2: arr[2] += arr[6]
  Thread 3: arr[3] += arr[7]
  结果: [0+4] [1+5] [2+6] [3+7] | (不再需要)

Step s=2: 步长=2，线程 0,1 工作
  Thread 0: arr[0] += arr[2]
  Thread 1: arr[1] += arr[3]
  结果: [0..3+4..7的一部分] [另一部分] | ...

Step s=1: 步长=1，线程 0 工作
  Thread 0: arr[0] += arr[1]
  结果: [最终和] ...

索引公式: if (tid < s) arr[tid] += arr[tid + s]
```

**优势**：
- 活跃线程始终连续（0,1,2,3 → 0,1 → 0），**无 warp divergence**
- 连续线程访问连续内存，**无 bank conflict**


## 3. Reduce Kernel 的演进：7 个版本

我们将实现一个对 `N = 2^24 = 16M` 个 float 求和的 kernel，逐步优化。

### Version 0: Interleaved Addressing with Divergent Branching

**最朴素的实现**

```cpp
__global__ void reduce_v0(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    // 每个线程从 global memory 加载一个元素到 shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // 树形归约
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // ❌ 问题：线程发散！
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 只有 thread 0 写回结果
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

理解变量的内存结构

![[assets/Pasted image 20251229150638.png]]

理解变量的数据流向

![[assets/Pasted image 20251229151159.png]]

![[assets/Pasted image 20251229151249.png]]


什么情况下需要syncthreads?
![[assets/Pasted image 20251229151721.png]]

**问题分析：**

```
Step s=1:  线程 0,2,4,6... 活跃，1,3,5,7... 空闲
           → 一个 warp (32线程) 中只有 16 个活跃
           → 50% 效率损失 + 分支发散

Step s=2:  线程 0,4,8,12... 活跃
           → 25% 效率

...以此类推
```

**性能瓶颈：**
- Warp divergence（同一 warp 内线程走不同分支）
- 大量线程空闲
- 条件判断 `tid % (2*s) == 0` 开销大

---

### Version 1: Interleaved Addressing with Bank Conflicts

**消除分支发散，但引入 Bank Conflict**

```cpp
__global__ void reduce_v1(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // 改进：连续的线程执行相同操作
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // 计算配对的索引
        int index = 2 * s * tid;
        
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**改进：**
- 前 N/2 个线程连续执行，消除了 warp divergence
- 但...引入了新问题：**Shared Memory Bank Conflict**

```
初始数据: sdata[0..7] = [a, b, c, d, e, f, g, h]

═══════════════════════════════════════════════════════════════════════
                         V0: tid % (2*s) == 0
═══════════════════════════════════════════════════════════════════════

s=1: 活跃线程是 tid % 2 == 0，即 tid = 0, 2, 4, 6
     
     tid:    0     1     2     3     4     5     6     7
           活跃   空闲  活跃   空闲  活跃   空闲  活跃   空闲
             │           │           │           │
             ▼           ▼           ▼           ▼
           [0]+[1]     [2]+[3]     [4]+[5]     [6]+[7]

     问题: 一个 warp 内，奇数线程空闲 → Warp Divergence!

s=2: 活跃线程是 tid % 4 == 0，即 tid = 0, 4
     
     tid:    0     1     2     3     4     5     6     7
           活跃   空闲  空闲  空闲  活跃   空闲  空闲  空闲
             │                       │
             ▼                       ▼
           [0]+[2]                 [4]+[6]

     问题: 更多线程空闲，divergence 更严重!

═══════════════════════════════════════════════════════════════════════
                      V1: index = 2 * s * tid  
═══════════════════════════════════════════════════════════════════════

s=1: index = 2 * 1 * tid = 2*tid
     
     tid:    0     1     2     3     4     5     6     7
           活跃   活跃  活跃   活跃  空闲   空闲  空闲   空闲
             │     │     │     │
             ▼     ▼     ▼     ▼
     index:  0     2     4     6
             │     │     │     │
             ▼     ▼     ▼     ▼
           [0]+[1] [2]+[3] [4]+[5] [6]+[7]

     改进: 前 4 个线程连续执行，后 4 个连续空闲 → 无 Divergence!

s=2: index = 2 * 2 * tid = 4*tid
     
     tid:    0     1     2     3     4     5     6     7
           活跃   活跃  空闲  空闲  空闲   空闲  空闲  空闲
             │     │
             ▼     ▼
     index:  0     4
             │     │
             ▼     ▼
           [0]+[2] [4]+[6]

     改进: 前 2 个线程连续执行 → 无 Divergence!
```

## 核心思想
```
V0 思路: 每个线程判断"我该不该工作"
         tid=0 工作，tid=1 不工作，tid=2 工作，tid=3 不工作...
         → 交错的活跃/空闲 → Divergence

V1 思路: 每个线程计算"我要操作哪个位置"
         tid=0 操作 index=0，tid=1 操作 index=2，tid=2 操作 index=4...
         → 前 N/2 个线程连续活跃 → 无 Divergence
```

## 为什么 V1 仍有问题？

V1 消除了 divergence，但引入了 **Bank Conflict**：
```
s=1 时:
  Thread 0 访问 sdata[0] 和 sdata[1]
  Thread 1 访问 sdata[2] 和 sdata[3]
  → 没问题

s=16 时: index = 32 * tid
  Thread 0 访问 sdata[0]  和 sdata[16]   → Bank 0, Bank 16
  Thread 1 访问 sdata[32] 和 sdata[48]   → Bank 0, Bank 16  ← 冲突!
  
  sdata[0]  在 Bank 0
  sdata[32] 在 Bank 0  (32 % 32 = 0)
  → 两个线程访问同一个 Bank 的不同地址 → 串行化!
```

**Bank Conflict 解释：**

Shared Memory 分成 32 个 bank（每 4 字节一个 bank）。当同一 warp 内的多个线程访问同一 bank 的不同地址时，访问会**串行化**。

```
Step s=1:
Thread 0 访问 sdata[0] 和 sdata[1]  → Bank 0, Bank 1
Thread 1 访问 sdata[2] 和 sdata[3]  → Bank 2, Bank 3
...没问题

Step s=16:
Thread 0 访问 sdata[0] 和 sdata[16]  → Bank 0, Bank 16 ✓
Thread 1 访问 sdata[32] 和 sdata[48] → Bank 0, Bank 16 ✗ 冲突!
...32-way bank conflict!
```

---

### Version 2: Sequential Addressing (消除 Bank Conflict)

**关键改进：改变归约方向**

```cpp
__global__ void reduce_v2(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // 改进：从大步长开始，逐步减半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**为什么消除了 Bank Conflict？**

```
blockDim.x = 256, s = 128:
Thread 0 访问 sdata[0] 和 sdata[128]   → Bank 0, Bank 0 (同一bank同一地址=广播)
Thread 1 访问 sdata[1] 和 sdata[129]   → Bank 1, Bank 1
...

s = 64:
Thread 0 访问 sdata[0] 和 sdata[64]    → Bank 0, Bank 0
...

连续线程访问连续内存，无冲突！
```

**访存模式对比：**

```
Version 1 (Interleaved):         Version 2 (Sequential):
Step 1: [0,1] [2,3] [4,5]...     Step 1: [0,128] [1,129] [2,130]...
Step 2: [0,2] [4,6] [8,10]...    Step 2: [0,64] [1,65] [2,66]...
→ 步长越来越大，冲突加剧          → 连续访问，无冲突
```

---

### Version 3: First Add During Load (减少 Global Memory 访问)

```cpp
__global__ void reduce_v3(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 改进：每个线程在加载时就做一次加法
    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();
    
    // 后续归约同 v2
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**效果分析：**

```
原来：N 个元素需要 N/blockDim.x 个 block
现在：N 个元素只需要 N/(blockDim.x*2) 个 block

→ Block 数量减半
→ 每个线程做更多工作
→ 更好地隐藏内存延迟
```

**扩展：可以让每个线程加载更多元素**

```cpp
// 每个线程加载 4 个元素
unsigned int i = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
float mySum = 0;
if (i < n) mySum += g_idata[i];
if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
if (i + 2*blockDim.x < n) mySum += g_idata[i + 2*blockDim.x];
if (i + 3*blockDim.x < n) mySum += g_idata[i + 3*blockDim.x];
```

![[assets/Pasted image 20251229161226.png]]



如何找最优呢？后面会介绍grid-strided loop

### Version 4: Unroll Last Warp (利用 Warp 内隐式同步)

**关键洞察**：当 s <= 32 时，所有活跃线程都在同一个 warp 内

在 CUDA 中，**同一 warp 内的线程天然同步执行**（SIMT），不需要 `__syncthreads()`！

```cpp
// Warp 内归约辅助函数（使用 volatile 防止编译器优化）
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
    
    // 只需要归约到 s > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 最后一个 warp 内的归约，无需同步
    if (tid < 32) warpReduce(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**为什么需要 `volatile`？**

没有 `volatile`，编译器可能会：
1. 将 `sdata[tid]` 缓存到寄存器
2. 多次操作后才写回 shared memory
3. 导致其他线程读到旧值

`volatile` 强制每次操作都真正访问 shared memory。

**现代替代方案：使用 `__shfl_down_sync`**（见 Version 6）

---

### Version 5: Complete Unroll (完全展开循环)

**当 blockDim.x 在编译时已知，可以完全展开循环**

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
    
    // 完全展开的归约循环
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

// 调用方式：
// reduce_v5<256><<<gridSize, 256, 256*sizeof(float)>>>(d_in, d_out, n);
```

**编译器优化：**

由于 `blockSize` 是编译时常量，编译器会：
1. 消除所有不满足条件的 `if` 分支
2. 完全展开循环
3. 生成最精简的指令序列

---

### Version 6: Warp Shuffle (现代 GPU 最佳实践)

**使用 Warp Shuffle 指令：零延迟，无需 shared memory**

从 Kepler 架构（CC 3.0）开始，CUDA 提供了 **warp shuffle** 指令：

```cpp
// T __shfl_down_sync(unsigned mask, T var, unsigned int delta);
// 让 lane i 获取 lane i+delta 的 var 值
```

```cpp
__device__ float warpReduceSum(float val) {
    // 0xffffffff 表示所有 32 个 lane 都参与
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    // 每个 warp 先内部归约
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    // Warp 0 的前几个线程收集各 warp 的结果
    __shared__ float shared[32];  // 最多 32 个 warp
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // 只有 warp 0 做最后归约
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__global__ void reduce_v6(float *g_idata, float *g_odata, int n) {
    float sum = 0;
    
    // Grid-stride loop：每个线程处理多个元素
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        sum += g_idata[i];
    }
    
    // Block 内归约
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) g_odata[blockIdx.x] = sum;
}
```

**Warp Shuffle 优势：**

| 特性 | Shared Memory | Warp Shuffle |
|------|--------------|--------------|
| 延迟 | ~5 cycles | ~1 cycle |
| 是否需要同步 | 是 | 否（warp内） |
| Bank conflict | 可能 | 不存在 |
| 资源消耗 | 占用 shared memory | 无 |
![[assets/Pasted image 20260102223044.png]]

```
T __shfl_down_sync(unsigned mask, T var, unsigned int delta);

// mask: 哪些 lane 参与 (0xffffffff = 全部 32 个)
// var:  要交换的值 (在寄存器中)
// delta: 从 lane+delta 获取值

// 返回值: lane i 获得 lane i+delta 的 var 值
//         如果 i+delta >= 32，返回自己的 var

### 图解 `__shfl_down_sync`
__shfl_down_sync(0xffffffff, val, 4):

Before:
Lane:    0    1    2    3    4    5    6    7   ...   28   29   30   31
val:    [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7] ... [a28][a29][a30][a31]

After (返回值):
Lane:    0    1    2    3    4    5    6    7   ...   28   29   30   31
result: [a4] [a5] [a6] [a7] [a8] [a9][a10][a11] ... [a28][a29][a30][a31]
                                                      ↑    ↑    ↑    ↑
                                                    超出范围，返回自己的值

Lane 0 得到了 Lane 4 的值
Lane 1 得到了 Lane 5 的值
...
Lane 27 得到了 Lane 31 的值
Lane 28-31 得到自己的值（因为 28+4=32 >= 32）
```


blockreducesum的实现
```
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // 最多 32 个 warp 的结果
    
    int lane = threadIdx.x % 32;  // warp 内位置
    int wid = threadIdx.x / 32;   // warp 编号
    
    // 第一层: 每个 warp 内部归约
    val = warpReduceSum(val);
    
    // 每个 warp 的 lane 0 写入 shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // 第二层: warp 0 归约所有 warp 的结果
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}
```

### 两层归约结构
```
假设 blockDim.x = 256 (8 个 warp)

┌─────────────────────────────────────────────────────────────────────────┐
│                        第一层: Warp 内归约                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Warp 0 (Thread 0-31):    32 个值 ──warpReduce──► sum_0 (在 Lane 0)    │
│  Warp 1 (Thread 32-63):   32 个值 ──warpReduce──► sum_1 (在 Lane 0)    │
│  Warp 2 (Thread 64-95):   32 个值 ──warpReduce──► sum_2 (在 Lane 0)    │
│  ...                                                                    │
│  Warp 7 (Thread 224-255): 32 个值 ──warpReduce──► sum_7 (在 Lane 0)    │
│                                                                         │
│  使用: warp shuffle (无 shared memory，无同步)                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      中间: 写入 Shared Memory                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  if (lane == 0) shared[wid] = val;                                     │
│                                                                         │
│  shared[0] = sum_0  (Thread 0 写入)                                    │
│  shared[1] = sum_1  (Thread 32 写入)                                   │
│  shared[2] = sum_2  (Thread 64 写入)                                   │
│  ...                                                                    │
│  shared[7] = sum_7  (Thread 224 写入)                                  │
│                                                                         │
│  __syncthreads();  // 确保所有 warp 都写完                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      第二层: Warp 0 归约 Warp 结果                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  // 只有 Warp 0 的前 8 个线程参与                                       │
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
│  → Warp 0 的 Lane 0 持有最终结果！                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 为什么只需要 32 个 shared memory？
```
最大 block 大小 = 1024 线程
1024 / 32 = 32 个 warp
所以最多只有 32 个 warp 结果需要存储

对比 V2-V5:
  需要 shared[blockDim.x] = 256 或 1024 个 float

V6:
  只需要 shared[32] = 32 个 float！
  
Shared memory 使用量: 1024 bytes → 128 bytes (8倍减少！)
```

**Grid-Stride Loop 解释：**

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
     i < n; 
     i += blockDim.x * gridDim.x)
```

- 每个线程不只处理一个元素，而是间隔 `gridSize * blockSize` 处理
- 优势：
  1. 同一份代码适用于任意大小的输入
  2. 可以调整 grid size 优化 occupancy
  3. 更好地利用内存带宽

```
// 核心三要素
for (int i = blockIdx.x * blockDim.x + threadIdx.x;  // 1. 起始位置
     i < n;                                          // 2. 边界
     i += blockDim.x * gridDim.x)                    // 3. 步长
gridDim.x = 2, blockDim.x = 4 (简化示例), n = 20

总线程数 = 2 * 4 = 8
步长 (stride) = 8

线程编号和起始 i:
  Block 0: Thread 0 → i=0, Thread 1 → i=1, Thread 2 → i=2, Thread 3 → i=3
  Block 1: Thread 0 → i=4, Thread 1 → i=5, Thread 2 → i=6, Thread 3 → i=7

Global Memory 索引:
  [ 0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 | 16 17 18 19 ]
    ─────────────────────   ───────────────────────   ───────────
           第1轮                    第2轮                第3轮
           (i)                   (i + 8)             (i + 16)

Thread 0 (Block 0): i = 0, 8, 16     → 处理 3 个元素
Thread 1 (Block 0): i = 1, 9, 17     → 处理 3 个元素  
Thread 2 (Block 0): i = 2, 10, 18    → 处理 3 个元素
Thread 3 (Block 0): i = 3, 11, 19    → 处理 3 个元素
Thread 0 (Block 1): i = 4, 12        → 处理 2 个元素 (20 > 20 停止)
Thread 1 (Block 1): i = 5, 13        → 处理 2 个元素
Thread 2 (Block 1): i = 6, 14        → 处理 2 个元素
Thread 3 (Block 1): i = 7, 15        → 处理 2 个元素

总计: 4*3 + 4*2 = 20 个元素 ✓
```


### Version 7: Cooperative Groups + atomicAdd

**最简洁的实现（CUDA 9.0+）**

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
    
    // 每个 warp 的 lane 0 原子加到结果
    if (warp.thread_rank() == 0) {
        atomicAdd(g_odata, sum);
    }
}
```

**关于 atomicAdd 的性能：**

在旧架构上，全局内存的 atomicAdd 很慢（串行化）。但现代 GPU 上：
- 硬件优化显著改善了原子操作性能
- 对于只有少量原子操作的场景（每个 warp 一次），开销可接受
- 代码极其简洁，易于维护

```
// ═══════════════════════════════════════════════════════════════════════
//                           传统方式
// ═══════════════════════════════════════════════════════════════════════

// Warp 内位置计算
int lane = threadIdx.x % 32;           // 手动计算
int wid = threadIdx.x / 32;            // 手动计算

// Warp shuffle
val += __shfl_down_sync(0xffffffff, val, offset);  // 手动指定 mask

// Block 同步
__syncthreads();                        // 全局函数


// ═══════════════════════════════════════════════════════════════════════
//                      Cooperative Groups 方式
// ═══════════════════════════════════════════════════════════════════════

// 获取线程组
cg::thread_block block = cg::this_thread_block();
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

// Warp 内位置
int lane = warp.thread_rank();          // 更清晰！
int wid = warp.meta_group_rank();       // 更清晰！

// Warp shuffle
val += warp.shfl_down(val, offset);     // 无需手动指定 mask！

// Block 同步
block.sync();                           // 面向对象风格
```

---

## 第二部分：直方图算子（Histogram Kernel 与原子竞争层次化私有化）

## Part 1: Histogram Kernel

https://leetgpu.com/challenges/histogramming

### 1.1 从Reduce到Histogram

上一讲我们学习了Reduce：将N个元素归约为1个值。Histogram可以看作是**多目标Reduce**：

```
Reduce:     N elements → 1 value      (所有元素归约到同一目标)
Histogram:  N elements → K bins       (元素按条件归约到K个不同目标)
```

| 对比维度 | Reduce | Histogram |
|---------|--------|-----------|
| 输出大小 | 1 | K (bins数量) |
| 写入目标 | 固定 | 数据依赖 |
| 并行难点 | 归约树设计 | **原子操作竞争** |
| Roofline | Memory-bound | Memory-bound + Atomic-bound |

核心区别：Reduce的写入目标是确定的，而Histogram的写入目标取决于输入数据值，这导致**多个线程可能同时更新同一个bin**。

### 1.2 Histogram的核心挑战：原子竞争

当数据分布集中（如大部分元素落入少数几个bin）时，原子操作严重串行化：

```
线程0 → bin[3] ─┐
线程1 → bin[3] ─┼─→ 串行执行！
线程2 → bin[3] ─┤
线程3 → bin[5] ─┘
```

### 1.3 解决方案：层次化私有化

核心思想：**减少竞争范围**，从global竞争→block内竞争→warp内竞争

```
┌─────────────────────────────────────────────────────────┐
│  Version 1: Global Atomic                               │
│  所有线程 → Global Memory (竞争最严重)                    │
├─────────────────────────────────────────────────────────┤
│  Version 2: Shared Memory Privatization                 │
│  Block内线程 → Shared Memory → Global Memory            │
│  竞争范围从全GPU缩小到单个Block (256线程)                  │
├─────────────────────────────────────────────────────────┤
│  Version 3: Warp-level + Local Accumulation             │
│  进一步减少atomic次数                                    │
└─────────────────────────────────────────────────────────┘
```

### 1.4 实现版本

#### Version 1: Naive (Baseline)

```cpp
// __global__ 修饰符：声明这是一个 GPU kernel 函数
//   - 由 CPU (host) 调用，在 GPU (device) 上执行
//   - 返回类型必须是 void
__global__ void histogram_v1_naive(
    // __restrict__ 关键字：告诉编译器这个指针是访问该内存的唯一方式
    //   - 保证 data 和 hist 指向的内存区域不重叠（no pointer aliasing）
    //   - 允许编译器进行更激进的优化（如循环展开、指令重排）
    //   - 类似于 C99 的 restrict，但在 CUDA 中使用双下划线
    const int* __restrict__ data,  // 输入数据数组（只读）
    int* __restrict__ hist,         // 输出直方图数组（读写）
    int n,                          // 输入数据的元素个数
    int num_bins                    // 直方图的 bin 数量
) {
    // 计算当前线程的全局索引
    // blockIdx.x: 当前 block 在 grid 中的索引
    // blockDim.x: 每个 block 中的线程数
    // threadIdx.x: 当前线程在 block 中的索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop 模式的步长
    // gridDim.x: grid 中 block 的总数
    // stride = 所有线程的总数，用于处理数据量大于线程数的情况
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop：每个线程处理多个元素
    // 这种模式的优点：
    //   1. 可以处理任意大小的输入数据
    //   2. 线程数可以独立于数据大小进行调优
    //   3. 保持良好的内存访问模式（相邻线程访问相邻内存）
    for (int i = idx; i < n; i += stride) {
        int bin = data[i];
        // 边界检查：确保 bin 值在有效范围内
        if (bin >= 0 && bin < num_bins) {
            // atomicAdd：原子加操作
            //   - 保证多个线程同时更新同一位置时的正确性
            //   - 缺点：当多个线程竞争同一个 bin 时会产生串行化
            //   - 这是 naive 版本的主要性能瓶颈
            atomicAdd(&hist[bin], 1);
        }
    }
}
```

**内存流动与延迟分析**

让我们逐步分析这个 kernel 中数据的内存流动过程：

1. **索引计算阶段**：`idx` 和 `stride` 这两个变量的计算完全在寄存器中完成。`blockIdx.x`、`blockDim.x`、`threadIdx.x` 和 `gridDim.x` 都是 CUDA 提供的内置变量，存储在特殊寄存器中，访问延迟极低（约 1 个时钟周期）。

2. **数据读取阶段**：`int bin = data[i]` 是整个 kernel 的第一个内存瓶颈。`data` 数组位于全局内存（Global Memory）中，访问延迟高达 400-800 个时钟周期。不过，由于我们使用了 grid-stride loop，相邻线程访问相邻内存地址，这形成了合并访问（Coalesced Access）模式。当一个 warp（32 个线程）同时访问连续的 32 个 int 时，这些请求会被合并成一次 128 字节的内存事务，大大提高了带宽利用率。读取的 `bin` 值会被存储在每个线程的寄存器中。

3. **边界检查阶段**：`if (bin >= 0 && bin < num_bins)` 的比较操作在寄存器中完成，延迟可忽略。这里可能有分支发散（Branch Divergence），但目前先忽略

4. **原子更新阶段**：`atomicAdd(&hist[bin], 1)` 是性能的核心瓶颈。这个操作涉及：
   - 首先，根据 `bin` 的值计算 `hist` 数组中对应元素的地址（寄存器操作）
   - 然后，发起一次全局内存的原子读-改-写操作
   - 原子操作本身的延迟与普通全局内存访问相当（400-800 周期），但问题在于串行化：当多个线程同时更新同一个 bin 时，这些操作必须排队执行
   - 如果数据分布不均匀（某些 bin 特别热门），竞争会更加严重，延迟可能累积到数千个周期

**延迟隐藏与性能特点**

GPU 通过大量并行线程来隐藏内存延迟。当一个 warp 等待内存访问完成时，调度器会切换到其他就绪的 warp 执行。然而，这个 naive 版本的问题在于：

- 全局内存的原子操作无法被有效隐藏，因为同一 bin 的更新必须串行
- 所有线程都在竞争同一个 `hist` 数组，形成严重的内存争用
- 当 bin 数量较少或数据分布集中时，性能会急剧下降

这就是为什么后续版本使用 Shared Memory 进行私有化：Shared Memory 的访问延迟只有约 20-30 个时钟周期，比全局内存快一个数量级，可以大幅减少原子操作的开销。

#### Version 2: Shared Memory Privatization ⭐

这是最实用的优化版本：

```cpp
__global__ void histogram_v2_shared(
    const int* __restrict__ data,
    int* __restrict__ hist,
    int n, int num_bins
) {
    extern __shared__ int s_hist[];  // 每个block的私有histogram
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Step 1: 初始化shared memory
    for (int i = tid; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Step 2: 在shared memory中累积 (block内竞争，比global快~10x)
    for (int i = idx; i < n; i += stride) {
        int bin = data[i];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();
    
    // Step 3: 归约到global memory (每个bin只需1次global atomic)
    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}
```

**为什么有效**：
- Shared memory atomic比global快约10倍
- 竞争从百万线程降到256线程/block
- 最终global atomic次数 = num_bins × num_blocks（而非n次）

**限制**：bins数量受shared memory限制（48KB → ~12K int bins）

#### Version 3: Local Accumulation (处理数据局部性)

当数据有局部性（连续元素倾向于落入相同bin）时：

```cpp
// 核心思想：利用数据的时间局部性，用寄存器累积连续相同的 bin
// 适用场景：当输入数据具有局部性（如图像像素、排序后的数据），连续元素往往落入相同 bin
// 优化原理：将多次原子操作合并为一次，减少原子操作的总次数
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

    // ========== 核心优化：本地累积器 ==========
    // last_bin: 记录上一次处理的 bin 索引（存储在寄存器中）
    // count: 累积计数器，记录连续相同 bin 的出现次数（存储在寄存器中）
    // 关键洞察：寄存器访问是免费的（1 周期），而 atomicAdd 代价高昂
    int last_bin = -1;
    int count = 0;

    for (int i = idx; i < n; i += stride) {
        int bin = data[i];
        if (bin == last_bin) {
            // 连续遇到相同 bin：只增加寄存器中的计数器
            // 这是纯寄存器操作，零内存访问开销
            count++;
        } else {
            // 遇到不同 bin：需要"刷新"之前的累积值到 shared memory
            // 只有在 bin 切换时才执行一次 atomicAdd，而不是每个元素都执行
            if (count > 0) atomicAdd(&s_hist[last_bin], count);
            last_bin = bin;  // 更新追踪的 bin
            count = 1;       // 重置计数器
        }
    }
    // 循环结束后，最后一批累积的计数还在寄存器中，需要最终刷新
    if (count > 0) atomicAdd(&s_hist[last_bin], count);

    __syncthreads();

    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}
// 性能分析：
// - 最好情况：数据完全有序，每个线程只需 1 次 atomicAdd（所有元素都在同一 bin）
// - 最坏情况：数据完全随机，退化为 Version 2 的性能（每个元素都触发 atomicAdd）
// - 额外开销：每次循环多了一次比较和条件分支，但这比 atomicAdd 便宜得多
```

### 1.5 性能特征与选择指南

| 场景 | 推荐版本 | 原因 |
|------|---------|------|
| bins < 12K | V2 Shared | 通用最优 |
| 数据有局部性 | V3 Local | 减少atomic次数 |
| bins > 12K | Multi-pass或CUB | 超出shared memory |
| 生产环境 | CUB库 | 高度优化 |

**Roofline 理论分析**

让我们用 Roofline 模型分析 Histogram kernel 的性能特征。

**符号定义**：
- $\pi$：GPU 峰值算力（FLOP/s）
- $\beta$：显存带宽（Byte/s）
- $N$：输入数据元素个数
- $B$：bin 的数量
- $s$：单个数据元素的字节数（如 int32 则 $s=4$）


**Histogram 的算术强度分析**：

对于 naive 版本，处理 $N$ 个元素：

- **内存访问量**：
  - 读取输入数据：$N \cdot s$ 字节
  - 原子更新直方图（读-改-写）：$N \cdot 2s$ 字节（最坏情况，每次都访问不同 bin）
  - 总计：$M = 3Ns$ 字节

- **计算量**：
  - 每个元素：边界检查 + 加法 ≈ $\alpha$ FLOPs（$\alpha \approx 2$）
  - 总计：$F = \alpha N$ FLOPs

- **算术强度**：
$$I_{hist} = \frac{F}{M} = \frac{\alpha N}{3Ns} = \frac{\alpha}{3s}$$

对于 int32（$s=4$），$I_{hist} = \frac{\alpha}{12} \ll 1$ FLOP/Byte

**Roofline 结论**：

由于现代 GPU 的 $I_{ridge} \gg 1$（通常 $I_{ridge} > 100$），而 $I_{hist} < 1$，因此：
$$I_{hist} \ll I_{ridge}$$

Histogram 处于严重的 **Memory-Bound** 区域，可达性能为：
$$P_{attainable} = I_{hist} \cdot \beta = \frac{\alpha \beta}{3s}$$

峰值算力利用率：
$$\eta = \frac{P_{attainable}}{\pi} = \frac{\alpha \beta}{3s\pi} = \frac{\alpha}{3s \cdot I_{ridge}} \ll 1$$

**考虑原子操作竞争**：

设 $\gamma \in (0, 1]$ 为原子操作的有效带宽系数（竞争越激烈，$\gamma$ 越小），实际性能为：
$$P_{real} = \gamma \cdot I_{hist} \cdot \beta = \frac{\gamma \alpha \beta}{3s}$$

**优化版本的 Roofline 视角**：

| 版本 | 优化效果 | Roofline 影响 |
|------|---------|--------------|
| V2 Shared | 用 Shared Memory 替代 Global Memory | 等效将 $\beta$ 提升 $\kappa$ 倍（$\kappa \approx 10\text{-}20$） |
| V3 Local | 寄存器累积，减少原子操作次数 | 将 $\gamma$ 提升至接近 1 |

**结论**：Histogram 的 $I \ll I_{ridge}$，始终是 memory-bound。优化策略：
1. **提升有效带宽**：使用更快的存储层级（Shared Memory），等效增大 $\beta$
2. **减少竞争**：私有化直方图，提升 $\gamma$
3. **减少访存次数**：本地累积，减小 $M$

---

---

## 第三部分：前缀和算子（Parallel Scan 与工作高效 Blelloch 算法）

## Part 2: Scan (Prefix Sum) Kernel

https://leetgpu.com/challenges/prefix-sum

### 2.1 什么是Scan

Scan（前缀和）是另一个核心并行原语，计算数组的累积操作：

```
输入:  [3, 1, 7, 0, 4, 1, 6, 3]

Exclusive Scan (不含当前元素):
输出:  [0, 3, 4, 11, 11, 15, 16, 22]
       ↑  ↑
       0  0+3

Inclusive Scan (含当前元素):  
输出:  [3, 4, 11, 11, 15, 16, 22, 25]
       ↑  ↑
       3  3+1
```

### 2.2 Scan的重要性

Scan是构建其他并行算法的基础：

| 应用 | 如何使用Scan |
|------|-------------|
| Stream Compaction | 标记→Scan→Scatter |
| Radix Sort | 计数→Scan→分配位置 |
| 稀疏矩阵 | CSR格式的row_ptr |
| 并行分配 | 计算每个线程的输出偏移 |

### 2.3 朴素并行Scan的问题

直观想法：每个元素独立计算前缀和

```cpp
// 错误！O(n²)复杂度
__global__ void scan_naive(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i = 0; i <= idx; i++) {  // 每个线程遍历[0, idx]
        sum += data[i];
    }
    data[idx] = sum;
}
```

问题：第i个线程做i次加法，总工作量O(n²)，完全没有利用并行性。

### 2.4 高效并行Scan：Blelloch算法

Blelloch算法分两个阶段，总工作量O(n)，跨度O(log n)：

```
Phase 1: Up-sweep (Reduce)
建立归约树，计算部分和

Phase 2: Down-sweep
从根向下传播，计算前缀和
```

#### 伪代码

```python
def blelloch_scan(x):
    """
    Blelloch 并行前缀和算法
    输入: x[0..n-1]，长度 n 必须是 2 的幂
    输出: exclusive prefix sum
    """
    n = len(x)

    # ========== Phase 1: Up-sweep (Reduce) ==========
    # 从叶子到根，构建归约树
    # 每一步将相邻元素对的和存储到右边元素的位置
    for d in range(log2(n)):           # d = 0, 1, ..., log2(n)-1
        stride = 2^(d+1)               # stride = 2, 4, 8, ...
        for i in parallel(0, n, stride):  # i = 0, stride, 2*stride, ...
            x[i + stride - 1] += x[i + stride/2 - 1]

    # 此时 x[n-1] 包含所有元素的总和

    # ========== Phase 2: Down-sweep ==========
    # 从根到叶子，利用归约树计算前缀和
    x[n-1] = 0                         # 将根设为 0（identity element）

    for d in range(log2(n)-1, -1, -1): # d = log2(n)-1, ..., 1, 0
        stride = 2^(d+1)               # stride = n, n/2, ..., 4, 2
        for i in parallel(0, n, stride):
            left = i + stride/2 - 1
            right = i + stride - 1

            temp = x[left]             # 保存左子节点的值
            x[left] = x[\right]         # 左子节点 = 父节点的值（来自上方）
            x[\right] += temp           # 右子节点 = 父节点值 + 原左子节点值

    return x  # 现在 x 包含 exclusive prefix sum
```

**复杂度分析**：
- 时间复杂度：$O(\log N)$ 步（每步内部并行）
- 工作复杂度：$O(N)$ 次加法操作
- 空间复杂度：$O(1)$ 额外空间（in-place 算法）

#### 图解（8个元素）

```
输入: [3, 1, 7, 0, 4, 1, 6, 3]

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
设置根为0: [3, 4, 7, 11, 4, 5, 6, 0]
                                  ↑

Step 1 (stride=4):
[3, 4, 7, 0, 4, 5, 6, 11]
           ↑            ↑
        交换并累加

Step 2 (stride=2):
[3, 4, 7, 0, 4, 5, 6, 11]
     ↓     ↓     ↓      ↓
[3, 0, 7, 4, 4, 11, 6, 16]

Step 3 (stride=1):
[0, 3, 4, 11, 11, 15, 16, 22]

输出 (Exclusive): [0, 3, 4, 11, 11, 15, 16, 22] ✓
```


```
new_left  = right           // 左子继承父节点传来的值
new_right = left + right    // 右子 = 继承值 + 左兄弟的子树和
树形视角

Up-sweep 结果（子树和）：        Down-sweep 传递（左边的和）：
        25                              0
       /  \                           /    \
     11    14                        0       11
    / \   / \                       / \     /   \
   4   7  5  9                     0   4   11    16
  /\  /\ /\  /\                   /\  /\   /\    /\
 3 1 7 0 4 1 6 3                 0 3 4 11 11 15 16 22


**规则**：
- **左子节点**：继承父节点的值（我左边 = 父亲左边）
- **右子节点**：父节点值 + 左兄弟的和（我左边 = 父亲左边 + 左兄弟）


temp[n-1] = 0;  // 在down-sweep开始前
因为是 **exclusive** scan——第一个元素的前缀和是 0（它左边没有任何元素）。这个 0 会在 down-sweep 过程中传播到位置 0。


输入:     [3, 1, 7, 0, 4, 1, 6, 3]

Up-sweep后: [3, 4, 7, 11, 4, 5, 6, 25]
                                   ↓ 设为0
           [3, 4, 7, 11, 4, 5, 6, 0]

Down-sweep:
  stride=4: [3, 4, 7, 0,  4, 5,  6, 11]   // 根层
  stride=2: [3, 0, 7, 4,  4, 11, 6, 16]   // 第二层  
  stride=1: [0, 3, 4, 11, 11, 15, 16, 22] // 叶子层 ✓
```

### 2.5 CUDA实现

#### Version 1: 单Block Scan (Blelloch)

```cpp
__global__ void scan_blelloch_single_block(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    
    // 加载到shared memory
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
    
    // 清除最后一个元素（为down-sweep准备）
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
    
    // 写回
    data[2*tid] = temp[2*tid];
    data[2*tid+1] = temp[2*tid+1];
}
```

**限制**：只能处理单个block大小的数据（通常≤2048元素）

#### Version 2: 多Block Scan (三阶段)

处理任意大小数组需要三个阶段：

```
阶段1: Block-level Scan
每个block独立scan自己的部分，保存block总和

阶段2: Scan Block Sums  
对所有block的总和做scan

阶段3: Add Block Offsets
每个block加上前面所有block的总和
```

```cpp
// 阶段1: 每个block scan并保存总和
__global__ void scan_blocks(int* data, int* block_sums, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * blockDim.x * 2;
    
    // 加载数据
    int ai = tid;
    int bi = tid + blockDim.x;
    temp[ai] = (block_offset + ai < n) ? data[block_offset + ai] : 0;
    temp[bi] = (block_offset + bi < n) ? data[block_offset + bi] : 0;
    
    // Blelloch scan (同上)
    // ... up-sweep ...
    // ... down-sweep ...
    
    __syncthreads();
    
    // 保存block总和
    if (tid == 0) {
        block_sums[bid] = temp[blockDim.x * 2 - 1];
    }
    
    // 写回scan结果
    if (block_offset + ai < n) data[block_offset + ai] = temp[ai];
    if (block_offset + bi < n) data[block_offset + bi] = temp[bi];
}

// 阶段3: 加上block偏移
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

#### Version 3: Work-Efficient with Bank Conflict Avoidance

Shared memory有bank conflict问题。添加padding避免：

```cpp
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

__global__ void scan_optimized(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    
    // 带padding的索引，避免bank conflict
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
    
    // ... 类似处理down-sweep ...
}
```


### 2.6 Scan的Roofline特征

```
内存访问: 读n + 写n = 2n
计算量: O(n) 加法
Arithmetic Intensity: ~0.25 ops/byte (int32)

结论: Memory-bound，但比histogram好（没有原子操作）
```

---

## 第四部分：状态空间模型（Mamba-1 / 2）中的并行 Scan 实战

## Part 3: Scan在Mamba 1.0中的应用

### 3.1 背景：State Space Model (SSM)

SSM是一种序列建模方法，可以看作连续时间系统的离散化：

```
连续形式:
  h'(t) = A·h(t) + B·x(t)     (状态更新)
  y(t)  = C·h(t) + D·x(t)     (输出)

离散化后:
  h_t = Ā·h_{t-1} + B̄·x_t    (线性递推！)
  y_t = C·h_t + D·x_t
```

**这是一个线性递推，正是Scan可以并行化的形式！**

### 3.2 SSM的两种计算模式

**模式1: Recurrent（顺序计算）**

```python
# O(L) 时间，O(1) 空间，但完全顺序
h = zeros(N)  # hidden state
for t in range(L):
    h = A @ h + B @ x[t]   # 必须等上一步完成
    y[t] = C @ h
```

- 推理时很高效：O(1) per token
- 训练时很慢：无法并行，GPU利用率低

**模式2: Convolution（并行计算）**

对于**时不变**SSM（A, B, C固定），可以展开成卷积：

```
y = x * K，其中 K = (CB̄, CĀB̄, CĀ²B̄, ...)
```

- 训练时高效：FFT卷积 O(L log L)
- 但要求A, B, C是常数（时不变）

### 3.3 Mamba的困境：Selective但不能用卷积

Mamba的核心创新是**Selective SSM**：让B, C, Δ依赖于输入

```python
# Selective SSM: 参数随输入变化
Δ_t = Linear(x_t)  # 离散化步长
B_t = Linear(x_t)  # 输入矩阵
C_t = Linear(x_t)  # 输出矩阵

h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B_t·x_t
y_t = C_t·h_t
```

**问题**：参数随时间变化 → 不再是时不变系统 → 卷积模式失效！

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mamba 的困境                                                               │
│                                                                             │
│  想要 Selective（性能好）→ B, C, Δ 必须依赖输入 → 时变系统                    │
│  想要 Fast Training     → 需要并行化 → 卷积要求时不变                        │
│                                                                             │
│  矛盾！传统方法只能二选一                                                    │
│                                                                             │
│  解决方案: Parallel Associative Scan                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Associative Scan：并行化线性递推

**关键洞察**：虽然参数时变，但递推仍然是**结合律**的！

SSM递推可以写成：
```
h_t = A_t · h_{t-1} + b_t

定义二元组: (A_t, b_t)
定义结合运算 ⊗: (A₂, b₂) ⊗ (A₁, b₁) = (A₂·A₁, A₂·b₁ + b₂)
```

**验证结合律**：

```
(A₃, b₃) ⊗ [(A₂, b₂) ⊗ (A₁, b₁)]
= (A₃, b₃) ⊗ (A₂·A₁, A₂·b₁ + b₂)
= (A₃·A₂·A₁, A₃·A₂·b₁ + A₃·b₂ + b₃)

[(A₃, b₃) ⊗ (A₂, b₂)] ⊗ (A₁, b₁)
= (A₃·A₂, A₃·b₂ + b₃) ⊗ (A₁, b₁)
= (A₃·A₂·A₁, A₃·A₂·b₁ + A₃·b₂ + b₃)

两者相等！满足结合律 ✓
```

### 3.5 Parallel Scan应用于SSM

有了结合律，就可以用Blelloch scan并行计算：

```
输入: [(A₁,b₁), (A₂,b₂), (A₃,b₃), (A₄,b₄), ...]

目标: 计算所有前缀积
  h₁ = (A₁,b₁)
  h₂ = (A₂,b₂) ⊗ (A₁,b₁)
  h₃ = (A₃,b₃) ⊗ (A₂,b₂) ⊗ (A₁,b₁)
  ...

使用 Blelloch Scan:
  Step 1 (Up-sweep): 构建部分积
  Step 2 (Down-sweep): 传播前缀积
  
  复杂度: O(L) work, O(log L) span
  可以在 O(log L) 步内完成！
```

**图示**：

```
Sequential (O(L) steps):
  h₁ → h₂ → h₃ → h₄ → h₅ → h₆ → h₇ → h₈
  
Parallel Scan (O(log L) steps):
  Step 1:  [1-2]   [3-4]   [5-6]   [7-8]     (4 pairs)
  Step 2:  [1-4]           [5-8]             (2 pairs)  
  Step 3:  [1-8]                             (1 pair)
  Down-sweep: 分发前缀积到每个位置
  
  总共 2·log₂(8) = 6 步，而非 8 步
```


```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mamba 的三大优化策略                                                        │
│                                                                             │
│  1. Kernel Fusion（算子融合）                                                │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ 传统: HBM → 离散化 → HBM → Scan → HBM → 输出 → HBM               │    │
│     │       (多次HBM读写，I/O瓶颈)                                      │    │
│     │                                                                 │    │
│     │ Fused: HBM → SRAM [离散化 + Scan + 输出] → HBM                   │    │
│     │        (一次读入，一次写出)                                       │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. Parallel Scan in SRAM                                                   │
│     • 不materialize中间状态到HBM                                            │
│     • 所有scan操作在SRAM中完成                                               │
│     • 只写最终输出到HBM                                                      │
│                                                                             │
│  3. Recomputation（重计算）                                                  │
│     • Forward: 不保存中间状态                                                │
│     • Backward: 重新计算需要的状态                                           │
│     • 用计算换内存                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.6 Mamba1 Kernel 核心代码解析

> Example source: https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan/

#### 核心文件结构

```
csrc/selective_scan/
├── selective_scan_common.h      # Associative scan 算子定义
├── selective_scan_fwd_kernel.cuh  # Forward pass kernel
├── selective_scan_bwd_kernel.cuh  # Backward pass kernel
└── reverse_scan.cuh             # 反向scan（用于梯度传播）
```

#### 1. Associative Scan 算子（核心！）

SSM递推公式: `h[t] = A * h[t-1] + B * x[t]`

**关键洞察**：这个递推可以表示为二元组 `(a, b)` 的结合运算：
- 状态表示为 `(decay, value)` = `(A, B*x)`
- 组合两个状态: `(a0, b0) ⊕ (a1, b1) = (a1*a0, a1*b0 + b1)`

```cuda
// selective_scan_common.h - 核心算子
// 这是Mamba能够并行化的数学基础！

template<>
struct SSMScanOp<float> {
    __device__ __forceinline__ float2 operator()(
        const float2 &ab0,  // (a0, b0) = 前一个状态
        const float2 &ab1   // (a1, b1) = 当前状态
    ) const {
        // 结合律: (a1*a0, a1*b0 + b1)
        // ab.x = decay factor (累积衰减)
        // ab.y = value contribution (累积输入)
        return make_float2(
            ab1.x * ab0.x,           // 累积decay: a1 * a0
            ab1.x * ab0.y + ab1.y    // 累积value: a1 * b0 + b1
        );
    }
};

// 复数版本（用于某些SSM变体）
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

// 跨chunk边界的状态传递
template <typename scalar_t>
struct SSMScanPrefixCallbackOp {
    using scan_t = std::conditional_t<
        std::is_same_v<scalar_t, float>, float2, float4>;
    scan_t running_prefix;  // 上一个chunk的最终状态

    __device__ scan_t operator()(scan_t block_aggregate) {
        scan_t old_prefix = running_prefix;
        // 将当前block的聚合结果与running prefix组合
        running_prefix = SSMScanOp<scalar_t>()(running_prefix, block_aggregate);
        return old_prefix;  // 返回给当前block使用
    }
};
```

#### 2. Forward Kernel 核心逻辑

```cuda
// selective_scan_fwd_kernel.cuh (简化版)

template<typename Ktraits>
__global__ void selective_scan_fwd_kernel(SSMParamsBase params) {
    // ========== 常量和配置 ==========
    constexpr int kNThreads = Ktraits::kNThreads;     // 线程数
    constexpr int kNItems = Ktraits::kNItems;         // 每线程处理的元素数
    constexpr int kChunkSize = kNThreads * kNItems;   // 2048 (典型值)

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;   // 每个block处理一个(batch, dim)

    // ========== Step 1: 加载参数 ==========
    // A: [D, N] - 状态转移矩阵（通常是负数，表示衰减）
    // delta: [B, L, D] - 时间步长（input-dependent）
    // B: [B, L, N] 或 [B, N] - 输入矩阵
    // C: [B, L, N] 或 [B, N] - 输出矩阵
    // u: [B, L, D] - 输入

    float A_val = A[dim_id * N + state_idx];  // 对每个state维度
    A_val *= LOG2E;  // 预乘log2(e)，使用exp2f更快

    // ========== Step 2: 按chunk处理序列 ==========
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        int chunk_offset = chunk * kChunkSize;

        // 2a. 加载这个chunk的数据到寄存器
        float delta_vals[kNItems], u_vals[kNItems];
        float B_vals[kNItems], C_vals[kNItems];

        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            int seq_idx = chunk_offset + threadIdx.x * kNItems + i;
            delta_vals[i] = delta[batch_id][seq_idx][dim_id];
            u_vals[i] = u[batch_id][seq_idx][dim_id];

            // Delta softplus (可选): delta = log(1 + exp(delta))
            if (kDeltaSoftplus) {
                delta_vals[i] = delta_vals[i] <= 20.f
                    ? log1pf(expf(delta_vals[i]))
                    : delta_vals[i];
            }
        }

        // 2b. 计算scan的输入: (decay, value) pairs
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
        // 使用CUB的BlockScan，配合自定义的SSMScanOp

        using BlockScanT = cub::BlockScan<float2, kNThreads,
                                          cub::BLOCK_SCAN_WARP_SCANS>;

        SSMScanPrefixCallbackOp<float> prefix_op(running_prefix);

        BlockScanT(smem_scan).InclusiveScan(
            thread_data,        // 输入: (decay, value) pairs
            thread_data,        // 输出: scan后的结果
            SSMScanOp<float>(), // 结合运算符
            prefix_op           // 处理跨chunk的状态传递
        );

        // 更新running_prefix用于下一个chunk
        running_prefix = prefix_op.running_prefix;

        // ========== Step 4: 计算输出 ==========
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            // thread_data[i].y 现在是 h[t]（隐藏状态）
            // output = C * h
            out_vals[i] += thread_data[i].y * C_vals[i];
        }

        // 写回HBM（只在所有state维度累加完后写）
    }
}
```

#### 3. 数学原理图解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Associative Scan 的数学基础                                                 │
│                                                                             │
│  递推公式: h[t] = a[t] * h[t-1] + b[t]                                      │
│                                                                             │
│  表示为二元组: (a, b) 其中 a=decay, b=input                                  │
│                                                                             │
│  组合运算 ⊕:                                                                │
│    (a0, b0) ⊕ (a1, b1) = (a1*a0, a1*b0 + b1)                               │
│                                                                             │
│  验证结合律:                                                                │
│    [(a0,b0) ⊕ (a1,b1)] ⊕ (a2,b2)                                           │
│    = (a1*a0, a1*b0+b1) ⊕ (a2,b2)                                           │
│    = (a2*a1*a0, a2*(a1*b0+b1)+b2)                                          │
│    = (a2*a1*a0, a2*a1*b0 + a2*b1 + b2)                                     │
│                                                                             │
│    (a0,b0) ⊕ [(a1,b1) ⊕ (a2,b2)]                                           │
│    = (a0,b0) ⊕ (a2*a1, a2*b1+b2)                                           │
│    = (a2*a1*a0, a2*a1*b0 + a2*b1 + b2)   ✓ 相等！                           │
│                                                                             │
│  因此可以用parallel prefix sum!                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Parallel Scan 执行过程 (8个元素示例)                                        │
│                                                                             │
│  输入: (a0,b0) (a1,b1) (a2,b2) (a3,b3) (a4,b4) (a5,b5) (a6,b6) (a7,b7)     │
│                                                                             │
│  Up-sweep (reduce):                                                         │
│  Level 0:  [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]            │
│              \    /        \    /        \    /        \    /              │
│  Level 1:   [0:1]         [2:3]         [4:5]         [6:7]               │
│                 \          /                \          /                   │
│  Level 2:       [0:3]                       [4:7]                          │
│                      \                    /                                │
│  Level 3:            [0:7] (全局聚合)                                       │
│                                                                             │
│  Down-sweep (distribute):                                                   │
│  把部分和传播回去，得到每个位置的inclusive scan结果                            │
│                                                                             │
│  输出: h[0]  h[1]  h[2]  h[3]  h[4]  h[5]  h[6]  h[7]                      │
│                                                                             │
│  复杂度: O(log L) 深度，O(L) 总工作量                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```


### 3.7 Mamba-2：从Scan到矩阵乘法

Mamba-1的parallel scan有个问题：**无法利用Tensor Core**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mamba-1 vs Mamba-2                                                         │
│                                                                             │
│  Mamba-1:                                                                   │
│  • 使用 parallel associative scan                                           │
│  • Scan操作是element-wise，无法用Tensor Core                                 │
│  • State dimension 限制为 N=16（更大会变慢）                                 │
│  • A100: 只用到 19 TFLOPS (FP32 arithmetic)                                 │
│                                                                             │
│  Mamba-2:                                                                   │
│  • 发现SSM可以写成structured matrix乘法                                      │
│  • 用矩阵乘法替代scan（可以用Tensor Core！）                                  │
│  • State dimension 可以扩展到 N=64, 128                                     │
│  • A100: 可用 312 TFLOPS (BF16 matmul) - 16x 提升！                         │
│                                                                             │
│                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

---

## 第五部分：Memory-Bound 典型模式库手写实战

## 3. Pattern Library：六类 Memory-Bound Kernel

以下将 memory-bound kernel 按访存模式归纳为六类。面对新的 kernel 时，首先判断其所属类别，然后按对应的原则组合进行优化。

---

### Pattern 1：Streaming（线性读写）

**典型场景**：`out[i] = f(in[i])`，elementwise map、向量加法

**访存特征**：顺序读取输入，顺序写入输出，无数据复用。

**适用原则**：B（coalescing）+ E（延迟隐藏）

**优化手段**：vectorized load/store（float4）、内存对齐、grid-stride loop、循环展开

```cpp
// Vectorized elementwise kernel
// 使用 float4 进行向量化加载和存储，每次内存事务搬运 16 字节
__global__ void vector_add_v4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n4  // n / 4，即 float4 元素个数
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

//// float4 的定义（大致）
// struct float4 {
//    float x, y, z, w;
//};

// 启动配置
// n 为 float 元素总数，需为 4 的倍数（否则需额外处理尾部）
// vector_add_v4<<<(n/4 + 255) / 256, 256>>>(a4, b4, c4, n/4);
```

> [!note] float4 的优化原理
> 使用 float4 后，每条 load/store 指令搬运 16 字节而非 4 字节。这减少了所需的 load/store 指令总数，使编译器能够更有效地安排指令流水线，提高指令级并行度（ILP）。一个 warp 的 32 个线程同时执行 float4 load 时，产生 4 个 128B 内存事务，总计搬运 512 字节。

---

### Pattern 2：Reorder / Permutation（重排类）

**典型场景**：Matrix Transpose

**核心矛盾**：读取方向与写入方向正交。若读取是 coalesced 的，则写入必然是 strided 的，反之亦然。

**解决方案**：使用 shared memory 作为中间缓冲区。读取时按行方向 coalesce 加载到 shared memory，写入时从 shared memory 按列方向 coalesce 写出。

**适用原则**：B（读写两端均需 coalesce）+ C（shared memory 作为重排缓存）+ D（避免 bank conflict）

```cpp
// Shared memory tiled 矩阵转置
// 输入 [M x N]，输出 [N x M]
#define TILE_DIM 32
#define BLOCK_ROWS 8  // block 尺寸: TILE_DIM x BLOCK_ROWS
                      // 每个 block 处理 TILE_DIM x TILE_DIM 的 tile

__global__ void transpose_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    // 列数 +1 作为 padding，消除 bank conflict（详见下方说明）
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Step 1: 按行方向从 global memory 加载到 shared memory（coalesced read）
    // 每个线程负责加载 TILE_DIM / BLOCK_ROWS = 4 行
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }

    __syncthreads();

    // Step 2: 坐标互换后，按行方向从 shared memory 写入 global memory（coalesced write）
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < M && (y + j) < N) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// 启动配置
// dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
// dim3 block(TILE_DIM, BLOCK_ROWS);
// transpose_optimized<<<grid, block>>>(input, output, M, N);
```

> [!important] Padding 消除 bank conflict
> Shared memory 由 32 个 bank 组成，每个 bank 宽度为 4 字节。若 tile 的列数恰好为 32，则同一列的所有元素将映射到同一个 bank，导致列方向访问时产生 32 路 bank conflict。将列数设为 `TILE_DIM + 1 = 33` 后，相邻行中同一列位置的元素错开一个 bank，从而消除冲突。这一技巧在涉及 shared memory 列访问的 kernel 中被广泛使用。

---

### Pattern 3：Stencil / Neighborhood（邻域复用类）

**典型场景**：1D/2D stencil、图像卷积、图像处理滤波器

**核心特征**：每个输入元素被多个相邻输出点复用。以 1D 3-point stencil 为例，每个输入元素被左、中、右三个输出点各读取一次。

**适用原则**：C（tile + halo 实现确定性复用）+ B（halo 加载时保持 coalescing）

```cpp
// 1D Stencil: out[i] = c0*in[i-R] + c1*in[i-R+1] + ... + c2R*in[i+R]
// R = stencil 半径，本例取 R = 4（9-point stencil）

#define RADIUS 4
#define BLOCK_SIZE 256
// 每个 block 计算 BLOCK_SIZE 个输出点
// 需加载 BLOCK_SIZE + 2*RADIUS 个输入点（含两端 halo 区域）

__constant__ float coeff[2 * RADIUS + 1];  // stencil 系数存入 constant memory

__global__ void stencil_1d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float smem[BLOCK_SIZE + 2 * RADIUS];

    int gidx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lidx = threadIdx.x + RADIUS;  // shared memory 内的偏移索引

    // Step 1: 加载中间区域
    smem[lidx] = (gidx < n) ? input[gidx] : 0.0f;

    // Step 2: 加载左侧 halo（前 RADIUS 个线程负责）
    if (threadIdx.x < RADIUS) {
        int halo_idx = gidx - RADIUS;
        smem[threadIdx.x] = (halo_idx >= 0) ? input[halo_idx] : 0.0f;
    }

    // Step 3: 加载右侧 halo（后 RADIUS 个线程负责）
    if (threadIdx.x >= BLOCK_SIZE - RADIUS) {
        int halo_idx = gidx + RADIUS;
        smem[lidx + RADIUS] = (halo_idx < n) ? input[halo_idx] : 0.0f;
    }

    __syncthreads();

    // Step 4: 从 shared memory 计算 stencil，无 global memory 访问
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

**内存流量对比**：

| 版本 | Global Memory 读取量 | 说明 |
|------|---------------------|------|
| Naive（无 shared memory） | $N \times (2R+1)$ | 每个输出点从 HBM 读取 $2R+1$ 个输入 |
| Tiled（使用 shared memory） | $N + 2R \times \text{num\_blocks}$ | 每个输入元素基本仅从 HBM 加载一次 |

当 stencil 半径 $R$ 越大时，数据复用程度越高，tiling 的收益越显著。

---

### Pattern 4：Indirection / Gather（读侧不规则）

**典型场景**：CSR 格式 SpMV、gather、embedding lookup

**核心困难**：间接寻址 `x[col_idx[j]]` 导致访问地址取决于数据内容，难以实现理想的 coalescing。

**适用原则**：A（将 index 的字节开销纳入计算）+ E（通过 warp-level 映射隐藏延迟）

稀疏矩阵-向量乘法 (SpMV)

```
CSR (Compressed Sparse Row) 用三个数组存储稀疏矩阵:
═══════════════════════════════════════════════════════════════════════════════

1. values[nnz]:   所有非零值，按行存储
2. col_idx[nnz]:  每个非零值对应的列号
3. row_ptr[M+1]:  每行在 values/col_idx 中的起始位置

对于上面的矩阵:

values[]:   [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
             ↑       ↑   ↑   ↑   ↑           ↑   ↑
            row0    row0 row1 row1 row2      row2 row3

col_idx[]:  [ 0,  2,  4,  1,  5,  0,  1,  2,  3,  5]
             对应每个非零元素的列号

row_ptr[]:  [ 0,  3,  5,  9, 10]
              ↑   ↑   ↑   ↑   ↑
             row0 row1 row2 row3 结束
             起始 起始 起始 起始

row_ptr[i] 到 row_ptr[i+1] 之间的索引就是第 i 行的非零元素
```


warp-per-row策略
```
核心思想: 一个 Warp (32个线程) 协作处理矩阵的一行
═══════════════════════════════════════════════════════════════════════════════

为什么不用 Thread-per-Row？
─────────────────────────────
如果一行有很多非零元素（比如 1000 个），单线程串行处理太慢

Warp-per-Row 的优势:
─────────────────────────────
1. 32 个线程并行处理一行的非零元素
2. 访问 values[] 和 col_idx[] 时地址连续 → Coalesced Access
3. 使用 Warp Shuffle 归约，不需要 Shared Memory
```

```cpp
// CSR 格式 SpMV: y = A * x
// CSR 存储: row_ptr[M+1], col_idx[nnz], values[nnz]
//
// 映射策略:
//   行长度较均匀: thread-per-row
//   行长度差异大: warp-per-row（本例采用此策略）

// Warp-per-row: 每个 warp（32 线程）处理矩阵的一行
// 优势:
//   1. warp 内线程访问连续的 col_idx 和 values（coalesced）
//   2. 使用 warp shuffle 归约，无需 shared memory 或 atomic
//   3. 32 个线程同时发起 load，有效隐藏延迟
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

    // warp 内 32 个线程以 stride 32 遍历该行的非零元素
    for (int j = row_start + lane; j < row_end; j += 32) {
        // col_idx[j], values[j]: 连续访问，coalesced
        // x[col_idx[j]]: 随机访问，依赖 L2 cache 和延迟隐藏
        sum += values[j] * x[col_idx[j]];
    }

    // Warp-level 归约（warp shuffle），无需 shared memory 或 barrier
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[warp_id] = sum;
    }
}

// 启动配置
// int threads_per_block = 256;  // 每个 block 包含 8 个 warp
// int num_blocks = (num_rows * 32 + threads_per_block - 1) / threads_per_block;
// spmv_csr_warp_per_row<<<num_blocks, threads_per_block>>>(...);
```

如果需要所有 lane 都拿到结果，可以用 __shfl_sync 广播，或者用 __shfl_xor_sync 做 butterfly 归约。但这里只需要写一个 y[warp_id]，所以 Lane 0 有结果就够了。

> [!note] 稀疏 kernel 的带宽上界
> 对于稀疏类 kernel，实际可达的带宽上限往往低于 HBM 的理论峰值。`x[col_idx[j]]` 的访问模式由矩阵的稀疏结构决定，无法在 kernel 层面完全控制。优化方向因此转为"降低访问的随机性"，例如对矩阵列进行 reorder 以提高向量 x 的访问局部性。

---

### Pattern 5：Scatter / Atomic（写侧不规则 + 争用）

**典型场景**：histogram、scatter-add、部分图算法

**核心困难**：写入地址取决于数据内容，热点目标（如高频 bin）导致原子操作严重串行化。

**适用原则**：D（层次化私有化）

本类 kernel 的优化方法已在上一讲 Histogram 的三个版本中详细展示。核心策略如下：

```
Level 0: Global atomic — 所有线程竞争全局内存
  ↓ 私有化
Level 1: Block 级 shared memory atomic — 竞争范围缩小至 256 线程
  ↓ 进一步私有化
Level 2: Warp 级 / 寄存器本地累积 — 竞争缩小至 32 线程或完全消除
  ↓ 最终归约
写回 global memory — atomic 调用次数从 N 降至 num_bins × num_blocks
```

---

### Pattern 6：Filter / Compaction（条件过滤类）

**典型场景**：stream compaction、去零操作、predicate filter

#### Stream Compaction 是什么？

**从数组中过滤出满足条件的元素，紧凑存储**

```
输入:  [ 3, -1, 4, 0, -2, 5, 0, 1 ]
条件:  元素 > 0
输出:  [ 3, 4, 5, 1 ]
```

#### 为什么难并行？

每个线程不知道自己的输出位置——**用 Exclusive Prefix Sum 解决**：

```
flag:   [ 1,  0,  1,  0,  0,  1,  0,  1 ]   ← 标记满足条件的
scan:   [ 0,  1,  1,  2,  2,  2,  3,  3 ]   ← exclusive scan

scan[i] = "在我之前有几个满足条件的" = 我的输出位置
```

#### 三步流程

| 步骤 | 做什么 | 代码 |
|-----|-------|------|
| **Flag** | 标记满足条件的元素 | `flag = (val > 0) ? 1 : 0` |
| **Scan** | Exclusive prefix sum (Blelloch算法) | Up-sweep + Down-sweep |
| **Scatter** | 写到正确位置 | `output[offset + scan[tid]] = val` |


```cpp
// Stream Compaction: 筛选 input 中 > 0 的元素，紧凑写入 output
// Fused 实现: 在单个 kernel 内完成 flag、block-level scan、scatter

#define BLOCK_SIZE 256

__global__ void stream_compaction(
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ output_count,  // 输出的元素总数
    int n
) {
    __shared__ int scan[BLOCK_SIZE];
    __shared__ int block_output_offset;

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // Step 1: Flag — 标记满足条件的元素
    int val = 0;
    int flag = 0;
    if (gid < n) {
        val = input[gid];
        flag = (val > 0) ? 1 : 0;
    }
    scan[tid] = flag;
    __syncthreads();

    // Step 2: Block-level exclusive scan（Blelloch 算法）

    // Up-sweep
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < BLOCK_SIZE) {
            scan[ai] += scan[ai - offset];
        }
        __syncthreads();
    }

    // 提取 block 内满足条件的元素总数，并通过一次 atomic 分配输出空间
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

    // Step 3: Scatter — 将满足条件的元素写入输出数组的正确位置
    if (gid < n && flag) {
        output[block_output_offset + scan[tid]] = val;
    }
}
```

> [!tip] 关键优化
> `atomicAdd(output_count, block_total)` 的调用次数为 num_blocks 而非 N。这是原则 D（减少争用）和原则 A（降低 atomic 的内存开销）的结合：每个 block 内部通过 scan 计算局部偏移，最后仅需一次 atomic 操作分配输出空间。

---

## 4. 优化 Checklist

对 memory-bound kernel 进行优化时，建议按以下顺序依次排查：

| 步骤 | 内容 | 观测手段 |
|------|------|----------|
| 1 | 计算算术强度与带宽上限 | Roofline 模型，手动计算 |
| 2 | 测量有效带宽 | Bytes / kernel_time，与理论峰值对比 |
| 3 | 检查合并访存 | ncu: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` |
| 4 | 检查争用与同步开销 | ncu: atomic throughput、barrier wait time |
| 5 | 调整 occupancy 与 ILP | grid-stride loop、循环展开、一线程多元素 |

此顺序的重要性在于：若 coalescing 未满足就去调整 occupancy，所获得的收益将非常有限。应首先确保访存模式正确，再进行更高层次的调优。

---

## 5. 总结

| 原则 | 优化目标 | 典型适用 kernel |
|------|----------|----------------|
| A 字节账本 | 准确量化内存流量 | 所有 kernel |
| B 合并访存 | warp 内连续地址访问，减少内存事务数 | streaming、transpose |
| C 显式复用 | 数据从 HBM 加载到 SRAM 后多次复用 | stencil、部分 sparse |
| D 减少争用 | 降低 barrier 与 atomic 的串行化开销 | histogram、scatter、compaction |
| E 延迟隐藏 | 通过并行度与 ILP 掩盖内存延迟 | sparse、gather |

前两讲中 Reduce、Histogram、Scan 的优化已经覆盖了以上所有原则（coalescing、first add during load、warp shuffle、ILP、grid-stride loop）。本讲的作用是将这些分散在具体例子中的技巧抽象为通用的分析框架。

面对新的 memory-bound kernel 时，首先判断其所属的 Pattern（1-6），然后按对应的原则组合制定优化策略。

---

---

## 第六部分：课后练习题与自测问答

### 模块 A：Reduce Kernel 演进与优化
### 练习 1：reduce kernel 为什么慢？

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">naive reduce 的主要问题是什么？</span></summary>

常见问题包括 global memory 访问不合并、每轮分支导致 warp divergence、跨 block 聚合依赖 atomic 或多 kernel launch、shared memory bank conflict，以及没有利用 warp-level primitive。优化路线通常是 block 内树形归约，再减少 divergence，最后用 warp shuffle 消除最后一个 warp 的 shared memory 同步。

</details>

### 练习 2：warp shuffle 的作用

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么最后 32 个元素适合用 shuffle reduce？</span></summary>

同一个 warp 内线程 lockstep 执行，可以用 shuffle 在寄存器之间直接交换数据，不必写 shared memory，也不需要 block-level barrier。这减少了 shared memory traffic 和同步开销。前提是参与线程 mask 正确，inactive lane 的值不能污染结果。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">reduce 为什么天然 arithmetic intensity 很低？</span></summary>

对 FP32 sum 来说，每个元素通常读 4 bytes，只贡献一次加法，AI 大约是 `1 FLOP / 4 bytes = 0.25 FLOP/Byte`。即使 kernel 写得很好，上界也主要由 HBM bandwidth 决定，而不是 FP32 或 Tensor Core peak。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">parallel reduction 的优化主线是什么？</span></summary>

先保证 global load 合并访存；再让每个 thread 在加载阶段多做一点累加，减少后续归约元素数；block 内用 shared memory 或 warp shuffle 做树形归约；最后把跨 block 汇总交给第二个 kernel 或 atomic。每一步都在减少同步、分支和内存流量。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">为什么 reduce 的最后阶段常用 warp-level primitive？</span></summary>

最后 32 个 lane 在同一个 warp 内 lockstep 执行，可以用 `shuffle` 在寄存器之间交换数据。这样不需要 shared memory round-trip，也不需要 `__syncthreads()`。但必须传正确 active mask，尤其是处理非 32 整数倍长度时。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">reduce kernel 优化后仍比 PyTorch/CUB 慢，通常差在哪里？</span></summary>

库实现会针对不同 dtype、长度、对齐、SM 架构选择不同策略，并处理向量化 load、多元素 per thread、warp/block 级归约、跨 block 汇总和 edge cases。手写教学 kernel 通常只覆盖一条路径，性能接近但不一定超过高度调参的库。

</details>

### 模块 B：Histogram 与 Scan 并行原语
### 练习 1：histogram 为什么比 reduce 难？

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">histogram 的并行瓶颈是什么？</span></summary>

reduce 的写入目标固定，histogram 的写入目标由数据值决定。多个线程可能同时更新同一个 bin，导致 atomic serialize。数据分布越偏，热点 bin 越严重。常见优化是先做 warp-local 或 block-local histogram，再合并到 global histogram。

</details>

### 练习 2：scan 的核心不变量

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">parallel scan 为什么要分 upsweep / downsweep？</span></summary>

upsweep 建立局部区间和，downsweep 把前缀偏移传播回每个位置。核心不变量是每个内部节点保存一个区间 aggregate，最后每个元素拿到它左侧所有元素的和。GPU 上通常会做 block 内 scan，再 scan block sums，最后把 block prefix 加回每个 block。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">histogram 和 reduce 都是聚合，为什么 histogram 更难？</span></summary>

reduce 的写入位置通常可控，block 内可以先局部归约再写少量结果；histogram 的 bin 由数据决定，多个 thread 可能同时写同一个 bin，需要 atomic 或 privatization。难点在冲突、skew、atomic 热点和局部 histogram 合并。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">scan 能并行的前提是什么？</span></summary>

scan 依赖一个 associative operator。普通加法满足结合律，所以可以先局部 prefix，再合并 block summary，再把前缀偏移加回去。很多 recurrence 也能改写成 associative pair operation；如果算子不满足结合律，就不能直接套 parallel prefix。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">exclusive scan 和 inclusive scan 有什么差别？</span></summary>

inclusive scan 的第 `i` 个输出包含 `x_i`；exclusive scan 的第 `i` 个输出只包含 `x_0 ... x_{i-1}`。很多 compaction 和 prefix offset 场景需要 exclusive scan，因为它直接给每个元素写入目标位置。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">什么时候用 privatized histogram？代价是什么？</span></summary>

当全局 atomic 冲突严重、bin 数不大时，可以让每个 block/warp 维护私有 histogram，再做合并。它减少全局 atomic 热点，但增加 shared memory 或临时 buffer 使用，并多出一次 merge。如果 bin 很多或分布均匀，privatization 不一定划算。

</details>

### 模块 C：Memory-Bound 通用调优经验
### 练习 1：字节账本

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">优化 memory-bound kernel 前为什么要先算 bytes？</span></summary>

memory-bound kernel 的时间下界近似是 bytes 除以 bandwidth。如果一个优化没有减少 HBM bytes，也没有提高访问合并度或 cache reuse，它很可能不会显著变快。字节账本还会暴露中间 tensor 写回、atomic read-modify-write、重复读取和 padding。

</details>

### 练习 2：什么时候不该 fusion？

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">kernel fusion 一定更快吗？</span></summary>

不一定。Fusion 能减少 HBM 往返和 launch overhead，但也可能增加 register pressure、降低 occupancy、破坏 library GEMM 的 Tensor Core 路径，或者让原本可复用的中间结果变成重复计算。判断标准是节省的 memory traffic 是否大于新增成本。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">memory-bound kernel 的“字节账本”应该怎么列？</span></summary>

按 tensor 逐项列 HBM read/write：输入读几次、中间结果是否写回、输出写几次、dtype 每元素多少 bytes、是否有额外 mask/scale/index。优化前后先比较 bytes，而不是只看 FLOPs。很多 fusion 的收益正是少写回一个中间张量。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">算子融合为什么常对 elementwise 链条有效？</span></summary>

elementwise 算子计算量小，但每个算子都会读输入、写输出。把 bias、activation、residual、dropout 这类操作融合后，中间值留在 register/SRAM，减少 HBM 往返。收益通常来自省 bandwidth，而不是省很多 arithmetic。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">什么时候 fusion 可能变慢？</span></summary>

fusion 会增加单个 kernel 的 register pressure、shared memory 使用和代码复杂度，可能降低 occupancy 或破坏 Tensor Core pipeline。把两个大 GEMM 强行融合也可能导致 tile 复用不划算。正确做法是先看 fusion 是否减少 HBM traffic，再看是否引入资源瓶颈。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">LayerNorm/RMSNorm 这类算子为什么常是 memory-bound？</span></summary>

它们每个元素只做少量加减乘除，但要读整行、做 reduction、再写回结果。计算强度不高，且 reduction 需要同步和多次访问。优化重点通常是向量化 load/store、一次 pass 尽量完成统计和归一化、减少中间写回。

</details>
