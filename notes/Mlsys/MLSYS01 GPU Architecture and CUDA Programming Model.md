# 01 · GPU 硬件体系、CUDA 编程模型与 Roofline 性能分析基石

本章作为 MLSYS 算子系统与底层优化的基石，建立起从 GPU 物理硬件、流式多处理器（SM）执行管线、CUDA 编程与执行模型，到 Roofline 性能分析框架与 Memory-Bound 调优核心原则的完整心智体系。

---

## 第一部分：GPU 硬件体系结构与核心组件

## More GPU

### Overview

![[assets/Pasted image 20251222135239.png]]
**图1：NVIDIA H100/B100 GPU 整体架构抽象图。** 展示了 GPU 的层次化内存与计算结构：多个流多处理器（SM 0, SM 1, ... SM N-1）并行排列，每个 SM 内包含 4 个 Tensor Core（负责矩阵乘法运算，贡献主要算力，类似 TPU 的 MXU）和 4 个 Warp Scheduler（SIMD 向量单元，包含 32 个 lane 即"CUDA Core"，同一 warp 内所有 lane 必须执行相同操作）。每个 SM 拥有 256KB 的 L1 Cache/SMEM（共享内存，可由程序员控制，类似 TPU VMEM 但更小）。所有 SM 共享 50MB 的 L2 Cache（硬件自动管理，提供更快的带宽）和底层的 HBM 高带宽显存（H100 为 80GB，B100 为 192GB，用于存储模型参数、激活值和优化器状态）。

![[assets/Pasted image 20251222135741.png]]
**图2：NVIDIA H100 单个 SM（流多处理器）内部详细架构图。** 每个 SM 包含 4 个处理块（Processing Block），共享 L1 指令缓存和 256KB L1 数据缓存/共享内存。每个处理块包含：L0 指令缓存、Warp Scheduler（每周期调度 32 线程）、Dispatch Unit、16384×32-bit 的寄存器文件，以及大量计算单元——16 个 INT32 单元、16 个 FP32 单元、8 个 FP64 单元、1 个第四代 Tensor Core、LD/ST（加载/存储）单元和 SFU（特殊函数单元）。底部还配备 Tensor Memory Accelerator（张量内存加速器）和 Tex（纹理单元）。这种设计使得 H100 能够高效地并行执行大规模矩阵运算和深度学习工作负载。


### 组件

#### GPU 计算组件总结

| 层级                 | 组件                            | 数量 (H100)     | 角色      | 负责的操作                            |
| ------------------ | ----------------------------- | ------------- | ------- | -------------------------------- |
| **GPU 级**          | GigaThread Engine             | 1             | 全局调度器   | 将 thread block 分配到各个 SM          |
| **SM 级**           | SM (Streaming Multiprocessor) | 132           | 独立计算单元  | 执行一个或多个 thread block，管理内部资源      |
| **SubPartition 级** | Warp Scheduler                | 4 per SM      | Warp 调度 | 从 warp pool 中选择 eligible warp 发射 |
|                    | Dispatch Unit                 | 2 per SubPart | 指令分发    | 读操作数、选执行单元、发射指令                  |
|                    | Scoreboard                    | 1 per SubPart | 依赖追踪    | 跟踪寄存器状态，检测数据冒险                   |
| **执行单元级**          | Tensor Core                   | 4 per SM      | 矩阵乘法    | GEMM，~1024 FLOPs/cycle，占 93%+ 算力 |
|                    | FP32 CUDA Cores               | 128 per SM    | 单精度浮点   | ReLU、pointwise ops、reduction     |
|                    | FP64 CUDA Cores               | 64 per SM     | 双精度浮点   | 科学计算（ML 中很少用）                    |
|                    | INT32 Cores                   | 64 per SM     | 整数运算    | 地址计算、索引、位操作                      |
|                    | Load/Store Units              | 32 per SM     | 内存访问    | 发起 load/store 请求，地址计算            |
|                    | SFU (Special Function Unit)   | 16 per SM     | 特殊函数    | sin, cos, exp, rsqrt 等超越函数       |
|                    | Texture Units                 | 4 per SM      | 纹理采样    | 图形渲染用，ML 中偶尔用于插值                 |

#### 计算组件层级关系

```
GPU
 └── GigaThread Engine (全局调度)
      └── SM ×132
           ├── Warp Pool (最多 64 warps 常驻)
           └── SubPartition ×4
                ├── Warp Scheduler ──► 选 warp
                ├── Dispatch Unit ×2 ──► 发指令
                └── Execution Units
                     ├── Tensor Core (矩阵乘)
                     ├── FP32 Cores ×32 (向量算术)
                     ├── INT32 Cores ×16
                     ├── FP64 Cores ×16
                     ├── LD/ST Units ×8
                     └── SFU ×4
```

#### GPU 存储组件总结

| 层级 | 组件 | 容量 (H100) | 带宽 | 延迟 | 作用域 | 用途 |
|------|------|-------------|------|------|--------|------|
| **片外** | HBM (显存) | 80 GB | 3.35 TB/s | ~400 cycles | 全局 | 模型权重、激活、大 tensor |
| | L2 Cache | 50 MB | ~12 TB/s | ~100 cycles | 全局 | 自动缓存 HBM 数据 |
| **SM 级** | SMEM (Shared Memory) | 256 KB per SM | ~33 TB/s | ~20 cycles | Block 内共享 | Tile 数据、线程间通信 |
| | L1 Cache | 与 SMEM 共享 | ~33 TB/s | ~20 cycles | SM 私有 | 自动缓存（可配置比例） |
| | TMEM (Tensor Memory) | B200 新增 | 极高 | 极低 | SubPart 私有 | 喂 Tensor Core 的专用缓存 |
| **线程级** | Register File | 64K ×32bit per SM | ~80 TB/s | 1 cycle | 线程私有 | 局部变量、中间结果 |
| | Local Memory | 溢出到 HBM | 同 HBM | 高 | 线程私有 | 寄存器溢出 (register spill) |
| **特殊** | Constant Memory | 64 KB | 广播优化 | ~4 cycles (cached) | 只读全局 | 常量参数、超参数 |
| | Texture Memory | 与 L1 共享 | 空间局部性优化 | 中等 | 只读全局 | 2D 空间数据访问 |

#### 存储层级金字塔

```
                    ┌─────────┐
                    │ Register│  64K×32bit/SM, 1 cycle, ~80 TB/s
                    │  File   │  线程私有
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  SMEM   │  256 KB/SM, ~20 cycles, ~33 TB/s
                    │L1 Cache │  Block 共享 / 自动缓存
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │L2 Cache │  50 MB, ~100 cycles, ~12 TB/s
                    │         │  全局共享，自动管理
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │   HBM   │  80 GB, ~400 cycles, 3.35 TB/s
                    │ (DRAM)  │  全局，持久存储
                    └─────────┘

容量:    小 ◄─────────────────────────────► 大
速度:    快 ◄─────────────────────────────► 慢
```

#### 各存储的典型使用场景

| 存储 | ML 中的典型用途 | 编程方式 |
|------|----------------|---------|
| **Register** | 累加器、循环变量、Tensor Core 输入输出 | 自动分配，局部变量 |
| **SMEM** | GEMM tiling、attention 的 K/V cache、reduction 中间结果 | `__shared__` 显式声明 |
| **L2** | 跨 SM 复用的数据（如同一 batch 的不同 head） | 自动，可用 `cudaAccessPolicyWindow` 提示 |
| **HBM** | 权重矩阵、输入输出 tensor、optimizer state | `cudaMalloc`，全局数组 |
| **Constant** | Layer 的超参数、lookup table | `__constant__` 声明 |


### 通过伪代码理解warp和dispatch的工作机制


```python
# ============ SM 内部结构 ============
class SM:
    def __init__(self):
        # 执行单元（以 Ampere 架构为例，每个 SM 有 4 个 sub-partition）
        self.sub_partitions = [SubPartition() for _ in range(4)]
        
        # 每个 sub-partition 有自己的 warp scheduler + dispatch unit
        
class SubPartition:
    def __init__(self):
        self.warp_scheduler = WarpScheduler()
        self.dispatch_units = [DispatchUnit(), DispatchUnit()]  # 通常 2 个
        
        # 执行单元
        self.int32_units = [INT32_ALU() for _ in range(16)]
        self.fp32_units = [FP32_ALU() for _ in range(16)]
        self.fp64_units = [FP64_ALU() for _ in range(8)]
        self.ld_st_units = [LoadStoreUnit() for _ in range(8)]
        self.sfu_units = [SpecialFuncUnit() for _ in range(4)]  # sin, cos, exp...
        self.tensor_cores = [TensorCore() for _ in range(1)]


# ============ Warp Scheduler ============
class WarpScheduler:
    """决定下一个周期执行哪个 warp"""
    
    def __init__(self):
        self.warp_pool = []  # 该 scheduler 管理的所有 warp（通常 8 个左右）
        
    def select_warps_to_issue(self):
        """每个周期选择可以发射的 warp"""
        
        ready_warps = []
        for warp in self.warp_pool:
            if self.is_warp_eligible(warp):
                ready_warps.append(warp)
        
        # 调度策略：GTO (Greedy Then Oldest), LRR (Loose Round Robin), 等
        selected = self.scheduling_policy(ready_warps)
        return selected  # 可能返回 1-2 个 warp（取决于 dispatch unit 数量）
    
    def is_warp_eligible(self, warp):
        """检查 warp 是否可以被调度"""
        
        if warp.is_finished():
            return False
            
        # 检查 scoreboard：指令的操作数是否就绪
        next_inst = warp.get_next_instruction()
        if not self.scoreboard.operands_ready(warp.id, next_inst):
            return False  # 数据依赖，stall
            
        # 检查结构冒险：目标执行单元是否可用
        if not self.check_structural_hazard(next_inst):
            return False
            
        # 检查是否在等待 barrier 同步
        if warp.waiting_at_barrier:
            return False
            
        return True
    
    def scheduling_policy(self, ready_warps):
        """调度策略示例：GTO - 优先让同一个 warp 连续执行"""
        if not ready_warps:
            return []
        
        # 优先选上次执行的 warp（局部性）
        if self.last_issued_warp in ready_warps:
            return [self.last_issued_warp]
        
        # 否则选最老的 ready warp
        return [min(ready_warps, key=lambda w: w.age)]


# ============ Dispatch Unit ============
class DispatchUnit:
    """把 warp scheduler 选中的指令分发到执行单元"""
    
    def dispatch(self, warp, instruction):
        """将指令分发到具体执行单元"""
        
        # 1. 从 Register File 读取操作数（32 个线程的数据）
        operands = self.read_operands(warp, instruction)
        # operands 是 32 份数据，每个 lane 一份
        
        # 2. 根据指令类型选择执行单元
        exec_unit = self.select_execution_unit(instruction)
        
        # 3. 发射到执行单元
        exec_unit.issue(warp.id, warp.active_mask, instruction, operands)
        
        # 4. 更新 scoreboard：标记目标寄存器为 pending
        self.scoreboard.mark_pending(warp.id, instruction.dest_reg)
        
    def select_execution_unit(self, instruction):
        match instruction.opcode:
            case 'FADD' | 'FMUL' | 'FFMA':
                return self.find_available(self.fp32_units)
            case 'IADD' | 'IMUL' | 'IMAD':
                return self.find_available(self.int32_units)
            case 'LD' | 'ST':
                return self.find_available(self.ld_st_units)
            case 'SIN' | 'COS' | 'EXP' | 'RCP':
                return self.find_available(self.sfu_units)
            case 'HMMA' | 'IMMA':  # Tensor Core ops
                return self.find_available(self.tensor_cores)


# ============ 执行单元 ============
class FP32_ALU:
    """FP32 执行单元 - SIMT 执行"""
    
    def issue(self, warp_id, active_mask, instruction, operands):
        """执行 32 个线程的计算"""
        
        results = [None] * 32
        for lane in range(32):
            if active_mask & (1 << lane):  # 只执行 active 的线程
                a = operands.src1[lane]
                b = operands.src2[lane]
                
                match instruction.opcode:
                    case 'FADD':
                        results[lane] = a + b
                    case 'FMUL':
                        results[lane] = a * b
                    case 'FFMA':
                        c = operands.src3[lane]
                        results[lane] = a * b + c
        
        # 写回 register file（流水线化，可能需要几个周期）
        self.writeback_queue.enqueue(warp_id, instruction.dest_reg, results)


# ============ 完整的每周期流程 ============
def sm_cycle(sub_partition):
    """每个时钟周期的流水线操作"""
    
    # Stage 1: Warp Scheduler 选择 warp
    selected_warps = sub_partition.warp_scheduler.select_warps_to_issue()
    
    # Stage 2: Dispatch Unit 分发指令
    for i, warp in enumerate(selected_warps):
        if i < len(sub_partition.dispatch_units):
            instruction = warp.fetch_next_instruction()
            sub_partition.dispatch_units[i].dispatch(warp, instruction)
            warp.pc += 1
    
    # Stage 3-N: 执行单元流水线执行（并行进行）
    for unit in all_execution_units(sub_partition):
        unit.pipeline_tick()
    
    # Writeback: 完成的结果写回 register file，更新 scoreboard
    sub_partition.process_writebacks()
```

## 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                         SM                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Sub-Partition (×4)                         │ │
│  │                                                         │ │
│  │   ┌──────────────┐                                      │ │
│  │   │ Warp Pool    │  (8 warps)                          │ │
│  │   │ W0 W1 W2 ... │                                      │ │
│  │   └──────┬───────┘                                      │ │
│  │          │ 哪个 warp ready?                              │ │
│  │          ▼                                              │ │
│  │   ┌──────────────┐                                      │ │
│  │   │Warp Scheduler│ ──选择 1-2 个 eligible warp          │ │
│  │   └──────┬───────┘                                      │ │
│  │          │                                              │ │
│  │          ▼                                              │ │
│  │   ┌──────────────┐    ┌──────────────┐                  │ │
│  │   │Dispatch Unit │    │Dispatch Unit │  (×2)            │ │
│  │   └──────┬───────┘    └──────┬───────┘                  │ │
│  │          │                   │                          │ │
│  │          ▼                   ▼                          │ │
│  │   ┌─────────────────────────────────────────────┐       │ │
│  │   │           Execution Units                    │       │ │
│  │   │  INT32  FP32  FP64  LD/ST  SFU  TensorCore  │       │ │
│  │   └─────────────────────────────────────────────┘       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

| 组件                 | 职责                     | 类比             |
| ------------------ | ---------------------- | -------------- |
| **Warp Scheduler** | 决定"谁来执行"，检查依赖、冒险、选择策略  | 调度员：选择下一个上场的选手 |
| **Dispatch Unit**  | 决定"怎么执行"，读操作数、选执行单元、发射 | 分配员：把选手送到正确的赛道 |

Cycle 1:  Warp_A: LD r1, [addr]     # 发起内存读取，需要等待 ~400 周期
Cycle 2:  Warp_B: ADD r2, r3, r4    # 切换到 B
Cycle 3:  Warp_C: MUL r5, r6, r7    # 切换到 C
...
Cycle 400: Warp_A: (内存返回)       # A 的数据到了
Cycle 401: Warp_A: ADD r8, r1, r9   # A 继续执行

### Tensor Core 的重要性

下面的数字用 H100 SXM 的量级做直觉对比：dense BF16 Tensor Core 约 990 TFLOPs，FP32 CUDA Core 约 60-66 TFLOPs。比例不是为了精确 benchmark，而是说明现代 ML workload 的主计算路径在哪里。

<div class="compute-share-card">
  <div class="compute-share-head">
    <span>H100 compute path</span>
    <strong>GEMM dominates ML throughput</strong>
  </div>
  <div class="compute-share-row tensor">
    <div class="compute-share-label">
      <strong>Tensor Core</strong>
      <span>BF16 dense GEMM</span>
    </div>
    <div class="compute-share-track"><span class="share-94"></span></div>
    <div class="compute-share-value">~990 TFLOPs · ~94%</div>
  </div>
  <div class="compute-share-row cuda">
    <div class="compute-share-label">
      <strong>CUDA Core</strong>
      <span>FP32 scalar/vector path</span>
    </div>
    <div class="compute-share-track"><span class="share-6"></span></div>
    <div class="compute-share-value">~60-66 TFLOPs · ~6%</div>
  </div>
  <div class="compute-share-foot">
    2:4 structured sparsity can roughly double Tensor Core peak, so the gap can be even larger for sparse-friendly GEMM.
  </div>
</div>

结论要更精确地说：Transformer 里的 projection、MLP、attention GEMM 需要尽量落到 Tensor Core；CUDA Core 仍然负责地址计算、分支、elementwise、reduction、softmax、normalization 和各种 glue code。优化 compute-bound kernel 时，如果主路径没有用上 Tensor Core，通常先不要谈更细的调度技巧。

---

## 第二部分：CUDA 编程模型与内核工程实战

## Q1 基础环境搭建与 hello world kernel

### 项目文件架构

```
project/
├── csrc/
│   ├── kernels.cpp     # C++ 声明 + pybind11 绑定
│   └── kernels.cu      # CUDA 内核实现
├── setup.py            # 使用 setuptools 编译扩展
└── main.py             # import 编译好的模块
```

> [!tip]
> - 快速实验用 `load_inline()`，代码嵌入 Python 字符串中
> - 正式项目用 `setup.py` + 分离文件，便于版本控制和调试
> - `.cu` 文件由 nvcc 编译，`.cpp` 文件由系统 C++ 编译器编译

### 内联编译方式

使用 PyTorch JIT 编译 CUDA 内核时，需要将代码分为两部分：

**C++ 声明（cpp_sources）** - 只包含函数声明，由普通 C++ 编译器编译：

```cpp
#include <torch/extension.h>

// 函数声明
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
```

**CUDA 源码（cuda_sources）** - 包含内核定义和调用内核的包装函数：

```cuda
// 内核定义
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 包装函数（必须在 .cu 文件中，因为使用了 <<<>>> 语法）
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), n
    );
    return c;
}
```

> [!important]
> `<<<>>>` 内核启动语法只能被 nvcc 编译，必须放在 `cuda_sources` 中，不能放在 `cpp_sources` 中。

### 编译与加载

```python
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name='cuda_kernels',
    cpp_sources=cpp_source,        # 只有声明
    cuda_sources=full_cuda_source,  # 内核 + 包装函数
    functions=['vector_add'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

# 使用编译好的内核
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = module.vector_add(a, b)
```

## Q2 理解 hello world kernel

### GPU 硬件架构 Intro

NVIDIA GPU 采用层次化的并行架构：

```
GPU
├── SM (Streaming Multiprocessor) × N    # 多个流式多处理器
│   ├── CUDA Cores × M                   # 每个 SM 有多个 CUDA 核心
│   ├── Shared Memory                    # 片上共享内存（快）
│   ├── L1 Cache                         # 一级缓存
│   └── Warp Scheduler                   # 线程束调度器
└── Global Memory (HBM/GDDR)             # 全局显存（慢）
```

**关键概念**：
- **SM**：独立的计算单元，可以同时运行多个线程块
- **Warp**：32 个线程组成一个 warp，是实际执行的最小单位，warp 内的线程同步执行相同指令
- **Shared Memory**：同一 block 内的线程共享，速度接近寄存器
- **Global Memory**：所有线程可访问，但延迟高（~400 cycles）

### CUDA 编程模型

CUDA 将线程组织为三层结构，与硬件对应：

```
Grid (网格)
├── Block 0                    # 线程块，映射到 SM
│   ├── Thread 0..31  (Warp 0) # 线程，映射到 CUDA Core
│   ├── Thread 32..63 (Warp 1)
│   └── ...
├── Block 1
└── ...
```

### 解析

```cuda
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**`__global__`**：声明这是一个 kernel 函数，从 CPU 调用，在 GPU 上执行

**内置变量**：

| 变量 | 含义 | 示例值 |
|------|------|--------|
| `blockIdx.x` | 当前 block 在 grid 中的索引 | 0, 1, 2, ... |
| `blockDim.x` | 每个 block 中的线程数 | 256 |
| `threadIdx.x` | 当前线程在 block 中的索引 | 0, 1, ..., 255 |

**全局索引计算**：
```
idx = blockIdx.x * blockDim.x + threadIdx.x

例如：blocks=4, threads=256, 总共 1024 个线程
Block 0: idx = 0*256 + 0..255  = 0..255
Block 1: idx = 1*256 + 0..255  = 256..511
Block 2: idx = 2*256 + 0..255  = 512..767
Block 3: idx = 3*256 + 0..255  = 768..1023
```

**边界检查** `if (idx < n)`：因为线程总数可能大于数据量，需要防止越界访问

### kernel 启动

```cuda
int threads = 256;
int blocks = (n + threads - 1) / threads;  // 向上取整
vector_add_kernel<<<blocks, threads>>>(a, b, c, n);
```

**`<<<blocks, threads>>>`**：CUDA 特有的 kernel 启动语法
- 第一个参数：grid 中的 block 数量
- 第二个参数：每个 block 中的线程数

**向上取整公式**：`(n + threads - 1) / threads` 确保有足够的线程覆盖所有数据
```
n=1000, threads=256
blocks = (1000 + 255) / 256 = 4
总线程数 = 4 * 256 = 1024 >= 1000 ✓
```

### 执行流程

```
CPU                          GPU
 │                            │
 ├─ 分配 GPU 内存 ────────────►│
 ├─ 拷贝数据到 GPU ───────────►│
 ├─ 启动 kernel ──────────────►├─ 调度 blocks 到 SMs
 │                            ├─ 每个 SM 执行 warps
 │                            ├─ 线程并行计算
 ├─ 等待完成 ◄────────────────┤
 ├─ 拷贝结果回 CPU ◄──────────┤
 │                            │
```

> [!tip]
> 选择 `threads=256` 是因为：
> 1. 是 32（warp size）的倍数，避免资源浪费
> 2. 足够大以隐藏内存延迟
> 3. 不超过硬件限制（通常 1024）

---

## 第三部分：Roofline 性能建模与理论瓶颈分析

## 1. Motivation

在深度学习中，我们经常遇到这样的困惑：
- 增大 batch size 有时能提速，有时却没用
- 同样的模型在不同硬件上表现差异巨大
- 某些算子（如 attention）特别慢，但矩阵乘法却很快
**Roofline 分析**提供了一个简洁的框架来回答这些问题：它告诉你当前的瓶颈是**算力**还是**带宽**，以及如何优化。
## 2. 核心定义

任何计算都可以分解为两部分时间：
$$T_{\text{math}} = \frac{\text{FLOPs}}{\text{Accelerator FLOPs/s}}$$
$$T_{\text{comms}} = \frac{\text{Bytes}}{\text{Bandwidth (Bytes/s)}}$$

| 符号        | 含义     | 例子 (TPU v5e)                 |
| --------- | ------ | ---------------------------- |
| FLOPs/s   | 芯片峰值算力 | $1.97 \times 10^{14}$ (bf16) |
| Bandwidth | HBM 带宽 | $8.2 \times 10^{11}$ bytes/s |

### 算术强度 (Arithmetic Intensity)
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}$$
这是 roofline 分析的核心概念：**每搬运一个 byte 的数据，能做多少次浮点运算**。
### 临界强度 (Critical Intensity)
$$\text{Critical Intensity} = \frac{\text{Peak FLOPs/s}}{\text{Peak Bandwidth}}$$
对于 TPU v5e MXU：
$$\frac{1.97 \times 10^{14}}{8.2 \times 10^{11}} \approx 240 \text{ FLOPs/byte}$$
### 2.4 Compute-bound vs Memory-bound

| 条件                                                              | 状态                         | 含义           |
| --------------------------------------------------------------- | -------------------------- | ------------ |
| $\text{Intensity}_{\text{algo}} > \text{Intensity}_{\text{hw}}$ | **Compute-bound**          | 算力被充分利用 ✓    |
| $\text{Intensity}_{\text{algo}} < \text{Intensity}_{\text{hw}}$ | **Memory/Bandwidth-bound** | 算力在等数据，被浪费 ✗ |

## 3. 分析方法与例子

### 3.1 Roofline 分析的系统方法

进行 Roofline 分析需要遵循以下步骤：
**Step 1: 确定硬件参数**
首先需要查阅目标硬件的规格：

| 硬件      | HBM 容量 | HBM 带宽 $\beta$           | bf16 算力 $\pi$         | int8 算力               | 临界强度 $I_c = \pi/\beta$ |
| ------- | ------ | ------------------------ | --------------------- | --------------------- | ---------------------- |
| TPU v5e | 16 GB  | $8.1 \times 10^{11}$ B/s | $1.97 \times 10^{14}$ | $3.94 \times 10^{14}$ | 243                    |
| TPU v5p | 96 GB  | $2.8 \times 10^{12}$ B/s | $4.59 \times 10^{14}$ | $9.18 \times 10^{14}$ | 164                    |
| TPU v6e | 32 GB  | $1.6 \times 10^{12}$ B/s | $9.20 \times 10^{14}$ | $1.84 \times 10^{15}$ | 575                    |

**Step 2: 计算算法的 FLOPs 和 Bytes**

对于给定算法，分别计算：
- $W$：总计算量（FLOPs）
- $Q$：总数据传输量（Bytes）= 读取 + 写回

**Step 3: 计算算术强度**
$$I_{\text{algo}} = \frac{W}{Q} \quad \text{(FLOPs/Byte)}$$
**Step 4: 判断瓶颈类型**

比较 $I_{\text{algo}}$ 与 $I_c$：
$$T_{\text{actual}} = \max\left( \underbrace{\frac{W}{\pi}}_{T_{\text{compute}}}, \underbrace{\frac{Q}{\beta}}_{T_{\text{memory}}} \right)$$
- 若 $I_{\text{algo}} > I_c$：**Compute-bound**，$T_{\text{actual}} = T_{\text{compute}}$
- 若 $I_{\text{algo}} < I_c$：**Memory-bound**，$T_{\text{actual}} = T_{\text{memory}}$

**Step 5: 计算硬件利用率**
$$\text{Efficiency} = \frac{\text{Achieved FLOPs/s}}{\pi} = \frac{W / T_{\text{actual}}}{\pi}$$
### 3.2 Roofline 图的理解
**两个区域的物理意义**：
- **斜线区域（Memory-bound）**：数据搬运是瓶颈，计算单元在"等数据"
  - 实际吞吐 = $I_{\text{algo}} \times \beta$（随强度线性增长）
- **平台区域（Compute-bound）**：计算是瓶颈，已达峰值算力
  - 实际吞吐 = $\pi$（不再增长）

![[assets/Pasted image 20251216210603.png]]
> *展示了两种不同运算强度的算法（算法 1 和算法 2）及其在不同带宽（BW1 和 BW2）下的理论峰值吞吐量。红色区域表示算法在两种带宽下均受限于带宽，浪费了硬件峰值 FLOPs/s 的一部分。黄色区域表示算法仅在较低带宽（BW1）下受限于带宽。绿色区域表示算法在所有带宽下均受限于计算能力。此处，我们已充分利用了加速器的峰值 FLOPs/s，增加带宽或提高运算强度均无益处。*

### 3.3 例子：Dot Product（向量点积）

**问题**：计算 `x · y`，其中 `x, y ∈ bf16[N]`，输出 `bf16[1]`

| 项目              | 计算过程                                                 | 结果           |
| --------------- | ---------------------------------------------------- | ------------ |
| 读取              | `x` 需要 $2N$ bytes，`y` 需要 $2N$ bytes (bf16 = 2 bytes) | $4N$         |
| 写回              | 输出 1 个 bf16 标量                                       | $2$          |
| **总 Bytes $Q$** | $4N + 2$                                             | $\approx 4N$ |
| FLOPs           | $N$ 次乘法 + $(N-1)$ 次加法                                | $\approx 2N$ |

$$I_{\text{dot}} = \frac{W}{Q} = \frac{2N}{4N + 2} \xrightarrow{N \to \infty} \frac{1}{2}$$
对于 TPU v5e，$I_c = 243$，而 $I_{\text{dot}} = 0.5 \ll 243$
**结论**：向量点积**永远是 memory-bound**，无论 $N$ 多大。这是因为每个元素只被使用一次（没有数据复用），算术强度有上界。
> [!warning]
> 这解释了为什么 elementwise 操作（如 ReLU、LayerNorm）通常需要通过 **kernel fusion** 来优化——单独执行时几乎总是 memory-bound。
### 3.4 例子：Matrix Multiplication（矩阵乘法）⭐

**问题**：计算 `C = A @ B`，其中 `A ∈ bf16[M, K]`，`B ∈ bf16[K, N]`，输出 `C ∈ bf16[M, N]`

| 项目              | 计算过程                                  | 结果          |
| --------------- | ------------------------------------- | ----------- |
| 读取 A            | $M \times K$ 个 bf16                   | $2MK$ bytes |
| 读取 B            | $K \times N$ 个 bf16                   | $2KN$ bytes |
| 写回 C            | $M \times N$ 个 bf16                   | $2MN$ bytes |
| **总 Bytes $Q$** | $2(MK + KN + MN)$                     |             |
| FLOPs           | 每个输出元素需要 $K$ 次乘法 + $K$ 次加法，共 $MN$ 个输出 | $2MNK$      |

> [!note] 为什么是 2MNK？
> 矩阵乘法 $C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}$ 对每个输出元素做 $K$ 次 multiply-add。
> 一次 multiply-add = 2 FLOPs，故总共 $2 \times M \times N \times K$ FLOPs。

$$I_{\text{matmul}} = \frac{2MNK}{2(MK + KN + MN)} = \frac{MNK}{MK + KN + MN}$$
**特殊情况分析**（设 $M = B$（batch），$K = D$（hidden），$N = F$（output））：

| 情况 | 条件 | 近似强度 | 物理意义 |
|------|------|----------|----------|
| Batch 推理 | $B \ll D, F$ | $I \approx B$ | batch size 决定是否 compute-bound |
| 方阵乘法 | $M = K = N$ | $I \approx \frac{N}{3}$ | 维度越大越好 |
| GEMV | $N = 1$ | $I \approx 1$ | 向量-矩阵乘，几乎总是 memory-bound |

对于 `bf16[B, D] @ bf16[D, F] → bf16[B, F]`（典型的 FFN 层）：
当 $B \ll D, F$ 时：
$$I \approx \frac{BDF}{DF} = B$$
在 TPU v5e 上，**Batch size $B > 243$** 时，matmul 变成 compute-bound。

**常见矩阵乘法 FLOPs 速查表**：

| 操作 | Shape | FLOPs | 说明 |
|------|-------|-------|------|
| GEMM | `[M,K] @ [K,N]` | $2MNK$ | 通用矩阵乘 |
| GEMV | `[M,K] @ [K,1]` | $2MK$ | 矩阵-向量乘，$I \approx 1$ |
| 方阵乘法 | `[N,N] @ [N,N]` | $2N^3$ | $I \approx N/3$ |
| Batch GEMM | `[B,M,K] @ [B,K,N]` | $2BMNK$ | 批量矩阵乘 |

### 3.5 例子：完整 Roofline 计算

**问题**：在 TPU v5e 上计算 `bf16[256, 4096] @ bf16[4096, 4096]`，分析其性能。

**已知参数**：
- $\pi = 1.97 \times 10^{14}$ FLOPs/s（bf16 峰值算力）
- $\beta = 8.1 \times 10^{11}$ B/s（HBM 带宽）
- $I_c = \pi / \beta = 243$ FLOPs/Byte

**Step 2: 计算量**
- $M = 256, K = 4096, N = 4096$
- $W = 2MNK = 2 \times 256 \times 4096 \times 4096 = 8.59 \times 10^9$ FLOPs
- $Q = 2(MK + KN + MN) = 2(256 \times 4096 + 4096 \times 4096 + 256 \times 4096)$
  $= 2(1.05 \times 10^6 + 1.68 \times 10^7 + 1.05 \times 10^6) = 3.77 \times 10^7$ Bytes

**Step 3: 算术强度**
$$I = \frac{8.59 \times 10^9}{3.77 \times 10^7} = 228 \text{ FLOPs/Byte}$$
**Step 4: 判断瓶颈**
- $I = 228 < I_c = 243$ → **Memory-bound**（略低于临界点）

**Step 5: 计算时间和效率**
- $T_{\text{compute}} = W / \pi = 8.59 \times 10^9 / 1.97 \times 10^{14} = 43.6 \mu s$
- $T_{\text{memory}} = Q / \beta = 3.77 \times 10^7 / 8.1 \times 10^{11} = 46.5 \mu s$
- $T_{\text{actual}} = \max(43.6, 46.5) = 46.5 \mu s$
- Efficiency = $43.6 / 46.5 = 93.8\%$

**结论**：虽然略微 memory-bound，但效率已达 94%，接近最优。

### 3.6 例子：Int8 量化 Matmul

**问题**：`int8[B, D] @ int8[D, F] → int8[B, F]`

**变化分析**：

| 项目 | bf16 版本 | int8 版本 | 变化 |
|------|-----------|-----------|------|
| 数据类型大小 | 2 bytes | 1 byte | $\times 0.5$ |
| 总 Bytes $Q$ | $2(BD + DF + BF)$ | $BD + DF + BF$ | $\times 0.5$ |
| 峰值算力 $\pi$ | $1.97 \times 10^{14}$ | $3.94 \times 10^{14}$ | $\times 2$ |
| FLOPs $W$ | $2BDF$ | $2BDF$ | 不变 |

**新的算术强度**：
$$I_{\text{int8}} = \frac{2BDF}{BD + DF + BF}$$
当 $B \ll D, F$ 时：
$$I_{\text{int8}} \approx \frac{2BDF}{DF} = 2B$$
**新的临界强度**：
$$I_c^{\text{int8}} = \frac{3.94 \times 10^{14}}{8.1 \times 10^{11}} = 486$$
**Compute-bound 条件**：
$$2B > 486 \implies B > 243$$
**结论**：
- Int8 的临界 batch size **仍然约为 243**（与 bf16 相同！）
- 但达到 compute-bound 后，**吞吐量翻倍**

> [!tip]
> 量化的主要收益是在 compute-bound 区域提升吞吐，而非改变临界点。

### 3.7 例子：混合精度（Int8 权重 + BF16 激活）

**问题**：`bf16[B, D] @ int8[D, F] → bf16[B, F]`

这种方案常用于推理优化：权重量化为 int8，但激活保持 bf16 精度。

**分析**：

| 项目 | 计算 |
|------|------|
| 读取激活 A | $2BD$ bytes (bf16) |
| 读取权重 B | $DF$ bytes (int8) |
| 写回输出 C | $2BF$ bytes (bf16) |
| **总 Bytes $Q$** | $2BD + DF + 2BF$ |
| FLOPs | $2BDF$（仍按 bf16 计算） |

**算术强度**：
$$I_{\text{mixed}} = \frac{2BDF}{2BD + DF + 2BF}$$
当 $B \ll D, F$ 且 $D \approx F$ 时：
$$I_{\text{mixed}} \approx \frac{2BDF}{DF} = 2B$$
**Compute-bound 条件**（使用 bf16 算力 $\pi = 1.97 \times 10^{14}$）：
$$2B > \frac{1.97 \times 10^{14}}{8.1 \times 10^{11}} = 243 \implies B > 122$$
**结论**：混合精度方案只需 **$B > 122$** 即可 compute-bound，比纯 bf16（$B > 243$）更容易达到！

### 3.8 不同内存层级的影响

TPU 有多层内存，带宽差异巨大：

| 内存类型 | 带宽 | 相对 HBM | 典型用途 |
|----------|------|----------|----------|
| VMEM (SRAM) | ~18 TB/s | 22× | Tile 内部计算 |
| HBM | ~0.8 TB/s | 1× | 主存储 |
| ICI (芯片间) | ~0.09 TB/s | 0.1× | 多芯片通信 |
| PCIe | ~0.015 TB/s | 0.02× | Host-Device 传输 |

**例子**：`int8[B, 4096] @ int8[16384, 4096]`

| 内存来源 | 临界 Batch Size |
|----------|-----------------|
| HBM | $B > 271$ |
| VMEM | $B > 11$ |

**结论**：如果权重可以 fit 进 VMEM，临界点降低 25 倍！这就是为什么 **tiling** 和 **weight caching** 如此重要。
### 3.9 Batch-Specific 权重矩阵（反面教材）

**问题**：如果每个 batch 元素有不同的权重矩阵：
`int8[B, D] @ int8[B, D, F] → int8[B, F]`

求算术强度。

**分析**：

这种情况在某些特殊场景出现（如 LoRA 的极端情况、per-sample adaptation）。

| 项目 | 标准 matmul | Batch-specific 权重 |
|------|-------------|---------------------|
| 读取 X | $BD$ | $BD$ |
| 读取 Y | $DF$ | $\mathbf{BDF}$ |
| 写回 Z | $BF$ | $BF$ |
| **总 Bytes** | $BD + DF + BF$ | $BD + BDF + BF$ |
| FLOPs | $2BDF$ | $2BDF$ |

**算术强度**：
$$I = \frac{2BDF}{BD + BDF + BF}$$
分母中 $BDF$ 占主导（因为 $D, F$ 通常很大）：
$$I \approx \frac{2BDF}{BDF} = 2$$
**结论**：
$$\boxed{I \approx 2 \text{ (常数)}}$$
> [!warning] 这是一个反面教材！
>
> 算术强度为常数（约为 2）意味着：
> - **永远是 memory-bound**，无论 batch size 多大
> - 每个权重元素只被使用一次，没有数据复用
> - 硬件利用率极低：$\text{Efficiency} = \frac{2}{486} \approx 0.4\%$
>
> **避免这种模式的方法**：
> - 尽量共享权重（标准 matmul）
> - 如果必须用不同权重，考虑分组/分块复用
> - 使用 LoRA 等低秩适配方法

### 3.10 GPU (H100) Roofline 分析

**问题**：使用 NVIDIA H100 的规格，计算临界 batch size。

**H100 SXM 规格**：

| 参数 | 数值 | 说明 |
|------|------|------|
| bf16 Tensor Core FLOPs | $1.979 \times 10^{15}$ | **带稀疏性** |
| 实际 bf16 FLOPs | $\sim 1 \times 10^{15}$ | 无稀疏性（除以 2） |
| HBM3 带宽 | 3.35 TB/s = $3.35 \times 10^{12}$ B/s | |
| HBM 容量 | 80 GB | |

> [!note] 关于稀疏性
> NVIDIA 宣传的 Tensor Core FLOPs 包含 2:4 结构化稀疏加速。
> 实际使用中，如果模型没有稀疏化，需要将官方数字除以 2。

**临界强度**：
$$I_c = \frac{\pi}{\beta} = \frac{1 \times 10^{15}}{3.35 \times 10^{12}} = 298 \text{ FLOPs/byte}$$
**临界 Batch Size**（当 $B \ll D, F$）：
$$B > I_c \implies \boxed{B > 298}$$
**与 TPU v5e 对比**：

| 硬件 | 峰值算力 $\pi$ | HBM 带宽 $\beta$ | 临界强度 $I_c$ | 临界 B |
|------|----------------|------------------|----------------|--------|
| TPU v5e | $1.97 \times 10^{14}$ | $8.1 \times 10^{11}$ | 243 | ~243 |
| H100 SXM | $1.0 \times 10^{15}$ | $3.35 \times 10^{12}$ | 298 | ~298 |
| **比值** | 5× | 4× | 1.2× | ~1.2× |

> [!important] 关键发现
> 尽管 H100 的绝对算力和带宽都远超 TPU v5e，但**临界 batch size 几乎相同**！
>
> 这是因为两者的"算力/带宽"比例接近（约 240-300 FLOPs/byte）。
> 这个比例由芯片架构决定，是现代 AI 加速器的共同特征。
## 4. 实践：代码与工具

### 4.1 PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

def torch_roofline(B, D, F, device='cuda'):
    x = torch.randn(B, D, dtype=torch.bfloat16, device=device)
    y = torch.randn(D, F, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(10):
        _ = x @ y
    torch.cuda.synchronize()
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    ) as prof:
        for _ in range(100):
            _ = x @ y
        torch.cuda.synchronize()
    
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=10
    ))
    
    # 导出 Chrome trace
    prof.export_chrome_trace("torch_trace.json")

torch_roofline(256, 4096, 4096)
```

### 4.2 NVIDIA Nsight 分析（需要root权限）

```bash
# 收集 roofline 数据
ncu --set roofline -o profile ./your_program

# 查看报告
ncu-ui profile.ncu-rep
```

### 4.3 分析hello world kernel/matmul kernel的roofline


![[assets/Pasted image 20251223165208.png]]
所有ops都在同一个位置


![[assets/Pasted image 20251223164911.png]]

这个是matmul的roofline curve，可以看到随着scale增大，逐渐从memory bound成为了compute bound(这里会跑到线上去,为什么？因为这张图其实是错的，这是cuda core的图，但是bf16的matmul会用到的是tensor core！)
### 5 小结

![[assets/Pasted image 20251223105527.png]]

* 点相对于 Ridge Point 的位置
	* 点在 Ridge Point 左侧（AI < Ridge Point）：
		* 算法处于 Memory-Bound 状态。性能瓶颈是内存带宽，计算单元在等待数据。理论最大性能 = 带宽 × AI。此时增加计算能力没有意义，因为数据供应不上。
		* 优化方向：减少内存访问（算子融合、量化、稀疏化）或提高数据复用（改变算法）。
	* 点在 Ridge Point 右侧（AI > Ridge Point）：
		* 算法处于 Compute-Bound 状态。性能瓶颈是计算能力，内存带宽有富余。理论最大性能 = 峰值算力。此时增加内存带宽没有意义，因为计算跟不上。
		* 优化方向：使用更高效的计算指令（Tensor Core）、提高并行度、减少指令依赖。
* 点相对于 Roofline 线的位置
	* 点在线上（效率 > 80%）：
		* 实现已经接近硬件极限，当前 AI 下几乎没有优化空间。如果还想提升性能，必须改变算法本身来提高 AI（比如算子融合），或者换更强的硬件。
	* 点在线下（效率 < 80%）：实现没有充分利用硬件，存在优化空间。需要诊断具体原因。
		* 如果在 Memory-Bound 区域且效率低，可能是：内存访问不连续（non-coalesced）、cache 命中率低、存在 bank conflict、数据对齐问题。
		* 如果在 Compute-Bound 区域且效率低，可能是：occupancy 不足、寄存器溢出、没有使用 Tensor Core、存在指令依赖导致流水线停顿。
	* 点在线上方：理论上不可能。如果测量结果显示点在 roofline 上方，说明测量有误或者 AI 计算错误。常见原因包括：没有算上 cache 效应导致实际内存访问量小于理论值、FLOPs 统计有遗漏、计时不准确。

---

## 第四部分：GPU 性能优化的五条核心工程原则

## 2. Memory-Bound 优化的五条核心原则

将 transpose、stencil、SpMV、histogram、compaction 等各类 kernel 的优化经验归纳后，可以提炼出以下五条通用原则：

### 原则 A：字节账本（Byte Accounting）

优化前需要准确计算 kernel 的总内存流量。这一步决定了后续优化是否对准了真正的瓶颈。

计算规则：**将所有读、写操作的字节数累加**，其中 atomic 操作本质是 read-modify-write，其内存流量需按 2-3 倍计算。

> [!note] 常见误区
> 将 FLOPs 减半但未减少内存访问量，kernel 的执行时间不会有任何改善。更糟糕的情况是：为减少计算而引入额外的中间数组，反而增加了内存流量，导致性能下降。

**例：向量加法的字节账本**
```
// C[i] = A[i] + B[i]，N 个 float
// 读：A (4N bytes) + B (4N bytes) = 8N bytes
// 写：C (4N bytes)
// 总内存流量 = 12N bytes
// 峰值带宽 900 GB/s → 理论下界 = 12N / 900G 秒
```

**例：Histogram 的字节账本（容易算错）**
```
// 输入：N 个 int（读 4N bytes）
// 输出：bins[] 使用 atomicAdd
// atomic = read + modify + write → 每次 ≈ 3×4 = 12 bytes
// 总流量 = 4N + 12N = 16N bytes（而非天真以为的 4N + 4N）
```

---

### 原则 B：合并访存（Coalescing）

理想的访存形态：**一个 warp 的 32 个线程访问连续的 128 字节**（以 float 为例）。

具体要求：
- warp 内的 lane 映射到连续的内存地址
- 整体访问模式尽可能接近顺序 streaming

合并访存是所有其他优化的前提。如果 coalescing 未满足，有效带宽的上限将被大幅削减。

**例：矩阵转置中的 coalescing 问题**
```
// 反面：按列读取，warp 内线程访问步长为 N
out[j][i] = in[i][j]   // in 按行读（coalesced）✓
                        // out 按列写（strided）✗ → 带宽利用率骤降

// 正面：借助 shared memory 中转
tile[threadIdx.y][threadIdx.x] = in[row][col]   // coalesced 读
__syncthreads()
out[col][row] = tile[threadIdx.x][threadIdx.y]   // coalesced 写
```

**例：AoS vs SoA**
```
// AoS（Array of Structs）— warp 读 x 时跨步为 sizeof(Point)
struct Point { float x, y, z; };
Point pts[N];            // pts[tid].x → stride=12 bytes ✗

// SoA（Struct of Arrays）— warp 读 x 时连续
float px[N], py[N], pz[N];
px[tid]                  // stride=4 bytes，完美 coalesced ✓
```

---

### 原则 C：显式复用（Tiling）

当 kernel 存在邻域或数据复用结构（如 stencil、卷积、部分稀疏局部算子）时：

- 不应依赖硬件 L1/L2 cache 的隐式命中
- 应通过 tiling（shared memory 或 register）将复用转化为确定性行为

核心思路：**将数据从 HBM 加载到 SRAM 后进行多次复用，避免重复的 HBM 访问**。

> [!info] 什么是 Stencil？
> Stencil（模板计算）是一种常见的计算模式：每个输出元素由其**自身及固定邻域内的输入元素**加权求和得到。半径 R 的 1D stencil 意味着 `out[i]` 依赖 `in[i-R] ... in[i+R]` 共 2R+1 个元素。典型应用包括有限差分（CFD/PDE 求解）、图像模糊/锐化（2D stencil）、音频滤波等。由于相邻输出点的输入窗口高度重叠，stencil 是 tiling 优化的经典场景。

**例：1D Stencil — 无 tiling vs 有 tiling**
```
// 无 tiling：每个输出点从 HBM 读 2R+1 个邻居
// 相邻线程的读取大量重叠 → 依赖 cache 命中，不可控
out[i] = Σ w[k] * in[i-R+k],  k=0..2R

// 有 tiling：block 协作加载一段 tile（含 halo）到 shared memory
__shared__ float tile[BLOCK + 2*R];
tile[threadIdx.x + R] = in[blockStart + threadIdx.x];
if (threadIdx.x < R) {               // 加载左右 halo
    tile[threadIdx.x] = in[blockStart - R + threadIdx.x];
    tile[BLOCK + R + threadIdx.x] = in[blockStart + BLOCK + threadIdx.x];
}
__syncthreads();
out[i] = Σ w[k] * tile[threadIdx.x + k];  // 全部命中 SRAM
```

---

### 原则 D：减少同步与争用（Sync/Contention）

Memory-bound kernel 的性能瓶颈往往不在于带宽本身，而在于：
- `__syncthreads()` 调用过于频繁，将流水线吞吐转化为串行等待
- 原子操作竞争（histogram、scatter），将并行写退化为串行写

上一讲 Histogram 中的"层次化私有化"是这一原则的典型应用：将竞争范围从 global 逐级缩小到 block、再到 warp，从而降低争用开销。

**例：Histogram 层次化私有化**
```
// 级别 1 — 全局 atomic（最大争用）
atomicAdd(&global_bins[val], 1);           // 所有线程竞争同一组 bins

// 级别 2 — block 私有 bins → 最后归约
__shared__ int local_bins[NUM_BINS];       // 每个 block 一份
atomicAdd(&local_bins[val], 1);            // 争用缩小到 block 内
__syncthreads();
atomicAdd(&global_bins[tid], local_bins[tid]);  // 一次性归约

// 级别 3 — warp 私有 bins（寄存器/shared memory 分区）
// 争用进一步缩小到 32 个线程内，几乎无冲突
```

---

### 原则 E：延迟隐藏（Latency Hiding）

当内存访问的高延迟无法避免时（如 SpMV 的随机访问），可通过以下方式隐藏延迟：
- **提高 occupancy**：增加同时驻留的 warp 数量，使更多 warp 能在内存等待期间被调度执行
- **增加 ILP（指令级并行）**：通过 unroll 和一线程多元素策略，使单线程同时发起更多 load 请求

Grid-stride loop 是实现这一原则的通用工程化手段。

**例：Grid-stride loop + ILP unroll**
```
// 基础版：每线程处理一个元素，occupancy 是唯一延迟隐藏手段
for (int i = tid; i < N; i += gridDim.x * blockDim.x)
    out[i] = f(in[i]);

// ILP 版：每线程同时发起多个 load，隐藏内存延迟
for (int i = tid; i < N; i += stride * 4) {
    float a = in[i];
    float b = in[i + stride];
    float c = in[i + stride*2];
    float d = in[i + stride*3];    // 4 个 load 同时 in-flight
    out[i]            = f(a);
    out[i + stride]   = f(b);
    out[i + stride*2] = f(c);
    out[i + stride*3] = f(d);
}
```

---

---

## 第五部分：课后练习题与自测问答

### 模块 A：CUDA 环境与 Extension 边界
### 练习 1：CUDA extension 文件边界

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么 kernel launch 必须放在 .cu 文件？</span></summary>

CUDA kernel launch 语法只能由 nvcc 解析。普通 C++ compiler 只能处理 binding、函数声明和 CPU 侧 wrapper。真正包含 global kernel 和 launch syntax 的代码应该放进 cuda source 或 .cu 文件。

</details>

### 练习 2：block / thread / warp 的映射

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">一个 block 有 256 threads，会被拆成多少个 warp？</span></summary>

一个 warp 是 32 threads，所以 256 threads 对应 8 个 warps。一个 block 会整体调度到某个 SM 上，block 内 warps 再由 warp scheduler 发射。优化时要同时看 block 数是否提供足够并行度，以及每个 block 是否占用太多 register 或 shared memory。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">CPU 代码、CUDA kernel、PyTorch binding 三层各自负责什么？</span></summary>

CPU 侧负责分配张量、检查 shape/dtype/device、调用 launcher；`.cu` 里的 kernel 负责 GPU 上每个 thread 实际做什么；PyTorch binding 负责把 Python 调用接到 C++/CUDA 实现上。面试里不要把三层混在一起：`pybind` 不执行并行计算，kernel 也不负责 Python API。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">为什么 kernel 里几乎总要写边界检查？</span></summary>

CUDA launch 通常按 block size 向上取整，实际线程数会大于数据元素数。如果没有 `if (idx < n)`，最后一个 block 的多余线程可能越界读写。这个 bug 在小样例里可能不报错，但会污染显存或造成 nondeterministic failure。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">给 `n=10000, threads=256`，block 数怎么算？为什么不是整除？</span></summary>

用 `(n + threads - 1) // threads` 向上取整，得到 `(10000 + 255) // 256 = 40`。如果只用 `n // threads = 39`，只能覆盖 9984 个元素，最后 16 个元素没有线程处理。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">一个 CUDA hello-world kernel 慢，先不要怀疑什么？应该先查什么？</span></summary>

先不要怀疑算法复杂度。入门 kernel 更常见的问题是 launch 配置太小、数据在 CPU/GPU 之间反复拷贝、没有同步导致计时错误、dtype/device 不一致、边界检查错误。先确认正确性、计时方式和数据流，再谈优化。

</details>

### 模块 B：SM 体系结构与执行调度
### 练习 1：Tensor Core vs CUDA Core

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么 GEMM 优化首先关心 Tensor Core？</span></summary>

现代训练和推理里的大头是矩阵乘：attention projection、MLP、MoE expert、QK 和 PV 都是 GEMM-like workload。H100 上 BF16 Tensor Core peak 远高于 FP32 CUDA Core peak。如果主计算没有落到 Tensor Core，通常说明数据类型、矩阵形状、layout、alignment 或 kernel lowering 有问题。

</details>

### 练习 2：SM 内部瓶颈定位

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">occupancy 很高但 Tensor Core 利用率低，可能是什么原因？</span></summary>

occupancy 高只表示有足够 warps 常驻，不保证发射的是高吞吐指令。Tensor Core 利用率低可能来自没有使用 mma/wgmma 路径、tile shape 不匹配、global/shared memory 供数不足、register spill、同步过多，或者 kernel 主体其实是 softmax/reduction 这类非 GEMM 操作。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">SM、warp、thread block 三者是什么关系？</span></summary>

thread block 是程序员 launch 的调度单位，一个 block 会被放到某个 SM 上执行；block 内线程按 32 个一组组成 warp；warp 是硬件实际发射指令的基本单位。优化时，block 数决定全局并行度，warp 数决定隐藏延迟的能力，SM 资源决定同时能驻留多少 block/warp。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">为什么 occupancy 高不等于 kernel 快？</span></summary>

occupancy 只表示常驻 warp 多，不表示这些 warp 能发出有用指令。kernel 仍可能被 memory bandwidth、register spill、shared memory bank conflict、同步、指令依赖或没有走 Tensor Core 限制。正确判断要结合 eligible warps、stall reason、memory throughput 和 tensor pipe utilization。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">Tensor Core 在 Transformer 里主要吃哪些算子？CUDA Core 还负责什么？</span></summary>

Tensor Core 主要吃 projection GEMM、MLP GEMM、MoE expert GEMM、attention 里的 QK/PV 等矩阵乘。CUDA Core 仍负责 elementwise、地址计算、mask、normalization、softmax、reduction 和控制逻辑。高性能 kernel 通常是 Tensor Core 做主计算，CUDA Core 做 glue code。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">一个 GEMM 理论上 compute-bound，但 Tensor Core 利用率低，排查顺序是什么？</span></summary>

先查 dtype 和 shape 是否能走 Tensor Core，再看矩阵维度是否对齐、layout 是否连续、tile shape 是否合理、shared memory pipeline 是否供数及时、寄存器是否溢出。最后用 profiler 看是否真的有 `mma`/`wgmma` 指令，而不是退化成普通 CUDA Core 路径。

</details>

### 模块 C：Roofline 分析与性能定界
### 练习 1：判断 memory-bound / compute-bound

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">给定 FLOPs、Bytes、peak FLOPs 和 bandwidth，怎么判断瓶颈？</span></summary>

先算 arithmetic intensity 等于 FLOPs 除以 Bytes，再算硬件 ridge point 等于 peak FLOPs 除以 bandwidth。如果前者小于后者，性能上界由带宽决定，属于 memory-bound；如果前者大于后者，性能上界接近峰值算力，属于 compute-bound。注意 BF16 matmul 应该用 Tensor Core peak。

</details>

### 练习 2：Roofline 误用

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么只看 achieved TFLOPs 可能误判？</span></summary>

memory-bound kernel 的 achieved TFLOPs 天然低，因为上界由带宽决定。正确做法是同时看 achieved bandwidth、arithmetic intensity、理论 roofline 上界和实际运行时间。reduce 的 TFLOPs 很低可能已经接近带宽上限；GEMM 的 TFLOPs 低才更可能表示 Tensor Core 没吃满。

</details>


### 复习自测：看这组题能不能讲完整篇

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">Roofline 里 arithmetic intensity 的分子和分母分别是什么？</span></summary>

分子是实际完成的 FLOPs，分母是从目标内存层级搬运的 Bytes。分析 GPU kernel 时通常先看 HBM bytes，因为很多瓶颈来自显存带宽。AI 越高，说明每读一个 byte 做的计算越多；AI 越低，越容易 memory-bound。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">为什么同一个 kernel 可能在小 shape memory-bound，在大 shape compute-bound？</span></summary>

shape 变大后数据复用可能提高。比如 GEMM 的 K/N/M 变大后，每个加载进来的 tile 会参与更多乘加，arithmetic intensity 上升，瓶颈可能从 HBM 带宽转向 Tensor Core 计算峰值。Roofline 不是给算子贴永久标签，而是分析某个 shape 和实现。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">点落在 roofline 上方通常说明什么？</span></summary>

通常说明测量或账本错了。常见原因是 FLOPs 算多、Bytes 算少、计时没有同步、使用了错误的硬件 peak、没有区分 Tensor Core 和 CUDA Core peak，或者 cache 命中让实际 HBM traffic 小于按理论张量大小估算的 bytes。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">优化 memory-bound 和 compute-bound kernel 的第一反应分别是什么？</span></summary>

memory-bound 先减少 HBM traffic 或提高访问效率：coalescing、fusion、cache reuse、量化、减少中间写回。compute-bound 先提高计算单元利用率：Tensor Core 路径、tile shape、pipeline、减少依赖和 register spill。方向反了会浪费时间。

</details>
