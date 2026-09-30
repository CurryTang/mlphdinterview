# 01 · GPU 硬件体系、CUDA 编程模型与 Roofline 性能分析

本节梳理 GPU 硬件拓扑、流式多处理器（SM）执行管线、CUDA 编程模型与存储层级映射，结合 Roofline 性能分析框架量化算子计算与访存瓶颈，并归纳访存受限（Memory-Bound）算子的五条优化原则。

---

## 第一部分：GPU 硬件体系结构与核心组件

### 1.1 GPU 全局拓扑与层次化存储架构（H100 / B100）

![[assets/Pasted image 20251222135239.png|图 1：NVIDIA H100/B100 GPU 整体架构抽象与层次化存储]]
> **图 1 核心架构解析：** 展示了 GPU 的层次化内存与计算结构。多个流式多处理器（SM 0, SM 1, ... SM N-1）并行排布，每个 SM 内部封装 4 个第四代 Tensor Core（主导密集矩阵乘累加，贡献 93%+ 峰值算力）与 4 个 Warp Scheduler（每个调度器对应 32-lane 向量管线）。每个 SM 独享 256KB 的极高速 L1 Cache / Shared Memory（可由程序员显式控制分配）。所有 SM 通过片上互联共享 50MB 的 L2 Cache，最底层由高带宽片外显存（HBM3，H100 为 80GB@3.35TB/s，B100 为 192GB@8TB/s）承载全局权重与激活。

### 1.2 SM（流式多处理器）微架构与处理块（Processing Block）剖析

![[assets/Pasted image 20251222135741.png|图 2：NVIDIA H100 单个 SM（Streaming Multiprocessor）内部详细微架构]]
> **图 2 微架构解析：** 单个 SM 内部划分为 4 个物理处理块（Processing Block / Sub-Partition），统一共享 L1 指令缓存与 256KB L1 数据缓存/共享内存池。每个处理块配备：独立的 L0 指令缓存、Warp 调度器（每周期可调度 32 线程）、双发射分发单元（Dispatch Unit）、$16384 \times 32\text{-bit}$ 寄存器堆，以及细粒度执行单元矩阵（16 个 INT32 单元、16 个 FP32 单元、8 个 FP64 单元、1 个第四代 Tensor Core、LD/ST 访存单元与 SFU 特殊函数单元）。底部集成硬件级张量内存加速器（TMA，Tensor Memory Accelerator）与纹理单元（Tex）。

### 1.3 核心硬件组件与存储层级全景

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


### 1.4 Warp 调度机制、流水线与 Dispatch 逻辑

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

<div class="hardware-arch-card">
  <div class="arch-header">
    <h4>SM 硬件执行管线与调度数据通路</h4>
    <span class="arch-badge">NVIDIA Hopper / Blackwell SM Pipeline</span>
  </div>
  <div class="sm-flow-grid">
    <div class="flow-step-box">
      <div class="flow-step-num">Step 1 · Warp Pool</div>
      <div class="flow-step-title">常驻线程束池 (Resident Warps)</div>
      <div class="flow-step-desc">每个 Sub-Partition 常驻多达 8-16 个 Warps（每个 SM 最多 64 个 Warps）。线程束在流水线中处于就绪、阻塞或等待同步状态。</div>
      <div class="flow-step-tags">
        <span class="flow-tag">32 Lanes / Warp</span>
        <span class="flow-tag">Zero-overhead switch</span>
      </div>
    </div>
    <div class="flow-step-box highlight">
      <div class="flow-step-num">Step 2 · Warp Scheduler</div>
      <div class="flow-step-title">调度仲裁器 (Arbiter & Selection)</div>
      <div class="flow-step-desc">每个周期由 Scoreboard 检查操作数就绪状态与结构冲突，采用 GTO（Greedy-Then-Oldest）或 LRR 策略挑出 1~2 个就绪 Warp。</div>
      <div class="flow-step-tags">
        <span class="flow-tag accent">GTO / LRR 策略</span>
        <span class="flow-tag">Scoreboard 追踪</span>
      </div>
    </div>
    <div class="flow-step-box">
      <div class="flow-step-num">Step 3 · Dispatch Unit (×2)</div>
      <div class="flow-step-title">双指令分发单元 (Dual-Issue)</div>
      <div class="flow-step-desc">从 64KB 寄存器堆读取源操作数，将无依赖指令分发到对应的计算单元管线，同时向 Scoreboard 注册写回依赖。</div>
      <div class="flow-step-tags">
        <span class="flow-tag">Dual Issue</span>
        <span class="flow-tag">1-Cycle Reg Read</span>
      </div>
    </div>
  </div>
  <div class="arch-subpartitions">
    <div class="flow-step-num">Step 4 · 多功能执行单元矩阵 (Execution Units)</div>
    <div class="subpartition-unit-list">
      <div class="unit-item tensor"><strong>Tensor Core (4th Gen)</strong><span>MMA / WGMMA (~94% 算力)</span></div>
      <div class="unit-item"><strong>FP32 CUDA Cores (×32)</strong><span>单精度浮点 / 激活 / 归约</span></div>
      <div class="unit-item"><strong>INT32 Cores (×16)</strong><span>寻址计算 / 索引 / 逻辑位移</span></div>
      <div class="unit-item"><strong>LD / ST Units (×8)</strong><span>HBM / L2 / SMEM 显存访存</span></div>
      <div class="unit-item"><strong>SFU (×4)</strong><span>exp / rsqrt / sin / cos</span></div>
    </div>
  </div>
</div>

| 组件 | 核心职责 | 工程类比与关键瓶颈 |
| :--- | :--- | :--- |
| **Warp Scheduler** | 决定“谁来执行”：每个周期评估 Scoreboard 依赖、硬件冒险与屏障同步，决定 Eligible Warps | 调度员：负责消除流水线 Bubble，通过交错就绪 Warp 实现延迟隐藏 |
| **Dispatch Unit** | 决定“怎么执行”：读取寄存器操作数、绑定功能单元并双发射指令 | 分派员：指令分发带宽决定了单周期的 IPC 上限 |

> [!example] 指令流水线周期级调度追踪（零开销线程束切换与延迟隐藏）
> ```bash
> Cycle 1:   Warp_A: LD r1, [addr]     # 发起 HBM 访存读取，需等待约 400 个时钟周期
> Cycle 2:   Warp_B: ADD r2, r3, r4    # 切换至就绪的 Warp_B，零开销上下文切换
> Cycle 3:   Warp_C: MUL r5, r6, r7    # 切换至就绪的 Warp_C，维持计算流水线打满
> ...
> Cycle 400: Warp_A: (Memory Ready)    # 显存数据抵达，Scoreboard 解锁目标寄存器
> Cycle 401: Warp_A: ADD r8, r1, r9    # Warp_A 重新满足 Eligible 条件被发射执行
> ```

### 1.5 Tensor Core 的算力主导地位与执行路径

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

### 2.1 基础环境搭建与工程架构（JIT 内联 vs Setup.py 扩展）

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

### 2.2 Vector Add 算子实现与底层执行模型映射

#### 1. 硬件架构映射

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

#### 2. CUDA 线程组织模型

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

#### 3. 内核代码解析与索引计算

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

#### 4. Kernel 启动配置与向上取整

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

#### 5. 端到端执行流程

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

### 3.1 性能建模动因与两段式耗时模型

在深度学习系统工程中，我们经常遇到如下核心问题：
- 增大 batch size 有时能成倍提升吞吐，有时却毫无改观；
- 同样的模型在不同硬件架构（如 A100 vs H100 vs TPU）上表现差异巨大；
- 某些算子（如 LayerNorm、Attention Softmax）用时漫长，但矩阵乘法却吞吐极高。

**Roofline 分析**提供了一个极简而完备的分析框架：它通过算法的**算术强度（Arithmetic Intensity）**与硬件的**临界强度（Critical Intensity / Ridge Point）**对比，明确揭示算子当前处于**算力受限（Compute-Bound）**还是**带宽受限（Memory-Bound）**状态，并指明唯一有效的优化路径。

### 3.2 核心定义与物理量标定

任何算子的执行时间在物理上均可解构为两部分：
$$T_{\text{math}} = \frac{\text{FLOPs}}{\text{Accelerator Peak FLOPs/s}}$$
$$T_{\text{comms}} = \frac{\text{Bytes}}{\text{Memory Bandwidth (Bytes/s)}}$$

| 物理符号 | 含义 | 硬件标定实例 (TPU v5e) | 硬件标定实例 (H100 SXM) |
| :--- | :--- | :--- | :--- |
| $\pi$ (Peak FLOPs/s) | 芯片理论峰值算力 | $1.97 \times 10^{14}$ (BF16 MXU) | $\sim 1.0 \times 10^{15}$ (BF16 Dense Tensor Core) |
| $\beta$ (Bandwidth) | 片外显存（HBM）物理带宽 | $8.2 \times 10^{11}$ Bytes/s (820 GB/s) | $3.35 \times 10^{12}$ Bytes/s (3.35 TB/s) |

#### 算术强度 (Arithmetic Intensity)
$$I_{\text{algo}} = \frac{\text{FLOPs}}{\text{Bytes}} \quad (\text{FLOPs/Byte})$$
每从存储层级搬运一个 Byte 的数据，能在计算单元内部完成多少次浮点运算。

#### 硬件临界强度 (Critical Intensity / Ridge Point)
$$I_c = \frac{\text{Peak FLOPs/s}}{\text{Peak Bandwidth}} = \frac{\pi}{\beta} \quad (\text{FLOPs/Byte})$$

- **TPU v5e MXU：** $I_c = \frac{1.97 \times 10^{14}}{8.1 \times 10^{11}} \approx 243 \text{ FLOPs/Byte}$
- **NVIDIA H100 SXM：** $I_c = \frac{1.0 \times 10^{15}}{3.35 \times 10^{12}} \approx 298 \text{ FLOPs/Byte}$

#### 瓶颈分类判据

| 判据条件 | 运行状态 | 物理本质与工程结论 |
| :--- | :--- | :--- |
| $I_{\text{algo}} \ge I_c$ | **Compute-Bound (算力受限)** | 硬件计算单元充分打满，显存总线带宽富余；优化重心是提升 Tensor Core 占比与指令级并行 |
| $I_{\text{algo}} < I_c$ | **Memory-Bound (带宽受限)** | 硬件计算单元大部分周期在等待数据喂入；优化重心是减少内存搬运（算子融合、量化）与增加数据复用 |

### 3.3 Roofline 分析五步法与理论吞吐区域

进行标准 Roofline 分析遵循以下五步：
- **Step 1: 确定目标硬件规格**（查阅标称 $\beta$ 与 $\pi$，计算 $I_c = \pi/\beta$）
- **Step 2: 计算算法的计算量 $W$（FLOPs）与访存量 $Q$（Bytes）**
- **Step 3: 计算算法算术强度** $I_{\text{algo}} = \frac{W}{Q}$
- **Step 4: 判断瓶颈类型与理论最短时间**：
  $$T_{\text{actual}} = \max\left( \underbrace{\frac{W}{\pi}}_{T_{\text{compute}}}, \underbrace{\frac{Q}{\beta}}_{T_{\text{memory}}} \right)$$
- **Step 5: 计算实际硬件利用率**：
  $$\text{Efficiency} = \frac{\text{Achieved FLOPs/s}}{\pi} = \frac{W / T_{\text{actual}}}{\pi}$$

![[assets/Pasted image 20251216210603.png|图 3：Roofline 性能边界与多层级带宽下的理论吞吐量分布]]
> **图 3 核心机制解读：** 展示了不同算术强度算法（算法 1 vs 算法 2）在不同硬件带宽（BW1 vs BW2）下的理论峰值吞吐量区域：
> - **红色区域（双重带宽受限）：** 算法在 BW1 与 BW2 下均处于倾斜的 Memory-Bound 区域，硬件峰值算力严重未饱满。
> - **黄色区域（单侧带宽受限）：** 算法仅在较低带宽（BW1）下受限，若切换至高带宽（BW2）即可跨越 Ridge Point 进入 Compute-Bound 平台。
> - **绿色区域（完全算力受限）：** 算法算术强度足够高，已达到硬件峰值 FLOPs/s 平台，此时单纯增加带宽不再带来任何实质性能收益。

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
### 3.4 Profiler 实战与 Roofline 实测定界

#### 3.4.1 PyTorch Profiler 导出 Chrome Trace 与 FLOPs 计数

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
    
    # 导出 Chrome trace 用于可视化时间线
    prof.export_chrome_trace("torch_trace.json")

torch_roofline(256, 4096, 4096)
```

#### 3.4.2 NVIDIA Nsight Compute (NCU) 硬件级 Roofline 剖析

```bash
# 收集 GPU 硬件计数器与 Roofline 剖析数据 (需具备驱动 profiling 权限)
ncu --set roofline -o profile ./your_program

# 启动 GUI 交互式分析器查看算子在 Roofline 图上的落点
ncu-ui profile.ncu-rep
```

#### 3.4.3 Vector Add 实测 Roofline（严格 Memory-Bound）

![[assets/Pasted image 20251223165208.png|图 4：BF16 Vector Add 在 RTX A5000 上的 Roofline 实测分布（Memory-Bound）]]
> **图 4 测量结论：** 不同规模（从 $n = 1\text{M}$ 至 $134\text{M}$）的向量加法算子全部紧密锚定在左侧斜率线上（实测达到 80%~90% 的峰值内存带宽）。因其算术强度 $I \approx 0.17 \text{ FLOPs/Byte}$ 远低于硬件 Ridge Point（72.4），算子完全受限于显存带宽，增大规模无法跨越至平台区。

#### 3.4.4 MatMul 实测 Roofline 演进与异常点剖析

![[assets/Pasted image 20251223164911.png|图 5：BF16 MatMul 在不同 Scale 下从 Memory-Bound 向 Compute-Bound 演进与异常点剖析]]
> **图 5 异常点与工程诊断要点：**
> 1. **图表标题异常说明：** 原图标题显示 `MatMul Roofline: □□□□ vs □□ (RTX A5000, BF16)`，系原测试脚本在 Linux 无中文字体环境下将“理论峰值 vs 测量性能”渲染为 Unicode 方块占位符（Tofu）。
> 2. **为什么大矩阵乘法突破了 Roofline 顶线（125%~179%）？**
>    - 图中蓝色顶线绘制的是 **CUDA Core (FP32) 的理论峰值算力**（约 65 TFLOPs）。
>    - 但现代 GPU 在执行 BF16 GEMM 时，底层硬件会自动走 **第四代 Tensor Core 硬件流水线**（其 BF16 Dense 算力上限在 A5000 上高达 130+ TFLOPs）。
>    - 因此当矩阵规模放大至 $1024 \times 1024$ 以上时，实际性能突破了绘制在图上的 CUDA Core“假天花板”。
>    - **工程教训：** 做 Roofline 分析时，硬件峰值 $\pi$ 必须严格对齐算子实际走的物理单元（Tensor Core vs CUDA Core），否则会导致效率计算超过 100% 的误判。

### 3.5 Roofline 瓶颈定界与优化决策矩阵

<div class="roofline-decision-card">
  <div class="decision-header">
    <h4>Roofline 性能优化决策矩阵</h4>
    <span class="decision-badge">Ridge Point 临界强度分水岭</span>
  </div>
  <div class="decision-grid">
    <div class="decision-branch memory">
      <span class="branch-badge">Memory-Bound 区域</span>
      <div class="branch-title">算术强度 AI &lt; Ridge Point</div>
      <code class="branch-formula">Achieved Throughput = AI × Bandwidth &lt; Peak FLOPs</code>
      <ul>
        <li><strong>首要瓶颈：</strong>显存总线带宽（HBM / DRAM），计算单元处于空转饥渴状态。</li>
        <li><strong>错误方向：</strong>增加计算核心数量或提升主频无收益。</li>
        <li><strong>核心优化战术：</strong>
          <ul>
            <li><strong>算子融合 (Kernel Fusion)：</strong>将 Elementwise/Norm/Activation 融于上游，避免中间张量往返写回 HBM。</li>
            <li><strong>显式切块复用 (Tiling)：</strong>将局部 Tile 搬运至 Shared Memory / Register，多次复用分摊访存。</li>
            <li><strong>精度量化 (Quantization)：</strong>FP32 $\to$ FP16/BF16 $\to$ INT8/FP4，直接缩减内存搬运字节数。</li>
            <li><strong>重计算 (Recomputation)：</strong>用轻量算力替代显存占用与搬运。</li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="decision-branch compute">
      <span class="branch-badge">Compute-Bound 区域</span>
      <div class="branch-title">算术强度 AI &ge; Ridge Point</div>
      <code class="branch-formula">Achieved Throughput &le; Peak FLOPs/s</code>
      <ul>
        <li><strong>首要瓶颈：</strong>硬件执行管线（ALU / Tensor Core）吞吐上限，访存带宽有充裕余量。</li>
        <li><strong>错误方向：</strong>单纯提升内存带宽或优化访存连续性无法进一步提升速度。</li>
        <li><strong>核心优化战术：</strong>
          <ul>
            <li><strong>启用专用加速核心：</strong>强制使用 Tensor Core / MMA / WGMMA 指令，释放 90%+ 算力。</li>
            <li><strong>提升指令级并行度 (ILP)：</strong>循环展开 (#pragma unroll)、多寄存器累加，掩盖执行管线延迟。</li>
            <li><strong>结构化稀疏 (Structured Sparsity)：</strong>采用 2:4 稀疏化指令倍增 Tensor Core 吞吐。</li>
            <li><strong>规避分支发散与停顿：</strong>消除 Warp Divergence，减少 RAW / WAR 数据冒险。</li>
          </ul>
        </li>
      </ul>
    </div>
  </div>
</div>

#### 点位相对于 Roofline 边界的位置判别

- **点位于 Ridge Point 左侧（$\text{AI} < I_c$）：** 处于 Memory-Bound 状态。瓶颈是内存带宽，计算单元等待数据。理论最大性能 $= \beta \times \text{AI}$。增加计算能力无意义，应主攻减少内存搬运（算子融合、量化）与增加数据复用。
- **点位于 Ridge Point 右侧（$\text{AI} > I_c$）：** 处于 Compute-Bound 状态。瓶颈是计算能力，内存带宽充足。理论最大性能 $= \pi$。增加带宽无意义，应主攻提高 Tensor Core 利用率、指令并行与流水线打满。
- **点在线上（效率 $> 80\%$）：** 算子实现已接近硬件物理极限，当前 AI 下优化空间极小。若想进一步提速，需通过算法级重构或算子融合提升 AI，或更换更高规格算力硬件。
- **点在线下（效率 $< 80\%$）：** 硬件利用率不足，存在显著优化空间：
  - **Memory-Bound 区域低效：** 常见于非合并访存（non-coalesced）、Shared Memory 存在 Bank Conflict、数据未对齐导致多次内存事务。
  - **Compute-Bound 区域低效：** 常见于 Occupancy 不足、寄存器溢出到 Local Memory、未走 Tensor Core 专用硬件、指令依赖导致流水线停顿。
- **点在线上方（物理不可能）：** 表明测试或计算存在错误。常见原因包括：未统计 L1/L2 缓存复用导致实际显存读取量远小于理论估算、FLOPs 统计遗漏、异步计时未执行 `cudaDeviceSynchronize()`、或硬件峰值选错（如用 CUDA Core 顶线评估 Tensor Core 算子）。

---

## 第四部分：GPU 性能优化的五条核心工程原则

将 transpose、stencil、SpMV、histogram、compaction 等各类 kernel 的优化经验归纳后，可以提炼出以下五条通用原则：

### 4.1 原则 A：字节账本（Byte Accounting）

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

### 4.2 原则 B：合并访存（Coalescing）

理想的访存形态：**一个 warp 的 32 个线程访问连续的 128 字节**（以 float 为例）。

具体要求：
- warp 内的 lane 映射到连续的内存地址
- 整体访问模式尽可能接近顺序 streaming

合并访存是所有其他优化的前提。如果 coalescing 未满足，有效带宽的上限将被大幅削减。

**例：矩阵转置中的 coalescing 问题**
```cuda
// 反面：按列读取，warp 内线程访问步长为 N
out[j][i] = in[i][j];   // in 按行读（coalesced）✓
                        // out 按列写（strided）✗ → 带宽利用率骤降

// 正面：借助 shared memory 中转
tile[threadIdx.y][threadIdx.x] = in[row][col];   // coalesced 读
__syncthreads();
out[col][row] = tile[threadIdx.x][threadIdx.y];   // coalesced 写
```

**例：AoS vs SoA**
```cpp
// AoS（Array of Structs）— warp 读 x 时跨步为 sizeof(Point)
struct Point { float x, y, z; };
Point pts[N];            // pts[tid].x → stride=12 bytes ✗

// SoA（Struct of Arrays）— warp 读 x 时连续
float px[N], py[N], pz[N];
px[tid]                  // stride=4 bytes，完美 coalesced ✓
```

---

### 4.3 原则 C：显式复用（Tiling）

当 kernel 存在邻域或数据复用结构（如 stencil、卷积、部分稀疏局部算子）时：

- 不应依赖硬件 L1/L2 cache 的隐式命中
- 应通过 tiling（shared memory 或 register）将复用转化为确定性行为

核心思路：**将数据从 HBM 加载到 SRAM 后进行多次复用，避免重复的 HBM 访问**。

> [!info] 什么是 Stencil？
> Stencil（模板计算）是一种常见的计算模式：每个输出元素由其**自身及固定邻域内的输入元素**加权求和得到。半径 R 的 1D stencil 意味着 `out[i]` 依赖 `in[i-R] ... in[i+R]` 共 2R+1 个元素。典型应用包括有限差分（CFD/PDE 求解）、图像模糊/锐化（2D stencil）、音频滤波等。由于相邻输出点的输入窗口高度重叠，stencil 是 tiling 优化的经典场景。

**例：1D Stencil — 无 tiling vs 有 tiling**
```cuda
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

### 4.4 原则 D：减少同步与争用（Sync / Contention）

Memory-bound kernel 的性能瓶颈往往不在于带宽本身，而在于：
- `__syncthreads()` 调用过于频繁，将流水线吞吐转化为串行等待
- 原子操作竞争（histogram、scatter），将并行写退化为串行写

上一讲 Histogram 中的"层次化私有化"是这一原则的典型应用：将竞争范围从 global 逐级缩小到 block、再到 warp，从而降低争用开销。

**例：Histogram 层次化私有化**
```cuda
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

### 4.5 原则 E：延迟隐藏（Latency Hiding）

当内存访问的高延迟无法避免时（如 SpMV 的随机访问），可通过以下方式隐藏延迟：
- **提高 occupancy**：增加同时驻留的 warp 数量，使更多 warp 能在内存等待期间被调度执行
- **增加 ILP（指令级并行）**：通过 unroll 和一线程多元素策略，使单线程同时发起更多 load 请求

Grid-stride loop 是实现这一原则的通用工程化手段。

**例：Grid-stride loop + ILP unroll**
```cuda
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

## 第五部分：课后练习题与自测问答

<details class="exercise">
<summary><span class="q-label">Q1</span> <span class="q-text">为什么 CUDA Kernel Launch 必须放在 .cu 文件中？</span></summary>

CUDA kernel launch 语法（`<<<>>>`）只能由 `nvcc` 解析。普通 C++ 编译器（如 g++、clang）只能处理 binding、函数声明和 CPU 侧 wrapper。真正包含 global kernel 和 launch 语法的代码必须放进 cuda source 或 `.cu` 文件中，通过 `nvcc` 进行预处理并降级生成 PTX / SASS 汇编。

</details>

<details class="exercise">
<summary><span class="q-label">Q2</span> <span class="q-text">一个 Thread Block 有 256 个线程，会被拆分成多少个 Warp？如何调度？</span></summary>

一个 Warp 固定为 32 个线程，因此 256 个线程对应 $256 / 32 = 8$ 个 Warps。整个 Thread Block 会作为整体原子地被调度分配到某个具体的 SM 上；Block 内的 8 个 Warps 再由 SM 内部的 4 个 Warp Scheduler 依据指令就绪状态动态发射。优化时既要保证 Block 数量充足打满 GPU 上的所有 SM，又要控制每个 Block 的寄存器和 Shared Memory 占用以避免降低 Occupancy。

</details>

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">CPU 代码、CUDA Kernel、PyTorch Binding 三层各自的核心职责是什么？</span></summary>

- **CPU 侧**：负责分配输入输出张量内存、校验 Shape/Dtype/Device 约束、计算 Grid/Block 维度并调用 Launcher。
- **CUDA Kernel（.cu）**：负责 GPU 端细粒度并行算子执行，定义每个 Thread/Warp 如何协作访问 Shared/Global 内存并完成数值计算。
- **PyTorch Binding（pybind11 / cpp_extension）**：负责将 Python 解释器的 Tensor 对象与底层 C++/CUDA 函数进行 ABI 级参数绑定与类型转换。三者不可混淆：PyTorch binding 不执行并行计算，Kernel 也不参与 Python 运行时。

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">为什么 CUDA Kernel 中几乎必须书写边界检查（Boundary Check）？</span></summary>

CUDA 启动 Grid 维度通常按 Block Size 向上取整（`ceil(N / BlockSize)`），实际分配的全局线程总数往往大于实际数据元素数 $N$。如果没有 `if (idx < n)` 边界保护，最后一个 Block 中超出数据范围的多余线程将产生非法的越界内存读写，这在小规模张量测试中可能侥幸未报段错误，但在真实训练推理中会引发内存污染或难以复现的静默数值错误。

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">给定 n=10000, threads=256，Block 数量如何计算？为什么不能直接向下整除？</span></summary>

必须使用向上取整公式 `(n + threads - 1) / threads`，即 `(10000 + 255) / 256 = 40` 个 Blocks。若错误地使用整除 `n / threads = 39`，则实际分配的线程数仅为 $39 \times 256 = 9984$，导致尾部剩余的 16 个元素无法被任何线程处理，造成算子计算结果截断缺失。

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">排查 CUDA Hello-World 类简易算子性能迟缓的正确路径是什么？</span></summary>

先不要怀疑算法时间复杂度。初级算子缓慢的核心病因通常包括：
1. Grid / Block launch 配置过小导致 SM 严重欠载；
2. 循环体内存在隐式的 CPU-GPU 张量来回拷贝（DtoH / HtoD）；
3. 统计执行耗时未在前后显式加入 `cudaDeviceSynchronize()` 导致计入了冷启动或异步开销；
4. 数据类型或设备未对齐导致的隐式转换。确认功能正确性、同步基准与端到端数据流后，再进入硬件微架构调优。

</details>

<details class="exercise">
<summary><span class="q-label">Q7</span> <span class="q-text">为什么现代大模型 GEMM 优化必须优先关注 Tensor Core？</span></summary>

现代大语言模型的核心计算瓶颈高度集中在密集矩阵乘法上（Attention 投影、MLP 前馈层、MoE 门控与专家层、FlashAttention 的 $QK^T$ 与 $PV$）。在 NVIDIA H100 架构上，BF16 Tensor Core 的理论峰值算力（$\sim 1000$ TFLOPs）是 FP32 CUDA Core（$\sim 67$ TFLOPs）的 15 倍以上。若核心算子未落入 Tensor Core 管线，算子性能在硬件物理层面上就已丢失一个数量级。

</details>

<details class="exercise">
<summary><span class="q-label">Q8</span> <span class="q-text">SM Occupancy 很高但 Tensor Core 利用率极低，根本原因可能是什么？</span></summary>

Occupancy 高仅代表 SM 内部常驻的活跃 Warp 数量充裕，但并不代表这些 Warp 正在发射高吞吐的张量指令。根本原因包括：
1. Kernel 底层未生成 `mma` 或 `wgmma` 硬件指令，退化为了普通的标量 FMA；
2. 矩阵分块 Tile 尺寸过小，导致 Tensor Core 硬件管线长期处于等待发射气泡状态；
3. Global/Shared Memory 供数带宽不足，Warp 因等待内存数据加载（Stall Long Scoreboard）而停顿；
4. 算子本身包含大量的 Softmax、LayerNorm 或 Reduction 等非矩阵乘胶水操作。

</details>

<details class="exercise">
<summary><span class="q-label">Q9</span> <span class="q-text">SM、Warp 与 Thread Block 三者之间的物理与逻辑映射关系是什么？</span></summary>

- **Thread Block**：程序员定义的逻辑组织与调度单位，启动时被原子分配至某一个具体的物理 SM，Block 在其生命周期内不可跨 SM 迁移。
- **Warp**：硬件执行与指令发射的基本原子单位（32 个线程锁步执行相同指令）。Block 被进一步划分为若干个 Warp。
- **SM（流式多处理器）**：包含物理寄存器堆、Shared Memory、Warp 调度器与执行管线的独立硬件芯片单元。一个 SM 上可并发驻留来自同一个或不同 Block 的多个 Warps。

</details>

<details class="exercise">
<summary><span class="q-label">Q10</span> <span class="q-text">为什么高 Occupancy 并不等价于高 Kernel 性能？</span></summary>

Occupancy 是用来“隐藏指令和内存访问延迟”的手段而非性能目的。当 Warp 数量已经足以掩盖流水线延迟（通常 40%~60% 的理论 Occupancy 即可满足）时，继续提升 Occupancy 并不会带来额外收益。相反，为了追求极高的 Occupancy 往往需要限制每个线程的寄存器使用量，极易诱发 Register Spill（寄存器溢出至高延迟 Local Memory），同时增加 Shared Memory 争用，最终反而导致算子吞吐大幅下降。

</details>

<details class="exercise">
<summary><span class="q-label">Q11</span> <span class="q-text">在 Transformer 架构中，Tensor Core 与 CUDA Core 的具体分工是什么？</span></summary>

- **Tensor Core**：专司高算术强度的 GEMM-like 主干算子，包括 Q/K/V 投影变换、MLP 的 Gate/Up/Down 投影、Attention 中的点积打分与加权聚合。
- **CUDA Core**：负责所有标量辅助、控制流与非矩阵计算，包括 RoPE 旋转位置编码、Softmax 指数归一化、RMSNorm / LayerNorm、SwiGLU 激活函数、内存重排与地址寻址计算。高性能 Kernel 通常采用 Tensor Core 负责重算力运算，CUDA Core 或专用单元负责轻量胶水逻辑。

</details>

<details class="exercise">
<summary><span class="q-label">Q12</span> <span class="q-text">一个理论上 Compute-Bound 的 GEMM 算子实测 Tensor Core 利用率低下，排查顺序是什么？</span></summary>

推荐排查步骤：
1. **指令验证**：通过 SASS 反汇编或 NCU 检查底层是否真正生成了 `HMMA` / `WGMMA` 指令；
2. **数据布局与对齐**：核验输入矩阵在 K 维度上是否为 16 字节对齐，访存连续性是否满足 Vectorized Load 要求；
3. **分块体系（Tiling）**：检查 Block Tile、Warp Tile 与 Thread Tile 形状是否匹配架构最佳实践；
4. **访存双缓冲与异步流水**：排查 Shared Memory 供数是否成为瓶颈，是否启用了 `cp.async` 隐藏数据加载延迟；
5. **寄存器压力**：排查是否存在 Register Spill 到 Local Memory。

</details>

<details class="exercise">
<summary><span class="q-label">Q13</span> <span class="q-text">给定 FLOPs、Bytes、峰值算力与带宽，如何严格判定算子受限类型？</span></summary>

1. 计算算法自身的算术强度：$I_{\text{algo}} = \frac{\text{FLOPs}}{\text{Bytes}}$；
2. 计算目标硬件平台的临界强度（Ridge Point）：$I_c = \frac{\text{Peak FLOPs/s}}{\text{Memory Bandwidth (Bytes/s)}}$；
3. **判定法则**：
   - 若 $I_{\text{algo}} < I_c$：判定为 **Memory-Bound**，性能上限由显存总线带宽决定，理论耗时为 $\frac{\text{Bytes}}{\beta}$；
   - 若 $I_{\text{algo}} \ge I_c$：判定为 **Compute-Bound**，性能上限由计算单元峰值吞吐决定，理论耗时为 $\frac{\text{FLOPs}}{\pi}$。对于 BF16 GEMM，$\pi$ 必须取 Tensor Core 的物理峰值。

</details>

<details class="exercise">
<summary><span class="q-label">Q14</span> <span class="q-text">为什么仅凭实测 Achieved TFLOPs 无法断定算子实现的好坏？</span></summary>

因为 Memory-Bound 类算子（如 Elementwise Add、LayerNorm、Softmax）的算术强度极低，受限于显存带宽物理极限，其理论上能够达到的 TFLOPs 本身就只有硬件峰值的数个百分点甚至更低。若某 Reduce 算子的实测带宽利用率已达 90% 的 HBM 理论上限，即便其 Achieved TFLOPs 仅有 2 TFLOPs，它也已经逼近物理极限；反之，若一个 GEMM 算子跑出 50 TFLOPs，但在拥有 1000 TFLOPs Tensor Core 的 H100 上其利用率仅为 5%，则是极其劣质的实现。必须结合 Achieved Bandwidth 与 Roofline 边界联合评定。

</details>

<details class="exercise">
<summary><span class="q-label">Q15</span> <span class="q-text">Roofline 分析中，算术强度的分子（FLOPs）与分母（Bytes）的物理本质是什么？</span></summary>

- **分子（FLOPs）**：算法本身在数学定义上必须完成的有效浮点运算次数（通常 1 次乘加 FMA 计为 2 FLOPs），独立于具体代码实现。
- **分母（Bytes）**：算子在执行过程中必须跨越特定硬件内存边界（通常指片外全局显存 HBM / DRAM）搬运的实际字节总量。算术强度反映了“每搬运 1 字节显存数据，能够支撑算法完成多少次浮点计算”，数值越高代表数据复用率越强。

</details>

<details class="exercise">
<summary><span class="q-label">Q16</span> <span class="q-text">为什么同一个算子在不同数据 Scale 下会发生瓶颈类型的物理转移？</span></summary>

以矩阵乘法为例，当 Batch Size 很小（如 $M=1$ 的 GEMV 推理解码阶段）时，每个权重矩阵参数从 HBM 载入后仅参与一次乘法，没有跨 Batch 的数据复用，算术强度约为 1 FLOPs/Byte，算子严格处于 Memory-Bound 区域；而随着 Batch Size 放大到数百或数千（Prompt 预填充或大 Batch 训练），每个权重元素在载入高速片上 SRAM 后被并发的数百个 Token 批量复用，算术强度线性上升至数百 FLOPs/Byte，跨越 Ridge Point 跃迁至 Compute-Bound 区域。

</details>

<details class="exercise">
<summary><span class="q-label">Q17</span> <span class="q-text">实际测试中落点突破 Roofline 理论顶线（计算效率超过 100%）的常见原因是什么？</span></summary>

主要原因包括：
1. **硬件峰值基准选错**：例如将 FP32 CUDA Core 峰值作为顶线，而实际运行的 BF16 算子触发了算力高达数倍的 Tensor Core 硬件管线；
2. **片上缓存命中未计入**：理论 Bytes 仅按全局张量尺寸估算，但大部分数据命中 L2 Cache 或 Shared Memory，实际发往 HBM 的物理流量远小于理论值；
3. **异步计时缺陷**：GPU 算子为异步提交，计时代码未插入 `cudaDeviceSynchronize()` 导致仅记录了 CPU Launch 开销；
4. **FLOPs 统计偏高**：公式估算的理论运算量包含了已被编译器死代码消除（DCE）的冗余逻辑。

</details>

<details class="exercise">
<summary><span class="q-label">Q18</span> <span class="q-text">针对 Memory-Bound 与 Compute-Bound 算子，第一反应的优化手段分别是什么？</span></summary>

- **Memory-Bound 算子（首要目标：削减显存字节账本与提升总线利用率）**：
  1. 实施算子融合（Kernel Fusion），将前后关联操作合并为一个 Kernel，消灭中间张量往返 HBM 的写入与读取；
  2. 实施数值量化（INT8 / FP8 / FP4），成倍削减数据传输位宽；
  3. 规避非合并访存（Coalescing）并消除 Shared Memory 的 Bank Conflict。
- **Compute-Bound 算子（首要目标：打满核心算力流水线与消除气泡）**：
  1. 切换至专用加速硬件通路（Tensor Core MMA / WGMMA 指令）；
  2. 展开循环（`#pragma unroll`）并发射多路累加寄存器，提升指令级并行度（ILP）；
  3. 采用 2:4 结构化稀疏加速；
  4. 消除分支发散（Warp Divergence），确保 32 个线程无停顿同步推进。

</details>

---

## 参考文献与拓展阅读

1. **Google DeepMind - *How To Scale Your Model*** (Jacob Austin, Sholto Douglas, Roy Frostig, et al., 2025): [Part 1: All About Rooflines](https://jax-ml.github.io/scaling-book).
   > 本章的 Roofline 五步法分析框架、TPU v5e/v5p/v6e 硬件物理量标定、Dot Product 与 GEMM 算术强度极限推导、量化与混合精度临界 Batch Size 推导、以及 Batch-Specific Weight 反面案例均源自该文献。
2. **Williams, S., Waterman, A., & Patterson, D. (2009)**. *Roofline: an insightful visual performance model for multicore architectures*. Communications of the ACM, 52(4), 65-76.
3. **NVIDIA Corporation (2022)**. *NVIDIA H100 Tensor Core GPU Architecture Whitepaper*.
4. **NVIDIA Corporation (2024)**. *CUDA C++ Programming Guide*.
5. **PyTorch Team (2024)**. *Custom C++ and CUDA Extensions Tutorial*.
