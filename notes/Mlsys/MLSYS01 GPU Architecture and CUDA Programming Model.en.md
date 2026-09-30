# 01 · GPU Architecture, CUDA Programming Model & Roofline Performance Analysis

This section details GPU hardware architecture, Streaming Multiprocessor (SM) execution pipelines, the CUDA programming model, and memory hierarchy mappings. It uses the Roofline performance model to quantify compute versus bandwidth bottlenecks and summarizes five core optimization principles for memory-bound kernels.

---

## Part 1: GPU Hardware Architecture & Core Components

### 1.1 Hierarchical Architecture Overview

![[assets/Pasted image 20251222135239.png|Figure 1: NVIDIA H100/B100 Top-Level Accelerator Architecture]]
> **Figure 1 Key Architectural Takeaways:** Shows the GPU's hierarchical memory and compute topology:
> - **Top-Level SM Array:** Multiple Streaming Multiprocessors (SM 0..SM N-1, up to 132 SMs on H100) are organized in parallel across GPCs.
> - **Compute Pipelines:** Each SM integrates 4 Tensor Cores (specialized MMA hardware delivering ~1024 FLOPs/cycle, dominating over 93% of dense DL compute) and 4 Warp Schedulers (each controlling 32 SIMD vector lanes).
> - **Memory Subsystem Hierarchy:** 256 KB L1/SMEM per SM provides low-latency (~20 cycles, ~33 TB/s aggregate) software-managed caching; all SMs share a 50 MB L2 Cache (~12 TB/s) backed by up to 80 GB HBM3 (3.35 TB/s peak).

![[assets/Pasted image 20251222135741.png|Figure 2: NVIDIA H100 Streaming Multiprocessor (SM) Microarchitecture]]
> **Figure 2 SM Microarchitecture Analysis:** Detailed breakdown of a single H100 SM:
> - **Four Sub-Partitions (Processing Blocks):** Each sub-partition operates independently with its own L0 Instruction Cache, Warp Scheduler (1 warp/cycle), Dual Dispatch Units, and a 16,384 × 32-bit Register File.
> - **Execution Units per Sub-Partition:** 16 INT32 cores, 16 FP32 cores, 8 FP64 cores, 1 Fourth-Gen Tensor Core, LD/ST units, and Special Function Units (SFUs).
> - **Asynchronous Engines:** Hardware TMA (Tensor Memory Accelerator) offloads tensor tiling and address calculation directly between global memory and shared memory without consuming SM registers or ALUs.

### 1.2 SM Compute & Memory Component Decomposition

#### Summary of GPU compute components

| Level | Component | Count (H100) | Role | Operations handled |
| ------------------ | ----------------------------- | ------------- | ------- | -------------------------------- |
| **GPU level** | GigaThread Engine | 1 | Global scheduler | Assigns thread blocks to SMs |
| **SM level** | SM (Streaming Multiprocessor) | 132 | Independent compute unit | Executes one or more thread blocks and manages internal resources |
| **SubPartition level** | Warp Scheduler | 4 per SM | Warp scheduling | Selects eligible warps from the warp pool to issue |
|                    | Dispatch Unit                 | 2 per SubPart | Instruction dispatch | Reads operands, selects execution units, issues instructions |
|                    | Scoreboard                    | 1 per SubPart | Dependency tracking | Tracks register state and detects data hazards |
| **Execution-unit level** | Tensor Core | 4 per SM | Matrix multiplication | GEMM, ~1024 FLOPs/cycle, accounting for 93%+ of compute |
|                    | FP32 CUDA Cores               | 128 per SM    | Single-precision floating point | ReLU, pointwise ops, reduction |
|                    | FP64 CUDA Cores               | 64 per SM     | Double-precision floating point | Scientific computing (rarely used in ML) |
|                    | INT32 Cores                   | 64 per SM     | Integer arithmetic | Address computation, indexing, bit operations |
|                    | Load/Store Units              | 32 per SM     | Memory access | Initiates load/store requests and performs address calculation |
|                    | SFU (Special Function Unit)   | 16 per SM     | Special functions | Transcendental functions such as sin, cos, exp, rsqrt |
|                    | Texture Units                 | 4 per SM      | Texture sampling | Used for graphics rendering, occasionally for interpolation in ML |

#### Hierarchy of compute components

```
GPU
 └── GigaThread Engine (global scheduling)
      └── SM ×132
           ├── Warp Pool (up to 64 resident warps)
           └── SubPartition ×4
                ├── Warp Scheduler ──► selects warps
                ├── Dispatch Unit ×2 ──► issues instructions
                └── Execution Units
                     ├── Tensor Core (matrix multiply)
                     ├── FP32 Cores ×32 (vector arithmetic)
                     ├── INT32 Cores ×16
                     ├── FP64 Cores ×16
                     ├── LD/ST Units ×8
                     └── SFU ×4
```

#### Summary of GPU memory components

| Level | Component | Capacity (H100) | Bandwidth | Latency | Scope | Use |
|------|------|-------------|------|------|--------|------|
| **Off-chip** | HBM (device memory) | 80 GB | 3.35 TB/s | ~400 cycles | Global | Model weights, activations, large tensors |
| | L2 Cache | 50 MB | ~12 TB/s | ~100 cycles | Global | Automatically caches HBM data |
| **SM level** | SMEM (Shared Memory) | 256 KB per SM | ~33 TB/s | ~20 cycles | Shared within a block | Tile data, inter-thread communication |
| | L1 Cache | Shared with SMEM | ~33 TB/s | ~20 cycles | Private to an SM | Automatic caching (configurable partitioning) |
| | TMEM (Tensor Memory) | New in B200 | Extremely high | Extremely low | Private to a SubPart | Dedicated cache for feeding Tensor Cores |
| **Thread level** | Register File | 64K ×32bit per SM | ~80 TB/s | 1 cycle | Private to a thread | Local variables, intermediate results |
| | Local Memory | Spills to HBM | Same as HBM | High | Private to a thread | Register spill |
| **Special** | Constant Memory | 64 KB | Broadcast-optimized | ~4 cycles (cached) | Read-only global | Constant parameters, hyperparameters |
| | Texture Memory | Shared with L1 | Optimized for spatial locality | Medium | Read-only global | 2D spatial data access |

#### Memory hierarchy pyramid

```
                    ┌─────────┐
                    │ Register│  64K×32bit/SM, 1 cycle, ~80 TB/s
                    │  File   │  Thread-private
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  SMEM   │  256 KB/SM, ~20 cycles, ~33 TB/s
                    │L1 Cache │  Block-shared / automatic cache
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │L2 Cache │  50 MB, ~100 cycles, ~12 TB/s
                    │         │  Globally shared, automatically managed
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │   HBM   │  80 GB, ~400 cycles, 3.35 TB/s
                    │ (DRAM)  │  Global, persistent storage
                    └─────────┘

Capacity: Small ◄─────────────────────────────► Large
Speed:    Fast ◄─────────────────────────────► Slow
```

#### Typical usage scenarios for each memory type

| Memory | Typical use in ML | Programming model |
|------|----------------|---------|
| **Register** | Accumulators, loop variables, Tensor Core inputs/outputs | Automatically allocated, local variables |
| **SMEM** | GEMM tiling, attention K/V cache, intermediate reduction results | Explicitly declared with `__shared__` |
| **L2** | Data reused across SMs (e.g., different heads from the same batch) | Automatic, with optional hints via `cudaAccessPolicyWindow` |
| **HBM** | Weight matrices, input/output tensors, optimizer state | `cudaMalloc`, global arrays |
| **Constant** | Layer hyperparameters, lookup tables | Declared with `__constant__` |


### Understanding the working mechanism of warp and dispatch through pseudocode


```python
# ============ SM internal structure ============
class SM:
    def __init__(self):
        # Execution units (using the Ampere architecture as an example, each SM has 4 sub-partitions)
        self.sub_partitions = [SubPartition() for _ in range(4)]
        
        # Each sub-partition has its own warp scheduler + dispatch units
        
class SubPartition:
    def __init__(self):
        self.warp_scheduler = WarpScheduler()
        self.dispatch_units = [DispatchUnit(), DispatchUnit()]  # typically 2
        
        # Execution units
        self.int32_units = [INT32_ALU() for _ in range(16)]
        self.fp32_units = [FP32_ALU() for _ in range(16)]
        self.fp64_units = [FP64_ALU() for _ in range(8)]
        self.ld_st_units = [LoadStoreUnit() for _ in range(8)]
        self.sfu_units = [SpecialFuncUnit() for _ in range(4)]  # sin, cos, exp...
        self.tensor_cores = [TensorCore() for _ in range(1)]


# ============ Warp Scheduler ============
class WarpScheduler:
    """Decide which warp executes in the next cycle"""
    
    def __init__(self):
        self.warp_pool = []  # all warps managed by this scheduler (typically around 8)
        
    def select_warps_to_issue(self):
        """Select warps that can issue each cycle"""
        
        ready_warps = []
        for warp in self.warp_pool:
            if self.is_warp_eligible(warp):
                ready_warps.append(warp)
        
        # Scheduling policy: GTO (Greedy Then Oldest), LRR (Loose Round Robin), etc.
        selected = self.scheduling_policy(ready_warps)
        return selected  # may return 1-2 warps (depending on the number of dispatch units)
    
    def is_warp_eligible(self, warp):
        """Check whether a warp can be scheduled"""
        
        if warp.is_finished():
            return False
            
        # Check the scoreboard: are the instruction operands ready?
        next_inst = warp.get_next_instruction()
        if not self.scoreboard.operands_ready(warp.id, next_inst):
            return False  # data dependency, stall
            
        # Check structural hazards: is the target execution unit available?
        if not self.check_structural_hazard(next_inst):
            return False
            
        # Check whether it is waiting at a barrier
        if warp.waiting_at_barrier:
            return False
            
        return True
    
    def scheduling_policy(self, ready_warps):
        """Example scheduling policy: GTO - prefer issuing the same warp consecutively"""
        if not ready_warps:
            return []
        
        # Prefer the previously issued warp (locality)
        if self.last_issued_warp in ready_warps:
            return [self.last_issued_warp]
        
        # Otherwise select the oldest ready warp
        return [min(ready_warps, key=lambda w: w.age)]


# ============ Dispatch Unit ============
class DispatchUnit:
    """Dispatch instructions selected by the warp scheduler to execution units"""
    
    def dispatch(self, warp, instruction):
        """Dispatch an instruction to a specific execution unit"""
        
        # 1. Read operands from the Register File (data for 32 threads)
        operands = self.read_operands(warp, instruction)
        # operands contains 32 values, one per lane
        
        # 2. Select the execution unit based on the instruction type
        exec_unit = self.select_execution_unit(instruction)
        
        # 3. Issue to the execution unit
        exec_unit.issue(warp.id, warp.active_mask, instruction, operands)
        
        # 4. Update the scoreboard: mark the destination register as pending
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


# ============ Execution units ============
class FP32_ALU:
    """FP32 execution unit - SIMT execution"""
    
    def issue(self, warp_id, active_mask, instruction, operands):
        """Execute computation for 32 threads"""
        
        results = [None] * 32
        for lane in range(32):
            if active_mask & (1 << lane):  # execute only active threads
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
        
        # Write back to the register file (pipelined, may take several cycles)
        self.writeback_queue.enqueue(warp_id, instruction.dest_reg, results)


# ============ Complete per-cycle flow ============
def sm_cycle(sub_partition):
    """Pipeline operations for each clock cycle"""
    
    # Stage 1: Warp Scheduler selects warps
    selected_warps = sub_partition.warp_scheduler.select_warps_to_issue()
    
    # Stage 2: Dispatch Unit dispatches instructions
    for i, warp in enumerate(selected_warps):
        if i < len(sub_partition.dispatch_units):
            instruction = warp.fetch_next_instruction()
            sub_partition.dispatch_units[i].dispatch(warp, instruction)
            warp.pc += 1
    
    # Stage 3-N: Execution-unit pipeline execution (in parallel)
    for unit in all_execution_units(sub_partition):
        unit.pipeline_tick()
    
    # Writeback: completed results write back to the register file and update the scoreboard
    sub_partition.process_writebacks()
```

### 1.3 Sub-Partition Execution Pipeline & Warp Scheduling Flow

<div class="hardware-arch-card">
  <div class="arch-card-header">
    <h4>NVIDIA Streaming Multiprocessor (SM) Execution Pipeline</h4>
    <span class="arch-badge">Warp Scheduling &amp; Dispatch Flow</span>
  </div>
  <div class="arch-subpartition-grid">
    <div class="arch-subpartition">
      <div class="subpart-title">Processing Sub-Partition (1 of 4 per SM)</div>
      <div class="arch-unit-group">
        <div class="unit-box pool">
          <span class="unit-tag">Warp Pool</span>
          <strong>Active Warps (up to 16 resident per Sub-Partition / 64 per SM)</strong>
        </div>
        <div class="unit-arrow">▼ Ready / Eligible Check (Scoreboard &amp; Barrier Filter)</div>
        <div class="unit-box scheduler">
          <span class="unit-tag">Warp Scheduler</span>
          <strong>Instruction Issue Logic (Greedy-Then-Oldest / Round-Robin)</strong>
        </div>
        <div class="unit-arrow">▼ Dual Issue per Cycle</div>
        <div class="dispatch-dual">
          <div class="unit-box dispatch">
            <span class="unit-tag">Dispatch Unit 0</span>
            <strong>Operand Collector &amp; Route</strong>
          </div>
          <div class="unit-box dispatch">
            <span class="unit-tag">Dispatch Unit 1</span>
            <strong>Operand Collector &amp; Route</strong>
          </div>
        </div>
        <div class="unit-arrow">▼ Issue to Execution Pipelines</div>
        <div class="execution-units-grid">
          <div class="exec-chip tensor">Tensor Core<br/><span>MMA / WGMMA</span></div>
          <div class="exec-chip fp32">FP32 Cores<br/><span>16 SIMD Lanes</span></div>
          <div class="exec-chip int32">INT32 Cores<br/><span>Address / Control</span></div>
          <div class="exec-chip ldst">LD/ST Units<br/><span>L1 / SMEM / Cache</span></div>
          <div class="exec-chip sfu">SFU Units<br/><span>sin / cos / exp / rsqrt</span></div>
          <div class="exec-chip fp64">FP64 Cores<br/><span>Double Precision</span></div>
        </div>
      </div>
    </div>
  </div>
</div>

| Component | Hardware Responsibility | Mental Model / Analogy |
| :--- | :--- | :--- |
| **Warp Scheduler** | Decides **"who executes"**: scans register scoreboard, checks barrier dependencies, and selects ready warps. | Dispatcher: picks the ready athlete from the dugout. |
| **Dispatch Unit** | Decides **"how to execute"**: collects operands from register files, arbitrates unit ports, and dispatches instructions. | Coordinator: guides the athlete onto the designated track. |

> [!example] Warp Latency Hiding Timeline
> ```cuda
> Cycle 1:   Warp_A: LD r1, [global_addr]  // Initiates global memory load (latches scoreboard, takes ~400 cycles)
> Cycle 2:   Warp_B: ADD r2, r3, r4        // Warp Scheduler instantly switches to Warp_B (Zero-overhead context switch)
> Cycle 3:   Warp_C: MUL r5, r6, r7        // Switches to Warp_C (no stall)
> ...
> Cycle 400: Warp_A: [Memory data returns] // Scoreboard marks Warp_A operands as ready
> Cycle 401: Warp_A: ADD r8, r1, r9        // Warp_A resumes execution seamlessly
> ```

### 1.4 The Dominance of Tensor Cores in Modern LLM Workloads

The numbers below use the H100 SXM scale for intuition: dense BF16 Tensor Core throughput is about 990 TFLOPs, while FP32 CUDA Core throughput is roughly 60-66 TFLOPs. The point is not a precise benchmark; it is to show where the main compute path of modern ML workloads lives.

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

The precise conclusion is: Transformer projection, MLP, and attention GEMMs should land on Tensor Cores whenever possible. CUDA Cores still handle address arithmetic, branches, elementwise ops, reductions, softmax, normalization, and glue code. For a compute-bound ML kernel, if the main path is not using Tensor Cores, lower-level scheduling tricks are usually the wrong first priority.

---

## Part 2: CUDA Programming Model & Kernel Engineering

### 2.1 CUDA Extension Build Pathways: load_inline vs setup.py

### Project file structure

```
project/
├── csrc/
│   ├── kernels.cpp     # C++ declarations + pybind11 bindings
│   └── kernels.cu      # CUDA kernel implementation
├── setup.py            # Build the extension with setuptools
└── main.py             # import the compiled module
```

> [!tip]
> - Quick experiment with `load_inline()`, code embedded in a Python string
> - Formal projects use `setup.py` + separated files to facilitate version control and debugging
> - `.cu` files are compiled by nvcc and `.cpp` files are compiled by the system C++ compiler

### Inline compilation method

When compiling a CUDA kernel using PyTorch JIT, you need to divide the code into two parts:

**C++ declarations (cpp_sources)** - Contains only function declarations, compiled by a normal C++ compiler:

```cpp
#include <torch/extension.h>

// Function declaration
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
```

**CUDA source code (cuda_sources)** - Contains kernel definitions and wrapper functions that call the kernel:

```cuda
// Kernel definition
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Wrapper function (must be in the .cu file because it uses the <<<>>> syntax)
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
> `<<<>>>` Kernel startup syntax can only be compiled by nvcc and must be placed in `cuda_sources`, not in `cpp_sources`.

### Compile and load

```python
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name='cuda_kernels',
    cpp_sources=cpp_source,        # declarations only
    cuda_sources=full_cuda_source,  # kernel + wrapper function
    functions=['vector_add'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

# Use the compiled kernel
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = module.vector_add(a, b)
```

### 2.2 Vector Add Operator Implementation and Execution Mapping

#### 1. Hardware Architecture Mapping

NVIDIA GPUs use a hierarchical parallel architecture:

```
GPU
├── SM (Streaming Multiprocessor) × N    # Multiple streaming multiprocessors
│   ├── CUDA Cores × M                   # Each SM has multiple CUDA cores
│   ├── Shared Memory                    # On-chip shared memory (fast)
│   ├── L1 Cache                         # Level-1 cache
│   └── Warp Scheduler                   # Warp scheduler
└── Global Memory (HBM/GDDR)             # Global device memory (slow)
```

**Key concepts**:
- **SM**: independent computing unit that can run multiple thread blocks at the same time
- **Warp**: 32 threads form a warp, which is the smallest unit of actual execution. The threads in the warp execute the same instructions synchronously
- **Shared Memory**: Threads in the same block share, the speed is close to the register
- **Global Memory**: accessible to all threads, but high latency (~400 cycles)

#### 2. CUDA Thread Hierarchy

CUDA organizes threads into a three-layer structure, corresponding to the hardware:

```
Grid
├── Block 0                    # Thread block, mapped to an SM
│   ├── Thread 0..31  (Warp 0) # Threads, mapped to CUDA cores
│   ├── Thread 32..63 (Warp 1)
│   └── ...
├── Block 1
└── ...
```

#### 3. Kernel Code Analysis & Index Computation

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

**`__global__`**: declares that this is a kernel function, called from the CPU and executed on the GPU

**Built-in variables**:

| Variables | Meaning | Example values ​​|
|------|------|--------|
| `blockIdx.x` | The index of the current block in the grid | 0, 1, 2, ... |
| `blockDim.x` | Number of threads in each block | 256 |
| `threadIdx.x` | The index of the current thread in block | 0, 1, ..., 255 |

**Global Index Calculation**:
```
idx = blockIdx.x * blockDim.x + threadIdx.x

Example: blocks=4, threads=256, for a total of 1024 threads
Block 0: idx = 0*256 + 0..255  = 0..255
Block 1: idx = 1*256 + 0..255  = 256..511
Block 2: idx = 2*256 + 0..255  = 512..767
Block 3: idx = 3*256 + 0..255  = 768..1023
```

**Boundary Check** `if (idx < n)`: Because the total number of threads may be greater than the amount of data, out-of-bounds access needs to be prevented

#### 4. Kernel Launch Configuration & Grid Ceil Sizing

```cuda
int threads = 256;
int blocks = (n + threads - 1) / threads;  // Round up
vector_add_kernel<<<blocks, threads>>>(a, b, c, n);
```

**`<<<blocks, threads>>>`**: CUDA-specific kernel startup syntax
- The first parameter: the number of blocks in the grid
- The second parameter: the number of threads in each block

**Round up formula**: `(n + threads - 1) / threads` ensures there are enough threads to cover all data
```
n=1000, threads=256
blocks = (1000 + 255) / 256 = 4
Total threads = 4 * 256 = 1024 >= 1000 ✓
```

#### 5. End-to-End Execution Flow

```
CPU                          GPU
 │                            │
 ├─ Allocate GPU memory ──────►│
 ├─ Copy data to GPU ─────────►│
 ├─ Launch kernel ────────────►├─ Schedule blocks onto SMs
 │                            ├─ Each SM executes warps
 │                            ├─ Threads compute in parallel
 ├─ Wait for completion ◄─────┤
 ├─ Copy results to CPU ◄─────┤
 │                            │
```

> [!tip]
> `threads=256` was chosen because:
> 1. Is a multiple of 32 (warp size) to avoid waste of resources
> 2. Large enough to hide memory latency
> 3. Does not exceed hardware limit (usually 1024)

---

## Part 3: Roofline Performance Modeling & Theoretical Bounds

### 3.1 Motivation & Two-Segment Execution Model

In deep learning systems engineering, we often encounter fundamental performance questions:
- Increasing batch size sometimes scales throughput linearly, but other times has zero effect;
- The exact same model shows vastly different scaling behavior across hardware architectures (e.g., A100 vs H100 vs TPU);
- Certain operators (e.g., LayerNorm, Attention Softmax) consume significant wall-clock time despite small parameter counts, whereas matrix multiplication achieves immense throughput.

**Roofline Analysis** establishes a concise yet rigorous diagnostic framework: by contrasting an algorithm's **Arithmetic Intensity** against the hardware platform's **Critical Intensity (Ridge Point)**, it pinpoints whether an operator is **Compute-Bound** or **Memory-Bound**, identifying the only mathematically viable optimization pathways.

### 3.2 Core Definitions & Physical Parameter Reference

The execution time of any operator is bounded by two physical components:
$$T_{\text{math}} = \frac{\text{FLOPs}}{\text{Accelerator Peak FLOPs/s}}$$
$$T_{\text{comms}} = \frac{\text{Bytes}}{\text{Memory Bandwidth (Bytes/s)}}$$

| Symbol | Physical Meaning | Reference Value (TPU v5e) | Reference Value (H100 SXM) |
| :--- | :--- | :--- | :--- |
| $\pi$ (Peak FLOPs/s) | Peak compute throughput | $1.97 \times 10^{14}$ (BF16 MXU) | $\sim 1.0 \times 10^{15}$ (BF16 Dense Tensor Core) |
| $\beta$ (Bandwidth) | Device memory (HBM) physical bandwidth | $8.2 \times 10^{11}$ Bytes/s (820 GB/s) | $3.35 \times 10^{12}$ Bytes/s (3.35 TB/s) |

#### Arithmetic Intensity
$$I_{\text{algo}} = \frac{\text{FLOPs}}{\text{Bytes}} \quad (\text{FLOPs/Byte})$$
Measures how many floating-point operations can be executed per byte of data transferred across the memory hierarchy.

#### Critical Intensity (Ridge Point)
$$I_c = \frac{\text{Peak FLOPs/s}}{\text{Peak Bandwidth}} = \frac{\pi}{\beta} \quad (\text{FLOPs/Byte})$$

- **TPU v5e MXU:** $I_c = \frac{1.97 \times 10^{14}}{8.1 \times 10^{11}} \approx 243 \text{ FLOPs/Byte}$
- **NVIDIA H100 SXM:** $I_c = \frac{1.0 \times 10^{15}}{3.35 \times 10^{12}} \approx 298 \text{ FLOPs/Byte}$

#### Bottleneck Classification Criteria

| Condition | Operational Regime | Physical Nature & Engineering Action |
| :--- | :--- | :--- |
| $I_{\text{algo}} \ge I_c$ | **Compute-Bound** | Hardware compute pipelines are fully saturated; memory bandwidth has surplus. Priority: Tensor Core utilization and instruction-level parallelism (ILP). |
| $I_{\text{algo}} < I_c$ | **Memory-Bound** | Compute units spend clock cycles stalled waiting for memory operands. Priority: eliminate memory traffic (operator fusion, quantization) and improve data reuse. |

### 3.3 Systematic 5-Step Roofline Methodology

To perform roofline analysis, follow these five systematic steps:

**Step 1: Determine target hardware parameters**
First, query the specifications of the target hardware ($\beta$ and $\pi$, then compute $I_c = \pi/\beta$):

| Hardware | HBM Capacity | HBM Bandwidth $\beta$ | bf16 Throughput $\pi$ | int8 Throughput | Critical Intensity $I_c = \pi/\beta$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TPU v5e | 16 GB  | $8.1 \times 10^{11}$ B/s | $1.97 \times 10^{14}$ | $3.94 \times 10^{14}$ | 243 |
| TPU v5p | 96 GB  | $2.8 \times 10^{12}$ B/s | $4.59 \times 10^{14}$ | $9.18 \times 10^{14}$ | 164 |
| TPU v6e | 32 GB  | $1.6 \times 10^{12}$ B/s | $9.20 \times 10^{14}$ | $1.84 \times 10^{15}$ | 575 |

**Step 2: Compute algorithmic compute work $W$ (FLOPs) and memory traffic $Q$ (Bytes)**
- $W$: total amount of computation (FLOPs)
- $Q$: total amount of data movement (Bytes) = reads + writes across HBM

**Step 3: Compute arithmetic intensity**
$$I_{\text{algo}} = \frac{W}{Q} \quad (\text{FLOPs/Byte})$$

**Step 4: Determine the bottleneck regime and execution lower bound**
$$T_{\text{actual}} = \max\left( \underbrace{\frac{W}{\pi}}_{T_{\text{compute}}}, \underbrace{\frac{Q}{\beta}}_{T_{\text{memory}}} \right)$$
- If $I_{\text{algo}} \ge I_c$: **Compute-bound**, $T_{\text{actual}} = T_{\text{compute}}$
- If $I_{\text{algo}} < I_c$: **Memory-bound**, $T_{\text{actual}} = T_{\text{memory}}$

**Step 5: Compute achieved hardware efficiency**
$$\text{Efficiency} = \frac{\text{Achieved FLOPs/s}}{\pi} = \frac{W / T_{\text{actual}}}{\pi}$$

![[assets/Pasted image 20251216210603.png|Figure 3: Roofline Performance Boundaries and Theoretical Throughput Across Multi-Tier Bandwidths]]
> **Figure 3 Architectural Insights:** Depicts theoretical peak throughput envelopes for algorithms with different arithmetic intensities across varying memory bandwidths (BW1 vs BW2):
> - **Red Region (Dual Memory-Bound):** Under both BW1 and BW2, the algorithm resides on the sloped bandwidth ceiling, leaving peak compute underutilized.
> - **Yellow Region (Single-Tier Bandwidth Bound):** The operator is bandwidth-constrained only under lower bandwidth (BW1); upgrading to higher bandwidth (BW2) transitions the operator across the Ridge Point into the compute ceiling.
> - **Green Region (Fully Compute-Bound):** Arithmetic intensity is high enough to saturate peak hardware FLOPs/s; additional bandwidth provides no execution speedup.

### 3.3 Example: Dot Product

**Problem**: compute `x · y`, where `x, y ∈ bf16[N]`, with output `bf16[1]`

| Item | Calculation | Result |
| --------------- | ---------------------------------------------------- | ------------ |
| Reads | `x` requires $2N$ bytes, `y` requires $2N$ bytes (bf16 = 2 bytes) | $4N$ |
| Writes | Output 1 bf16 scalar | $2$ |
| **Total Bytes $Q$** | $4N + 2$ | $\approx 4N$ |
| FLOPs | $N$ multiplies + $(N-1)$ adds | $\approx 2N$ |

$$I_{\text{dot}} = \frac{W}{Q} = \frac{2N}{4N + 2} \xrightarrow{N \to \infty} \frac{1}{2}$$
For TPU v5e, $I_c = 243$, while $I_{\text{dot}} = 0.5 \ll 243$
**Conclusion**: vector dot product is **always memory-bound**, no matter how large $N$ is. This is because each element is used only once (no data reuse), so arithmetic intensity has an upper bound.
> [!warning]
> This explains why elementwise operations (such as ReLU and LayerNorm) usually need **kernel fusion** for optimization—when run individually, they are almost always memory-bound.
### 3.4 Example: Matrix Multiplication ⭐

**Problem**: compute `C = A @ B`, where `A ∈ bf16[M, K]`, `B ∈ bf16[K, N]`, and output `C ∈ bf16[M, N]`

| Item | Calculation | Result |
| --------------- | ------------------------------------- | ----------- |
| Read A | $M \times K$ bf16 values | $2MK$ bytes |
| Read B | $K \times N$ bf16 values | $2KN$ bytes |
| Write C | $M \times N$ bf16 values | $2MN$ bytes |
| **Total Bytes $Q$** | $2(MK + KN + MN)$ | |
| FLOPs | Each output element needs $K$ multiplies + $K$ adds, across $MN$ outputs | $2MNK$ |

> [!note] Why is it 2MNK?
> Matrix multiplication $C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}$ performs $K$ multiply-adds for each output element.
> One multiply-add = 2 FLOPs, so the total is $2 \times M \times N \times K$ FLOPs.

$$I_{\text{matmul}} = \frac{2MNK}{2(MK + KN + MN)} = \frac{MNK}{MK + KN + MN}$$
**Special-case analysis** (let $M = B$ (batch), $K = D$ (hidden), $N = F$ (output)):

| Case | Condition | Approximate Intensity | Physical Meaning |
|------|------|----------|----------|
| Batched inference | $B \ll D, F$ | $I \approx B$ | batch size determines whether it is compute-bound |
| Square matrix multiply | $M = K = N$ | $I \approx \frac{N}{3}$ | larger dimensions are better |
| GEMV | $N = 1$ | $I \approx 1$ | vector-matrix multiply, almost always memory-bound |

For `bf16[B, D] @ bf16[D, F] → bf16[B, F]` (a typical FFN layer):
When $B \ll D, F$:
$$I \approx \frac{BDF}{DF} = B$$
On TPU v5e, when **Batch size $B > 243$**, matmul becomes compute-bound.

**Common matrix multiplication FLOPs quick reference**:

| Operation | Shape | FLOPs | Notes |
|------|-------|-------|------|
| GEMM | `[M,K] @ [K,N]` | $2MNK$ | General matrix multiplication |
| GEMV | `[M,K] @ [K,1]` | $2MK$ | Matrix-vector multiply, $I \approx 1$ |
| Square matrix multiply | `[N,N] @ [N,N]` | $2N^3$ | $I \approx N/3$ |
| Batch GEMM | `[B,M,K] @ [B,K,N]` | $2BMNK$ | Batched matrix multiplication |

### 3.5 Example: Full Roofline Calculation

**Problem**: analyze the performance of `bf16[256, 4096] @ bf16[4096, 4096]` on TPU v5e.

**Given parameters**:
- $\pi = 1.97 \times 10^{14}$ FLOPs/s (bf16 peak throughput)
- $\beta = 8.1 \times 10^{11}$ B/s (HBM bandwidth)
- $I_c = \pi / \beta = 243$ FLOPs/Byte

**Step 2: Workload calculation**
- $M = 256, K = 4096, N = 4096$
- $W = 2MNK = 2 \times 256 \times 4096 \times 4096 = 8.59 \times 10^9$ FLOPs
- $Q = 2(MK + KN + MN) = 2(256 \times 4096 + 4096 \times 4096 + 256 \times 4096)$
  $= 2(1.05 \times 10^6 + 1.68 \times 10^7 + 1.05 \times 10^6) = 3.77 \times 10^7$ Bytes

**Step 3: Arithmetic intensity**
$$I = \frac{8.59 \times 10^9}{3.77 \times 10^7} = 228 \text{ FLOPs/Byte}$$
**Step 4: Determine the bottleneck**
- $I = 228 < I_c = 243$ → **Memory-bound** (slightly below the critical point)

**Step 5: Compute time and efficiency**
- $T_{\text{compute}} = W / \pi = 8.59 \times 10^9 / 1.97 \times 10^{14} = 43.6 \mu s$
- $T_{\text{memory}} = Q / \beta = 3.77 \times 10^7 / 8.1 \times 10^{11} = 46.5 \mu s$
- $T_{\text{actual}} = \max(43.6, 46.5) = 46.5 \mu s$
- Efficiency = $43.6 / 46.5 = 93.8\%$

**Conclusion**: although it is slightly memory-bound, the efficiency already reaches 94%, which is near-optimal.

### 3.6 Example: Int8 Quantized Matmul

**Problem**: `int8[B, D] @ int8[D, F] → int8[B, F]`

**Change analysis**:

| Item | bf16 version | int8 version | Change |
|------|-----------|-----------|------|
| Data type size | 2 bytes | 1 byte | $\times 0.5$ |
| Total Bytes $Q$ | $2(BD + DF + BF)$ | $BD + DF + BF$ | $\times 0.5$ |
| Peak throughput $\pi$ | $1.97 \times 10^{14}$ | $3.94 \times 10^{14}$ | $\times 2$ |
| FLOPs $W$ | $2BDF$ | $2BDF$ | Unchanged |

**New arithmetic intensity**:
$$I_{\text{int8}} = \frac{2BDF}{BD + DF + BF}$$
When $B \ll D, F$:
$$I_{\text{int8}} \approx \frac{2BDF}{DF} = 2B$$
**New critical intensity**:
$$I_c^{\text{int8}} = \frac{3.94 \times 10^{14}}{8.1 \times 10^{11}} = 486$$
**Compute-bound condition**:
$$2B > 486 \implies B > 243$$
**Conclusion**:
- The critical batch size for Int8 is **still about 243** (the same as bf16!)
- But once it reaches the compute-bound regime, **throughput doubles**

> [!tip]
> The main benefit of quantization is higher throughput in the compute-bound regime, not shifting the critical point.

### 3.7 Example: Mixed Precision (Int8 Weights + BF16 Activations)

**Problem**: `bf16[B, D] @ int8[D, F] → bf16[B, F]`

This scheme is commonly used for inference optimization: weights are quantized to int8, while activations remain in bf16.

**Analysis**:

| Item | Calculation |
|------|------|
| Read activation A | $2BD$ bytes (bf16) |
| Read weight B | $DF$ bytes (int8) |
| Write output C | $2BF$ bytes (bf16) |
| **Total Bytes $Q$** | $2BD + DF + 2BF$ |
| FLOPs | $2BDF$ (still counted in bf16) |

**Arithmetic intensity**:
$$I_{\text{mixed}} = \frac{2BDF}{2BD + DF + 2BF}$$
When $B \ll D, F$ and $D \approx F$:
$$I_{\text{mixed}} \approx \frac{2BDF}{DF} = 2B$$
**Compute-bound condition** (using bf16 throughput $\pi = 1.97 \times 10^{14}$):
$$2B > \frac{1.97 \times 10^{14}}{8.1 \times 10^{11}} = 243 \implies B > 122$$
**Conclusion**: with mixed precision, only **$B > 122$** is needed to become compute-bound, making it easier to reach than pure bf16 ($B > 243$)!

### 3.8 The Impact of Different Memory Hierarchies

TPUs have multiple memory levels with dramatically different bandwidths:

| Memory Type | Bandwidth | Relative to HBM | Typical Use |
|----------|------|----------|----------|
| VMEM (SRAM) | ~18 TB/s | 22× | Computation within a tile |
| HBM | ~0.8 TB/s | 1× | Main storage |
| ICI (inter-chip) | ~0.09 TB/s | 0.1× | Multi-chip communication |
| PCIe | ~0.015 TB/s | 0.02× | Host-device transfer |

**Example**: `int8[B, 4096] @ int8[16384, 4096]`

| Memory Source | Critical Batch Size |
|----------|-----------------|
| HBM | $B > 271$ |
| VMEM | $B > 11$ |

**Conclusion**: if the weights can fit into VMEM, the critical point drops by 25×. This is why **tiling** and **weight caching** are so important.
### 3.9 Batch-Specific Weight Matrices (A Cautionary Example)

**Problem**: suppose each batch element has a different weight matrix:
`int8[B, D] @ int8[B, D, F] → int8[B, F]`

Find the arithmetic intensity.

**Analysis**:

This situation arises in some special scenarios (such as extreme cases of LoRA or per-sample adaptation).

| Item | Standard matmul | Batch-specific weights |
|------|-------------|---------------------|
| Read X | $BD$ | $BD$ |
| Read Y | $DF$ | $\mathbf{BDF}$ |
| Write Z | $BF$ | $BF$ |
| **Total Bytes** | $BD + DF + BF$ | $BD + BDF + BF$ |
| FLOPs | $2BDF$ | $2BDF$ |

**Arithmetic intensity**:
$$I = \frac{2BDF}{BD + BDF + BF}$$
The $BDF$ term dominates the denominator (because $D$ and $F$ are usually large):
$$I \approx \frac{2BDF}{BDF} = 2$$
**Conclusion**:
$$\boxed{I \approx 2 \text{ (constant)}}$$
> [!warning] This is a cautionary example!
>
> An arithmetic intensity that is constant (about 2) means:
> - It is **always memory-bound**, regardless of batch size
> - Each weight element is used only once, with no data reuse
> - Hardware utilization is extremely low: $\text{Efficiency} = \frac{2}{486} \approx 0.4\%$
>
> **Ways to avoid this pattern**:
> - Share weights whenever possible (standard matmul)
> - If different weights are required, consider grouped/tiled reuse
> - Use low-rank adaptation methods such as LoRA

### 3.10 GPU (H100) Roofline Analysis

**Problem**: using the NVIDIA H100 specifications, compute the critical batch size.

**H100 SXM specifications**:

| Parameter | Value | Notes |
|------|------|------|
| bf16 Tensor Core FLOPs | $1.979 \times 10^{15}$ | **with sparsity** |
| Actual bf16 FLOPs | $\sim 1 \times 10^{15}$ | without sparsity (divide by 2) |
| HBM3 bandwidth | 3.35 TB/s = $3.35 \times 10^{12}$ B/s | |
| HBM capacity | 80 GB | |

> [!note] About sparsity
> NVIDIA's advertised Tensor Core FLOPs include 2:4 structured sparsity acceleration.
> In practice, if the model is not sparse, the official number should be divided by 2.

**Critical intensity**:
$$I_c = \frac{\pi}{\beta} = \frac{1 \times 10^{15}}{3.35 \times 10^{12}} = 298 \text{ FLOPs/byte}$$
**Critical batch size** (when $B \ll D, F$):
$$B > I_c \implies \boxed{B > 298}$$
**Comparison with TPU v5e**:

| Hardware | Peak throughput $\pi$ | HBM bandwidth $\beta$ | Critical intensity $I_c$ | Critical B |
|------|----------------|------------------|----------------|--------|
| TPU v5e | $1.97 \times 10^{14}$ | $8.1 \times 10^{11}$ | 243 | ~243 |
| H100 SXM | $1.0 \times 10^{15}$ | $3.35 \times 10^{12}$ | 298 | ~298 |
| **Ratio** | 5× | 4× | 1.2× | ~1.2× |

> [!important] Key finding
> Although the H100 has much higher absolute compute throughput and bandwidth than the TPU v5e, the **critical batch size is almost the same**!
>
> This is because the two devices have similar "compute/bandwidth" ratios (about 240-300 FLOPs/byte).
> This ratio is determined by chip architecture and is a common characteristic of modern AI accelerators.
### 3.4 Profiler Practice & Hardware Roofline Validation

#### 3.4.1 PyTorch Profiler: Chrome Trace & FLOPs Counter

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
    
    # Export Chrome trace for timeline inspection
    prof.export_chrome_trace("torch_trace.json")

torch_roofline(256, 4096, 4096)
```

#### 3.4.2 NVIDIA Nsight Compute (NCU) Hardware Roofline Profiling

```bash
# Collect GPU hardware performance counters & roofline analysis (requires profiling permissions)
ncu --set roofline -o profile ./your_program

# Launch NCU interactive GUI to inspect operator placement on the roofline plot
ncu-ui profile.ncu-rep
```

#### 3.4.3 Vector Add Empirical Roofline (Strictly Memory-Bound)

![[assets/Pasted image 20251223165208.png|Figure 4: BF16 Vector Add Roofline Empirical Distribution on RTX A5000 (Memory-Bound)]]
> **Figure 4 Measurement Analysis:** Across varying array lengths ($n = 1\text{M}$ to $134\text{M}$), vector add data points firmly anchor along the left-hand bandwidth slope (saturating 80%~90% of peak HBM bandwidth). Because its arithmetic intensity ($I \approx 0.17 \text{ FLOPs/Byte}$) sits far below the hardware ridge point ($72.4$), the operator is strictly bound by memory bandwidth; scaling tensor sizes cannot bridge the operator onto the compute plateau.

#### 3.4.4 MatMul Empirical Roofline Evolution & Diagnostic Anomalies

![[assets/Pasted image 20251223164911.png|Figure 5: BF16 MatMul Scaling Transition from Memory-Bound to Compute-Bound & Diagnostic Anomalies]]
> **Figure 5 Diagnostic Anomalies & Systems Insights:**
> 1. **Font Glyphs / Tofu Box Explanation:** The original plot title displays `MatMul Roofline: □□□□ vs □□ (RTX A5000, BF16)` because the benchmarking script ran in a Linux headless environment missing CJK system fonts, rendering Chinese characters ("理论峰值 vs 测量性能") as fallback tofu squares.
> 2. **Why do large GEMM points physically exceed the Roofline ceiling (125%~179%)?**
>    - The blue horizontal ceiling represents the **theoretical peak throughput of FP32 CUDA Cores** (~65 TFLOPs).
>    - However, modern NVIDIA architectures automatically route BF16 GEMM instructions through the specialized **Fourth-Gen Tensor Core pipeline** (with BF16 dense throughput reaching ~130+ TFLOPs on the RTX A5000).
>    - Consequently, once matrix dimensions scale to $1024 \times 1024$ and beyond, empirical throughput naturally surges past the CUDA Core "false ceiling".
>    - **Key Engineering Takeaway:** When performing Roofline validation, ensure theoretical peak $\pi$ strictly corresponds to the actual execution pipeline (Tensor Core vs CUDA Core); otherwise, calculated hardware efficiency will erroneously exceed 100%.

### 3.5 Roofline Classification & Optimization Decision Matrix

<div class="roofline-decision-card">
  <div class="decision-header">
    <h4>Roofline Performance Optimization Decision Matrix</h4>
    <span class="decision-badge">Ridge Point Boundary Demarcation</span>
  </div>
  <div class="decision-grid">
    <div class="decision-branch memory">
      <span class="branch-badge">Memory-Bound Regime</span>
      <div class="branch-title">Arithmetic Intensity AI &lt; Ridge Point</div>
      <code class="branch-formula">Achieved Throughput = AI × Bandwidth &lt; Peak FLOPs</code>
      <ul>
        <li><strong>Primary Bottleneck:</strong> Memory bus bandwidth (HBM / DRAM); compute ALUs spend cycles starved of operands.</li>
        <li><strong>Futile Direction:</strong> Increasing compute core counts or core clocks delivers negligible speedup.</li>
        <li><strong>Core Optimization Tactics:</strong>
          <ul>
            <li><strong>Operator Fusion (Kernel Fusion):</strong> Fuse elementwise, normalization, and activation ops into preceding GEMMs to eliminate roundtrip HBM writes.</li>
            <li><strong>Explicit Tiling:</strong> Stage local tiles into Shared Memory / Registers to maximize intra-block data reuse.</li>
            <li><strong>Precision Quantization:</strong> FP32 $\to$ FP16/BF16 $\to$ INT8/FP4, halving memory bandwidth demand.</li>
            <li><strong>Recomputation:</strong> Trade low-cost compute cycles to avoid large intermediate tensor spills.</li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="decision-branch compute">
      <span class="branch-badge">Compute-Bound Regime</span>
      <div class="branch-title">Arithmetic Intensity AI &ge; Ridge Point</div>
      <code class="branch-formula">Achieved Throughput &le; Peak FLOPs/s</code>
      <ul>
        <li><strong>Primary Bottleneck:</strong> Execution pipeline saturation (ALU / Tensor Core peak); memory bus has surplus bandwidth.</li>
        <li><strong>Futile Direction:</strong> Memory streaming optimizations or wider bandwidth do not accelerate execution.</li>
        <li><strong>Core Optimization Tactics:</strong>
          <ul>
            <li><strong>Leverage Dedicated Tensor Pipelines:</strong> Enforce MMA / WGMMA hardware instructions, unlocking 90%+ of silicon FLOPS.</li>
            <li><strong>Instruction-Level Parallelism (ILP):</strong> Loop unrolling (#pragma unroll) with multiple accumulator registers to hide execution latency.</li>
            <li><strong>Structured Sparsity:</strong> Utilize 2:4 structured sparse matrix multiplication instructions to double throughput.</li>
            <li><strong>Eliminate Warp Divergence:</strong> Align control flow across all 32 lanes to avoid execution serialization.</li>
          </ul>
        </li>
      </ul>
    </div>
  </div>
</div>

#### Diagnostic Interpretation: Position Relative to Roofline Boundaries

- **Point situated to the left of the Ridge Point ($\text{AI} < I_c$):** Strictly Memory-Bound. The limiting resource is memory bandwidth, while execution units wait on operands. Theoretical performance ceiling $= \beta \times \text{AI}$. Adding compute cores is futile; prioritize memory traffic minimization (operator fusion, quantization) and cache reuse.
- **Point situated to the right of the Ridge Point ($\text{AI} > I_c$):** Strictly Compute-Bound. The limiting resource is pipeline issue throughput, with surplus bandwidth. Theoretical ceiling $= \pi$. Upgrading bandwidth provides no gain; prioritize Tensor Core utilization, instruction unrolling, and latency hiding.
- **Point on the ceiling line ($\text{Efficiency} > 80\%$):** Implementation is operating near physical hardware limits; negligible room for micro-optimizations under current AI. Further acceleration requires algorithmic reformulation (e.g., fusion to increase AI) or hardware upgrades.
- **Point substantially below the ceiling line ($\text{Efficiency} < 80\%$):** Hardware is underutilized; actionable optimization targets exist:
  - **Memory-Bound Inefficiencies:** Non-coalesced memory access patterns, Shared Memory bank conflicts, or unaligned data layouts triggering redundant memory transactions.
  - **Compute-Bound Inefficiencies:** Suboptimal occupancy, register spilling to local memory, failing to invoke Tensor Core pipelines, or raw data dependency stalls.
- **Point above the ceiling line (Physically Impossible):** Signifies benchmarking or accounting errors: uncounted L1/L2 cache reuse causing actual HBM traffic to be smaller than theoretical estimates, inaccurate FLOPs formulas, omitted host synchronization (`cudaDeviceSynchronize()`), or selecting an inappropriate hardware peak (e.g., using CUDA Core ceiling for Tensor Core operations).

---

## Part 4: Five Core Principles for GPU Optimization

Summarizing optimization experience across kernels such as transpose, stencil, SpMV, histogram, and compaction, we can extract the following five general principles:

### 4.1 Principle A: Byte Accounting

Before optimizing, you need an accurate estimate of the kernel's total memory traffic. This step determines whether subsequent optimization work is actually targeting the real bottleneck.

Rule of thumb: **sum the bytes of all read and write operations**, and remember that an atomic operation is fundamentally a read-modify-write, so its memory traffic should be counted as 2-3x.

> [!note] A common pitfall
> Cutting FLOPs in half without reducing memory accesses does not improve kernel runtime at all. Worse, reducing computation by introducing extra intermediate arrays can increase memory traffic and actually hurt performance.

**Example: Byte accounting for vector addition**
```cuda
// C[i] = A[i] + B[i], N floats
// Read: A (4N bytes) + B (4N bytes) = 8N bytes
// Write: C (4N bytes)
// Total memory traffic = 12N bytes
// Peak bandwidth 900 GB/s -> theoretical lower bound = 12N / 900G seconds
```

**Example: Byte accounting for Histogram (easy to miscalculate)**
```cuda
// Input: N ints (read 4N bytes)
// Output: bins[] uses atomicAdd
// atomic = read + modify + write -> each update is about 3x4 = 12 bytes
// Total traffic = 4N + 12N = 16N bytes (not the naive 4N + 4N)
```

---

### 4.2 Principle B: Coalescing

The ideal memory access pattern is: **the 32 threads in a warp access a contiguous 128-byte segment** (for `float`, for example).

More concretely:
- lanes within a warp should map to consecutive memory addresses
- the overall access pattern should be as close to sequential streaming as possible

Coalescing is the prerequisite for all other optimizations. If coalescing is not satisfied, the upper limit of effective bandwidth drops dramatically.

**Example: The coalescing issue in matrix transpose**
```cuda
// Bad case: read by column, warp threads access with stride N
out[j][i] = in[i][j];   // in read by row (coalesced) ✓
                        // out written by column (strided) ✗ -> bandwidth utilization drops sharply

// Good case: use shared memory as a staging buffer
tile[threadIdx.y][threadIdx.x] = in[row][col];   // coalesced read
__syncthreads();
out[col][row] = tile[threadIdx.x][threadIdx.y];   // coalesced write
```

**Example: AoS vs SoA**
```cpp
// AoS (Array of Structs) — when a warp reads x, the stride is sizeof(Point)
struct Point { float x, y, z; };
Point pts[N];            // pts[tid].x → stride=12 bytes ✗

// SoA (Struct of Arrays) — when a warp reads x, accesses are contiguous
float px[N], py[N], pz[N];
px[tid]                  // stride=4 bytes, perfectly coalesced ✓
```

---

### 4.3 Principle C: Explicit Reuse (Tiling)

When a kernel has neighborhood structure or data reuse (such as stencil, convolution, or some sparse local operators):

- do not rely on implicit hits in the hardware L1/L2 cache
- use tiling (shared memory or registers) to turn reuse into deterministic behavior

The core idea is: **load data from HBM into SRAM and reuse it multiple times to avoid repeated HBM accesses**.

> [!info] What is a Stencil?
> A stencil is a common computational pattern in which each output element is computed as a weighted sum of **the input element itself and input elements in a fixed neighborhood**. A 1D stencil with radius R means that `out[i]` depends on `in[i-R] ... in[i+R]`, for a total of 2R+1 elements. Typical applications include finite differences (CFD/PDE solvers), image blur/sharpening (2D stencil), audio filtering, and more. Because the input windows of neighboring output points overlap heavily, stencil is a classic use case for tiling.

**Example: 1D stencil — without tiling vs with tiling**
```cuda
// Without tiling: each output point reads 2R+1 neighbors from HBM
// Neighboring threads have heavily overlapping reads -> depends on cache hits, not controllable
out[i] = Σ w[k] * in[i-R+k],  k=0..2R

// With tiling: the block cooperatively loads a tile segment (including halo) into shared memory
__shared__ float tile[BLOCK + 2*R];
tile[threadIdx.x + R] = in[blockStart + threadIdx.x];
if (threadIdx.x < R) {               // load left and right halo
    tile[threadIdx.x] = in[blockStart - R + threadIdx.x];
    tile[BLOCK + R + threadIdx.x] = in[blockStart + BLOCK + threadIdx.x];
}
__syncthreads();
out[i] = Σ w[k] * tile[threadIdx.x + k];  // all hits come from SRAM
```

---

### 4.4 Principle D: Reduce Synchronization & Contention (Sync / Contention)

For memory-bound kernels, the performance bottleneck is often not the bandwidth itself, but rather:
- overly frequent `__syncthreads()` calls, which turn pipeline throughput into serialized waiting
- contention on atomic operations (histogram, scatter), which degrades parallel writes into serialized writes

The "hierarchical privatization" strategy in the previous Histogram lecture is a canonical application of this principle: reduce the scope of contention progressively from global to block to warp, thereby lowering contention overhead.

**Example: Hierarchical privatization for Histogram**
```cuda
// Level 1 — global atomic (maximum contention)
atomicAdd(&global_bins[val], 1);           // all threads contend for the same set of bins

// Level 2 — block-private bins -> final reduction
__shared__ int local_bins[NUM_BINS];       // one copy per block
atomicAdd(&local_bins[val], 1);            // contention shrinks to within the block
__syncthreads();
atomicAdd(&global_bins[tid], local_bins[tid]);  // one-shot reduction

// Level 3 — warp-private bins (registers/shared-memory partitioning)
// contention shrinks further to within 32 threads, with almost no conflicts
```

---

### 4.5 Principle E: Latency Hiding

When high memory latency is unavoidable (for example, the random accesses in SpMV), latency can be hidden in the following ways:
- **increase occupancy**: raise the number of resident warps so that more warps can be scheduled while others are waiting on memory
- **increase ILP (instruction-level parallelism)**: use unrolling and multiple-elements-per-thread strategies so that each thread can issue more load requests concurrently

The grid-stride loop is a general engineering pattern for realizing this principle.

**Example: Grid-stride loop + ILP unrolling**
```cuda
// Basic version: each thread handles one element; occupancy is the only latency-hiding mechanism
for (int i = tid; i < N; i += gridDim.x * blockDim.x)
    out[i] = f(in[i]);

// ILP version: each thread issues multiple loads at once to hide memory latency
for (int i = tid; i < N; i += stride * 4) {
    float a = in[i];
    float b = in[i + stride];
    float c = in[i + stride*2];
    float d = in[i + stride*3];    // 4 loads in flight at the same time
    out[i]            = f(a);
    out[i + stride]   = f(b);
    out[i + stride*2] = f(c);
    out[i + stride*3] = f(d);
}
```

---

## Part 5: Practice Exercises & Review Self-Checks

<details class="exercise">
<summary><span class="q-label">Q1</span> <span class="q-text">Why must CUDA kernel launches be placed in .cu files rather than .cpp files?</span></summary>

CUDA kernel launch syntax `<<<blocks, threads>>>` can only be parsed by the `nvcc` compiler. Standard host C++ compilers (g++, clang) only handle bindings, function declarations, and CPU-side wrappers. True CUDA source code containing `__global__` functions and launch syntax must reside in `.cu` source files to generate PTX and SASS assembly.

</details>

<details class="exercise">
<summary><span class="q-label">Q2</span> <span class="q-text">A thread block has 256 threads. How many warps does it contain, and how are they scheduled?</span></summary>

A warp consists of 32 threads, so 256 threads correspond to $256 / 32 = 8$ warps. The thread block is scheduled atomically as a whole onto a specific physical SM; the 8 warps within the block are then dynamically issued by the 4 warp schedulers inside the SM based on instruction readiness. Tuning must balance having enough blocks to keep all SMs occupied without exhausting registers or Shared Memory.

</details>

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">What are the respective responsibilities of CPU code, CUDA kernels, and PyTorch bindings?</span></summary>

- **CPU Host Code**: Allocates tensor memory, verifies Shape/Dtype/Device constraints, computes Grid/Block launch configurations, and invokes launchers.
- **CUDA Kernels (.cu)**: Executes fine-grained parallel computation on the GPU, defining how threads and warps cooperate to access Shared/Global memory.
- **PyTorch Binding (pybind11 / cpp_extension)**: Bridges Python tensor objects to underlying C++/CUDA functions across the ABI boundary. Never conflate these: `pybind` performs no computation, and kernels never manage Python runtime objects.

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">Why is boundary checking almost always necessary in CUDA kernels?</span></summary>

Kernel launches round up the grid size `ceil(N / BlockSize)`. The total allocated threads often exceed the array length $N$. Without `if (idx < n)`, trailing threads in the final block perform illegal out-of-bounds reads/writes. While small unit tests might silently pass without page faults, in real production runs this causes memory corruption or silent numerical defects.

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">Given n=10000 and threads=256, how do you calculate block count? Why not integer division?</span></summary>

You must use the ceiling formula `(n + threads - 1) / threads`, yielding `(10000 + 255) / 256 = 40` blocks. If you use floor division `10000 / 256 = 39`, only $39 \times 256 = 9984$ threads are launched, leaving the final 16 elements completely unprocessed and corrupting the output tensor.

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">What is the correct diagnostic roadmap when a CUDA hello-world kernel runs unexpectedly slow?</span></summary>

Do not start by questioning algorithmic complexity. Novice kernel slowdowns usually stem from:
1. Launch configuration is too small, leaving SMs severely underutilized;
2. Unintended implicit CPU-GPU data transfers (DtoH / HtoD) inside loops;
3. Omitted `cudaDeviceSynchronize()` during timing, capturing cold-start or asynchronous latency;
4. Hidden type conversions due to mismatched dtypes. First confirm functional correctness, synchronization, and memory flow before tuning hardware microarchitecture.

</details>

<details class="exercise">
<summary><span class="q-label">Q7</span> <span class="q-text">Why does modern LLM GEMM optimization prioritize Tensor Cores above all else?</span></summary>

Large language model compute is heavily dominated by dense matrix multiplications (Attention projections, MLP layers, MoE experts, FlashAttention $QK^T$ and $PV$). On NVIDIA H100 SXM, BF16 Tensor Core peak throughput (~1000 TFLOPs) is roughly 15× higher than FP32 CUDA Core peak (~67 TFLOPs). If the core compute does not land on Tensor Cores, you lose an order of magnitude of hardware capability immediately.

</details>

<details class="exercise">
<summary><span class="q-label">Q8</span> <span class="q-text">What causes high SM Occupancy but very low Tensor Core utilization?</span></summary>

High occupancy only means many warps reside on the SM; it does not ensure they are issuing high-throughput instructions. Root causes include:
1. The kernel fails to lower to `mma` or `wgmma` assembly instructions, falling back to scalar FMA;
2. Tile sizes are too small, causing pipeline bubbles between Tensor Core issues;
3. Memory starvation: Global or Shared Memory cannot feed data fast enough, causing warps to stall on scoreboard dependencies;
4. Workloads dominated by non-GEMM logic like LayerNorm, Softmax, or reductions.

</details>

<details class="exercise">
<summary><span class="q-label">Q9</span> <span class="q-text">What is the physical and logical mapping among SM, Warp, and Thread Block?</span></summary>

- **Thread Block**: The programmer-defined logical scheduling unit, assigned atomically to a specific SM for its entire lifetime.
- **Warp**: The hardware instruction issue and execution atom (32 threads lock-stepping the same instruction). Blocks are subdivided into warps.
- **SM (Streaming Multiprocessor)**: The independent physical compute unit containing register files, Shared Memory, warp schedulers, and execution units. Multiple warps from one or more blocks reside concurrently on an SM.

</details>

<details class="exercise">
<summary><span class="q-label">Q10</span> <span class="q-text">Why is high occupancy not synonymous with high kernel performance?</span></summary>

Occupancy is merely a mechanism to hide latency, not an optimization objective. Once enough warps reside on an SM to hide memory and pipeline latency (typically 40%~60% theoretical occupancy is sufficient), further occupancy increases yield zero speedup. Striving for 100% occupancy often requires restricting registers per thread, triggering catastrophic register spilling into local memory and aggravating Shared Memory pressure.

</details>

<details class="exercise">
<summary><span class="q-label">Q11</span> <span class="q-text">In Transformer architectures, what is the exact operational division between Tensor Cores and CUDA Cores?</span></summary>

- **Tensor Cores**: Heavy matrix multiplication operators with high arithmetic intensity: Q/K/V projections, MLP Gate/Up/Down projections, Attention score computation and value aggregation.
- **CUDA Cores**: Scalar, control-flow, and memory-bound operations: RoPE positional embeddings, Softmax, RMSNorm/LayerNorm, SwiGLU activations, indexing, and memory layout transformations. High-performance kernels use Tensor Cores for compute and CUDA Cores as glue logic.

</details>

<details class="exercise">
<summary><span class="q-label">Q12</span> <span class="q-text">What is the systematic debugging sequence for a theoretically compute-bound GEMM with poor Tensor Core utilization?</span></summary>

Recommended sequence:
1. **Instruction Verification**: Check SASS or NCU to verify that `HMMA` / `WGMMA` instructions are actually issued;
2. **Alignment & Layout**: Ensure K-dimension is 16-byte aligned and memory layout supports 128-bit vectorized loads;
3. **Tiling Dimensions**: Verify Block Tile, Warp Tile, and Thread Tile match architecture recommendations;
4. **Asynchronous Double-Buffering**: Confirm Shared Memory staging is not bottlenecking compute, leveraging `cp.async` on modern architectures;
5. **Register Pressure**: Check for register spills to local memory.

</details>

<details class="exercise">
<summary><span class="q-label">Q13</span> <span class="q-text">Given FLOPs, Bytes, peak compute, and memory bandwidth, how do you formally determine the bottleneck regime?</span></summary>

1. Compute arithmetic intensity: $I_{\text{algo}} = \frac{\text{FLOPs}}{\text{Bytes}}$;
2. Compute hardware critical intensity (Ridge Point): $I_c = \frac{\text{Peak FLOPs/s}}{\text{Memory Bandwidth (Bytes/s)}}$;
3. **Decision Rule**:
   - If $I_{\text{algo}} < I_c$: The operator is **Memory-Bound**, bounded by memory bus bandwidth, with theoretical runtime $\frac{\text{Bytes}}{\beta}$;
   - If $I_{\text{algo}} \ge I_c$: The operator is **Compute-Bound**, bounded by peak pipeline throughput, with theoretical runtime $\frac{\text{FLOPs}}{\pi}$. For BF16 GEMM, $\pi$ must be Tensor Core peak.

</details>

<details class="exercise">
<summary><span class="q-label">Q14</span> <span class="q-text">Why does achieved TFLOPs alone fail to characterize kernel implementation quality?</span></summary>

Memory-bound operators (e.g. Elementwise Add, LayerNorm, Softmax) have inherently low arithmetic intensity; their theoretical ceiling is governed by memory bandwidth, resulting in achieved TFLOPs that are naturally only a fraction of hardware compute peak. A reduction achieving 90% of theoretical HBM bandwidth is near-optimal even if it only hits 2 TFLOPs. Conversely, a GEMM hitting 50 TFLOPs on an H100 with 1000 TFLOPs capability represents an abysmal 5% utilization. Performance must be evaluated relative to the Roofline boundary and memory bandwidth utilization.

</details>

<details class="exercise">
<summary><span class="q-label">Q15</span> <span class="q-text">What are the physical definitions of FLOPs (numerator) and Bytes (denominator) in Roofline analysis?</span></summary>

- **FLOPs (Numerator)**: The algorithmic, mathematically required floating-point operations (typically 1 FMA = 2 FLOPs), independent of implementation quirks.
- **Bytes (Denominator)**: The total physical data traffic that must cross the target memory boundary (typically off-chip HBM/DRAM) during execution. Arithmetic intensity indicates how many floating-point calculations are supported per byte transferred from memory.

</details>

<details class="exercise">
<summary><span class="q-label">Q16</span> <span class="q-text">Why does the same operator transition between bottleneck regimes as problem scale grows?</span></summary>

Consider matrix multiplication: when Batch Size is 1 (GEMV during autoregressive generation), each weight element loaded from HBM is used once with no cross-batch reuse, giving $I \approx 1 \text{ FLOP/Byte}$ (strictly Memory-Bound). When Batch Size scales to hundreds or thousands (Prompt prefill or training), each weight tile staged into on-chip SRAM is reused across all batch elements, scaling arithmetic intensity linearly until it surpasses the Ridge Point into the Compute-Bound regime.

</details>

<details class="exercise">
<summary><span class="q-label">Q17</span> <span class="q-text">What are the common causes of benchmarked data points exceeding 100% Roofline efficiency?</span></summary>

Common causes include:
1. **Selecting the wrong hardware peak baseline**: e.g. plotting FP32 CUDA Core peak while running BF16 code that executes on higher-throughput Tensor Cores;
2. **Unaccounted on-chip cache reuse**: Estimating theoretical bytes from tensor sizes while actual HBM traffic is greatly reduced due to L2/SRAM hits;
3. **Asynchronous timing errors**: Forgetting `cudaDeviceSynchronize()`, measuring only CPU launch latency;
4. **Overestimated FLOP counts**: Counting operations eliminated by compiler dead code elimination (DCE).

</details>

<details class="exercise">
<summary><span class="q-label">Q18</span> <span class="q-text">What are the primary optimization instincts for Memory-Bound vs Compute-Bound operators?</span></summary>

- **Memory-Bound Operators (Goal: Minimize memory traffic and maximize bus saturation)**:
  1. Operator Fusion (Kernel Fusion) to keep intermediate activations on-chip and eliminate DRAM roundtrips;
  2. Precision Quantization (INT8 / FP8 / FP4) to shrink byte traffic;
  3. Enforce memory coalescing and eliminate Shared Memory bank conflicts.
- **Compute-Bound Operators (Goal: Saturate compute pipelines and eliminate execution stalls)**:
  1. Switch to dedicated hardware acceleration pathways (Tensor Core MMA / WGMMA);
  2. Loop unrolling (`#pragma unroll`) and multiple accumulator registers to boost ILP;
  3. Leverage 2:4 structured sparsity;
  4. Eliminate warp divergence to keep all 32 lanes synchronous without idle cycles.

</details>

---

## References & Further Reading

1. **Google DeepMind - *How To Scale Your Model*** (Jacob Austin, Sholto Douglas, Roy Frostig, et al., 2025): [Part 1: All About Rooflines](https://jax-ml.github.io/scaling-book).
   > The Roofline methodology, TPU parameter references, arithmetic intensity derivations for Dot Product and GEMM, critical batch size calculations under quantization and mixed precision, and the batch-specific weight counter-example in this chapter are derived from this work.
2. **Williams, S., Waterman, A., & Patterson, D. (2009)**. *Roofline: an insightful visual performance model for multicore architectures*. Communications of the ACM, 52(4), 65-76.
3. **NVIDIA Corporation (2022)**. *NVIDIA H100 Tensor Core GPU Architecture Whitepaper*.
4. **NVIDIA Corporation (2024)**. *CUDA C++ Programming Guide*.
5. **PyTorch Team (2024)**. *Custom C++ and CUDA Extensions Tutorial*.
