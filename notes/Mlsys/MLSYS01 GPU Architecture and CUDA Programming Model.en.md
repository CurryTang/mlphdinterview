# 01 · GPU Architecture, CUDA Programming Model & Roofline Performance Analysis

This chapter serves as the fundamental foundation of MLSYS operator engineering and low-level systems. It establishes a complete mental model bridging GPU physical hardware, Streaming Multiprocessor (SM) execution pipelines, the CUDA programming and execution model, Roofline performance modeling, and the five core engineering principles of memory-bound optimization.

---

## Part 1: GPU Hardware Architecture & Core Components

## More GPU

### Overview

![[assets/Pasted image 20251222135239.png]]
**Figure 1: Abstract diagram of the overall NVIDIA H100/B100 GPU architecture.** It shows the GPU's hierarchical memory and compute structure: multiple streaming multiprocessors (SM 0, SM 1, ... SM N-1) are arranged in parallel, and each SM contains 4 Tensor Cores (responsible for matrix multiplication and contributing most of the compute throughput, analogous to the TPU MXU) and 4 Warp Schedulers (SIMD vector units containing 32 lanes, i.e., "CUDA Cores"; all lanes within the same warp must execute the same operation). Each SM has 256KB of L1 Cache/SMEM (shared memory that can be controlled by the programmer, similar to TPU VMEM but smaller). All SMs share a 50MB L2 Cache (automatically managed by hardware to provide faster bandwidth) and the underlying HBM high-bandwidth memory (80GB on H100, 192GB on B100), which stores model parameters, activations, and optimizer states.

![[assets/Pasted image 20251222135741.png]]
**Figure 2: Detailed internal architecture of a single NVIDIA H100 SM (streaming multiprocessor).** Each SM contains 4 processing blocks that share an L1 instruction cache and a 256KB L1 data cache/shared memory. Each processing block contains an L0 instruction cache, a Warp Scheduler (scheduling 32 threads per cycle), a Dispatch Unit, a 16384×32-bit register file, and many compute units—16 INT32 units, 16 FP32 units, 8 FP64 units, 1 fourth-generation Tensor Core, LD/ST (load/store) units, and an SFU (special function unit). The bottom also includes a Tensor Memory Accelerator and Tex (texture units). This design enables the H100 to efficiently execute large-scale matrix operations and deep learning workloads in parallel.


### Components

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

## Flowchart

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
│  │          │ Which warp is ready?                         │ │
│  │          ▼                                              │ │
│  │   ┌──────────────┐                                      │ │
│  │   │Warp Scheduler│ ──selects 1-2 eligible warps         │ │
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

| Component | Responsibility | Analogy |
| ------------------ | ---------------------- | -------------- |
| **Warp Scheduler** | Decides "who executes," checks dependencies and hazards, and chooses a policy | Dispatcher: chooses the next player to send in |
| **Dispatch Unit**  | Decides "how to execute," reads operands, selects execution units, and issues instructions | Coordinator: sends the player to the correct lane |

Cycle 1:  Warp_A: LD r1, [addr]     # initiates memory read, must wait ~400 cycles
Cycle 2:  Warp_B: ADD r2, r3, r4    # switch to B
Cycle 3:  Warp_C: MUL r5, r6, r7    # switch to C
...
Cycle 400: Warp_A: (memory returns)       # A's data has arrived
Cycle 401: Warp_A: ADD r8, r1, r9   # A resumes execution

### Importance of Tensor Cores

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

## Q1 Basic environment setup and hello world kernel

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

## Q2 Understand hello world kernel

### GPU Hardware Architecture Intro

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

### CUDA Programming Model

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

### Analysis

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

### kernel startup

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

### Execution process

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

## 1. Motivation

In deep learning, we often run into the following confusion:
- Increasing batch size sometimes speeds things up, but sometimes has no effect
- The same model can behave very differently on different hardware
- Some operators (such as attention) are especially slow, while matrix multiplication is very fast
**Roofline analysis** provides a concise framework for answering these questions: it tells you whether the current bottleneck is **compute** or **bandwidth**, and how to optimize for it.
## 2. Core Definitions

Any computation can be decomposed into two time components:
$$T_{\text{math}} = \frac{\text{FLOPs}}{\text{Accelerator FLOPs/s}}$$
$$T_{\text{comms}} = \frac{\text{Bytes}}{\text{Bandwidth (Bytes/s)}}$$

| Symbol    | Meaning | Example (TPU v5e) |
| --------- | ------ | ---------------------------- |
| FLOPs/s   | Peak chip compute throughput | $1.97 \times 10^{14}$ (bf16) |
| Bandwidth | HBM bandwidth | $8.2 \times 10^{11}$ bytes/s |

### Arithmetic Intensity
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}$$
This is the core concept in roofline analysis: **how many floating-point operations can be performed per byte of data moved**.
### Critical Intensity
$$\text{Critical Intensity} = \frac{\text{Peak FLOPs/s}}{\text{Peak Bandwidth}}$$
For the TPU v5e MXU:
$$\frac{1.97 \times 10^{14}}{8.2 \times 10^{11}} \approx 240 \text{ FLOPs/byte}$$
### 2.4 Compute-bound vs Memory-bound

| Condition | Regime | Meaning |
| --------------------------------------------------------------- | -------------------------- | ------------ |
| $\text{Intensity}_{\text{algo}} > \text{Intensity}_{\text{hw}}$ | **Compute-bound**          | Compute is fully utilized ✓ |
| $\text{Intensity}_{\text{algo}} < \text{Intensity}_{\text{hw}}$ | **Memory/Bandwidth-bound** | Compute waits on data and is wasted ✗ |

## 3. Methodology and Examples

### 3.1 A Systematic Method for Roofline Analysis

To perform roofline analysis, follow these steps:
**Step 1: Determine hardware parameters**
First, look up the specifications of the target hardware:

| Hardware | HBM Capacity | HBM Bandwidth $\beta$ | bf16 Throughput $\pi$ | int8 Throughput | Critical Intensity $I_c = \pi/\beta$ |
| ------- | ------ | ------------------------ | --------------------- | --------------------- | ---------------------- |
| TPU v5e | 16 GB  | $8.1 \times 10^{11}$ B/s | $1.97 \times 10^{14}$ | $3.94 \times 10^{14}$ | 243                    |
| TPU v5p | 96 GB  | $2.8 \times 10^{12}$ B/s | $4.59 \times 10^{14}$ | $9.18 \times 10^{14}$ | 164                    |
| TPU v6e | 32 GB  | $1.6 \times 10^{12}$ B/s | $9.20 \times 10^{14}$ | $1.84 \times 10^{15}$ | 575                    |

**Step 2: Compute the algorithm's FLOPs and Bytes**

For a given algorithm, compute separately:
- $W$: total amount of computation (FLOPs)
- $Q$: total amount of data movement (Bytes) = reads + writes

**Step 3: Compute arithmetic intensity**
$$I_{\text{algo}} = \frac{W}{Q} \quad \text{(FLOPs/Byte)}$$
**Step 4: Determine the bottleneck type**

Compare $I_{\text{algo}}$ with $I_c$:
$$T_{\text{actual}} = \max\left( \underbrace{\frac{W}{\pi}}_{T_{\text{compute}}}, \underbrace{\frac{Q}{\beta}}_{T_{\text{memory}}} \right)$$
- If $I_{\text{algo}} > I_c$: **Compute-bound**, $T_{\text{actual}} = T_{\text{compute}}$
- If $I_{\text{algo}} < I_c$: **Memory-bound**, $T_{\text{actual}} = T_{\text{memory}}$

**Step 5: Compute hardware efficiency**
$$\text{Efficiency} = \frac{\text{Achieved FLOPs/s}}{\pi} = \frac{W / T_{\text{actual}}}{\pi}$$
### 3.2 Understanding the Roofline Plot
**Physical meaning of the two regions**:
- **Sloped region (Memory-bound)**: data movement is the bottleneck; compute units are "waiting for data"
  - Actual throughput = $I_{\text{algo}} \times \beta$ (grows linearly with intensity)
- **Flat region (Compute-bound)**: computation is the bottleneck; peak throughput has been reached
  - Actual throughput = $\pi$ (no longer increases)

![[assets/Pasted image 20251216210603.png]]
> *This figure shows two algorithms with different arithmetic intensities (Algorithm 1 and Algorithm 2) and their theoretical peak throughput under different bandwidths (BW1 and BW2). The red region indicates that the algorithm is bandwidth-limited under both bandwidth settings, leaving part of the hardware peak FLOPs/s unused. The yellow region indicates that the algorithm is bandwidth-limited only under the lower bandwidth (BW1). The green region indicates that the algorithm is compute-limited under all bandwidth settings. At this point, the accelerator's peak FLOPs/s is fully utilized, and neither increasing bandwidth nor increasing arithmetic intensity provides further benefit.*

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
## 4. Practice: Code and Tools

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
    
    # Export Chrome trace
    prof.export_chrome_trace("torch_trace.json")

torch_roofline(256, 4096, 4096)
```

### 4.2 NVIDIA Nsight Analysis (requires root privileges)

```bash
# Collect roofline data
ncu --set roofline -o profile ./your_program

# View report
ncu-ui profile.ncu-rep
```

### 4.3 Analyze the Roofline of the hello world Kernel / Matmul Kernel


![[assets/Pasted image 20251223165208.png]]
All ops are at the same position.


![[assets/Pasted image 20251223164911.png]]

This is the roofline curve for matmul. You can see that as the scale increases, it gradually transitions from memory-bound to compute-bound (why does it end up on the line here? Because this figure is actually wrong: it is a CUDA Core plot, but bf16 matmul uses Tensor Cores!)
### 5 Summary

![[assets/Pasted image 20251223105527.png]]

* Position of a point relative to the Ridge Point
	* Point to the left of the Ridge Point (AI < Ridge Point):
		* The algorithm is in the Memory-Bound regime. The performance bottleneck is memory bandwidth, and the compute units are waiting for data. The theoretical maximum performance = bandwidth × AI. In this case, increasing compute capability does not help, because data cannot be supplied fast enough.
		* Optimization direction: reduce memory accesses (operator fusion, quantization, sparsification) or improve data reuse (change the algorithm).
	* Point to the right of the Ridge Point (AI > Ridge Point):
		* The algorithm is in the Compute-Bound regime. The performance bottleneck is compute capability, and memory bandwidth has headroom. The theoretical maximum performance = peak compute throughput. In this case, increasing memory bandwidth does not help, because computation cannot keep up.
		* Optimization direction: use more efficient compute instructions (Tensor Core), improve parallelism, and reduce instruction dependencies.
* Position of a point relative to the Roofline
	* Point on the line (efficiency > 80%):
		* The implementation is already close to the hardware limit, leaving almost no room for optimization at the current AI. If you still want higher performance, you must change the algorithm itself to increase AI (for example through operator fusion), or switch to stronger hardware.
	* Point below the line (efficiency < 80%): the implementation does not fully utilize the hardware, so there is room for optimization. You need to diagnose the specific reason.
		* If it is in the Memory-Bound region and efficiency is low, possible causes include: non-coalesced memory accesses, low cache hit rate, bank conflicts, or data alignment issues.
		* If it is in the Compute-Bound region and efficiency is low, possible causes include: insufficient occupancy, register spilling, not using Tensor Cores, or instruction dependencies causing pipeline stalls.
	* Point above the line: theoretically impossible. If measurements show a point above the roofline, then either the measurement is wrong or the AI calculation is wrong. Common causes include: not accounting for cache effects so actual memory traffic is smaller than the theoretical value, missing FLOPs in the count, or inaccurate timing.

---

## Part 4: Five Core Principles for GPU Optimization

## 2. Five Core Principles for Memory-Bound Optimization

Summarizing optimization experience across kernels such as transpose, stencil, SpMV, histogram, and compaction, we can extract the following five general principles:

### Principle A: Byte Accounting

Before optimizing, you need an accurate estimate of the kernel's total memory traffic. This step determines whether subsequent optimization work is actually targeting the real bottleneck.

Rule of thumb: **sum the bytes of all read and write operations**, and remember that an atomic operation is fundamentally a read-modify-write, so its memory traffic should be counted as 2-3x.

> [!note] A common pitfall
> Cutting FLOPs in half without reducing memory accesses does not improve kernel runtime at all. Worse, reducing computation by introducing extra intermediate arrays can increase memory traffic and actually hurt performance.

**Example: Byte accounting for vector addition**
```
// C[i] = A[i] + B[i], N floats
// Read: A (4N bytes) + B (4N bytes) = 8N bytes
// Write: C (4N bytes)
// Total memory traffic = 12N bytes
// Peak bandwidth 900 GB/s -> theoretical lower bound = 12N / 900G seconds
```

**Example: Byte accounting for Histogram (easy to miscalculate)**
```
// Input: N ints (read 4N bytes)
// Output: bins[] uses atomicAdd
// atomic = read + modify + write -> each update is about 3x4 = 12 bytes
// Total traffic = 4N + 12N = 16N bytes (not the naive 4N + 4N)
```

---

### Principle B: Coalescing

The ideal memory access pattern is: **the 32 threads in a warp access a contiguous 128-byte segment** (for `float`, for example).

More concretely:
- lanes within a warp should map to consecutive memory addresses
- the overall access pattern should be as close to sequential streaming as possible

Coalescing is the prerequisite for all other optimizations. If coalescing is not satisfied, the upper limit of effective bandwidth drops dramatically.

**Example: The coalescing issue in matrix transpose**
```
// Bad case: read by column, warp threads access with stride N
out[j][i] = in[i][j]   // in read by row (coalesced) ✓
                        // out written by column (strided) ✗ -> bandwidth utilization drops sharply

// Good case: use shared memory as a staging buffer
tile[threadIdx.y][threadIdx.x] = in[row][col]   // coalesced read
__syncthreads()
out[col][row] = tile[threadIdx.x][threadIdx.y]   // coalesced write
```

**Example: AoS vs SoA**
```
// AoS (Array of Structs) — when a warp reads x, the stride is sizeof(Point)
struct Point { float x, y, z; };
Point pts[N];            // pts[tid].x → stride=12 bytes ✗

// SoA (Struct of Arrays) — when a warp reads x, accesses are contiguous
float px[N], py[N], pz[N];
px[tid]                  // stride=4 bytes, perfectly coalesced ✓
```

---

### Principle C: Explicit Reuse (Tiling)

When a kernel has neighborhood structure or data reuse (such as stencil, convolution, or some sparse local operators):

- do not rely on implicit hits in the hardware L1/L2 cache
- use tiling (shared memory or registers) to turn reuse into deterministic behavior

The core idea is: **load data from HBM into SRAM and reuse it multiple times to avoid repeated HBM accesses**.

> [!info] What is a Stencil?
> A stencil is a common computational pattern in which each output element is computed as a weighted sum of **the input element itself and input elements in a fixed neighborhood**. A 1D stencil with radius R means that `out[i]` depends on `in[i-R] ... in[i+R]`, for a total of 2R+1 elements. Typical applications include finite differences (CFD/PDE solvers), image blur/sharpening (2D stencil), audio filtering, and more. Because the input windows of neighboring output points overlap heavily, stencil is a classic use case for tiling.

**Example: 1D stencil — without tiling vs with tiling**
```
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

### Principle D: Reduce Synchronization and Contention (Sync/Contention)

For memory-bound kernels, the performance bottleneck is often not the bandwidth itself, but rather:
- overly frequent `__syncthreads()` calls, which turn pipeline throughput into serialized waiting
- contention on atomic operations (histogram, scatter), which degrades parallel writes into serialized writes

The "hierarchical privatization" strategy in the previous Histogram lecture is a canonical application of this principle: reduce the scope of contention progressively from global to block to warp, thereby lowering contention overhead.

**Example: Hierarchical privatization for Histogram**
```
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

### Principle E: Latency Hiding

When high memory latency is unavoidable (for example, the random accesses in SpMV), latency can be hidden in the following ways:
- **increase occupancy**: raise the number of resident warps so that more warps can be scheduled while others are waiting on memory
- **increase ILP (instruction-level parallelism)**: use unrolling and multiple-elements-per-thread strategies so that each thread can issue more load requests concurrently

The grid-stride loop is a general engineering pattern for realizing this principle.

**Example: Grid-stride loop + ILP unrolling**
```
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

---

## Part 5: Practice Exercises & Review Self-Checks

### Module A: CUDA Environment & Extension Boundaries

<details class="exercise">
<summary><span class="q-label">Exercise 1</span> <span class="q-text">Why must CUDA kernel launches be placed in .cu files rather than .cpp files?</span></summary>

CUDA kernel launch syntax `<<<blocks, threads>>>` can only be parsed by the `nvcc` compiler. Standard host C++ compilers only handle bindings, function declarations, and CPU-side wrappers. True CUDA source code containing `__global__` functions and launch syntax must reside in `.cu` source files.

</details>

<details class="exercise">
<summary><span class="q-label">Exercise 2</span> <span class="q-text">A thread block has 256 threads. How many warps does it contain?</span></summary>

A warp consists of 32 threads, so 256 threads correspond to 8 warps. The thread block is scheduled as a whole onto an SM, and its warps are issued by the warp schedulers. Performance tuning requires checking whether block count provides sufficient grid parallelism and whether each block uses too many registers or shared memory.

</details>

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">What are the respective responsibilities of CPU code, CUDA kernels, and PyTorch bindings?</span></summary>

CPU code manages tensor allocation, validates shapes/dtypes/devices, and calls launchers; CUDA kernels define thread-level execution logic on the GPU; PyTorch C++ bindings expose CUDA operations to Python. Never conflate these layers in interviews: `pybind` performs no computation, and kernels never manage Python APIs.

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">Why is boundary checking almost always necessary in CUDA kernels?</span></summary>

Kernel launches round up the grid size to multiples of the block size `(n + threads - 1) / threads`. The total number of threads launched often exceeds the array length. Without `if (idx < n)`, the trailing threads in the final block would access out-of-bounds memory, leading to memory corruption or silent nondeterministic failures.

</details>

### Module B: SM Architecture & Execution Scheduling

<details class="exercise">
<summary><span class="q-label">Exercise 1</span> <span class="q-text">Why does GEMM optimization focus on Tensor Cores first?</span></summary>

Modern deep learning training and inference workloads are dominated by matrix multiplications: attention projections, MLPs, MoE experts, QK, and PV. On H100, dense BF16 Tensor Core peak throughput (~990 TFLOPs) dwarfs FP32 CUDA Core peak (~60-66 TFLOPs). If compute does not land on Tensor Cores, memory bandwidth, layout, alignment, or lowering must be investigated first.

</details>

<details class="exercise">
<summary><span class="q-label">Exercise 2</span> <span class="q-text">What causes high occupancy but low Tensor Core utilization?</span></summary>

High occupancy only means many warps reside on the SM, not that they issue high-throughput instructions. Low Tensor Core utilization stems from not using `mma`/`wgmma` instructions, tile shape mismatches, memory starvation, register spilling, excessive synchronization, or non-GEMM workloads (e.g. softmax or reductions).

</details>

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">What is the relationship between SM, Warp, and Thread Block?</span></summary>

A thread block is the programmer's launch unit assigned to an SM; threads within a block are grouped into 32-thread warps; warps are the physical units issued by the hardware schedulers. Block count dictates global parallelism, warp count enables latency hiding, and SM resources determine occupancy.

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">Why does high occupancy not necessarily equal high kernel performance?</span></summary>

Occupancy only measures resident warps, not useful instruction issue rates. A kernel can still be stalled by memory bandwidth, register spills, shared memory bank conflicts, thread barriers, or execution on low-throughput scalar paths. True evaluation requires inspecting eligible warps, stall reasons, memory throughput, and tensor pipe utilization.

</details>

### Module C: Roofline Analysis & Bottleneck Identification

<details class="exercise">
<summary><span class="q-label">Exercise 1</span> <span class="q-text">Given FLOPs, Bytes, peak FLOPs, and memory bandwidth, how do you diagnose the performance bottleneck?</span></summary>

Compute arithmetic intensity $I = \text{FLOPs} / \text{Bytes}$ and critical intensity $I_c = \pi / \beta$. If $I < I_c$, the kernel is memory-bound; if $I > I_c$, it is compute-bound. For BF16 GEMM, peak FLOPs must reflect Tensor Core throughput.

</details>

<details class="exercise">
<summary><span class="q-label">Exercise 2</span> <span class="q-text">Why can evaluating only achieved TFLOPs lead to incorrect conclusions?</span></summary>

Memory-bound kernels naturally attain low TFLOPs because their ceiling is limited by memory bandwidth. Achieving a fraction of peak compute while saturating 90%+ of memory bandwidth represents near-optimal performance for a memory-bound operator.

</details>

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">What do the numerator and denominator represent in Arithmetic Intensity?</span></summary>

The numerator is the total floating-point operations executed; the denominator is the total bytes transferred across the targeted memory hierarchy (typically HBM). Higher intensity means more computation per transferred byte.

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">Why does a kernel transition from memory-bound to compute-bound as tensor dimensions scale?</span></summary>

Scaling tensor dimensions increases data reuse. In GEMM, larger $M, N, K$ allow tiled data to be reused across more multiply-accumulate operations, raising arithmetic intensity past the hardware ridge point.

</details>
