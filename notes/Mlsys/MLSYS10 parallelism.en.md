# MLSYS10 · Distributed Training Parallelism

> [!info] Overview
> This tutorial provides a detailed introduction to common parallel training paradigms in deep learning, including data parallelism, fully sharded data parallelism (FSDP), tensor parallelism, and pipeline parallelism. The material is adapted from Google DeepMind's Scaling Book and explained together with GPU hardware characteristics and practical implementations in the Hugging Face Picotron framework.

---

## Table of Contents

1. [[#1. Introduction and Background]]
2. [[#2. GPU Hardware Fundamentals and Communication Primitives]]
3. [[#3. Sharded Matrices and Matrix Multiplication]]
4. [[#4. Data Parallelism]]
5. [[#5. Fully Sharded Data Parallelism (FSDP/ZeRO)]]
6. [[#6. Tensor Parallelism]]
7. [[#7. Pipeline Parallelism]]
8. [[#8. Hybrid Parallel Strategies]]
8½. [[#8½. N-D Parallelism Panorama: Understanding Transformer Parallel Decomposition from a Single-GPU Perspective]]
9. [[#9. Picotron Design Analysis]]
10. [[#10. Summary and Best Practices]]
11. [[#11. Exercises]]

---

## 1. Introduction and Background

### 1.1 Why Do We Need Distributed Training?

When we train large language models (LLM), we face the following core challenges:

> [!important] Key Challenges
> 1. **Memory limits**: model parameters, optimizer states, and activations do not fit in the memory of a single GPU
> 2. **Compute bottlenecks**: a single GPU does not provide enough compute to finish training in a reasonable time
> 3. **Communication overhead**: data transfer across multiple GPUs can become a performance bottleneck

For example, for a 70B-parameter model:
- Parameters themselves (bf16): $70 \times 10^9 \times 2 = 140\text{GB}$
- Adam optimizer states (fp32): $70 \times 10^9 \times 8 = 560\text{GB}$
- Total: about 700 GB, far exceeding the memory of a single H100 (80 GB)

### 1.2 Notation

This tutorial uses the following notation:

| Symbol | Meaning |
|------|------|
| $D$ | `d_model` (hidden dimension / residual stream dimension) |
| $F$ | `d_ff` (feed-forward network dimension) |
| $B$ | Batch size (total number of tokens in the batch) |
| $T$ | Sequence length |
| $L$ | Number of model layers |
| $C$ | FLOPs/s per chip |
| $W$ | Network bandwidth (bidirectional) |
| $X, Y, Z$ | Number of chips on each grid axis |

### 1.3 A Simplified Transformer Layer

To simplify the analysis, we approximate the Transformer layer as a stack of MLP blocks:

```
Input: In[B, D]
    ↓
    ├─→ Win[D, F] → Tmp[B, F] (up projection)
    ↓
    └─→ Wout[F, D] → Out[B, D] (down projection)
```

> [!note] Note
> For large models, Attention only accounts for about 1/3 of FLOPs, and MLP accounts for 2/3. This simplification is therefore a reasonable approximation.

---

## 2. GPU Hardware Fundamentals and Communication Primitives

> **Why understand the hardware first?** Every parallel strategy is fundamentally a trade-off between **computation** and **communication**. Whether a strategy is worthwhile depends on whether its communication cost can be hidden behind computation—and making that comparison requires the actual numerical values of GPU throughput (FLOPs/s) and interconnect bandwidth. The hardware intuition built in this section is the "unit-conversion foundation" for the entire parallel analysis framework.

### 2.1 NVIDIA H100 Hardware Specs

Before diving into parallel strategies, we need to understand the key hardware parameters of GPUs:

| Spec | H100 SXM | H100 PCIe | A100 |
|------|----------|-----------|------|
| Memory capacity | 80 GB HBM3 | 80 GB HBM2e | 80 GB HBM2e |
| Memory bandwidth | 3.35 TB/s | 2.0 TB/s | 2.0 TB/s |
| FP16 throughput | 989 TFLOPS | 756 TFLOPS | 312 TFLOPS |
| BF16 throughput | 989 TFLOPS | 756 TFLOPS | 312 TFLOPS |
| FP8 throughput | 1,979 TFLOPS | 1,513 TFLOPS | - |
| NVLink bandwidth | 900 GB/s | 600 GB/s (NVL) | 600 GB/s |

> [!tip] Key ratio: arithmetic intensity
> **Arithmetic intensity** = $C / W_{mem}$ indicates how many FLOPs are needed per byte transferred to "hide" transfer latency
> 
> For H100 SXM:
> - Memory arithmetic intensity: $989 \times 10^{12} / 3.35 \times 10^{12} \approx 295$ (bf16)
> - NVLink arithmetic intensity: $989 \times 10^{12} / 9 \times 10^{11} \approx 1100$ (bf16)

### 2.2 GPU Topology

Modern datacenter GPUs are typically connected as follows:

```
┌─────────────────────────────────────────────────────┐
│                    DGX H100 Node                     │
│  ┌─────┐  NVSwitch  ┌─────┐  NVSwitch  ┌─────┐      │
│  │ GPU0├────────────┤ GPU1├────────────┤ GPU2│...   │
│  └─────┘   900GB/s  └─────┘   900GB/s  └─────┘      │
│     ↓                  ↓                  ↓          │
│           PCIe / NVLink Switch System               │
└─────────────────────────────────────────────────────┘
                           ↓
                    InfiniBand / RoCE
                    (200-400 Gb/s per node)
                           ↓
┌─────────────────────────────────────────────────────┐
│                   Other Nodes                        │
└─────────────────────────────────────────────────────┘
```

**Three bandwidth tiers**:
1. **Intra-node NVLink**: ~900 GB/s (H100 SXM)
2. **Inter-node InfiniBand**: ~50-100 GB/s
3. **Datacenter network**: ~25 GB/s

### 2.3 Core Communication Primitives (Collective Operations)

Distributed training relies on the following core communication operations:

#### 2.3.1 AllGather

**Function**: collect shards from all devices so every device has the full data

```
Device 0: [A0]     →   Device 0: [A0, A1, A2, A3]
Device 1: [A1]     →   Device 1: [A0, A1, A2, A3]
Device 2: [A2]     →   Device 2: [A0, A1, A2, A3]
Device 3: [A3]     →   Device 3: [A0, A1, A2, A3]
```

**Notation**: $\text{AllGather}_X([A_X, B]) \rightarrow [A, B]$

**Cost**: $T = \frac{V}{W_{bidirectional}}$, where $V$ is the total data volume


> **AllGather Ring Algorithm**: $N$ devices are arranged in a ring, each device holds $V/N$ bytes of data. Each step sends the current block to the right and receives the block to the left. The two-way ring transmits to the left and right simultaneously:
> $$T_\text{hop} = \frac{2V}{N \cdot W_\text{ici}}, \quad T_\text{total} = \frac{N}{2} \cdot T_\text{hop} = \frac{V}{W_\text{ici}}$$
> **Key Insight**: AllGather time is **irrelevant** to $N$ of devices (in bandwidth-limited mode)!

**Latency correction**: when the data volume per hop is small, the per-hop latency $T_\text{min} \approx 1\,\mu\text{s}$ becomes the bottleneck:

$$T_\text{hop} = \max\!\left[T_\text{min},\ \frac{2V}{N \cdot W_\text{ici}}\right] \quad \Rightarrow \quad T_\text{total} = \max\!\left[\frac{T_\text{min} \cdot N}{2},\ \frac{V}{W_\text{ici}}\right]$$

For TPU v5e ($W_\text{ici} = 4.5 \times 10^{10}$ B/s), the latency threshold is around 45 kB: arrays smaller than this size are latency bound.

**Multi-axis AllGather**: if AllGather runs simultaneously over multiple mesh axes $\{X_1, X_2, \ldots\}$, the effective bandwidth increases proportionally:

$$T_\text{total} = \max\!\left[\frac{T_\text{min} \cdot \sum |X_i|}{2},\ \frac{V}{W_\text{ici} \cdot N_\text{axes}}\right]$$

![AllGather measured bandwidth (TPU v5e 8×16): about 95% peak above 10 MB](https://jax-ml.github.io/scaling-book/assets/img/all-gather-bandwidth.png)

> [!example] AllGather time estimate
>
> Grid: TPU v5e, `{'X': 8, 'Y': 4}`, ICI two-way bandwidth $W = 4.5 \times 10^{10}$ B/s
>
> **(a)** `AllGather_Y([E_Y, F])`, $E = 2048$, $F = 8192$, bfloat16
> - holds `bf16[512, 8192]` = 8.4 MB per device, total array 33.6 MB
> - Time (bandwidth limited): $T = 33.6\text{ MB} / 4.5 \times 10^{10} \approx 747\,\mu\text{s}$ (actual measurement includes overhead of about 680 μs)
>
> **(b)** Same settings, $E = 256$, $F = 256$
> - `bf16[64, 256]` = 32 kB < 45 kB threshold held per device → **Latency bound**
> - Time: $T \approx T_\text{min} \times (Y/2) = 1\,\mu\text{s} \times 2 = 2\,\mu\text{s}$ (actually measured about 8 μs)

#### 2.3.2 ReduceScatter

**Function**: first reduce (sum), then scatter across devices

```
Device 0: [A0, B0, C0, D0]   →   Device 0: [A0+A1+A2+A3]
Device 1: [A1, B1, C1, D1]   →   Device 1: [B0+B1+B2+B3]
Device 2: [A2, B2, C2, D2]   →   Device 2: [C0+C1+C2+C3]
Device 3: [A3, B3, C3, D3]   →   Device 3: [D0+D1+D2+D3]
```

**Notation**: $\text{ReduceScatter}_{X,K}([A, K]\{U_X\}) \rightarrow [A, K_X]$

**Cost**: same as AllGather
> **Dual relationship between ReduceScatter and AllGather** (Kronecker product perspective):
>
> Define the broadcast operator $\text{broadcast} = \mathbf{u} \otimes I_n$, and the reduction operator $\text{reduce} = \mathbf{u}^T \otimes I_n$ ($\mathbf{u} = (1,\ldots,1)^T$), then:
> - $\text{AllGather} = \text{broadcast} \otimes I_p$
> - $\text{ReduceScatter} = \text{reduce} \otimes I_p$
>
> Since $(\mathbf{u} \otimes I_n)^T = \mathbf{u}^T \otimes I_n$, we have $\text{AllGather}^T = \text{ReduceScatter}$.
>
> This means that the gradient of AllGather in backpropagation is ReduceScatter and vice versa - this is a mathematical necessity, not a coincidence.

#### 2.3.3 AllReduce

**Function**: sum the data across all devices and replicate the result to every device

```
Device 0: [A0]   →   Device 0: [A0+A1+A2+A3]
Device 1: [A1]   →   Device 1: [A0+A1+A2+A3]
Device 2: [A2]   →   Device 2: [A0+A1+A2+A3]
Device 3: [A3]   →   Device 3: [A0+A1+A2+A3]
```

> [!important] Key relationship
> **AllReduce = ReduceScatter + AllGather**
> 
> Therefore, AllReduce takes 2 times as much time as AllGather: $T = \frac{2V}{W}$

#### 2.3.4 AllToAll

**Function**: transpose the sharded dimension

```
Device 0: [A0, B0]   →   Device 0: [A0, A1]
Device 1: [A1, B1]   →   Device 1: [B0, B1]
```

**Notation**: $\text{AllToAll}_{X, J}([A, B_X]) \rightarrow [A_X, B]$

**Cost**: about one-quarter of AllGather

> **Why is AllToAll 4 times faster than AllGather? **(two-way ring)
>
> - **AllGather**: Each piece of data needs to reach all $N-1$ other devices, the total transmission volume of each link in the one-way ring $\propto V(1-1/N)$
> - **AllToAll**: The data block of device $i$ only needs to be sent to device $j$ (taking $j-i$ steps), the total link load $\propto V \cdot \frac{N(N-1)/2}{N^2} \approx V/2$
> - One-way ratio: AllToAll/AllGather $= 1/2$
>
> When optimizing in both directions: AllGather is only 2 times faster (each direction shares half of the traffic); AllToAll is 4 times faster (each block takes the shortest path $\min(j-i, N-(j-i))$, and the average distance is further halved):
> $$T_\text{AllToAll} = \frac{T_\text{AllGather}}{4} \quad \text{(bidirectional ring)}$$

#### 2.3.5 Summary of Communication Operations

| Operation | Description | Symbol | Time consuming |
|------|------|------|------|
| AllGather | Collect shards, remove subscripts | $[A_X, B] → [A, B]$ | $V / W$ |
| ReduceScatter | Reduce and scatter | $[A, B]\{U_X\} → [A_X, B]$ | $V / W$ |
| AllReduce | Full Reduce | $[A_X, B]\{U_Y\} → [A_X, B]$ | $2V / W$ |
| AllToAll | Transpose shards | $[A, B_X] → [A_X, B]$ | $V / (4W)$ |

![Comparison of four collective communication primitives](https://jax-ml.github.io/scaling-book/assets/img/all-collectives.png)

---

## 3. Sharded Matrices and Matrix Multiplication

> **Why start with sharded matrix multiplication?** The vast majority of LLM compute (about 90%) comes from matrix multiplication (QKV projections, MLP layers, and so on). Once you understand how to multiply sharded matrices efficiently, you can systematically derive all parallel strategies—data parallelism, tensor parallelism, and FSDP are all fundamentally different sharding choices for matrix multiplication, each corresponding to a different communication-computation trade-off. The original material develops the full theory here: [Sharded Matrices and How to Multiply Them](https://jax-ml.github.io/scaling-book/sharding/).

### 3.1 Sharding Notation System

We use named-axis notation to describe how tensors are sharded over the device mesh:

![Sharding example: array of global shape (4,128) on 4 devices, per-device local shape (2,64)](https://jax-ml.github.io/scaling-book/assets/img/sharding-example.png)

- **Device Mesh**: defines how physical devices are organized
  ```python
  mesh = DeviceMesh("cuda", (4, 2))  # 4×2 device mesh
  mesh = DeviceMesh("cuda", (4, 2), mesh_dim_names=("X", "Y"))
  ```

- **Sharding Spec**: describes how each tensor dimension maps to a mesh axis

  > **Notation intuition**: I, J, K, ... are the tensor's **logical dimension names**, while X, Y, Z, ... are the device mesh's **physical axis names**. A subscript binds the two together:
  > ```
> A [ I_X , J_Y ]
> ↑ ↑ ↑ ↑ ↑
> array dimension physical axis dimension physical axis
> (row) (cut along X) (column) (cut along Y)
  > ```
  > No subscript (for example `J`) means that dimension is **not sharded** and is fully replicated on every device.

  ```
  A[I_X, J_Y]  # I dimension sharded along X, J dimension sharded along Y
  A[I_XY, J]   # I dimension sharded across the flattened X and Y axes
  A[I, J]      # fully replicated (no sharding)
  ```

> [!example] Sharding example
>
> For a tensor of shape `[1024, 4096]` on a mesh `{'X': 8, 'Y': 2}`:
>
> | Sharding spec | Per-device shape | Total memory multiplier |
> |----------|-----------|-----------|
> | $A[I, J]$ | [1024, 4096] | 16× |
> | $A[I_X, J]$ | [128, 4096] | 2× |
> | $A[I_X, J_Y]$ | [128, 2048] | 1× |
> | $A[I_{XY}, J]$ | [64, 4096] | 1× |
>
> **Total memory multiplier = the product of the device counts along mesh axes that are not used for sharding** (that is, the number of replicas). Axes used for sharding do not replicate the data; unused axes store the full tensor on each device and therefore replicate it:
> - $A[I, J]$: neither X nor Y is used → replicated 8×2 = **16 copies**
> - $A[I_X, J]$: X shards I and Y is unused → replicated 1×2 = **2 copies**
> - $A[I_X, J_Y]$: both X and Y are used for sharding → replicated 1×1 = **1 copy**
> - $A[I_{XY}, J]$: both X and Y shard I → replicated 1×1 = **1 copy**

**JAX code example**:

```python
import jax
import jax.numpy as jnp

# Create a 4×2 device mesh (requires 8 devices)
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# Define a sharding-spec helper
def P(*args):
    return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# Create sharded arrays (JAX handles communication automatically and transparently)
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))   # A[I_X, J_Y]
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))  # B[J, K_Y]

# Sharded matrix multiplication (JAX automatically inserts required collectives)
y = jax.jit(
    lambda A, B: jnp.einsum('BD,DF->BF', A, B),
    out_shardings=P('X', 'Y')
)(A, B)
```

> [!note] JAX sharding transparency
> Sharded arrays behave exactly like ordinary arrays: you can apply arbitrary operations, and the JAX compiler automatically infers and inserts the required communication primitives.

**PyTorch equivalent** (`DTensor`, PyTorch 2.0+):

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate

# Initialize the process group (requires 8 processes)
dist.init_process_group(backend="nccl")

# Create a 4×2 device mesh
mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("X", "Y"))

# A[I_X, J_Y]: dimension 0 sharded along X, dimension 1 sharded along Y
A = distribute_tensor(
    torch.zeros(8, 2048, dtype=torch.bfloat16),
    mesh,
    placements=[Shard(0), Shard(1)]
)

# B[J, K_Y]: dimension 0 replicated across X, dimension 1 sharded along Y
B = distribute_tensor(
    torch.zeros(2048, 8192, dtype=torch.bfloat16),
    mesh,
    placements=[Replicate(), Shard(1)]
)

# Sharded matrix multiplication (DTensor automatically infers and inserts communication)
y = torch.einsum('BD,DF->BF', A, B)  # y is automatically sharded as [Shard(0), Shard(1)]
```

> [!note] JAX ↔ PyTorch DTensor correspondence
> | JAX `PartitionSpec` | PyTorch `placements` |
> |---------------------|----------------------|
> | `P('X', 'Y')` | `[Shard(0), Shard(1)]` |
> | `P(None, 'Y')` | `[Replicate(), Shard(1)]` |
> | `P(None, None)` | `[Replicate(), Replicate()]` |
> | `{U_X}` (partial sum) | `[Partial(), ...]` |

> [!example] Pop Quiz 1: 2D sharded-memory calculation
>
> Array `fp32[1024, 4096]`, sharding spec $A[I_{XY}, J]$, mesh `{'X': 8, 'Y': 2}`
>
> - Per-device local shape: `fp32[64, 4096]` ($1024 / (8 \times 2) = 64$)
> - Per-device memory: $64 \times 4096 \times 4 = 1\text{ MiB}$
> - H100 load time (3.35 TB/s): $10^6 / 3.35 \times 10^{12} \approx 0.3\,\mu\text{s}$ (longer in practice once overhead is included)

> [!example] Pop Quiz 2: total memory with replicated sharding
>
> Array `int8[128, 2048]`, sharding spec $A[I_{XY}, J]$, mesh `{'X': 2, 'Y': 8, 'Z': 2}` (32 devices total)
>
> - Sharding only uses the X and Y axes (16 devices total), so the **Z axis (2 devices) is fully replicated**
> - Per-device local shape: `int8[8, 2048]` ($128 / (2 \times 8) = 8$)
> - Per-device memory: $8 \times 2048 \times 1 = 16\text{ KiB}$
> - **Total memory**: $16\text{ KiB} \times 32\text{ devices} = 512\text{ KiB}$ (twice the original 256 KiB because the Z axis introduces one extra replica)

### 3.2 Four Cases of Sharded Matrix Multiplication

When performing sharded matrix multiplication $C = A \cdot B$, the communication pattern depends on how the inputs are sharded:

#### Case 1: Neither contraction dimension is sharded

$$A[I_X, J] \cdot B[J, K_Y] \rightarrow C[I_X, K_Y]$$

**No communication required.** Each device can perform the local multiplication independently.

```python
# PyTorch example
# A: [batch/X, d_model], B: [d_model, d_ff/Y] → C: [batch/X, d_ff/Y]
local_C = torch.matmul(local_A, local_B)
```

#### Case 2: The contraction dimension of one input is sharded

$$A[I, J_X] \cdot B[J, K] \rightarrow C[I, K]$$

**Requires AllGather**: first gather A, then perform the local multiplication

```python
# First AllGather A
full_A = all_gather(local_A, dim=1)  # [I, J_X] → [I, J]
# Then local multiplication
local_C = torch.matmul(full_A, local_B)
```

#### Case 3: Both inputs have the contraction dimension sharded along the same axis

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K]\{U_X\}$$

**Local multiplication produces partial sums, so AllReduce is required**:

```python
# Local multiplication (partial sums)
partial_C = torch.matmul(local_A, local_B)  # each device gets a partial result
# AllReduce sum
full_C = all_reduce(partial_C, op=SUM)
```

> [!note] Optimization: use ReduceScatter instead of AllReduce
> If you need a sharded result afterward, you can use ReduceScatter:
> $$C[I, K]\{U_X\} \xrightarrow{\text{ReduceScatter}} C[I, K_X]$$
> This saves half the communication volume.

#### Case 4: Two non-contraction dimensions are sharded along the same axis (invalid)

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X] \quad \text{❌ Invalid!}$$

**You must AllGather one of the inputs first**:

```python
# Option 1: AllGather A
full_A = all_gather(local_A, dim=0)
local_C = torch.matmul(full_A, local_B)  # C[I, K_X]

# Option 2: AllGather B
full_B = all_gather(local_B, dim=1)
local_C = torch.matmul(local_A, full_B)  # C[I_X, K]
```

### 3.3 Communication-Computation Overlap (Collective Matmul)

The key optimization is to **perform computation while communication is in flight**.

```
Timeline:
├── AllGather chunk 0 ──┬── AllGather chunk 1 ──┬── AllGather chunk 2 ──┤
                     │                    │                    │
                     └── MatMul chunk 0 ─────┴── MatMul chunk 1 ─────┴── MatMul chunk 2
```

In PyTorch, this is implemented via CUDA streams:

```python
import torch
import torch.distributed as dist

# Create a dedicated communication stream
comm_stream = torch.cuda.Stream()
comp_stream = torch.cuda.current_stream()

# Chunk the tensor
chunks = tensor.chunk(num_chunks, dim=0)

for i, chunk in enumerate(chunks):
    # Launch AllGather on the communication stream
    with torch.cuda.stream(comm_stream):
        gathered_chunk = all_gather_async(chunk)
    
    # Process the previous chunk on the compute stream at the same time
    if i > 0:
        result_chunks[i-1] = compute(gathered_chunks[i-1])
    
    # Wait for the current chunk's AllGather to finish
    comp_stream.wait_stream(comm_stream)
    gathered_chunks[i] = gathered_chunk
```

---

## 4. Data Parallelism

> **Motivation**: Data parallelism is the most natural form of parallelism: split the dataset, let each GPU hold a full copy of the model, run forward and backward passes independently, and finally synchronize gradients with AllReduce. Its advantages are simplicity of implementation (PyTorch DDP is essentially a one-line wrapper) and zero communication in the forward pass. The key question is: when does gradient AllReduce become the bottleneck? The answer depends on the relationship between per-GPU batch size and the hardware compute-to-bandwidth ratio.

> [!note] Relationship to Chapters 2 and 3
> Chapter 2 provides the **vocabulary** (primitives such as AllGather and AllReduce, plus their costs), while Chapter 3 provides the **grammar** (given a sharding pattern, determine which primitive is required). Starting in Chapter 4, we apply this toolkit to concrete problems: choose a sharding scheme for the Transformer → use the rules from Chapter 3 to derive the required communication → use the formulas from Chapter 2 to estimate communication time → compare against compute time. **This is the unified analysis framework for every parallel strategy, and the rest of the tutorial follows it.**

Sharding selection for data parallelism: **B (batch) sharding along X, weights fully replicated**.

| | Forward pass | Backward pass |
|--|---------|---------|
| Sharding form | $\text{In}[B_X, D] \cdot W[D, F]$ | Gradient $\nabla W[D,F]\ \{U_X\}$ |
| Corresponding case from Chapter 3 | Case 1 (the contraction dimension D is unsharded) → **No communication required** | Partial sums must be reduced → **AllReduce** |
| Communication overhead | 0 | $4DF / W$ |

The following chapters (FSDP and TP) simply change the sharding choice. The communication primitives then change accordingly, but the analysis framework stays the same.

### 4.1 Basic Principle

**Data parallelism** is the simplest parallel strategy:

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

```
┌──────────────────────────────────────────────────────────┐
│                  Data Parallelism Diagram                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   GPU 0              GPU 1              GPU 2            │
│  ┌───────┐          ┌───────┐          ┌───────┐         │
│  │Batch 0│          │Batch 1│          │Batch 2│         │
│  │(B/3)  │          │(B/3)  │          │(B/3)  │         │
│  └───┬───┘          └───┬───┘          └───┬───┘         │
│      ↓                  ↓                  ↓              │
│  ┌───────┐          ┌───────┐          ┌───────┐         │
│  │  Full  │         │  Full  │         │  Full  │        │
│  │Weights │         │Weights │         │Weights │        │
│  │ (W)   │          │ (W)   │          │ (W)   │         │
│  └───────┘          └───────┘          └───────┘         │
│      ↓                  ↓                  ↓              │
│  ┌───────┐          ┌───────┐          ┌───────┐         │
│  │Grad 0 │←─────────┼──AllReduce──────→│Grad 2 │         │
│  └───────┘          └───────┘          └───────┘         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Algorithm Details

**Forward pass** (no communication):

```python
def forward_pass(input_shard, W_in, W_out):
    # input_shard: [B/X, D]
    # W_in, W_out: fully replicated
    tmp = input_shard @ W_in       # [B/X, F]
    output = tmp @ W_out           # [B/X, D]
    return output
```

**Backward pass** (requires AllReduce):

```python
def backward_pass(dL_dOutput, input_shard, tmp, W_in, W_out):
    # Compute local gradients
    dL_dW_out_local = tmp.T @ dL_dOutput        # [F, D] partial sum
    dL_dTmp = dL_dOutput @ W_out.T              # [B/X, F]
    dL_dW_in_local = input_shard.T @ dL_dTmp    # [D, F] partial sum
    dL_dInput = dL_dTmp @ W_in.T                # [B/X, D]
    
    # AllReduce gradients (can overlap with the next layer's compute)
    dL_dW_out = all_reduce(dL_dW_out_local)     # [F, D] full gradient
    dL_dW_in = all_reduce(dL_dW_in_local)       # [D, F] full gradient
    
    return dL_dInput, dL_dW_in, dL_dW_out
```

### 4.3 Communication Analysis

Each layer requires 2 AllReduces:

$$T_\text{comms} = \frac{2 \times 2 \times 2 \times D \times F}{W_\text{NVLink}} = \frac{8DF}{W}$$

Compute time:

$$T_\text{math} = \frac{8 \times B \times D \times F}{X \times C}$$

> [!important] Compute-bound condition
> When $T_\text{math} > T_\text{comms}$, the system is compute-bound (the ideal regime):
> 
> $$\frac{B}{X} > \frac{C}{W}$$
> 
> For H100 SXM: $C/W \approx 989 \times 10^{12} / 9 \times 10^{11} \approx 1100$
> 
> That is, the batch size on each GPU must exceed roughly 1100 tokens to utilize compute resources efficiently.

### 4.4 PyTorch DDP Implementation

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize the process group
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Wrap the model
model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])

# Training loop - DDP handles gradient synchronization automatically
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()  # DDP automatically AllReduces gradients here
    optimizer.step()
```

---

## 5. Fully Sharded Data Parallelism (FSDP/ZeRO)

### 5.1 Motivation and Principle

**FSDP** (Fully Sharded Data Parallel), also known as **ZeRO-3**, addresses the memory limitations of pure data parallelism:

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

Core idea: **parameters, gradients, and optimizer states are all sharded along the data-parallel dimension**.

```
┌──────────────────────────────────────────────────────────┐
│                       FSDP Diagram                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   GPU 0              GPU 1              GPU 2            │
│  ┌───────┐          ┌───────┐          ┌───────┐         │
│  │Batch 0│          │Batch 1│          │Batch 2│         │
│  └───┬───┘          └───┬───┘          └───┬───┘         │
│      │                  │                  │              │
│  ┌───┴───┐          ┌───┴───┐          ┌───┴───┐         │
│  │W shard0│         │W shard1│         │W shard2│        │
│  │ (1/3) │          │ (1/3) │          │ (1/3) │         │
│  └───────┘          └───────┘          └───────┘         │
│      │                  │                  │              │
│      ├──────────AllGather────────────────┤              │
│      ↓                  ↓                  ↓              │
│  ┌───────┐          ┌───────┐          ┌───────┐         │
│  │ full W │         │ full W │         │ full W │        │
│  │(temp.) │         │(temp.) │         │(temp.) │        │
│  └───┬───┘          └───┬───┘          └───┬───┘         │
│      ↓ Forward compute   ↓                  ↓              │
│      ↓ Discard full W    ↓                  ↓              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Three Phases of ZeRO: Accurate Memory Analysis

Under mixed-precision training, the memory footprint per parameter is 16 bytes/parameter:

```
Memory breakdown per parameter:

  bf16 param   fp32 grad    fp32 master   fp32 m      fp32 v
  ┌─────────┐ ┌─────────┐  ┌─────────┐  ┌─────────┐ ┌─────────┐
  │  2 bytes│ │  4 bytes│  │  4 bytes│  │  4 bytes│ │  4 bytes│
  └─────────┘ └─────────┘  └──────────────────────────────────┘
   parameter    gradient     ←──────── Adam optimizer states: 12 bytes ────────→
```

Take a **7B-parameter model with N=8 GPUs** as an example (pure DDP requires 112 GB/GPU):

```
                    Param(2B)   Grad(4B)    Optimizer(12B) Per-GPU Total
─────────────────────────────────────────────────────────────────
DDP (unsharded)    14 GB       28 GB       84 GB         126 GB  ❌
ZeRO-1 (shard OS)  14 GB       28 GB       84/8=10.5 GB  52.5 GB
ZeRO-2 (shard G+OS)14 GB      28/8=3.5 GB 84/8=10.5 GB  28  GB
ZeRO-3/FSDP      14/8=1.75GB 28/8=3.5 GB 84/8=10.5 GB  15.75GB ✓
─────────────────────────────────────────────────────────────────
Note: activations are not sharded in any stage (use activation recomputation separately to save memory)
```

| Stage | What is sharded | Additional communication | Recommended scenario |
|------|----------|----------|----------|
| ZeRO-1 | Optimizer states | No additional communication | Optimizer state is the bottleneck |
| ZeRO-2 | + Gradients | No additional communication | Gradients also no longer fit |
| ZeRO-3 (FSDP) | + Parameters | Two extra `AllGather`s in the forward pass | Even parameters do not fit |

### 5.4 Communication Analysis

Compared to pure data parallelism:

| Operation | Data Parallelism | FSDP |
|------|----------|------|
| Forward communication | 0 | 2 × AllGather(W) |
| Backward communication | 2 × AllReduce(∇W) | 2 × AllGather(W) + 2 × ReduceScatter(∇W) |
| Total traffic | $4 \times 2DF$ ​​| $4 \times 2DF$ ​​|

> [!important] Key insight
> The total communication volume of FSDP is **the same** as in pure data parallelism.
> 
> This is because AllReduce = AllGather + ReduceScatter.
> 
> But FSDP dramatically reduces memory usage.

### 5.5 PyTorch FSDP Implementation

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# Define the wrapping policy
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock}
)

# Mixed-precision configuration
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16
)

# Wrap the model
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
```

### 5.6 FSDP Communication Timeline

FSDP spreads communication across the full forward/backward pass, whereas DDP communicates in a single burst after the backward pass finishes:

```
DDP (naive):
Forward ──────────────────────────────────────── Backward ────────── AllReduce ──▶

FSDP:
Forward: [AG W1][compute][free W1][AG W2][compute][free W2]...
Backward:[AG W_L][compute][RS ∇W_L][free][AG W_{L-1}][compute][RS ∇W_{L-1}]...

AG = AllGather (reconstruct weights)  RS = ReduceScatter (reduce + shard gradients)
free = immediately release full weights (the key to memory savings)
```

**FSDP communication volume = DDP communication volume**, but it is distributed across more points in time, making it easier to overlap with computation.

### 5.7 ZeRO++: Communication volume halved again

The communication bottleneck in ZeRO-3 is cross-node AllGather, since inter-node bandwidth is only about one-tenth of intra-node bandwidth. ZeRO++ introduces three optimizations:

```
ZeRO++ optimization 1: quantized AllGather (qG)
  bf16 weights → int8 quantization → cross-node AllGather (half the data volume) → dequantization
  Cost: slight accuracy loss

ZeRO++ optimization 2: hierarchical AllGather (hpZ)
  First do intra-node AllGather (NVLink, fast) → each node completes its own local computation
  Cost: uses tp× more weight memory within each node

ZeRO++ optimization 3: quantized ReduceScatter (qRS)
  Gradients are transmitted after quantization → stored after dequantization
  Cost: gradient precision loss (usually not a major issue)
```

### 5.8 Interview FAQs

> [!question] FSDP does not reduce activation memory. What should I do?
> Activations (the intermediate results for each batch) are not sharded in any ZeRO stage and therefore remain full-sized on each GPU. You need to apply **activation recomputation (gradient checkpointing)** separately: do not save activations in the forward pass, and recompute them during the backward pass. The cost is about 33% extra compute in exchange for a large reduction in activation memory.

> [!question] When should I choose FSDP, and when should I choose TP?
> - FSDP: solves memory problems, with communication mainly in the backward pass; efficient for large batches
> - TP: improves efficiency for small batches, but communicates at every matmul; requires NVLink
> - In practice: start with FSDP, and add TP if the per-GPU batch size is below about 1000 tokens

### 5.9 When to Use FSDP

> [!tip] Good use cases for FSDP
> - Model size exceeds single GPU memory
> - Per-GPU batch size is large enough ($B/X > C/W$)
> - You want to scale without modifying model code

> [!warning] Limitations of FSDP
> - The compute-bound condition is the same as for data parallelism
> - On H100: batch size must exceed about 1100 tokens per GPU
> - If you need a smaller batch size, you must combine it with tensor parallelism

---

## 6. Tensor Parallelism

> **Motivation**: FSDP is efficient only when the per-GPU batch size exceeds roughly $C/W \approx 1100$ tokens. In inference ($B=1$) or other small-batch regimes, that condition is hard to satisfy, and FSDP becomes communication-bound. Tensor parallelism takes a different approach: **do not shard the data; shard the weights**—split the FFN width dimension $F$ or the number of attention heads across devices. Then the efficiency condition becomes $F > Y \times C/W$, so compute can still be utilized efficiently even when $B=1$. The trade-off is that every matrix multiplication requires communication, so TP must run over high-bandwidth intra-node NVLink.

### 6.1 Basic Principle

**Tensor parallelism** (also known as Megatron sharding) shards model dimensions:

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

Core idea: **shard model dimensions rather than data dimensions**.

```
┌──────────────────────────────────────────────────────────┐
│                 Tensor Parallelism Diagram                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│        Input [B, D]                                      │
│            │                                             │
│            ├──AllGather──┐                               │
│            ↓              ↓                               │
│   GPU 0: In[B,D]   GPU 1: In[B,D]                        │
│            │              │                               │
│            ↓              ↓                               │
│   ┌────────────┐  ┌────────────┐                         │
│   │W_in[D,F/2] │  │W_in[D,F/2] │   (column parallel)    │
│   └──────┬─────┘  └─────┬──────┘                         │
│          ↓              ↓                                 │
│   Tmp[B, F/2]    Tmp[B, F/2]                             │
│          │              │                                 │
│          ↓              ↓                                 │
│   ┌────────────┐  ┌────────────┐                         │
│   │W_out[F/2,D]│  │W_out[F/2,D]│   (row parallel)       │
│   └──────┬─────┘  └─────┬──────┘                         │
│          │              │                                 │
│          └──ReduceScatter──┐                             │
│                    ↓                                      │
│            Out[B, D/2] (sharded)                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 6.2 How ReduceScatter Works in TP

After a row-parallel multiplication, each GPU holds a partial sum that must be merged. Take `tp=2` and output dimension `D=4` as an example:

```
After the row-parallel matmul, each GPU has computed one part:

GPU 0 partial: Tmp0[B, F/2] @ W0[F/2, D] = C0[B, D=4]
GPU 1 partial: Tmp1[B, F/2] @ W1[F/2, D] = C1[B, D=4]

                C0          C1          True Out = C0 + C1
GPU 0 holds: [1, 2, 3, 4]             → [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
GPU 1 holds:            [5, 6, 7, 8]   (each position is the sum of contributions from both GPUs)
```

**AllReduce approach** (original Megatron): each GPU broadcasts its partial result to the other, then both GPUs add the two partials together → both GPUs get the full `[6, 8, 10, 12]`, doubling memory usage.

**ReduceScatter approach** (SP version): split the output along the D dimension, and let each GPU sum only the slice it is responsible for:

```
Step 1: Exchange data
  GPU 0 sends the second half of C0, [3,4], to GPU 1
  GPU 1 sends the first half of C1, [5,6], to GPU 0

Step 2: Locally add the slice each GPU is responsible for
  GPU 0 first half: [1,2] + [5,6] = [6, 8]       ← correct answer for the first D/2
  GPU 1 second half: [3,4] + [7,8] = [10, 12]    ← correct answer for the second D/2

Result:
  GPU 0 holds Out[B, :D/2] = [6, 8]    (SP state: sharded along D)
  GPU 1 holds Out[B, D/2:] = [10, 12]
```

**Key point**: ReduceScatter does two things at once: ① it sums the partial results (**Reduce**), and ② it stores the output in sharded form (**Scatter**). This exactly matches the SP state required by the next LayerNorm: one communication, two purposes, zero extra cost.

### 6.3 Column Parallelism and Row Parallelism

#### Column Parallel

Split the weight matrix along columns:

$$W = [W_0 | W_1 | ... | W_{n-1}]$$

- Input: copied to all GPUs
- Output: Each GPU holds a portion of the output
- No communication required (until full output is required)

#### Row Parallel

Split the weight matrix along the rows:

$$W = \begin{bmatrix} W_0 \\ W_1 \\ \vdots \\ W_{n-1} \end{bmatrix}$$

- Input: must be sharded
- Output: Requires AllReduce or ReduceScatter aggregation

### 6.3 Tensor Parallelism in the MLP Layer

The Transformer's MLP maps naturally onto a column-parallel + row-parallel decomposition:

```
┌─────────────────────────────────────────────────┐
│              MLP Tensor Parallelism              │
├─────────────────────────────────────────────────┤
│                                                 │
│   Input [B, D]                                  │
│       │                                         │
│       │ (replicated)                             │
│       ↓                                         │
│   ┌───────────┐     ┌───────────┐              │
│   │ W_up      │     │ W_gate    │  (column parallel) │
│   │ [D, F/Y]  │     │ [D, F/Y]  │              │
│   └─────┬─────┘     └─────┬─────┘              │
│         │                 │                     │
│         ↓                 ↓                     │
│   hidden [B, F/Y]   gate [B, F/Y]              │
│         │                 │                     │
│         └────── × ────────┘  (element-wise)     │
│                 │                               │
│                 ↓                               │
│         ┌─────────────┐                        │
│         │ W_down      │  (row parallel)         │
│         │ [F/Y, D]    │                        │
│         └──────┬──────┘                        │
│                │                               │
│                ↓                               │
│         partial [B, D]                         │
│                │                               │
│         AllReduce / ReduceScatter              │
│                │                               │
│                ↓                               │
│         Output [B, D] or [B, D_Y]               │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 6.4 Tensor Parallelism in the Attention Layer

```
┌─────────────────────────────────────────────────┐
│           Attention Tensor Parallelism           │
├─────────────────────────────────────────────────┤
│                                                 │
│   Input [B, S, D]                               │
│       │                                         │
│       │ (replicated)                             │
│       ↓                                         │
│   ┌─────┐  ┌─────┐  ┌─────┐                    │
│   │ W_Q │  │ W_K │  │ W_V │   (column parallel) │
│   │[D,H]│  │[D,H]│  │[D,H]│   H = n_heads/Y    │
│   └──┬──┘  └──┬──┘  └──┬──┘                    │
│      │        │        │                        │
│      ↓        ↓        ↓                        │
│  Q[B,S,H]  K[B,S,H]  V[B,S,H]                  │
│      │        │        │                        │
│      └────Attention────┘                        │
│              │                                  │
│              ↓                                  │
│        Attn_out [B, S, H]                       │
│              │                                  │
│          ┌───┴───┐                             │
│          │ W_O   │  (row parallel)              │
│          │[H, D] │                             │
│          └───┬───┘                             │
│              │                                  │
│         AllReduce                               │
│              │                                  │
│              ↓                                  │
│        Output [B, S, D]                         │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 6.5 Algorithm Details

```python
def tensor_parallel_forward(input_shard, W_in, W_out):
    """
    input_shard: [B, D/Y] - sharded along the D dimension
    W_in: [D, F/Y] - column parallel
    W_out: [F/Y, D] - row parallel
    """
    # AllGather the input
    input_full = all_gather(input_shard, dim=-1)  # [B, D]
    
    # Column-parallel matmul (no communication)
    tmp = input_full @ W_in  # [B, F/Y]
    
    # Row-parallel matmul (produces partial sums)
    output_partial = tmp @ W_out  # [B, D] {U_Y}
    
    # ReduceScatter
    output_shard = reduce_scatter(output_partial, dim=-1)  # [B, D/Y]
    
    return output_shard
```

### 6.6 Communication Analysis

$$T_\text{math} = \frac{4BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{4BD}{W}$$

> [!important] Compute-bound condition
> $$\frac{F}{Y \cdot C} > \frac{1}{W} \Rightarrow F > Y \cdot \frac{C}{W}$$
> 
> For H100 SXM: $C/W \approx 1100$
> 
> Therefore $Y < F / 1100$
> 
> For LLaMA-70B ($F \approx 28672$): $Y_\text{max} \approx 26$

> [!tip] Key difference
> - **Data parallelism**: limited by batch size
> - **Tensor parallelism**: limited by model width, independent of batch size

### 6.7 Residual Connections

The subtlest design point in TP is **how to keep the residual path (`x + sublayer(x)`) correct**.

```
Standard Transformer layer (TP=2):

x [B,S,D] (replicated)
│
├──────────────────────────────────────┐  ← residual branch (unchanged)
│                                      │
▼                                      │
LayerNorm (local, no communication)    │
│                                      │
▼ AllGather(SP) → [B, S/cp, D]        │
│                                      │
├── Q_proj[D, D/2] → Q[B,S,D/2]      │  ← ColumnParallel: local matmul, no communication
├── K_proj[D, D/2] → K[B,S,D/2]      │
└── V_proj[D, D/2] → V[B,S,D/2]      │
         │                             │
     Attention (local, each GPU computes D/2 heads) │
         │                             │
     O_proj[D/2, D] (row parallel)    │
         │                             │
     ReduceScatter → [B, S/(cp·sp), D]│  ← sum partial products and enter SP at the same time
         │                             │
         └──────────── + ─────────────┘  ← residual addition (both in SP state, shapes match)
                       │
                  [B, S/(cp·sp), D]

Key point: the RowParallel ReduceScatter output has exactly the same shape as the residual, so they can be added directly with no extra communication
```

### 6.8 Why TP Must Stay Within a Node

TP incurs **2** collective communications per Transformer layer (one for Attention and one for the MLP), so a 32-layer model performs **64** communication rounds per forward pass.

```
Communication latency estimate (per AllGather, data [B,S,D] = 128×4096×8192×2 = 8MB):

NVLink  (900 GB/s): 8MB / 900GB/s ≈ 9μs    ← acceptable
InfiniBand (25 GB/s): 8MB / 25GB/s ≈ 320μs ← 640μs per layer, 32 layers = 20ms
Single-layer compute time (H100): ~2ms (when B=128)
→ Cross-node TP: communication time >> compute time, completely impractical
```

### 6.9 Interview FAQs

> [!question] What is the difference between column-parallel and row-parallel communication patterns?
> - **Column parallel** (split by output dimension): input is replicated, local matrix multiplication runs independently, and output is naturally sharded → **no communication in the forward pass**
> - **Row parallel** (split by input dimension): input is already sharded (from the previous column-parallel layer), local matrix multiplication produces partial sums → **the forward pass requires ReduceScatter**
> - In backpropagation, the two swap roles: column parallel backward needs AllReduce, while row parallel backward needs AllGather

> [!question] What is the upper limit of TP? Why not increase TP indefinitely?
> The efficiency condition is $F > Y \times C/W$, that is, $Y < F / (C/W)$. On H100, $C/W \approx 1100$, and for LLaMA-70B, $F = 28672$, so the theoretical upper limit is $Y \approx 26$. In practice, people usually stop at 8 (the number of GPUs in a node). Beyond that, communication time exceeds compute time.

> [!question] How does TP handle the embedding layer?
> The embedding table $[V, D]$ is large (for LLaMA-3: 128K × 8K ≈ 2 GB). Vocab Parallel lets each GPU hold $[V/\text{tp}, D]$. In the forward pass, each GPU looks up only its own vocabulary range (tokens outside that range contribute 0), then an AllReduce produces the full embedding. The LM head shares weights with the embedding matrix (transposed), so it can reuse the same sharding without extra communication.

> [!question] Can TP and FSDP be used together? How are they combined?
> Yes—this is the most common combination. Let TP=`t` and FSDP=`d`, so the total number of GPUs is `t×d`.
> The `t` GPUs inside each FSDP group run TP (intra-node NVLink), while the `d` FSDP groups run data parallelism across nodes (InfiniBand).
> From FSDP's perspective, the "weight" is already reduced to `1/t` by TP; it is then sharded again across `d` devices, so each GPU stores `1/(t×d)` of the original weights.

---

## 7. Pipeline Parallelism

> **Motivation**: Both TP and FSDP require high-bandwidth links (intra-node NVLink ~900 GB/s) and are therefore ill-suited to cross-node communication (inter-node InfiniBand ~200 Gb/s, roughly 10-100× slower). When the model must span multiple nodes, pipeline parallelism is often the better option: **partition the model by layers across nodes and communicate only activations at stage boundaries**. Since activations are much smaller than weights, this minimizes cross-node bandwidth demand. The trade-off is the introduction of "pipeline bubbles" (GPU idle time), which must be mitigated with microbatch scheduling.

### 7.1 Basic Principle

**Pipeline parallelism** distributes the layers of the model to different devices:

```
┌──────────────────────────────────────────────────────────┐
│                 Pipeline Parallelism Diagram              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   GPU 0         GPU 1         GPU 2         GPU 3       │
│  ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐     │
│  │Layers│ ───→ │Layers│ ───→ │Layers│ ───→ │Layers│     │
│  │ 0-7  │      │ 8-15 │      │16-23 │      │24-31 │     │
│  └──────┘      └──────┘      └──────┘      └──────┘     │
│     ↑                                          │         │
│     │               Backward pass               │         │
│     └──────────────────────────────────────────┘         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 7.2 Pipeline Bubbles: Root Cause

**Bubble = time the GPU is idle waiting for upstream and downstream data**.

Take P=4 stages and a single batch as an example:

```
Timeline (each cell = 1 unit of forward or backward time):

         1    2    3    4    5    6    7    8
GPU 0: [ F0 ]                         [ B0 ]
GPU 1:        [ F0 ]              [ B0 ]
GPU 2:               [ F0 ]  [ B0 ]
GPU 3:                    [ F0 ][ B0 ]
             ←──── bubble ────→
             After GPU 0 finishes F0,
             it must wait for GPU 3 to finish before backpropagating

Bubble sources:
  - Warm-up phase: GPU 0 passes data to GPU 1, then can only wait
  - Cool-down phase: GPU 3 passes gradients back to GPU 2, GPU 2 passes them to GPU 1...
  - GPU 0 must wait for the gradients to return before it can do B0, so it is completely idle meanwhile
```

**Bubble ratio (the most serious in a single batch)** = $(P-1)$ idle units / $2P$ total units = $(P-1)/2P \approx 50\%$

---

### 7.3 Solution: Microbatches + Scheduling

Split a batch into $M$ microbatches to keep the pipeline busy during the warm-up/cool-down period:

#### GPipe / AFAB (All-Forward-All-Backward)

```
P=4 stages, M=4 microbatches, each cell represents one F or B:

         1    2    3    4    5    6    7    8    9   10   11
GPU 0: [ F0 ][ F1 ][ F2 ][ F3 ]  .    .    . [ B3 ][ B2 ][ B1 ][ B0 ]
GPU 1:        [ F0 ][ F1 ][ F2 ][ F3 ]  .    .    . [ B3 ][ B2 ][ B1 ][ B0 ]
GPU 2:               [ F0 ][ F1 ][ F2 ][ F3 ]  .    .    . [ B3 ][ B2 ][ B1 ][ B0 ]
GPU 3:                     [ F0 ][ F1 ][ F2 ][ F3 ][ B3 ][ B2 ][ B1 ][ B0 ]
                                               ↑
                                           GPU 3 finishes the last F
                                           and starts B immediately (no bubble!)

Bubble (.) exists only between the end of GPU 0's F3 and the start of B3 = P-1 = 3 units
Total time = M + (P-1) + M = 2M + (P-1) = 11 units
Ideal time = 2M = 8 units
Bubble ratio = (P-1)/(2M+P-1) ≈ P/(2M)    → for M=4,P=4, about 27%

⚠️ Memory issue: before GPU 0 starts B, the activations of all M=4 microbatches accumulate in memory
→ Activation memory ∝ M × layer_size (grows linearly with M)
```

#### 1F1B (One-Forward-One-Backward)

**Core idea: as soon as a backward step can run, run it immediately instead of waiting for all forward steps to finish.**

```
P=4, M=8 (microbatches), each cell = 1 F or B:

         Warm-up        Steady state (alternating 1F1B)     Cool-down
         ←──P-1──→   ←──────── M=8 ────────→   ←──P-1──→
GPU 0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]
GPU 1:     [F0][F1][F2][B0][F3][B1][F4][B2][F5][B3][F6][B4][B5][B6][B7]
GPU 2:         [F0][F1][B0][F2][B1][F3][B2][F4][B3][F5][B4][B5][B6][B7]
GPU 3:             [F0][B0][F1][B1][F2][B2][F3][B3][F4][B4][B5][B6][B7]
                                                              ↑
                                                      In steady state, GPU 0 is always busy
Bubble = GPU 3 waiting in warm-up + GPU 0 waiting in cool-down ≈ (P-1) units (half before, half after)

Bubble ratio = (P-1)/(M+P-1) ≈ P/M (inversely proportional to M; larger M is better)

Memory advantage: at any time, each GPU stores activations for at most P microbatches
→ Activation memory ∝ P × layer_size (independent of M!)
```

**GPipe vs 1F1B comparison:**

```
                   Bubble ratio      Activation memory
GPipe (AFAB)      (P-1)/(2M+P-1)   M × layer activations
1F1B              (P-1)/(M+P-1)    P × layer activations  ← large memory savings
```

> 1F1B bubble is slightly larger, but the activation memory is reduced from $O(M)$ to $O(P)$, **In large model training $M \gg P$, the memory saving is far more important than the bubble**.

---

### 7.4 Interleaved 1F1B (Virtual Stage)

**Question**: Even if there are M micro-batches, the bubble ratio $(P-1)/M$ is still significant when P is large (such as P=16, M=32, bubbles ≈ 47%).

**Solution**: Each GPU undertakes $V$ **discontinuous virtual stages** (interleaved chunks), which is equivalent to dividing P into finer pipelines:

```
Standard 1F1B (P=4, 1 contiguous layer chunk per GPU):
GPU 0: Layer 0-7    GPU 1: Layer 8-15   GPU 2: Layer 16-23  GPU 3: Layer 24-31

Interleaved 1F1B (P=4, V=2, 2 non-contiguous layer chunks per GPU):
GPU 0: Layer 0-3 and Layer 16-19
GPU 1: Layer 4-7 and Layer 20-23
GPU 2: Layer 8-11 and Layer 24-27
GPU 3: Layer 12-15 and Layer 28-31

→ The effective pipeline depth becomes P×V=8, reducing the bubble ratio by a factor of V:
  Bubble = (P-1)/(V×M+P-1) ≈ P/(V×M)
```

**Cost**: Each micro-batch has to go through $V$ additional stage switches → **The number of P2P communications increases by $V$ times**.

```
Trade-off:
  V=1 (standard): bubble P/M, communication 2×P times per microbatch
  V=2: bubble P/(2M), communication 4×P times per microbatch
  V=4: bubble P/(4M), communication 8×P times per microbatch

In practice: V=2 or V=4 is common; larger V incurs too much communication overhead.
```

---

### 7.5 Interview FAQs

> [!question] What is the nature of pipeline bubbles, and how can we quantify them?
> Bubbles are GPU idles during pipeline warm-up/cool-down periods. Standard 1F1B bubble ratio = $(P-1)/(M+P-1)$. For bubbles < 5%, $M > 20(P-1) \approx 20P$ is required. For example, when PP=8, 160 micro-batches are required.

> [!question] What is the core advantage of 1F1B over GPipe?
> **Not bubbles (the two are similar), but activated memory**. GPipe needs to save the activation values ​​of M micro-batches at the same time ($O(M)$ memory), and in the steady state of 1F1B, only P activation values ​​are in flight ($O(P)$ memory). In large model training, usually $M \gg P$, saving activation memory is crucial.

> [!question] What data is exchanged between PP stages, and how large is it?
> Forward: activation value $[B/\text{dp}, S/\text{cp}, D]$, size $= B \cdot S \cdot D \cdot 2$ bytes (bf16). Reverse: Gradient of the same shape. This is typically only a few MB compared to the weight size (several GB), which is why PP can be interconnected with low bandwidth across nodes.

> [!question] How should I choose `M` (number of microbatches) and `P` (number of pipeline stages)?
> - Increase M: reduce bubbles, but the batch size of each micro-batch becomes smaller (may affect statistical efficiency)
> - Increase P: A larger model can be trained, but as the bubbles increase, M needs to be increased simultaneously
> - Rule of thumb: $M \geq 4P$ (make bubbles < 25%), typically $M = 8P$ to $M = 16P$

> [!question] What is the bubble formula for interleaved 1F1B, and what is its cost?
> Bubble ratio $(P-1)/(V \cdot M + P-1) \approx P/(VM)$, reduced by $V$ times. Cost: $V$ times more P2P communications per micro-batch (more stage boundaries). In practice $V = 2$ is a common choice, balancing bubbles and communication overhead.

### 7.6 1F1B Scheduling Pseudocode

```
# Scheduling logic for each stage (pseudocode)

warmup_steps = P - my_rank - 1   # smaller rank means longer warm-up

# Warm-up phase: forward only, fill the pipeline
for i in 0..warmup_steps:
    x = recv_forward() if not first_stage else microbatch[i]
    y = forward(x)
    send_forward(y) if not last_stage
    save(x, y)                   # save activations for backward

# Steady-state phase: alternate 1F1B
for i in warmup_steps..M:
    x = recv_forward() if not first_stage else microbatch[i]
    y = forward(x)
    send_forward(y) if not last_stage

    dy = recv_backward() if not last_stage else loss_grad
    dx = backward(saved_x, saved_y, dy)
    send_backward(dx) if not first_stage

# Cool-down phase: backward only, drain the pipeline
for i in 0..warmup_steps:
    dy = recv_backward() if not last_stage else loss_grad
    dx = backward(saved_x, saved_y, dy)
    send_backward(dx) if not first_stage

optimizer.step()

# Note: first_stage/last_stage get data or loss gradients directly
# Other stages only do recv/send and remain agnostic to the data source (modular design)
```

---

## 8. Hybrid Parallel Strategies

### 8.1 3D Parallelism

When actually training large models, multiple parallel strategies are usually combined:

```
┌──────────────────────────────────────────────────────────────┐
│                    3D Parallelism Diagram                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                    ┌─────────────────────────────────────┐   │
│                    │         Pipeline Parallel            │   │
│                    │  Stage 0        Stage 1              │   │
│                    │  ┌──────┐       ┌──────┐             │   │
│    Data Parallel   │  │      │──────→│      │             │   │
│    Replica 0       │  │      │       │      │             │   │
│                    │  └──────┘       └──────┘             │   │
│                    │  ↕ TP ↕         ↕ TP ↕               │   │
│                    │  ┌──────┐       ┌──────┐             │   │
│                    │  │      │       │      │             │   │
│                    │  └──────┘       └──────┘             │   │
│                    └─────────────────────────────────────┘   │
│                                                              │
│                    ┌─────────────────────────────────────┐   │
│                    │         Pipeline Parallel            │   │
│                    │  Stage 0        Stage 1              │   │
│                    │  ┌──────┐       ┌──────┐             │   │
│    Data Parallel   │  │      │──────→│      │             │   │
│    Replica 1       │  │      │       │      │             │   │
│                    │  └──────┘       └──────┘             │   │
│                    │  ↕ TP ↕         ↕ TP ↕               │   │
│                    │  ┌──────┐       ┌──────┐             │   │
│                    │  │      │       │      │             │   │
│                    │  └──────┘       └──────┘             │   │
│                    └─────────────────────────────────────┘   │
│                                                              │
│  Total GPUs = DP × PP × TP = 2 × 2 × 2 = 8                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 FSDP + TP Combination

The most commonly used combination is FSDP (data parallelism) + tensor parallelism:

$$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

**Advantages**:

- FSDP movement weight, TP movement activation value
- As TP increases, FSDP's AllGather becomes smaller (because activation values ​​are fragmented)
- As FSDP increases, TP’s AllGather becomes smaller (because the batch is fragmented)

### 8.3 Optimal Configuration

**Goal**: Minimize communication time, maintain computational bottlenecks

$$T_\text{FSDP comms} = \frac{4DF}{Y \cdot W \cdot M_X}$$

$$T_\text{TP comms} = \frac{4BD}{X \cdot W \cdot M_Y}$$

**Optimal FSDP size**:

$$X_\text{opt} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N}$$

where $N$ is the total number of GPUs.

> [!example] Configuration example
> For LLaMA-70B ($F \approx 28672$), $B = 2M$ tokens, $N = 64$ GPUs:
> 
> $$X_\text{opt} = \sqrt{\frac{2 \times 10^6}{28672} \cdot 1 \cdot 64} \approx 67$$
> 
> Select $X = 64$ (FSDP), $Y = 1$ (no TP)

### 8.4 Process Group Management in Picotron

Picotron uses `ProcessGroupManager` to uniformly manage 4D parallel process group allocation. The arrangement order of the devices is `DP → CP → TP → PP`. The coordinates of each rank can be directly calculated by taking the modulus of integer division; the communication group of each dimension is created by enumerating all combinations of other dimensions. For specific design details, see [[#9, Picotron Practical Combat: Building a Distributed Training Framework from Scratch]] §9.2.

---

## 8½. N-D Parallelism Panorama: Understanding Transformer Parallel Decomposition from a Single-GPU Perspective

> This section draws on Ailing Zhang's blog [Visualizing Parallelism in Transformer](https://ailzhang.github.io/posts/distributed-compute-in-transformer/), which offers an intuitive way to understand parallelism from the perspective of a **single GPU**. Earlier sections treated DP, FSDP, TP, and PP separately, but in real large-model training a single Transformer forward pass typically interleaves **5-6 parallel strategies at once**. This section unifies them.

### 8½.1 "Local Shape" Thinking: Seeing the World from Inside a Single GPU

The golden rule for understanding parallelism is: **imagine yourself inside a single GPU**. You only hold one shard of the global tensor, and you need to know:
- What is the shape of the data I currently hold?
- What am I missing to complete my computation, and who do I need to communicate with?

**Local activation shape formula**:

$$\text{local shape} = \left[\frac{B}{\text{dp}}, \; \frac{S}{\text{cp} \times \text{sp}}, \; D\right]$$

- $B / \text{dp}$: data parallelism shards the batch
- $S / (\text{cp} \times \text{sp})$: Context Parallel and Sequence Parallel jointly shard the sequence
- $D$: the hidden dimension stays intact (TP shards the weight's F dimension, not D)

Each parallel strategy cuts into different dimensions, which is why they can be combined orthogonally:

| Symbol | Parallel strategy | What is sharded? | Which layers does it apply to? |
|------|---------|-----------|-------------|
| dp | Data Parallel | Batch ($B$) | All layers |
| tp | Tensor Parallel | FFN/Head dimensions of weights ($F$, $n\_heads$) | Attention, MLP |
| sp | Sequence Parallel | Sequence dimension of activations ($S$), **only in element-wise operations** | LayerNorm, Dropout, residual connections |
| cp | Context Parallel | Sequence dimension ($S$), **in Attention QKV calculation** | Attention |
| ep | Expert Parallel | Number of experts ($E$) | MoE layers |
| vp | Vocab Parallel | Vocabulary Dimension ($V$) | Embedding, Loss |

### 8½.2 Sequence Parallel (SP): A natural partner for TP

**Problem**: Tensor Parallel (TP) can only shard matrix-multiplication operations, because matrix multiplication can be split along a dimension. But Transformers also contain many **element-wise operations** (LayerNorm, Dropout, residual additions, activation functions). These operations have very small weights or no weights at all, so TP is not worthwhile for them. In pure TP, such operations can only run on **full, unsharded activations**, which wastes memory.

**SP solution**: for element-wise operations, **shard activations along the sequence dimension**.

Key insight: element-wise operations are independent across positions, so each GPU can process only its local sequence slice with no communication.

```
TP region (matrix multiplication)   SP region (element-wise)
┌────────────────┐          ┌────────────────┐
│ Each GPU holds full S │       │ Each GPU holds S/sp │
│ Weights sharded by F  │  ←→   │ Activations sharded by S │
│ AllGather(sp)         │ convert │ ReduceScatter       │
└────────────────┘          └────────────────┘
```

**Converting between SP and TP**:
- **Entering the TP region** (for example, the matrix multiplies in Attention or the MLP): use `AllGather(sp)` to reconstruct the full sequence
- **Leaving the TP region** (back to LayerNorm and other element-wise ops): use `ReduceScatter` to both sum TP partials and scatter along the sequence dimension

The clever part is that TP's row-parallel layer already needs `ReduceScatter` to sum partial products. SP simply makes that same `ReduceScatter` **also perform sequence sharding**—one communication, two purposes, zero extra overhead.

### 8½.3 Context Parallel (CP): The savior of long sequences

**Problem**: The compute and memory complexity of Self-Attention is $O(S^2)$. When the sequence is very long (for example, 128K tokens), the $S \times S$ attention matrix is still huge even if other dimensions are sharded. SP can only shard the sequence in element-wise operations. Inside Attention, **each position must see all other positions**, so you cannot simply split along S.

**CP solution**: also shard along the sequence dimension inside Attention, but use **Ring Attention**—each GPU holds a `Query` slice of length $S/\text{cp}$ and completes full attention by **circulating KV blocks around a ring**.

```
GPU 0: Q[0:S/4]     GPU 1: Q[S/4:S/2]    GPU 2: Q[S/2:3S/4]   GPU 3: Q[3S/4:S]
      │                    │                     │                    │
      └── Ring: pass KV ──→ ──→ ──→ ──→ ──→ ──→ ──→ ──→ ──→ ──→ ──→─┘
```

At each step, each GPU computes local attention using its own Q and the KV block it currently holds, then forwards that KV block to the next GPU in the ring. After `cp` rounds, every GPU has seen the full KV set, and attention is complete.

**CP vs. SP**:
- **SP**: shards the sequence dimension for element-wise operations (LayerNorm, Dropout), with no cross-position communication
- **CP**: shards the sequence dimension for Attention, using Ring Attention to share KV across positions

Both are divided into $S$ dimensions, so they are multiplied in the local shape formula: $S / (\text{cp} \times \text{sp})$.

### 8½.4 Expert Parallel (EP): MoE’s exclusive parallelism

In **Mixture of Experts (MoE)** models, the MLP layer is replaced by multiple "experts" (each expert is an independent MLP), and each token activates only its top-k experts.

**EP approach**: place different experts on different GPUs.

```
Token routing (Router) decides which expert each token goes to
         │
    AllToAll(ep)         ← send tokens to the GPUs hosting the corresponding experts
         │
    Each GPU computes independently ← experts on each GPU process their assigned tokens
    (can use TP internally)
         │
    AllToAll(ep)         ← send the computed results back to the GPUs where the tokens originated
         │
    Continue subsequent computation
```

**Core communication**: two `AllToAll`s—one to send tokens to experts, and one to send results back. AllToAll is a "transpose" operation: input is sharded by token while output is sharded by expert, or vice versa.

**EP bottleneck**: AllToAll requires **pairwise communication among all GPUs** (unlike AllGather/ReduceScatter, which can be optimized with rings), so the network becomes the main bottleneck. This is why MoE models can have lower active parameter counts but still incur very high communication cost.

### 8½.5 Vocab Parallel (VP): Sharding of Embedding and Loss

LLM vocabularies are usually large (for example, LLaMA 3: 128K). The embedding table has size $V \times D$ (128K × 8K ≈ 1 GB in bf16), so fully replicating it on every GPU is inefficient.

**VP approach**: each GPU holds only a shard of the vocabulary, $[V/\text{vp}]$.

**Embedding forward**:
```
Input token IDs: [B, S]
         │
    Each GPU looks up its own embedding shard (tokens not owned by it get 0)
         │
    ReduceScatter         ← sum to obtain the full embedding while switching to SP sharding
         │
    Output: [B, S/sp, D]  ← enter the SP state
```

**Loss computation** (Cross-Entropy with VP):

Cross-Entropy requires a softmax over the full vocabulary, but each GPU has only $V/\text{vp}$ logits. Additional communication is therefore required to compute the global softmax denominator:

```
local logits: [B, S, V/vp]
         │
    AllReduce(max)        ← find the global maximum (numerical stability)
    AllReduce(sum)        ← compute the global exp-sum (softmax denominator)
         │
    Local log-softmax     ← complete locally using the global statistics
         │
    AllReduce(sum)        ← aggregate the loss
```

The original blog contains several excellent figures; see [Ailing Zhang's blog](https://ailzhang.github.io/posts/distributed-compute-in-transformer/) for the **complete Transformer parallelism panorama**, which shows the communication patterns across all layers:

![Complete Transformer parallel panorama: DP/TP/SP/CP/EP/VP interleaving](https://ailzhang.github.io/posts/distributed-compute-in-transformer/overview.svg)

## 9. Picotron Design Analysis

[Picotron](https://github.com/huggingface/picotron) is Hugging Face's educational 4D parallel framework. Its core design philosophy is: **each parallel strategy is an independent orthogonal dimension, and process groups compose them without interference**.

---

### 9.1 Core Abstraction: 4D Device Grid

All processes are arranged on a 4D grid, and each process has a unique coordinate `(dp, tp, pp, cp)`:

```
                    PP Stage 0          PP Stage 1
                ┌───────────────┐   ┌───────────────┐
                │  TP=0  TP=1   │   │  TP=0  TP=1   │
    DP replica 0│  GPU0  GPU1   │──▶│  GPU4  GPU5   │
                │               │   │               │
    DP replica 1│  GPU2  GPU3   │──▶│  GPU6  GPU7   │
                └───────────────┘   └───────────────┘
                      Node 0              Node 1
```

**Each type of parallelism corresponds to one slice of the grid:**

```
TP group = same row (same dp, pp, cp; different tp) → high bandwidth within a node
DP group = same column (same tp, pp, cp; different dp) → cross-node
PP group = depth direction (same dp, tp, cp; different pp) → point-to-point
```

**Rank mapping formula** (Picotron convention: `dp → cp → tp → pp`):

```
global_rank = dp * (cp*tp*pp) + cp * (tp*pp) + tp * pp + pp_rank
```
### 9.2 Process Group Manager (ProcessGroupManager)

When a process starts, it computes its 4 coordinates from its `global_rank` and joins the corresponding communication subgroups:

```
# Pseudocode
class ProcessGroupManager:
    def __init__(dp, tp, pp, cp):
        rank = dist.get_rank()

        # Decode coordinates
        self.dp_rank = rank // (cp*tp*pp)
        self.cp_rank = (rank // (tp*pp)) % cp
        self.tp_rank = (rank // pp) % tp
        self.pp_rank = rank % pp

        # Create subgroup communicators for each dimension
        self.dp_group = new_group([ranks with the same (tp,pp,cp) position])
        self.tp_group = new_group([ranks with the same (dp,pp,cp) position])
        self.pp_group = new_group([ranks with the same (dp,tp,cp) position])
```

After that, the process only needs to use `pgm.tp_group`, `pgm.dp_group`, and so on, without reasoning about global ranks directly.
### 9.3 TP Design: Model Surgery

TP does not modify the training loop. Instead, it **replaces the model's linear layers** so that each layer natively supports sharded computation:

```
Original model                    After TP transformation
─────────────────────────────────────────────────────────
Linear(D → F)          →    ColumnParallelLinear(D → F/tp)
                                  (no communication, output is naturally sharded)

Linear(F → D)          →    RowParallelLinear(F/tp → D)
                                  (ReduceScatter summation)
─────────────────────────────────────────────────────────
```

**Forward data flow (using `tp=4` as an example):**

```
Input x[B, D] ──────────────────────────────────────────
              ↓ Replicated to 4 GPUs (or AllGathered from SP)
  GPU0: x[B,D]   GPU1: x[B,D]   GPU2: x[B,D]   GPU3: x[B,D]
      │               │               │               │
      ▼ W_in[D,F/4]   ▼               ▼               ▼
  [B, F/4]        [B, F/4]        [B, F/4]        [B, F/4]
      │               │               │               │
      ▼ W_out[F/4,D]  ▼               ▼               ▼
  [B, D]{U}       [B, D]{U}       [B, D]{U}       [B, D]{U}
      └───────────────┴───────────────┴───────────────┘
                          ReduceScatter
                              ↓
                      [B, D/4] (continue in the SP state)
```

**Layer replacement pseudocode:**
```
# Iterate over all model layers and replace in place
for layer in model.layers:
    layer.mlp.gate_proj  = ColumnParallel(original_layer)   # column parallel
    layer.mlp.up_proj    = ColumnParallel(original_layer)
    layer.mlp.down_proj  = RowParallel(original_layer)      # row parallel
    layer.attn.q/k/v     = ColumnParallel(original_layer)
    layer.attn.o_proj    = RowParallel(original_layer)
```
### 9.4 DP Design: Bucketed Gradient AllReduce

Naive approach: wait for the backward pass of all layers to finish, then AllReduce all gradients at once → communication and computation are completely serialized.

Picotron's approach: **group parameters into buckets by size, and trigger asynchronous AllReduce** as soon as a bucket's gradients are ready during backward, overlapping communication with the backward compute of later layers:

```
Backward order (from back to front):

Time ──────────────────────────────────────────────────────────────▶

Layer N backward: ████████
              └──▶ Bucket 3 AllReduce: ░░░░░░░░░░░░
Layer N-1 backward:         ████████
                        └──▶ Bucket 2 AllReduce: ░░░░░░░░░░░░
Layer N-2 backward:                     ████████
                                    └──▶ Bucket 1 AllReduce: ░░░░░░
                                                     ↑
                                              ████ = compute
                                              ░░░░ = communication (async)
```

**Key design: Bucketing in reverse order** (the parameters of the last layer are placed in the first Bucket) to ensure that communication can be triggered immediately as soon as the gradient is generated.

```
# Pseudocode
params_reversed = reversed(model.parameters())  # backward order
for param in params_reversed:
    current_bucket.add(param)
    if current_bucket.size > BUCKET_SIZE:
        create_next_bucket()

# Register hooks: trigger when gradients are ready
for param in bucket:
    param.register_hook(lambda:
        if bucket.is_full(): async AllReduce(bucket.grads)
    )
```

---

### 9.5 PP Design: Stage Partitioning + 1F1B Scheduling

**Stage partitioning**: distribute the model's `L` layers evenly across `pp` stages, so each process stores only `L/pp` layers:

```
32-layer model, pp=4:

Stage 0 (GPU0): Layer  0-7   ──send_fwd──▶
Stage 1 (GPU1): Layer  8-15             ──send_fwd──▶
Stage 2 (GPU2): Layer 16-23                         ──send_fwd──▶
Stage 3 (GPU3): Layer 24-31  (compute loss)
                                         ◀──send_bwd──
                             ◀──send_bwd──
                 ◀──send_bwd──
```

Only activation values ​​(forward) and gradients (reverse) are transferred between processes, and the amount of data = `[B/dp, S/cp, D]`, which is much smaller than the weight.

**1F1B scheduling** (see Chapter 7 §7.3): in steady state, each stage alternates between forward and backward work to minimize GPU idle bubbles. Picotron implements this scheduler directly; PP users only need to provide `forward_step` and `loss_fn`.

---

### 9.6 Orthogonality of the Four Parallel Dimensions

```
┌─────────────┬──────────────┬────────────────┬──────────────────┐
│             │ What is split│ Where comm happens│ Comm frequency │
├─────────────┼──────────────┼────────────────┼──────────────────┤
│ DP          │  Batch (B)   │  Within DP group │  Once per step   │
│             │              │  AllReduce ∇W   │  (after backward) │
├─────────────┼──────────────┼────────────────┼──────────────────┤
│ TP          │  Weight dims │  Within TP group │  Twice per layer │
│             │  (F, heads)  │  AllGather/RS   │  (per matmul)    │
├─────────────┼──────────────┼────────────────┼──────────────────┤
│ PP          │  Model depth │ Between adjacent │  Per microbatch  │
│             │              │  P2P Send/Recv  │                  │
├─────────────┼──────────────┼────────────────┼──────────────────┤
│ CP          │  Sequence S  │  Within CP group │  Once per attn layer │
│             │              │  Ring KV passing│                  │
└─────────────┴──────────────┴────────────────┴──────────────────┘

Placement principles:
  TP → within node (most frequent communication, needs high-bandwidth NVLink)
  PP → across nodes (least communication, InfiniBand is sufficient)
  DP → anywhere (AllReduce can overlap with backward)
  CP → prefer within node (Ring Attention is latency-sensitive)
```

---

## 10. Summary and Best Practices

### 10.1 Parallel Strategy Selection Guide

```
┌───────────────────────────────────────────────────────────────┐
│                  Parallel Strategy Decision Tree             │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Does the model fit on a single GPU?                         │
│       │                                                       │
│       ├── Yes → use data parallelism (DDP)                   │
│       │                                                       │
│       └── No → model parameters + optimizer > single-GPU memory? │
│                    │                                          │
│                    ├── Yes → use FSDP (ZeRO-3)               │
│                    │        │                                 │
│                    │        └── per-GPU batch size > C/W?    │
│                    │                  │                       │
│                    │                  ├── Yes → pure FSDP    │
│                    │                  │                       │
│                    │                  └── No → FSDP + TP     │
│                    │                                          │
│                    └── No → use tensor parallelism (TP)      │
│                                                               │
│  Multi-node training?                                        │
│       │                                                       │
│       └── Consider adding pipeline parallelism (PP)          │
│           - Place PP across nodes (low bandwidth)            │
│           - Place TP within nodes (high-bandwidth NVLink)    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

References

**Sharding and Communication Primitives**
- Austin et al. (2025) [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) — the main reference for Chapters 2 and 3 of this tutorial, with a systematic treatment of sharded matrix multiplication
- Gibiansky (2017) [Bringing HPC Techniques to Deep Learning](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) — the Ring AllReduce algorithm and the Chapter 2 communication-primitives background

**Data Parallelism / FSDP**
- Rajbhandari et al. (2020) [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — ZeRO-1/2/3, foundational for Chapter 5
- Zhao et al. (2023) [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) — PyTorch FSDP implementation details

**Tensor Parallelism / Sequence Parallelism**
- Shoeybi et al. (2019) [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — column parallelism + row parallelism, foundational for Chapter 6
- Korthikanti et al. (2022) [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Sequence Parallel (SP) combined with TP; referenced in Chapter 8½'s SP discussion

**Pipeline Parallelism**
- Huang et al. (2019) [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) — GPipe / AFAB scheduling
- Narayanan et al. (2021) [Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2104.04473) — 1F1B scheduling, referenced in Chapter 7

**Context Parallel / Ring Attention**
- Liu et al. (2023) [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) — referenced in Chapter 8½'s CP discussion

**Collective Matmul (Communication-Computation Overlap)**
- Wang et al. (2022) [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://dl.acm.org/doi/10.1145/3567955.3567959) — referenced in Chapter 3 on collective matmul

### Code Implementations
- [Picotron](https://github.com/huggingface/picotron) — the educational 4D parallel framework referenced throughout this tutorial
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — NVIDIA official TP/PP implementation
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) — ZeRO series implementation
- [PyTorch DTensor](https://pytorch.org/docs/stable/distributed.tensor.html) — PyTorch sharded tensor API (used in the Chapter 3 code examples)
- [Mosaic GPU Collective Matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html) — JAX Pallas collective matmul implementation

### Online Resources
- [How To Scale Your Model (JAX Scaling Book)](https://jax-ml.github.io/scaling-book/) — the primary reference for this tutorial
- [Visualizing Parallelism in Transformer](https://ailzhang.github.io/posts/distributed-compute-in-transformer/) — Ailing Zhang (Meta PyTorch), the main reference for Chapter 8½, including six SVG overviews covering overview / embedding / attention / mlp / moe / loss
- [Picotron Tutorial Playlist](https://www.youtube.com/playlist?list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S) — accompanying video tutorial

---
