# 08 · Distributed Training Parallelism Paradigms & NCCL Communication

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
12. [[#12. References & Further Reading]]

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

In distributed deep learning, individual computing accelerators (GPUs or TPUs) hold local shards of model parameters, gradients, optimizer states, or activation tensors. To execute end-to-end training and inference, hardware devices must cooperatively exchange data. Modern deep learning frameworks orchestrate this distributed communication via a set of fundamental building blocks known as **collective communication primitives**.

In distributed systems engineering and performance modeling, evaluating whether a parallelization strategy is scalable hinges on one critical task: **accurately modeling the runtime cost of each collective primitive on the physical network topology, and comparing it against local compute time (GEMM / Attention) to determine whether execution is compute-bound or communication-bound**.

The analytical cost models below are derived on the standard **logical ring topology (or multi-dimensional torus)**—the foundational model established in Baidu Ring-AllReduce and formalized in Google DeepMind's *How to Scale Your Model* (Austin et al., 2025), which also underlies NVIDIA NCCL's ring-based implementations.

#### 2.3.1 AllGather

**Function**: Each device begins with a local tensor shard; upon completion, every device holds the reconstructed full global tensor.

```
Device 0: [A0]     →   Device 0: [A0, A1, A2, A3]
Device 1: [A1]     →   Device 1: [A0, A1, A2, A3]
Device 2: [A2]     →   Device 2: [A0, A1, A2, A3]
Device 3: [A3]     →   Device 3: [A0, A1, A2, A3]
```

**Notation**: $\text{AllGather}_X([A_X, B]) \rightarrow [A, B]$ (gather along sharded dimension $A$ across device grid axis $X$).

**Primary Use Cases**:
1. **FSDP / ZeRO-3**: Before executing forward pass for a specific Transformer block, the parameters sharded across data-parallel ranks are gathered just-in-time into a complete weight matrix, and freed immediately after computation.
2. **Tensor Parallelism (TP) / Sequence Parallelism (SP)**: Megatron sequence parallelism gathers sequence shards into full-length sequences before computing self-attention or MLP projections.

**Cost Model and Step-by-Step Derivation**:

Let $N$ be the number of participating devices (ranks), and $V$ be the total size of the final gathered tensor in bytes. Prior to communication, each device holds a local shard of size $V/N$.

1. **Uni-directional Ring Derivation**:
   Arrange $N$ devices into a logical ring. In each hop (step), device $i$ sends its current shard (size $V/N$) to its clockwise neighbor while receiving a new shard from its counter-clockwise neighbor.
   To ensure all $N$ devices receive all $N$ shards, exactly $N-1$ steps are required.
   If the uni-directional link bandwidth is $W_\text{uni}$, each hop takes $\frac{V/N}{W_\text{uni}}$. For large $N$, $N-1 \approx N$:
   $$T_\text{uni} = (N - 1) \cdot \frac{V/N}{W_\text{uni}} \approx N \cdot \frac{V/N}{W_\text{uni}} = \frac{V}{W_\text{uni}}$$

2. **Bidirectional Ring / Torus Derivation**:
   Modern hardware interconnects (NVIDIA NVLink, Google TPU ICI, PCIe full-duplex) support simultaneous bidirectional full-duplex transmission. To saturate total bidirectional link bandwidth $W_\text{bidir}$ (bandwidth per direction is $W_\text{bidir}/2$), communication libraries split each device's shard into two halves of size $\frac{V}{2N}$, transmitting one half clockwise and the other counter-clockwise.
   Both directions require $(N-1)/2 \approx N/2$ hops to cover the ring:
   $$T_\text{hop} = \frac{V / (2N)}{W_\text{bidir} / 2} = \frac{2V}{N \cdot W_\text{bidir}}$$
   Across $N/2$ hops, the total communication time evaluates to:
   $$T_\text{total} = \frac{N}{2} \cdot T_\text{hop} = \frac{N}{2} \cdot \frac{2V}{N \cdot W_\text{bidir}} = \frac{V}{W_\text{bidir}}$$

> [!important] Key Insight: AllGather time is independent of device count $N$ in the bandwidth-bound regime
> In the bandwidth-saturated regime, $N$ strictly cancels out of the formula! **The total AllGather transfer duration depends exclusively on the full tensor volume $V$ and the link bandwidth $W$, completely independent of the number of participating devices $N$**.
> Although increasing $N$ scales up hop count proportionally, the data transferred per step ($V/N$) shrinks by the exact same factor, resulting in zero net change in total transfer time.

3. **Latency-bound Correction**:
   The derivation above assumes each transmission hop is dominated by raw wire transfer time. In physical networks, every cross-device transfer incurs fixed software protocol stack and hardware synchronization latency $T_\text{min}$ ($\sim 0.5 - 1\,\mu\text{s}$ for NVLink, $\sim 1.5 - 3\,\mu\text{s}$ for InfiniBand, $\sim 1\,\mu\text{s}$ for TPU ICI).
   When the total tensor $V$ is small, or $N$ is very large such that the per-hop payload $\frac{2V}{N}$ drops to just a few kilobytes, the transfer time falls below $T_\text{min}$. The step duration becomes bottlenecked by latency:
   $$T_\text{hop} = \max\!\left[ T_\text{min},\ \frac{2V}{N \cdot W} \right] \quad \Rightarrow \quad T_\text{total} = \max\!\left[ \frac{T_\text{min} \cdot N}{2},\ \frac{V}{W} \right]$$
   - **Latency Crossover Threshold**: Setting $\frac{T_\text{min} \cdot N}{2} = \frac{V}{W}$ yields the critical shard size $\frac{V}{N} = \frac{T_\text{min} \cdot W}{2}$.
   - **Hardware Comparisons**:
     - **TPU v5e** ($W = 4.5 \times 10^{10}\text{ B/s},\ T_\text{min} \approx 1\,\mu\text{s}$): Crossover threshold $\approx 22.5 - 45\text{ kB}$. Shards smaller than this fall into the latency-bound regime.
     - **NVIDIA H100 SXM5** ($W = 900\text{ GB/s},\ T_\text{min} \approx 0.8\,\mu\text{s}$): Crossover threshold $\approx 360\text{ kB}$.
     - **Systems Engineering Takeaway**: Never trigger fine-grained AllGather calls on tiny tensors individually. Small tensors must be fused into multi-megabyte buckets (Bucket Fusion) to force execution into the bandwidth-bound regime.

4. **Multi-axis Parallel AllGather**:
   In multi-dimensional topologies (2D/3D Torus or intra-node NVLink + inter-node InfiniBand), device grids span multiple orthogonal coordinate axes $\{X_1, X_2, \ldots\}$. When tensors are sharded and gathered across multiple mesh axes concurrently, transfers utilize physically independent link channels:
   $$T_\text{total} = \max\!\left[ \frac{T_\text{min} \cdot \sum |X_i|}{2},\ \frac{V}{W \cdot N_\text{axes}} \right]$$
   Effective bandwidth scales linearly with the number of concurrent axes $N_\text{axes}$.

![AllGather measured bandwidth (TPU v5e 8×16): about 95% peak above 10 MB](https://jax-ml.github.io/scaling-book/assets/img/all-gather-bandwidth.png)

> [!example] Numerical AllGather Estimation Examples
>
> Grid setup: TPU v5e, `{'X': 8, 'Y': 4}`, ICI bidirectional bandwidth $W = 4.5 \times 10^{10}\text{ B/s}$.
>
> **(a) Scenario A (Bandwidth-bound)**: `AllGather_Y([E_Y, F])`, $E = 2048$, $F = 8192$, bfloat16.
> - Shard per device: `bf16[512, 8192]` = 8.4 MB, total array $V = 33.6\text{ MB}$.
> - Transfer time (bandwidth-bound): $T = 33.6\text{ MB} / (4.5 \times 10^{10}\text{ B/s}) \approx 747\,\mu\text{s}$ (measured hardware runtime $\approx 680\,\mu\text{s}$).
>
> **(b) Scenario B (Latency-bound)**: Same network setup, reduced shape $E = 256$, $F = 256$.
> - Shard per device: `bf16[64, 256]` = 32 kB < 45 kB threshold → **Latency-bound**.
> - Theoretical time: $T \approx T_\text{min} \times (Y/2) = 1\,\mu\text{s} \times 2 = 2\,\mu\text{s}$ (measured runtime dominated by scheduling overhead at $\approx 8\,\mu\text{s}$).

#### 2.3.2 ReduceScatter

**Function**: All devices start with local tensors of identical shape (e.g., local gradient contributions computed during backpropagation). The operation elementwise reduces (sums) them and scatters equal-sized slices to each device, leaving each rank with only one shard of the globally reduced tensor.

```
Device 0: [A0, B0, C0, D0]   →   Device 0: [A0+A1+A2+A3]
Device 1: [A1, B1, C1, D1]   →   Device 1: [B0+B1+B2+B3]
Device 2: [A2, B2, C2, D2]   →   Device 2: [C0+C1+C2+C3]
Device 3: [A3, B3, C3, D3]   →   Device 3: [D0+D1+D2+D3]
```

**Notation**: $\text{ReduceScatter}_{X,K}([A, K]\{U_X\}) \rightarrow [A, K_X]$ (reduce over unreduced dimension $K$ across axis $X$ and shard into $K_X$).

**Cost Model**: Identical to AllGather ($T = \frac{V}{W_\text{bidir}}$), as the ring communication steps and data volumes are exactly symmetric.

> [!note] Mathematical Duality with AllGather (Kronecker Perspective)
> Defining broadcast $\text{broadcast} = \mathbf{u} \otimes I_n$ and reduce $\text{reduce} = \mathbf{u}^T \otimes I_n$ ($\mathbf{u} = (1,\ldots,1)^T$):
> - $\text{AllGather} = \text{broadcast} \otimes I_p$
> - $\text{ReduceScatter} = \text{reduce} \otimes I_p$
>
> Since $(\mathbf{u} \otimes I_n)^T = \mathbf{u}^T \otimes I_n$, we have $\text{AllGather}^T = \text{ReduceScatter}$.
>
> **Physical Consequence**: **The backward adjoint derivative of an AllGather forward operation is mathematically a ReduceScatter**, and vice versa. In FSDP, gathering weights during forward automatically duals into reduce-scattering gradients during backward.

#### 2.3.3 AllReduce

**Function**: Elementwise sums data across all devices and leaves the fully reduced global tensor replicated across all devices.

```
Device 0: [A0]   →   Device 0: [A0+A1+A2+A3]
Device 1: [A1]   →   Device 1: [A0+A1+A2+A3]
Device 2: [A2]   →   Device 2: [A0+A1+A2+A3]
Device 3: [A3]   →   Device 3: [A0+A1+A2+A3]
```

**Notation**: $\text{AllReduce}_{X}([A_X, B]\{U_Y\}) \rightarrow [A_X, B]$

> [!important] Ring AllReduce Two-Stage Decomposition
> Ring AllReduce is canonically factored into two consecutive stages:
> $$\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$$
> 1. **Stage 1 (ReduceScatter)**: Reduce and scatter shards across devices ($T_1 = \frac{V}{W_\text{bidir}}$).
> 2. **Stage 2 (AllGather)**: Gather and broadcast the reduced shards to all ranks ($T_2 = \frac{V}{W_\text{bidir}}$).
> 
> Therefore, AllReduce takes exactly twice the duration of an AllGather:
> $$T_\text{AllReduce} = \frac{2V}{W_\text{bidir}}$$

#### 2.3.4 AllToAll

**Function**: Transposes the sharding dimensions (dimension exchange). Each device splits its data into $N$ equal blocks and sends block $j$ directly to device $j$.

```
Device 0: [A0, B0]   →   Device 0: [A0, A1]
Device 1: [A1, B1]   →   Device 1: [B0, B1]
```

**Notation**: $\text{AllToAll}_{X, J}([A, B_X]) \rightarrow [A_X, B]$

**Primary Use Cases**:
1. **Mixture-of-Experts (MoE) Expert Parallelism**: Routers dispatch tokens to designated expert GPUs and receive processed representations back via AllToAll.
2. **Hybrid Parallelism Axis Transformation**: Transposing between sequence-parallel and tensor-parallel sharding formats.

**Cost Analysis (Bidirectional Ring)**:
- In AllGather, each shard traverses all $N-1$ other nodes to cover the entire ring.
- In AllToAll, device $i$'s shard only travels to target device $j$. Along a bidirectional ring with shortest-path routing, the average travel distance is only $N/4$ hops (half of AllGather's $N/2$).
- With bidirectional concurrency, the total transfer time is approximately 1/4 of AllGather:
  $$T_\text{AllToAll} = \frac{T_\text{AllGather}}{4} = \frac{V}{4W_\text{bidir}}$$

#### 2.3.5 Summary of Communication Operations

| Operation | Core Function | Notation | Bidirectional Ring Runtime | Typical Parallel Paradigm |
|:---|:---|:---|:---|:---|
| **AllGather** | Gather shards into full tensor | $[A_X, B] \rightarrow [A, B]$ | $\frac{V}{W_\text{bidir}}$ | FSDP weight reconstruction, SP sequence concat |
| **ReduceScatter** | Reduce elementwise & scatter shards | $[A, B]\{U_X\} \rightarrow [A_X, B]$ | $\frac{V}{W_\text{bidir}}$ | FSDP backward gradient reduce, TP column-to-row |
| **AllReduce** | Global reduction replicated everywhere | $[A, B]\{U_X\} \rightarrow [A, B]$ | $\frac{2V}{W_\text{bidir}}$ | DDP gradient sync, Megatron TP output combine |
| **AllToAll** | Transpose sharding dimensions | $[A, B_X] \rightarrow [A_X, B]$ | $\frac{V}{4W_\text{bidir}}$ | MoE token routing / dispatch, 2D grid transform |

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

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X] \quad \text{(Dimension Conflict, Invalid)}$$

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

$$\text{local shape} = \left[ \frac{B}{\text{dp}}, \; \frac{S}{\text{cp} \times \text{sp}}, \; D \right]$$

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

---

## 11. Exercises

<details class="exercise">
<summary><span class="q-label">Q1</span> <span class="q-text">Why cannot a 70B parameter model fit on a single GPU for training?</span></summary>

Storing 70B parameters in BF16 alone requires $\sim 140\text{ GB}$, exceeding the memory of an 80 GB H100. During mixed-precision AdamW training, optimizer states require FP32 master weights (4 bytes), first momentum $m$ (4 bytes), and second momentum $v$ (4 bytes), totaling 12 bytes/param. Along with gradients (2 to 4 bytes/param), static model states demand $16 - 18\text{ bytes/param} \times 70\text{B} \approx 1.12 - 1.26\text{ TB}$. Adding dynamic activation memory and workspace buffers places the model far beyond single-device capacity, mandating FSDP/ZeRO-3, Tensor Parallelism, Pipeline Parallelism, or hybrid schemes.

</details>

<details class="exercise">
<summary><span class="q-label">Q2</span> <span class="q-text">Why is Tensor Parallelism (TP) strictly suited for intra-node deployment?</span></summary>

Tensor Parallelism partitions matrix multiplications within individual Transformer layers. Every forward and backward pass through a single layer requires collective communication (AllReduce or ReduceScatter + AllGather). This results in extraordinarily high communication frequency and microsecond-level latency sensitivity.
Only intra-node NVLink / NVSwitch fabrics (delivering 900 GB/s to 1.8 TB/s bidirectional bandwidth and sub-microsecond latency) can sustain such frequent synchronization without stalling CUDA cores. Running TP across nodes over InfiniBand introduces latency penalties that cause massive compute idle time.

</details>

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">What states do ZeRO-1, ZeRO-2, and ZeRO-3 (FSDP) shard respectively?</span></summary>

The ZeRO family partitions model training states along the data-parallel dimension:
- **ZeRO-1**: Shards only the **optimizer states**. Model parameters and gradients remain fully replicated across all GPUs.
- **ZeRO-2**: Shards **gradients** in addition to optimizer states. Each rank retains only the gradient slice corresponding to its optimizer partition.
- **ZeRO-3 (FSDP)**: Shards **model parameters** as well. Ranks store only their assigned parameter shard, materializing full weights temporarily via AllGather before layer execution and freeing them immediately afterward.

| Stage | Parameters (P) | Gradients (G) | Optimizer States (OS) | Communication Overhead | Memory Reduction |
|:---|:---|:---|:---|:---|:---|
| **DDP** | Replicated | Replicated | Replicated | Baseline (one backward AllReduce) | None |
| **ZeRO-1** | Replicated | Replicated | Sharded | Equal to DDP | $\sim 4\times$ parameter memory saved |
| **ZeRO-2** | Replicated | Sharded | Sharded | Equal to DDP | Gradients + optimizer states saved |
| **ZeRO-3 / FSDP** | Sharded | Sharded | Sharded | $\approx 1.5\times$ DDP | Near-linear reduction across all states |

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">ZeRO memory accounting for a 7B model on 8 GPUs</span></summary>

Assuming mixed-precision AdamW (BF16 parameters: 2 bytes, FP32 gradients: 4 bytes, FP32 master weights + $m$ + $v$: 12 bytes), the static model states for a 7B parameter model evaluate to:

```text
Parameters (BF16):       7B * 2  = 14 GB
Gradients (FP32):        7B * 4  = 28 GB
Optimizer States:        7B * 12 = 84 GB
DDP Per-GPU Baseline:              126 GB
```

When sharding across 8 GPUs (DP = 8):
- **ZeRO-1**: Optimizer states sharded ($84 / 8 = 10.5\text{ GB}$). Total = $14 + 28 + 10.5 = \mathbf{52.5\text{ GB}}$.
- **ZeRO-2**: Gradients ($28 / 8 = 3.5\text{ GB}$) and optimizer states sharded. Total = $14 + 3.5 + 10.5 = \mathbf{28\text{ GB}}$.
- **ZeRO-3**: All three sharded (parameters: $14 / 8 = 1.75\text{ GB}$, gradients: $3.5\text{ GB}$, optimizer: $10.5\text{ GB}$). Total = $\mathbf{15.75\text{ GB}}$.

(Note: Excludes dynamic activations, temporary AllGather buffers, and CUDA workspace).

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">Why does FSDP still require AllGather if parameters are sharded?</span></summary>

Standard linear layers perform dense GEMMs between the local batch activations and the layer weight matrix. Computing the correct local activation outputs requires the complete weight matrix. Since FSDP stores only $1/N$ of the layer's weights permanently, it must perform an AllGather immediately before layer computation to assemble the full weight matrix. Once the layer's GEMM completes, the full weight buffer is instantly freed, keeping peak memory bounded by just-in-time materialization.

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">Why is FSDP communication volume comparable to DDP?</span></summary>

In DDP, backward execution performs a full-gradient AllReduce:
$$\text{Comm}_\text{DDP} = 2 \cdot V_\text{grad}$$

In FSDP (ZeRO-3):
1. Forward AllGather of layer weights: $V_\text{param}$
2. Backward AllGather of layer weights: $V_\text{param}$
3. Backward ReduceScatter of layer gradients: $V_\text{grad}$

Assuming uniform 16-bit precision ($V_\text{param} = V_\text{grad} = V$):
$$\text{Comm}_\text{FSDP} = 3V = 1.5 \cdot \text{Comm}_\text{DDP}$$
FSDP incurs only $1.5\times$ the communication volume of DDP. Furthermore, because communication is broken down into per-layer AllGathers and ReduceScatters, it overlaps naturally with adjacent layer computations via prefetching streams.

</details>

<details class="exercise">
<summary><span class="q-label">Q7</span> <span class="q-text">How to choose the optimal wrapping granularity for FSDP?</span></summary>

Wrapping granularity balances peak memory usage against communication scheduling overhead:
- **Too coarse (e.g., one wrap for the entire model)**: AllGather reconstructs all model parameters upfront, eliminating memory savings.
- **Too fine (e.g., wrapping every single Linear layer)**: Triggers thousands of tiny collective operations, incurring severe kernel launch and network latency penalties ($T_\text{min}$).
- **Best Practice (Per Transformer Block)**: Wrapping at the Transformer decoder layer boundary provides the sweet spot. Each block contains tens to hundreds of megabytes—saturating link bandwidth—while enabling seamless prefetching of block $i+1$ during block $i$'s computation.

</details>

<details class="exercise">
<summary><span class="q-label">Q8</span> <span class="q-text">Do FSDP and activation checkpointing solve the same problem?</span></summary>

No. They target orthogonal memory bottlenecks:
- **FSDP (ZeRO-3)** targets **static model states** (parameters, gradients, optimizer states), which scale linearly with parameter count $P$ and are invariant to batch size or sequence length.
- **Activation Checkpointing** targets **dynamic forward activations**, which scale with batch size $B$, sequence length $S$, and hidden size $H$. It drops intermediate activations during forward and recomputes them on-demand during backward at the expense of $\sim 30\% - 33\%$ compute overhead.

Production LLM training combines both: FSDP accommodates hundred-billion-scale model weights, while activation checkpointing prevents OOMs during long-context training.

</details>

<details class="exercise">
<summary><span class="q-label">Q9</span> <span class="q-text">When is pure FSDP insufficient, requiring Tensor Parallelism (TP)?</span></summary>

TP is required when:
1. **Single-layer GEMM weights or activations exceed single-GPU memory**: FSDP must materialize full weights for at least one layer. If an enormous hidden dimension or vocabulary projection cannot fit on one GPU, FSDP cannot run.
2. **Tiny Batch Size ($B=1$ inference or RLHF rollout generation)**: FSDP requires per-GPU batch size $> C/W$ to hide communication behind GEMM compute. At $B=1$, FSDP becomes heavily communication-bound, whereas TP splits the weight dimensions to parallelize compute directly.
3. **Latency-critical single-step execution**: TP reduces per-layer wall-clock time by distributing GEMM arithmetic across multiple GPUs concurrently.

</details>

<details class="exercise">
<summary><span class="q-label">Q10</span> <span class="q-text">What are the engineering trade-offs in FSDP checkpointing?</span></summary>

FSDP partitions model state across GPUs, presenting two checkpointing strategies:
1. **Full State Dict**: All ranks gather their shards to Rank 0 (or CPU RAM) to output a single consolidated checkpoint file.
   - *Pros*: Universal compatibility with downstream inference and evaluation scripts.
   - *Cons*: High risk of Rank 0 OOM and long save times that stall training.
2. **Sharded State Dict**: Each GPU writes its local parameter and optimizer shards directly to distributed storage in parallel (e.g., PyTorch Distributed Checkpoint).
   - *Pros*: Scalable, fast, asynchronous saving with zero rank bottlenecks.
   - *Cons*: Checkpoints depend on cluster topology metadata; changing GPU count requires resharding utilities.

</details>

<details class="exercise">
<summary><span class="q-label">Q11</span> <span class="q-text">Which dimensions do TP, PP, EP, and CP shard in the Transformer computational graph?</span></summary>

| Paradigm | Dimension Sharded | Core Collective | Primary Bottleneck Solved |
|:---|:---|:---|:---|
| **TP (Tensor Parallel)** | Hidden / Channel dimensions ($H, 4H$) | AllReduce / ReduceScatter + AllGather | Oversized single layers & GEMM latency |
| **PP (Pipeline Parallel)** | Model depth (Transformer layers $L$) | P2P Send / Recv | Cross-node model capacity limits |
| **EP (Expert Parallel)** | MoE expert pool (Expert ID $E$) | AllToAll (dynamic token dispatch/combine) | Massive parameter scaling in sparse MoEs |
| **CP (Context Parallel)** | Sequence / context dimension ($S$) | Ring-Attention P2P or AllGather | Quadratic $O(S^2)$ attention memory on 128k+ contexts |

</details>

<details class="exercise">
<summary><span class="q-label">Q12</span> <span class="q-text">Why does Megatron MLP use column-parallel followed by row-parallel linear layers?</span></summary>

In an MLP layer: $Y = \text{GELU}(X \cdot W_\text{in}) \cdot W_\text{out}$.
1. **Column-Parallel First Layer**: $W_\text{in}$ is partitioned by columns into $[W_{\text{in}, 1}, W_{\text{in}, 2}]$. Each rank computes $Z_i = \text{GELU}(X \cdot W_{\text{in}, i})$. Since GELU is an elementwise non-linear function, each GPU computes its activation shard locally without any cross-device communication.
2. **Row-Parallel Second Layer**: $W_\text{out}$ is partitioned by rows. Each rank multiplies its local activation shard $Z_i$ by its weight row slice $W_{\text{out}, i}$.
3. **Single Output AllReduce**: Summing the partial results $Y = \sum Y_i$ requires only a single AllReduce at the very end of the block.

This pairing delays cross-GPU communication until after the entire MLP, avoiding communication on the widest intermediate activation dimension ($4H$).

</details>

<details class="exercise">
<summary><span class="q-label">Q13</span> <span class="q-text">Why does Pipeline Parallelism incur bubbles, and how can they be mitigated?</span></summary>

Pipeline bubbles occur due to sequential data dependencies across stages:
- Early microbatches must progress forward before downstream stages can begin execution.
- Downstream stages must complete forward execution and compute loss before upstream stages can receive backward gradients.

For GPipe (AFAB schedule), the bubble fraction is $F_\text{bubble} = \frac{P-1}{M+P-1}$.
Mitigation techniques:
1. **Increase microbatches ($M \gg P$)**: Dilutes idle bubble time, but increases activation memory.
2. **1F1B Scheduling**: Interleaves one forward step with one backward step once steady-state is reached, bounding in-flight activations to $P$ microbatches.
3. **Interleaved 1F1B**: Assigns multiple non-contiguous virtual stages per physical GPU to further shrink bubble overhead.

</details>

<details class="exercise">
<summary><span class="q-label">Q14</span> <span class="q-text">How does Expert Parallelism (EP) differ from Tensor Parallelism (TP)?</span></summary>

- **Sharding Logic**: TP shards dense matrices uniformly; every token participates in every sharded GEMM across all TP GPUs. EP shards the pool of sparse expert FFNs; individual tokens are dynamically routed by gating networks to only their Top-$k$ selected experts.
- **Communication Pattern**: TP uses deterministic, dense collectives (AllReduce / ReduceScatter) with rigid latency constraints. EP uses dynamic token dispatch and combination via AllToAll operations:
  $$\text{Local Tokens} \xrightarrow{\text{Router}} \text{AllToAll Send} \to \text{Expert FFN} \to \text{AllToAll Return} \to \text{Combine}$$
  EP introduces unique challenges including token load imbalance, straggler tail latency, and inter-node AllToAll bandwidth saturation.

</details>

<details class="exercise">
<summary><span class="q-label">Q15</span> <span class="q-text">How does Context Parallelism (CP) differ from standard Sequence Parallelism (SP)?</span></summary>

Megatron Sequence Parallelism (SP) does not split the core attention computation: it shards sequences only across LayerNorm and Dropout, gathering back full sequence representations before Attention and MLP GEMMs. Thus, standard SP cannot break the single-GPU memory barrier for massive context lengths.
Context Parallelism (CP) partitions long sequences across GPUs throughout the entire Transformer block. For self-attention, CP employs Ring Attention: each rank computes local attention while asynchronously rotating Key/Value blocks in a logical ring, computing partial softmax reductions without ever gathering the full sequence on any single device.

</details>

<details class="exercise">
<summary><span class="q-label">Q16</span> <span class="q-text">How should TP, PP, and DP dimensions be mapped across a 64-GPU cluster?</span></summary>

On a 64-GPU cluster (8 nodes $\times$ 8 H100s, 900 GB/s NVLink intra-node, 400G InfiniBand inter-node), dimensions are assigned to align communication intensity with physical bandwidth:
```text
Intra-Node (8 GPUs):      TP = 8   (saturates intra-node NVLink / NVSwitch mesh)
Inter-Node Dimension 1:   PP = 4   (partitions depth into 4 pipeline stages across nodes)
Inter-Node Dimension 2:   DP = 2   (replicates 2 pipeline instances for data parallelism)
Total GPUs:              TP * PP * DP = 8 * 4 * 2 = 64 GPUs
```
- **TP=8 within node**: TP generates high-frequency, latency-sensitive collectives that demand NVLink's sub-microsecond latency.
- **PP=4 across nodes**: PP exchanges only boundary activations and gradients at microbatch boundaries, easily tolerating network latency.
- **DP=2 outer layer**: Gradients are overlapped asynchronously with backward execution.

</details>

<details class="exercise">
<summary><span class="q-label">Q17</span> <span class="q-text">Why cannot large-scale model training rely on a single parallelism paradigm?</span></summary>

Every single parallelism strategy hits physical hardware limits:
- **Pure DP/FSDP**: Does not partition single-layer GEMMs; fails when layer weights or activations exceed single-card memory, and suffers low efficiency at tiny per-device batch sizes.
- **Pure TP**: Communication overhead scales with depth and rank count; extending TP beyond an 8-GPU NVLink boundary severely degrades throughput.
- **Pure PP**: High stage counts inflate bubble fractions ($F_\text{bubble} = \frac{P-1}{M+P-1}$) and introduce severe activation memory pressure.
- **Pure EP**: Applies only to MoE layers; dense attention layers remain unparallelized.
- **Pure CP**: Solves long context scaling, but does not address parameter or optimizer memory.

Consequently, training frontier models requires multi-dimensional 3D/4D hybrid parallelism.

</details>

<details class="exercise">
<summary><span class="q-label">Q18</span> <span class="q-text">What four diagnostic questions evaluate a distributed parallelism scheme?</span></summary>

To diagnose any distributed architecture proposal, verify four fundamental physical dimensions:
1. **Which dimension of the computational graph is partitioned?** (Batch, hidden channel, model depth, expert pool, or sequence context?)
2. **When and how frequently does communication occur?** (Per GEMM, per Transformer layer, per microbatch, or per optimization step?)
3. **Which physical network layer carries the communication?** (Intra-node NVLink/NVSwitch, inter-node InfiniBand/RoCE, or cross-switch datacenter fabric?)
4. **Which category of memory is saved, and what computational/communication trade-off is incurred?** (Static parameters/optimizer states vs. dynamic activations? Does it trade off FLOPs, network bandwidth, or pipeline bubble latency?)

</details>

---

## 12. References & Further Reading

### Foundational Literature & Sharding Theory
- **Google DeepMind Scaling Book**: Austin, J., Douglas, S., Frostig, R., et al. (2025). [How to Scale Your Model](https://jax-ml.github.io/scaling-book/). The primary reference for Chapters 2 and 3, formalizing sharded matrix multiplication notation, collective communication runtime models, and multi-axis parallelism constraints.
- **Ring AllReduce Foundations**: Gibiansky, A. (2017). [Bringing HPC Techniques to Deep Learning](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/). Baidu Silicon Valley AI Lab. Ring-based collective communication derivation and latency modeling.

### Data Parallelism & Memory Sharding (DP / FSDP / ZeRO)
- **ZeRO**: Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054). SC20. ZeRO-1/2/3 memory accounting and sharding strategies.
- **PyTorch FSDP**: Zhao, Y., Gu, A., Varma, R., et al. (2023). [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277). VLDB 2023. Industrial FSDP implementation, prefetching, and communication-computation overlap.

### Tensor Parallelism & Sequence Parallelism (TP / SP)
- **Megatron-LM**: Shoeybi, M., Patwary, M., Puri, R., et al. (2019). [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). Column-parallel and row-parallel linear operator decompositions.
- **Sequence Parallelism**: Korthikanti, V., Casper, J., Dey, S., et al. (2022). [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198). Integrating Sequence Parallelism (SP) with TP.

### Pipeline Parallelism (PP)
- **GPipe**: Huang, Y., Cheng, Y., Bapna, A., et al. (2019). [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965). NeurIPS 2019. GPipe and AFAB pipeline scheduling.
- **1F1B Scheduling**: Narayanan, D., Shoeybi, M., Zheng, C., et al. (2021). [Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2104.04473). ICML 2021. 1F1B and Interleaved 1F1B schedules.

### Long Context & Sparse MoE Parallelism (CP / EP)
- **Context Parallelism / Ring Attention**: Liu, H., Yan, M., Zaharia, M., & Abbeel, P. (2023). [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889). ICLR 2024.
- **Switch Transformers / MoE**: Fedus, W., Zoph, B., & Shazeer, N. (2022). [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961). JMLR 2022.

### Communication-Computation Overlap & Visualizations
- **Collective Matmul**: Wang, Y., et al. (2022). [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://dl.acm.org/doi/10.1145/3567955.3567959). ASPLOS 2023.
- **Visualizing Parallelism in Transformer**: Zhang, A. (2024). [Visualizing Parallelism in Transformer](https://ailzhang.github.io/posts/distributed-compute-in-transformer/). Meta PyTorch. Reference for Chapter 8½ architecture diagrams.

### Code Repositories
- [Picotron (Hugging Face)](https://github.com/huggingface/picotron) — Educational 4D hybrid parallelism framework
- [Megatron-LM (NVIDIA)](https://github.com/NVIDIA/Megatron-LM) — NVIDIA official TP/PP/SP implementation
- [DeepSpeed (Microsoft)](https://github.com/microsoft/DeepSpeed) — ZeRO-1/2/3 and ZeRO-Offload implementation
- [PyTorch DTensor](https://pytorch.org/docs/stable/distributed.tensor.html) — PyTorch distributed sharded tensor API
