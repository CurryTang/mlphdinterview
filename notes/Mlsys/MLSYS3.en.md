# Roofline Analysis

> Core question: why does an algorithm run in 50 ms rather than 5 ms or 500 ms? Where exactly is the bottleneck?
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

Ref: 
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
