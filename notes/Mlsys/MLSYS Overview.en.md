# MLSYS Overview: Learning Path and Directory

The MLSYS section is organized by system hierarchy, ranging from GPU/CUDA to kernels, distributed training, inference serving, post-training infrastructure, and long-context model architectures. This entry page is designed to help you quickly determine the reading order and identify dependencies:

```text
Which note should I read today?
What kind of system problem does each note solve?
What are the dependencies between them?
```

## Directory

1. [Learning Roadmap](#learning-roadmap)
2. [GPU and CUDA Fundamentals](#gpu-and-cuda-fundamentals)
3. [Kernel Optimization](#kernel-optimization)
4. [Training and Inference Systems](#training-and-inference-systems)
5. [Precision and Quantization](#precision-and-quantization)
6. [Post-training / Efficient Attention / KV Cache / Inference / MoE](#post-training-efficient-attention-kv-cache-inference-moe)
7. [Where did the LLM Interview Prep go?](#where-did-the-llm-interview-prep-go)

## Learning Roadmap

MLSYS system problems generally unfold along this chain:

```text
GPU architecture
-> CUDA programming model
-> memory / compute bound analysis
-> kernel optimization
-> distributed training
-> inference serving
-> precision / quantization
```

Recommended sequence:

```text
MLSYS1 -> MLSYS2 -> MLSYS3 -> MLSYS4/5/6 -> MLSYS7/8/9
       -> MLSYS10 -> MLSYS11/12 -> MLSYS13 -> MLSYS14/15/16/17/18
```

If you are short on time, prioritize these:

| Goal | Recommended Notes |
|---|---|
| GPU / CUDA Fundamentals | [[MLSYS1]], [[MLSYS2]] |
| Performance Analysis Framework | [[MLSYS3]] |
| Kernel Optimization Patterns | [[MLSYS4]], [[MLSYS5]], [[MLSYS6]] |
| GEMM / compute-bound | [[MLSYS7 Compute-Bound Kernel (1)]], [[MLSYS8 Compute-Bound Kernel (2)]], [[MLSYS9 Compute-bound kernel (3)]] |
| Distributed Training | [[MLSYS10 parallelism]] |
| Inference Systems | [[MLSYS11 nano-vllm-1]], [[MLSYS12 nano-vllm-2]] |
| Quantization and Precision | [[MLSYS13 Quantization and precision]] |
| Post-training / RL Infra | [[MLSYS14 Post-Training Infra]] |
| Efficient attention / Long-context architectures | [[MLSYS15 Efficient Attention Modern Architectures|MLSYS15 Efficient Attention]] |
| KV cache / Long-context inference | [[MLSYS15 KV Cache Prefix Caching IndexShare|MLSYS16 KV Cache]] |
| Inference Acceleration | [[MLSYS15 LLM Inference Speculative Decoding DFlash|MLSYS17 Inference]] |
| MoE Systems | [[MLSYS16 Modern MoE SonicMoE|MLSYS18 MoE Systems]] |

## GPU and CUDA Fundamentals

### [[MLSYS1|MLSYS1 · Introduction to GPU Architecture]]

First, establish the GPU execution model:

- Relationship between SM / warp / thread block
- Levels of the memory hierarchy
- Why GPUs are suitable for high-throughput parallelism

### [[MLSYS2|MLSYS2 · CUDA Programming Model]]

Mapping the hardware model to CUDA code:

- grid / block / thread indexing
- shared memory
- synchronization
- Basic intuition for memory coalescing

## Kernel Optimization

### [[MLSYS3|MLSYS3 · Roofline Analysis]]

Roofline is the diagnostic tool for all subsequent kernel optimizations.

You should first ask:

```text
Is this operator memory-bound or compute-bound?
Is the bottleneck bandwidth, memory access patterns, or compute utilization?
```

### [[MLSYS4|MLSYS4 · CUDA Reduce Kernel]]

Reduction is the best exercise for mastering CUDA optimization fundamentals:

- tree reduction
- warp divergence
- bank conflict
- unroll
- warp shuffle

### [[MLSYS5|MLSYS5 · Histogram & Scan]]

Parallel primitives are the building blocks for many advanced operators.

Focus on:

- histogram atomic / privatization
- scan prefix-sum concepts
- work-efficient vs step-efficient

### [[MLSYS6|MLSYS6 · Memory-Bound Kernel Optimization]]

Optimizations centered around memory-bound operators:

- coalescing
- vectorized load/store
- shared memory tiling
- Reducing global memory traffic

## Compute-Bound Kernel

### [[MLSYS7 Compute-Bound Kernel (1)|MLSYS7 · Compute-Bound Kernel (1)]]

The entry point for compute-bound kernels, usually starting with GEMM / matmul concepts.

### [[MLSYS8 Compute-Bound Kernel (2)|MLSYS8 · Compute-Bound Kernel (2)]]

Further breakdown of optimizations like tiling, register blocking, and shared memory reuse.

### [[MLSYS9 Compute-bound kernel (3)|MLSYS9 · Compute-bound kernel (3)]]

Details closer to real-world performance troubleshooting:

- occupancy
- arithmetic intensity
- tensor core / MMA
- pipeline thinking

## Training and Inference Systems

### [[MLSYS10 parallelism|MLSYS10 · Distributed Training Parallelism Paradigms]]

The core of training systems is the parallel strategy:

- data parallel
- tensor parallel
- pipeline parallel
- ZeRO / FSDP
- communication vs computation overlap

### [[MLSYS11 nano-vllm-1|MLSYS11 · nano-vllm Deep Dive (1)]]

The first part of inference systems, focusing on:

- prefill / decode
- KV cache
- attention FLOPs / memory traffic
- Why serving systems differ from training systems

### [[MLSYS12 nano-vllm-2|MLSYS12 · nano-vllm Deep Dive (2)]]

The second part of inference systems, focusing on:

- paged KV cache
- continuous batching
- prefix cache
- block table
- CUDA graph / flash attention integration

## Precision and Quantization

### [[MLSYS13 Quantization and precision|MLSYS13 · Quantization and Precision]]

The main thread for precision and quantization:

- FP32 / FP16 / BF16 / FP8
- weight-only quantization
- KV cache quantization
- Stability issues in low-precision training

## Post-training / Efficient Attention / KV Cache / Inference / MoE

### [[MLSYS14 Post-Training Infra|MLSYS14 · Post-Training Infra]]

Post-training belongs to the MLSYS main thread because it discusses the system architecture composed of training, inference, and environment services:

- rollout / training / reward / weight sync
- RL infra frameworks like veRL, slime, SkyRL, AReaL
- Agentic RL workloads like SearchR1, terminal agent, sandbox

### [[MLSYS15 Efficient Attention Modern Architectures|MLSYS15 · Efficient Attention: Modern Long-Context Architectures]]

The efficient attention main thread:

- Dense, linear, sparse, and hybrid attention from an associative memory perspective
- Dynamic sparse routing in DeepSeek DSA, DMA, and DHSA
- Hybrid / recurrent attention implementations in Qwen3-Next, Kimi Linear, and MiniMax-M1
- System constraints under prefill, decode, cache manager, and speculative decoding

### [[MLSYS15 KV Cache Prefix Caching IndexShare|MLSYS16 · KV Cache: Memory Management, Prefix Reuse, and IndexShare]]

The KV cache module supplements the cache layer of inference systems:

- What exactly is stored in the KV cache
- Boundaries of PagedAttention, prefix cache, and RadixAttention
- KV cache capacity, quantization, eviction, and transmission
- Implementation paths for GLM-5.2 IndexShare / IndexCache in Transformers, ATOM/vLLM

### [[MLSYS15 LLM Inference Speculative Decoding DFlash|MLSYS17 · Inference: Parallel Decoding and Draft Verification]]

Decoding acceleration in inference systems:

- Exact sampling in speculative decoding
- Drafter designs in Medusa, EAGLE, and DFlash
- How to enable and evaluate spec decode in vLLM / SGLang

### [[MLSYS16 Modern MoE SonicMoE|MLSYS18 · MoE Systems: Routing, Communication, and Kernels]]

MoE topics in training and inference systems:

- router, top-k, capacity, load balance
- expert parallel, all-to-all, grouped GEMM
- How SonicMoE optimizes kernels and activation memory for fine-grained sparse MoE

## Where did the LLM Interview Prep go?

The `LLM Interview Prep` section now only contains self-assessment/interview-style content. Systematic tutorials remain in the `MLSYS` main thread.

```text
LLM Interview Prep
```

Entry points:

- [Open LLM Interview Prep section](#where-did-the-llm-interview-prep-go)
- [[MLSYS15 RL Infra 自测 35 问|Reinforcement Learning Exercises]]

After this split:

```text
MLSYS = GPU / kernel / training / inference / precision / post-training / attention / MoE systems
LLM Interview Prep = RL infra self-check / interview drills
LeetCode = data structure & algorithm patterns
```
