# MLSYS Overview: Learning Roadmap & Curriculum

The MLSYS track spans the full spectrum of machine learning systems: starting from low-level GPU hardware and CUDA operator engineering, through high-performance Attention and KV Cache memory architectures, to LLM inference engines, acceleration/quantization, ultra-scale distributed training parallelism, and MoE / reinforcement learning cluster infrastructure.

---

## 5 Core Modules & 10 In-Depth Chapters

### Module 1: GPU Hardware Architecture & CUDA Operator Engineering
- [[MLSYS01 GPU Architecture and CUDA Programming Model|01 · GPU Architecture, CUDA Programming Model & Roofline Performance Analysis]]: SM microarchitecture, pipeline simulation, CUDA indexing, Roofline bounds, and 5 tuning principles.
- [[MLSYS02 Parallel Primitives and Memory-Bound Kernels|02 · Parallel Primitives & Memory-Bound Operator Optimization]]: 7 versions of Reduce, Histogram privatization, Blelloch Scan, Mamba associative scan, Vectorized float4, Transpose padding, and RMSNorm.
- [[MLSYS03 Compute-Bound Kernels and GEMM Optimization|03 · Compute-Bound Kernels & GEMM Optimization]]: Naive to 2D Tiling, Register Tiling, Double Buffering, Tensor Core MMA, Triton Conv2D Implicit GEMM, and Epilogue activation fusion.

### Module 2: High-Performance Attention & Long Context
- [[MLSYS04 Efficient Attention and Long Context|04 · Efficient Attention Evolution & Long-Context Operators]]: Online Softmax mathematical proof, FlashAttention-2 Triton code, FA3 Hopper features, Ring Attention, Striped Attention, and DeepSeek MLA.
- [[MLSYS05 KV Cache Memory Management and Prefix Caching|05 · KV Cache Memory Management, Prefix Caching & PagedAttention]]: Memory fragmentation, PagedAttention block mapping, SGLang Radix Tree, Chunked Prefill, and IndexShare.

### Module 3: LLM Inference Engines & Serving Acceleration
- [[MLSYS06 LLM Inference Engine Architecture nano-vllm|06 · LLM Inference Engine Architecture: Deep Dive into nano-vllm]]: Deconstructing vLLM internals: LLMEngine, Sequence state machine, preemptive Scheduler, BlockManager Copy-on-Write, and ModelRunner.
- [[MLSYS07 LLM Inference Acceleration and Quantization|07 · LLM Inference Acceleration, Speculative Decoding & Quantization]]: Autoregressive decoding bottlenecks, PTQ/QAT, SmoothQuant, AWQ, GPTQ, FP8, and speculative decoding (Medusa, EAGLE-3, DFlash).

### Module 4: Large-Scale Distributed Training & Cluster Infra
- [[MLSYS08 Distributed Training Parallelism and Communication|08 · Distributed Training Parallelism Paradigms & NCCL Communication]]: 3D/4D parallelism (DP/DDP, ZeRO-1/2/3, FSDP, Megatron TP/PP/SP, Ulysses) and NCCL ring communication analytical models.
- [[MLSYS09 MoE Systems and Post-Training Cluster Infra|09 · MoE Systems & Post-Training RL Cluster Infrastructure]]: DeepSeekMoE fine-grained experts, All-to-All communication, SonicMoE, and Post-Training RL systems (TRL to Forge/Ray, PPO/GRPO decoupling).

### Module 5: Classic Tech Report Deep Dives
- [[MLSYS10 Classic Tech Report MAI-Thinking-1|10 · Classic Tech Report Reading: MAI-Thinking-1 Architecture & Slide Notes]]: End-to-end deconstruction of Microsoft AI's 109-page technical report and 80-slide deck on trillion-parameter MoE and asynchronous RL systems: Dense/MoE 4x communication reduction, Ladder scaling methodology, Chinchilla TPP>>20 and data repetition boundaries, Mixing Laws, Token-level GRPO wide clipping, Rocket 4,096+768 split RL engine, and exact numerics replay.

---

## Recommended Learning Path

```text
01 (GPU & CUDA & Roofline)
  │
  ├──► 02 (Memory-Bound Operators & Reduce/Scan)
  ├──► 03 (Compute-Bound & GEMM / Triton)
  │      │
  │      └──► 04 (FlashAttention & MLA)
  │             └──► 05 (PagedAttention & KV Cache)
  │                    └──► 06 (nano-vllm Serving Engine Internals)
  │                           └──► 07 (Quantization & Speculative Decoding)
  │
  └──► 08 (Distributed Training Parallelism & NCCL)
         └──► 09 (MoE Systems & Post-Training RL Infra)
                └──► 10 (Classic Tech Report Deep Dive: MAI-Thinking-1 MoE & RL Systems)
```
