# MLSYS Overview：学习路径与核心目录

MLSYS 板块从 GPU 底层硬件与 CUDA 算子工程出发，贯穿高性能 Attention、KV Cache 显存系统、LLM 推理引擎、解码加速与量化，延伸至超大规模分布式训练并行体系与 MoE / 强化学习集群 Infra。

---

## 4 大核心模块与 9 篇深度教程

### 模块 1：GPU 硬件架构与 CUDA 算子工程
- [[MLSYS01 GPU Architecture and CUDA Programming Model|01 · GPU 硬件体系、CUDA 编程模型与 Roofline 性能分析基石]]：从 SM 微架构、流水线仿真、CUDA 线程层级映射，到 Roofline 理论上下界与调优五原则。
- [[MLSYS02 Parallel Primitives and Memory-Bound Kernels|02 · 经典并行原语与 Memory-Bound 算子手撕指南]]：Reduce 7 版本演进、Histogram 私有化、Scan（Blelloch 与 Mamba 关联扫描）、Vectorized float4、Transpose padding 与 RMSNorm。
- [[MLSYS03 Compute-Bound Kernels and GEMM Optimization|03 · Compute-Bound 算子与 GEMM 演进完全指南]]：从 Naive GEMM 到 2D Tiling、Register Tiling、Double Buffering、Tensor Core MMA、Triton Conv2D Implicit GEMM 与 Epilogue 融合。

### 模块 2：高性能 Attention 算子与长上下文
- [[MLSYS04 Efficient Attention and Long Context|04 · FlashAttention 演进与长上下文算子]]：Online Softmax 数学证明、FlashAttention-2 Triton 实现、FA3 Hopper 优化、Ring Attention、Striped Attention 与 DeepSeek MLA。
- [[MLSYS05 KV Cache Memory Management and Prefix Caching|05 · KV Cache 显存管理、前缀复用与 PagedAttention]]：显存碎片根因分析、PagedAttention 物理分页、SGLang Radix Tree 前缀树、Chunked Prefill 与 IndexShare。

### 模块 3：大模型推理系统与服务加速
- [[MLSYS06 LLM Inference Engine Architecture nano-vllm|06 · LLM 推理引擎内核架构：nano-vllm 源码完全解构]]：解构 vLLM 核心，涵盖 LLMEngine、Sequence 状态机、抢占式 Scheduler、BlockManager 写时复制与 ModelRunner。
- [[MLSYS07 LLM Inference Acceleration and Quantization|07 · 推理解码加速、投机采样与低比特量化全景]]：自回归解码访存瓶颈、PTQ/QAT、SmoothQuant、AWQ、GPTQ、FP8，以及投机采样（Medusa、EAGLE-3 与 DFlash）。

### 模块 4：大规模分布式训练与集群 Infra
- [[MLSYS08 Distributed Training Parallelism and Communication|08 · 分布式训练并行范式全景与 NCCL 通信]]：3D/4D 并行体系（DP/DDP、ZeRO-1/2/3、FSDP、Megatron TP/PP/SP、Ulysses）与 NCCL 环形通信数学模型。
- [[MLSYS09 MoE Systems and Post-Training Cluster Infra|09 · MoE 稀疏系统与后训练强化学习集群架构]]：DeepSeekMoE 细粒度专家、All-to-All 通信加速、SonicMoE，以及后训练强化学习（TRL 到 Forge/Ray、PPO/GRPO 系统解耦）。

---

## 推荐学习主线

```text
01 (GPU & CUDA & Roofline)
  │
  ├──► 02 (Memory-Bound 算子 & Reduce/Scan)
  ├──► 03 (Compute-Bound & GEMM / Triton)
  │      │
  │      └──► 04 (FlashAttention & MLA)
  │             └──► 05 (PagedAttention & KV Cache)
  │                    └──► 06 (nano-vllm 推理引擎源码)
  │                           └──► 07 (量化与投机采样加速)
  │
  └──► 08 (分布式训练并行与 NCCL 通信)
         └──► 09 (MoE 稀疏系统与后训练强化学习 Infra)
```
