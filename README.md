# ML PhD Interview Notes

Live site: [https://currytang.github.io/mlphdinterview/](https://currytang.github.io/mlphdinterview/)

## 中文

这是一个面向 ML / LLM 方向面试复习的笔记站。内容包含以下核心板块：

- **MLSYS**（4 大模块 · 9 篇深度教程）：
  - 模块 1：GPU 硬件体系、CUDA 编程模型与 Roofline 分析；经典并行原语（Reduce/Scan/Histogram）与 Memory-Bound 优化；Compute-Bound 算子与 GEMM 优化（Shared Memory / 寄存器分块 / Double Buffering / Tensor Core / Triton）。
  - 模块 2：高性能 Attention 算子演进（Online Softmax / FlashAttention-1/2/3 / Ring Attention / MLA）；KV Cache 显存管理、前缀复用与 PagedAttention。
  - 模块 3：LLM 推理引擎架构（nano-vllm 源码完全解构）；解码加速、投机采样（Medusa / EAGLE / DFlash）与低比特量化（INT8/INT4/FP8/AWQ/GPTQ）。
  - 模块 4：大规模分布式训练并行体系（TP / PP / DP / ZeRO / FSDP）与 NCCL 环形通信；MoE 稀疏系统（DeepSeekMoE / SonicMoE）与后训练强化学习集群 Infra。
- **ML Coding & 八股**（大模型、传统机器学习与工业算法从零手撕）：
  - **Transformer 基础**：从零手写 LLM 循环、架构变体与算力分析、注意力算子全家桶、架构扩展（LoRA/ViT/MoE）、推理解码策略。
  - **LLM 核心理论与后训练**：架构选型与混合精度训练、RLHF 与偏好对齐全景（PPO/DPO）、RLVR 推理模型与 Agentic RL、极简 Agent 核心循环与架构、强化学习基础设施自测 35 问。
  - **LLM 代码实战**：量化与偏好损失函数手撕实战。
  - **基础知识八股与原理**：核心机制八股与基础算子手撕、数据预处理与经典损失函数、现代优化器（从 SGD/AdamW 到 Muon）。
  - **树模型与统计学习**：树模型集成与分布漂移检验。
  - **工业业务算法**：推荐与搜索多阶段全链路（数据流、多路召回、精排与多目标融合、重排决策与在线实验、生成式推荐与 Agentic Search，共 20 章）。
- **Quant**：概率、期望与示性变量、鞅与停时理论、布朗运动与随机微积分、博弈论、衍生品定价、线性回归几何视角与 C++ 系统实战。
- **System Design**：后端系统设计、分布式存储、异步消息系统、秒杀与通知架构、LLM serving、Kubernetes 训练调度控制面（配套实验 [LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)）。
- **LeetCode Core Skills**：双指针、滑动窗口、单调栈、二分查找、回溯决策树、贪心模式、区间动态规划等核心题型。
- **找工**：Neolab 列表与求职资源。

Kubernetes 章节的配套实验在 [`project/LLMTrainLab/`](project/LLMTrainLab/)（独立仓库 [CurryTang/LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)）。

如果你发现内容有错误、表达不清楚、公式渲染问题，或者想补充更好的例题 / 面试题，欢迎提 issue 或 PR。

## English

This is a comprehensive interview-preparation site for ML, LLM, Quant, and Systems roles, organized across six core tracks:

- **MLSYS** (4 Modules · 9 In-Depth Chapters):
  - Module 1: GPU hardware architecture, CUDA programming model & Roofline analysis; Parallel primitives & Memory-bound kernels (Reduce/Scan/Histogram); Compute-bound kernels & GEMM optimization (Shared memory, register tiling, tensor cores, Triton).
  - Module 2: FlashAttention evolution (Online Softmax, FA1/2/3, Ring Attention, MLA); KV cache memory management, prefix caching & PagedAttention.
  - Module 3: LLM inference engines (nano-vllm codebase walkthrough); inference acceleration, speculative decoding & low-bit quantization (INT8/INT4/FP8/AWQ/GPTQ).
  - Module 4: Distributed training parallelism (TP/PP/DP/ZeRO/FSDP) & NCCL analytical models; MoE sparse systems & post-training RL cluster infra.
- **ML Coding & Fundamentals** (From-Scratch Implementations & Theory):
  - **Transformer Basics**: Build an LLM from scratch, architecture variants & FLOPs analysis, attention mechanism zoo, architecture extensions (LoRA/ViT/MoE), inference decoding strategies.
  - **LLM Core & Post-Training**: Architecture decisions & mixed-precision systems, RLHF & preference alignment (PPO to DPO), RLVR & agentic RL, minimalist agent core loop, 35 self-check questions on RL infra.
  - **Hands-on Code**: Quantization & preference loss implementations.
  - **Foundations & Core Mechanics**: Core mechanics & foundational operators, data preprocessing & loss functions, modern optimizers (from SGD/AdamW to Muon).
  - **Tree Models & Statistics**: Tree ensembles & distribution shift testing.
  - **Industrial Algorithms**: End-to-end multi-stage recommendation and search pipeline, multi-channel retrieval, ranking & score fusion, reranking, online A/B experimentation, generative recommendation & agentic search (20 chapters).
- **Quant**: Probability, expectations, martingales, Brownian motion & stochastic calculus, game theory, financial derivatives, FWL geometry, and C++ systems.
- **System Design**: Backend architecture, storage systems, async messaging, flash sales, notifications, LLM serving, and Kubernetes cluster scheduling (paired with [LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)).
- **LeetCode Core Skills**: Two pointers, sliding window, monotonic stack, binary search, backtracking decision trees, greedy patterns, dynamic programming.
- **Jobs**: Curated Neolab lists and job hunting resources.

Corrections, issue reports, and contributions are welcome. If something is wrong, unclear, outdated, or missing a useful example, feel free to open an issue or PR.

## Local Development

```bash
npm install
npm run dev
```

## Checks

```bash
npm test
npm run lint
npm run build
```
