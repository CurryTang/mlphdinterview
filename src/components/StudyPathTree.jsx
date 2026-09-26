import React, { useState } from 'react';

export const studyPathStages = [
  {
    stageId: 'stage-01',
    levelNumber: '01',
    title: { zh: '筑基与数学基础', en: 'Foundations & Numerical Primitives' },
    subtitle: {
      zh: '数据工程、时序防泄露、数值稳定性与 Tokenization 编解码底座',
      en: 'Data engineering, leakage defense, numerical stability & tokenization foundations',
    },
    icon: '🌱',
    nodes: [
      {
        id: 'node-01-preprocessing',
        tracks: ['all', 'traditional-ml', 'recsys'],
        title: { zh: '数据预处理、特征工程与时序防泄露机制', en: 'Data Preprocessing, Feature Engineering & Leakage Defense' },
        subtitle: { zh: '面试高频陷阱：缺失值填充、Standardization 几何意义与数据穿越规避', en: 'Imputation, standardization geometry & time-series leakage defense' },
        difficulty: 'Easy',
        badge: { zh: '核心八股', en: 'Must-Know' },
        type: 'theory',
        targetTutorialId: 'MLCoding00 ML Basics Data Preprocessing Loss Functions.md',
        checklist: {
          zh: [
            '缺失值处理准则：数值型中位数/均值 vs 类别型占位符与指示变量',
            'Standardization (Z-score) vs MinMax 几何影响与 Hessian 条件数关联',
            '时序样本切分时绝对禁止使用未来信息（Data Leakage / Lookahead Bias）',
            '类别特征高基数编码（Target Encoding）与防过拟合平滑先验',
          ],
          en: [
            'Missing value policies: median/mean vs category indicator tokens',
            'Standardization vs MinMax: impact on Hessian condition number',
            'Time-series split leakage defense (preventing lookahead bias)',
            'High-cardinality target encoding with smoothing priors',
          ],
        },
      },
      {
        id: 'node-02-numerics-loss',
        tracks: ['all', 'traditional-ml', 'recsys', 'llm'],
        title: { zh: '数值防溢出 Log-Sum-Exp 与经典损失函数全景', en: 'Numerical Stability, Log-Sum-Exp & Loss Functions Zoo' },
        subtitle: { zh: 'Log-Sum-Exp 技巧原理、NaN/梯度爆炸根因排查，与 BCE/Focal/Triplet 损失', en: 'Log-Sum-Exp trick, NaN root causes, and BCE/Focal/Triplet loss derivations' },
        difficulty: 'Medium',
        badge: { zh: '手撕代码', en: 'Coding & Theory' },
        type: 'code',
        targetTutorialId: 'MLCoding00 ML Basics Data Preprocessing Loss Functions.md',
        checklist: {
          zh: [
            'Softmax / CrossEntropy 中减去最大值的 Log-Sum-Exp 绝对防溢出推导',
            '训练中 Loss 突变为 NaN 的五大根因（学习率、未归一化、未截断、深层梯度、除零）',
            'MSE vs MAE 梯度动力学与对异常值敏感度',
            'BCE 与 Focal Loss 调制因子 (1-p_t)^γ 对正负极度不平衡与难负例的挖掘机制',
          ],
          en: [
            'Log-Sum-Exp trick in Softmax/CrossEntropy for preventing overflow/underflow',
            'Five root causes of NaN losses (learning rate, unnormalized data, clipping, div-by-zero)',
            'MSE vs MAE gradient dynamics and outlier sensitivity',
            'Focal Loss modulation factor (1-p_t)^γ for class imbalance & hard negative mining',
          ],
        },
      },
      {
        id: 'node-03-bpe-tokenizer',
        tracks: ['all', 'llm'],
        title: { zh: 'Unicode Pretokenization 与 BPE 分词器从零手撕', en: 'Unicode Pretokenization & Byte-level BPE from Scratch' },
        subtitle: { zh: 'GPT-4 正则表达式预分词、字节对贪心合并、Tokenizer 编解码与自回归训练循环', en: 'GPT-4 regex splits, byte-pair merge rules, tokenizer codec & autoregressive loop' },
        difficulty: 'Medium',
        badge: { zh: '手撕代码', en: 'Coding' },
        type: 'code',
        targetTutorialId: 'MLCoding01 Unicode Pretokenization.md',
        checklist: {
          zh: [
            'UTF-8 变长字节编码规则与 Unicode 码点转换',
            'GPT-4 分词正则拆分规则（缩写、字母、数字、空白分流）',
            'BPE 词表统计频次与 Byte Pair 贪心合并算法纯手写',
            '自回归单步训练循环与 cross_entropy next-token loss 计算',
          ],
          en: [
            'UTF-8 variable-length byte representation & Unicode codepoints',
            'GPT-4 regex pre-tokenization split rules for contractions & digits',
            'Vocabulary frequency counting & greedy pair merging implementation',
            'Autoregressive single-step training loop with cross-entropy next-token loss',
          ],
        },
      },
    ],
  },
  {
    stageId: 'stage-02',
    levelNumber: '02',
    title: { zh: '核心算子从零手撕', en: 'Core Operators from Scratch' },
    subtitle: {
      zh: '从归一化、卷积到多头与组查询注意力全家桶，彻底摒弃黑盒调用',
      en: 'From normalization and convolutions to attention operator zoo from scratch',
    },
    icon: '⚡',
    nodes: [
      {
        id: 'node-04-norm-conv',
        tracks: ['all', 'traditional-ml'],
        title: { zh: '归一化全家桶与经典算子手写 (BatchNorm/RMSNorm/Conv2d)', en: 'Normalization Zoo, Dropout & Conv2d from Scratch' },
        subtitle: { zh: 'BatchNorm 运行均值方差与 eval 模式陷阱，LayerNorm vs RMSNorm，Conv2d im2col 向量化', en: 'BatchNorm running statistics trap, LayerNorm vs RMSNorm, and Conv2d im2col' },
        difficulty: 'Medium',
        badge: { zh: '纯手撕代码', en: 'Coding' },
        type: 'code',
        targetTutorialId: 'MLCoding02 GELU BatchNorm Conv2d Linear Regression.md',
        checklist: {
          zh: [
            'BatchNorm 训练模式批次统计量与评估模式 running_mean/var 动量更新',
            'LayerNorm 计算均值与方差 vs RMSNorm 仅除以均方根（计算与显存加速）',
            'Inverted Dropout 训练时缩放 1/(1-p) 保持期望一致与测试阶段无操作',
            'Conv2d 滑动窗口展开为 im2col GEMM 矩阵连乘向量化',
          ],
          en: [
            'BatchNorm train-time batch stats vs eval-time running mean/var momentum trap',
            'LayerNorm centering & scaling vs RMSNorm root-mean-square speedup',
            'Inverted Dropout scaling 1/(1-p) for expectation consistency & zero-overhead eval',
            'Conv2d sliding window unfolding to im2col GEMM vectorization',
          ],
        },
      },
      {
        id: 'node-05-attention-zoo',
        tracks: ['all', 'llm'],
        title: { zh: '注意力算子全家桶从零手撕 (MHA / GQA / Sliding Window / KV Cache)', en: 'Attention Operator Zoo from Scratch (MHA, GQA, Sliding Window, KV Cache)' },
        subtitle: { zh: '拒绝官方黑盒调用！显式 W_Q, W_K, W_V, W_O、einops rearrange、key padding mask 维度广播与 KV 缓存复用', en: 'No blackboxes! Explicit projections, einops rearrange, key padding mask broadcasting & KV Cache' },
        difficulty: 'Hard',
        badge: { zh: '绝对高频手撕', en: 'Must-Code' },
        type: 'code',
        targetTutorialId: 'MLCoding03 Attention Variants GQA Sliding Window KV Cache.md',
        checklist: {
          zh: [
            '显式 nn.Linear 投影与 einops.rearrange "b t (h d) -> b h t d"',
            'Key padding mask 维度广播 mask[:, None, None, :] 与 masked_fill(-inf)',
            'GQA 组查询注意力：K/V 头数小于 Q 头数时的 repeat_interleave 维度扩展',
            'Sliding Window 局部因果掩码带状生成与矩阵裁剪',
            'KVCacheAttention 增量更新与显存复用（避免全长重复计算）',
          ],
          en: [
            'Explicit nn.Linear projections and einops.rearrange "b t (h d) -> b h t d"',
            'Key padding mask broadcasting mask[:, None, None, :] & masked_fill(-inf)',
            'GQA Grouped Query Attention: expanding K/V heads to match Q via repeat_interleave',
            'Sliding Window local causal band generation & matrix masking',
            'KVCacheAttention incremental append & memory reuse for decoding',
          ],
        },
      },
      {
        id: 'node-06-transformer-flops',
        tracks: ['all', 'llm', 'mlsys'],
        title: { zh: 'Transformer 架构变体、FLOPs 分解与 KV 显存估算', en: 'Transformer Variants, FLOPs Breakdown & KV Cache Roofline' },
        subtitle: { zh: 'MHA vs MQA vs GQA 张量形状对比、前向推断 FLOPs 理论推导与上下文长序列显存模型', en: 'MHA vs MQA vs GQA tensor shapes, forward FLOPs derivation & long context KV memory models' },
        difficulty: 'Medium',
        badge: { zh: '架构与算力', en: 'Architecture' },
        type: 'theory',
        targetTutorialId: 'MLCoding01B Transformer Architecture Variants Attention FLOPs KV Cache.md',
        checklist: {
          zh: [
            'MHA、MQA、GQA 在 Q/K/V 权重参数量与 KV Cache 显存占用的精确量化公式',
            '一次前向传播中 Attention 与 MLP 层的 FLOPs 分解推导（2N 乘加）',
            '上下文长度 T 膨胀时，KV Cache 显存随批量大小 B 与层数 L 的线性增长公式',
            'FlashAttention 分块切分（Tiling）与 SRAM 访存复杂度从 O(T^2) 降至 O(T)',
          ],
          en: [
            'Quantitative memory formulas: MHA vs MQA vs GQA parameters & KV Cache sizes',
            'Forward pass FLOPs breakdown for Attention vs MLP blocks (2N multiply-adds)',
            'Linear growth formula of KV cache memory as context length T scales',
            'FlashAttention tiling & reducing HBM access complexity from O(T^2) to O(T)',
          ],
        },
      },
      {
        id: 'node-07-peft-moe',
        tracks: ['all', 'llm'],
        title: { zh: 'PEFT 参数高效微调 (LoRA) 与 MoE 混合专家路由', en: 'PEFT Low-Rank Adaptation (LoRA) & MoE Routing' },
        subtitle: { zh: 'LoRA 低秩矩阵分解 W = W_0 + (B @ A) * α/r，ViT Patch Embedding，Top-k Softmax 门控路由', en: 'LoRA low-rank decomposition W = W_0 + (B @ A) * α/r, ViT patches, and MoE Top-k routing' },
        difficulty: 'Hard',
        badge: { zh: '手撕代码', en: 'Coding' },
        type: 'code',
        targetTutorialId: 'MLCoding04 LoRA ViT Patch Embedding MoE.md',
        checklist: {
          zh: [
            'LoRA 权重低秩分解：A ~ N(0, 1/r), B = 0 确保初始状态无偏',
            '推理阶段权重合并 (Weight Merging) 实现零延迟引入',
            'MoE 门控网络 Top-k Softmax 路由、Load Balancing 辅助损失与 Capacity Factor',
            'ViT 图像 Patch Embedding 展开为 2D 卷积或全连接映射',
          ],
          en: [
            'LoRA rank decomposition: A ~ N(0, 1/r), B = 0 for zero-initial-perturbation',
            'Inference weight merging for zero added serving latency',
            'MoE Top-k Softmax routing, auxiliary load balancing loss & capacity factor',
            'Vision Transformer (ViT) patch embedding as 2D convolution projection',
          ],
        },
      },
    ],
  },
  {
    stageId: 'stage-03',
    levelNumber: '03',
    title: { zh: '树模型、集成算法与现代优化器', en: 'Ensembles, Drift Testing & Modern Optimizers' },
    subtitle: {
      zh: '从决策树严格凹性推导到随机森林手撕、分布漂移检验与 AdamW / Muon 谱正交优化器',
      en: 'From tree concavity proofs to Random Forest from scratch, drift testing & AdamW/Muon optimizers',
    },
    icon: '🌲',
    nodes: [
      {
        id: 'node-08-tree-ensembles',
        tracks: ['all', 'traditional-ml', 'recsys'],
        title: { zh: '决策树分裂凹性证明、随机森林手写与分布漂移检验', en: 'Tree Splitting Concavity, Random Forest from Scratch & Drift Testing' },
        subtitle: { zh: 'Gini/Entropy 严格凹性与 Jensen 不等式，分类错误率增益假死反例，RF Bootstrap + OOB，KS/PSI/C2ST 检验', en: 'Gini/Entropy strict concavity proofs, classification error failure, RF from scratch & drift testing' },
        difficulty: 'Medium',
        badge: { zh: '数学推导 & 代码', en: 'Theory & Code' },
        type: 'code',
        targetTutorialId: 'MLCoding09 Data Science Statistical Testing Distribution Drift C2ST.md',
        checklist: {
          zh: [
            '基尼系数与信息熵二阶导数处处负定（严格凹函数）与 Jensen 不等式增益保证',
            '分类错误率分段线性导致分裂增益假死（Zero Gain）反例推导',
            '随机森林 Bootstrap 样本抽样与特征随机子空间纯手撕',
            '袋外评估 (OOB Score) 无偏估计泛化误差与无需 Hold-out 验证集原理',
            'Kolmogorov-Smirnov 单特征检验、PSI 稳定性指标与 C2ST 分类器双样本漂移检验',
          ],
          en: [
            'Strict concavity of Gini & Entropy (negative definite second derivatives) & Jensen inequality',
            'Zero-gain counterexample proving why classification error fails as a splitting criterion',
            'Random Forest bootstrap sampling & random feature subspace implementation from scratch',
            'Out-of-Bag (OOB) evaluation for unbiased generalization estimation',
            'Kolmogorov-Smirnov test, Population Stability Index (PSI) & C2ST classifier drift test',
          ],
        },
      },
      {
        id: 'node-09-optimizers-muon',
        tracks: ['all', 'traditional-ml', 'llm'],
        title: { zh: '现代优化器全景：从 SGD 动量到 AdamW 与 Muon 谱正交纯手写', en: 'Modern Optimizers: SGD Momentum to AdamW & Muon Spectral Orthogonalization' },
        subtitle: { zh: 'Nesterov 加速梯度动力学、Adam 二阶矩偏差校正与 AdamW 解耦权重衰减，Muon 优化器与 Newton-Schulz 极分解手写', en: 'Nesterov acceleration, Adam second moments, AdamW decoupled decay & Muon Newton-Schulz from scratch' },
        difficulty: 'Hard',
        badge: { zh: '前沿理论与代码', en: 'Frontier Code' },
        type: 'code',
        targetTutorialId: 'MLCoding10 Modern Optimizers SGD AdamW Muon.md',
        checklist: {
          zh: [
            'SGD 经典动量与 Nesterov 预测未来梯度物理动力学机制',
            '标准 Adam 将 L2 正则放进梯度引发的衰减失真 vs AdamW 解耦权重衰减',
            'Adam 二阶矩偏差校正 beta2 稀疏梯度陷阱与梯度截断时机',
            '前沿 Muon 优化器原理：利用 Newton-Schulz 迭代求极分解正交投影纯手写',
          ],
          en: [
            'Classical Momentum vs Nesterov Accelerated Gradient physical mechanics',
            'Adam coupled L2 decay distortion vs AdamW decoupled weight decay',
            'Adam second-moment bias correction beta2 sparse update pitfalls',
            'Muon optimizer: Newton-Schulz polar decomposition orthogonal projection from scratch',
          ],
        },
      },
    ],
  },
  {
    stageId: 'stage-04',
    levelNumber: '04',
    title: { zh: '后训练对齐、长程推理与智能体', en: 'Post-Training, Reasoning & Autonomous Agents' },
    subtitle: {
      zh: '从 DPO/PPO 偏好对齐到 DeepSeek-R1 GRPO 规则推理与 300 行极简智能体循环',
      en: 'From DPO/PPO alignment to DeepSeek-R1 GRPO reasoning & 300-line minimalist agent loops',
    },
    icon: '🎯',
    nodes: [
      {
        id: 'node-10-decoding-quant',
        tracks: ['all', 'llm', 'mlsys'],
        title: { zh: '推理解码加速策略与 INT8 动态量化算子手写', en: 'Decoding Strategies & INT8 Dynamic Quantization from Scratch' },
        subtitle: { zh: 'Top-k / Top-p / Temperature 采样，投机解码草稿验证核，INT8 AbsMax 矩阵量化实现', en: 'Top-k/Top-p/Temp sampling, speculative decoding verification kernel & INT8 quantization' },
        difficulty: 'Medium',
        badge: { zh: '手撕代码', en: 'Coding' },
        type: 'code',
        targetTutorialId: 'MLCoding05 Sampling Beam Search Speculative Decoding.md',
        checklist: {
          zh: [
            'Temperature 缩放 logits、Top-k 截断与 Top-p 累积概率核采样实现',
            '投机解码 (Speculative Decoding) 草稿模型与大模型验证核判定逻辑手撕',
            'INT8 动态量化 scale/zero_point 估算与定点矩阵反量化还原',
          ],
          en: [
            'Temperature scaling, Top-k cutoff & Top-p cumulative probability nucleus sampling',
            'Speculative decoding draft generation and large model verification kernel from scratch',
            'INT8 dynamic quantization scale/zero_point estimation and dequantization math',
          ],
        },
      },
      {
        id: 'node-11-rlhf-ppo-dpo',
        tracks: ['all', 'llm'],
        title: { zh: 'RLHF 偏好对齐全景：PPO 4 模型架构到 DPO/SimPO 损失函数纯手写', en: 'RLHF Alignment Panorama: PPO 4-Model Topology to DPO/SimPO Loss from Scratch' },
        subtitle: { zh: 'Actor/Critic/Ref/Reward 通信拓扑，PPO 裁剪目标与 GAE，DPO 隐式奖励闭式推导与 DPO Loss 实现', en: 'Actor/Critic/Ref/Reward topology, PPO clipped objective & GAE, and DPO implicit reward derivation' },
        difficulty: 'Hard',
        badge: { zh: '前沿核心考点', en: 'Frontier Core' },
        type: 'code',
        targetTutorialId: 'MLCoding06B RLHF Preference Alignment PPO DPO.md',
        checklist: {
          zh: [
            'PPO 4 模型架构：Actor、Critic、Reference、Reward 协同与显存负载分析',
            'PPO 裁剪目标 clip(r_t, 1-ε, 1+ε) 与 GAE 广义优势估计对齐税分析',
            'DPO 隐式奖励闭式解代入 Bradley-Terry 模型彻底规避 Critic 网络的数学证明',
            'DPO 损失函数、GRPO 优势计算纯手写（MLCoding06）',
          ],
          en: [
            'PPO 4-model topology: Actor, Critic, Reference, Reward interaction & memory costs',
            'PPO clipped objective clip(r_t, 1-ε, 1+ε), GAE advantage & the alignment tax',
            'DPO mathematical proof bypassing Critic network via Bradley-Terry substitution',
            'From-scratch implementation of DPO loss and GRPO advantage kernels',
          ],
        },
      },
      {
        id: 'node-12-rlvr-deepseek-grpo',
        tracks: ['all', 'llm'],
        title: { zh: 'RLVR、DeepSeek-R1 推理模型演进与 GRPO 算法推导', en: 'RLVR, DeepSeek-R1 Reasoning Evolution & GRPO Derivation' },
        subtitle: { zh: 'Cold-Start SFT ➔ Reasoning RL ➔ Rejection Sampling 演进，GRPO 移除 Critic 与规则验证器机制', en: 'Cold-start SFT ➔ Reasoning RL ➔ Rejection sampling, critic-free GRPO & verifiable rewards' },
        difficulty: 'Hard',
        badge: { zh: '顶会前沿必考', en: 'Must-Know Hotspot' },
        type: 'theory',
        targetTutorialId: 'MLCoding06C RLVR Reasoning GRPO Agentic RL.md',
        checklist: {
          zh: [
            'DeepSeek-R1 完整演进路径：R1-Zero 纯强化学习顿悟现象 ➔ Cold-Start 格式引导 ➔ Rejection Sampling',
            'GRPO (Group Relative Policy Optimization) 移除 Critic 价值网络原理',
            '从同一 Prompt 采样 G 个输出计算组内相对均值与方差归一化优势推导',
            '规则验证器（Accuracy Reward + Format Reward）替代不稳定 Reward Model 机制',
          ],
          en: [
            'DeepSeek-R1 evolution: R1-Zero pure RL aha moments ➔ Cold-start formatting ➔ Rejection sampling',
            'GRPO mathematical derivation: eliminating the Critic network via group sampling',
            'Group-relative advantage normalization across G sampled completions per prompt',
            'Rule-based verifiable rewards (accuracy + format) replacing brittle reward models',
          ],
        },
      },
      {
        id: 'node-13-agent-loop',
        tracks: ['all', 'llm'],
        title: { zh: '基于 Pi Agent 的 300 行极简智能体循环与状态机', en: 'Minimalist 300-Line Agent Loop & State Machine' },
        subtitle: { zh: 'ReAct (Thought ➔ Action ➔ Observation) 核心状态机，Tool Call JSON 校验与执行路由，长程上下文压缩', en: 'ReAct state machine, tool call JSON schema validation, execution routing & context pruning' },
        difficulty: 'Hard',
        badge: { zh: '工程手撕', en: 'System Code' },
        type: 'code',
        targetTutorialId: 'MLCoding08 Minimalist Agent Loop Architecture 300 Lines Pi Agent.md',
        checklist: {
          zh: [
            'ReAct 状态机循环：Prompt 构建 ➔ LLM 生成 ➔ 工具提取 ➔ 环境执行 ➔ 结果追加',
            'Tool Calling JSON Schema 结构定义与异常格式容错自愈策略',
            '长轮次交互中的上下文滑动窗口与重要观察记忆修剪',
            '多工具并行执行与中断挂起恢复机制',
          ],
          en: [
            'ReAct state machine cycle: prompt construct ➔ LLM generate ➔ tool extract ➔ execute ➔ append',
            'Tool calling JSON schema definition with resilient format repair strategies',
            'Context pruning & sliding window memory management across multi-turn sessions',
            'Parallel tool calling execution & interruption suspension/resumption flow',
          ],
        },
      },
    ],
  },
  {
    stageId: 'stage-05',
    levelNumber: '05',
    title: { zh: '系统规模扩展与工业级落地', en: 'High-Scale ML Systems & RecSys' },
    subtitle: {
      zh: '从 GPU 算子与 3D 分布式并行训练，到工业级推荐排序漏斗与因果 A/B 测试闭环',
      en: 'From GPU kernels & 3D parallelism to industrial multi-stage RecSys & causal A/B testing',
    },
    icon: '🚀',
    nodes: [
      {
        id: 'node-14-recsys-funnel',
        tracks: ['all', 'recsys'],
        title: { zh: '工业级推荐排序、长序列建模与因果 A/B 评估', en: 'Industrial RecSys Funnel, Long Sequences & Causal A/B Testing' },
        subtitle: { zh: '双塔召回 ➔ 精排 ➔ 生成式重排链路，DIN/SIM 序列建模，多目标 MMoE，A/B 测试 CUPED 方差缩减手写', en: 'Multi-stage funnel, DIN/SIM long history, multi-task MMoE, and CUPED variance reduction' },
        difficulty: 'Hard',
        badge: { zh: '业务实战', en: 'Production System' },
        type: 'code',
        targetTutorialId: 'MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md',
        checklist: {
          zh: [
            '推荐与搜索多阶段漏斗：粗排、精排与生成式多样性重排（DPP）',
            '长行为序列建模：DIN 目标驱动注意力与 SIM 两阶段硬检索/软注意力',
            '多目标预估架构：Shared-Bottom、MMoE 与 PLE 负迁移解决机制',
            '在线 A/B 实验因果推断：CUPED 利用前验协变量缩减方差代码实现',
          ],
          en: [
            'Multi-stage funnel: pre-ranking, heavy ranking, and generative reranking with DPP',
            'Long user sequence modeling: DIN target attention & SIM two-stage search',
            'Multi-objective ranking: Shared-Bottom, MMoE, and PLE negative transfer mitigation',
            'Online A/B causal evaluation: CUPED pre-experiment covariate variance reduction from scratch',
          ],
        },
      },
      {
        id: 'node-15-mlsys-parallelism',
        tracks: ['all', 'mlsys', 'llm'],
        title: { zh: 'ML Systems、3D 分布式并行与推理服务引擎', en: 'ML Systems, 3D Parallelism & Inference Engines' },
        subtitle: { zh: 'Megatron-LM 张量并行 (TP)、流水线并行 (PP) 与 ZeRO-1/2/3，Continuous Batching 与 nano-vllm', en: 'Megatron TP/PP/DP, ZeRO memory sharding, continuous batching & nano-vllm engine walkthrough' },
        difficulty: 'Hard',
        badge: { zh: '系统高难度', en: 'Infra' },
        type: 'system',
        targetTutorialId: 'MLSYS10 parallelism.md',
        checklist: {
          zh: [
            '张量并行 (TP) 中 RowParallelLinear 与 ColumnParallelLinear 通信原语分析',
            '流水线并行 (PP) 气泡率 (Bubble Rate) 与 1F1B 调度算法',
            'ZeRO-1（优化器状态）、ZeRO-2（梯度）、ZeRO-3（参数）显存切分收益推导',
            'Continuous Batching 与 PagedAttention 解决显存碎片与并发吞吐瓶颈',
          ],
          en: [
            'Tensor Parallelism (TP) RowParallelLinear & ColumnParallelLinear communication primitives',
            'Pipeline Parallelism (PP) bubble rate & 1F1B schedule derivation',
            'ZeRO-1/2/3 memory sharding gains for optimizer states, gradients, and model weights',
            'Continuous batching & PagedAttention eliminating memory fragmentation in high-concurrency serving',
          ],
        },
      },
    ],
  },
];

const trackOptions = [
  { id: 'all', labelZh: '🌟 全景学习树', labelEn: '🌟 Full Roadmap' },
  { id: 'llm', labelZh: '🤖 LLM & Reasoning 专精', labelEn: '🤖 LLM & Alignment' },
  { id: 'traditional-ml', labelZh: '🌲 Traditional ML 核心', labelEn: '🌲 Classical ML & Math' },
  { id: 'mlsys', labelZh: '⚙️ ML Systems & 分布式', labelEn: '⚙️ ML Systems & Infra' },
  { id: 'recsys', labelZh: '📊 工业推荐与因果推断', labelEn: '📊 RecSys & Causal ML' },
];

export default function StudyPathTree({ language = 'zh', onSelectTutorial }) {
  const isEn = language === 'en';
  const [selectedTrack, setSelectedTrack] = useState('all');
  const [expandedNodes, setExpandedNodes] = useState(() => {
    // Expand the first two stages by default
    const set = new Set();
    studyPathStages.slice(0, 2).forEach((stage) => {
      stage.nodes.forEach((node) => set.add(node.id));
    });
    return set;
  });

  const toggleNode = (nodeId) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };

  const expandAll = () => {
    const all = new Set();
    studyPathStages.forEach((stage) => {
      stage.nodes.forEach((node) => all.add(node.id));
    });
    setExpandedNodes(all);
  };

  const collapseAll = () => {
    setExpandedNodes(new Set());
  };

  // Filter nodes according to selected track
  const filteredStages = studyPathStages.map((stage) => ({
    ...stage,
    filteredNodes: stage.nodes.filter(
      (node) => selectedTrack === 'all' || node.tracks.includes(selectedTrack),
    ),
  })).filter((stage) => stage.filteredNodes.length > 0);

  const totalMatchingNodes = filteredStages.reduce(
    (sum, stage) => sum + stage.filteredNodes.length,
    0,
  );

  return (
    <section className="study-tree-section" id="study-tree" aria-labelledby="study-tree-heading">
      <div className="section-heading tree-header-container">
        <div className="tree-title-group">
          <p className="eyebrow">{isEn ? 'CURRICULUM TREE' : '结构化进阶技能树'}</p>
          <h2 id="study-tree-heading">
            {isEn ? 'Interactive Study Path Tree' : '交互式技能学习路径树'}
          </h2>
          <p className="section-lead">
            {isEn
              ? 'A phased hierarchical study tree designed for ML PhD qualifiers, AI research scientists, and LLM Infra interview preparation.'
              : '专为 ML PhD 资格考、大厂算法研究员与 LLM Infra 高频面试打造的阶梯式学习树。按阶段递进学习，包含手撕代码、理论八股与工业实战。'}
          </p>
        </div>

        <div className="tree-toolbar">
          <div className="tree-track-pills" role="tablist">
            {trackOptions.map((track) => (
              <button
                key={track.id}
                role="tab"
                aria-selected={selectedTrack === track.id}
                className={`tree-track-pill ${selectedTrack === track.id ? 'active' : ''}`}
                onClick={() => setSelectedTrack(track.id)}
                type="button"
              >
                {isEn ? track.labelEn : track.labelZh}
              </button>
            ))}
          </div>

          <div className="tree-action-controls">
            <span className="tree-count-badge">
              {isEn ? `${totalMatchingNodes} Topics in Track` : `当前专精共 ${totalMatchingNodes} 个核心知识节点`}
            </span>
            <button type="button" className="tree-control-btn" onClick={expandAll}>
              {isEn ? 'Expand All' : '全部展开'}
            </button>
            <button type="button" className="tree-control-btn" onClick={collapseAll}>
              {isEn ? 'Collapse All' : '全部收起'}
            </button>
          </div>
        </div>
      </div>

      <div className="study-tree-timeline">
        {filteredStages.map((stage) => (
          <div key={stage.stageId} className="tree-stage-block">
            <div className="stage-milestone-marker">
              <span className="milestone-icon" aria-hidden="true">{stage.icon}</span>
              <div className="milestone-info">
                <div className="milestone-badge">STAGE {stage.levelNumber}</div>
                <h3 className="milestone-title">{isEn ? stage.title.en : stage.title.zh}</h3>
                <p className="milestone-sub">{isEn ? stage.subtitle.en : stage.subtitle.zh}</p>
              </div>
            </div>

            <div className="stage-nodes-container">
              {stage.filteredNodes.map((node) => {
                const isExpanded = expandedNodes.has(node.id);
                return (
                  <div
                    key={node.id}
                    className={`tree-node-card ${isExpanded ? 'is-expanded' : 'is-collapsed'} type-${node.type}`}
                  >
                    <div
                      className="node-header"
                      onClick={() => toggleNode(node.id)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          toggleNode(node.id);
                        }
                      }}
                      aria-expanded={isExpanded}
                    >
                      <div className="node-title-area">
                        <div className="node-meta-row">
                          <span className={`node-type-pill pill-${node.type}`}>
                            {isEn ? node.badge.en : node.badge.zh}
                          </span>
                          <span className={`node-difficulty-badge diff-${node.difficulty.toLowerCase()}`}>
                            {node.difficulty}
                          </span>
                        </div>
                        <h4 className="node-title">{isEn ? node.title.en : node.title.zh}</h4>
                        <p className="node-subtitle">{isEn ? node.subtitle.en : node.subtitle.zh}</p>
                      </div>

                      <div className="node-toggle-actions">
                        <button
                          type="button"
                          className="node-read-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onSelectTutorial && node.targetTutorialId) {
                              onSelectTutorial(node.targetTutorialId);
                            }
                          }}
                        >
                          {isEn ? 'Read Note →' : '查阅笔记 →'}
                        </button>
                        <span className="expand-indicator" aria-hidden="true">
                          {isExpanded ? '▲' : '▼'}
                        </span>
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="node-content-body">
                        <div className="node-checklist-wrap">
                          <div className="checklist-heading">
                            <span className="checklist-icon" aria-hidden="true">🎯</span>
                            <strong>{isEn ? 'Key Examination Points & Mental Model' : '面试高频拆解与必考自查'}</strong>
                          </div>
                          <ul className="node-checklist">
                            {(isEn ? node.checklist.en : node.checklist.zh).map((item, idx) => (
                              <li key={idx} className="checklist-item">
                                <span className="check-bullet" aria-hidden="true">✓</span>
                                <span>{item}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
