import React, { useState } from 'react';

export const knowledgeMapDomains = [
  {
    id: 'llm-arch',
    category: 'llm',
    categoryLabel: { zh: '大模型核心', en: 'LLM Core' },
    title: { zh: 'Transformer 与注意力核心架构', en: 'Transformer & Attention Architecture' },
    icon: '🧠',
    badge: { zh: '核心底座', en: 'Core Backbone' },
    description: {
      zh: '从 Unicode 预分词与 BPE 词表构建，到多头注意力 (MHA/GQA) 纯手写、KV Cache 硬件显存瓶颈与 LoRA/MoE 架构扩展。',
      en: 'From Unicode pre-tokenization & BPE vocabulary construction to from-scratch attention (MHA/GQA), KV Cache memory rooflines, and LoRA/MoE extensions.',
    },
    keyConcepts: [
      'Byte-level BPE',
      'Decoder-Only',
      'Attention Zoo (MHA/GQA/Window)',
      'KV Cache Roofline',
      'LoRA PEFT',
      'MoE Routing',
    ],
    keyNotes: [
      {
        id: 'MLCoding03 Attention Variants GQA Sliding Window KV Cache.md',
        title: { zh: '注意力算子从零手撕 (MHA/GQA/Sliding/KV)', en: 'Attention Zoo from Scratch (MHA/GQA/KV)' },
        difficulty: 'Hard',
        type: 'code',
      },
      {
        id: 'MLCoding01B Transformer Architecture Variants Attention FLOPs KV Cache.md',
        title: { zh: 'Transformer 架构变体、FLOPs 与 KV Cache 估算', en: 'Transformer Variants, FLOPs & KV Cache Roofline' },
        difficulty: 'Medium',
        type: 'theory',
      },
      {
        id: 'MLCoding01 Unicode Pretokenization.md',
        title: { zh: 'Unicode Pretokenization 与 BPE 从零实现', en: 'Unicode Pretokenization & BPE from Scratch' },
        difficulty: 'Medium',
        type: 'code',
      },
      {
        id: 'MLCoding04 LoRA ViT Patch Embedding MoE.md',
        title: { zh: 'LoRA 低秩微调、ViT Patch 与 MoE 混合专家', en: 'LoRA PEFT, ViT Patch Embedding & MoE' },
        difficulty: 'Hard',
        type: 'code',
      },
    ],
    sectionId: 'mlcoding',
  },
  {
    id: 'llm-alignment',
    category: 'llm',
    categoryLabel: { zh: '大模型核心', en: 'LLM Core' },
    title: { zh: '后训练偏好对齐与 RLVR 推理 (DeepSeek-R1 范式)', en: 'Post-Training Alignment & RLVR Reasoning' },
    icon: '🎯',
    badge: { zh: '前沿热点', en: 'Frontier' },
    description: {
      zh: 'RLHF 偏好建模与 PPO 4 模型通信架构，DPO/IPO/KTO 隐式奖励闭式转化，以及 DeepSeek-R1 演进路径与 GRPO 规则可验证奖励。',
      en: 'RLHF preference modeling with PPO 4-model topology, DPO/IPO/KTO closed-form rewards, and DeepSeek-R1 evolution with GRPO verifiable rewards.',
    },
    keyConcepts: [
      'DPO / IPO / KTO / SimPO',
      'PPO Actor-Critic & GAE',
      'GRPO Group Advantage',
      'DeepSeek-R1 Paradigm',
      'Rule-Based Verifiable Rewards (RLVR)',
      'Cold-Start SFT',
    ],
    keyNotes: [
      {
        id: 'MLCoding06C RLVR Reasoning GRPO Agentic RL.md',
        title: { zh: 'RLVR 推理模型与 GRPO 算法推导', en: 'RLVR Reasoning Models & GRPO Derivation' },
        difficulty: 'Hard',
        type: 'theory',
      },
      {
        id: 'MLCoding06B RLHF Preference Alignment PPO DPO.md',
        title: { zh: 'RLHF 偏好对齐与 PPO/DPO 全景', en: 'RLHF Preference Alignment & PPO/DPO Panorama' },
        difficulty: 'Hard',
        type: 'theory',
      },
      {
        id: 'MLCoding06 INT8 Quantization DPO GRPO PPO OPD Loss.md',
        title: { zh: 'DPO / GRPO / PPO 损失函数纯手写', en: 'From-Scratch DPO / GRPO / PPO Loss Functions' },
        difficulty: 'Hard',
        type: 'code',
      },
      {
        id: 'MLSYS15 RL Infra 自测 35 问.md',
        title: { zh: '强化学习基础设施 35 问自测', en: 'RL Infrastructure 35-Question Self-Test' },
        difficulty: 'Medium',
        type: 'quiz',
      },
    ],
    sectionId: 'mlcoding',
  },
  {
    id: 'inference-agents',
    category: 'llm',
    categoryLabel: { zh: '大模型核心', en: 'LLM Core' },
    title: { zh: '推理解码加速与自主智能体系统', en: 'Inference Serving & Autonomous Agents' },
    icon: '⚡',
    badge: { zh: '工程手写', en: 'System & Code' },
    description: {
      zh: 'Top-k / Top-p / 投机采样解码加速，INT8 动态量化与反量化矩阵算子，300 行极简 Pi Agent 核心循环与 ReAct 状态机。',
      en: 'Top-k / Top-p / speculative decoding acceleration, INT8 dynamic quantization operators, and a 300-line minimalist Pi Agent loop with ReAct state machines.',
    },
    keyConcepts: [
      'Top-k / Top-p / Temp Sampling',
      'Speculative Decoding',
      'INT8 Dynamic Quantization',
      'Minimalist 300-line Agent Loop',
      'ReAct State Machine',
      'Tool Calling JSON Schema',
    ],
    keyNotes: [
      {
        id: 'MLCoding05 Sampling Beam Search Speculative Decoding.md',
        title: { zh: '推理解码策略：采样与投机解码从零手写', en: 'Decoding Strategies: Sampling & Speculative Decoding' },
        difficulty: 'Medium',
        type: 'code',
      },
      {
        id: 'MLCoding08 Minimalist Agent Loop Architecture 300 Lines Pi Agent.md',
        title: { zh: '智能体架构：基于 Pi Agent 的 300 行核心循环', en: 'Minimalist 300-Line Agent Loop & State Machine' },
        difficulty: 'Hard',
        type: 'code',
      },
      {
        id: 'MLSYS15 LLM Inference Speculative Decoding DFlash.md',
        title: { zh: '并行解码与草稿验证机制', en: 'Parallel Decoding & Draft Verification' },
        difficulty: 'Medium',
        type: 'system',
      },
      {
        id: 'BusinessAlgorithm06 Agentic Search.md',
        title: { zh: 'RAG 与 Agentic 搜索系统', en: 'RAG & Agentic Search Systems' },
        difficulty: 'Medium',
        type: 'system',
      },
    ],
    sectionId: 'mlcoding',
  },
  {
    id: 'traditional-ml-core',
    category: 'traditional-ml',
    categoryLabel: { zh: '传统机器学习', en: 'Traditional ML' },
    title: { zh: '机器学习核心机制、八股与经典算子', en: 'Core ML Fundamentals & Classical Operators' },
    icon: '⚙️',
    badge: { zh: '高频必考', en: 'High Frequency' },
    description: {
      zh: '数据预处理规范与时序泄露防御，数值防溢出 Log-Sum-Exp 与 NaN 排查，经典损失全景（BCE/Focal/Triplet），BatchNorm/RMSNorm/Dropout 机制与 Conv2d 卷积手写。',
      en: 'Preprocessing best practices & leakage defense, numerical Log-Sum-Exp & NaN debugging, loss zoo (BCE/Focal/Triplet), normalization zoo (BatchNorm/RMSNorm/Dropout), and Conv2d from scratch.',
    },
    keyConcepts: [
      'Feature Imputation & Standardization',
      'Log-Sum-Exp & NaN Debugging',
      'Loss Zoo (MSE/BCE/Focal/Triplet)',
      'BatchNorm vs RMSNorm vs LayerNorm',
      'Inverted Dropout',
      'Conv2d im2col GEMM',
    ],
    keyNotes: [
      {
        id: 'MLCoding00 ML Basics Data Preprocessing Loss Functions.md',
        title: { zh: '数据预处理、防泄露与经典损失函数全景', en: 'Data Preprocessing, Leakage & Loss Functions' },
        difficulty: 'Easy',
        type: 'theory',
      },
      {
        id: 'MLCoding02 GELU BatchNorm Conv2d Linear Regression.md',
        title: { zh: 'BatchNorm / RMSNorm / Dropout / Conv2d 纯手写', en: 'Norm Zoo, Dropout & Conv2d from Scratch' },
        difficulty: 'Medium',
        type: 'code',
      },
      {
        id: 'MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.md',
        title: { zh: '混合精度训练体系与架构选型基础', en: 'Mixed-Precision Training & Architecture Choices' },
        difficulty: 'Medium',
        type: 'theory',
      },
    ],
    sectionId: 'mlcoding',
  },
  {
    id: 'trees-ensembles',
    category: 'traditional-ml',
    categoryLabel: { zh: '传统机器学习', en: 'Traditional ML' },
    title: { zh: '树模型体系、随机森林手撕与漂移检验', en: 'Tree Models, Ensembles & Drift Testing' },
    icon: '🌲',
    badge: { zh: '工业基石', en: 'Industrial Pillar' },
    description: {
      zh: '决策树分裂不纯度严格凹性证明（Gini/Entropy），随机森林 Bootstrap 重采样与 OOB 评估手撕，XGBoost 二阶展开与 LightGBM，多元分布漂移检验体系（KS/PSI/C2ST）。',
      en: 'Strict concavity proofs for Gini & Entropy, Bagging & Random Forest from scratch with OOB evaluation, XGBoost 2nd-order Taylor, and distribution drift testing (KS/PSI/C2ST).',
    },
    keyConcepts: [
      'Gini Impurity & Entropy Concavity',
      'Bootstrap Aggregation & Subspaces',
      'Out-of-Bag (OOB) Generalization',
      'XGBoost 2nd-order Taylor Expansion',
      'Kolmogorov-Smirnov & PSI',
      'C2ST Classifier Two-Sample Test',
    ],
    keyNotes: [
      {
        id: 'MLCoding09 Data Science Statistical Testing Distribution Drift C2ST.md',
        title: { zh: '树模型体系、随机森林手写与分布漂移检验', en: 'Decision Trees, Random Forest & Drift Testing' },
        difficulty: 'Medium',
        type: 'code',
      },
      {
        id: 'MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md',
        title: { zh: '工业精排模型体系与树模型应用', en: 'Industrial Ranking Models & Trees' },
        difficulty: 'Hard',
        type: 'system',
      },
    ],
    sectionId: 'mlcoding',
  },
  {
    id: 'optimization-dynamics',
    category: 'traditional-ml',
    categoryLabel: { zh: '传统机器学习', en: 'Traditional ML' },
    title: { zh: '现代优化器动力学与矩阵谱正交方法', en: 'Optimization Dynamics & Spectral Orthogonalization' },
    icon: '🚀',
    badge: { zh: '数学推导', en: 'Theory & Code' },
    description: {
      zh: 'SGD 动量与 Nesterov 动力学机制，AdamW 解耦权重衰减、二阶矩偏差校正与稀疏陷阱，Muon 优化器与 Newton-Schulz 极分解正交投影从零实现。',
      en: 'SGD Momentum & Nesterov acceleration, AdamW decoupled weight decay & second moment bias correction, and the Muon optimizer with Newton-Schulz polar orthogonalization from scratch.',
    },
    keyConcepts: [
      'SGD Momentum & Nesterov',
      'AdamW Decoupled Weight Decay',
      'Second Moment Bias Correction',
      'Sparse Update Traps',
      'Muon Optimizer',
      'Newton-Schulz Polar Factorization',
    ],
    keyNotes: [
      {
        id: 'MLCoding10 Modern Optimizers SGD AdamW Muon.md',
        title: { zh: '从 SGD 动量到 AdamW 与 Muon 谱正交纯手写', en: 'From SGD Momentum to AdamW & Muon from Scratch' },
        difficulty: 'Hard',
        type: 'code',
      },
    ],
    sectionId: 'mlcoding',
  },
  {
    id: 'mlsys-infra',
    category: 'mlsys',
    categoryLabel: { zh: '系统与算子', en: 'ML Systems' },
    title: { zh: 'GPU 算子工程、3D 并行与推理系统', en: 'GPU Kernels, 3D Parallelism & Serving' },
    icon: '🛠️',
    badge: { zh: '高难度系统', en: 'High Scale' },
    description: {
      zh: 'GPU 内存层次与 Roofline 分析，CUDA Reduce/Scan 原语，Megatron-LM 3D 分布式并行（TP/PP/DP/ZeRO），Continuous Batching 与 PagedAttention 虚拟显存映射。',
      en: 'GPU memory hierarchies & Roofline modeling, CUDA parallel primitives, Megatron-LM 3D parallelism (TP/PP/DP/ZeRO), continuous batching & PagedAttention virtual memory mapping.',
    },
    keyConcepts: [
      'GPU SRAM / HBM Roofline',
      'CUDA Shared Memory & Tiling',
      'Megatron 3D Parallelism (TP/PP/DP)',
      'ZeRO-1/2/3 Memory Sharding',
      'Continuous Batching & PagedAttention',
      'nano-vllm Scheduler Loop',
    ],
    keyNotes: [
      {
        id: 'MLSYS1.md',
        title: { zh: 'GPU 体系结构入门', en: 'GPU Architecture Fundamentals' },
        difficulty: 'Easy',
        type: 'system',
      },
      {
        id: 'MLSYS3.md',
        title: { zh: 'Roofline 性能瓶颈分析', en: 'Roofline Performance Bottleneck Analysis' },
        difficulty: 'Medium',
        type: 'theory',
      },
      {
        id: 'MLSYS10 parallelism.md',
        title: { zh: '分布式训练 3D 并行范式', en: '3D Distributed Parallelism Training Paradigms' },
        difficulty: 'Hard',
        type: 'system',
      },
      {
        id: 'MLSYS11 nano-vllm-1.md',
        title: { zh: 'nano-vllm 推理引擎源码精读', en: 'nano-vllm Inference Engine Source Code Walkthrough' },
        difficulty: 'Hard',
        type: 'code',
      },
    ],
    sectionId: 'mlsys',
  },
  {
    id: 'recsys-funnel',
    category: 'recsys',
    categoryLabel: { zh: '工业搜索推荐', en: 'Industrial RecSys' },
    title: { zh: '搜索推荐多阶段链路与因果 A/B 实验', en: 'Multi-Stage Funnel & Causal A/B Testing' },
    icon: '📊',
    badge: { zh: '业务实战', en: 'Production' },
    description: {
      zh: '多阶段漏斗链路（双塔召回、精排、重排），DIN 局部注意力与 SIM 超长序列检索，多目标预估（MMoE/PLE），A/B 在线实验与 CUPED 方差缩减技术。',
      en: 'Multi-stage funnel (retrieval, ranking, rerank), DIN local attention & SIM long sequence retrieval, multi-task ranking (MMoE/PLE), A/B testing & CUPED variance reduction.',
    },
    keyConcepts: [
      'Two-Tower Vector Retrieval',
      'DIN / SIM User Sequence Modeling',
      'MMoE / PLE Multi-Task Learning',
      'Generative Reranking & DPP',
      'A/B Testing & CUPED Variance Reduction',
    ],
    keyNotes: [
      {
        id: 'MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md',
        title: { zh: '精排模型、长序列、生成式重排与因果推断', en: 'Heavy Ranking, Long Sequences, Reranking & Causal ML' },
        difficulty: 'Hard',
        type: 'code',
      },
      {
        id: 'Business Algorithm TODO.md',
        title: { zh: '推荐与搜索多阶段链路导论', en: 'Multi-Stage Funnel Architecture Overview' },
        difficulty: 'Easy',
        type: 'system',
      },
      {
        id: 'BusinessAlgorithm01B Vector Retrieval.md',
        title: { zh: '双塔模型、负采样与向量检索', en: 'Two-Tower Retrieval, Negative Sampling & ANN' },
        difficulty: 'Medium',
        type: 'system',
      },
      {
        id: 'BusinessAlgorithm02 Ranking.md',
        title: { zh: '排序目标与离线评价指标', en: 'Ranking Objectives & Offline Metrics' },
        difficulty: 'Medium',
        type: 'theory',
      },
    ],
    sectionId: 'business-algorithm',
  },
  {
    id: 'quant-math',
    category: 'quant',
    categoryLabel: { zh: '量化与算法', en: 'Quant & Algorithms' },
    title: { zh: '量化概率统计、随机微积分与高频算法', en: 'Quant Probability, Stochastic Calculus & Algo' },
    icon: '📐',
    badge: { zh: '硬核数学', en: 'Math & Algo' },
    description: {
      zh: '离散概率期望、Markov 链与吸收时间、高维积分、Black-Scholes 期权定价公式，以及 LeetCode 单调栈、滑动窗口与动态规划经典模式。',
      en: 'Discrete probability expectation, Markov chains & absorbing times, high-dimensional integration, Black-Scholes pricing, and LeetCode algorithmic patterns.',
    },
    keyConcepts: [
      'Expectation & Recurrence',
      'Markov Chains & Transition Matrices',
      'Stochastic Calculus & Ito Lemma',
      'Monotonic Stack & Sliding Window',
      'Dynamic Programming Invariants',
    ],
    keyNotes: [
      {
        id: 'Quant01 Expectation Counting Multinomial.md',
        title: { zh: '期望、计数与递推基础', en: 'Expectation, Counting & Recurrence' },
        difficulty: 'Easy',
        type: 'theory',
      },
      {
        id: 'Quant02 Markov Chains Expected Time.md',
        title: { zh: 'Markov 链与吸收时间计算', en: 'Markov Chains & Absorbing Time' },
        difficulty: 'Medium',
        type: 'theory',
      },
      {
        id: 'CoreSkills17 Stack MinStack Monotonic Stack.md',
        title: { zh: '单调栈模板与高频算法手写', en: 'Monotonic Stack Patterns & Implementations' },
        difficulty: 'Medium',
        type: 'code',
      },
    ],
    sectionId: 'quant',
  },
];

const categoryFilterOptions = [
  { id: 'all', labelZh: '全部领域', labelEn: 'All Domains' },
  { id: 'llm', labelZh: '大模型核心', labelEn: 'LLM Core' },
  { id: 'traditional-ml', labelZh: '传统机器学习', labelEn: 'Traditional ML' },
  { id: 'mlsys', labelZh: '系统与算子', labelEn: 'ML Systems' },
  { id: 'recsys', labelZh: '工业搜索推荐', labelEn: 'Industrial RecSys' },
  { id: 'quant', labelZh: '量化与高频算法', labelEn: 'Quant & Math' },
];

export default function KnowledgeMap({ language = 'zh', onSelectTutorial, onSelectSection }) {
  const isEn = language === 'en';
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  const filteredDomains = knowledgeMapDomains.filter((domain) => {
    if (selectedCategory !== 'all' && domain.category !== selectedCategory) {
      return false;
    }
    if (!searchTerm.trim()) return true;
    const term = searchTerm.toLowerCase();
    const title = (domain.title[language] || domain.title.zh).toLowerCase();
    const desc = (domain.description[language] || domain.description.zh).toLowerCase();
    const concepts = domain.keyConcepts.some((c) => c.toLowerCase().includes(term));
    return title.includes(term) || desc.includes(term) || concepts;
  });

  return (
    <section className="knowledge-map-section" id="knowledge-map" aria-labelledby="map-heading">
      <div className="section-heading map-header-container">
        <div className="map-title-group">
          <p className="eyebrow">{isEn ? 'KNOWLEDGE TOPOLOGY' : '全景知识图谱'}</p>
          <h2 id="map-heading">
            {isEn ? 'Interactive Knowledge Map' : '全景知识地图'}
          </h2>
          <p className="section-lead">
            {isEn
              ? 'A topological roadmap connecting foundational theory, scratch operator code, ML systems, and industrial AI.'
              : '以领域拓扑为脉络，串联底层理论、核心算子从零手写、系统推理加速与工业级实践。点击任意代表笔记即可直达精读。'}
          </p>
        </div>

        <div className="map-toolbar">
          <div className="map-filter-pills" role="tablist">
            {categoryFilterOptions.map((opt) => (
              <button
                key={opt.id}
                role="tab"
                aria-selected={selectedCategory === opt.id}
                className={`map-filter-pill ${selectedCategory === opt.id ? 'active' : ''}`}
                onClick={() => setSelectedCategory(opt.id)}
                type="button"
              >
                {isEn ? opt.labelEn : opt.labelZh}
              </button>
            ))}
          </div>

          <div className="map-search-box">
            <span className="search-icon" aria-hidden="true">🔍</span>
            <input
              type="search"
              placeholder={isEn ? 'Filter topics, e.g. Attention, GRPO, AdamW...' : '筛选领域或概念，如 Attention、GRPO、AdamW...'}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="map-search-input"
            />
          </div>
        </div>
      </div>

      <div className="map-domain-grid">
        {filteredDomains.map((domain) => (
          <div key={domain.id} className={`map-domain-card domain-cat-${domain.category}`}>
            <div className="domain-card-header">
              <div className="domain-icon-wrap">
                <span className="domain-icon" aria-hidden="true">{domain.icon}</span>
                <div>
                  <span className="domain-category-badge">
                    {isEn ? domain.categoryLabel.en : domain.categoryLabel.zh}
                  </span>
                  <h3 className="domain-title">{isEn ? domain.title.en : domain.title.zh}</h3>
                </div>
              </div>
              <span className="domain-status-badge">
                {isEn ? domain.badge.en : domain.badge.zh}
              </span>
            </div>

            <p className="domain-description">
              {isEn ? domain.description.en : domain.description.zh}
            </p>

            <div className="domain-concepts-wrap">
              <span className="domain-concepts-label">{isEn ? 'Core Concepts:' : '核心知识点:'}</span>
              <div className="domain-concept-tags">
                {domain.keyConcepts.map((concept) => (
                  <span key={concept} className="concept-tag">{concept}</span>
                ))}
              </div>
            </div>

            <div className="domain-notes-section">
              <div className="domain-notes-header">
                <span className="domain-notes-title">{isEn ? 'Featured Notes' : '代表性核心笔记'}</span>
                <button
                  type="button"
                  className="domain-section-link"
                  onClick={() => onSelectSection && onSelectSection(domain.sectionId)}
                >
                  {isEn ? 'All Notes →' : '进入该板块 →'}
                </button>
              </div>

              <div className="domain-notes-list">
                {domain.keyNotes.map((note) => (
                  <button
                    key={note.id}
                    type="button"
                    className="domain-note-pill"
                    onClick={() => onSelectTutorial && onSelectTutorial(note.id)}
                  >
                    <span className={`note-type-indicator type-${note.type}`}>
                      {note.type === 'code' ? '⚡' : note.type === 'theory' ? '📐' : note.type === 'system' ? '🛠️' : '📝'}
                    </span>
                    <span className="domain-note-name">
                      {isEn ? note.title.en : note.title.zh}
                    </span>
                    <span className="note-difficulty-tag">{note.difficulty}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
