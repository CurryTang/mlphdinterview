# ML Coding 07 · 工业级机器学习体系：推荐精排、长序列建模、生成式重排与 A/B 测试因果推断

在工业界机器学习（Industrial Machine Learning）与大规模互联网推荐系统（Large-Scale Recommender Systems）中，算法工程师不仅要精通模型底层架构设计与损失函数优化，更要深刻理解**从海量稀疏样本流、终身行为序列建模、全屏生成式重排决策，到多维指标评测体系与在线 A/B 实验因果推断**的完整工业级闭环。

本篇将工业界端到端核心知识体系凝练为八大结构化逻辑模块：
1. **推荐精排基座模型体系与多目标负迁移（Seesaw Effect）全方位治理**
2. **训练样本规模构建、负采样范式与数学概率还原公式推导**
3. **工业级长行为序列建模五大架构对比、时延权衡与陈旧噪声治理**
4. **电商生成式重排全链路（候选 Token 化、前缀约束束搜索、Listwise 目标与 P99 ≤ 20ms 时延熔断）**
5. **多维度指标金字塔体系（ROC-AUC, GAUC 用户/请求分组、无偏估计、校准曲线 PCOC 与看板设计）**
6. **工业级在线实验与 A/B 测试全生命周期规范（CUPED 方差缩减、SRM 检验与决策红线）**
7. **经典实战案例库：注册漏斗 $2 \times 2$ 全因子实验、商详页改版转化归因与极小样本分层推断**
8. **资深算法专家高频面试标准答题模板与方法论**

---

## 模块一：推荐排序基座模型家族与多目标“跷跷板效应”治理

```text
工业级推荐精排模型演进三维谱系：
┌────────────────────────────────────────────────────────────────────────┐
│ 1. 显式与隐式特征交叉 (Feature Interaction)                            │
│ • 浅层线性 ➔ FM / FFM ➔ Wide&Deep ➔ DeepFM ➔ DCN-v2 (低秩张量交叉)    │
│ • 核心能力: 捕捉高阶非线性特征组合，兼顾泛化记忆与特征组合能力        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. 用户长短期行为序列建模 (User Behavior Modeling)                     │
│ • Pooling (Sum/Mean) ➔ DIN (目标注意力 Target Attention) ➔             │
│ • SIM (两阶段超长序列检索: 硬检索 Hard-search + 软检索 Soft-search)    │
│ • 核心能力: 从用户数百乃至数万条历史交互序列中，动态提取与候选物关联的兴趣│
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. 多目标联合学习与负荷路由 (Multi-Task Learning, MTL)                 │
│ • Shared-Bottom ➔ MMoE (多门控混合专家) ➔ PLE (渐进式分层抽取路由)    │
│ • 核心能力: 解决多任务之间的梯度冲突与负迁移 (Negative Transfer)       │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 经典工业级基座模型三大支柱（含业务算法专篇深度链接与伪代码解析）

工业界精排模型必须在极严苛的 $P99 \le 15\text{ms}$ 延迟下完成高维交叉与数十万用户行为的精准激活。以下三大支柱的具体实现与 PyTorch 伪代码均已深度沉淀在业务算法专篇中：

#### 支柱一：特征交叉基座（Feature Interaction Backbone）
> 📘 **详见专篇**：[第 11 章 · 特征交叉、粗排与个性化](file:///Users/czk/Documents/mlsysnotes/MLSYS_tutorial/notes/BusinessAlgorithm/BusinessAlgorithm02C%20Feature%20Interaction.md)（含 FM / DCN-v2 / DLRM 完整代码）
- **DCN-v2 (Deep & Cross Network v2)**：引入低秩矩阵分解（$\mathbf{W}_l = \mathbf{U}_l \mathbf{V}_l^T$）与 MoE 子空间门控，以 $\mathcal{O}(d \cdot r)$ 极低复杂度计算显式高阶多项式特征交叉：
  $$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l$$
- **DLRM (Deep Learning Recommendation Model)**：Meta 经典架构，将特征解耦为 Dense Bottom MLP 与 Sparse Embedding Tables，通过显式矩阵点积（`torch.bmm(E, E.T)`）提取上三角交互特征送入 Top MLP。

#### 支柱二：行为序列建模基座（User Behavior Sequence Backbone）
> 📘 **详见专篇**：[第 12 章 · 用户行为序列建模](file:///Users/czk/Documents/mlsysnotes/MLSYS_tutorial/notes/BusinessAlgorithm/BusinessAlgorithm02D%20User%20Sequences.md)（含 DIN 目标注意力与 SIM 两阶段检索代码）
- **DIN (Deep Interest Network)**：利用候选 Item $\mathbf{q}$ 对历史序列 $[\mathbf{h}_1, \dots, \mathbf{h}_L]$ 计算 **Target Attention**：$\mathbf{u}(\mathbf{q}) = \sum \alpha_j \mathbf{h}_j$，其中 $\alpha_j = \text{MLP}([\mathbf{q}, \mathbf{h}_j, \mathbf{q}-\mathbf{h}_j, \mathbf{q}\odot\mathbf{h}_j])$；
- **SIM (Search-based Interest Model)**：两阶段解耦——第一阶段 **Hard Search** 按同类目粗筛 Top-50，第二阶段在小集合上结合时间差 $\Delta t$ 进行精细 Target Attention，突破数万步长序列建模瓶颈。

#### 支柱三：多目标排序基座（Multi-Task Learning Backbone）
> 📘 **详见专篇**：[第 10 章 · 多目标学习与分值融合](file:///Users/czk/Documents/mlsysnotes/MLSYS_tutorial/notes/BusinessAlgorithm/BusinessAlgorithm02B%20Multi-Objective%20Ranking.md)（含 MMoE 与 PLE 分层解耦专家代码）
- **MMoE (Multi-gate Mixture-of-Experts)**：多任务共享 Experts 池，各任务使用 Softmax 门控加权 $\mathbf{h}_t = \sum g_{t,e} f_e(\mathbf{x})$；
- **PLE (Progressive Layered Extraction)**：显式区分为 **Task-Specific 独占专家** 与 **Shared 共享专家**，物理隔离不同任务表征，彻底阻断 CTR 与 CVR 之间的负迁移与跷跷板效应。

### 2. 多目标表征冲突与“跷跷板效应（Seesaw Effect）”底层机理

```text
多目标表征冲突三大根因示意:
┌────────────────────────────────────────────────────────────────────────┐
│ 根因 1: 梯度方向负冲突 (Gradient Conflict: cos(g_A, g_B) < 0)           │
│ • Task A (点击) 梯度推动参数向"吸引人标题"方向更新                     │
│ • Task B (完播/购买) 梯度推动参数向"深度高质量内容"方向更新 ➔ 共享参数撕裂│
├────────────────────────────────────────────────────────────────────────┤
│ 根因 2: 梯度量级与样本频次压制 (Magnitude & Frequency Imbalance)       │
│ • 高频正样本 Task (CTR 约 5%) 梯度范数 ≫ 低频稀疏 Task (CVR 约 0.1%)   │
│ • 共享 Embedding 与底层权重被高频任务完全主导，低频核心任务严重欠拟合  │
├────────────────────────────────────────────────────────────────────────┤
│ 根因 3: 样本空间不一致与选择偏差 (Sample Space Mismatch)               │
│ • CTR 在全量曝光空间训练: D_imp                                         │
│ • CVR 传统上仅在点击后样本空间训练: D_click (非随机丢弃 ➔ 样本选择偏差)│
└────────────────────────────────────────────────────────────────────────┘
```

#### 六大多目标架构与优化方案深度对比

| 解决方案类别 | 核心方法 | 拓扑机制与数学定义 | 负迁移防御力 | 工业界核心洞察 |
|---|---|---|---|---|
| **架构路由** | **Shared-Bottom** | 共享单一底层 MLP，顶部分叉 Task Towers。 | **极差（0 分）** | 任务间梯度强行对冲，跷跷板效应最剧烈。 |
| **架构路由** | **MMoE** | 共享 Experts 池，每个任务拥有独立的 Softmax 门控：$g_t(x) = \text{Softmax}(W_t x)$。 | **中等** | 实现了动态软路由，但所有 Expert 仍是全局共享的，弱相关任务仍会争抢容量。 |
| **架构路由** | **PLE (Progressive Extraction)** | 显式解耦为 **Task-Specific 独占专家** 与 **Shared 共享专家**，分层渐进抽取。 | **卓越（SOTA）** | **任务私有特征与共享特征严格物理隔离**，彻底阻断不同任务间的负迁移。 |
| **损失平衡** | **Uncertainty Weighting** | 建模同方差不确定性 $\sigma_k^2$：<br>$$\mathcal{L} = \sum \left( \frac{1}{2\sigma_k^2}\mathcal{L}_k + \ln \sigma_k \right)$$ | **优良** | 自动根据任务方差与噪声动态调整损失权重，避免人工粗暴调参。 |
| **梯度手术** | **PCGrad (Gradient Projection)** | 当梯度冲突 $\mathbf{g}_i \cdot \mathbf{g}_j < 0$ 时，将 $\mathbf{g}_i$ 正交投影到 $\mathbf{g}_j$ 的法平面：<br>$$\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2}\mathbf{g}_j$$ | **极强** | 从梯度动力学层面消除相互抵消的破坏性分量。 |
| **决策融合** | **Constrained Pareto Fusion** | 拉格朗日乘子约束优化与实时 PID 控权：<br>$$\max \text{GMV} \text{ s.t. } \text{CTR} \ge \text{CTR}_0$$ | **线上闭环** | 替代静态超参融合，自适应追踪业务护栏。 |

---

## 模块二：训练样本规模、负采样方案与概率还原数学推导

在工业界，精排模型（Ranking）与召回模型（Retrieval）在样本构建和采样范式上存在本质区别：

```text
召回 (Retrieval) vs 精排 (Ranking) 样本构建本质差异：
┌───────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 环节                  │ 样本池来源与负样本采样策略                                             │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 召回阶段 (Retrieval)  │ • 候选空间: 全库数亿级候选 Item。                                      │
│                       │ • 负样本机制: 全库随机负采样 (Random Negatives) + 曝光未点击弱负例     │
│                       │   + 粗排截断硬负例 (Hard Negatives) + Cross-Batch In-batch Negatives。 │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 精排阶段 (Ranking)    │ • 候选空间: 经前序召回与粗排筛选后的 Top 1000~3000 候选集。            │
│                       │ • 正样本: 用户真实产生交互的行为（如点击、点赞、购买、完播）。         │
│                       │ • 负样本: 真正展现给用户但用户未产生交互的曝光（Impression-Unclicked）。│
│                       │ • 严禁机制: 精排严禁随意引入全库随机未曝光负样本（否则破坏真实曝光分布）│
└───────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

### 1. 训练样本规模（Sample Size & Streaming Pipeline）
- **滚动时间窗口（Rolling Horizon）**：通常采用近 **14 天 ~ 30 天** 的全量曝光与点击日志（规模常达 **数十亿至数百亿条样本**）；
- **样本新鲜度与在线流式更新（Online Streaming Update）**：
  - 离线 Batch 训练更新天级基座大模型（Daily Checkpoints）；
  - 线上准实时流式计算管道（Flink/Kafka）在数分钟内拼接曝光与点击日志，对模型权重与 Embedding 表执行流式增量微调（Streaming Hourly / Real-time Updates），捕捉突发热点。

---

### 2. 负样本下采样（Negative Downsampling）与概率还原数学推导

#### 为什么工业级精排必须做负样本下采样？
在信息流或电商推荐中，真实点击率通常极低（如 $p \approx 1\% \sim 5\%$），正负样本极度不平衡（$1:20 \sim 1:100$）。
1. **节省训练计算资源与存储**：下采样负样本可将训练集体积缩减 70%~90%，大幅提高梯度迭代吞吐量；
2. **防止梯度被海量易分负样本淹没**。

#### 下采样率对预估概率的系统性扭曲
假设我们将未点击负样本以采样率 $w \in (0, 1]$ 进行随机下采样（保留比例为 $w$），而正样本全部保留。
下采样后，模型学到的条件概率 $\hat{p} = P(Y=1 \mid X, \text{sampled})$ 会被**人为大幅拉高**：

$$\hat{p} = \frac{P(Y=1 \mid X)}{P(Y=1 \mid X) + w \cdot P(Y=0 \mid X)} = \frac{p}{p + w(1-p)}$$

#### 线上推理时的概率校准还原公式（Probability Recovery Formula）
在竞价广告（$eCPM = pCTR \times pCVR \times \text{Bid}$）或需要精准概率融合的业务中，必须在模型输出后通过解析反函数将 $\hat{p}$ **精确还原为真实自然世界概率 $p$**：

$$p = \frac{\hat{p}}{\hat{p} + \frac{1 - \hat{p}}{w}}$$

```text
概率还原验证示例：
若真实 CTR p = 0.01 (1%)，负样本下采样保留率 w = 0.1 (10%):
1. 训练集上模型观测到的 CTR: p_hat = 0.01 / (0.01 + 0.1 * 0.99) = 0.01 / 0.109 = 0.0917 (9.17%)
2. 线上推理时通过公式还原: p = 0.0917 / (0.0917 + (1 - 0.0917) / 0.1) = 0.01 (精准复原为 1%!)
```

---

## 模块三：工业级长行为序列建模五大架构体系深度剖析

```text
工业级推荐序列建模五大架构演进脉络：
┌────────────────────────────────────────────────────────────────────────┐
│ 1. 截断自注意力 (Truncated Transformers: SASRec / BERT4Rec / BST)        │
│ • 机制: 截断至最近 N=50~100 行为, 标准 O(N²) Self-Attention             │
├────────────────────────────────────────────────────────────────────────┤
│ 2. 压缩记忆网络 (Memory / Compressive Networks: MIMN / Neural Memory)  │
│ • 机制: 维护固定槽位记忆矩阵 M ∈ R^(C×d), 增量写入与压缩, O(1) 在线读取  │
├────────────────────────────────────────────────────────────────────────┤
│ 3. 终身目标注意力 (Lifelong Target-Attention: DIN / DIEN)               │
│ • 机制: 以候选 Item 为 Query, 对全量 L 序列直接执行 Target-Attention    │
├────────────────────────────────────────────────────────────────────────┤
│ 4. 分层多分辨率池化 (Hierarchical Multi-Resolution Pooling: HPMN)       │
│ • 机制: Session级 ➔ 天级 ➔ 月级多粒度聚合, 结合指数时间衰减核          │
├────────────────────────────────────────────────────────────────────────┤
│ 5. 两阶段检索增强历史 (Retrieval-Augmented Histories: SIM / ETA / UBR)  │
│ • 机制: Hard/Soft 快速粗筛 Top-M (M≈50) ➔ 精细 Target-Attention (万级解耦)│
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 算力、显存与存储开销多维权衡矩阵

| 架构体系 | 算力复杂度 (FLOPs / Query) | 显存与通信带宽 (Memory Bound) | 在线时延 (P99) | 最大承载序列长度 $L$ | 工业界核心应用位置 |
|---|---|---|---|---|---|
| **截断 Transformer<br>(SASRec/BST)** | $\mathcal{O}(K \cdot N^2 \cdot d)$ | **高**（自注意力矩阵随 $N$ 平方激增） | 中等（$10 \sim 20\text{ms}$） | $N \le 100$ | 粗排 / 短期即时兴趣捕获 |
| **压缩记忆网络<br>(MIMN)** | $\mathcal{O}(K \cdot C \cdot d)$ | **极低**（仅存槽位矩阵 $C \ll L$） | **极快**（$< 3\text{ms}$） | $L \ge 10,000$ | 超高 QPS 召回 / 粗排 |
| **全量 Target-Attention<br>(DIN)** | $\mathcal{O}(K \cdot L \cdot d)$ | **极高**（每次请求拉取 $L$ 维全量向量） | **极慢**（超标，$> 50\text{ms}$） | $L \le 200$ | 中短序列精排 |
| **两阶段检索增强<br>(SIM Hard Search)** | $\mathcal{O}(K \cdot M \cdot d)$<br>$(M \ll L)$ | **极低**（仅按需传输 Top-50 命中特征） | **极快**（$5 \sim 8\text{ms}$） | **$L \ge 50,000+$** | **工业级精排第一绝对主力** |
| **两阶段检索增强<br>(SIM Soft / ETA)** | $\mathcal{O}(K \cdot M \cdot d + \text{LSH})$ | **中等**（需维护用户向量索引） | 优良（$8 \sim 12\text{ms}$） | **$L \ge 10,000+$** | 跨类目跨域长序列精排 |

---

### 2. 陈旧事件噪声与生命周期治理

1. **偶然误触与冲动点击**：通过停留时长阈值（Dwell Time > 10s）与购买加购信号过滤；
2. **生命周期跃迁与概念漂移**：引入显式时间差编码（Time-Delta Embedding: $\Delta t = t_{\text{now}} - t_{\text{event}}$）与指数衰减核（$e^{-\lambda \Delta t}$）；
3. **耐用品复购饱和**：配置品类后置抑制 Mask（Post-Purchase Suppression）。

---

## 模块四：电商生成式重排全链路（Generative Reranking Pipeline）

```text
电商生成式重排全链路流程:
【精排输出 Top 50 候选商品池】
       │
       ▼
【1. 候选 Token 化与序列化 (Candidate Serialization)】
• 紧凑 Slot Tokens <C_01> ~ <C_50> + 用户画像 Prefix Embeddings
       │
       ▼
【2. 约束束搜索生成器 (Prefix-Conditioned Beam Search)】
• 6 层轻量 Decoder-Only Transformer (隐层 256, KV Cache 复用)
• 硬约束 Masked Softmax (杜绝重复商品与非法 ID)
       │
       ▼
【3. 最终 6~10 个商品排列 (Final Slate: π = [π₁, π₂, ..., πₖ])】
```

### 1. 为什么需要重排？（Pointwise 的三大固有缺陷）
1. **商品间同质化蚕食（Cannibalization）**：同屏 5 双相似跑鞋导致视觉疲劳与选择困难；
2. **缺乏全屏搭配互补性（Complementarity）**：无法构造“手机 + 手机壳 + 无线耳机”跨类目搭配；
3. **价格锚点与对比心理失衡（Price Anchoring）**。

### 2. 候选序列化与前缀约束解码
- 采用局部临时占位符 `<C_01>` 到 `<C_50>`，每步解码仅生成 1 个 Token；
- **硬约束 Masked Softmax**：在每步解码时，将已选商品及非法 ID 的 Logits 强行设为 $-\infty$，彻底杜绝重复生成与幻觉；
- **Plackett-Luce 与全屏奖励优化**：
  $$\mathcal{L}_{\text{Plackett-Luce}} = -\sum_{k=1}^K \log \left( \frac{\exp(s_{\pi_k})}{\sum_{j=k}^K \exp(s_{\pi_j})} \right)$$
  $$R(\pi) = \sum_{k=1}^K \gamma^{k-1} (\text{Click}_k \cdot \text{Margin}_k + \text{GMV}_k) - \lambda \cdot \text{Redundancy}(\pi)$$

### 3. P99 ≤ 20ms 时延治理与评估全矩阵
- **时延保障**：4~6 层小模型、**Prefix KV Cache 跨步复用**、**18ms 超时熔断无缝降级**（回退到精排基线原序）；
- **离线指标**：Slate-NDCG@K、屏内多样性（Intra-List Diversity, ILD）、类目覆盖度；
- **线上核心指标**：用户人均 GMV、整屏点击率（Slate CTR）、订单转化率、P99 时延与降级率。

---

## 模块五：多维度指标金字塔体系与训练看板设计

```text
推荐系统多层级指标评估金字塔：
                     ▲
                    / \     【顶层：商业与北极星业务指标 Business Metrics】
                   /   \    • DAU / MAU, 人均时长, 购买 GMV, D7/D30 留存, 创作者生态多样性
                  /─────                 /       \   【中层：分片与公平性指标 Slice & Guardrail Metrics】
                /         \  • 新老用户分群, 冷启动 Item, 细分类目切片, P99 耗时
               /───────────              /             \ 【底层：算法与排序离线指标 Ranking & Offline Metrics】
             /               \• GAUC (User/Request 分组), Global AUC, LogLoss, PCOC, ECE, NDCG@K
            └─────────────────┘
```

### 1. ROC-AUC 与 GAUC（Grouped AUC）数学定义
- **全局 ROC-AUC**：$\text{AUC} = \frac{1}{|\mathcal{D}^+| \cdot |\mathcal{D}^-|} \sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \left( \mathbb{I}(s_i > s_j) + \frac{1}{2} \mathbb{I}(s_i = s_j) \right)$（易受跨用户活跃度偏倚混淆）；
- **Grouped AUC (GAUC)**：
  $$\text{GAUC} = \frac{\sum_{g \in \mathcal{G}, \, n_g^+ > 0, \, n_g^- > 0} w_g \cdot \text{AUC}_g}{\sum_{g \in \mathcal{G}, \, n_g^+ > 0, \, n_g^- > 0} w_g}, \quad w_g = n_g \text{ (曝光数)}$$
  - **User-Grouped GAUC**：衡量跨会话个性化偏好，受早晚/工作日意图漂移影响；
  - **Request-Grouped GAUC**：衡量单次刷新同屏 Slate 内的相对优劣，**精排模型的第一黄金离线指标**。

### 2. 负采样下的指标不变性与重要性加权
- **AUC / GAUC**：在均匀随机负采样下具有单调保序不变性，是真实全量 AUC 的渐近无偏估计；
- **LogLoss**：依赖先验分布，必须引入负样本 $\frac{1}{w}$ 重要性加权才能得到无偏估计。

### 3. 校准度度量（PCOC / ECE / Brier Score）
- **PCOC（Predictive-over-Observed Ratio）**：$\text{PCOC} = \frac{\sum \hat{p}_i}{\sum y_i}$（$=1.0$ 完美校准）；
- **ECE（Expected Calibration Error）**：$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) 
ight|$。

### 4. 工业级训练看板设计（Dashboard Instrumentation）

```text
工业级推荐排序训练仪表盘 4 栏标准布局 (Dashboard Instrumentation):
┌────────────────────────────────────────────────────────────────────────┐
│ 面板 1：优化动力学与损失收敛 (Optimization & Loss Dynamics)             │
│ • [Train/Val Weighted LogLoss] • [Gradient L2-Norm] • [Learning Rate] │
├────────────────────────────────────────────────────────────────────────┤
│ 面板 2：排序分辨力指标 (Ranking & Discrimination Metrics) ★核心排序能力 │
│ • [Request-GAUC (一刷内排序)]  • [User-GAUC (用户级排序)]              │
│ • [Global AUC (全局混淆基线)]  • [NDCG@5 / MRR (首屏位置感知)]          │
├────────────────────────────────────────────────────────────────────────┤
│ 面板 3：概率校准与可靠性 (Calibration & Probabilistic Reliability)      │
│ • [PCOC Ratio (全盘高低估比)]  • [ECE (分桶校准误差)]                   │
│ • [Reliability Plot (10-Bin 动态可靠性散点图)] • [Raw vs Recovered CTR]│
├────────────────────────────────────────────────────────────────────────┤
│ 面板 4：重要切片与商业代理 (Cohort Slices & Guardrail Proxies)          │
│ • [Cold-start User GAUC (冷启动)] • [Long-tail Item GAUC (长尾内容)]    │
│ • [High-Activity vs Low-Activity GAUC] • [P99 Inference Latency Proxy] │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 模块六：工业级在线实验与 A/B 测试全生命周期规范

```text
A/B 测试因果验证全生命周期:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. 实验设计: 假设检验 (H0, H1), 样本量估算 (Power ≥ 80%), 分流正交层     │
├────────────────────────────────────────────────────────────────────────┤
│ 2. 实验执行: SRM 卡方拟合优度检验, CUPED 方差缩减 (Var·(1-ρ²)), 灰度放量 │
├────────────────────────────────────────────────────────────────────────┤
│ 3. 决策发布: 跑满 14 天完整周周期, 严禁偷看 (No Peeking), 护栏指标无退化 │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 为什么“离线涨点，线上平甚至跌点”？
1. **反馈闭环与自选择偏差（Selection Bias）**：离线日志由旧策略筛选，新模型推荐的高潜新内容缺乏历史正反馈记录；
2. **特征时序穿越与服务延迟（Training-Serving Skew）**：离线特征泄漏，线上毫秒级实时特征延迟；
3. **系统时延与 P99 降级损耗**：模型变重导致 P99 延迟增加 50ms 触发截断降级；
4. **多目标冲突**：单目标 CTR 模型引发标题党诱导点击，损害转化率与长期留存。

### 2. CUPED 方差缩减数学原理
$$\hat{Y}_{\text{CUPED}} = Y - \theta(X - \mathbb{E}[X]), \quad \text{其中 } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
$$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - 
ho^2)$$
若实验前后的相关系数 $
ho = 0.8$，则方差直接骤降 $64\%$，在不增加流量的前提下将所需样本量降低近 3 倍！

---

## 模块七：经典实战案例库与小样本分层因果推断

### 案例一：注册漏斗 $2 \times 2$ 全因子实验设计（Sign-Up Funnel）

```text
2x2 全因子实验矩阵 (Full Factorial Design):
                       按钮颜色 (Color)
                 红色 (Red)         蓝色 (Blue)
              ┌─────────────────┬─────────────────┐
  顶部 (Top)  │   Control (T0)  │  Treatment (T1) │
              │   红色 + 顶部   │   蓝色 + 顶部   │
按钮位置      ├─────────────────┼─────────────────┤
(Position)    │  Treatment (T2) │  Treatment (T3) │
  底部 (Bottom)│  红色 + 底部   │   蓝色 + 底部   │
              └─────────────────┴─────────────────┘
```

- **交互效应估计量**：$\hat{eta}_3 = (T_3 - T_1) - (T_2 - T_0) = T_3 - T_2 - T_1 + T_0$；
- **防稀释触发曝光**：底部按钮通过 Viewport Intersection Observer 仅在滚动进入可视视口时记录曝光。

---

### 案例二：电商商详页 (PDP) 改版（CTR 涨，CVR 平）转化归因与小样本推断

```text
电商转化漏斗分解与流量稀释效应:
【全盘列表曝光 Impression】 (100,000)
       │
       ▼  CTR (Control: 5.0% ➔ Treatment: 5.6%, 相对 +12%)
【进入商详页 PDP Pageviews】 (Control: 5,000 ➔ Treatment: 5,600)
       │
       ▼  CVR (Control: 10.0% ➔ Treatment: 10.0%, 相对 +0%)
【最终下单成交 Purchases】 (Control: 500 ➔ Treatment: 560, 净增 +60 单! 全盘净增长 +12%!)
```

1. **全盘无条件成交净增（CTCVR 增长）**：
   $$\text{CTCVR} = \frac{\text{总购买订单数}}{\text{总展示曝光数}} = \text{CTR} \times \text{CVR} \implies \text{净增长 } +12\%$$
2. **流量稀释效应（Traffic Dilution）**：吸纳了原本不会点击的低意向边缘访客。在分母被稀释的情况下仍维持 CVR 平稳，证明商详页承接转化能力强劲；
3. **极小样本分层三大科学推断方案（$N \approx 100$, Power $< 20\%$）**：
   - **经验贝叶斯部分池化（Empirical Bayes Partial Pooling / Shrinkage）**：
     $$\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \cdot \mu_{\text{grand}} + (1 - B) \cdot ar{Y}_{\text{small}}, \quad \text{其中 } B = \frac{\sigma_{\text{small}}^2}{\sigma_{\text{small}}^2 + \tau^2}$$
   - **CUPED 引入前置 30 天历史消费协变量**：削减方差至 $28\%$，等效放大有效样本量 3.5 倍；
   - **非参数精确置换检验（Exact Permutation Test）** 与后验超越概率 $P(\theta_T > \theta_C \mid \text{Data}) > 0.90$。

---

## 模块八：工业级核心技术高频自测单选题（Interactive Self-Test Quizzes）

<details class="exercise">
<summary><span class="q-label">Q1 · 精排负采样与概率还原</span> <span class="q-text">某信息流推荐精排模型真实点击率 $p = 1\%$。为了加速训练，对未点击负样本以保留率 $w = 10\%$（$w=0.1$）进行均匀下采样，正样本全部保留。在模型输出预测值 $\hat{p} = 0.0917$（$9.17\%$）后，线上竞价与融分系统应如何精确还原真实自然概率 $p$？</span></summary>

- [ ] **A.** 直接使用 $\hat{p} \times w = 0.0917 \times 0.1 = 0.00917$ 进行粗略估计。
- [ ] **B.** 使用线性放缩公式 $p = \frac{\hat{p}}{w} = 0.917$。
- [x] **C.** 使用解析还原公式 $p = \frac{\hat{p}}{\hat{p} + \frac{1 - \hat{p}}{w}}$，计算得出精准自然概率 $p = 0.01$（$1.0\%$）。
- [ ] **D.** 下采样仅改变 LogLoss 绝对值，不影响输出概率的绝对标定，无需做任何概率还原。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **先验扭曲**：在负采样率 $w$ 下，模型学到的条件概率被拉高为 $\hat{p} = \frac{p}{p + w(1-p)}$；
>   2. **精准逆变换**：解反函数可得 $p = \frac{\hat{p}}{\hat{p} + \frac{1 - \hat{p}}{w}}$。代入 $\hat{p} = 0.0917$ 与 $w = 0.1$：
>      $$p = \frac{0.0917}{0.0917 + \frac{1 - 0.0917}{0.1}} = \frac{0.0917}{0.0917 + 9.083} = \frac{0.0917}{9.1747} = 0.01 \quad (1.0\%)$$
>   3. **业务后果**：如果不做还原直接传入 $eCPM = pCTR \times \text{Bid}$ 竞价排序，出价与预估 CTR 将虚高近 10 倍，瞬间击穿广告主预算。
</details>

<details class="exercise">
<summary><span class="q-label">Q2 · 多目标学习与负迁移治理</span> <span class="q-text">在电商多目标精排模型中（同时预测点击 CTR 与购买 CVR），频繁出现“点击率显著上涨但下单成交量严重下滑”的跷跷板效应（Negative Transfer）。下列哪种架构与优化方案在理论与工程上<strong>最能彻底阻断不同任务私有特征间的梯度撕裂</strong>？</span></summary>

- [ ] **A.** Shared-Bottom 架构：让底层所有参数由两个任务均摊反向传播梯度。
- [ ] **B.** 静态加权 Loss 调参：通过人工微调 Loss 权重（如 $\mathcal{L} = 0.8\mathcal{L}_{\text{CTR}} + 0.2\mathcal{L}_{\text{CVR}}$）。
- [x] **C.** PLE (Progressive Layered Extraction) 架构结合 PCGrad 梯度正交投影：物理隔离任务私有专家与共享专家，并在梯度负冲突（$\cos(\mathbf{g}_i, \mathbf{g}_j) < 0$）时进行法平面正交投影。
- [ ] **D.** 采用单任务训练并在离线阶段直接相乘（$p\text{CTCVR} = p\text{CTR} \times p\text{CVR}$）。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **Shared-Bottom 的缺陷**：底层硬共享导致点击梯度与转化梯度相互抵消（$\cos(\mathbf{g}_A, \mathbf{g}_B) < 0$），高频 CTR 完全主导参数更新；
>   2. **PLE 的物理隔离优势**：PLE 显式将 Experts 划分为 Task-Specific 独占专家与 Shared 共享专家，各任务私有表征不再受到其他任务梯度的直接污染；
>   3. **PCGrad 动态保障**：当不同任务在共享层出现破坏性负方向分量时，将梯度正交投影到冲突梯度的法平面，彻底消除梯度对抗。
</details>

<details class="exercise">
<summary><span class="q-label">Q3 · 离线评估指标 GAUC 解析</span> <span class="q-text">在评估工业级推荐精排模型时，为什么算法团队通常以 <strong>Request-Grouped GAUC（按单次刷新请求分组的 GAUC）</strong> 作为第一核心排序北极星，而非全局 Global AUC 或 User-Grouped GAUC？</span></summary>

- [ ] **A.** Request-GAUC 的数值通常比 Global AUC 更高，便于向上汇报展示涨点效果。
- [ ] **B.** Global AUC 会受到不同用户样本量不平衡的影响，而 User-GAUC 完全没有这一问题。
- [x] **C.** Global AUC 混合了跨用户的全局打分偏倚（辛普森悖论）；User-GAUC 混杂了用户跨时段的意图漂移；而 Request-GAUC 严格评估单次刷新同屏候选物（Within-Slate）内部的精准相对排序。
- [ ] **D.** Request-GAUC 能够直接替代 PCOC 和 ECE 来评估模型的概率绝对校准能力。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **Global AUC 的致命缺陷**：若模型学会“给高活用户打高分、低活用户打低分”，Global AUC 很高但单次列表内部排序完全随机；
>   2. **User-GAUC vs. Request-GAUC**：User-GAUC 将用户早上的未点击项与晚上的点击项混合计算，引入了昼夜意图漂移噪声；而 Request-GAUC 严格限制在单次刷新展现的 6~10 个同屏物料中，**百分之百对齐了用户决策瞬间面临的真实选择现场**；
>   3. **D 选项错误**：GAUC 具有单调保序不变性，只能衡量排序分辨力（Discrimination），无法评估绝对概率校准度（Calibration）。
</details>

<details class="exercise">
<summary><span class="q-label">Q4 · 在线实验与小样本因果推断</span> <span class="q-text">电商商详页（PDP）UI 改版 A/B 实验显示：列表到商详页点击率 CTR 提升 $+12\%$（$p < 0.01$），但商详页内转化率 CVR 保持平稳（$\Delta \text{CVR} \approx 0\%$，$p = 0.65$）。在黑金大客户分层（$N \approx 100$）样本量极小时，下列决策与推断方法中<strong>最科学严谨的是</strong>：</span></summary>

- [ ] **A.** 因为大客户层 $t$ 检验 $p > 0.05$，说明新 UI 对高价值客户无效，应立即下线改版。
- [ ] **B.** 条件 CVR 没有提升说明改版失败，单次曝光带来的实际商业回报没有增长。
- [x] **C.** 全盘总成交量 $\text{CTCVR} = \text{CTR} \times \text{CVR}$ 实际上净增长了 $+12\%$；CVR 持平源于新增边缘访客的“流量稀释效应”；大客户小样本层应采用<strong>经验贝叶斯部分池化（Empirical Bayes Shrinkage）</strong>与前置消费协变量 CUPED 进行稳健推断。
- [ ] **D.** 延长实验至 6 个月以上，直到大客户层样本量达到数十万后自然拒绝原假设。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **漏斗全局因果净增**：$\text{CTCVR} = \text{CTR} \times \text{CVR} = 1.12 \times 1.0 = 1.12$（全盘总订单量净增 $+12\%$）；
>   2. **流量稀释因果机制**：改版吸引了额外原本不点击的低意图用户，在分母稀释下仍维持原有高转化率，证明落地页承接能力强劲；
>   3. **小样本推断**：$p > 0.05$ 仅代表统计功效不足（$\text{Power} < 20\%$），绝非无效证明。采用经验贝叶斯收缩模型 $\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \mu_{\text{grand}} + (1-B) \bar{Y}_{\text{small}}$ 跨层借力，是工业界防误判的金标准。
</details>
