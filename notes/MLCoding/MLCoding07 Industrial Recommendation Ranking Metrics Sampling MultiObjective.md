# ML Coding 07 · 工业级推荐排序全景：模型体系、负采样概率还原、GAUC 指标与多目标跷跷板治理

在工业级推荐系统（Industrial Recommender Systems）与计算广告工程中，构建高吞吐、高精度、多目标协同的排序（Ranking）体系是核心技术壁垒。面对面试官或架构评审时，**如何系统性阐述基座模型选型（Base Model Family）、训练样本规模与负采样方案（Sampling Scheme & Probability Recovery）、金字塔指标体系（ROC-AUC, GAUC by User/Request, PCOC Calibration, Slices），以及多目标负迁移与跷跷板效应治理（PLE, PCGrad, Constrained Pareto Fusion）**，是资深算法专家的关键分水岭。

本篇系统梳理工业级推荐排序全链路四大核心支柱：
1. **排序基座模型家族架构（Base Model Family: 特征交叉、序列建模与多任务学习）**
2. **训练样本规模构建、负采样范式与概率校准数学推导（Sampling Scheme & Probability Recovery）**
3. **多维度指标金字塔体系（ROC-AUC、GAUC 用户/请求分组、无偏估计、校准曲线 PCOC 与切片指标）**
4. **多目标负迁移机理与跷跷板效应全方位治理（Shared-Bottom, MMoE, PLE, PCGrad 与 PID 约束融合）**
5. **训练看板仪表盘设计（Dashboard Instrumentation）与高频面试标准答题模板**

---

## 模块一：推荐排序基座模型家族（Base Model Family）

工业级推荐排序模型通常处理**百亿级高维稀疏特征（ID Features）**与**稠密数值特征（Dense Features）**，其演进脉络聚焦于三大核心技术轴线：

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

### 1. 经典工业级基座模型三大支柱

1. **特征交叉基座（Feature Interaction Backbone）**：
   - **DCN-v2 (Deep & Cross Network v2)**：通过显式多项式交叉层（Cross Network）结合低秩矩阵分解（Low-Rank MoE Matrix Multiplication），在极低延迟下逼近任意高阶特征交叉；
   - **DLRM (Deep Learning Recommendation Model)**：Meta 开源的标准架构，将特征明确解耦为用于稀疏 ID 的 Embedding 表与用于数值特征的 Bottom MLP，通过显式 Dot-product 交互后喂入 Top MLP。
2. **行为序列建模基座（User Behavior Sequence Backbone）**：
   - **DIN (Deep Interest Network)**：利用候选 Item 与历史行为序列计算 **Target Attention**，克服了传统 Pooling 对用户多样化兴趣的静态压缩损失；
   - **SIM (Search-based Interest Model)**：将用户行为序列扩展到数万步（Long-Term Sequence），第一阶段使用类目/Tag 倒排索引做 **Sub-sequence 快速检索（Hard Search）**，第二阶段在小集合上执行精细 Attention（Soft Search）。
3. **多目标排序基座（Multi-Task Learning Backbone）**：
   - **MMoE (Multi-gate Mixture-of-Experts)**：为每个任务分配独立的 Softmax 门控网络，动态加权共享 Expert；
   - **PLE (Progressive Layered Extraction)**：将专家网络严格区分为 **任务独占专家（Task-Specific Experts）** 与 **全局共享专家（Shared Experts）**，彻底阻断了点击（CTR）与转化（CVR）等相关性较弱任务间的负迁移现象。

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
在信息流或电商推荐中，真实点击率通常极低（如 $p pprox 1\% \sim 5\%$），正负样本极度不平衡（$1:20 \sim 1:100$）。
1. **节省训练计算资源与存储**：下采样负样本可将训练集体积缩减 70%~90%，大幅提高梯度迭代吞吐量；
2. **防止梯度被海量易分负样本淹没**。

#### 下采样率对预估概率的系统性扭曲
假设我们将未点击负样本以采样率 $w \in (0, 1]$ 进行随机下采样（保留比例为 $w$），而正样本全部保留。
下采样后，模型学到的条件概率 $\hat{p} = P(Y=1 \mid X, 	ext{sampled})$ 会被**人为大幅拉高**：

$$\hat{p} = rac{P(Y=1 \mid X)}{P(Y=1 \mid X) + w \cdot P(Y=0 \mid X)} = rac{p}{p + w(1-p)}$$

#### 线上推理时的概率校准还原公式（Probability Recovery Formula）
在竞价广告（$eCPM = pCTR 	imes pCVR 	imes 	ext{Bid}$）或需要精准概率融合的业务中，必须在模型输出后通过解析反函数将 $\hat{p}$ **精确还原为真实自然世界概率 $p$**：

$$p = rac{\hat{p}}{\hat{p} + rac{1 - \hat{p}}{w}}$$

```text
概率还原验证示例：
若真实 CTR p = 0.01 (1%)，负样本下采样保留率 w = 0.1 (10%):
1. 训练集上模型观测到的 CTR: p_hat = 0.01 / (0.01 + 0.1 * 0.99) = 0.01 / 0.109 = 0.0917 (9.17%)
2. 线上推理时通过公式还原: p = 0.0917 / (0.0917 + (1 - 0.0917) / 0.1) = 0.01 (精准复原为 1%!)
```

---

## 模块三：ROC-AUC、GAUC（按用户/按请求分组）与多维指标深度剖析

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

#### (1) 全局 ROC-AUC（Global AUC）的 Wilcoxon-Mann-Whitney 统计学定义
ROC 曲线下面积在统计学上严格等价于：**从测试集随机抽取一个正样本 $i \in \mathcal{D}^+$ 和一个负样本 $j \in \mathcal{D}^-$，模型的预测打分 $s_i$ 大于 $s_j$ 的概率**：

$$	ext{AUC} = rac{1}{|\mathcal{D}^+| \cdot |\mathcal{D}^-|} \sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \left( \mathbb{I}(s_i > s_j) + rac{1}{2} \mathbb{I}(s_i = s_j) ight)$$

- **缺陷（Simpsons-like Bias）**：Global AUC 混合了跨用户的全部样本。如果模型仅仅学会了“给高活跃用户打整体高分、给低活跃用户打整体低分”，其 Global AUC 也能达到 0.85，但对**任何单一用户展示列表内部的相对优劣排序完全无效**！

#### (2) Grouped AUC (GAUC) 的数学定义
为了剥离跨用户之间的基线活跃度差异，工业界提出 **GAUC**：在每个独立分组 $g \in \mathcal{G}$ 内部单独计算 $	ext{AUC}_g$，然后按曝光量 $n_g$ 或正样本量 $n_g^+$ 进行加权平均：

$$	ext{GAUC} = rac{\sum_{g \in \mathcal{G}, \, n_g^+ > 0, \, n_g^- > 0} w_g \cdot 	ext{AUC}_g}{\sum_{g \in \mathcal{G}, \, n_g^+ > 0, \, n_g^- > 0} w_g}, \quad 	ext{其中 } w_g = n_g 	ext{ (曝光数) 或 } n_g^+ 	ext{ (点击数)}$$

---

### 2. 按用户分组（User-level）vs. 按请求/Session 分组（Request-level）统计量差异

```text
GAUC 分组层级统计量对比：
┌───────────────────────┬───────────────────────────────────┬───────────────────────────────────┐
│ 统计维度              │ 按用户分组 (User-Grouped GAUC)    │ 按请求/会话分组 (Request-GAUC)    │
├───────────────────────┼───────────────────────────────────┼───────────────────────────────────┤
│ 分组单元 (Unit g)     │ 唯一 user_id                      │ 唯一 request_id / session_id      │
│ 跨样本时空跨度        │ 跨越该用户在测试期内的多次访问与刷新│ 严格限制在单次刷新展现的一刷 Slate│
│ 衡量核心能力          │ 用户全局偏好识别度 (跨会话整体排序) │ 单次上下文下候选项区分度 (同屏竞争) │
│ 上下文与时变干扰      │ 受用户在早晚/工作日的意图漂移影响 │ 完全消除了时间、网络、设备等瞬时干扰│
│ 工业界最佳实践        │ 适合评估长周期个性化推荐与召回    │ **精排模型最严密纯粹的离线指标**   │
└───────────────────────┴───────────────────────────────────┴───────────────────────────────────┘
```

- **为何两者会发生背离？**
  - 一个用户在上午搜索“办公软件”（点击了几个专业工具），晚上浏览“搞笑短视频”（点击了几个娱乐内容）。
  - 如果计算 **User-GAUC**，上午的未点击视频与晚上的点击视频会被放在一起做成对比较，引入了跨 Session 意图漂移的噪音；
  - 而 **Request-GAUC** 只比较用户在“某一次刷新展示的 6 个候选物中”点了哪一个，**严格对齐了用户在决策瞬间面临的选择现场**。

---

### 3. 负采样下的无偏估计（Unbiased Estimation with Sampled Negatives）

- **排序指标（AUC / GAUC）的无偏性**：
  在对未点击负样本进行**均匀随机下采样（Uniform Negative Downsampling）**时，由于正样本对 $(i)$ 和负样本对 $(j)$ 的大小相对顺序在期望上不被改变：
  $$\mathbb{E}_{	ext{sampled}}[\mathbb{I}(s_i > s_j) \mid Y_i=1, Y_j=0] = \mathbb{P}(s_i > s_j \mid Y_i=1, Y_j=0)$$
  因此，**在下采样数据上直接计算的 AUC 和 GAUC 是真实全量数据 AUC/GAUC 的渐近无偏估计**，无需额外乘以加权系数！
- **似然与损失指标（LogLoss）的重要性加权无偏估计（Importance Weighting）**：
  相比之下，交叉熵损失（LogLoss）强烈依赖正负样本比例。若要在采样数据上得到全量分布的无偏损失估计，必须使用重要性权重（Importance Weight $rac{1}{w}$ 为负样本加权）：
  $$\mathcal{L}_{	ext{unbiased}} = -rac{1}{N^+} \sum_{i \in \mathcal{D}^+} \log \hat{p}_i - rac{1}{w \cdot N^-} \sum_{j \in \mathcal{D}^-} \log(1 - \hat{p}_j)$$

---

### 4. 排序（Ranking）vs. 校准（Calibration）的本质解耦

- **排序能力（Discrimination / Ranking）**：
  - **性质**：对预测分数的任何**严格单调递增变换保持不变**（若将所有分数全部平方或加上 100，AUC 和 GAUC 保持绝对一致）；
  - **局限**：一个模型的 AUC 即使高达 0.95，其输出的预估值可能全部挤在 $[0.0001, 0.0002]$ 之间。
- **校准能力（Reliability / Calibration）**：
  - **性质**：要求模型预估的置信度在数值上精确等于真实发生概率，即 $\mathbb{E}[Y \mid \hat{p}] = \hat{p}$；
  - **商业严重性**：在竞价广告（$eCPM = pCTR 	imes pCVR 	imes 	ext{Bid}$）与主站多目标融分中，如果高估 $50\%$，会导致广告主预算半小时内被快速耗尽、竞价出价严重虚高，直接破坏商业生态。
- **扩展校准度量**：
  - **PCOC（Predictive-over-Observed Calibration Ratio）**：$	ext{PCOC} = rac{\sum \hat{p}_i}{\sum y_i}$（$=1.0$ 完美校准）；
  - **ECE（Expected Calibration Error，预期校准误差）**：将样本按预测概率分为 $M$ 个等宽/等频分桶 $B_m$：
    $$	ext{ECE} = \sum_{m=1}^M rac{|B_m|}{N} \left| 	ext{acc}(B_m) - 	ext{conf}(B_m) ight|$$
  - **Brier Score（均方概率误差）**：$	ext{Brier} = rac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$，同时兼顾排序分辨力与绝对校准度。

---

## 模块四：多目标排序负迁移机理与跷跷板效应治理（Multi-Objective Ranking）

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

### 1. 六大多目标架构与优化方案对比

| 解决方案类别 | 核心方法 | 拓扑机制与数学定义 | 负迁移防御力 | 工业界核心洞察 |
|---|---|---|---|---|
| **架构路由** | **Shared-Bottom** | 共享单一底层 MLP，顶部分叉 Task Towers。 | **极差（0 分）** | 任务间梯度强行对冲，跷跷板效应最剧烈。 |
| **架构路由** | **MMoE** | 共享 Experts 池，每个任务拥有独立的 Softmax 门控：$g_t(x) = 	ext{Softmax}(W_t x)$。 | **中等** | 实现了动态软路由，但所有 Expert 仍是全局共享的，弱相关任务仍会争抢容量。 |
| **架构路由** | **PLE (Progressive Extraction)** | 显式解耦为 **Task-Specific 独占专家** 与 **Shared 共享专家**，分层渐进抽取。 | **卓越（SOTA）** | **任务私有特征与共享特征严格物理隔离**，彻底阻断不同任务间的负迁移。 |
| **损失平衡** | **Uncertainty Weighting** | 建模同方差不确定性 $\sigma_k^2$：<br>$$\mathcal{L} = \sum \left( rac{1}{2\sigma_k^2}\mathcal{L}_k + \ln \sigma_k ight)$$ | **优良** | 自动根据任务方差与噪声动态调整损失权重，避免人工粗暴调参。 |
| **梯度手术** | **PCGrad (Gradient Projection)** | 当梯度冲突 $\mathbf{g}_i \cdot \mathbf{g}_j < 0$ 时，将 $\mathbf{g}_i$ 正交投影到 $\mathbf{g}_j$ 的法平面：<br>$$\mathbf{g}_i \leftarrow \mathbf{g}_i - rac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2}\mathbf{g}_j$$ | **极强** | 从梯度动力学层面消除相互抵消的破坏性分量。 |
| **决策融合** | **Constrained Pareto Fusion** | 拉格朗日乘子约束优化与实时 PID 控权：<br>$$\max 	ext{GMV} 	ext{ s.t. } 	ext{CTR} \ge 	ext{CTR}_0$$ | **线上闭环** | 替代静态超参融合，自适应追踪业务护栏。 |

---

## 模块五：工业级训练看板仪表盘设计与防混淆指南（Dashboard Instrumentation）

```text
工业级推荐排序训练仪表盘 4 栏标准布局 (Dashboard Instrumentation):
┌────────────────────────────────────────────────────────────────────────┐
│ 面板 1：优化动力学与损失收敛 (Optimization & Loss Dynamics)             │
│ • [Train/Val Weighted LogLoss] • [Gradient L2-Norm] • [Learning Rate] │
│ • [Embedding Table Norm] • [FP16 Loss Scale / Overflow Status]         │
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

### 看板防混淆三大铁律（Anti-Confusion Guardrails）
1. **命名严格隔离**：严禁直接简写为 `AUC`！必须显式标明 `AUC/Global`、`GAUC/By_Request` 与 `GAUC/By_User`；
2. **校准与排序双红线**：设置双门禁报警机制——若 `GAUC` 提升 $+0.003$ 但 `PCOC` 偏离 $[0.98, 1.02]$ 区间，仪表盘自动亮红灯警告，禁止自动推流到灰度环境；
3. **切片差异可视化**：在面板 4 中默认展示 `GAUC_delta(Cold_Start) - GAUC_delta(Overall)`，防止高活大盘均值掩盖冷启动子群的严重衰退（辛普森悖论防御）。
