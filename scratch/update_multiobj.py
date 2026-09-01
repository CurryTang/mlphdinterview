import os

zh_content = """# 多目标排序全景：负迁移机理、模型架构演进与跷跷板效应治理

在工业级推荐系统（Industrial Recommender Systems）与计算广告中，业务往往需要同时优化多个相互冲突或相关性各异的业务指标。例如：短视频推荐需同时兼顾**点击率（pCTR）、长读完播率（pLongView）、点赞收藏互动率（pInteract）与负反馈率（pDislike）**；电商推荐需同时优化**点击率（pCTR）、加购率（pCart）、购买转化率（pCVR）与客单成交额（pGMV）**。

当多个任务通过底层共享表征（Shared Representations）联合训练时，经常会观察到**“任务 A 效果提升，但任务 B 发生严重回退”的跷跷板效应（Seesaw Effect / Negative Transfer）**。

本篇系统拆解多目标排序体系三大核心支柱：
1. **多任务表征冲突与跷跷板效应底层根因（Gradient Conflicts, Magnitude Dominance & Selection Bias）**
2. **六大主流多目标模型与优化方案横向对比（Shared-Bottom, MMoE, PLE, Uncertainty Weighting, PCGrad, Constrained Fusion）**
3. **跷跷板效应消融实验诊断矩阵与线上实时护栏熔断机制（Ablations & Online Guardrails）**

---

## 模块一：多目标任务为何会在共享表征中相互打架？（跷跷板底层根因）

```text
多目标表征冲突三大根因示意:
┌────────────────────────────────────────────────────────────────────────┐
│ 根因 1: 梯度方向负冲突 (Gradient Conflict: cos(g_A, g_B) < 0)           │
│ • Task A (点击) 梯度推动参数向"吸引人标题"方向更新                     │
│ • Task B (完播) 梯度推动参数向"深度高质量内容"方向更新 ➔ 共享参数被撕裂 │
├────────────────────────────────────────────────────────────────────────┤
│ 根因 2: 梯度量级与样本频次压制 (Magnitude & Frequency Imbalance)       │
│ • 高频正样本 Task (CTR 约 5%) 梯度范数 ≫ 低频稀疏 Task (CVR 约 0.1%)   │
│ • 共享 Embedding 与底层 MLP 权重被高频任务完全主导，低频任务欠拟合      │
├────────────────────────────────────────────────────────────────────────┤
│ 根因 3: 样本空间不一致与选择偏差 (Sample Space Mismatch)               │
│ • CTR 在全量曝光空间训练: D_imp                                         │
│ • CVR 传统上仅在点击后样本空间训练: D_click (非随机丢弃 ➔ 样本选择偏差)│
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 梯度方向负冲突（Negative Gradient Cosine Similarity）
当两个任务的损失函数 $\mathcal{L}_A$ 与 $\mathcal{L}_B$ 对共享参数 $W_{\text{shared}}$ 计算梯度时：

$$\mathbf{g}_A = \nabla_{W_{\text{shared}}} \mathcal{L}_A, \quad \mathbf{g}_B = \nabla_{W_{\text{shared}}} \mathcal{L}_B$$

若 $\cos(\mathbf{g}_A, \mathbf{g}_B) < 0$（即 $\mathbf{g}_A \cdot \mathbf{g}_B < 0$），两个任务的梯度方向形成钝角甚至反向对冲。此时参数更新 $\Delta W \propto -(\mathbf{g}_A + \mathbf{g}_B)$ 必然会导致至少一个任务的损失上升，形成**典型的跷跷板恶性循环**。

### 2. 梯度量级失衡与高频任务压制（Frequency Dominance）
高频行为（如展示点击，正样本率 5%）的样本量和更新步数通常是低频稀疏行为（如高单价购买或付费转化，正样本率 0.1%）的数十倍乃至上百倍。如果不加约束，共享层参数会被点击任务的梯度彻底“洗脑”，低频重要任务的语义表征空间被严重挤压。

---

## 模块二：六大多目标模型架构与优化策略深度横向对比

```text
多目标解决方案三维谱系:
┌───────────────────────────────────┬───────────────────────────────────┐
│ 1. 架构级特征路由 (Architecture)  │ Shared-Bottom ➔ MMoE ➔ PLE (渐进分层)│
├───────────────────────────────────┼───────────────────────────────────┤
│ 2. 优化级梯度与损失平衡 (Loss/Grad)│ 动态不确定性加权 ➔ PCGrad 梯度投影    │
├───────────────────────────────────┼───────────────────────────────────┤
│ 3. 决策级分数融合 (Score Fusion)  │ 静态加权 ➔ 帕累托约束优化 / PID 控制  │
└───────────────────────────────────┴───────────────────────────────────┘
```

### 1. 架构级方案对比（Shared-Bottom vs. MMoE vs. PLE）

| 架构方案 | 网络拓扑与路由机制 | 负迁移防御能力 | 优势（Pros） | 缺陷与局限（Cons） |
|---|---|---|---|---|
| **Shared-Bottom** | 所有任务完全共享单一底层 MLP，顶部分叉 Task Towers。 | **极差**（无防御能力） | 结构最简单，计算开销极低。 | 任务间梯度强行对冲，跷跷板效应最剧烈。 |
| **MMoE<br>(Multi-gate MoE)** | 共享一组 Experts，每个任务拥有独立的 Softmax 门控网络 $g_t(x) = \text{Softmax}(W_t x)$。 | **一般**（部分缓解） | 实现了动态软路由加权，自适应分配专家。 | **所有 Expert 依然是全局共享的**，弱相关任务间仍会争夺 Expert 容量。 |
| **PLE<br>(Progressive Extraction)** | 显式解耦为 **任务独占专家（Task-Specific Experts）** 与 **全局共享专家（Shared Experts）**，分层渐进抽取。 | **卓越（SOTA）**（彻底阻断负迁移） | **任务私有特征与共享特征严格物理隔离**；高层逐步融合，彻底消除负迁移。 | 参数量与前向计算开销略微增加（约 15%~25%）。 |

---

### 2. 优化与损失级方案（Uncertainty Weighting & PCGrad）

#### (1) 同方差不确定性加权（Homoscedastic Uncertainty Weighting, Kendall et al.）
传统固定超参数加权 $\mathcal{L} = \sum w_k \mathcal{L}_k$ 极其依赖人工网格调参。Uncertainty Weighting 将每个任务的不确定性 $\sigma_k^2$ 建模为可学习参数：

$$\mathcal{L}_{\text{total}} = \sum_{k=1}^K \left( \frac{1}{2\sigma_k^2} \mathcal{L}_k + \ln \sigma_k \right)$$

- **自适应平衡机制**：当任务 $k$ 噪声大、方差 $\sigma_k^2$ 高时，自动缩减其损失权重 $\frac{1}{2\sigma_k^2}$，同时对数正则项 $\ln \sigma_k$ 防止权重退化为 0。

#### (2) 冲突梯度正交投影（PCGrad: Projecting Conflicting Gradients）
当检测到任务 $i$ 与任务 $j$ 的梯度发生冲突时（$\mathbf{g}_i \cdot \mathbf{g}_j < 0$），将 $\mathbf{g}_i$ **正交投影到 $\mathbf{g}_j$ 的法平面上**：

$$\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j$$

- **效果**：消除了对任务 $j$ 有害的分量，同时最大限度保留了任务 $i$ 的原更新方向，从梯度动力学层面彻底切断负对冲。

---

### 3. 决策级分数融合（Constrained Fusion & Pareto Optimization）

即使底层多任务模型预估出了极准的 $\hat{p}_{\text{CTR}}$ 与 $\hat{p}_{\text{CVR}}$，线上如何将它们融合成单一排序分 $S$ 依然是业务核心：
- **传统朴素加权**：$S = \hat{p}_{\text{CTR}} \times \hat{p}_{\text{CVR}}^\alpha$（超参 $\alpha$ 静态固化，遇大促或流量波动易失效）；
- **带约束的拉格朗日优化与 PID 控制**：
  $$\max \mathbb{E}[\text{GMV}] \quad \text{s.t.} \quad \text{CTR} \ge \text{CTR}_0, \quad \text{Dislike Rate} \le \tau$$
  在线服务维护实时 PID 控制器，根据最近 5 分钟的实际 CTR 动态调整拉格朗日乘子 $\lambda(t)$，保证在满足生态护栏的前提下实现主商业目标最大化。

---

## 模块三：跷跷板效应的消融诊断矩阵与线上实时护栏体系

```text
多目标跷跷板效应离线诊断与线上防御闭环:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. 离线消融诊断 (Offline Diagnostic Ablations)                         │
│ • 单任务上界基准 (Single-Task Upper Bounds): 训练 K 个独立单模型       │
│ • 任务梯度余弦相似度矩阵: 全局追踪 cos(g_i, g_j) 负相关比例            │
│ • 任务 GAUC 增益矩阵: ΔGAUC_k = GAUC_MTL(k) - GAUC_Single(k)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. 线上实时护栏与自适应熔断 (Online Guardrails & Adaptive Throttling)  │
│ • 实时 PID 动态控权: 每分钟基于线上大盘指标动态微调融合权重 λ          │
│ • 自动熔断降级 (Circuit Breaker): 关键护栏指标下跌超阈值自动回退       │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 离线消融诊断标准四步法（The 4-Step Offline Ablation Protocol）

1. **单任务性能天花板测定（Single-Task Benchmarks）**：
   为每个任务独立训练一个专属的单任务模型（完全无参数共享），记录其 $\text{GAUC}_{\text{Single}, k}$ 作为该任务的理论上界；
2. **多任务相对增益测算（Task Delta Matrix）**：
   计算 $\Delta \text{GAUC}_k = \text{GAUC}_{\text{MTL}, k} - \text{GAUC}_{\text{Single}, k}$。若出现“任务 A $\Delta \text{GAUC} = +0.008$，但任务 B $\Delta \text{GAUC} = -0.006$”，直接证实严重负迁移；
3. **梯度冲突热力图追踪（Gradient Cosine Heatmap）**：
   在训练过程中每 1000 步记录一次各任务梯度余弦相似度 $\cos(\mathbf{g}_i, \mathbf{g}_j)$。若负相关步数占比超过 30%，必须引入 PLE 或 PCGrad；
4. **子群切片差异性检验（Slice Heterogeneity）**：
   检查高活跃用户与低活跃用户、冷启动 Item 上的多目标表现，排查局部跷跷板现象。

---

### 2. 线上实时护栏与自适应控权机制（Online Guardrails）

1. **动态 PID 控权闭环**：
   设目标为最大化 GMV 且确保大盘 CTR 相对跌幅不超过 $-1\%$。线上收集实时滑动窗口（如 5 分钟）的 $\text{CTR}_{\text{online}}$：
   $$e(t) = \text{CTR}_{\text{online}}(t) - \text{CTR}_{\text{target}}$$
   $$\lambda(t) = \lambda(t-1) + K_p e(t) + K_i \int e(\tau) d\tau + K_d \frac{de(t)}{dt}$$
   动态调整融合公式中的 CTR 权重 $\lambda(t)$；
2. **自动熔断回路（Circuit Breakers）**：
   若线上核心护栏指标（如负反馈率上升 $> 5\%$ 或核心类目 GMV 下跌 $> 2\%$）持续 3 个监控周期，系统自动将分流灰度流量无缝降级回退到基线策略，并向值班团队发送 P0 告警。
"""

with open("notes/BusinessAlgorithm/BusinessAlgorithm02B Multi-Objective Ranking.md", "w", encoding="utf-8") as f:
    f.write(zh_content)
print("Successfully updated BusinessAlgorithm02B Chinese note")
