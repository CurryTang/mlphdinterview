import re

# ==============================================================================
# 1. Update BusinessAlgorithm02C (Feature Interaction) with DCN-v2 & DLRM Pseudocode
# ==============================================================================
with open("notes/BusinessAlgorithm/BusinessAlgorithm02C Feature Interaction.md", "r", encoding="utf-8") as f:
    ba02c = f.read()

dcn_dlrm_zh = r"""### 11.3 DCN 与 DCN-v2 (Deep & Cross Network)

DCN 的核心是显式 Cross Layer：

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l$$

在 **DCN-v2** 中，为了降低全连接矩阵 $\mathbf{W}_l \in \mathbb{R}^{d \times d}$ 的计算开销并捕捉低秩子空间，引入了**低秩矩阵分解（Low-Rank Matrix Decomposition: $\mathbf{W}_l = \mathbf{U}_l \mathbf{V}_l^T$）与混合门控路由（MoE-style Subspaces）**：

#### 伪代码实现：DCN-v2 Cross Network 前向
```python
import torch
import torch.nn as nn

class CrossNetworkV2(nn.Module):
    def __init__(self, in_features, num_layers=3, low_rank=32, num_experts=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        # 低秩分解: W_l = U_l @ V_l.T, U in R^(d x r), V in R^(d x r)
        self.u_list = nn.ParameterList([
            nn.Parameter(torch.randn(num_experts, in_features, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.v_list = nn.ParameterList([
            nn.Parameter(torch.randn(num_experts, in_features, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.gate_list = nn.ModuleList([
            nn.Linear(in_features, num_experts, bias=False) for _ in range(num_layers)
        ])
        self.bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features)) for _ in range(num_layers)
        ])

    def forward(self, x0):
        # x0: [batch_size, in_features]
        xl = x0
        for l in range(self.num_layers):
            # 1. 动态计算 Expert 门控权重: [batch_size, num_experts, 1]
            gate = torch.softmax(self.gate_list[l](xl), dim=-1).unsqueeze(-1)
            # 2. 低秩矩阵变换: (xl @ V) @ U.T
            # xl: [B, 1, d] -> xl @ V: [B, E, r] -> (xl @ V) @ U.T: [B, E, d]
            xl_expanded = xl.unsqueeze(1) # [B, 1, d]
            v = self.v_list[l] # [E, d, r]
            u = self.u_list[l] # [E, d, r]
            low_rank_out = torch.einsum('bmd,edr->ber', xl_expanded, v) # [B, E, r]
            expert_out = torch.einsum('ber,edr->bed', low_rank_out, u)   # [B, E, d]
            # 3. MoE 专家加权汇聚: [B, d]
            moe_out = (expert_out * gate).sum(dim=1)
            # 4. 显式特征交叉与残差相加
            xl = x0 * (moe_out + self.bias_list[l]) + xl
        return xl
```

---

### 11.4 DLRM (Deep Learning Recommendation Model)

Meta DLRM 架构将特征解耦为数值 Dense 特征与稀疏 Sparse 类别特征：
1. **Bottom MLP**：将连续稠密特征映射为 $d$ 维稠密向量；
2. **Embedding Tables**：为各 Sparse 类别 ID 查表得到 $d$ 维嵌入向量；
3. **Explicit Dot-Product Triu Interaction**：计算所有 $d$ 维向量之间的两两点积（Dot Product），提取上三角矩阵元素并拉平；
4. **Top MLP**：将点积交叉结果与原始 Bottom 特征拼接后送入 Top MLP 输出预估概率。

#### 伪代码实现：DLRM 特征交互与前向
```python
class DLRM(nn.Module):
    def __init__(self, embedding_sizes, dense_in_dim, embed_dim=64):
        super().__init__()
        # 1. Sparse 嵌入表
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim) for num_classes in embedding_sizes
        ])
        # 2. Bottom MLP 处理 Dense 数值特征
        self.bottom_mlp = nn.Sequential(
            nn.Linear(dense_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        # 3. 计算两两点积组合数: (num_sparse + 1) * num_sparse / 2
        num_fields = len(embedding_sizes) + 1
        num_interactions = (num_fields * (num_fields - 1)) // 2
        # 4. Top MLP 最终分类
        self.top_mlp = nn.Sequential(
            nn.Linear(num_interactions + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, dense_x, sparse_x):
        # dense_x: [B, dense_in_dim], sparse_x: [B, num_sparse]
        v_dense = self.bottom_mlp(dense_x).unsqueeze(1) # [B, 1, embed_dim]
        v_sparse = [emb(sparse_x[:, i]).unsqueeze(1) for i, emb in enumerate(self.embeddings)]
        # 全部字段向量拼接: [B, num_fields, embed_dim]
        all_embeddings = torch.cat([v_dense] + v_sparse, dim=1)
        # 批量矩阵乘法计算两两内积: [B, num_fields, num_fields]
        dot_interactions = torch.bmm(all_embeddings, all_embeddings.transpose(1, 2))
        # 提取上三角非对角线元素 (去冗余): [B, num_interactions]
        triu_indices = torch.triu_indices(all_embeddings.size(1), all_embeddings.size(1), offset=1)
        flat_interactions = dot_interactions[:, triu_indices[0], triu_indices[1]]
        # 拼接原始 Bottom 特征后喂入 Top MLP
        top_input = torch.cat([flat_interactions, v_dense.squeeze(1)], dim=-1)
        return self.top_mlp(top_input)
```
"""

if "### 11.3 DCN 与 DCN-v2" not in ba02c:
    dcn_pos = ba02c.find("### 11.3 DCN")
    if dcn_pos != -1:
        ba02c = ba02c[:dcn_pos] + dcn_dlrm_zh + "\n\n### 11.5 LHUC、SENet 与 FiBiNET\n" + ba02c[ba02c.find("### 11.4 LHUC") + len("### 11.4 LHUC"):]
        with open("notes/BusinessAlgorithm/BusinessAlgorithm02C Feature Interaction.md", "w", encoding="utf-8") as f:
            f.write(ba02c)
        print("Updated BusinessAlgorithm02C.md with DCN-v2 & DLRM pseudocode")

# ==============================================================================
# 2. Update BusinessAlgorithm02D (User Sequences) with DIN & SIM Pseudocode
# ==============================================================================
with open("notes/BusinessAlgorithm/BusinessAlgorithm02D User Sequences.md", "r", encoding="utf-8") as f:
    ba02d = f.read()

din_sim_zh = r"""### 12.3 DIN (Deep Interest Network)

DIN 核心创新是 **Target Attention**：用候选 Item $\mathbf{q}$ 与用户历史行为序列 $[\mathbf{h}_1, \dots, \mathbf{h}_L]$ 计算自适应注意力权重：

$$\alpha_j = \text{MLP}([\mathbf{h}_j, \mathbf{q}, \mathbf{h}_j - \mathbf{q}, \mathbf{h}_j \odot \mathbf{q}]), \quad \mathbf{u}(\mathbf{q}) = \sum_{j=1}^L \alpha_j \mathbf{h}_j$$

#### 伪代码实现：DIN 目标注意力前向
```python
import torch
import torch.nn as nn

class DINAttention(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64):
        super().__init__()
        # 输入维度: [q, h, q - h, q * h] -> 4 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1) # 输出标量注意力权重
        )

    def forward(self, query, history, mask=None):
        # query: [B, embed_dim], history: [B, L, embed_dim], mask: [B, L] (1=有效, 0=padding)
        B, L, d = history.shape
        q_expanded = query.unsqueeze(1).expand(B, L, d) # [B, L, d]
        # 拼接 4 种交互特征
        interaction = torch.cat([
            q_expanded, history, q_expanded - history, q_expanded * history
        ], dim=-1) # [B, L, 4*d]
        # 计算未归一化注意力得分 (DIN 不做 Softmax 归一化以保留总体兴趣强度)
        scores = self.mlp(interaction).squeeze(-1) # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask, 0.0)
        # 加权求和得到用户针对该候选物的动态兴趣表征
        user_interest = torch.bmm(scores.unsqueeze(1), history).squeeze(1) # [B, d]
        return user_interest
```

---

### 12.4 SIM (Search-based Interest Model: Hard & Soft Search)

面对 $L \ge 10,000$ 的终身超长序列，SIM 采用两阶段解耦策略：
1. **Hard Search（类目硬检索）**：在数万行为中快速过滤出与候选 Item 同类目（Sub-category）的 Top-$M$（$M \approx 50$）子序列；
2. **Soft Attention（精细注意力）**：在 $M$ 维子序列上结合时间差 Embedding（$\Delta t$）执行加权注意力。

#### 伪代码实现：SIM 检索增强长序列前向
```python
class SIMSequenceModel(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.time_delta_emb = nn.Embedding(100, embed_dim) # 时间差分桶嵌入
        self.attention = DINAttention(embed_dim * 2, hidden_dim=64)

    def forward(self, cand_id, cand_cat, user_hist_ids, user_hist_cats, user_hist_times, item_embed_table):
        # 1. 第一阶段: Hard Search (类目硬筛选 Top-M)
        # 筛选出与 cand_cat 匹配的历史行为索引
        match_mask = (user_hist_cats == cand_cat.unsqueeze(1)) # [B, L]
        # 取最近 M=50 个匹配项
        # 2. 第二阶段: 拼接商品向量与时间差向量进行精细 Target Attention
        cand_vec = item_embed_table(cand_id) # [B, d]
        # 假定选出 Top-M 的 history_ids 和 time_deltas
        hist_vec = item_embed_table(user_hist_ids[:, :50]) # [B, 50, d]
        time_vec = self.time_delta_emb(user_hist_times[:, :50]) # [B, 50, d]
        combined_hist = torch.cat([hist_vec, time_vec], dim=-1) # [B, 50, 2*d]
        combined_cand = torch.cat([cand_vec, torch.zeros_like(cand_vec)], dim=-1) # [B, 2*d]
        # 执行目标注意力
        return self.attention(combined_cand, combined_hist)
```
"""

if "class DINAttention" not in ba02d:
    din_pos = ba02d.find("### 12.3 DIN")
    if din_pos != -1:
        ba02d = ba02d[:din_pos] + din_sim_zh + "\n\n### 12.5 训练中的时间问题\n" + ba02d[ba02d.find("### 12.5 训练中的时间问题") + len("### 12.5 训练中的时间问题"):]
        with open("notes/BusinessAlgorithm/BusinessAlgorithm02D User Sequences.md", "w", encoding="utf-8") as f:
            f.write(ba02d)
        print("Updated BusinessAlgorithm02D.md with DIN & SIM pseudocode")

# ==============================================================================
# 3. Update BusinessAlgorithm02B (Multi-Objective) with MMoE & PLE Pseudocode
# ==============================================================================
with open("notes/BusinessAlgorithm/BusinessAlgorithm02B Multi-Objective Ranking.md", "r", encoding="utf-8") as f:
    ba02b = f.read()

mmoe_ple_zh = r"""### 10.3 MMoE (Multi-gate Mixture-of-Experts)

MMoE 为每个任务设计独立的门控网络，在共享的 Experts 集合上实现软路由：

$$\mathbf{h}_t(\mathbf{x}) = \sum_{e=1}^E g_{t,e}(\mathbf{x}) f_e(\mathbf{x}), \quad \mathbf{g}_t(\mathbf{x}) = \text{Softmax}(\mathbf{W}_t \mathbf{x})$$

#### 伪代码实现：MMoE 多门控混合专家前向
```python
import torch
import torch.nn as nn

class MMoE(nn.Module):
    def __init__(self, in_features, num_experts=4, expert_dim=64, num_tasks=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        # 共享 Experts 专家网络池
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, expert_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        # 各任务独占门控网络 (Softmax Gating)
        self.task_gates = nn.ModuleList([
            nn.Linear(in_features, num_experts) for _ in range(num_tasks)
        ])
        # 顶层任务塔 (Task Towers)
        self.task_towers = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        # x: [B, in_features]
        # 1. 计算所有专家前向输出: [B, num_experts, expert_dim]
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1)
        # 2. 分别为每个任务进行门控加权汇聚
        task_predictions = []
        for t in range(self.num_tasks):
            gate_weights = torch.softmax(self.task_gates[t](x), dim=-1).unsqueeze(-1) # [B, num_experts, 1]
            task_rep = (expert_outputs * gate_weights).sum(dim=1) # [B, expert_dim]
            task_pred = self.task_towers[t](task_rep) # [B, 1]
            task_predictions.append(task_pred)
        return task_predictions # [pCTR, pCVR]
```

---

### 10.4 PLE (Progressive Layered Extraction: 渐进式分层抽取)

PLE 彻底解耦了**任务独占专家（Task-Specific Experts）**与**全局共享专家（Shared Experts）**，阻断弱相关任务间的负迁移：

#### 伪代码实现：PLE 分层解耦专家前向
```python
class PLECustomExtractionLayer(nn.Module):
    def __init__(self, in_features, num_task_experts=2, num_shared_experts=2, expert_dim=64, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks
        # 各任务独占专家
        self.task_experts = nn.ModuleList([
            nn.ModuleList([nn.Linear(in_features, expert_dim) for _ in range(num_task_experts)])
            for _ in range(num_tasks)
        ])
        # 全局共享专家
        self.shared_experts = nn.ModuleList([
            nn.Linear(in_features, expert_dim) for _ in range(num_shared_experts)
        ])
        # 各任务私有门控 (只在任务专家 + 共享专家上做 Softmax)
        total_task_experts = num_task_experts + num_shared_experts
        self.task_gates = nn.ModuleList([
            nn.Linear(in_features, total_task_experts) for _ in range(num_tasks)
        ])
        # 共享门控 (在全部专家上做 Softmax)
        total_all_experts = num_task_experts * num_tasks + num_shared_experts
        self.shared_gate = nn.Linear(in_features, total_all_experts)

    def forward(self, task_inputs, shared_input):
        # 计算各专家输出
        task_exp_outs = [[exp(task_inputs[t]) for exp in self.task_experts[t]] for t in range(self.num_tasks)]
        shared_exp_outs = [exp(shared_input) for exp in self.shared_experts]
        
        # 1. 任务私有路由
        task_next_inputs = []
        for t in range(self.num_tasks):
            pool = torch.stack(task_exp_outs[t] + shared_exp_outs, dim=1) # [B, task_exp + shared_exp, d]
            gate = torch.softmax(self.task_gates[t](task_inputs[t]), dim=-1).unsqueeze(-1)
            task_rep = (pool * gate).sum(dim=1)
            task_next_inputs.append(task_rep)
            
        # 2. 共享路由 (汇聚全局)
        all_pool = torch.stack([item for sublist in task_exp_outs for item in sublist] + shared_exp_outs, dim=1)
        shared_gate = torch.softmax(self.shared_gate(shared_input), dim=-1).unsqueeze(-1)
        shared_next_input = (all_pool * shared_gate).sum(dim=1)
        
        return task_next_inputs, shared_next_input
```
"""

if "class PLECustomExtractionLayer" not in ba02b:
    mmoe_pos = ba02b.find("### 10.3 MMoE")
    if mmoe_pos != -1:
        ba02b = ba02b[:mmoe_pos] + mmoe_ple_zh + "\n\n### 10.5 ESMM 与转化漏斗\n" + ba02b[ba02b.find("### 10.4 ESMM 与转化漏斗") + len("### 10.4 ESMM 与转化漏斗"):]
        with open("notes/BusinessAlgorithm/BusinessAlgorithm02B Multi-Objective Ranking.md", "w", encoding="utf-8") as f:
            f.write(ba02b)
        print("Updated BusinessAlgorithm02B.md with MMoE & PLE pseudocode")

# ==============================================================================
# 4. Update MLCoding07 with explicit direct markdown links and pseudocode summary
# ==============================================================================
with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md", "r", encoding="utf-8") as f:
    content_07 = f.read()

target_base_models_zh = r"""### 1. 经典工业级基座模型三大支柱（含业务算法专篇深度链接与伪代码解析）

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
"""

old_base_start = content_07.find("### 1. 经典工业级基座模型三大支柱")
old_base_end = content_07.find("### 2. 多目标表征冲突与“跷跷板效应")
if old_base_start != -1 and old_base_end != -1:
    content_07 = content_07[:old_base_start] + target_base_models_zh + "\n" + content_07[old_base_end:]
    with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md", "w", encoding="utf-8") as f:
        f.write(content_07)
    print("Updated MLCoding07.md with links and model summaries")

# Update English note
with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.en.md", "r", encoding="utf-8") as f:
    content_07_en = f.read()

target_base_models_en = r"""### 1. The 3 Architectural Pillars (with Links to Business Algorithm Notes & Pseudocode)

Industrial precision rankers must execute bounded high-order feature interactions and activate tens of thousands of user behavior events under strict $P99 \le 15\text{ms}$ SLAs. The detailed architectural designs and PyTorch pseudocode implementations are documented across the Business Algorithm curriculum:

#### Pillar 1: Feature Interaction Backbone
> 📘 **Detailed Guide**: [Chapter 11 · Feature Interaction, Coarse Ranking, and Personalization](file:///Users/czk/Documents/mlsysnotes/MLSYS_tutorial/notes/BusinessAlgorithm/BusinessAlgorithm02C%20Feature%20Interaction.en.md) (with FM, DCN-v2 & DLRM code)
- **DCN-v2 (Deep & Cross Network v2)**: Employs low-rank decomposition ($\mathbf{W}_l = \mathbf{U}_l \mathbf{V}_l^T$) with MoE subspace gating to compute explicit bounded high-order polynomial cross terms with $\mathcal{O}(d \cdot r)$ efficiency: $\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l$;
- **DLRM (Deep Learning Recommendation Model)**: Meta's open-source architecture that explicitly decouples dense continuous features (Bottom MLP) from sparse categorical IDs (Embedding Tables), extracting upper-triangular dot-product interactions (`torch.bmm(E, E.T)`) into a Top MLP.

#### Pillar 2: User Behavior Sequence Modeling Backbone
> 📘 **Detailed Guide**: [Chapter 12 · User Behavior Sequences](file:///Users/czk/Documents/mlsysnotes/MLSYS_tutorial/notes/BusinessAlgorithm/BusinessAlgorithm02D%20User%20Sequences.en.md) (with DIN Target-Attention & SIM Two-Stage Retrieval code)
- **DIN (Deep Interest Network)**: Employs candidate query $\mathbf{q}$ over historical actions $[\mathbf{h}_1, \dots, \mathbf{h}_L]$ to compute **Target Attention**: $\mathbf{u}(\mathbf{q}) = \sum \alpha_j \mathbf{h}_j$ where $\alpha_j = \text{MLP}([\mathbf{q}, \mathbf{h}_j, \mathbf{q}-\mathbf{h}_j, \mathbf{q}\odot\mathbf{h}_j])$;
- **SIM (Search-based Interest Model)**: Two-stage decoupling—**Hard Search** filters Top-50 category-matched actions, followed by fine-grained soft attention with time-delta embeddings $\Delta t$, scaling to 50,000+ lifelong actions.

#### Pillar 3: Multi-Task Learning (MTL) Backbone
> 📘 **Detailed Guide**: [Chapter 10 · Multi-Objective Learning and Score Fusion](file:///Users/czk/Documents/mlsysnotes/MLSYS_tutorial/notes/BusinessAlgorithm/BusinessAlgorithm02B%20Multi-Objective%20Ranking.en.md) (with MMoE & PLE decoupled routing code)
- **MMoE (Multi-gate Mixture-of-Experts)**: Shared expert pool with task-specific Softmax gating $\mathbf{h}_t = \sum g_{t,e} f_e(\mathbf{x})$;
- **PLE (Progressive Layered Extraction)**: Physical isolation of **Task-Specific Experts** and **Shared Experts**, eliminating negative transfer and seesaw degradation between CTR and CVR.
"""

old_base_start_en = content_07_en.find("### 1. The 3 Architectural Pillars")
old_base_end_en = content_07_en.find("### Multi-Objective Representation Conflicts")
if old_base_start_en != -1 and old_base_end_en != -1:
    content_07_en = content_07_en[:old_base_start_en] + target_base_models_en + "\n\n" + content_07_en[old_base_end_en:]
    with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.en.md", "w", encoding="utf-8") as f:
        f.write(content_07_en)
    print("Updated MLCoding07.en.md with links and model summaries")

