# 多目标学习与分数融合

## 第 10 章 多目标学习与分数融合

### 10.1 为什么会有多目标

短视频平台可能同时关心点击、播放时长、完播、点赞、关注和负反馈。电商关心点击、加购、下单和成交额。搜索还要考虑相关性、质量、时效、地域和个性化。

把所有目标粗暴加成一个标签，会丢掉结构。分别训练多个模型又会重复计算，并让低频任务缺少数据。多任务学习就在这两端之间找平衡。

这里有两件经常被混在一起的事：

```text
多任务学习：怎样共享表示，同时预测 CTR、时长、CVR 等目标
分数融合：线上最终怎样用这些预估决定一个 item 的排序分
```

MMoE、Shared-Bottom 和 ESMM 主要解决前一件事。加法、乘法、rank fusion 或学习式融合解决后一件事。多个 head 都预测得更准，不代表最终列表一定更好；如果融合权重让一个高频目标压过质量和负反馈，模型结构再漂亮也会走偏。

反过来，融合公式也救不了坏标签。点击、时长和购买的观察窗口、采样率与校准口径不同，必须先让每个 head 的目标可解释，再讨论权重。线上调权是在明确的目标之间做产品取舍，不是弥补训练数据没有定义清楚。

### 10.2 Shared-Bottom

最简单的结构共享底层：

```text
features -> shared network -> task A tower
                           -> task B tower
                           -> task C tower
```

总损失：

```math
\mathcal L=\sum_t \lambda_t\mathcal L_t.
```

问题是任务梯度可能冲突。点击偏好标题吸引力，长时长偏好内容持续价值，它们不总朝同一方向更新共享参数。

### 10.3 MMoE (Multi-gate Mixture-of-Experts)

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


### 10.5 ESMM 与转化漏斗


电商 CVR 只在点击后可观察。若只用点击样本训练 CVR，训练分布与全量曝光分布不同。

ESMM 利用：

```math
P(\text{click and conversion})
=P(\text{click})P(\text{conversion}\mid\text{click}),
```

在全量曝光空间联合学习 CTR 和 CTCVR，再由二者关系约束 CVR。它缓解样本选择偏差与转化稀疏，但仍依赖模型假设和数据口径，不代表反事实问题完全解决。

### 10.5 时长建模与短视频偏置消除

播放时长既有零膨胀（大量未点击或快速划走），又受视频本身物理长度的强烈制约。

#### 1. YouTube (2016) 加权逻辑回归 (Weighted Logistic Regression) 的数学推导
YouTube 2016 精排模型提出了一种精妙的方案：**用分类损失端到端无偏拟合连续观看时长**，避免了回归损失（MSE）对极端长时长的敏感性，同时不丢失未点击样本的信息。

**加权训练机制：**
- 对正样本（发生点击并播放的展现），将其样本权重设为实际观看时长 $T_i$（秒数）；
- 对负样本（未点击或刚曝光即划走的展现），将其样本权重设为 1。

**严密数学推导：**
设测试集共有 $N$ 个曝光展现，其中正样本数为 $N_+$，负样本数为 $N_- = N - N_+$。
加权交叉熵优化下，模型学到的对数几率比 (Odds) 为正负样本加权和之比：

```math
\operatorname{Odds} = \frac{p}{1-p} = \frac{\sum_{i \in \text{pos}} T_i}{\sum_{j \in \text{neg}} 1} = \frac{\sum_{i \in \text{pos}} T_i}{N - N_+} = \frac{\frac{1}{N}\sum_{i \in \text{pos}} T_i}{1 - \frac{N_+}{N}} = \frac{\mathbb{E}[T]}{1 - p_{\text{click}}}
```

在推荐与广告系统中，单条内容的点击率天然较低（$p_{\text{click}} \ll 1$，如 $p_{\text{click}} \approx 0.02 \sim 0.05$），因此分母 $1 - p_{\text{click}} \approx 1$：

```math
\operatorname{Odds} \approx \mathbb{E}[T]
```

由于逻辑回归预测的概率为 $p = \sigma(z) = \frac{1}{1 + e^{-z}}$，其对数几率比恒等于最后一层 Logit 的指数：$\operatorname{Odds} = e^z$。

**核心工程收益：**
- **训练阶段**：直接使用带样本权重的常规二元交叉熵损失（BCEWithLogitsLoss(weight=...)）；
- **推理阶段**：**直接计算最后一层线性激活的指数 $\hat{T} = e^z$ 作为期望观看时长的无偏预估**！无需负样本过滤，无对数转换偏置，直接统一分类与时长预测。

#### 2. 短视频时长悖论 (Duration Bias Paradox) 与长短视频博弈
在短视频（TikTok / Reels / 快手）场景中，直接优化绝对时长或完播率会引发严重的生态扭曲：
- **完播率悖论 (Completion Rate Trap)**：15 秒视频播放 12 秒，完播率为 80%；180 秒深度视频播放 45 秒，完播率仅 25%。若只以完播率为目标，推荐池会迅速塌陷为清一色的 5 秒无脑超短片；
- **时长悖论 (Duration Trap)**：若只优化绝对播放秒数，45 秒远胜 12 秒，模型会强推冗长、拖沓的低质长视频，引发用户划走率飙升与留存下滑。

**工业级解决方案：分桶标准化 (Bucket-wise Z-Score)**
将视频按物理长度 $L$ 切分为互斥区间（如 `[0, 15s)`, `[15, 30s)`, `[30, 60s)`, `[60, 180s)`, `[180s, +inf)`）。
在各个分桶内统计全站真实播放时长的均值 $\mu_L$ 与标准差 $\sigma_L$，对标签做 Z-score 归一化：

```math
\tilde{T} = \frac{T - \mu_L}{\sigma_L}
```

通过将绝对物理时长剥离，模型在同一长度分桶内比拼**相对超越度**，兼顾短视频的高完播节奏与中长视频的深度沉浸，生态分布更健康。

### 10.6 分数融合

模型输出通常不能直接线性相加。CTR 可能在 `[0, 0.2]`，时长预测是秒，CVR 更稀疏。先做校准，再讨论融合。

常见形式：

```math
S
=w_1f_1(\hat p_{\text{click}})
+w_2f_2(\hat t)
+w_3f_3(\hat p_{\text{conversion}})
-w_4\hat p_{\text{negative}}.
```

`f_t` 可以是 log、幂函数、分段函数或分位数映射。权重不只靠离线搜索，最终需要在线实验。

课程里的几种典型融合各有侧重点：

```math
S_{\text{add}}
=p_{\text{click}}+w_1p_{\text{like}}+w_2p_{\text{share}}+\cdots,
```

```math
S_{\text{rank}}
=\sum_j\frac{w_j}{r_j+\beta_j},
```

```math
S_{\text{commerce}}
=p_{\text{click}}^{\alpha}
\times p_{\text{cart}}^{\beta}
\times p_{\text{pay}}^{\gamma}
\times \operatorname{price}^{\delta}.
```

第一种依赖校准后的数值尺度；第二种只看各目标的候选内名次，尺度更稳但丢失分差；电商乘法形式贴合曝光到支付的漏斗，任何一项接近零都会强烈压低总分。

另一条路是学习融合模型，把各目标分数和上下文作为输入。但它仍要有训练标签，且更难解释目标权衡。业务强约束最好保留在重排或规则层。

### 10.7 概率校准与负采样还原

如果模型输出 0.2，在该预估区间的样本中确实有 20% 真正点击，分数就是完全校准的。

#### 1. 为什么校准对工业界至关重要？
- **多目标分数融合**：加权求和 $w_1 p_{\text{click}} + w_2 p_{\text{like}}$ 的前提是两者的概率数值处于同一客观物理尺度。若新版模型使得 $p_{\text{click}}$ 系统性上浮 50%，融分比例被彻底破坏；
- **广告竞价 (eCPM)**：广告按 $\text{eCPM} = 1000 \times \text{pCTR} \times \text{bid}$ 排序扣费。若预估概率偏高 20%，广告主预算将被过快透支，平台产生巨额超成本赔付。

#### 2. 校准评测指标：E/O (Expected over Observed / COP)
```math
\operatorname{E/O} = \frac{\sum_{i=1}^N \hat{p}_i}{\sum_{i=1}^N y_i}
```
$\operatorname{E/O} > 1$ 表示系统性高估，$\operatorname{E/O} < 1$ 表示系统性低估。工业界要求整体及核心分桶的 $\operatorname{E/O} \in [0.98, 1.02]$。常用算法包括 **Platt Scaling (逻辑回归校准)**、**Isotonic Regression (保序回归)** 与 **分桶平滑映射**。

#### 3. 负样本降采样的理论还原推导
为节省存储与算力，工业界常对海量未点击负样本执行负采样（Downsampling）。若正样本全保留，负样本采样率为 $\alpha$（保留比例 $\alpha \in (0, 1)$）：
- 真实数据集中几率比为：$\text{Odds} = \frac{N_+}{N_-}$；
- 采样后训练集负样本变为 $\alpha N_-$，采样数据集上的几率比为：
  $$\text{Odds}_s = \frac{N_+}{\alpha N_-} = \frac{1}{\alpha} \text{Odds} \implies \text{Odds} = \alpha \cdot \text{Odds}_s$$
- 采样集上模型预估概率为 $p_s$，则 $\text{Odds}_s = \frac{p_s}{1 - p_s}$。代入真实概率公式 $p = \frac{\text{Odds}}{1 + \text{Odds}}$：

```math
p = \frac{\alpha \frac{p_s}{1 - p_s}}{1 + \alpha \frac{p_s}{1 - p_s}} = \frac{\alpha p_s}{1 - p_s + \alpha p_s}
```

只做降采样、不做几率还原，会使模型输出系统性虚高几十倍，直接摧毁后续所有多目标融合与出价公式。

### 10.8 从排序损失到偏好优化

BCE 判断单个 pair，BPR 比较一对 item，InfoNCE 让一个正例与一组候选竞争。三者都利用正负反馈，比较粒度和负样本来源不同。

生成式推荐把比较单位扩展到 token 或完整序列。next-token CE 与整个词表竞争，DPO 比较 chosen/rejected 序列，policy gradient 用 advantage 给 rollout 加权。RL 的低 advantage rollout 不能简单当成固定负样本，因为候选由当前 policy 产生，样本权重也会随训练变化。细节见 [[BusinessAlgorithm05 Generative Recommendation.md#18.9 从正负样本到 RL|生成式推荐中的偏好优化]]。

### 10.9 本章自测

1. Shared-Bottom 的负迁移从哪里来？
2. MMoE 的 gate 可以怎样诊断？
3. ESMM 解决了 CVR 的哪两个问题？
4. 为什么直接预测播放秒数会偏长视频？
5. AUC 不变时，校准为什么仍可能改善线上融合？
6. BCE、BPR、InfoNCE 的比较粒度有什么不同？

<details>
<summary>参考答案</summary>

1. 多个任务共享同一表示时，梯度方向可能冲突，数据量大的任务还会主导参数更新，使其他任务变差。
2. 看不同任务和样本上的 gate 分布、expert 使用率、熵与负载；长期只选一个 expert 或所有 gate 完全相同都值得检查。
3. 它用曝光→点击和点击→转化的联合建模缓解 CVR 样本选择偏差，并利用全曝光空间缓解只在点击样本上训练造成的数据稀疏。
4. 秒数上界随视频长度增长，模型容易把长度当作收益。可预测完播率、分桶时长或使用带长度归一化的目标。
5. AUC 只看相对顺序。融合多个目标时需要概率尺度可比，校准能避免某个头仅因分数偏大而压过其他目标。
6. BCE 判断单个样本；BPR 比较一对正负 item；InfoNCE 让正例与一组 batch 或采样候选共同竞争。

</details>
