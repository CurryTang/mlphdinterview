# 用户行为序列

## 第 12 章 用户行为序列

行为序列不是越长越好，把全部日志交给 Transformer 也不等于完成了建模。模型真正要估计的是用户此刻的状态：哪些兴趣还有效，刚才的行为是否改变了意图，当前候选应该读取哪一段历史。这个估计受存储和延迟预算约束。

事件进入模型前就已经有取舍。误触、自动连播和主动收藏不能同权；连续观看同一作者的十条内容，也未必提供了十份独立证据。行为强度、去重、session 边界、时间间隔和负反馈处理，常常比换一层网络更早影响效果。

线上通常同时保留两种用户表示。通用 user embedding 每个请求算一次，适合召回和大候选粗筛；候选相关表示用当前 item 查询历史，表达更细，但每个候选都要重复计算。DIN、SIM 以及更长的序列模型，只是在这组效果与成本的取舍上选择了不同位置。

### 12.1 平均池化丢掉了什么

用户看过篮球、做饭、音乐和旅行内容。把所有 item embedding 平均，会得到一个模糊的"总体兴趣"，却不知道当前候选与哪段历史相关，也忽略时间顺序。

序列模型主要解决三件事：

- 不同行为权重不同；
- 当前候选需要读取不同历史；
- 兴趣随时间演化。

### 12.2 Last-N

最简单的做法取最近 N 个行为。它便宜，也往往比复杂模型想象中更强。

可以加入：

- 行为类型权重；
- 时间衰减；
- 去重与连续播放压缩；
- 有效播放阈值；
- 类目或作者分组。

Last-N 的 N 不是越大越好。长历史带来噪声、存储与服务成本，也可能把旧兴趣重新放大。

### 12.3 DIN (Deep Interest Network)

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


### 12.5 训练中的时间问题


序列模型最危险的 bug 是时间穿越。

样本发生在时间 `t`，历史只能使用 `t` 之前可见的行为。统计特征、画像和 item 热度也要按 `t` 截止。离线直接读当前用户全量历史，指标会漂亮得不真实。

还要处理：

- 同一 session 内标签泄漏；
- 重复曝光；
- 行为日志乱序；
- 延迟到达；
- 负反馈和无效播放；
- 训练截断与线上截断不一致。

### 12.6 实时更新

序列模型的价值经常来自最近几次行为。若用户刚看完滑雪视频，特征服务五分钟后才更新，模型再复杂也反应不过来。

常见做法是：

- 长期序列离线存储；
- 短期行为流式更新；
- 在线拼接并去重；
- 对缺失或迟到行为降级；
- 记录特征版本用于回放。

模型参数也可以做增量更新：凌晨基于完整窗口训练全量模型，白天按小时消费新日志做小步更新。增量模型降低兴趣变化的响应时间，但会遇到延迟标签、灾难性遗忘和错误数据快速扩散。线上必须能回退到最近一次全量 checkpoint，并分别记录全量与增量数据版本。

### 12.7 本章自测

1. Last-N 平均池化丢失了哪些信息？
2. DIN 的用户表示为什么依赖候选？
3. SIM 为什么要两阶段处理长序列？
4. 怎样检查序列特征是否时间穿越？
5. 线上短期行为更新失败时如何降级？

<details>
<summary>参考答案</summary>

1. 平均池化会丢掉顺序、时间间隔、重复强度和兴趣变化，也无法区分同一组行为的不同发生顺序。
2. DIN 用候选 item 作为 query 对历史行为做 attention，所以同一用户面对不同候选会得到不同兴趣表示。
3. 它先用较便宜的机制从长历史中检索与候选相关的子序列，再对短子序列做精细建模，避免全量序列交互的成本。
4. 对每条样本验证所有行为时间戳都早于请求时间，并用 point-in-time join 回放；还要检查离线聚合窗口是否包含未来事件。
5. 回退到较旧但有版本的短期序列，或只使用长期画像与热门特征，同时记录降级比例并避免把缺失值伪装成真实空历史。

</details>
