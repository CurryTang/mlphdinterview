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

---

### 12.4B SASRec (Self-Attentive Sequential Recommendation, 2018)

DIN 采用候选驱动的局部注意力（Target Attention），能够解决“多样兴趣激活”问题，但序列内部物品之间**没有发生任何相互交互**，无法捕捉“用户买了手机 $\to$ 买了手机壳 $\to$ 接下来需要买充电线”这类严密的时间转移与因果演化。

**SASRec** 首次将 Transformer 的因果自注意力机制引入序列推荐：

```math
\mathbf{S} = \operatorname{Self-Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{M}\right)\mathbf{V}
```

#### 1. 核心架构设计
1. **因果掩码 (Causal Triangular Mask, $\mathbf{M}$)**：
   - 推荐属于时间序列自回归预测。在预测时刻 $t$ 的下一步意图时，模型**严禁看到时刻 $t+1$ 之后的未来行为**；
   - 掩码矩阵定义为上三角无穷小：
     $$M_{ij} = \begin{cases} 0, & i \ge j \\ -\infty, & i < j \end{cases}$$
2. **位置编码 (Positional Embeddings)**：
   - 为捕捉行为先后顺序，将每个位置赋予可学习的位置向量 $\mathbf{P} \in \mathbb{R}^{L \times d}$；
   - 序列输入表征为：$\hat{\mathbf{E}} = [\mathbf{e}_1 + \mathbf{p}_1, \mathbf{e}_2 + \mathbf{p}_2, \dots, \mathbf{e}_L + \mathbf{p}_L]$。
3. **点式前馈网络 (Pointwise Feed-Forward Network, FFN)**：
   - 每个自注意力层后接入两层全连接与 ReLU，赋予模型非线性建模能力，并伴随残差连接与 LayerNorm。
4. **DIN vs SASRec 的本质区别**：
   - **DIN（排序专属）**：用户向量依赖特定候选物品 $v_{\text{cand}}$，一次前向只能评估一个候选，计算复杂度随候选集规模线性放大，仅适合精排；
   - **SASRec（召回与排序通用）**：序列最后一个时间步输出的隐藏状态 $\mathbf{h}_L$ 完整编码了用户到目前为止的时序上下文意图。$\mathbf{h}_L$ **与具体候选无关**，可以直接作为用户向量推入 ANN 向量库进行全库最近邻检索（极速生成式序列召回），也能作为全局动态特征直接输入精排。

#### 伪代码实现：SASRec 因果自注意力前向
```python
import torch
import torch.nn as nn

class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, causal_mask):
        # x: [batch_size, seq_len, hidden_dim]
        # causal_mask: [seq_len, seq_len], 上三角全为 True (阻止注意力流向未来)
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=causal_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, hidden_dim=64, num_heads=2, num_blocks=2):
        super().__init__()
        self.max_len = max_len
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embeddings = nn.Embedding(max_len, hidden_dim)
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq_ids):
        # seq_ids: [batch_size, seq_len]
        batch_size, seq_len = seq_ids.size()
        positions = torch.arange(seq_len, device=seq_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # 1. 词嵌入与可学习位置编码相加
        x = self.item_embeddings(seq_ids) + self.pos_embeddings(positions)
        
        # 2. 构造因果掩码 (上三角布尔矩阵)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq_ids.device), diagonal=1).bool()
        
        # 3. 堆叠自注意力层
        for block in self.blocks:
            x = block(x, causal_mask)
            
        final_seq = self.final_norm(x) # [batch_size, seq_len, hidden_dim]
        # 取最后一个有效时间步作为全序列用户兴趣表征向量
        user_repr = final_seq[:, -1, :] # [batch_size, hidden_dim]
        return user_repr
```

---

### 12.5 序列时间表征演进：从单调衰减到兴趣时钟 (Circadian Rhythm & Interest Clock)

在短视频、资讯流与电商等信息流推荐中，用户的消费意图存在强烈的**生理节律与作息周期（Circadian Rhythm）**。“兴趣时钟”的核心是将传统的“单调距离衰减”升级为“单调趋势衰减（Trend Decay）+ 周期谐振（Periodic Resonance）”的双轨制时间表征体系。

#### 1. 传统单调时间距离建模的缺陷

大多数经典序列推荐模型（如早期 DIN、BST、SIM 的 Soft Search）仅计算行为发生时刻与当前时刻的单调时间差：

$$\Delta t = t_{\text{target}} - t_{\text{action}}$$

并通过对数分桶后学习离散嵌入：$\mathbf{e}_{\Delta t} = \text{Embedding}(\text{Bucket}(\log(1 + \Delta t)))$。

这种建模隐含了一个强假设：**物理时间流逝越久，历史行为与当前意图的相关度必然单调递减**。但在信息流推荐中，这一假设频繁失效：
- **作息心智割裂**：用户昨晚 23:30 深度消费的“助眠白噪音/情感电台”，其与今晚 23:30 的相关度，远高于今天上午 09:00 通勤时顺手划过的“突发财经早报”。
- **负面稀释效应**：若纯粹依赖 $\Delta t$ 的单调衰减，昨夜高价值的同周期深度兴趣会被长达 14 个小时的物理时间跨度强行抑制，导致夜间模型无法及时复现用户的作息心智。

#### 2. 兴趣时钟的双重解耦表征

兴趣时钟从两个互补的物理维度解耦时间动态：

```text
[时间建模双轨制架构]

                 ┌── 1. 绝对时钟槽位 (Wall-Clock): 学习全天 0:00~23:59 各时段固有心智
时间戳投影 ──────┤
                 └── 2. 相对圆周相位差 (Circular Delta): 衡量目标与历史在 24h 圆盘上的最短弧距
```

- **维度一：绝对时钟槽位（Wall-Clock Time）**
  将绝对时间戳投影到以 24 小时为闭环周期的圆盘上（0:00 ~ 23:59）：
  - **连续正余弦调和基底（Continuous Harmonic Basis）**：
    $$e_{\text{sin}} = \sin\left(2\pi \cdot \frac{\text{minute\_of\_day}}{1440}\right), \quad e_{\text{cos}} = \cos\left(2\pi \cdot \frac{\text{minute\_of\_day}}{1440}\right)$$
    有效消除 23:59 与 00:01 在欧式空间中的拓扑截断，保持平滑连续。
  - **离散槽位嵌入（Discrete Clock Slot Embedding）**：
    工业界高吞吐推荐通常采用等间隔离散切分，如全天划分 96 个槽位（每 15 分钟一槽）或 48 个槽位（每 30 分钟一槽），为每个槽位分配独立的稠密向量 $\mathbf{e}_{\text{clock\_slot}} \in \mathbb{R}^d$。

- **维度二：相对时钟相位差（Circular Clock Delta）**
  计算历史时刻与当前预测时刻在 24 小时时钟圆盘上的**最短圆周角弧距（Shortest Circular Arc Distance）**：
  $$\Delta \tau_i = \min\left(\vert{}h_{\text{target}} - h_i\vert{},\; 24 - \vert{}h_{\text{target}} - h_i\vert{}\right)$$
  若当前请求时刻为凌晨 01:00（$h_{\text{target}} = 1$），历史行为发生于昨夜 23:00（$h_i = 23$），虽然物理时间差 $\Delta t = 26 \text{ 小时}$，但圆周时钟相位差仅为 $\Delta \tau = 2 \text{ 小时}$，模型可准确激活“深夜同时间段兴趣谐振”。

#### 3. 融合机制：Embedding 级拼接 vs. Attention Bias 干预

- **Embedding 级融合（输入侧粗干预）**：
  $$\mathbf{x}_i = \mathbf{e}_{\text{item}} + \mathbf{e}_{\text{action}} + \mathbf{e}_{\text{clock\_slot}} + \mathbf{e}_{\Delta t}$$
  将时钟特征作为辅助侧信息拼入序列单元。其局限在于：通过后续全连接层隐式交互，无法强约束注意力打分直接响应周期接近性。
- **Attention Bias 级干预（注意力侧显式干预，工业界收益最高）**：
  在 Query-Key 交互矩阵中直接注入可学习的时钟圆周偏置标量：
  $$\text{Score}(Q, K_i) = \frac{Q K_i^T}{\sqrt{d}} - \lambda \cdot \log(1 + \Delta t_i) + \text{Bias}_{\text{circ}}(\Delta \tau_i)$$
  该公式实现了“衰减阻尼与周期共振”的数学统一：基础项 $-\lambda \log(1 + \Delta t)$ 维持长程兴趣漂移的自然衰减，而偏置项 $+\text{Bias}_{\text{circ}}$ 则在相位接近（$\Delta \tau \to 0$）时，将同作息段的高质量历史权重精准“拉回”。

#### 4. 生产级 PyTorch 实现：InterestClockAttention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterestClockAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_clock_slots: int = 96):
        """
        num_clock_slots: 一天划分为多少个时间槽（96 表示每 15 分钟一个槽位）
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_clock_slots = num_clock_slots

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 1. 周期时钟槽位的绝对 Embedding (0 ~ num_clock_slots-1)
        self.clock_slot_embed = nn.Embedding(num_clock_slots, d_model)

        # 2. 相对时钟圆周差的 Attention Bias (最大差值为 num_clock_slots // 2)
        # 例如 96 槽位，最大最短圆周差为 48，映射到各头独立的偏置标量
        max_circ_diff = num_clock_slots // 2 + 1
        self.circ_bias_table = nn.Embedding(max_circ_diff, num_heads)

    def _extract_clock_slots(self, timestamps: torch.Tensor) -> torch.Tensor:
        """从 Unix 时间戳提取一天中的槽位索引 (0 ~ num_clock_slots - 1)"""
        seconds_in_day = timestamps % 86400
        slot_duration = 86400 / self.num_clock_slots
        return (seconds_in_day / slot_duration).long()

    def _calc_circular_delta(self, target_slots: torch.Tensor, hist_slots: torch.Tensor) -> torch.Tensor:
        """计算目标时刻与历史时刻在时钟圆盘上的最短距离"""
        abs_diff = torch.abs(target_slots - hist_slots)  # [B, 1, L]
        circ_diff = torch.minimum(abs_diff, self.num_clock_slots - abs_diff)
        return circ_diff

    def forward(
        self,
        target_item: torch.Tensor,       # [B, 1, d_model]
        target_ts: torch.Tensor,         # [B, 1] (Unix 时间戳)
        hist_items: torch.Tensor,        # [B, L, d_model]
        hist_ts: torch.Tensor,           # [B, L] (Unix 时间戳)
        mask: torch.Tensor = None        # [B, 1, L] (True 表示有效)
    ) -> torch.Tensor:
        B, L, _ = hist_items.shape

        # 1. 提取时间槽索引
        tgt_slot = self._extract_clock_slots(target_ts)       # [B, 1]
        hist_slot = self._extract_clock_slots(hist_ts)        # [B, L]

        # 2. 在 Input 端叠加绝对时钟表征
        tgt_input = target_item + self.clock_slot_embed(tgt_slot)
        hist_input = hist_items + self.clock_slot_embed(hist_slot)

        # 3. 线性投影与多头分拆
        Q = self.q_proj(tgt_input).view(B, 1, self.num_heads, self.d_k).transpose(1, 2)   # [B, H, 1, d_k]
        K = self.k_proj(hist_input).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        V = self.v_proj(hist_input).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]

        # 4. 基础 Attention 打分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, 1, L]

        # 5. 注入时钟相对圆周偏置 (Clock Attention Bias)
        circ_diff = self._calc_circular_delta(tgt_slot.unsqueeze(2), hist_slot.unsqueeze(1))  # [B, 1, L]
        clock_bias = self.circ_bias_table(circ_diff.squeeze(1))  # [B, L, H]
        clock_bias = clock_bias.permute(0, 2, 1).unsqueeze(2)    # [B, H, 1, L]

        scores = scores + clock_bias

        # 6. Padding Mask 与加权聚合
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, L]
        out = torch.matmul(attn_weights, V)       # [B, H, 1, d_k]
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        return out
```

#### 5. 工业落地的三大工程实践细节

1. **本地时区归一化（Local Timezone Normalization）**：
   原始日志中的时间戳均为 UTC 时间。提取时钟槽位前，必须根据客户端地理位置或时区偏置（Timezone Offset）将 UTC 转换为**用户本地壁钟时间（Local Wall-Clock Time）**。否则相同 UTC 时间在纽约（深夜）与北京（中午）会发生灾难性的节律反相。
2. **多尺度层级时钟（Hierarchical Clocks: Hour + Weekday）**：
   工业成熟模型通常构建双层时钟嵌入：
   - **Hour Clock（24h 周期）**：刻画早间通勤、午餐放空、深夜助眠的昼夜节律；
   - **Weekday Clock（7d 周期）**：刻画“工作日快节奏消费”与“周末深度沉浸探索”的跨天心智漂移。两者偏置在打分矩阵中线性叠加。
3. **线上 Serving 零损耗优化（Offline Pre-quantization & In-memory Cache）**：
   在实时召回/精排链路中，当前请求的 Target Slot 仅需计算 1 次；而历史行为各 Item 的时钟 Slot 会在事件进入实时特征流（Flink/Kafka）时预先完成量化并存储在 Feature Store 中。在线打分仅需单次数组寻址与查表，对线上 P99 耗时影响趋近于 0。

---

### 12.6 训练中的时间问题

序列模型最危险的 bug 是时间穿越。

样本发生在时间 `t`，历史只能使用 `t` 之前可见的行为。统计特征、画像和 item 热度也要按 `t` 截止。离线直接读当前用户全量历史，指标会漂亮得不真实。

还要处理：

- 同一 session 内标签泄漏；
- 重复曝光；
- 行为日志乱序；
- 延迟到达；
- 负反馈和无效播放；
- 训练截断与线上截断不一致。

---

### 12.7 实时更新

序列模型的价值经常来自最近几次行为。若用户刚看完滑雪视频，特征服务五分钟后才更新，模型再复杂也反应不过来。

常见做法是：

- 长期序列离线存储；
- 短期行为流式更新；
- 在线拼接并去重；
- 对缺失或迟到行为降级；
- 记录特征版本用于回放。

模型参数也可以做增量更新：凌晨基于完整窗口训练全量模型，白天按小时消费新日志做小步更新。增量模型降低兴趣变化的响应时间，但会遇到延迟标签、灾难性遗忘和错误数据快速扩散。线上必须能回退到最近一次全量 checkpoint，并分别记录全量与增量数据版本。

---

### 12.8 本章自测

1. Last-N 平均池化丢失了哪些信息？
2. DIN 的用户表示为什么依赖候选？
3. SIM 为什么要两阶段处理长序列？
4. 传统单调时间差（$\Delta t$）建模在处理生理节律时有什么根本缺陷？兴趣时钟是如何解决的？
5. 怎样检查序列特征是否时间穿越？
6. 线上短期行为更新失败时如何降级？

<details>
<summary>参考答案</summary>

1. 平均池化会丢掉顺序、时间间隔、重复强度和兴趣变化，也无法区分同一组行为的不同发生顺序。
2. DIN 用候选 item 作为 query 对历史行为做 attention，所以同一用户面对不同候选会得到不同兴趣表示。
3. 它先用较便宜的机制从长历史中检索与候选相关的子序列，再对短子序列做精细建模，避免全量序列交互的成本。
4. 传统 $\Delta t$ 强假设物理时间越久相关度必然越低，导致昨夜高价值的同周期深度兴趣（如助眠情感电台）被物理跨度过度稀释；兴趣时钟通过解耦“绝对壁钟槽位”与“24 小时圆周相位差（$\Delta \tau$）”，在 Attention 打分中施加周期谐振偏置，实现同作息时段历史的精准拉回。
5. 对每条样本验证所有行为时间戳都早于请求时间，并用 point-in-time join 回放；还要检查离线聚合窗口是否包含未来事件。
6. 回退到较旧但有版本的短期序列，或只使用长期画像与热门特征，同时记录降级比例并避免把缺失值伪装成真实空历史。

</details>
