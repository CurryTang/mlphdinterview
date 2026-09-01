# -*- coding: utf-8 -*-

# Read current Chinese file
with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md", "r", encoding="utf-8") as f:
    zh_content = f.read()

pseudocode_section_zh = r"""### 2. 五大架构核心算法与 PyTorch 伪代码实现（可折叠查看）

<details class="exercise" open>
<summary><span class="q-label">架构 1 · 伪代码</span> <span class="q-text">截断自注意力架构（SASRec / BST 行为序列自注意力网络）</span></summary>

```python
import torch
import torch.nn as nn

class SASRecTruncatedTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, num_layers=2, max_len=50, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, seq_embeddings, mask=None):
        # seq_embeddings: [B, N, d] (截断最近 N=50 个行为)
        # mask: [B, N] (1=有效, 0=padding)
        B, N, d = seq_embeddings.shape
        positions = torch.arange(N, device=seq_embeddings.device).unsqueeze(0).expand(B, N)
        x = self.layer_norm(seq_embeddings + self.pos_emb(positions))
        
        # 构造因果下三角掩码 (Causal Mask)
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        padding_mask = (~mask) if mask is not None else None
        
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask) # [B, N, d]
        user_recent_rep = out[:, -1, :] # 提取序列最末尾 Token 的上下文表征 [B, d]
        return user_recent_rep
```
</details>

<details class="exercise">
<summary><span class="q-label">架构 2 · 伪代码</span> <span class="q-text">压缩记忆网络（MIMN 神经槽位记忆矩阵）</span></summary>

```python
class MIMNMemoryNetwork(nn.Module):
    def __init__(self, embed_dim=64, num_slots=8):
        super().__init__()
        self.num_slots = num_slots # 压缩为 C=8 个兴趣槽位
        self.embed_dim = embed_dim
        # 槽位写入控制器 (NTM / GRU Write Controller)
        self.write_gate = nn.Linear(embed_dim * 2, num_slots)
        self.erase_gate = nn.Linear(embed_dim, embed_dim)
        self.add_gate = nn.Linear(embed_dim, embed_dim)

    def forward(self, user_history_stream, initial_memory=None):
        # user_history_stream: [B, L, d] (超长流式行为 L >= 10000)
        # initial_memory: [B, num_slots, d] (常驻于线上 KV/Redis 中的用户槽位状态)
        B, L, d = user_history_stream.shape
        M = initial_memory if initial_memory is not None else torch.zeros(B, self.num_slots, d, device=user_history_stream.device)
        
        # 增量流式写入与遗忘更新 (可在 Flink/Spark 离线或近线流式执行)
        for t in range(L):
            xt = user_history_stream[:, t, :] # [B, d]
            # 计算写入当前行为在各 Slot 上的注意力权重
            w = torch.softmax(self.write_gate(torch.cat([xt, M.mean(dim=1)], dim=-1)), dim=-1) # [B, C]
            # 擦除与新增
            erase = torch.sigmoid(self.erase_gate(xt)).unsqueeze(1) # [B, 1, d]
            add = torch.tanh(self.add_gate(xt)).unsqueeze(1)        # [B, 1, d]
            M = M * (1.0 - w.unsqueeze(-1) * erase) + (w.unsqueeze(-1) * add)
            
        return M # 输出压缩槽位矩阵 [B, C, d]，在线精排时以 O(1) 复杂度直接读取
```
</details>

<details class="exercise">
<summary><span class="q-label">架构 3 · 伪代码</span> <span class="q-text">终身目标注意力架构（DIN 全量 Target Attention）</span></summary>

```python
class DINFullTargetAttention(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.activation_unit = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1) # 输出局部注意力得分
        )

    def forward(self, candidate_query, lifelong_history, mask=None):
        # candidate_query: [B, d], lifelong_history: [B, L, d] (全量历史序列)
        B, L, d = lifelong_history.shape
        q_exp = candidate_query.unsqueeze(1).expand(B, L, d) # [B, L, d]
        
        # 4 种交互特征拼接: [q, h, q - h, q * h]
        interaction = torch.cat([q_exp, lifelong_history, q_exp - lifelong_history, q_exp * lifelong_history], dim=-1)
        scores = self.activation_unit(interaction).squeeze(-1) # [B, L]
        
        if mask is not None:
            scores = scores.masked_fill(~mask, 0.0)
            
        # 未归一化加权池化 (保留用户总兴趣强度)
        user_interest = torch.bmm(scores.unsqueeze(1), lifelong_history).squeeze(1) # [B, d]
        return user_interest
```
</details>

<details class="exercise">
<summary><span class="q-label">架构 4 · 伪代码</span> <span class="q-text">分层多分辨率池化架构（HPMN 多时间粒度分层聚合）</span></summary>

```python
class HPMNHierarchicalPooling(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.decay_lambda = nn.Parameter(torch.tensor([0.05])) # 可学习的时间衰减系数

    def forward(self, session_seq, daily_seq, monthly_seq, time_deltas_daily, time_deltas_monthly):
        # session_seq: [B, N_sess, d] (实时会话级最近行为)
        # daily_seq: [B, N_days, d] (按天聚合的中期行为)
        # monthly_seq: [B, N_months, d] (按月聚合的长期基础兴趣)
        
        # 1. 实时层: 均值池化
        h_session = session_seq.mean(dim=1) # [B, d]
        
        # 2. 中期层: 指数时间衰减池化 exp(-lambda * delta_t)
        decay_daily = torch.exp(-torch.clamp(self.decay_lambda, min=1e-4) * time_deltas_daily).unsqueeze(-1) # [B, N_days, 1]
        h_daily = (daily_seq * decay_daily).sum(dim=1) / (decay_daily.sum(dim=1) + 1e-6) # [B, d]
        
        # 3. 长期层: 时间加权池化
        decay_monthly = torch.exp(-torch.clamp(self.decay_lambda, min=1e-4) * time_deltas_monthly).unsqueeze(-1)
        h_monthly = (monthly_seq * decay_monthly).sum(dim=1) / (decay_monthly.sum(dim=1) + 1e-6) # [B, d]
        
        # 4. 多分辨率表征级联拼接
        h_user_hierarchical = torch.cat([h_session, h_daily, h_monthly], dim=-1) # [B, 3*d]
        return h_user_hierarchical
```
</details>

<details class="exercise">
<summary><span class="q-label">架构 5 · 伪代码</span> <span class="q-text">两阶段检索增强架构（SIM Hard/Soft Search 与时序软注意力）</span></summary>

```python
class SIMRetrievalAugmentedModel(nn.Module):
    def __init__(self, embed_dim=64, top_m=50):
        super().__init__()
        self.top_m = top_m
        self.time_delta_emb = nn.Embedding(100, embed_dim) # 时间跨度分桶嵌入
        self.attention = DINFullTargetAttention(embed_dim * 2, hidden_dim=64)

    def forward(self, cand_item_id, cand_category_id, user_lifelong_ids, user_lifelong_cats, user_lifelong_times, item_embed_table):
        # 阶段 1: Hard Search (按同类目从万级历史中极速筛选 Top-M 行为)
        # cand_category_id: [B], user_lifelong_cats: [B, L]
        match_mask = (user_lifelong_cats == cand_category_id.unsqueeze(1)) # [B, L]
        
        # 阶段 2: 提取命中子序列并拼接时间差向量进行精细 Target Attention
        cand_vec = item_embed_table(cand_item_id) # [B, d]
        # 假定取出匹配的前 M=50 个 items 与其对应的时间差分桶 times
        hist_vec = item_embed_table(user_lifelong_ids[:, :self.top_m]) # [B, M, d]
        time_vec = self.time_delta_emb(user_lifelong_times[:, :self.top_m]) # [B, M, d]
        
        combined_hist = torch.cat([hist_vec, time_vec], dim=-1) # [B, M, 2*d]
        combined_cand = torch.cat([cand_vec, torch.zeros_like(cand_vec)], dim=-1) # [B, 2*d]
        
        # 精细 Target-Attention 输出
        return self.attention(combined_cand, combined_hist)
```
</details>
"""

# Replace in Chinese file
old_part_start = zh_content.find("### 2. 陈旧事件噪声与生命周期治理")
if old_part_start != -1:
    zh_content = zh_content[:old_part_start] + pseudocode_section_zh + "\n\n### 3. 陈旧事件噪声与生命周期治理" + zh_content[old_part_start + len("### 2. 陈旧事件噪声与生命周期治理"):]
    with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md", "w", encoding="utf-8") as f:
        f.write(zh_content)
    print("Updated Chinese MLCoding07 with Module 3 collapsable pseudocode")

# Read English file
with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.en.md", "r", encoding="utf-8") as f:
    en_content = f.read()

pseudocode_section_en = r"""### 2. Five Long-Sequence Architectures PyTorch Pseudocode (Collapsible)

<details class="exercise" open>
<summary><span class="q-label">Arch 1 · Pseudocode</span> <span class="q-text">Truncated Self-Attention (SASRec / BST Truncated Transformer)</span></summary>

```python
import torch
import torch.nn as nn

class SASRecTruncatedTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, num_layers=2, max_len=50, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, seq_embeddings, mask=None):
        B, N, d = seq_embeddings.shape
        positions = torch.arange(N, device=seq_embeddings.device).unsqueeze(0).expand(B, N)
        x = self.layer_norm(seq_embeddings + self.pos_emb(positions))
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        padding_mask = (~mask) if mask is not None else None
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return out[:, -1, :] # context representation of latest action [B, d]
```
</details>

<details class="exercise">
<summary><span class="q-label">Arch 2 · Pseudocode</span> <span class="q-text">Compressive Memory Networks (MIMN Neural Slot Memory Matrix)</span></summary>

```python
class MIMNMemoryNetwork(nn.Module):
    def __init__(self, embed_dim=64, num_slots=8):
        super().__init__()
        self.num_slots = num_slots # compressed into C=8 slots
        self.write_gate = nn.Linear(embed_dim * 2, num_slots)
        self.erase_gate = nn.Linear(embed_dim, embed_dim)
        self.add_gate = nn.Linear(embed_dim, embed_dim)

    def forward(self, user_history_stream, initial_memory=None):
        B, L, d = user_history_stream.shape
        M = initial_memory if initial_memory is not None else torch.zeros(B, self.num_slots, d, device=user_history_stream.device)
        for t in range(L):
            xt = user_history_stream[:, t, :]
            w = torch.softmax(self.write_gate(torch.cat([xt, M.mean(dim=1)], dim=-1)), dim=-1) # [B, C]
            erase = torch.sigmoid(self.erase_gate(xt)).unsqueeze(1)
            add = torch.tanh(self.add_gate(xt)).unsqueeze(1)
            M = M * (1.0 - w.unsqueeze(-1) * erase) + (w.unsqueeze(-1) * add)
        return M # Slot matrix [B, C, d], read online with O(1) complexity
```
</details>

<details class="exercise">
<summary><span class="q-label">Arch 3 · Pseudocode</span> <span class="q-text">Lifelong Target-Attention (DIN Full Sequence Target Attention)</span></summary>

```python
class DINFullTargetAttention(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.activation_unit = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, candidate_query, lifelong_history, mask=None):
        B, L, d = lifelong_history.shape
        q_exp = candidate_query.unsqueeze(1).expand(B, L, d)
        interaction = torch.cat([q_exp, lifelong_history, q_exp - lifelong_history, q_exp * lifelong_history], dim=-1)
        scores = self.activation_unit(interaction).squeeze(-1) # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask, 0.0)
        return torch.bmm(scores.unsqueeze(1), lifelong_history).squeeze(1) # [B, d]
```
</details>

<details class="exercise">
<summary><span class="q-label">Arch 4 · Pseudocode</span> <span class="q-text">Hierarchical Multi-Resolution Pooling (HPMN Multi-Scale Decay)</span></summary>

```python
class HPMNHierarchicalPooling(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.decay_lambda = nn.Parameter(torch.tensor([0.05]))

    def forward(self, session_seq, daily_seq, monthly_seq, time_deltas_daily, time_deltas_monthly):
        h_session = session_seq.mean(dim=1)
        decay_daily = torch.exp(-torch.clamp(self.decay_lambda, min=1e-4) * time_deltas_daily).unsqueeze(-1)
        h_daily = (daily_seq * decay_daily).sum(dim=1) / (decay_daily.sum(dim=1) + 1e-6)
        decay_monthly = torch.exp(-torch.clamp(self.decay_lambda, min=1e-4) * time_deltas_monthly).unsqueeze(-1)
        h_monthly = (monthly_seq * decay_monthly).sum(dim=1) / (decay_monthly.sum(dim=1) + 1e-6)
        return torch.cat([h_session, h_daily, h_monthly], dim=-1) # [B, 3*d]
```
</details>

<details class="exercise">
<summary><span class="q-label">Arch 5 · Pseudocode</span> <span class="q-text">Retrieval-Augmented Lifelong History (SIM Hard/Soft Search)</span></summary>

```python
class SIMRetrievalAugmentedModel(nn.Module):
    def __init__(self, embed_dim=64, top_m=50):
        super().__init__()
        self.top_m = top_m
        self.time_delta_emb = nn.Embedding(100, embed_dim)
        self.attention = DINFullTargetAttention(embed_dim * 2, hidden_dim=64)

    def forward(self, cand_item_id, cand_category_id, user_lifelong_ids, user_lifelong_cats, user_lifelong_times, item_embed_table):
        # Stage 1: Hard Search category filtering
        match_mask = (user_lifelong_cats == cand_category_id.unsqueeze(1))
        # Stage 2: Target-Attention with time-delta embeddings
        cand_vec = item_embed_table(cand_item_id)
        hist_vec = item_embed_table(user_lifelong_ids[:, :self.top_m])
        time_vec = self.time_delta_emb(user_lifelong_times[:, :self.top_m])
        combined_hist = torch.cat([hist_vec, time_vec], dim=-1)
        combined_cand = torch.cat([cand_vec, torch.zeros_like(cand_vec)], dim=-1)
        return self.attention(combined_cand, combined_hist)
```
</details>
"""

old_part_start_en = en_content.find("## Module 4: E-Commerce Generative Reranking Pipeline")
if old_part_start_en != -1:
    en_content = en_content[:old_part_start_en] + pseudocode_section_en + "\n\n" + en_content[old_part_start_en:]
    with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.en.md", "w", encoding="utf-8") as f:
        f.write(en_content)
    print("Updated English MLCoding07 with Module 3 collapsable pseudocode")

