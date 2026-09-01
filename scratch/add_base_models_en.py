import os

# 1. Update BusinessAlgorithm02C.en.md
with open("notes/BusinessAlgorithm/BusinessAlgorithm02C Feature Interaction.en.md", "r", encoding="utf-8") as f:
    ba02c_en = f.read()

dcn_dlrm_en = r"""### 11.3 DCN & DCN-v2 (Deep & Cross Network)

The standard Cross Layer in DCN is defined as:

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l$$

In **DCN-v2**, low-rank matrix decomposition ($\mathbf{W}_l = \mathbf{U}_l \mathbf{V}_l^T$) and MoE-style subspace routing are introduced to reduce compute while capturing high-order feature combinations:

#### Pseudocode: DCN-v2 Cross Network Forward Pass
```python
import torch
import torch.nn as nn

class CrossNetworkV2(nn.Module):
    def __init__(self, in_features, num_layers=3, low_rank=32, num_experts=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        # Low-rank decomposition: W_l = U_l @ V_l.T
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
        xl = x0
        for l in range(self.num_layers):
            gate = torch.softmax(self.gate_list[l](xl), dim=-1).unsqueeze(-1) # [B, E, 1]
            xl_expanded = xl.unsqueeze(1) # [B, 1, d]
            v = self.v_list[l] # [E, d, r]
            u = self.u_list[l] # [E, d, r]
            low_rank_out = torch.einsum('bmd,edr->ber', xl_expanded, v) # [B, E, r]
            expert_out = torch.einsum('ber,edr->bed', low_rank_out, u)   # [B, E, d]
            moe_out = (expert_out * gate).sum(dim=1) # [B, d]
            xl = x0 * (moe_out + self.bias_list[l]) + xl
        return xl
```

---

### 11.4 DLRM (Deep Learning Recommendation Model)

Meta's DLRM architecture explicitly decouples dense continuous features from sparse categorical embeddings:
1. **Bottom MLP**: Transforms continuous dense features into a $d$-dimensional vector;
2. **Embedding Tables**: Maps sparse categorical IDs into $d$-dimensional embeddings;
3. **Explicit Dot-Product Triu Interaction**: Computes pairwise inner products across all $d$-dim vectors;
4. **Top MLP**: Concatenates interaction dot-products with the original bottom representation to generate probabilities.

#### Pseudocode: DLRM Architecture Forward Pass
```python
class DLRM(nn.Module):
    def __init__(self, embedding_sizes, dense_in_dim, embed_dim=64):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim) for num_classes in embedding_sizes
        ])
        self.bottom_mlp = nn.Sequential(
            nn.Linear(dense_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        num_fields = len(embedding_sizes) + 1
        num_interactions = (num_fields * (num_fields - 1)) // 2
        self.top_mlp = nn.Sequential(
            nn.Linear(num_interactions + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, dense_x, sparse_x):
        v_dense = self.bottom_mlp(dense_x).unsqueeze(1)
        v_sparse = [emb(sparse_x[:, i]).unsqueeze(1) for i, emb in enumerate(self.embeddings)]
        all_embeddings = torch.cat([v_dense] + v_sparse, dim=1) # [B, num_fields, d]
        dot_interactions = torch.bmm(all_embeddings, all_embeddings.transpose(1, 2))
        triu_indices = torch.triu_indices(all_embeddings.size(1), all_embeddings.size(1), offset=1)
        flat_interactions = dot_interactions[:, triu_indices[0], triu_indices[1]]
        top_input = torch.cat([flat_interactions, v_dense.squeeze(1)], dim=-1)
        return self.top_mlp(top_input)
```
"""

if "class CrossNetworkV2" not in ba02c_en:
    dcn_pos_en = ba02c_en.find("### 11.3 DCN")
    if dcn_pos_en != -1:
        ba02c_en = ba02c_en[:dcn_pos_en] + dcn_dlrm_en + "\n\n### 11.5 LHUC, SENet, and FiBiNET\n" + ba02c_en[ba02c_en.find("### 11.4 LHUC") + len("### 11.4 LHUC"):]
        with open("notes/BusinessAlgorithm/BusinessAlgorithm02C Feature Interaction.en.md", "w", encoding="utf-8") as f:
            f.write(ba02c_en)
        print("Updated BusinessAlgorithm02C.en.md")

# 2. Update BusinessAlgorithm02D.en.md
with open("notes/BusinessAlgorithm/BusinessAlgorithm02D User Sequences.en.md", "r", encoding="utf-8") as f:
    ba02d_en = f.read()

din_sim_en = r"""### 12.3 DIN (Deep Interest Network)

DIN introduces **Target Attention** to compute dynamic user interest representation $\mathbf{u}(\mathbf{q})$ conditioned on candidate query $\mathbf{q}$:

$$\alpha_j = \text{MLP}([\mathbf{h}_j, \mathbf{q}, \mathbf{h}_j - \mathbf{q}, \mathbf{h}_j \odot \mathbf{q}]), \quad \mathbf{u}(\mathbf{q}) = \sum_{j=1}^L \alpha_j \mathbf{h}_j$$

#### Pseudocode: DIN Target-Attention Forward Pass
```python
import torch
import torch.nn as nn

class DINAttention(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query, history, mask=None):
        B, L, d = history.shape
        q_expanded = query.unsqueeze(1).expand(B, L, d)
        interaction = torch.cat([
            q_expanded, history, q_expanded - history, q_expanded * history
        ], dim=-1) # [B, L, 4*d]
        scores = self.mlp(interaction).squeeze(-1) # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask, 0.0)
        user_interest = torch.bmm(scores.unsqueeze(1), history).squeeze(1) # [B, d]
        return user_interest
```

---

### 12.4 SIM (Search-based Interest Model: Hard & Soft Search)

For lifelong user sequences ($L \ge 10,000$), SIM employs a two-stage decoupled search architecture:
1. **Hard Search**: Fast sub-sequence retrieval filtering top-$M$ ($M \approx 50$) category-matched historical events;
2. **Soft Attention**: Fine-grained Target Attention combined with time-delta embeddings $\Delta t$.

#### Pseudocode: SIM Long-Sequence Forward Pass
```python
class SIMSequenceModel(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.time_delta_emb = nn.Embedding(100, embed_dim)
        self.attention = DINAttention(embed_dim * 2, hidden_dim=64)

    def forward(self, cand_id, cand_cat, user_hist_ids, user_hist_cats, user_hist_times, item_embed_table):
        cand_vec = item_embed_table(cand_id) # [B, d]
        # Stage 1: Hard Search extracts top-50 items matching cand_cat
        hist_vec = item_embed_table(user_hist_ids[:, :50]) # [B, 50, d]
        time_vec = self.time_delta_emb(user_hist_times[:, :50]) # [B, 50, d]
        combined_hist = torch.cat([hist_vec, time_vec], dim=-1) # [B, 50, 2*d]
        combined_cand = torch.cat([cand_vec, torch.zeros_like(cand_vec)], dim=-1) # [B, 2*d]
        # Stage 2: Target-Attention
        return self.attention(combined_cand, combined_hist)
```
"""

if "class DINAttention" not in ba02d_en:
    din_pos_en = ba02d_en.find("### 12.3 DIN")
    if din_pos_en != -1:
        ba02d_en = ba02d_en[:din_pos_en] + din_sim_en + "\n\n### 12.5 Time Issues in Training\n" + ba02d_en[ba02d_en.find("### 12.5 Time Issues in Training") + len("### 12.5 Time Issues in Training"):]
        with open("notes/BusinessAlgorithm/BusinessAlgorithm02D User Sequences.en.md", "w", encoding="utf-8") as f:
            f.write(ba02d_en)
        print("Updated BusinessAlgorithm02D.en.md")

# 3. Update BusinessAlgorithm02B.en.md
with open("notes/BusinessAlgorithm/BusinessAlgorithm02B Multi-Objective Ranking.en.md", "r", encoding="utf-8") as f:
    ba02b_en = f.read()

mmoe_ple_en = r"""### 10.3 MMoE (Multi-gate Mixture-of-Experts)

MMoE deploys dedicated Softmax gating networks per task over shared expert layers:

$$\mathbf{h}_t(\mathbf{x}) = \sum_{e=1}^E g_{t,e}(\mathbf{x}) f_e(\mathbf{x}), \quad \mathbf{g}_t(\mathbf{x}) = \text{Softmax}(\mathbf{W}_t \mathbf{x})$$

#### Pseudocode: MMoE Architecture Forward Pass
```python
import torch
import torch.nn as nn

class MMoE(nn.Module):
    def __init__(self, in_features, num_experts=4, expert_dim=64, num_tasks=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, expert_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.task_gates = nn.ModuleList([
            nn.Linear(in_features, num_experts) for _ in range(num_tasks)
        ])
        self.task_towers = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1) # [B, E, d]
        task_predictions = []
        for t in range(self.num_tasks):
            gate_weights = torch.softmax(self.task_gates[t](x), dim=-1).unsqueeze(-1)
            task_rep = (expert_outputs * gate_weights).sum(dim=1)
            task_pred = self.task_towers[t](task_rep)
            task_predictions.append(task_pred)
        return task_predictions
```

---

### 10.4 PLE (Progressive Layered Extraction)

PLE physically decouples **Task-Specific Experts** from **Shared Experts**, eliminating negative transfer across weakly correlated objectives:

#### Pseudocode: PLE Extraction Layer Forward Pass
```python
class PLECustomExtractionLayer(nn.Module):
    def __init__(self, in_features, num_task_experts=2, num_shared_experts=2, expert_dim=64, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_experts = nn.ModuleList([
            nn.ModuleList([nn.Linear(in_features, expert_dim) for _ in range(num_task_experts)])
            for _ in range(num_tasks)
        ])
        self.shared_experts = nn.ModuleList([
            nn.Linear(in_features, expert_dim) for _ in range(num_shared_experts)
        ])
        total_task_experts = num_task_experts + num_shared_experts
        self.task_gates = nn.ModuleList([
            nn.Linear(in_features, total_task_experts) for _ in range(num_tasks)
        ])
        total_all_experts = num_task_experts * num_tasks + num_shared_experts
        self.shared_gate = nn.Linear(in_features, total_all_experts)

    def forward(self, task_inputs, shared_input):
        task_exp_outs = [[exp(task_inputs[t]) for exp in self.task_experts[t]] for t in range(self.num_tasks)]
        shared_exp_outs = [exp(shared_input) for exp in self.shared_experts]
        
        # 1. Task-Specific Routing
        task_next_inputs = []
        for t in range(self.num_tasks):
            pool = torch.stack(task_exp_outs[t] + shared_exp_outs, dim=1)
            gate = torch.softmax(self.task_gates[t](task_inputs[t]), dim=-1).unsqueeze(-1)
            task_next_inputs.append((pool * gate).sum(dim=1))
            
        # 2. Shared Global Routing
        all_pool = torch.stack([item for sublist in task_exp_outs for item in sublist] + shared_exp_outs, dim=1)
        shared_gate = torch.softmax(self.shared_gate(shared_input), dim=-1).unsqueeze(-1)
        shared_next_input = (all_pool * shared_gate).sum(dim=1)
        
        return task_next_inputs, shared_next_input
```
"""

if "class PLECustomExtractionLayer" not in ba02b_en:
    mmoe_pos_en = ba02b_en.find("### 10.3 MMoE")
    if mmoe_pos_en != -1:
        ba02b_en = ba02b_en[:mmoe_pos_en] + mmoe_ple_en + "\n\n### 10.5 ESMM and Conversion Funnels\n" + ba02b_en[ba02b_en.find("### 10.4 ESMM") + len("### 10.4 ESMM"):]
        with open("notes/BusinessAlgorithm/BusinessAlgorithm02B Multi-Objective Ranking.en.md", "w", encoding="utf-8") as f:
            f.write(ba02b_en)
        print("Updated BusinessAlgorithm02B.en.md")

