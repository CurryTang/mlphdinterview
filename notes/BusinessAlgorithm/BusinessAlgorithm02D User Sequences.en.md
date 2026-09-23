# User Behavior Sequences

## Chapter 12: User Behavior Sequences

A behavior sequence is not better simply because it is longer, and feeding every log event into a Transformer does not finish the modeling problem. The model estimates the user's current state: which interests remain active, whether a recent action changed intent, and which part of history matters to the current candidate. Storage and latency limit how it can make that estimate.

The first choices happen before the model. An accidental tap, autoplay, and an explicit save should not have equal weight. Watching ten items from one creator may not provide ten independent pieces of evidence. Behavior strength, deduplication, session boundaries, time gaps, and negative feedback often affect the result before another network layer does.

Production systems commonly keep two user representations. A general user embedding is computed once per request and works well for retrieval or a large candidate set. A candidate-conditioned representation queries history with the current item and captures finer intent, but repeats work for every candidate. DIN, SIM, and longer-sequence models choose different points on this quality-cost tradeoff.

### 12.1 What Does Average Pooling Lose?

A user has viewed basketball, cooking, music, and travel content. Averaging all item embeddings yields a fuzzy "overall interest," but it doesn't know which part of the history is relevant to the current candidate, nor does it account for temporal order.

Sequence models primarily solve three things:

- Different behaviors have different weights;
- The current candidate needs to read different parts of the history;
- Interests evolve over time.

### 12.2 Last-N

The simplest approach takes the last N behaviors. It is inexpensive and often stronger than complex models suggest.

One can add:

- Behavioral type weights;
- Time decay;
- Deduplication and continuous playback compression;
- Effective view thresholds;
- Category or author grouping.

For Last-N, bigger is not always better. Long histories introduce noise, storage, and service costs, and may re-amplify old interests.

### 12.3 DIN (Deep Interest Network)

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

---

### 12.4B SASRec (Self-Attentive Sequential Recommendation, 2018)

DIN uses candidate-driven local attention (Target Attention) to activate relevant historical subsets. However, **no interaction occurs between historical items themselves**, making it impossible to capture strict temporal transitions and causal progressions (e.g., "user bought a phone $\to$ bought a phone case $\to$ now needs a charging cable").

**SASRec** first introduced Transformer causal self-attention to sequential recommendation:

```math
\mathbf{S} = \operatorname{Self-Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{M}\right)\mathbf{V}
```

#### 1. Core Architectural Innovations
1. **Causal Triangular Mask ($\mathbf{M}$)**:
   - Sequential recommendation is an autoregressive task. When predicting intent at step $t$, the model **must not access future actions at step $t+1$ and beyond**;
   - The causal attention mask is defined as an upper-triangular negative infinity matrix:
     $$M_{ij} = \begin{cases} 0, & i \ge j \\ -\infty, & i < j \end{cases}$$
2. **Positional Embeddings**:
   - To encode event order without recurrent bottlenecks, learnable positional embeddings $\mathbf{P} \in \mathbb{R}^{L \times d}$ are added to item vectors:
     $$\hat{\mathbf{E}} = [\mathbf{e}_1 + \mathbf{p}_1, \mathbf{e}_2 + \mathbf{p}_2, \dots, \mathbf{e}_L + \mathbf{p}_L]$$
3. **Pointwise Feed-Forward Network (FFN)**:
   - Each self-attention block is followed by two dense layers with ReLU activation, residual connections, and LayerNorm.
4. **DIN vs. SASRec: Fundamental Systems Difference**:
   - **DIN (Ranking-Only)**: The user representation depends on a specific candidate item $v_{\text{cand}}$. Forward compute scales linearly with candidate set size $O(C \times L)$, restricting it to fine ranking over a small candidate set;
   - **SASRec (Dual-Use for Retrieval & Ranking)**: The final hidden state $\mathbf{h}_L$ at the last sequence position encodes the user's complete dynamic intent. Because $\mathbf{h}_L$ is **candidate-independent**, it can be pushed directly into an ANN index for millisecond-scale generative sequential retrieval, or passed as a dense dynamic feature to fine ranking.

#### Pseudocode: SASRec Causal Forward Implementation
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
        # causal_mask: [seq_len, seq_len], upper triangular True mask
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
        
        # 1. Sum item embeddings and learnable position embeddings
        x = self.item_embeddings(seq_ids) + self.pos_embeddings(positions)
        
        # 2. Construct upper-triangular causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq_ids.device), diagonal=1).bool()
        
        # 3. Stack causal self-attention blocks
        for block in self.blocks:
            x = block(x, causal_mask)
            
        final_seq = self.final_norm(x) # [batch_size, seq_len, hidden_dim]
        # Last step state serves as the universal user sequence embedding
        user_repr = final_seq[:, -1, :] # [batch_size, hidden_dim]
        return user_repr
```

---

### 12.5 Temporal Issues in Training

The most dangerous bug in sequence models is time leakage.

If a sample occurs at time `t`, the history can only use behaviors visible before `t`. Statistical features, profiles, and item popularity must also be cut off at `t`. If one reads the full history of the current user offline, the metrics will be unrealistically good.

One must also handle:

- Label leakage within the same session;
- Repeated exposure;
- Out-of-order behavior logs;
- Delayed arrival;
- Negative feedback and ineffective views;
- Inconsistencies between training truncation and online truncation.

### 12.6 Real-Time Updates

The value of sequence models often comes from the most recent behaviors. If a user just finished watching a skiing video, and the feature service only updates five minutes later, no matter how complex the model is, it cannot react.

Common practices include:

- Offline storage for long-term sequences;
- Streaming updates for short-term behaviors;
- Online concatenation and deduplication;
- Degradation for missing or late behaviors;
- Recording feature versions for replay.

Model parameters can also update incrementally: train a full model on a complete window overnight, then consume fresh logs for small hourly updates. This shortens response time to interest shifts but introduces delayed labels, catastrophic forgetting, and rapid propagation of bad data. Serving must be able to fall back to the latest full checkpoint and track full and incremental data versions separately.

### 12.7 Chapter Self-Test

1. What information does Last-N average pooling lose?
2. Why does DIN's user representation depend on the candidate?
3. Why does SIM use a two-stage process for long sequences?
4. How can one check if sequence features have time leakage?
5. How to degrade when online short-term behavior updates fail?

<details>
<summary>Reference answers</summary>

1. Mean pooling loses order, time gaps, repetition, and interest transitions.
2. DIN uses the candidate as the attention query, so the same user receives a different representation for each candidate.
3. SIM cheaply retrieves a candidate-relevant subsequence from long history, then models that shorter sequence in detail.
4. Verify every event precedes the request and replay features with a point-in-time join; aggregation windows must not include future events.
5. Fall back to an older versioned sequence or long-term features, record the degradation rate, and do not disguise missing data as a genuine empty history.

</details>
