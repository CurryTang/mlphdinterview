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

### 12.5 Temporal Representation Evolution: From Monotonic Decay to the Interest Clock (Circadian Rhythm)

In recommendation domains like short-video feeds and news aggregators, user consumption intent is profoundly governed by **circadian rhythms and physiological schedules**. The "Interest Clock" replaces simple monotonic time decay with a dual-track framework: **Trend Decay + Periodic Resonance**.

#### 1. Pathology of Monotonic Time Distance Decay

Most classical sequential recommendation models (such as early DIN, BST, and SIM's Soft Search) calculate a unidirectional time delta:

$$\Delta t = t_{\text{target}} - t_{\text{action}}$$

which is discretized via logarithmic bucketing: $\mathbf{e}_{\Delta t} = \text{Embedding}(\text{Bucket}(\log(1 + \Delta t)))$.

This introduces a flawed assumption: **as physical elapsed time increases, historical relevance to the current query must strictly decrease**. In real-world feeds, this assumption regularly breaks down:
- **Circadian Intent Disconnection**: A user's deep engagement with "sleep-aid ambient audio / late-night emotional radio" at 23:30 last night is far more relevant to tonight at 23:30 than to the "morning breaking financial news" casually browsed during their 09:00 morning commute.
- **Over-dilution of High-Value History**: Monotonic $\Delta t$ decay severely suppresses last night's deep interest due to a 14-hour physical time gap, preventing the nighttime model from reproducing circadian intent.

#### 2. Dual-Scale Disentangled Representation

The Interest Clock disentangles temporal dynamics across two complementary physical dimensions:

```text
[Dual-Track Temporal Representation Architecture]

                     ┌── 1. Absolute Wall-Clock Slot: Learns intrinsic mindset across 0:00~23:59
Timestamp Projection ┤
                     └── 2. Circular Clock Delta: Measures shortest circular arc on a 24-hour dial
```

- **Dimension 1: Absolute Wall-Clock Time**
  Projects Unix timestamps onto a closed 24-hour cycle (0:00 ~ 23:59):
  - **Continuous Harmonic Basis**:
    $$e_{\text{sin}} = \sin\left(2\pi \cdot \frac{\text{minute\_of\_day}}{1440}\right), \quad e_{\text{cos}} = \cos\left(2\pi \cdot \frac{\text{minute\_of\_day}}{1440}\right)$$
    Eliminates the Euclidean discontinuity between 23:59 and 00:01, ensuring topological smoothness.
  - **Discrete Clock Slot Embedding**:
    Production recommendation engines typically bucket the day into fixed intervals, such as 96 slots (15 minutes per slot) or 48 slots (30 minutes per slot), learning a dense embedding $\mathbf{e}_{\text{clock\_slot}} \in \mathbb{R}^d$ per slot.

- **Dimension 2: Circular Clock Delta**
  Computes the **shortest circular arc distance** between the historical event time and the current target request time on the 24-hour dial:
  $$\Delta \tau_i = \min\left(\vert{}h_{\text{target}} - h_i\vert{},\; 24 - \vert{}h_{\text{target}} - h_i\vert{}\right)$$
  If the target request occurs at 01:00 AM ($h_{\text{target}} = 1$) and a historical interaction happened at 23:00 PM yesterday ($h_i = 23$), the physical delta is $\Delta t = 26 \text{ hours}$, yet the circular phase difference is only $\Delta \tau = 2 \text{ hours}$. This triggers a strong "same-time-window interest resonance".

#### 3. Fusion Paradigm: Embedding Concatenation vs. Attention Bias

- **Embedding-Level Fusion (Input-side Coarse Conditioning)**:
  $$\mathbf{x}_i = \mathbf{e}_{\text{item}} + \mathbf{e}_{\text{action}} + \mathbf{e}_{\text{clock\_slot}} + \mathbf{e}_{\Delta t}$$
  While straightforward, passing side embeddings through subsequent dense layers relies on implicit interaction and cannot guarantee attention weights directly scale with temporal proximity.
- **Attention Bias-Level Intervention (Attention-side Explicit Steering, Production SOTA)**:
  Directly injects a learnable circular clock bias into the Query-Key attention matrix:
  $$\text{Score}(Q, K_i) = \frac{Q K_i^T}{\sqrt{d}} - \lambda \cdot \log(1 + \Delta t_i) + \text{Bias}_{\text{circ}}(\Delta \tau_i)$$
  This formulation unites physical damping with periodic resonance: $-\lambda \log(1 + \Delta t)$ handles long-term interest drift, while $+\text{Bias}_{\text{circ}}$ elevates past high-quality items whenever the circadian phase aligns ($\Delta \tau \to 0$).

#### 4. Production PyTorch Implementation: InterestClockAttention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterestClockAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_clock_slots: int = 96):
        """
        num_clock_slots: Number of bins across 24 hours (96 slots = 15 mins per slot)
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_clock_slots = num_clock_slots

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 1. Absolute periodic clock slot embedding (0 ~ num_clock_slots - 1)
        self.clock_slot_embed = nn.Embedding(num_clock_slots, d_model)

        # 2. Relative circular delta attention bias table (max arc = num_clock_slots // 2)
        max_circ_diff = num_clock_slots // 2 + 1
        self.circ_bias_table = nn.Embedding(max_circ_diff, num_heads)

    def _extract_clock_slots(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Extracts 24h slot indices from Unix timestamps (0 ~ num_clock_slots - 1)"""
        seconds_in_day = timestamps % 86400
        slot_duration = 86400 / self.num_clock_slots
        return (seconds_in_day / slot_duration).long()

    def _calc_circular_delta(self, target_slots: torch.Tensor, hist_slots: torch.Tensor) -> torch.Tensor:
        """Calculates shortest circular arc distance on the 24h dial"""
        abs_diff = torch.abs(target_slots - hist_slots)  # [B, 1, L]
        circ_diff = torch.minimum(abs_diff, self.num_clock_slots - abs_diff)
        return circ_diff

    def forward(
        self,
        target_item: torch.Tensor,       # [B, 1, d_model]
        target_ts: torch.Tensor,         # [B, 1] (Unix timestamp)
        hist_items: torch.Tensor,        # [B, L, d_model]
        hist_ts: torch.Tensor,           # [B, L] (Unix timestamp)
        mask: torch.Tensor = None        # [B, 1, L] (True indicates valid position)
    ) -> torch.Tensor:
        B, L, _ = hist_items.shape

        # 1. Extract slot indices
        tgt_slot = self._extract_clock_slots(target_ts)       # [B, 1]
        hist_slot = self._extract_clock_slots(hist_ts)        # [B, L]

        # 2. Superimpose absolute clock embeddings at input layer
        tgt_input = target_item + self.clock_slot_embed(tgt_slot)
        hist_input = hist_items + self.clock_slot_embed(hist_slot)

        # 3. Linear projections and multi-head reshaping
        Q = self.q_proj(tgt_input).view(B, 1, self.num_heads, self.d_k).transpose(1, 2)   # [B, H, 1, d_k]
        K = self.k_proj(hist_input).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        V = self.v_proj(hist_input).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]

        # 4. Scaled dot-product attention score
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, 1, L]

        # 5. Inject circular clock attention bias
        circ_diff = self._calc_circular_delta(tgt_slot.unsqueeze(2), hist_slot.unsqueeze(1))  # [B, 1, L]
        clock_bias = self.circ_bias_table(circ_diff.squeeze(1))  # [B, L, H]
        clock_bias = clock_bias.permute(0, 2, 1).unsqueeze(2)    # [B, H, 1, L]

        scores = scores + clock_bias

        # 6. Padding mask and weighted aggregation
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, L]
        out = torch.matmul(attn_weights, V)       # [B, H, 1, d_k]
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        return out
```

#### 5. Industrial Systems Engineering Details

1. **Local Timezone Normalization**:
   Raw timestamps in logs are typically UTC. Slot computation must translate UTC into the **user's local wall-clock time** based on client IP/GPS or timezone offsets; otherwise, identical UTC timestamps between London (midnight) and Beijing (morning) yield inverted circadian phases.
2. **Hierarchical Clocks (Hour + Weekday)**:
   Mature industrial architectures combine two nested cycles:
   - **Hour Clock (24h period)**: Captures diurnal morning-commute, midday-break, and late-night sleep-aid habits.
   - **Weekday Clock (7d period)**: Captures shifts between weekday goal-driven consumption and weekend leisure browsing. Biases from both tiers are linearly summed into the attention logits.
3. **Zero-Latency Serving Optimization (Pre-quantization & In-memory Feature Store)**:
   Target slot is computed once per request. Historical item slots are pre-quantized in streaming ETL (Flink/Kafka) upon event arrival and stored as lightweight uint8 attributes in the Feature Store. Online inference reduces to integer subtraction and array lookups, introducing virtually zero P99 latency overhead.

---

### 12.6 Temporal Issues in Training

The most dangerous bug in sequence models is time leakage.

If a sample occurs at time `t`, the history can only use behaviors visible before `t`. Statistical features, profiles, and item popularity must also be cut off at `t`. If one reads the full history of the current user offline, the metrics will be unrealistically good.

One must also handle:

- Label leakage within the same session;
- Repeated exposure;
- Out-of-order behavior logs;
- Delayed arrival;
- Negative feedback and ineffective views;
- Inconsistencies between training truncation and online truncation.

---

### 12.7 Real-Time Updates

The value of sequence models often comes from the most recent behaviors. If a user just finished watching a skiing video, and the feature service only updates five minutes later, no matter how complex the model is, it cannot react.

Common practices include:

- Offline storage for long-term sequences;
- Streaming updates for short-term behaviors;
- Online concatenation and deduplication;
- Degradation for missing or late behaviors;
- Recording feature versions for replay.

Model parameters can also update incrementally: train a full model on a complete window overnight, then consume fresh logs for small hourly updates. This shortens response time to interest shifts but introduces delayed labels, catastrophic forgetting, and rapid propagation of bad data. Serving must be able to fall back to the latest full checkpoint and track full and incremental data versions separately.

---

### 12.8 Chapter Self-Test

1. What information does Last-N average pooling lose?
2. Why does DIN's user representation depend on the candidate?
3. Why does SIM use a two-stage process for long sequences?
4. What is the fundamental flaw of monotonic time decay ($\Delta t$) in modeling circadian rhythms, and how does the Interest Clock resolve it?
5. How can one check if sequence features have time leakage?
6. How to degrade when online short-term behavior updates fail?

<details>
<summary>Reference answers</summary>

1. Mean pooling loses order, time gaps, repetition, and interest transitions.
2. DIN uses the candidate as the attention query, so the same user receives a different representation for each candidate.
3. SIM cheaply retrieves a candidate-relevant subsequence from long history, then models that shorter sequence in detail.
4. Monotonic $\Delta t$ strongly assumes relevance decays monotonically over time, over-diluting valuable same-period historical interests (e.g. sleep-aid audio late at night); the Interest Clock decouples absolute wall-clock slots from circular clock phase differences ($\Delta \tau$), injecting periodic resonance bias into attention logits to retrieve matching-phase historical items.
5. Verify every event precedes the request and replay features with a point-in-time join; aggregation windows must not include future events.
6. Fall back to an older versioned sequence or long-term features, record the degradation rate, and do not disguise missing data as a genuine empty history.

</details>
