# ML Coding 08 · Industrial Long-Sequence Modeling & E-Commerce Generative Reranking

In modern recommender systems and e-commerce search, **Long-Sequence User Behavior Modeling** and **Generative Reranking** represent two critical frontier directions—capturing lifelong latent preferences on the input side and optimizing joint slate utility on the output side.

This note systematically covers two major modules:
1. **The 5 Core Long-Sequence Architectural Paradigms (Truncated Transformers, Compressive Memory Networks, Lifelong Target Attention, Hierarchical Pooling & Two-Stage Retrieval SIM/ETA)**
2. **Compute/Memory/Latency Tradeoffs, Stale Noise Governance & SLA Decision Tree**
3. **E-Commerce Generative Reranking Pipeline (Candidate Slot Serialization, Constrained Beam Search, Listwise Objectives, P99 ≤ 20ms Latency Budgeting & Evaluation Metrics)**

---

## Module 1: Industrial Long-Sequence Modeling Paradigms

```text
The 5-Dimensional Sequence Modeling Spectrum:
1. Truncated Transformers (SASRec/BST): Last N=50~100, O(N²) attention (amnesia)
2. Compressive Memory (MIMN): Fixed slot matrix M ∈ R^(C×d), O(1) read (lossy compression)
3. Lifelong Target Attention (DIN): Candidate query over full L items (O(K·L) explodes)
4. Hierarchical Pooling (HPMN): Session ➔ Daily ➔ Monthly pyramidal pooling
5. Retrieval-Augmented Histories (SIM/ETA): Two-stage Hard/Soft search ➔ Exact Attention (SOTA)
```

### Tradeoff Matrix
- **SIM Hard Search**: $O(K \cdot M \cdot d)$ complexity, fetches only Top-50 IDs, $5 \sim 8	ext{ms}$ P99 latency, scales to **$L \ge 50,000+$** actions.

---

## Module 2: E-Commerce Generative Reranking Pipeline

```text
End-to-End Generative Reranking Pipeline:
[Precision Ranking Top-50 Output]
       │
       ▼
[1. Candidate Serialization: Slot Tokens <C_01> ~ <C_50> + User Prefix]
       │
       ▼
[2. Constrained Beam Search Generator (6-Layer Decoder-Only, Masked Softmax)]
       │
       ▼
[3. Final Display Slate: π = [π₁, π₂, ..., π_K] (K = 6~10)]
```

### 1. Inherent Flaws of Pointwise Rankers
- Cannibalization across identical top items;
- Inability to build cross-category complementary bundles ("Phone + Case + Earbuds");
- Disrupted price anchoring psychology.

### 2. Training Objectives & Production Latency Budgeting
- **Plackett-Luce Loss**: $\mathcal{L}_{	ext{List}} = -\sum_{k=1}^K \log \left( rac{\exp(s_{\pi_k})}{\sum_{j=k}^K \exp(s_{\pi_j})} ight)$
- **Slate Reward RL**: $R(\pi) = \sum \gamma^{k-1}(	ext{Click}_k \cdot 	ext{Margin}_k + 	ext{GMV}_k) - \lambda \cdot 	ext{Redundancy}(\pi)$
- **P99 ≤ 20ms SLA**: Prefix KV Cache sharing across beam steps + 18ms hard timeout fallback to precision ranking order.
