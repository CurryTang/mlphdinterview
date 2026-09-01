import os

en_content = """# Multi-Objective Ranking: Negative Transfer Mechanisms, Architectural Evolution & Seesaw Mitigation

In modern industrial recommender systems and computational advertising, systems must simultaneously balance multiple competing business objectives. For instance, short-video platforms balance **click-through rate (pCTR), long-view/completion rate (pLongView), social interaction rate (pInteract), and negative feedback rate (pDislike)**; e-commerce rankers balance **pCTR, add-to-cart (pCart), purchase conversion rate (pCVR), and gross merchandise value (pGMV)**.

When multiple objectives are trained jointly over shared representations, engineers frequently observe the **Seesaw Effect (Negative Transfer)**—where optimizing Task A directly causes significant performance degradation in Task B.

This note systematically covers 3 foundational pillars of multi-objective ranking:
1. **Root Causes of Shared Representation Conflicts (Gradient Conflicts, Magnitude Dominance & Sample Selection Bias)**
2. **Horizontal Comparison of 6 Mainstream Multi-Task Architectures & Optimization Strategies (Shared-Bottom, MMoE, PLE, Uncertainty Weighting, PCGrad & Constrained Fusion)**
3. **Offline Diagnostic Ablation Framework & Real-Time Online Guardrail Circuit Breakers**

---

## Module 1: Why Targets Conflict in Shared Representations (The Seesaw Effect)

```text
The 3 Root Causes of Multi-Objective Representation Conflicts:
┌────────────────────────────────────────────────────────────────────────┐
│ Root Cause 1: Gradient Directional Conflicts (cos(g_A, g_B) < 0)       │
│ • Task A (CTR) pushes parameters toward clickbait/catchy headlines     │
│ • Task B (Completion) pushes parameters toward long-form depth         │
│ • The shared representation is torn apart, hurting both objectives.    │
├────────────────────────────────────────────────────────────────────────┤
│ Root Cause 2: Gradient Magnitude & Frequency Imbalance                 │
│ • High-frequency task (CTR ~ 5% positive) gradient norm ≫ CVR (~ 0.1%) │
│ • Shared layers are dominated by CTR; rare conversion tasks starve.    │
├────────────────────────────────────────────────────────────────────────┤
│ Root Cause 3: Sample Space Mismatch & Selection Bias                   │
│ • CTR is trained on the entire impression space D_imp                  │
│ • CVR is traditionally trained only on clicked samples D_click         │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Directional Gradient Conflicts
When loss functions $\mathcal{L}_A$ and $\mathcal{L}_B$ compute gradients with respect to shared parameters $W_{\text{shared}}$:

$$\mathbf{g}_A = \nabla_{W_{\text{shared}}} \mathcal{L}_A, \quad \mathbf{g}_B = \nabla_{W_{\text{shared}}} \mathcal{L}_B$$

If $\cos(\mathbf{g}_A, \mathbf{g}_B) < 0$ ($\mathbf{g}_A \cdot \mathbf{g}_B < 0$), the gradients point in conflicting directions. Parameter updates $\Delta W \propto -(\mathbf{g}_A + \mathbf{g}_B)$ will inevitably increase the loss of at least one task.

### 2. Frequency & Magnitude Domination
High-frequency interactions generate orders of magnitude more gradient updates than rare conversion events. Without explicit normalization, the shared embedding and bottom MLP parameters are hijacked by the high-frequency task, leading to under-fitting on critical sparse objectives.

---

## Module 2: Comparison of 6 Core Architectures & Optimization Strategies

```text
The 3-Dimensional Solution Matrix:
┌───────────────────────────────────┬───────────────────────────────────┐
│ 1. Architectural Feature Routing  │ Shared-Bottom ➔ MMoE ➔ PLE (SOTA) │
├───────────────────────────────────┼───────────────────────────────────┤
│ 2. Optimization & Loss Balancing  │ Uncertainty Weighting ➔ PCGrad    │
├───────────────────────────────────┼───────────────────────────────────┤
│ 3. Score Fusion & Decision Layer  │ Static Weights ➔ Constrained PID  │
└───────────────────────────────────┴───────────────────────────────────┘
```

### 1. Architectural Routing (Shared-Bottom vs. MMoE vs. PLE)

| Architecture | Routing Mechanism | Negative Transfer Defense | Pros | Cons / Limitations |
|---|---|---|---|---|
| **Shared-Bottom** | Single shared bottom MLP; branches into task towers at the top. | **Very Poor** (Zero isolation) | Minimal compute and parameter footprint. | Severe gradient conflicts; worst seesaw effect. |
| **MMoE<br>(Multi-gate MoE)** | Shared pool of experts; dedicated softmax gating per task: $g_t(x) = \text{Softmax}(W_t x)$. | **Moderate** (Partial mitigation) | Dynamic sample-level soft routing across experts. | **All experts remain globally shared**; weakly correlated tasks still fight over capacity. |
| **PLE<br>(Progressive Extraction)** | Explicitly separates **Task-Specific Experts** from **Shared Experts** with multi-level extraction. | **SOTA** (Eliminates negative transfer) | **Physical isolation of private vs shared representations**; completely eliminates negative transfer. | ~15-25% higher parameter and FLOPs overhead. |

---

### 2. Optimization & Gradient Balancing (Uncertainty Weighting & PCGrad)

#### (1) Homoscedastic Uncertainty Weighting (Kendall et al.)
Replaces arbitrary manual loss weights with learned task-dependent uncertainty parameters $\sigma_k^2$:

$$\mathcal{L}_{\text{total}} = \sum_{k=1}^K \left( \frac{1}{2\sigma_k^2} \mathcal{L}_k + \ln \sigma_k \right)$$

- **Mechanism**: Automatically attenuates the loss weight $\frac{1}{2\sigma_k^2}$ of noisy or high-variance tasks while the logarithmic barrier $\ln \sigma_k$ prevents weights from collapsing to zero.

#### (2) Projecting Conflicting Gradients (PCGrad)
When gradients conflict ($\mathbf{g}_i \cdot \mathbf{g}_j < 0$), project $\mathbf{g}_i$ onto the normal plane of $\mathbf{g}_j$:

$$\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j$$

- **Effect**: Eliminates the destructive component acting against Task $j$ while preserving Task $i$'s original trajectory.

---

### 3. Decision-Level Score Fusion & Pareto Constrained Optimization
Instead of brittle static formulas ($S = pCTR \times pCVR^\alpha$), production systems formulate score fusion as a constrained optimization problem:

$$\max \mathbb{E}[\text{GMV}] \quad \text{s.t.} \quad \text{CTR} \ge \text{CTR}_0, \quad \text{Dislike Rate} \le \tau$$

Online servers run real-time PID controllers that continuously adjust Lagrangian multipliers $\lambda(t)$ every 5 minutes to track ecosystem guardrails dynamically.

---

## Module 3: Offline Diagnostic Ablation Matrix & Online Guardrail Architecture

```text
Diagnostic and Production Defense Loop:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Offline Diagnostic Ablations                                        │
│ • Single-Task Upper Bounds: Train K dedicated independent models       │
│ • Gradient Cosine Matrix: Track percentage of negative cos(g_i, g_j)   │
│ • Task GAUC Delta Matrix: ΔGAUC_k = GAUC_MTL(k) - GAUC_Single(k)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Online Guardrails & Adaptive Throttling                             │
│ • Real-time PID Fusion Control: Dynamically adjusts weights per minute │
│ • Automated Circuit Breaker: Drops to baseline if guardrails regress   │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Offline Ablation Protocol
1. **Single-Task Upper Bounds**: Train independent, unshared models for each task to establish theoretical performance ceilings $\text{GAUC}_{\text{Single}, k}$;
2. **Relative Delta Matrix**: Measure $\Delta \text{GAUC}_k = \text{GAUC}_{\text{MTL}, k} - \text{GAUC}_{\text{Single}, k}$. A pattern where Task A is $+0.008$ while Task B is $-0.006$ directly confirms severe negative transfer;
3. **Gradient Conflict Monitoring**: Compute pairwise cosine similarities across training iterations; if negative steps exceed $30\%$, transition to PLE or PCGrad;
4. **Subgroup Slice Consistency**: Audit metrics across new vs. active users to catch localized seesaw degradation.

### 2. Online Guardrail Circuit Breakers
- **PID-Controlled Feedback Loop**: Dynamically adapts fusion weights based on real-time sliding window telemetry;
- **Automated Canary Circuit Breaker**: If online negative feedback rate rises $>5\%$ or GMV drops $>2\%$ over 3 consecutive monitoring windows, the canary bucket automatically aborts and falls back to the baseline control model.
"""

with open("notes/BusinessAlgorithm/BusinessAlgorithm02B Multi-Objective Ranking.en.md", "w", encoding="utf-8") as f:
    f.write(en_content)
print("Successfully updated BusinessAlgorithm02B English note")
