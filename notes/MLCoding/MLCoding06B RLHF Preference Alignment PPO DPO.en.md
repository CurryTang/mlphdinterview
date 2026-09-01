# ML Coding 06B · RLHF & Preference Alignment: From Reward Modeling & PPO 4-Model System to DPO/IPO/KTO/SimPO & Alignment Tax

In the modern large language model (LLM) lifecycle, pre-training provides vast world knowledge and next-token generation capability, but the base model remains an unaligned completion engine. To transform it into a helpful, honest, and harmless assistant, **Post-Training Preference Alignment** is the definitive cornerstone of production LLM engineering.

This guide provides a comprehensive, mathematically rigorous breakdown across 5 key pillars:
1. **The Classical 3-Stage RLHF Pipeline (SFT $\to$ Reward Modeling $\to$ PPO Policy Optimization)**
2. **PPO 4-Model Concurrent Architecture & Generalized Advantage Estimation (GAE)**
3. **Direct Preference Optimization (DPO): Closed-Form Implicit Reward Derivation & Gradient Dynamics**
4. **The Modern Alignment Family Taxonomy (DPO vs IPO vs KTO vs ORPO vs SimPO)**
5. **Production Alignment Traps & Mitigations (Reward Hacking, Verbosity Bias, Over-Refusal & Alignment Tax)**

---

## Module 1: The Classical 3-Stage RLHF Pipeline

```text
The 3-Stage RLHF Workflow:
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Supervised Fine-Tuning (SFT)                                  │
│ • Data: High-quality curated instruction-response pairs (Prompt, Resp) │
│ • Goal: Impart basic instruction-following and dialogue formatting     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Reward Modeling (RM)                                          │
│ • Data: Multiple model candidate responses ranked by humans (y_w ≻ y_l)│
│ • Goal: Train scalar scoring model r_ψ(x, y) approximating human taste │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Reinforcement Learning Policy Optimization (PPO)              │
│ • Mechanism: Policy generates -> RM scores -> KL penalty & GAE advantage│
│ • Goal: Iteratively update policy weights to maximize human preference │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Stage 2: Bradley-Terry Preference Modeling & Reward Loss

Humans find it difficult to assign absolute, calibrated scalar scores (e.g. 8.7/10) to open-ended text, but excel at **pairwise comparisons** ($y_w \succ y_l$).

#### Bradley-Terry Preference Probability
Given prompt $x$, human-preferred winner $y_w$, and dispreferred loser $y_l$, assuming an underlying latent scalar reward $r^*(x, y)$, the preference probability follows the Bradley-Terry model:

$$P(y_w \succ y_l \mid x) = \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \r\right) = \frac{1}{1 + e^{-(r_\psi(x, y_w) - r_\psi(x, y_l))}}$$

#### Reward Model Objective (Binary Ranking Loss)
Given preference dataset $\mathcal{D} = \{(x, y_w, y_l)\}$, the reward model $r_\psi$ is trained by minimizing negative log-likelihood:

$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \r\right) \r\right]$$

---

## Module 2: PPO 4-Model System Architecture & GAE Advantage

### 1. The 4-Model Concurrent Runtime Topology in PPO

```text
PPO 4-Model Concurrency Topology:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Actor Model (π_θ, Policy):                                          │
│    • Status: Active Training (Full Backprop & Optimizer Updates)       │
│    • Role: Receives Prompt x, generates autoregressive response y      │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Critic / Value Model (V_ϕ):                                         │
│    • Status: Active Training (Full Backprop & Optimizer Updates)       │
│    • Role: Estimates state baseline V(s_t) to compute GAE advantages   │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Reward Model (r_ψ):                                                 │
│    • Status: Frozen (Inference Only)                                   │
│    • Role: Scores full sequence (x, y) with scalar preference value    │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Reference Model (π_ref, SFT Baseline):                              │
│    • Status: Frozen (Inference Only)                                   │
│    • Role: Computes per-token baseline log-probs for KL regularization │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Token-Level Rewards & Dynamic KL Regularization

To prevent the policy $\pi_\theta$ from exploiting blind spots in the reward model (Reward Hacking) or drifting away from natural language, per-token KL penalties are assigned:

$$R_t = \begin{cases} -\beta \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}, & t < T \\ r_\psi(x, y) - \beta \log \frac{\pi_\theta(y_T \mid x, y_{<T})}{\pi_{\text{ref}}(y_T \mid x, y_{<T})}, & t = T \text{ (sequence end)} \end{cases}$$

---

### 3. Generalized Advantage Estimation (GAE)

To balance **variance** and **bias** in value estimation, PPO computes token-level advantage $\hat{A}_t$ via GAE:

1. **Temporal Difference (TD) Error**:
   $$\delta_t^V = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
2. **GAE Exponential Decay Accumulation**:
   $$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V$$

---

### 4. PPO-Clip Objective

$$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$$

$$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \r\right) \r\right]$$

---

## Module 3: Direct Preference Optimization (DPO) Closed-Form Derivation

Rafailov et al. (NeurIPS 2023) revolutionized LLM alignment with **DPO**: by reparameterizing the latent reward as a closed-form function of the policy's log probabilities, **DPO completely eliminates the need for separate Reward and Critic networks!**

```text
PPO 4-Model RL System vs DPO 2-Model Binary Classification:
┌────────────────────────────────────────────────────────┐
│ PPO: Actor + Critic + Reward + Reference (4 Models)    │
│ Pipeline: Sampling -> Reward -> GAE -> Policy Clipping │
└───────────────────────────┬────────────────────────────┘
                            ▼ Radical Simplification
┌────────────────────────────────────────────────────────┐
│ DPO: Trained Policy π_θ + Frozen Reference π_ref (2 M) │
│ Pipeline: Offline closed-form cross-entropy loss       │
└────────────────────────────────────────────────────────┘
```

### 1. The Step-by-Step Mathematical Derivation

#### Step 1: Analytical Optimal Policy for KL-Regularized RL
The standard KL-regularized RL objective is:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot \mid x)} \left[ r(x, y) \r\right] - \beta \mathbb{D}_{\text{KL}}(\pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))$$

Using calculus of variations, the optimal policy $\pi^*$ has an exact analytical solution:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \r\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \r\right)$ is the partition function.

#### Step 2: Inverting for the Implicit Reward Function
Taking the natural logarithm and rearranging yields:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

#### Step 3: Substitution into Bradley-Terry Model (Partition Function Cancels Out)
Substitute the implicit reward formulation into the Bradley-Terry preference probability:

$$P(y_w \succ y_l \mid x) = \sigma\left( r(x, y_w) - r(x, y_l) \r\right)$$

$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \r\right) - \left( \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \r\right)$$

**Crucial Insight**: The intractable partition function $\beta \log Z(x)$ cancels out cleanly!

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

#### Step 4: The Closed-Form DPO Loss
Parameterizing the optimal policy with $\pi_\theta$, we obtain the DPO objective:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \r\right) \r\right]$$

---

### 2. DPO Dynamic Gradient Weighting

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \mathbb{E} \left[ \underbrace{\sigma\left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \r\right)}_{\text{Dynamic Weight } w(x, y_w, y_l)} \cdot \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \r\right) \r\right]$$

- When the model is severely mistaken ($\hat{r}_\theta(y_w) \ll \hat{r}_\theta(y_l)$), $w \to 1$, applying maximum gradient to push up $y_w$ and down $y_l$;
- When the model has already mastered the preference ($\hat{r}_\theta(y_w) \gg \hat{r}_\theta(y_l)$), $w \to 0$, naturally preventing overfitting.

---

## Module 4: The Alignment Family Taxonomy

| Alignment Paradigm | Key Mechanism & Innovation | Loss Formulation | Reference Model | Production Pros & Best Use Case |
|---|---|---|---|---|
| **PPO** | Actor-Critic RL with GAE Advantage | $\mathbb{E}[\min(r_t A_t, \text{clip} \cdot A_t)]$ | **Required (4 models)** | High online exploration, complex multi-turn dynamic rewards |
| **DPO** | Implicit reward reparameterization | $-\log \sigma(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$ | **Required (2 models)** | **Industry standard for general alignment**, stable, lightweight |
| **IPO** | Quadratic regularizer on log-ratio differences | $(\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} - \frac{1}{2\tau})^2$ | **Required (2 models)** | Prevents DPO overfitting and distribution collapse |
| **KTO** | Grounded in Prospect Theory, binary feedback | Optimize utility on individual inputs independently | **Required (2 models)** | **No paired preference data required**, uses upvote/downvote logs |
| **ORPO** | Monolithic SFT + Odds-Ratio penalty | $\mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{OddsRatio}}$ | **None (1 model)** | Single-stage SFT + alignment with zero reference model overhead |
| **SimPO** | Length-normalized average log-prob with margin | $-\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l) - \gamma \r\right)$ | **None (1 model)** | **SOTA on AlpacaEval 2.0**, reference-free & inherently mitigates verbosity |

---

## Module 5: Production Alignment Pitfalls & LLM-as-a-Judge Evaluation

### 1. The 4 Major Production Alignment Failure Modes

```text
The 4 Major Production Alignment Failure Modes:
┌───────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Pitfall                   │ Mechanism & Production Mitigation                                      │
├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Reward Hacking         │ Policy exploits reward model shortcuts (superficial formatting).       │
│                           │ ➔ Defense: Tight KL budget $\beta$; length penalty in reward modeling. │
├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Verbosity Bias         │ Reward models favor unnecessarily long, verbose answers.               │
│                           │ ➔ Defense: Use SimPO length normalization; balance data length distributions.│
├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. Alignment Tax          │ Degradation of raw reasoning, math, and code capabilities after RLHF.   │
│                           │ ➔ Defense: Replay 10%–20% pre-training / math reasoning data during alignment.│
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. Over-Refusal           │ False positives on benign prompts containing sensitive words (e.g. "kill").│
│                           │ ➔ Defense: Hard-negative boundary training with datasets like XSTest.   │
└───────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 2. LLM-as-a-Judge Evaluation Protocols & Three Inherent Biases

In automated alignment evaluation and preference dataset filtering, frontier LLMs (such as GPT-4o and Claude-3.5-Sonnet) are widely deployed as automated judges (LLM-as-a-Judge) for pairwise comparison and score rubrics. However, evaluator models exhibit three prominent systematic biases:

```text
LLM-as-a-Judge Biases & Mitigation Protocols:
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Bias Category           │ Mechanism & Industrial Mitigation Strategy                             │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Position Bias        │ Tendency to favor Candidate 1 over Candidate 2.                        │
│                         │ ➔ Mitigation: Pairwise position swapping (evaluating both (A,B) and    │
│                         │   (B,A)) and averaging scores.                                         │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Verbosity Bias       │ Tendency to assign higher scores to longer, highly-formatted responses │
│                         │ regardless of factual substance.                                       │
│                         │ ➔ Mitigation: Length-penalized rubrics and strict word-count limits.   │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. Self-Enhancement Bias│ Favoring responses generated by the judge model's own architectural    │
│                         │ family due to shared latent representational preferences.              │
│                         │ ➔ Mitigation: Multi-judge consensus panels and reference-grounded eval.│
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## Module 6: Key Interview FAQs

### Q1: Why does DPO mathematically eliminate the separate Reward Model?
> **Answer**: Under KL-regularized RL, the optimal policy $\pi^*$ and latent reward $r^*(x, y)$ have an exact analytical equivalence: $r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$. When substituted into the Bradley-Terry preference difference, the partition function $Z(x)$ cancels out entirely. Thus, policy log-odds directly represent relative preference probabilities, reducing RL to standard binary classification without any reward network.

### Q2: What is Alignment Tax, and how is it mitigated in production pipelines?
> **Answer**: Alignment Tax refers to the degradation of complex reasoning, math problem-solving, and code generation performance when a model is aggressively optimized for safety/chat preference. It is mitigated by:
> 1. **Data Replay**: Mixing 10%–20% raw pre-training and reasoning SFT data into preference optimization;
> 2. **Decoupled Post-Training**: Adopting a multi-stage paradigm (like DeepSeek-R1) where reasoning RL with verifiable rewards (RLVR) is trained first, followed by a minimal, gentle general alignment stage.
