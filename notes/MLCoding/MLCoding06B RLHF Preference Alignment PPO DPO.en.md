# 06B · RLHF & Preference Alignment: PPO to DPO

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

$$P(y_w \succ y_l \mid x) = \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) = \frac{1}{1 + e^{-(r_\psi(x, y_w) - r_\psi(x, y_l))}}$$

#### Reward Model Objective (Binary Ranking Loss)
Given preference dataset $\mathcal{D} = \{(x, y_w, y_l)\}$, the reward model $r_\psi$ is trained by minimizing negative log-likelihood:

$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]$$

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

### 3. Generalized Advantage Estimation (GAE) and the Bias-Variance Tradeoff

In policy gradient algorithms, the variance of the gradient estimator directly determines training stability and sample efficiency. GAE (Schulman et al., 2015) introduces exponential decay weighting to construct a smooth interpolation between single-step Temporal Difference (TD) and full-trajectory Monte Carlo returns.

#### 1. Temporal Difference Error and $k$-Step Advantage Estimators

Let $V_\phi(s)$ denote the learned value network. The single-step Temporal Difference (TD) error is defined as:
$$\delta_t^V = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

Expanding across different time horizons yields $k$-step advantage estimators:
$$\hat{A}_t^{(1)} = \delta_t^V = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
$$\hat{A}_t^{(2)} = \delta_t^V + \gamma \delta_{t+1}^V = R_t + \gamma R_{t+1} + \gamma^2 V_\phi(s_{t+2}) - V_\phi(s_t)$$
$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l R_{t+l} + \gamma^k V_\phi(s_{t+k}) - V_\phi(s_t)$$
$$\hat{A}_t^{(\infty)} = \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = \sum_{l=0}^\infty \gamma^l R_{t+l} - V_\phi(s_t)$$

#### 2. GAE Exponential Weighting and Recurrence

GAE is defined as the exponentially weighted average of all $k$-step advantage estimators parameterized by $\lambda \in [0, 1]$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat{A}_t^{(k)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

For a finite trajectory of length $T$, GAE satisfies a backward recursive formulation with $O(T)$ complexity, enabling efficient parallel reverse scans on GPUs:
$$\hat{A}_t^{\text{GAE}} = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}^{\text{GAE}}$$

#### 3. The Bias-Variance Tradeoff Governed by $\lambda$

The hyperparameter $\lambda \in [0, 1]$ directly arbitrates between empirical environment sampling variance and value network modeling bias:

- **$\lambda = 0$ (Single-Step TD Limit / Low Variance, High Bias)**:
  $$\hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t^V = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
  - **Minimal Variance**: Relies strictly on the immediate reward and a single state transition, accumulating zero future exploration noise;
  - **High Bias**: Estimates are entirely bounded by the accuracy of the critic network $V_\phi$. If the critic is under-trained or systematically shifted, that estimation error is 100% transmitted into policy gradient updates, inducing persistent drift.
- **$\lambda = 1$ (Full Monte Carlo Limit / High Variance, Zero Bias)**:
  $$\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^\infty \gamma^l R_{t+l} - V_\phi(s_t)$$
  - **Zero Theoretical Bias**: Returns reflect actual sampled environment rollouts; subtracting state baseline $V_\phi(s_t)$ does not alter the mathematical expectation of policy gradients ($\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot V(s)] = 0$);
  - **Severe Variance**: Compounding stochasticity across prolonged actions and transitions causes gradient variance to explode, demanding massive batch sizes to converge stably.
- **$\lambda \in (0, 1)$ (Production Sweet Spot)**:
  - The geometric decay factor $(\gamma \lambda)^l$ assigns dominant weights to near-term empirical returns while exponentially suppressing far-future noise;
  - Accepts modest, bounded critic bias in exchange for orders-of-magnitude variance reduction. Production LLM RLHF commonly sets $\gamma = 1.0, \lambda \in [0.95, 0.98]$.

---

### 4. Trust Region Policy Optimization (TRPO) Foundations

In Vanilla Policy Gradients, parameters are updated directly along the Euclidean gradient direction: $\theta_{\text{new}} = \theta_{\text{old}} + \alpha \nabla_\theta J(\theta)$. This framework suffers from a fatal step-size dilemma: minute learning rates cause training stagnation, while excessive step sizes thrust parameters into poor policy regimes, generating corrupt rollout trajectories that cause irreversible **Policy Collapse**. TRPO (Schulman et al., 2015) established rigorous monotonic improvement guarantees to resolve this issue.

#### 1. What TRPO Optimizes: Surrogate Objective and Monotonic Improvement Theorem

Let $\eta(\pi) = \mathbb{E}_{\tau \sim \pi}[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)]$ denote the expected return. By the Kakade & Langford policy improvement identity:
$$\eta(\pi) = \eta(\pi_{\text{old}}) + \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t A^{\pi_{\text{old}}}(s_t, a_t) \right] = \eta(\pi_{\text{old}}) + \sum_s \rho_\pi(s) \sum_a \pi(a \mid s) A^{\pi_{\text{old}}}(s, a)$$

Because the state visitation frequency $\rho_\pi(s)$ of the unexecuted new policy is inaccessible prior to rollout, TRPO substitutes it with the old state distribution $\rho_{\pi_{\text{old}}}(s)$, forming the **Surrogate Objective**:
$$L_{\pi_{\text{old}}}(\pi) = \eta(\pi_{\text{old}}) + \sum_s \rho_{\pi_{\text{old}}}(s) \sum_a \pi(a \mid s) A^{\pi_{\text{old}}}(s, a) = \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi(a \mid s)}{\pi_{\text{old}}(a \mid s)} A^{\pi_{\text{old}}}(s, a) \right]$$

Schulman et al. proved a guaranteed theoretical lower bound:
$$\eta(\pi) \ge L_{\pi_{\text{old}}}(\pi) - C \cdot D_{\text{KL}}^{\max}(\pi_{\text{old}}, \pi), \quad \text{where } C = \frac{4 \epsilon \gamma}{(1 - \gamma)^2}, \quad \epsilon = \max_{s, a} |A^{\pi_{\text{old}}}(s, a)|$$

**Theoretical Guarantee**: As long as the KL divergence between $\pi_{\text{old}}$ and $\pi$ is strictly bounded, maximizing the surrogate objective $L_{\pi_{\text{old}}}(\pi)$ guarantees that true expected return $\eta(\pi)$ improves monotonically.

#### 2. Why a KL Trust Region Is Necessary: Parameter Space vs. Probability Manifold

Standard gradient updates evaluate step size via Euclidean distance $\|\Delta \theta\|_2$ in parameter space. However, neural network parameter space is highly non-Euclidean with respect to the generated action distribution manifold:
- In regions of high curvature, infinitesimal parameter shifts $\|\Delta \theta\|_2 < 10^{-4}$ can trigger catastrophic flips in action logits;
- Once a destructive update damages the policy, the agent samples corrupted rollouts, unable to escape local degradation.

Hence, trust region constraints must be enforced on the **probability distribution manifold**. TRPO formulates this as a hard constrained optimization problem over state-averaged KL divergence:
$$\max_\theta \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A^{\pi_{\theta_{\text{old}}}}(s, a) \right] \quad \text{s.t.} \quad \bar{D}_{\text{KL}}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta) \le \delta$$

#### 3. How TRPO Is Approximated: Taylor Expansion, Fisher Information Matrix, and Conjugate Gradients (CG)

Because this constrained optimization problem lacks an exact closed form, TRPO computes a local Taylor expansion around $\theta_{\text{old}}$:
1. **First-order expansion of the objective**:
   $$L(\theta) \approx L(\theta_{\text{old}}) + g^T (\theta - \theta_{\text{old}}), \quad g = \nabla_\theta L(\theta)\big|_{\theta = \theta_{\text{old}}}$$
2. **Second-order expansion of the KL constraint**:
   $$\bar{D}_{\text{KL}}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta) \approx \frac{1}{2} (\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}})$$
   (Because the KL divergence is zero and minimal at $\theta = \theta_{\text{old}}$, its first derivative vanishes; its Hessian is the Fisher Information Matrix $F$):
   $$F = \mathbb{E}_{s \sim \rho, a \sim \pi} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)^T \right]$$

Applying Lagrangian duality yields the **Natural Policy Gradient** update direction:
$$\Delta \theta = \theta - \theta_{\text{old}} = \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g$$

##### Scalable Approximation Mechanics
With millions to billions of parameters $d$, explicitly forming $F \in \mathbb{R}^{d \times d}$ and inverting it ($O(d^3)$) is computationally impossible. TRPO utilizes two engineering techniques:
- **Conjugate Gradient (CG) Algorithm**: Directly solves $F x = g$ without computing $F^{-1}$. CG iterates in Krylov subspaces and only evaluates Fisher-vector products ($F v$). By identity $F v = \nabla_\theta \left( (\nabla_\theta \bar{D}_{\text{KL}})^T v \right)$, each matrix-vector product requires just two backpropagation passes, converging to high accuracy in 10–20 iterations;
- **Backtracking Line Search**: Due to higher-order Taylor truncation error, TRPO evaluates exponentially decayed steps $\theta = \theta_{\text{old}} + \alpha^j \Delta \theta$ ($\alpha \in (0, 1)$), verifying that the unapproximated objective improves ($L(\theta) \ge L(\theta_{\text{old}})$) and strictly honors the exact unexpanded KL constraint ($\bar{D}_{\text{KL}} \le \delta$).

---

### 5. Proximal Policy Optimization (PPO-Clip) and Lower-Bound Clipping

While TRPO guarantees monotonic improvement, second-order Fisher calculations, conjugate gradients, and line searches resist distributed parallelism and cannot natively leverage first-order adaptive optimizers like Adam/AdamW. PPO (Schulman et al., 2017) resolves this by formulating a purely first-order differentiable clipped surrogate objective.

#### 1. Simplifications Introduced by PPO over TRPO

1. **Second-order to pure first-order optimization**: Eliminates Fisher matrix computations, CG iterations, and line searches, executing standard backpropagation with Adam/AdamW;
2. **Hard constraints to differentiable clipping**: Replaces constrained Lagrangian optimization with an inline piecewise clipped objective;
3. **Multi-epoch minibatch reuse**: While TRPO typically updates parameters only once per rollout batch, PPO's clipped ratio protects against policy drift, enabling multi-epoch minibatch SGD updates on the same rollout data and drastically increasing sample efficiency.

#### 2. The PPO-Clip Objective and Pessimistic Lower Bound

Define the importance sampling probability ratio:
$$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$$

The PPO-Clip objective is formulated as:
$$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

The outer $\min$ constructs a conservative **Pessimistic Lower Bound**, demonstrating asymmetric gating depending on the sign of $\hat{A}_t$:

- **Positive Advantage ($\hat{A}_t > 0$, action outperforms baseline, encourage probability increase)**:
  $$\min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) = \min\left( r_t \hat{A}_t, \, (1+\epsilon)\hat{A}_t \right)$$
  - When $r_t \le 1+\epsilon$: Objective equals $r_t \hat{A}_t$, pushing $\pi_\theta(a \mid s)$ upwards with positive gradient;
  - When $r_t > 1+\epsilon$: Objective is plateaued to constant $(1+\epsilon)\hat{A}_t$, dropping gradient to zero;
  - **Mechanism**: **Prevents excessive reward amplification**. Prohibits the policy from taking overly aggressive steps on favorable rollouts, avoiding policy collapse.
- **Negative Advantage ($\hat{A}_t < 0$, action underperforms baseline, suppress probability)**:
  - Because $\hat{A}_t$ is negative, inequalities invert:
  $$\min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) = \min\left( r_t \hat{A}_t, \, (1-\epsilon)\hat{A}_t \right)$$
  - When $r_t \ge 1-\epsilon$: Objective equals $r_t \hat{A}_t$, delivering negative gradient to suppress poor actions;
  - When $r_t < 1-\epsilon$: The smaller term is $(1-\epsilon)\hat{A}_t$ due to negative multiplication, freezing gradients to zero;
  - **Mechanism**: **Prevents excessive penalty collapses**. Avoids driving probabilities to zero abruptly, preserving exploration entropy and numerical stability.
- **Why taking $\min$ is strictly necessary**:
  - Without $\min$, relying solely on $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t$ would erroneously inflate the objective if an action worsened ($r_t < 1-\epsilon$ when $\hat{A}_t > 0$);
  - Taking $\min$ ensures that whenever an update underperforms unclipped expectations, the lower (pessimistic) estimate bounds optimization.

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

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot \mid x)} \left[ r(x, y) \right] - \beta \mathbb{D}_{\text{KL}}(\pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))$$

Using calculus of variations, the optimal policy $\pi^*$ has an exact analytical solution:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$ is the partition function.

#### Step 2: Inverting for the Implicit Reward Function
Taking the natural logarithm and rearranging yields:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

#### Step 3: Substitution into Bradley-Terry Model (Partition Function Cancels Out)
Substitute the implicit reward formulation into the Bradley-Terry preference probability:

$$P(y_w \succ y_l \mid x) = \sigma\left( r(x, y_w) - r(x, y_l) \right)$$

$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right)$$

**Crucial Insight**: The intractable partition function $\beta \log Z(x)$ cancels out cleanly!

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

#### Step 4: The Closed-Form DPO Loss
Parameterizing the optimal policy with $\pi_\theta$, we obtain the DPO objective:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

---

### 2. DPO Dynamic Gradient Weighting

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \mathbb{E} \left[ \underbrace{\sigma\left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \right)}_{\text{Dynamic Weight } w(x, y_w, y_l)} \cdot \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right) \right]$$

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
| **SimPO** | Length-normalized average log-prob with margin | $-\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l) - \gamma \right)$ | **None (1 model)** | **SOTA on AlpacaEval 2.0**, reference-free & inherently mitigates verbosity |

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

### Q3: What is the core optimization objective of TRPO? Why is a KL trust region required, and how is it approximated via Conjugate Gradients?
> **Answer**:
> 1. **Optimization Objective**: Optimizes the Surrogate Objective derived from the Kakade-Langford policy improvement identity:
>    $$L_{\pi_{\text{old}}}(\pi) = \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi(a \mid s)}{\pi_{\text{old}}(a \mid s)} A^{\pi_{\text{old}}}(s, a) \right]$$
>    By the theoretical monotonic lower bound $\eta(\pi) \ge L_{\pi_{\text{old}}}(\pi) - C \cdot D_{\text{KL}}^{\max}(\pi_{\text{old}}, \pi)$, keeping the KL divergence between old and new policies small guarantees monotonic improvement in true expected return;
> 2. **Why a KL Trust Region Is Necessary**:
>    - Standard policy gradients enforce Euclidean step sizes in parameter space. However, neural network parameter space is highly non-linear with respect to output probability distributions; minuscule parameter steps along steep directions can trigger catastrophic distribution collapses;
>    - Once a policy collapses, all subsequent rollout samples become degenerate noise, preventing the policy from recovering;
>    - Enforcing hard KL constraints $\bar{D}_{\text{KL}}(\pi_{\text{old}} \parallel \pi) \le \delta$ bounds updates directly on the probability manifold;
> 3. **How It Is Scalably Approximated**:
>    - First-order Taylor expands the objective ($g = \nabla_\theta L$) and second-order expands the KL constraint (Hessian equals Fisher Information Matrix $F$);
>    - Yields the Natural Policy Gradient step $\Delta \theta \propto F^{-1} g$;
>    - To avoid the intractable $O(d^3)$ inversion of $F$, **Conjugate Gradients (CG)** solves $F x = g$ iteratively using Fisher-vector products ($F v$) computed via two backward passes;
>    - A **Backtracking Line Search** then confirms actual objective gain and validates that the exact, unapproximated KL constraint holds.

### Q4: What key simplifications does PPO make compared to TRPO? How does the PPO-Clip objective operate mathematically and intuitively?
> **Answer**:
> 1. **Key Simplifications over TRPO**:
>    - **Second-Order to Pure First-Order**: Eliminates Fisher Information Matrix construction, Conjugate Gradient iterations, and line searches, executing standard backpropagation with Adam/AdamW;
>    - **Hard Constraints to Clipped Surrogate**: Replaces constrained optimization with an unconstrained, piecewise-differentiable objective;
>    - **High Sample Reuse**: TRPO typically permits only one update per batch of experience; PPO uses the clipped importance sampling ratio to safely perform multiple epochs of minibatch SGD updates on the same rollout data, dramatically improving throughput and sample efficiency;
> 2. **PPO-Clip Mechanics (Pessimistic Lower Bound)**:
>    - Objective: $\mathcal{L}_{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$, where $r_t = \frac{\pi_\theta}{\pi_{\text{old}}}$;
>    - **Favorable Actions ($\hat{A}_t > 0$)**: When $r_t$ exceeds $1+\epsilon$, it is clipped to $(1+\epsilon)\hat{A}_t$ with zero gradient, **preventing over-incentivizing already positive actions** and preserving policy stability;
>    - **Unfavorable Actions ($\hat{A}_t < 0$)**: When $r_t$ falls below $1-\epsilon$, negative multiplication causes $\min$ to select the clipped term $(1-\epsilon)\hat{A}_t$ with zero gradient, **preventing catastrophic penalty collapses** that crush exploration entropy;
>    - The outer $\min$ guarantees a conservative pessimistic bound across all off-policy deviations.

### Q5: How does the hyperparameter $\lambda$ in Generalized Advantage Estimation (GAE) balance bias and variance? What do $\lambda=0$ and $\lambda=1$ correspond to?
> **Answer**:
> 1. **Tradeoff Principle**: GAE is defined as $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$, representing an exponentially weighted average over all $k$-step advantage estimates parameterized by $\lambda \in [0, 1]$;
> 2. **$\lambda = 0$ (Single-Step TD Limit)**:
>    - $\hat{A}_t = \delta_t^V = R_t + \gamma V(s_{t+1}) - V(s_t)$;
>    - **Minimal Variance**: Only incorporates single-step environment reward and transition noise;
>    - **High Bias**: Completely reliant on value network $V(s_{t+1})$ accuracy. Any critic approximation error is fully transmitted into the policy gradient;
> 3. **$\lambda = 1$ (Full Monte Carlo Limit)**:
>    - $\hat{A}_t = \sum_{l=0}^\infty \gamma^l R_{t+l} - V(s_t)$;
>    - **Zero Theoretical Bias**: Measures empirical cumulative returns directly from environment rollouts; subtracting baseline $V(s_t)$ leaves the gradient expectation unbiased;
>    - **Extreme Variance**: Compounding stochasticity across long trajectories leads to noisy gradient estimates that destabilize training;
> 4. **Production Practice**: $\lambda \in (0, 1)$ (such as $\lambda = 0.95$ in LLM RLHF) exponentially dampens long-term sampling noise, trading minor critic modeling bias for substantial variance reduction.
