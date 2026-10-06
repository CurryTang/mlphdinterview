# 06C · RLVR, Reasoning Models & Agentic RL

As generative AI enters the era of "System 2 Slow Thinking" and "Autonomous Agents", traditional human-preference RLHF (e.g. standard PPO and DPO) has reached fundamental scaling limits. The modern frontier—represented by **OpenAI o1 / o3**, **DeepSeek-R1**, and modern **Agentic Tool-Use RL**—shifts the reinforcement learning paradigm from "mimicking human conversational style" to **"Reinforcement Learning with Verifiable Rewards (RLVR)"**.

This guide provides a comprehensive breakdown across 6 key pillars:
1. **The RLVR Paradigm Shift: How Deterministic Rule-Based Verifiers Eradicate Reward Hacking**
2. **DeepSeek-R1 Core Architecture: Group Relative Policy Optimization (GRPO) Mathematical Derivation**
3. **Pure RL Emergence: Long Chain-of-Thought (CoT), Self-Reflection, Backtracking & The "Aha Moment"**
4. **DeepSeek-R1 4-Stage Production Recipe (Cold-Start $\to$ Reasoning RL $\to$ Rejection SFT $\to$ Universal RL)**
5. **On-Policy Distillation (OPD) & On-Policy Self-Distillation (OPSD) Landscape**
6. **Agentic RL: Multi-Turn Environment Sandboxes, Process Reward Models (PRM vs ORM) & MCTS**

---

## Module 1: The RLVR Paradigm Shift—From Subjective Taste to Verifiable Truth

```text
Classical RLHF (Subjective Preference) vs Modern RLVR (Deterministic Verifiable Rewards):
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Classical RLHF (Chat / Creative Writing / Safety):                  │
│    • Reward Source: Neural Reward Model (Subject to approximation bias)│
│    • Bottleneck: Reward Hacking, superficial verbosity, human ceiling  │
│    • Task Domain: Open Q&A, translation, roleplay, summarization       │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Modern RLVR (Math / Coding / Logic / Formal Proofs / Agents):       │
│    • Reward Source: Deterministic external verifiers (Python, Unit Test)│
│    • Advantage: 100% immune to reward hacking, binary truth, self-play │
│    • Task Domain: LeetCode, AIME/MATH, Lean 4 proofs, SQL, Bash tasks  │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Why Are Math & Code the Ultimate Frontier for RLVR?

In conversational tasks, determining which essay is "better" is noisy and subjective. However, in math and coding:
1. **Deterministic Ground-Truth Verifiability**:
   - Algorithms have unit tests with strict time and space complexity constraints;
   - Math problems possess single, exact boxed answers or formal symbolic solutions (SymPy);
   - Formal mathematics uses compilers (Lean 4, Isabelle, Coq) for interactive theorem verification.
2. **Infinite Exploration & Search Space**:
   - The model can discover novel, ingenious solutions that humans have never written;
   - **Completely breaks free from the human ceiling imposed by SFT training datasets.**

---

## Module 2: Group Relative Policy Optimization (GRPO) Mathematical Derivation

In traditional PPO, training on long chains of thought (Long CoT, up to 32K tokens) leads to **memory blowup and training collapse**:
- **Critic Value Network Instability**: Estimating per-token value across tens of thousands of tokens has immense variance, leading to Critic divergence;
- **GPU Memory Footprint**: The Critic network doubles memory consumption and requires retaining all intermediate activations.

DeepSeek's **GRPO** completely **eliminates the Critic value network** by employing **Group Normalized Advantage Estimation**:

```text
PPO 4-Model Overhead vs GRPO Lightweight Grouped Architecture:
┌────────────────────────────────────────────────────────────────────────┐
│ Classical PPO: Actor (Policy) + Critic (Value) + Reward + Reference    │
│ • Massive VRAM footprint; Critic gradients OOM on 32K token CoT        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼ GRPO Revolution
┌────────────────────────────────────────────────────────────────────────┐
│ Modern GRPO: Actor Policy Model + Rule Verifiers / Ref Model           │
│ • Mechanism: Sample G outputs per prompt, normalize advantage in-group │
│ • Benefit: Zero Critic network, >50% VRAM savings, scales to 32K+ CoT  │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 1. GRPO Mathematical Derivation

#### Step 1: Group Sampling
For each query $q$, GRPO samples a group of $G$ distinct candidate outputs from the old policy $\pi_{\theta_{\text{old}}}$:

$$\{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$$

#### Step 2: Rule Scoring & Group-Relative Advantage Normalization
External verifiable rule engines score each candidate response with rewards $\{r_1, r_2, \dots, r_G\}$.

Compute the group mean $\mu_q$ and standard deviation $\sigma_q$:

$$\mu_q = \frac{1}{G} \sum_{i=1}^G r_i, \quad \sigma_q = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \mu_q)^2}$$

The scalar advantage $A_i$ for candidate $o_i$ is computed via group normalization:

$$A_i = \frac{r_i - \mu_q}{\sigma_q + \epsilon}$$

> **Geometric Intuition**: If a candidate outperforms the group mean ($r_i > \mu_q$), $A_i > 0$ and its tokens are reinforced. If below average, it is penalized. **Group baselining inherently neutralizes prompt difficulty variance!**

#### Step 3: GRPO Clipped Objective Function
Broadcasting $A_i$ across all tokens in trajectory $o_i$, the GRPO optimization loss is:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})} A_i, \text{clip}\left( \frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right\}$$

where the per-token unbiased KL estimator follows Schulman's form:

$$\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})_t = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

---

## Module 3: Pure RL Emergence of Long CoT & Self-Reflection

In DeepSeek-R1-Zero's pure RL experiment (zero human SFT data, purely query prompts + GRPO rule rewards), the model **spontaneously exhibited emergent System 2 reasoning behaviors**:

```text
Spontaneous Reasoning Behaviors Under Pure RL:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Self-Correction & Backtracking:                                     │
│    • Output: "Wait, let me double check this step..."                  │
│    • Detecting logical errors in previous lines and branching anew     │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Multi-Hypothesis Exploration:                                       │
│    • Output: "Alternatively, let's consider another approach..."       │
│    • Cross-validating multiple solution paths before finalizing        │
├────────────────────────────────────────────────────────────────────────┤
│ 3. The "Aha Moment":                                                   │
│    • Discovering subtle edge cases after long derivations and pivoting │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Test-Time Compute Scaling:                                          │
│    • Thinking tokens naturally scale with question difficulty (to 32K) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Module 4: DeepSeek-R1 4-Stage Production Recipe

To eliminate pure RL defects (language mixing, poor formatting, excessive loops), DeepSeek-R1 employs a structured 4-stage pipeline:

```text
DeepSeek-R1 4-Stage Training Pipeline:
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Cold-Start Long-CoT SFT                                       │
│ • Data: Thousands of curated, readable, long reasoning demonstrations  │
│ • Goal: Impart clean linguistic formatting and initial reflection      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Reasoning RL (RLVR via GRPO)                                  │
│ • Tasks: Large-scale Math (AIME/MATH), Competitive Coding, Logic       │
│ • Mechanism: Accuracy reward + format compliance reward                │
│ • Goal: Drive deep reasoning, self-correction, and long CoT expansion │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Rejection Sampling & Multi-Task Mixed SFT                     │
│ • Data: High-reward trajectories from Stage 2 + General Chat/Writing   │
│ • Goal: Solidify reasoning while restoring general conversation quality│
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Universal RL for All Scenarios                                │
│ • Mechanism: Rule rewards (for math/code) + Preference Model (safety)  │
│ • Goal: Achieve SOTA reasoning aligned with human values               │
└────────────────────────────────────────────────────────────────────────┘
### 2. Knowledge Distillation Paradigms: White-Box Logits vs. Black-Box Sequence SFT (DeepSeek-R1-Distill)

Once a frontier reasoning model (such as DeepSeek-R1 671B) is trained via large-scale RL, how can its reasoning prowess be transferred into lightweight edge models (e.g., 1.5B / 7B / 8B / 14B / 32B)? Two distillation paradigms exist in production:

| Distillation Mode | Core Mechanism | Pros | Cons / Limitations | Industrial Case Studies |
|---|---|---|---|---|
| **White-Box (Logit-based KD)** | Student aligns its output logit probability distribution with the teacher's via KL divergence:<br>$$\mathcal{L}_{KD} = D_{KL}(P_{\text{teacher}} \parallel P_{\text{student}})$$ | Retains rich "Dark Knowledge" (probability correlations over non-top-1 candidates). | Requires **strictly identical tokenizer vocabulary** between teacher and student; impossible over commercial APIs. | Same-family model compression (e.g., LLaMA-3-70B $\to$ LLaMA-3-8B). |
| **Black-Box (Sequence-Level SFT)** | Teacher generates massive synthetic problem-solving and long-CoT reasoning trajectories; student fine-tunes via standard SFT. | **Zero architectural constraints** (e.g., distilling DeepSeek-R1 into Qwen-2.5 or LLaMA-3); supports commercial API generation. | Loses soft-target distribution information; heavily dependent on deduplication and rigorous verification. | **DeepSeek-R1-Distill-Qwen/LLaMA Series** (800K curated R1 reasoning traces fine-tuning open-source models). |

> 💡 **Core Finding from DeepSeek-R1**:
> Distilling 800K synthetic long-CoT reasoning traces directly into smaller dense models (such as Qwen-2.5-32B) yields competitive performance (DeepSeek-R1-Distill-32B) that significantly outperforms training those smaller models with pure RL from scratch. This demonstrates that **complex reasoning capabilities are highly distillable and transferable across model architectures!**
>
> However, offline static distillation (Off-Policy KD) suffers from fundamental theoretical limitations: the student model is never trained on its own generated prefixes, meaning early inference slips compound quadratically; moreover, forcing small models to fit every teacher mode induces hallucinations. These bottlenecks have driven the modern post-training frontier toward **On-Policy Distillation (OPD)** and **Privileged On-Policy Self-Distillation (OPSD)**.

---

## Module 5: On-Policy Distillation (OPD) & On-Policy Self-Distillation (OPSD) Landscape

```text
Offline Distillation (Off-Policy KD) vs On-Policy Distillation (OPD):
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Classical Offline Distillation (Off-Policy KD):                     │
│    • Data Source: Fixed, static dataset D generated by the teacher     │
│    • Mechanism: Teacher grades text the student would never generate   │
│    • Bottleneck: Student never sees its own mistakes; compounding      │
│                  quadratic errors O(T^2); mode-smearing hallucinations │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Modern On-Policy Distillation (OPD):                                │
│    • Data Source: Student's own real-time rollout y ~ π_θ (On-Policy)  │
│    • Mechanism: Teacher evaluates and scores the student's own trace   │
│                 token-by-token using reverse KL divergence             │
│    • Advantage: Trains error recovery in-distribution; error grows     │
│                 linearly O(T); suppresses catastrophic forgetting      │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. The Distillation Paradigm Shift: Mathematical Formulation of Off-Policy vs On-Policy KD

#### Standard (Off-Policy) Knowledge Distillation
Classical knowledge distillation defines its objective function as the expected forward KL divergence over a pre-collected, static dataset $\mathcal{D}$:

$$\mathcal{L}_{\text{KD}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{D}_{\text{KL}}(\pi_T(\cdot \mid x) \parallel \pi_\theta(\cdot \mid x)) \right]$$

All training sequences are generated in advance by the teacher model $\pi_T$ or taken from fixed human corpora. Throughout training, the student model $\pi_\theta$ **is evaluated exclusively on gold-standard sentences that it would never produce under its own parameter distribution**. As a result, the student never encounters its own intermediate errors during training and never acquires the ability to backtrack or recover from local reasoning flaws.

#### On-Policy Knowledge Distillation (OPD)
On-Policy Distillation transfers sampling control directly to the student policy itself:

$$\mathcal{L}_{\text{OnKD}} = \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot \mid x)} \left[ D(\pi_T(\cdot \mid x, y_{<t}),\, \pi_\theta(\cdot \mid x, y_{<t})) \right]$$

Given a query $x$, the active student model $\pi_\theta$ first samples a complete trajectory $y \sim \pi_\theta(\cdot \mid x)$. The teacher model $\pi_T$ then grades and aligns **that exact sequence produced by the student** token by token. The label "On-Policy" refers strictly to **the data generation policy being the current active student policy $\pi_\theta$**.

---

### 2. Core Mathematical Foundation: KL Divergence Asymmetry & Mode Behavior

The Kullback-Leibler (KL) divergence quantifies the extra information entropy lost when approximating reference distribution $P$ with $Q$:

$$\mathbb{D}_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

KL divergence is fundamentally **asymmetric**. The penalty explodes toward infinity whenever $P(x)$ is large while $Q(x) \approx 0$ (because $\frac{P(x)}{Q(x)} \to \infty$). Conversely, if $P(x) \approx 0$ while $Q(x)$ is large, the penalty is negligible. Thus: **whichever distribution sits in the reference position ($P$) is the one the approximating distribution ($Q$) is mathematically forced to cover**.

```text
Forward KL (Mode-Covering / SFT) vs Reverse KL (Mode-Seeking / RL):
┌────────────────────────────────────────────────────────────────────────┐
│ Forward KL: D_KL(π_T || π_θ)                                           │
│ • Penalty Focus: Student missing any teacher mode (Zero-Avoiding)      │
│ • Behavior: Forced to cover every peak of the teacher (Mode-Covering)  │
│ • Low Capacity Penalty: Small student smears mass into the gaps,       │
│                         producing fluent nonsense and hallucinations   │
├────────────────────────────────────────────────────────────────────────┤
│ Reverse KL: D_KL(π_θ || π_T)                                           │
│ • Penalty Focus: Student producing tokens the teacher hates            │
│ • Behavior: Allowed to drop unrepresentable modes and commit to one    │
│             clean, coherent peak (Mode-Seeking / Zero-Forcing)         │
│ • Optimization Win: Preserves logical consistency and high fidelity    │
└────────────────────────────────────────────────────────────────────────┘
```

| Divergence Metric | Forward KL ($\mathbb{D}_{\text{KL}}(\pi_T \parallel \pi_\theta)$) | Reverse KL ($\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_T)$) |
|---|---|---|
| **Penalty Focus** | Heavily punishes the student for missing teacher modes | Heavily punishes the student for mass where teacher probability is $\approx 0$ |
| **Geometric Behavior** | **Mode-Covering / Zero-Avoiding** | **Mode-Seeking / Zero-Forcing** |
| **Small Student Capacity** | Forced to cover all modes; lacking parameters, it smears mass across gaps $\to$ **Fluent nonsense and hallucinations**. | Drops unrepresentable secondary modes; focuses on one coherent solution $\to$ **High fidelity and rigorous logic**. |
| **Training Parallel** | Corresponds to standard Supervised Fine-Tuning (SFT). | Corresponds to Reinforcement Learning (RL) optimization. |

```kl-divergence-modes-demo
```

---

### 3. Why On-Policy Beats Off-Policy: Three Foundational Theoretical Arguments

#### Argument 1: Catastrophic Forgetting Suppression & RL's Razor
- **The Destructive Jump of SFT**: SFT relies on forward KL to enforce mode coverage across the entire support. This forces large, unconstrained parameter jumps across weight space, altering pre-trained internal representations and triggering severe **catastrophic forgetting**.
- **Local Reweighting in Reverse KL**: Reverse KL evaluated on-policy merely reweights the student's own sampled actions according to teacher scores.
- **RL's Razor**: In policy gradient optimization, each update step represents the **minimal local perturbation (minimal divergence nudge)** necessary to increase expected reward. The policy stays tightly bound to the neighborhood of its initial capability distribution, preserving existing capabilities while acquiring new reasoning skills.

#### Argument 2: Compounding Errors & DAgger Theorem ($\mathcal{O}(T)$ vs $\mathcal{O}(T^2)$)
In long-horizon reasoning (1000+ token chains) and multi-turn agent interaction, off-policy imitation learning is inherently brittle:
- Under classical imitation learning theory (Ross & Bagnell, DAgger Theorem), an off-policy student trained only on teacher prefixes will, upon making a single mistake at step $t$, enter an **Out-of-Distribution (OOD)** state never encountered in training;
- Because it has never practiced error recovery, subsequent errors compound rapidly, and total trajectory error scales **quadratically** with rollout length: $\mathcal{O}(T^2)$;
- **OPD Linear Error Growth**: In on-policy distillation, the student practices directly on **its own generated prefixes**. Intermediate deviations remain in-distribution; the model learns backtracking and error recovery, bounding compounding errors to **linear growth**: $\mathcal{O}(T)$.

#### Argument 3: Dense Feedback vs. Sparse 0/1 Outcome Rewards (Terminal-Bench Case Study)
In complex agentic environments (such as Terminal-Bench shell execution or multi-file coding):
- **The Collapse of Outcome Rewards (ORM)**: A 1000+ token rollout concludes with a single terminal status code (0/1). When the episode fails, credit assignment is impossible: the entire rollout yields zero informative gradient signal;
- **OPD Per-Token Dense Supervision**: Under OPD, the teacher model scores the student's trace token by token, providing **dense targets** across every reasoning step. **A single sparse bit per episode expands into thousands of dense supervisory gradients**, identifying the exact token where reasoning derailed even within failed trajectories.

---

### 4. Algorithmic Landscape & Evolution: MiniLLM, GKD, and DistiLLM

```text
Evolution of Foundational OPD Frameworks:
MiniLLM (2023)                GKD (2023)                     DistiLLM (2024)
┌───────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│ KD is Formulated as RL│ ──► │ Unified 2-Knob Spectrum│ ──► │ Industrial Stability   │
│ • Reverse KL into RL  │     │ • Data dial: λ ∈ [0,1] │     │ • Skew KL for variance │
│ • Teacher log-P reward│     │ • Divergence selector  │     │ • Replay buffer reuse  │
└───────────────────────┘     └────────────────────────┘     └────────────────────────┘
```

#### 1. MiniLLM (Gu et al., 2023) — "Knowledge Distillation is Really RL"
Standard offline data cannot directly optimize reverse KL divergence. MiniLLM formulated reverse KL distillation as reinforcement learning policy gradient optimization:
1. The student generates an on-policy sequence $y \sim \pi_\theta(\cdot \mid x)$;
2. The teacher's log-probability serves as the scalar reward: $R(y) = \log \pi_T(y \mid x)$;
3. Policy gradient updates push the student toward modes favored by the teacher, mathematically allowing the smaller student to prune unrepresentable modes.

#### 2. GKD (Generalized Knowledge Distillation, Agarwal et al., 2023) — "One Dial Spectrum"
GKD unified offline KD and on-policy KD into a single continuous formulation governed by two control knobs:
1. **Data Dial $\lambda \in [0, 1]$**:
   - $\lambda = 0$: Standard offline dataset KD;
   - $\lambda = 1$: Fully on-policy student generation (MiniLLM-style);
   - $\lambda \in (0, 1)$: Blended mixture of offline expert traces and on-policy rollouts.
2. **Divergence Selector**:
   - Smoothly selects between Forward KL, Jensen-Shannon Divergence (JSD), and Reverse KL.
   - Traditional KD and MiniLLM represent opposite corners of this two-dimensional hyperparameter grid.

#### 3. DistiLLM (Ko et al., 2024) — "Engineering Stability & Computational Efficiency"
GKD encountered two major hurdles in production: heavy-tailed gradient variance during early training when student and teacher diverge, and extreme compute overhead from generating fresh rollouts at every step. DistiLLM introduced two solutions:
- **Skew KL for Gradient Stability**:
  Blends the student and teacher distributions prior to divergence computation:
  $$\pi_\alpha = (1-\alpha) \pi_\theta + \alpha \pi_T$$
  Injecting a smoothing baseline into the denominator eliminates numerical singularities where $Q(x) \to 0$, shrinking gradient variance and stabilizing early convergence.
- **Adaptive Sample Schedule & Replay Buffer**:
  Maintains an experience replay buffer of past student rollouts. Monitors distribution staleness, reusing cached rollouts when policy drift is low and regenerating fresh samples only when necessary, slashing rollout generation compute by over 60%.

#### 4. Cross-Tokenizer Plumbing
When distilling across heterogeneous model families (e.g., DeepSeek-R1 to Qwen-2.5 or LLaMA-3), token vocabularies, special control tokens, and chat templates differ:
- Directly aligning unaligned logit vectors corrupts training;
- Training requires sequence-level text re-projection or probability mass remapping across vocabulary token boundaries.

---

### 5. Privileged Information & Self-Distillation (OPSD)

```text
The Two-Pass Self-Distillation Mechanism (OPSD):
┌────────────────────────────────────────────────────────────────────────┐
│ Single Model Weights π_θ (No second model, saving 40%–60% VRAM):       │
│                                                                        │
│ Pass 1 (Teacher View): Prompt x + Privileged Information (PI) ──► π_θ  │
│                                                              │         │
│                                                      Per-Token Gap     │
│                                                              ▼         │
│ Pass 2 (Student View): Prompt x alone ──────────────────────► π_θ      │
│                                                                        │
│ • Core Insight: The teacher's advantage is not model scale, but        │
│                 an information asymmetry unavailable at test time.    │
└────────────────────────────────────────────────────────────────────────┘
```

#### Motivation & Premise
Frontier research teams face two practical bottlenecks:
1. **No External Teacher Available**: When the target model is already the frontier state-of-the-art model, no superior model exists to distill from;
2. **GPU Memory Saturation**: Retaining two 70B+ or 600B+ models concurrently in GPU memory is cost-prohibitive.

**OPSD Core Insight**: A teacher model's edge is often **not structural scale, but Privileged Information (PI)** unavailable to the student at inference time (e.g. worked solutions, reference docs, or system prompts like "be concise"). Supplying that privileged context to the same model turns it into its own teacher.

#### The Two-Pass Mechanism
Using a **single set of model weights $\pi_\theta$**, the training loop executes two forward passes:
1. The student generates a candidate solution $\hat{y} \sim \pi_\theta(\cdot \mid x)$;
2. **Pass 1 (Teacher View)**: Condition the model on $x + \text{Privileged Information (PI)}$ and compute token probabilities over $\hat{y}$;
3. **Pass 2 (Student View)**: Condition the model on the unaugmented prompt $x$ alone and compute token probabilities over $\hat{y}$;
4. The per-token probability divergence between Pass 1 and Pass 2 provides a **free, dense, high-signal gradient target**, reducing GPU memory consumption by 40%–60%.

#### Three Representative OPSD Archetypes

```text
Spectrum of Privileged Information (PI) Transferability:
Instance-Specific (Overfitting, Poor Transfer) ◄──────────────► Shared-Rule (High Generalization)
┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│ 1. SDR (Answer Key)    │   │ 2. CRISP (Shared Rule) │   │ 3. RLSD (PI + Verifier)│
│ • Hint: Solution key   │   │ • Hint: "be concise"   │   │ • Distillation = size  │
│ • Risk: Overfits task  │   │ • Difficulty-adaptive  │   │ • Verifier = direction │
└────────────────────────┘   └────────────────────────┘   └────────────────────────┘
```

1. **Self-Distilled Reasoner (SDR) — Hint = Reference Solution**:
   - **Mechanism**: Pass 1 is provided the complete reference solution. The model's reasoning improves dramatically, and this enhanced trajectory is distilled into Pass 2 (unaugmented prompt);
   - **Limitation**: The hint is **instance-specific**. The student tends to memorize problem-specific shortcuts, resulting in poor out-of-distribution transfer to novel problems lacking solutions.
2. **CRISP — Hint = Universal Shared Rule**:
   - **Mechanism**: The hint is a task-agnostic meta-instruction applied uniformly across all queries (e.g. "be concise");
   - **Difficulty-Aware Compression**: Aggressively compresses token redundancy on easy problems where succinct reasoning succeeds, while softening compression on hard problems to prevent logical chain collapse. Because the rule is shared, acquired skills generalize across tasks.
3. **RLSD — Hint Combined with Deterministic Verifier (PI + Verifier)**:
   - **Motivation**: Privileged hints boost model confidence, but confidence $\neq$ correctness. A misleading hint causes self-distillation to train the model to be confidently wrong;
   - **Division of Labor**:
     - **Self-distillation provides update magnitude**: Sets the per-token scale of parameter adjustment;
     - **External verifier provides update direction (sign)**: If the privileged rollout passes deterministic unit tests, apply a positive step $(+)$; if the privileged rollout fails, flip the gradient sign $(-)$ to push the policy away from that reasoning failure.

---

### 6. Fragilities & Deep Failure Modes

```text
Core Failure Modes in Self-Distillation:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. The Privileged Information Gap (PI Gap):                            │
│    P(correct | x, hint) ≠ P(correct | x)                               │
│    • Teacher exhibits unearned confidence from hidden privileged data. │
│    • Student forced to mimic confidence becomes Confidently Wrong.     │
│    • Epistemic Suppression: Strips self-correcting hedging phrases.    │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Estimator & Distribution Drift:                                     │
│    • Rock Tokens: ~18% of tokens remain at high loss forever.          │
│    • Prefix Drift: Out-of-support rollouts cause teacher extrapolation │
│                    noise, leading to degenerative student feedback.    │
└────────────────────────────────────────────────────────────────────────┘
```

#### 1. Estimator-Side Fragilities
- **Token-Level KL Proxy Bias**: Sampled per-token KL divergence is a biased surrogate for trajectory-level sequence KL; over long horizons (1000+ tokens), approximation error compounds;
- **"Rock Tokens"**: In reasoning rollouts, approximately 18% of tokens (syntax punctuation, formatting delimiters) exhibit permanently elevated loss, consuming gradient budget without imparting reasoning skill;
- **Tokenizer Misalignment**: Special token mismatches or boundary misalignments corrupt supervisory signals.

#### 2. Teacher-Side Fragilities
- **Prefix Drift**: If the student wanders outside the teacher's high-probability support region, the teacher extrapolates onto out-of-distribution noise; the student then distills this noise, degrading performance;
- **Teachability Collapse**: If the teacher's capability is only marginally superior to the student's, output distributions overlap closely and gradient magnitudes approach zero. Amplifying this weak signal turns dense supervision into dense noise.

#### 3. Deep Conceptual Failures: The PI Gap & Epistemic Suppression
- **The Privileged Information Gap (PI Gap)**:
  $$P(\text{correct} \mid x, \text{hint}) \neq P(\text{correct} \mid x)$$
  The teacher's confidence is anchored in privileged data hidden at inference. Forcing the deployed model to replicate this confidence trains it to be **confidently wrong**;
- **Epistemic Suppression**:
  Aggressive distillation deletes epistemic hedging words ("perhaps", "double check this step", "consider alternative"), stripping the model of its spontaneous self-correction capability;
- **Diversity Collapse**:
  Probability mass concentrates into a single deterministic mode. While single-sample accuracy (Pass@1) may increase, exploration diversity over multiple samples (Pass@$k$) drops precipitously.

#### 4. Practical Prescription
- **Average Over Hint Ensembles**: Sample diverse variations of valid hints and average supervisory gradients to isolate transferable reasoning skills;
- **Prefer Shared-Rule Hints**: **Shared-rule hints (system prompts, reasoning styles) generalize cleanly**, whereas **instance-specific hints (direct solutions) overfit**.

---

### 7. Frontier Industrial Workflows & Best Practices

In modern frontier post-training pipelines (such as DeepSeek-V4, Qwen3, GLM-5, and MiMo-V2-Flash), OPD serves as a standard post-training consolidation phase:
1. **Two-Stage Consolidation Pipeline**:
   - **Stage 1 (RL Exploration)**: Pre-trained foundation models scale reasoning ceilings via RLVR / GRPO, discovering emergent deep thinking behaviors;
   - **Stage 2 (On-Policy Distillation)**: OPD transfers this capability into deployable edge models (1.5B/7B/14B/32B), preserving self-reflection and backtracking while compressing token footprint.
2. **Multi-Teacher OPD ("Specialize, Then Unify")**:
   Specialized expert teachers are trained independently for competitive math, software engineering, and multi-turn tool interaction. During edge model training, multiple teachers concurrently evaluate the student's on-policy traces, distilling domain-specific competencies into a single unified generalist.

---

## Module 6: Agentic RL—Multi-Turn Environment Interaction

Extending RL from single-turn text generation to interactive multi-turn tool calling creates **Agentic RL**:

```text
Agentic RL Multi-Turn Environment Interaction:
┌──────────────┐      Observation (Tool Output / Error / DOM)     ┌──────────────┐
│              │ ◄─────────────────────────────────────────────── │              │
│  Agent (LLM) │                                                  │  Sandbox /   │
│  Policy π_θ  │ ───────────────────────────────────────────────► │  Environment │
│              │ ───────────────────────────────────────────────► │              │
└──────────────┘       Action (Bash Command / Python Script)      └──────────────┘
       ▲                                                                 │
       └────────────────── Environment Reward r_env ─────────────────────┘
```

### 1. Outcome Reward Model (ORM) vs Process Reward Model (PRM) & OPD Synergies

- **Outcome Reward Model (ORM)**:
  - Evaluates only terminal success (0/1 binary reward upon task completion);
  - **Bottleneck**: Long-horizon trajectories (dozens of steps, thousands of tokens) suffer from severe **credit assignment failure**—it cannot pinpoint which tool invocation caused final failure.
- **Process Reward Model (PRM)**:
  - Scores every intermediate thought and tool action, improving Monte Carlo Tree Search (MCTS) and Best-of-$N$ pruning efficiency.
- **Synergy with OPD Dense Supervision**:
  In long-horizon agent traces (e.g. Terminal-Bench), ORM yields only 1 bit per episode, while PRM annotation is expensive. Applying **OPD (Module 5)** allows a frontier teacher to grade student trajectories token-by-token, supplying dense gradients across the entire interactive sequence at zero additional human annotation cost.

### 2. Environment Sandboxes & Reward Design

| Sandbox Domain | Agent Action Types | Deterministic Verification Reward Formulation |
|---|---|---|
| **Python Coding** | Write code, debug, execute unit tests | `r = (Passed Tests / Total Tests) - 0.1 * SyntaxError - 0.05 * Latency` |
| **Bash / DevOps** | Shell commands, process management | `r = Target State Flag (Exit Code == 0 & Output Match)` |
| **SQL Database** | Query optimization, joins, aggregation | `r = DataFrame Exact Hash Match with Ground-Truth` |
| **Web Browser** | Click, scroll, form fill, DOM extract | `r = Target DOM State Verification + Success Flag` |

---

## Module 7: Key Interview FAQs

### Q1: Why does GRPO eliminate the Critic network in PPO?
> **Answer**: In long-sequence reasoning (up to 32K tokens), maintaining an Actor-sized Critic network causes catastrophic GPU memory blowup (OOM) and suffers from high value-estimation variance. GRPO generates a group of $G$ outputs for each query and calculates advantages purely by normalizing rewards across the group ($A_i = \frac{r_i - \mu}{\sigma + \epsilon}$), completely removing the Critic network while maintaining stable policy gradients.

### Q2: What is RLVR, and how does it fundamentally differ from standard RLHF?
> **Answer**: RLVR (Reinforcement Learning with Verifiable Rewards) uses deterministic external verification engines (unit test execution, symbolic math engines, compilers) rather than subjective neural reward models to score responses. Unlike RLHF—which is vulnerable to reward hacking (e.g. producing superficial verbose prose)—RLVR provides objective, unhackable feedback, allowing models to achieve super-human reasoning via self-exploration and test-time compute scaling.

### Q3: Why does Reverse KL outperform Forward KL in On-Policy Distillation (OPD)? What is the core mode behavior difference?
> **Answer**:
> 1. **Mathematical Asymmetry**: $\mathbb{D}_{\text{KL}}(P \parallel Q)$ penalizes regions where $P$ is large but $Q \approx 0$. Forward KL puts the teacher in the reference position, forcing the student to cover all teacher peaks (Mode-Covering). When student capacity is constrained, it smears probability mass into low-density valleys, generating fluent nonsense and hallucinations;
> 2. **Mode-Seeking in Reverse KL**: Reverse KL puts the student in the reference position, penalizing the student for generating tokens the teacher rejects. It allows the small student to drop unrepresentable secondary modes and concentrate on a single coherent, high-confidence peak, ensuring rigorous logical consistency.

### Q4: What is RL's Razor? How does OPD prevent catastrophic forgetting and quadratic error accumulation?
> **Answer**:
> 1. **RL's Razor & Forgetting Prevention**: SFT (Forward KL) forces distant parameter jumps across weight space, corrupting pre-trained features. Policy gradients in Reverse KL represent the minimal local perturbation required to increase expected reward, preserving pre-trained general capabilities;
> 2. **Compounding Error Suppression (DAgger Theorem)**: In off-policy imitation, early mistakes push the student onto unvisited prefixes, compounding errors quadratically $\mathcal{O}(T^2)$. OPD trains on the student's own generated prefixes, enabling error recovery and constraining error growth to linear $\mathcal{O}(T)$.

### Q5: What are the Privileged Information Gap (PI Gap) and Epistemic Suppression in OPSD? How are they mitigated?
> **Answer**:
> 1. **The PI Gap**: $P(\text{correct} \mid x, \text{hint}) \neq P(\text{correct} \mid x)$. Teacher confidence stems from privileged data unavailable at test time; forcing the unprivileged student to match that confidence leads to models being confidently wrong;
> 2. **Epistemic Suppression**: Aggressively compressing rollouts strips self-correcting hedging phrases ("maybe", "double check"), degrading exploration diversity (Pass@$k$ collapse);
> 3. **Mitigations**:
>    - Employ RLSD: Use self-distillation for update magnitude and an external verifier to dictate gradient sign;
>    - Avoid instance-specific hints (which overfit); use shared-rule hints (system prompts) and average over hint ensembles.
