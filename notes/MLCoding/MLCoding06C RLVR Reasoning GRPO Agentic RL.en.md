# ML Coding 06C · RLVR, Reasoning Models & Agentic RL: DeepSeek-R1 Paradigm, GRPO Derivation, Rule Verifiers & Agentic RL

As generative AI enters the era of "System 2 Slow Thinking" and "Autonomous Agents", traditional human-preference RLHF (e.g. standard PPO and DPO) has reached fundamental scaling limits. The modern frontier—represented by **OpenAI o1 / o3**, **DeepSeek-R1**, and modern **Agentic Tool-Use RL**—shifts the reinforcement learning paradigm from "mimicking human conversational style" to **"Reinforcement Learning with Verifiable Rewards (RLVR)"**.

This guide provides a comprehensive breakdown across 5 key pillars:
1. **The RLVR Paradigm Shift: How Deterministic Rule-Based Verifiers Eradicate Reward Hacking**
2. **DeepSeek-R1 Core Architecture: Group Relative Policy Optimization (GRPO) Mathematical Derivation**
3. **Pure RL Emergence: Long Chain-of-Thought (CoT), Self-Reflection, Backtracking & The "Aha Moment"**
4. **DeepSeek-R1 4-Stage Production Recipe (Cold-Start $\to$ Reasoning RL $\to$ Rejection SFT $\to$ Universal RL)**
5. **Agentic RL: Multi-Turn Environment Sandboxes, Process Reward Models (PRM vs ORM) & MCTS**

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
```

---

## Module 5: Agentic RL—Multi-Turn Environment Interaction

Extending RL from single-turn text output to interactive multi-turn tool use creates **Agentic RL**:

```text
Agentic RL Multi-Turn Environment Interaction:
┌──────────────┐      Observation (Tool Output / Error / DOM)     ┌──────────────┐
│              │ ◄─────────────────────────────────────────────── │              │
│  Agent (LLM) │                                                  │  Sandbox /   │
│  Policy π_θ  │ ───────────────────────────────────────────────► │  Environment │
└──────────────┘       Action (Bash Command / Python Script)      └──────────────┘
       ▲                                                                 │
       └────────────────── Environment Reward r_env ─────────────────────┘
```

### Outcome Reward Model (ORM) vs Process Reward Model (PRM)
- **ORM**: Provides a binary $0/1$ reward only upon final trajectory completion. Suffers from the **credit assignment problem** over long tool trajectories.
- **PRM**: Scores each intermediate thought and tool execution step, dramatically improving Monte Carlo Tree Search (MCTS) and Best-of-$N$ pruning efficiency.

---

## Module 6: Key Interview FAQs

### Q1: Why does GRPO eliminate the Critic network in PPO?
> **Answer**: In long-sequence reasoning (up to 32K tokens), maintaining an Actor-sized Critic network causes catastrophic GPU memory blowup (OOM) and suffers from high value-estimation variance. GRPO generates a group of $G$ outputs for each query and calculates advantages purely by normalizing rewards across the group ($A_i = \frac{r_i - \mu}{\sigma + \epsilon}$), completely removing the Critic network while maintaining stable policy gradients.

### Q2: What is RLVR, and how does it fundamentally differ from standard RLHF?
> **Answer**: RLVR (Reinforcement Learning with Verifiable Rewards) uses deterministic external verification engines (unit test execution, symbolic math engines, compilers) rather than subjective neural reward models to score responses. Unlike RLHF—which is vulnerable to reward hacking (e.g. producing superficial verbose prose)—RLVR provides objective, unhackable feedback, allowing models to achieve super-human reasoning via self-exploration and test-time compute scaling.
