# Reinforcement Learning Exercises

## A · Algorithm Basics (Q1-Q10)

<details class="exercise">
<summary><span class="q-label">Q1</span> <span class="q-text">Why use Actor-Critic instead of a pure Critic method?</span></summary>

**Core tension**: A pure Critic method (such as DQN) directly learns $Q(s,a)$, then selects an action with $\arg\max_a Q(s,a)$. That works in discrete action spaces, but an LLM's action space is the entire vocabulary (32K-128K tokens), so $\arg\max$ is intractable.

The value of the Actor is that it directly parameterizes the policy $\pi_\theta(a|s)$ and uses gradient ascent to optimize expected reward, without enumerating the action space. The value of the Critic is that it provides a low-variance advantage estimate; otherwise pure policy gradients have enormous variance and require many samples to converge.

**Mapped to LLM RL**: Actor = the language model itself (outputs the token distribution); Critic = an independent value head (predicts the expected reward of the current prefix). GRPO further shows that when the group is large enough, you can omit the Critic and use relative scores within the group as the advantage while still keeping variance low enough.

**Extension**: -> [[MLSYS14 Post-Training Infra#III. Minimal Algorithm Background: PPO and GRPO]]

</details>

<details class="exercise">
<summary><span class="q-label">Q2</span> <span class="q-text">What is the relationship among KL divergence, cross-entropy, and MLE?</span></summary>

The three are closely related. Start from the most basic identity:

$$
\text{KL}(P \| Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)} = H(P, Q) - H(P)
$$

- $H(P,Q)$ is cross-entropy, and $H(P)$ is entropy
- MLE is equivalent to minimizing cross-entropy $H(P_{\text{data}}, Q_\theta)$, because $H(P_{\text{data}})$ is a constant
- **Therefore MLE ≡ minimizing KL$(P_{\text{data}} \| Q_\theta)$**

**Role in RL**:
- The KL penalty term $\text{KL}(\pi_\theta \| \pi_{\text{ref}})$ in PPO/GRPO prevents the policy from drifting too far from the initial SFT model and avoids reward hacking
- DPO turns the RLHF objective into an MLE problem by training an implicit alignment reward function through the cross-entropy of preference pairs
- In GRPO, KL can be placed inside the reward ($r' = r - \beta \text{KL}$) or inside the loss (explicit regularization). The two are equivalent in principle but have different numerical behavior

</details>

<details class="exercise">
<summary><span class="q-label">Q3</span> <span class="q-text">How should reward be designed in different RL scenarios?</span></summary>

**Core principle**: the reward signal should be strong, only moderately sparse, and impossible to exploit through reward hacking.

**By scenario**:

| Scenario | Reward Type | Example |
|------|------------|------|
| Mathematical reasoning | Outcome reward (terminal) | Final answer correct/incorrect (0/1 or continuous score) |
| Code generation | Execution-based | Test-case pass rate |
| Dialogue / RLHF | RM score | Human preference model score |
| Agentic tasks | Sparse terminal + shaped | Task completion + intermediate step rewards |
| Long reasoning | Process reward | Quality score for each step in the reasoning chain |

**Common design problems**:
- **Too sparse**: if only the final correct answer gets reward, early training receives almost no gradient signal
- **Too dense**: if every step has a reward, reward shaping becomes easy to exploit and the model learns to game intermediate scores
- **Verifiability**: math and code can be verified programmatically and are naturally resistant to hacking; open-ended QA requires an RM, which itself may be attacked

**The rise of RLVR (Verifiable Reward)**: use rules or programs as reward verifiers instead of an RM, thereby avoiding RM reward hacking. A large part of DeepSeek-R1's success depends on verifiable rewards in math and coding tasks.

</details>

<details class="exercise">
<summary><span class="q-label">Q4</span> <span class="q-text">How are importance sampling (IS) and rejection sampling applied in RL?</span></summary>

**Importance sampling (IS)**: when the training data comes from an old policy $\pi_{\text{old}}$ but we want to optimize a new policy $\pi_\theta$, we correct the expectation with an IS ratio:

$$
\mathbb{E}_{\pi_\theta}[f] \approx \mathbb{E}_{\pi_{\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} f\right]
$$

This is exactly where $r_t(\theta)$ in the PPO objective comes from. IS is the algorithmic foundation of async RL: it allows training on slightly stale data as long as the IS ratio does not explode, which clipping controls.

**Rejection sampling**: generate multiple candidate sequences and keep only some of them according to an acceptance rule:
- The most common form is: generate G candidates and keep only those that pass a quality threshold, such as reward > threshold
- Group sampling in GRPO is a soft form of rejection sampling: samples are not discarded, but higher-reward samples get larger weights through normalized advantages
- Rejection Sampling Fine-Tuning (RFT): generate K candidates at each step and use only the correct answers for SFT, without doing RL

**System-side impact**: the IS ratio requires recomputing logprobs during training, which means an extra forward pass through the actor. That is one cost PPO has over GRPO, since GRPO can directly use the logprobs recorded during rollout for IS.

</details>

<details class="exercise">
<summary><span class="q-label">Q5</span> <span class="q-text">How is advantage computed in PPO and GRPO? Why subtract a baseline? Is standard-deviation normalization necessary?</span></summary>

**PPO's advantage (GAE)**:
$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T}(\gamma\lambda)^l(r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l}))
$$
This requires a Critic network to estimate $V(s_t)$ at every token position. The parameter $\lambda$ controls the bias-variance tradeoff: $\lambda=0$ reduces to TD, and $\lambda=1$ reduces to Monte Carlo returns.

**GRPO's advantage (Group Relative)**:
$$
\hat{A}_i = \frac{r_i - \mu_g}{\sigma_g}
$$
where $\mu_g, \sigma_g$ are the mean and standard deviation of rewards within the same group, meaning the G completions for the same prompt.

**Why subtract a baseline**: the variance of the policy gradient comes from the absolute scale of $r_t$. Subtracting a baseline $b$, whether a constant or a state-dependent $V(s_t)$, **does not change the gradient expectation** because $\mathbb{E}[\nabla \log\pi \cdot b] = 0$, but it greatly reduces variance.

**Is standard-deviation normalization necessary?** This is a debated point in GRPO and is the origin of Dr.GRPO:
- **In favor**: normalization unifies the advantage scale across prompts, so the learning rate is not affected by reward scale
- **Against**: if every answer in a group gets the same reward, then $\sigma_g = 0$, which causes division-by-zero or forces clipping; if all answers are correct, normalization makes the advantage 0 and discards the learning signal from correct samples
- Dr.GRPO's fix: handle degenerate groups such as full-correct and full-wrong groups specially, either by dropping them or restoring variance

</details>

<details class="exercise">
<summary><span class="q-label">Q6</span> <span class="q-text">How does exploration differ between RL training and scaling at test time?</span></summary>

**Exploration during RL training**: explore the action space through policy stochasticity, meaning sampling with temperature > 0. Higher temperature means more exploration and more diversity, but also more noise. Group sampling in GRPO is essentially repeated random exploration on the same problem to find high-reward paths.

**Exploration in test-time scaling (TTS)**: the model is fixed and parameters are not updated. The solution space is explored through methods such as:
- **Best-of-N sampling**: generate N candidates and choose the best one with a verifier or RM
- **Beam search / Diverse beam search**: maintain multiple candidate paths
- **Process reward model guidance**: score each step with a PRM and prune poor intermediate steps
- **MCTS (tree search)**: more systematic exploration, but with extremely high compute cost

**Key difference**: exploration during training updates model parameters and aims to find a good policy; exploration in TTS does not update parameters and aims to find a good output at inference time. TTS compute can scale as a multiple of the model parameter count, which is the essence of how models like o1 and R1 "think more."

</details>

<details class="exercise">
<summary><span class="q-label">Q7</span> <span class="q-text">How does PPO clipping work? Why take the min? What happens without clipping? How is CISPO different?</span></summary>

**PPO clipping mechanism**:

$$
\mathcal{L}_{\text{clip}} = \mathbb{E}_t\left[\min\left(\underbrace{r_t \hat{A}_t}_{\text{original objective}},\ \underbrace{\text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t}_{\text{clipped version}}\right)\right]
$$

**Why take the min (a pessimistic lower bound)**:
- If $\hat{A}_t > 0$ and the action is good, we want to increase $\pi_\theta$ to enlarge $r_t$, but clipping prevents $r_t$ from exceeding $1+\epsilon$ and stops over-exploiting a single sample
- If $\hat{A}_t < 0$ and the action is bad, we want to decrease $\pi_\theta$, but clipping prevents $r_t$ from dropping below $1-\epsilon$ and stops over-penalization
- Taking the min enforces a bounded amount of progress: regardless of the sign of the advantage, the size of a single gradient update is constrained

**What happens without clipping**: the policy may over-optimize on a single good sample, collapse the distribution, and then reward hack, because the RM or verifier only sees outputs inside the training distribution.

**CISPO** (Clipped IS Policy Optimization, the algorithm paired with MiniMax Forge):
- On top of IS correction, it changes the clip range from a fixed $\epsilon$ to one that is adjusted dynamically based on staleness: larger staleness -> tighter clipping
- The goal is that in a fully async system, the IS ratio itself already reflects the extent of policy drift, so dynamic clipping is more reasonable
- By contrast, standard PPO clipping assumes $\pi_{\text{old}} \approx \pi_\theta$, meaning a near on-policy regime; CISPO explicitly handles off-policy conditions

</details>

<details class="exercise">
<summary><span class="q-label">Q8</span> <span class="q-text">Why add a KL penalty in GRPO? How is KL computed? Why do DAPO and GSPO remove it?</span></summary>

**Purpose of the KL penalty**: prevent the policy from drifting too far from the initial SFT model while optimizing reward, which can cause:
1. Lower language quality, such as gibberish generation or reward-hacking patterns
2. Catastrophic forgetting of knowledge learned during SFT

**Two places to compute KL**:

| Location | Formula | Characteristics |
|------|------|------|
| Inside the reward (original GRPO) | $r' = r - \beta \text{KL}[\pi_\theta \|\|\pi_{\text{ref}}]$ | Computed per token; affects advantage estimation |
| Outside the loss (explicit regularization) | $\mathcal{L} = \mathcal{L}_{\text{GRPO}} + \beta\text{KL}$ | The gradient acts directly on parameters; numerically more stable |

**Why DAPO removes KL**:
- For tasks with strong verifiable rewards, such as math and code, the KL penalty limits the model's ability to move far away from the SFT distribution, but that departure is necessary because the SFT model does not naturally perform long CoT reasoning
- Experiments show that removing KL and moderately increasing the clip range, or using a clip lower bound instead, works better

**GSPO's KL improvement**: replace token-level KL with sequence-level KL, which reduces the unreasonable contribution of high-entropy tokens such as spaces.

</details>

<details class="exercise">
<summary><span class="q-label">Q9</span> <span class="q-text">In LLM training, what happens if the loss is all-reduced multiple times?</span></summary>

This is a systems debugging question that tests understanding of numerical behavior in distributed training.

**Normal case**: under distributed data parallelism such as DDP or ZeRO, each rank computes the loss on its local batch, then gradients are all-reduced and averaged to obtain the global average gradient.

**What goes wrong if the loss itself is all-reduced** rather than the gradients:
- If the code mistakenly all-reduces the loss value itself, the loss is amplified by $N$, where $N$ is the number of ranks, and the gradient is also amplified by $N$
- Effect: the effective learning rate becomes $N$ times larger than intended, so training quickly diverges and the loss becomes NaN or explodes

**What goes wrong if gradients are all-reduced multiple times**:
- The gradients are averaged multiple times, and each all-reduce divides by the rank count
- The final gradient is shrunk by a factor of $N^k$, where $k$ is the number of extra all-reduces, so the effective learning rate becomes tiny and training almost stops updating

**Special RL scenario**: during GRPO training, rollout data must be distributed across ranks. If reward or advantage is accidentally all-reduced across ranks during distribution, perhaps because the intent was to compute a global mean or std, and then loss computation performs another all-reduce, double all-reduction occurs. The symptom is abnormally low training loss because the gradients are too small, or reward fails to improve.

</details>

<details class="exercise">
<summary><span class="q-label">Q10</span> <span class="q-text">What is the reward function in DPO? Can reward hacking happen? How can it be mitigated?</span></summary>

**DPO's implicit reward**: DPO merges the RM + RL two-stage RLHF pipeline into one objective. Its loss is equivalent to doing RL with the following implicit reward:

$$
r_{\text{DPO}}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

That is, the log-ratio of the current policy relative to the reference policy, multiplied by $\beta$, the inverse of the KL coefficient.

**Risk of reward hacking in DPO**:
- Because the implicit reward is directly determined by the policy parameters, the policy can artificially increase reward by lowering the reference probability of dispreferred samples instead of truly generating better outputs
- In practice this appears as the model becoming too conservative, avoiding any sequence the reference model considers unlikely, or length hacking, where reward is manipulated by changing sequence length

**Mitigations**:
- **SimPO**: remove the reference model and instead use length-normalized log-probability as the implicit reward, which is more resistant to length hacking
- **IPO** (Identity PO): directly optimize preference probability and bypass explicit reward-function modeling
- **Online DPO**: continuously generate new preference pairs during training instead of relying on a fixed dataset, making it more like online RLHF
- **KTO**: does not require pairwise data and uses only binary feedback, good/bad, with a more robust implicit reward design

</details>

---

## B · Advanced Algorithms (Q11-Q19)

<details class="exercise">
<summary><span class="q-label">Q11</span> <span class="q-text">What methods address the train-inference mismatch of MoE models, and what is the underlying idea?</span></summary>

**Root problem**: the MoE model's router, meaning top-k gating, is implemented independently on the inference side, such as vLLM or SGLang, and on the training side, such as Megatron. Floating-point precision differences can cause different expert choices in boundary cases, which means training logprobs for a different sequence.

**Three main categories of handling methods**:

1. **Router inconsistency**: during inference, a token is routed to Expert A, but during training the forward pass routes it to Expert B because the two implementations have slightly different floating-point behavior in softmax or top-k. This creates a systematic bias in the IS ratio.

   Fix -> **Keep Routing**: record each token's expert routing index on the inference side, pass it along with the trajectory as auxiliary data, and force the training side to replay the same routing, bypassing the router network and directly selecting the expert.

2. **Sampling mask mismatch**: during inference, top-p or top-k truncates the vocabulary so only the highest-probability part is sampled, but training computes logprobs over the full 32K vocabulary, so the normalization differs.

   Fix -> **Keep Sampling Mask**: record the sampling mask on the inference side, meaning which tokens were truncated, and apply the same mask to the logits before softmax during training so that logprob normalization matches inference.

3. **Token-level ratio noise**: with highly sparse MoE, router jitter makes the single-token logprob ratio noisy, especially during long GRPO or PPO training.

   Fix -> **GSPO / sequence-level ratio**: use a response-level likelihood ratio for clipping and reduce the effect of route jitter at the single-token level. This does not replay the router; instead it changes the optimization objective from token-level to sequence-level.

-> [[MLSYS14 Post-Training Infra#6.4 Train-Inference Mismatch]]

</details>

<details class="exercise">
<summary><span class="q-label">Q12</span> <span class="q-text">How should group size, learning rate, PPO epochs, and generation length be chosen in RL training?</span></summary>

**Group size G**:
- Larger G -> lower advantage variance because the statistics are more reliable, but each batch requires generating G times as many sequences, so rollout cost rises linearly
- Practical range: G=4 when resources are limited up to G=32 for long-reasoning tasks; DAPO recommends G=8-16
- If task difficulty is very high and most completions are wrong, you need a larger G to sample enough correct examples

**Learning rate**:
- RL learning rates are usually 5-10x smaller than SFT, about 1e-6 to 5e-6, whereas SFT often uses 1e-5
- The reason is that RL updates are not as uniform as SFT updates: high-variance advantages cause gradient fluctuations, so a small learning rate prevents single-step blowups

**PPO epochs**: how many times each batch is reused for training:
- Standard PPO: 2-4 epochs per batch
- Every extra epoch increases IS-ratio drift because $\pi_\theta$ is changing while $\pi_{\text{old}}$ stays fixed; clipping becomes tighter and the gradient approaches 0 later in training
- GRPO typically uses only 1 epoch because group data volume is already large enough

**Generation length**:
- If max length in training is too short, reward gets truncated and the model learns to stop early instead of solving the problem
- If it is too long, memory and time cost explode and the long tail worsens
- In practice, it is usually set to 2-3x the expected output length; DAPO uses dynamic truncation, meaning samples over a threshold are truncated but receive 0 reward instead of being discarded

</details>

<details class="exercise">
<summary><span class="q-label">Q13</span> <span class="q-text">Compared with GRPO, what do Dr.GRPO, DAPO, GSPO, CISPO, SAPO, DPPO, MaxRL, and SimKO change, and what are their limitations?</span></summary>

| Method | Core change | Problem solved | Limitation |
|------|---------|------------|------|
| **Dr.GRPO** | Special handling for degenerate groups, meaning all-correct or all-wrong groups | Avoids division-by-zero and fake gradients | Adds data-filtering complexity |
| **DAPO** | Removes KL, changes to clip lower bound, token-level loss | Lets the model deviate substantially from SFT | No KL constraint, so stability depends on clip settings |
| **GSPO** | Sequence-level KL instead of token-level KL | Reduces KL contribution from high-entropy tokens | Sequence-level KL gradient estimation is more complex |
| **CISPO** | IS clipping adjusted dynamically with staleness | Off-policy correction under async RL | Requires accurate staleness estimation |
| **SAPO** | Adaptive clip threshold based on the IS-ratio distribution | Avoids a fixed $\epsilon$ being suboptimal across tasks | Extra statistics overhead |
| **DPPO** | Distributed PPO with a distributed critic | Reduces the critic memory bottleneck | Higher communication cost |
| **MaxRL** | MiniMax's agentic RL algorithm, paired with Forge | Ultra-long context and multi-agent settings | Not open source |
| **SimKO** | Simplified KL-regularized objective | Moves KL from reward to loss, improving numerical stability | Sensitive to hyperparameters |

**Key pattern**: almost all of these methods are adjusting three knobs: (1) the location and strength of the KL penalty, (2) the range and form of clipping, and (3) the way advantage is computed or normalized. There is no universal solution; the right choice depends on the task and scale.

</details>

<details class="exercise">
<summary><span class="q-label">Q14</span> <span class="q-text">How do TRPO, DPPO, and AReaL impose trust-region constraints on the RL objective?</span></summary>

**TRPO (Trust Region Policy Optimization)**: applies a trust region through an exact KL constraint:
$$
\max_\theta \mathbb{E}\left[\frac{\pi_\theta}{\pi_{\text{old}}} A\right] \quad \text{s.t.} \quad \text{KL}[\pi_{\text{old}} \| \pi_\theta] \leq \delta
$$
It is solved with second-order methods, using an approximate inverse of the Fisher information matrix. The compute cost is extremely high, so it is unusable for LLMs.

**PPO's trust region**: clipping replaces the exact KL constraint. It is a first-order method whose practical effect is close to TRPO. Clipping is a soft trust region: updates are not penalized while the ratio stays inside $[1-\epsilon, 1+\epsilon]$.

**DPPO (Distributed PPO)**: the trust-region constraint is the same as PPO's, but the problem it solves is the critic bottleneck in large-scale distributed settings, where critic parameter count is about the same as actor parameter count and creates large memory pressure. DPPO distributes critic storage and computation while leaving the trust-region constraint unchanged.

**AReaL's trust region**: decoupled clipping, where the larger the staleness $s$, the tighter the clipping, as discussed in Q7 and CISPO. From the trust-region perspective, larger staleness means a larger gap between $\pi_\theta$ and the $\pi_{\text{old}}$ that generated the samples, so updates must be more conservative.

</details>

<details class="exercise">
<summary><span class="q-label">Q15</span> <span class="q-text">Can RL fundamentally expand the capability boundary of LLMs?</span></summary>

This is an open academic discussion question without a definitive answer, but several key arguments matter:

**The perspective that says "yes, it can expand capabilities"**:
- DeepSeek-R1: RL unlocked long reasoning chains that SFT could not directly produce
- AlphaGo and AlphaCode: RL exceeded human performance on closed tasks, suggesting expansion is possible
- In theory, RL's exploration mechanism can discover solution paths that are not covered by the SFT dataset

**The opposing or limiting perspective**:
- Research from Tsinghua IIIS and others, including ICLR 2026 work, argues that under strictly controlled verifiable environments, RL training on LLMs redistributes behavior within the model's existing capability distribution rather than creating fundamentally new abilities
- RL cannot teach a model knowledge it completely lacks from scratch, such as a math theorem it has never seen
- Reward hacking: the model may find shortcuts that are semantically wrong but still satisfy the verifier's formal criteria

**Practical conclusion**: RL is very effective in settings where the model already has latent potential but needs the right incentive to output it reliably, such as mathematical reasoning. It is much less effective when the model truly lacks the knowledge. RL is a **capability releaser**, not a **knowledge injector**.

</details>

<details class="exercise">
<summary><span class="q-label">Q16</span> <span class="q-text">Based on work such as ProRL, how should we think about the scaling boundary of RL training?</span></summary>

**The core finding of ProRL (Prolonged RL Training)**: performance improves with training steps according to a power law similar to pretraining, but there is a **capability emergence threshold**: below a certain amount of compute there is almost no improvement, and above it performance rises quickly.

**Key variables in scaling**:
- **Number of rollouts**: more samples mean better gradient estimates and more stable learning
- **Task-difficulty distribution**: tasks that are too easy, where everything is correct, and too hard, where everything is wrong, both provide little learning signal; curriculum learning matters
- **Quality of the reward signal**: the more reliable the reward, the more effective scaling becomes; RM reward hacking is a hard upper bound on scaling

**How RL scaling is similar to and different from pretraining scaling**:
- Same: more compute generally yields better performance, and a power law exists
- Different: RL scaling happens on top of a fixed base model and is limited by that base model's capability boundary; unlike pretraining, it cannot accumulate unlimited new knowledge

**Practical advice**: first validate the reward signal and the algorithm with small-scale RL on the chosen task. Before scaling, make sure rollout data quality is good, especially that the rejection-sample ratio is neither too high nor too low.

</details>

<details class="exercise">
<summary><span class="q-label">Q17</span> <span class="q-text">How does OPD (On-Policy Distillation) improve on traditional RL and SFT? What are its application scenarios?</span></summary>

**Core idea of OPD**: the student model generates sequences using its current policy, while a teacher model, typically a stronger one, provides token-level logit guidance. The student learns the teacher's distribution rather than one-hot labels.

**Advantages over SFT**:
- SFT's teacher forcing does not handle distribution shift, because it sees ground truth during training but its own outputs at inference time
- OPD uses prefixes generated by the student itself, and the teacher supplies only token-level soft labels through KD, so the student learns how to correct itself under its own distribution

**Advantages over RL**:
- RL requires a reward function, which is difficult to design outside verifiable tasks
- OPD directly uses teacher logits as the reward signal, avoiding manual reward design

**Application scenarios**:
- Knowledge distillation: a large model such as GPT-5 -> a small model while preserving on-policy behavior
- Continual learning: self-improvement using outputs from the current model version plus logits from a better version
- RLCD (RL from Contrast Distillation): combining OPD with preference learning

</details>

<details class="exercise">
<summary><span class="q-label">Q18</span> <span class="q-text">At what stage of training does LLM reasoning ability emerge?</span></summary>

**Evidence summary**:

| Stage | How reasoning ability appears |
|------|---------------|
| Pretraining | Implicit reasoning ability exists, as seen in few-shot CoT work, but is unreliable; the ability is determined by the training data |
| SFT (on CoT data) | Significantly improves reasoning stability; the model learns the "think first, answer later" format |
| **RL / RLVR** | **Unlocks autonomous exploration of reasoning paths**: DeepSeek-R1-Zero shows that long reasoning chains can emerge through pure RL without SFT |

**Key experiment (DeepSeek-R1-Zero)**: applying GRPO directly to a base model without SFT led the model to spontaneously learn:
- An "Aha moment": changing its line of thought midway and revisiting wrong reasoning
- Longer reasoning chains, meaning it automatically extends token usage
- Self-verification

**Academic dispute**: is this emergence a genuinely new capability, or is RL merely unlocking a capability already present from pretraining that SFT failed to activate? The mainstream view currently leans toward the latter, that RL is a capability releaser, but this does not reduce RL's practical value.

</details>

<details class="exercise">
<summary><span class="q-label">Q19</span> <span class="q-text">From DeepSeek R1 to V3.2 and then to a future V4, what RL-related improvements were introduced? How is RL different in MoE?</span></summary>

**The evolution path of DeepSeek / MoE RL**:

| Version | RL-related improvement |
|------|------------|
| R1-Zero | Pure GRPO from base, no SFT, proving that RL can directly induce reasoning ability |
| R1 | Added SFT warm-up, meaning cold start, plus RL; rejection sampling to supplement SFT data; multi-stage training |
| V3 (MoE) | RL on an MoE architecture, which requires handling Expert Parallelism; introduced routing replay |
| V3.2 | Further RL scaling; longer CoT; agentic scenarios |

**Special challenges of RL in MoE**:

1. **Expert Parallelism x Rollout**: during rollout, an MoE model needs AllToAll communication for token routing. This is extremely inefficient in the decode phase, where batch sizes are small. Rollout therefore usually uses TP instead of EP to avoid AllToAll latency.

2. **Router drift**: RL training changes router parameters, which can lead to expert load imbalance where a small number of experts are overused. An auxiliary load-balancing loss is required.

3. **MoE mismatch**: as discussed in Q11, router inconsistency is the core engineering issue in MoE RL. Routing Replay is the system-side solution, while GSPO is the algorithm-objective-side solution.

4. **Amplified memory pressure**: the total expert parameter count in MoE is much larger than the active parameter count. Under EP, per-GPU memory may stay flat, but communication rises significantly. RL's own memory pressure, actor + ref + KV, is then stacked on top of EP's communication pressure.

</details>

---

## C · Single-Node and Memory (Q20/Q22/Q26)

<details class="exercise">
<summary><span class="q-label">Q20</span> <span class="q-text">How should memory pressure be broken down during GRPO training?</span></summary>

Do not memorize a fixed answer in terms of "how many copies of the model." GRPO memory pressure depends on the colocated or disaggregated layout, the FSDP or ZeRO sharding strategy, whether the reference is shared, whether the rollout engine stays resident, and the rollout context length, group size, and KV-cache policy.

A more robust way to answer is to break memory usage into four categories:

| Category | Main contents | Optimization direction |
|------|----------|----------|
| Model state on the training side | Actor parameters, gradients, optimizer states, and possibly reference or old-policy-logprob-related state | FSDP / ZeRO-3 sharding, optimizer-state sharding, reference sharing |
| State on the inference side | Rollout-engine weight replica, KV cache, paged-cache metadata, request buffers | Co-locate rollout and training to reuse weights, prefix sharing, KV paging, sleep/offload |
| Backpropagation state | Activations, microbatch buffers, sequence-packing buffers | Activation checkpointing, sequence packing, microbatch tuning |
| Transient communication buffers | All-gather, reduce-scatter, weight-sync buckets, reshard buffers | Bucket-size tuning, overlapped communication, bucketized weight sync |

Compared with PPO, GRPO removes the critic, but group sampling expands each prompt into G completions, so KV cache, activations, and reward-computation wait time all increase. As a result, GRPO's total memory pressure is not necessarily lower than PPO's. The key question is whether the memory saved by removing the critic exceeds the memory added by rollout, KV, and activation growth.

</details>

<details class="exercise">
<summary><span class="q-label">Q22</span> <span class="q-text">INT8 vs FP8: which precision is recommended for training and inference respectively, and what are the tradeoffs?</span></summary>

See the full analysis in [[MLSYS14 Post-Training Infra#6.5 Precision Special Topic: INT8 vs FP8]].

**Short answer**:

- **Training recommends FP8**: H100 FP8 matrix multiplication is 2x faster than BF16; scaling factors are required; the usual combination is E4M3 for forward and E5M2 for backward
- **Inference recommends INT8 weight-only**: no calibration is needed; halving weight size reduces HBM bandwidth pressure; decode is memory-bound, so INT8 directly improves throughput

**Additional RL consideration**: using INT8 on the rollout side and FP8 on the training side introduces mismatch, because logprobs are computed at different precisions. Some frameworks, such as ROLL, provide a unified precision configuration to address this.

</details>

<details class="exercise">
<summary><span class="q-label">Q26</span> <span class="q-text">How is backpropagation implemented in large-scale multi-node RL training?</span></summary>

See the multi-node backpropagation section in [[MLSYS14 Post-Training Infra#6.6 MoE x RL]].

**Short answer**:

| Parallelism mode | Gradient communication method | Key technique |
|----------|------------|---------|
| Data Parallel (DP) | AllReduce gradients | Gradient bucketing, meaning communication packed by bucket, with overlap between compute and communication inside each bucket |
| ZeRO-3 / FSDP | ReduceScatter for gradients + AllGather for parameters | AllGather parameters in the forward pass and ReduceScatter gradients in the backward pass |
| Tensor Parallel (TP) | AllReduce in Attention and FFN | AllReduce between column-parallel and row-parallel partitions |
| Pipeline Parallel (PP) | P2P send/recv gradients | 1F1B scheduling, where gradients are passed backward across stages |

**RL-specific aspect**: RL loss comes from advantage-weighted log-probabilities, so the rollout-time logprobs and token-level advantages must be passed to the training side together with the data. This adds extra overhead in data shuffling and packing, which is exactly what slime's Data Buffer is designed to handle.

</details>

---

## D · Rollout Systems (Q21/Q23/Q24/Q25/Q28)

<details class="exercise">
<summary><span class="q-label">Q21</span> <span class="q-text">What are the strategies for KV-cache transfer and multi-GPU communication in distributed inference?</span></summary>

**Scenario**: prefill and decode are deployed separately, meaning PD separation. After prefill is completed on one batch of GPUs, the KV cache must be transferred to the decode GPUs to continue autoregressive generation.

**The challenge of KV-cache transfer**: KV-cache size is $2 \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len} \times n_{\text{layers}}$, which can reach several GB for an 8K sequence on a 32B model. The transfer requires cross-machine network transmission, and the latency is non-negligible.

**Mainstream solutions**:

| Solution | Mechanism | Suitable scenario |
|------|------|---------|
| NCCL P2P | Direct GPU-to-GPU sending through DMA | Same cluster with high-bandwidth NVLink or InfiniBand |
| RDMA (Mooncake/InfiniBand) | CPU bypass, RDMA write directly into the decode GPU | High-performance cross-machine transfer with low latency |
| Disaggregated KV store | Store KV in CPU memory, such as Mooncake, and let decode GPUs fetch it on demand | Ultra-long context where KV exceeds GPU memory |
| Avoid transfer | Co-locate prefill and decode, meaning co-serving | Small-scale, simpler setups |

**KV-cache communication in RL**: AReaL's re-prefill strategy does not transfer KV and instead recomputes under the new weights each time. slime and SGLang's prefix sharing allows the G completions in GRPO to share prefill KV, greatly reducing prefill communication volume.

</details>

<details class="exercise">
<summary><span class="q-label">Q23</span> <span class="q-text">What is the long-tail problem in RL rollout, and how can it be solved?</span></summary>

See the full analysis in [[MLSYS14 Post-Training Infra#6.1 Long-Tail Rollouts and Countermeasures]].

**Short answer**:

The long-tail problem means that a small number of ultra-long sequences in a batch slow down the whole batch. A 1% fraction of 32K-token sequences can leave 99% of the GPUs idle for 4x longer.

**Priority order of solutions**:
1. **Partial rollout / interruptible rollout**: best for agents, multi-round tool use, and long reasoning chains. It preserves the learning signal from slow samples, but requires maintaining prefixes, policy versions, and recovery state.
2. **Length-aware scheduling**: best when prompt lengths differ significantly and task shape is stable. It barely changes the training distribution.
3. **Over-sampling + earliest-completion selection**: best for large-scale GRPO when stable step time matters, but it must avoid biasing the model toward short answers.
4. **Truncation / rejection sampling**: best for early experiments and strictly formatted tasks. It is the simplest engineering option, but wastes sampling compute and may introduce distribution bias.

</details>

<details class="exercise">
<summary><span class="q-label">Q24</span> <span class="q-text">What new problems does continuous batching introduce in RL training? How do vLLM and SGLang differ?</span></summary>

See the full analysis in [[MLSYS14 Post-Training Infra#6.2 New Problems Continuous Batching Introduces in RL]].

**Summary of the key differences**:

- **New RL problems caused by continuous batching**: episode-boundary alignment is difficult; the triggering time of process rewards is complex; KV-eviction pressure is high
- **The most important vLLM vs SGLang difference**: SGLang's RadixAttention is naturally friendly to prefix sharing in GRPO, so the G completions for the same prompt share prefill KV; vLLM requires manual prefix-caching configuration

</details>

<details class="exercise">
<summary><span class="q-label">Q25</span> <span class="q-text">How do you measure utilization in vLLM and SGLang? How do you evaluate KV-cache utilization in RL training?</span></summary>

**Core vLLM metrics**:
- `gpu_cache_usage_perc`: KV-cache page utilization. If it is below 50%, that indicates waste or a low prefix-caching hit rate
- `num_running_seqs`: number of concurrent sequences. Higher means better GPU utilization
- `num_waiting_reqs`: queued backlog. If it stays above 0, inference is the bottleneck

**Core SGLang metrics**:
- `cache_hit_rate`: prefix-cache hit rate. In GRPO it should exceed 80%, because the G completions share the same prompt
- `/get_server_info` API: returns queue depth, tokens per second, and cache stats in real time

**RL-specific things to watch**:
- **Abort ratio**: in interruptible-rollout frameworks, a high abort ratio means training and rollout speeds are badly mismatched
- **Reuse rate** (slime's OPSM): when gradients are computed only on the tokens of the best completion, this reflects effective token utilization
- **Reward variance within group**: variance near 0 means the task is too easy or too hard, so the advantage signal has failed

</details>

<details class="exercise">
<summary><span class="q-label">Q28</span> <span class="q-text">In AReaL or other partial-rollout frameworks, is the KV cache from the previous policy kept?</span></summary>

**AReaL's choice: no, it is not kept; it uses re-prefill.**

**Reason**: the KV cache computed by the old policy used the old Q/K/V projection weights. After the new policy updates the weights, continuing decode with the old KV cache creates attention inconsistency: new Q attends to old K, which is physically invalid. It will not raise an error, but it produces wrong logprobs and degraded generation quality.

**Cost of re-prefill**: the prefix of the partial sequence must be prefetched again, meaning another forward pass. The cost is usually negligible because prefill is compute-bound and runs much faster than decode.

**Comparison with other solutions**:
- **Magistral (Mistral)** uses a weird strategy: keep the KV cache and continue decoding directly under the new weights, effectively creating a mixed KV where one part was generated by the old weights and another part by the new weights. It is efficient, but the algorithmic correctness is questionable.
- **SkyRL / slime abort + prefix-resume**: interrupt the sequence, save the prefix token sequence rather than the KV cache, and rerun prefill from the prefix under the new weights. This is similar to AReaL and is the more correct choice.

</details>

---

## E · Async Frameworks (Q27/Q32/Q33/Q34/Q35)

<details class="exercise">
<summary><span class="q-label">Q27</span> <span class="q-text">What async RL frameworks exist, and what synchronization bottleneck does each one solve?</span></summary>

See the full analysis in [[MLSYS14 Post-Training Infra#6.3 Async RL Design Space]].

**Quick reference table**:

| Framework | Core bottleneck solved | Key mechanism |
|------|---------------|---------|
| **AReaL** | Long-tail sequences + idle time during weight sync | Interruptible rollout + re-prefill + staleness-aware PPO |
| **slime (async)** | Weight-sync dead time | Bucketed NCCL + abort-in-flight + buffer queue |
| **ROLL Flash** | Multi-round waiting in agentic tasks | Async reward server + episode-level buffer |
| **PipelineRL** | Extreme weight-sync overhead | Per-forward-pass weight swap, meaning updates at token granularity |
| **PRIME-RL** | Large staleness across regions or providers | Version tracking + depth bound + IS in one package |
| **veRL (async)** | Basic train-rollout decoupling | Disaggregated deployment + bounded buffer |

</details>

<details class="exercise">
<summary><span class="q-label">Q32</span> <span class="q-text">How do AReaL and slime differ in their understanding of the RL rollout bottleneck?</span></summary>

See the detailed analysis in [[MLSYS14 Post-Training Infra#5.5 AReaL (Ant Group & Tsinghua IIIS)]] and [[MLSYS14 Post-Training Infra#5.4 slime (THUDM/Zhipu)]].

**One-sentence summary**:
- **slime**: the bottleneck is dead time during weight sync plus inference-engine efficiency -> faster transfer through bucketed NCCL + a better engine through SGLang
- **AReaL**: the bottleneck is that long-tail sequences slow down the whole batch -> interruptible rollout so slow sequences do not block everything

Both views are correct; they target different scenarios, medium-length RLVR versus ultra-long CoT or agentic settings.

</details>

<details class="exercise">
<summary><span class="q-label">Q33</span> <span class="q-text">How should staleness be understood in fully async RL, and what values are typical in practice?</span></summary>

**Mathematical definition of staleness**: if samples were generated with policy version $\pi_k$ and the current training policy is $\pi_{k+s}$, then the step gap $s$ between them is staleness.

**Effect of staleness**:
- Algorithmically: the IS ratio $\pi_{k+s}/\pi_k$ drifts away from 1, so the gradient estimate becomes biased
- Magnitude of the bias: the larger $s$ is, the larger the gap between $\pi_{k+s}$ and $\pi_k$, the wider the IS-ratio distribution, and the higher the variance
- AReaL experiments: when $s \leq 8$, final performance is not affected because IS-ratio drift stays within the clipping range

**Typical practical values**:
- Synchronous systems: $s = 0$, strictly on-policy
- Double-buffer: $s \in \{0, 1\}$
- Bounded queue with depth K: $s \in [0, K]$, with mean about $K/2$
- **AReaL in practice**: mean $s \approx 2-4$ steps, max $s \leq 8$
- PRIME-RL across regions: $s$ can reach 10+ steps, so IS correction and version gating are required

**Control methods**:
1. Upper bound on buffer depth, which is a hard constraint and drops data beyond the limit
2. IS-weight clipping, restricting $r_t$ to $[c_{\min}, c_{\max}]$, for example $[0.1, 10]$
3. Actively balancing rollout and training speed so the mean staleness stays stable

</details>

<details class="exercise">
<summary><span class="q-label">Q34</span> <span class="q-text">How does data flow through slime? How is it integrated with Megatron? How is the loss computed?</span></summary>

See the full data-flow explanation in [[MLSYS14 Post-Training Infra#5.4 slime (THUDM/Zhipu)]].

**Short answer** (6 steps):

1. **SGLang generation**: the rollout engine receives a prompt and outputs the completion, per-token logprobs, and finish reason
2. **Reward computation**: the Data Buffer calls a reward verifier, meaning rules or an RM, and obtains sequence-level or token-level reward
3. **Advantage computation**: the Data Buffer computes GRPO advantage within the group through z-score normalization
4. **Data packing**: the Data Buffer packs `{prompt, completion, logprobs, advantage}` into a format Megatron can accept, using variable-length sequence packing to maximize token utilization per batch
5. **Megatron forward/backward**:
   - Forward: recompute logprobs $\pi_\theta$ using the current actor, while also running the reference model $\pi_{\text{ref}}$
   - Loss = $\text{GRPO}_{\text{clip}}(\pi_\theta / \pi_{\text{old}}, \hat{A}) + \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})$, with optional OPSM masking so gradients are computed only on the tokens of the best completion
6. **Weight broadcast**: after Megatron finishes the optimizer step, it broadcasts to the SGLang cluster through **bucketed NCCL**, one 1GB bucket at a time; SGLang then hot-updates through the `update_weights` API

**OPSM masking**: compute loss only on the tokens of the highest-reward completion in the group, and set the gradient of all other completions to 0. Physically, this means learning only from the correct sample and not learning punishment from wrong samples, under the assumption that the punishment signal is too noisy.

</details>

<details class="exercise">
<summary><span class="q-label">Q35</span> <span class="q-text">If you had to choose among veRL, TRL, Unsloth, AReaL, and slime, how would you choose?</span></summary>

See the full decision tree in [[MLSYS14 Post-Training Infra#7.1 Q35 Direct Answer]].

**Quick decision**:

| Need | Recommendation |
|------|------|
| Single machine, quick validation | **TRL** |
| Single-machine LoRA, maximum efficiency | **Unsloth** for SFT-like tasks, or TRL |
| Multi-node, <=70B, standard RLHF | **veRL** with colocate + FSDP |
| Multi-node, 100B+ dense or MoE | **slime** with Megatron + SGLang, the most production-validated option |
| Long CoT, outputs >8K, maximum throughput | **AReaL** with fully async + interruptible rollout |
| Agentic, multi-tool, complex scaffold | **ROLL** with mature five-role agentic support |

**Unsloth's position**: it focuses on single-GPU LoRA training efficiency for SFT and DPO. It does not support multi-node RL and is not suitable for large-scale RL training. Comparing it directly to veRL or slime is a category error.

</details>

---

## F · Advanced Systems (Q29/Q30/Q31)

<details class="exercise">
<summary><span class="q-label">Q29</span> <span class="q-text">How does Expert Parallelism affect MoE throughput?</span></summary>

See the EP analysis in [[MLSYS14 Post-Training Infra#6.6 MoE x RL]].

**Core formula**: the key overhead of EP is AllToAll token routing, and its communication volume is:
$$
\text{AllToAll volume} = \text{batch\_size} \times d_{\text{model}} \times 2 \times \text{top-k}
$$

**EP break-even point**: when the average number of tokens processed per expert, $n_{\text{tokens/expert}}$, is at least 8-16, expert compute time is enough to overlap AllToAll communication. Below this threshold, which is common in decode where batch size is often only 1-8 tokens per expert, the communication cost of EP exceeds its benefit.

**Practical conclusion**:
- Prefill, with large batches: EP gives significant benefits and communication can be overlapped
- Decode, with small batches: EP is usually avoided and TP is used instead; each expert may process only 1-2 tokens, so AllToAll latency is unacceptable
- RL rollout is almost entirely decode -> rollout is generally recommended to use TP instead of EP

</details>

<details class="exercise">
<summary><span class="q-label">Q30</span> <span class="q-text">How should compute-communication overlap be designed for long-context training? What is the difference between Megatron and FSDP in parallel strategy?</span></summary>

See the long-context and parallel-strategy analysis in [[MLSYS14 Post-Training Infra#6.6 MoE x RL]].

**Core techniques for compute-communication overlap**:

| Technique | Principle | Suitable scenario |
|------|------|---------|
| Overlap AllReduce with matrix multiplication | Launch async AllReduce and immediately start computing the next layer | DP, which requires gradient bucketing |
| AllGather prefetch (FSDP) | During backward, pre-AllGather the next layer's parameters | FSDP ZeRO-3 |
| Ring attention (CP) | Under Sequence Parallelism, GPUs pass KV around the ring, and each local Q attends to the KV it receives | Ultra-long sequences |
| Hide pipeline bubbles | 1F1B + interleaving; fill bubbles with microbatches | PP scenarios |

**Core difference between Megatron and FSDP**:
- **Megatron**: natively supports arbitrary combinations of TP + PP + CP + EP; provides fine-grained pipeline schedules such as 1F1B, interleaved 1F1B, and virtual pipeline; implements MoE EP correctly
- **FSDP2**: supports only DP + TP + CP, with no PP and no EP; simpler and easier to use, suitable for <=70B dense models; has native ZeRO-style sharding support

**Conclusion**: if you need an ultra-large MoE or any setup that requires PP, you must use Megatron. For research settings or models <=70B, FSDP is more convenient.

</details>

<details class="exercise">
<summary><span class="q-label">Q31</span> <span class="q-text">How do you implement deterministic execution? What is batch invariance? What causes it? Can atomic add solve the problem?</span></summary>

**Definition of batch invariance**: given the same input token sequence, the model should produce exactly the same logprob regardless of batch size. Violating batch invariance creates systematic error in RL IS ratios.

**Sources that break batch invariance**:

1. **Nondeterminism from atomic add**: on GPUs, parallel threads use `atomicAdd` to sum into shared memory. Floating-point addition is not associative, so different thread schedules produce different results. This affects operations such as LayerNorm, where mean and variance require reductions, and softmax.

2. **Different padding behavior at different batch sizes**: if padding tokens are handled inconsistently, attention masks differ and outputs differ.

3. **Dropout**: a nondeterministic operation that should be disabled at RL inference time, though training forward passes usually keep dropout enabled.

4. **Numerical nondeterminism in Flash Attention**: different Flash Attention versions handle blocked softmax precision somewhat differently.

**Can atomic add solve it?** **Partially**:

- `atomicAdd` can be replaced by a deterministic ordered accumulation, but that is slower; PyTorch provides `torch.use_deterministic_algorithms(True)` to force deterministic operators
- However, enforcing determinism disables some efficient kernels, such as parts of cuDNN, and **significantly reduces training speed**, typically by 10-30%
- **Practical choice**: most frameworks do not force determinism. Instead, they tolerate small mismatch through IS clipping rather than chasing perfect determinism

**Real solution**: Keep Routing + Keep Sampling Mask, as in Q11, or use exactly the same operator implementation so training and inference share one kernel stack.

</details>

---

Reference: [Sheriyuo · RL Interview Questions 2026](https://x.com/sheriyuo/status/2063295181131247674)

Back to main document: [[MLSYS14 Post-Training Infra]]
