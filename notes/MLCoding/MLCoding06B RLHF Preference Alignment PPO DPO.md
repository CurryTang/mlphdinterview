# 06B · RLHF 与偏好对齐全景

在大语言模型（LLM）的生命周期中，预训练赋予了模型海量的世界知识与语言建模能力，但模型此时仍然只是一个“下一个词补全机器”。为了使模型具备指令遵循（Instruction-Following）、安全合规（Harmlessness）以及真实有用（Helpfulness & Honesty）的人类意图对齐能力，**后训练人类偏好对齐（Post-Training Alignment）** 构成了现代大模型工程最关键的技术壁垒。

本篇系统梳理偏好对齐的 5 大核心体系：
1. **经典三阶段 RLHF 流水线（SFT $\to$ Reward Modeling $\to$ RL 策略优化）**
2. **PPO 4 模型并发系统架构（Actor / Critic / Reward / Reference）与 GAE 优势估计推导**
3. **DPO（Direct Preference Optimization）闭式隐式奖励数学推导与梯度动态**
4. **现代对齐全家桶演进（DPO vs IPO vs KTO vs ORPO vs SimPO）**
5. **对齐陷阱与生产治理（奖励黑客 Reward Hacking、长度偏见 Verbosity Bias 与对齐税 Alignment Tax）**

---

## 模块一：经典三阶段 RLHF 流水线全景

```text
经典三阶段 RLHF 演进流水线：
┌────────────────────────────────────────────────────────────────────────┐
│ 阶段一：监督微调 (Supervised Fine-Tuning, SFT)                         │
│ • 数据: 高质量人工标注的指令-回答对 (Prompt, Response)                │
│ • 目标: 激发模型遵循指令的基本格式与对话能力                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 阶段二：奖励建模 (Reward Modeling, RM)                                 │
│ • 数据: 对同一 Prompt 的多个模型回答，标注人类偏好排名 (y_w ≻ y_l)      │
│ • 目标: 训练标量奖励打分模型 r_ψ(x, y)，模拟人类偏好价值判断          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 阶段三：强化学习策略优化 (RL Fine-Tuning via PPO)                      │
│ • 机制: 策略模型生成回答 -> RM 给标量奖赏 -> 计算 KL 惩罚与 GAE 优势   │
│ • 目标: PPO 迭代更新策略模型参数，最大化人类偏好期望奖励               │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 阶段二：Bradley-Terry 偏好模型与奖励建模损失

人类很难对单个回答打出绝对准确的分数（如 8.7 分），但非常擅长在两个回答中进行**成对相对优劣比较（Pairwise Preference Comparison）**。

#### Bradley-Terry 偏好概率建模
给定提示词 $x$，人类偏好的优质回答为 $y_w$（winner），劣质回答为 $y_l$（loser）。假设存在一个真实的隐式标量奖励函数 $r^*(x, y)$，人类偏好 $y_w$ 优于 $y_l$ 的概率服从 Bradley-Terry 模型：

$$P(y_w \succ y_l \mid x) = \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) = \frac{1}{1 + e^{-(r_\psi(x, y_w) - r_\psi(x, y_l))}}$$

#### 奖励模型目标损失函数（Binary Ranking Loss）
为了训练参数为 $\psi$ 的奖励模型 $r_\psi$，我们在偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$ 上最大化对数似然，即最小化负对数似然损失：

$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]$$

- **梯度行为**：当奖励模型预测错误（$r_\psi(x, y_w) < r_\psi(x, y_l)$）时，$\sigma(\cdot)$ 导数很大，强力将 $r_\psi(x, y_w)$ 调高、将 $r_\psi(x, y_l)$ 压低；
- **$K$-way 排名扩展**：对于包含 $K$ 个候选回答的排序列表，可将其拆分为 $\binom{K}{2}$ 个二元对联合训练。

---

## 模块二：PPO 4 模型并发系统架构与 GAE 优势估计

### 1. PPO-style RLHF 的 4 大并发模型架构

在阶段三的 PPO 训练中，系统必须在 GPU 集群中同时维护 **4 个不同职责的大模型**：

```text
PPO 4 模型并发运行时拓扑：
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Actor Model (π_θ, 策略模型):                                        │
│    • 状态: 激活训练 (梯度反向传播更新)                                 │
│    • 职责: 接收 Prompt x，自回归采样生成 Response y                    │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Critic / Value Model (V_ϕ, 价值模型):                               │
│    • 状态: 激活训练 (梯度反向传播更新)                                 │
│    • 职责: 评估每个 Token 状态的基线期望收益 V(s_t)，用于计算 GAE 优势 │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Reward Model (r_ψ, 奖励模型):                                       │
│    • 状态: 冻结 (Inference Only)                                       │
│    • 职责: 对生成的全序列 (x, y) 给出外部标量偏好打分                  │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Reference Model (π_ref, 初始 SFT 参考模型):                         │
│    • 状态: 冻结 (Inference Only)                                       │
│    • 职责: 计算每个 Token 的参考概率，提供 KL 散度惩罚，防止策略走偏   │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Token-Level 奖励与 KL 散度动态惩罚

为了防止策略模型 $\pi_\theta$ 过度迎合奖励模型的漏洞而产生严重的分布漂移（Reward Hacking）或退化为不可读的胡言乱语，RLHF 在每个 Token 步引入了**参考策略 KL 散度惩罚**：

$$R_t = \begin{cases} -\beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})_t = -\beta \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}, & t < T \\ r_\psi(x, y) - \beta \log \frac{\pi_\theta(y_T \mid x, y_{<T})}{\pi_{\text{ref}}(y_T \mid x, y_{<T})}, & t = T \text{ (序列末尾)} \end{cases}$$

- $\beta$ 是 KL 惩罚系数（超参数，通常取 $0.01 \sim 0.1$）；
- 标量外部奖励 $r_\psi(x, y)$ 只在最后一个 Token 释放，中间 Token 的即时奖励纯粹由 KL 散度惩罚构成。

---

### 3. 广义优势估计（GAE, Generalized Advantage Estimation）与偏置–方差权衡

在策略梯度方法中，优势函数 $\hat{A}(s, a) = Q(s, a) - V(s)$ 用于度量特定动作相对平均水准的净收益。由于真实状态价值与动作价值不可直接获知，GAE（Schulman et al., 2015）通过引入指数衰减加权，在单步时序差分（TD）与全轨迹蒙特卡洛（Monte Carlo）回报之间构建了连续的插值权衡。

#### 1. 时序差分误差与 $k$ 步优势估计

设状态价值网络为 $V_\phi(s)$，单步时序差分误差（TD Error）定义为：
$$\delta_t^V = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

当向前推进不同步数时，可推导出对应跨度的 $k$ 步优势估计：
$$\hat{A}_t^{(1)} = \delta_t^V = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
$$\hat{A}_t^{(2)} = \delta_t^V + \gamma \delta_{t+1}^V = R_t + \gamma R_{t+1} + \gamma^2 V_\phi(s_{t+2}) - V_\phi(s_t)$$
$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l R_{t+l} + \gamma^k V_\phi(s_{t+k}) - V_\phi(s_t)$$
$$\hat{A}_t^{(\infty)} = \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = \sum_{l=0}^\infty \gamma^l R_{t+l} - V_\phi(s_t)$$

#### 2. GAE 指数加权平均与递推形式

GAE 定义为所有 $k$ 步优势估计关于参数 $\lambda \in [0, 1]$ 的指数加权平均：
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat{A}_t^{(k)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

在有限长度轨迹（长度为 $T$）中，GAE 满足反向递推形式，计算复杂度为 $O(T)$，适合 GPU 倒序向量化扫描：
$$\hat{A}_t^{\text{GAE}} = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}^{\text{GAE}}$$

#### 3. $\lambda$ 在偏置（Bias）与方差（Variance）上的权衡机制

超参数 $\lambda \in [0, 1]$ 决定了估计量在环境转移随机性与价值网络模型误差之间的分配：

- **$\lambda = 0$（极端单步 TD / 低方差，高偏差）**：
  $$\hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t^V = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
  - **方差极小**：仅依赖当前单步即时环境奖励与一次状态转移，不累积后续长程随机性；
  - **偏差极高**：估计完全受限于价值网络 $V_\phi$ 本身。若 Critic 预测存在偏差（如训练早期未收敛），该偏差将全额注入策略梯度，导致更新方向持续偏移。
- **$\lambda = 1$（极端蒙特卡洛 MC / 高方差，无偏差）**：
  $$\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^\infty \gamma^l R_{t+l} - V_\phi(s_t)$$
  - **无理论偏差**：回报完全由真实环境轨迹的采样总收益给出；减去基线 $V_\phi(s_t)$ 不改变策略梯度的期望（$\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot V(s)] = 0$）；
  - **方差极大**：长序列中每一步动作采样随机性与环境转移噪声不断累积相乘，梯度方差随轨迹长度急剧膨胀，需要大量采样才能压低方差。
- **$\lambda \in (0, 1)$（连续折中平衡）**：
  - 几何级数衰减因子 $(\gamma \lambda)^l$ 使模型对近处的确定性收益赋予高权重，对远期高方差噪声进行指数级抑制；
  - 用轻微受控的 Critic 模型偏差，换取方差的大幅下降。在大语言模型 RLHF 训练中，通常配置 $\gamma = 1.0, \lambda \in [0.95, 0.98]$。

---

### 4. 信任域策略优化（TRPO, Trust Region Policy Optimization）理论根基

在标准策略梯度（Vanilla Policy Gradient）中，参数更新直接在欧几里得参数空间中沿梯度方向移动：$\theta_{\text{new}} = \theta_{\text{old}} + \alpha \nabla_\theta J(\theta)$。这种做法存在根本缺陷：步长过小导致收敛停滞，步长过大则可能导致策略瞬间退化进入低收益分布，采集到的后续样本全盘失真，引发不可逆的策略崩溃（Policy Collapse）。TRPO（Schulman et al., 2015）给出了具备单调提升保证的理论解法。

#### 1. 要优化什么：Surrogate 替代目标与单调提升定理

强化学习目标是最大化期望回报 $\eta(\pi) = \mathbb{E}_{\tau \sim \pi}[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)]$。根据 Kakade & Langford 策略改进恒等式：
$$\eta(\pi) = \eta(\pi_{\text{old}}) + \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t A^{\pi_{\text{old}}}(s_t, a_t) \right] = \eta(\pi_{\text{old}}) + \sum_s \rho_\pi(s) \sum_a \pi(a \mid s) A^{\pi_{\text{old}}}(s, a)$$

由于新策略的状态访问分布 $\rho_\pi(s)$ 无法在采样前获取，TRPO 将其替换为旧策略状态分布 $\rho_{\pi_{\text{old}}}(s)$，构造**替代目标函数（Surrogate Objective）**：
$$L_{\pi_{\text{old}}}(\pi) = \eta(\pi_{\text{old}}) + \sum_s \rho_{\pi_{\text{old}}}(s) \sum_a \pi(a \mid s) A^{\pi_{\text{old}}}(s, a) = \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi(a \mid s)}{\pi_{\text{old}}(a \mid s)} A^{\pi_{\text{old}}}(s, a) \right]$$

Schulman 等人给出了真实回报与替代目标之间的理论下界：
$$\eta(\pi) \ge L_{\pi_{\text{old}}}(\pi) - C \cdot D_{\text{KL}}^{\max}(\pi_{\text{old}}, \pi), \quad \text{其中 } C = \frac{4 \epsilon \gamma}{(1 - \gamma)^2}, \quad \epsilon = \max_{s, a} |A^{\pi_{\text{old}}}(s, a)|$$

**理论保证**：只要新旧策略之间的 KL 散度足够小，最大化替代目标 $L_{\pi_{\text{old}}}(\pi)$ 必然能够保证真实策略期望回报 $\eta(\pi)$ 单调递增。

#### 2. 为什么需要 KL 信任域：参数空间与分布流形的几何失真

标准梯度下降在参数空间度量欧氏步长 $\|\Delta \theta\|_2$，但深度神经网络参数与其输出的动作概率分布之间是非线性映射：
- 在某些高曲率参数子空间，极微小的欧氏位移 $\|\Delta \theta\|_2 < 10^{-4}$ 即可引起动作概率分布的断崖式剧变；
- 一旦某次过大更新破坏了策略分布，策略将持续生成劣质动作，导致后续收集到的轨迹全为无效样本，无法自行恢复。

因此，步长限制必须建立在**概率分布流形**上。TRPO 将理论目标转化为平均状态 KL 散度的硬约束优化问题：
$$\max_\theta \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A^{\pi_{\theta_{\text{old}}}}(s, a) \right] \quad \text{s.t.} \quad \bar{D}_{\text{KL}}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta) \le \delta$$

#### 3. 怎么近似计算：二阶展开、Fisher 信息矩阵与共轭梯度（CG）

该带非线性约束的优化问题无法直接闭式求解，TRPO 在当前参数 $\theta_{\text{old}}$ 处进行局部泰勒展开：
1. **目标函数一阶展开**：
   $$L(\theta) \approx L(\theta_{\text{old}}) + g^T (\theta - \theta_{\text{old}}), \quad g = \nabla_\theta L(\theta)\big|_{\theta = \theta_{\text{old}}}$$
2. **KL 约束二阶展开**：
   $$\bar{D}_{\text{KL}}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta) \approx \frac{1}{2} (\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}})$$
   （由于 $\theta = \theta_{\text{old}}$ 时 KL 散度为 0 且取得极小值，一阶导恒为 0；二阶导即为 Fisher 信息矩阵 $F$）：
   $$F = \mathbb{E}_{s \sim \rho, a \sim \pi} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)^T \right]$$

通过拉格朗日乘子法，可推导得**自然策略梯度（Natural Policy Gradient）**解析解：
$$\Delta \theta = \theta - \theta_{\text{old}} = \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g$$

##### 工业近似求解机制
当参数维度 $d$ 达到数百万至数十亿时，显式构建 $F \in \mathbb{R}^{d \times d}$ 并直接求逆 $F^{-1}$ 需 $O(d^3)$ 复杂度，计算不可行。TRPO 采用两步工程机制：
- **共轭梯度法（Conjugate Gradient, CG）**：线性方程组 $F x = g$ 无需显式求逆。CG 算法在 Krylov 子空间内迭代搜索，单次迭代仅需计算 Fisher-向量积（$F v$）。利用恒等式 $F v = \nabla_\theta \left( (\nabla_\theta \bar{D}_{\text{KL}})^T v \right)$，仅需两次反向传播即可算出向量积，通常经过 10~20 步 CG 迭代即可获得高精度解 $x \approx F^{-1} g$；
- **回溯线搜索（Backtracking Line Search）**：由于泰勒展开存在高阶近似误差，沿解出方向尝试衰减步长 $\theta = \theta_{\text{old}} + \alpha^j \Delta \theta$（$\alpha \in (0, 1)$），逐次验证是否同时满足目标函数提升（$L(\theta) \ge L(\theta_{\text{old}})$）与未近似的真实 KL 散度硬约束（$\bar{D}_{\text{KL}} \le \delta$）。

---

### 5. 近端策略优化（PPO-Clip）与截断目标函数

虽然 TRPO 具备优异的单调收敛理论保障，但二阶 Fisher 矩阵计算、共轭梯度和线搜索使得它难以并行化，且与现代基于一阶自动微分的深度学习优化器（如 Adam/AdamW）难以融合。PPO（Schulman et al., 2017）通过设计一阶可微的截断替代目标，彻底取代了复杂的二阶优化流程。

#### 1. 相对 TRPO 简化了什么

1. **从二阶降为纯一阶计算**：彻底去除 Fisher 信息矩阵、共轭梯度迭代（CG）与回溯线搜索，直接使用一阶梯度与 Adam 优化器完成更新；
2. **硬约束转为可微截断**：不再求解复杂的带约束拉格朗日优化，直接在目标函数内施加截断机制；
3. **支持多轮 Minibatch 样本复用**：TRPO 单次采集的数据通常只能执行一次参数更新；而 PPO 引入重要性采样比与截断保护，允许在同一批 Rollout 轨迹上执行多个 Epoch 的 Minibatch 随机梯度下降更新，大幅提升样本利用率与吞吐。

#### 2. PPO-Clip 目标函数与悲观下界解析

定义重要性采样比（Importance Sampling Ratio）：
$$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$$

PPO-Clip 目标函数为：
$$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中外层的 $\min$ 操作构建了保守的**悲观下界（Pessimistic Lower Bound）**。该目标函数随优势值 $\hat{A}_t$ 的正负呈现非对称保护机制：

- **正优势样本（$\hat{A}_t > 0$，动作优于基准，需要增加采样概率）**：
  $$\min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) = \min\left( r_t \hat{A}_t, \, (1+\epsilon)\hat{A}_t \right)$$
  - 当 $r_t \le 1+\epsilon$ 时：目标函数为 $r_t \hat{A}_t$，梯度正常流动，推动 $\pi_\theta$ 概率增加；
  - 当 $r_t > 1+\epsilon$ 时：目标函数被截断为常数 $(1+\epsilon)\hat{A}_t$，对参数 $\theta$ 的梯度归零；
  - **核心机制**：**防止对好动作“过度奖励”**。避免单批样本因奖励极高使策略迈出过大步子，破坏策略分布稳定性。
- **负优势样本（$\hat{A}_t < 0$，动作劣于基准，需要降低采样概率）**：
  - 由于 $\hat{A}_t$ 为负数，原函数与截断函数的大小关系翻转：
  $$\min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) = \min\left( r_t \hat{A}_t, \, (1-\epsilon)\hat{A}_t \right)$$
  - 当 $r_t \ge 1-\epsilon$ 时：目标函数为 $r_t \hat{A}_t$，产生负梯度压低不良动作概率；
  - 当 $r_t < 1-\epsilon$ 时：由于负负相乘，较小项为 $(1-\epsilon)\hat{A}_t$。目标函数再次被截断为常数，梯度归零；
  - **核心机制**：**防止对劣动作“过度惩罚”**。避免单步更新将某些动作概率直接压至极小甚至归零，造成探索能力退化与数值溢出。
- **为什么必须取 $\min$ 形成悲观下界**：
  - 若仅使用单纯的截断项 $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t$：当 $\hat{A}_t > 0$ 但更新导致 $r_t < 1-\epsilon$（策略反而退步）时，截断项会虚报目标值；
  - 外层 $\min$ 确保了当新策略表现劣于未截断预期时，选择更悲观的估计值，形成单向防御屏障。

## 模块三：DPO（Direct Preference Optimization）闭式隐式奖励数学推导

尽管 PPO 在理论上完备，但工程上面临着**4 个模型常驻显存、训练极不稳定、Critic 网络难以收敛、超参数极其敏感**等严重痛点。

Rafailov 等人（NeurIPS 2023）提出的 **DPO** 彻底颠覆了这一范式：**利用数学解析推导，直接将奖励函数重参数化为策略模型本身的输出对数概率，从而完全废除了独立的 Reward Model 与 Critic 价值网络！**

```text
PPO 4 模型强化学习闭环 vs DPO 单一分类对数损失：
┌────────────────────────────────────────────────────────┐
│ PPO: Actor + Critic + Reward + Reference (4 模型并发)  │
│ 流程: 复杂强化学习循环 -> 采样 -> 打分 -> GAE -> 截断更新│
└───────────────────────────┬────────────────────────────┘
                            ▼ 革命性简化
┌────────────────────────────────────────────────────────┐
│ DPO: 仅保留训练模型 π_θ 与冻结参考模型 π_ref (2 个模型)│
│ 流程: 离线直接计算闭式二元交叉熵损失，极速稳定收敛      │
└────────────────────────────────────────────────────────┘
```

### 1. DPO 核心数学推导（The Mathematical Derivation）

#### 第一步：带 KL 正则项的强化学习最优策略解析解
标准的 RL 优化目标为：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot \mid x)} \left[ r(x, y) \right] - \beta \mathbb{D}_{\text{KL}}(\pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))$$

通过变分法（Calculus of Variations）可严格求解出其最优策略 $\pi^*$ 的闭式解：

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$ 为配分函数（Partition Function）。

#### 第二步：反解真实隐式奖励函数（Implicit Reward）
对上述公式两边取自然对数并重新整理，可得到真实奖励 $r(x, y)$ 与最优策略 $\pi^*$ 的解析映射关系：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

#### 第三步：代入 Bradley-Terry 偏好模型（配分函数奇迹相消）
将隐式奖励公式代入 Bradley-Terry 偏好对数概率公式中：

$$P(y_w \succ y_l \mid x) = \sigma\left( r(x, y_w) - r(x, y_l) \right)$$

$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right)$$

注意到，**依赖于输入 $x$ 的配分函数 $\beta \log Z(x)$ 在相减过程中被精确抵消！**

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

#### 第四步：构建 DPO 策略损失函数
直接用可学习的策略模型 $\pi_\theta$ 替代最优策略 $\pi^*$，构建全样本的负对数似然损失：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

---

### 2. DPO 梯度的自适应加权动态

对参数 $\theta$ 求梯度，可揭示 DPO 的内在优化动力学：

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \mathbb{E} \left[ \underbrace{\sigma\left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \right)}_{\text{动态权重系数 } w(x, y_w, y_l)} \cdot \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right) \right]$$

- **当模型预测严重错误时（$\hat{r}_\theta(x, y_w) \ll \hat{r}_\theta(x, y_l)$）**：
  权重 $w \to 1$，梯度以最大力度增大 $y_w$ 的概率、压低 $y_l$ 的概率；
- **当模型已经完全掌握正确偏好时（$\hat{r}_\theta(x, y_w) \gg \hat{r}_\theta(x, y_l)$）**：
  权重 $w \to 0$，梯度自动衰减至零，避免对已经正确分类的样本过度更新。

---

## 模块四：现代偏好对齐全家桶技术对比

在大模型发展历程中，针对 DPO 的泛化性、参考模型依赖性与数据需求，衍生出了丰富的对齐算法家族：

| 对齐算法 | 核心机制与创新点 | 目标损失形式 | Reference Model 需求 | 工业界核心优势与适用场景 |
|---|---|---|---|---|
| **PPO** | 经典强化学习 Actor-Critic + GAE 优势估计 | $\mathbb{E}[\min(r_t A_t, \text{clip} \cdot A_t)]$ | **必须 (4 模型)** | 在线探索能力强，适合超长多轮对话与复杂动态奖励环境 |
| **DPO** | 隐式奖励重参数化，闭式成对分类损失 | $-\log \sigma(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$ | **必须 (2 模型)** | **工业界通用对齐绝对主流**，训练极度稳定，显存开销小 |
| **IPO** | 在 DPO 损失上增加平方正则项，防止策略过拟合偏好数据 | $(\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} - \frac{1}{2\tau})^2$ | **必须 (2 模型)** | 解决 DPO 在极端偏好对上的概率发散与过拟合问题 |
| **KTO** | 基于前景理论（Prospect Theory），支持单点二元标签（Like/Dislike） | 分别对单个好样本与坏样本优化效用期望 | **必须 (2 模型)** | **无需成对数据**，适用于实际业务中只有点赞/点踩日志的冷启动对齐 |
| **ORPO** | 将 SFT 交叉熵与优势几率比（Odds Ratio）惩罚融为一体 | $\mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{OddsRatio}}$ | **无需 (1 模型)** | 单阶段完成 SFT + 对齐，彻底消除了参考模型显存开销 |
| **SimPO** | 使用生成长度归一化的平均 Logits 差，结合 Target Margin 惩罚 | $-\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l) - \gamma \right)$ | **无需 (1 模型)** | **当前开源评测 SOTA**，彻底摆脱参考模型，且内生性消除长度偏见 |

## 模块五：RLHF 对齐陷阱、评测体系与生产治理实战

### 1. 偏好对齐四大生产陷阱与防御体系

```text
偏好对齐四大生产陷阱与防御体系：
┌───────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 陷阱类型                  │ 现象机理与生产防御治理手段                                             │
├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. 奖励黑客 (Reward Hack) │ 模型钻奖励模型的空子，生成空洞客套、排版华丽但毫无信息的无用长文。     │
│                           │ ➔ 防御: 严格限制 KL 散度 $\beta$；在 RM 中引入长度惩罚与规则打分约束。 │
├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. 长度偏见 (Verbosity)   │ 奖励模型和人类裁判天然偏好长回答（长文本显得更加详尽）。               │
│                           │ ➔ 防御: 采用 SimPO 的长度归一化奖赏；在训练数据中强制注入长负样本与短正样本。│
├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. 对齐税 (Alignment Tax) │ 经历安全对齐后，基础逻辑推理、数学解题与代码能力发生严重回退。         │
│                           │ ➔ 防御: 在对齐阶段混合一定比例（10%~20%）的高质量通用预训练与推理数据。│
├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. 过度拒答 (Over-Refusal)│ 模型对包含“爆炸”、“杀毒”、“攻击”等中性词的合法学术提问产生过度恐慌并拒答。│
│                           │ ➔ 防御: 构建大规模边界对抗安全数据集（XSTest），专门训练模型的合规辨识度。│
└───────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 2. LLM-as-a-Judge 评测体系与三大固有偏见治理

在后训练对齐效果的自动化评测与成对偏好标注中，工业界广泛使用能力更强的前沿大模型（如 GPT-4o、Claude-3.5-Sonnet）作为裁判（LLM-as-a-Judge）对候选回答进行多维度评分或成对比较（Pairwise Comparison）。然而，裁判模型存在以下三大严重固有偏见：

```text
LLM-as-a-Judge 三大固有偏见与工业界防御对策：
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 偏见类型 (Biases)       │ 现象机理与工业界标准防御手段                                           │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. 位置偏见 (Position)  │ 倾向于给排在前面的候选者（Candidate 1）打更高分。                      │
│                         │ ➔ 防御手段: 进行 Pairwise 位置对调 (Swap Order) 双向打分并取平均。    │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. 冗长偏见 (Verbosity) │ 倾向于给篇幅更长、排版更丰富但可能废话连篇的回答打高分。               │
│                         │ ➔ 防御手段: 在 Prompt 中明确长度约束，或对字数进行长度惩罚正则化。    │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. 自我偏好 (Self-Enhance) 倾向于给自己模型家族生成的回答打更高分（由于自注意力特征喜好一致）。│
│                         │ ➔ 防御手段: 引入多裁判委员会交叉盲审，或使用带标准参考答案的裁判 Prompt│
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 模块六：面试高频必背速答题

### Q1：为什么 DPO 可以在数学上完全舍弃独立的 Reward Model？
> **答**：
> 1. 根据 KL 正则化强化学习的数学极值条件，最优策略 $\pi^*$ 与真实隐式奖励 $r^*(x, y)$ 存在严格的闭式解析映射：$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$；
> 2. 当我们将该解析式代入 Bradley-Terry 偏好模型中计算两个回答的奖励差值时，难以计算的配分函数 $Z(x)$ 被精确抵消；
> 3. 因此，我们可以直接用策略模型本身的对数似然比来表达偏好概率，将强化学习策略优化直接转化为单阶段的二元分类对数损失，无需训练和存储任何独立的 Reward Model。

### Q2：什么是对齐税（Alignment Tax）？如何从训练数据与策略层面进行缓解？
> **答**：
> 1. **定义**：模型在经过激进的人类偏好对齐（RLHF/DPO）后，由于策略分布被强行压缩到安全和人类偏好的狭窄子空间内，导致模型的通用基础能力（如代码编写、复杂多跳数学推理、知识泛化）发生退化的现象；
> 2. **缓解方案**：
>    - **数据回放（Data Replay）**：在对齐损失中混合 10%~20% 的预训练高质量语言建模与数学代码 SFT 数据；
>    - **多阶段解耦（Decoupled Post-Training）**：采用类似 DeepSeek-R1 的路径——先使用可验证奖励（RLVR）最大化释放数学代码推理能力，最后再用轻量温和的偏好对齐微调安全与通用人设。

### Q3：TRPO 的优化目标是什么？为什么需要引入 KL 信任域？如何通过共轭梯度（CG）近似计算？
> **答**：
> 1. **优化目标**：基于 Kakade-Langford 策略改进恒等式构建的替代目标函数（Surrogate Objective）：
>    $$L_{\pi_{\text{old}}}(\pi) = \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi(a \mid s)}{\pi_{\text{old}}(a \mid s)} A^{\pi_{\text{old}}}(s, a) \right]$$
>    根据理论单调提升下界 $\eta(\pi) \ge L_{\pi_{\text{old}}}(\pi) - C \cdot D_{\text{KL}}^{\max}(\pi_{\text{old}}, \pi)$，只要新旧策略的 KL 散度受控，最大化该替代目标即可严格保证真实期望回报单调提升；
> 2. **为什么需要 KL 信任域**：
>    - 标准策略梯度在参数空间施加欧氏距离步长限制，但深度神经网络参数与其输出动作概率分布之间是非线性的。某些高曲率方向上极小的参数步长即可造成策略分布断崖式剧变；
>    - 一旦单步更新过大进入劣质分布，后续采样数据将全盘恶化，导致不可逆的“策略崩溃（Policy Collapse）”；
>    - 因此必须直接在概率分布流形上施加信任域硬约束：$\bar{D}_{\text{KL}}(\pi_{\text{old}} \parallel \pi) \le \delta$；
> 3. **如何近似计算**：
>    - 对目标函数做一阶泰勒展开（梯度 $g = \nabla_\theta L$），对 KL 约束做二阶泰勒展开（Hessian 矩阵即为 Fisher 信息矩阵 $F$）；
>    - 转化为二次约束极值问题，解析解为自然策略梯度方向 $\Delta\theta \propto F^{-1} g$；
>    - 针对高维参数下 $F^{-1}$ 求逆复杂度 $O(d^3)$ 无法承受的问题，使用**共轭梯度法（CG）**直接求解线性方程 $F x = g$，每次迭代仅需通过两次反向传播计算 Hessian-向量积 $F v$；
>    - 求解出方向后，利用**回溯线搜索（Backtracking Line Search）**验证目标函数真实提升并确保满足未近似的真实 KL 散度硬约束。

### Q4：PPO 相比 TRPO 做了哪些关键简化？PPO-Clip 目标函数的裁剪机制在数学和直觉上是如何工作的？
> **答**：
> 1. **相对 TRPO 的关键简化**：
>    - **从二阶降为纯一阶**：舍弃了 Fisher 信息矩阵计算、共轭梯度迭代（CG）与回溯线搜索，改造为纯一阶可微目标，直接兼容现代一阶自适应优化器（Adam/AdamW）；
>    - **硬约束转为截断软约束**：无需维护严格的拉格朗日乘子与硬性 KL 信任域；
>    - **样本高复用率**：TRPO 每次采集数据仅能更新一次；PPO 通过重要性采样比截断，支持在同一批采集的轨迹上执行多个 Epoch 的 Minibatch 随机梯度更新，显著提升样本吞吐与利用率；
> 2. **Clip 目标的运行机制（非对称悲观下界）**：
>    - 目标函数：$\mathcal{L}_{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$，其中 $r_t = \frac{\pi_\theta}{\pi_{\text{old}}}$；
>    - **好动作（$\hat{A}_t > 0$）**：当 $r_t$ 增加超过 $1+\epsilon$ 时被截断为常数 $(1+\epsilon)\hat{A}_t$，梯度归零，**防止对优质动作过度奖励**而冲垮策略稳定性；
>    - **坏动作（$\hat{A}_t < 0$）**：当 $r_t$ 降低低于 $1-\epsilon$ 时，负负相乘使 $\min$ 选中截断项 $(1-\epsilon)\hat{A}_t$，梯度归零，**防止对劣质动作过度惩罚**导致概率崩溃趋零；
>    - 外层 $\min$ 确保在任何外推异常情况下均采取悲观下界评估，构成稳健的单向防护。

### Q5：广义优势估计（GAE）中的超参数 $\lambda$ 是如何调节偏差（Bias）与方差（Variance）的？$\lambda=0$ 和 $\lambda=1$ 分别对应什么物理极限？
> **答**：
> 1. **权衡原理**：GAE 定义为 $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$，本质是将所有不同跨度的 $k$ 步优势估计进行参数为 $\lambda \in [0, 1]$ 的几何级数指数加权平均；
> 2. **$\lambda = 0$（单步 TD 极限）**：
>    - $\hat{A}_t = \delta_t^V = R_t + \gamma V(s_{t+1}) - V(s_t)$；
>    - **低方差**：仅涉及当前单步的环境转移与即时奖励，方差最小；
>    - **高偏差**：估计完全取决于价值网络 $V(s_{t+1})$ 的准确度；如果 Critic 未收敛或存在预估偏误，该偏差将 100% 污染策略梯度；
> 3. **$\lambda = 1$（全轨迹蒙特卡洛 MC 极限）**：
>    - $\hat{A}_t = \sum_{l=0}^\infty \gamma^l R_{t+l} - V(s_t)$；
>    - **无偏差**：优势值基于真实采样的全局累计回报，减去状态基线 $V(s_t)$ 不改变梯度的无偏性；
>    - **高方差**：整条序列上所有随机探索动作与状态转移的方差连乘累加，方差极大，导致训练剧烈抖动；
> 4. **工程选型**：$\lambda \in (0, 1)$（如大模型 RLHF 常用 $\lambda=0.95$）通过指数衰减压制远期噪声，用轻微的 Critic 模型偏差换取方差的大幅衰减。
