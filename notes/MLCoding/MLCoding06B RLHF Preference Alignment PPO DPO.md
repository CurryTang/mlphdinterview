# ML Coding 06B · RLHF 与偏好对齐全景：从 Reward Model、PPO 4 模型架构到 DPO/IPO/KTO/SimPO 与对齐税

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

### 3. 广义优势估计（GAE, Generalized Advantage Estimation）

为了在价值估计的**方差（Variance）**与**偏差（Bias）**之间取得最优平衡，PPO 使用 GAE 计算每个 Token 的优势值 $\hat{A}_t$。

1. **时序差分误差（TD Error）**：
   $$\delta_t^V = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
2. **GAE 指数衰减加权累加**：
   $$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V$$
   - $\lambda = 0$ 时退化为单步 TD 估计（低方差，高偏差）；
   - $\lambda = 1$ 时退化为全序列 Monte Carlo 估计（高方差，无偏差）；
   - 工业界常取 $\gamma = 1.0, \lambda = 0.95$。

---

### 4. PPO-Clip 策略截断目标函数

$$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$$

$$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

PPO 通过将概率比率 $r_t(\theta)$ 限制在 $[1-\epsilon, 1+\epsilon]$（如 $\epsilon=0.2$）内，防止单步策略更新幅度过大导致策略崩溃。

---

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
| **SimPO** | 使用生成长度归一化的平均 Logits 差，结合 Target Margin 惩罚 | $-\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l) - \gamma\right)$ | **无需 (1 模型)** | **当前开源评测 SOTA**，彻底摆脱参考模型，且内生性消除长度偏见 |

---

## 模块五：RLHF 对齐陷阱与生产治理实战

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
