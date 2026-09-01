# ML Coding 06 · 量化与对齐：INT8 量化和 DPO / GRPO / PPO / OPD 损失函数

这一篇覆盖 ML coding 面试里"推理效率"和"post-training 对齐"这两块高频考点。它们的共同点是：都可以写成一个不依赖 autograd 内部机制的纯数值函数，面试官通常只要求你写出 forward 公式并解释背后的推导，而不是完整训练一个模型。

## 模块九：量化与对齐损失函数

### Exercise 1 · INT8 Quantization

`Int8Linear` 考的不是"把浮点数变成整数"这个动作本身，而是三件事：per-channel scale 怎么定义、为什么 `scale` 不能是 `nn.Parameter`、以及量化误差的理论上界怎么推。

对权重矩阵 `W` 的每一个输出通道（也就是每一行）做对称量化：

```text
scale[o]      = max(|W[o, :]|) / 127
W_int8[o, i]  = clamp(round(W[o, i] / scale[o]), -127, 127)
```

这是"对称"量化，因为零点固定在 0，不需要额外存 `zero_point`（非对称量化才需要零点偏移，通常用在激活值这种非对称分布上，比如 ReLU 之后全是非负数）。per-channel 而不是 per-tensor 的原因很直接：如果整个权重矩阵共用一个 scale，某一行数值特别大就会把其余行的量化精度全部拖垮，per-channel 把这个耦合去掉了。

`scale`（以及非对称量化里的 `zero_point`）必须注册成 `buffer` 而不是 `nn.Parameter`：它们是从权重统计量里直接算出来的确定性函数，不参与反向传播，也不该被 optimizer 更新；注册成 `Parameter` 会让 `.parameters()` 里混进不该被梯度下降碰的量，state_dict 的语义也会变得混乱。

误差上界的推导很短：`round` 的误差落在 `[-0.5, 0.5]` 个量化步长内，反量化之后误差就是 `|round(x) - x| * scale <= scale / 2`。这是面试里最常被追问的一句话，回答"误差和 scale 成正比，且有 `scale/2` 的硬上界"比背代码更重要。

#### Quick Coding：`Int8Linear`

```python
class Int8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

<details>
<summary>参考答案</summary>

```python
import torch
from torch import nn

class Int8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        # weight: (out_features, in_features)，量化前是训练好的、冻结的浮点权重
        scale = we\right.abs().amax(dim=1, keepdim=True) / 127.0
        w_int8 = torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8)

        self.register_buffer("w_int8", w_int8)
        self.register_buffer("scale", scale.squeeze(-1))  # (out_features,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)；这里只量化了权重，激活仍用浮点，
        # 是最常见的 "weight-only INT8" 变体（推理时省显存，不是省算力）
        y = torch.einsum("...i,oi->...o", x, self.w_int8.float())
        return y * self.scale
```

```python
# 数值自检（用 NumPy 复现同一套逻辑，避免依赖本机的 PyTorch 安装）
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(size=(6, 17)) * 3.7
scale = np.abs(W).max(axis=1, keepdims=True) / 127.0
W_int8 = np.clip(np.round(W / scale), -127, 127)
W_dq = W_int8 * scale

max_err_per_row = np.abs(W - W_dq).max(axis=1)
bound = scale.squeeze(-1) / 2.0 + 1e-9
assert np.all(max_err_per_row <= bound)  # 每一行的最大误差都不超过 scale/2
```

</details>

### Exercise 2 · DPO Loss

DPO（Direct Preference Optimization）要解决的问题是：给定同一个 prompt 下"更好"（chosen）和"更差"（rejected）的两条回答，怎么用一个监督式的 loss，把 RLHF 里"训 reward model + PPO rollout"这一整套流程压缩掉。

推导的起点是 KL 正则化的 RL 目标：

```text
max_pi  E_{y~pi(.|x)}[r(x,y)] - beta * KL(pi(.|x) || pi_ref(.|x))
```

这个目标存在闭式最优解：

```text
pi*(y|x) = pi_ref(y|x) * exp(r(x,y) / beta) / Z(x)
```

反解出隐式奖励：

```text
r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

`log Z(x)` 只依赖 prompt，不依赖具体回答 `y`。把这个隐式奖励代入 Bradley-Terry 偏好模型 `P(chosen > rejected) = sigmoid(r(chosen) - r(rejected))`，`log Z(x)` 恰好在做差时被消掉，剩下的就是 DPO loss：

```text
loss = -log sigmoid(
    beta * [(logp_chosen - logp_ref_chosen) - (logp_rejected - logp_ref_rejected)]
)
```

这一步消去解释了为什么 DPO 不需要显式的 reward model：reward 已经被参数化成"policy 和 reference 的 log 概率比"，训练这个比值本身就是在训练一个隐式 reward。参考模型的两项同时也起到了 per-prompt baseline 的作用：如果 chosen/rejected 的长度或难度差异导致 `pi_ref` 本身给出的概率就不同，减去 `logp_ref` 之后这部分差异会被抵消，避免模型学到"更短的回答更好"这种和偏好无关的捷径。

#### Quick Coding：`dpo_loss`

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float):
    ...
```

<details>
<summary>参考答案</summary>

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -torch.nn.functional.logsigmoid(logits).mean()
```

```python
# 数值自检：margin 越大，loss 应该严格越小（单调性）
import numpy as np

def dpo_loss_np(pol_c, pol_r, ref_c, ref_r, beta):
    margin = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    return -np.log(1 / (1 + np.exp(-margin)))

beta = 0.1
ref_c, ref_r = -12.0, -13.0
base   = dpo_loss_np(np.array([-10.0]), np.array([-13.0]), np.array([ref_c]), np.array([ref_r]), beta)
better = dpo_loss_np(np.array([-9.0]),  np.array([-13.0]), np.array([ref_c]), np.array([ref_r]), beta)
assert better < base  # chosen 相对 ref 的 log 概率更大 -> margin 更大 -> loss 更小

worse_reject = dpo_loss_np(np.array([-10.0]), np.array([-11.0]), np.array([ref_c]), np.array([ref_r]), beta)
assert worse_reject > base  # rejected 相对 ref 的 log 概率变大（margin 缩小）-> loss 更大
```

</details>

### Exercise 3 · GRPO Loss

GRPO（Group Relative Policy Optimization）解决的是 PPO 里最贵的一块：value function（critic）网络。PPO 需要一个和 policy 同量级的 critic 来估计 baseline，这个网络本身的训练和推理都要占用大量算力。

GRPO 的做法是：对同一个 prompt 采样一组（比如 K=8）回答，直接用这组内部的统计量替代 value function 的作用：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
```

组内均值就是这组样本的经验 baseline，组内标准差把不同 prompt 之间奖励量纲不一致的问题也一并归一化了。算出 `A_i` 之后，剩下的优化目标和 PPO 完全一样，还是下面 Exercise 4 的 clipped surrogate objective。这也是这道题最容易被问到的一句话总结：**GRPO 不是一个新的 loss 形式，它只是换了一种更便宜的 advantage 估计方式，外层的 clip 机制原封不动地照搬 PPO**。

#### Quick Coding：`grpo_loss`

```python
def grpo_loss(logps, old_logps, rewards, group_ids, eps: float, clip_ratio: float):
    ...
```

<details>
<summary>参考答案</summary>

```python
def grpo_loss(logps, old_logps, rewards, group_ids, eps: float, clip_ratio: float):
    advantages = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        mask = group_ids == g
        r = rewards[mask]
        advantages[mask] = (r - r.mean()) / (r.std() + eps)

    ratio = torch.exp(logps - old_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()
```

```python
# 数值自检：组内 advantage 的均值应约为 0，标准差约为 1
import numpy as np

rewards = np.array([1.0, 3.0, 2.0, 5.0, 0.0])
eps = 1e-6
A = (rewards - rewards.mean()) / (rewards.std() + eps)
manual = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)
assert np.allclose(A, manual)
assert abs(A.mean()) < 1e-6
assert abs(A.std() - 1.0) < 1e-3
```

</details>

### Exercise 4 · PPO Loss

PPO 的 clipped surrogate objective 要回答的核心问题是：为什么普通的 policy gradient（`-mean(ratio * A)`）不稳定，clip 又是怎么把这个问题按住的。

重要性采样比 `ratio = exp(new_logp - old_logp)` 衡量的是"新策略相对旧策略，对这个动作的偏好变化了多少"。如果不加约束地按 `ratio * A` 更新，一旦某个动作的 `ratio` 因为一次更新变得很大，下一次更新会基于一个已经和真实分布差很远的重要性权重继续外推，容易一步错、步步错（这也是 trust-region 方法要解决的问题）。

```text
loss = -mean( min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) )
```

这里有一个经典的面试陷阱：**clip 的哪一侧生效，取决于 `A` 的符号**。

- `A > 0`（这个动作比平均水平好，想增加它的概率）：只有当 `ratio` 超过 `1+eps` 时，`min()` 才会选中被压低的 `clip` 分支。也就是说，clip 在阻止你把一个"已经在变好"的动作概率进一步无限推高。
- `A < 0`（这个动作比平均水平差，想降低它的概率）：只有当 `ratio` 低于 `1-eps` 时，`min()` 才会选中 `clip` 分支。这次 clip 是在阻止你把概率压得太低太快。

一句话记忆：**clip 永远是在阻止"已经朝正确方向移动太多"的更新继续加速，而不是阻止移动本身**。这也是为什么两侧的判定条件是镜像的。

#### Quick Coding：`ppo_loss`

```python
def ppo_loss(new_logps, old_logps, advantages, clip_ratio: float):
    ...
```

<details>
<summary>参考答案</summary>

```python
def ppo_loss(new_logps, old_logps, advantages, clip_ratio: float):
    ratio = torch.exp(new_logps - old_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()
```

```python
# 数值自检：验证非对称 clip 的生效条件
import numpy as np

clip_ratio = 0.2

def which_bound_binds(r, A):
    unclipped = r * A
    clipped = np.clip(r, 1 - clip_ratio, 1 + clip_ratio) * A
    return min(unclipped, clipped), clipped < unclipped

# A > 0：只有 r > 1+eps 时 clip 分支才会被 min() 选中
for r in np.linspace(0.5, 2.0, 7):
    m, clip_active = which_bound_binds(r, A=1.0)
    assert clip_active == (r > 1 + clip_ratio)

# A < 0：只有 r < 1-eps 时 clip 分支才会被 min() 选中
for r in np.linspace(0.5, 2.0, 7):
    m, clip_active = which_bound_binds(r, A=-1.0)
    assert clip_active == (r < 1 - clip_ratio)
```

</details>

### Exercise 5 · OPD Loss

OPD（On-Policy Distillation）和标准知识蒸馏（KD）的区别，是这道题唯一真正的考点。标准 KD 在离线的教师训练数据（或教师采样出的数据）上，计算正向 KL `KL(teacher || student)`；OPD 反过来：从**学生自己当前的策略**里采样序列，然后在这些学生生成的 token 上计算反向 KL `KL(student || teacher)`。

```text
KL(student || teacher) = E_{token ~ student}[ log p_student(token) - log p_teacher(token) ]
```

这个期望是对学生自己的采样分布取的，所以 loss 的蒙特卡洛估计就是"在学生生成的 token 上，`log p_student - log p_teacher` 的均值"。

两个问题决定了这道题的深度：

1. **为什么用反向 KL 而不是正向 KL？** 反向 KL 是 mode-seeking 的：最小化 `KL(student||teacher)`会驱使学生把概率质量集中到教师认为合理的区域，但不强迫学生覆盖教师所有的高概率区域，学生只需要对自己实际会生成的东西负责。正向 KL 是 mode-covering 的，会强迫学生在教师分布的每个峰上都留一些概率质量，即使那是学生自己从来不会走到的路径。
2. **为什么一定要"on-policy"（从学生采样）？** 如果继续在教师的离线数据上算反向 KL，学生在训练时看到的 token 序列和它推理时自己生成的序列分布不一致（exposure bias）：训练时学生总是被"喂"正确前缀，测试时它要在自己犯错后的前缀上继续生成。从学生自己的 rollout 里采样，能让蒸馏信号直接针对学生真实会遇到的分布纠偏，而不是针对一个它推理时用不上的分布。

KL 的不对称性本身也是一个常见追问点：`KL(P||Q) != KL(Q||P)`，下面的数值自检会直接算出两侧的值来确认这一点。

#### Quick Coding：`opd_loss`

```python
def opd_loss(student_logits, teacher_logits, sampled_tokens):
    ...
```

<details>
<summary>参考答案</summary>

```python
def opd_loss(student_logits, teacher_logits, sampled_tokens):
    # student_logits, teacher_logits: (batch, seq_len, vocab)
    # sampled_tokens: (batch, seq_len)，从学生自己的分布里采样得到
    log_p_student = torch.log_softmax(student_logits, dim=-1)
    log_p_teacher = torch.log_softmax(teacher_logits, dim=-1)

    log_p_s = torch.gather(log_p_student, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    log_p_t = torch.gather(log_p_teacher, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

    return (log_p_s - log_p_t).mean()  # 蒙特卡洛估计 KL(student || teacher)
```

```python
# 数值自检：KL 的不对称性，以及 opd_loss 的蒙特卡洛估计是否收敛到真值
import numpy as np

outcomes = np.arange(5)
p_student = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
p_teacher = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

kl_fwd = np.sum(p_teacher * (np.log(p_teacher) - np.log(p_student)))  # KL(teacher||student)
kl_rev = np.sum(p_student * (np.log(p_student) - np.log(p_teacher)))  # KL(student||teacher)
assert not np.isclose(kl_fwd, kl_rev)  # 两个方向的 KL 确实不相等

rng = np.random.default_rng(0)
for n in [100, 10_000, 1_000_000]:
    samples = rng.choice(len(p_student), size=n, p=p_student)
    est = np.mean(np.log(p_student[samples]) - np.log(p_teacher[samples]))
    # n 越大，est 应该越接近 kl_rev
assert abs(est - kl_rev) < 0.01  # 用最后一轮（n=1_000_000）的估计值检查收敛
```

</details>

## 小结：这五道题在考什么

| 题目 | 表面考点 | 真正考点 |
| --- | --- | --- |
| INT8 Quantization | 量化公式 | per-channel 的必要性、`buffer` vs `Parameter`、误差上界推导 |
| DPO | loss 公式 | 从 KL 正则化 RL 目标反解隐式 reward，`log Z(x)` 如何被消去 |
| GRPO | advantage 归一化 | 用组内统计量替代 value function 的动机 |
| PPO | clip 语法 | clip 在 `A` 正负两种情况下分别保护哪一侧 |
| OPD | 反向 KL 公式 | mode-seeking vs mode-covering、on-policy 采样为什么必要 |

如果只能记一句话：这五道题都在问"这个方法省掉了什么组件，又是通过什么数学变换省掉的"，而不是单纯要求你背出最终公式。
