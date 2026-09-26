# 现代优化器 · 从 SGD 动量到 Adam/AdamW 与 Muon 谱正交投影：预条件几何、二阶矩与稀疏陷阱

> **导读**：优化算法是大模型预训练与工业级机器学习系统的“发动机”。很多工程师只把优化器当作一行简单的 `optimizer.step()`，但在大模型分布式预训练（超大 Batch）、推荐系统（非平稳与高维稀疏特征）以及前沿基座训练（谱正交化）中，优化器的数学机制直接决定了模型是稳定收敛还是中途发散。
>
> 本篇系统梳理从 **SGD**、**动量机制**、**AdaGrad**、**RMSProp**、**Adam/AdamW** 到 2024~2025 年爆火的 **Muon (Momentum Orthogonal Optimizer)** 的演进逻辑；深度剖析一阶与二阶矩、偏差修正、有效步长机制、稀疏 Embedding 陷阱与超大 Batch 泛化灾难；并给出纯 PyTorch 的生产级代码实现与面试高频深度追问答题框架。

---

## 00. 交互轨迹对比：病态峡谷中的优化动力学

在真实的深度神经网络与 Transformer 损失曲面中，不同维度的曲率通常极度不均衡（Hessian 矩阵的条件数 $\kappa = \lambda_{\max}/\lambda_{\min} \gg 1$，形成狭窄的“病态峡谷”）。不同优化器在峡谷中的寻优行为存在本质分歧：

```optimizer-trajectory-demo
```

---

## 01. 经典基石：从 SGD 到动量 SGD (SGD with Momentum)

### 1. 朴素 SGD 的几何困境与病态曲率

标准随机梯度下降更新公式为：

$$\theta_{t+1} = \theta_t - \eta g_t, \quad \text{其中 } g_t = \nabla_\theta \mathcal{L}_B(\theta_t)$$

- **各向同性（Isotropic）步长假定**：SGD 对所有参数维度使用统一的学习率 $\eta$。
- **病态峡谷困境**：在二次逼近下 $\mathcal{L}(\theta) \approx \frac{1}{2} \theta^\top H \theta$。若 Hessian 矩阵的最大特征值 $\lambda_{\max}$ 与最小特征值 $\lambda_{\min}$ 差异巨大（条件数 $\kappa = \lambda_{\max} / \lambda_{\min} \gg 1$）：
  - 为了保证在曲率极大（坡度极陡）的垂直峡谷壁方向不发散，学习率必须被限制在 $\eta < \frac{2}{\lambda_{\max}}$；
  - 但在平缓的谷底主轴方向（曲率 $\lambda_{\min}$），该学习率导致每步前进步长极其微弱，模型陷入在两侧陡壁来回剧烈震荡、谷底前进停滞的局面。

### 2. Polyak 经典重球动量 (Heavy-ball Momentum)

动量机制模拟物理世界中重球滚下山坡的过程，引入一阶动量缓冲 $m_t$：

$$m_t = \beta m_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta m_t$$

- **频域滤波效应**：展开递推式可知 $m_t = \sum_{\tau=0}^{t-1} \beta^\tau g_{t-\tau}$。
  - 在高频震荡维度上，连续步的梯度方向不断反向（$g_t$ 与 $g_{t-1}$ 异号），累加时相互抵消；
  - 在谷底低频平缓维度上，梯度方向持续一致，动量不断同向叠加，速度放大为约 $\frac{1}{1-\beta}$ 倍（当 $\beta=0.9$ 时加速 10 倍）。

<details>
<summary><strong>深入探究：为什么理想步长是 1/λᵢ？二次几何解耦、秩坍塌与无法直接计算的物理硬伤</strong></summary>

### 核心数学推导：为什么理想步长是 $1/\lambda_i$？

考察损失函数在极小值点 $\theta^*$ 附近的局部二次近似（不妨平移坐标系使 $\theta^* = \mathbf{0}, f(\theta^*) = 0$）：

$$f(\theta) = \frac{1}{2} \theta^\top H \theta$$

其中 $H \in \mathbb{R}^{d \times d}$ 为对称正定（SPD）的 Hessian 矩阵。梯度为 $g(\theta) = \nabla f(\theta) = H \theta$。

---

#### 1. 特征正交分解与几何解耦

由于 $H$ 是实对称矩阵，由谱定理可对其进行正交特征分解：

$$H = Q \Lambda Q^\top = \sum_{i=1}^d \lambda_i v_i v_i^\top$$

* $Q = [v_1, v_2, \dots, v_d]$ 为由特征向量组成的正交矩阵（$Q^\top Q = I$），各 $v_i$ 互为正交基底；
* $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_d)$ 为特征值对角矩阵，每个 $\lambda_i > 0$ 代表沿该特征向量方向的主曲率（二阶导数大小）。

引入正交坐标变换 $z = Q^\top \theta$（即把原始参数投影到 Hessian 的特征坐标系下），目标函数被完全解耦为 $d$ 个独立的一维二次函数之和：

$$f(\theta) = \frac{1}{2} (Q z)^\top H (Q z) = \frac{1}{2} z^\top (Q^\top H Q) z = \frac{1}{2} z^\top \Lambda z = \sum_{i=1}^d \frac{1}{2} \lambda_i z_i^2$$

梯度在各特征轴上的投影为：

$$g_z = \nabla_z f(z) = \Lambda z \implies g_z^{(i)} = \lambda_i z_i$$

---

#### 2. 一维坐标上的动力学与单步置零

若允许对每个特征方向赋予独立的步长 $\eta_i$，沿方向 $v_i$ 的单步梯度下降为：

$$z_{t+1}^{(i)} = z_t^{(i)} - \eta_i g_z^{(i)} = z_t^{(i)} - \eta_i \lambda_i z_t^{(i)} = (1 - \eta_i \lambda_i) z_t^{(i)}$$

误差的单步收缩倍率（Contraction Factor）为：

$$\rho_i(\eta_i) = |1 - \eta_i \lambda_i|$$

* **最快收敛（单步置零）：**
  要想让该方向的误差瞬间归零（即 $z_{t+1}^{(i)} = 0$），只需令收缩因子 $1 - \eta_i \lambda_i = 0$，解得：
  $$\eta_i^* = \frac{1}{\lambda_i}$$
  此时仅需一步迭代，参数就精确落在了该方向抛物线的最低点（谷底底端）。
* **稳定下降的绝对边界：**
  迭代不发散的充要条件是收缩模长小于 1：
  $$|1 - \eta_i \lambda_i| < 1 \iff -1 < 1 - \eta_i \lambda_i < 1 \iff 0 < \eta_i < \frac{2}{\lambda_i}$$

---

### 从一维线搜索（Line Search）视角理解

假设当前位置为 $\theta$，沿特征方向 $v_i$ 移动步长 $\alpha$：

$$\phi(\alpha) = f(\theta - \alpha v_i)$$

利用泰勒展开（由于是二次模型，展开精确成立）：

$$\phi(\alpha) = f(\theta) - \alpha \nabla f(\theta)^\top v_i + \frac{1}{2} \alpha^2 v_i^\top H v_i$$

注意到 $H v_i = \lambda_i v_i$，且 $v_i^\top v_i = 1$，展开式简化为关于标量 $\alpha$ 的一元凸二次函数：

$$\phi(\alpha) = f(\theta) - \alpha (\nabla f(\theta)^\top v_i) + \frac{1}{2} \alpha^2 \lambda_i$$

为了让目标函数下降最多，对 $\alpha$ 求导并令一阶导为 0：

$$\phi'(\alpha) = - (\nabla f(\theta)^\top v_i) + \alpha \lambda_i = 0 \implies \alpha^* = \frac{\nabla f(\theta)^\top v_i}{\lambda_i}$$

若写成梯度步长形式 $\Delta \theta = -\eta_i (\nabla f(\theta)^\top v_i) v_i$，则最优标量比例正是：

$$\eta_i^* = \frac{1}{\lambda_i}$$

**物理直觉：**
* **曲率大（$\lambda_i$ 极大）：** 碗口极陡，坡度虽然陡峭（梯度大），但极小值点离得很近。稍不留神就会冲上对面的峭壁，因此必须迈小步（$\eta_i \sim \frac{1}{\lambda_i}$ 极小）。
* **曲率小（$\lambda_i$ 极小）：** 峡谷极平缓，坡度虽然平淡（梯度小），但极小值点在很远的地方。如果步子不大，漫长的平原将永远走不完，因此需要迈大步（$\eta_i \sim \frac{1}{\lambda_i}$ 极大）。

---

### 背景知识与深层拓展

#### 1. 牛顿法（Newton's Method）的几何本质
为什么二阶牛顿法在二次曲面上能一步收敛？牛顿更新公式为：

$$\Delta \theta_{\text{Newton}} = - H^{-1} \nabla f(\theta)$$

将 $H^{-1}$ 在特征基底 $v_i$ 下展开：

$$H^{-1} = Q \Lambda^{-1} Q^\top = \sum_{i=1}^d \frac{1}{\lambda_i} v_i v_i^\top$$

代入梯度：

$$\Delta \theta_{\text{Newton}} = - \sum_{i=1}^d \frac{1}{\lambda_i} v_i (v_i^\top \nabla f(\theta))$$

牛顿法本质上是在每个互相正交的特征方向 $v_i$ 上，自动分配了精确等于 $\frac{1}{\lambda_i}$ 的理想步长。

#### 2. 标准 SGD 的“各向同性折磨”与条件数瓶颈
标准一阶梯度下降（SGD）被迫用一个标量学习率 $\eta$ 统一指挥所有维度：
* 系统的**稳定性上限**被最陡峭的壁面绑架：为了保证整个系统不发散，必须保证所有特征方向收缩，即 $\eta < \frac{2}{\lambda_{\max}}$；
* 在平缓的谷底主轴方向（曲率 $\lambda_{\min}$），其实际收缩因子为：
  $$1 - \eta \lambda_{\min} \approx 1 - \frac{2 \lambda_{\min}}{\lambda_{\max}} = 1 - \frac{2}{\kappa}$$
  其中 $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ 为曲率条件数。
* 当 $\kappa = 10^4$ 时，每一步只能消除万分之二的残差，陷入“陡峭方向左右剧烈晃动，平缓方向像在沥青中爬行”的困境。

#### 3. 现代深度学习优化器的应对路径
* **动量法（Polyak Momentum）：** 无法改变步长，但通过一阶递推的共轭复根动力学，把高频震荡相互抵消、低频速度相干叠加，将收缩步数从 $\mathcal{O}(\kappa)$ 加速到 $\mathcal{O}(\sqrt{\kappa})$。
* **Adam / RMSProp（对角预条件）：** 计算梯度的二阶矩 $v_t \approx g^2$。在二次模型上，$\mathbb{E}[g_i^2]$ 在经验上扮演了对角 Hessian $\text{diag}(H)$ 的代理，通过除以 $\sqrt{v_t}$ 试图模仿 $H_{ii}^{-1}$，拉平各坐标轴的步长差距。
* **Muon（谱正交化）：** 放弃逐元素对角缩放，通过 Newton-Schulz 迭代把矩阵参数梯度投影到正交流形 $UV^\top$，在矩阵层面上将全谱奇异值强行归一化为 1.0，直接将参数空间的谱条件数降为 1。

---

### 阻碍直接计算最优步长的四个根本物理与统计硬伤

高维确实是一道不可逾越的物理硬伤，但就算算力无限，“直接算出各方向最优步长”在深度学习的非凸几何与随机训练环境下依然无法直接成立。阻碍直接计算最优步长的根本原因可以分为以下四个层面：

#### 1. 维度之壁：显存与算力的立方级爆炸
在二次模型中，要想给所有特征方向分配精准的最优步长 $\eta_i = \frac{1}{\lambda_i}$，数学本质等价于直接求解牛顿更新步 $\Delta \theta = -H^{-1} g$ 或对 Hessian 矩阵 $H$ 进行特征值分解。
* **显存爆炸（$\mathcal{O}(d^2)$）：**
  以一个 7B（$d \approx 7 \times 10^9$）参数的语言模型为例：
  $$H \in \mathbb{R}^{d \times d} \implies (7 \times 10^9)^2 \approx 4.9 \times 10^{19} \text{ 个元素}$$
  以 float32 存储该矩阵需要近 **200 EB（Exabytes）** 的显存，而整台 8 卡 H100 服务器的显存总和仅有 640 GB。即便只存储，也是物理不可能的。
* **算力爆炸（$\mathcal{O}(d^3)$）：**
  对 $d \times d$ 矩阵进行求逆或特征正交分解的计算复杂度为 $\mathcal{O}(d^3)$。对于千万级以上参数的网络，一次迭代可能就需要全球算力集群计算数月。

#### 2. 统计随机性与“秩坍塌”（Rank Deficiency）
深度学习使用 Mini-batch 训练。真实损失函数的 Hessian 矩阵是全数据集曲率的期望值：
$$H = \mathbb{E}_{x \sim \mathcal{D}} [\nabla^2 \ell(x; \theta)]$$
但在实际训练的单步迭代中，只能通过当前 Batch 计算经验曲率（以最常用的经验 Fisher / Gauss-Newton 矩阵近似为例）：
$$\hat{H} = \frac{1}{B} \sum_{k=1}^B g_k g_k^\top$$
* 一个 Batch 的样本量 $B$ 通常在 $10^2 \sim 10^4$ 之间，而参数量 $d$ 在 $10^7 \sim 10^{11}$ 之间（$B \ll d$）。
* $B$ 个外积矩阵相加，其**代数秩最大只有 $B$**。
* 这意味着计算出来的 Hessian 有超过 $99.99\%$ 的特征值严格为 0（极端不可逆）。在退化的零曲率空间中，所谓的“理论最优步长” $\frac{1}{\lambda_i} = \frac{1}{0}$ 发生除零崩溃，根本不存在数学定义。

#### 3. 非凸几何灾难：负特征值与鞍点陷阱
二次模型假设损失曲面是一个处处向上弯曲的凸抛物面（$H \succ 0$，所有 $\lambda_i > 0$）。但在深层神经网络的真实损失曲面上：
* **非凸与负曲率（$\lambda_i < 0$）：** 网络存在大量鞍点和局部极大值。在这些区域，Hessian 矩阵必定存在负特征值。
* **反向冲向极大值：** 若机械代入 $\eta_i = \frac{1}{\lambda_i}$，当 $\lambda_i < 0$ 时，更新量符号反转：
  $$\Delta z^{(i)} = - \left(\frac{1}{\lambda_i}\right) (\lambda_i z^{(i)}) = -z^{(i)}$$
  一阶泰勒展开下不仅不会下降，反而会迎面冲向鞍点或局部极大值脊线，导致优化器剧烈发散。必须引入复杂的阻尼矩阵（Damping，如 Levenberg-Marquardt）或信赖域（Trust Region）截断，进一步放大了计算成本。

#### 4. 为什么退而求其次的“一维线搜索”依然行不通？
既然不能算全空间的矩阵逆，如果固定当前的一阶梯度方向 $p = -\nabla f(\theta)$，只沿着这一个方向做一维线搜索（Exact Line Search），找到让单步下降最多的最优标量步长 $\eta^*$：
1. **评估成本过高：** 一维线搜索（如回溯法、二分法、插值法）为了找到二次谷底，每一步需要评估 3~5 次损失值（即 3~5 次前向传播 Forward Pass）。深度学习的瓶颈就在前向与反向传播上，多做几次前向的开销足以让 SGD 迈出更多有效步。
2. **批次噪声失效：** 在 Mini-batch $B_t$ 上通过多次试探精准找到的最佳步长 $\eta^*$，在换到下一个 Mini-batch $B_{t+1}$ 时曲面形态完全变了，前一步费尽心力找到的精确极小点在新 Batch 上直接失效。

---

### 现代优化器的妥协解法：如何绕过这个物理极限

既然精确全局计算不可能，工业界走出了三条“局部与结构化近似”的道路：
* **Adam / RMSProp（对角近似）：**
  彻底放弃交叉项（令所有非对角曲率 $H_{ij} = 0$），只维护 $d$ 维的对角二阶矩 $v_t \approx \text{diag}(g^2)$ 作为各坐标轴曲率的代理，将存储与计算降回线性复杂度 $\mathcal{O}(d)$。
* **K-FAC / Shampoo（Kronecker 分解）：**
  利用全连接层和卷积层的张量积结构，把巨大的权重梯度矩阵近似为两个小矩阵的 Kronecker 积 $H \approx A \otimes B$，将千万级矩阵求逆拆解为两个小矩阵分别求逆 $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$。
* **Muon（矩阵流形谱正交投影）：**
  不再尝试估计全局 Hessian，而是针对 2D 权重矩阵直接用五阶 Newton-Schulz 迭代进行奇异值全谱归一化（将所有奇异值归一为 1.0），在 GPU Tensor Core 上纯靠 5 次极速 GEMM 矩阵乘法，避开 SVD 与矩阵求逆，以最低开销消除了层内的谱病态性。

</details>

---

## 01B. 凸优化与梯度下降的收敛性推导：裂项相消 (Telescoping Sum) 与上界证明

在优化理论与深度学习系统的面试考核中，对梯度下降及其动量变体的理论性质分析往往直击本质。本节给出基于**裂项相消（Telescoping Sum）**技巧的经典收敛率严格数学推导，并推导病态二次曲面下条件数 $\kappa$ 对 SGD 与动量加速的根本制约。

### 1. $L$-光滑函数的下降引理 (Descent Lemma) 与梯度范数裂项求和

设目标函数 $f: \mathbb{R}^d \to \mathbb{R}$ 连续可微，且其梯度为 $L$-Lipschitz 连续（即 $L$-光滑）：

$$\|\nabla f(x) - \nabla f(y)\| \le L \|x - y\|, \quad \forall x, y \in \mathbb{R}^d$$

#### 下降引理（Descent Lemma）推导

由微积分基本定理，沿线段积分可得：

$$f(y) - f(x) = \int_0^1 \langle \nabla f(x + \tau(y - x)), y - x \rangle d\tau$$

在积分内部加减 $\nabla f(x)$：

$$f(y) - f(x) = \langle \nabla f(x), y - x \rangle + \int_0^1 \langle \nabla f(x + \tau(y - x)) - \nabla f(x), y - x \rangle d\tau$$

应用 Cauchy-Schwarz 不等式及梯度的 $L$-Lipschitz 连续性：

$$|f(y) - f(x) - \langle \nabla f(x), y - x \rangle| \le \int_0^1 \|\nabla f(x + \tau(y - x)) - \nabla f(x)\| \|y - x\| d\tau$$
$$\le \int_0^1 L \tau \|y - x\|^2 d\tau = \frac{L}{2} \|y - x\|^2$$

移项即得**下降引理（Descent Lemma）**：

$$f(y) \le f(x) + \langle \nabla f(x), y - x \rangle + \frac{L}{2} \|y - x\|^2$$

#### 单步充分下降量

代入一阶梯度下降更新步 $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$，令 $x = \theta_t, y = \theta_{t+1}$，则位移向量 $y - x = -\eta \nabla f(\theta_t)$：

$$f(\theta_{t+1}) \le f(\theta_t) - \eta \|\nabla f(\theta_t)\|^2 + \frac{L \eta^2}{2} \|\nabla f(\theta_t)\|^2 = f(\theta_t) - \eta \left(1 - \frac{L\eta}{2}\right) \|\nabla f(\theta_t)\|^2$$

选取步长 $\eta \le \frac{1}{L}$（当选取最优步长 $\eta = \frac{1}{L}$ 时，下降量最大）：

$$f(\theta_{t+1}) \le f(\theta_t) - \frac{1}{2L} \|\nabla f(\theta_t)\|^2 \iff \|\nabla f(\theta_t)\|^2 \le 2L [f(\theta_t) - f(\theta_{t+1})]$$

#### 裂项相消求和 (Telescoping Sum)

对所有迭代步 $t = 0, 1, \dots, T-1$ 累加求和：

$$\sum_{t=0}^{T-1} \|\nabla f(\theta_t)\|^2 \le 2L \sum_{t=0}^{T-1} [f(\theta_t) - f(\theta_{t+1})]$$

展开右侧累加式，所有中间项首尾正负相消（Telescoping Cancellation）：

$$\sum_{t=0}^{T-1} [f(\theta_t) - f(\theta_{t+1})] = [f(\theta_0) - f(\theta_1)] + [f(\theta_1) - f(\theta_2)] + \dots + [f(\theta_{T-1}) - f(\theta_T)] = f(\theta_0) - f(\theta_T)$$

若目标函数存在全局下界 $f^*$（即 $f(\theta_T) \ge f^*$）：

$$\sum_{t=0}^{T-1} \|\nabla f(\theta_t)\|^2 \le 2L [f(\theta_0) - f^*]$$

两边同除以 $T$，并取迭代过程中的最小梯度范数平方：

$$\min_{0 \le t < T} \|\nabla f(\theta_t)\|^2 \le \frac{1}{T} \sum_{t=0}^{T-1} \|\nabla f(\theta_t)\|^2 \le \frac{2L (f(\theta_0) - f^*)}{T} = O\left(\frac{1}{T}\right)$$

> **核心结论**：对于任意 $L$-光滑函数（无需凸性假设），常数步长梯度下降必然在 $T$ 步内以 $O(1/T)$ 速率使梯度范数平方收敛，即达到 $\|\nabla f(\theta)\| \le \epsilon$ 的驻点最多需要 $T = O(1/\epsilon^2)$ 步。

---

### 2. 凸函数距离势函数裂项相消与 SGD 的 $O(1/\sqrt{T})$ 上界证明

在凸优化设定下，通过构造到最优解 $\theta^*$ 的距离势函数（Lyapunov Potential），可利用裂项相消推导出函数值误差 $f(\theta) - f^*$ 的收敛上界。

#### 势函数展开

设目标函数 $f$ 为凸函数，且梯度（或次梯度）有界：$\|g_t\| \le G$。
定义势函数 $\Phi_t = \frac{1}{2} \|\theta_t - \theta^*\|^2$。考察更新步 $\theta_{t+1} = \theta_t - \eta g_t$：

$$\|\theta_{t+1} - \theta^*\|^2 = \|\theta_t - \eta g_t - \theta^*\|^2 = \|\theta_t - \theta^*\|^2 - 2\eta \langle g_t, \theta_t - \theta^* \rangle + \eta^2 \|g_t\|^2$$

由凸性一阶充要条件 $\langle g_t, \theta_t - \theta^* \rangle \ge f(\theta_t) - f(\theta^*) = f(\theta_t) - f^*$：

$$\|\theta_{t+1} - \theta^*\|^2 \le \|\theta_t - \theta^*\|^2 - 2\eta (f(\theta_t) - f^*) + \eta^2 G^2$$

移项整理出瞬时次优差距（Suboptimality Gap）：

$$f(\theta_t) - f^* \le \frac{1}{2\eta} \left( \|\theta_t - \theta^*\|^2 - \|\theta_{t+1} - \theta^*\|^2 \right) + \frac{\eta}{2} G^2$$

#### 裂项相消求和 (Telescoping Sum)

对 $t = 0, 1, \dots, T-1$ 累加求和：

$$\sum_{t=0}^{T-1} (f(\theta_t) - f^*) \le \frac{1}{2\eta} \sum_{t=0}^{T-1} \left( \|\theta_t - \theta^*\|^2 - \|\theta_{t+1} - \theta^*\|^2 \right) + \frac{\eta}{2} \sum_{t=0}^{T-1} G^2$$

中间所有距离项发生**首尾裂项相消**：

$$\sum_{t=0}^{T-1} \left( \|\theta_t - \theta^*\|^2 - \|\theta_{t+1} - \theta^*\|^2 \right) = \|\theta_0 - \theta^*\|^2 - \|\theta_T - \theta^*\|^2 \le \|\theta_0 - \theta^*\|^2$$

代入得总误差上界（记初始距离 $R = \|\theta_0 - \theta^*\|$）：

$$\sum_{t=0}^{T-1} (f(\theta_t) - f^*) \le \frac{R^2}{2\eta} + \frac{\eta T G^2}{2}$$

#### Jensen 不等式与最优步长选取

定义历次参数的遍历平均解（Ergodic Average / Polyak-Ruppert 平均）$\bar{\theta}_T = \frac{1}{T} \sum_{t=0}^{T-1} \theta_t$。由凸函数性质与 Jensen 不等式：

$$f(\bar{\theta}_T) - f^* \le \frac{1}{T} \sum_{t=0}^{T-1} (f(\theta_t) - f^*) \le \frac{R^2}{2\eta T} + \frac{\eta G^2}{2}$$

右侧关于学习率 $\eta$ 为凸函数。对其求导并令导数为 0：

$$\frac{\partial}{\partial \eta} \left( \frac{R^2}{2\eta T} + \frac{\eta G^2}{2} \right) = -\frac{R^2}{2\eta^2 T} + \frac{G^2}{2} = 0 \implies \eta^* = \frac{R}{G \sqrt{T}}$$

代入最优步长 $\eta^*$：

$$f(\bar{\theta}_T) - f^* \le \frac{R^2}{2 \left(\frac{R}{G\sqrt{T}}\right) T} + \frac{\left(\frac{R}{G\sqrt{T}}\right) G^2}{2} = \frac{R G}{2\sqrt{T}} + \frac{R G}{2\sqrt{T}} = \frac{R G}{\sqrt{T}} = O\left(\frac{1}{\sqrt{T}}\right)$$

> **核心结论**：对于仅有一阶梯度信息的随机凸优化，SGD 的全局收敛速率界为 $O(1/\sqrt{T})$。达到 $\epsilon$ 精度所需迭代步数为 $T = O(1/\epsilon^2)$，这是信息论下不可逾越的极小极大下界（Minimax Rate）。

---

### 3. 强凸二次型条件数 $\kappa$ 与收缩率严格推导 ($\kappa$ 瓶颈)

在真实的深度神经网络损失局部极小值附近，函数常被局部二次逼近所支配：

$$\mathcal{L}(\theta) = \frac{1}{2} \theta^\top H \theta$$

其中 Hessian 矩阵对称正定 $H \succ 0$，特征值谱为 $\mu = \lambda_{\min} \le \dots \le \lambda_{\max} = L$，定义曲率条件数 $\kappa = \frac{L}{\mu}$。

#### 坐标解耦与单步收缩率

设最优点为 $\theta^* = \mathbf{0}$。SGD 更新式为：

$$\theta_{t+1} = \theta_t - \eta H \theta_t = (I - \eta H) \theta_t$$

利用 $H$ 的正交特征分解 $H = Q \Lambda Q^\top$，在特征向量坐标系下各维度完全解耦：

$$\theta_{t+1}^{(i)} = (1 - \eta \lambda_i) \theta_t^{(i)}$$

为了保证所有特征维度均不发散，必须满足谱半径约束：

$$|1 - \eta \lambda_i| < 1, \quad \forall i \iff -1 < 1 - \eta \lambda_{\max} < 1 \implies \eta < \frac{2}{\lambda_{\max}} = \frac{2}{L}$$

整个系统的最差单步收缩因子（Contraction Factor）由极端特征值决定：

$$\rho(\eta) = \max_{\lambda \in [\mu, L]} |1 - \eta \lambda| = \max(|1 - \eta \mu|, |1 - \eta L|)$$

最优步长必须平衡陡峭维度的下界与平缓维度的上界：

$$1 - \eta^* \mu = \eta^* L - 1 \implies \eta^* = \frac{2}{L + \mu}$$

代入最优步长，得到 SGD 在二次型下的最优单步收缩因子：

$$\rho_{\text{SGD}}^* = \frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1} = 1 - \frac{2}{\kappa + 1} \approx 1 - \frac{2}{\kappa}$$

#### 步数复杂度

要使误差缩小到原始的 $\epsilon$（即 $\|\theta_T - \theta^*\| \le \epsilon \|\theta_0 - \theta^*\|$）：

$$\left(1 - \frac{2}{\kappa}\right)^T \le \epsilon \implies T \ln\left(1 - \frac{2}{\kappa}\right) \le \ln \epsilon \implies T \ge \frac{\kappa}{2} \ln\left(\frac{1}{\epsilon}\right) = O\left(\kappa \log \frac{1}{\epsilon}\right)$$

> **物理困境**：当条件数 $\kappa = 10,000$ 时，SGD 单步收缩率仅为 $0.9998$，每步只能消除万分之二的残差，需要上万步才能完成收敛。

---

### 4. Polyak 重球动量加速推导：从 $\kappa$ 到 $\sqrt{\kappa}$ 的平方根加速

Polyak 重球法引入动量缓冲 $m_t = \beta m_{t-1} + H \theta_t$，$\theta_{t+1} = \theta_t - \eta m_t$。消去中间变量 $m_t$ 得二阶差分方程：

$$\theta_{t+1} = (1 + \beta - \eta H) \theta_t - \beta \theta_{t-1}$$

#### 增广状态转移矩阵与特征多项式

将二阶递推写为状态空间矩阵形式：

$$\begin{bmatrix} \theta_{t+1} \\ \theta_t \end{bmatrix} = \begin{bmatrix} (1 + \beta)I - \eta H & -\beta I \\ I & 0 \end{bmatrix} \begin{bmatrix} \theta_t \\ \theta_{t-1} \end{bmatrix}$$

对于任意特征值 $\lambda \in [\mu, L]$，该转移矩阵的特征方程为：

$$\det \begin{bmatrix} (1 + \beta - \eta \lambda) - \rho & -\beta \\ 1 & -\rho \end{bmatrix} = \rho^2 - (1 + \beta - \eta \lambda) \rho + \beta = 0$$

判别式为：

$$\Delta(\lambda) = (1 + \beta - \eta \lambda)^2 - 4\beta$$

#### 共轭复根与全谱等距衰减

当参数使得对于全谱 $\lambda \in [\mu, L]$ 均有 $\Delta(\lambda) \le 0$ 时，特征根为一对**共轭复数**：

$$\rho_{1, 2} = \frac{(1 + \beta - \eta \lambda) \pm i \sqrt{4\beta - (1 + \beta - \eta \lambda)^2}}{2}$$

其复模长精确等于：

$$|\rho| = \sqrt{\rho_1 \rho_2} = \sqrt{\beta}$$

> **代数本质**：在共轭复根区域内，转移矩阵的谱半径**与具体的曲率 $\lambda$ 完全无关**，所有特征维度的能量衰减率被强行拉平成完全相等的常数 $\sqrt{\beta}$！

为使 $\Delta(\lambda) \le 0$ 在整个区间 $[\mu, L]$ 恒成立，最优参数选取使得区间两端点刚好触碰判别式零点：

$$1 + \beta - \eta \mu = 2\sqrt{\beta}, \quad 1 + \beta - \eta L = -2\sqrt{\beta}$$

解此二元一次方程组，解得最优阻尼 $\beta^*$ 与最优学习率 $\eta^*$：

$$\beta^* = \left( \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} \right)^2 = \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^2$$
$$\eta^* = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}$$

#### 加速收敛率

Polyak 动量的最优收缩因子为：

$$\rho_{\text{Mom}}^* = \sqrt{\beta^*} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \approx 1 - \frac{2}{\sqrt{\kappa}}$$

所需步数复杂度为：

$$T_{\text{Mom}} = O\left(\sqrt{\kappa} \log \frac{1}{\epsilon}\right)$$

| 算法 | 收缩因子 $\rho^*$ | $\kappa = 10,000$ 时的收缩因子 | 相对步数复杂度 |
| :--- | :--- | :--- | :--- |
| **SGD** | $\frac{\kappa - 1}{\kappa + 1} \approx 1 - \frac{2}{\kappa}$ | $0.9998$ | $O(\kappa \log(1/\epsilon)) \approx 10,000$ 步 |
| **Polyak 动量** | $\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \approx 1 - \frac{2}{\sqrt{\kappa}}$ | $0.9802$ | $O(\sqrt{\kappa} \log(1/\epsilon)) \approx 100$ 步 (**提速 100 倍**) |

---

### 5. 对角预条件 (Adam) 与矩阵谱正交 (Muon) 在旋转曲面下的几何分歧

#### 对角预条件化 (Adam) 的坐标依赖局限

Adam 维护二阶矩 $v_t \approx \text{diag}(g_t^2)$，通过对角矩阵预条件化参数更新：

$$\Delta \theta = -P_{\text{Adam}}^{-1} g_t, \quad P_{\text{Adam}} = \text{diag}(\sqrt{v_t} + \epsilon)$$

- **坐标轴对齐时**：若 Hessian 矩阵为对角阵（各参数坐标正交无关），对角预条件矩阵 $P_{\text{Adam}} \approx \text{diag}(H)^{1/2}$ 能完美拉伸各个坐标轴，将狭长椭圆等高线压缩为各向同性正圆，此时有效条件数 $\kappa_{\text{eff}} \approx 1.0$。
- **旋转曲面退化（交叉耦合项存在）**：若 Hessian 矩阵包含非对角耦合元素（$H = R_\phi \Lambda R_\phi^\top$），由于对角矩阵无法表达坐标系的旋转基底，Adam 仅能沿固定的直角坐标轴进行独立缩放。这导致更新轨迹退化为类似 $L_\infty$ 范数超立方体的锯齿状折线，在旋转峡谷的次级方向来回超调。

#### Muon 谱正交投影的代数优势

针对 Transformer 中占据主导参数量的 2D 线性层矩阵权重 $W \in \mathbb{R}^{m \times n}$，Muon 通过极分解直接计算动量矩阵的正交因子：

$$\mathcal{O}(M) = U V^\top = M (M^\top M)^{-1/2}$$

- **严格酉旋转不变性 (Unitary Invariance)**：对任意正交旋转矩阵 $Q_1, Q_2$，均有 $\mathcal{O}(Q_1 M Q_2) = Q_1 \mathcal{O}(M) Q_2$。Muon 的更新几何由矩阵本身的算子结构决定，彻底免疫坐标轴选取偏差。
- **奇异值全谱规整**：奇异值分解为 $M = \sum \sigma_i u_i v_i^\top$。经过极分解后，$\mathcal{O}(M) = \sum 1.0 \cdot u_i v_i^\top$，所有正交特征方向的能量增益严格恒等于 1.0，从根本上消除了矩阵参数空间内部的谱病态性。

---

## 02. 深度拆解 Adam：矩估计、偏差修正与有效步长

Kingma & Ba (ICLR 2015) 提出的 **Adam (Adaptive Moment Estimation)** 是过去十年深度学习中使用最广泛的自适应优化器。

### 1. 一阶矩与二阶矩的统计物理本质

Adam 在每个时间步维护两个状态向量：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(一阶矩估计：均值 / 速度)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(二阶未中心化矩估计：方差代理 / 能量)}$$

- **一阶矩 $m_t$**：梯度的指数移动加权平均（EMA），代表梯度的**主导方向（Velocity）**。
- **二阶矩 $v_t$**：梯度逐元素平方 $g_t \odot g_t$ 的 EMA，代表参数在各个坐标轴上的**历史振幅/不确定性（Energy / Variance Proxy）**。若某个维度的梯度经常很大，则 $v_t$ 很大；若几乎没有梯度，则 $v_t$ 趋近于 0。

### 2. 偏差修正 (Bias Correction) 的数学推导

#### 为什么必须做偏差修正？

在初始化阶段，动量向量通常设为全零：$m_0 = \mathbf{0}, v_0 = \mathbf{0}$。这会导致在训练初期，特别是在前几步，$m_t$ 与 $v_t$ 严重向零偏置（冷启动低估）。

#### 严格期望推导

以一阶矩为例，展开递推展开式：

$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$

两边取数学期望。在局部平稳假设下（即前 $t$ 步的真实梯度期望近似不变 $\mathbb{E}[g_i] \approx \mathbb{E}[g_t]$）：

$$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right] \approx \mathbb{E}[g_t] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$

利用等比数列求和公式 $\sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}$：

$$\mathbb{E}[m_t] \approx \mathbb{E}[g_t] (1 - \beta_1) \cdot \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g_t] (1 - \beta_1^t)$$

同理可得：

$$\mathbb{E}[v_t] \approx \mathbb{E}[g_t^2] (1 - \beta_2^t)$$

#### 为什么二阶矩 $v_t$ 的偏差修正尤其致命？

典型的超参数为 $\beta_2 = 0.999$。在第 $t=1$ 步时：

$$1 - \beta_2^1 = 1 - 0.999 = 0.001$$

若不作修正，计算出来的 $v_1$ 会被真实值人为压小 **1000 倍**！
在更新参数时，$v_t$ 出现在分母 $\sqrt{v_t}$ 位置。没有偏差修正时，初始步长的分母被缩小了 $\sqrt{1000} \approx 31.6$ 倍，**导致初始有效步长暴增 30 余倍，直接造成第一步参数更新溢出导致模型 NaN 发散**！

因此，无偏估计修正项为：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### 3. 有效每参数步长 (Effective Per-Parameter Step Size)

最终更新公式为：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

#### SignSGD 等价性与步长有界性

考虑极端情况：假设某个坐标维度上的梯度方向保持高度一致（即 $g_1 \approx g_2 \approx \dots \approx g$）：

$$\hat{m}_t \approx g, \quad \hat{v}_t \approx g^2$$
$$\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} \approx \frac{g}{\sqrt{g^2}} = \text{sign}(g)$$

此时，Adam 在该坐标上的单步更新量精确恒等于：

$$\Delta \theta = -\eta \cdot \text{sign}(g)$$

- **有效步长严格有界**：对每个参数而言，$|\Delta \theta| \lesssim \eta$。这赋予了 Adam 极佳的稳定性，任何维度的梯度爆发都不会导致超过 $\eta$ 的不可控突变。
- **尺度不变性（Scale Invariance）**：若将损失函数乘以常数 $c$（$\mathcal{L}' = c \mathcal{L}$），梯度变为 $c g$，分子 $m_t$ 放大 $c$ 倍，分母 $\sqrt{v_t}$ 同样放大 $c$ 倍，两者在比值中完美抵消。这意味着 Adam 对损失尺度的缩放完全免疫。
- **对角 Hessian 逆预条件化**：二阶矩 $\sqrt{v_t}$ 在几何上扮演了经验对角 Hessian $\text{diag}(H^2)^{1/4} \approx H_{ii}^{1/2}$ 的逆矩阵角色，自动拉伸平缓维度、压缩陡峭维度。

### 4. 典型超参数工业级配置

| 超参数 | 经典默认值 (CV/经典NLP) | 现代 LLM 预训练 (LLaMA/Mistral) | 核心原理与设计考量 |
| :--- | :--- | :--- | :--- |
| **学习率 $\eta$** | $10^{-3}$ | $1.5 \times 10^{-4} \sim 3 \times 10^{-4}$ | 大模型参数规模极大时，为防止注意力 Logits 饱和，基准学习率调低一个数量级 |
| **$\beta_1$** | $0.9$ | $0.9$ | 动量指数滑动平均衰减率，对应平均平滑窗口为 $\frac{1}{1-\beta_1} = 10$ 步 |
| **$\beta_2$** | $0.999$ | **$0.95 \sim 0.98$** | **重大演进**：$0.999$ 对应 $1000$ 步的超长历史记忆。在动态剧烈变化的 Transformer 预训练曲面上，过大的 $\beta_2$ 会滞后感知当前的曲率变化。调小至 $0.95$ 能更快淘汰陈旧方差 |
| **$\epsilon$** | $10^{-8}$ (float32) | **$10^{-6} \sim 10^{-5}$ (bf16/fp16)** | 在混合精度训练中，$10^{-8}$ 极易遭遇浮点下溢为 $0$，必须调大以保证数值安全性 |

---

## 03. Adam 的三大工业级致命陷阱

### 陷阱一：稀疏 Embedding 表中的 $\epsilon$ 陷阱与 Decay 陷阱

在大模型 Vocabulary Embedding（数万到数十万行）与推荐系统 ID 特征 Embedding（数千万到数十亿行）中，梯度高度稀疏（单批次只有被激活的少量 Token/Item 会产生非零梯度）：

#### 1. $\epsilon$ 诱发的梯度爆炸

设某长尾稀疏特征（如罕见生僻词或冷门商品 ID）在前 100,000 个 Step 中从未被采样过，其二阶矩 $v_t$ 持续按 $\beta_2$ 衰减至极限接近 0（$v_t \approx 0$）。
当它突然在某个 Step 被采样到，并产生了一个微弱的梯度 $g = 10^{-4}$ 时：

$$\Delta \theta = -\eta \cdot \frac{g}{\sqrt{0} + \epsilon} = -\eta \cdot \frac{10^{-4}}{10^{-8}} = -\eta \cdot 10^4$$

一个本应微不足道的冷门特征梯度，由于分母仅剩下 $\epsilon=10^{-8}$，**被瞬间放大了 $10^8$ 倍**！这直接导致该行 Embedding 权重被炸飞成巨大数值，在后续 Softmax 处引发整网 NaN。

- **工业解法**：
  1. 将 $\epsilon$ 移入根号内：$\sqrt{v_t + \epsilon}$；
  2. 采用 `torch.optim.SparseAdam`：**仅针对当前批次实际出现的行执行矩估计与更新**，未出现的行保持 $m_t, v_t$ 冻结，绝不空转衰减。

#### 2. 全局 Weight Decay 陷阱

若使用标准的密集 AdamW 对稀疏 Embedding 表优化，未被采样的数千万行虽然梯度为 0，但解耦权重衰减 $\theta_{t+1} = \theta_t (1 - \eta \lambda)$ 会在每一个 step 对**全表所有未访问行无差别执行连续萎缩**。几万步后，长尾特征即便未曾犯错也会被活活衰减成全零向量。

---

### 陷阱二：超大 Batch 训练下的泛化崩溃 (Large-Batch Pitfalls)

当分布式训练将全局 Batch Size 扩充到数万至数十万（例如 $B \ge 32\text{k} \sim 64\text{k}$）以追求硬件扩展率时，Adam 常常发生严重的泛化性能崩溃：

1. **随机梯度噪声坍塌**：小 Batch 下的 SGD/Adam 伴随天然的梯度方差噪声，该噪声能充当隐式正则化，帮助优化轨迹从陡峭的鞍点或较差的尖锐局部极小值（Sharp Minima）中逃逸到平坦极小值（Flat Minima）。在超大 Batch 下，梯度噪声几乎消失，Adam 极易收敛到尖锐坑底，导致训练 Loss 极低而测试集指标崩溃。
2. **超立方体顶点跳跃（Sign 效应放大）**：由于 $\Delta \theta \approx -\eta \cdot \text{sign}(g)$，在极低方差下，每个参数几乎都在步长边界 $\pm \eta$ 的超立方体顶点跳跃，缺乏连续模长自适应。
3. **层间更新比失衡**：在超大网络中，靠近输入端的底层特征范数 $\|\theta_l\|$ 与靠近输出的顶层范数可差数十倍。Adam 仅按坐标维度归一化，无法感知整层的相对更新尺度。
- **工业解法**：在大 Batch 预训练中切换至 **LAMB (Layer-wise Adaptive Moments for Batch Training)**，在 Adam 的基础上引入层级信任比（Trust Ratio）：

$$r_l = \frac{\|\theta_l\|_2}{\|u_l\|_2}, \quad \text{其中 } u_l = \frac{\hat{m}_l}{\sqrt{\hat{v}_l} + \epsilon} + \lambda \theta_l$$
$$\theta_{l, t+1} = \theta_{l, t} - \eta \cdot \phi(r_l) \cdot u_l$$

---

### 陷阱三：非平稳推荐系统 (Non-Stationary RecSys Data) 的滞后震荡

推荐系统（CTR/CVR 预估）与自然语言大模型预训练的根本差异在于：**文本语言规律是静态的，而推荐系统数据分布是非平稳的（Non-Stationary Streaming Data）**。用户的实时兴趣漂移、运营活动上线、热点爆款视频的突发性爆发，都会导致真实条件概率 $P_t(y|x)$ 随时间剧烈漂移。

#### Adam 的滞后死穴

Adam 的二阶矩滑动衰减因子 $\beta_2=0.999$ 隐式假设了数据分布具有至少数百至数千步的“平稳窗口”：
1. **热点突发时的幽灵刹车**：某商品昨日极其平淡，其二阶矩 $v_t$ 很大（累积了大量的历史非点击方差）。当今日由于突发事件该商品变成热搜，高密度的点击梯度涌入时，陈旧的历史大 $v_t$ 会充当“幽灵刹车”，死死压制当前正向梯度的更新，导致模型无法快速冷启动该爆款。
2. **热点退潮时的过度发散**：当爆款降温后，过去几千步积累的小方差未能及时更新，导致残留的大步长在降温后频繁超调。
- **工业解法**：在工业级流式推荐精排中，业界普遍采用 **FTRL-Proximal (Follow-The-Regularized-Leader)** 或动态重置二阶矩的优化器，直接以单样本累加的广义二阶梯度为依据，配合显式 $L_1$ 正则化产出高稀疏、瞬时响应的权重。

---

## 04. 理论跃迁：从标量、对角预条件到 Muon 谱正交优化器

优化器的演进历史本质上是**如何设计参数更新的预条件变换（Preconditioning）的历史**：

| 优化算法 | 预条件矩阵形态 | 矩阵空间几何解释 | 显存/计算代价 | 局限与适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | 标量标量：$P = \frac{1}{\eta} I$ | **各向同性球体**：所有方向一视同仁 | 0 额外状态 | 病态曲率下沿窄谷严重震荡 |
| **Adam / AdamW** | 坐标轴对角矩阵：$P = \text{diag}(\sqrt{v})$ | **坐标对齐椭球**：独立缩放每个标量维度 | $2\times$ 参数量显存 ($m, v$) | 将 2D 矩阵拆散为 1D 数组，无视矩阵流形结构 |
| **Muon (2024)** | **矩阵谱正交投影**：$\mathcal{O}(M) = U V^\top$ | **等距正交变换**：所有奇异值归一化为 1.0 | 仅 $1\times$ 参数量显存 (动量 $M$) | **专为 2D 核心权重矩阵设计**，训练收敛提速 1.5~2 倍 |

```mermaid
flowchart LR
    A["SGD: 标量缩放 -η·G"] --> B["Adam: 对角预条件 -η·diag(v)^(-1/2)·G"]
    B --> C["Muon: 矩阵谱正交化 -η·Orthogonal(M)"]
    
    style A fill:#ff4757,stroke:#333,color:#fff
    style B fill:#38bdf8,stroke:#333,color:#000
    style C fill:#10b981,stroke:#333,color:#fff
```

### 1. Muon (Momentum Orthogonal Optimizer) 的数学机理

由 Keller Jordan 于 2024 年末提出、并在开源大模型 pretraining 中引发轰动的 **Muon**，针对现代 Transformer 中占据 95% 以上 FLOPs 的 2D 线性层矩阵权重（$W \in \mathbb{R}^{m \times n}$，如 QKV 投影、FFN 上升下降矩阵），彻底抛弃了逐元素对角缩放。

#### 极分解 (Polar Decomposition) 与谱归一化

任何实矩阵 $M \in \mathbb{R}^{m \times n}$（设 $m \le n$）的奇异值分解（SVD）为 $M = U \Sigma V^\top$。其极分解的正交因子为：

$$\mathcal{O}(M) = U V^\top$$

- **能量等距守恒（Isometry）**：正交投影矩阵 $U V^\top$ 的所有奇异值 $\sigma_i$ **全部精确恒等于 1.0**！
- **杜绝任何方向的特征塌缩**：Adam 独立缩放坐标轴，容易把某些方向的权重更新压制过小或放大过大；Muon 保证了动量矩阵在所有正交谱方向上的能量传播是绝对均匀、等距同构的。

### 2. 硬件加速：Newton-Schulz 五阶多项式极速迭代

如果每个 Step 都在 GPU 上对 $4096 \times 4096$ 的矩阵做一次真实的 SVD 分解，其 $O(n^3)$ 的立方复杂度将彻底击垮训练吞吐。
Muon 的精妙之处在于：**采用五阶 Newton-Schulz 迭代，纯粹依靠 5 次矩阵乘法（GEMM）在 Tensor Core 上以极高算力吞吐逼近极分解**！

迭代算法如下：
1. 初始尺度归一化：$X_0 = \frac{M}{\|M\|_F + \epsilon}$
2. 五阶多项式迭代 5 步：
   $$A = X_k X_k^\top$$
   $$B = b A + c A^2$$
   $$X_{k+1} = a X_k + B X_k$$
   其中常数 $(a, b, c) = (3.4445, -4.7750, 2.0315)$ 经过严密的多项式逼近优化，保证在零点附近的收敛速度最大化。
3. 形状自适应步长缩放：最终更新矩阵乘上宽高比系数 $\sqrt{\max(1, m/n)}$。

---

## 05. 完整代码实现：Muon 与 AdamW 混合优化器

在工业级实践中，优化器采用**分工混合配置**：
- **Muon**：负责全体隐藏层 2D 矩阵（Attention 投影矩阵、FFN 门控与升降矩阵）；
- **AdamW**：负责 1D 向量（RMSNorm 增益参数、Bias）与不适宜矩阵正交化的 Token Embedding 表。

```python
import torch
import torch.nn as nn
from typing import List, Dict

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    通过五阶 Newton-Schulz 迭代计算矩阵 G 的第零次幂（极分解正交因子 U * V^T）。
    纯依靠矩阵乘法，Tensor Core 原生极速执行。
    """
    assert G.ndim == 2, "Muon 仅支持 2D 权重矩阵"
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # 强制在 bfloat16 或 float32 精度下运算
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (X.norm() + eps)
    
    # 保证行数 <= 列数以降低内积矩阵维度
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        
    if transposed:
        X = X.T
        
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon (Momentum Orthogonal Optimizer) 纯 PyTorch 实现。
    专用于大模型中的 2D 线性层权重。
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # 仅维护单一动量缓冲区，相比 Adam 节省整整一半显存状态！
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                # Nesterov 加速修正
                update_grad = g.add(buf, alpha=momentum) if nesterov else buf
                
                # 矩阵正交化投影
                ortho_update = zeropower_via_newtonschulz5(update_grad, steps=ns_steps)
                
                # 宽高比缩放校准
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.data.add_(ortho_update, alpha=-lr * scale)

        return loss

def setup_hybrid_optimizers(model: nn.Module, lr_muon: float = 0.02, lr_adam: float = 3e-4, weight_decay: float = 0.01):
    """
    工业级混合优化器分配：
    - 2D 内部核心权重矩阵 -> Muon
    - 1D 向量与 Embedding -> AdamW
    """
    muon_params: List[nn.Parameter] = []
    adam_decay_params: List[nn.Parameter] = []
    adam_no_decay_params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 仅针对内部二维隐藏权重应用 Muon；排除 Embedding 表
        if p.ndim == 2 and "token_embedding" not in name and "lm_head" not in name:
            muon_params.append(p)
        elif p.ndim >= 2:
            adam_decay_params.append(p)
        else:
            adam_no_decay_params.append(p)

    opt_muon = Muon(muon_params, lr=lr_muon, momentum=0.95)
    opt_adam = torch.optim.AdamW([
        {"params": adam_decay_params, "weight_decay": weight_decay},
        {"params": adam_no_decay_params, "weight_decay": 0.0},
    ], lr=lr_adam, betas=(0.9, 0.95))

    return opt_muon, opt_adam
```

---

## 06. 面试高频深度八股与结构化作答

### Q1：Adam 的一阶矩和二阶矩在统计物理上代表什么？偏差修正为什么在数学上是必需的？
- **答题核心**：
  1. **一阶矩 $m_t$**：梯度的指数滑动平均，代表局部参数更新的速度与主导方向（过滤高频随机震荡，累加共识方向）。
  2. **二阶矩 $v_t$**：未中心化方差的指数滑动平均，代表各坐标维度的能量与历史震荡剧烈程度（用于对角预条件归一化）。
  3. **偏差修正的本质**：由于 $m_0=0, v_0=0$ 冷启动，直接展开时 $\mathbb{E}[m_t] = \mathbb{E}[g_t](1 - \beta_1^t)$。尤其对于 $\beta_2=0.999$，首步时 $1 - \beta_2^1 = 0.001$。若无偏差修正，分母 $\sqrt{v_t}$ 会被无故低估 1000 倍，初始步长被恶性放大 30 余倍，直接导致第一步更新爆炸为 NaN。

---

### Q2：为什么说 Adam 在梯度方向稳定时等价于 SignSGD？这会带来什么利弊？
- **答题核心**：
  1. **数学等价**：当梯度方向连续相同时，$\hat{m}_t \approx g$，$\hat{v}_t \approx g^2$，更新项 $\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} \approx \text{sign}(g)$，步长退化为常数 $\eta$。
  2. **优势**：各维度单步更新量严格上界为 $\eta$，具备损失尺度不变性（Loss 缩放 $c$ 倍完全不影响步长），对梯度异常尖刺有天然免疫性。
  3. **劣势**：在大 Batch 或复杂曲面上，SignSGD 本质是在超立方体的顶点之间震荡跳转，缺乏连续模长的细粒度自适应，极易在尖锐局部极小值附近卡住，导致泛化性能劣于精细调节的 SGD。

---

### Q3：稀疏 Embedding 表使用 Adam 训练为什么容易崩溃？工业界如何避坑？
- **答题核心**：
  1. **$\epsilon$ 陷阱**：长尾稀疏 Token 长期未更新，$v_t$ 衰减至极限接近 0。一旦突然出现，分母仅剩 $\epsilon=10^{-8}$，梯度被瞬时放大 $10^8$ 倍触发 NaN。
  2. **Decay 陷阱**：密集 AdamW 会在每一个 Step 对整张 Embedding 表执行连续权重衰减，把未出现的长尾词特征强行萎缩为零。
  3. **解决方案**：将 $\epsilon$ 置于根号内 $\sqrt{v_t + \epsilon}$ 并调大至 $10^{-5}$；采用 `SparseAdam`，仅对当前 batch 命中的稀疏行执行矩累加与衰减。

---

### Q4：超大 Batch 训练（如 64k）中，为什么不能通过无脑调大 Adam 学习率来加速？LAMB 解决了什么问题？
- **答题核心**：
  1. **噪声缺失与层间失衡**：大 Batch 导致随机梯度方差过小，Adam 的 SignSGD 行为收敛到泛化极差的 Sharp Minima；同时底层与顶层的梯度模长差异极大，全局统一学习率会导致某些层发散而某些层欠拟合。
  2. **LAMB 机制**：在 Adam 的基础上引入层级自适应信任比 $r_l = \frac{\|\theta_l\|_2}{\|u_l\|_2}$，将各层的有效更新尺度直接绑定为该层参数自身的模长比例，成功将 BERT/LLM 预训练 Batch Size 扩展至数万级别且保持稳定收敛。

---

### Q5：实时推荐系统（RecSys CTR 预估）为什么普遍使用 FTRL 而弃用 Adam？
- **答题核心**：
  1. **非平稳数据分布**：推荐日志随时间动态漂移。Adam 的 $\beta_2=0.999$（约千步记忆）在爆款突发时形成“幽灵刹车”，在退潮时产生滞后震荡。
  2. **稀疏性与在线推断**：工业推荐系统依靠亿级特征参数，线上显存必须要求权重高度稀疏（绝大多数不重要特征权重严格为 0）。Adam 无法产生严格稀疏解；而 FTRL-Proximal 融合了 $L_1$ 正则与坐标级对偶平均，既能秒级响应新特征，又能保持 90% 以上的参数稀疏度以支持超低延迟线上推断。

---

### Q6：AdamW 和 Adam 的核心分歧是什么？为什么不能对 LayerNorm/RMSNorm 的可学习参数做 Weight Decay？
- **答题核心**：
  1. **L2 正则 vs 解耦衰减**：Adam 把正则化项 $\lambda \theta$ 当作梯度处理，导致频繁更新的大梯度参数衰减被分母压小，小梯度参数被过度衰减；AdamW 剥离正则化，直接执行 $\theta \leftarrow \theta(1 - \eta \lambda)$。
  2. **参数分组保护**：LayerNorm/RMSNorm 的 $\gamma$ 属于 1D 缩放增益向量（非矩阵投影）。若对其施加 Weight Decay，$\gamma$ 会被持续压低逼近 0，导致残差信号幅度整体坍缩，破坏深度网络的恒等信息流动。

---

### Q7：2024~2025 年爆火的 Muon 优化器为什么比 AdamW 快那么多？其谱归一化的数学原理是什么？
- **答题核心**：
  1. **矩阵流形视角**：Adam 把二维矩阵拉平成一维数组做独立坐标缩放，破坏了矩阵变换的代数特征；Muon 针对 2D 权重矩阵，通过极分解将其动量投影到正交流形 $\mathcal{O}(M) = U V^\top$。
  2. **奇异值恒等归一**：极分解矩阵的所有奇异值精确等于 1.0，保证更新在所有特征正交方向上能量完全均等传播，无任何主方向塌缩或爆炸。
  3. **硬件极致优化**：无需昂贵的 SVD，纯粹依靠 5 次五阶 Newton-Schulz 矩阵乘法迭代（GEMM）在 Tensor Core 上极速逼近正交投影，实测大模型预训练耗时削减 30%~50%。
---

### Q8：如何利用裂项相消（Telescoping Sum）推导梯度下降的 O(1/T) 与凸优化 SGD 的 O(1/√T) 收敛率界？
- **答题核心**：
  1. **光滑函数的下降引理**：由梯度 $L$-Lipschitz 条件积分展开得 $f(\theta_{t+1}) \le f(\theta_t) - \frac{1}{2L} \|\nabla f(\theta_t)\|^2$。移项得 $\|\nabla f(\theta_t)\|^2 \le 2L[f(\theta_t) - f(\theta_{t+1})]$。
  2. **梯度范数裂项相消**：累加 $t=0 \dots T-1$，右侧 $[f(\theta_0) - f(\theta_1)] + \dots + [f(\theta_{T-1}) - f(\theta_T)]$ 裂项相消为 $f(\theta_0) - f(\theta_T) \le f(\theta_0) - f^*$。两边除以 $T$ 得 $\min_{t} \|\nabla f(\theta_t)\|^2 \le \frac{2L(f(\theta_0)-f^*)}{T} = O(1/T)$。
  3. **凸优化距离势函数裂项**：展开欧氏距离 $\|\theta_{t+1} - \theta^*\|^2 \le \|\theta_t - \theta^*\|^2 - 2\eta (f(\theta_t) - f^*) + \eta^2 G^2$。累加 $T$ 步后中间距离项裂项相消，得到 $\sum (f(\theta_t) - f^*) \le \frac{R^2}{2\eta} + \frac{\eta T G^2}{2}$。由 Jensen 不等式并令关于 $\eta$ 导数为 0，解得最优步长 $\eta^* = \frac{R}{G\sqrt{T}}$，代入即得 $f(\bar{\theta}_T) - f^* \le \frac{RG}{\sqrt{T}} = O(1/\sqrt{T})$。

---

### Q9：从特征值角度严格推导，为什么 Polyak 动量能把病态二次曲面的收敛步数从 O(κ) 加速到 O(√κ)？为什么 Adam 在旋转曲面上会退化？
- **答题核心**：
  1. **SGD 的条件数瓶颈**：二次型更新 $\theta_{t+1} = (I - \eta H)\theta_t$。最优步长 $\eta = \frac{2}{L+\mu}$ 决定单步收缩因子 $\rho = \frac{\kappa-1}{\kappa+1} \approx 1 - \frac{2}{\kappa}$，步数复杂度为 $O(\kappa \log(1/\epsilon))$。
  2. **动量的共轭复根神迹**：增广状态转移矩阵的特征方程为 $\rho^2 - (1+\beta-\eta\lambda)\rho + \beta = 0$。当判别式 $\Delta \le 0$ 时，两根为共轭复根，模长恒等于常数 $|\rho| = \sqrt{\beta}$（与具体曲率 $\lambda$ 完全解耦）。令区间端点重合解得最优 $\beta^* = (\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1})^2$，收缩率提升为 $1 - \frac{2}{\sqrt{\kappa}}$，复杂度降为 $O(\sqrt{\kappa} \log(1/\epsilon))$。
  3. **Adam 的对角旋转缺陷**：Adam 仅用对角阵 $\text{diag}(\sqrt{v})$ 预条件化，本质是沿直角坐标系缩放。当 Hessian 存在旋转耦合（非对角非零）时，对角矩阵无法旋转坐标基底，更新轨迹在交叉方向产生锯齿超调；而 Muon 的极分解 $\mathcal{O}(M) = U V^\top$ 具备严格的酉旋转不变性，全谱奇异值归一化为 1.0，彻底消除矩阵空间的病态旋转干扰。
