# Quant 11 · 鞅、停时与随机游走：Wald 等式、鞅构造与最优时停

在顶尖量化对冲基金（Jane Street、Optiver、Citadel、SIG、Jump Trading、Two Sigma）的核心数学与概率面试中，**鞅（Martingale）与最优停时定理（Optional Stopping Theorem, OST）**是最具降维打击能力的王牌工具。

本讲把 Quant 10 里已经用过的工具（鞅、无套利概率）往数学基础的方向全面补全：鞅的严格定义、停时的定义、最优停时定理（OST）成立的三大充分条件与经典反例、Wald 等式族、鞅的四大构造模板、一维随机游走（Gambler's Ruin）的全套对称与不对称解析解，以及最优时停理论在量化高频真题（秘书问题 37% 定律、掷骰子博弈、美式期权、模式等待时间与赌场鞅、无放回抽卡的最优停止与 Doob 分解）中的深度实战应用。

```text
遇到随机游走 / 等待时间 / 最优时停问题的 5 步心智模型：
1. 确认过程是否为鞅：计算一步条件期望 E[X_{n+1} | F_n]，检验其是否恒等于 X_n（公平游戏性质）。
2. 确认停止时刻是否为合法停时：判断"要不要在时刻 n 停下"只能使用截止到时刻 n 的信息，严禁偷看未来。
3. 检验 OST 适用条件：在应用 E[X_T] = E[X_0] 前，先检查是否满足三大充分条件之一（T 有界、停止过程有界、或 E[T] < ∞ 且增量有界）；防止掉入 E[T] = ∞ 的未定式陷阱。
4. 匹配并构造目标鞅：
   - 求吸收概率 P(Hit a) -> 构造指数几何鞅 M_n = (q/p)^{S_n} 或调和函数鞅 f(X_n)；
   - 求无漂移对称游走期望时间 E[T] -> 构造二次方差鞅 M_n = S_n^2 - n 或应用二阶 Wald 等式；
   - 求带漂移游走期望时间 E[T] -> 构造线性漂移消除鞅 M_n = S_n - nμ 或应用一阶 Wald 等式；
   - 求字符串模式等待时间 E[T_Pattern] -> 构造赌场赌资累计净利润鞅（Li's Martingale）。
5. 最优时停与单步前瞻：若是多阶段择时决策，利用 Doob 分解 Y_n = N_n + A_n 将收益过程拆解为纯鞅与可预测漂移项；若单步增量 Δ_n 具有单调吸收性（Monotone Stopping），单步前瞻（1-SLA）首次 Δ_n <= 0 处即为全局最优停止点！
```

---

## 交互实验室：鞅论、随机游走与最优时停

```martingale-rw-demo
```

---

## 模块一：鞅（Martingale）的严格数学定义与性质

### 1. 严格数学三条件

设 $\{\mathcal{F}_n\}$ 是一列不断递增的信息流（filtration，$\mathcal{F}_n \subseteq \mathcal{F}_{n+1}$，代表“到时刻 $n$ 为止已知的全部历史信息”）。随机过程 $\{X_n\}$ 相对于 $\{\mathcal{F}_n\}$ 是**鞅（Martingale）**，需要同时满足三个条件：

$$
\text{(i) } X_n \text{ 是 } \mathcal{F}_n\text{-可测的} \qquad \text{(ii) } \mathbb{E}[|X_n|] < \infty \qquad \text{(iii) } \mathbb{E}[X_{n+1} \mid \mathcal{F}_n] = X_n
$$

- **条件 (i)**：$X_n$ 的取值完全由截止到时刻 $n$ 的信息决定，不包含未知变量。
- **条件 (ii)**：纯技术性的可积性要求，确保条件期望具有良定义的数学意义。
- **条件 (iii)（核心）**：站在时刻 $n$ 往前看一步，对 $X_{n+1}$ 的最优条件预测就是当前值 $X_n$ 本身。过程既不系统性向上漂移，也不系统性向下衰减，这正是“**公平博弈（Fair Game）**”的严格数学刻画。

### 2. 塔性质与固定时刻期望不变性

反复应用条件期望的**塔性质（Tower Property）** $\mathbb{E}[\mathbb{E}[\cdot \mid \mathcal{F}_{n+1}] \mid \mathcal{F}_n] = \mathbb{E}[\cdot \mid \mathcal{F}_n]$，可将一步预测递推至任意多步：

$$\mathbb{E}[X_m \mid \mathcal{F}_n] = X_n \quad (\forall m > n)$$

对全空间取无条件期望，立即导出鞅的一个核心守恒推论：

$$\mathbb{E}[X_n] = \mathbb{E}[X_0] \quad (\forall n \ge 0)$$

即鞅在任意固定确定性时刻的期望值永远守恒。

---

### 例题 1：二次方差鞅的构造与证明

设 $S_n = \sum_{i=1}^n X_i$（$S_0=0$）是简单对称随机游走，$X_i$ 独立同分布，$\mathbb{P}(X_i = 1) = \mathbb{P}(X_i = -1) = 1/2$。证明 $M_n = S_n^2 - n$ 是关于自然信息流 $\mathcal{F}_n = \sigma(X_1, \dots, X_n)$ 的鞅。

**严格证明**：
$M_n$ 显然是 $\mathcal{F}_n$-可测的且满足可积性。关键验证条件 (iii)：

$$\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = \mathbb{E}[(S_n + X_{n+1})^2 - (n+1) \mid \mathcal{F}_n] = \mathbb{E}[S_n^2 + 2S_n X_{n+1} + X_{n+1}^2 - n - 1 \mid \mathcal{F}_n]$$

由于 $S_n$ 是 $\mathcal{F}_n$-可测的常数可提出，而 $X_{n+1}$ 独立于 $\mathcal{F}_n$ 且 $\mathbb{E}[X_{n+1}] = 0$，$X_{n+1}^2 \equiv (\pm 1)^2 = 1$：

$$= S_n^2 + 2S_n \cdot \mathbb{E}[X_{n+1}] + \mathbb{E}[X_{n+1}^2] - n - 1 = S_n^2 + 2S_n(0) + 1 - n - 1 = S_n^2 - n = M_n$$

得证。

> [!TIP]
> **直觉本质**：$S_n^2$ 的期望平均以速度 $1$ 线性递增（每走一步贡献方差 $1$），减去时间项 $n$ 恰好抵消掉了系统性方差增长，剩下的部分便成为零漂移的公平鞅。

**常见追问 / 面试陷阱**：
> 追问：“$S_n$ 本身是不是鞅？”
> 答：是，$\mathbb{E}[S_{n+1} \mid \mathcal{F}_n] = S_n + \mathbb{E}[X_{n+1}] = S_n$。但 $S_n$ 只包含**一阶线性信息**（只能解吸收概率），而 $S_n^2 - n$ 包含了**二阶方差信息**，是求解期望吸收时间 $\mathbb{E}[T]$ 的王牌工具。

---

## 模块二：停时（Stopping Time）与信息流

### 1. 严格定义

随机变量 $T$（取值在 $\{0, 1, 2, \dots\} \cup \{\infty\}$）是关于信息流 $\{\mathcal{F}_n\}$ 的**停时（Stopping Time）**，若对任意 $n \ge 0$，事件：

$$\{T \le n\} \in \mathcal{F}_n$$

即事件 $\{T \le n\}$ 的真假完全由截止到时刻 $n$ 的信息决定。直觉上：**“要不要在当前时刻停下”的决策，只能依据当下和过去发生的信息做出，绝不能透支或偷看未来的信息**。

- **典型合法停时**：首次到达某个状态的时刻（如 $T = \min\{n : S_n = 5\}$），因为在时刻 $n$ 判断是否已触碰 5 仅需检查已知轨迹 $S_1, \dots, S_n$。
- **典型非法非停时**：过程最后一次归零的时刻。因为要确认某次归零是不是“最后一次”，必须预知未来是否还会归零，这是典型的透视未来信息。

---

### 例题 2：停时合法性判定

设 $S_n$ 为简单随机游走，分析下列两个时刻是否为合法停时：
- (a) $T_1 = \min\{n : S_n = 5\}$（首次到达 5 的时刻）
- (b) $T_2 = T_1 - 1$（首次到达 5 的**前一步**）

**解答**：
- $T_1$ 是合法停时：$\{T_1 \le n\} = \bigcup_{k=1}^n \{S_k = 5\} \in \mathcal{F}_n$。
- $T_2$ **不是停时**：在时刻 $n$ 要判断 $T_2 = n$，等价于判断“$S_n \ne 5$ 且 $S_{n+1} = 5$”，这需要依赖时刻 $n+1$ 才能揭晓的未来随机变量 $S_{n+1}$，故违背定义。

**常见追问 / 面试陷阱**：
> 追问：“两个停时的和、最小值、最大值是否仍是停时？”
> 答：是的。若 $T_1, T_2$ 是停时，则 $T_1 \wedge T_2 = \min(T_1, T_2)$、$T_1 \vee T_2 = \max(T_1, T_2)$ 以及 $T_1 + T_2$ 均仍为合法停时。例如 $\{T_1 \wedge T_2 \le n\} = \{T_1 \le n\} \cup \{T_2 \le n\} \in \mathcal{F}_n$。

---

## 模块三：最优停时定理（Optional Stopping Theorem）与三大充分条件

### 1. 定理陈述

设 $\{X_n\}$ 是关于 $\{\mathcal{F}_n\}$ 的鞅，$T$ 是合法停时。若满足下列**三大充分条件中的任意一条**，则停时时刻的期望等于初始期望：

$$\mathbb{E}[X_T] = \mathbb{E}[X_0]$$

```text
Doob 最优停时定理（OST）的三大充分条件（满足其一即可）：
条件 (A) 停时几乎必然有界：存在确定常数 K < ∞，使得 P(T <= K) = 1；
条件 (B) 停止过程几乎必然有界：存在确定常数 M < ∞，使得对所有 n，|X_{T ∧ n}| <= M a.s.；
条件 (C) 期望时间有限且增量有界：E[T] < ∞，且存在常数 c < ∞ 使得每步条件增量 |X_{n+1} - X_n| <= c a.s.。
```

---

### 例题 3：经典反例与致命陷阱（自由游走首中 1）

考虑标准一维对称随机游走 $S_n = \sum_{i=1}^n X_i$（$S_0=0$），定义停时 $T = \min\{n : S_n = 1\}$ 为首次到达 $+1$ 的时刻。
显然停止时 $S_T \equiv 1$，故 $\mathbb{E}[S_T] = 1$。
但初始期望 $\mathbb{E}[S_0] = 0$。如果盲目套用 $\mathbb{E}[S_T] = \mathbb{E}[S_0]$，会得出荒谬矛盾：$1 = 0$！

**深入病理剖析（为什么 OST 条件全部破裂？）**：
1. **条件 (A) 破裂**：游走没有时间上限，$T$ 无界；
2. **条件 (B) 破裂**：在到达 $+1$ 之前，游走可以向负方向漂移到任意深的负数（$-10^6, -10^9$），停止过程 $S_{T \wedge n}$ 向下无界；
3. **条件 (C) 破裂**：虽然由一维游走常返性可知 $\mathbb{P}(T < \infty) = 1$（游走必定最终到达 1），但**期望停止时间是无穷大** $\mathbb{E}[T] = \infty$！

> [!WARNING]
> **量化面试警钟**：只要看到无吸收边界的单边停时问题，必须高度警惕 $\mathbb{E}[T] = \infty$ 导致的 OST 失效！

---

## 模块四：Wald 等式族全景深度解析

Wald 等式是鞅论与 OST 在独立同分布（i.i.d.）求和下的直接推论。

### 1. 一阶 Wald 等式（Wald's First Identity）

#### 定理陈述
设 $X_1, X_2, \dots$ 为 i.i.d. 随机变量，$\mathbb{E}[|X_1|] < \infty$，记均值 $\mu = \mathbb{E}[X_1]$。设 $T$ 为停时且 $\mathbb{E}[T] < \infty$。则随机和 $S_T = \sum_{i=1}^T X_i$ 满足：

$$\mathbb{E}[S_T] = \mathbb{E}[T] \cdot \mathbb{E}[X_1]$$

#### 严格数学证明（示性函数展开与 Fubini 积分）
将随机停止和 $S_T$ 展开为示性函数级数：

$$S_T = \sum_{n=1}^\infty X_n \mathbf{1}_{\{T \ge n\}}$$

注意到事件 $\{T \ge n\} = \{T \le n - 1\}^c \in \mathcal{F}_{n-1}$ 完全由前 $n-1$ 步决定。因此，**随机变量 $X_n$ 与示性变量 $\mathbf{1}_{\{T \ge n\}}$ 严格独立**！

利用 Fubini-Tonelli 定理交换求和与数学期望积分（由 $\mathbb{E}[T] < \infty$ 和 $\mathbb{E}[|X_1|] < \infty$ 保证绝对收敛）：

$$\mathbb{E}[S_T] = \sum_{n=1}^\infty \mathbb{E}\left[ X_n \mathbf{1}_{\{T \ge n\}} \right] = \sum_{n=1}^\infty \mathbb{E}[X_n] \cdot \mathbb{E}\left[ \mathbf{1}_{\{T \ge n\}} \right] = \mathbb{E}[X_1] \sum_{n=1}^\infty \mathbb{P}(T \ge n)$$

根据离散非负随机变量尾概率积分公式 $\mathbb{E}[T] = \sum_{n=1}^\infty \mathbb{P}(T \ge n)$，立即得出：

$$\mathbb{E}[S_T] = \mathbb{E}[X_1] \cdot \mathbb{E}[T]$$

证毕。

---

### 2. 二阶 Wald 等式（Wald's Second Identity）

若二阶矩有限 $\mathbb{E}[X_1^2] < \infty$ 且 $\mathbb{E}[T] < \infty$，记方差 $\sigma^2 = \text{Var}(X_1) = \mathbb{E}[X_1^2] - \mu^2$，则：

$$\mathbb{E}\left[ (S_T - T\mu)^2 \right] = \sigma^2 \mathbb{E}[T]$$

特别地，当步长均值 $\mu = 0$ 的对称游走时：

$$\mathbb{E}[S_T^2] = \sigma^2 \mathbb{E}[T]$$

**证明**：构造二次方差鞅 $M_n = (S_n - n\mu)^2 - n\sigma^2$。应用 OST 令 $\mathbb{E}[M_T] = \mathbb{E}[M_0] = 0$，即刻得出结论。

---

### 3. Wald 指数等式（Wald's Exponential Identity）

设矩母函数（MGF）$M(\theta) = \mathbb{E}[e^{\theta X_1}]$ 存在。定义几何指数鞅：

$$M_n(\theta) = \frac{e^{\theta S_n}}{(M(\theta))^n}$$

在一致可积条件下应用 OST 即得：

$$\mathbb{E}\left[ \frac{e^{\theta S_T}}{(M(\theta))^T} \right] = 1$$

- **对 $\theta$ 求一阶导令 $\theta \to 0$**：直接导出 Wald 一阶等式 $\mathbb{E}[S_T] = \mu \mathbb{E}[T]$；
- **对 $\theta$ 求二阶导令 $\theta \to 0$**：直接导出 Wald 二阶等式 $\mathbb{E}[(S_T - T\mu)^2] = \sigma^2 \mathbb{E}[T]$；
- **求解停时拉普拉斯变换**：令 $M(\theta) = e^{-s}$，可直接求解 $\mathbb{E}[e^{-s T}]$！

---

## 模块五：什么时候用鞅？量化问题分类与构造万能 Cheatsheet

### 1. 识别“适合用鞅论解决”的核心特征

在量化面试与概率博弈中，当遇到以下**6 大场景特征**时，鞅论与 OST 往往能实现极其优美的降维打击：

1. **首达与吸收概率（First Hitting & Ruin Probabilities）**：已知初态，求过程首次触碰边界 $A$ 先于边界 $B$ 的概率；
2. **期望等待与退出步数（Expected Exit / Waiting Times）**：求过程首次触碰某个停止区域的平均步数 $\mathbb{E}[T]$；
3. **最优时停与边际搜索成本（Optimal Stopping & Search Costs）**：每步有固定继续成本或罚没风险，需要在“锁定利润”与“博取未来”之间做权衡；
4. **序列模式匹配（Sequential Pattern Matching）**：投掷硬币或字符串流中，求特定目标子串（如 $HHT, HTTH$）首次出现的平均等待时间或竞速胜率；
5. **极限定理与有界比例演化（Pólya's Urn & Proportion Processes）**：无放回抽样、罐模型比例、选民模型等比例演化问题；
6. **分支繁殖与灭绝概率（Branching & Extinction）**：每个粒子独立繁衍，求后代种群规模期望与灭绝概率。

---

### 2. 量化面试鞅构造万能决策矩阵（Master Martingale Cheatsheet）

| 问题场景分类 | 推荐构造的鞅形式 $M_n$ / $M_t$ | 输出的核心代数关系 / 求解公式 | 验证要点与常见陷阱 |
| :--- | :--- | :--- | :--- |
| **一维对称游走破产概率** | $M_n = S_n$ | $a \mathbb{P}_a + (-b)(1 - \mathbb{P}_a) = S_0 \implies \mathbb{P}_a = \frac{b + S_0}{a + b}$ | 边界有界，$\mathbb{E}[T] < \infty$ 自动满足 |
| **一维对称游走期望步数** | $M_n = S_n^2 - n$ | $\mathbb{E}[S_T^2] - \mathbb{E}[T] = 0 \implies \mathbb{E}[T] = a \cdot b$ | $\mu = 0$ 且 $\sigma^2 = 1$，注意二阶 Wald 等式 |
| **一维非对称游走破产概率** | $M_n = (q/p)^{S_n}$ | $\mathbb{P}_a (q/p)^a + (1 - \mathbb{P}_a)(q/p)^{-b} = 1$ | 根源于 $p(q/p) + q(p/q) = 1$ |
| **一维非对称游走期望步数** | $M_n = S_n - n(p - q)$ | $\mathbb{E}[T] = \frac{\mathbb{E}[S_T] - S_0}{p - q} = \frac{a \mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$ | 漂移率分母 $p - q \ne 0$，一阶 Wald 等式 |
| **单边自由首达等待时间** | Wald 指数鞅 $e^{\theta S_n} / M(\theta)^n$ | $\mathbb{E}[s^{T_a}] = \left( \frac{1 - \sqrt{1 - 4pqs^2}}{2ps} \right)^a$ | 自由单边 $\mathbb{P}(T < \infty) = 1$ 但 $\mathbb{E}[T] = \infty$（零漂移陷阱） |
| **序列模式匹配等待时间** | 赌场赌资累积净利润鞅 | $\mathbb{E}[T_A] = (A * A)_2 = \sum 2^k \mathbf{1}_{\{\text{前缀}=\text{后缀}\}}$ | Penney's Game 非传递性：后手总能构造克制序列 |
| **无放回抽样最优停止** | 比例鞅 $M_n = \frac{I_n}{N - n}$ 与 Doob 分解 | $\Delta_n = \mathbb{E}[Y_{n+1} - Y_n \mid \mathcal{F}_n]$，在 $\Delta_n \le 0$ 时停下 | 1-SLA 单步前瞻法，Chow-Robbins 单调停止定理 |
| **带固定重掷成本最优停止** | 超额收益 Wald 鞅 $\sum (Z_k - \mu_n)$ | $\mu_{n^*} = \mathbb{E}[\max(X - n^*, 0)] = c \implies \mathbb{E}[\text{Payoff}] = n^* + c$ | 边际收益等于边际成本（无差异均衡） |
| **带清零危险的停掷决策** | 截断吸收下鞅 $Y_n = S_n \mathbf{1}_{\{T > n\}}$ | 在安全区内 $\mathbb{E}[Y_{n+1} - Y_n \mid \mathcal{F}_n] = \mu_D > 0$ | 证明 $\sup_\tau \mathbb{E}[Y_\tau] > S_0$，得出必须继续投掷 |
| **连续漂移扩散首达概率** | 指数鞅 $M_t = \exp\left( -\frac{2\mu}{\sigma^2} X_t \right)$ | $\mathbb{P}(X_\tau = b) = \frac{e^{\gamma a} - 1}{e^{\gamma a} - e^{-\gamma b}}$（$\gamma = \frac{2\mu}{\sigma^2}$） | 漂移布朗运动 $X_t = \mu t + \sigma W_t$ 上的尺度变换 |
| **连续漂移扩散期望时间** | 线性鞅 $N_t = X_t - \mu t$ | $\mathbb{E}[\tau] = \frac{\mathbb{E}[X_\tau]}{\mu} = \frac{b p_b - a(1 - p_b)}{\mu}$ | 消除扩散项，由均值漂移项主导平均时间 |
| **分支过程灭绝概率** | 规模归一鞅 $Z_n / m^n$ 与母函数鞅 $s^{Z_n}$ | 灭绝概率 $\pi$ 是母函数不动点 $G(s) = s$ 的最小非负根 | $m \le 1$ 几乎必然灭绝（除平凡情况），$m > 1$ 部分存活 |
| **波利亚罐比例演化** | 比例鞅 $M_n = \frac{R_n}{R + B + n c}$ | $M_n \to M_\infty$ 几乎必然收敛，且 $M_\infty \sim \text{Beta}\left( \frac{R}{c}, \frac{B}{c} \right)$ | 鞅收敛定理（Martingale Convergence Theorem） |
| **无放回翻牌等期望划分** | 示性函数对称性 / 局部比例鞅 | 翻出第 1 张红牌期望步数 $\frac{B+R+1}{R+1}$，最后剩余黑牌 $\frac{B}{R+1}$ | $R$ 张红牌将 $B$ 张黑牌均匀分割为 $R+1$ 个等长区间 |

---

## 模块六：九大量化面试真题与实战博弈全景精解（9 High-Frequency Quant Problems & Rigorous Derivations）

---

### 真题 1：带固定重掷成本的最优停止与 Wald 鞅（Search Cost / Cost to Reroll via Wald Martingale）

> **通用题目建模（Optiver / Jane Street / IMC 量化笔试压轴题）**：
> 
> 设 $X_1, X_2, \dots$ 为独立同分布的随机变量，取值于离散集合 $\{1, 2, \dots, K\}$（服从离散均匀分布，$\mathbb{P}(X_k = x) = \frac{1}{K}$，或任意已知分布 $p_x$）。
> 规则如下：
> - 每次掷出点数 $X_k$，你可以选择**接受当前点数并终止游戏**，或者**支付固定重掷费 $c > 0$ 重新掷一次**；
> - 若在第 $N$ 次掷骰子后决定停止，前 $N-1$ 次每次支付了 $c$ 元重掷费，最终净收益为：
> 
> $$
> \text{Payoff} = X_N - c(N - 1)
> $$
> 
> - 试用鞅论与 Wald 等式求解：**最优停止阈值 $n^*$ 的一般判定方程是什么？期望净收益是多少？**

#### 1. 属于什么类型的鞅？
本题属于 **超额收益随机和的 Wald 鞅（Excess Return Wald Martingale）与边际成本无差异条件**。

#### 2. 记号与超额收益 Wald 鞅构造
由于各次投掷独立同分布，最优策略必然是**固定阈值策略**：设定一个阈值 $n$，当且仅当 $X_k > n$ 时停止。
停时定义为：

$$
N = \inf\{ k \ge 1 : X_k > n \}
$$

考虑每次掷骰子相较于保留阈值 $n$ 的“超额收益（Excess Return）”：

$$
Z_k = \max(X_k - n, 0)
$$

$Z_k$ 是独立同分布的非负随机变量，记其单步期望值为 $\mu_n = \mathbb{E}[Z_k] = \mathbb{E}[\max(X - n, 0)]$。
构造标准的 **Wald 鞅**：

$$
M_m = \sum_{k=1}^m (Z_k - \mu_n)
$$

#### 3. 应用可选停止定理（OST）
因为停止概率 $p = \mathbb{P}(X > n) > 0$，停时 $N$ 服从几何分布，$\mathbb{E}[N] = 1/p < \infty$，且增量 $|Z_k - \mu_n|$ 有界，完全满足 OST 条件 (C)：

$$
\mathbb{E}[M_N] = M_0 = 0 \implies \mathbb{E}\left[ \sum_{k=1}^N Z_k \right] = \mu_n \mathbb{E}[N]
$$

根据停时 $N$ 的定义：
- 对于前 $N-1$ 步（$k < N$），都有 $X_k \le n$，因此 $Z_k = 0$；
- 只有在第 $N$ 步，才会出现 $X_N > n$，此时 $Z_N = X_N - n$。

因此整个随机和级数**直接塌缩为单项**：

$$
\sum_{k=1}^N Z_k = Z_N = X_N - n
$$

代入 OST 结论立即得到关键期望等式：

$$
\mathbb{E}[X_N - n] = \mu_n \mathbb{E}[N] \implies \mathbb{E}[X_N] = n + \mu_n \mathbb{E}[N]
$$

#### 4. 确定通用最优阈值 $n^*$
将 $\mathbb{E}[X_N]$ 代入净收益的期望表达式：

$$
\mathbb{E}[\text{Payoff}] = \mathbb{E}[X_N - c(N - 1)] = \mathbb{E}[X_N] - c\mathbb{E}[N] + c = n + c + (\mu_n - c)\mathbb{E}[N]
$$

- 由于 $\mathbb{E}[N] > 0$，要使期望净收益最大化：
  - 若 $\mu_n > c$：继续重掷带来的超额期望收益大于成本 $c$，应当提高保留阈值 $n$ 继续重掷；
  - 若 $\mu_n < c$：每多重掷一步期望净亏损，随着 $\mathbb{E}[N]$ 变大会严重拉低总收益；
  - 因此最优阈值 $n^*$ 落在**边际超额收益恰好覆盖边际重掷成本的无差异均衡点**：

$$
\boxed{\mu_{n^*} = \mathbb{E}[\max(X - n^*, 0)] \approx c}
$$

此时最大化期望净收益直接简化为常数：

$$
\boxed{\mathbb{E}[\text{Payoff}^*] = n^* + c}
$$

#### 5. $K$ 面均匀骰子与成本 $c$ 的通用解析解
对于取值在 $\{1, 2, \dots, K\}$ 的均匀离散分布：

$$
\mu_n = \mathbb{E}[\max(X - n, 0)] = \frac{1}{K} \sum_{x=n+1}^K (x - n) = \frac{1}{K} \sum_{j=1}^{K-n} j = \frac{(K - n)(K - n + 1)}{2K}
$$

令 $\mu_n \approx c$，解出连续近似阈值：

$$
(K - n)^2 + (K - n) \approx 2Kc \implies K - n^* \approx \frac{-1 + \sqrt{1 + 8Kc}}{2} \implies \boxed{n^* \approx K + \frac{1}{2} - \sqrt{2Kc + \frac{1}{4}}}
$$

**特例代入验证（$K=6$ 面标准骰子，重掷费 $c=1$ 元）**：
- 计算各整数阈值的 $\mu_n$：
  - $n = 3$：$\mu_3 = \frac{(6-3)(6-3+1)}{12} = \frac{3 \times 4}{12} = 1 = c$（边际收益与成本精确相等！）；
  - $n = 4$：$\mu_4 = \frac{2 \times 3}{12} = 0.5 < 1$（成本高于收益，不可取）。
- **最优决策**：选择 $n^* = 3$（即掷出点数 $\ge 4$ 停下，否则重掷）；
- **最大期望收益**：$\mathbb{E}[\text{Payoff}] = n^* + c = 3 + 1 = \mathbf{4}$ 元！

---

### 真题 2：字符串模式等待时间与李氏赌场鞅（Pattern Waiting Times & Li's Martingale）

> **经典面试题（Jane Street / Optiver 压轴真题）**：
> 
> 投掷均匀硬币（正面 $H$，反面 $T$），求目标序列 **$HHT$** 首次出现的期望投掷次数 $\mathbb{E}[T_{HHT}]$。并解释为什么它与 $HTH$ 以及 $HHH$ 的等待时间不同。

#### 1. 属于什么类型的鞅？
本题属于 **赌场赌资累积净利润鞅（Li's Casino Bankroll Martingale）与自重叠特征多项式（Autocorrelation Polynomial）**。

#### 2. 赌场赌资累积净利润鞅模型
- 假设在每一个时间步 $n=1, 2, \dots$，都有一位新赌徒进场，投入 $\$1$ 本金；
- 第 $n$ 位赌徒押注第 $n$ 步掷出 $H$：若猜错，本金输光立刻离场；若猜中，连本带利获得 $\$2$，并把全部 $\$2$ 押在第 $n+1$ 步掷出 $H$；
- 若第 $n+1$ 步再次猜中，资金翻倍至 $\$4$，并将全部 $\$4$ 押在第 $n+2$ 步掷出 $T$；若猜中拿到 $\$8$ 离场（因为模式 $HHT$ 已经完整出现，游戏终止）。
- 赌场的累积净利润 $M_n = (\text{所有进场赌徒的总投入}) - (\text{赌场支付给所有赌徒的总奖金})$ 是期望为 0 的**公平鞅**。

#### 3. 应用 OST 求解期望时间
当模式 $HHT$ 在停时 $T$ 首次出现时，共有 $T$ 位赌徒进场投入了 $\$T$。此时检查留在场上的活跃赌徒：
1. **在时刻 $T-2$ 进场的赌徒**：依次押中了 $H, H, T$，成功完成整个模式，赢得 $\$2^3 = \$8$；
2. **在时刻 $T-1$ 进场的赌徒**：其押注的前两位是 $H, H$，而实际出现的是模式的后两位 $H, T$（第 2 位不匹配），已输光出局，奖金为 $\$0$；
3. **在时刻 $T$ 进场的赌徒**：其押注的第 1 位是 $H$，而实际出现的是模式的最后一位 $T$，输光出局，奖金为 $\$0$。

由 OST $\mathbb{E}[M_T] = \mathbb{E}[M_0] = 0$：

$$
\mathbb{E}[T - 8 - 0 - 0] = 0 \implies \boxed{\mathbb{E}[T_{HHT}] = 8}
$$

#### 4. 常见模式对比与自重叠多项式
- **$HHT$**（无非平凡自重叠）：$\mathbb{E}[T_{HHT}] = 2^3 = 8$
- **$HTH$**（存在长度为 1 的自重叠 `H`）：$\mathbb{E}[T_{HTH}] = 2^3 + 2^1 = 10$
- **$HHH$**（存在长度为 1, 2 的自重叠 `H`, `HH`）：$\mathbb{E}[T_{HHH}] = 2^3 + 2^2 + 2^1 = 14$

---

### 真题 3：无放回抽卡惩罚模型与 Doob 分解（Sampling Without Replacement & 1-SLA）

> **通用题目建模（Akuna / SIG / Citadel 量化面试真题）**：
> 
> 桌面上有一副洗匀的牌，共 **$B$ 张蓝牌** 与 **$1$ 张红牌**（卡池总牌数为 $N = B + 1$ 张）。
> 规则如下：
> - 你每次从牌堆中**无放回**地抽取一张牌；
> - 每抽到一张蓝牌，你的累计得分 $+1$，你可以选择**立即停止游戏并带走当前所有得分**，或者选择**继续抽下一张**；
> - 如果抽到红牌，游戏**强制立即结束**，并且你的最终收益判定为 **$-(n-1)$**（即倒扣前面已抽出的全部 $n-1$ 分）；
> - 试用鞅论与 Doob 分解求解：**最优停止阈值 $n^*$ 的通用公式是什么？最大期望总收益是多少？**

#### 1. 属于什么类型的鞅？
本题属于 **无放回抽样中的比例鞅（Base Proportion Martingale）与可预测增量 Doob 分解（Doob Decomposition of Reward Process）**。

#### 2. 基础比例鞅的构造（Base Martingale）
设红牌被抽出的轮数记为停时 $T \in \{1, 2, \dots, B+1\}$。由于洗牌均匀，红牌等可能出现在第 $1$ 到第 $B+1$ 个位置，即 $\mathbb{P}(T = k) = \frac{1}{B+1}$。
引入指示变量 $I_n = \mathbf{1}_{\{T > n\}}$，表示前 $n$ 步全部抽中蓝牌（未出红牌）。
构造比例过程：

$$
M_n = \frac{I_n}{B + 1 - n} \quad (0 \le n \le B)
$$

**严格验证鞅性质**：
在已知历史信息 $\mathcal{F}_n$ 且 $I_n = 1$（前 $n$ 步未出红牌）的条件下，剩余 $B+1-n$ 张牌中包含 $1$ 张红牌、$B-n$ 张蓝牌。下一步不出红牌的条件概率为：

$$
\mathbb{E}[I_{n+1} \mid \mathcal{F}_n] = I_n \cdot \mathbb{P}(T > n+1 \mid T > n) = I_n \cdot \frac{B - n}{B + 1 - n}
$$

两边除以剩余牌数 $(B + 1) - (n + 1) = B - n$：

$$
\mathbb{E}\left[ \frac{I_{n+1}}{B + 1 - (n + 1)} \;\middle|\; \mathcal{F}_n \right] = \frac{I_n \cdot \frac{B - n}{B + 1 - n}}{B - n} = \frac{I_n}{B + 1 - n} = M_n
$$

而在 $I_n = 0$（已经出了红牌）时，$I_{n+1} \equiv 0$，等式 $0 = 0$ 恒成立。
因此 $M_n$ 是一个**标准的严格离散鞅**。

#### 3. 收益过程与 Doob 分解（Doob Decomposition）
定义在第 $n$ 步主动停下的累计收益过程为 $Y_n$。
在 $I_n = 1$（当前尚未抽到红牌，手握 $n$ 分）时，计算**再抽一张牌**的单步条件期望增量 $\Delta_n$：

$$
\Delta_n = \mathbb{E}[Y_{n+1} - Y_n \mid \mathcal{F}_n] = \underbrace{\frac{B - n}{B + 1 - n} \cdot (n + 1)}_{\text{抽中蓝牌，得分变为 } n+1} + \underbrace{\frac{1}{B + 1 - n} \cdot (-n)}_{\text{抽中红牌，罚没变为 } -n} - \underbrace{n}_{\text{当前收益}}
$$

将通分后的分子展开并化简：

$$
(B - n)(n + 1) - n - n(B + 1 - n) = (B + Bn - n - n^2) - n - (Bn + n - n^2) = B - 3n
$$

因此，得到极其优雅的通用单步增量公式：

$$
\Delta_n = \mathbb{E}[Y_{n+1} - Y_n \mid \mathcal{F}_n] = \frac{B - 3n}{B + 1 - n} I_n
$$

由此给出收益过程 $Y_n$ 的 **Doob 显式分解**：

$$
Y_n = N_n + A_n, \quad \text{其中 } A_n = \sum_{k=0}^{n-1} \frac{B - 3k}{B + 1 - k} I_k
$$

其中 $N_n$ 为纯鞅，$A_n$ 为可预测累积漂移过程。

#### 4. 单步前瞻法（1-SLA）与通用最优停止策略
分析单步漂移项 $\Delta_n = \frac{B - 3n}{B + 1 - n}$ 的正负号变化：
- **当 $B - 3n > 0 \implies n < B/3$ 时**：$\Delta_n > 0$（处于严格下鞅区间，继续抽牌具有正期望边际收益，应该继续）；
- **当 $B - 3n \le 0 \implies n \ge B/3$ 时**：$\Delta_n \le 0$（进入上鞅区间，继续抽牌具有负期望边际收益，绝不继续！）。

由于单步增量 $\Delta_n$ 的分子 $B - 3n$ 随 $n$ 严格单调递减，一旦 $n \ge \lfloor B/3  floor$ 后，未来所有步的 $\Delta_k \le 0$ 恒成立。根据 **Chow-Robbins 单调停止定理（Monotone Stopping Theorem）**，单步前瞻准则（1-Step Look-Ahead）保证了局部最优解就是全局最优解！

> **通用最优策略（Optimal Stopping Policy）**：
> **当手头累积抽到 $n^*$ 张蓝牌时，立即选择停止游戏**，其中通用最优阈值为：
> 
> $$
> \boxed{n^* = \left\lfloor \frac{B}{3}  ight floor}
> $$
> 
> - **核心速记要诀**：**在单红牌全部倒扣罚没规则下，最优停止点永远落在总蓝牌数的 $1/3$ 处！**

#### 5. 通用期望收益闭式解（Exact Expected Payout Formula）
采用在 $n^*$ 步主动停止的策略：
- **情况 A（前 $n^*$ 步全为蓝牌）**：概率为 $\frac{\binom{B}{n^*}}{\binom{B+1}{n^*}} = \frac{B+1-n^*}{B+1}$，主动退出获得收益 $+n^*$ 分；
- **情况 B（红牌出现在第 $k$ 步，$1 \le k \le n^*$）**：概率恒为 $\frac{1}{B+1}$，收益为 $-(k-1)$ 分。

计算全期望：

$$
\mathbb{E}[\text{Payoff}] = \frac{B + 1 - n^*}{B + 1} \cdot n^* + \sum_{k=1}^{n^*} \frac{1}{B + 1} \cdot (-(k - 1)) = \frac{n^*(B + 1 - n^*)}{B + 1} - \frac{1}{B + 1} \frac{(n^* - 1)n^*}{2}
$$

通分化简分子：

$$
2n^*(B + 1 - n^*) - n^*(n^* - 1) = n^*(2B + 2 - 2n^* - n^* + 1) = n^*(2B + 3 - 3n^*)
$$

得到通用的期望收益闭式公式：

$$
\boxed{\mathbb{E}[\text{Payoff}] = \frac{n^*(2B + 3 - 3n^*)}{2(B + 1)}}
$$

**特例代入验证（$B=9$ 张蓝牌，$1$ 张红牌）**：
- 最优停止阈值：$n^* = \lfloor 9/3  floor = 3$；
- 期望收益：$\mathbb{E}[\text{Payoff}] = \frac{3 \times (2 \times 9 + 3 - 3 \times 3)}{2 \times (9 + 1)} = \frac{3 \times 12}{20} = \frac{36}{20} = \mathbf{1.8}$ 分！

---

### 真题 4：带漂移布朗运动双边界吸收概率与期望时间（Drifted BM Two-Sided Exit & Scale Function）

> **通用题目建模（Citadel / Two Sigma / Morgan Stanley Strats 面试）**：
> 
> 设资产对数收益率遵循漂移布朗运动 $X_t = \mu t + \sigma W_t$（其中 $X_0 = 0, \mu > 0, \sigma > 0$）。设定止损线 $-a < 0$ 和止盈线 $b > 0$。定义双边吸收停时 $\tau = \inf\{t \ge 0 : X_t = -a \text{ 或 } X_t = b\}$。
> 试用鞅方法求解：触碰止盈线 $b$ 的概率 $p_b = \mathbb{P}(X_\tau = b)$ 以及平均退出时间 $\mathbb{E}[\tau]$。

#### 1. 属于什么类型的鞅？
本题属于 **连续状态扩散过程中的指数几何鞅（Exponential MGF Martingale）与线性漂移消除鞅（Linear Drift Martingale）**。

#### 2. 指数鞅求解触碰概率
构造指数鞅 $M_t = \exp(-\gamma X_t)$，代入 $X_t$ 并令 $t$ 的系数为 0：

$$
\mathbb{E}[e^{-\gamma(\mu t + \sigma W_t)}] = \exp\left( -\gamma \mu t + \frac{1}{2}\gamma^2 \sigma^2 t \right) = 1 \implies -\gamma \mu + \frac{1}{2}\gamma^2 \sigma^2 = 0 \implies \gamma = \frac{2\mu}{\sigma^2}
$$

由于过程在 $[-a, b]$ 内有界，由 OST $\mathbb{E}[M_\tau] = M_0 = 1$：

$$
p_b e^{-\gamma b} + (1 - p_b) e^{\gamma a} = 1 \implies \boxed{p_b = \frac{e^{\gamma a} - 1}{e^{\gamma a} - e^{-\gamma b}}} \quad \left( \gamma = \frac{2\mu}{\sigma^2} \right)
$$

#### 3. 线性鞅求解期望退出时间
构造线性漂移消除鞅 $N_t = X_t - \mu t$。由 OST $\mathbb{E}[N_\tau] = 0$：

$$
\mathbb{E}[X_\tau] - \mu \mathbb{E}[\tau] = 0 \implies \boxed{\mathbb{E}[\tau] = \frac{\mathbb{E}[X_\tau]}{\mu} = \frac{b \cdot p_b - a(1 - p_b)}{\mu}}
$$

---

### 真题 5：双瓶放球取球博弈与对称随机游走鞅（2-Urn Ball Placement & Symmetric Random Walk Martingale）

> **通用题目建模（Jane Street / Two Sigma 压轴博弈题）**：
> 
> 桌上有两个空瓶（1 号瓶与 2 号瓶）。游戏共有 $N$ 轮操作（假设总轮数 $N = 2m$ 为偶数，如 $N=100$）。
> 规则如下：
> - 在每一轮开始前，系统会**以各 $1/2$ 的概率随机选定其中一个瓶子**；
> - 玩家在每轮操作前必须决定本轮是**“放球（Deposit a ball）”**还是**“取球（Retrieve a ball）”**；
> - 如果选择放球，系统将向选中的瓶子中放入 1 个球（球源无限）；
> - 如果选择取球，系统从选中的瓶子中取出 1 个球：若该瓶子非空，取球成功，玩家获得 1 分；若该瓶子为空，取球失败，获得 0 分；
> - 试用对换论证与随机游走鞅证明：**最优策略是什么？最大期望成功取球数是多少？**

#### 1. 属于什么类型的鞅？
本题属于 **对换论证策略降维（Pathwise Exchange Argument）+ 二项卷积（Binomial Convolution）+ 对称随机游走绝对值一阶矩鞅（Symmetric Random Walk Absolute First Moment）**。

#### 2. 策略结构简化（对换论证）
首先证明最优策略一定是**“先全部放球，后全部取球”**：
- 假设在某一轮策略中存在“先取球、后放球”的相邻子序列；
- 若将这两轮的决策顺序对换为“先放球、后取球”，在所有可能的瓶子随机选择样本路径（Sample Paths）下：
  - 若原序列在取球时瓶子恰好为空（取球失败），先放球则使瓶子非空，从而使得取球转化为成功；
  - 若原序列取球时瓶子已经非空，先放后取对最终球数与后续状态没有任何负面影响；
- 因此，“先放后取”在**每一条样本路径上的收益均逐点弱占优（Pointwise Dominates）**于任何交错策略。
- 结论：最优策略必然是：**前 $k$ 轮全选择放球，后 $N - k$ 轮全选择取球**（$0 \le k \le N$）。

#### 3. 收益过程与二项分布表示
设在前 $k$ 次放球中，有 $X$ 次随机放入了 1 号瓶，则有 $k - X$ 次放入了 2 号瓶；
在后 $N - k$ 次取球中，有 $R$ 次从 1 号瓶取，则有 $(N - k) - R$ 次从 2 号瓶取。
由于每轮选中 1 号瓶的概率独立为 $1/2$：

$$
X \sim \text{Bin}\left( k, \frac{1}{2} \right), \quad R \sim \text{Bin}\left( N - k, \frac{1}{2} \right), \quad X \text{ 与 } R \text{ 相互独立}
$$

从 1 号瓶和 2 号瓶成功取出的球数分别为 $\min(X, R)$ 和 $\min(k - X, (N - k) - R)$。总成功取球数 $Y$ 为：

$$
Y = \min(X, R) + \min(k - X, (N - k) - R)
$$

#### 4. 代数转化与二项卷积（Binomial Convolution）
应用经典恒等式 $\min(a, b) = \frac{a + b - |a - b|}{2}$：

$$
Y = \frac{(X + R) + (k - X + N - k - R) - |X - R| - |(k - X) - (N - k - R)|}{2} = \frac{N - \left( |X - R| + |(2k - N) - (X - R)| \right)}{2}
$$

引入核心卷积随机变量：

$$
Z = X + (N - k - R)
$$

注意：$X \sim \text{Bin}(k, 1/2)$，而 $(N - k) - R \sim \text{Bin}(N - k, 1/2)$。由二项分布的独立卷积性质：

$$
Z \sim \text{Bin}\left(k + (N - k), \frac{1}{2} \right) = \text{Bin}\left( N, \frac{1}{2} \right)
$$

**惊人发现：$Z$ 的概率分布完全与策略参数 $k$ 无关，永远服从总轮数为 $N$ 的标准二项分布！**
将 $X - R = Z - (N - k)$ 代回收益公式：

$$
Y = \frac{N}{2} - \frac{|Z - (N - k)| + |Z - k|}{2}
$$

#### 5. 三角不等式优化最优轮数 $k^*$
利用实数绝对值三角不等式 $|A| + |B| \ge |A - B|$：

$$
|Z - (N - k)| + |Z - k| \ge |(N - k) - k| = |N - 2k|
$$

当且仅当 $N - k = k \implies \boxed{k^* = \frac{N}{2} = m}$ 时，下界 $|N - 2k| = 0$ 取得全局极小，损耗被完全对称抵消！
此时收益公式化为极其优美的形式：

$$
Y = \frac{N}{2} - \left| Z - \frac{N}{2}  ight| = m - |Z - m|
$$

#### 6. 随机游走鞅与绝对值一阶矩精确计算
定义 $N$ 步标准对称随机游走鞅 $S_N = \sum_{i=1}^N \xi_i$（其中 $\mathbb{P}(\xi_i = \pm 1) = 1/2$）。
由于 $Z \sim \text{Bin}(N, 1/2)$，有同分布恒等式 $2Z - N \stackrel{d}{=} S_N$，因此 $|Z - m| \stackrel{d}{=} \frac{1}{2}|S_N|$。
总期望收益为：

$$
\mathbb{E}[Y] = m - \frac{1}{2} \mathbb{E}[|S_{2m}|]
$$

对于长度为 $2m$ 的简单对称随机游走，其绝对值的一阶矩有严格的组合数解析恒等式：

$$
\mathbb{E}[|S_{2m}|] = 2m \binom{2m}{m} 2^{-2m}
$$

代入即得**最大期望成功取球数的通用精确闭式解**：

$$
\boxed{\mathbb{E}[Y] = m - m \binom{2m}{m} 2^{-2m} \approx m - \sqrt{\frac{m}{\pi}}}
$$

**特例数值代入（$N = 100$ 轮，即 $m = 50$）**：
- 最优策略：**前 50 轮全选择放球，后 50 轮全选择取球**；
- 利用 Stirling 公式逼近 $\binom{100}{50} 2^{-100} \approx \frac{1}{\sqrt{50\pi}}$：

$$
\mathbb{E}[Y] = 50 - 50 \binom{100}{50} 2^{-100} \approx 50 - \sqrt{\frac{50}{\pi}} \approx 50 - 3.989 = \mathbf{46.011}
$$

---

### 真题 6：无放回翻牌经典——第一张红牌与最后一张红牌（Card Drawing Without Replacement Symmetry）

> **通用题目建模（Jane Street / SIG 高频面试题）**：
> 
> 一副洗匀的牌共有 **$R$ 张红牌** 与 **$B$ 张黑牌**（总牌数 $R + B$ 张），一张一张依次翻开。
> 1. 求翻出**第一张红牌**时，已经翻开的总牌数期望 $\mathbb{E}[T_1]$；
> 2. 求翻出**最后一张红牌**时，牌堆中**剩余黑牌数**的期望 $\mathbb{E}[\text{Remaining Black}]$。

#### 1. 属于什么类型的鞅？
本题属于 **示性变量置换对称性（Exchangeability & Indicator Symmetry）与区间等期望划分原理**。

#### 2. 示性变量对称性与等期望划分（Indicator Symmetry）
$R$ 张红牌在随机洗牌中将所有 $B$ 张黑牌分割成 $R + 1$ 个“间隔区（Bins）”：
- $I_0$：第 1 张红牌之前的黑牌数；
- $I_1$：第 1 张与第 2 张红牌之间的黑牌数；
- $\dots$
- $I_R$：最后一张红牌之后的黑牌数（即剩余黑牌数）。

由全排列的完全置换对称性，所有 $R + 1$ 个间隔中的黑牌数是**同分布且等期望的**：

$$
\mathbb{E}[I_0] = \mathbb{E}[I_1] = \dots = \mathbb{E}[I_R]
$$

由于所有间隔中的黑牌总和恒为 $B$：

$$
\sum_{k=0}^R \mathbb{E}[I_k] = (R + 1) \mathbb{E}[I_0] = B \implies \boxed{ \mathbb{E}[I_k] = \frac{B}{R + 1} }
$$

#### 3. 求解问题（1）与（2）
1. **翻出第一张红牌时的总翻开牌数**：
   包含第一张红牌之前的全部黑牌（数量为 $I_0$）以及那张红牌本身（$+1$）：
   $$\mathbb{E}[T_1] = \mathbb{E}[I_0] + 1 = \frac{B}{R + 1} + 1 = \boxed{\frac{B + R + 1}{R + 1}}$$
2. **翻出最后一张红牌时的剩余黑牌数**：
   恰好对应最后一个间隔 $I_R$：
   $$\mathbb{E}[\text{Remaining Black}] = \mathbb{E}[I_R] = \boxed{\frac{B}{R + 1}}$$

**特例代入（标准扑克牌 $R=26$ 红，$B=26$ 黑）**：
- 翻到第 1 张红牌的平均步数：$\mathbb{E}[T_1] = \frac{26 + 26 + 1}{26 + 1} = \frac{53}{27} \approx \mathbf{1.963}$ 张；
- 翻到最后一张红牌时剩余黑牌数：$\mathbb{E}[\text{Remaining Black}] = \frac{26}{27} \approx \mathbf{0.963}$ 张！

---

### 真题 7：带完全平方数清零风险的停掷决策与吸收下鞅（Square Hazard & Absorbing Submartingale）

> **通用题目建模（Citadel / Jump Trading 量化面试真题）**：
> 
> 玩家在数轴上累加投掷点数，当前累计和为 $S_0$。每次投掷的点数增量为 $D_k \in \{1, 2, \dots, K\}$（均值为 $\mu_D = \mathbb{E}[D_k] > 0$），累计和序列为 $S_n = S_0 + \sum_{k=1}^n D_k$。
> 规则如下：
> - 数轴上分布着一系列“危险陷阱点” $\mathcal{H} = \{H_1, H_2, \dots\}$（例如完全平方数序列 $1^2, 2^2, 3^2, \dots$）；
> - 记当前所处位置 $S_0$ 与下一个危险点 $H$ 的距离为 $d = H - S_0$（假设 $1 \le d \le K$）；
> - 若某一投掷后**累计和恰好等于危险点（$S_n = H$）**，游戏强制结束，之前的所有累计得分**瞬间清零且永久吸收为 0**；若越过危险点（$S_n > H$），则脱离当前危险；
> - 问：**玩家在当前 $S_0$ 应该选择“立即停下带走 $S_0$”，还是“继续投掷”？如何用下鞅与 OST 建立严格的期望下界证明？**

#### 1. 属于什么类型的鞅？
本题属于 **带吸收态的截断下鞅（Absorbing Submartingale with Safe-Zone Drift）与两步停止策略下界构造**。

#### 2. 过程定义与安全区下鞅性
定义危险停时（首次踩中危险点 $H$ 的时刻）：

$$
T = \inf\{n \ge 1 : S_n = H\}
$$

定义收益过程（若踩中 $H$ 则直接清零并永久吸收为 0）：

$$
Y_n = S_n \cdot \mathbf{1}_{\{T > n\}}
$$

考察越过危险点 $H$ 之后的“安全区间”：设下一个危险点为 $H_{\text{next}}$。
当 $S_n \in [H + 1, H_{\text{next}} - K]$ 时，由于单步最大步长为 $K$，距离下一个危险点至少为 $H_{\text{next}} - (H_{\text{next}} - K) = K$，因此下一掷**踩中下一个危险点的概率严格为 0**：

$$
\mathbb{P}(S_{n+1} \in \mathcal{H} \mid S_n \in [H+1, H_{\text{next}} - K]) = 0
$$

在该安全区间内，单步条件期望增量严格等于平均步长：

$$
\mathbb{E}[Y_{n+1} - Y_n \mid \mathcal{F}_n] = \mathbb{E}[D_{n+1}] = \mu_D > 0
$$

这证明在安全区域内，$Y_n$ 是一个**严格递增的下鞅（Submartingale）**！

#### 3. 构建两步停止策略 $\tau$ 并应用下鞅 OST
为了证明“继续投掷优于立即停下”，我们构造一个具体的**两步停止策略 $\tau$**：
1. **第 1 步（$n=1$）**：必须投掷第 1 次；
2. **第 2 步（$n=2$）**：若第 1 步未踩中 $H$（即 $D_1 \ne d$），此时 $S_1 \in [H+1, S_0+K]$ 已经安全越过了危险点 $H$，且处于严格下鞅安全区内，因此**至少再投掷第 2 次**，随后立刻停止锁定利润。

停时 $\tau$ 的定义：
- 若 $D_1 = d$（踩中 $H$），$\tau = 1$，收益 $Y_1 = 0$；
- 若 $D_1 \ne d$（越过 $H$），$\tau = 2$。

**计算该策略的期望收益 $\mathbb{E}[Y_\tau]$**：

$$
\mathbb{E}[Y_\tau] = \mathbb{P}(D_1 = d) \cdot 0 + \mathbb{P}(D_1 \ne d) \cdot \mathbb{E}[Y_2 \mid D_1 \ne d]
$$

在 $D_1 \ne d$ 的条件下，$S_1$ 落在安全区，其条件期望为：

$$
\mathbb{E}[S_1 \mid D_1 \ne d] = S_0 + \mathbb{E}[D_1 \mid D_1 \ne d] = S_0 + \frac{\sum_{k \ne d} k}{K - 1}
$$

第二步在安全区继续投掷一次，由下鞅性：

$$
\mathbb{E}[Y_2 \mid D_1 \ne d] = \mathbb{E}[S_1 \mid D_1 \ne d] + \mathbb{E}[D_2] = S_0 + \frac{\sum_{k \ne d} k}{K - 1} + \mu_D
$$

代入总期望得到通用下界公式：

$$
\boxed{\mathbb{E}[Y_\tau] = \frac{K - 1}{K} \left( S_0 + \frac{\sum_{k \ne d} k}{K - 1} + \mu_D \right) = \frac{K - 1}{K} S_0 + \frac{1}{K} \left( \sum_{k \ne d} k \right) + \frac{K - 1}{K} \mu_D}
$$

#### 4. 特例数值验证（$S_0 = 35$，$K = 6$ 面骰子，危险点 $H = 36 = 6^2$）
- 距离危险点 $d = 36 - 35 = 1$（踩中 36 的点数为 $D_1 = 1$）；
- 下一个完全平方数为 $49 = 7^2$。越过 36 后的最大位置为 $35 + 6 = 41$，距离 49 至少 $49 - 41 = 8 > 6$，完全处于安全区；
- $\mu_D = 3.5$，$\mathbb{P}(D_1 \ne 1) = 5/6$；
- 条件期望 $\mathbb{E}[S_1 \mid D_1 \ge 2] = 35 + \frac{2+3+4+5+6}{5} = 35 + 4 = 39$；
- 第二步期望 $\mathbb{E}[Y_2 \mid D_1 \ge 2] = 39 + 3.5 = 42.5$；
- 策略期望收益：

$$
\mathbb{E}[Y_\tau] = \frac{1}{6} \times 0 + \frac{5}{6} \times 42.5 = \frac{212.5}{6} \approx \mathbf{35.4167}
$$

- **严格结论**：
  - 若**立即停下**，收益为确定性的 $35$；
  - 若**继续投掷**，仅采用极其简单的两步策略就能获得期望收益 $35.42 > 35$；
  - 因此最优停止值必然满足 $\sup_\tau \mathbb{E}[Y_\tau] \ge 35.42 > 35$，**玩家必须选择继续投掷（Continue Rolling）**！

---

### 真题 8：波利亚罐模型与鞅极限定理（Pólya's Urn & Proportion Martingales）

> **通用题目建模（Jane Street / SIG 经典概率面试题）**：
> 
> 罐中初始有 $R$ 个红球与 $B$ 个黑球。每次随机抽出一球，观察颜色后将该球放回，并**额外放入 $c$ 个同色球**。
> 设第 $n$ 步罐中红球比例为 $M_n$。证明 $M_n$ 是鞅，并求 $M_n$ 当 $n \to \infty$ 时的极限分布。

#### 1. 属于什么类型的鞅？
本题属于 **有界比例鞅（Bounded Proportion Martingale）与 Doob 鞅收敛定理（Martingale Convergence Theorem）**。

#### 2. 严格鞅性验证
设第 $n$ 步红球数为 $R_n$，总球数为 $T_n = R + B + n c$。红球比例为 $M_n = \frac{R_n}{T_n}$。
在第 $n+1$ 步，抽中红球的条件概率为 $M_n$（此时红球变为 $R_n + c$），抽中黑球概率为 $1 - M_n$（红球数保持 $R_n$）：

$$
\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = M_n \cdot \frac{R_n + c}{T_n + c} + (1 - M_n) \cdot \frac{R_n}{T_n + c} = \frac{M_n(R_n + c) + R_n - M_n R_n}{T_n + c} = \frac{M_n c + R_n}{T_n + c}
$$

将 $R_n = M_n T_n$ 代入分子：

$$
= \frac{M_n c + M_n T_n}{T_n + c} = \frac{M_n (T_n + c)}{T_n + c} = M_n
$$

因此红球比例 $M_n$ 是一个取值在 $[0, 1]$ 之间的**严格有界鞅**！

#### 3. 极限分布
由 Doob 有界鞅收敛定理，$M_n \to M_\infty$ 几乎必然收敛。利用矩母匹配或经典 Beta-Binomial 极限可得：

$$
\boxed{M_\infty \sim \text{Beta}\left( \frac{R}{c}, \frac{B}{c} \right)}
$$

特别地，当初始 $R=1, B=1, c=1$ 时，极限比例服从 $(0, 1)$ 上的**标准连续均匀分布 $\text{Uniform}(0, 1)$**！

---

### 真题 9：分支过程与群体灭绝概率（Galton-Watson Branching Process & Extinction Probability）

> **通用题目建模（Citadel / Two Sigma 统计物理与随机过程真题）**：
> 
> 设单个粒子独立繁殖，后代数量 $K$ 的概率母函数为 $G(s) = \mathbb{E}[s^K]$，平均后代数 $m = \mathbb{E}[K] = G'(1)$。第 $n$ 代种群总数为 $Z_n$（$Z_0 = 1$）。
> 试用鞅论证明灭绝概率 $\pi = \mathbb{P}(\lim_{n \to \infty} Z_n = 0)$ 是方程 $G(s) = s$ 在 $[0, 1]$ 上的最小非负解。

#### 1. 属于什么类型的鞅？
本题属于 **种群规模归一化鞅（Normalized Branching Martingale）与概率母函数有界鞅（Probability Generating Function Martingale）**。

#### 2. 规模归一化鞅与母函数鞅
1. **规模归一化鞅**：$M_n = \frac{Z_n}{m^n}$ 是非负鞅（$\mathbb{E}[Z_{n+1} \mid Z_n] = m Z_n$），由鞅收敛定理知 $M_n \to M_\infty$ 几乎必然收敛。
2. **母函数鞅**：设 $s \in [0, 1]$ 是不动点方程 $G(s) = s$ 的任意解。构造过程 $Y_n = s^{Z_n}$：

$$
\mathbb{E}[Y_{n+1} \mid \mathcal{F}_n] = \mathbb{E}\left[ s^{\sum_{i=1}^{Z_n} K_i} \;\middle|\; Z_n \right] = (G(s))^{Z_n} = s^{Z_n} = Y_n
$$

因此 $Y_n = s^{Z_n}$ 是一个**严格有界鞅**（$0 \le Y_n \le 1$）！

#### 3. 应用有界鞅收敛定理
对有界鞅 $Y_n$ 应用支配收敛：

$$
\mathbb{E}[Y_n] = \mathbb{E}[Y_0] = s^1 = s
$$

当 $n \to \infty$ 时，$Z_n$ 要么趋于 0（灭绝，概率为 $\pi$），要么趋于 $\infty$（繁衍爆发，概率为 $1-\pi$）：

$$
\lim_{n \to \infty} s^{Z_n} = \mathbf{1}_{\{\text{灭绝}\}} \cdot s^0 + \mathbf{1}_{\{\text{爆发}\}} \cdot 0 = \mathbf{1}_{\{\text{灭绝}\}}
$$

取期望得 $\mathbb{E}[\mathbf{1}_{\{\text{灭绝}\}}] = \pi = s$。故灭绝概率 $\pi$ 严格满足不动点方程：

$$
\boxed{\pi = G(\pi)}
$$

