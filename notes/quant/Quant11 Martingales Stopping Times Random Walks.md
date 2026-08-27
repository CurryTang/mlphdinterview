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

$$\mathbb{E}[S_T] = \sum_{n=1}^\infty \mathbb{E}\left[X_n \mathbf{1}_{\{T \ge n\}}\right] = \sum_{n=1}^\infty \mathbb{E}[X_n] \cdot \mathbb{E}\left[\mathbf{1}_{\{T \ge n\}}\right] = \mathbb{E}[X_1] \sum_{n=1}^\infty \mathbb{P}(T \ge n)$$

根据离散非负随机变量尾概率积分公式 $\mathbb{E}[T] = \sum_{n=1}^\infty \mathbb{P}(T \ge n)$，立即得出：

$$\mathbb{E}[S_T] = \mathbb{E}[X_1] \cdot \mathbb{E}[T]$$

证毕。

---

### 2. 二阶 Wald 等式（Wald's Second Identity）

若二阶矩有限 $\mathbb{E}[X_1^2] < \infty$ 且 $\mathbb{E}[T] < \infty$，记方差 $\sigma^2 = \text{Var}(X_1) = \mathbb{E}[X_1^2] - \mu^2$，则：

$$\mathbb{E}\left[(S_T - T\mu)^2\right] = \sigma^2 \mathbb{E}[T]$$

特别地，当步长均值 $\mu = 0$ 的对称游走时：

$$\mathbb{E}[S_T^2] = \sigma^2 \mathbb{E}[T]$$

**证明**：构造二次方差鞅 $M_n = (S_n - n\mu)^2 - n\sigma^2$。应用 OST 令 $\mathbb{E}[M_T] = \mathbb{E}[M_0] = 0$，即刻得出结论。

---

### 3. Wald 指数等式（Wald's Exponential Identity）

设矩母函数（MGF）$M(\theta) = \mathbb{E}[e^{\theta X_1}]$ 存在。定义几何指数鞅：

$$M_n(\theta) = \frac{e^{\theta S_n}}{(M(\theta))^n}$$

在一致可积条件下应用 OST 即得：

$$\mathbb{E}\left[\frac{e^{\theta S_T}}{(M(\theta))^T}\right] = 1$$

- **对 $\theta$ 求一阶导令 $\theta \to 0$**：直接导出 Wald 一阶等式 $\mathbb{E}[S_T] = \mu \mathbb{E}[T]$；
- **对 $\theta$ 求二阶导令 $\theta \to 0$**：直接导出 Wald 二阶等式 $\mathbb{E}[(S_T - T\mu)^2] = \sigma^2 \mathbb{E}[T]$；
- **求解停时拉普拉斯变换**：令 $M(\theta) = e^{-s}$，可直接求解 $\mathbb{E}[e^{-s T}]$！

---

## 模块五：鞅的构造艺术与四大通用模板

| 鞅模板名称 | 数学形式 $M_n$ | 适用目标题型 | 输出的核心代数关系 |
| :--- | :--- | :--- | :--- |
| **1. 线性漂移消除鞅** | $S_n - n\mu$ | 带漂移随机游走的期望时间 $\mathbb{E}[T]$ | $\mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{\mu}$ |
| **2. 二次方差修正鞅** | $(S_n - n\mu)^2 - n\sigma^2$ | 无漂移对称游走的期望时间 $\mathbb{E}[T]$ | $\mathbb{E}[T] = \frac{\mathbb{E}[S_T^2]}{\sigma^2}$ |
| **3. 指数几何鞅** | $\left(\frac{q}{p}\right)^{S_n}$ 或 $\frac{e^{\theta S_n}}{(M(\theta))^n}$ | 非对称游走的双边吸收概率 $\mathbb{P}(\text{Hit } a)$ | $\mathbb{P}_a \left(\frac{q}{p}\right)^a + (1 - \mathbb{P}_a)\left(\frac{q}{p}\right)^{-b} = 1$ |
| **4. 调和特征函数鞅** | $f(X_n)$，满足 $(P - I)f = 0$ | 通用有限状态马尔可夫链吸收概率 | $\mathbb{E}[f(X_T)] = f(X_0)$ |

### 1. 模板 1：线性漂移消除鞅（Drift Correction）
当过程每步有恒定漂移 $\mathbb{E}[S_{n+1} - S_n \mid \mathcal{F}_n] = \mu \ne 0$ 时：
$$M_n = S_n - n\mu \implies \mathbb{E}[S_T] - \mu \mathbb{E}[T] = 0 \implies \mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{\mu}$$

### 2. 模板 2：二次方差修正鞅（Variance Correction）
当均值为零 $\mu = 0$ 时，减去累积方差项 $n\sigma^2$：
$$M_n = S_n^2 - n\sigma^2 \implies \mathbb{E}[S_T^2] - \sigma^2 \mathbb{E}[T] = 0 \implies \mathbb{E}[T] = \frac{\mathbb{E}[S_T^2]}{\sigma^2}$$

### 3. 模板 3：指数几何鞅（Exponential MGF Martingale）
针对非对称二项步长（$\mathbb{P}(+1)=p, \mathbb{P}(-1)=q \ne p$），解特征方程 $\mathbb{E}[\lambda^{X_1}] = p\lambda + q/\lambda = 1 \implies \lambda = q/p$。
故 $M_n = (q/p)^{S_n}$ 为严格鞅，直接用于求解非对称吸收概率。

### 4. 模板 4：调和函数鞅（Harmonic Function for Markov Chains）
若马氏链状态函数满足差分方程 $\sum_y P(x, y) f(y) = f(x)$，则 $M_n = f(X_n)$ 为鞅，联立边界条件即可解出吸收概率。

---

## 模块六：一维随机游走中的鞅论全解（Gambler's Ruin）

设粒子从原点 $S_0 = 0$ 出发，每步向上概率 $p$，向下概率 $q = 1 - p$。
双边吸收停时定义为：$T = \min\{n \ge 0 : S_n = a \text{ 或 } S_n = -b\}$（$a, b \in \mathbb{N}^+$）。

### 1. 简单对称随机游走（$p = 1/2$）

#### (1) 求解吸收概率 $\mathbb{P}(S_T = a)$
构造鞅 $M_n = S_n$。过程被严格约束在 $[-b, a]$，满足 OST 条件 (B)：
$$\mathbb{E}[S_T] = \mathbb{E}[S_0] = 0 \implies a \cdot \mathbb{P}_a + (-b) \cdot (1 - \mathbb{P}_a) = 0$$
$$\mathbb{P}_a = \frac{b}{a + b}, \quad \mathbb{P}_{-b} = \frac{a}{a + b}$$

#### (2) 求解期望吸收时间 $\mathbb{E}[T]$
构造二次方差鞅 $M_n = S_n^2 - n$。应用 OST：
$$\mathbb{E}[S_T^2 - T] = 0 \implies \mathbb{E}[T] = \mathbb{E}[S_T^2] = a^2 \left(\frac{b}{a+b}\right) + (-b)^2 \left(\frac{a}{a+b}\right) = \frac{ab(a+b)}{a+b} = a \cdot b$$

> [!NOTE]
> **对称游走速算公式**：
> - 胜率：$\mathbb{P}(\text{到达 } +a) = \frac{b}{a+b}$（反比于距离目标的远近）；
> - 期望步数：$\mathbb{E}[T] = a \cdot b$（直接等于上下边界距离的乘积）。

---

### 2. 非对称随机游走（$p \ne 1/2$）

#### (1) 求解吸收概率 $\mathbb{P}(S_T = a)$
构造指数几何鞅 $M_n = (q/p)^{S_n}$。应用 OST：
$$\mathbb{E}[(q/p)^{S_T}] = (q/p)^0 = 1 \implies \mathbb{P}_a (q/p)^a + (1 - \mathbb{P}_a)(q/p)^{-b} = 1$$
$$\mathbb{P}_a = \frac{1 - (q/p)^{-b}}{(q/p)^a - (q/p)^{-b}} = \frac{(q/p)^b - 1}{(q/p)^{a+b} - 1}$$

#### (2) 求解期望吸收时间 $\mathbb{E}[T]$
单步均值 $\mu = p - q \ne 0$。构造线性漂移鞅 $M_n = S_n - n(p - q)$。应用 OST：
$$\mathbb{E}[S_T - T(p - q)] = 0 \implies \mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{p - q} = \frac{a \mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$$

---

### 3. 常返性、瞬变性与 OST 条件的系统呼应

- **对称游走（$p=1/2$）**：是**常返**的（以概率 1 访问任意状态），但无界自由游走的期望回归时间是无穷大 $\mathbb{E}[T] = \infty$（这是自由单边停时 OST 失效的根源）；
- **不对称游走（$p \ne 1/2$）**：是**瞬变**的（以概率 1 趋向正负无穷之一）；
- **加了双边吸收边界后**：无论是对称还是非对称，停时 $T$ 几乎必然有限且 $\mathbb{E}[T] < \infty$，停止过程被严格约束在 $[-b, a]$ 之间，完全满足 OST 充分条件。

---

## 模块七：最优时停理论与 4 大高频面试真题

### 最优时停架构与斯奈尔包络（Snell Envelope）
- **目标**：寻求停时 $T^*$ 使得期望收益最大：$V_0 = \sup_{T \in \mathcal{T}} \mathbb{E}[Z_T]$。
- **倒推递推式**：
  $$U_N = Z_N$$
  $$U_n = \max\left(Z_n, \mathbb{E}[U_{n+1} \mid \mathcal{F}_n]\right), \quad n = N-1, \dots, 0$$
- **最优规则**：首次即时收益不低于继续持有价值时停止：$T^* = \min\{n \ge 0 : Z_n = U_n\}$。

---

### 真题 1：经典秘书问题（The Secretary Problem / $37\%$ 法则）

> **原题描述（Citadel / SIG / Jane Street 高频）**：
> 有 $n$ 位候选人按随机顺序面试，每次面试后必须立即决定录取或拒绝且不可反悔。只能观察到相对排名。求选到**全局第一名**的最大概率策略。

#### 精确离散求和推导
最优策略必为阈值策略：前 $k-1$ 人只观察不选，从第 $k$ 人开始选取第一个超越前期峰值的候选人。
第 $j$ 个人（$j \ge k$）被录取且为全局最优的充要条件是：
1. 全局最佳恰排在第 $j$ 位（概率 $1/n$）；
2. 前 $j-1$ 个人中的相对最佳者出现在前 $k-1$ 个人的观察区中（概率 $\frac{k-1}{j-1}$）。

成功概率为：
$$\mathbb{P}(\text{Success} \mid k) = \sum_{j=k}^n \frac{1}{n} \cdot \frac{k-1}{j-1} = \frac{k-1}{n} \sum_{j=k}^n \frac{1}{j-1}$$

#### 连续极限分析（$n \to \infty$）
令 $x = k/n$，利用黎曼和积分逼近：
$$f(x) = x \int_x^1 \frac{1}{t} dt = -x \ln x$$
求导极值点：$f'(x) = -\ln x - 1 = 0 \implies x^* = \frac{1}{e} \approx 36.8\%$，最大成功概率同样为 $1/e \approx 36.8\%$。

---

### 真题 2：有限次掷骰子最优时停（Sequential Die Rolling Game）

> **原题描述（Optiver / Jane Street 交易员笔试真题）**：
> 最多可掷均匀 6 面骰子 $N$ 次。每次可选择接受点数获得 $\$X$ 并结束，或放弃重掷。求公允价值及决策临界值。

#### 逆向归纳推导
记 $v_k$ 为剩余 $k$ 次机会时的最大期望收益：
1. **剩余 1 次（$k=1$）**：$v_1 = \mathbb{E}[X] = 3.5$。
2. **剩余 2 次（$k=2$）**：掷出 $X > 3.5$（4、5、6）接受，$\le 3$ 重掷：
   $$v_2 = \frac{1}{6}(3.5 \times 3 + 4 + 5 + 6) = \frac{25.5}{6} = 4.25$$
3. **剩余 3 次（$k=3$）**：掷出 $X > 4.25$（5、6）接受，$\le 4$ 重掷：
   $$v_3 = \frac{1}{6}(4.25 \times 4 + 5 + 6) = \frac{28}{6} \approx 4.667$$
4. **剩余 4 次（$k=4$）**：$v_4 = \frac{1}{6}(4.667 \times 4 + 5 + 6) \approx 4.944$。

---

### 真题 3：美式期权提前行权与最优时停边界（American Options）

> **面试核心考点（Two Sigma / Morgan Stanley Strats 面试）**：
> 1. 为什么无红利美式 Call **永远不应提前行权**？
> 2. 为什么美式 Put 存在最优提前行权边界 $S^*(t)$？

#### (1) 美式 Call：无套利与鞅性证明
由贴现标的资产的鞅性质与 Jensen 不等式：
$$C(S_t, t) \ge \mathbb{E}^\mathbb{Q}[e^{-r(\tau - t)}(S_\tau - K) \mid \mathcal{F}_t] \ge S_t - K e^{-r(\tau - t)} > S_t - K \quad (\forall r > 0)$$
期权市价 $C(S_t, t)$ 严格大于立即行权价值 $S_t - K$，提前行权会白白浪费时间价值与下行保护，直接卖出期权永远更优。

#### (2) 美式 Put：提前行权边界 $S^*(t)$
当标的资产暴跌至极低价（$S_t \to 0$）时，立即行权可提前拿到现金 $\$K$ 并在剩余时间内赚取无风险利息 $rK > 0$；而继续等待最多也只能拿到 $\$K$。当利息收益覆盖期权时间价值时，立即行权是最优的。

---

### 真题 4：硬币序列等待时间与李氏赌场鞅（Li's Martingale & Penney's Game）

> **原题描述（Jane Street / Optiver 压轴面试题）**：
> 抛掷均匀硬币，求首次出现 `HTTH` 与 `HTHT` 的期望投掷次数及差异原因。

#### 赌场赌资累积净利润鞅模型
在每轮投掷前，有一位新赌徒进场投入 \$1。每猜中一位序列字符以 $1{:}2$ 公正赔率翻倍，直至完整匹配目标序列拿走 $\$2^m$ 离场；猜错输光出局。
赌场的累积净利润 $M_n = (\text{总投入本金}) - (\text{总支付奖金})$ 是期望为 0 的鞅。

当模式 $A$ 在时刻 $T$ 首次出现时，共 $T$ 位赌徒进场。此时持有正奖金的赌徒，其进场时刻刚好对应模式 $A$ 的**前缀与后缀完全重合的长度 $k$**（奖金为 $\$2^k$）。
根据 OST $\mathbb{E}[M_T] = 0$：

$$\mathbb{E}[T_A] = (A * A)_2 = \sum_{k=1}^m 2^k \cdot \mathbf{1}_{\{\text{Prefix}(A, k) = \text{Suffix}(A, k)\}}$$

#### 心算对比与直觉
1. **模式 $A = \text{HTTH}$**：
   - 重合判定：$k=1$（`H`=`H`，$\to 2^1=2$）；$k=2, 3$（不重合）；$k=4$（`HTTH`=`HTTH`，$\to 2^4=16$）。
   - $\mathbb{E}[T_{\text{HTTH}}] = 16 + 2 = 18$。
2. **模式 $B = \text{HTHT}$**：
   - 重合判定：$k=1$（不重合）；$k=2$（`HT`=`HT`，$\to 2^2=4$）；$k=3$（不重合）；$k=4$（`HTHT`=`HTHT`，$\to 2^4=16$）。
   - $\mathbb{E}[T_{\text{HTHT}}] = 16 + 4 = 20$。

> **自重叠与成簇直觉**：`HTHT` 具有周期为 2 的自重叠性，容易在短时间内连续成簇（Cluster）出现。在相同长期频率下，**成簇出现会导致没有模式出现的“空白等待间隔”被显著拉长**，因此首次出现的期望等待时间更大！

---

### 真题 5：无放回抽卡的最优停止与 Doob 分解（Sampling Without Replacement & 1-SLA）

> **原题描述（Akuna / SIG / Citadel 量化面试真题）**：
> 
> 桌面上有一副洗匀的牌，共 **10 张牌**（包含 **9 张蓝牌** 和 **1 张红牌**）。
> 规则如下：
> - 你每次从牌堆中**无放回**地抽取一张牌；
> - 如果抽到蓝牌，你的当前得分累加 $1$ 分，你可以选择**立即停止游戏并带走当前所有得分**，或者选择**继续抽下一张**；
> - 如果抽到红牌，游戏**强制立即结束**，并且你的最终收益判定为 **$-(n-1)$**（即倒扣前面已抽出的全部 $n-1$ 分）；
> - 问：**最优停止策略是什么？在最优策略下的期望总收益是多少？**

#### 1. 记号与过程定义
- 设红牌被抽出的轮数记为随机变量（停时）$T \in \{1, 2, \dots, 10\}$。由于洗牌均匀，红牌等可能出现在第 $1$ 到第 $10$ 个位置，$P(T = k) = \frac{1}{10}$（$k=1,\dots,10$）；
- 引入指示变量 $I_n = \mathbf{1}_{\{T > n\}}$，表示前 $n$ 步全部抽中蓝牌（未出红牌）；
- 若在第 $n$ 步未出红牌且选择主动停下（$I_n=1$），累计收益为 $n$；
- 若在第 $t \le n$ 步抽出了红牌，游戏强制终止，收益为 $-(t-1)$。

#### 2. 基础比例鞅的构造（Base Martingale）
在无放回抽取单张目标牌的场景中，构造关键比例过程：

$$
M_n = \frac{I_n}{10 - n} \quad (0 \le n \le 9)
$$

**验证鞅性质**：
在已知 $\mathcal F_n$ 且 $I_n = 1$（前 $n$ 步未出红牌）的条件下，剩余 $10-n$ 张牌中包含 $1$ 张红牌、$9-n$ 张蓝牌。下一步不出红牌的条件概率为：

$$
\mathbb E[I_{n+1} \mid \mathcal F_n] = I_n \cdot P(T > n+1 \mid T > n) = I_n \cdot \frac{9-n}{10-n}
$$

两边除以剩余牌数 $10-(n+1) = 9-n$：

$$
\mathbb E\left[ \frac{I_{n+1}}{10-(n+1)} \;\middle|\; \mathcal F_n \right] = \frac{I_n \cdot \frac{9-n}{10-n}}{9-n} = \frac{I_n}{10-n} = M_n
$$

而在 $I_n = 0$（已经出了红牌）时，$I_{n+1} \equiv 0$，等式 $0 = 0$ 显然成立。
因此 $M_n$ 是关于信息流 $\{\mathcal F_n\}$ 的**标准严格离散鞅**。

#### 3. 收益过程与 Doob 分解
定义在第 $n$ 步主动停下的累计收益过程为 $Y_n$。
在 $I_n = 1$（当前尚未抽到红牌，手握 $n$ 分）时，计算**再抽一张牌**的单步条件期望增量 $\Delta_n$：

$$
\Delta_n = \mathbb E[Y_{n+1} - Y_n \mid \mathcal F_n] = \underbrace{\frac{9-n}{10-n} \cdot (n+1)}_{\text{抽中蓝牌，得分变为 } n+1} + \underbrace{\frac{1}{10-n} \cdot (-n)}_{\text{抽中红牌，罚没变为 } -n} - \underbrace{n}_{\text{当前收益}}
$$

将通分后的分子展开并化简：

$$
(9-n)(n+1) - n - n(10-n) = (9 + 8n - n^2) - n - (10n - n^2) = 9 - 3n
$$

因此，得到极其优雅的单步增量公式：

$$
\Delta_n = \mathbb E[Y_{n+1} - Y_n \mid \mathcal F_n] = \frac{9 - 3n}{10 - n} I_n
$$

由此给出收益过程 $Y_n$ 的 **Doob 显式分解**：

$$
Y_n = N_n + A_n, \quad \text{其中 } A_n = \sum_{k=0}^{n-1} \frac{9-3k}{10-k} I_k
$$

其中 $N_n$ 为纯鞅，$A_n$ 为可预测累积漂移过程。

#### 4. 单步前瞻法（1-SLA）与最优停止策略
分析单步漂移项 $\Delta_n = \frac{9-3n}{10-n}$ 的正负号变化：
- **$n = 0$**（手握 0 分）：$\Delta_0 = \frac{9}{10} > 0$（下鞅区间，继续抽）；
- **$n = 1$**（手握 1 分）：$\Delta_1 = \frac{6}{9} = \frac{2}{3} > 0$（下鞅区间，继续抽）；
- **$n = 2$**（手握 2 分）：$\Delta_2 = \frac{3}{8} > 0$（下鞅区间，继续抽）；
- **$n = 3$**（手握 3 分）：$\Delta_3 = \frac{9-9}{7} = 0$（临界点，继续抽与立即停期望严格相等）；
- **$n \ge 4$**（手握 $\ge 4$ 分）：$9 - 3n < 0 \implies \Delta_n < 0$（期望净亏损，进入严格上鞅区间，绝不继续！）。

由于单步增量 $\Delta_n$ 的分子 $9-3n$ 随 $n$ 严格单调递减，一旦 $n \ge 3$ 后未来所有步的 $\Delta_k \le 0$ 恒成立，单调停止条件（Chow-Robbins Theorem）严格满足！

> **最优策略（Optimal Stopping Policy）**：
> **当手头累积达到 3 分（抽中 3 张蓝牌）时，立即选择停止游戏**（在 $n=3$ 与 $n=4$ 停下期望相同，推荐 $n^*=3$ 锁定利润，降低收益方差）。

#### 5. 期望收益计算
采用在 $n=3$ 步主动停止的策略：
- **情况 A（前 3 步全为蓝牌）**：概率为 $\frac{9}{10} \times \frac{8}{9} \times \frac{7}{8} = \frac{7}{10}$，主动退出，获得收益 $+3$ 分；
- **情况 B（红牌出现在第 1 步）**：概率为 $\frac{1}{10}$，收益为 $-(1-1) = 0$ 分；
- **情况 C（红牌出现在第 2 步）**：概率为 $\frac{1}{10}$，收益为 $-(2-1) = -1$ 分；
- **情况 D（红牌出现在第 3 步）**：概率为 $\frac{1}{10}$，收益为 $-(3-1) = -2$ 分。

计算全期望：

$$
\mathbb E[\text{Payoff}] = \frac{7}{10} \times 3 + \frac{1}{10} \times 0 + \frac{1}{10} \times (-1) + \frac{1}{10} \times (-2) = \frac{21 - 0 - 1 - 2}{10} = \frac{18}{10} = 1.8 \text{ 分}
$$

#### 6. 通用化推广：$B$ 蓝 $1$ 红卡池的最优停止公式
若卡池推广到一般的 $B$ 张蓝牌与 $1$ 张红牌（总牌数 $B+1$ 张）：
- 单步条件增量为 $\Delta_n = \frac{B - 3n}{B+1-n} I_n$
- 令分子 $B - 3n \le 0$，立即得到通用最优停止阈值：

$$
\boxed{n^* = \left\lfloor \frac{B}{3} \right\rfloor}
$$

- 规则记忆：**在单红牌罚没规则下，最优停止点永远落在总蓝牌数的 $1/3$ 处！**

---

## 模块八：量化面试极速自查矩阵

| 问题场景 | 推荐鞅构造 | 解析公式 / 求解定理 | 验证要点与陷阱 |
| :--- | :--- | :--- | :--- |
| **一维对称游走破产概率** | $M_n = S_n$ | $\mathbb{P}_a = \frac{b}{a + b}$ | 边界有界，$\mathbb{E}[T] < \infty$ 自动满足 |
| **一维对称游走期望步数** | $M_n = S_n^2 - n$ | $\mathbb{E}[T] = a \cdot b$ | 二阶 Wald 等式，$\mu = 0$ 且 $\sigma^2 = 1$ |
| **一维非对称游走破产概率** | $M_n = (q/p)^{S_n}$ | $\mathbb{P}_a = \frac{(q/p)^b - 1}{(q/p)^{a+b} - 1}$ | 利用 $p(q/p) + q(p/q) = 1$ 验证鞅性质 |
| **一维非对称游走期望步数** | $M_n = S_n - n(p - q)$ | $\mathbb{E}[T] = \frac{a\mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$ | 分母 $p - q \ne 0$，一阶 Wald 等式 |
| **单边吸收等待时间** | Wald 指数鞅 $e^{\theta S_n} / M(\theta)^n$ | $\mathbb{E}[s^{T_a}] = \left(\frac{1 - \sqrt{1 - 4pqs^2}}{2ps}\right)^a$ | $\mathbb{P}(T < \infty) = 1$ 但 $\mathbb{E}[T] = \infty$（零漂移陷阱） |
| **序列模式等待时间** | 赌场赌资净利润鞅 | $\mathbb{E}[T_A] = (A * A)_2 = \sum 2^k \mathbf{1}_{\{\text{前缀=后缀}\}}$ | Penney's Game 非传递性：后手总可构造优势前缀 |
| **连续/离散最优时停** | 斯奈尔包络（Snell Envelope） | $U_n = \max(Z_n, \mathbb{E}[U_{n+1} \mid \mathcal{F}_n])$ | 倒推逆向归纳法寻找最佳临界阈值 |

---

## 模块九：快速选择题巩固

```quiz
title: 快速选择题 1
question: 随机过程 X_n 是鞅，其定义中最核心的一条是：
answer: C
A. X_n 的方差不随时间变化
B. X_n 严格单调
C. E[X_{n+1} | F_n] = X_n，即给定当前信息，下一步的最优预测就是当前值
D. X_n 的分布不随时间变化
explanation: 鞅的核心条件是条件期望的"公平游戏"性质：站在时刻 n 往前看一步，对 X_{n+1} 的最优预测就是 X_n 本身，不涉及方差不变或分布不变这些更强的要求。
```

```quiz
title: 快速选择题 2
question: 关于停时的定义，下列说法正确的是：
answer: B
A. 停时可以依赖任意未来信息，只要它是一个有限的随机变量
B. T 是停时要求"是否 T <= n"这件事只能由截止到时刻 n 的信息决定
C. 停时必须是一个确定性的常数
D. 随机游走的最后一次归零时刻是一个合法的停时
explanation: 停时的定义就是"决定是否在时刻n停下"只能用当下及之前的信息判断，不能偷看未来；D 是经典的非停时例子，因为判断"是不是最后一次归零"需要知道未来是否还会归零。
```

```quiz
title: 快速选择题 3
question: 简单对称随机游走 S_n 首次到达水平 1 的时刻 T 满足下列哪个性质，导致直接套用 E[S_T]=E[S_0] 会出错？
answer: D
A. T 不是一个合法的停时
B. S_n 根本不是鞅
C. T 是有界的
D. T 几乎必然有限，但期望 E[T] 是无穷大，且停止过程无界
explanation: T 确实是合法停时，S_n 确实是鞅；问题在于 T 虽然几乎必然有限（对称随机游走常返），但不满足 OST 的任何一个充分条件（不有界、停止过程无界、E[T]=无穷），所以 E[S_T]=E[S_0] 没有理论保证，实际算出来也确实不成立（1 != 0）。
```

```quiz
title: 快速选择题 4
question: 最优停时定理的三个常见充分条件中，"条件 C"具体要求的是：
answer: A
A. E[T] < 无穷，且鞅的每一步增量有界
B. T 必须是一个常数
C. 鞅本身必须是有界的
D. 停时必须几乎必然等于 0
explanation: 条件 C 允许过程本身无界、时间线无限长，但要求平均等待时间有限，并且每一步变化幅度有一个统一的上界，这样才能保证期望不被"极端但小概率"的长时间路径主导。
```

```quiz
title: 快速选择题 5
question: 对称简单随机游走从 0 出发，吸收边界为 -a 和 b。先到达 b 的概率是：
answer: B
A. 1/2，与 a、b 无关
B. a / (a+b)
C. b / (a+b)
D. a·b / (a+b)
explanation: 对 S_n 这个鞅用 OST 得 E[S_T]=0，S_T 只能取 -a 或 b，解方程 -a(1-p_b)+b·p_b=0 得 p_b=a/(a+b)（先到 b 的概率），边界离起点越远，先到达它的概率越小。
```

```quiz
title: 快速选择题 6
question: 对称简单随机游走从 0 出发，吸收边界为 -a 和 b，期望吸收时间 E[T] 是：
answer: C
A. a + b
B. (a+b)^2
C. a·b
D. 需要额外条件才能确定
explanation: 对鞅 M_n=S_n^2-n 用 OST 得 E[T]=E[S_T^2]，代入 S_T^2 的两种取值和对应概率算出 E[T]=a·b，这是一个只依赖 a、b 乘积的简洁结果。
```

```quiz
title: 快速选择题 7
question: 不对称随机游走（每步向上概率 p、向下概率 q=1-p，p≠q）中，为什么不能直接对 S_n 用 OST 求吸收概率？
answer: B
A. 因为不对称随机游走没有停时的概念
B. 因为 p≠q 时 S_n 存在系统性漂移，不再是鞅，E[X_{n+1}]=p-q≠0
C. 因为不对称随机游走一定不满足 OST 的有界性条件
D. 因为不对称情形下期望吸收时间总是无穷大
explanation: S_n 是不是鞅只取决于每一步的条件期望增量是否为 0；不对称情形下增量期望是 p-q≠0，S_n 本身有系统性漂移不是鞅，需要换成指数鞅 r^{S_n}（r=q/p）才能重新应用 OST。
```

```quiz
title: 快速选择题 8
question: 指数鞅 M_n = r^{S_n} 中，r 的取值 r=q/p 是如何确定的？
answer: A
A. 唯一满足 E[r^{X_{n+1}}] = p·r + q/r = 1 的非平凡取值
B. 任意大于 1 的常数都可以
C. r 必须等于 p/q，不能是 q/p
D. r 由吸收边界 a、b 的具体数值决定
explanation: 要让 r^{S_n} 是鞅，唯一需要满足的条件是 E[r^{X_{n+1}}]=1，代入 X_{n+1}=+1（概率p）或-1（概率q）得方程 pr+q/r=1，解出 r=q/p（r=1 是平凡解，对应对称情形退化回 S_n 本身）。
```

```quiz
title: 快速选择题 9
question: 关于一维随机游走的常返性与瞬变性，下列说法正确的是：
answer: C
A. 对称和不对称随机游走都是常返的
B. 对称和不对称随机游走都是瞬变的
C. 对称随机游走是常返的（但期望回归时间无穷大），不对称随机游走是瞬变的
D. 常返性只取决于吸收边界的设置，与 p 是否等于 1/2 无关
explanation: 对称简单随机游走以概率 1 回到任意有限水平（常返），但期望回归时间是无穷大；不对称随机游走（p≠q）几乎必然趋向正负无穷之一，不会以概率 1 回到起点（瞬变）。这正是模块三 OST 反例出现的根源。
```

```quiz
title: 快速选择题 10
question: 关于"OST 反例"和"加了吸收边界的 Gambler's Ruin"这两个设定，下列说法最准确的是：
answer: D
A. 两者本质相同，OST 在两种设定下都不成立
B. 两者本质相同，OST 在两种设定下都成立
C. Gambler's Ruin 设定下 OST 不成立，自由随机游走下 OST 成立
D. 自由随机游走单边停时（如首中 1）不满足 OST 条件，而加了双边吸收边界后停时的期望有限、增量有界，重新满足 OST 条件
explanation: 关键区别在于停时本身的性质：自由游走下"首次碰到某个单侧水平"的停时期望无穷大、停止过程无界；一旦加上双边吸收边界，停时几乎必然在有限期望步数内发生，且过程本身被边界限制在有界范围内，满足 OST 的充分条件。同一个随机游走在不同问题设定下，停时是否满足 OST 条件可以完全不同。
```
