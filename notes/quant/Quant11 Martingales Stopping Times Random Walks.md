# Quant 11 · 鞅、停时与随机游走：Wald 等式、鞅构造与最优时停

在顶尖量化对冲基金（Jane Street、Optiver、Citadel、SIG、Jump Trading、Two Sigma）的核心数学与概率面试中，**鞅（Martingale）与最优停时定理（Optional Stopping Theorem, OST）**是最具降维打击能力的王牌工具。

面对多阶段随机游走、吸收边界、等待时间、序列出现模式（Pattern Matching）或多轮最优决策问题，传统的状态转移矩阵与递归方程往往伴随极其繁琐的代数运算，而在高维或连续场景下极易失效。**鞅论的核心威力在于：通过构造满足无漂移特性的随机过程，将整个动态演化过程的终局期望直接“一键定格”在初始状态上（$\mathbb{E}[M_T] = \mathbb{E}[M_0]$），实现跨维度的降维求解。**

本章系统解构 Wald 等式族、鞅的四大构造模板、一维随机游走的全套鞅论解法，以及最优时停理论在经典面试题（秘书问题、掷骰子博弈、美式期权、赌场赌资鞅）中的应用。

```text
遇到随机游走/等待时间/最优时停问题的 4 步心智模型：
1. 识别停时与信息流：明确停止时间 T 的定义（如首次触碰双边边界、首次匹配目标字符序列），验证 T 是否几乎必然有限（P(T < ∞) = 1）以及是否满足 OST 适用条件。
2. 匹配并构造目标鞅：
   - 求吸收概率 P(Hit A) -> 构造指数鞅 M_n = (q/p)^{S_n} 或调和函数鞅 f(X_n)；
   - 求零漂移对称游走的期望时间 E[T] -> 构造二次方差鞅 M_n = S_n^2 - n 或应用二阶 Wald 等式；
   - 求带漂移游走的期望时间 E[T] -> 构造线性漂移鞅 M_n = S_n - nμ 或应用一阶 Wald 等式；
   - 求字符串模式等待时间 E[T_Pattern] -> 构造赌场赌资累计净利润鞅（Li's Martingale）。
3. 应用最优停时定理（OST）：令 E[M_T] = E[M_0]，将终局状态的代数加权和直接联立求解。
4. 最优时停逆向归纳：若是多阶段择时决策（如买卖、雇佣、博弈），构建斯奈尔包络（Snell Envelope），通过倒推逆向归纳法锁定最优临界阈值。
```

---

## 模块一：Wald 等式族全景深度解析

Wald 等式（Wald's Identities）是鞅论在独立同分布（i.i.d.）随机变量求和与随机停止时间下的直接推论，也是量化面试中推导随机游走、分支过程与复合泊松过程期望的核心利器。

```martingale-rw-demo
```

### 1. 一阶 Wald 等式（Wald's First Identity）

#### 定理陈述
设 $X_1, X_2, \dots$ 为独立同分布（i.i.d.）的随机变量，其一阶绝对矩有限 $\mathbb{E}[|X_1|] < \infty$，记均值 $\mu = \mathbb{E}[X_1]$。设 $T$ 是关于自然信息流 $\mathcal{F}_n = \sigma(X_1, \dots, X_n)$ 的停时（Stopping Time），且期望停止时间有限 $\mathbb{E}[T] < \infty$。令 $S_n = \sum_{i=1}^n X_i$，则随机求和 $S_T = \sum_{i=1}^T X_i$ 的数学期望满足：

$$\mathbb{E}[S_T] = \mathbb{E}[T] \cdot \mathbb{E}[X_1]$$

#### 严格数学证明（基于示性函数与独立性展开）
将随机停止求和 $S_T$ 展开为示性函数的形式：

$$S_T = \sum_{n=1}^\infty X_n \mathbf{1}_{\{T \ge n\}}$$

注意到事件 $\{T \ge n\} = \{T \le n - 1\}^c$。根据停时的定义，事件 $\{T \le n - 1\}$ 完全由前 $n-1$ 步的历史信息 $\mathcal{F}_{n-1} = \sigma(X_1, \dots, X_{n-1})$ 决定。因此，**随机变量 $X_n$ 与示性事件 $\mathbf{1}_{\{T \ge n\}}$ 相互独立**！

利用 Fubini-Tonelli 定理交换求和与数学期望积分（由 $\mathbb{E}[T] < \infty$ 和 $\mathbb{E}[|X_1|] < \infty$ 保证绝对可积性）：

$$\mathbb{E}[S_T] = \sum_{n=1}^\infty \mathbb{E}\left[X_n \mathbf{1}_{\{T \ge n\}}\right] = \sum_{n=1}^\infty \mathbb{E}[X_n] \cdot \mathbb{E}\left[\mathbf{1}_{\{T \ge n\}}\right] = \mathbb{E}[X_1] \sum_{n=1}^\infty \mathbb{P}(T \ge n)$$

根据离散非负整值随机变量期望的尾概率求和公式 $\mathbb{E}[T] = \sum_{n=1}^\infty \mathbb{P}(T \ge n)$，立即得到：

$$\mathbb{E}[S_T] = \mathbb{E}[X_1] \cdot \mathbb{E}[T]$$

证毕。

> [!WARNING]
> **致命陷阱：为什么 $\mathbb{E}[T] < \infty$ 是绝对不可或缺的前提？**
> 
> 考虑标准一维对称简单随机游走（$X_i = \pm 1$ 各占 $1/2$ 概率），从 $0$ 出发，定义停时 $T = \min\{n : S_n = 1\}$ 为首次到达 $+1$ 的时刻。
> 显然 $S_T \equiv 1$，故 $\mathbb{E}[S_T] = 1$。
> 但每一步增量的均值 $\mathbb{E}[X_1] = 0$。如果盲目套用 Wald 等式，会得出荒谬结论：
> $$1 = \mathbb{E}[S_T] = \mathbb{E}[T] \cdot \mathbb{E}[X_1] = \mathbb{E}[T] \cdot 0 = 0 \quad (\text{矛盾！})$$
> **错误根源**：虽然由常返性可知 $\mathbb{P}(T < \infty) = 1$（游走必定最终到达 1），但**期望停止时间是无穷大** $\mathbb{E}[T] = \infty$。此时 $\infty \cdot 0$ 为未定式，Wald 一阶等式的前提条件破裂。

---

### 2. 二阶 Wald 等式（Wald's Second Identity）

#### 定理陈述
在上述 i.i.d. 条件下，若二阶矩有限 $\mathbb{E}[X_1^2] < \infty$ 且 $\mathbb{E}[T] < \infty$，记方差 $\sigma^2 = \text{Var}(X_1) = \mathbb{E}[X_1^2] - \mu^2$，则随机求和的二阶矩满足：

$$\mathbb{E}\left[(S_T - T\mu)^2\right] = \sigma^2 \mathbb{E}[T]$$

特别地，当步长均值为零（$\mu = 0$）的对称游走时：

$$\mathbb{E}[S_T^2] = \sigma^2 \mathbb{E}[T]$$

#### 鞅视角下的极速证明
定义离散时间过程 $M_n = (S_n - n\mu)^2 - n\sigma^2$。我们验证 $M_n$ 是一个鞅：

$$\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = \mathbb{E}[(S_n + X_{n+1} - (n+1)\mu)^2 \mid \mathcal{F}_n] - (n+1)\sigma^2$$
$$= \mathbb{E}[((S_n - n\mu) + (X_{n+1} - \mu))^2 \mid \mathcal{F}_n] - (n+1)\sigma^2$$
$$= (S_n - n\mu)^2 + 2(S_n - n\mu)\underbrace{\mathbb{E}[X_{n+1} - \mu]}_{0} + \underbrace{\mathbb{E}[(X_{n+1} - \mu)^2]}_{\sigma^2} - (n+1)\sigma^2$$
$$= (S_n - n\mu)^2 + \sigma^2 - n\sigma^2 - \sigma^2 = (S_n - n\mu)^2 - n\sigma^2 = M_n$$

根据 Doob 最优停时定理（OST），在 $\mathbb{E}[T] < \infty$ 且增量有界下，$\mathbb{E}[M_T] = \mathbb{E}[M_0] = 0$，即：

$$\mathbb{E}\left[(S_T - T\mu)^2 - T\sigma^2\right] = 0 \implies \mathbb{E}\left[(S_T - T\mu)^2\right] = \sigma^2 \mathbb{E}[T]$$

---

### 3. Wald 指数等式（Wald's Exponential Identity）

设矩母函数（MGF）$M(\theta) = \mathbb{E}[e^{\theta X_1}]$ 在某个开区间内收敛存在。定义几何指数过程：

$$M_n(\theta) = \frac{e^{\theta S_n}}{(M(\theta))^n}$$

由于增量独立同分布，易知 $M_n(\theta)$ 是一个均值为 1 的非负鞅。对满足一致可积条件的停时 $T$，应用 OST 即得 **Wald 指数等式**：

$$\mathbb{E}\left[\frac{e^{\theta S_T}}{(M(\theta))^T}\right] = 1$$

> [!TIP]
> **Wald 指数等式的两大实战妙用**：
> 1. **对 $\theta$ 求一阶导并令 $\theta \to 0$**：直接导出 Wald 一阶等式 $\mathbb{E}[S_T] = \mu \mathbb{E}[T]$；
> 2. **对 $\theta$ 求二阶导并令 $\theta \to 0$**：直接导出 Wald 二阶等式 $\mathbb{E}[(S_T - T\mu)^2] = \sigma^2 \mathbb{E}[T]$；
> 3. **求解连续越界（Overshoot）与拉普拉斯联合分布**：令 $M(\theta) = e^{-s}$，即可直接解出停时 $T$ 的母函数 $\mathbb{E}[e^{-s T}]$！

---

## 模块二：鞅的构造艺术与四大通用模板

在量化面试中，最考察功底的一步是：**如何针对给定的问题背景，无中生有地构造出恰到好处的鞅？**

```html
<table>
  <thead>
    <tr>
      <th style="width: 22%;">鞅模板名称</th>
      <th style="width: 28%;">数学形式 $M_n$</th>
      <th style="width: 25%;">适用目标题型</th>
      <th style="width: 25%;">输出的核心代数关系</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. 线性漂移消除鞅</strong></td>
      <td>$S_n - n\mu$</td>
      <td>带漂移随机游走的期望时间 $\mathbb{E}[T]$</td>
      <td>$\mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{\mu}$</td>
    </tr>
    <tr>
      <td><strong>2. 二次方差修正鞅</strong></td>
      <td>$(S_n - n\mu)^2 - n\sigma^2$</td>
      <td>无漂移对称游走的期望时间 $\mathbb{E}[T]$</td>
      <td>$\mathbb{E}[T] = \frac{\mathbb{E}[S_T^2]}{\sigma^2}$</td>
    </tr>
    <tr>
      <td><strong>3. 指数几何鞅</strong></td>
      <td>$\left(\frac{q}{p}\right)^{S_n}$ 或 $\frac{e^{\theta S_n}}{(M(\theta))^n}$</td>
      <td>非对称游走的双边吸收概率 $\mathbb{P}(\text{Hit } a)$</td>
      <td>$\mathbb{P}_a \left(\frac{q}{p}\right)^a + (1 - \mathbb{P}_a)\left(\frac{q}{p}\right)^{-b} = 1$</td>
    </tr>
    <tr>
      <td><strong>4. 调和特征函数鞅</strong></td>
      <td>$f(X_n)$，满足 $(P - I)f = 0$</td>
      <td>通用有限状态马尔可夫链吸收概率</td>
      <td>$\mathbb{E}[f(X_T)] = f(X_0)$</td>
    </tr>
  </tbody>
</table>
```

### 1. 模板 1：线性漂移修正鞅（Drift Correction）
当过程 $S_n$ 每一步有非零恒定漂移 $\mathbb{E}[S_{n+1} - S_n \mid \mathcal{F}_n] = \mu \ne 0$ 时，直接减去累积漂移项 $n\mu$：
$$M_n = S_n - n\mu$$
- **验证**：$\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = S_n + \mu - (n+1)\mu = S_n - n\mu = M_n$。
- **解法**：$\mathbb{E}[S_T] - \mu \mathbb{E}[T] = 0 \implies \mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{\mu}$。

### 2. 模板 2：二次方差修正鞅（Variance Correction）
当过程均值为零（$\mu = 0$），线性鞅只能给出 $\mathbb{E}[S_T] = 0$（无法提取出 $T$）。此时将状态平方，减去累积方差项 $n\sigma^2$：
$$M_n = S_n^2 - n\sigma^2$$
- **验证**：$\mathbb{E}[S_{n+1}^2 - (n+1)\sigma^2 \mid \mathcal{F}_n] = S_n^2 + 2 S_n \cdot 0 + \sigma^2 - (n+1)\sigma^2 = S_n^2 - n\sigma^2 = M_n$。
- **解法**：$\mathbb{E}[S_T^2] - \sigma^2 \mathbb{E}[T] = 0 \implies \mathbb{E}[T] = \frac{\mathbb{E}[S_T^2]}{\sigma^2}$。

### 3. 模板 3：指数几何鞅（Exponential MGF Martingale）
针对非对称二项步长（$\mathbb{P}(X_i = +1) = p, \mathbb{P}(X_i = -1) = q = 1 - p \ne p$），寻找常数 $\lambda \ne 1$ 使得 $M_n = \lambda^{S_n}$ 成为鞅：
$$\mathbb{E}[\lambda^{X_1}] = p \lambda^1 + q \lambda^{-1} = 1 \implies p\lambda^2 - \lambda + q = 0 \implies (\lambda - 1)(p\lambda - q) = 0$$
非平凡解为 $\lambda = \frac{q}{p}$。
故 $M_n = \left(\frac{q}{p}\right)^{S_n}$ 是一个严格鞅。
- **解法**：利用 $\mathbb{E}\left[\left(\frac{q}{p}\right)^{S_T}\right] = \left(\frac{q}{p}\right)^{S_0} = 1$，瞬间解出非对称破产概率 $\mathbb{P}_a$。

### 4. 模板 4：调和函数鞅（Harmonic Function for Markov Chains）
设离散马尔可夫链转移矩阵为 $P$。若函数 $f: \mathcal{S} \to \mathbb{R}$ 满足调和条件 $(Pf)(x) = f(x)$，即：
$$\sum_{y \in \mathcal{S}} P(x, y) f(y) = f(x)$$
则随机序列 $M_n = f(X_n)$ 是一个鞅！
- **解法**：边界条件由吸收态 $f(\text{Goal}) = 1, f(\text{Fail}) = 0$ 确定，求解差分方程 $(P - I)f = 0$ 即可获得精确的吸收概率公式。

---

## 模块三：一维随机游走中的鞅论全解

设粒子从原点 $S_0 = 0$ 出发，每一步以概率 $p$ 移动 $+1$，以概率 $q = 1 - p$ 移动 $-1$。
定义双边吸收停时：
$$T = \min\{n \ge 0 : S_n = a \text{ 或 } S_n = -b\} \quad (a, b \in \mathbb{N}^+)$$

### 1. 简单对称随机游走（$p = 1/2$）

#### (1) 求解吸收概率 $\mathbb{P}(S_T = a)$
构造鞅 $M_n = S_n$。
由于停时截断值被严格限制在 $[-b, a]$ 之间，满足有界性条件，应用 OST：
$$\mathbb{E}[S_T] = \mathbb{E}[S_0] = 0$$
令 $\mathbb{P}_a = \mathbb{P}(S_T = a)$，则 $\mathbb{P}(S_T = -b) = 1 - \mathbb{P}_a$。
$$a \cdot \mathbb{P}_a + (-b) \cdot (1 - \mathbb{P}_a) = 0 \implies (a + b)\mathbb{P}_a = b \implies \mathbb{P}_a = \frac{b}{a + b}, \quad \mathbb{P}_{-b} = \frac{a}{a + b}$$

#### (2) 求解期望吸收时间 $\mathbb{E}[T]$
构造二次方差鞅 $M_n = S_n^2 - n$（此时 $\sigma^2 = 1, \mu = 0$）。
应用 OST：
$$\mathbb{E}[S_T^2 - T] = \mathbb{E}[S_0^2 - 0] = 0 \implies \mathbb{E}[T] = \mathbb{E}[S_T^2]$$
将吸收概率代入：
$$\mathbb{E}[S_T^2] = a^2 \cdot \mathbb{P}_a + (-b)^2 \cdot (1 - \mathbb{P}_a) = a^2 \left(\frac{b}{a+b}\right) + b^2 \left(\frac{a}{a+b}\right) = \frac{ab(a + b)}{a + b} = a \cdot b$$

> [!NOTE]
> **对称游走的心算秒杀公式**：
> - 胜率：$\mathbb{P}(\text{到达 } +a) = \frac{b}{a+b}$（反比于离目标的距离）；
> - 期望步数：$\mathbb{E}[T] = a \cdot b$（直接等于上下边界距离的乘积！例如从 0 出发在 $[-5, +5]$ 之间游走，期望步数正是 $5 \times 5 = 25$ 步）。

---

### 2. 非对称随机游走（$p \ne 1/2$）

#### (1) 求解吸收概率 $\mathbb{P}(S_T = a)$
构造指数几何鞅 $M_n = \left(\frac{q}{p}\right)^{S_n}$。
应用 OST：
$$\mathbb{E}\left[\left(\frac{q}{p}\right)^{S_T}\right] = \left(\frac{q}{p}\right)^{S_0} = 1$$
$$\mathbb{P}_a \left(\frac{q}{p}\right)^a + (1 - \mathbb{P}_a)\left(\frac{q}{p}\right)^{-b} = 1$$
解得：
$$\mathbb{P}_a = \frac{1 - (q/p)^{-b}}{(q/p)^a - (q/p)^{-b}} = \frac{1 - (p/q)^b}{1 - (p/q)^{a+b}} \cdot \left(\frac{p}{q}\right)^a = \frac{(q/p)^b - 1}{(q/p)^{a+b} - 1}$$

#### (2) 求解期望吸收时间 $\mathbb{E}[T]$
单步均值 $\mu = \mathbb{E}[X_1] = p(+1) + q(-1) = p - q \ne 0$。
构造线性漂移鞅 $M_n = S_n - n(p - q)$。
应用 OST：
$$\mathbb{E}[S_T - T(p - q)] = 0 \implies \mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{p - q}$$
将终局状态期望 $\mathbb{E}[S_T] = a \mathbb{P}_a - b(1 - \mathbb{P}_a)$ 代入：
$$\mathbb{E}[T] = \frac{a \mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$$

---

## 模块四：最优时停理论与 4 大高频面试真题

### 最优时停问题（Optimal Stopping Problem）的数学架构
- **目标**：给定适应于信息流 $\{\mathcal{F}_n\}$ 的收益过程 $\{Z_n\}_{n=0}^N$，寻找一个停时 $T^*$ 使得期望收益最大：
  $$V_0 = \sup_{T \in \mathcal{T}} \mathbb{E}[Z_T]$$
- **斯奈尔包络（Snell Envelope）**：定义值过程序列 $\{U_n\}$：
  $$U_N = Z_N$$
  $$U_n = \max\left(Z_n, \mathbb{E}[U_{n+1} \mid \mathcal{F}_n]\right), \quad n = N-1, N-2, \dots, 0$$
- **最优规则**：首次出现“当前即时收益 $Z_n$ 等于继续持有期望价值 $U_n$”时停下：
  $$T^* = \min\{n \ge 0 : Z_n = U_n\}$$

---

### 真题 1：经典秘书问题（The Secretary Problem / $37\%$ 法则）

> **原题描述（Citadel / SIG / Jane Street 高频）**：
> 有 $n$ 位候选人按完全随机的顺序依次前来面试。你必须在每次面试结束后**立即决定录用或拒绝**，且一旦拒绝便不可反悔召回。
> 面试官只能观察到当前候选人与之前所有已面试候选人的**相对优劣排名**。
> **目标**：制定一个最优策略，使得选到**全局综合排名第一（唯一最好）**候选人的概率最大。

#### 最优策略形式
由逆向归纳法可证，最优策略必为**阈值策略（Threshold Strategy）**：
- **观察阶段**：前 $k-1$ 个人一律拒绝，仅作为基准样本，记录其中的历史最高分 $M_{k-1}$；
- **遴选阶段**：从第 $k$ 个人开始，一旦遇到第一个比前 $k-1$ 个人都要优秀的人，立即录用并结束游戏；若直到第 $n$ 个人仍未出现，则被迫录用第 $n$ 个人。

#### 概率精确推导
记事件 $S$ 为“成功选到全局最好的第 $j$ 位候选人”。
第 $j$ 个人（$j \ge k$）被录用且为全局最优，必须同时满足两个独立事件：
1. 全局最好的人恰好排在第 $j$ 位（概率为 $\frac{1}{n}$）；
2. 前 $j-1$ 个人中的相对最高分，必须落在前 $k-1$ 个人的观察区内（否则在 $k \sim j-1$ 之间就会触发停止条件提前录用了次优者），该概率为 $\frac{k-1}{j-1}$。

因此，固定阈值 $k$ 时的成功概率为：
$$\mathbb{P}(\text{Success} \mid k) = \sum_{j=k}^n \mathbb{P}(\text{第 } j \text{ 人是最好的且前 } j-1 \text{ 人的最好者在前 } k-1 \text{ 人中})$$
$$= \sum_{j=k}^n \frac{1}{n} \cdot \frac{k-1}{j-1} = \frac{k-1}{n} \sum_{j=k}^n \frac{1}{j-1}$$

#### 渐近极限分析（$n \to \infty$）
令 $x = \frac{k}{n} \in (0, 1)$，利用黎曼和积分逼近调和级数：
$$\sum_{j=k}^n \frac{1}{j-1} \approx \int_k^n \frac{1}{t} dt = \ln\left(\frac{n}{k}\right) = -\ln x$$
因此成功概率函数转化为关于连续比例 $x$ 的平滑函数：
$$f(x) = x (-\ln x) = -x \ln x$$

对 $x$ 求一阶导数寻找极值点：
$$f'(x) = -\ln x - 1 = 0 \implies \ln x = -1 \implies x^* = \frac{1}{e} \approx 0.3679 \approx 36.8\%$$
此时全局最大成功概率同样为：
$$f(x^*) = \frac{1}{e} \approx 36.8\%$$

> **量化面试结论**：无论 $n$ 多大，先拒绝前 $36.8\%$ 的人，随后选择第一个超越前期峰值的候选人，能以高达 $36.8\%$ 的概率精确命中全局第一！

---

### 真题 2：有限次掷骰子最优时停（Sequential Die Rolling Game）

> **原题描述（Optiver / Jane Street 交易员笔试真题）**：
> 你最多可以掷一枚均匀的 6 面骰子 $N$ 次。每次掷出点数后，你可以选择：
> 1. 接受当前点数 $X$，获得 $\$X$ 现金并结束游戏；
> 2. 放弃当前点数，继续掷下一次（若到达第 $N$ 次则必须接受第 $N$ 次的点数）。
> **求游戏的公允价值（期望收益）及每一步的最优决策规则。**

#### 逆向归纳法动态推导
记 $v_k$ 为**在还剩 $k$ 次投掷机会时**的最大期望收益：

1. **还剩 1 次机会（$k = 1$）**：
   无得选，只能接受本次投掷：
   $$v_1 = \mathbb{E}[X] = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = 3.5$$

2. **还剩 2 次机会（$k = 2$）**：
   如果当前掷出点数 $X > v_1 = 3.5$（即掷出 4、5、6），选择拿钱离场；若 $X < 3.5$（即掷出 1、2、3），选择重掷（后续期望为 $v_1$）：
   $$v_2 = \mathbb{E}[\max(X, 3.5)] = \frac{1}{6}\left(3.5 \times 3 + 4 + 5 + 6\right) = \frac{10.5 + 15}{6} = \frac{25.5}{6} = 4.25$$
   **决策规则**：掷出 $\ge 4$ 停止，$\le 3$ 继续。

3. **还剩 3 次机会（$k = 3$）**：
   当前掷出点数 $X > v_2 = 4.25$（即掷出 5、6）时停止；若 $X \le 4$ 时重掷（期望为 $v_2 = 4.25$）：
   $$v_3 = \mathbb{E}[\max(X, 4.25)] = \frac{1}{6}\left(4.25 \times 4 + 5 + 6\right) = \frac{17 + 11}{6} = \frac{28}{6} \approx 4.667$$
   **决策规则**：掷出 $\ge 5$ 停止，$\le 4$ 继续。

4. **还剩 4 次机会（$k = 4$）**：
   $$v_4 = \mathbb{E}[\max(X, 4.667)] = \frac{1}{6}\left(4.667 \times 4 + 5 + 6\right) = \frac{18.667 + 11}{6} = \frac{29.667}{6} \approx 4.944$$
   **决策规则**：掷出 $\ge 5$ 停止，$\le 4$ 继续。

---

### 真题 3：美式期权提前行权与最优时停边界（American Options）

> **面试核心考点（Two Sigma / Morgan Stanley Strats 面试）**：
> 1. 为什么无红利美式看涨期权（American Call without Dividends）**永远不应该提前行权**？
> 2. 为什么美式看跌期权（American Put）存在最优提前行权边界 $S^*(t)$？

#### (1) 美式看涨期权：严密无套利证明
美式看涨期权价值定价为最优停时问题：
$$C_{\text{Amer}}(S_t, t) = \sup_{\tau \in [t, \mathcal{T}]} \mathbb{E}^\mathbb{Q}\left[e^{-r(\tau - t)} (S_\tau - K)^+ \mid \mathcal{F}_t\right]$$

利用 Jensen 不等式与贴现标的资产价格的鞅性质 $\mathbb{E}^\mathbb{Q}[e^{-r(\tau - t)} S_\tau \mid \mathcal{F}_t] = S_t$：
$$C(S_t, t) \ge \mathbb{E}^\mathbb{Q}[e^{-r(\tau - t)} (S_\tau - K) \mid \mathcal{F}_t] \ge S_t - K e^{-r(\tau - t)} > S_t - K \quad (\forall r > 0, \tau > t)$$

提前行权获得的即时内在价值是 $S_t - K$。
而由于 $C(S_t, t) > S_t - K$，**持有期权本身的市价严格大于立即行权的所得**。行权会白白扔掉时间价值（Time Value）以及对下行风险的保护，在市场上直接卖出期权永远比行权更优！

#### (2) 美式看跌期权：存在最优边界 $S^*(t)$ 的经济学机理
看跌期权的内在价值为 $K - S_t$。当标的资产价格暴跌至接近 $0$（$S_t \to 0$）时：
- 若立即行权，可立即获得 $\$K$ 现金，存入银行在剩余时间里赚取无风险利息 $rK > 0$；
- 若继续等待，最多也只能在到期时拿到 $\$K$（因为股价不能低于 0），反而损失了持有期间的现金利息。
因此当 $S_t \le S^*(t)$ 时，利息收益超过了期权的时间价值，此时提前行权是最优的。

---

### 真题 4：硬币序列等待时间与赌场赌资鞅（ABRACADABRA / Li's Martingale & Penney's Game）

> **原题描述（Jane Street / Optiver 压轴面试题）**：
> 不断抛掷一枚均匀硬币（正反面各 $1/2$）。
> 1. 首次出现序列 `HTTH` 的期望投掷次数 $\mathbb{E}[T_{\text{HTTH}}]$ 是多少？
> 2. 首次出现序列 `HTHT` 的期望投掷次数 $\mathbb{E}[T_{\text{HTHT}}]$ 是多少？为什么两者长度相同但期望时间不同？

#### 赌场赌资鞅构造法（Li's Martingale Theorem）
设想一个虚拟赌场，游戏规则如下：
1. 在每一轮投掷硬币之前，都有一个**新的赌徒**带着 **\$1** 进场；
2. 进场的第 $n$ 个赌徒，在第 $n$ 轮投掷前押 \$1 在目标序列的第 1 个字符（如 `H`）上；
   - 赔率为公正赔率 $1{:}2$（猜中翻倍变成 \$2，猜错输光出局）；
3. 若猜中，他在第 $n+1$ 轮将手头的全部 **\$2** 押在序列的第 2 个字符（如 `T`）上；
4. 依次类推，直到猜中整个长度为 $m$ 的序列并拿走 **\$$2^m$** 离场；一旦任何一步猜错，立即输光所有累积本金并离开。

**鞅的定义**：由于每一步都是公平博彩，**赌场的累积净利润** $M_n = (\text{所有赌徒投入的本金}) - (\text{赌场向所有赌徒支付的奖金})$ 是一个期望为 0 的鞅（$M_0 = 0$）！

当目标模式 $A$ 在时刻 $T$ 首次完整出现时：
- 一共有 $T$ 位赌徒进场，总投入本金为 **\$T**；
- 此时哪些赌徒手头持有正资金？
  - 在时刻 $T - m + 1$ 进场的赌徒，刚好完整匹配了整个模式 $A$，手里持有 **\$$2^m$**；
  - 在时刻 $T - k + 1$（$1 \le k < m$）进场的赌徒，手里持有 **\$$2^k$** 的充要条件是：**模式 $A$ 的长度为 $k$ 的前缀与长度为 $k$ 的后缀完全重合！**

根据停时定理 $\mathbb{E}[M_T] = 0$：
$$\mathbb{E}[T] - \sum_{k=1}^m 2^k \cdot \mathbf{1}_{\{\text{Prefix}(A, k) = \text{Suffix}(A, k)\}} = 0$$
$$\implies \mathbb{E}[T_A] = \sum_{k=1}^m 2^k \cdot \mathbf{1}_{\{\text{Prefix}(A, k) = \text{Suffix}(A, k)\}} = (A * A)_2$$

#### 极速心算对比
1. **模式 $A = \text{HTTH}$**（长度 $m = 4$）：
   - $k=1$: 前缀 `H` vs 后缀 `H` $\implies$ **匹配（重合）** $\to 2^1 = 2$；
   - $k=2$: 前缀 `HT` vs 后缀 `TH` $\implies$ 不匹配 $\to 0$；
   - $k=3$: 前缀 `HTT` vs 后缀 `TTH` $\implies$ 不匹配 $\to 0$；
   - $k=4$: 前缀 `HTTH` vs 后缀 `HTTH` $\implies$ **匹配** $\to 2^4 = 16$；
   $$\mathbb{E}[T_{\text{HTTH}}] = 2^4 + 2^1 = 16 + 2 = 18$$

2. **模式 $B = \text{HTHT}$**（长度 $m = 4$）：
   - $k=1$: 前缀 `H` vs 后缀 `T` $\implies$ 不匹配 $\to 0$；
   - $k=2$: 前缀 `HT` vs 后缀 `HT` $\implies$ **匹配（重合）** $\to 2^2 = 4$；
   - $k=3$: 前缀 `HTH` vs 后缀 `THT` $\implies$ 不匹配 $\to 0$；
   - $k=4$: 前缀 `HTHT` vs 后缀 `HTHT` $\implies$ **匹配** $\to 2^4 = 16$；
   $$\mathbb{E}[T_{\text{HTHT}}] = 2^4 + 2^2 = 16 + 4 = 20$$

> **深度直觉解释**：
> 为什么 `HTHT` 的平均等待时间（20）比 `HTTH`（18）更长？
> 因为 `HTHT` 具有周期为 2 的**自身高度重叠性（Self-Overlap）**。一旦 `HTHT` 出现了一次，它在短时间内发生连续成簇（Cluster）出现的概率更高。而在相同长期频率下，**成簇出现会导致无模式出现的“空白间隔”被拉长**，因此首次到达的期望等待时间更大！

---

## 模块五：量化面试极速自查矩阵

```html
<table>
  <thead>
    <tr>
      <th style="width: 25%;">问题场景</th>
      <th style="width: 25%;">推荐鞅构造</th>
      <th style="width: 25%;">解析公式 / 求解定理</th>
      <th style="width: 25%;">验证要点与陷阱</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>一维对称游走破产概率</strong></td>
      <td>$M_n = S_n$</td>
      <td>$\mathbb{P}_a = \frac{b}{a + b}$</td>
      <td>边界有界，$\mathbb{E}[T] < \infty$ 自动满足</td>
    </tr>
    <tr>
      <td><strong>一维对称游走期望步数</strong></td>
      <td>$M_n = S_n^2 - n$</td>
      <td>$\mathbb{E}[T] = a \cdot b$</td>
      <td>二阶 Wald 等式，$\mu = 0$ 且 $\sigma^2 = 1$</td>
    </tr>
    <tr>
      <td><strong>一维非对称游走破产概率</strong></td>
      <td>$M_n = (q/p)^{S_n}$</td>
      <td>$\mathbb{P}_a = \frac{(q/p)^b - 1}{(q/p)^{a+b} - 1}$</td>
      <td>利用 $p(q/p) + q(p/q) = 1$ 验证鞅性质</td>
    </tr>
    <tr>
      <td><strong>一维非对称游走期望步数</strong></td>
      <td>$M_n = S_n - n(p - q)$</td>
      <td>$\mathbb{E}[T] = \frac{a\mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$</td>
      <td>分母 $p - q \ne 0$，一阶 Wald 等式</td>
    </tr>
    <tr>
      <td><strong>单边吸收等待时间</strong></td>
      <td>Wald 指数鞅 $e^{\theta S_n} / M(\theta)^n$</td>
      <td>$\mathbb{E}[s^{T_a}] = \left(\frac{1 - \sqrt{1 - 4pqs^2}}{2ps}\right)^a$</td>
      <td>$\mathbb{P}(T < \infty) = 1$ 但 $\mathbb{E}[T] = \infty$（零漂移陷阱）</td>
    </tr>
    <tr>
      <td><strong>序列模式等待时间</strong></td>
      <td>赌场赌资净利润鞅</td>
      <td>$\mathbb{E}[T_A] = (A * A)_2 = \sum 2^k \mathbf{1}_{\{\text{前缀=后缀}\}}$</td>
      <td>Penney's Game 非传递性：后手总可构造优势前缀</td>
    </tr>
    <tr>
      <td><strong>连续/离散最优时停</strong></td>
      <td>斯奈尔包络（Snell Envelope）</td>
      <td>$U_n = \max(Z_n, \mathbb{E}[U_{n+1} \mid \mathcal{F}_n])$</td>
      <td>倒推逆向归纳法寻找最佳临界阈值</td>
    </tr>
  </tbody>
</table>
```
