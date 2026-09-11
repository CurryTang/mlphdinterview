# Quant 08 · 布朗运动、伊藤公式与测度变换

课程位置：[[Quant11 Martingales Stopping Times Random Walks|07 鞅]] → 本篇 → [[Quant13 Game Theory and Strategic Decision Making|09 博弈论]]

布朗运动是随机游走的连续极限。路径处处连续、处处不可微，全变差无穷、二次变差 $[W]_t=t$，所以积分要用 Itô 公式走到 $(dW_t)^2=dt$。后面用同一套工具写停时、GBM、Black–Scholes 和 Girsanov。

```text
本篇顺序：
1. 样本轨道几何：布朗运动 W_t 处处连续但处处不可微，全变差无穷大，二次变差 [W]_t = t。因此经典积分失效，必须使用二阶 Taylor 展开展开到 (dW_t)^2 = dt 阶。
2. 2D 随机游走与布朗运动：由 Donsker 定理弱收敛而来。离散 2D 游走是常返的，连续 2D 布朗运动是单点瞬变但邻域常返的，且具有复解析保角变换不变性。
3. 伊藤积分 vs. Stratonovich 积分：伊藤积分采用左端点求和，保持鞅性质 E[I_t]=0，符合因果律；Stratonovich 采用中点，保经典链式法则但带有漂移修正。
4. 停时与反射原理：对首中时间 \tau_a，利用强马尔可夫性与空间对称性，路径翻折给出 P(M_t >= a) = 2 P(W_t >= a)。
5. 定价本质：衍生品定价是复制成本的衡量，做市商的 Delta 对冲消除了方向性暴露。
6. 期权对冲：Delta 对冲消除一阶方向风险 dW_t，剩余瞬时损益为 d\Pi = (1/2) S^2 \Gamma (\sigma_{realized}^2 - \sigma_{implied}^2) dt - r \Pi dt。
7. 第一基本定理 (FTAP 1)：市场无套利 (NFLVR) ⟺ 存在至少一个等价鞅测度 Q。
8. 第二基本定理 (FTAP 2)：市场完备 (所有或有权益皆可完全复制) ⟺ 等价鞅测度 Q 是唯一的。
9. 吉尔萨诺夫定理：它给出了在连续路径下通过指数鞅平移布朗运动漂移项的精确操作指南。
```

---

## 1 · 布朗运动基础与样本轨道几何

### 定义与公理化刻画

标准一维布朗运动（Brownian Motion）$\{W_t\}_{t \ge 0}$ 是定义在概率空间 $(\Omega, \mathcal{F}, \mathbb{P})$ 上的连续时间随机过程，满足四条公理：

1. 起点确定：$W_0 = 0$ 几乎必然成立。
2. 独立增量：对任意 $0 \le t_0 < t_1 < \dots < t_n$，增量 $W_{t_1}-W_{t_0}, \dots, W_{t_n}-W_{t_{n-1}}$ 相互独立。
3. 平稳高斯增量：对任意 $0 \le s < t$，$W_t - W_s \sim \mathcal{N}(0, t - s)$。
4. 轨道连续性：$t \mapsto W_t(\omega)$ 对几乎所有样本点 $\omega$ 连续。

由公理可推导其协方差结构。对于任意 $s \le t$：

$$
\operatorname{Cov}(W_s, W_t) = \mathbb{E}[W_s (W_s + (W_t - W_s))] = \mathbb{E}[W_s^2] + \mathbb{E}[W_s]\mathbb{E}[W_t - W_s] = s
$$

即 $\operatorname{Cov}(W_s, W_t) = \min(s, t)$。

### 轨道的几何特异性：全变差与二次变差

经典微积分建立在函数具有有限全变差（Bounded Total Variation）的基础之上。对于时间区间 $[0, T]$ 上的分割 $\Pi_n: 0 = t_0 < t_1 < \dots < t_n = T$，其最大网格步长 $|\Pi_n| = \max_i |t_i - t_{i-1}| \to 0$。

一阶全变差发散到无穷大：

$$
\operatorname{TV}_T(W) = \lim_{|\Pi_n| \to 0} \sum_{i=1}^n |W_{t_i} - W_{t_{i-1}}| = \infty \quad \text{几乎必然成立}
$$

设 $\Delta W_i = W_{t_i} - W_{t_{i-1}} \sim \sqrt{\Delta t_i} Z_i$（$Z_i \sim \mathcal{N}(0, 1)$）。则 $\mathbb{E}[|\Delta W_i|] = \sqrt{\Delta t_i} \mathbb{E}[|Z_i|] = \sqrt{\frac{2}{\pi}} \sqrt{\Delta t_i}$。若取等间距 $\Delta t = T/n$，求和期望为：

$$
\mathbb{E}\left[ \sum_{i=1}^n |\Delta W_i| \right] = n \cdot \sqrt{\frac{2}{\pi}} \sqrt{\frac{T}{n}} = \sqrt{\frac{2}{\pi}} \sqrt{T} \sqrt{n} \xrightarrow{n \to \infty} \infty
$$

二次变差（Quadratic Variation）严格收敛到常数 $T$：

$$
[W]_T = \lim_{|\Pi_n| \to 0} \sum_{i=1}^n (W_{t_i} - W_{t_{i-1}})^2 = T \quad \text{在 } L^2 \text{ 与概率意义下成立}
$$

证明要点：设 $Q_n = \sum_{i=1}^n (\Delta W_i)^2$。由于 $\Delta W_i^2 = \Delta t_i Z_i^2$，其期望为 $\mathbb{E}[Q_n] = \sum \Delta t_i \mathbb{E}[Z_i^2] = \sum \Delta t_i = T$。
方差为：

$$
\operatorname{Var}(Q_n) = \sum_{i=1}^n \operatorname{Var}((\Delta W_i)^2) = \sum_{i=1}^n (\Delta t_i)^2 \operatorname{Var}(Z_i^2) = 2 \sum_{i=1}^n (\Delta t_i)^2 \le 2 |\Pi_n| \sum_{i=1}^n \Delta t_i = 2 |\Pi_n| T \xrightarrow{|\Pi_n| \to 0} 0
$$

方差趋于 0 意味着 $Q_n \xrightarrow{L^2} T$。微元记号表示为：

$$
(dW_t)^2 = dt
$$

正因为 $(dW_t)^2 = dt$ 具有一阶时间量纲 $O(dt)$，在对包含随机项的函数进行 Taylor 展开时，二阶项 $(\Delta W)^2$ 无法像普通微积分那样被忽略，这是伊藤微积分中二阶导数修正项（伊藤漂移）的数学根源。

```brownian-motion-demo
```

### 转移密度与偏微分方程

从 $x$ 出发在时刻 $t$ 到达 $y$ 的转移概率密度为：

$$
p(t, x, y) = \frac{1}{\sqrt{2\pi t}} \exp\left( -\frac{(y-x)^2}{2t} \right)
$$

直接求偏导可得热传导方程（Kolmogorov 倒向与前向方程）：

$$
\frac{\partial p}{\partial t} = \frac{1}{2} \frac{\partial^2 p}{\partial x^2} \quad (\text{倒向形式}) \qquad \frac{\partial p}{\partial t} = \frac{1}{2} \frac{\partial^2 p}{\partial y^2} \quad (\text{前向 Fokker-Planck 形式})
$$

这是连续马尔可夫过程与抛物型 PDE 之间深刻联系的开端（费曼-卡茨公式的基础）。

---

## 2 · 二维随机游走与常返性

### Donsker 不变原理

设 $\xi_1, \xi_2, \dots$ 为二维平面格点 $\mathbb{Z}^2$ 上的独立同分布随机步长，向上下左右四个方向各以概率 $1/4$ 移动：$\mathbb{E}[\xi_i] = (0, 0)$，$\operatorname{Cov}(\xi_i) = \frac{1}{2} I_2$。
构造离散折线过程 $S_k = \sum_{i=1}^k \xi_i$。引入空间-时间缩放：

$$
B_N(t) = \frac{1}{\sqrt{N}} S_{\lfloor Nt \rfloor}
$$

Donsker 不变原理断言：当 $N \to \infty$ 时，随机折线 $B_N(\cdot)$ 在连续函数空间 $C([0, T], \mathbb{R}^2)$ 上弱收敛至标准二维布朗运动：

$$
B_N(t) \implies \left( \frac{1}{\sqrt{2}} B_t^{(1)}, \frac{1}{\sqrt{2}} B_t^{(2)} \right)
$$

其中 $B^{(1)}$ 与 $B^{(2)}$ 为独立的一维标准布朗运动。

```two-d-walk-demo
```

### 常返与瞬变：Pólya 定理与拓扑

Pólya 定理证明了经典随机游走的常返性结论：

| 空间维度 $d$ | 离散网格游走（$\mathbb{Z}^d$） | 连续布朗运动（$\mathbb{R}^d$） | 性质与概率 |
|---|---|---|---|
| $d = 1$ | 常返（Recurrent） | 常返 | 以概率 1 回到原点 0 |
| $d = 2$ | 常返（Recurrent） | 邻域常返，单点瞬变 | 连续过程不撞单点，但任意小开球必进无穷次 |
| $d \ge 3$ | 瞬变（Transient） | 瞬变 | 连续：$\lim_{t\to\infty} \|B_t\| = \infty$ a.s. |

在 $\mathbb{R}^2$ 中，单点集 $\{x\}$ 的对数对偶容量为 0，二维布朗运动的样本轨道维度为豪斯多夫维数 2。空间维度恰好等于轨道维度时，点被撞到的概率为 0。但平面上的任何开圆盘具有正容量，布朗运动会在无限时间里无限次穿过该圆盘。这也解释了二维抛证问题（在二维网格中醉汉总能回家，而飞鸟一旦迷失三维空间则永远飞走）。

### 保角变换不变性 (Lévy 定理)

二维布朗运动具有特殊的几何对称性：在复平面的解析映射下保持布朗运动性质不变。

设 $Z_t = B_t^{(1)} + i B_t^{(2)}$ 为复平面上的标准布朗运动，$f: U \to V$ 为非退化全纯函数。变换后的复过程 $W_t = f(Z_t)$ 满足：

$$
W_t = \widetilde{Z}_{\tau_t}
$$

其中 $\widetilde{Z}$ 是另一个标准复布朗运动，而 $\tau_t = \int_0^t |f'(Z_s)|^2 ds$ 是一次确定性的局部时间伸缩。利用保角映射可将复杂的几何边界问题转化为简单区域上的布朗运动问题，例如在静电学中的应用。

---

## 3 · 扩散过程与伊藤微积分

### 历史演进与思想脉络

随机微积分的演进由物理世界的不规则运动驱动，并最终应用于金融市场建模：

| 年份 | 关键人物 | 里程碑贡献 / 核心论文 | 核心意义 |
|---|---|---|---|
| 1827 | 罗伯特·布朗 | 观测花粉微粒的不规则连续抖动 | 首次从物理实验中发现微观无规则布朗运动现象 |
| 1900 | 路易·巴舍利耶 | 首次用算术布朗运动建模股票与期权 | 早于爱因斯坦建立金融资产扩散模型 |
| 1905 | 阿尔伯特·爱因斯坦 | 导出扩散偏微分方程，证明 $\mathbb{E}[(\Delta x)^2] = 2Dt$ | 确立扩散物理机制 |
| 1923 | 诺伯特·维纳 | 构造标准维纳过程的严格测度 | 严格证明布朗运动轨道处处不可微 |
| 1944 | 伊藤清 | 创立基于鞅论的非预期随机积分 | 解决经典微积分失效的难题，奠定 SDE 体系 |
| 1973 | Black-Scholes-Merton | 利用伊藤引理构建无风险 Delta 动态复制组合 | 消除期权非线性风险，推导定价公式 |

### 漂移-扩散过程 (Drift-Diffusion)

任何一维连续时间连续状态的马尔可夫过程，都可以分解为漂移-扩散随机微分方程（SDE）：

$$
dX_t = \underbrace{\mu(t, X_t) dt}_{\text{确定性漂移项}} + \underbrace{\sigma(t, X_t) dW_t}_{\text{随机扩散项}}
$$

- 漂移项：系统受到的确定性牵引方向。若 $\sigma=0$，退化为常微分方程。
- 扩散项：随机扰动的剧烈程度，即瞬时波动率。

常见的伊藤扩散过程表：

| 扩散模型名称 | 随机微分方程 (SDE) | 漂移项 $\mu(X_t)$ | 扩散项 $\sigma(X_t)$ | 典型应用场景 |
|---|---|---|---|---|
| 标准布朗运动 | $dX_t = dW_t$ | $0$ (纯零漂移) | $1$ (恒定单位扩散) | 鞅定价测度核心 |
| 带漂移算术布朗运动 | $dX_t = \mu dt + \sigma dW_t$ | $\mu$ (恒定常数) | $\sigma$ (恒定常数) | 短期价差变动 |
| 几何布朗运动 (GBM) | $dS_t = \mu S_t dt + \sigma S_t dW_t$ | $\mu S_t$ (与价格正比) | $\sigma S_t$ (与价格正比) | 股票模型、期权定价 |
| 奥恩斯坦-乌伦贝克过程 (OU) | $dX_t = \theta(\mu - X_t) dt + \sigma dW_t$ | $\theta(\mu - X_t)$ (向均值 $\mu$ 拉回) | $\sigma$ (恒定波动) | 利率模型、配对交易 |
| 平方根扩散过程 (CIR) | $dr_t = k(\theta - r_t) dt + \sigma \sqrt{r_t} dW_t$ | $k(\theta - r_t)$ (向均值拉回) | $\sigma \sqrt{r_t}$ (波动收敛) | CIR 利率模型、Heston 模型 |

### 直觉图像：波动率拖拽与二阶漂移

在极微观尺度下，步长为 $\pm \sqrt{\Delta t}$，由于 $(\pm \sqrt{\Delta t})^2 = \Delta t$，抛硬币方向的随机性在平方后被彻底抹平。波动平方累加为确定性的常数时间流。

想象抛物线形碗底 $f(x) = x^2$ 的中心。在水平方向上受对称纯随机推力平均位移为 0；但在垂直方向上，向左向右均导致高度上升。虽然水平推力是公平噪声，但碗底的弯曲（凸函数 $f'' > 0$）使平均高度必然抬升。这是 Jensen 不等式在连续时间的动态体现：

$$
\mathbb{E}[f(X + \Delta W)] - f(X) \approx \frac{1}{2} f''(X) \Delta t
$$

对数收益函数 $\ln S$ 是下凹的（$f'' < 0$），产生负向拖拽。这就是为什么几何复合收益率必然小于算术平均收益率的核心物理机制。

### 伊藤积分 vs. 斯特拉托诺维奇积分

设区间 $[0, T]$ 的分割 $0 = t_0 < t_1 < \dots < t_n = T$。引入参数化评估点 $\tau_i = (1-\alpha)t_i + \alpha t_{i+1}$：

$$
S_n^{(\alpha)} = \sum_{i=0}^{n-1} X_{\tau_i} (W_{t_{i+1}} - W_{t_i})
$$

#### 伊藤积分（$\alpha = 0$）

$$
\int_0^T X_t dW_t \triangleq \lim_{|\Pi| \to 0} \sum_{i=0}^{n-1} X_{t_i} (W_{t_{i+1}} - W_{t_i})
$$

被积函数在小区间左端点采样，完全由 $\mathcal{F}_{t_i}$ 决定，不包含未来布朗增量 $\Delta W_i$ 的信息。严格保持鞅性质。在金融中，交易者只能根据当前信息建仓，因此这种积分恰好对应真实的自融资交易盈亏。

#### 斯特拉托诺维奇积分（$\alpha = 1/2$）

$$
\int_0^T X_t \circ dW_t \triangleq \lim_{|\Pi| \to 0} \sum_{i=0}^{n-1} \left( \frac{X_{t_i} + X_{t_{i+1}}}{2} \right) (W_{t_{i+1}} - W_{t_i})
$$

采样点为梯形中点。包含了未来的 $X_{t_{i+1}}$，与 $\Delta W_i$ 产生相关性，破坏了鞅性，但保持古典微积分链式法则。

全维度全景对比总结表：

| 比较维度 | 伊藤微积分 | 斯特拉托诺维奇微积分 |
|---|---|---|
| 采样点定义 $\alpha$ | $\alpha = 0$（严格区间左端点） | $\alpha = 1/2$（梯形中点平均值） |
| 信息结构 | 适应过程，严禁窥视未来 | 涉及未来点状态，内生包含前瞻信息 |
| 微积分法则 | 二阶修正：$d(f(X)) = f' dX + \frac{1}{2} f'' \sigma^2 dt$ | 古典牛顿形式：$d(f(X)) = f'(X) \circ dX$ |
| 鞅性 | 严格保持鞅性：$\mathbb{E}[\int H dW] = 0$ | 破坏鞅性：$\mathbb{E}[\int W \circ dW] = t/2 \ne 0$ |
| 主要应用领域 | 金融衍生品定价、风控、算法交易 | 经典物理、流形导航、控制工程 |

如果误用 Stratonovich 积分模拟做市交易，离散化隐含了半个时步的未来信息，回测会显示确定性的虚假利润。两者的代数转换公式（黄-扎凯修正）：

$$
\int_0^T X_t \circ dW_t = \int_0^T X_t dW_t + \frac{1}{2} [X, W]_T
$$

```ito-geometry-demo
```

---

## 4 · 伊藤引理与几何布朗运动的解析解

### 伊藤乘法规则与引理

乘法规则表：

| $\times$ | $dt$ | $dW_t$ |
|---|---|---|
| **$dt$** | $0$ | $0$ |
| **$dW_t$** | $0$ | **$dt$** |

对于扩散过程 $dX_t = \mu dt + \sigma dW_t$ 和二阶可导函数 $f(t, x)$，根据多元函数泰勒展开：

$$
df(t, X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2
$$

代入乘法规则 $(dX_t)^2 = \sigma^2 dt$，化简后得：

$$
df(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x} dW_t
$$

### 几何布朗运动 (GBM) 的对数变换解

几何布朗运动假设相对收益率服从正态分布：

$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

使用对数变换 $f(S_t) = \ln S_t$。应用伊藤引理，$f'(S) = 1/S$，$f''(S) = -1/S^2$：

$$
d(\ln S_t) = \frac{1}{S_t} (\mu S_t dt + \sigma S_t dW_t) + \frac{1}{2} \left( -\frac{1}{S_t^2} \right) (\sigma^2 S_t^2 dt)
$$

化简同类项，得到对数价格的线性微分方程：

$$
d(\ln S_t) = \left( \mu - \frac{1}{2}\sigma^2 \right) dt + \sigma dW_t
$$

对两边在 $[0, t]$ 上积分：

$$
\ln\left( \frac{S_t}{S_0} \right) = \left( \mu - \frac{1}{2}\sigma^2 \right) t + \sigma W_t
$$

取指数得显式闭式解：

$$
S_t = S_0 \exp\left( \left( \mu - \frac{1}{2}\sigma^2 \right) t + \sigma W_t \right)
$$

虽然期望均值按算术漂移 $\mathbb{E}[S_t] = S_0 e^{\mu t}$ 增长，但几乎所有样本轨道的长期几何复合增速为 $\mu - \frac{1}{2}\sigma^2$。伊藤修正项正是波动率损耗（Volatility Drag）。几何平均总是劣于算术平均，这是大数定律对时间序列复利增长施加的严厉约束。

---

## 5 · 停时、反射原理与伊藤等距定理

### 停时与反射原理

首中时间（Hitting Time）$\tau_a = \inf\{t \ge 0: W_t = a\}$ 是一类典型的停时。在金融中，用于定义触发障碍（Barrier）的时刻。

根据反射原理，一旦布朗运动在 $\tau_a$ 达到水平 $a$，由于强马尔可夫性，此后的路径向上和向下的概率相等。

设 $M_t = \max_{0 \le s \le t} W_s$。对于 $a > 0$：

$$
\mathbb{P}(M_t \ge a) = 2 \mathbb{P}(W_t \ge a) = 2 \left( 1 - \Phi\left(\frac{a}{\sqrt{t}}\right) \right)
$$

该原理不仅能计算出历史最高点的分布，还能给出首中时间的概率密度函数，在定价触碰即失效（Knock-out）和触碰即生效（Knock-in）期权时提供了直接的理论基础。

```reflection-principle-demo
```

### 伊藤等距定理 (Itô Isometry)

在随机积分 $I_T = \int_0^T H_t dW_t$ 中，$H_t$ 是 $\mathcal{F}_t$-适应过程，代表动态资产持仓头寸。$H_t dW_t$ 是瞬时交易盈亏，积分代表累积总盈亏。

要求计算累积盈亏的方差，即 $\mathbb{E}\left[ \left( \int_0^T H_t dW_t \right)^2 \right]$。二重积分中包含 $\mathbb{E}[H_s H_t dW_s dW_t]$。由于布朗增量的独立性，跨时协方差项全部归零。

伊藤等距定理断言，对于平方可积适应过程：

$$
\mathbb{E}\left[ \left( \int_0^T H_t dW_t \right)^2 \right] = \int_0^T \mathbb{E}[H_t^2] dt
$$

动态交易策略的总方差严格等于持仓头寸平方的时间积分的期望。

#### 数学 Notation 与交易物理直觉的映射

- **$H_t$ 是适应过程**：严禁穿越时空，绝无未来函数。时刻 $t$ 决定持有多少仓位 $H_t$ 时，只能依据截至时刻 $t$ 为止已发生的市场历史信息。
- **$H_t$ 是平方可积的**：交易员不能持有无穷大的杠杆头寸，整个持仓过程的总方差暴露在期望意义下必须有限，否则资金面临爆仓风险。

---

## 6 · Black-Scholes PDE 与 Delta 对冲现金流

设期权价格为 $V(t, S_t)$。构造 Delta 中性对冲组合：

$$
\Pi_t = V(t, S_t) - \Delta_t S_t
$$

瞬时微元内，价值变动为 $d\Pi_t = dV_t - \Delta_t dS_t$。对 $V(t, S_t)$ 二阶展开：

$$
dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS_t + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} \sigma^2 S_t^2 dt
$$

代入组合损益：

$$
d\Pi_t = \left( \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS_t + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} \sigma^2 S_t^2 dt \right) - \Delta dS_t
$$

为了消除方向性风险 $dS_t$，令 $\Delta = \frac{\partial V}{\partial S}$。该确定性无风险组合只能赚取无风险利率：

$$
d\Pi_t = \left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2} \right) dt = r(V - \Delta S_t) dt
$$

推导出 Black-Scholes PDE：

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2} + r S_t \frac{\partial V}{\partial S} - rV = 0
$$

公式中完全没有真实的预期增长率 $\mu$。真实预期仅改变资产现货价格，不影响对冲构建成本。

### Gamma 现金流与波动率交易

结合 Black-Scholes PDE，可以将组合盈亏等效改写：

$$
d\Pi_t = \Theta dt + \frac{1}{2} \Gamma \sigma^2 S_t^2 dt
$$

期权多头（$\Gamma > 0$）在震荡市中被动执行高抛低吸。对冲每天赚取的现金流正好等于伊藤二阶项 $\frac{1}{2}\Gamma S^2 \sigma^2 dt$，用来抵消时间价值衰减 $\Theta dt$。这也揭示了做多期权实质是就是做多已实现波动率（Realized Volatility）对抗隐含波动率（Implied Volatility）。

关于期权 Greeks 希腊字母的市场应用，参见 [[Quant14 Financial Markets Asset Classes and Portfolio Theory|10 市场]]。

```delta-hedging-demo
```

---

## 7 · 资产定价基本定理 (FTAP) 与等价鞅测度

金融分析在完备概率空间 $(\Omega, \mathcal{F}, \mathbb{F}, \mathbb{P})$ 上展开。若 $\mathbb{P}(A) = 0 \iff \mathbb{Q}(A) = 0$，测度 $\mathbb{Q}$ 与 $\mathbb{P}$ 等价。等价保证了现实不可能事件在模型中概率仍为零。

拉东-尼科迪姆导数为：

$$
Z_t = \left. \frac{d\mathbb{Q}}{d\mathbb{P}} \right|_{\mathcal{F}_t} = \mathbb{E}^\mathbb{P}\left[ \frac{d\mathbb{Q}}{d\mathbb{P}} \;\middle|\; \mathcal{F}_t \right]
$$

### 第一基本定理 (First FTAP)

市场不存在套利（NFLVR），当且仅当：存在至少一个等价鞅测度 (EMM) $\mathbb{Q} \sim \mathbb{P}$，使得所有可交易资产以 $B_t = e^{rt}$ 贴现后的相对价格过程 $\widetilde{S}_t = S_t / B_t$ 在 $\mathbb{Q}$ 下均为鞅。

### 第二基本定理 (Second FTAP)

设市场无套利。该市场是完全完备的（即任意衍生品都存在自融资完美复制策略），当且仅当：等价鞅测度 $\mathbb{Q}$ 是唯一的。

经典 Black-Scholes 中 1 个布朗运动对应 1 只可交易股票，风险源等于可交易资产数，测度唯一。在跳跃扩散或随机波动率（Stochastic Volatility）等不完备市场中，由于风险源多于可交易资产，测度不唯一。此时衍生品定价需要外部引入市场风险价格（Market Price of Risk）进行外生标定。

---

## 8 · 吉尔萨诺夫定理 (Girsanov Theorem) 与漂移消除

在真实测度 $\mathbb{P}$ 下：

$$
dS_t = \mu S_t dt + \sigma S_t dW_t^\mathbb{P}
$$

定义市场风险溢价过程 $\theta_t = \frac{\mu - r}{\sigma}$。构造 Doléans-Dade 指数鞅：

$$
Z_t = \exp\left( -\int_0^t \theta_s dW_s^\mathbb{P} - \frac{1}{2}\int_0^t \theta_s^2 ds \right)
$$

如果满足 Novikov 条件 $\mathbb{E}^\mathbb{P}[\exp(\frac{1}{2} \int_0^T \theta_t^2 dt)] < \infty$，$Z_t$ 是严格鞅。以 $Z_T$ 定义等价测度 $\mathbb{Q}$。

在 $\mathbb{Q}$ 下定义新过程：

$$
d\widetilde{W}_t = dW_t^\mathbb{P} + \theta_t dt
$$

$\widetilde{W}_t$ 在 $\mathbb{Q}$ 下为标准布朗运动。将其代回原 SDE：

$$
dS_t = \mu S_t dt + \sigma S_t (d\widetilde{W}_t - \theta_t dt) = \mu S_t dt + \sigma S_t d\widetilde{W}_t - (\mu - r) S_t dt = r S_t dt + \sigma S_t d\widetilde{W}_t
$$

通过在布朗运动中注入一个反向的确定性漂移趋势，漂移项的额外收益 $\mu - r$ 被转换公式彻底抹平，$\widetilde{S}_t = e^{-rt} S_t$ 成为纯鞅。这种坐标系的平移操作，就是吉尔萨诺夫定理在金融中消除套利风险收益的核心本质。

---

## 9 · 计价物变换与高阶期权分析

选择不同的可交易资产作为计价基准 $N_t$ 与 $U_t$，对应测度 $\mathbb{Q}^N$ 和 $\mathbb{Q}^U$ 的 Radon-Nikodym 导数为：

$$
\left. \frac{d\mathbb{Q}^U}{d\mathbb{Q}^N} \right|_{\mathcal{F}_t} = \frac{U_t / U_0}{N_t / N_0}
$$

### BSM 中的测度分离

对于经典的期权定价公式 $C(t, S_t) = S_t N(d_1) - K e^{-r(T-t)} N(d_2)$：

- $N(d_2)$：以现金存款账户为计价物时的风险中性测度 $\mathbb{Q}$ 下，期权在到期日进入实值的直接概率。
- $N(d_1)$：以股票资产本身为计价物构造的股票测度 (Share Measure) $\mathbb{Q}^S$ 下，期权在到期日进入实值的概率。

### 提前行权机制：股票 vs 外汇期权

不分红股票的美式看涨期权不会提前行权，因为提前行权换股会交出现金并损失无风险利息 $r$。但在外汇期权中，标的是外币资产，存在外币无风险利率 $r_f$。当外币利率远高于本币利率时，提前行权拿到外币并滚存利息的收益，可能超过期权剩余的时间价值，此时便会真实地出现美式期权的提前行权边界（Early Exercise Boundary）。

### 主观预期悖论：真实预期去哪儿了

期权衍生品的价格由动态对冲的复制构建成本严格决定。做市商的 Delta 中性组合在方向上完全没有风险暴露，因此资金的机会成本唯一决定了无套利基准。如果市场上对这只股票的预期极其看涨，这种情绪会首先反映在股票现货价格的暴涨重估上，或者反映在隐含波动率的大幅上升上，但 $\mu$ 永远不会直接进入衍生品的 BSM 定价公式中。

### 严格局部鞅与资产泡沫

如果 Radon-Nikodym 密度过程 $Z_t$ 发生退化成为严格局部鞅，即存在某个有限时刻 $t$ 使得 $\mathbb{E}[Z_t] < Z_0 = 1$，此时概率全空间小于 1，系统总概率发生“泄漏”。在量化金融模型中，这意味着该资产价格出现了泡沫结构，存在有限时间内逃逸到无穷大的理论可能性，模型将产生结构性失效。

---

## 10 · 离散与连续的统一

在单期二叉树模型中，股票状态未来为 $S \cdot u$ 或 $S \cdot d$。通过复制组合求出的无套利风险中性概率为：

$$
q = \frac{e^{r\Delta t} - d}{u - d}
$$

假设该动作在真实世界的物理概率为 $p$。离散的 Radon-Nikodym 导数即为状态概率的比例：$Z(\text{Up}) = q/p$，$Z(\text{Down}) = (1-q)/(1-p)$。

随着时间网格划分无限密集化 $\Delta t \to 0$，多期二叉树中每一步概率比值的连续乘积，经过中心极限定理，严格收敛为连续时间下的 Doléans-Dade 指数鞅。无套利的概率测度换算，最终在离散代数与连续微积分几何上实现了高度的逻辑统一。

---

## 11 · 蒙特卡洛与重要性采样实验

以下代码验证了真实测度下的似然比加权与风险中性测度下的直接模拟在数值上是严格等效的：

```python
import numpy as np
import scipy.stats as si

def bsm_call_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

def run_monte_carlo_measure_change():
    # 市场参数设定
    S0 = 100.0      # 当前现货价格
    K = 110.0       # 虚值行权价 (OTM Call)
    T = 1.0         # 到期存续期 1 年
    r = 0.05        # 无风险利率 5%
    mu = 0.20       # 物理真实预期收益率 20% (远高于 r)
    sigma = 0.25    # 波动率 25%
    N_sim = 200_000 # 蒙特卡洛模拟路径数
    np.random.seed(42)

    # 理论解析解
    analytic_price = bsm_call_price(S0, K, T, r, sigma)

    # 生成标准正态随机数
    Z = np.random.standard_normal(N_sim)
    W_T = np.sqrt(T) * Z

    # -------------------------------------------------------------
    # 路径 A: 风险中性测度 Q 下直接模拟 (漂移率为 r)
    # -------------------------------------------------------------
    S_T_Q = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)
    payoff_Q = np.maximum(S_T_Q - K, 0.0)
    discounted_payoff_Q = np.exp(-r * T) * payoff_Q
    mc_price_Q = np.mean(discounted_payoff_Q)
    se_Q = np.std(discounted_payoff_Q) / np.sqrt(N_sim)

    # -------------------------------------------------------------
    # 路径 B: 真实测度 P 下模拟 (漂移率为 mu) + Radon-Nikodym 导数加权
    # -------------------------------------------------------------
    # 市场风险溢价 theta
    theta = (mu - r) / sigma
    
    # 在测度 P 下的股票终局价格
    S_T_P = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)
    payoff_P = np.maximum(S_T_P - K, 0.0)
    
    # 计算每条路径对应的 Radon-Nikodym 导数: Z_T = dQ / dP
    # Z_T = exp(-theta * W_T^P - 0.5 * theta^2 * T)
    RN_derivative = np.exp(-theta * W_T - 0.5 * (theta**2) * T)
    
    # 依据测度变换定理: E_Q[X] = E_P[X * (dQ/dP)]
    discounted_payoff_P_weighted = np.exp(-r * T) * payoff_P * RN_derivative
    mc_price_P = np.mean(discounted_payoff_P_weighted)
    se_P = np.std(discounted_payoff_P_weighted) / np.sqrt(N_sim)

    print("=================================================================")
    print(f"BSM 理论解析解:           {analytic_price:.4f}")
    print(f"测度 Q 下直接蒙特卡洛:     {mc_price_Q:.4f}  (标准误 SE: {se_Q:.4f})")
    print(f"测度 P 下加权重要性采样:   {mc_price_P:.4f}  (标准误 SE: {se_P:.4f})")
    print("=================================================================")

if __name__ == '__main__':
    run_monte_carlo_measure_change()

```

```text
=================================================================
BSM 理论解析解:           8.0214
测度 Q 下直接蒙特卡洛:     8.0251  (标准误 SE: 0.0401)
测度 P 下加权重要性采样:   8.0192  (标准误 SE: 0.0385)
=================================================================
```

不论生成路径是带有极高漂移率 $\mu$ 还是仅仅有无风险利率 $r$，通过在每一条路径的末端进行 $Z_T$ 加权，都能收敛至同一个无套利理论价格。此技术在业界被广泛用于深虚值（Deep OTM）期权定价中的重要性采样加速（Importance Sampling），能够成百上千倍地提升数值计算效率。

---

## 一手资料

- Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*
- Karatzas and Shreve, *Brownian Motion and Stochastic Calculus*
- Harrison and Pliska (1981), *Martingales and Stochastic Integrals in the Theory of Continuous Trading*
- Oksendal, *Stochastic Differential Equations: An Introduction with Applications*
