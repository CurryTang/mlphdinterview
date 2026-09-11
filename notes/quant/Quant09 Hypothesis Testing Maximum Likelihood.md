# Quant 06 · 假设检验与似然：方向、边界与偏差方差

课程位置：[[Quant06 High Dimensional Integral Dominated Convergence|05 高维积分与大数定律]] → 本篇 → [[Quant11 Martingales Stopping Times Random Walks|07 鞅、停时与下注]]

处理参数估计与检验的一般顺序：

```text
1. p 值：看观测值相对原假设期望的方向，决定检验侧。
2. 似然函数：写出 L(theta)。支撑集不依赖参数则求导找驻点；依赖则看单调性找边界最大值。
3. 似然比：化简比值，合并样本统计量找充分统计量。
4. 矩估计：总体矩表述为参数，用样本矩替换求解。
5. 比较估计量：均方误差 = 偏差平方 + 方差。
```

---

## 1 · p 值与检验方向

### 10 次掷硬币中观测到 8 次正面

设 $X$ 为 10 次独立公平硬币的正面向上面数，原假设 $H_0:p=\frac12$，$\mathbb E[X]=10\times\frac12=5$。观测值 $X=8$ 明显大于期望，因此走右尾检验。

p 值为原假设下观测到当前或更极端结果的概率：

$$
p=\mathbb P(X\ge 8)=\frac{\binom{10}{8}+\binom{10}{9}+\binom{10}{10}}{2^{10}}=\frac{45+10+1}{1024}=\frac{56}{1024}\approx0.0547
$$

### 检验方向的原则

由备择假设 $H_1$ 决定使用哪侧尾部。观测值本身只用于判断偏离方向，公式选择完全取决于 $H_1$：

- $H_1:\theta>\theta_0$：右尾检验，$p=\mathbb P(T\ge t_{\text{obs}}\mid H_0)$
- $H_1:\theta<\theta_0$：左尾检验，$p=\mathbb P(T\le t_{\text{obs}}\mid H_0)$
- $H_1:\theta\ne\theta_0$：双尾检验，$p=2\min\big(\mathbb P(T\ge t_{\text{obs}}),\mathbb P(T\le t_{\text{obs}})\big)$

双尾公式中的“乘二”在分布关于均值对称时精确。分布不对称时，更严格的定义是把似然不超过观测值似然的所有结果计入。

### 非 1/2 概率的右尾检验

54 张牌（含 2 张大小王）放回抽取 5 次，观测到 3 张大小王。原假设每次抽到大小王概率 $p_0=\frac{2}{54}=\frac{1}{27}$。$X\sim\mathrm{Binomial}(5,\frac1{27})$，$\mathbb E[X]=5\times\frac1{27}\approx0.185$。

观测 $X=3$ 远大于期望。备择假设 $H_1:p>\frac1{27}$，走右尾检验：

$$
p\text{-value}=\mathbb P(X\ge3)=\sum_{k=3}^5\binom5k\left( \frac1{27} \right)^k\left( \frac{26}{27} \right)^{5-k}=\frac{6891}{14348907}\approx0.00048
$$

### 左尾与双尾

如果观测到 $X=2$（$n=10, p_0=\frac12$），小于期望 5，备择假设 $H_1:p<\frac12$，走左尾检验：

$$
p\text{-value}=\mathbb P(X\le2)=\frac{\binom{10}{0}+\binom{10}{1}+\binom{10}{2}}{2^{10}}=\frac{56}{1024}\approx0.0547
$$

这个数值和右尾 p 值相等，因为二项分布在此处对称。若不预设方向，$H_1:p\ne\frac12$，走双尾检验：

$$
p\text{-value}=2\times\mathbb P(X\ge8)=2\times\frac{56}{1024}\approx0.1094
$$

---

## 2 · 似然函数与最大似然估计

### 均值参数化的指数分布

设 $x_1,\ldots,x_n$ 独立同分布于均值为 $\beta$ 的指数分布，密度 $f(x\mid\beta)=\frac1\beta e^{-x/\beta}$。支撑集 $[0,\infty)$ 不依赖 $\beta$。似然函数：

$$
L(\beta)=\prod_{i=1}^n\frac1\beta e^{-x_i/\beta}=\beta^{-n}e^{-\sum_i x_i/\beta}
$$

对数似然：

$$
\ell(\beta)=-n\ln\beta-\frac{\sum_i x_i}{\beta}
$$

求导置零：

$$
\ell'(\beta)=-\frac n\beta+\frac{\sum_i x_i}{\beta^2}=0\quad\Longrightarrow\quad \hat\beta=\bar X
$$

二阶导数 $\ell''(\bar X)<0$，确认为最大值。使用均值参数化时 MLE 直接是 $\bar X$，如果是速率参数化则为 $1/\bar X$。

### 边界上的 MLE：$\mathrm{Unif}[\theta,2\theta]$

设 $x_1,\ldots,x_n \overset{i.i.d.}{\sim} \mathrm{Unif}[\theta,2\theta]$，求 $\theta$ 的 MLE 及偏差与方差。

似然函数包含两个指示函数：

$$
L(\theta)=\theta^{-n}\mathbf 1\{\theta\le X_{(1)}\}\mathbf 1\{2\theta\ge X_{(n)}\}
$$

合法定义域为 $\theta\in\left[ \frac{X_{(n)}}2,\,X_{(1)} \right]$。在此区间上，$\theta^{-n}$ 严格递减，似然在左端点最大：

$$
\boxed{\hat\theta=\frac{X_{(n)}}2}
$$

求偏差与方差：利用 $X_{(n)}=\theta(1+U_{(n)})$，其中 $U_{(n)}$ 为 $n$ 个独立 $\mathrm{Unif}[0,1]$ 的最大值。

$$
\mathbb E[U_{(n)}]=\frac n{n+1},\qquad \mathrm{Var}(U_{(n)})=\frac n{(n+1)^2(n+2)}
$$

一阶矩：

$$
\mathbb E[X_{(n)}]=\theta\left( 1+\frac n{n+1} \right)=\theta\cdot\frac{2n+1}{n+1}
$$

$$
\mathbb E[\hat\theta]=\frac12\mathbb E[X_{(n)}]=\theta\cdot\frac{2n+1}{2(n+1)}
$$

$$
\mathrm{Bias}(\hat\theta)=-\frac{\theta}{2(n+1)}
$$

方差不受平移影响，只由缩放系数决定：

$$
\mathrm{Var}(\hat\theta)=\frac14\mathrm{Var}(X_{(n)})=\frac{n\,\theta^2}{4(n+1)^2(n+2)}
$$

无偏修正：$\tilde\theta=\frac{2(n+1)}{2n+1}\hat\theta=\frac{(n+1)X_{(n)}}{2n+1}$。

---

## 3 · 似然比与充分统计量

比较 $N(0,\sigma^2)$ 和 $N(\mu,\sigma^2)$，方差已知，原假设 $H_0:\mu=0$，备择假设 $H_1:\mu>0$。

似然比：

$$
T=\frac{L_1}{L_0}=\prod_{i=1}^n\frac{\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2} \right)}{\exp\left( -\frac{x_i^2}{2\sigma^2} \right)}
$$

指数展开相减：

$$
-\frac{(x_i-\mu)^2}{2\sigma^2}+\frac{x_i^2}{2\sigma^2}=\frac{\mu x_i}{\sigma^2}-\frac{\mu^2}{2\sigma^2}
$$

$$
T=\exp\left( \frac{\mu}{\sigma^2}\sum_{i=1}^n x_i-\frac{n\mu^2}{2\sigma^2} \right)
$$

由于 $\mu/\sigma^2>0$，$T$ 是 $\sum_i x_i$（或 $\bar X$）的严格递增函数。事件 "$T>c$" 等价于 "$\bar X>c'$"。拒绝域由 $\bar X$ 决定，$\bar X$ 是充分统计量。

### 混合备择假设

如果备择假设改成 “N 组独立测量中恰好有一组（未知哪组）均值平移了 $A$”，由于具体组别未知，$H_1$ 的似然是对 $N$ 个选择求全概率混合：

$$
L_1=\frac1N\sum_{j=1}^N\prod_{i=1}^N f\big(x_i-A\cdot\mathbf 1\{i=j\}\big)
$$

提取公共因子 $\prod_i f(x_i)$，剩余部分是逐项似然比 $\frac{f(x_j-A)}{f(x_j)}$ 的加权求和，其代数结构与前面完全一致。

---

## 4 · 偏差与方差对比：MLE vs 矩估计

对于密度 $f(x\mid\theta)=\frac{3x^2}{\theta^3}$（$0\le x\le\theta$），分别求矩估计与 MLE，并比较。

### 矩估计

先确认密度合法：积分 $\int_0^\theta 3x^2/\theta^3 dx = 1$。一阶总体矩：

$$
\mathbb E[X]=\int_0^\theta x\cdot\frac{3x^2}{\theta^3}\,dx=\frac{3\theta}4
$$

令样本均值等于总体矩：

$$
\bar X=\frac{3\theta}4\quad\Longrightarrow\quad \hat\theta_C=\frac43\bar X
$$

后面对比方差还需要 $X$ 的方差。二阶矩：

$$
\mathbb E[X^2]=\int_0^\theta x^2\cdot\frac{3x^2}{\theta^3}\,dx=\frac{3\theta^2}5
$$

$$
\mathrm{Var}(X)=\frac{3\theta^2}5-\left( \frac{3\theta}4 \right)^2=\frac{3\theta^2}5-\frac{9\theta^2}{16}=\frac{3\theta^2}{80}
$$

$\hat\theta_C$ 是样本均值的线性变换，方差按平方缩放：

$$
\mathrm{Var}(\hat\theta_C)=\left( \frac43 \right)^2\cdot\frac{\mathrm{Var}(X)}n=\frac{16}9\cdot\frac{3\theta^2}{80n}=\frac{\theta^2}{15n}
$$

### MLE

似然函数在定义域 $\theta\ge X_{(n)}$ 上递减：

$$
L(\theta)=\frac{3^n\prod x_i^2}{\theta^{3n}}\mathbf 1\{\theta\ge X_{(n)}\}
$$

最大值在左端点：$\hat\theta_A=X_{(n)}$。

求分布与矩：

$$
F(x)=\left( \frac x\theta \right)^3 \Longrightarrow f_{X_{(n)}}(t)=\frac{3n\,t^{3n-1}}{\theta^{3n}}
$$

$$
\mathbb E[X_{(n)}]=\int_0^\theta t\cdot\frac{3n\,t^{3n-1}}{\theta^{3n}}\,dt=\frac{3n}{3n+1}\theta,\qquad \mathrm{Bias}(\hat\theta_A)=-\frac{\theta}{3n+1}\approx-\frac{\theta}{3n}
$$

$$
\mathbb E[X_{(n)}^2]=\int_0^\theta t^2\cdot\frac{3n\,t^{3n-1}}{\theta^{3n}}\,dt=\frac{3n}{3n+2}\theta^2
$$

$$
\mathrm{Var}(\hat\theta_A)=\frac{3n}{3n+2}\theta^2-\left( \frac{3n}{3n+1} \right)^2\theta^2
$$

令 $m=3n$，通分：

$$
\frac{m}{m+2}-\frac{m^2}{(m+1)^2}=\frac{m(m+1)^2-m^2(m+2)}{(m+2)(m+1)^2}=\frac{m}{(m+2)(m+1)^2}
$$

代入 $m=3n$：

$$
\mathrm{Var}(\hat\theta_A)=\frac{3n\,\theta^2}{(3n+2)(3n+1)^2}\approx\frac{\theta^2}{9n^2}
$$

### 比较结论

| 估计量 | 偏差 | 方差 | 均方误差 ($n$ 较大) |
|---|---|---|---|
| MLE ($X_{(n)}$) | $\approx-\frac{\theta}{3n}$ | $\approx\frac{\theta^2}{9n^2}$ | $\approx\frac{2\theta^2}{9n^2}$ |
| 矩估计 ($\frac43\bar X$) | $0$ | $\frac{\theta^2}{15n}$ | $\frac{\theta^2}{15n}$ |

MLE 虽有偏，但方差为 $O(1/n^2)$。大样本下，其均方误差远小于无偏但方差为 $O(1/n)$ 的矩估计。

---

## 5 · 估计与检验核心检查单

1. **支撑集是否依赖参数**：依赖就不要求导，直接判断似然作为参数函数的单调性，从边界读出最大值；不依赖就走标准路径，对数似然求导置零，二阶导数确认最大值。
2. **p 值的方向判断**：先看观测值相对原假设下期望值是偏大还是偏小，再决定往哪一侧算尾部。不等号是否取等（$\ge$ 还是 $>$）、单尾还是双尾，要在写公式前确认。
3. **似然比的充分统计量**：化简后的似然比是否是某个样本统计量（如均值、和）的单调函数？如果是，该统计量即为充分统计量，拒绝域完全由它决定。
4. **矩估计的标准流程**：写出总体矩表达式，用样本矩替换，最后解出参数。比较方差时，别忘了先求单观测的方差再按线性系数缩放。
5. **比较估计量的完整标准**：只看无偏性是片面的。完整的比较标准是“均方误差 = 偏差平方 + 方差”。有偏估计量若方差足够小，在大样本下经常整体更优。

一句话概括核心：

> 似然函数的形状先于似然函数的导数。支撑集依赖参数时先看单调性后看边界；不依赖参数时才轮到求导。无论走哪条路，最终评价估计量优劣的标准始终是均方误差，而不是单一片面的无偏标签。
