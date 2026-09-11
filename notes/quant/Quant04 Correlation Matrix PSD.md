# Quant 04 · 协方差、正态与相关矩阵

课程位置：[[Quant03 Continuous Distribution Geometry Transform|03 连续分布]] → 本篇 → [[Quant06 High Dimensional Integral Dominated Convergence|05 高维积分]]

协方差量化了两个随机变量的线性同动趋势。去掉单位影响后，它就是相关系数。相关矩阵是多个变量之间相关系数的集合，它最重要的数学约束是半正定（Positive Semidefinite, PSD）。

---

## 1 · 协方差在量什么

设 $X, Y$ 是两个随机变量。协方差定义为：

$$
\operatorname{Cov}(X,Y) = \mathbb{E}\left[ (X-\mathbb{E}X)(Y-\mathbb{E}Y) \right]
$$

它衡量两个变量偏离均值时，是否经常同向变化。

| 情况 | 直觉 | 协方差符号 |
|---|---|---|
| $X$ 大于均值时，$Y$ 也常大于均值 | 同涨同跌 | 正 |
| $X$ 大于均值时，$Y$ 常小于均值 | 一个涨一个跌 | 负 |
| 没有稳定线性关系 | 线性同动弱 | 接近 0 |

轻量图像可以这样记：

```text
positive covariance        negative covariance        near zero covariance

y                          y                          y
|        *                 | *                        |   *    *
|      *                   |   *                      | *   *
|    *                     |     *                    |      *
|  *                       |       *                  | *       *
+--------- x               +--------- x               +--------- x
```

协方差有一个缺点：它受单位影响。把美元换成美分，协方差会放大很多。处理数值问题时，通常需要将其标准化。

---

## 2 · 相关系数是标准化后的协方差

相关系数定义为：

$$
\operatorname{corr}(X,Y) = \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
$$

其中，标准差表示为：

$$
\sigma_X=\sqrt{\operatorname{Var}(X)}
$$

$$
\sigma_Y=\sqrt{\operatorname{Var}(Y)}
$$

也可以先把变量本身进行标准化处理：

$$
Z_X=\frac{X-\mathbb{E}X}{\sigma_X}
$$

$$
Z_Y=\frac{Y-\mathbb{E}Y}{\sigma_Y}
$$

经过这一步，标准化变量的均值为 0，方差为 1。此时：

$$
\operatorname{corr}(X,Y)=\operatorname{Cov}(Z_X,Z_Y)
$$

相关系数可以看作“去掉度量单位以后，两个变量线性同动的纯粹强度”。

```text
raw variable X
  -> center: X - E[X]
  -> scale: divide by sigma_X
  -> standardized Z_X
  -> Cov(Z_X, Z_Y) = corr(X,Y)
```

### 为什么相关系数一定在 [-1, 1]

标准化后，$\operatorname{Var}(Z_X)=\operatorname{Var}(Z_Y)=1$。
考虑它们的任意线性组合，方差必须非负。对任意实数 $t$：

$$
\operatorname{Var}(Z_X-tZ_Y)\ge0
$$

展开方差公式：

$$
\operatorname{Var}(Z_X) - 2t\operatorname{Cov}(Z_X,Z_Y) + t^2\operatorname{Var}(Z_Y) \ge 0
$$

代入已知值：

$$
1-2t\operatorname{Cov}(Z_X,Z_Y)+t^2 \ge 0
$$

记 $ ho=\operatorname{Cov}(Z_X,Z_Y)$。这是一个关于 $t$ 的二次多项式：

$$
t^2 - 2 ho t + 1 \ge 0
$$

既然它对所有 $t$ 都大于等于零，其判别式必须小于等于零：

$$
(-2 ho)^2 - 4 \le 0
$$

解得：

$$
 ho^2 \le 1
$$

也就是：

$$
-1 \le \operatorname{corr}(X,Y) \le 1
$$

注意 $ ho=0$ 只表示没有线性相关关系，并不等同于两个变量独立。独立一定意味着协方差为 0，但协方差为 0 推不出独立。

---

## 3 · 相关矩阵与半正定约束

当有 $n$ 个随机变量 $X_1,\ldots,X_n$ 时，可以把它们全部标准化为 $Z_i$。相关矩阵 $R$ 就是这些 $Z_i$ 的协方差矩阵：

$$
R_{ij} = \operatorname{corr}(X_i,X_j) = \operatorname{Cov}(Z_i,Z_j)
$$

相关矩阵必须满足三条严格的代数性质：

| 性质 | 来源 |
|---|---|
| 对称 | $\operatorname{corr}(X_i,X_j)=\operatorname{corr}(X_j,X_i)$ |
| 对角线为 1 | 每个变量与自身的相关系数是 1 |
| 半正定 (PSD) | 任意随机变量的线性组合方差非负 |

半正定（PSD）是多变量相关性最核心的约束。取任意实数权重 $a_1,\ldots,a_n$，构成一个新的随机变量：

$$
W = a_1Z_1 + \cdots + a_nZ_n
$$

方差必须非负：

$$
\operatorname{Var}(W) = \operatorname{Var}\left( \sum_{i=1}^n a_iZ_i \right) \ge 0
$$

展开双重求和：

$$
\operatorname{Var}(W) = \sum_{i=1}^n\sum_{j=1}^n a_i a_j \operatorname{Cov}(Z_i,Z_j) = a^\top R a
$$

这证明了，对任意向量 $a$，二次型都有：

$$
a^\top R a \ge 0
$$

这就是矩阵半正定的严格定义。

```text
choose weights a1,...,an
        |
        v
linear combination W = sum ai Zi
        |
        v
variance Var(W) cannot be negative
        |
        v
a^T R a >= 0 for every a
        |
        v
R is PSD
```

---

## 4 · 极值推论：等相关下界

给定 $n$ 个变量，如果它们两两之间的相关系数全都相等，设为 $ ho$，那么 $ ho$ 最小能是多少？

构造一个全 1 的权重向量 $a = (1, 1, \dots, 1)^\top$。相关矩阵 $R$ 必须满足半正定条件：

$$
a^\top R a = \sum_{i=1}^n \sum_{j=1}^n R_{ij} \ge 0
$$

矩阵 $R$ 中，对角线上有 $n$ 个 1，非对角线上有 $n(n-1)$ 个 $ ho$：

$$
a^\top R a = n + n(n-1) ho \ge 0
$$

化简得：

$$
n(n-1) ho \ge -n
$$

解出下界：

$$
 ho \ge -\frac{1}{n-1}
$$

这给出了等相关矩阵的严格约束。例如：
- $n=3$ 时，下界是 $-1/2$。
- $n=4$ 时，下界是 $-1/3$。

随着变量数量增加，下界逐渐趋向于 0。当 $n$ 很大时，不可能构造出一组两两呈现强负相关的随机变量。

如果问题是求“所有两两相关系数之和的最小值”，不论相关系数是否相等，都可以直接应用同一不等式：

$$
\operatorname{Var}(Z_1 + \cdots + Z_n) = n + 2 \sum_{1\le i<j\le n} \operatorname{corr}(X_i,X_j) \ge 0
$$

由此得到下界：

$$
\sum_{1\le i<j\le n} \operatorname{corr}(X_i,X_j) \ge -\frac{n}{2}
$$

---

## 5 · 三变量相关系数边界

如果已知变量之间的部分相关关系，可以用 PSD 约束推导未知的相关系数。已知 $X$ 与 $Y$ 的相关系数为 $ ho_{12}$，$Y$ 与 $Z$ 的相关系数为 $ ho_{23}$，如何限制 $X$ 与 $Z$ 的相关系数 $ ho_{13}$？

写出这三个变量的 $3 \times 3$ 相关矩阵：

$$
R = \begin{pmatrix} 1 &  ho_{12} &  ho_{13} \\  ho_{12} & 1 &  ho_{23} \\  ho_{13} &  ho_{23} & 1 \end{pmatrix}
$$

既然 $R$ 是 PSD，它的所有主子式都必须非负。特别是整个矩阵的行列式必须非负：

$$
\det(R) \ge 0
$$

展开行列式：

$$
1 + 2 ho_{12} ho_{23} ho_{13} -  ho_{12}^2 -  ho_{23}^2 -  ho_{13}^2 \ge 0
$$

将其整理为关于 $ ho_{13}$ 的二次不等式：

$$
 ho_{13}^2 - 2 ho_{12} ho_{23} ho_{13} + ( ho_{12}^2 +  ho_{23}^2 - 1) \le 0
$$

这是一个开口向上的抛物线，要在零点之间取值。解二次方程，得到 $ ho_{13}$ 的闭区间范围：

$$
 ho_{13} \in \left[  ho_{12} ho_{23} - \sqrt{(1- ho_{12}^2)(1- ho_{23}^2)},\  ho_{12} ho_{23} + \sqrt{(1- ho_{12}^2)(1- ho_{23}^2)} \right]
$$

几何上，相关系数可以看作随机向量在空间中的夹角余弦。已知两个夹角，第三个夹角自然会受到空间几何的三角不等式限制。

---

## 6 · Cholesky 分解与相关正态模拟

给定两个独立的标准正态变量 $U, V \overset{i.i.d.}{\sim} N(0,1)$，如何构造出相关系数为 $ ho$ 的二维正态变量 $(X,Y)$？

可以直接使用相关矩阵的 Cholesky 分解进行线性变换：

$$
\begin{pmatrix} X \\ Y \end{pmatrix}
=
\begin{pmatrix} 1 & 0 \\  ho & \sqrt{1- ho^2} \end{pmatrix}
\begin{pmatrix} U \\ V \end{pmatrix}
$$

展开形式为：

$$
X = U
$$

$$
Y =  ho U + \sqrt{1- ho^2} V
$$

检验均值、方差和协方差：

$$
\mathbb{E}[X] = \mathbb{E}[U] = 0
$$

$$
\mathbb{E}[Y] =  ho \mathbb{E}[U] + \sqrt{1- ho^2} \mathbb{E}[V] = 0
$$

$$
\operatorname{Var}(X) = \operatorname{Var}(U) = 1
$$

$$
\operatorname{Var}(Y) =  ho^2 \operatorname{Var}(U) + (1- ho^2) \operatorname{Var}(V) = 1
$$

$$
\operatorname{Cov}(X,Y) = \operatorname{Cov}(U,  ho U + \sqrt{1- ho^2} V) =  ho \operatorname{Var}(U) =  ho
$$

因为 $X, Y$ 的方差都为 1，它们的协方差等于相关系数：

$$
\operatorname{corr}(X,Y) =  ho
$$

这个变换把 $(U,V)$ 平面上独立的圆对称分布，线性拉伸成了具有特定倾斜方向的椭圆分布。由于使用的是线性算子，它保持了联合正态性（joint normality）。

---

## 7 · 二维正态的符号相关期望

利用上述构造，可以精确计算二维标准正态的符号乘积期望：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
$$

其中 $(X,Y)$ 的相关系数为 $ ho$。连续正态分布取值为 0 的概率为 0，所以符号乘积必定是 1 或 -1。

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = 1 \cdot P(\text{同号}) + (-1) \cdot P(\text{异号}) = P(\text{同号}) - P(\text{异号})
$$

利用全概率公式，将异号概率替换掉：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = 2P(\text{同号}) - 1
$$

根据二维正态分布关于原点的对称性，$X,Y$ 落在第一象限和第三象限的概率相等：

$$
P(X>0, Y>0) = P(X<0, Y<0)
$$

因此：

$$
P(\text{同号}) = 2P(X>0, Y>0)
$$

记 $p = P(X>0, Y>0)$，则期望化简为：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = 4p - 1
$$

### 在独立平面中计算概率

将 Cholesky 变换代入 $p$ 的计算：

$$
p = P(X>0, Y>0) = P\left(U>0,\  ho U + \sqrt{1- ho^2} V > 0\right)
$$

现在问题转移到了 $(U,V)$ 平面。$U,V$ 是独立标准正态，其联合密度函数为：

$$
f(u,v) = \frac{1}{2\pi} e^{-(u^2+v^2)/2}
$$

这个密度在几何上是完美的圆对称形式，只依赖于到原点的距离。对于任何过原点的扇形区域，其概率质量严格等于扇形角度占完整圆周的比例，也就是 $\theta / 2\pi$。

不等式系统定义了两个半平面：

1. $U > 0$：边界线为 $U=0$（即 $V$ 轴）。它保留了右半平面。
2. $ ho U + \sqrt{1- ho^2} V > 0$：边界线为 $V = -\frac{ ho}{\sqrt{1- ho^2}} U$。

令 $\alpha = \arcsin ho$。经过原点的边界线 $V = -\tan(\alpha) U$ 与 $U$ 轴正半轴的夹角正好是 $-\alpha$。而第一条边界线 $U=0$ 对应正向的 $\pi/2$。

这两个半平面的交集构成了一个扇形，其开角为：

$$
\frac{\pi}{2} + \alpha = \frac{\pi}{2} + \arcsin ho
$$

于是概率值为：

$$
p = \frac{\frac{\pi}{2} + \arcsin ho}{2\pi} = \frac{1}{4} + \frac{\arcsin ho}{2\pi}
$$

### 代回期望公式

将 $p$ 的表达式代回期望公式：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = 4\left(\frac{1}{4} + \frac{\arcsin ho}{2\pi}\right) - 1
$$

得到干净的最终结论：

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = \frac{2}{\pi} \arcsin ho
$$

这个结果重度依赖独立高斯分布的旋转不变性（圆对称），并不适用于相关系数同为 $ ho$ 的任意分布。

---

## 8 · 快速复习

```text
Covariance
= E[(X - E[X])(Y - E[Y])]

Correlation
= Cov(X, Y) / (sigma_X * sigma_Y)

Correlation Matrix R
= symmetric, diagonals are 1, and PSD (Positive Semidefinite)

PSD Meaning
= variance of any linear combination is non-negative
= a^T R a >= 0 for all vectors a

Equicorrelation Lower Bound
= rho >= -1 / (n - 1)

Cholesky for Bivariate Normal
X = U
Y = rho U + sqrt(1 - rho^2) V

Normal Sign Correlation Expectation
E[sgn(X)sgn(Y)] = (2/pi) arcsin(rho)
```

---

## 一手资料

- Zhou, *A Practical Guide to Quantitative Finance Interviews*
