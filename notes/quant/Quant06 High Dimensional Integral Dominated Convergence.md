# Quant 05 · 高维积分：大数定律与控制收敛

课程位置：[[Quant04 Correlation Matrix PSD|04 协方差、正态与相关矩阵]] → 本篇 → [[Quant09 Hypothesis Testing Maximum Likelihood|06 假设检验与似然]]

考虑极限：

$$
I_n=\int_{[0,1]^n}
\frac{x_1^2+x_2^2+\cdots+x_n^2}
{x_1+x_2+\cdots+x_n}
\,dx_1\cdots dx_n,
\qquad
\lim_{n\to\infty} I_n\;=?
$$

答案是：

$$
\boxed{\lim_{n\to\infty}I_n=\frac23}
$$

这里需要完成两次视角转换：

1. 把单位立方体上的积分看成独立均匀随机变量的期望。
2. 用大数定律求出被积函数的几乎处处极限，再用控制收敛定理交换极限与期望。

---

## 1 · 动态 3D 直觉：积分是平均高度

```high-dimensional-integral-demo
```

先切到 `n = 2 曲面`。此时：

$$
f_2(x_1,x_2)=\frac{x_1^2+x_2^2}{x_1+x_2}
$$

在单位正方形上均匀撒点，每个点竖起的高度是 $f_2(x_1,x_2)$。二重积分就是这张曲面的平均高度，因为底面积为 1。

再切到 `n → ∞ 云团`。保留被积函数真正需要的三个统计量：

$$
\bar X_n=\frac1n\sum_{i=1}^n X_i,
\qquad
Q_n=\frac1n\sum_{i=1}^n X_i^2,
\qquad
R_n=\frac{Q_n}{\bar X_n}
$$

点云中的一个点代表一组随机样本 $(X_1,\ldots,X_n)$，它的三个坐标是 $(\bar X_n,Q_n,R_n)$。拖动维度滑杆会看到：

$$
(\bar X_n,Q_n,R_n)
\longrightarrow
\left( \frac12,\frac13,\frac23 \right)
$$

这正是大数定律的几何形状：维度越高，样本统计量越集中。图中的点云平均高度是固定伪随机样本给出的 Monte Carlo 估计，用来展示趋势。

### 显而易见的界

对 $0\le x_i\le1$，有 $x_i^2\le x_i$，所以：

$$
0\le
\frac{\sum_{i=1}^n x_i^2}{\sum_{i=1}^n x_i}
\le1
$$

还可以把它写成加权平均：

$$
\frac{\sum_i x_i^2}{\sum_i x_i}
=
\sum_i \frac{x_i}{\sum_jx_j}\,x_i
$$

权重 $x_i/\sum_jx_j$ 非负且总和为 1。因此函数值一定落在 $[0,1]$ 内，且较大的 $x_i$ 会得到更大的权重。这个 $[0,1]$ 界稍后就是控制函数的来源。

---

## 2 · 把高维积分变成期望

令：

$$
X_1,X_2,\ldots\overset{i.i.d.}{\sim}\operatorname{Unif}[0,1]
$$

联合密度在 $[0,1]^n$ 上等于 1，所以：

$$
I_n
=
\mathbb E\left[ \frac{X_1^2+\cdots+X_n^2}{X_1+\cdots+X_n} \right]
$$

把分子、分母同时除以 $n$：

$$
I_n
=\mathbb E[R_n],
\qquad
R_n=
\frac{\frac1n\sum_{i=1}^nX_i^2}
{\frac1n\sum_{i=1}^nX_i}
=\frac{Q_n}{\bar X_n}
$$

注意有限 $n$ 时不能直接把期望穿过比值：

$$
\mathbb E\left[ \frac{Q_n}{\bar X_n} \right]
\ne
\frac{\mathbb E[Q_n]}{\mathbb E[\bar X_n]}
$$

虽然右侧恰好是 $2/3$，但它不是 $I_n$。我们需要证明当 $n\to\infty$ 时，这个比值本身趋近 $2/3$。

---

## 3 · 大数定律给出逐点极限

均匀分布的前两阶矩是：

$$
\mathbb E[X_1]=\int_0^1x\,dx=\frac12,
\qquad
\mathbb E[X_1^2]=\int_0^1x^2\,dx=\frac13
$$

强大数定律分别作用于 $X_i$ 和 $X_i^2$：

$$
\bar X_n=\frac1n\sum_{i=1}^nX_i
\xrightarrow{a.s.}\frac12
$$

$$
Q_n=\frac1n\sum_{i=1}^nX_i^2
\xrightarrow{a.s.}\frac13
$$

两个收敛事件的交集仍是概率 1。分母极限 $1/2>0$，连续映射定理给出：

$$
R_n=\frac{Q_n}{\bar X_n}
\xrightarrow{a.s.}
\frac{1/3}{1/2}=\frac23
$$

大数定律解决了每一条典型无限样本序列上函数值的去向。但要求的是这些函数值的期望 $\mathbb E[R_n]$，逐点收敛不足以直接交换极限和期望，需要控制收敛定理。

---

## 4 · 控制收敛定理 (DCT)

设 $f_n$ 是测度空间 $(\Omega,\mathcal F,\mu)$ 上的可测函数。若：

1. $f_n\to f$ 几乎处处；
2. 存在一个与 $n$ 无关的可积函数 $g$，使得对所有 $n$ 都有 $|f_n|\le g$ 几乎处处。

那么：

$$
\lim_{n\to\infty}\int_\Omega f_n\,d\mu
=
\int_\Omega f\,d\mu
$$

即几乎处处收敛加上统一的可积控制，即可交换极限与积分。

仅有 $f_n(\omega)\to f(\omega)$ 不能排除函数在趋窄的区域上长出更高的尖峰，携带不消失的面积。控制函数 $g$ 的作用就是排除这种逃逸质量。在概率空间中，如果所有 $f_n$ 都满足 $|f_n|\le C$，可直接取常数控制函数 $g\equiv C$。

---

## 5 · 交换极限与期望

### 条件 1：几乎处处收敛

$$
R_n\xrightarrow{a.s.}\frac23
$$

### 条件 2：找到统一控制函数

因为 $0\le X_i\le1$，所以 $X_i^2\le X_i$。当分母非零时：

$$
0\le R_n=\frac{\sum_iX_i^2}{\sum_iX_i}\le1
$$

分母为零发生在 $X_1=\cdots=X_n=0$，这是概率为 0 的集合。可约定此集合上 $R_n=0$。因此可取：

$$
g(\omega)\equiv1
$$

它与 $n$ 无关，且在概率空间上可积。

### 完成计算

控制收敛定理给出：

$$
\begin{aligned}
\lim_{n\to\infty}I_n
&=\lim_{n\to\infty}\mathbb E[R_n]\\
&=\mathbb E\left[ \lim_{n\to\infty}R_n \right]\\
&=\mathbb E\left[ \frac23 \right]\\
&=\frac23
\end{aligned}
$$

---

## 6 · 严谨细节：统一积分空间

DCT 要求 $f_n$ 定义在同一个空间上。原题积分域是 $[0,1]^n$，严格做法是使用无限乘积概率空间 $\Omega=[0,1]^{\mathbb N}$ 和 $\mathbb P=\lambda^{\otimes\mathbb N}$。坐标映射 $X_i(\omega)=\omega_i$ 是一列 i.i.d. 的 $\operatorname{Unif}[0,1]$。在同一个 $\Omega$ 上定义：

$$
R_n(\omega)=
\frac{\omega_1^2+\cdots+\omega_n^2}
{\omega_1+\cdots+\omega_n}
$$

$R_n$ 只依赖前 $n$ 个坐标：

$$
\mathbb E_\mathbb P[R_n]
=
\int_{[0,1]^n}
\frac{\sum_{i=1}^n x_i^2}{\sum_{i=1}^n x_i}
\,dx_1\cdots dx_n
=I_n
$$

这使得大数定律和 DCT 可以严格使用。引入 i.i.d. 均匀变量已隐含了此构造。

---

## 7 · 证明模板

遇到高维积分求极限：

```text
高维积分
  ↓ 写成 i.i.d. 样本的期望
E[Φ(经验均值)]
  ↓ 大数定律
经验均值 → 总体均值 (a.s.)
  ↓ 连续映射
被积函数 → 常数 (a.s.)
  ↓ 统一可积上界
控制收敛，交换 limit 与 expectation
```

压缩证明：

$$
I_n=\mathbb E\left[ \frac{\overline{X^2}_n}{\bar X_n} \right],
\qquad X_i\overset{i.i.d.}{\sim}U[0,1]
$$

$$
\bar X_n\to\frac12,
\qquad
\overline{X^2}_n\to\frac13
\quad a.s.
$$

$$
\frac{\overline{X^2}_n}{\bar X_n}\to\frac23
\quad a.s.,
\qquad
0\le\frac{\overline{X^2}_n}{\bar X_n}\le1
$$

$$
\therefore\quad
I_n\to\frac23
\qquad\text{by DCT}
$$

---

## 8 · 常见误区

| 误区 | 问题 | 正确处理 |
| --- | --- | --- |
| $\mathbb E[A/B]=\mathbb E[A]/\mathbb E[B]$ | 仅当相互独立时成立 | 证明 $A_n/B_n$ 整体收敛 |
| 逐点收敛后直接交换积分 | 无法排除尖峰逃逸 | 找与 $n$ 无关的可积控制函数 |
| 在随 $n$ 变化的 $[0,1]^n$ 上套 DCT | 空间不统一 | 统一到 $[0,1]^{\mathbb N}$ |
| 忽略分母为 0 | 存在未定义点 | 在零测集上任意定义 |
| 变动的控制函数 | 不符合 DCT 条件 | 取固定的 $g\equiv 1$ |
