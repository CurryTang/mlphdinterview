# Quant 6 · 高维积分：大数定律与控制收敛

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

这道题的核心不是硬算高维积分，而是完成两次视角转换：

1. 把单位立方体上的积分看成独立均匀随机变量的期望；
2. 用大数定律求出被积函数的几乎处处极限，再用控制收敛定理交换“极限”和“期望”。

---

## 1. 动态 3D 直觉：积分是平均高度

```high-dimensional-integral-demo
```

先切到 `n = 2 曲面`。此时：

$$
f_2(x_1,x_2)=\frac{x_1^2+x_2^2}{x_1+x_2}
$$

在单位正方形上均匀撒点，每个点竖起的高度是 $f_2(x_1,x_2)$。二重积分就是这张曲面的**平均高度**，因为底面的面积正好为 1。

再切到 `n → ∞ 云团`。不要尝试画 $n$ 个坐标轴，而是保留被积函数真正需要的三个统计量：

$$
\bar X_n=\frac1n\sum_{i=1}^n X_i,
\qquad
Q_n=\frac1n\sum_{i=1}^n X_i^2,
\qquad
R_n=\frac{Q_n}{\bar X_n}.
$$

点云中的一个点代表一组随机样本 $(X_1,\ldots,X_n)$，它的三个坐标是 $(\bar X_n,Q_n,R_n)$。拖动维度滑杆会看到：

$$
(\bar X_n,Q_n,R_n)
\longrightarrow
\left(\frac12,\frac13,\frac23\right).
$$

这正是大数定律的几何形状：维度越高，样本统计量越集中。图中的点云平均高度是固定伪随机样本给出的 Monte Carlo 估计，用来展示趋势，不参与严格证明。

### 一个立刻可见的界

对 $0\le x_i\le1$，有 $x_i^2\le x_i$，所以：

$$
0\le
\frac{\sum_{i=1}^n x_i^2}{\sum_{i=1}^n x_i}
\le1.
$$

还可以把它写成加权平均：

$$
\frac{\sum_i x_i^2}{\sum_i x_i}
=
\sum_i \frac{x_i}{\sum_jx_j}\,x_i.
$$

权重 $x_i/\sum_jx_j$ 非负且总和为 1。因此函数值一定落在 $[0,1]$ 内，而且较大的 $x_i$ 会得到更大的权重。这个 $[0,1]$ 界稍后就是控制函数的来源。

---

## 2. 把高维积分变成期望

令：

$$
X_1,X_2,\ldots\overset{i.i.d.}{\sim}\operatorname{Unif}[0,1].
$$

因为联合密度在 $[0,1]^n$ 上等于 1，所以：

$$
I_n
=
\mathbb E\left[
\frac{X_1^2+\cdots+X_n^2}{X_1+\cdots+X_n}
\right].
$$

把分子、分母同时除以 $n$：

$$
I_n
=\mathbb E[R_n],
\qquad
R_n=
\frac{\frac1n\sum_{i=1}^nX_i^2}
{\frac1n\sum_{i=1}^nX_i}
=\frac{Q_n}{\bar X_n}.
$$

注意：有限 $n$ 时不能把期望直接穿过比值：

$$
\mathbb E\left[\frac{Q_n}{\bar X_n}\right]
\ne
\frac{\mathbb E[Q_n]}{\mathbb E[\bar X_n]}.
$$

右边虽然恰好是 $2/3$，但它不是 $I_n$。我们只能证明当 $n\to\infty$ 时，这个比值本身趋近 $2/3$。

---

## 3. 第一步：大数定律给出逐点极限

均匀分布的前两阶矩是：

$$
\mathbb E[X_1]=\int_0^1x\,dx=\frac12,
\qquad
\mathbb E[X_1^2]=\int_0^1x^2\,dx=\frac13.
$$

强大数定律分别作用于 $X_i$ 和 $X_i^2$：

$$
\bar X_n=\frac1n\sum_{i=1}^nX_i
\xrightarrow{a.s.}\frac12,
$$

$$
Q_n=\frac1n\sum_{i=1}^nX_i^2
\xrightarrow{a.s.}\frac13.
$$

两个收敛事件的交集仍然是概率 1。又因为分母的极限 $1/2>0$，连续映射定理给出：

$$
R_n=\frac{Q_n}{\bar X_n}
\xrightarrow{a.s.}
\frac{1/3}{1/2}=\frac23.
$$

大数定律解决的是“每一条典型无限样本序列上，函数值走向哪里”。但题目问的是这些函数值的平均，也就是 $\mathbb E[R_n]$。逐点收敛本身通常不足以交换极限和期望，这一步需要控制收敛定理。

---

## 4. 什么是控制收敛定理

### 定理

设 $f_n$ 是同一个测度空间 $(\Omega,\mathcal F,\mu)$ 上的可测函数。若：

1. $f_n\to f$ 几乎处处；
2. 存在一个**与 $n$ 无关**的可积函数 $g$，使得对所有 $n$ 都有 $|f_n|\le g$ 几乎处处；

那么：

$$
\lim_{n\to\infty}\int_\Omega f_n\,d\mu
=
\int_\Omega f\,d\mu.
$$

也就是：

$$
\boxed{
\text{几乎处处收敛}
+
\text{统一的可积控制}
\Longrightarrow
\text{可以交换极限与积分}
}
$$

### “控制”到底控制什么

仅有 $f_n(\omega)\to f(\omega)$，不排除函数在越来越小的区域上长出越来越高的尖峰。尖峰对每个固定点最终会消失，但它携带的总面积可能不消失。

例如在 $[0,1]$ 上：

$$
f_n(x)=n\mathbf 1_{(0,1/n)}(x).
$$

对几乎所有 $x$，有 $f_n(x)\to0$；但是：

$$
\int_0^1 f_n(x)\,dx=1
$$

始终不趋近于 0。控制函数 $g$ 的作用，就是排除这种“越来越窄、越来越高，但面积不消失”的逃逸质量。

在概率空间中，如果所有 $f_n$ 都满足 $|f_n|\le C$，可以直接取常数控制函数 $g\equiv C$，因为：

$$
\int_\Omega C\,d\mathbb P=C<\infty.
$$

这也是有界收敛定理的常见用法。

---

## 5. 本题如何使用控制收敛

### 条件 1：几乎处处收敛

上一节已经由强大数定律得到：

$$
R_n\xrightarrow{a.s.}\frac23.
$$

### 条件 2：找到统一控制函数

因为 $0\le X_i\le1$，所以 $X_i^2\le X_i$。当分母非零时：

$$
0\le R_n=\frac{\sum_iX_i^2}{\sum_iX_i}\le1.
$$

分母为零只可能发生在 $X_1=\cdots=X_n=0$，这是概率为 0 的集合。为了让函数处处有定义，可以约定在这个集合上 $R_n=0$。

因此可取：

$$
g(\omega)\equiv1.
$$

它与 $n$ 无关，而且在概率空间上可积：

$$
\mathbb E[g]=1.
$$

### 交换极限与期望

控制收敛定理给出：

$$
\begin{aligned}
\lim_{n\to\infty}I_n
&=\lim_{n\to\infty}\mathbb E[R_n]\\
&=\mathbb E\left[\lim_{n\to\infty}R_n\right]\\
&=\mathbb E\left[\frac23\right]\\
&=\frac23.
\end{aligned}
$$

---

## 6. 严谨细节：积分空间的维数一直在变

控制收敛定理要求 $f_n$ 定义在**同一个空间**上，而原题的积分域分别是 $[0,1]^n$。严格写法是使用无限乘积概率空间：

$$
\Omega=[0,1]^{\mathbb N},
\qquad
\mathbb P=\lambda^{\otimes\mathbb N},
$$

其中 $\lambda$ 是 $[0,1]$ 上的 Lebesgue 概率测度。坐标映射：

$$
X_i(\omega)=\omega_i
$$

就是一列 i.i.d. 的 $\operatorname{Unif}[0,1]$ 随机变量。

在同一个 $\Omega$ 上定义：

$$
R_n(\omega)=
\frac{\omega_1^2+\cdots+\omega_n^2}
{\omega_1+\cdots+\omega_n}.
$$

$R_n$ 只依赖前 $n$ 个坐标，因此：

$$
\mathbb E_\mathbb P[R_n]
=
\int_{[0,1]^n}
\frac{\sum_{i=1}^n x_i^2}{\sum_{i=1}^n x_i}
\,dx_1\cdots dx_n
=I_n.
$$

现在所有 $R_n$ 都在同一个概率空间上，大数定律和控制收敛定理才可以被严格地使用。

面试中如果题目重点是概率方法，可以说“令 $X_1,X_2,\ldots$ 为同一概率空间上的 i.i.d. 均匀变量”，这句话已经隐含完成了统一空间的构造。

---

## 7. 一页式证明模板

遇到形如：

$$
\int_{[0,1]^n}
\Phi\left(\frac1n\sum_i h_1(x_i),\ldots,
\frac1n\sum_i h_k(x_i)\right)dx
$$

的高维积分，可以依次检查：

```text
高维积分
  ↓ 写成 i.i.d. 样本的期望
E[Φ(经验均值)]
  ↓ 大数定律
经验均值 → 总体均值（a.s.）
  ↓ 连续映射
被积函数 → 常数（a.s.）
  ↓ 找统一可积上界 / 验证一致可积性
控制收敛，交换 limit 与 expectation
```

本题压缩成四行就是：

$$
I_n=\mathbb E\left[\frac{\overline{X^2}_n}{\bar X_n}\right],
\qquad X_i\overset{i.i.d.}{\sim}U[0,1],
$$

$$
\bar X_n\to\frac12,
\qquad
\overline{X^2}_n\to\frac13
\quad a.s.,
$$

$$
\frac{\overline{X^2}_n}{\bar X_n}\to\frac23
\quad a.s.,
\qquad
0\le\frac{\overline{X^2}_n}{\bar X_n}\le1,
$$

$$
\therefore\quad
I_n\to\frac23
\qquad\text{by DCT}.
$$

---

## 8. 常见误区

| 误区 | 问题 | 正确处理 |
| --- | --- | --- |
| $\mathbb E[A/B]=\mathbb E[A]/\mathbb E[B]$ | 一般不成立 | 先证明 $A_n/B_n$ 本身收敛 |
| 逐点收敛后直接交换积分 | 可能有尖峰逃逸 | 找与 $n$ 无关的可积控制函数 |
| 直接在 $[0,1]^n$ 上套 DCT | 定义域随 $n$ 改变 | 统一到 $[0,1]^{\mathbb N}$ 概率空间 |
| 忽略分母为 0 | 函数在原点未定义 | 在零测集上任意定义，例如定义为 0 |
| 把 $g_n=R_n$ 当控制函数 | 控制函数不能随 $n$ 变化 | 本题取固定的 $g\equiv1$ |

```quiz
title: Dominated Convergence Check
question: 本题里，哪一项真正提供了控制收敛定理所需的“统一控制”？
answer: C
A. 分子和分母同时除以 n
B. 样本均值几乎处处收敛
C. 对所有 n 都有 0 ≤ R_n ≤ 1，因此可取 g ≡ 1
D. 积分区域的体积等于 1
explanation: 大数定律负责逐点（几乎处处）收敛；0 ≤ R_n ≤ 1 才提供与 n 无关且可积的控制函数。概率空间总质量为 1 则保证 g ≡ 1 可积。
```
