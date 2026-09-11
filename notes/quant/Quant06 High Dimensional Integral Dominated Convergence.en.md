# Quant 05 · High Dimensional Integral: Law of Large Numbers and Dominated Convergence

Course location: [[Quant04 Correlation Matrix PSD.en|04 Correlation]] → This note → [[Quant09 Hypothesis Testing Maximum Likelihood.en|06 Hypothesis Testing]]

Consider the limit:

$$
I_n=\int_{[0,1]^n}
\frac{x_1^2+x_2^2+\cdots+x_n^2}
{x_1+x_2+\cdots+x_n}
\,dx_1\cdots dx_n,
\qquad
\lim_{n\to\infty} I_n\;=?
$$

The answer is:

$$
\boxed{\lim_{n\to\infty}I_n=\frac23}
$$

This requires two perspective shifts:

1. View the integral over the unit hypercube as the expectation of independent uniform random variables.
2. Use the Law of Large Numbers (LLN) to find the almost sure limit of the integrand, then apply the Dominated Convergence Theorem (DCT) to swap the limit and expectation.

---

## 1 · Dynamic 3D intuition: integral as average height

```high-dimensional-integral-demo
```

First, look at the `n = 2 surface`:

$$
f_2(x_1,x_2)=\frac{x_1^2+x_2^2}{x_1+x_2}
$$

If you uniformly sample points on the unit square, the height at each point is $f_2(x_1,x_2)$. The double integral is exactly the average height of this surface, since the base area is 1.

Next, switch to the `n → ∞ cloud`. Instead of trying to visualize $n$ axes, keep the three statistics that matter for the integrand:

$$
\bar X_n=\frac1n\sum_{i=1}^n X_i,
\qquad
Q_n=\frac1n\sum_{i=1}^n X_i^2,
\qquad
R_n=\frac{Q_n}{\bar X_n}
$$

A point in the cloud represents a random sample $(X_1,\ldots,X_n)$, with coordinates $(\bar X_n,Q_n,R_n)$. Drag the dimension slider to see:

$$
(\bar X_n,Q_n,R_n)
\longrightarrow
\left( \frac12,\frac13,\frac23 \right)
$$

This is the geometric manifestation of the LLN: as dimensions increase, sample statistics concentrate.

### An immediate bound

For $0\le x_i\le1$, we have $x_i^2\le x_i$, so:

$$
0\le
\frac{\sum_{i=1}^n x_i^2}{\sum_{i=1}^n x_i}
\le1
$$

This bound of $[0,1]$ will soon provide the dominating function.

---

## 2 · Converting the integral to expectation

Let:

$$
X_1,X_2,\ldots\overset{i.i.d.}{\sim}\operatorname{Unif}[0,1]
$$

Since the joint density on $[0,1]^n$ is 1:

$$
I_n
=
\mathbb E\left[ \frac{X_1^2+\cdots+X_n^2}{X_1+\cdots+X_n} \right]
$$

Divide both numerator and denominator by $n$:

$$
I_n
=\mathbb E[R_n],
\qquad
R_n=
\frac{\frac1n\sum_{i=1}^nX_i^2}
{\frac1n\sum_{i=1}^nX_i}
=\frac{Q_n}{\bar X_n}
$$

Note that for finite $n$, we cannot pull the expectation through the ratio:

$$
\mathbb E\left[ \frac{Q_n}{\bar X_n} \right]
\ne
\frac{\mathbb E[Q_n]}{\mathbb E[\bar X_n]}
$$

We must instead prove that the ratio itself converges to $2/3$ as $n\to\infty$.

---

## 3 · LLN provides the pointwise limit

The first two moments of the uniform distribution are:

$$
\mathbb E[X_1]=\int_0^1x\,dx=\frac12,
\qquad
\mathbb E[X_1^2]=\int_0^1x^2\,dx=\frac13
$$

The Strong Law of Large Numbers (SLLN) applies to $X_i$ and $X_i^2$:

$$
\bar X_n=\frac1n\sum_{i=1}^nX_i
\xrightarrow{a.s.}\frac12
$$

$$
Q_n=\frac1n\sum_{i=1}^nX_i^2
\xrightarrow{a.s.}\frac13
$$

Since the denominator limit $1/2>0$, the Continuous Mapping Theorem yields:

$$
R_n=\frac{Q_n}{\bar X_n}
\xrightarrow{a.s.}
\frac{1/3}{1/2}=\frac23
$$

The LLN determines where the function value goes on typical infinite sample sequences. However, pointwise convergence is not enough to exchange limit and expectation; we need the DCT.

---

## 4 · Dominated Convergence Theorem (DCT)

Let $f_n$ be measurable functions on a measure space $(\Omega,\mathcal F,\mu)$. If:

1. $f_n\to f$ almost everywhere;
2. There exists an integrable function $g$ **independent of $n$** such that $|f_n|\le g$ almost everywhere for all $n$.

Then:

$$
\lim_{n\to\infty}\int_\Omega f_n\,d\mu
=
\int_\Omega f\,d\mu
$$

The dominating function $g$ prevents escaping mass (e.g., spikes that get taller and narrower but retain area). In a probability space, if $|f_n|\le C$, we can just use the constant $g\equiv C$.

---

## 5 · Swapping limit and expectation

### Condition 1: a.s. convergence

We established:

$$
R_n\xrightarrow{a.s.}\frac23
$$

### Condition 2: dominating function

For non-zero denominators:

$$
0\le R_n=\frac{\sum_iX_i^2}{\sum_iX_i}\le1
$$

The denominator is zero only when $X_1=\cdots=X_n=0$, a probability zero event where we can define $R_n=0$. We choose:

$$
g(\omega)\equiv1
$$

which is independent of $n$ and integrable.

### The swap

By DCT:

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

## 6 · Rigorous detail: unifying the space

DCT requires the space to be fixed, but the integral domain $[0,1]^n$ changes with $n$. The rigorous approach uses the infinite product probability space $\Omega=[0,1]^{\mathbb N}$ and $\mathbb P=\lambda^{\otimes\mathbb N}$. The coordinate maps $X_i(\omega)=\omega_i$ are i.i.d. $\operatorname{Unif}[0,1]$.

On this shared $\Omega$:

$$
R_n(\omega)=
\frac{\omega_1^2+\cdots+\omega_n^2}
{\omega_1+\cdots+\omega_n}
$$

Since $R_n$ depends only on the first $n$ coordinates, $\mathbb E_\mathbb P[R_n] = I_n$. Stating "let $X_1, X_2, \ldots$ be i.i.d. on a probability space" implicitly handles this construction.

---

## 7 · Proof template

For high-dimensional limits of the form:

```text
High-dimensional integral
  ↓ Express as expectation of i.i.d.
E[Φ(empirical mean)]
  ↓ LLN
Empirical mean → population mean (a.s.)
  ↓ Continuous mapping
Integrand → constant (a.s.)
  ↓ Find dominating integrable bound
DCT: swap limit and expectation
```

Compressed proof:

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

## 8 · Common pitfalls

| Pitfall | Issue | Fix |
| --- | --- | --- |
| $\mathbb E[A/B]=\mathbb E[A]/\mathbb E[B]$ | Generally false | Show $A_n/B_n$ converges |
| Swap integral directly after pointwise limit | Escaping spikes | Find $n$-independent bound |
| Apply DCT on changing $[0,1]^n$ | Space varies with $n$ | Unify to $[0,1]^{\mathbb N}$ |
| Ignore zero denominator | Undefined value | Define arbitrarily on measure zero set |
| Dominating function varies with $n$ | Violates DCT | Use fixed $g\equiv 1$ |
