# Quant 6 · High-Dimensional Integrals: The Law of Large Numbers and Dominated Convergence

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

The core of this problem is not evaluating a high-dimensional integral directly, but making two changes of perspective:

1. View the integral over the unit cube as an expectation of independent uniform random variables.
2. Use the law of large numbers to find the almost-sure limit of the integrand, then use the dominated convergence theorem to interchange the limit and expectation.

---

## 1. Dynamic 3D intuition: An integral is an average height

```high-dimensional-integral-demo
```

First switch to the `n = 2 surface`. Then:

$$
f_2(x_1,x_2)=\frac{x_1^2+x_2^2}{x_1+x_2}
$$

Scatter points uniformly over the unit square and assign each point the height $f_2(x_1,x_2)$. The double integral is the **average height** of this surface because the base has area 1.

Next switch to the `n → ∞ cloud`. Instead of trying to draw $n$ coordinate axes, retain the three statistics the integrand actually needs:

$$
\bar X_n=\frac1n\sum_{i=1}^n X_i,
\qquad
Q_n=\frac1n\sum_{i=1}^n X_i^2,
\qquad
R_n=\frac{Q_n}{\bar X_n}.
$$

Each point in the cloud represents one random sample $(X_1,\ldots,X_n)$, with coordinates $(\bar X_n,Q_n,R_n)$. Moving the dimension slider shows:

$$
(\bar X_n,Q_n,R_n)
\longrightarrow
\left(\frac12,\frac13,\frac23\right).
$$

This is the geometric form of the law of large numbers: sample statistics become more concentrated as the dimension grows. The cloud's average height is a Monte Carlo estimate from fixed pseudorandom samples. It illustrates the trend but is not part of the rigorous proof.

### An immediately visible bound

For $0\le x_i\le1$, we have $x_i^2\le x_i$, so:

$$
0\le
\frac{\sum_{i=1}^n x_i^2}{\sum_{i=1}^n x_i}
\le1.
$$

It can also be written as a weighted average:

$$
\frac{\sum_i x_i^2}{\sum_i x_i}
=
\sum_i \frac{x_i}{\sum_jx_j}\,x_i.
$$

The weights $x_i/\sum_jx_j$ are nonnegative and sum to 1. The function therefore lies in $[0,1]$, with larger $x_i$ receiving greater weight. This $[0,1]$ bound will later provide the dominating function.

---

## 2. Convert the high-dimensional integral into an expectation

Let:

$$
X_1,X_2,\ldots\overset{i.i.d.}{\sim}\operatorname{Unif}[0,1].
$$

Because the joint density equals 1 on $[0,1]^n$:

$$
I_n
=
\mathbb E\left[
\frac{X_1^2+\cdots+X_n^2}{X_1+\cdots+X_n}
\right].
$$

Divide both numerator and denominator by $n$:

$$
I_n
=\mathbb E[R_n],
\qquad
R_n=
\frac{\frac1n\sum_{i=1}^nX_i^2}
{\frac1n\sum_{i=1}^nX_i}
=\frac{Q_n}{\bar X_n}.
$$

For finite $n$, expectation cannot pass directly through a ratio:

$$
\mathbb E\left[\frac{Q_n}{\bar X_n}\right]
\ne
\frac{\mathbb E[Q_n]}{\mathbb E[\bar X_n]}.
$$

Although the right-hand side happens to equal $2/3$, it is not $I_n$. We can only prove that the ratio itself approaches $2/3$ as $n\to\infty$.

---

## 3. Step one: The law of large numbers gives the pointwise limit

The first two moments of the uniform distribution are:

$$
\mathbb E[X_1]=\int_0^1x\,dx=\frac12,
\qquad
\mathbb E[X_1^2]=\int_0^1x^2\,dx=\frac13.
$$

Apply the strong law of large numbers separately to $X_i$ and $X_i^2$:

$$
\bar X_n=\frac1n\sum_{i=1}^nX_i
\xrightarrow{a.s.}\frac12,
$$

$$
Q_n=\frac1n\sum_{i=1}^nX_i^2
\xrightarrow{a.s.}\frac13.
$$

The intersection of these two convergence events still has probability 1. Since the denominator's limit satisfies $1/2>0$, the continuous mapping theorem gives:

$$
R_n=\frac{Q_n}{\bar X_n}
\xrightarrow{a.s.}
\frac{1/3}{1/2}=\frac23.
$$

The law of large numbers determines where the function value goes along each typical infinite sample sequence. The problem, however, asks for the average of these values, $\mathbb E[R_n]$. Pointwise convergence alone is generally insufficient to interchange a limit and expectation. That step requires the dominated convergence theorem.

---

## 4. The dominated convergence theorem

### Theorem

Let $f_n$ be measurable functions on the same measure space $(\Omega,\mathcal F,\mu)$. Suppose:

1. $f_n\to f$ almost everywhere;
2. there is an integrable function $g$ **independent of $n$** such that $|f_n|\le g$ almost everywhere for every $n$.

Then:

$$
\lim_{n\to\infty}\int_\Omega f_n\,d\mu
=
\int_\Omega f\,d\mu.
$$

That is:

$$
\boxed{
\text{almost-everywhere convergence}
+
\text{uniform integrable domination}
\Longrightarrow
\text{the limit and integral may be interchanged}
}
$$

### What domination controls

Knowing only that $f_n(\omega)\to f(\omega)$ does not rule out increasingly tall spikes on increasingly small regions. The spike eventually disappears at each fixed point, but its total area may not disappear.

For example, on $[0,1]$:

$$
f_n(x)=n\mathbf 1_{(0,1/n)}(x).
$$

For almost every $x$, $f_n(x)\to0$, but:

$$
\int_0^1 f_n(x)\,dx=1
$$

never approaches 0. The dominating function $g$ rules out this escaping mass, which becomes narrower and taller while retaining nonvanishing area.

On a probability space, if every $f_n$ satisfies $|f_n|\le C$, we may simply choose the constant dominating function $g\equiv C$, because:

$$
\int_\Omega C\,d\mathbb P=C<\infty.
$$

This is also the common use of the bounded convergence theorem.

---

## 5. Applying dominated convergence here

### Condition 1: Almost-sure convergence

The previous section used the strong law of large numbers to establish:

$$
R_n\xrightarrow{a.s.}\frac23.
$$

### Condition 2: Find a uniform dominating function

Since $0\le X_i\le1$, we have $X_i^2\le X_i$. Whenever the denominator is nonzero:

$$
0\le R_n=\frac{\sum_iX_i^2}{\sum_iX_i}\le1.
$$

The denominator can be zero only when $X_1=\cdots=X_n=0$, an event of probability 0. To define the function everywhere, set $R_n=0$ on that event.

We may therefore choose:

$$
g(\omega)\equiv1.
$$

It is independent of $n$ and integrable on the probability space:

$$
\mathbb E[g]=1.
$$

### Interchange the limit and expectation

The dominated convergence theorem gives:

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

## 6. A rigorous detail: The dimension of the integration space changes

The dominated convergence theorem requires the $f_n$ to be defined on the **same space**, while the original integration domains are $[0,1]^n$. A rigorous formulation uses the infinite product probability space:

$$
\Omega=[0,1]^{\mathbb N},
\qquad
\mathbb P=\lambda^{\otimes\mathbb N},
$$

where $\lambda$ is Lebesgue probability measure on $[0,1]$. The coordinate maps:

$$
X_i(\omega)=\omega_i
$$

form an i.i.d. sequence of $\operatorname{Unif}[0,1]$ random variables.

Define on the same $\Omega$:

$$
R_n(\omega)=
\frac{\omega_1^2+\cdots+\omega_n^2}
{\omega_1+\cdots+\omega_n}.
$$

$R_n$ depends only on the first $n$ coordinates, so:

$$
\mathbb E_\mathbb P[R_n]
=
\int_{[0,1]^n}
\frac{\sum_{i=1}^n x_i^2}{\sum_{i=1}^n x_i}
\,dx_1\cdots dx_n
=I_n.
$$

Now every $R_n$ lives on the same probability space, so the law of large numbers and the dominated convergence theorem apply rigorously.

If a probability-method interview question is the focus, saying "let $X_1,X_2,\ldots$ be i.i.d. uniform variables on the same probability space" already implies this common-space construction.

---

## 7. One-page proof template

For a high-dimensional integral of the form:

$$
\int_{[0,1]^n}
\Phi\left(\frac1n\sum_i h_1(x_i),\ldots,
\frac1n\sum_i h_k(x_i)\right)dx,
$$

check the following in order:

```text
high-dimensional integral
  ↓ write as an expectation of i.i.d. samples
E[Φ(empirical means)]
  ↓ law of large numbers
empirical means → population means (a.s.)
  ↓ continuous mapping
integrand → a constant (a.s.)
  ↓ find a uniform integrable bound / verify uniform integrability
dominated convergence, interchange limit and expectation
```

For this problem, the proof compresses to four lines:

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

## 8. Common mistakes

| Mistake | Problem | Correct treatment |
| --- | --- | --- |
| $\mathbb E[A/B]=\mathbb E[A]/\mathbb E[B]$ | Not true in general | First prove that $A_n/B_n$ itself converges |
| Interchanging an integral immediately after pointwise convergence | Escaping spikes may exist | Find an integrable dominating function independent of $n$ |
| Applying DCT directly on $[0,1]^n$ | The domain changes with $n$ | Use the common probability space $[0,1]^{\mathbb N}$ |
| Ignoring a zero denominator | The function is undefined at the origin | Define it arbitrarily on the measure-zero set, for example as 0 |
| Using $g_n=R_n$ as the dominating function | The dominating function cannot vary with $n$ | Use the fixed function $g\equiv1$ here |

```quiz
title: Dominated Convergence Check
question: In this problem, which fact provides the "uniform domination" required by the dominated convergence theorem?
answer: C
A. Dividing both the numerator and denominator by n
B. The sample mean converges almost surely
C. For every n, 0 ≤ R_n ≤ 1, so we may choose g ≡ 1
D. The integration region has volume 1
explanation: The law of large numbers provides pointwise (almost-sure) convergence; 0 ≤ R_n ≤ 1 provides an integrable dominating function independent of n. The probability space having total mass 1 ensures that g ≡ 1 is integrable.
```
