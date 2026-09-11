# Quant 06 · Hypothesis Testing and Likelihood: Directions, Boundaries, and Bias-Variance

Course location: [[Quant06 High Dimensional Integral Dominated Convergence.en|05 High Dimensional Integral]] → This note → [[Quant11 Martingales Stopping Times Random Walks.en|07 Martingales]]

Standard procedure for parameter estimation and testing:

```text
1. p-value: Check the direction of the observation relative to the expected value under the null hypothesis to determine the test tail.
2. Likelihood function: Write L(theta). If the support does not depend on the parameter, take the derivative to find the stationary point. If it does, analyze monotonicity to find the maximum at the boundary.
3. Likelihood ratio: Simplify the ratio, isolate the sample-dependent parts to identify the sufficient statistic.
4. Method of Moments (MoM): Express population moments as functions of parameters, then substitute sample moments to solve.
5. Comparing estimators: Mean Squared Error (MSE) = Bias^2 + Variance.
```

---

## 1 · p-value and test direction

### 8 heads in 10 coin flips

Let $X$ be the number of heads in 10 independent fair coin flips. Null hypothesis $H_0:p=\frac12$, $\mathbb E[X]=5$. The observation $X=8$ is greater than the expectation, so we use a right-tailed test.

The p-value is the probability of observing a result at least as extreme under the null hypothesis:

$$
p=\mathbb P(X\ge 8)=\frac{\binom{10}{8}+\binom{10}{9}+\binom{10}{10}}{2^{10}}=\frac{56}{1024}\approx0.0547
$$

### Principles for determining test direction

The alternative hypothesis $H_1$ determines which tail to use:

- $H_1:\theta>\theta_0$: Right-tailed test, $p=\mathbb P(T\ge t_{\text{obs}}\mid H_0)$
- $H_1:\theta<\theta_0$: Left-tailed test, $p=\mathbb P(T\le t_{\text{obs}}\mid H_0)$
- $H_1:\theta\ne\theta_0$: Two-tailed test, $p=2\min\big(\mathbb P(T\ge t_{\text{obs}}),\mathbb P(T\le t_{\text{obs}})\big)$

The "multiply by two" in the two-tailed formula is exact when the distribution is symmetric around the mean.

### Right-tailed test with non-1/2 probability

Draw 5 cards with replacement from a 54-card deck (including 2 Jokers). We observe 3 Jokers. The null hypothesis is that the probability of drawing a Joker is $p_0=\frac{1}{27}$. $X\sim\mathrm{Binomial}(5,\frac1{27})$, $\mathbb E[X]=\frac{5}{27}\approx0.185$.

The observation $X=3$ is much larger than the expectation. The alternative hypothesis $H_1:p>\frac1{27}$ leads to a right-tailed test:

$$
p\text{-value}=\mathbb P(X\ge3)=\sum_{k=3}^5\binom5k\left( \frac1{27} \right)^k\left( \frac{26}{27} \right)^{5-k}\approx0.00048
$$

### Left-tailed and two-tailed tests

If we observed $X=2$ ($n=10, p_0=\frac12$), which is less than the expected 5, the alternative $H_1:p<\frac12$ requires a left-tailed test:

$$
p\text{-value}=\mathbb P(X\le2)=\frac{56}{1024}\approx0.0547
$$

If no direction is specified ($H_1:p\ne\frac12$), we use a two-tailed test:

$$
p\text{-value}=2\times\mathbb P(X\ge8)=2\times\frac{56}{1024}\approx0.1094
$$

---

## 2 · Likelihood function and Maximum Likelihood Estimation

### Exponential distribution parameterized by mean

Let $x_1,\ldots,x_n$ be i.i.d. samples from an exponential distribution with mean $\beta$, density $f(x\mid\beta)=\frac1\beta e^{-x/\beta}$. The support $[0,\infty)$ does not depend on $\beta$. The likelihood function is:

$$
L(\beta)=\prod_{i=1}^n\frac1\beta e^{-x_i/\beta}=\beta^{-n}e^{-\sum_i x_i/\beta}
$$

Log-likelihood:

$$
\ell(\beta)=-n\ln\beta-\frac{\sum_i x_i}{\beta}
$$

Taking the derivative and setting it to zero:

$$
\ell'(\beta)=-\frac n\beta+\frac{\sum_i x_i}{\beta^2}=0\quad\Longrightarrow\quad \hat\beta=\bar X
$$

The second derivative $\ell''(\bar X)<0$ confirms this is a maximum.

### MLE on the boundary: $\mathrm{Unif}[\theta,2\theta]$

Let $x_1,\ldots,x_n \overset{i.i.d.}{\sim} \mathrm{Unif}[\theta,2\theta]$. Find the MLE of $\theta$, its bias, and variance.

Likelihood function:

$$
L(\theta)=\theta^{-n}\mathbf 1\{\theta\le X_{(1)}\}\mathbf 1\{2\theta\ge X_{(n)}\}
$$

The valid domain is $\theta\in\left[ \frac{X_{(n)}}2,\,X_{(1)} \right]$. On this interval, $\theta^{-n}$ is strictly decreasing, so the likelihood is maximized at the left endpoint:

$$
\boxed{\hat\theta=\frac{X_{(n)}}2}
$$

To find the bias and variance, use $X_{(n)}=\theta(1+U_{(n)})$, where $U_{(n)}$ is the maximum of $n$ independent $\mathrm{Unif}[0,1]$ variables.

$$
\mathbb E[U_{(n)}]=\frac n{n+1},\qquad \mathrm{Var}(U_{(n)})=\frac n{(n+1)^2(n+2)}
$$

$$
\mathbb E[\hat\theta]=\frac12\mathbb E[X_{(n)}]=\frac{\theta}{2}\left( 1+\frac n{n+1} \right)=\theta\cdot\frac{2n+1}{2(n+1)}
$$

$$
\mathrm{Bias}(\hat\theta)=-\frac{\theta}{2(n+1)}
$$

$$
\mathrm{Var}(\hat\theta)=\frac14\mathrm{Var}(X_{(n)})=\frac{n\,\theta^2}{4(n+1)^2(n+2)}
$$

Unbiased correction: $\tilde\theta=\frac{2(n+1)}{2n+1}\hat\theta=\frac{(n+1)X_{(n)}}{2n+1}$.

---

## 3 · Likelihood ratio and sufficient statistics

Compare $N(0,\sigma^2)$ and $N(\mu,\sigma^2)$ with known variance. Null hypothesis $H_0:\mu=0$, alternative hypothesis $H_1:\mu>0$.

Likelihood ratio:

$$
T=\frac{L_1}{L_0}=\prod_{i=1}^n\frac{\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2} \right)}{\exp\left( -\frac{x_i^2}{2\sigma^2} \right)}
$$

Expanding and subtracting the exponents:

$$
-\frac{(x_i-\mu)^2}{2\sigma^2}+\frac{x_i^2}{2\sigma^2}=\frac{\mu x_i}{\sigma^2}-\frac{\mu^2}{2\sigma^2}
$$

$$
T=\exp\left( \frac{\mu}{\sigma^2}\sum_{i=1}^n x_i-\frac{n\mu^2}{2\sigma^2} \right)
$$

Since $\mu/\sigma^2>0$, $T$ is a strictly increasing function of $\sum_i x_i$ (or $\bar X$). The event "$T>c$" is equivalent to "$\bar X>c'$". The rejection region is entirely determined by $\bar X$, making it a sufficient statistic.

---

## 4 · Bias and variance comparison: MLE vs MoM

For the density $f(x\mid\theta)=\frac{3x^2}{\theta^3}$ ($0\le x\le\theta$), find the Method of Moments (MoM) estimator and MLE, then compare them.

### Method of Moments

First population moment:

$$
\mathbb E[X]=\int_0^\theta x\cdot\frac{3x^2}{\theta^3}\,dx=\frac{3\theta}4
$$

Equate sample mean to population moment:

$$
\bar X=\frac{3\theta}4\quad\Longrightarrow\quad \hat\theta_C=\frac43\bar X
$$

To compare variances, we need the variance of $X$. Second moment:

$$
\mathbb E[X^2]=\int_0^\theta x^2\cdot\frac{3x^2}{\theta^3}\,dx=\frac{3\theta^2}5
$$

$$
\mathrm{Var}(X)=\frac{3\theta^2}5-\left( \frac{3\theta}4 \right)^2=\frac{3\theta^2}5-\frac{9\theta^2}{16}=\frac{3\theta^2}{80}
$$

$\hat\theta_C$ is a linear transformation of the sample mean, so its variance scales by the squared coefficient:

$$
\mathrm{Var}(\hat\theta_C)=\left( \frac43 \right)^2\cdot\frac{\mathrm{Var}(X)}n=\frac{16}9\cdot\frac{3\theta^2}{80n}=\frac{\theta^2}{15n}
$$

### MLE

The likelihood function is strictly decreasing on the domain $\theta\ge X_{(n)}$:

$$
L(\theta)=\frac{3^n\prod x_i^2}{\theta^{3n}}\mathbf 1\{\theta\ge X_{(n)}\}
$$

The maximum occurs at the left endpoint: $\hat\theta_A=X_{(n)}$.

Distributions and moments:

$$
F(x)=\left( \frac x\theta \right)^3 \Longrightarrow f_{X_{(n)}}(t)=\frac{3n\,t^{3n-1}}{\theta^{3n}}
$$

$$
\mathbb E[X_{(n)}]=\frac{3n}{3n+1}\theta,\qquad \mathrm{Bias}(\hat\theta_A)=-\frac{\theta}{3n+1}\approx-\frac{\theta}{3n}
$$

$$
\mathbb E[X_{(n)}^2]=\frac{3n}{3n+2}\theta^2
$$

$$
\mathrm{Var}(\hat\theta_A)=\frac{3n}{3n+2}\theta^2-\left( \frac{3n}{3n+1} \right)^2\theta^2
$$

Let $m=3n$, finding a common denominator:

$$
\frac{m}{m+2}-\frac{m^2}{(m+1)^2}=\frac{m(m+1)^2-m^2(m+2)}{(m+2)(m+1)^2}=\frac{m}{(m+2)(m+1)^2}
$$

Substituting $m=3n$:

$$
\mathrm{Var}(\hat\theta_A)=\frac{3n\,\theta^2}{(3n+2)(3n+1)^2}\approx\frac{\theta^2}{9n^2}
$$

### Conclusion

| Estimator | Bias | Variance | MSE (large $n$) |
|---|---|---|---|
| MLE ($X_{(n)}$) | $\approx-\frac{\theta}{3n}$ | $\approx\frac{\theta^2}{9n^2}$ | $\approx\frac{2\theta^2}{9n^2}$ |
| MoM ($\frac43\bar X$) | $0$ | $\frac{\theta^2}{15n}$ | $\frac{\theta^2}{15n}$ |

While the MLE is biased, its variance is $O(1/n^2)$. In large samples, its mean squared error is significantly smaller than the unbiased MoM estimator, which has a variance of $O(1/n)$.

---

## 5 · Core checklist for estimation and testing

1. **Does the support depend on the parameter?** If it does, avoid derivatives. Analyze the monotonicity of the likelihood as a function of the parameter and find the maximum at the boundary. If it does not, follow the standard path: take the derivative of the log-likelihood, set it to zero, and confirm the maximum with the second derivative.
2. **Direction of p-value**: Compare the observation to the expected value under the null hypothesis to decide which tail to calculate. Always confirm the direction, equality signs ($\ge$ vs. $>$), and whether the test is one-tailed or two-tailed before writing the formula.
3. **Likelihood ratio and sufficient statistics**: After simplifying, is the likelihood ratio a monotonic function of a sample statistic (like the sample mean or sum)? If so, that statistic is sufficient, and it completely determines the rejection region.
4. **Standard MoM workflow**: Express population moments, substitute them with sample moments, and solve for the parameter. When comparing variances, compute the variance of a single observation first, then scale by the linear coefficients.
5. **Holistic comparison of estimators**: Relying solely on unbiasedness is incomplete. The proper metric is "Mean Squared Error = Bias^2 + Variance." A biased estimator with a sufficiently small variance often performs better overall, especially in large samples.

Core summary in one sentence:

> The shape of the likelihood function precedes its derivative. Look at monotonicity and boundaries when support depends on the parameter; only rely on derivatives when it does not. Regardless of the path, the ultimate metric for an estimator's quality is always the mean squared error, not merely the label of unbiasedness.
