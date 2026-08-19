# Quant 9 · Hypothesis Testing and Maximum Likelihood: Direction, Boundaries, and the Bias-Variance Tradeoff

This chapter covers hypothesis testing and parameter estimation together: p-values and test direction, maximum likelihood estimation (including the case where the support depends on the parameter, so calculus fails and a boundary argument is needed instead), likelihood ratios and sufficient statistics, the method of moments, and a closing comparison of MLE versus method-of-moments bias and variance. The processing order stays fixed throughout: write the likelihood (or the test statistic), decide whether a calculus argument or a boundary argument applies, then read off the estimator's bias and variance. Several problems reuse the distribution of order statistics; readers of the previous chapter (Order Statistics: CDF Differentiation and Conditional Truncation) will recognize the machinery immediately, and this chapter restates whatever is needed without assuming that derivation is still open on the page.

```text
1. p-value: check whether the observed value sits above or below its expected value under the null, then decide which side to integrate.
2. Likelihood: write L(theta), check whether the support depends on the parameter. If not, differentiate ln L and set it to zero. If it does, check whether the likelihood is monotonic in the parameter and read the maximum off a boundary.
3. Likelihood ratio: simplify the ratio and see whether the sample-dependent part collapses into a single statistic; monotonicity in that statistic identifies a sufficient statistic.
4. Method of moments: write the population moment as a function of the parameter, substitute the sample moment, solve for the parameter.
5. Comparing estimators: compute bias and variance, then compare using mean squared error = bias squared + variance.
```

---

## Module 1: p-Values and Test Direction

### 1. p-Value for 8 Heads in 10 Fair-Coin Flips

**Idea**: First fix the expectation under the null hypothesis. Whether the observed value sits above or below that expectation determines which tail the test integrates. Here the null-hypothesis expectation is 5, and the observed value of 8 sits well above it, so this calls for a right-tail test.

**Derivation**: Let $X$ be the number of heads in 10 independent fair coin flips, with null hypothesis $H_0:p=\frac12$, so $\mathbb E[X]=10\times\frac12=5$. Observing $X=8$, far above 5, sets the test direction to the right tail.

The p-value is defined as the probability, under the null hypothesis, of observing the current result or something more extreme; here "more extreme" means at least 8 heads:

$$
p=\mathbb P(X\ge 8)=\frac{\binom{10}{8}+\binom{10}{9}+\binom{10}{10}}{2^{10}}=\frac{45+10+1}{1024}=\frac{56}{1024}\approx0.0547
$$

**Takeaway**: Test direction is determined by which way the observed value deviates from its expected value under the null, not by the wording of the problem statement.

> The p-value is defined as the probability, under the null hypothesis, of observing the current result or something more extreme. Deciding the test direction (one-tailed or two-tailed, and which side to integrate) starts from whether the observed value sits above or below its expected value under the null. The most common mistakes are dropping the equality (using $>$ instead of $\ge$) and using a two-tailed test where a one-tailed test is called for.

### Choosing the Test Direction and Three Formulas

Let $T$ be the test statistic (in a discrete counting scenario, this is often just the count $X$ itself), with the distribution of $T$ under $H_0$ known and the observed value $t_{\text{obs}}$. Which tail to use is determined by the direction of the alternative hypothesis $H_1$:

- $H_1:\theta>\theta_0$ (the parameter is larger than the null claims): right-tailed test, $p=\mathbb P(T\ge t_{\text{obs}}\mid H_0)$
- $H_1:\theta<\theta_0$ (the parameter is smaller than the null claims): left-tailed test, $p=\mathbb P(T\le t_{\text{obs}}\mid H_0)$
- $H_1:\theta\ne\theta_0$ (the parameter differs from the null's claimed value, with no assumed direction): two-tailed test, $p=2\min\big(\mathbb P(T\ge t_{\text{obs}}\mid H_0),\ \mathbb P(T\le t_{\text{obs}}\mid H_0)\big)$ (capped at 1 if the result exceeds 1)

The choice among the three formulas is determined entirely by the direction of $H_1$, not by the observed value itself; the observed value is only used to determine whether it sits above or below the expected value under $H_0$, which then fixes which side to integrate. When a problem does not state a direction for $H_1$ and only asks whether the parameter equals a given value, treat it as two-tailed.

The "double the tail" step in the two-tailed formula is exact when the null distribution of $T$ is symmetric about its mean (for example $\mathrm{Binomial}(n,\frac12)$ or a normal distribution), since the two tail probabilities are then equal and doubling one gives their sum. When the distribution is not symmetric (for example the $p=\frac1{27}$ binomial in problem 2 below), doubling is a common approximation; the more rigorous definition sums the probabilities of all outcomes whose likelihood does not exceed the observed value's likelihood. Either convention is acceptable in an interview as long as it is stated explicitly.

### 2. Right-Tailed Test for the Proportion of Jokers in a 54-Card Deck

**Idea**: The null hypothesis here is not $p=\frac12$ but an arbitrary given proportion, which shows that the right-tailed formula applies to any binomial distribution and is not limited to the $p=\frac12$ special case.

**Derivation**: A 54-card deck (including 2 Jokers) is drawn from with replacement 5 times, and 3 Jokers are observed out of the 5 draws. The null hypothesis is that the deck holds exactly 2 Jokers, so the probability of drawing a Joker on a given draw is $p_0=\dfrac2{54}=\dfrac1{27}$. Let $X$ be the number of Jokers drawn in the 5 draws, so $X\sim\mathrm{Binomial}(5,\frac1{27})$, with $\mathbb E[X]=5\times\frac1{27}=\frac5{27}\approx0.185$.

Observing $X=3$, well above the expected $0.185$, sets the alternative hypothesis to $H_1:p>\frac1{27}$ (the true proportion of Jokers is higher), calling for a right-tailed test:

$$
p\text{-value}=\mathbb P(X\ge3)=\sum_{k=3}^5\binom5k\left(\frac1{27}\right)^k\left(\frac{26}{27}\right)^{5-k}=\frac{6891}{14348907}\approx0.00048
$$

**Takeaway**: The right-tailed formula $\mathbb P(T\ge t_{\text{obs}})$ does not require the null-hypothesis success probability to be $\frac12$; write down the correct distribution of $T$ under $H_0$ (here $\mathrm{Binomial}(5,\frac1{27})$) and the formula itself carries over unchanged.

### 3. Testing Whether a Coin's Heads Probability Is Significantly Low

**Idea**: With the alternative hypothesis switched to "the parameter is smaller than claimed," the right-tailed formula becomes a left-tailed formula. Reusing problem 1's exact setup ($n=10,p_0=\frac12$) with the observed value moved to the other side of the distribution makes the comparison direct.

**Derivation**: Let $X$ be the number of heads in 10 independent coin flips, with null hypothesis $H_0:p=\frac12$, so $\mathbb E[X]=5$. This time the observed value is $X=2$, well below 5, setting the alternative hypothesis to $H_1:p<\frac12$ and calling for a left-tailed test:

$$
p\text{-value}=\mathbb P(X\le2)=\frac{\binom{10}{0}+\binom{10}{1}+\binom{10}{2}}{2^{10}}=\frac{1+10+45}{1024}=\frac{56}{1024}\approx0.0547
$$

This value equals problem 1's right-tailed p-value exactly, because $\mathrm{Binomial}(10,\frac12)$ is symmetric about 5, so $X=2$ and $X=8$ are mirror-image positions and $\mathbb P(X\le2)=\mathbb P(X\ge8)$.

**Takeaway**: The left-tailed formula reverses the inequality direction of the right-tailed formula, replacing $\mathbb P(T\ge t_{\text{obs}})$ with $\mathbb P(T\le t_{\text{obs}})$. Under a symmetric distribution, two observed values equally far from the mean on opposite sides give the same one-tailed p-value.

### 4. Testing Whether a Coin Is Fair (Two-Tailed)

**Idea**: The data is identical to problem 1's ($n=10$, 8 heads observed), but the alternative hypothesis changes from "the heads probability is higher" to "the heads probability is not equal to $\frac12$," with no assumed direction. The same data under different hypotheses gives different p-values; the test direction is determined by the hypothesis, not the data.

**Derivation**: $X\sim\mathrm{Binomial}(10,\frac12)$, with $X=8$ observed. The alternative hypothesis is $H_1:p\ne\frac12$, calling for a two-tailed test. Since $\mathrm{Binomial}(10,\frac12)$ is symmetric about its mean, the two tail probabilities are equal, so problem 1's one-tailed probability is doubled:

$$
p\text{-value}=2\times\mathbb P(X\ge8)=2\times\frac{56}{1024}=\frac{112}{1024}\approx0.1094
$$

Equivalently, this can be written directly as the sum of both tail probabilities: $\mathbb P(X\ge8)+\mathbb P(X\le2)=\dfrac{56}{1024}+\dfrac{56}{1024}=\dfrac{112}{1024}$.

**Takeaway**:

> For the same observed data, the right-tailed test gives $p\approx0.0547$ and the two-tailed test gives $p\approx0.1094$, exactly double. The test direction is determined by how the alternative hypothesis is stated; before writing a test, confirm whether the problem asks "is it higher/lower" or "is it not equal."

---

## Module 2: The Likelihood Function and Maximum Likelihood Estimation

### 5. Maximum Likelihood Estimation for $\mathrm{Exp}(\lambda)$

**Idea**: The support of the exponential distribution, $[0,\infty)$, does not depend on $\lambda$, so this is the standard case: write the log-likelihood, differentiate and set it to zero to find the stationary point, then confirm a maximum with the second derivative.

**Derivation**: Let $x_1,\ldots,x_n$ be an independent sample from $\mathrm{Exp}(\lambda)$, with density $f(x\mid\lambda)=\lambda e^{-\lambda x}$. The likelihood function:

$$
L(\lambda)=\prod_{i=1}^n \lambda e^{-\lambda x_i}=\lambda^n e^{-\lambda\sum_i x_i}
$$

Taking logs:

$$
\ell(\lambda)=n\ln\lambda-\lambda\sum_i x_i
$$

Differentiating with respect to $\lambda$ and setting the result to zero:

$$
\ell'(\lambda)=\frac{n}{\lambda}-\sum_i x_i=0\quad\Longrightarrow\quad \hat\lambda=\frac{n}{\sum_i x_i}=\frac{1}{\bar X}
$$

The second derivative $\ell''(\lambda)=-\dfrac{n}{\lambda^2}<0$ confirms this is a maximum.

**Takeaway**: When the support does not depend on the parameter, maximum likelihood estimation follows the standard path: differentiate the log-likelihood, set it to zero, and confirm a maximum with the second derivative.

### 6. Maximum Likelihood Estimation for $\mathrm{Unif}[0,\theta]$: Bias, Variance, and an Unbiased Correction

**Idea**: The support $[0,\theta]$ here depends on $\theta$, so the standard differentiate-and-set-to-zero approach fails (differentiating the likelihood with respect to $\theta$ gives an expression that is never zero). The correct approach is to write the likelihood as a function of $\theta$ with its valid domain marked out, then observe its monotonicity on that domain directly, and read the maximum off the boundary.

**Derivation**: Let $x_1,\ldots,x_n$ be iid $\mathrm{Unif}[0,\theta]$, with density $f(x\mid\theta)=\frac1\theta\mathbf 1\{0\le x\le\theta\}$. The likelihood function:

$$
L(\theta)=\prod_{i=1}^n\frac1\theta\mathbf 1\{0\le x_i\le\theta\}=\theta^{-n}\mathbf 1\{\theta\ge X_{(n)}\}
$$

The indicator requires $\theta$ to be at least as large as every observed data point, that is, at least as large as the sample maximum $X_{(n)}$; this is exactly the valid domain $\theta\ge X_{(n)}$.

On that domain, $\theta^{-n}$ is a strictly decreasing function of $\theta$, so the likelihood is maximized at the smallest valid $\theta$, the left endpoint of the domain:

$$
\boxed{\hat\theta=X_{(n)}}
$$

The sample maximum itself is the maximum likelihood estimator. This step needs no differentiation; the monotonicity observation alone is enough.

Finding the bias and variance of $\hat\theta$ requires the distribution of $X_{(n)}$. A single observation has CDF $F(t)=t/\theta$, so following the standard CDF-differentiation approach:

$$
F_{X_{(n)}}(t)=\left(\frac t\theta\right)^n,\qquad f_{X_{(n)}}(t)=\frac{n\,t^{n-1}}{\theta^n},\qquad 0\le t\le\theta
$$

The first moment:

$$
\mathbb E[X_{(n)}]=\int_0^\theta t\cdot\frac{n\,t^{n-1}}{\theta^n}\,dt=\frac{n}{\theta^n}\cdot\frac{\theta^{n+1}}{n+1}=\frac{n}{n+1}\theta
$$

$$
\mathrm{Bias}(\hat\theta)=\mathbb E[\hat\theta]-\theta=\frac{n}{n+1}\theta-\theta=-\frac{\theta}{n+1}
$$

The bias is negative: the sample maximum can never exceed the true $\theta$, so $\hat\theta$ systematically underestimates $\theta$.

The second moment, from the same integral with an extra factor of $t$:

$$
\mathbb E[X_{(n)}^2]=\int_0^\theta t^2\cdot\frac{n\,t^{n-1}}{\theta^n}\,dt=\frac{n}{n+2}\theta^2
$$

The variance:

$$
\mathrm{Var}(\hat\theta)=\frac{n}{n+2}\theta^2-\left(\frac{n}{n+1}\theta\right)^2=\theta^2\left[\frac{n}{n+2}-\frac{n^2}{(n+1)^2}\right]
$$

Putting the bracket over a common denominator $(n+2)(n+1)^2$:

$$
\frac{n}{n+2}-\frac{n^2}{(n+1)^2}=\frac{n(n+1)^2-n^2(n+2)}{(n+2)(n+1)^2}=\frac{n\left[(n+1)^2-n(n+2)\right]}{(n+2)(n+1)^2}=\frac{n}{(n+2)(n+1)^2}
$$

since $(n+1)^2-n(n+2)=n^2+2n+1-n^2-2n=1$. So:

$$
\mathrm{Var}(\hat\theta)=\frac{n\,\theta^2}{(n+1)^2(n+2)}
$$

Finally, an unbiased estimator can be built by scaling $\hat\theta$ up in proportion to its bias:

$$
\tilde\theta=\frac{n+1}{n}X_{(n)},\qquad \mathbb E[\tilde\theta]=\frac{n+1}{n}\cdot\frac{n}{n+1}\theta=\theta
$$

**Takeaway**: A parameter-dependent support is the signal for a boundary argument. Once identified, do not attempt to differentiate; check directly whether the likelihood is increasing or decreasing as a function of the parameter, and the maximum sits at the corresponding endpoint of the valid domain.

> When the support of the likelihood depends on the parameter (as in $\mathrm{Unif}[0,\theta]$), the likelihood is typically maximized at a boundary of the parameter's domain rather than at a stationary point where the derivative vanishes. Differentiating in that case produces an expression that is never zero. The correct approach is to observe directly that the likelihood is a monotonic function of the parameter and read the maximum off the boundary.

---

## Module 3: Likelihood Ratios and Sufficient Statistics

### 7. The Likelihood Ratio for $N(0,1)$ versus $N(\mu,1)$ and Identifying the Sufficient Statistic

**Idea**: The likelihood ratio test statistic is the ratio of two likelihoods. Simplifying it into exponential form usually reveals that it is a monotonic function of some sample statistic, and that statistic is the sufficient statistic.

**Derivation**: Let $x_1,\ldots,x_n$ be iid normal with known variance 1, with null hypothesis $H_0:\mu=0$ and alternative $H_1:\mu=\mu_0>0$ (writing $\mu$ for $\mu_0$). The ratio of the two likelihoods:

$$
T=\frac{L_1}{L_0}=\prod_{i=1}^n\frac{\exp\left(-\dfrac{(x_i-\mu)^2}2\right)}{\exp\left(-\dfrac{x_i^2}2\right)}
$$

Expanding and subtracting the exponents of each term:

$$
-\frac{(x_i-\mu)^2}2+\frac{x_i^2}2=-\frac{x_i^2-2\mu x_i+\mu^2}2+\frac{x_i^2}2=\mu x_i-\frac{\mu^2}2
$$

Summing over $i$ and exponentiating:

$$
T=\exp\left(\sum_{i=1}^n\left[\mu x_i-\frac{\mu^2}2\right]\right)=\exp\left(\mu\sum_{i=1}^n x_i-\frac{n\mu^2}2\right)
$$

For $\mu>0$, $T$ is a strictly increasing function of $\sum_i x_i$, and the term $-n\mu^2/2$ in the exponent is a constant that does not depend on the data. So the event "$T>c$" is equivalent to "$\sum_i x_i>c''$", which is equivalent to "$\bar X>c'$" for some threshold $c'$. The rejection region of the likelihood ratio test is determined entirely by the sample mean $\bar X$: all the information the sample carries for distinguishing $\mu=0$ from $\mu>0$ is contained in the single statistic $\bar X$, which is sufficient for this testing problem.

A related, more elaborate variant is worth a brief mention. If the alternative instead states that exactly one of $N$ observations (unknown which one) has had its mean shifted by $A$, then since the identity of the shifted observation is unknown, the likelihood under $H_1$ is a mixture over the $N$ equally likely choices:

$$
L_1=\frac1N\sum_{j=1}^N\prod_{i=1}^N f\big(x_i-A\cdot\mathbf 1\{i=j\}\big)
$$

Factoring out the common product $\prod_i f(x_i)$ leaves a $1/N$-weighted sum of per-observation likelihood ratios $\dfrac{f(x_j-A)}{f(x_j)}$. This per-term ratio uses exactly the same algebra as the main derivation above; the only difference is an outer sum, by the law of total probability, over which observation might have been shifted:

$$
\frac{f(x_j-A)}{f(x_j)}=\exp\left[\frac{A(x_j-\mu)}{\sigma^2}-\frac{A^2}{2\sigma^2}\right]
$$

**Takeaway**: If a simplified likelihood ratio can be written as a monotonic function of some statistic, that statistic is sufficient. A mixture alternative does not change the per-observation likelihood-ratio algebra; it only adds an outer sum over the unknown category by the law of total probability.

> For the normal distribution (and most exponential-family distributions), expanding the squared terms in a likelihood ratio almost always lets the sample-dependent part collapse into a function of the sample mean or sample sum. If the likelihood ratio is a monotonic function of that statistic, the statistic is sufficient.

---

## Module 4: The Method of Moments

### 8. Method of Moments for $f(x\mid\theta)=\dfrac{2x}{\theta^2}$ (for $0\le x\le\theta$)

**Idea**: The method of moments follows a fixed path: compute the population moment as a function of the parameter, replace it with the corresponding sample moment, and solve for the parameter.

**Derivation**: First confirm this is a valid density:

$$
\int_0^\theta\frac{2x}{\theta^2}\,dx=\frac{2}{\theta^2}\cdot\frac{\theta^2}2=1
$$

The first moment:

$$
\mathbb E[X]=\int_0^\theta x\cdot\frac{2x}{\theta^2}\,dx=\frac{2}{\theta^2}\cdot\frac{\theta^3}3=\frac{2\theta}3
$$

Setting the sample mean equal to this population moment and solving for $\theta$:

$$
\bar X=\frac{2\theta}3\quad\Longrightarrow\quad \hat\theta_C=\frac32\bar X
$$

This estimator is unbiased:

$$
\mathbb E[\hat\theta_C]=\frac32\cdot\frac{2\theta}3=\theta
$$

Comparing variances later requires $\mathrm{Var}(X)$. The second moment:

$$
\mathbb E[X^2]=\int_0^\theta x^2\cdot\frac{2x}{\theta^2}\,dx=\frac{2}{\theta^2}\cdot\frac{\theta^4}4=\frac{\theta^2}2
$$

$$
\mathrm{Var}(X)=\frac{\theta^2}2-\left(\frac{2\theta}3\right)^2=\frac{\theta^2}2-\frac{4\theta^2}9=\frac{9\theta^2-8\theta^2}{18}=\frac{\theta^2}{18}
$$

Since $\hat\theta_C=\frac32\bar X$ is a linear rescaling of the sample mean, its variance scales accordingly:

$$
\mathrm{Var}(\hat\theta_C)=\left(\frac32\right)^2\cdot\frac{\mathrm{Var}(X)}n=\frac94\cdot\frac{\theta^2}{18n}=\frac{\theta^2}{8n}
$$

**Takeaway**: The method-of-moments procedure never changes: write the population moment, set it equal to the sample moment, solve for the parameter. Variance computations follow the same pattern: find the variance of a single observation first, then rescale by the linear coefficient relating the estimator to the sample mean.

---

## Module 5: Worked Example: Comparing MLE and Method-of-Moments Bias and Variance

### 9. Comparing MLE and Method of Moments for $f(x\mid\theta)=\dfrac{2x}{\theta^2}$

**Idea**: Module 4 already gives the method-of-moments estimator $\hat\theta_C$ for this density. Repeating problem 6's boundary argument for the same density gives the maximum likelihood estimator $\hat\theta_A$. Comparing bias, variance, and mean squared error for the two settles which is preferable.

**Derivation**:

Step 1, maximum likelihood estimation. The likelihood function:

$$
L(\theta)=\prod_{i=1}^n\frac{2x_i}{\theta^2}\mathbf 1\{\theta\ge x_i\}=\frac{2^n\prod_i x_i}{\theta^{2n}}\mathbf 1\{\theta\ge X_{(n)}\}
$$

As in problem 6, this is a strictly decreasing function of $\theta$ on the valid domain $\theta\ge X_{(n)}$, so the maximum is at the left endpoint:

$$
\hat\theta_A=X_{(n)}
$$

Step 2, the distribution of $X_{(n)}$ under this density. A single observation has CDF:

$$
F(x)=\int_0^x\frac{2t}{\theta^2}\,dt=\left(\frac x\theta\right)^2
$$

so:

$$
F_{X_{(n)}}(t)=\left(\frac t\theta\right)^{2n},\qquad f_{X_{(n)}}(t)=\frac{2n\,t^{2n-1}}{\theta^{2n}}
$$

Step 3, moments and bias:

$$
\mathbb E[X_{(n)}]=\int_0^\theta t\cdot\frac{2n\,t^{2n-1}}{\theta^{2n}}\,dt=\frac{2n}{2n+1}\theta,\qquad \mathrm{Bias}(\hat\theta_A)=-\frac{\theta}{2n+1}\approx-\frac{\theta}{2n}\text{ for large }n
$$

$$
\mathbb E[X_{(n)}^2]=\int_0^\theta t^2\cdot\frac{2n\,t^{2n-1}}{\theta^{2n}}\,dt=\frac{2n}{2n+2}\theta^2=\frac{n}{n+1}\theta^2
$$

$$
\mathrm{Var}(\hat\theta_A)=\frac{n}{n+1}\theta^2-\frac{4n^2}{(2n+1)^2}\theta^2=\frac{n(2n+1)^2-4n^2(n+1)}{(n+1)(2n+1)^2}\theta^2
$$

Expanding the numerator: $n(4n^2+4n+1)-4n^3-4n^2=4n^3+4n^2+n-4n^3-4n^2=n$. So:

$$
\mathrm{Var}(\hat\theta_A)=\frac{n\,\theta^2}{(n+1)(2n+1)^2}\approx\frac{\theta^2}{4n^2}\text{ for large }n
$$

Step 4, comparing the two estimators:

| Estimator | Bias | Variance | MSE (large-$n$ approximation) |
|---|---|---|---|
| $\hat\theta_A$ (MLE, $=X_{(n)}$) | $-\dfrac{\theta}{2n+1}\approx-\dfrac\theta{2n}$ | $\dfrac{n\theta^2}{(n+1)(2n+1)^2}\approx\dfrac{\theta^2}{4n^2}$ | $\approx\dfrac{\theta^2}{2n^2}$ |
| $\hat\theta_C$ (MOM, $=\frac32\bar X$) | $0$ | $\dfrac{\theta^2}{8n}$ | $\dfrac{\theta^2}{8n}$ |

The MLE's bias is of order $O(1/n)$ and vanishes as the sample size grows. Its variance is of order $O(1/n^2)$, an order of magnitude smaller than the method-of-moments estimator's $O(1/n)$ variance. Once $n$ is large enough, an $O(1/n^2)$ mean squared error is far smaller than an $O(1/n)$ one, so the MLE, despite being biased, wins decisively over the unbiased method-of-moments estimator in mean-squared-error terms.

**Takeaway**: Comparing two estimators cannot stop at whether each is biased or unbiased; the comparison has to reach mean squared error. A biased estimator with sufficiently small variance often performs better overall in large samples.

> A maximum likelihood estimator, even when biased, typically has bias of order $O(1/n)$, which vanishes as the sample size grows (asymptotic unbiasedness). Its variance is often of order $O(1/n^2)$, an order of magnitude smaller than the $O(1/n)$ variance typical of a method-of-moments estimator. In terms of mean squared error (MSE = bias squared + variance), the MLE, despite being biased, frequently outperforms an unbiased method-of-moments estimator with larger variance by a wide margin.

---

## Module 6: Final Checklist Before an Interview

1. Does the support depend on the parameter? If so, do not differentiate; check the monotonicity of the likelihood as a function of the parameter and read the maximum off a boundary. If not, follow the standard path: differentiate the log-likelihood, set it to zero, and confirm a maximum with the second derivative.
2. Is the p-value's direction correct? Check whether the observed value sits above or below its expected value under the null before deciding which side to integrate. Confirm whether the inequality includes equality ($\ge$ versus $>$) and whether the test is one-tailed or two-tailed before writing the formula.
3. After simplifying a likelihood ratio, is it a monotonic function of some sample statistic (sample mean, sample sum)? If so, that statistic is sufficient, and the rejection region is determined entirely by it.
4. Are all the method-of-moments steps complete: population moment expressed in terms of the parameter, substitution of the sample moment, solving for the parameter? If comparing variances, remember to find the variance of a single observation first, then rescale by the linear coefficient.
5. When comparing two estimators, does the conclusion stop at biased versus unbiased? The complete comparison standard is mean squared error = bias squared + variance; a biased estimator with sufficiently small variance often performs better overall.

One sentence to keep in mind:

> The shape of the likelihood function comes before its derivative: when the support depends on the parameter, check monotonicity first and read the boundary second; only when the support is parameter-free does differentiating for a stationary point apply. Whichever path is used, the final standard of comparison is mean squared error, not the single label of biased versus unbiased.
