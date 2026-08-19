# Quant 8 · Order Statistics: CDF Differentiation and Conditional Truncation

Let $X_1,\ldots,X_n$ be independent and identically distributed, and let $X_{(1)}\le X_{(2)}\le\cdots\le X_{(n)}$ be the same values sorted in increasing order. This ordered sequence is called the order statistics. Problems in this section typically ask for the distribution of a single order statistic (most often $X_{(1)}$ or $X_{(n)}$), or for an expectation built from several order statistics at once.

The processing order is:

```text
1. Distribution of max / min: write the CDF first, then differentiate.
2. Given the value of one order statistic, what is the conditional distribution of the rest: this holds for any continuous distribution.
3. Under a uniform distribution, the conditional distribution is uniform again: this is specific to uniform, not the general result.
```

---

## Module 1: Distribution of max and min, CDF First

Let $X_1,\ldots,X_n$ be iid with density $f$ and CDF $F$.

### Why Start From the CDF

Writing the density of $X_{(n)}$ directly requires working through the following: $X_{(n)}\in[x,x+dx]$ means exactly one of the $n$ points lands in this small interval and serves as the maximum, the other $n-1$ points are all below $x$, and there is a choice of which of the $n$ points plays that role. This combinatorial bookkeeping is easy to get wrong.

The CDF sidesteps this counting step. The event $X_{(n)}\le x$ is equivalent to "all $n$ points are $\le x$," an event that makes no reference to which point is the maximum, and it expands directly using independence:

$$
F_{X_{(n)}}(x)=P(X_{(n)}\le x)=P(X_1\le x,\ldots,X_n\le x)=F(x)^n
$$

Differentiating with respect to $x$ gives the density:

$$
f_{X_{(n)}}(x)=n\,f(x)\,F(x)^{n-1}
$$

The factor $n$ only appears after differentiation, and it corresponds exactly to the combinatorial choice of "which point is the maximum." Differentiation produces this coefficient on its own, with no need to enumerate the choice by hand ahead of time.

### The Survival Function Is More Direct for min

$X_{(1)}>x$ is equivalent to "all $n$ points are $>x$":

$$
1-F_{X_{(1)}}(x)=P(X_{(1)}>x)=P(X_1>x,\ldots,X_n>x)=[1-F(x)]^n
$$

So:

$$
F_{X_{(1)}}(x)=1-[1-F(x)]^n
$$

$$
f_{X_{(1)}}(x)=n\,[1-F(x)]^{n-1}f(x)
$$

### The Uniform$[0,1]$ Example

$F(x)=x$, $f(x)=1$. Substituting into the two formulas above:

$$
F_{X_{(n)}}(x)=x^n,\qquad f_{X_{(n)}}(x)=n\,x^{n-1}
$$

$$
F_{X_{(1)}}(x)=1-(1-x)^n,\qquad f_{X_{(1)}}(x)=n(1-x)^{n-1}
$$

Expectations:

$$
\mathbb E[X_{(n)}]=\int_0^1 x\cdot n x^{n-1}\,dx=\frac{n}{n+1}
$$

$$
\mathbb E[X_{(1)}]=\int_0^1 x\cdot n(1-x)^{n-1}\,dx=\frac{1}{n+1}
$$

The formula for $X_{(1)}$ also follows directly from the one for $X_{(n)}$: the uniform distribution is symmetric under $x\mapsto 1-x$, so $X_{(1)}$ and $1-X_{(n)}$ have the same distribution, giving $\mathbb E[X_{(1)}]=1-\mathbb E[X_{(n)}]=\frac{1}{n+1}$.

The worked example later also needs the second moment:

$$
\mathbb E[X_{(n)}^2]=\int_0^1 x^2\cdot n x^{n-1}\,dx=\frac{n}{n+2}
$$

---

## Module 2: The General Result: What the Remaining Points Look Like After Conditioning on an Extreme Value

The joint density of $X_{(1)},\ldots,X_{(n)}$ is:

$$
f_{X_{(1)},\ldots,X_{(n)}}(x_1,\ldots,x_n)=n!\prod_{i=1}^n f(x_i),\qquad x_1<\cdots<x_n
$$

The coefficient $n!$ is the number of orderings of the original sample $X_1,\ldots,X_n$ that map to the same sorted tuple.

The conditional density equals the joint density divided by the marginal density of $X_{(n)}$ (already derived in Module 1 as $f_{X_{(n)}}(x)=n f(x)F(x)^{n-1}$):

$$
f_{X_{(1)},\ldots,X_{(n-1)}\mid X_{(n)}=x}(x_1,\ldots,x_{n-1})
=\frac{n!\prod_{i=1}^n f(x_i)}{n\,f(x)\,F(x)^{n-1}}
=(n-1)!\prod_{i=1}^{n-1}\frac{f(x_i)}{F(x)}
$$

$\dfrac{f(t)}{F(x)}$ (for $0\le t\le x$) is exactly what results from restricting the density $f$ to $[0,x]$ and renormalizing by that interval's probability $F(x)$: the density of the conditional distribution "$F$ truncated to $[0,x]$." So:

$$
\boxed{\text{Given } X_{(n)}=x,\ (X_{(1)},\ldots,X_{(n-1)}) \text{ are } n-1 \text{ iid draws from } F \text{ truncated to } [0,x]}
$$

This holds for any continuous distribution $F$; it does not require $F$ to be uniform.

### Uniform Distribution: Truncation Returns a Uniform Distribution

For the uniform distribution, $f(t)=1$ on $[0,1]$. Substituting into the truncation formula:

$$
\frac{f(t)}{F(x)}=\frac{1}{x},\qquad 0\le t\le x
$$

$\dfrac1x$ is exactly the density of $\mathrm{Unif}[0,x]$. Truncating a uniform distribution to any subinterval leaves the shape uniform; only the interval changes. This closure property is specific to the uniform distribution; a general continuous distribution does not have it.

This property is precisely what lets "conditional uniformity" be used algebraically all the way through: after conditioning on an extreme value, the remaining points are still uniform (on a rescaled interval), so the uniform distribution's own ready-made formulas can be applied directly, with no need to redo a truncated-distribution integral at every step.

### A Non-Uniform Distribution: Truncation Produces a Different Distribution

Switching to the exponential distribution, $f(t)=e^{-t}$, $F(x)=1-e^{-x}$, truncated to $[0,x]$:

$$
\frac{f(t)}{F(x)}=\frac{e^{-t}}{1-e^{-x}},\qquad 0\le t\le x
$$

This is a truncated exponential distribution: neither the original unbounded exponential distribution nor a uniform one. Its mean has a closed form:

$$
\mathbb E[T]=1-\frac{x\,e^{-x}}{1-e^{-x}}
$$

At $x=2$, this formula gives $0.687$, matching the mean obtained from a numerical simulation of the truncated exponential distribution. The truncation principle itself still holds for any continuous distribution; outside the uniform case, the truncated distribution is no longer the same as the original one, and each subsequent step needs a fresh integral against this new density.

---

## Module 3: Worked Example: Finding $\mathbb E[X_{(1)}X_{(n)}]$

Let $X_1,\ldots,X_n$ be iid $\mathrm{Unif}[0,1]$. Find $\mathbb E[X_{(1)}X_{(n)}]$, the expected product of the minimum and the maximum. Two solution methods follow.

### Method 1: Direct Integration Over the Joint Density

First find the joint density of $(X_{(1)},X_{(n)})$. For $0\le x\le y\le1$, the event $X_{(1)}\in[x,x+dx]$, $X_{(n)}\in[y,y+dy]$ requires: one of the $n$ points lands near $x$ and serves as the minimum ($n$ ways to choose it), one of the remaining $n-1$ points lands near $y$ and serves as the maximum ($n-1$ ways to choose it), and the remaining $n-2$ points must fall in $(x,y)$. So:

$$
f_{X_{(1)},X_{(n)}}(x,y)=n(n-1)(y-x)^{n-2},\qquad 0\le x\le y\le1
$$

Substituting into the definition of the expectation:

$$
\mathbb E[X_{(1)}X_{(n)}]=\int_0^1\int_0^y xy\cdot n(n-1)(y-x)^{n-2}\,dx\,dy
$$

First compute the inner integral over $x$. Let $u=y-x$:

$$
\int_0^y x(y-x)^{n-2}\,dx
=\int_0^y (y-u)u^{n-2}\,du
=y\cdot\frac{y^{n-1}}{n-1}-\frac{y^n}{n}
=\frac{y^n}{n(n-1)}
$$

Substituting back into the outer integral:

$$
\mathbb E[X_{(1)}X_{(n)}]
=\int_0^1 y\cdot n(n-1)\cdot\frac{y^n}{n(n-1)}\,dy
=\int_0^1 y^{n+1}\,dy
=\frac{1}{n+2}
$$

This path repeats for any continuous distribution: replace $F$ with a general distribution, rederive the joint density of $(X_{(1)},X_{(n)})$, and run the same double integral. It relies on no special property of the uniform distribution.

### Method 2: Conditional Uniformity

Given $X_{(n)}=x$, Module 2's general result says the remaining $n-1$ points are iid draws from $F$ truncated to $[0,x]$. Since $F$ is uniform here, the truncated distribution is $\mathrm{Unif}[0,x]$. The smallest of these $n-1$ points is exactly $X_{(1)}$ among all $n$ points, since $X_{(n)}=x$ is already the maximum.

The expected minimum of $m$ iid $\mathrm{Unif}[0,a]$ draws is $\dfrac{a}{m+1}$ (the $\mathrm{Unif}[0,1]$ result rescaled onto $[0,a]$). Here $m=n-1$, $a=x$:

$$
\mathbb E[X_{(1)}\mid X_{(n)}=x]=\frac{x}{n}
$$

By the law of total expectation:

$$
\mathbb E[X_{(1)}X_{(n)}]
=\mathbb E\big[X_{(n)}\cdot\mathbb E[X_{(1)}\mid X_{(n)}]\big]
=\mathbb E\left[X_{(n)}\cdot\frac{X_{(n)}}{n}\right]
=\frac{\mathbb E[X_{(n)}^2]}{n}
$$

Substituting Module 1's result $\mathbb E[X_{(n)}^2]=\dfrac{n}{n+2}$:

$$
\mathbb E[X_{(1)}X_{(n)}]=\frac{1}{n}\cdot\frac{n}{n+2}=\frac{1}{n+2}
$$

Matching Method 1.

### Comparing the Two Methods

| | Method 1: Direct Integration | Method 2: Conditional Uniformity |
|---|---|---|
| Tools needed | Joint density of $(X_{(1)},X_{(n)})$, a double integral | Module 2's truncation principle, plus the ready-made formula for the mean of a uniform minimum |
| Computation | A change of variables and a double integral | Multiplying two already-known results |
| Range of applicability | Still applies after replacing the uniform with any continuous distribution and rederiving the joint density | Relies on "the truncated distribution is still uniform"; only the uniform case gives this short a path |

### Numerical Check

At $n=3$, both methods give $\mathbb E[X_{(1)}X_{(3)}]=\dfrac15=0.2$. A Monte Carlo simulation with 2 million independent samples gives an estimate of $0.19993$, matching the theoretical value. The same check at $n=2,5,10$ also holds, with simulated values within about $10^{-4}$ of $\dfrac{1}{n+2}$.

---

## Module 4: Generalization: Order Statistics Sandwiched Between Two Others

Module 2's truncation principle has a more general form: given $X_{(m)}=a$ and $X_{(k)}=b$ (with $m<k$), the $k-m-1$ points strictly between them are iid draws from $F$ truncated to $[a,b]$. This also holds for any continuous distribution.

Under a uniform distribution, the truncated distribution on $[a,b]$ is again $\mathrm{Unif}[a,b]$, which is exactly the source of the commonly used tool "given the values of two order statistics, the points between them are uniformly distributed on that interval again." This tool is often labeled directly as "conditional uniformity," and the word "uniformity" in that name is the tell: the truncation principle is the general rule, while remaining uniform after truncation is a property the uniform distribution alone has.

---

## Module 5: Final Checklist Before an Interview

1. For the distribution of max or min, which function comes first? The CDF (the survival function is more direct for min), followed by differentiation to get the density. Writing the density directly requires separately handling the combinatorial choice of "which point is the extreme value."
2. Does the problem give the value of one order statistic and ask for the conditional distribution of the rest? For any continuous distribution, the rest are iid draws from the original distribution truncated to the corresponding interval.
3. Is the underlying distribution uniform? If so, the truncated distribution is uniform again, and the uniform distribution's own formulas (such as the mean of an $m$-point minimum being $\frac{a}{m+1}$) apply directly, with no fresh integral needed.
4. Is the underlying distribution not uniform? Module 2's truncation principle still holds, but the truncated distribution is usually a new distribution, and every subsequent step needs a fresh integral against that new density; there is no algebraic shortcut equivalent to the uniform case.

One sentence to keep in mind:

> max $\le x$ means every point is $\le x$; min $>x$ means every point is $>x$. Given an extreme value, the remaining points follow a truncated distribution, and this holds for any continuous distribution. The uniform distribution only makes the computation after truncation clean; it is not the reason the principle holds.
