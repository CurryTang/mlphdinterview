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

## Module 5: Warm-Up

### 1. Expected Max and Min of Three Uniforms

**Approach**: The max follows the same route as Module 1 (CDF, then differentiate). The min only needs an expectation, not a density, so the tail-integral formula for nonnegative random variables skips a step.

**Derivation**: Let $X_1,X_2,X_3$ be iid $\mathrm{Unif}[0,1]$.

The max has CDF $F_M(x)=x^3$ and density $f_M(x)=3x^2$:

$$
\mathbb E[M]=\int_0^1 x\cdot 3x^2\,dx=\frac34
$$

For the min, skip the density entirely and use $\mathbb E[T]=\int_0^\infty P(T>t)\,dt$, valid whenever $T\ge0$. Since $P(m>t)=P(X_1>t,X_2>t,X_3>t)=(1-t)^3$:

$$
\mathbb E[m]=\int_0^1(1-t)^3\,dt=\frac14
$$

Both values match Module 1's general formulas $\mathbb E[X_{(n)}]=\frac{n}{n+1}$ and $\mathbb E[X_{(1)}]=\frac{1}{n+1}$ at $n=3$.

**Takeaway**:

> For a nonnegative random variable, $\mathbb E[T]=\int_0^\infty P(T>t)\,dt$ always holds. When only the expectation of a minimum is needed, this route is usually faster than deriving the density first, since the survival function $P(\min>t)$ is a direct product of $P(\text{every point}>t)$.

### 2. The Min/Max Asymmetry for Exponentials

**Approach**: The survival function of the min is still a product of independent exponential survival functions, so its shape does not change. The max requires expanding $(1-e^{-\lambda x})^n$, which is no longer a single exponential term. Working with the spacings between order statistics avoids that expansion.

**Derivation**: Let $X_1,\ldots,X_n$ be iid $\mathrm{Exp}(\lambda)$.

For the min: $P(m>x)=\prod_{i=1}^n e^{-\lambda x}=e^{-n\lambda x}$, so $m\sim\mathrm{Exp}(n\lambda)$ and $\mathbb E[m]=\dfrac{1}{n\lambda}$. The distribution keeps its exponential shape; only the rate changes to $n\lambda$.

For the max: let $X_{(1)}<\cdots<X_{(n)}$ be the order statistics and $D_k=X_{(k)}-X_{(k-1)}$ (with $X_{(0)}=0$). Memorylessness of the exponential distribution says that, given $X_{(k-1)}$, the points among the remaining $n-k+1$ that exceed it, shifted down by $X_{(k-1)}$, are again iid $\mathrm{Exp}(\lambda)$. Their minimum is $D_k$, distributed $\mathrm{Exp}((n-k+1)\lambda)$, and $D_1,\ldots,D_n$ are mutually independent. Then $X_{(n)}=\sum_{k=1}^n D_k$:

$$
\boxed{\mathbb E[X_{(n)}]=\sum_{k=1}^n\frac{1}{(n-k+1)\lambda}=\frac1\lambda\sum_{j=1}^n\frac1j=\frac{H_n}{\lambda}}
$$

where $H_n=\sum_{j=1}^n\frac1j$ is the $n$-th harmonic number. At $n=3,\lambda=1$: $H_3=1+\frac12+\frac13=\frac{11}{6}\approx1.833$, matching a Monte Carlo estimate of $1.833$; $\mathbb E[m]=\frac13\approx0.333$ matches the simulated $0.333$ as well.

**Takeaway**:

> The min of $n$ independent exponential random variables is exponential again, with rates adding. The max is not exponential, and $\mathbb E[\max]$ involves the harmonic number $H_n$. This asymmetry is a common interview follow-up.

### 3. $\mathbb E[X_{(k)}]$: Density Integration versus Spacing Exchangeability

**Approach**: Method 1 extends the density-based approach of Modules 1 and 2 directly. Method 2 never uses the explicit form of that density, only the exchangeability of the spacings and the constraint that they sum to a constant.

**Derivation**: Let $X_1,\ldots,X_n$ be iid $\mathrm{Unif}[0,1]$.

Method 1 (density):

$$
f_{X_{(k)}}(x)=\frac{n!}{(k-1)!(n-k)!}x^{k-1}(1-x)^{n-k}
$$

This is the density of $\mathrm{Beta}(k,n-k+1)$, with mean $\dfrac{k}{k+(n-k+1)}=\dfrac{k}{n+1}$.

Method 2 (spacings and exchangeability): split $[0,1]$ using the $n$ points into $n+1$ spacings $Y_1,\ldots,Y_{n+1}$ (with $Y_1=X_{(1)}$, $Y_i=X_{(i)}-X_{(i-1)}$, $Y_{n+1}=1-X_{(n)}$). Their joint density on the simplex $\{y_i\ge0,\sum y_i=1\}$ is the constant $n!$, a symmetric function of the coordinates, so the spacings are exchangeable: they share a common marginal distribution, and its exact form is not needed.

Take expectations of both sides of $\sum_{i=1}^{n+1}Y_i=1$, and use exchangeability to set all the $\mathbb E[Y_i]$ equal:

$$
(n+1)\,\mathbb E[Y_1]=1\ \Longrightarrow\ \mathbb E[Y_i]=\frac1{n+1},\quad i=1,\ldots,n+1
$$

Since $X_{(k)}=Y_1+\cdots+Y_k$, linearity gives:

$$
\boxed{\mathbb E[X_{(k)}]=\frac{k}{n+1}}
$$

Both methods agree.

**Takeaway**:

> The expectation of an order statistic often needs only two facts: the spacings are exchangeable, and they sum to a constant. The actual distribution of each spacing is not required.

### 4. $\mathbb E|X-Y|$ for Two Uniforms

**Approach**: $|X-Y|$ is exactly the range of two points, the max minus the min, so plugging $n=2$ into Module 1's formulas gives the answer directly, without setting up a fresh double integral over the absolute value.

**Derivation**: Let $X,Y$ be iid $\mathrm{Unif}[0,1]$. Since $|X-Y|=X_{(2)}-X_{(1)}$, substituting $n=2$:

$$
\mathbb E|X-Y|=\mathbb E[X_{(2)}]-\mathbb E[X_{(1)}]=\frac23-\frac13=\frac13
$$

As a cross-check, integrate the absolute value directly:

$$
\mathbb E|X-Y|=2\int_0^1\int_0^x(x-y)\,dy\,dx=2\int_0^1\frac{x^2}{2}\,dx=\int_0^1x^2\,dx=\frac13
$$

Both routes agree.

**Takeaway**: The range equals the max minus the min, so many "expected absolute difference" questions reduce directly to a known order-statistics formula instead of requiring a new double integral each time.

### 5. Probability At Least One Exceeds 0.9

**Approach**: Computing "at least one" directly involves inclusion-exclusion. Its complement, "all of them stay at or below 0.9," multiplies out immediately by independence.

**Derivation**: Let 5 points be iid $\mathrm{Unif}[0,1]$.

$$
P(\text{all}\le0.9)=0.9^5=0.59049
$$

$$
P(\text{at least one}>0.9)=1-0.9^5=0.40951
$$

**Takeaway**: "At least one exceeds a threshold" events are handled most directly through the complement plus independence; this is the same structure as $P(\max\le x)=F(x)^n$ from Module 1.

---

## Module 6: Spacings and Exchangeability

### 6. Expected Largest Segment from Two Cut Points

**Approach**: $\max_i Y_i$ is a union of three events rather than a single random variable, so its density is awkward to write directly. Expanding the survival function $P(\max>t)$ by inclusion-exclusion, then applying the same tail-integral formula used in Problem 1, is more direct.

**Derivation**: Let $U_1,U_2$ be iid $\mathrm{Unif}[0,1]$ cut points splitting $[0,1]$ into three segments $Y_1,Y_2,Y_3$.

Survival function of a single segment: $Y_1$ (from 0 to the smaller cut point) exceeds $t$ exactly when both cut points exceed $t$:

$$
P(Y_1>t)=P(U_1>t,U_2>t)=(1-t)^2
$$

Problem 3 already established that the three segments are exchangeable, so $P(Y_i>t)=(1-t)^2$ for $i=1,2,3$ as well, with no separate computation needed for $Y_2,Y_3$.

Two segments exceeding $t$ simultaneously: taking $Y_1,Y_2$ as an example, the condition is $U_{(1)}>t$ and $U_{(2)}-U_{(1)}>t$. Shifting $V_1=U_{(1)}-t$, $V_2=U_{(2)}-2t$ turns the two constraints into $0<V_1<V_2$, and $U_{(2)}\le1$ gives $V_2<1-2t$: this is exactly two points' order statistics confined to an interval of length $1-2t$, with probability $(1-2t)_+^2$. Exchangeability guarantees the same result for any pair of segments:

$$
P(Y_i>t,Y_j>t)=(1-2t)_+^2,\qquad i\ne j
$$

All three segments exceeding $t$: the same shift argument, with one more constraint $1-U_{(2)}>t$, confines the two points to an interval of length $1-3t$:

$$
P(Y_1>t,Y_2>t,Y_3>t)=(1-3t)_+^2
$$

Inclusion-exclusion gives the survival function of the max:

$$
\boxed{P\big(\max_iY_i>t\big)=3(1-t)^2-3(1-2t)_+^2+(1-3t)_+^2}
$$

Tail-integrating over $t$ (the three terms switch off in pieces, at $t=1/2$ and $t=1/3$):

$$
\mathbb E[\max_iY_i]=3\int_0^1(1-t)^2\,dt-3\int_0^{1/2}(1-2t)^2\,dt+\int_0^{1/3}(1-3t)^2\,dt=1-\frac12+\frac19=\frac{11}{18}
$$

Keeping only the last term gives the expected minimum segment: $\mathbb E[\min_iY_i]=\int_0^{1/3}(1-3t)^2dt=\dfrac19$, matching Problem 9's formula $1/(n+1)^2$ at $n=2$. The three expectations should satisfy:

| Segment | Expectation |
|---|---|
| Largest | $11/18$ |
| Middle | $5/18$ |
| Smallest | $1/9=2/18$ |
| Total | $1$ |

The middle segment's expectation does not need its own derivation; it follows from $1-11/18-1/9=5/18$. Monte Carlo estimates: largest $0.6110$, smallest $0.1111$, middle $0.2778$, all matching the theoretical values.

**Takeaway**:

> After computing the expectations for a set of order statistics or spacings, adding them up and checking against the known total (here, 1) is the standard way to catch computation errors.

### 7. Probability Three Segments Form a Triangle

**Approach**: Whether three segments form a triangle depends only on whether the longest one exceeds half the total length. Substituting this threshold into Problem 6's inclusion-exclusion formula, the two intersection terms vanish automatically at $t=1/2$, because two segments cannot simultaneously exceed half the total when the total is fixed.

**Derivation**: Three segments $a,b,c$ with $a+b+c=1$ form a triangle exactly when each side is less than the sum of the other two, i.e. less than $1-\text{that side}$, i.e. less than $1/2$. They fail to form a triangle exactly when some segment is $\ge1/2$.

Substituting $t=1/2$ into Problem 6's formula: $(1-2t)_+=(1-1)_+=0$ and $(1-3t)_+=(1-\tfrac32)_+=0$, so both intersection terms drop out. The reason: two segments each exceeding $1/2$ would already sum to more than 1, while the three segments together sum to exactly 1, a contradiction. So the events $\{Y_i>1/2\}$ for $i=1,2,3$ are pairwise mutually exclusive.

> When a constraint's sum is fixed and a single event requires exceeding half of that total, no two such events can occur together; the intersection terms in an inclusion-exclusion expansion vanish automatically, leaving a single sum.

Mutual exclusivity means the probability of the union is just the sum of the individual probabilities:

$$
P(\text{no triangle})=\sum_{i=1}^3P(Y_i>1/2)=3\cdot\left(\frac12\right)^2=\frac34,\qquad P(\text{triangle})=\frac14
$$

A Monte Carlo simulation gives $0.2501$, matching $1/4$.

Generalizing to $n-1$ cut points and $n$ segments: the single-segment survival function becomes $P(Y_i>1/2)=(1/2)^{n-1}$ (the exponent 2 in Problem 6's $(1-t)^2$ was the number of cut points, which generalizes to $n-1$), and mutual exclusivity holds for any $n$:

$$
P(\text{can form an } n\text{-gon})=1-\frac{n}{2^{n-1}}
$$

**Takeaway**: The triangle (and more generally, convex polygon) feasibility criterion, "the longest side is less than the sum of the rest," translates into a tail probability for a single segment; mutual exclusivity then collapses an inclusion-exclusion sum into a single term. This pattern recurs whenever a constraint sums to a constant and a single event requires exceeding half of it.

### 8. Probability Three Circle Points Share a Semicircle

**Approach**: Translate the positions of $n$ points on a circle into spacing language: they split the circumference (length 1) into $n$ arcs, again exchangeable and summing to 1. "All points lie within some common semicircle" is equivalent to "some arc has length $\ge1/2$," the same tail-probability question as the line-segment models in Problems 6 and 7.

**Derivation**: Let 3 points be iid on a circle of circumference 1. Let $A_i$ denote the event that the arc of length $1/2$ starting at point $i$ and going clockwise contains every other point, i.e. the semicircle starting at point $i$ covers all points. Given point $i$, each of the other $n-1$ points independently lands in that fixed semicircle with probability $1/2$:

$$
P(A_i)=\left(\frac12\right)^{n-1}
$$

The $A_i$ are pairwise mutually exclusive: the $n$ arc lengths always sum to 1, and two different points both being the start of a semicircle covering everyone would require two arcs, each of length $\ge1/2$, whose total cannot exceed 1, a contradiction (the same mechanism as the mutual-exclusivity argument in Problem 7). So:

$$
P(\text{common semicircle exists})=\sum_{i=1}^nP(A_i)=\frac{n}{2^{n-1}}
$$

At $n=3$ this gives $3/4$; a Monte Carlo simulation gives $0.7502$, matching.

Problems 7 and 8 ask about the same underlying event, whether some arc or segment reaches length $\ge1/2$, just phrased as its non-occurrence and occurrence respectively, making them complements:

| | Event | Probability |
|---|---|---|
| Problem 7 | No segment $\ge1/2$ (an $n$-gon can be formed) | $1-n/2^{n-1}$ |
| Problem 8 | Some arc $\ge1/2$ (a common semicircle exists) | $n/2^{n-1}$ |

**Takeaway**:

> Once translated into spacing language, a circle-point problem and a line-segment-splitting problem are the same problem: the $n$ arcs produced by $n$ points on a circle and the $n$ spacings produced by $n-1$ cut points on a line share the same exchangeable, fixed-sum structure, so the same tail-probability toolkit applies to both directly.

### 9. Expected Smallest of $n+1$ Spacings

**Approach**: Find the survival function of the minimum spacing using the same "reserve length $t$, let the remaining points distribute freely" technique from Problem 6, but reserving from all $n+1$ spacings at once, which reduces the problem to $n$ points distributed freely over a shortened interval.

**Derivation**: Let $n$ points be iid $\mathrm{Unif}[0,1]$ with order statistics $U_{(1)}<\cdots<U_{(n)}$. All $n+1$ spacings exceeding $t$ is equivalent to:

$$
U_{(1)}>t,\quad U_{(2)}-U_{(1)}>t,\quad\ldots,\quad U_{(n)}-U_{(n-1)}>t,\quad 1-U_{(n)}>t
$$

Shift $V_i=U_{(i)}-i\,t$ for $i=1,\ldots,n$. Substituting into each constraint in turn gives:

$$
0<V_1<V_2<\cdots<V_n<1-(n+1)t
$$

which is exactly $n$ points' order statistics confined to an interval of length $1-(n+1)t$. The shift preserves measure, so:

$$
P(\min_iY_i>t)=(1-(n+1)t)_+^n,\qquad 0\le t\le\frac1{n+1}
$$

Substituting $u=1-(n+1)t$ and tail-integrating:

$$
\mathbb E[\min_iY_i]=\int_0^{1/(n+1)}(1-(n+1)t)^n\,dt=\frac1{n+1}\int_0^1u^n\,du=\boxed{\frac{1}{(n+1)^2}}
$$

At $n=3$ this gives $1/16=0.0625$, matching a Monte Carlo estimate of $0.0625$.

The case $n=1$ (one cut point, two segments) is directly checkable: $\mathbb E[\min(U,1-U)]=2\int_0^{1/2}u\,du=\dfrac14$, matching the formula $1/(1+1)^2$.

**Takeaway**: This "reserve length, then let the remaining points move freely after a shift" technique is the same method used in Problem 6 to reserve from one, two, or three segments individually; here it is applied to all $n+1$ segments at once.

### 10. Expected Range $\mathbb E[X_{(n)}-X_{(1)}]$

**Approach**: The range is the largest order statistic minus the smallest, so it follows directly by differencing Problem 3's result $\mathbb E[X_{(k)}]=k/(n+1)$; equivalently, it can be viewed as 1 minus the expectations of the two edge spacings.

**Derivation**: Let $n$ points be iid $\mathrm{Unif}[0,1]$.

$$
\mathbb E[X_{(n)}-X_{(1)}]=\mathbb E[X_{(n)}]-\mathbb E[X_{(1)}]=\frac{n}{n+1}-\frac1{n+1}=\frac{n-1}{n+1}
$$

Equivalently, the range equals 1 minus the leftmost and rightmost spacings: $X_{(n)}-X_{(1)}=1-Y_1-Y_{n+1}$. Problem 3 already showed every spacing has expectation $1/(n+1)$:

$$
\mathbb E[X_{(n)}-X_{(1)}]=1-\frac2{n+1}=\frac{n-1}{n+1}
$$

Both routes give the same result. At $n=2$, this evaluates to $1/3$, exactly the answer to $\mathbb E|X-Y|$ from Problem 4; the two questions ask for the same quantity.

**Takeaway**: Once the range is written as 1 minus the two edge spacings, Problem 3's spacing-expectation result applies directly, with no need to integrate the joint density of $(X_{(1)},X_{(n)})$ from scratch.

---

## Module 7: Conditional Uniformity and the Tower Property

Problems 11 through 14 build directly on Module 2's truncation principle and Module 3's conditional uniformity method, applying them to conditional expectations, correlations, and cross moments of order statistics. None of the truncation machinery is rederived; it is only redeployed on new problems.

### 11. Conditional Expectation of the Minimum Given the Maximum

**Idea**: This is exactly the intermediate result already used inside Module 3's Method 2, isolated here as its own problem. Work out the $n=3$ special case first, then generalize. Matching Module 2's truncation principle, specialized to the uniform distribution (conditional uniformity): given $X_{(n)}=x$, the remaining $n-1$ points are iid $\mathrm{Unif}[0,x]$, so the problem reduces to the expected minimum of that new batch of points, and Module 1's formula for the mean of an $m$-point uniform minimum, $a/(m+1)$, applies directly.

**Derivation**: Start with $n=3$. Given $X_{(3)}=x$, the remaining 2 points are iid $\mathrm{Unif}[0,x]$; the smaller of these two is exactly $X_{(1)}$ among all three points, since $X_{(3)}=x$ is already the maximum and cannot be exceeded. Taking $m=2, a=x$:

$$
\mathbb E[X_{(1)}\mid X_{(3)}=x]=\frac{x}{2+1}=\frac{x}{3}
$$

The general $n$ case follows the identical argument: given $X_{(n)}=x$, the remaining $n-1$ points are iid $\mathrm{Unif}[0,x]$, and the smallest of them is $X_{(1)}$. Taking $m=n-1, a=x$:

$$
\boxed{\mathbb E[X_{(1)}\mid X_{(n)}=x]=\frac{x}{n}}
$$

Equivalently, $\mathbb E[X_{(1)}\mid X_{(n)}]=X_{(n)}/n$. Module 3's Method 2 takes this result, multiplies by $X_{(n)}$, and applies the tower property to get $\mathbb E[X_{(1)}X_{(n)}]$.

**Takeaway**:

> Given the maximum $X_{(n)}=x$, the conditional expectation of the minimum among the remaining $n-1$ points is $x/n$. This single line is the complete application of the conditional uniformity method to an expectation calculation.

### 12. Verifying the Order-Statistic Correlation for n = 2

**Idea**: The mean and variance follow directly from the standard $\text{Beta}$ distribution formulas. Recomputing the covariance by integrating the joint density of $(X_{(1)},X_{(2)})$ would be tedious, but at $n=2$ the minimum times the maximum equals the product of the two original independent variables, since the order statistics are just a relabeling of the same two numbers by size; this bypasses the joint distribution of the order statistics entirely.

**Derivation**: Means (Module 1, $n=2$):

$$
\mathbb E[X_{(1)}]=\frac{1}{n+1}=\frac13,\qquad \mathbb E[X_{(2)}]=\frac{n}{n+1}=\frac23
$$

Variances: at $n=2$, $X_{(1)}\sim\text{Beta}(1,2)$ and $X_{(2)}\sim\text{Beta}(2,1)$, and both have the same variance:

$$
\text{Var}(X_{(1)})=\text{Var}(X_{(2)})=\frac{n}{(n+1)^2(n+2)}=\frac{2}{9\cdot4}=\frac{1}{18}
$$

The covariance is the key step. Let $X,Y\sim\mathrm{Unif}[0,1]$ be the two original independent draws, so $X_{(1)}=\min(X,Y)$, $X_{(2)}=\max(X,Y)$, and

$$
X_{(1)}X_{(2)}=\min(X,Y)\cdot\max(X,Y)=XY
$$

The right side is a product of the original independent variables. This step relies on the independence of $X,Y$; the order statistics $X_{(1)},X_{(2)}$ themselves are correlated and their product cannot be split this way directly:

$$
\mathbb E[X_{(1)}X_{(2)}]=\mathbb E[XY]=\mathbb E[X]\,\mathbb E[Y]=\frac14
$$

So:

$$
\text{Cov}(X_{(1)},X_{(2)})=\frac14-\frac13\cdot\frac23=\frac14-\frac29=\frac1{36}
$$

$$
\boxed{\text{corr}(X_{(1)},X_{(2)})=\frac{1/36}{1/18}=\frac12}
$$

matching $1/n$ at $n=2$. A Monte Carlo simulation at $n=2$ gives an estimated correlation of $0.4998$, in agreement.

**Takeaway**:

> The order statistics $\{X_{(1)},\ldots,X_{(n)}\}$ are simply a reordering of the original iid sample $\{X_1,\ldots,X_n\}$. The expectation of any symmetric function of the sample (a sum, a product, a sum of squares) is unchanged by sorting, and this observation often makes it possible to skip the distribution of the order statistics entirely and compute directly on the original independent variables.

### 13. Conditional Distribution of an Order Statistic Sandwiched Between Two Others

**Idea**: This is the $n=3$ special case of Module 4's sandwiched-order-statistic result: given $X_{(1)}=a$ and $X_{(3)}=b$, only one point, $X_{(2)}$, lies between them, and the truncation principle gives its conditional distribution directly, with no need to rederive the conditional density from scratch.

**Derivation**: The joint density of the order statistics of $n=3$ iid $\mathrm{Unif}[0,1]$ draws (Module 2's formula $n!\prod f(x_i)$, with $f\equiv1$ under the uniform distribution) is the constant

$$
f_{X_{(1)},X_{(2)},X_{(3)}}(x_1,x_2,x_3)=3!=6,\qquad 0\le x_1<x_2<x_3\le1
$$

Fixing $x_1=a, x_3=b$, this joint density remains the same constant $6$ as a function of $x_2\in(a,b)$. Renormalizing over $x_2\in(a,b)$, i.e. dividing by the interval length $b-a$:

$$
f_{X_{(2)}\mid X_{(1)}=a,X_{(3)}=b}(x_2)=\frac{6}{6\cdot(b-a)}=\frac{1}{b-a},\qquad a<x_2<b
$$

$$
\boxed{X_{(2)}\mid(X_{(1)}=a,X_{(3)}=b)\sim\mathrm{Unif}[a,b]}
$$

This matches Module 4's general result exactly: given the order statistics on either side, the points between them are iid draws from the original distribution truncated to $[a,b]$, and a truncated uniform distribution is uniform again.

**Takeaway**:

> The proof of conditional uniformity reduces to one sentence: the joint density of a uniform distribution is constant, and fixing some of the coordinates leaves the conditional density on the remaining coordinates constant as well, over the shrunk interval, which is again a uniform distribution.

### 14. Expected Product of the Two Smallest Order Statistics, General n

**Idea**: Module 3's worked example computes the expected product of the minimum and maximum, $\mathbb E[X_{(1)}X_{(n)}]$. Here the product is of two adjacent order statistics, $\mathbb E[X_{(1)}X_{(2)}]$; direct integration over the joint density of $(X_{(1)},X_{(2)})$ still works (Module 3's Method 1 path), but converting to spacing variables and squaring the sum-to-one constraint solves it algebraically, with no joint density or double integral required.

**Derivation**: Let $Y_1,\ldots,Y_{n+1}$ be the $n+1$ spacings produced by $n$ points on $[0,1]$ ($Y_1=X_{(1)}$, $Y_i=X_{(i)}-X_{(i-1)}$ for $i\ge2$). Then

$$
X_{(1)}=Y_1,\qquad X_{(2)}=Y_1+Y_2,\qquad X_{(1)}X_{(2)}=Y_1^2+Y_1Y_2
$$

Step 1, the marginal second moment of a single spacing. The $n+1$ spacings are exchangeable, and each has marginal $Y_i\sim\text{Beta}(1,n)$, with mean $\mathbb E[Y]=\dfrac1{n+1}$ and variance $\text{Var}(Y)=\dfrac{n}{(n+1)^2(n+2)}$, so

$$
\mathbb E[Y^2]=\text{Var}(Y)+\mathbb E[Y]^2=\frac{n}{(n+1)^2(n+2)}+\frac{1}{(n+1)^2}=\frac{n+(n+2)}{(n+1)^2(n+2)}=\frac{2}{(n+1)(n+2)}
$$

Step 2, the cross moment $\mathbb E[Y_iY_j]$ ($i\ne j$). The $n+1$ spacings satisfy $\sum_{i=1}^{n+1}Y_i=1$; squaring both sides and taking expectations:

$$
1=\mathbb E\left[\left(\sum_{i=1}^{n+1}Y_i\right)^2\right]=(n+1)\,\mathbb E[Y^2]+(n+1)n\,\mathbb E[Y_iY_j]
$$

Expanding the square gives $n+1$ squared terms and $(n+1)n$ ordered cross-product terms, all with the same expectation by exchangeability. Substituting $\mathbb E[Y^2]$ and solving:

$$
1=\frac{2}{n+2}+(n+1)n\,\mathbb E[Y_iY_j]\ \Longrightarrow\ \mathbb E[Y_iY_j]=\frac{1-\frac{2}{n+2}}{n(n+1)}=\frac{\frac{n}{n+2}}{n(n+1)}=\frac{1}{(n+1)(n+2)}
$$

Step 3, combine:

$$
\mathbb E[X_{(1)}X_{(2)}]=\mathbb E[Y_1^2]+\mathbb E[Y_1Y_2]=\frac{2}{(n+1)(n+2)}+\frac{1}{(n+1)(n+2)}=\boxed{\frac{3}{(n+1)(n+2)}}
$$

Checking $n=2$: $\dfrac{3}{3\cdot4}=\dfrac14$, matching the value of $\mathbb E[X_{(1)}X_{(2)}]$ from Problem 12. Monte Carlo simulations at $n=3,4,5$ give $0.1501, 0.1001, 0.0714$ against theoretical values $0.1500, 0.1000, 0.0714$.

**Takeaway**:

> For a set of exchangeable random variables satisfying $\sum_i Y_i=1$, squaring both sides of this constraint and taking expectations solves directly for the cross moment $\mathbb E[Y_iY_j]$ between any two components, with no need for a double integral over the joint density or the covariance matrix.

---

## Module 8: Linearity and Indicator Variables

Problems 15 through 17 solve three superficially different counting-expectation problems with the same tool: decompose the count into a sum of indicator variables, find each indicator's expectation (usually a simple probability from symmetry or exchangeability), then sum using linearity. None of these steps requires the indicators to be independent.

### 15. Expected Number of Records in an iid Sequence

**Idea**: Encode "is position $k$ a record" as an indicator variable $I_k$, use exchangeability to get $\mathbb P(I_k=1)$ directly, then sum with linearity. Whether the $I_k$ are independent of one another plays no role in this step.

**Derivation**: Let $X_1,\ldots,X_n$ be iid continuous random variables (continuity only rules out ties). Define

$$
I_k=\mathbf 1\{X_k>\max(X_1,\ldots,X_{k-1})\},\qquad k=1,\ldots,n
$$

with $I_1=1$ by convention, since the first position is always a record. $I_k=1$ exactly when the largest of the first $k$ values happens to sit at position $k$. Since $X_1,\ldots,X_k$ are iid and thus exchangeable, the maximum is equally likely to fall at any of these $k$ positions, so

$$
\mathbb P(I_k=1)=\frac1k
$$

The total number of records is $\sum_{k=1}^n I_k$, and by linearity (no independence assumption needed among the $I_k$):

$$
\mathbb E[\#\text{records}]=\mathbb E\left[\sum_{k=1}^n I_k\right]=\sum_{k=1}^n\mathbb P(I_k=1)=\boxed{\sum_{k=1}^n\frac1k=H_n}
$$

where $H_n$ is the $n$-th harmonic number. A Monte Carlo simulation at $n=10$ gives an estimate of $2.929$, matching $H_{10}=2.9290$.

**Takeaway**:

> Indicator variables need not be independent; linearity of expectation holds for a sum regardless. The $I_k$ in this problem are clearly dependent on one another, since an early maximum directly affects every later $I_k$, yet $\mathbb E[\sum_k I_k]=\sum_k \mathbb E[I_k]$ still holds without modification.

### 16. Expected Number of Local Maxima in a Random Permutation

**Idea**: Whether a position is a local maximum depends only on the relative order of the values at that position and its neighbors, not on their actual magnitudes. Write each position's local-maximum status as an indicator, compute the probability separately for endpoints and interior positions, and sum with linearity.

**Derivation**: Let $a_1,\ldots,a_n$ be a uniformly random permutation of $\{1,\ldots,n\}$. For an interior position $2\le i\le n-1$, $a_i$ is a local maximum exactly when it is the largest of the three values $\{a_{i-1},a_i,a_{i+1}\}$. The relative order of these three values is uniformly random, so the probability that the largest one lands in the middle position, position $i$ itself, is

$$
\mathbb P(I_i=1)=\frac13,\qquad 2\le i\le n-1
$$

For the two endpoints $i=1,n$, only a single neighbor comparison is needed, and by symmetry

$$
\mathbb P(I_1=1)=\mathbb P(I_n=1)=\frac12
$$

By linearity:

$$
\mathbb E[\#\text{local maxima}]=2\cdot\frac12+(n-2)\cdot\frac13=\boxed{\frac{n+1}{3}}
$$

Exact enumeration over all permutations at $n=3,4,5$ gives average local-maxima counts of $1.3333, 1.6667, 2.0$, matching $(n+1)/3$ exactly.

**Takeaway**:

> Whether a position is a local maximum depends only on the relative order of the neighboring values, not on their actual magnitudes; the probability follows directly from a symmetry argument on the relative arrangement, with no reference to the underlying numeric distribution.

### 17. Expected Number of Mutual Nearest Neighbors

**Idea**: First identify each point's nearest neighbor, then check whether that relationship is mutual. For $n=3$, each endpoint's only possible nearest neighbor is the middle point, so the problem reduces to which side the middle point's nearest neighbor falls on, determined by the relative size of the two spacings, whose probability follows from exchangeability.

**Derivation**: Sort the three points as $X_{(1)}<X_{(2)}<X_{(3)}$, with spacings $Y_2=X_{(2)}-X_{(1)}$ and $Y_3=X_{(3)}-X_{(2)}$.

$X_{(1)}$'s only possible nearest neighbor is $X_{(2)}$, the only other point on its side; likewise $X_{(3)}$'s only possible nearest neighbor is $X_{(2)}$.

$X_{(2)}$'s own nearest neighbor depends on which spacing is smaller: it is $X_{(1)}$ if $Y_2<Y_3$, and $X_{(3)}$ otherwise.

$X_{(2)}$ is always a mutual nearest neighbor: whichever of $X_{(1)}$ or $X_{(3)}$ it points to, that point's only possible nearest neighbor is $X_{(2)}$ itself, so the relationship is automatically mutual, giving

$$
\mathbb P(X_{(2)}\text{ is a mutual nearest neighbor})=1
$$

$X_{(1)}$ is a mutual nearest neighbor exactly when $X_{(2)}$ points back to it, i.e. $Y_2<Y_3$; by exchangeability of the two spacings, this has probability $1/2$. Likewise $X_{(3)}$ is a mutual nearest neighbor with probability $1/2$.

By linearity:

$$
\mathbb E[\#\text{mutual nearest neighbors}]=1+\frac12+\frac12=\boxed{2}
$$

A full mutual-nearest-neighbor simulation gives an estimate exactly equal to $2.0$.

**Takeaway**: Whether a point qualifies as a mutual nearest neighbor decomposes into a set of indicator variables. Some points (here, $X_{(2)}$) have only one candidate nearest neighbor and satisfy mutuality automatically, while the rest require an actual comparison of spacings to get a probability. This decomposition turns a problem that looks like it needs a joint distribution into simple probability calculations on a handful of random variables.

---

## Module 9: Synthesis and Geometric Applications

The three problems in this module apply the tools developed in earlier modules to slightly more involved geometric and discrete settings: whether cut segments can form a polygon, the area of a random polygon inscribed in a circle, and the expected maximum of a discrete random variable.

---

### 18. Probability That Four Segments Form a Quadrilateral

**Idea**: Recast the segment problem as a spacing problem for order statistics. Three cut points split a unit-length segment into 4 pieces, and these 4 pieces form a quadrilateral (rather than degenerating into a folded line) if and only if the longest piece is no more than half the total length. This criterion, and the general formula it produces, was already derived earlier in this chapter: the probability that $n$ segments can form an $n$-gon is

$$
P(\text{can form})=1-\frac{n}{2^{n-1}}
$$

Substituting $n=4$ gives the answer directly, and a direct computation serves as a check.

**Derivation**: Let $Y_1,Y_2,Y_3,Y_4$ be the 4 segment lengths (the 4 spacings produced by 3 cut points), with $\sum_i Y_i=1$. The event "cannot form a quadrilateral" is equivalent to "some $Y_i>\frac12$." Since the total length is 1, two segments cannot both exceed $\frac12$ at the same time, so the 4 events $\{Y_i>\frac12\}$, $i=1,\ldots,4$, are pairwise mutually exclusive. This is the same summation argument used earlier in the chapter to derive the general $n$-gon formula.

As a direct check, compute a single event's probability: $Y_i$ is one spacing among 3 uniform cut points, with marginal distribution $\mathrm{Beta}(1,3)$, so $P(Y_i>x)=(1-x)^3$ (all 3 cuts must avoid a specific interval of length $\frac12$). At $x=\frac12$:

$$
P\left(Y_i>\frac12\right)=\left(\frac12\right)^3=\frac18
$$

By mutual exclusivity,

$$
P(\text{cannot form})=\sum_{i=1}^4 P\left(Y_i>\frac12\right)=4\times\frac18=\frac12
$$

so

$$
\boxed{P(\text{can form a quadrilateral})=1-\frac12=\frac12}
$$

Checking against the general formula: at $n=4$, $1-\dfrac{4}{2^3}=1-\dfrac48=\dfrac12$, matching. A Monte Carlo simulation with 2 million trials gives an estimate of $0.5001$, consistent with the theoretical value.

**Takeaway**: The $n$-gon criterion (longest side at most half the total length) combined with a sum over mutually exclusive events gives a closed form $1-n/2^{n-1}$ for any $n$, with no need to redo inclusion-exclusion or a multiple integral for each new $n$. At $n=3$ this reduces to the classical broken-stick triangle problem, with probability $\frac14$.

---

### 19. Expected Area of a Random Triangle Inscribed in a Circle

**Idea**: Three points are placed independently and uniformly on a unit circle (by angle). Find the expected area of the triangle they form. Use rotational symmetry to fix one point, convert the other points' angles into spacings, write the area as a sum of sines of those spacings, and take the expectation using the spacings' marginal distribution.

**Derivation**:

> This same template applies to any "expected area of a $k$-point random polygon on a circle" problem: fix one point using rotational symmetry, convert the remaining points' angular differences into spacings, write the area as $\frac12\sum\sin(\text{spacing})$, and compute $\mathbb E[\sin(\cdot)]$ from the spacings' marginal (Beta) distribution. Changing $k$ only changes the Beta parameters of the spacing distribution and the final coefficient, not the method.

Step 1 (rotational symmetry): rotating the whole configuration does not change the area, so fix the first point at angle 0 and let the other two angles be $U_1,U_2\stackrel{iid}\sim\mathrm{Unif}[0,2\pi]$.

Step 2 (sort into spacings): let $0=\theta_0<\theta_1<\theta_2<2\pi$ be the sorted angles (0 together with the order statistics of $U_1,U_2$), and define the three arc spacings

$$
\varphi_1=\theta_1,\qquad \varphi_2=\theta_2-\theta_1,\qquad \varphi_3=2\pi-\theta_2
$$

These spacings are exchangeable and sum to $2\pi$, the same family of object as the circle spacings in Problem 8 earlier in this chapter, except that here the actual angle values are needed rather than just whether one spacing exceeds a threshold.

Step 3 (decomposing the area around the center): using the center as a common vertex, split the triangle into 3 smaller triangles, each formed by the center and one side of the original triangle (i.e., two adjacent vertices). Both sides from the center have length 1 (the radius), with included angle $\varphi_i$, so each has area $\frac12\cdot1\cdot1\cdot\sin\varphi_i$:

$$
A=\frac12\sum_{i=1}^3\sin\varphi_i
$$

When the center falls outside the triangle, one $\varphi_i>\pi$, and the corresponding $\sin\varphi_i$ automatically turns negative, subtracting off exactly the over-counted area. So this formula holds for any configuration without needing to split into cases based on whether the center is inside or outside the triangle.

Step 4 (linearity of expectation and exchangeability):

$$
\mathbb E[A]=\frac12\times3\times\mathbb E[\sin\varphi_1]=\frac32\mathbb E[\sin\varphi_1]
$$

Step 5 (distribution of $\varphi_1$): $\varphi_1/(2\pi)$ is a single spacing among 2 uniform points, distributed as $\mathrm{Beta}(1,2)$ with density $2(1-u)$ on $u\in[0,1]$. Converting back with $x=2\pi u$:

$$
f_{\varphi_1}(x)=\frac1\pi\left(1-\frac{x}{2\pi}\right),\qquad 0\le x\le2\pi
$$

Step 6 (integrate):

$$
\mathbb E[\sin\varphi_1]=\frac1\pi\int_0^{2\pi}\sin x\,dx-\frac{1}{2\pi^2}\int_0^{2\pi}x\sin x\,dx
$$

The first integral is over a full period and equals 0. The second, via integration by parts:

$$
\int_0^{2\pi}x\sin x\,dx=\big[-x\cos x+\sin x\big]_0^{2\pi}=-2\pi
$$

so

$$
\mathbb E[\sin\varphi_1]=0-\frac{1}{2\pi^2}\times(-2\pi)=\frac1\pi
$$

Substituting back into Step 4:

$$
\boxed{\mathbb E[A]=\frac32\times\frac1\pi=\frac{3}{2\pi}\approx0.4775}
$$

As a check, simulating 3 random angles directly and computing the triangle's area via the shoelace formula gives a Monte Carlo estimate of $0.47726$, close to $3/(2\pi)=0.47746$.

The same method applied to 4 random points on the circle gives the expected area of the resulting quadrilateral. Here $\mathbb E[A]=4\times\frac12\mathbb E[\sin\varphi_1]=2\mathbb E[\sin\varphi_1]$, where $\varphi_1$ is now a spacing among 3 uniform points with density $\dfrac{3}{2\pi}\left(1-\dfrac{x}{2\pi}\right)^2$; the corresponding integral gives $\mathbb E[\sin\varphi_1]=3/(2\pi)$, so $\mathbb E[A]=3/\pi$. The computation has the identical structure to the triangle case, only the order of the spacing distribution and the outer coefficient change; the full integral is not repeated here.

**Takeaway**: the core decomposition for random-polygon-area-on-a-circle problems is fixing one point by rotational symmetry, writing the area as a sum of sines of the spacings, and taking the expectation via the spacings' marginal distribution. This three-step structure does not depend on the number of points; the point count only changes the Beta parameters of the spacing distribution and the outer coefficient.

---

### 20. Expected Maximum of Three Dice

**Idea**: Three independent fair six-sided dice. Find $\mathbb E[M]$ for the maximum $M$. Since $M$ takes values in the finite integer set $\{1,\ldots,6\}$, the tail-sum formula is more direct than writing out the pmf and summing by definition.

**Derivation**:

> The expectation of a discrete random variable can also be computed with a tail-sum formula: $\mathbb E[M]=\sum_{k\ge1}\mathbb P(M\ge k)$ when $M$ takes positive integer values. This is the discrete analogue of the continuous formula $\mathbb E[T]=\int_0^\infty \mathbb P(T>t)\,dt$, and it should be the first tool considered for maximum/minimum problems over a finite integer set, rather than writing out the pmf and summing directly.

For $M\in\{1,\ldots,6\}$,

$$
\mathbb E[M]=\sum_{k=1}^{6}\mathbb P(M\ge k)
$$

The complement of $M\ge k$ is $M\le k-1$, meaning all 3 dice show $\le k-1$:

$$
\mathbb P(M\ge k)=1-\mathbb P(M\le k-1)=1-\left(\frac{k-1}{6}\right)^3
$$

Substituting into the sum:

$$
\mathbb E[M]=\sum_{k=1}^6\left[1-\frac{(k-1)^3}{216}\right]=6-\frac{0+1+8+27+64+125}{216}=6-\frac{225}{216}
$$

Simplifying:

$$
\boxed{\mathbb E[M]=6-\frac{25}{24}=\frac{119}{24}\approx4.958}
$$

A simulation of 3 million trials of 3 dice gives a mean of $4.9583$, matching $119/24=4.9583$ exactly.

**Takeaway**: whenever a random variable takes values in a finite (or countable) set of integers, the tail-sum formula $\mathbb E[M]=\sum_k\mathbb P(M\ge k)$ is usually faster than deriving the pmf and summing by definition, particularly when $\mathbb P(M\ge k)$ can be written directly from the complementary event "all variables are below some value" (here, all 3 dice showing $\le k-1$).

---

## Module 10: Final Checklist Before an Interview

1. For the distribution of max or min, which function comes first? The CDF (the survival function is more direct for min), followed by differentiation to get the density. Writing the density directly requires separately handling the combinatorial choice of "which point is the extreme value."
2. Does the problem give the value of one order statistic and ask for the conditional distribution of the rest? For any continuous distribution, the rest are iid draws from the original distribution truncated to the corresponding interval.
3. Is the underlying distribution uniform? If so, the truncated distribution is uniform again, and the uniform distribution's own formulas (such as the mean of an $m$-point minimum being $\frac{a}{m+1}$) apply directly, with no fresh integral needed.
4. Is the underlying distribution not uniform? Module 2's truncation principle still holds, but the truncated distribution is usually a new distribution, and every subsequent step needs a fresh integral against that new density; there is no algebraic shortcut equivalent to the uniform case.

One sentence to keep in mind:

> max $\le x$ means every point is $\le x$; min $>x$ means every point is $>x$. Given an extreme value, the remaining points follow a truncated distribution, and this holds for any continuous distribution. The uniform distribution only makes the computation after truncation clean; it is not the reason the principle holds.
