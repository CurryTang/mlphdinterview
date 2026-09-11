# Quant 03 · Continuous distributions and order statistics

Course: [[Quant02 Markov Chains Expected Time|02 Markov Chains]] → This note → [[Quant04 Correlation Matrix PSD|04 Correlation Matrix]]

There are three common tools for handling continuous random variables: writing the CDF, integrating geometric regions, and applying variable transformations. This note combines these tools with order statistics.

---

## 1 · From Density and CDF to Geometry

When finding the distribution of $T=g(X,Y,Z)$, looking for the density directly is error-prone. The first step is always to write the CDF:

$$
F_T(t)=P(T\le t)
$$

The next step is to translate the event $T\le t$ into regions, conditional probabilities, or integrals in terms of the original variables. Under a uniform distribution, the probability is directly equal to the area or volume of the geometric region satisfying the condition.

### 1.1 Product of Uniforms

Let $A, B \sim U[0,1]$ be independent. Find the CDF of $U=AB$:

$$
F_U(u) = P(AB \le u)
$$

Since $(A,B)$ is uniformly distributed over the unit square $[0,1]\times[0,1]$, this is equivalent to calculating the area to the left of the curve $a = u/b$. The integral must be split because the region is truncated by the boundary:

```text
b
1 |█████████████████░░░░░░░░
  |███████████░░░░░░░░░░░░░░
  |████████░░░░░░░░░░░░░░░░░
u |█████████████████████████
  |█████████████████████████
0 +------------------------- a
    0        u/b           1

█ = region where ab <= u
```

Splitting the integral by $b$:

| Fixed $b$ | Condition on the horizontal line | Length of $a$ satisfying the condition |
| --- | --- | --- |
| $0<b\le u$ | $u/b\ge1$ | Entire segment $[0,1]$, length $1$ |
| $u<b\le1$ | $u/b<1$ | $0\le a\le u/b$, length $u/b$ |

$$
F_U(u) = \int_0^u 1\,db + \int_u^1 \frac{u}{b}\,db = u - u\ln u, \qquad 0<u<1
$$

### 1.2 Volume of a Simplex

If $X_1,\ldots,X_n \sim U[0,1]$ are independent, the inequality $\sum_{i=1}^n X_i < 1$ defines a standard simplex.
Under a uniform distribution, the probability is simply the volume of the simplex:

```text
          x3
          |
          |\
          | \
          |  \
          |___\____ x2
         /
        /
       x1

x1 + x2 + x3 < 1
```

The volume can be proved by recursive slicing. Define $V_n(t)$ as the volume of the simplex $\sum_{i=1}^n x_i < t$. Fixing the last coordinate $x_n=s$, the available budget for the remaining $n-1$ coordinates is $t-s$:

$$
V_n(t) = \int_0^t V_{n-1}(t-s)\,ds
$$

Knowing $V_1(t)=t$, and assuming $V_{n-1}(u) = \frac{u^{n-1}}{(n-1)!}$:

$$
V_n(t) = \int_0^t \frac{(t-s)^{n-1}}{(n-1)!}\,ds
$$

Substitute $u=t-s$:

$$
V_n(t) = \int_0^t \frac{u^{n-1}}{(n-1)!}\,du = \frac{t^n}{n!}
$$

Therefore, the probability of the sum being less than $1$ is:

$$
P\left(\sum_{i=1}^n X_i < 1\right) = V_n(1) = \frac{1}{n!}
$$

---

## 2 · Variable Transformations and the Jacobian

For a strictly monotonic univariate function $Y=g(X)$, differentiating the CDF gives the density transformation:

$$
f_Y(y) = f_X(x) \left| \frac{dx}{dy} \right|
$$

For a multivariate bijective transformation $(U,V) = g(X,Y)$, conservation of probability mass requires the absolute value of the Jacobian determinant:

$$
f_{U,V}(u,v) = f_{X,Y}(x,y) |J|^{-1}
$$

where $J = \frac{\partial(u,v)}{\partial(x,y)}$. The Jacobian matrix maps irregular regions into more regular coordinate systems.

### 2.1 Standard Transformation: Logarithms and Products

When dealing with products or powers on $[0,1]$, taking the negative logarithm $-\ln$ is a standard monotonic transformation.
If $X \sim U[0,1]$, let $R=-\ln X$, then:

$$
P(R \le r) = P(-\ln X \le r) = P(X \ge e^{-r}) = 1 - e^{-r}
$$

This is exactly the CDF of an $\mathrm{Exp}(1)$ distribution. Products turn into sums under logarithms:

$$
-\ln(XY) = (-\ln X) + (-\ln Y)
$$

If $X,Y \sim U[0,1]$ are independent, $-\ln(XY)$ is the sum of two independent $\mathrm{Exp}(1)$ variables, which follows a $\mathrm{Gamma}(2,1)$ distribution. Its density is:
$$
f_G(g) = g e^{-g}, \quad g>0
$$
This algebraic trick drastically simplifies multiple integrals.

### 2.2 Comprehensive Transformation Example: $T=|XY|^{|Z|}$

Let $X,Y,Z\sim U[-1,1]$ be independent. Find the CDF of $T=|XY|^{|Z|}$.

Since $|X|, |Y|, |Z|$ all follow $U[0,1]$, the problem simplifies to $T=(AB)^C$, where $A,B,C \sim U[0,1]$.
Take the negative logarithm:

$$
-\ln T = C[-\ln(AB)] = C \cdot G
$$

where $C \sim U[0,1]$ and $G \sim \mathrm{Gamma}(2,1)$.
For $0<t<1$, let $a=-\ln t$. The event $T\le t$ is equivalent to $CG \ge a$.
Fix $G=g$. When $g \ge a$, the requirement $C \ge a/g$ has a probability of $1 - a/g$ under $U[0,1]$:

$$
P(CG\ge a)=\int_a^\infty \left( 1-\frac{a}{g} \right)g e^{-g}\,dg
$$

Expand and integrate:

$$
\int_a^\infty g e^{-g}\,dg = (a+1)e^{-a}
$$
$$
\int_a^\infty a e^{-g}\,dg = a e^{-a}
$$

Subtracting these gives:
$$
P(CG\ge a) = e^{-a} = t
$$

Therefore, $F_T(t) = t$, meaning $|XY|^{|Z|} \sim U[0,1]$.

### 2.3 $T=\max(|XY|,|Z|)$ and $T=\min(|XY|,|Z|)$

Transformations combining extremes and products are handled differently:

For $T=\max(|XY|,|Z|)$:
$$
\max(|XY|,|Z|)\le t \iff |XY|\le t \text{ and } |Z|\le t
$$
Since the CDF of $A=|XY|$ is $t-t\ln t$:
$$
F_T(t) = (t-t\ln t) \cdot t = t^2(1-\ln t)
$$

For $T=\min(|XY|,|Z|)$, using the survival function is more direct:
$$
\min(|XY|,|Z|)>t \iff |XY|>t \text{ and } |Z|>t
$$
$$
F_T(t) = 1 - [1-(t-t\ln t)][1-t]
$$

### 2.4 Other Similar Transformations

**$T=|XYZ|$**
Taking the negative logarithm results in the sum of three $\mathrm{Exp}(1)$ variables, giving $\mathrm{Gamma}(3,1)$:
$$
-\ln T = (-\ln |X|) + (-\ln |Y|) + (-\ln |Z|)
$$
Resulting in $F_T(t) = t\left(1-\ln t+\frac{(\ln t)^2}{2}\right)$.

**$T=|X|^{|Y|}$**
$$
P(A^B \le t) = \int_0^1 P(A \le t^{1/b})\,db = \int_0^1 t^{1/b}\,db
$$
This integral usually does not need further expansion; writing this step is considered the correct CDF form in practice.

---

## 3 · Order Statistics: CDFs of Extremes

Let $X_1,\ldots,X_n$ be independent and identically distributed, with density $f$ and CDF $F$. Sorting them in increasing order gives $X_{(1)}\le X_{(2)}\le\cdots\le X_{(n)}$.

Finding the distribution of the maximum $X_{(n)}$ should start from the CDF to avoid combinatorial errors:

$$
F_{X_{(n)}}(x) = P(X_1 \le x, \ldots, X_n \le x) = F(x)^n
$$

Differentiating gives the density:

$$
f_{X_{(n)}}(x) = n f(x) F(x)^{n-1}
$$

For the minimum $X_{(1)}$, start from the survival function:

$$
1-F_{X_{(1)}}(x) = P(X_1 > x, \ldots, X_n > x) = [1-F(x)]^n
$$

$$
f_{X_{(1)}}(x) = n f(x) [1-F(x)]^{n-1}
$$

### 3.1 Expected Extremes of $U[0,1]$

$$
\mathbb{E}[X_{(n)}] = \int_0^1 x\cdot n x^{n-1}\,dx = \frac{n}{n+1}
$$

$$
\mathbb{E}[X_{(1)}] = \int_0^1 x\cdot n(1-x)^{n-1}\,dx = \frac{1}{n+1}
$$

For $\mathbb{E}[X_{(1)}]$, symmetry offers a quick derivation: under a uniform distribution, $X_{(1)}$ and $1-X_{(n)}$ have the same distribution, so $\mathbb{E}[X_{(1)}] = 1 - \frac{n}{n+1} = \frac{1}{n+1}$.

Second moments are important for covariance calculations:

$$
\mathbb{E}[X_{(n)}^2] = \int_0^1 x^2\cdot n x^{n-1}\,dx = \frac{n}{n+2}
$$

---

## 4 · Joint Distribution and Spacings

The joint density of all order statistics introduces the permutation factor $n!$:

$$
f_{X_{(1)},\ldots,X_{(n)}}(x_1,\ldots,x_n) = n! \prod_{i=1}^n f(x_i), \qquad x_1 < \cdots < x_n
$$

### 4.1 Marginal Distribution of $X_{(k)}$

For the uniform distribution, integrating out the variables on both sides gives the marginal density of the intermediate point $X_{(k)}$:

$$
f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!} x^{k-1} (1-x)^{n-k}
$$

This is the $\mathrm{Beta}(k, n-k+1)$ distribution. Since the mean of a Beta distribution is $\frac{\alpha}{\alpha+\beta}$, its expectation is $\frac{k}{n+1}$.

### 4.2 Exchangeability and Expectations of Spacings

On the interval $[0,1]$, $n$ random points split the segment into $n+1$ spacings:
$$Y_1=X_{(1)}, \quad Y_i=X_{(i)}-X_{(i-1)}, \quad Y_{n+1}=1-X_{(n)}$$

These $n+1$ spacings are jointly uniform over the simplex $\sum Y_i = 1$, and they are completely symmetric (exchangeable). Thus, the expectation of every spacing is equal:

$$
\mathbb{E}[Y_i] = \frac{1}{n+1}
$$

Using spacings, we can bypass multiple integrals. For the expected range:

$$
\mathbb{E}[X_{(n)} - X_{(1)}] = 1 - \mathbb{E}[Y_1] - \mathbb{E}[Y_{n+1}] = 1 - \frac{2}{n+1} = \frac{n-1}{n+1}
$$

Cross moments can also be found. Because $Y_i \sim \mathrm{Beta}(1,n)$, its mean is $1/(n+1)$ and variance is $n/((n+1)^2(n+2))$, meaning its second moment is:
$$
\mathbb{E}[Y_i^2] = \frac{2}{(n+1)(n+2)}
$$

Since $\sum_{i=1}^{n+1} Y_i = 1$, squaring both sides and taking expectations:

$$
(n+1)\mathbb{E}[Y_i^2] + (n+1)n\mathbb{E}[Y_i Y_j] = 1
$$

Substituting the second moment directly solves for the expected product algebraically as $\mathbb{E}[Y_i Y_j] = \frac{1}{(n+1)(n+2)}$, avoiding integrating a joint distribution.

---

## 5 · Conditional Distributions and Truncation

After conditioning on an extreme value, the distributions of the remaining points are truncated. This is the most powerful structural tool for order statistics.

Dividing the joint density by the marginal density of the maximum:

$$
f_{X_{(1)},\ldots,X_{(n-1)} \mid X_{(n)}=x}(x_1,\ldots,x_{n-1}) = \frac{n!\prod_{i=1}^n f(x_i)}{n f(x) F(x)^{n-1}} = (n-1)! \prod_{i=1}^{n-1} \frac{f(x_i)}{F(x)}
$$

This result holds for any continuous distribution:
Given $X_{(n)}=x$, the remaining $n-1$ points can be treated as independent and identically distributed samples drawn from the original distribution truncated to $[0,x]$.

More generally, given $X_{(m)}=a$ and $X_{(k)}=b$ (with $m<k$), the $k-m-1$ points strictly between them are iid draws from $F$ truncated to $[a,b]$.

### 5.1 Applying Conditional Uniformity to Cross Moments

When the original distribution is uniform, $f(t)/F(x) = 1/x$, and the truncated distribution on $[0,x]$ is still $\mathrm{Unif}[0,x]$. This closure property is unique to the uniform distribution.

Combining this property with the tower property computes expectations like $\mathbb{E}[X_{(1)}X_{(n)}]$:

$$
\mathbb{E}[X_{(1)}X_{(n)}] = \mathbb{E}[X_{(n)} \mathbb{E}[X_{(1)} \mid X_{(n)}]]
$$

Conditioned on $X_{(n)}$, the remaining $n-1$ points follow $\mathrm{Unif}[0,X_{(n)}]$. Their minimum is exactly $X_{(1)}$ for the entire sample. Therefore:

$$
\mathbb{E}[X_{(1)} \mid X_{(n)}] = \frac{X_{(n)}}{(n-1)+1} = \frac{X_{(n)}}{n}
$$

Substituting back into the outer expectation:

$$
\mathbb{E}[X_{(1)}X_{(n)}] = \mathbb{E}\left[X_{(n)} \frac{X_{(n)}}{n}\right] = \frac{\mathbb{E}[X_{(n)}^2]}{n} = \frac{1}{n} \cdot \frac{n}{n+2} = \frac{1}{n+2}
$$

This is significantly faster than computing the double integral directly.

### 5.2 Memorylessness and Spacings of Exponential Distributions

The exponential distribution is invariant under left-truncation and shifting. The minimum of $n$ independent $\mathrm{Exp}(\lambda)$ is $\mathrm{Exp}(n\lambda)$.
Given $X_{(k-1)}$, the portions of the remaining points that exceed it, shifted down by $X_{(k-1)}$, are again independent $\mathrm{Exp}(\lambda)$.
Thus, the spacings between consecutive order statistics are mutually independent exponential distributions:

$$
D_k = X_{(k)} - X_{(k-1)} \sim \mathrm{Exp}((n-k+1)\lambda)
$$

This independence means the expected maximum $X_{(n)}$ can be written directly as the sum of expected spacings, introducing the harmonic number $H_n$:

$$
\mathbb{E}[X_{(n)}] = \sum_{k=1}^n \frac{1}{(n-k+1)\lambda} = \frac{1}{\lambda} \sum_{j=1}^n \frac{1}{j} = \frac{H_n}{\lambda}
$$

This asymmetry is a structural feature of the exponential distribution. The minimum is exponential, but the maximum is not.

---

## 6 · Indicator Variables and Linearity: Local Maxima and Records

Complex discrete counting problems can be simplified by breaking them down into indicator variables. Indicator variables do not need to be mutually independent for the linearity of expectation to hold.

### 6.1 Expected Number of Records in an IID Sequence

Let $X_1,\ldots,X_n$ be continuous and iid. Let $I_k=1$ if the $k$-th value is the maximum of the first $k$ values (a "record").
Since the first $k$ values are exchangeable, the probability that the maximum falls exactly in the last position is $1/k$:

$$
\mathbb{P}(I_k=1) = \frac{1}{k}
$$

Even though future records depend on past states, the expected total is:

$$
\mathbb{E}[\text{records}] = \sum_{k=1}^n \mathbb{E}[I_k] = \sum_{k=1}^n \frac{1}{k} = H_n
$$

### 6.2 Local Maxima in a Random Permutation

In a random permutation from $1$ to $n$, whether a position is a local maximum depends only on its relative order with neighbors.
For an interior position $i$, the probability of being greater than both left and right neighbors equals the probability that the maximum of three values lands in the middle:

$$
\mathbb{P}(I_i=1) = \frac{1}{3}
$$

For the two endpoints, they only need to be greater than their single neighbor, with probability $1/2$.
The total expected number of local maxima is:

$$
\mathbb{E}[\text{local maxima}] = 2\cdot\frac{1}{2} + (n-2)\cdot\frac{1}{3} = \frac{n+1}{3}
$$

---

## 7 · Geometric Problems with Spacing Constraints

The spacing model can solve probabilities for forming polygons.

### 7.1 Probability That Three Segments Form a Triangle

Randomly picking two cut points on $[0,1]$ forms three segments $Y_1,Y_2,Y_3$. They form a triangle if and only if no segment exceeds $1/2$.
The event of failing to form a triangle means some segment $>1/2$. Since the total sum is $1$, it is impossible for two segments to simultaneously be $>1/2$. Thus, these events are mutually exclusive.
For a single segment, both points must fall in an area of size $1/2$ avoiding the segment: $P(Y_i > 1/2) = (1/2)^2 = 1/4$.

$$
P(\text{cannot form}) = \sum_{i=1}^3 P(Y_i > 1/2) = 3 \times \frac{1}{4} = \frac{3}{4}
$$

$$
P(\text{can form}) = 1 - \frac{3}{4} = \frac{1}{4}
$$

### 7.2 All Points on a Semicircle

Placing $n$ points on a circle splits the circumference into $n$ arcs. All points falling in the same semicircle is equivalent to one arc length being $>1/2$.
Similarly, due to mutual exclusivity:

$$
P(\text{common semicircle}) = n \left(\frac{1}{2}\right)^{n-1}
$$

Both problems utilize the principle that when the sum is fixed, events demanding more than half the total are mutually exclusive, greatly simplifying inclusion-exclusion. This allows us to complete the calculation without expanding complex intersection terms.

---

## Reference

- Zhou, *A Practical Guide to Quantitative Finance Interviews*
- Crack, *Heard on the Street*
- [Order Statistic (Wikipedia)](https://en.wikipedia.org/wiki/Order_statistic)
- [Jacobian matrix and determinant (Wikipedia)](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
