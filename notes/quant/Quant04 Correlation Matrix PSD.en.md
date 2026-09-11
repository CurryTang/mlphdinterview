# Quant 04 · Covariance, Gaussians, and correlation matrices

Course: [[Quant03 Continuous Distribution Geometry Transform|03 Continuous distribution]] → This note → [[Quant06 High Dimensional Integral Dominated Convergence|05 High-dimensional integral]]

Covariance quantifies the tendency of two random variables to move together. When normalized to remove unit dependence, it becomes correlation. A correlation matrix collects the pairwise correlations of multiple variables; its most important mathematical constraint is being positive semidefinite (PSD).

---

## 1 · Covariance and correlation

For two random variables $X$ and $Y$, covariance is defined as:

$$
\operatorname{Cov}(X,Y) = \mathbb{E}\left[ (X-\mathbb{E}X)(Y-\mathbb{E}Y) \right]
$$

It measures whether the two variables tend to deviate from their means in the same direction.

| Situation | Intuition | Sign of covariance |
|---|---|---|
| When $X$ is above its mean, $Y$ is also often above its mean | Rise and fall together | Positive |
| When $X$ is above its mean, $Y$ is often below its mean | Move in opposite directions | Negative |
| No stable linear relationship | Weak linear comovement | Close to 0 |

A lightweight mental model:

```text
positive covariance        negative covariance        near zero covariance

y                          y                          y
|        *                 | *                        |   *    *
|      *                   |   *                      | *   *
|    *                     |     *                    |      *
|  *                       |       *                  | *       *
+--------- x               +--------- x               +--------- x
```

Covariance has a flaw: it scales with units. Converting dollars to cents greatly increases the covariance. Handling numerical data requires standardization.

---

## 2 · Correlation is standardized covariance

Correlation is defined as:

$$
\operatorname{corr}(X,Y) = \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
$$

where standard deviations are:

$$
\sigma_X=\sqrt{\operatorname{Var}(X)}
$$

$$
\sigma_Y=\sqrt{\operatorname{Var}(Y)}
$$

Alternatively, the variables can be standardized first:

$$
Z_X=\frac{X-\mathbb{E}X}{\sigma_X}
$$

$$
Z_Y=\frac{Y-\mathbb{E}Y}{\sigma_Y}
$$

After this step, the standardized variables have a mean of 0 and a variance of 1. Then:

$$
\operatorname{corr}(X,Y)=\operatorname{Cov}(Z_X,Z_Y)
$$

Correlation represents the pure strength of linear comovement between two variables after stripping away units.

```text
raw variable X
  -> center: X - E[X]
  -> scale: divide by sigma_X
  -> standardized Z_X
  -> Cov(Z_X, Z_Y) = corr(X,Y)
```

### Why correlation is always bounded in [-1, 1]

After standardization, $\operatorname{Var}(Z_X)=\operatorname{Var}(Z_Y)=1$. 
Consider any linear combination of them; the variance must be non-negative. For any real number $t$:

$$
\operatorname{Var}(Z_X-tZ_Y)\ge0
$$

Expanding the variance formula:

$$
\operatorname{Var}(Z_X) - 2t\operatorname{Cov}(Z_X,Z_Y) + t^2\operatorname{Var}(Z_Y) \ge 0
$$

Substitute the known values:

$$
1-2t\operatorname{Cov}(Z_X,Z_Y)+t^2 \ge 0
$$

Let $ ho=\operatorname{Cov}(Z_X,Z_Y)$. This creates a quadratic polynomial in terms of $t$:

$$
t^2 - 2 ho t + 1 \ge 0
$$

Since it is greater than or equal to zero for all $t$, its discriminant must be less than or equal to zero:

$$
(-2 ho)^2 - 4 \le 0
$$

Solving this yields:

$$
 ho^2 \le 1
$$

Which means:

$$
-1 \le \operatorname{corr}(X,Y) \le 1
$$

Note that $ ho=0$ only implies the absence of a linear relationship; it is not equivalent to independence. Independence guarantees a covariance of 0, but a covariance of 0 does not guarantee independence.

---

## 3 · Correlation matrices and positive semidefiniteness

Given $n$ random variables $X_1,\ldots,X_n$, standardize them all to $Z_i$. The correlation matrix $R$ is the covariance matrix of these $Z_i$:

$$
R_{ij} = \operatorname{corr}(X_i,X_j) = \operatorname{Cov}(Z_i,Z_j)
$$

The correlation matrix must satisfy three strict algebraic properties:

| Property | Source |
|---|---|
| Symmetric | $\operatorname{corr}(X_i,X_j)=\operatorname{corr}(X_j,X_i)$ |
| Diagonal is 1 | The correlation coefficient of a variable with itself is 1 |
| Positive Semidefinite (PSD) | The variance of any linear combination of random variables is non-negative |

Positive Semidefiniteness (PSD) is the core mathematical constraint for multi-variable correlation. Choose arbitrary real weights $a_1,\ldots,a_n$ and form a new random variable:

$$
W = a_1Z_1 + \cdots + a_nZ_n
$$

Its variance must be non-negative:

$$
\operatorname{Var}(W) = \operatorname{Var}\left( \sum_{i=1}^n a_iZ_i \right) \ge 0
$$

Expanding the double sum:

$$
\operatorname{Var}(W) = \sum_{i=1}^n\sum_{j=1}^n a_i a_j \operatorname{Cov}(Z_i,Z_j) = a^\top R a
$$

This proves that for any arbitrary vector $a$, the quadratic form is:

$$
a^\top R a \ge 0
$$

This is the exact definition of a positive semidefinite matrix.

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

## 4 · Equicorrelation lower bound

Given $n$ variables, if all pairwise correlation coefficients are equal to $ ho$, what is the smallest possible value for $ ho$?

Construct an all-ones weight vector $a = (1, 1, \dots, 1)^\top$. The correlation matrix $R$ must satisfy the PSD constraint:

$$
a^\top R a = \sum_{i=1}^n \sum_{j=1}^n R_{ij} \ge 0
$$

In the matrix $R$, there are $n$ ones on the diagonal and $n(n-1)$ entries of $ ho$ off the diagonal:

$$
a^\top R a = n + n(n-1) ho \ge 0
$$

Simplifying this gives:

$$
n(n-1) ho \ge -n
$$

Solving for the lower bound:

$$
 ho \ge -\frac{1}{n-1}
$$

This provides a strict constraint for equicorrelation matrices. For example:
- For $n=3$, the lower bound is $-1/2$.
- For $n=4$, the lower bound is $-1/3$.

As the number of variables increases, the lower bound approaches 0. When $n$ is large, it is impossible to construct a set of random variables that are all strongly negatively correlated with each other.

If the goal is to find the minimum possible sum of all pairwise correlations, regardless of whether they are equal, the same inequality applies directly:

$$
\operatorname{Var}(Z_1 + \cdots + Z_n) = n + 2 \sum_{1\le i<j\le n} \operatorname{corr}(X_i,X_j) \ge 0
$$

Resulting in the lower bound:

$$
\sum_{1\le i<j\le n} \operatorname{corr}(X_i,X_j) \ge -\frac{n}{2}
$$

---

## 5 · Three-variable correlation bounds

When partial correlation information is known, the PSD constraint can be used to derive limits for the remaining correlations. If the correlation between $X$ and $Y$ is $ ho_{12}$ and between $Y$ and $Z$ is $ ho_{23}$, how is the correlation between $X$ and $Z$, $ ho_{13}$, restricted?

Write out the $3 \times 3$ correlation matrix for these three variables:

$$
R = \begin{pmatrix} 1 &  ho_{12} &  ho_{13} \\  ho_{12} & 1 &  ho_{23} \\  ho_{13} &  ho_{23} & 1 \end{pmatrix}
$$

Since $R$ is PSD, all its principal minors must be non-negative. Specifically, the determinant of the entire matrix must be non-negative:

$$
\det(R) \ge 0
$$

Expanding the determinant:

$$
1 + 2 ho_{12} ho_{23} ho_{13} -  ho_{12}^2 -  ho_{23}^2 -  ho_{13}^2 \ge 0
$$

Rearranging this into a quadratic inequality in terms of $ ho_{13}$:

$$
 ho_{13}^2 - 2 ho_{12} ho_{23} ho_{13} + ( ho_{12}^2 +  ho_{23}^2 - 1) \le 0
$$

This is a parabola opening upwards, constrained between its roots. Solving the quadratic equation yields a closed interval for $ ho_{13}$:

$$
 ho_{13} \in \left[  ho_{12} ho_{23} - \sqrt{(1- ho_{12}^2)(1- ho_{23}^2)},\  ho_{12} ho_{23} + \sqrt{(1- ho_{12}^2)(1- ho_{23}^2)} \right]
$$

Geometrically, correlation coefficients can be interpreted as the cosine of the angle between random vectors in space. Knowing two angles naturally restricts the third due to triangle inequality limits in spatial geometry.

---

## 6 · Cholesky decomposition and correlated Gaussian simulation

Given two independent standard normal variables $U, V \overset{i.i.d.}{\sim} N(0,1)$, how can we construct bivariate standard normals $(X,Y)$ with a correlation coefficient of $ ho$?

This is done through a linear transformation using the Cholesky decomposition of the correlation matrix:

$$
\begin{pmatrix} X \\ Y \end{pmatrix}
=
\begin{pmatrix} 1 & 0 \\  ho & \sqrt{1- ho^2} \end{pmatrix}
\begin{pmatrix} U \\ V \end{pmatrix}
$$

Expanded form:

$$
X = U
$$

$$
Y =  ho U + \sqrt{1- ho^2} V
$$

Check the means, variances, and covariances:

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

Since the variances of both $X$ and $Y$ are 1, their covariance equals the correlation coefficient:

$$
\operatorname{corr}(X,Y) =  ho
$$

This transformation linearly stretches the independent, circularly symmetric distribution in the $(U,V)$ plane into an elliptical distribution with a specific tilt. Since it relies purely on a linear operator, it preserves joint normality.

---

## 7 · Expected sign correlation of bivariate normals

Using the construction above, we can precisely calculate the expected value of the product of signs for bivariate standard normals:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
$$

where $(X,Y)$ have a correlation of $ ho$. The probability of a continuous normal distribution taking exactly the value 0 is 0, so the product of signs must be either 1 or -1.

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = 1 \cdot P(\text{same sign}) + (-1) \cdot P(\text{opposite sign}) = P(\text{same sign}) - P(\text{opposite sign})
$$

Using the law of total probability to replace the opposite sign probability:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = 2P(\text{same sign}) - 1
$$

Due to the bivariate normal distribution's symmetry around the origin, the probability of falling in the first quadrant equals the third quadrant:

$$
P(X>0, Y>0) = P(X<0, Y<0)
$$

Therefore:

$$
P(\text{same sign}) = 2P(X>0, Y>0)
$$

Let $p = P(X>0, Y>0)$. The expectation simplifies to:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = 4p - 1
$$

### Calculating probability using sectors

Substitute the Cholesky transform into the calculation for $p$:

$$
p = P(X>0, Y>0) = P\left(U>0,\  ho U + \sqrt{1- ho^2} V > 0\right)
$$

The calculation moves to the independent $(U,V)$ plane. Because $U,V$ are independent standard normals, their joint density function is:

$$
f(u,v) = \frac{1}{2\pi} e^{-(u^2+v^2)/2}
$$

This density is perfectly circularly symmetric; it depends strictly on distance from the origin and not on the angle. For any sector region starting from the origin, its probability mass strictly equals the ratio of its angle to the full circle, $\theta / 2\pi$.

The two inequalities define two half-planes:

1. $U > 0$: The boundary is $U=0$ (the $V$-axis), keeping the right half-plane.
2. $ ho U + \sqrt{1- ho^2} V > 0$: The boundary line is $V = -\frac{ ho}{\sqrt{1- ho^2}} U$.

Let $\alpha = \arcsin ho$. The boundary line $V = -\tan(\alpha) U$ makes an angle of $-\alpha$ with the positive $U$-axis. The first boundary line $U=0$ corresponds to an angle of $\pi/2$.

The intersection of these two half-planes forms a sector with an angle of:

$$
\frac{\pi}{2} + \alpha = \frac{\pi}{2} + \arcsin ho
$$

So the probability $p$ is:

$$
p = \frac{\frac{\pi}{2} + \arcsin ho}{2\pi} = \frac{1}{4} + \frac{\arcsin ho}{2\pi}
$$

### Final expectation result

Substitute the expression for $p$ back into the expectation formula:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = 4\left(\frac{1}{4} + \frac{\arcsin ho}{2\pi}\right) - 1
$$

Simplifying gives the final result:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)] = \frac{2}{\pi} \arcsin ho
$$

This result relies heavily on the rotational invariance (circular symmetry) of independent Gaussians and does not arbitrarily translate to other non-normal distributions that happen to share a correlation of $ ho$.

---

## 8 · Fast review

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
