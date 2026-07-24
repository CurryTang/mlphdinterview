# Quant 5 · Normal Distribution: Bivariate Normal, Cholesky, and Sign Correlation

This lecture revolves around a classic problem:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
$$

where $(X,Y)$ is a bivariate standard normal distribution with correlation coefficient $\rho$. The answer is:

$$
\boxed{
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
\frac{2}{\pi}\arcsin\rho
}
$$

The cleanest way to solve this is not by direct integration, but by transforming the correlated bivariate normal into two independent standard normals and then using circular symmetry to convert the probability into the angle of a sector.

---

## 1. Univariate Standard Normal

The density of a standard normal $N(0,1)$ is:

$$
\phi(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}
$$

It has three fundamental properties:

| Property | Meaning |
| --- | --- |
| Symmetric about 0 | $P(X>0)=P(X<0)=1/2$ |
| Mean is 0 | Positive and negative deviations cancel out |
| Variance is 1 | Used as a standard scale |

The probability of a continuous normal variable falling on an exact point is 0, so:

$$
P(X=0)=0
$$

Therefore, $\operatorname{sgn}(X)$ only needs to consider the sign.

---

## 2. Bivariate Standard Normal and Correlation Coefficient

If $(X,Y)$ is a bivariate standard normal, and:

$$
\mathbb{E}X=\mathbb{E}Y=0,
\qquad
\operatorname{Var}(X)=\operatorname{Var}(Y)=1,
\qquad
\operatorname{corr}(X,Y)=\rho
$$

Then its covariance matrix is:

$$
\Sigma=
\begin{pmatrix}
1 & \rho\\
\rho & 1
\end{pmatrix}
$$

$\rho$ controls the tilt of the ellipse:

| $\rho$ | Intuition |
| --- | --- |
| $\rho>0$ | Ellipse stretches along $y=x$; variables are more likely to have the same sign |
| $\rho=0$ | Circularly symmetric; variables are independent |
| $\rho<0$ | Ellipse stretches along $y=-x$; variables are more likely to have opposite signs |

A standard construction for a correlated bivariate normal is:

$$
U,V\overset{i.i.d.}{\sim}N(0,1),
\qquad U\perp V
$$

Define:

$$
X=U,\qquad
Y=\rho U+\sqrt{1-\rho^2}V
$$

Then $(X,Y)$ is a bivariate standard normal with correlation coefficient $\rho$.

<figure class="quant-svg-figure quant-svg-wide">
<svg viewBox="0 0 1060 420" role="img" aria-label="Cholesky transform from independent normal variables to correlated normal variables">
  <defs>
    <marker id="arrow-normal-1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#24485a" />
    </marker>
    <linearGradient id="normal-ellipse" x1="0" x2="1">
      <stop offset="0%" stop-color="#dff1ed" />
      <stop offset="100%" stop-color="#f8ead2" />
    </linearGradient>
    <filter id="normal-small-shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="7" stdDeviation="6" flood-color="#18313f" flood-opacity="0.09" />
    </filter>
  </defs>
  <rect x="18" y="18" width="1024" height="384" rx="18" fill="#fbfcfa" stroke="#d0dce0" />
  <text x="70" y="62" class="quant-svg-title">independent standard normal</text>
  <text x="680" y="62" class="quant-svg-title">correlated normal</text>
  <g transform="translate(248 220)">
    <line x1="-142" y1="0" x2="152" y2="0" stroke="#5f7b88" stroke-width="2.4" marker-end="url(#arrow-normal-1)" />
    <line x1="0" y1="140" x2="0" y2="-150" stroke="#5f7b88" stroke-width="2.4" marker-end="url(#arrow-normal-1)" />
    <circle cx="0" cy="0" r="108" fill="#e9f4f1" stroke="#2c6b7f" stroke-width="3.5" />
    <circle cx="0" cy="0" r="64" fill="none" stroke="#cfe1e1" stroke-width="2" />
    <line x1="-82" y1="82" x2="82" y2="-82" stroke="#e08d3c" stroke-width="5" stroke-linecap="round" />
    <text x="158" y="8" class="quant-svg-label">U</text>
    <text x="10" y="-152" class="quant-svg-label">V</text>
    <rect x="-122" y="122" width="244" height="38" rx="10" fill="#ffffff" stroke="#d6e2e5" />
    <text x="-103" y="146" class="quant-svg-note">level sets are circles</text>
  </g>
  <g transform="translate(530 214)" filter="url(#normal-small-shadow)">
    <rect x="-112" y="-58" width="224" height="116" rx="14" fill="#ffffff" stroke="#d4e0e3" />
    <line x1="-82" y1="-4" x2="82" y2="-4" stroke="#24485a" stroke-width="3.5" marker-end="url(#arrow-normal-1)" />
    <text x="-70" y="-26" class="quant-svg-label">Cholesky map</text>
    <text x="-82" y="35" class="quant-svg-formula">X = U</text>
    <text x="-82" y="58" class="quant-svg-formula">Y = rho U + sqrt(1-rho^2) V</text>
  </g>
  <g transform="translate(800 220)">
    <line x1="-152" y1="0" x2="164" y2="0" stroke="#5f7b88" stroke-width="2.4" marker-end="url(#arrow-normal-1)" />
    <line x1="0" y1="140" x2="0" y2="-150" stroke="#5f7b88" stroke-width="2.4" marker-end="url(#arrow-normal-1)" />
    <g transform="rotate(-28)">
      <ellipse cx="0" cy="0" rx="146" ry="68" fill="url(#normal-ellipse)" stroke="#2c6b7f" stroke-width="3.5" />
      <ellipse cx="0" cy="0" rx="86" ry="40" fill="none" stroke="#d7dfdd" stroke-width="2" />
      <line x1="-118" y1="0" x2="118" y2="0" stroke="#e08d3c" stroke-width="5" stroke-linecap="round" />
    </g>
    <text x="170" y="8" class="quant-svg-label">X</text>
    <text x="10" y="-152" class="quant-svg-label">Y</text>
    <rect x="-128" y="122" width="256" height="38" rx="10" fill="#ffffff" stroke="#d6e2e5" />
    <text x="-108" y="146" class="quant-svg-note">level sets become ellipses</text>
  </g>
</svg>
<figcaption>Independent standard normals are circularly symmetric in the (U,V) plane. The Cholesky linear transformation stretches the circular level sets into the ellipses of a correlated bivariate normal; the orange lines indicate the principal directions, and straight lines remain straight after the linear transformation.</figcaption>
</figure>

---

## 3. Why This Construction Is Correct

First, check the means:

$$
\mathbb{E}X=0,\qquad
\mathbb{E}Y=\rho\mathbb{E}U+\sqrt{1-\rho^2}\mathbb{E}V=0
$$

Next, check the variances:

$$
\operatorname{Var}(Y)
=
\rho^2\operatorname{Var}(U)
+
(1-\rho^2)\operatorname{Var}(V)
=
1
$$

The covariance is:

$$
\operatorname{Cov}(X,Y)
=
\operatorname{Cov}(U,\rho U+\sqrt{1-\rho^2}V)
=
\rho
$$

Because $U,V$ are independent, $\operatorname{Cov}(U,V)=0$. Since the variances of $X$ and $Y$ are both 1:

$$
\operatorname{corr}(X,Y)=\rho
$$

In matrix form:

$$
\begin{pmatrix}
X\\
Y
\end{pmatrix}
=
\begin{pmatrix}
1&0\\
\rho&\sqrt{1-\rho^2}
\end{pmatrix}
\begin{pmatrix}
U\\
V
\end{pmatrix}
$$

The relevant correlation matrix is:

$$
\begin{pmatrix}
1&\rho\\
\rho&1
\end{pmatrix}
$$

The factor shown in the previous equation is its Cholesky factor.

This step requires joint normality. Knowing only that $X$ and $Y$ are individually standard normal with correlation $\rho$ is not enough to derive this linear representation.

---

## 4. Converting the Product of Signs to Same-Sign Probability

Because $P(X=0)=P(Y=0)=0$:

```text
same sign:
  sgn(X)sgn(Y) = 1

opposite sign:
  sgn(X)sgn(Y) = -1
```

Therefore:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
P(\text{same sign})-P(\text{opposite sign})
$$

The total probability is 1, so:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
2P(\text{same sign})-1
$$

The bivariate normal is symmetric about the origin:

$$
P(X>0,Y>0)=P(X<0,Y<0)
$$

Let:

$$
p=P(X>0,Y>0)
$$

Then:

$$
P(\text{same sign})=2p
$$

So:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
4p-1
$$

---

## 5. Calculating $p$ Using the Independent Normal Plane

From the Cholesky representation:

$$
X=U,\qquad
Y=\rho U+\sqrt{1-\rho^2}V
$$

So:

$$
p
=
P(U>0,\ \rho U+\sqrt{1-\rho^2}V>0)
$$

Now consider the $(U,V)$ plane. Because $U,V$ are independent standard normals, their joint density is:

$$
f(u,v)=\frac1{2\pi}e^{-(u^2+v^2)/2}
$$

This density depends only on the radius:

$$
r=\sqrt{u^2+v^2}
$$

It does not depend on the angle. Therefore, for a sector region passing through the origin, the probability depends only on the angle of the sector:

$$
P((U,V)\text{ falls in a sector with angle }\theta)
=
\frac{\theta}{2\pi}
$$

---

## 6. Where Does the Sector Angle Come From?

The two conditions give two half-planes:

$$
U>0
$$

and:

$$
\rho U+\sqrt{1-\rho^2}V>0
$$

The boundary line of the second condition is:

$$
V=-\frac{\rho}{\sqrt{1-\rho^2}}U
$$

Let:

$$
\alpha=\arcsin\rho
$$

Then the angle between this boundary line and the positive $U$-axis is $-\alpha$. The first boundary $U=0$ corresponds to an angle of $\pi/2$. The angle of the sector formed by the intersection of the two half-planes is:

$$
\frac{\pi}{2}+\alpha
=
\frac{\pi}{2}+\arcsin\rho
$$

<figure class="quant-svg-figure quant-svg-wide">
<svg viewBox="0 0 1060 520" role="img" aria-label="Sector geometry for the probability P of X greater than zero and Y greater than zero">
  <defs>
    <marker id="arrow-normal-2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#24485a" />
    </marker>
    <filter id="normal-card-shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="8" stdDeviation="7" flood-color="#18313f" flood-opacity="0.10" />
    </filter>
    <linearGradient id="sector-fill" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f7c66f" stop-opacity="0.86" />
      <stop offset="100%" stop-color="#df8f38" stop-opacity="0.46" />
    </linearGradient>
  </defs>
  <rect x="18" y="18" width="1024" height="484" rx="18" fill="#fbfcfa" stroke="#d0dce0" />
  <text x="52" y="58" class="quant-svg-title">sector in the independent (U,V) plane</text>
  <text x="52" y="88" class="quant-svg-note">example: rho = 0.5, alpha = arcsin(rho) = 30 degrees</text>
  <g transform="translate(330 286)">
    <circle cx="0" cy="0" r="182" fill="#f8fbfa" stroke="#d7e3e6" stroke-width="2" />
    <circle cx="0" cy="0" r="122" fill="none" stroke="#e8eff1" stroke-width="2" />
    <circle cx="0" cy="0" r="64" fill="none" stroke="#eef3f4" stroke-width="2" />
    <path d="M0 0 L157.6 91 A182 182 0 0 0 0 -182 Z" fill="url(#sector-fill)" stroke="#d1812c" stroke-width="3" />
    <line x1="-214" y1="0" x2="222" y2="0" stroke="#52707d" stroke-width="2" marker-end="url(#arrow-normal-2)" />
    <line x1="0" y1="206" x2="0" y2="-218" stroke="#52707d" stroke-width="2" marker-end="url(#arrow-normal-2)" />
    <line x1="-188" y1="-108.5" x2="188" y2="108.5" stroke="#24485a" stroke-width="3" stroke-dasharray="10 8" />
    <line x1="0" y1="190" x2="0" y2="-190" stroke="#24485a" stroke-width="3" />
    <path d="M77.9 45 A90 90 0 0 0 0 -90" fill="none" stroke="#9b4c18" stroke-width="5" stroke-linecap="round" />
    <path d="M64 0 A64 64 0 0 1 55.4 32" fill="none" stroke="#9b4c18" stroke-width="3" />
    <rect x="45" y="-141" width="112" height="32" rx="8" fill="#fff8e8" stroke="#edc77f" />
    <text x="60" y="-120" class="quant-svg-note">U&gt;0, Y&gt;0</text>
    <rect x="15" y="-214" width="74" height="28" rx="7" fill="#ffffff" stroke="#d0dce0" />
    <text x="27" y="-195" class="quant-svg-label">U = 0</text>
    <rect x="-190" y="-143" width="74" height="28" rx="7" fill="#ffffff" stroke="#d0dce0" />
    <text x="-178" y="-124" class="quant-svg-label">Y = 0</text>
    <text x="230" y="6" class="quant-svg-label">U</text>
    <text x="8" y="-225" class="quant-svg-label">V</text>
    <text x="45" y="-70" class="quant-svg-formula">pi/2 + alpha</text>
    <text x="76" y="52" class="quant-svg-formula">-alpha</text>
  </g>
  <g transform="translate(628 130)" filter="url(#normal-card-shadow)">
    <rect x="0" y="0" width="355" height="290" rx="16" fill="#ffffff" stroke="#d4e0e3" />
    <text x="24" y="42" class="quant-svg-label">What the picture is doing</text>
    <circle cx="33" cy="82" r="5" fill="#24485a" />
    <text x="52" y="87" class="quant-svg-note">U = 0 keeps the right half-plane.</text>
    <circle cx="33" cy="120" r="5" fill="#24485a" />
    <text x="52" y="125" class="quant-svg-note">Y = 0 is the slanted dashed line.</text>
    <circle cx="33" cy="158" r="5" fill="#d1812c" />
    <text x="52" y="163" class="quant-svg-note">The overlap is the shaded sector.</text>
    <line x1="24" y1="188" x2="331" y2="188" stroke="#e5edef" />
    <text x="24" y="222" class="quant-svg-formula">sector angle = pi/2 + alpha</text>
    <text x="24" y="252" class="quant-svg-formula">p = (pi/2 + alpha) / (2pi)</text>
    <text x="24" y="282" class="quant-svg-formula">alpha = arcsin(rho)</text>
  </g>
</svg>
<figcaption>In the (U,V) plane, the joint density is circularly symmetric. The intersection of U &gt; 0 and Y &gt; 0 is a sector, and the probability is equal to the sector angle divided by 2π.</figcaption>
</figure>

So:

$$
p
=
\frac{\frac{\pi}{2}+\arcsin\rho}{2\pi}
=
\frac14+\frac{\arcsin\rho}{2\pi}
$$

---

## 7. Substituting Back into the Sign Product Expectation

We previously obtained:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
4p-1
$$

Substituting:

$$
p=\frac14+\frac{\arcsin\rho}{2\pi}
$$

We get:

$$
\mathbb{E}[\operatorname{sgn}(X)\operatorname{sgn}(Y)]
=
4\left(\frac14+\frac{\arcsin\rho}{2\pi}\right)-1
=
\frac{2}{\pi}\arcsin\rho
$$

---

## 8. Special Value Check

| $\rho$ | Case | Formula Result |
| --- | --- | --- |
| $0$ | Independent, same and opposite signs are equally likely | $0$ |
| $1$ | $Y=X$, always same sign | $1$ |
| $-1$ | $Y=-X$, always opposite sign | $-1$ |

Substituting into the formula:

$$
\frac{2}{\pi}\arcsin 0=0
$$

$$
\frac{2}{\pi}\arcsin 1=1
$$

$$
\frac{2}{\pi}\arcsin(-1)=-1
$$

All are consistent with intuition.

---

## 9. Structural Summary

The entire derivation relies on two points.

First, a correlated bivariate normal can be written as:

$$
X=U,\qquad
Y=\rho U+\sqrt{1-\rho^2}V
$$

Second, the independent standard normal plane is circularly symmetric. The probability of any sector passing through the origin is determined solely by its angle.
