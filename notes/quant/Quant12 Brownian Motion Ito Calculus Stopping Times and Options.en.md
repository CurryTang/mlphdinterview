# Quant 12 · Brownian Motion, Itô Calculus, Stopping Times & Option Trading

This tutorial systematically unpacks the core continuous-time stochastic analysis framework in quantitative finance: from the continuous limit of 1D and 2D random walks (Brownian motion and Donsker's Invariance Principle), to sample path geometric irregularities (infinite first-order variation and finite quadratic variation), to Itô's Lemma and the structural divergence between Itô and Stratonovich integrals, continuous-time stopping times, exponential martingales, and the Reflection Principle, and finally to Geometric Brownian Motion, Black-Scholes-Merton PDE, Delta hedging Gamma PnL, and volatility signature arbitrage.

```text
Core Mental Model & Roadmap:
1. Path Geometry: Brownian motion W_t is almost surely continuous everywhere but nowhere differentiable. Its total variation is infinite, while its quadratic variation is deterministic [W]_t = t. Ordinary Riemann-Stieltjes calculus collapses; one must Taylor-expand to order (dW_t)^2 = dt.
2. 2D Random Walk & Donsker Limit: Discrete grid walks weakly converge to 2D Brownian motion. Discrete 2D walk is recurrent (Pólya's theorem). Continuous 2D Brownian motion is point-transient (never hits an exact pre-assigned point, capacity 0) but neighborhood-recurrent (visits any open ball B_eps(x) infinitely often), and possesses conformal invariance under analytic maps (Paul Lévy's Theorem).
3. Itô vs. Stratonovich Geometry: Itô evaluates at left endpoints (non-anticipating/adapted), maintaining the martingale property with E[I_t] = 0 (essential for financial causality and no-arbitrage). Stratonovich evaluates at midpoints, preserving standard chain rules but injecting an anticipating drift correction of +(1/2) dt.
4. Stopping Times & Reflection Principle: For the hitting time \tau_a = inf{t: W_t = a}, strong Markov property and spatial mirror symmetry yield P(M_t >= a) = 2 P(W_t >= a), leading directly to the Lévy distribution for first passage times.
5. Option Hedging & Volatility Signature: Delta hedging eliminates first-order directional risk dW_t. The instantaneous hedging PnL is d\Pi = (1/2) S^2 \Gamma (\sigma_{realized}^2 - \sigma_{implied}^2) dt - r \Pi dt. Long Gamma is fundamentally a bet that realized volatility will exceed implied volatility.
```

---

## Module 1: Basics & Path Geometry of Brownian Motion

### 1. Axiomatic Characterization

A standard one-dimensional Brownian motion (Wiener process) $\{W_t\}_{t \ge 0}$ defined on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$ satisfies:

$$
\begin{aligned}
\text{(i) Starting Point:} &\ W_0 = 0 \text{ almost surely} \\
\text{(ii) Independent Increments:} &\ \forall 0 \le t_0 < t_1 < \dots < t_n,\ \text{increments } W_{t_1}-W_{t_0}, \dots, W_{t_n}-W_{t_{n-1}} \text{ are mutually independent} \\
\text{(iii) Stationary Gaussian Increments:} &\ \forall 0 \le s < t,\ W_t - W_s \sim \mathcal{N}(0, t - s) \\
\text{(iv) Sample Path Continuity:} &\ t \mapsto W_t(\omega) \text{ is continuous for almost all } \omega
\end{aligned}
$$

The covariance kernel for any $s, t \ge 0$ (assuming $s \le t$) is:

$$
\operatorname{Cov}(W_s, W_t) = \mathbb{E}[W_s W_t] = \mathbb{E}[W_s (W_s + W_t - W_s)] = \mathbb{E}[W_s^2] + 0 = s = \min(s, t)
$$

### 2. Path Geometry: Nowhere Differentiability and Quadratic Variation

For a partition $\Pi_n: 0 = t_0 < t_1 < \dots < t_n = T$ with mesh size $|\Pi_n| = \max_i |t_i - t_{i-1}| \to 0$:

#### (1) First-Order Total Variation Diverges to Infinity

$$
\operatorname{TV}_T(W) = \lim_{|\Pi_n| \to 0} \sum_{i=1}^n |W_{t_i} - W_{t_{i-1}}| = \infty \quad \text{almost surely}
$$

**Proof sketch**: Let $\Delta W_i = W_{t_i} - W_{t_{i-1}} = \sqrt{\Delta t_i} Z_i$ ($Z_i \sim \mathcal{N}(0, 1)$). Then $\mathbb{E}[|\Delta W_i|] = \sqrt{\frac{2}{\pi}} \sqrt{\Delta t_i}$. For uniform partitions $\Delta t = T/n$:

$$
\mathbb{E}\left[ \sum_{i=1}^n |\Delta W_i| \right] = n \cdot \sqrt{\frac{2}{\pi}} \sqrt{\frac{T}{n}} = \sqrt{\frac{2}{\pi}} \sqrt{T} \sqrt{n} \xrightarrow{n \to \infty} \infty
$$

#### (2) Quadratic Variation Converges Deterministically to $T$

$$
[W]_T = \lim_{|\Pi_n| \to 0} \sum_{i=1}^n (W_{t_i} - W_{t_{i-1}})^2 = T \quad \text{in } L^2 \text{ and in probability}
$$

**Proof**: Let $Q_n = \sum_{i=1}^n (\Delta W_i)^2$. The expectation is $\mathbb{E}[Q_n] = \sum \Delta t_i = T$. The variance is:

$$
\operatorname{Var}(Q_n) = \sum_{i=1}^n (\Delta t_i)^2 \operatorname{Var}(Z_i^2) = 2 \sum_{i=1}^n (\Delta t_i)^2 \le 2 |\Pi_n| \sum \Delta t_i = 2 |\Pi_n| T \xrightarrow{|\Pi_n| \to 0} 0
$$

In differential notation:

$$
(dW_t)^2 = dt
$$

> **Key Quant Takeaway**: Because $(dW_t)^2 = dt$ has the same first-order scale as $dt$, any Taylor expansion of a function along a Brownian path must include second derivatives.

```brownian-motion-demo
```

### 3. Symmetries and Invariant Transformations

1. **Brownian Scaling**: $\forall c > 0$, $X_t = \frac{1}{\sqrt{c}} W_{ct}$ is a standard Brownian motion.
2. **Time Inversion**: $Y_t = t W_{1/t}$ (with $Y_0 = 0$) is a standard Brownian motion.
3. **Spatial Reflection**: $Z_t = -W_t$ is a standard Brownian motion.

### 4. Transition Density & Heat Equation

The transition probability density from $x$ to $y$ across time $t$ is:

$$
p(t, x, y) = \frac{1}{\sqrt{2\pi t}} \exp\left( -\frac{(y-x)^2}{2t} \right)
$$

It directly satisfies the Heat Equation / Kolmogorov backward & forward PDE:

$$
\frac{\partial p}{\partial t} = \frac{1}{2} \frac{\partial^2 p}{\partial x^2} \quad (\text{Backward}) \qquad \frac{\partial p}{\partial t} = \frac{1}{2} \frac{\partial^2 p}{\partial y^2} \quad (\text{Forward / Fokker-Planck})
$$

---

## Module 2: 2D Random Walk & Brownian Motion

### 1. Donsker's Invariance Principle (Functional Central Limit Theorem)

Let $\xi_1, \xi_2, \dots$ be i.i.d. step vectors on the discrete integer lattice $\mathbb{Z}^2$, choosing each cardinal direction with probability $1/4$: $\mathbb{E}[\xi_i] = (0, 0)$, $\operatorname{Cov}(\xi_i) = \frac{1}{2} I_2$. Construct the piecewise linear random walk $S_k = \sum_{i=1}^k \xi_i$ and the rescaled process:

$$
B_N(t) = \frac{1}{\sqrt{N}} S_{\lfloor Nt \rfloor}
$$

Donsker's Invariance Principle states that as $N \to \infty$, $B_N(\cdot)$ converges weakly in $C([0, T], \mathbb{R}^2)$ to standard two-dimensional Brownian motion:

$$
B_N(t) \implies \left( \frac{1}{\sqrt{2}} B_t^{(1)}, \frac{1}{\sqrt{2}} B_t^{(2)} \right)
$$

```two-d-walk-demo
```

### 2. Recurrence vs. Transience: Pólya's Theorem and Topological Finesse

| Spatial Dimension $d$ | Discrete Grid Walk ($\mathbb{Z}^d$) | Continuous Brownian Motion ($\mathbb{R}^d$) | Properties |
| :--- | :--- | :--- | :--- |
| **$d = 1$** | **Recurrent** | **Recurrent** | Returns to origin almost surely, $\mathbb{E}[\tau] = \infty$ (null recurrent) |
| **$d = 2$** | **Recurrent** | **Neighborhood Recurrent, Point Transient** | Discrete: $P(\text{return}) = 1$; Continuous: $P(\exists t>0: B_t = 0) = 0$, but enters any open disk $B_\epsilon(x)$ infinitely often |
| **$d \ge 3$** | **Transient** | **Transient** | Discrete $d=3$: $P(\text{return}) \approx 0.340537$; Continuous: $\lim_{t\to\infty} \|B_t\| = \infty$ a.s. |

### 3. Conformal Invariance (Paul Lévy's Theorem)

Let $Z_t = B_t^{(1)} + i B_t^{(2)}$ be standard complex Brownian motion, and let $f: U \to V$ be a non-degenerate holomorphic (analytic) function. Then:

$$
f(Z_t) = \widetilde{Z}_{\tau_t}
$$

where $\widetilde{Z}$ is another standard complex Brownian motion and $\tau_t = \int_0^t |f'(Z_s)|^2 ds$ is a strictly increasing time change. Conformal mappings preserve the Brownian motion structure, enabling geometric transformations of complex boundary value problems.

---

## Module 3: Itô Calculus & Itô Geometry (Intuition First)

### 1. Historical Evolution & Timeline: From Pollen Grains to Modern Quantitative Finance

Stochastic calculus is not an abstract game of pure symbols; it is a profound revolution born out of physical particle diffusion and forged into the modern language of global quantitative trading:

| Year | Key Pioneer | Milestone Contribution & Publication | Significance to Quantitative Finance & Stochastic Analysis |
| :---: | :--- | :--- | :--- |
| **1827** | **Robert Brown** | Microscopic observation of continuous chaotic jittering of pollen grains in water | First physical discovery of macroscopic continuous stochastic motion |
| **1900** | **Louis Bachelier** | PhD thesis *Théorie de la Spéculation* models Paris stock exchange options via Brownian motion | **Predated Einstein by 5 years** in modeling asset price diffusion; father of mathematical finance |
| **1905** | **Albert Einstein** | Derives macroscopic diffusion PDE from molecular collisions: $\mathbb{E}[(\Delta x)^2] = 2Dt$ | Proved physics of diffusion and enabled experimental measurement of Avogadro's number |
| **1923** | **Norbert Wiener** | Establishes rigorous Wiener measure on path space $C[0, \infty)$ | Rigorously proved that Brownian sample paths are **continuous everywhere, nowhere differentiable** |
| **1944–1951** | **Kiyosi Itô** | Constructs non-anticipating stochastic integral via $L^2$ isometry & derives **Itô's Lemma** | Solved the breakdown of classical calculus on rough paths, creating modern SDE theory |
| **1973** | **Black-Scholes-Merton** | Replaces arithmetic BM with GBM; uses Itô's Lemma to build self-financing Delta hedges | Eliminated market risk to derive the Nobel Prize-winning option pricing & dynamic hedging formula |

---

### 2. What is a Diffusion Process & Itô Diffusion? (Foundations)

In quantitative finance and stochastic calculus, "**Diffusion Process**" and "**Itô Diffusion**" are fundamental concepts that are often used loosely. Here is the precise physical, mathematical, and financial breakdown:

#### (1) Physical Intuition: What is a Basic Diffusion?
* **Real-World Analogy**: When a drop of ink is released into still water, microscopic water molecules bombard ink particles in chaotic thermal motion (Brownian motion), causing ink to spread out from high to low concentration (Fick's laws).
* **Microscopic vs. Macroscopic Duality**:
  - **Microscopic Level (Single Particle)**: Follows a jagged **Brownian motion / random walk** $W_t$;
  - **Macroscopic Level (Probability Density / Concentration $\rho(t, x)$)**: Governed by the deterministic **Heat / Diffusion PDE**:
    $$\frac{\partial \rho}{\partial t} = \frac{1}{2}\sigma^2 \frac{\partial^2 \rho}{\partial x^2}$$
* **Core Essence of Diffusion**: **The systematic expansion of uncertainty (variance) over time**. For standard Brownian motion, variance expands linearly with time: $\operatorname{Var}(W_t) = t$.

---

#### (2) Mathematical Formulation: The Drift-Diffusion Decomposition
Asset prices in real financial markets are neither deterministic straight lines nor pure zero-trend noise; they are a continuous mixture of **deterministic trend + random fluctuations**.

Any continuous-path continuous-state Markov process can be decomposed into a **Drift-Diffusion Stochastic Differential Equation (SDE)**:

$$\boxed{dX_t = \underbrace{\mu(t, X_t) dt}_{\text{Deterministic Drift Term}} + \underbrace{\sigma(t, X_t) dW_t}_{\text{Stochastic Diffusion Term}}}$$

* **Drift Term ($\mu(t, X_t)$)**:
  - **Physics**: External forces, gravity, fluid flow pulling the particle in a deterministic direction. If noise is turned off ($\sigma=0$), it reduces to an ODE $\frac{dX_t}{dt} = \mu(t, X_t)$;
  - **Finance**: The expected return, risk-free interest rate drift, or **mean-reverting gravitational pull**.
* **Diffusion Term ($\sigma(t, X_t)$)**:
  - **Physics**: The intensity/amplitude of microscopic thermal noise collisions;
  - **Finance**: The asset's **instantaneous volatility**, representing market price vibration per unit time.

---

#### (3) Mathematical Definition: What is an "Itô Diffusion"?
A stochastic process $X = \{X_t : t \ge 0\}$ is rigorously defined as an **Itô Diffusion** if it satisfies three foundational pillars:
1. **Driven by Brownian SDE**: It is the strong/weak solution to the integral equation:
   $$X_t = X_0 + \int_0^t \mu(s, X_s) ds + \int_0^t \sigma(s, X_s) dW_s$$
   where the second integral is a non-anticipating **Itô Stochastic Integral**;
2. **Strong Markov Property**: The process is "memoryless". Conditioned on the present state $X_\tau$ at stopping time $\tau$, the future evolution is completely independent of the past path history;
3. **Continuous Sample Paths**: Almost all trajectories $t \mapsto X_t(\omega)$ are continuous functions (**No Jumps**). If discontinuous jumps exist, it becomes an Itô-Lévy jump-diffusion.

---

#### (4) Master Cheatsheet: 5 Iconic Itô Diffusions in Quantitative Finance

| Model Name | Stochastic Differential Equation (SDE) | Drift $\mu(X_t)$ | Diffusion $\sigma(X_t)$ | Key Quantitative Application |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Brownian Motion** | $dX_t = dW_t$ | $0$ (Zero drift) | $1$ (Unit diffusion) | Pure noise baseline, Martingale pricing foundation |
| **Arithmetic BM with Drift** | $dX_t = \mu dt + \sigma dW_t$ | $\mu$ (Constant) | $\sigma$ (Constant) | Bachelier option pricing model, short-term spread dynamics |
| **Geometric Brownian Motion (GBM)** | $dS_t = \mu S_t dt + \sigma S_t dW_t$ | $\mu S_t$ (Linear in price) | $\sigma S_t$ (Percentage volatility) | **Black-Scholes Stock & Index Option Model** |
| **Ornstein-Uhlenbeck (OU) Process** | $dX_t = \theta(\mu - X_t) dt + \sigma dW_t$ | $\theta(\mu - X_t)$ (Mean-reversion pull) | $\sigma$ (Constant) | **Vasicek Short Rate Model**, Statistical Arbitrage Pairs Trading |
| **Cox-Ingersoll-Ross (CIR) Process** | $dr_t = k(\theta - r_t) dt + \sigma \sqrt{r_t} dW_t$ | $k(\theta - r_t)$ (Mean-reversion pull) | $\sigma \sqrt{r_t}$ (Vanishes at 0, strictly non-negative) | **CIR Interest Rate Model**, **Heston Stochastic Volatility Model** |

---

### 3. Why Classical Calculus Fails: 3 Core Intuitive & Physical Mental Models

#### Mental Model 1: Why does $(dW_t)^2 = dt$ become a 100% deterministic constant?
> **The Coin Toss & Step Size Intuition**:
> Imagine tossing a coin every $\Delta t$ seconds. To make the total variance after 1 second equal to 1, the step size cannot be $\pm \Delta t$ (otherwise variance after $N = 1/\Delta t$ steps would vanish to 0). **The step size must be $\pm \sqrt{\Delta t}$**!
> 
> Now look at the squared step:
> $$(\pm \sqrt{\Delta t})^2 = +\Delta t$$
> **Regardless of whether the coin lands Heads ($+1$) or Tails ($-1$), squaring wipes out the sign and completely erases all randomness!**
> 
> Summing up all microscopic squared steps over 1 second:
> $$\sum_{i=1}^{1/\Delta t} (\Delta W_i)^2 = \left(\frac{1}{\Delta t}\right) \times \Delta t = 1 \text{ second}$$
> 
> **Core Insight**: On an infinitesimal scale, the square of random noise is a **100% deterministic, vibration-free constant flow of time**! $(dW_t)^2 = dt$ is not an approximation—it is an exact law of nature!

---

#### Mental Model 2: The Parabolic Bowl (Why is there a $+\frac{1}{2} f''(x) dt$ correction?)
> **The Bowl Bottom Intuition**:
> Imagine standing at the bottom of a parabolic bowl $f(x) = x^2$ at $x=0$:
> - A symmetric gust of wind pushes you randomly Left ($1$ meter) or Right ($1$ meter) with $50/50$ probability;
> - **Horizontally**: Your average displacement is $\frac{+1 + (-1)}{2} = 0$ (no drift);
> - **Vertically (Height)**: Moving Right lifts you up $(+1)^2 = 1$ meter; moving Left **also lifts you up $(-1)^2 = 1$ meter**!
> - **Result**: Even though horizontal wind is fair pure noise, the upward curvature of the bowl (convexity $f''(x) > 0$) **systematically pumps your average height upward by 1 meter on every jitter!**
> 
> This is **Jensen's Inequality operating in real-time continuous dynamics**:
> $$\mathbb{E}[f(X + \Delta W)] - f(X) \approx \underbrace{f'(X)\mathbb{E}[\Delta W]}_{= 0} + \frac{1}{2} f''(X) \underbrace{\mathbb{E}[(\Delta W)^2]}_{= \Delta t} = \frac{1}{2} f''(X) \Delta t$$
> 
> - The curvier the bowl ($f''(x)$), the more violent the jitter ($\sigma^2$), the faster the upward drift!
> - Under a concave ceiling ($f''(x) < 0$), symmetric noise systematically drags your expected value downward!

---

#### Mental Model 3: Volatility Drag ($-\frac{1}{2}\sigma^2 dt$: How Volatility Destroys Compound Growth)
> **Practical Trading & Portfolio Intuition**:
> Why is the actual geometric compounding return always lower than the arithmetic return by $-\frac{1}{2}\sigma^2$?
> 
> Consider a classic leveraged investment:
> - Day 1: Asset surges $+50\%$ ($100 \to 150$);
> - Day 2: Asset crashes $-50\%$ ($150 \to 75$);
> - **Arithmetic Average Return**: $\frac{+50\% - 50\%}{2} = 0\%$ (looks like break-even);
> - **Actual Portfolio Value**: Drops from $\$100$ to $\$75$—a net **$-25\%$ loss**!
> 
> That phantom $-25\%$ loss is discrete **Volatility Drag**!
> In continuous stochastic calculus, the logarithm $f(S) = \ln S$ is strictly concave ($f''(S) = -1/S^2 < 0$). Under continuous Brownian jitter, Itô's Lemma generates the exact geometric downward drag of **$-\frac{1}{2}\sigma^2 dt$**!

---

### 4. Itô's Lemma: Multiplication Table & 3-Step Interview Rule

For an Itô diffusion: $dX_t = \mu(t, X_t) dt + \sigma(t, X_t) dW_t$.


#### Itô Multiplication Table (Only Noise $\times$ Noise produces Time)

| $\times$ | $dt$ | $dW_t$ | Memory Intuition |
| :---: | :---: | :---: | :--- |
| **$dt$** | $0$ | $0$ | $dt \cdot dt \sim O(dt^2)$, negligible |
| **$dW_t$** | $0$ | **$dt$** | Two $\sqrt{dt}$ terms multiply to first-order time $dt$ |

#### Rapid 3-Step Recipe (Solve any derivative drift in 3 seconds):
1. **Step 1 (Standard Newton Differential)**: $\frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t$;
2. **Step 2 (Add the Itô Curvature Kick)**: $+\frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2$;
3. **Step 3 (Substitute $(dX_t)^2 = \sigma^2 dt$ & Collect $dt$ terms)**:

$$\boxed{df(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x} dW_t}$$

---

### 5. Itô vs. Stratonovich Integrals: Master Guide, Conversions & the Wong-Zakai Theorem

In the historical development of stochastic calculus, the choice of evaluation point in the stochastic Riemann sum sparked two major mathematical schools and practical paradigms: the **Itô Integral** and the **Stratonovich Integral**.

```ito-geometry-demo
```

---

#### (1) Historical Background & Philosophical Division

* **The Kiyosi Itô Camp (1944 · Martingale Property & Temporal Causality First)**:
  - **Motivation**: Create a rigorous measure-theoretic stochastic calculus for Markov processes and physical diffusions.
  - **Core Philosophy**: **Time flows forward; looking into the future is strictly forbidden (No Anticipation)**. At time $t$, trading decisions or physical forces can only depend on filtration $\mathcal{F}_t$.
  - **Trade-off**: Sacrifices the standard Newton-Leibniz chain rule, requiring the second-order curvature correction in **Itô's Lemma**.
* **The Ruslan Stratonovich Camp (1966 & D. L. Fisk 1963 · Geometric Symmetry & Chain Rule First)**:
  - **Motivation**: Provide a stochastic calculus for physical dynamical systems, robotics, and differential geometry that preserves classical calculus rules.
  - **Core Philosophy**: **Calculus must be coordinate-invariant on smooth manifolds**, requiring symmetric midpoint evaluations.
  - **Trade-off**: **Loses the Martingale property**, creating phantom drift out of pure symmetric white noise.

---

#### (2) Mathematical Formulations: Left Endpoint vs. Midpoint Riemann Sums

Partition the interval $[0, T]$ as $0 = t_0 < t_1 < \dots < t_n = T$ with mesh size $|\Pi| = \max_i |t_{i+1} - t_i| \to 0$. Introduce evaluation point $\tau_i = (1-\alpha)t_i + \alpha t_{i+1}$ for $\alpha \in [0, 1]$:

$$S_n^{(\alpha)} = \sum_{i=0}^{n-1} X_{\tau_i} (W_{t_{i+1}} - W_{t_i})$$

```mermaid
graph TD
    subgraph "Parameterized Stochastic Riemann Sum S_n^(α)"
    A["Interval Element [t_i, t_{i+1}]"] --> B["α = 0 (Left Endpoint) : X(t_i) · ΔW_i<br/><b>【Itô Integral】</b><br/>Non-anticipating · Preserves Martingale Property E[I]=0"]
    A --> C["α = 1/2 (Midpoint Average) : ½[X(t_i) + X(t_{i+1})] · ΔW_i<br/><b>【Stratonovich Integral】</b><br/>Symmetric · Preserves Standard Chain Rule"]
    A --> D["α = 1 (Right Endpoint) : X(t_{i+1}) · ΔW_i<br/><b>【Backward Itô Integral】</b><br/>Full look-ahead bias"]
    end
```

* **Itô Integral ($\alpha = 0$, denoted $\int_0^T X_t dW_t$)**:
  $$\int_0^T X_t dW_t \triangleq \lim_{|\Pi| \to 0} \sum_{i=0}^{n-1} X_{t_i} (W_{t_{i+1}} - W_{t_i})$$
  - The integrand $X_{t_i}$ is evaluated at the **left endpoint**, making it $\mathcal{F}_{t_i}$-measurable and **completely independent** of the forward increment $\Delta W_i = W_{t_{i+1}} - W_{t_i}$.
* **Stratonovich Integral ($\alpha = 1/2$, denoted $\int_0^T X_t \circ dW_t$)**:
  $$\int_0^T X_t \circ dW_t \triangleq \lim_{|\Pi| \to 0} \sum_{i=0}^{n-1} \left( \frac{X_{t_i} + X_{t_{i+1}}}{2} \right) (W_{t_{i+1}} - W_{t_i})$$
  - The integrand is evaluated at the **trapezoidal midpoint**, creating an endogenous correlation between the integrand and future Brownian increment $\Delta W_i$.

---

#### (3) The Definitive Calculation Comparison: $\int_0^T W_t dW_t$ vs. $\int_0^T W_t \circ dW_t$

##### 1. Itô Integral Calculation:
Using $W_{t_i}(W_{t_{i+1}} - W_{t_i}) = \frac{1}{2}(W_{t_{i+1}}^2 - W_{t_i}^2) - \frac{1}{2}(W_{t_{i+1}} - W_{t_i})^2$:
$$\sum_{i=0}^{n-1} W_{t_i} \Delta W_i = \frac{1}{2} \underbrace{\sum_{i=0}^{n-1} (W_{t_{i+1}}^2 - W_{t_i}^2)}_{\text{Telescoping Sum} = W_T^2 - W_0^2 = W_T^2} - \frac{1}{2} \underbrace{\sum_{i=0}^{n-1} (\Delta W_i)^2}_{\text{Quadratic Variation} \to T}$$
$$\boxed{\int_0^T W_t dW_t = \frac{1}{2} W_T^2 - \frac{1}{2} T}$$
* **Expectation Check (Martingale Property)**:
  $$\mathbb{E}\left[ \int_0^T W_t dW_t \right] = \frac{1}{2} \mathbb{E}[W_T^2] - \frac{1}{2}T = \frac{1}{2}T - \frac{1}{2}T = \mathbf{0}$$
  Expected profit from pure noise is strictly zero (no free lunch).

##### 2. Stratonovich Integral Calculation:
$$\sum_{i=0}^{n-1} \left( \frac{W_{t_i} + W_{t_{i+1}}}{2} \right) (W_{t_{i+1}} - W_{t_i}) = \frac{1}{2} \sum_{i=0}^{n-1} (W_{t_{i+1}}^2 - W_{t_i}^2) = \frac{1}{2} W_T^2$$
$$\boxed{\int_0^T W_t \circ dW_t = \frac{1}{2} W_T^2}$$
* **Standard Classical Form**: Identical to Newton calculus $\int x dx = \frac{1}{2}x^2$.
* **Expectation Trap (Loses Martingale Property)**:
  $$\mathbb{E}\left[ \int_0^T W_t \circ dW_t \right] = \frac{1}{2} \mathbb{E}[W_T^2] = \mathbf{\frac{1}{2} T \ne 0}$$
  **Manufactures a fictitious positive deterministic drift of $+\frac{1}{2}T$ out of pure zero-mean noise!**

---

#### (4) Conversion Formulas & the Drift Correction

Itô and Stratonovich calculus are connected by an exact **algebraic conversion identity**:

$$\boxed{\int_0^T X_t \circ dW_t = \int_0^T X_t dW_t + \frac{1}{2} [X, W]_T}$$

For semimartingale $dX_t = \mu_t dt + \sigma_t dW_t$, the quadratic covariation is $d[X, W]_t = \sigma_t dt$.

##### 1. Differential Level:
$$X_t \circ dW_t = X_t dW_t + \frac{1}{2} \sigma_t dt$$

##### 2. SDE Level Conversion:
* **Stratonovich SDE $\to$ Itô SDE**:
  Given Stratonovich SDE: $dX_t = \underline{b}(X_t) dt + \sigma(X_t) \circ dW_t$
  The equivalent Itô SDE adds the **Wong-Zakai drift correction**:
  $$\boxed{dX_t = \left( \underline{b}(X_t) + \frac{1}{2} \sigma(X_t) \sigma'(X_t) \right) dt + \sigma(X_t) dW_t}$$
* **Itô SDE $\to$ Stratonovich SDE**:
  Given Itô SDE: $dX_t = \mu(X_t) dt + \sigma(X_t) dW_t$
  The equivalent Stratonovich SDE is:
  $$\boxed{dX_t = \left( \mu(X_t) - \frac{1}{2} \sigma(X_t) \sigma'(X_t) \right) dt + \sigma(X_t) \circ dW_t}$$

---

#### (5) Wong-Zakai Theorem: Why Physics Prefers Stratonovich

Ideal mathematical "white noise" (correlation time $\tau = 0$) does not exist in nature; all physical disturbances are **smooth colored noise** with tiny correlation time $\epsilon > 0$.

> **Wong-Zakai Theorem (1965)**:
> Let $W_t^{(\epsilon)}$ be a smooth, differentiable approximation of Brownian motion (e.g., via an Ornstein-Uhlenbeck mollifier).
> Consider the standard ODE driven by smooth noise:
> $$\frac{dX_t^{(\epsilon)}}{dt} = b(X_t^{(\epsilon)}) + \sigma(X_t^{(\epsilon)}) \dot{W}_t^{(\epsilon)}$$
> As correlation time $\epsilon \to 0$, the solution $X_t^{(\epsilon)}$ **converges in probability to the Stratonovich SDE, NEVER the Itô SDE!**
> $$\lim_{\epsilon \to 0} X_t^{(\epsilon)} = X_t^{\text{Stratonovich}}$$

* **Physical Intuition**:
  Physical bodies possess inertia and cannot respond instantaneously with zero delay. Physical damping naturally smooths microscopic collisions into an average midpoint response.

---

#### (6) Comprehensive Comparison Matrix (Master Table)

| Dimension | Itô Calculus | Stratonovich Calculus |
| :--- | :--- | :--- |
| **Sampling Point $\alpha$** | $\alpha = 0$ (Left endpoint) | $\alpha = 1/2$ (Trapezoidal midpoint) |
| **Information Filtration** | $\mathcal{F}_{t_i}$-adapted, **Non-anticipating (Strict Causality)** | Evaluates $X_{t_{i+1}}$, contains forward look-ahead |
| **Chain Rule Form** | **Second-Order Itô's Lemma**: $d(f) = f' dX + \frac{1}{2}f''\sigma^2 dt$ | **Classical Newton Chain Rule**: $d(f) = f'(X) \circ dX$ |
| **Martingale Property** | **Strictly Preserved**: $\mathbb{E}\left[\int H dW\right] = 0$ | **Violated**: $\mathbb{E}\left[\int W \circ dW\right] = \frac{T}{2} \ne 0$ |
| **Manifold Invariance** | Second-order drift breaks tensor covariance | **Preserves Lie group and Riemannian symmetries** |
| **Numerical Schemes** | Euler-Maruyama Scheme | Heun's Method / Stratonovich Runge-Kutta |
| **Primary Domain** | **Quantitative Trading, Derivatives Pricing, Risk Hedging** | **Statistical Physics, Robotics, Manifold Navigation** |

---

#### (7) Quant Interview Classic Questions & Traps

> **Question 1: What fatal flaw happens if a backtesting engine uses Stratonovich calculus for high-frequency market making?**
> * **Answer**:
>   It creates a **fictitious "infinite money glitch"**! Stratonovich discrete sampling peeks ahead half a tick. On pure zero-mean noise, backtests will report a guaranteed riskless profit of $+\frac{1}{2}\sigma^2 dt$ per second, which will catastrophically collapse in live execution.
>
> **Question 2: Convert the standard GBM Itô SDE $dS_t = \mu S_t dt + \sigma S_t dW_t$ into Stratonovich form.**
> * **Derivation**:
>   - Diffusion term $\sigma(S) = \sigma S \implies \sigma'(S) = \sigma$;
>   - Correction term: $\frac{1}{2}\sigma(S)\sigma'(S) = \frac{1}{2}(\sigma S)(\sigma) = \frac{1}{2}\sigma^2 S$;
>   - Stratonovich drift subtracts this term: $\underline{b}(S) = (\mu - \frac{1}{2}\sigma^2) S$;
>   - **Final Stratonovich SDE**:
>     $$\boxed{dS_t = \left(\mu - \frac{1}{2}\sigma^2\right) S_t dt + \sigma S_t \circ dW_t}$$


---

## Module 4: Classic Applications & Analytical Tools of Itô Calculus

Having established the foundational theory of Itô stochastic calculus and Itô's Lemma, this module examines three quintessential applications in financial mathematics and quantitative modeling. These applications demonstrate how the second-order curvature term $\frac{1}{2}\sigma^2 f''(x) dt$ governs asset dynamics, option risk harvesting, and variance calculations.

---

### 1. Application 1: Closed-Form Solution of Geometric Brownian Motion (GBM) via Log-Transform

#### (1) Financial Motivation: Why Arithmetic Brownian Motion Fails
In 1900, Louis Bachelier modeled stock prices with Arithmetic Brownian Motion (ABM): $dS_t = \mu dt + \sigma dW_t$. However, ABM suffers from two major economic flaws:
1. **Negative Asset Prices**: Normal distributions have unbounded support $(-\infty, +\infty)$, which violates the limited liability of equities;
2. **Constant Absolute Dollar Volatility**: ABM assumes dollar fluctuations are identical whether a stock is trading at $\$10$ or $\$1000$. In reality, investors think in terms of **percentage returns**.

In 1965, Nobel laureate Paul Samuelson introduced **Geometric Brownian Motion (GBM)**, modeling relative instantaneous returns as normally distributed:

$$\frac{dS_t}{S_t} = \mu dt + \sigma dW_t \iff dS_t = \mu S_t dt + \sigma S_t dW_t \quad (S_0 > 0)$$

where $\mu$ is the expected drift return and $\sigma > 0$ is percentage volatility.

#### (2) Deriving the Analytical Closed-Form Solution via Itô's Lemma
Because state variable $S_t$ appears inside the diffusion term, standard Riemann integration is impossible. We apply the **logarithmic transformation**:
* Consider smooth non-linear map $f(S) = \ln S$;
* Derivatives: $f'(S) = \frac{1}{S}$, $f''(S) = -\frac{1}{S^2}$;
* Apply Itô's Lemma to $f(S_t) = \ln S_t$:
  $$d(\ln S_t) = f'(S_t) dS_t + \frac{1}{2} f''(S_t) (dS_t)^2 = \frac{1}{S_t} (\mu S_t dt + \sigma S_t dW_t) + \frac{1}{2} \left( -\frac{1}{S_t^2} \right) (\sigma^2 S_t^2 dt)$$
* Simplify to obtain a linear drift-diffusion equation for log-prices:
  $$d(\ln S_t) = \left( \mu - \frac{1}{2}\sigma^2 \right) dt + \sigma dW_t$$
* Integrate directly across $[0, t]$:
  $$\ln\left(\frac{S_t}{S_0}\right) = \left( \mu - \frac{1}{2}\sigma^2 \right) t + \sigma W_t$$
* Exponentiate both sides to reach the **exact analytical solution**:
  $$\boxed{S_t = S_0 \exp\left( \left( \mu - \frac{1}{2}\sigma^2 \right) t + \sigma W_t \right)}$$

#### (3) Statistical Moments & Interpretation of Volatility Drag
* **Log-Normal Distribution**: $\ln(S_t / S_0) \sim \mathcal{N}\left( (\mu - \frac{1}{2}\sigma^2)t, \sigma^2 t \right)$;
* **Expected Price**: Using the moment generating function $\mathbb{E}[e^{\sigma W_t}] = e^{\frac{1}{2}\sigma^2 t}$:
  $$\mathbb{E}[S_t] = S_0 e^{(\mu - \frac{1}{2}\sigma^2)t} \mathbb{E}[e^{\sigma W_t}] = S_0 e^{(\mu - \frac{1}{2}\sigma^2)t} e^{\frac{1}{2}\sigma^2 t} = S_0 e^{\mu t}$$
* **Trading Insight**: While the ensemble arithmetic expectation grows at rate $\mu$, almost every individual trajectory compound growth rate is penalized by the deterministic downward **Volatility Drag $-\frac{1}{2}\sigma^2$**.

---

### 2. Application 2: Option Long Gamma & Delta Hedging Cash Flow (Gamma Scalping)

#### (1) Financial Background & Market Maker Dynamics
In modern options trading, market makers and proprietary desks construct **Delta-Neutral Portfolios** to neutralize first-order directional price risk.

A market maker holding a long derivative position $V(t, S_t)$ hedges directional exposure by shorting $\Delta_t = \frac{\partial V}{\partial S}$ shares of the underlying stock:

$$\Pi_t = V(t, S_t) - \Delta_t S_t$$

#### (2) Second-Order PnL Decomposition
In an infinitesimal interval $dt$, portfolio change is $d\Pi_t = dV(t, S_t) - \Delta_t dS_t$.
Expanding $V(t, S_t)$ via Itô's Lemma:

$$dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS_t + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} (dS_t)^2 = \Theta dt + \Delta dS_t + \frac{1}{2} \Gamma \sigma^2 S_t^2 dt$$

Substituting into portfolio PnL:

$$d\Pi_t = \left( \Theta dt + \Delta dS_t + \frac{1}{2} \Gamma \sigma^2 S_t^2 dt \right) - \Delta dS_t = \underbrace{\Theta dt}_{\text{Time Decay Loss (Theta)}} + \underbrace{\frac{1}{2} \Gamma \sigma^2 S_t^2 dt}_{\text{Second-Order Volatility Cash Inflow (Gamma)}}$$

```mermaid
graph LR
    A["Underlying Stock Fluctuation (Random Walk)"] --> B["Stock Rallies ↑"]
    A --> C["Stock Dips ↓"]
    B --> D["Delta Increases → Hedging Algorithm MUST [Sell Stock High]"]
    C --> E["Delta Decreases → Hedging Algorithm MUST [Buy Stock Low]"]
    D --> F["Captures continuous automated [Buy-Low-Sell-High] Cash Flow: + 1/2 Γ S² σ² dt"]
    E --> F
```

#### (3) The Trader's Intuition: Volatility Harvesting (Gamma Scalping)
* **Long Gamma Position ($\Gamma > 0$)**:
  - When the stock surges, Delta rises, forcing the algorithm to sell stock at high prices to rebalance to neutral;
  - When the stock drops, Delta falls, forcing the algorithm to buy stock at low prices to rebalance;
* **Conclusion**: Market volatility mechanically forces the dynamic delta hedger to **Buy Low and Sell High**!
* The continuous cash harvested from this rebalancing is exactly equal to $\frac{1}{2}\Gamma S^2 \sigma^2 dt$, which compensates for daily Theta time decay.

---

### 3. Application 3: Itô Isometry & Stochastic Integral Variance Calculations

#### (1) Mathematical Background & Motivation
In portfolio mean-variance optimization and stochastic control, we frequently need to compute the variance of stochastic integrals $I_T = \int_0^T H_t dW_t$.
Using naive real analysis expansion $\mathbb{E}[(\int H dW)^2]$ involves tedious double integrals over random martingale differentials. Itô's isometry maps the $L^2$ norm on stochastic sample space directly to the deterministic $L^2$ time norm.

#### (2) Theorem Statement & Key Identity

> **Itô Isometry Theorem**:
> For any square-integrable adapted process $H_t$ with $\mathbb{E}\left[\int_0^T H_t^2 dt\right] < \infty$:
> 
> $$\boxed{\mathbb{E}\left[ \left( \int_0^T H_t dW_t \right)^2 \right] = \int_0^T \mathbb{E}[H_t^2] dt}$$
> 
> Since $\mathbb{E}\left[\int_0^T H_t dW_t\right] = 0$, the variance is given directly by:
> 
> $$\boxed{\operatorname{Var}\left( \int_0^T H_t dW_t \right) = \int_0^T \mathbb{E}[H_t^2] dt}$$

#### (3) Concrete Quantitative Examples
* **Example 1: Variance of $\int_0^T t dW_t$**
  - Integrand is deterministic $H_t = t$;
  - By Itô Isometry:
    $$\operatorname{Var}\left( \int_0^T t dW_t \right) = \int_0^T t^2 dt = \frac{1}{3} T^3$$
* **Example 2: Variance in Vasicek Short-Rate Model $\int_0^T e^{\kappa t} dW_t$**
  - Integrand is $H_t = e^{\kappa t}$;
  - By Itô Isometry:
    $$\operatorname{Var}\left( \int_0^T e^{\kappa t} dW_t \right) = \int_0^T e^{2\kappa t} dt = \frac{e^{2\kappa T} - 1}{2\kappa}$$

---

## Module 5: Derivatives Landscape, Stopping Times & Extreme Values

### 1. Foundations: The Financial Derivatives Landscape (Forwards/Futures vs. Options)

Before diving into stochastic stopping times and extreme value theory, we must establish the structural architecture of modern financial derivatives:

```mermaid
graph TD
    A["Financial Derivatives Architecture"] --> B["Linear Derivatives<br/><b>【Forwards & Futures】</b><br/>Two-sided mandatory obligation<br/>Linear Payoff: S_T - K"]
    A --> C["Non-Linear Derivatives<br/><b>【Options】</b><br/>Asymmetric Right vs. Obligation<br/>Convex Payoff: max(S_T - K, 0)"]
    
    C --> D["Classification by Exercise Mechanism"]
    D --> E["European Options<br/>Exercisable ONLY on maturity T<br/><b>【Fixed-Endpoint Conditional Expectation】</b>"]
    D --> F["American Options<br/>Exercisable at ANY stopping time τ ≤ T<br/><b>【Optimal Stopping & Free Boundary Problem】</b>"]
    D --> G["Path-Dependent Exotic Options<br/>Barrier Options / Asian Options<br/><b>【First Hitting Times & Running Extrema】</b>"]
```

#### (1) Linear Contracts: Forwards & Futures
* **Core Mechanism**: Both parties enter a binding agreement to buy or sell the underlying asset at a predetermined delivery price $K$ on maturity date $T$;
* **Symmetric Obligation**: **Both parties are legally obligated to execute (Obligation)**; neither party can unilaterally walk away;
* **Terminal Payoff**:
  - **Long Position**: $\text{Payoff} = S_T - K$ (a straight line);
  - **Short Position**: $\text{Payoff} = K - S_T$;
* **Initial Premium**: Under no-arbitrage equilibrium, entering a forward contract costs $\$0$ upfront.

#### (2) Non-Linear Contracts: Options
* **Core Mechanism**: The buyer pays an upfront **Premium** for the **Right (but NOT obligation)** to buy or sell the underlying asset at strike $K$; the seller collects the premium and assumes the **passive obligation** to fulfill the trade if exercised;
* **Asymmetric Convex Payoffs**:
  - **Call Option (Right to Buy)**: Payoff is $\max(S_T - K, 0)$;
  - **Put Option (Right to Sell)**: Payoff is $\max(K - S_T, 0)$.

---

### 2. Exercise Styles: European, American & the Natural Mapping to Stopping Times

#### (1) European Options
* **Definition**: Can **ONLY be exercised on maturity date $T$**;
* **Pricing Formula**: Expressed as a terminal conditional expectation under risk-neutral measure $\mathbb{Q}$:
  $$V_{\text{Eur}}(t, S_t) = e^{-r(T-t)} \mathbb{E}^\mathbb{Q}[\Phi(S_T) \mid \mathcal{F}_t]$$

#### (2) American Options & the Optimal Stopping Problem
* **Definition**: Can be exercised at **any continuous stopping time $\tau \in [t, T]$** prior to maturity;
* **Mathematical Pricing Formulation**: Because the rational option holder chooses an exercise stopping time that maximizes expected payoff, American option valuation is an **Optimal Stopping Problem**:
  $$V_{\text{Am}}(t, S_t) = \sup_{\tau \in [t, T]} \mathbb{E}^\mathbb{Q} \left[ e^{-r(\tau - t)} \Phi(S_\tau) \;\middle|\; \mathcal{F}_t \right]$$
* **Classic Quant Interview Question: Why is it NEVER optimal to early exercise an American Call on a non-dividend-paying stock?**
  - **Rigorous Proof**: The European call price satisfies the lower bound inequality:
    $$C_{\text{Eur}}(t, S_t) \ge S_t - K e^{-r(T-t)} > S_t - K \quad (\text{when } r > 0 \text{ and } T > t)$$
  - If the holder early exercises at $t$, they receive only the intrinsic value $S_t - K$;
  - If the holder sells the option in the secondary market, they receive the fair value $C(t, S_t) > S_t - K$;
  - **Conclusion**: Early exercise throws away both **Time Value** and the **interest earned by delaying payment of strike $K$**. Thus, $C_{\text{Am}}(t, S) = C_{\text{Eur}}(t, S)$!
  - **Counter-example (American Put)**: For an American Put on a crashing stock (Deep ITM, $S \to 0$), **early exercise IS optimal**! Exercising early allows the holder to collect cash $K$ immediately and earn risk-free bank interest, which outweighs the negligible remaining time value. There exists a time-dependent **Optimal Early Exercise Boundary $S^*(t)$**.

#### (3) Path-Dependent Exotic Options & Brownian Extremes
* **Barrier Options**: Knock-in or knock-out events are triggered when the asset price breaches barrier level $B$, governed by the **First Hitting Time $\tau_B = \inf\{t \ge 0 : S_t = B\}$**;
* **Lookback Options**: Payoffs depend on the historical extreme $\max_{0 \le t \le T} S_t$ or $\min S_t$, governed by the **Reflection Principle and Running Maximum distributions**.

---

### 3. Exponential Martingales & Wald's Identity

For any $\theta \in \mathbb{R}$, the Doléans-Dade exponential martingale:

$$
M_t^\theta = \exp\left( \theta W_t - \frac{1}{2} \theta^2 t \right)
$$

satisfies $dM_t^\theta = \theta M_t^\theta dW_t$. Under the Optional Stopping Theorem (OST):

$$
\mathbb{E}\left[ \exp\left( \theta W_\tau - \frac{1}{2}\theta^2 \tau \right) \right] = 1 \quad (\text{Wald's Martingale Identity})
$$

---

### 4. Two-Sided Absorbing Boundaries (Continuous Gambler's Ruin)

For $\tau = \inf\{t \ge 0 : W_t \notin (-b, a)\}$ ($a, b > 0$):
1. **Hitting Probability**: Applying OST to martingale $W_t$:
   $$\mathbb{E}[W_\tau] = 0 \implies a P(W_\tau = a) - b (1 - P(W_\tau = a)) = 0 \implies \boxed{P(\text{hit } a \text{ first}) = \frac{b}{a+b}}$$
2. **Expected Exit Time**: Applying OST to martingale $W_t^2 - t$:
   $$\mathbb{E}[W_\tau^2 - \tau] = 0 \implies \boxed{\mathbb{E}[\tau] = \mathbb{E}[W_\tau^2] = a^2 \frac{b}{a+b} + b^2 \frac{a}{a+b} = ab}$$

---

### 5. The Reflection Principle & Running Maximum Distribution

Let $M_t = \max_{0 \le s \le t} W_s$ and $\tau_a = \inf\{s \ge 0 : W_s = a\}$ for $a > 0$.

```reflection-principle-demo
```

**Geometric Reflection Argument**:
The event $\{M_t \ge a\}$ is identical to $\{\tau_a \le t\}$.
At hitting time $\tau_a$, the path reaches $a$. By the **Strong Markov Property**, the residual path $\widetilde{W}_s = W_{\tau_a + s} - a$ ($s \ge 0$) is an independent Brownian motion. By spatial mirror symmetry across level $a$:

$$
\mathbb{P}(W_t \ge a \mid \tau_a \le t) = \mathbb{P}(W_t \le a \mid \tau_a \le t) = \frac{1}{2}
$$

Yielding the **Reflection Principle Formula**:

$$
\mathbb{P}(M_t \ge a) = \mathbb{P}(\tau_a \le t) = 2 \mathbb{P}(W_t \ge a) = 2 \left( 1 - \Phi\left( \frac{a}{\sqrt{t}} \right) \right)
$$

Differentiating with respect to $t$ gives the **Lévy Distribution** density for hitting time $\tau_a$:

$$
f_{\tau_a}(t) = \frac{d}{dt} \mathbb{P}(\tau_a \le t) = \frac{a}{\sqrt{2\pi t^3}} \exp\left( -\frac{a^2}{2t} \right) \quad (t > 0)
$$

---

## Module 6: Black-Scholes-Merton Model, Greeks & Quant Trading Applications

### 6.1 Option Terminology, Moneyness & Value Decomposition

Before deriving partial differential equations, we define the foundational mechanics of option contracts:

#### (1) The 5 Core Pricing Inputs
1. **Spot Price ($S$)**: Current market price of the underlying asset;
2. **Strike Price ($K$)**: Agreed contract execution price at delivery;
3. **Time to Maturity ($\tau = T - t$)**: Annualized time remaining until contract expiration;
4. **Risk-Free Interest Rate ($r$)**: Continuous annualized riskless borrowing/lending rate;
5. **Volatility ($\sigma$)**: Annualized standard deviation of percentage returns.

#### (2) Option Value Decomposition: Intrinsic Value vs. Time Value
Any option market price $V$ decomposes uniquely into:

$$\boxed{\text{Total Option Value } V = \text{Intrinsic Value} + \text{Time Value (Extrinsic Value)}}$$

* **Intrinsic Value**: The immediate payoff if exercised right now: $\max(S - K, 0)$ for Call, $\max(K - S, 0)$ for Put;
* **Time Value**: The market premium paid for the probability that future volatility will move the option deeper in-the-money. Time value decays monotonically to zero as $t \to T$.

#### (3) The 3 States of Moneyness
* **In-The-Money (ITM)**: Intrinsic value $> 0$. Call: $S > K$; Put: $S < K$;
* **At-The-Money (ATM)**: $S \approx K$. Intrinsic value $\approx 0$. **Time value, Gamma (curvature), and Vega (volatility sensitivity) are all maximized at ATM**;
* **Out-Of-The-Money (OTM)**: Intrinsic value $= 0$. Call: $S < K$; Put: $S > K$. The contract is 100% pure time value and expires worthless if $S$ fails to cross $K$.

---

### 6.2 Underlying Asset Dynamics: Geometric Brownian Motion (GBM)


In the Black-Scholes framework, the underlying price process $\{S_t\}_{t \ge 0}$ follows a Geometric Brownian Motion (GBM) stochastic differential equation:

$$
\frac{dS_t}{S_t} = \mu dt + \sigma dW_t \iff dS_t = \mu S_t dt + \sigma S_t dW_t
$$

where $\mu \in \mathbb{R}$ is the expected rate of return (drift), $\sigma > 0$ is the volatility (diffusion), and $W_t$ is a standard Brownian motion.

**Analytical Log-Normal Solution**:
Applying Itô's Lemma to $f(S) = \ln S$ (with $f' = 1/S$ and $f'' = -1/S^2$):

$$
d(\ln S_t) = \frac{1}{S_t} dS_t - \frac{1}{2 S_t^2} (dS_t)^2 = \left( \mu - \frac{1}{2}\sigma^2 \right) dt + \sigma dW_t
$$

Integrating over $[0, t]$:

$$
\ln\left( \frac{S_t}{S_0} \right) = \left( \mu - \frac{1}{2}\sigma^2 \right) t + \sigma W_t \implies \boxed{S_t = S_0 \exp\left( \left( \mu - \frac{1}{2}\sigma^2 \right) t + \sigma W_t \right)}
$$

Thus, $\ln(S_t/S_0) \sim \mathcal{N}\left( (\mu - \frac{1}{2}\sigma^2)t, \sigma^2 t \right)$, meaning $S_t$ is log-normally distributed with moments:

$$
\mathbb{E}[S_t] = S_0 e^{\mu t}, \qquad \operatorname{Var}(S_t) = S_0^2 e^{2\mu t} \left( e^{\sigma^2 t} - 1 \right)
$$

---

### 6.2 Core Assumptions of the Black-Scholes-Merton World

1. **Asset Dynamics**: The underlying price follows a GBM with constant drift $\mu$ and constant volatility $\sigma$;
2. **Frictionless Markets**: Zero transaction costs, zero taxes, and zero bid-ask spread with continuous trading of arbitrary fractional shares;
3. **Unrestricted Short Selling**: Full access to borrowing and shorting at the risk-free rate;
4. **Constant Risk-Free Rate**: Lending and borrowing occur at a constant rate $r > 0$;
5. **No Arbitrage**: No riskless free lunches exist;
6. **European Style**: The derivative can only be exercised at maturity $T$, with no discrete dividend payments before $T$ (extendable to continuous dividend yield $q$).

---

### 6.3 Dual Derivations of the Black-Scholes PDE

Let $V(t, S)$ denote the fair price of a European derivative with terminal payoff $\Phi(S_T)$ at maturity $T$.

#### Method 1: No-Arbitrage Delta Replicating Portfolio (The Classical BSM Approach)

Construct a portfolio $\Pi_t$: **Long 1 unit of the derivative $V(t, S_t)$, Short $\Delta_t$ shares of the underlying $S_t$**:

$$
\Pi_t = V(t, S_t) - \Delta_t S_t
$$

Over an infinitesimal interval $dt$:

$$
d\Pi_t = dV(t, S_t) - \Delta_t dS_t
$$

By Itô's Lemma, expanding $V(t, S_t)$ to order $dt$:

$$
dV = \left( \frac{\partial V}{\partial t} + \mu S \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt + \sigma S \frac{\partial V}{\partial S} dW_t
$$

Substituting $dV$ and $dS_t = \mu S dt + \sigma S dW_t$ into $d\Pi_t$:

$$
d\Pi_t = \left( \frac{\partial V}{\partial t} + \mu S \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - \Delta_t \mu S \right) dt + \sigma S \left( \frac{\partial V}{\partial S} - \Delta_t \right) dW_t
$$

To eliminate the stochastic noise $dW_t$ completely, set the **Delta Hedge**:

$$
\Delta_t = \frac{\partial V}{\partial S}
$$

The portfolio becomes locally riskless. Under the no-arbitrage principle, its return must equal the risk-free rate: $d\Pi_t = r \Pi_t dt = r (V - \Delta_t S) dt$. Equating both sides yields the **Black-Scholes-Merton PDE**:

$$
\boxed{\frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = r V}
$$

> **Crucial Insight**: The subjective expected drift $\mu$ disappears entirely! The option value is purely determined by volatility $\sigma$, interest rate $r$, spot price $S$, strike $K$, and time to expiry $\tau = T - t$.

---

#### Method 2: Risk-Neutral Martingale Pricing & Feynman-Kac Theorem

By Girsanov's Theorem, under the equivalent martingale measure $\mathbb{Q}$ (the risk-neutral measure):

$$
dS_t = r S_t dt + \sigma S_t d\widetilde{W}_t \quad (\widetilde{W}_t \text{ is a } \mathbb{Q}\text{-Brownian motion})
$$

The discounted asset $e^{-rt}S_t$ is a $\mathbb{Q}$-martingale. The no-arbitrage price is the discounted expected payoff:

$$
V(t, S_t) = e^{-r(T-t)} \mathbb{E}^\mathbb{Q}\left[ \Phi(S_T) \;\middle|\; \mathcal{F}_t \right]
$$

By the **Feynman-Kac Theorem**, this conditional expectation solves the BSM PDE with terminal condition $V(T, S) = \Phi(S)$.

---

### 6.4 Analytical Closed-Form Derivation of the Call and Put Formulas

Consider a European Call option with payoff $\Phi(S_T) = \max(S_T - K, 0)$ and time to maturity $\tau = T - t$.
Under $\mathbb{Q}$, the terminal price is $S_T = S_t \exp\left( (r - \frac{1}{2}\sigma^2)\tau + \sigma\sqrt{\tau} Z \right)$ where $Z \sim \mathcal{N}(0, 1)$.

$$
C(S_t, t) = e^{-r\tau} \int_{-\infty}^\infty \max\left( S_t e^{(r - \frac{1}{2}\sigma^2)\tau + \sigma\sqrt{\tau} z} - K, 0 \right) \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz
$$

#### Step 1: Integration Lower Bound
$S_T > K$ if and only if $z > -d_2$, where:

$$
\boxed{d_2 = \frac{\ln(S_t/K) + (r - \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}}
$$

#### Step 2: Decomposition into Asset and Cash Integrals

$$
C(S_t, t) = \underbrace{e^{-r\tau} \int_{-d_2}^\infty S_t e^{(r - \frac{1}{2}\sigma^2)\tau + \sigma\sqrt{\tau} z} \frac{e^{-z^2/2}}{\sqrt{2\pi}} dz}_{I_1} - \underbrace{K e^{-r\tau} \int_{-d_2}^\infty \frac{e^{-z^2/2}}{\sqrt{2\pi}} dz}_{I_2}
$$

#### Step 3: Evaluating Cash Integral $I_2$

$$
I_2 = K e^{-r\tau} P(Z \ge -d_2) = K e^{-r\tau} \Phi(d_2)
$$

#### Step 4: Evaluating Asset Integral $I_1$ via Completing the Square

$$
I_1 = S_t e^{-\frac{1}{2}\sigma^2\tau} \int_{-d_2}^\infty \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{z^2 - 2\sigma\sqrt{\tau} z}{2} \right) dz
$$

Completing the square: $z^2 - 2\sigma\sqrt{\tau} z = (z - \sigma\sqrt{\tau})^2 - \sigma^2\tau$. The factor $e^{\frac{1}{2}\sigma^2\tau}$ cancels $e^{-\frac{1}{2}\sigma^2\tau}$ exactly:

$$
I_1 = S_t \int_{-d_2}^\infty \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(z - \sigma\sqrt{\tau})^2}{2} \right) dz
$$

Substituting $u = z - \sigma\sqrt{\tau}$ changes the lower bound to $-d_2 - \sigma\sqrt{\tau} \triangleq -d_1$:

$$
d_1 = d_2 + \sigma\sqrt{\tau} = \frac{\ln(S_t/K) + (r + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}} \implies I_1 = S_t \Phi(d_1)
$$

#### Step 5: Master Black-Scholes Pricing Formula

$$
\boxed{C(S, t) = S \Phi(d_1) - K e^{-r(T-t)} \Phi(d_2)}
$$

Via Put-Call Parity $P = C - S + K e^{-r\tau}$:

$$
\boxed{P(S, t) = K e^{-r(T-t)} \Phi(-d_2) - S \Phi(-d_1)}
$$

---

### 6.5 Financial and Probabilistic Interpretations of $d_1$ and $d_2$

```mermaid
flowchart TD
  Formula["C = S·Φ(d_1) - K·e^(-rτ)·Φ(d_2)"]
  Term1["S·Φ(d_1)<br/>Discounted expected stock value upon exercise<br/>(Hedge Position Value = Δ·S)"]
  Term2["K·e^(-rτ)·Φ(d_2)<br/>Discounted expected cash payment reserve<br/>(Discounted Strike × Exercise Probability)"]
  D2["Φ(d_2) = Q(S_T ≥ K)<br/>Risk-neutral probability of finishing in-the-money"]
  D1["Φ(d_1) = Delta<br/>Share hedge ratio<br/>(Exercise probability under Share Measure Q^S)"]

  Formula --> Term1
  Formula --> Term2
  Term2 --> D2
  Term1 --> D1
```

1. **$\Phi(d_2) = \mathbb{Q}(S_T \ge K)$**: The exact risk-neutral probability that the option finishes in-the-money (ITM). $K e^{-r\tau} \Phi(d_2)$ represents the discounted cash reserve needed to deliver the strike price upon exercise.
2. **$\Phi(d_1) = \Delta = \frac{\partial C}{\partial S}$**: The replicating delta (number of underlying shares to hold). Under the Share Measure $\mathbb{Q}^S$ (taking $S_t$ as numeraire), $\Phi(d_1) = \mathbb{Q}^S(S_T \ge K)$ is the probability of exercise.
3. **The Core Derivative Identity**:

$$
\boxed{S \phi(d_1) = K e^{-r\tau} \phi(d_2)}
$$

This identity guarantees that cross terms vanish when calculating Greeks.

---

### 6.6 Put-Call Parity

For European options with identical strike $K$ and maturity $T$:

$$
\boxed{C_t - P_t = S_t - K e^{-r(T-t)}}
$$

---

### 6.7 Black-Scholes Greeks: Multi-Dimensional Risk Decomposition for Market Makers

#### (1) Why do Quantitative Traders Define "The Greeks"?
An option price is a highly non-linear multivariate surface: $V = V(S, t, \sigma, r, K)$.
A trading desk manages portfolios of thousands of option contracts across strikes and maturities. Traders cannot intuitively visualize five-dimensional surfaces; they use **Multivariate Taylor Expansions** to decompose aggregate portfolio risk into first- and second-order directional sensitivities known across Wall Street as **The Greeks**:

$$dV \approx \underbrace{\frac{\partial V}{\partial S}}_{\Delta} dS + \underbrace{\frac{1}{2} \frac{\partial^2 V}{\partial S^2}}_{\Gamma} (dS)^2 + \underbrace{\frac{\partial V}{\partial t}}_{\Theta} dt + \underbrace{\frac{\partial V}{\partial \sigma}}_{\text{Vega}} d\sigma + \underbrace{\frac{\partial V}{\partial r}}_{\rho} dr$$

```mermaid
graph TD
    V["Option Price Non-Linear Surface V(S, t, σ, r, K)"] --> D["Delta (Δ = ∂V/∂S)<br/><b>First-Order Directional Risk</b><br/>Hedge ratio · Underlying share exposure"]
    V --> G["Gamma (Γ = ∂²V/∂S²)<br/><b>Second-Order Curvature Risk</b><br/>Rate of change of Delta · Volatility harvesting source"]
    V --> T["Theta (Θ = ∂V/∂t)<br/><b>Time Value Decay Rate</b><br/>Daily rent paid/collected for optionality"]
    V --> VE["Vega (𝒱 = ∂V/∂σ)<br/><b>Implied Volatility Sensitivity</b><br/>VIX / market panic index exposure"]
    V --> R["Rho (ρ = ∂V/∂r)<br/><b>Interest Rate Sensitivity</b><br/>Central bank rate hike/cut exposure"]
```

---

#### (2) Comprehensive Greeks Master Formula Table

| Greek | Core Financial & Trading Meaning | Call Formula | Put Formula | Key Properties & Trader Rules of Thumb |
| :--- | :--- | :--- | :--- | :--- |
| **Delta ($\Delta$)** | Price Sensitivity $\frac{\partial V}{\partial S}$ | $\Phi(d_1)$ | $\Phi(d_1) - 1 = -\Phi(-d_1)$ | Call $\in (0, 1)$, Put $\in (-1, 0)$; ATM $\approx 0.5$ |
| **Gamma ($\Gamma$)** | Convexity / Curvature $\frac{\partial^2 V}{\partial S^2}$ | $\frac{\phi(d_1)}{S \sigma \sqrt{\tau}}$ | $\frac{\phi(d_1)}{S \sigma \sqrt{\tau}}$ | Strictly positive and identical ($\Gamma > 0$); peaks at ATM |
| **Vega ($\mathcal{V}$)** | Volatility Sensitivity $\frac{\partial V}{\partial \sigma}$ | $S \sqrt{\tau} \phi(d_1)$ | $S \sqrt{\tau} \phi(d_1)$ | Strictly positive and identical ($\mathcal{V} > 0$); decays near expiry |
| **Theta ($\Theta$)** | Daily Time Decay $\frac{\partial V}{\partial t}$ | $-\frac{S\phi(d_1)\sigma}{2\sqrt{\tau}} - r K e^{-r\tau}\Phi(d_2)$ | $-\frac{S\phi(d_1)\sigma}{2\sqrt{\tau}} + r K e^{-r\tau}\Phi(-d_2)$ | Negative for long positions (buyers bleed cash daily, sellers collect rent) |
| **Rho ($\rho$)** | Interest Rate Sensitivity $\frac{\partial V}{\partial r}$ | $K \tau e^{-r\tau} \Phi(d_2)$ | $-K \tau e^{-r\tau} \Phi(-d_2)$ | Call $\rho > 0$, Put $\rho < 0$ |

---

#### (3) The Market Maker's Conservation Law: The Eternal Theta-Gamma Trade-off

Substituting all Greeks back into the Black-Scholes PDE yields the fundamental **Theta-Gamma Equilibrium Law**:

$$\boxed{\Theta + \frac{1}{2}\sigma^2 S^2 \Gamma = r(V - S\Delta)}$$

* **Physical & Trading Intuition**:
  There is no free lunch in options trading:
  1. **Long Gamma ($\Gamma > 0$)**:
     - You gain an automated cash machine that harvests buy-low-sell-high profits as the stock oscillates ($+\frac{1}{2}\Gamma S^2 \sigma^2 dt > 0$);
     - In exchange, you must pay continuous rent to the market in the form of daily time value decay ($\Theta < 0$);
  2. **Short Gamma ($\Gamma < 0$)**:
     - You act as an insurance company, steadily collecting time value rent every day ($\Theta > 0$ generates positive carry);
     - In exchange, you sit on a negative convexity time bomb: if a black swan event occurs, negative Gamma accelerates your losses exponentially.


---

### 6.8 Dynamic Delta Hedging & Volatility Arbitrage

In practical market making, options are priced with implied vol $\sigma_I$, while the underlying moves with realized vol $\sigma_R$:

$$
\boxed{d\Pi_t = \frac{1}{2} S_t^2 \Gamma_t \left( \sigma_R^2 - \sigma_I^2 \right) dt}
$$

```delta-hedging-demo
```

- **Long Gamma ($\Gamma > 0$)**: Earns positive alpha when realized volatility exceeds implied volatility ($\sigma_R > \sigma_I$).
- **Short Gamma ($\Gamma < 0$)**: Harvests steady Theta decay in calm regimes ($\sigma_R < \sigma_I$), but carries tail risk during volatility shocks.

---

### Module 7: Deep Quant Interview Problems, Generalizations & Scaffolded Practice (Core & Bridge Problems)

### Part 1: The 3 Core Interview Problems & Parametric Generalizations

---

#### Problem 1: Correlation of Brownian Time & Stochastic Integrals

> **Problem Statement (SIG / Akuna / Citadel Core Interview Question)**:
> 
> Let $W_t$ be a standard one-dimensional Brownian motion ($W_0 = 0$). Over the time horizon $[0, T]$, define two random variables:
> 
> $$
> X = \int_0^T W_t dt, \qquad Y = \int_0^T t dW_t
> $$
> 
> Find the correlation coefficient $\operatorname{Corr}(X, Y)$.

**Step-by-Step Derivation & Solution**:

**Step 1: Stochastic Integration by Parts**
Applying Itô's Lemma to $f(t, W_t) = t W_t$:

$$
d(t W_t) = t dW_t + W_t dt + (dt)(dW_t) = t dW_t + W_t dt
$$

Integrating over $[0, T]$:

$$
T W_T - 0 = \int_0^T t dW_t + \int_0^T W_t dt \implies T W_T = X + Y \implies X = T W_T - Y
$$

Alternatively, using $W_t = \int_0^t dW_s$ and changing the order of integration (stochastic Fubini):

$$
X = \int_0^T \left( \int_0^t dW_s \right) dt = \int_0^T \left( \int_s^T dt \right) dW_s = \int_0^T (T - t) dW_t
$$

Both $X$ and $Y$ are Itô integrals of deterministic functions with respect to standard Brownian motion, so $(X, Y)$ is jointly Gaussian with zero mean: $\mathbb{E}[X] = \mathbb{E}[Y] = 0$.

**Step 2: Variances and Covariance via Itô Isometry**
1. **Variance of $X$**:
   $$\operatorname{Var}(X) = \mathbb{E}\left[ \left( \int_0^T (T - t) dW_t \right)^2 \right] = \int_0^T (T - t)^2 dt = \left[ -\frac{(T - t)^3}{3} \right]_0^T = \frac{T^3}{3}$$
2. **Variance of $Y$**:
   $$\operatorname{Var}(Y) = \mathbb{E}\left[ \left( \int_0^T t dW_t \right)^2 \right] = \int_0^T t^2 dt = \left[ \frac{t^3}{3} \right]_0^T = \frac{T^3}{3}$$
3. **Covariance of $X$ and $Y$**:
   $$\operatorname{Cov}(X, Y) = \mathbb{E}[X Y] = \int_0^T (T - t) t dt = \int_0^T (T t - t^2) dt = \frac{T^3}{2} - \frac{T^3}{3} = \frac{T^3}{6}$$

**Step 3: Correlation Coefficient**

$$
\operatorname{Corr}(X, Y) = \frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X) \operatorname{Var}(Y)}} = \frac{\frac{T^3}{6}}{\sqrt{\frac{T^3}{3} \cdot \frac{T^3}{3}}} = \frac{\frac{T^3}{6}}{\frac{T^3}{3}} = \boxed{\frac{1}{2}}
$$

**Consistency Check**:
$\operatorname{Var}(X + Y) = \operatorname{Var}(T W_T) = T^2 \operatorname{Var}(W_T) = T^3 = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y) = \frac{T^3}{3} + \frac{T^3}{3} + 2\left(\frac{T^3}{6}\right) = T^3$. Exact match!

---

#### Problem 2: 2D Brownian Motion Boundary Hitting Distribution (Generalized to Starting Point $(x_0, y_0)$)

> **Problem Statement (Jane Street / Two Sigma Capstone Interview Question)**:
> 
> A standard 2D Brownian motion $(X_t, Y_t)$ starts at an arbitrary point $(x_0, y_0)$ in the right half-plane ($x_0 > 0, y_0 \in \mathbb{R}$).
> It stops upon first hitting the $y$-axis, with stopping time $\tau = \inf\{t > 0 : X_t = 0\}$.
> 1. Find the probability that the stopping point $(0, Y_\tau)$ lies on the **positive $y$-axis ($Y_\tau > 0$)**: $\mathbb{P}(Y_\tau > 0)$;
> 2. Derive the full probability density function $f_{Y_\tau}(u)$ of the stopping ordinate $Y_\tau$;
> 3. Verify the special case where the starting position is $(1, 1)$.

**Step-by-Step Derivation & Dual Perspectives**:

**Perspective 1: Independent Component Convolution & Lévy-Cauchy Mixture**
1. **Distribution of First Hitting Time $\tau$**:
   Since $X_t$ and $Y_t$ are independent, $\tau$ is the first hitting time of $0$ for the 1D horizontal Brownian motion $X_t$ starting at $x_0 > 0$. By the reflection principle, $\tau$ follows the **Lévy distribution**:
   $$f_\tau(t) = \frac{x_0}{\sqrt{2\pi t^3}} \exp\left( -\frac{x_0^2}{2t} \right) \quad (t > 0)$$
2. **Conditional Distribution of $Y_\tau \mid (\tau = t)$**:
   $Y_\tau \mid (\tau = t) \sim \mathcal{N}(y_0, t)$.
3. **Marginal Density of $Y_\tau$**:
   $$f_{Y_\tau}(u) = \int_0^\infty \frac{1}{\sqrt{2\pi t}} \exp\left( -\frac{(u - y_0)^2}{2t} \right) \cdot \frac{x_0}{\sqrt{2\pi t^3}} \exp\left( -\frac{x_0^2}{2t} \right) dt = \frac{x_0}{2\pi} \int_0^\infty \frac{1}{t^2} \exp\left( -\frac{x_0^2 + (u - y_0)^2}{2t} \right) dt$$
   Substituting $v = \frac{x_0^2 + (u - y_0)^2}{2t}$ gives:
   $$f_{Y_\tau}(u) = \boxed{\frac{1}{\pi} \frac{x_0}{x_0^2 + (u - y_0)^2}}$$
   This is the **Cauchy distribution** $\text{Cauchy}(y_0, x_0)$ with location $y_0$ and scale $x_0$!
4. **Probability of Hitting the Positive $y$-Axis**:
   $$\mathbb{P}(Y_\tau > 0) = \int_0^\infty \frac{x_0}{\pi (x_0^2 + (u - y_0)^2)} du = \left[ \frac{1}{\pi} \arctan\left( \frac{u - y_0}{x_0} \right) \right]_0^\infty = \boxed{\frac{1}{2} + \frac{1}{\pi} \arctan\left( \frac{y_0}{x_0} \right)}$$

**Perspective 2: Harmonic Measure & Conformal Angle Invariance**
- Let $u(x, y) = \mathbb{P}_{(x, y)}(Y_\tau > 0)$. By the strong Markov property of Brownian motion, $u(x, y)$ is **harmonic** ($\Delta u = 0$) in the right half-plane $\mathbb{H} = \{x > 0\}$.
- Boundary conditions along $x = 0$: $u(0, y) = 1$ for $y > 0$, and $u(0, y) = 0$ for $y < 0$.
- In polar coordinates, the angle $\theta = \arctan(y/x) \in (-\pi/2, \pi/2)$ is harmonic ($\text{Im}(\ln z) = \theta$).
- Matching boundary limits directly yields:
  $$u(x_0, y_0) = \frac{1}{\pi}\left( \theta + \frac{\pi}{2} \right) = \boxed{\frac{1}{2} + \frac{1}{\pi} \arctan\left( \frac{y_0}{x_0} \right)}$$

**Special Case Verification**:
- **Starting at $(1, 1)$**:
  $$\mathbb{P}(Y_\tau > 0) = \frac{1}{2} + \frac{1}{\pi}\arctan(1) = \frac{1}{2} + \frac{1}{4} = \boxed{\frac{3}{4} = 75\%}$$
- **Starting on the $x$-axis ($y_0 = 0$)**: $\mathbb{P} = 1/2 = 50\%$ (by vertical reflection symmetry).
- **Asymptotics**: As $y_0 \to +\infty$, $\mathbb{P} \to 1$; as $y_0 \to -\infty$, $\mathbb{P} \to 0$.

---

#### Problem 3: Brownian Bridge General Conditional Distribution & Regression (Generalized to $W_T = x$)

> **Problem Statement (Citadel / Optiver / Jump Trading Core Interview Question)**:
> 
> Let $W_t$ be a standard Brownian motion with $W_0 = 0$. Given the terminal condition $W_T = x$ ($T > 0, x \in \mathbb{R}$), for any intermediate time $t \in (0, T)$:
> 1. Find the conditional expectation $\mathbb{E}[W_t \mid W_T = x]$;
> 2. Find the conditional variance $\operatorname{Var}(W_t \mid W_T = x)$ and write the full conditional distribution;
> 3. Define the Brownian Bridge process $B_t = W_t - \frac{t}{T} W_T$, prove that $B_t$ is independent of $W_T$, and compute its covariance function $\operatorname{Cov}(B_s, B_t)$;
> 4. Verify the special case $W_0 = 0, W_2 = 1$ to find $\mathbb{E}[W_{1/2} \mid W_2 = 1]$ and $\operatorname{Var}(W_{1/2} \mid W_2 = 1)$.

**Step-by-Step Derivation & Solution**:

**Step 1: Bivariate Gaussian Conditioning Formula**
The vector $(W_t, W_T)^T$ is jointly Gaussian:

$$
\begin{pmatrix} W_t \\ W_T \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} t & t \\ t & T \end{pmatrix} \right)
$$

1. **Conditional Expectation**:
   $$\mathbb{E}[W_t \mid W_T = x] = \mathbb{E}[W_t] + \frac{\operatorname{Cov}(W_t, W_T)}{\operatorname{Var}(W_T)} (x - \mathbb{E}[W_T]) = 0 + \frac{t}{T}(x - 0) = \boxed{\frac{t}{T} x}$$
2. **Conditional Variance**:
   $$\operatorname{Var}(W_t \mid W_T = x) = \operatorname{Var}(W_t) - \frac{(\operatorname{Cov}(W_t, W_T))^2}{\operatorname{Var}(W_T)} = t - \frac{t^2}{T} = \boxed{\frac{t(T - t)}{T}}$$
3. **Full Conditional Distribution**:
   $$\boxed{W_t \mid (W_T = x) \sim \mathcal{N}\left( \frac{t}{T} x, \frac{t(T - t)}{T} \right)}$$

**Step 2: Orthogonal Increments Representation**
Let $X = W_t \sim \mathcal{N}(0, t)$ and $Y = W_T - W_t \sim \mathcal{N}(0, T - t)$ be independent increments.
Let $Z = W_T = X + Y$. Linear regression of $X$ on $Z$ yields $X = \beta Z + \epsilon$ where:

$$
\beta = \frac{\operatorname{Cov}(X, Z)}{\operatorname{Var}(Z)} = \frac{t}{T}, \qquad \epsilon = X - \frac{t}{T} Z = \frac{T - t}{T} X - \frac{t}{T} Y
$$

$\operatorname{Cov}(\epsilon, Z) = \frac{T - t}{T} t - \frac{t}{T}(T - t) = 0$. In joint Gaussian distributions, zero covariance implies statistical independence. Thus $\epsilon$ is independent of $W_T$ with $\operatorname{Var}(\epsilon) = \frac{t(T - t)}{T}$.

**Step 3: Brownian Bridge Covariance Function ($0 \le s \le t \le T$)**
For $B_t = W_t - \frac{t}{T} W_T$:

$$
\operatorname{Cov}(B_s, B_t) = \operatorname{Cov}(W_s, W_t) - \frac{t}{T}\operatorname{Cov}(W_s, W_T) - \frac{s}{T}\operatorname{Cov}(W_T, W_t) + \frac{st}{T^2}\operatorname{Var}(W_T) = s - \frac{st}{T} - \frac{st}{T} + \frac{st}{T} = \boxed{s \left( 1 - \frac{t}{T} \right)}
$$

**Step 4: Special Case ($T=2, t=1/2, x=1$)**
- $\mathbb{E}[W_{1/2} \mid W_2 = 1] = \frac{1/2}{2} \times 1 = \boxed{\frac{1}{4}}$
- $\operatorname{Var}(W_{1/2} \mid W_2 = 1) = \frac{(1/2)(3/2)}{2} = \boxed{\frac{3}{8}}$

---

### Part 2: Scaffolded Practice Problems for Deep Intuition

---

#### Practice 1: Cross-Covariances of Stochastic Integrals
**Problem**: Find $\operatorname{Cov}(W_T, X)$ and $\operatorname{Cov}(W_T, Y)$ for $X = \int_0^T W_t dt$ and $Y = \int_0^T t dW_t$.
**Solution**:
- $\operatorname{Cov}(W_T, X) = \int_0^T (T - t) dt = \frac{T^2}{2}$.
- $\operatorname{Cov}(W_T, Y) = \int_0^T t dt = \frac{T^2}{2}$.
- Sum: $\operatorname{Cov}(W_T, X + Y) = \operatorname{Cov}(W_T, T W_T) = T \operatorname{Var}(W_T) = T^2 = \frac{T^2}{2} + \frac{T^2}{2}$.

---

#### Practice 2: Conformal Wedge Exit Probabilities
**Problem**: For 2D Brownian motion in a wedge $D = \{r e^{i\theta} : r > 0, 0 < \theta < \alpha\}$ starting at $(r_0, \theta_0)$, find the probability of hitting the upper ray ($\theta = \alpha$) before the lower ray ($\theta = 0$).
**Solution**:
By scale invariance and harmonicity of $\theta$, $u(r_0, \theta_0) = \boxed{\frac{\theta_0}{\alpha}}$, independent of the initial radius $r_0$.

---

#### Practice 3: Three-Point Brownian Conditioning
**Problem**: For $0 < s < t < u < T$, given $W_s = a$ and $W_u = b$, find $\mathbb{E}[W_t \mid W_s = a, W_u = b]$.
**Solution**:
By the Markov property, over $[s, u]$ the trajectory is a bridge connecting $a$ and $b$:
$$\mathbb{E}[W_t \mid W_s = a, W_u = b] = \boxed{\frac{u - t}{u - s} a + \frac{t - s}{u - s} b}$$

---

#### Practice 4: Maximum of a Brownian Bridge (Kolmogorov-Smirnov Statistic)
**Problem**: For standard Brownian bridge $B_t$ on $[0, 1]$ ($B_0 = B_1 = 0$), find $\mathbb{P}(\max_{0 \le t \le 1} B_t \ge y)$ for $y > 0$.
**Solution**:
By the reflection principle conditioned on $W_1 = 0$, reflected paths end at $2y$:
$$\mathbb{P}\left( \max_{0 \le t \le 1} B_t \ge y \right) = \frac{\phi(2y)}{\phi(0)} = \boxed{e^{-2y^2}} \quad (y > 0)$$

---

#### Practice 5: Two-Sided Exit for Drifted Brownian Motion
**Problem**: $X_t = \mu t + \sigma W_t$ with barriers $-a < 0 < b$. Find hitting probability $p_b$ and expected exit time $\mathbb{E}[\tau]$.
**Solution**:
Using exponential martingale $M_t = \exp\left( -\frac{2\mu}{\sigma^2} X_t \right)$ with $\gamma = \frac{2\mu}{\sigma^2}$:
$$p_b = \boxed{\frac{e^{\gamma a} - 1}{e^{\gamma a} - e^{-\gamma b}}}, \qquad \mathbb{E}[\tau] = \boxed{\frac{b p_b - a(1 - p_b)}{\mu}}$$

---
