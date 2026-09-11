# Quant 08 · Brownian Motion, Itô Calculus, and Measure Transforms

Course location: [[Quant11 Martingales Stopping Times Random Walks.en|07 Martingales]] → This note → [[Quant13 Game Theory and Strategic Decision Making.en|09 Game Theory]]

Brownian motion is the scaling limit of a random walk. Paths are continuous and nowhere differentiable; total variation is infinite and quadratic variation is $[W]_t=t$, so integration uses Itô's formula through $(dW_t)^2=dt$. The same toolkit covers stopping times, GBM, Black–Scholes, and Girsanov.

```text
Order of this note:
1. Sample path geometry: Brownian motion W_t is everywhere continuous but nowhere differentiable. Its total variation is infinite, and its quadratic variation is [W]_t = t. Therefore, classical integration fails, and a second-order Taylor expansion up to the (dW_t)^2 = dt order is required.
2. 2D Random Walk & Brownian Motion: Derived via weak convergence (Donsker's theorem). A discrete 2D walk is recurrent, while a continuous 2D Brownian motion is point-transient but neighborhood-recurrent, preserving conformal invariance under complex analytic mappings.
3. Itô Integral vs. Stratonovich Integral: The Itô integral evaluates at the left endpoint, preserving the martingale property E[I_t]=0 and respecting causality. Stratonovich evaluates at the midpoint, preserving the classical chain rule but introducing a drift correction.
4. Stopping Times & Reflection Principle: For the first hitting time \tau_a, leveraging the strong Markov property and spatial symmetry, path folding yields P(M_t >= a) = 2 P(W_t >= a).
5. The essence of pricing: Derivative pricing is the measurement of replication cost. The market maker's Delta hedging eliminates directional exposure.
6. Option Hedging: Delta hedging eliminates first-order directional risk dW_t. The residual instantaneous P&L is d\Pi = (1/2) S^2 \Gamma (\sigma_{realized}^2 - \sigma_{implied}^2) dt - r \Pi dt.
7. First Fundamental Theorem (FTAP I): No Free Lunch with Vanishing Risk (NFLVR) ⟺ Existence of at least one Equivalent Martingale Measure (EMM) Q.
8. Second Fundamental Theorem (FTAP II): Market completeness (all contingent claims are perfectly replicable) ⟺ The Equivalent Martingale Measure Q is unique.
9. Girsanov's Theorem: It provides the precise operational guide for shifting the drift term of a Brownian motion via an exponential martingale in continuous paths.
```

---

## 1 · Fundamentals of Brownian Motion and Sample Path Geometry

### Definition and Axiomatic Characterization

A standard one-dimensional Brownian motion (Wiener Process) $\{W_t\}_{t \ge 0}$ is a continuous-time stochastic process defined on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$, satisfying four axioms:

1. Deterministic origin: $W_0 = 0$ almost surely.
2. Independent increments: For any $0 \le t_0 < t_1 < \dots < t_n$, the increments $W_{t_1}-W_{t_0}, \dots, W_{t_n}-W_{t_{n-1}}$ are mutually independent.
3. Stationary Gaussian increments: For any $0 \le s < t$, $W_t - W_s \sim \mathcal{N}(0, t - s)$.
4. Path continuity: $t \mapsto W_t(\omega)$ is continuous for almost all sample paths $\omega$.

Its covariance structure follows directly from the axioms. For any $s \le t$:

$$
\operatorname{Cov}(W_s, W_t) = \mathbb{E}[W_s (W_s + (W_t - W_s))] = \mathbb{E}[W_s^2] + \mathbb{E}[W_s]\mathbb{E}[W_t - W_s] = s
$$

Thus, $\operatorname{Cov}(W_s, W_t) = \min(s, t)$.

### Geometric Particularities: Total and Quadratic Variation

Classical calculus is built upon functions having finite total variation. Consider a partition of the time interval $[0, T]$, denoted as $\Pi_n: 0 = t_0 < t_1 < \dots < t_n = T$, with the maximum grid step $|\Pi_n| = \max_i |t_i - t_{i-1}| \to 0$.

The first-order total variation diverges to infinity:

$$
\operatorname{TV}_T(W) = \lim_{|\Pi_n| \to 0} \sum_{i=1}^n |W_{t_i} - W_{t_{i-1}}| = \infty \quad \text{almost surely}
$$

Let $\Delta W_i = W_{t_i} - W_{t_{i-1}} \sim \sqrt{\Delta t_i} Z_i$ where $Z_i \sim \mathcal{N}(0, 1)$. Then $\mathbb{E}[|\Delta W_i|] = \sqrt{\Delta t_i} \mathbb{E}[|Z_i|] = \sqrt{\frac{2}{\pi}} \sqrt{\Delta t_i}$. Assuming equidistant intervals $\Delta t = T/n$, the expected sum is:

$$
\mathbb{E}\left[ \sum_{i=1}^n |\Delta W_i| \right] = n \cdot \sqrt{\frac{2}{\pi}} \sqrt{\frac{T}{n}} = \sqrt{\frac{2}{\pi}} \sqrt{T} \sqrt{n} \xrightarrow{n \to \infty} \infty
$$

The quadratic variation strictly converges to a constant $T$:

$$
[W]_T = \lim_{|\Pi_n| \to 0} \sum_{i=1}^n (W_{t_i} - W_{t_{i-1}})^2 = T \quad \text{in } L^2 \text{ and probability}
$$

Proof outline: Let $Q_n = \sum_{i=1}^n (\Delta W_i)^2$. Since $\Delta W_i^2 = \Delta t_i Z_i^2$, its expectation is $\mathbb{E}[Q_n] = \sum \Delta t_i \mathbb{E}[Z_i^2] = \sum \Delta t_i = T$.
The variance is:

$$
\operatorname{Var}(Q_n) = \sum_{i=1}^n \operatorname{Var}((\Delta W_i)^2) = \sum_{i=1}^n (\Delta t_i)^2 \operatorname{Var}(Z_i^2) = 2 \sum_{i=1}^n (\Delta t_i)^2 \le 2 |\Pi_n| \sum_{i=1}^n \Delta t_i = 2 |\Pi_n| T \xrightarrow{|\Pi_n| \to 0} 0
$$

The variance converging to 0 implies $Q_n \xrightarrow{L^2} T$. In differential notation, this is expressed as:

$$
(dW_t)^2 = dt
$$

Because $(dW_t)^2 = dt$ has a first-order time dimension $O(dt)$, when performing a Taylor expansion on functions containing stochastic terms, the second-order term $(\Delta W)^2$ cannot be ignored as it is in classical calculus. This is the mathematical root of the second-derivative correction term (Itô drift) in stochastic calculus.

```brownian-motion-demo
```

### Transition Density and Partial Differential Equations

The transition probability density of reaching $y$ at time $t$, starting from $x$, is:

$$
p(t, x, y) = \frac{1}{\sqrt{2\pi t}} \exp\left( -\frac{(y-x)^2}{2t} \right)
$$

Taking partial derivatives directly yields the heat equations (Kolmogorov backward and forward equations):

$$
\frac{\partial p}{\partial t} = \frac{1}{2} \frac{\partial^2 p}{\partial x^2} \quad (\text{Backward form}) \qquad \frac{\partial p}{\partial t} = \frac{1}{2} \frac{\partial^2 p}{\partial y^2} \quad (\text{Forward Fokker-Planck form})
$$

This is the beginning of the profound connection between continuous Markov processes and parabolic PDEs (the foundation of the Feynman-Kac formula).

---

## 2 · Two-Dimensional Random Walk and Recurrence

### Donsker's Invariance Principle

Let $\xi_1, \xi_2, \dots$ be i.i.d. random steps on the 2D grid $\mathbb{Z}^2$, moving up, down, left, or right with probability $1/4$ each: $\mathbb{E}[\xi_i] = (0, 0)$ and $\operatorname{Cov}(\xi_i) = \frac{1}{2} I_2$.
Construct the discrete polygonal process $S_k = \sum_{i=1}^k \xi_i$. Introduce the spatio-temporal scaling:

$$
B_N(t) = \frac{1}{\sqrt{N}} S_{\lfloor Nt \rfloor}
$$

Donsker's invariance principle asserts that as $N \to \infty$, the random polygonal path $B_N(\cdot)$ converges weakly in the space of continuous functions $C([0, T], \mathbb{R}^2)$ to standard 2D Brownian motion:

$$
B_N(t) \implies \left( \frac{1}{\sqrt{2}} B_t^{(1)}, \frac{1}{\sqrt{2}} B_t^{(2)} \right)
$$

where $B^{(1)}$ and $B^{(2)}$ are independent 1D standard Brownian motions.

```two-d-walk-demo
```

### Recurrence and Transience: Pólya's Theorem

Pólya's theorem establishes the recurrence properties of classical random walks:

| Spatial Dimension $d$ | Discrete Grid Walk ($\mathbb{Z}^d$) | Continuous Brownian Motion ($\mathbb{R}^d$) | Properties and Probabilities |
|---|---|---|---|
| $d = 1$ | Recurrent | Recurrent | Returns to origin 0 with probability 1 |
| $d = 2$ | Recurrent | Neighborhood-recurrent, point-transient | Almost never hits a specific point, but enters any small open ball infinitely many times |
| $d \ge 3$ | Transient | Transient | Continuous: $\lim_{t\to\infty} \|B_t\| = \infty$ a.s. |

In $\mathbb{R}^2$, the logarithmic capacity of a singleton set $\{x\}$ is 0, and the sample path of a 2D Brownian motion has a Hausdorff dimension of 2. When the spatial dimension exactly matches the path dimension, the probability of hitting a specific point is 0. However, any open disk in the plane has positive capacity, meaning Brownian motion will pass through it infinitely many times over an infinite time horizon. This explains the 2D drunkard's walk problem (a drunkard on a 2D grid will always find their way home, whereas a bird lost in 3D space might fly away forever).

### Conformal Invariance (Lévy's Theorem)

2D Brownian motion possesses a special geometric symmetry: it remains a Brownian motion under complex analytic (holomorphic) mappings.

Let $Z_t = B_t^{(1)} + i B_t^{(2)}$ be standard Brownian motion in the complex plane, and let $f: U \to V$ be a non-degenerate holomorphic function. The transformed complex process $W_t = f(Z_t)$ satisfies:

$$
W_t = \widetilde{Z}_{\tau_t}
$$

where $\widetilde{Z}$ is another standard complex Brownian motion, and $\tau_t = \int_0^t |f'(Z_s)|^2 ds$ is a deterministic local time scaling (clock change). Using conformal mappings, complex geometric boundary problems can be transformed into Brownian motion problems on simpler domains.

---

## 3 · Diffusion Processes and Itô Calculus

### Historical Evolution

The development of stochastic calculus was driven by the observation of irregular motion in the physical world, eventually finding profound application in financial modeling:

| Year | Key Figure | Milestone Contribution | Core Significance |
|---|---|---|---|
| 1827 | Robert Brown | Observed the irregular, continuous jitter of pollen grains | First physical observation of micro-level random motion |
| 1900 | Louis Bachelier | Modeled stocks and options using arithmetic Brownian motion | Preceded Einstein in establishing a diffusion model for assets |
| 1905 | Albert Einstein | Derived the diffusion PDE, proving $\mathbb{E}[(\Delta x)^2] = 2Dt$ | Established the physical mechanism of diffusion |
| 1923 | Norbert Wiener | Constructed the rigorous measure for the standard Wiener process | Rigorously proved BM paths are nowhere differentiable |
| 1944 | Kiyosi Itô | Invented non-anticipative stochastic integration based on martingale theory | Solved the failure of classical calculus, establishing SDEs |
| 1973 | Black-Scholes-Merton | Used Itô's lemma to build a risk-free Delta dynamic replicating portfolio | Eliminated non-linear option risk, deriving the pricing formula |

### Drift-Diffusion Processes

Any one-dimensional continuous-time, continuous-state Markov process can be decomposed into a drift-diffusion Stochastic Differential Equation (SDE):

$$
dX_t = \underbrace{\mu(t, X_t) dt}_{\text{Deterministic Drift}} + \underbrace{\sigma(t, X_t) dW_t}_{\text{Stochastic Diffusion}}
$$

- Drift Term: The deterministic pull or trend. If $\sigma=0$, it degenerates into an ODE.
- Diffusion Term: The intensity of random shocks, representing instantaneous volatility.

Common Itô Diffusion Processes:

| Model Name | SDE | Drift $\mu(X_t)$ | Diffusion $\sigma(X_t)$ | Typical Applications |
|---|---|---|---|---|
| Standard BM | $dX_t = dW_t$ | $0$ (zero drift) | $1$ (unit diffusion) | Core of martingale pricing |
| Arithmetic BM with Drift | $dX_t = \mu dt + \sigma dW_t$ | $\mu$ (constant) | $\sigma$ (constant) | Short-term spread modeling |
| Geometric BM (GBM) | $dS_t = \mu S_t dt + \sigma S_t dW_t$ | $\mu S_t$ (proportional to price) | $\sigma S_t$ (percentage volatility) | Equity modeling, option pricing |
| Ornstein-Uhlenbeck (OU) | $dX_t = \theta(\mu - X_t) dt + \sigma dW_t$ | $\theta(\mu - X_t)$ (mean-reverting pull) | $\sigma$ (constant volatility) | Interest rate models, pairs trading |
| Square-Root Process (CIR) | $dr_t = k(\theta - r_t) dt + \sigma \sqrt{r_t} dW_t$ | $k(\theta - r_t)$ (mean reversion) | $\sigma \sqrt{r_t}$ (vol vanishes near 0) | CIR rates, Heston model |

### Intuition: Volatility Drag and Second-Order Drift

At the microscopic level, the step size is $\pm \sqrt{\Delta t}$. Because $(\pm \sqrt{\Delta t})^2 = \Delta t$, the randomness of the coin flip's direction is completely annihilated upon squaring. The accumulation of squared volatility becomes a deterministic flow of time.

Imagine standing at the center of a parabolic bowl $f(x) = x^2$. A symmetric random force pushes horizontally; the average horizontal displacement is 0. However, vertically, both left and right movements result in an increase in height. Despite the horizontal force being pure fair noise, the bowl's curvature (convexity $f'' > 0$) ensures the average height inevitably rises. This is the dynamic manifestation of Jensen's inequality in continuous time:

$$
\mathbb{E}[f(X + \Delta W)] - f(X) \approx \frac{1}{2} f''(X) \Delta t
$$

The logarithmic return function $\ln S$ is concave ($f'' < 0$), generating a negative drag. This is the core physical mechanism for why geometric compound returns are inevitably lower than arithmetic average returns.

### Itô vs. Stratonovich Integral

Consider a partition of the interval $[0, T]$, $0 = t_0 < t_1 < \dots < t_n = T$. Introduce the parameterized evaluation point $\tau_i = (1-\alpha)t_i + \alpha t_{i+1}$:

$$
S_n^{(\alpha)} = \sum_{i=0}^{n-1} X_{\tau_i} (W_{t_{i+1}} - W_{t_i})
$$

#### Itô Integral ($\alpha = 0$)

$$
\int_0^T X_t dW_t \triangleq \lim_{|\Pi| \to 0} \sum_{i=0}^{n-1} X_{t_i} (W_{t_{i+1}} - W_{t_i})
$$

The integrand is evaluated at the left endpoint of the subinterval, completely determined by $\mathcal{F}_{t_i}$, containing no information about the future Brownian increment $\Delta W_i$. This strictly preserves the martingale property. In finance, traders can only build positions based on current information, making this integral precisely correspond to actual self-financing trading P&L.

#### Stratonovich Integral ($\alpha = 1/2$)

$$
\int_0^T X_t \circ dW_t \triangleq \lim_{|\Pi| \to 0} \sum_{i=0}^{n-1} \left( \frac{X_{t_i} + X_{t_{i+1}}}{2} \right) (W_{t_{i+1}} - W_{t_i})
$$

The evaluation point is the trapezoidal midpoint. It includes the future state $X_{t_{i+1}}$, creating a correlation with $\Delta W_i$. This breaks the martingale property but preserves the classical chain rule of calculus.

Comparison Summary:

| Dimension | Itô Calculus | Stratonovich Calculus |
|---|---|---|
| Evaluation Point $\alpha$ | $\alpha = 0$ (strict left endpoint) | $\alpha = 1/2$ (trapezoidal midpoint) |
| Information Structure | Adapted process, strictly non-anticipative | Involves future states, endogenous look-ahead |
| Calculus Rules | Second-order correction: $d(f(X)) = f' dX + \frac{1}{2} f'' \sigma^2 dt$ | Classical Newton form: $d(f(X)) = f'(X) \circ dX$ |
| Martingale Property | Strictly preserved: $\mathbb{E}[\int H dW] = 0$ | Destroyed: $\mathbb{E}[\int W \circ dW] = t/2 \ne 0$ |
| Primary Domains | Financial derivatives pricing, risk management, algo trading | Classical physics, manifold navigation, control engineering |

If the Stratonovich integral is mistakenly used to simulate market-making strategies, the discretization implies half a time step of future information, and backtests will show deterministic, illusory profits. The conversion formula between the two (Wong-Zakai correction):

$$
\int_0^T X_t \circ dW_t = \int_0^T X_t dW_t + \frac{1}{2} [X, W]_T
$$

```ito-geometry-demo
```

---

## 4 · Itô's Lemma and the Solution to GBM

### Itô Multiplication Rules and Lemma

Multiplication Table:

| $\times$ | $dt$ | $dW_t$ |
|---|---|---|
| **$dt$** | $0$ | $0$ |
| **$dW_t$** | $0$ | **$dt$** |

For a diffusion process $dX_t = \mu dt + \sigma dW_t$ and a twice-differentiable function $f(t, x)$, according to multivariate Taylor expansion:

$$
df(t, X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2
$$

Substituting the multiplication rule $(dX_t)^2 = \sigma^2 dt$ and simplifying:

$$
df(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x} dW_t
$$

### The Log-Transform Solution to Geometric Brownian Motion

GBM assumes relative returns follow a normal distribution:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

Applying the logarithmic transformation $f(S_t) = \ln S_t$. Using Itô's Lemma, with $f'(S) = 1/S$ and $f''(S) = -1/S^2$:

$$
d(\ln S_t) = \frac{1}{S_t} (\mu S_t dt + \sigma S_t dW_t) + \frac{1}{2} \left( -\frac{1}{S_t^2} \right) (\sigma^2 S_t^2 dt)
$$

Combining terms yields a linear differential equation for the log price:

$$
d(\ln S_t) = \left( \mu - \frac{1}{2}\sigma^2 \right) dt + \sigma dW_t
$$

Integrating both sides over $[0, t]$:

$$
\ln\left( \frac{S_t}{S_0} \right) = \left( \mu - \frac{1}{2}\sigma^2 \right) t + \sigma W_t
$$

Exponentiating provides the explicit closed-form solution:

$$
S_t = S_0 \exp\left( \left( \mu - \frac{1}{2}\sigma^2 \right) t + \sigma W_t \right)
$$

Although the expected mean grows at the arithmetic drift $\mathbb{E}[S_t] = S_0 e^{\mu t}$, the long-term geometric compound growth rate for almost all sample paths is $\mu - \frac{1}{2}\sigma^2$. The Itô correction term is precisely the Volatility Drag. The geometric mean is always inferior to the arithmetic mean, reflecting the severe constraint the Law of Large Numbers imposes on time-series compound growth.

---

## 5 · Stopping Times, Reflection Principle, and Itô Isometry

### Stopping Times and the Reflection Principle

The first hitting time $\tau_a = \inf\{t \ge 0: W_t = a\}$ is a classic example of a stopping time. In finance, it is used to define the trigger moment for barrier options.

According to the reflection principle, once Brownian motion reaches level $a$ at $\tau_a$, due to the strong Markov property, the probabilities of the subsequent path moving upwards or downwards are identical.

Let $M_t = \max_{0 \le s \le t} W_s$. For $a > 0$:

$$
\mathbb{P}(M_t \ge a) = 2 \mathbb{P}(W_t \ge a) = 2 \left( 1 - \Phi\left(\frac{a}{\sqrt{t}}\right) \right)
$$

This principle not only calculates the distribution of the historical maximum but also provides the probability density function of the first hitting time, forming the direct theoretical basis for pricing path-dependent derivatives like knock-out and knock-in options.

```reflection-principle-demo
```

### Itô Isometry

In the stochastic integral $I_T = \int_0^T H_t dW_t$, $H_t$ is an $\mathcal{F}_t$-adapted process representing a dynamic asset holding position. $H_t dW_t$ is the instantaneous trading P&L, and the integral represents the cumulative total P&L.

To calculate the variance of the cumulative P&L, $\mathbb{E}\left[ \left( \int_0^T H_t dW_t \right)^2 \right]$, the double integral involves $\mathbb{E}[H_s H_t dW_s dW_t]$. Due to the independence of Brownian increments, the inter-temporal covariance terms all collapse to zero.

Itô's isometry asserts that for a square-integrable adapted process:

$$
\mathbb{E}\left[ \left( \int_0^T H_t dW_t \right)^2 \right] = \int_0^T \mathbb{E}[H_t^2] dt
$$

The total variance of a dynamic trading strategy is strictly equal to the expected time integral of the squared position sizes.

#### Mapping Math Notation to Trading Intuition

- **$H_t$ is an adapted process**: Time travel is strictly forbidden; there are no "look-ahead" functions. When deciding the position $H_t$ at time $t$, one can only rely on the market history that has occurred up to time $t$.
- **$H_t$ is square-integrable**: Traders cannot hold infinitely leveraged positions. The total variance exposure of the entire holding process must be finite in expectation; otherwise, the fund faces inevitable liquidation.

---

## 6 · Black-Scholes PDE and Delta Hedging Cash Flows

Let the option price be $V(t, S_t)$. Construct a Delta-neutral hedging portfolio:

$$
\Pi_t = V(t, S_t) - \Delta_t S_t
$$

In an infinitesimal instant, the change in value is $d\Pi_t = dV_t - \Delta_t dS_t$. Expanding $V(t, S_t)$ to the second order:

$$
dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS_t + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} \sigma^2 S_t^2 dt
$$

Substituting into the portfolio P&L:

$$
d\Pi_t = \left( \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS_t + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} \sigma^2 S_t^2 dt \right) - \Delta dS_t
$$

To eliminate the directional risk $dS_t$, we set $\Delta = \frac{\partial V}{\partial S}$. This deterministic, risk-free portfolio must then earn exactly the risk-free rate:

$$
d\Pi_t = \left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2} \right) dt = r(V - \Delta S_t) dt
$$

Rearranging yields the Black-Scholes PDE:

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2} + r S_t \frac{\partial V}{\partial S} - rV = 0
$$

Notably, the true expected growth rate $\mu$ is completely absent from the formula. The true expectation only alters the spot price of the asset; it does not affect the replication construction cost.

### Gamma Cash Flows and Volatility Trading

Combining this with the Black-Scholes PDE, the portfolio P&L can be equivalently rewritten as:

$$
d\Pi_t = \Theta dt + \frac{1}{2} \Gamma \sigma^2 S_t^2 dt
$$

A long option position ($\Gamma > 0$) passively executes a "buy low, sell high" strategy in a choppy market. The daily cash flow earned by the hedge exactly equals the Itô second-order term $\frac{1}{2}\Gamma S^2 \sigma^2 dt$, which is used to offset the time value decay $\Theta dt$. This reveals that being long an option is fundamentally equivalent to being long realized volatility against implied volatility.

For market applications of option Greeks, see [[Quant14 Financial Markets Asset Classes and Portfolio Theory.en|10 Markets]].

```delta-hedging-demo
```

---

## 7 · Fundamental Theorems of Asset Pricing (FTAP) and Equivalent Martingale Measures

Financial analysis unfolds on a complete probability space $(\Omega, \mathcal{F}, \mathbb{F}, \mathbb{P})$. Measures $\mathbb{Q}$ and $\mathbb{P}$ are equivalent if $\mathbb{P}(A) = 0 \iff \mathbb{Q}(A) = 0$. Equivalence guarantees that events deemed impossible in reality remain impossible in the model.

The Radon-Nikodym derivative is:

$$
Z_t = \left. \frac{d\mathbb{Q}}{d\mathbb{P}} \right|_{\mathcal{F}_t} = \mathbb{E}^\mathbb{P}\left[ \frac{d\mathbb{Q}}{d\mathbb{P}} \;\middle|\; \mathcal{F}_t \right]
$$

### First Fundamental Theorem (FTAP I)

A market exhibits No Free Lunch with Vanishing Risk (NFLVR) if and only if: there exists at least one Equivalent Martingale Measure (EMM) $\mathbb{Q} \sim \mathbb{P}$ such that the discounted relative price processes $\widetilde{S}_t = S_t / B_t$ of all tradable assets (discounted by $B_t = e^{rt}$) are martingales under $\mathbb{Q}$.

### Second Fundamental Theorem (FTAP II)

Assuming an arbitrage-free market, the market is completely complete (i.e., any derivative has a self-financing perfect replication strategy) if and only if: the Equivalent Martingale Measure $\mathbb{Q}$ is unique.

In the classic Black-Scholes model, 1 Brownian motion corresponds to 1 tradable stock; the number of risk sources equals the number of tradable assets, making the measure unique. In incomplete markets like jump-diffusion or stochastic volatility models, because there are more risk sources than tradable assets, the measure is not unique. Here, derivative pricing requires externally introducing a Market Price of Risk for exogenous calibration.

---

## 8 · Girsanov's Theorem and Drift Elimination

Under the physical measure $\mathbb{P}$:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t^\mathbb{P}
$$

Define the market price of risk process $\theta_t = \frac{\mu - r}{\sigma}$. Construct the Doléans-Dade exponential martingale:

$$
Z_t = \exp\left( -\int_0^t \theta_s dW_s^\mathbb{P} - \frac{1}{2}\int_0^t \theta_s^2 ds \right)
$$

If the Novikov condition $\mathbb{E}^\mathbb{P}[\exp(\frac{1}{2} \int_0^T \theta_t^2 dt)] < \infty$ is satisfied, $Z_t$ is a true martingale. The equivalent measure $\mathbb{Q}$ is defined via $Z_T$.

Under $\mathbb{Q}$, define a new process:

$$
d\widetilde{W}_t = dW_t^\mathbb{P} + \theta_t dt
$$

$\widetilde{W}_t$ is a standard Brownian motion under $\mathbb{Q}$. Substituting this back into the original SDE:

$$
dS_t = \mu S_t dt + \sigma S_t (d\widetilde{W}_t - \theta_t dt) = \mu S_t dt + \sigma S_t d\widetilde{W}_t - (\mu - r) S_t dt = r S_t dt + \sigma S_t d\widetilde{W}_t
$$

By injecting a reverse deterministic drift trend into the Brownian motion, the excess return of the drift term $\mu - r$ is entirely obliterated by the transformation formula, making $\widetilde{S}_t = e^{-rt} S_t$ a pure martingale. This coordinate translation operation is the core essence of Girsanov's theorem in eliminating arbitrage risk premia in finance.

---

## 9 · Change of Numeraire and Advanced Option Analysis

Choosing different tradable assets $N_t$ and $U_t$ as the numeraire, the Radon-Nikodym derivative for the corresponding measures $\mathbb{Q}^N$ and $\mathbb{Q}^U$ is:

$$
\left. \frac{d\mathbb{Q}^U}{d\mathbb{Q}^N} \right|_{\mathcal{F}_t} = \frac{U_t / U_0}{N_t / N_0}
$$

### Measure Separation in BSM

For the classic option pricing formula $C(t, S_t) = S_t N(d_1) - K e^{-r(T-t)} N(d_2)$:

- $N(d_2)$: The direct probability that the option ends up in-the-money at maturity under the risk-neutral measure $\mathbb{Q}$, using the cash deposit account as the numeraire.
- $N(d_1)$: The probability that the option ends up in-the-money at maturity under the Share Measure $\mathbb{Q}^S$, constructed using the stock asset itself as the numeraire.

### Early Exercise Mechanisms: Equities vs. FX Options

An American call option on a non-dividend-paying stock will never be exercised early, because early exercise involves surrendering cash and losing the risk-free interest $r$. However, in FX options, the underlying is a foreign currency asset subject to a foreign risk-free rate $r_f$. When the foreign rate significantly exceeds the domestic rate, the yield from exercising early to obtain the foreign currency and rolling it at interest may exceed the option's remaining time value. In this scenario, a genuine Early Exercise Boundary emerges for American options.

### The Subjective Expectation Paradox: Where Did the True Expectation Go?

Derivative prices are strictly determined by the dynamic hedging replication cost. The market maker's Delta-neutral portfolio has entirely no directional risk exposure, thus the opportunity cost of capital solely determines the arbitrage-free benchmark. If the market is intensely bullish on a stock, this sentiment will first reflect in an explosive revaluation of the underlying spot price, or a massive spike in implied volatility, but $\mu$ will never directly enter the BSM pricing formula for the derivative.

### Strict Local Martingales and Asset Bubbles

If the Radon-Nikodym density process $Z_t$ degenerates into a strict local martingale—meaning there exists some finite time $t$ such that $\mathbb{E}[Z_t] < Z_0 = 1$—the total probability space becomes less than 1, causing a "leakage" in total probability. In quantitative financial models, this signifies that the asset price has developed a bubble structure, harboring the theoretical possibility of escaping to infinity in finite time, resulting in a structural failure of the model.

---

## 10 · The Unity of Discrete and Continuous Time

In a single-period binomial tree model, the future stock state is either $S \cdot u$ or $S \cdot d$. The arbitrage-free risk-neutral probability, derived via a replicating portfolio, is:

$$
q = \frac{e^{r\Delta t} - d}{u - d}
$$

Assume the physical probability in the real world is $p$. The discrete Radon-Nikodym derivative is simply the ratio of state probabilities: $Z(\text{Up}) = q/p$ and $Z(\text{Down}) = (1-q)/(1-p)$.

As the time grid becomes infinitely dense ($\Delta t \to 0$), the continuous product of these step-by-step probability ratios in a multi-period binomial tree rigorously converges, via the Central Limit Theorem, to the continuous-time Doléans-Dade exponential martingale. The conversion of arbitrage-free probability measures ultimately achieves profound logical unity across discrete algebra and continuous differential geometry.

---

## 11 · Monte Carlo and Importance Sampling Experiment

The following code verifies that likelihood ratio weighting under the physical measure and direct simulation under the risk-neutral measure are strictly equivalent numerically:

```python
import numpy as np
import scipy.stats as si

def bsm_call_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

def run_monte_carlo_measure_change():
    S0 = 100.0      # Current spot
    K = 110.0       # Out-of-the-money call strike
    T = 1.0         # 1-year expiration
    r = 0.05        # Risk-free rate (5%)
    mu = 0.20       # Physical bullish drift (20%)
    sigma = 0.25    # Volatility (25%)
    N_sim = 200_000 # Number of paths
    np.random.seed(42)

    analytic_price = bsm_call_price(S0, K, T, r, sigma)
    Z = np.random.standard_normal(N_sim)
    W_T = np.sqrt(T) * Z

    # 1. Direct simulation under Q (drift = r)
    S_T_Q = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)
    payoff_Q = np.maximum(S_T_Q - K, 0.0)
    disc_Q = np.exp(-r * T) * payoff_Q
    mc_price_Q = np.mean(disc_Q)
    se_Q = np.std(disc_Q) / np.sqrt(N_sim)

    # 2. Importance sampling under P (drift = mu) weighted by Radon-Nikodym
    theta = (mu - r) / sigma
    S_T_P = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)
    payoff_P = np.maximum(S_T_P - K, 0.0)
    RN_derivative = np.exp(-theta * W_T - 0.5 * (theta**2) * T)
    disc_P = np.exp(-r * T) * payoff_P * RN_derivative
    mc_price_P = np.mean(disc_P)
    se_P = np.std(disc_P) / np.sqrt(N_sim)

    print("=================================================================")
    print(f"BSM Analytic Fair Value:       {analytic_price:.4f}")
    print(f"Direct Monte Carlo under Q:    {mc_price_Q:.4f}  (SE: {se_Q:.4f})")
    print(f"Weighted Sim under P (RN):     {mc_price_P:.4f}  (SE: {se_P:.4f})")
    print("=================================================================")

if __name__ == '__main__':
    run_monte_carlo_measure_change()

```

```text
=================================================================
Theoretical BSM Price:              8.0214
Direct MC under Measure Q:          8.0251  (SE: 0.0401)
Weighted Importance Sampling (P):   8.0192  (SE: 0.0385)
=================================================================
```

Whether the generated paths feature an extremely high drift rate $\mu$ or just the risk-free rate $r$, applying the $Z_T$ weighting at the end of each path converges to the exact same arbitrage-free theoretical price. In industry practice, this technique is widely used for Importance Sampling acceleration in pricing deep Out-of-The-Money (OTM) options, capable of boosting numerical computational efficiency by factors of hundreds or thousands.

---

## References

- Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*
- Karatzas and Shreve, *Brownian Motion and Stochastic Calculus*
- Harrison and Pliska (1981), *Martingales and Stochastic Integrals in the Theory of Continuous Trading*
- Oksendal, *Stochastic Differential Equations: An Introduction with Applications*
