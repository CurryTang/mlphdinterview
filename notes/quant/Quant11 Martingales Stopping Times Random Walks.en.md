# Quant 11 · Martingales, Stopping Times & Random Walks: Wald's Identity, Martingale Construction & Optimal Stopping

In core quantitative research and trading interviews at top hedge funds (Jane Street, Optiver, Citadel, SIG, Jump Trading, Two Sigma), **Martingales and Doob's Optional Stopping Theorem (OST)** are the ultimate weapons for dimensional reduction.

When faced with multi-stage random walks, absorbing boundaries, expected waiting times, pattern occurrences, or sequential optimal decision problems, traditional Markov transition matrices and difference equations quickly become algebraically intractable. **The superpower of Martingale Theory lies in constructing a driftless stochastic process that collapses the expected payoff of a complex dynamic evolution directly onto its initial boundary condition ($\mathbb{E}[M_T] = \mathbb{E}[M_0]$).**

This chapter thoroughly deconstructs Wald's identities, the 4 master martingale templates, complete solutions to 1D random walks, and optimal stopping theory applied to high-frequency interview classics (Secretary Problem, Sequential Die Rolling, American Options, and Li's Casino Martingale).

```text
4-Step Executive Framework for Random Walks & Optimal Stopping:
1. Identify the Stopping Time & Filtration: Clearly define T (e.g. hitting boundaries, matching a sequence). Verify almost sure finiteness P(T < ∞) = 1 and OST admissibility conditions.
2. Select and Construct the Martingale:
   - For hitting probabilities P(Hit A) -> Construct Exponential Martingale M_n = (q/p)^{S_n} or Harmonic Martingale f(X_n);
   - For expected exit times of symmetric walks -> Construct Quadratic Variance Martingale M_n = S_n^2 - n;
   - For expected exit times of biased walks -> Construct Linear Drift Martingale M_n = S_n - nμ;
   - For string pattern waiting times -> Construct Li's Casino Bankroll Martingale.
3. Apply Optional Stopping Theorem (OST): Equate E[M_T] = E[M_0] to solve for target variables algebraically.
4. Backward Induction for Optimal Stopping: If making sequential stopping decisions, build the Snell Envelope and use dynamic programming to locate optimal exercise boundaries.
```

---

## Module 1: Deep Dive into Wald's Identities

Wald's identities are foundational results stemming from martingale theory applied to i.i.d. sums stopped at random times, serving as the computational backbone for random walks, branching processes, and renewal theory.

```martingale-rw-demo
```

### 1. Wald's First Identity

#### Theorem Statement
Let $X_1, X_2, \dots$ be independent and identically distributed (i.i.d.) random variables with finite first absolute moment $\mathbb{E}[|X_1|] < \infty$ and mean $\mu = \mathbb{E}[X_1]$. Let $T$ be a stopping time with respect to the filtration $\mathcal{F}_n = \sigma(X_1, \dots, X_n)$ with finite expected stopping time $\mathbb{E}[T] < \infty$. Then the randomly stopped sum $S_T = \sum_{i=1}^T X_i$ satisfies:

$$\mathbb{E}[S_T] = \mathbb{E}[T] \cdot \mathbb{E}[X_1]$$

#### Rigorous Mathematical Proof
Expand the stopped sum $S_T$ using indicator functions:

$$S_T = \sum_{n=1}^\infty X_n \mathbf{1}_{\{T \ge n\}}$$

Notice that the event $\{T \ge n\} = \{T \le n - 1\}^c \in \mathcal{F}_{n-1}$. By the definition of stopping times, whether stopping occurs before step $n$ depends purely on the history up to $n-1$. Hence, **$X_n$ is completely independent of $\mathbf{1}_{\{T \ge n\}}$**!

Applying the Fubini-Tonelli theorem to interchange expectation and infinite summation:

$$\mathbb{E}[S_T] = \sum_{n=1}^\infty \mathbb{E}\left[X_n \mathbf{1}_{\{T \ge n\}}\right] = \sum_{n=1}^\infty \mathbb{E}[X_n] \cdot \mathbb{E}\left[\mathbf{1}_{\{T \ge n\}}\right] = \mathbb{E}[X_1] \sum_{n=1}^\infty \mathbb{P}(T \ge n)$$

Using the tail probability formula for discrete non-negative random variables $\mathbb{E}[T] = \sum_{n=1}^\infty \mathbb{P}(T \ge n)$:

$$\mathbb{E}[S_T] = \mathbb{E}[X_1] \cdot \mathbb{E}[T]$$

$\blacksquare$

> [!WARNING]
> **Fatal Interview Pitfall: Why is $\mathbb{E}[T] < \infty$ strictly necessary?**
> 
> Consider a standard 1D symmetric random walk ($X_i = \pm 1$ with probability $1/2$), starting at $S_0 = 0$. Define $T = \min\{n : S_n = 1\}$ as the first hitting time of $+1$.
> Clearly $S_T \equiv 1$, so $\mathbb{E}[S_T] = 1$.
> However, $\mathbb{E}[X_1] = 0$. If one naively applied Wald's first identity:
> $$1 = \mathbb{E}[S_T] = \mathbb{E}[T] \cdot \mathbb{E}[X_1] = \mathbb{E}[T] \cdot 0 = 0 \quad (\text{Contradiction!})$$
> **Root Cause**: Although recurrence ensures $\mathbb{P}(T < \infty) = 1$, the **expected hitting time is infinite** $\mathbb{E}[T] = \infty$. Here $\infty \cdot 0$ is an indeterminate form, violating Wald's integrability hypothesis.

---

### 2. Wald's Second Identity

#### Theorem Statement
Under the same i.i.d. assumptions, if $\mathbb{E}[X_1^2] < \infty$ and $\mathbb{E}[T] < \infty$, let $\mu = \mathbb{E}[X_1]$ and $\sigma^2 = \text{Var}(X_1) = \mathbb{E}[X_1^2] - \mu^2$. Then:

$$\mathbb{E}\left[(S_T - T\mu)^2\right] = \sigma^2 \mathbb{E}[T]$$

In particular, for zero-drift symmetric walks ($\mu = 0$):

$$\mathbb{E}[S_T^2] = \sigma^2 \mathbb{E}[T]$$

#### Proof via Martingale Representation
Define the discrete-time process $M_n = (S_n - n\mu)^2 - n\sigma^2$. We check that $M_n$ is a martingale:

$$\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = \mathbb{E}[(S_n + X_{n+1} - (n+1)\mu)^2 \mid \mathcal{F}_n] - (n+1)\sigma^2$$
$$= (S_n - n\mu)^2 + 2(S_n - n\mu)\underbrace{\mathbb{E}[X_{n+1} - \mu]}_{0} + \underbrace{\mathbb{E}[(X_{n+1} - \mu)^2]}_{\sigma^2} - (n+1)\sigma^2 = M_n$$

By Doob's Optional Stopping Theorem, under $\mathbb{E}[T] < \infty$ and bounded increments:

$$\mathbb{E}[M_T] = \mathbb{E}[M_0] = 0 \implies \mathbb{E}\left[(S_T - T\mu)^2\right] = \sigma^2 \mathbb{E}[T]$$

---

### 3. Wald's Exponential Identity

Let the moment-generating function $M(\theta) = \mathbb{E}[e^{\theta X_1}]$ exist on an open interval around 0. Define the exponential process:

$$M_n(\theta) = \frac{e^{\theta S_n}}{(M(\theta))^n}$$

Because the increments are i.i.d., $M_n(\theta)$ is a non-negative martingale with mean 1. Under OST admissibility:

$$\mathbb{E}\left[\frac{e^{\theta S_T}}{(M(\theta))^T}\right] = 1$$

---

## Module 2: The Art of Martingale Construction & 4 Master Templates

In quantitative interviews, the key question is: **How do we construct the exact right martingale tailored to the problem?**

```html
<table>
  <thead>
    <tr>
      <th style="width: 22%;">Template Name</th>
      <th style="width: 28%;">Mathematical Form $M_n$</th>
      <th style="width: 25%;">Target Application</th>
      <th style="width: 25%;">Core Resulting Equation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. Linear Drift Correction</strong></td>
      <td>$S_n - n\mu$</td>
      <td>Expected stopping time $\mathbb{E}[T]$ for biased walks</td>
      <td>$\mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{\mu}$</td>
    </tr>
    <tr>
      <td><strong>2. Quadratic Variance Correction</strong></td>
      <td>$(S_n - n\mu)^2 - n\sigma^2$</td>
      <td>Expected stopping time $\mathbb{E}[T]$ for zero-drift walks</td>
      <td>$\mathbb{E}[T] = \frac{\mathbb{E}[S_T^2]}{\sigma^2}$</td>
    </tr>
    <tr>
      <td><strong>3. Geometric Exponential MGF</strong></td>
      <td>$\left(\frac{q}{p}\right)^{S_n}$ or $\frac{e^{\theta S_n}}{(M(\theta))^n}$</td>
      <td>Two-sided exit probabilities $\mathbb{P}(\text{Hit } a)$ for asymmetric walks</td>
      <td>$\mathbb{P}_a \left(\frac{q}{p}\right)^a + (1 - \mathbb{P}_a)\left(\frac{q}{p}\right)^{-b} = 1$</td>
    </tr>
    <tr>
      <td><strong>4. Harmonic Eigenfunction</strong></td>
      <td>$f(X_n)$, where $(P - I)f = 0$</td>
      <td>General finite-state Markov chain absorption</td>
      <td>$\mathbb{E}[f(X_T)] = f(X_0)$</td>
    </tr>
  </tbody>
</table>
```

### 1. Template 1: Linear Drift Correction
When $S_n$ has constant non-zero drift $\mathbb{E}[S_{n+1} - S_n \mid \mathcal{F}_n] = \mu \ne 0$:
$$M_n = S_n - n\mu \implies \mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{\mu}$$

### 2. Template 2: Quadratic Variance Correction
When the process has zero drift ($\mu = 0$):
$$M_n = S_n^2 - n\sigma^2 \implies \mathbb{E}[T] = \frac{\mathbb{E}[S_T^2]}{\sigma^2}$$

### 3. Template 3: Geometric / Exponential Martingale
For discrete asymmetric steps ($\mathbb{P}(X_i = +1) = p, \mathbb{P}(X_i = -1) = q = 1 - p \ne p$):
$$\mathbb{E}[\lambda^{X_1}] = 1 \implies \lambda = \frac{q}{p} \implies M_n = \left(\frac{q}{p}\right)^{S_n}$$

### 4. Template 4: Harmonic Functions for Markov Chains
For transition matrix $P$, any vector $f$ satisfying $(Pf)(x) = f(x)$ turns $f(X_n)$ into a martingale, solving boundary hitting probabilities directly via boundary conditions $f(\text{Goal}) = 1, f(\text{Fail}) = 0$.

---

## Module 3: Complete Solutions to 1D Random Walks

Consider a particle starting at $S_0 = 0$, moving $+1$ with probability $p$ and $-1$ with probability $q = 1 - p$.
Define the two-sided exit stopping time:
$$T = \min\{n \ge 0 : S_n = a \text{ or } S_n = -b\} \quad (a, b \in \mathbb{N}^+)$$

### 1. Simple Symmetric Random Walk ($p = 1/2$)

#### (1) Hitting Probability $\mathbb{P}(S_T = a)$
Construct martingale $M_n = S_n$. Applying OST:
$$\mathbb{E}[S_T] = \mathbb{E}[S_0] = 0 \implies a \mathbb{P}_a + (-b)(1 - \mathbb{P}_a) = 0 \implies \mathbb{P}_a = \frac{b}{a + b}, \quad \mathbb{P}_{-b} = \frac{a}{a + b}$$

#### (2) Expected Duration $\mathbb{E}[T]$
Construct quadratic martingale $M_n = S_n^2 - n$. Applying OST:
$$\mathbb{E}[S_T^2 - T] = 0 \implies \mathbb{E}[T] = \mathbb{E}[S_T^2] = a^2 \left(\frac{b}{a+b}\right) + b^2 \left(\frac{a}{a+b}\right) = a \cdot b$$

> [!NOTE]
> **Symmetric Walk Mental Shortcut**:
> - Win probability $\mathbb{P}(+a) = \frac{b}{a+b}$ (inversely proportional to distance);
> - Expected steps $\mathbb{E}[T] = a \cdot b$ (the product of distances to both barriers! E.g. starting at 0 between $[-5, +5]$ takes $5 \times 5 = 25$ steps).

---

### 2. Asymmetric Random Walk ($p \ne 1/2$)

#### (1) Hitting Probability $\mathbb{P}(S_T = a)$
Construct exponential martingale $M_n = \left(\frac{q}{p}\right)^{S_n}$. Applying OST:
$$\mathbb{E}\left[\left(\frac{q}{p}\right)^{S_T}\right] = 1 \implies \mathbb{P}_a \left(\frac{q}{p}\right)^a + (1 - \mathbb{P}_a)\left(\frac{q}{p}\right)^{-b} = 1$$
$$\mathbb{P}_a = \frac{1 - (q/p)^{-b}}{(q/p)^a - (q/p)^{-b}} = \frac{(q/p)^b - 1}{(q/p)^{a+b} - 1}$$

#### (2) Expected Duration $\mathbb{E}[T]$
Single step drift $\mu = p - q \ne 0$. Construct drift martingale $M_n = S_n - n(p - q)$.
$$\mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{p - q} = \frac{a \mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$$

---

## Module 4: Optimal Stopping Theory & 4 High-Frequency Interview Classics

### Mathematical Architecture of Optimal Stopping
- **Goal**: Find stopping rule $T^*$ maximizing expected payoff:
  $$V_0 = \sup_{T \in \mathcal{T}} \mathbb{E}[Z_T]$$
- **Snell Envelope**: Backward induction sequence:
  $$U_N = Z_N, \quad U_n = \max\left(Z_n, \mathbb{E}[U_{n+1} \mid \mathcal{F}_n]\right)$$
- **Optimal Rule**: Stop at the first time current value equals continuation value:
  $$T^* = \min\{n \ge 0 : Z_n = U_n\}$$

---

### Classic 1: The Secretary Problem ($37\%$ Rule)

> **Interview Prompt (Citadel / SIG / Jane Street)**:
> $n$ candidates arrive sequentially in random order. After each interview, you must immediately hire or reject irrevocably.
> You only observe the relative ranking among candidates seen so far.
> **Goal**: Maximize the probability of hiring the single best candidate.

#### Exact Probability Derivation
The optimal strategy rejects the first $k-1$ candidates as a baseline (recording best $M_{k-1}$), then hires the very first candidate who beats $M_{k-1}$.
$$\mathbb{P}(\text{Success} \mid k) = \sum_{j=k}^n \frac{1}{n} \cdot \frac{k-1}{j-1} = \frac{k-1}{n} \sum_{j=k}^n \frac{1}{j-1}$$

#### Continuous Asymptotic Limit ($n \to \infty$)
Let $x = k/n$. The Riemann sum converges to:
$$f(x) = x \int_x^1 \frac{1}{t} dt = -x \ln x$$
Setting $f'(x) = -\ln x - 1 = 0 \implies x^* = \frac{1}{e} \approx 36.8\%$.
The maximum success probability is also $1/e \approx 36.8\%$.

---

### Classic 2: Sequential Die Rolling Game

> **Interview Prompt (Optiver / Jane Street)**:
> You can roll a fair 6-sided die up to $N$ times. At each roll, you can stop and take $\$X$, or discard and roll again. Find fair values $v_k$ and stopping rules.

#### Backward Induction
1. **1 roll remaining**: $v_1 = \mathbb{E}[X] = 3.5$.
2. **2 rolls remaining**: $v_2 = \mathbb{E}[\max(X, 3.5)] = \frac{3 \times 3.5 + 4 + 5 + 6}{6} = 4.25$ (Stop on $\ge 4$).
3. **3 rolls remaining**: $v_3 = \mathbb{E}[\max(X, 4.25)] = \frac{4 \times 4.25 + 5 + 6}{6} = 4.667$ (Stop on $\ge 5$).
4. **4 rolls remaining**: $v_4 = \mathbb{E}[\max(X, 4.667)] = \frac{4 \times 4.667 + 5 + 6}{6} = 4.944$ (Stop on $\ge 5$).

---

### Classic 3: American Options as Optimal Stopping

- **Non-dividend American Call**:
  $$C_{\text{Amer}}(S_t, t) = \sup_{\tau} \mathbb{E}^\mathbb{Q}[e^{-r(\tau - t)}(S_\tau - K)^+] \ge S_t - K e^{-r(\tau - t)} > S_t - K$$
  Early exercise forfeits positive time value; holding or selling in market is always strictly superior.
- **American Put**:
  When $S_t \to 0$, exercising immediately frees up $\$K$ cash earning positive interest $rK > 0$, establishing an optimal early exercise boundary $S^*(t)$.

---

### Classic 4: Pattern Occurrence & Li's Casino Martingale

> **Interview Prompt (Jane Street / Optiver)**:
> Find the expected tosses until `HTTH` vs `HTHT` appears.

#### Li's Casino Bankroll Argument
At each toss $n$, a new gambler enters with \$1 and sequentially bets on the pattern characters with 2x payoff. Total casino net profit $M_n$ is a martingale with $M_0 = 0$.
At stopping time $T$, total input is $\$T$, and surviving gamblers are those whose entry time matches prefix-suffix overlaps!
$$\mathbb{E}[T_A] = \sum_{k=1}^m 2^k \cdot \mathbf{1}_{\{\text{Prefix}(A, k) = \text{Suffix}(A, k)\}} = (A * A)_2$$

- **For $A = \text{HTTH}$**: Overlaps at $k=1$ (`H`=`H`) and $k=4$ (`HTTH`=`HTTH`) $\implies \mathbb{E}[T] = 2^4 + 2^1 = 18$.
- **For $B = \text{HTHT}$**: Overlaps at $k=2$ (`HT`=`HT`) and $k=4$ (`HTHT`=`HTHT`) $\implies \mathbb{E}[T] = 2^4 + 2^2 = 20$.

`HTHT` takes longer because self-overlapping patterns cluster together, stretching out the dry waiting intervals between clusters.

---

## Module 5: Interview Fast-Recall Matrix

```html
<table>
  <thead>
    <tr>
      <th style="width: 25%;">Problem Type</th>
      <th style="width: 25%;">Recommended Martingale</th>
      <th style="width: 25%;">Analytical Formula</th>
      <th style="width: 25%;">Validation & Pitfalls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Symmetric Ruin Probability</strong></td>
      <td>$M_n = S_n$</td>
      <td>$\mathbb{P}_a = \frac{b}{a + b}$</td>
      <td>Bounded boundary satisfies OST</td>
    </tr>
    <tr>
      <td><strong>Symmetric Exit Time</strong></td>
      <td>$M_n = S_n^2 - n$</td>
      <td>$\mathbb{E}[T] = a \cdot b$</td>
      <td>Wald 2nd identity with $\mu = 0, \sigma^2 = 1$</td>
    </tr>
    <tr>
      <td><strong>Asymmetric Ruin Probability</strong></td>
      <td>$M_n = (q/p)^{S_n}$</td>
      <td>$\mathbb{P}_a = \frac{(q/p)^b - 1}{(q/p)^{a+b} - 1}$</td>
      <td>Check $p(q/p) + q(p/q) = 1$</td>
    </tr>
    <tr>
      <td><strong>Asymmetric Exit Time</strong></td>
      <td>$M_n = S_n - n(p - q)$</td>
      <td>$\mathbb{E}[T] = \frac{a\mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$</td>
      <td>Denominator $p - q \ne 0$</td>
    </tr>
    <tr>
      <td><strong>Pattern Waiting Time</strong></td>
      <td>Casino Net Bankroll Martingale</td>
      <td>$\mathbb{E}[T_A] = \sum 2^k \mathbf{1}_{\{\text{Prefix=Suffix}\}}$</td>
      <td>Penney's game non-transitivity</td>
    </tr>
    <tr>
      <td><strong>Optimal Stopping</strong></td>
      <td>Snell Envelope</td>
      <td>$U_n = \max(Z_n, \mathbb{E}[U_{n+1} \mid \mathcal{F}_n])$</td>
      <td>Backward induction for threshold boundaries</td>
    </tr>
  </tbody>
</table>
```
