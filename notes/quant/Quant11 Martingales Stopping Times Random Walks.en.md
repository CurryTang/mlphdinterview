# Quant 11 · Martingales, Stopping Times & Random Walks: Wald's Identity, Martingale Construction & Optimal Stopping

In the core mathematical and probability interviews at top quantitative hedge funds and proprietary trading firms (Jane Street, Optiver, Citadel, SIG, Jump Trading, Two Sigma), **Martingales and the Optional Stopping Theorem (OST)** are among the most powerful dimensional-reduction mathematical weapons.

This tutorial completes the foundational theory behind the martingale tools introduced in Quant 10: rigorous definitions of martingales and stopping times, the three sufficient conditions of OST and classic counterexamples, the Wald equation family, the four master martingale construction templates, the complete analytical solution for one-dimensional random walks (Gambler's Ruin), and optimal stopping theory applied to classic quantitative interview problems (the Secretary Problem 37% rule, sequential die rolling games, American options early exercise boundaries, pattern waiting times with Li's casino bankroll martingale, and sampling without replacement with Doob decomposition).

```text
5-Step Mental Framework for Martingales & Optimal Stopping:
1. Verify Martingale Property: Check if the one-step conditional expectation E[X_{n+1} | F_n] equals X_n (the fair game property).
2. Verify Valid Stopping Time: Ensure the decision to stop at step n depends solely on information available up to step n (no peeking into the future).
3. Validate OST Conditions: Before applying E[X_T] = E[X_0], verify at least one of the three sufficient conditions holds (bounded T, bounded stopped process, or E[T] < ∞ with bounded increments) to avoid the E[T] = ∞ trap.
4. Select & Construct Target Martingale:
   - For hitting probability P(Hit a) -> Construct exponential martingale M_n = (q/p)^{S_n} or harmonic martingale f(X_n);
   - For symmetric zero-drift expected exit time E[T] -> Construct quadratic variance martingale M_n = S_n^2 - n or apply Wald's second identity;
   - For drifted walk expected exit time E[T] -> Construct linear drift-cancelling martingale M_n = S_n - nμ or apply Wald's first identity;
   - For pattern occurrence waiting time E[T_Pattern] -> Construct Li's casino net profit martingale.
5. Optimal Stopping & 1-Step Look-Ahead: Decompose reward process Y_n = N_n + A_n into martingale N_n and predictable drift A_n; if the 1-step increment Δ_n has the monotonic absorbing property (Chow-Robbins), the 1-step look-ahead rule stopping at first Δ_n <= 0 achieves the global optimum!
```

---

## Interactive Lab: Martingales, Random Walks & Optimal Stopping

```martingale-rw-demo
```

---

## Module 1: Mathematical Foundations of Martingales

### 1. The Three Strict Conditions

Let $\{\mathcal{F}_n\}$ be a filtration ($\mathcal{F}_n \subseteq \mathcal{F}_{n+1}$, representing the historical information known up to time $n$). A stochastic process $\{X_n\}$ is a **martingale** with respect to $\{\mathcal{F}_n\}$ if it satisfies:

$$
\text{(i) } X_n \text{ is } \mathcal{F}_n\text{-measurable} \qquad \text{(ii) } \mathbb{E}[|X_n|] < \infty \qquad \text{(iii) } \mathbb{E}[X_{n+1} \mid \mathcal{F}_n] = X_n
$$

- **Condition (i)**: The value of $X_n$ is completely determined by information up to time $n$.
- **Condition (ii)**: Standard integrability ensuring the conditional expectation is well-defined.
- **Condition (iii) (Core)**: Given all information up to step $n$, the best prediction of the next step $X_{n+1}$ is the current state $X_n$. This mathematically defines a **fair game**.

### 2. Tower Property and Fixed-Time Conservation

By applying the **tower property** $\mathbb{E}[\mathbb{E}[\cdot \mid \mathcal{F}_{n+1}] \mid \mathcal{F}_n] = \mathbb{E}[\cdot \mid \mathcal{F}_n]$ iteratively:

$$\mathbb{E}[X_m \mid \mathcal{F}_n] = X_n \quad (\forall m > n)$$

Taking the unconditional expectation yields:

$$\mathbb{E}[X_n] = \mathbb{E}[X_0] \quad (\forall n \ge 0)$$

---

### Example 1: Quadratic Variance Martingale Proof

Let $S_n = \sum_{i=1}^n X_i$ ($S_0=0$) be a simple symmetric random walk where $X_i \in \{-1, +1\}$ with probability $1/2$. Prove that $M_n = S_n^2 - n$ is a martingale with respect to $\mathcal{F}_n = \sigma(X_1, \dots, X_n)$.

**Proof**:
$M_n$ is $\mathcal{F}_n$-measurable and integrable. We verify condition (iii):

$$\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = \mathbb{E}[(S_n + X_{n+1})^2 - (n+1) \mid \mathcal{F}_n] = \mathbb{E}[S_n^2 + 2S_n X_{n+1} + X_{n+1}^2 - n - 1 \mid \mathcal{F}_n]$$

Since $S_n \in \mathcal{F}_n$, $X_{n+1}$ is independent of $\mathcal{F}_n$ with $\mathbb{E}[X_{n+1}]=0$, and $X_{n+1}^2 \equiv 1$:

$$= S_n^2 + 2S_n \cdot \mathbb{E}[X_{n+1}] + \mathbb{E}[X_{n+1}^2] - n - 1 = S_n^2 + 0 + 1 - n - 1 = S_n^2 - n = M_n$$

Q.E.D.

---

## Module 2: Stopping Times and Information Flow

### 1. Formal Definition

A random variable $T \in \{0, 1, 2, \dots\} \cup \{\infty\}$ is a **stopping time** with respect to $\{\mathcal{F}_n\}$ if for every $n \ge 0$:

$$\{T \le n\} \in \mathcal{F}_n$$

Intuition: **The decision to stop at step $n$ must depend solely on information realized up to time $n$, without peeking into future outcomes**.

- **Valid Stopping Time**: The first hitting time $T = \min\{n : S_n = 5\}$.
- **Invalid Non-Stopping Time**: The last return to zero, which requires knowing all future steps.

---

### Example 2: Stopping Time Validity

- (a) $T_1 = \min\{n : S_n = 5\}$ is a valid stopping time since $\{T_1 \le n\} = \bigcup_{k=1}^n \{S_k = 5\} \in \mathcal{F}_n$.
- (b) $T_2 = T_1 - 1$ (the step before hitting 5) is **not** a stopping time, because knowing whether $T_2 = n$ requires inspecting $S_{n+1}$.

---

## Module 3: Optional Stopping Theorem (OST) & 3 Sufficient Conditions

### 1. Theorem Statement

If $\{X_n\}$ is a martingale and $T$ is a stopping time satisfying **any one of the following three conditions**, then:

$$\mathbb{E}[X_T] = \mathbb{E}[X_0]$$

```text
Doob's Optional Stopping Theorem (OST) Sufficient Conditions:
Condition (A) Bounded Stopping Time: P(T <= K) = 1 for some deterministic constant K < ∞;
Condition (B) Bounded Stopped Process: |X_{T ∧ n}| <= M for all n and some constant M < ∞;
Condition (C) Finite Expected Time & Bounded Increments: E[T] < ∞ and |X_{n+1} - X_n| <= c a.s.
```

---

### Example 3: Classic Counterexample (Unbounded Random Walk)

Let $S_n$ be a standard symmetric random walk from $0$, and define $T = \min\{n : S_n = 1\}$.
Upon stopping, $S_T \equiv 1$, so $\mathbb{E}[S_T] = 1$.
However, $\mathbb{E}[S_0] = 0$. Applying $\mathbb{E}[S_T] = \mathbb{E}[S_0]$ blindly yields $1 = 0$!

**Why OST Fails**:
1. $T$ is not bounded (Condition A fails).
2. $S_{T \wedge n}$ can drift arbitrarily far to the negative side (Condition B fails).
3. Even though $\mathbb{P}(T < \infty) = 1$ due to recurrence, **the expected time is infinite** $\mathbb{E}[T] = \infty$ (Condition C fails).

---

## Module 4: The Wald Equation Family

### 1. Wald's First Identity

Let $X_1, X_2, \dots$ be i.i.d. with mean $\mu = \mathbb{E}[X_1]$. If $T$ is a stopping time with $\mathbb{E}[T] < \infty$, then:

$$\mathbb{E}[S_T] = \mathbb{E}[T] \cdot \mathbb{E}[X_1]$$

**Proof**:
Expand $S_T = \sum_{n=1}^\infty X_n \mathbf{1}_{\{T \ge n\}}$. Since $\{T \ge n\} \in \mathcal{F}_{n-1}$, $X_n$ is independent of $\mathbf{1}_{\{T \ge n\}}$. By Fubini-Tonelli:

$$\mathbb{E}[S_T] = \sum_{n=1}^\infty \mathbb{E}[X_n] \mathbb{P}(T \ge n) = \mathbb{E}[X_1] \mathbb{E}[T]$$

---

### 2. Wald's Second Identity

If $\mathbb{E}[X_1^2] < \infty$ and $\mathbb{E}[T] < \infty$, let $\sigma^2 = \text{Var}(X_1)$:

$$\mathbb{E}\left[ (S_T - T\mu)^2 \right] = \sigma^2 \mathbb{E}[T]$$

For zero-drift walks ($\mu = 0$): $\mathbb{E}[S_T^2] = \sigma^2 \mathbb{E}[T]$.

---

### 3. Wald's Exponential Identity

## Module 4: The Wald Equation Family

Wald's identities are direct consequences of martingale theory and Doob's Optional Stopping Theorem (OST) applied to sums of independent and identically distributed (i.i.d.) random variables.

### 1. Wald's First Identity

#### Theorem Statement
Let $X_1, X_2, \dots$ be i.i.d. random variables with finite expectation $\mathbb{E}[|X_1|] < \infty$, and denote the mean by $\mu = \mathbb{E}[X_1]$. Let $T$ be a stopping time with $\mathbb{E}[T] < \infty$. Then the stopped random sum $S_T = \sum_{i=1}^T X_i$ satisfies:

$$\mathbb{E}[S_T] = \mathbb{E}[T] \cdot \mathbb{E}[X_1]$$

#### Rigorous Proof (Indicator Expansion & Fubini Integration)
Expand the stopped sum $S_T$ as a series of indicator functions:

$$S_T = \sum_{n=1}^\infty X_n \mathbf{1}_{\{T \ge n\}}$$

Notice that the event $\{T \ge n\} = \{T \le n - 1\}^c \in \mathcal{F}_{n-1}$ is determined entirely by information up to time $n-1$. Therefore, **the random variable $X_n$ is strictly independent of the indicator variable $\mathbf{1}_{\{T \ge n\}}$**!

Applying the Fubini-Tonelli Theorem to interchange expectation and summation (justified by $\mathbb{E}[T] < \infty$ and $\mathbb{E}[|X_1|] < \infty$):

$$\mathbb{E}[S_T] = \sum_{n=1}^\infty \mathbb{E}\left[ X_n \mathbf{1}_{\{T \ge n\}} \right] = \sum_{n=1}^\infty \mathbb{E}[X_n] \cdot \mathbb{E}\left[ \mathbf{1}_{\{T \ge n\}} \right] = \mathbb{E}[X_1] \sum_{n=1}^\infty \mathbb{P}(T \ge n)$$

Using the tail sum formula for discrete non-negative random variables $\mathbb{E}[T] = \sum_{n=1}^\infty \mathbb{P}(T \ge n)$:

$$\mathbb{E}[S_T] = \mathbb{E}[X_1] \cdot \mathbb{E}[T]$$

Q.E.D.

---

### 2. Wald's Second Identity

If $\mathbb{E}[X_1^2] < \infty$ and $\mathbb{E}[T] < \infty$, denote the variance by $\sigma^2 = \text{Var}(X_1) = \mathbb{E}[X_1^2] - \mu^2$. Then:

$$\mathbb{E}\left[ (S_T - T\mu)^2 \right] = \sigma^2 \mathbb{E}[T]$$

In particular, for zero-drift symmetric walks ($\mu = 0$):

$$\mathbb{E}[S_T^2] = \sigma^2 \mathbb{E}[T]$$

**Proof**: Construct the quadratic variance martingale $M_n = (S_n - n\mu)^2 - n\sigma^2$. Applying OST gives $\mathbb{E}[M_T] = \mathbb{E}[M_0] = 0$, immediately yielding the result.

---

### 3. Wald's Exponential Identity

Let $M(\theta) = \mathbb{E}[e^{\theta X_1}]$ be the moment generating function (MGF). The geometric process:

$$M_n(\theta) = \frac{e^{\theta S_n}}{(M(\theta))^n}$$

is a martingale. Under uniform integrability conditions, applying OST yields:

$$\mathbb{E}\left[ \frac{e^{\theta S_T}}{(M(\theta))^T} \right] = 1$$

- **Differentiating with respect to $\theta$ and setting $\theta \to 0$**: Recovers Wald's first identity $\mathbb{E}[S_T] = \mu \mathbb{E}[T]$;
- **Differentiating twice with respect to $\theta$ and setting $\theta \to 0$**: Recovers Wald's second identity $\mathbb{E}[(S_T - T\mu)^2] = \sigma^2 \mathbb{E}[T]$;
- **Solving Stopping Time Laplace Transforms**: Setting $M(\theta) = e^{-s}$ directly yields $\mathbb{E}[e^{-s T}]$!

---

## Module 5: When to Use Martingales? Master Decision Cheatsheet

### 1. Identifying Problems Suited for Martingale Theory

In quantitative finance interviews and stochastic decision making, martingales and OST provide optimal dimension reduction across **6 primary problem archetypes**:

1. **First Hitting & Absorption Probabilities**: Given an initial state, compute the probability of hitting boundary $A$ before boundary $B$;
2. **Expected Absorption & Exit Times**: Compute the mean number of steps $\mathbb{E}[T]$ required to reach a designated boundary region;
3. **Optimal Stopping & Search Costs**: Balance the tradeoff between terminating now to lock in profit vs paying a continuation/rerolling cost;
4. **Sequential Pattern Matching**: In a stream of coin tosses or characters, find the expected waiting time or racing win rate for patterns (e.g., $HHT$ vs $HTH$);
5. **Proportion Evolution & Limit Theorems**: Sampling without replacement, Pólya urns, voter models, and bounded proportional dynamics;
6. **Branching & Population Extinction**: Individual reproductive branching, total progeny, and extinction probabilities (Galton-Watson processes).

---

### 2. Quantitative Master Martingale Cheatsheet

| Problem Archetype | Recommended Martingale Form $M_n$ / $M_t$ | Governing Algebraic Identity / Theorem | Verification Points & Traps |
| :--- | :--- | :--- | :--- |
| **Symmetric Walk Ruin Probability** | $M_n = S_n$ | $a \mathbb{P}_a + (-b)(1 - \mathbb{P}_a) = S_0 \implies \mathbb{P}_a = \frac{b + S_0}{a + b}$ | Bounded state space guarantees OST Condition B |
| **Symmetric Walk Expected Time** | $M_n = S_n^2 - n$ | $\mathbb{E}[S_T^2] - \mathbb{E}[T] = 0 \implies \mathbb{E}[T] = a \cdot b$ | $\mu = 0, \sigma^2 = 1$; 2nd Wald identity |
| **Asymmetric Walk Ruin Probability** | $M_n = (q/p)^{S_n}$ | $\mathbb{P}_a (q/p)^a + (1 - \mathbb{P}_a)(q/p)^{-b} = 1$ | Derived from $p(q/p) + q(p/q) = 1$ |
| **Asymmetric Walk Expected Time** | $M_n = S_n - n(p - q)$ | $\mathbb{E}[T] = \frac{\mathbb{E}[S_T] - S_0}{p - q} = \frac{a \mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$ | Non-zero denominator $p - q \ne 0$; 1st Wald identity |
| **One-Sided Free Hitting Time** | Wald Exponential Martingale $e^{\theta S_n} / M(\theta)^n$ | $\mathbb{E}[s^{T_a}] = \left( \frac{1 - \sqrt{1 - 4pqs^2}}{2ps} \right)^a$ | $\mathbb{P}(T < \infty) = 1$ but $\mathbb{E}[T] = \infty$ (zero-drift trap) |
| **Pattern Waiting Times** | Casino bankroll net profit martingale | $\mathbb{E}[T_A] = (A * A)_2 = \sum 2^k \mathbf{1}_{\{\text{Prefix}=\text{Suffix}\}}$ | Penney's game non-transitivity |
| **Sampling Without Replacement** | Proportion martingale $M_n = \frac{I_n}{N - n}$ & Doob decomposition | $\Delta_n = \mathbb{E}[Y_{n+1} - Y_n \mid \mathcal{F}_n]$; stop when $\Delta_n \le 0$ | 1-SLA rule; Chow-Robbins Monotone Stopping Theorem |
| **Optimal Stopping with Search Cost** | Excess return Wald martingale $\sum (Z_k - \mu_n)$ | $\mu_{n^*} = \mathbb{E}[\max(X - n^*, 0)] = c \implies \mathbb{E}[\text{Payoff}] = n^* + c$ | Marginal benefit equals marginal search cost |
| **Absorbing Hazard Roll-or-Stop** | Absorbing submartingale $Y_n = S_n \mathbf{1}_{\{T > n\}}$ | In safe zone $\mathbb{E}[Y_{n+1} - Y_n \mid \mathcal{F}_n] = \mu_D > 0$ | Proves $\sup_\tau \mathbb{E}[Y_\tau] > S_0 \implies$ continue rolling |
| **Continuous Drifted Diffusion Exit** | Exponential martingale $M_t = \exp\left( -\frac{2\mu}{\sigma^2} X_t \right)$ | $\mathbb{P}(X_\tau = b) = \frac{e^{\gamma a} - 1}{e^{\gamma a} - e^{-\gamma b}}$ ($\gamma = \frac{2\mu}{\sigma^2}$) | Scale function for Brownian motion with drift |
| **Continuous Drifted Expected Time** | Linear martingale $N_t = X_t - \mu t$ | $\mathbb{E}[\tau] = \frac{\mathbb{E}[X_\tau]}{\mu} = \frac{b p_b - a(1 - p_b)}{\mu}$ | Drift dominates mean exit time |
| **Branching Process Extinction** | Normalized size $Z_n / m^n$ & generating function $s^{Z_n}$ | Extinction probability $\pi$ is minimal non-negative root of $G(s) = s$ | Almost sure extinction if $m \le 1$ |
| **Pólya's Urn Proportion Limit** | Proportion martingale $M_n = \frac{R_n}{R + B + n c}$ | $M_n \to M_\infty$ almost surely, with $M_\infty \sim \text{Beta}\left( \frac{R}{c}, \frac{B}{c} \right)$ | Martingale Convergence Theorem |
| **Card Drawing Without Replacement** | Indicator symmetry / exchangeability | Expected cards to 1st red is $\frac{B+R+1}{R+1}$; remaining black is $\frac{B}{R+1}$ | $R$ red cards partition $B$ black cards into $R+1$ equal bins |

---

### Module 6: 9 Core Quant Interview Problems & Rigorous Derivations

---

### Problem 1: Optimal Stopping with Search / Reroll Cost & Wald Martingale

> **General Problem Statement (Optiver / Jane Street / IMC Core Interview Question)**:
> 
> Let $X_1, X_2, \dots$ be i.i.d. draws from a discrete uniform distribution over $\{1, 2, \dots, K\}$ (or general distribution $p_x$).
> Rules:
> - After each roll $X_k$, you may choose to **accept $X_k$ and terminate**, or **pay a fixed reroll fee $c > 0$ to roll again**;
> - If stopped at roll $N$, the net payout is $\text{Payoff} = X_N - c(N - 1)$;
> - **Find the general equation for the optimal threshold $n^*$ and the maximum expected net payout.**

#### 1. Martingale Classification
This problem belongs to **Excess Return Wald Martingales with Marginal Search Cost Indifference**.

#### 2. Excess Return Wald Martingale Construction
The optimal policy is a threshold rule: stop when $X_k > n$.
Let $N = \inf\{k \ge 1 : X_k > n\}$. Define the excess return $Z_k = \max(X_k - n, 0)$ with mean $\mu_n = \mathbb{E}[Z_k]$.
Construct the Wald martingale:

$$
M_m = \sum_{k=1}^m (Z_k - \mu_n)
$$

Applying OST gives $\mathbb{E}[M_N] = 0 \implies \mathbb{E}[\sum_{k=1}^N Z_k] = \mu_n \mathbb{E}[N]$.
Since $Z_k = 0$ for $k < N$ and $Z_N = X_N - n$:

$$
\sum_{k=1}^N Z_k = X_N - n \implies \mathbb{E}[X_N] = n + \mu_n \mathbb{E}[N]
$$

#### 3. Determining Optimal Threshold $n^*$
Substituting into expected net payoff:

$$
\mathbb{E}[\text{Payoff}] = \mathbb{E}[X_N] - c\mathbb{E}[N] + c = n + c + (\mu_n - c)\mathbb{E}[N]
$$

To maximize payoff, the optimal threshold $n^*$ equates marginal expected excess return to marginal reroll cost:

$$
\boxed{\mu_{n^*} = \mathbb{E}[\max(X - n^*, 0)] \approx c} \implies \boxed{\mathbb{E}[\text{Payoff}^*] = n^* + c}
$$

For a $K$-sided fair die: $\mu_n = \frac{(K - n)(K - n + 1)}{2K} \approx c$.

**Special Case ($K=6, c=1$)**:
- $n=3 \implies \mu_3 = \frac{3 \times 4}{12} = 1 = c$;
- Optimal strategy: stop if $X \ge 4$; expected net payoff is $n^* + c = 3 + 1 = \mathbf{4}$.

---

### Problem 2: Pattern Waiting Times & Li's Casino Martingale

> **General Problem Statement (Jane Street / Optiver Signature Problem)**:
> 
> In a sequence of fair coin tosses ($H$ and $T$), find the expected waiting time $\mathbb{E}[T_{HHT}]$ for pattern **$HHT$** to appear. Explain why it differs from $HTH$ and $HHH$.

#### 1. Martingale Classification
This problem belongs to **Li's Casino Bankroll Net Profit Martingale & Autocorrelation Polynomials**.

#### 2. Casino Bankroll Martingale Setup
- At each time step $n=1, 2, \dots$, a new gambler enters the casino, investing $\$1$;
- Gambler $n$ bets on $H$: if wrong, loses $\$1$ and exits; if correct, capital doubles to $\$2$, bet on $H$ at step $n+1$;
- If step $n+1$ is $H$, capital doubles to $\$4$, bet on $T$ at step $n+2$; if correct, wins $\$8$ and exits (the game ends).
- The casino's cumulative net profit $M_n = (\text{total bets}) - (\text{total payouts})$ is a fair martingale.

#### 3. Applying OST
When $HHT$ appears at stopping time $T$, $T$ gamblers have entered investing $\$T$. Checking active gamblers:
1. Gambler entering at $T-2$: matched $H, H, T$, wins $\$2^3 = \$8$;
2. Gambler entering at $T-1$: bet $H, H$, while actual rolls were $H, T$, wins $\$0$;
3. Gambler entering at $T$: bet $H$, while actual roll was $T$, wins $\$0$.

By OST:

$$
\mathbb{E}[T - 8 - 0 - 0] = 0 \implies \boxed{\mathbb{E}[T_{HHT}] = 8}
$$

#### 4. Pattern Autocorrelation Comparison
- **$HHT$** (no self-overlap): $\mathbb{E}[T_{HHT}] = 2^3 = 8$;
- **$HTH$** (overlap `H` of length 1): $\mathbb{E}[T_{HTH}] = 2^3 + 2^1 = 10$;
- **$HHH$** (overlaps `H`, `HH` of length 1, 2): $\mathbb{E}[T_{HHH}] = 2^3 + 2^2 + 2^1 = 14$.

---

### Problem 3: Sampling Without Replacement with Penalty & Doob Decomposition

> **General Problem Statement (Akuna / SIG / Citadel Quant Interview)**:
> 
> A shuffled deck contains **$B$ blue cards** and **$1$ red card** (total $N = B + 1$ cards).
> Rules:
> - You draw cards one by one **without replacement**;
> - Each blue card drawn adds $+1$ point to your score. You may choose to **stop immediately and keep your accumulated points**, or **continue drawing**;
> - If you draw the red card, the game terminates immediately with penalty **$-(n-1)$** (you forfeit all previously accumulated $n-1$ points);
> - **Find the general optimal stopping threshold $n^*$ and the maximum expected payout formula.**

#### 1. Martingale Classification
This problem belongs to **Proportion Martingales in Sampling Without Replacement and Predictable Doob Drift Decomposition**.

#### 2. Base Martingale Construction
Let $T \in \{1, 2, \dots, B+1\}$ be the round in which the red card is drawn ($P(T = k) = \frac{1}{B+1}$ for all $k$).
Let $I_n = \mathbf{1}_{\{T > n\}}$ indicate that all first $n$ cards were blue.
Define the proportion process:

$$
M_n = \frac{I_n}{B + 1 - n} \quad (0 \le n \le B)
$$

**Martingale Verification**:
Conditioned on $\mathcal{F}_n$ and $I_n = 1$, there remain $B+1-n$ cards ($1$ red, $B-n$ blue):

$$
\mathbb{E}[I_{n+1} \mid \mathcal{F}_n] = I_n \cdot \frac{B - n}{B + 1 - n}
$$

Dividing both sides by $(B + 1) - (n + 1) = B - n$:

$$
\mathbb{E}\left[ \frac{I_{n+1}}{B + 1 - (n + 1)} \;\middle|\; \mathcal{F}_n \right] = \frac{I_n}{B + 1 - n} = M_n
$$

When $I_n = 0$, $I_{n+1} \equiv 0$. Thus $M_n$ is a strict discrete-time martingale.

#### 3. Payoff Process & Doob Decomposition
Let $Y_n$ be the payoff upon stopping at step $n$. Conditioned on $I_n = 1$ (currently holding $n$ points), the 1-step conditional expected reward increment is:

$$
\Delta_n = \mathbb{E}[Y_{n+1} - Y_n \mid \mathcal{F}_n] = \frac{B - n}{B + 1 - n}(n + 1) + \frac{1}{B + 1 - n}(-n) - n = \frac{B - 3n}{B + 1 - n} I_n
$$

The explicit Doob decomposition is $Y_n = N_n + A_n$ where:

$$
A_n = \sum_{k=0}^{n-1} \frac{B - 3k}{B + 1 - k} I_k
$$

#### 4. 1-Step Look-Ahead (1-SLA) & Optimal Stopping Rule
Analyzing $\Delta_n = \frac{B - 3n}{B + 1 - n}$:
- For $n < B/3$: $\Delta_n > 0$ (Submartingale regime, continue);
- For $n \ge B/3$: $\Delta_n \le 0$ (Supermartingale regime, stop!).

Because $B - 3n$ is strictly decreasing, the stopping region is absorbing. By the **Chow-Robbins Monotone Stopping Theorem**, the 1-SLA rule achieves the exact global optimum:

$$
\boxed{n^* = \left\lfloor \frac{B}{3}  ight floor}
$$

> **Rule of Thumb**: Under full forfeiture penalty, always stop after drawing **one-third** of the total blue cards!

#### 5. General Expected Payout Formula
Under the threshold policy $n^*$:

$$
\mathbb{E}[\text{Payoff}] = \frac{B + 1 - n^*}{B + 1} \cdot n^* - \sum_{k=1}^{n^*} \frac{1}{B + 1} (k - 1) = \boxed{\frac{n^*(2B + 3 - 3n^*)}{2(B + 1)}}
$$

**Special Case ($B=9, R=1$)**:
- $n^* = \lfloor 9/3  floor = 3$;
- $\mathbb{E}[\text{Payoff}] = \frac{3(18 + 3 - 9)}{20} = \frac{36}{20} = \mathbf{1.8}$ points.

---

### Problem 4: Drifted Brownian Motion Two-Sided Exit & Scale Function

> **General Problem Statement (Citadel / Two Sigma / Morgan Stanley Strats)**:
> 
> Let log-asset prices follow drifted Brownian motion $X_t = \mu t + \sigma W_t$ ($X_0 = 0, \mu > 0, \sigma > 0$). Stop-loss is $-a < 0$ and take-profit is $b > 0$. Stopping time is $\tau = \inf\{t \ge 0 : X_t = -a \text{ or } X_t = b\}$.
> Find hitting probability $p_b = \mathbb{P}(X_\tau = b)$ and expected exit time $\mathbb{E}[\tau]$.

#### 1. Martingale Classification
This problem belongs to **Exponential MGF Martingales & Linear Drift Cancellation in Continuous Diffusions**.

#### 2. Exponential Martingale for Hitting Probability
Construct $M_t = \exp(-\gamma X_t)$ with $\gamma = \frac{2\mu}{\sigma^2}$. By OST $\mathbb{E}[M_\tau] = 1$:

$$
p_b e^{-\gamma b} + (1 - p_b) e^{\gamma a} = 1 \implies \boxed{p_b = \frac{e^{\gamma a} - 1}{e^{\gamma a} - e^{-\gamma b}}} \quad \left( \gamma = \frac{2\mu}{\sigma^2} \right)
$$

#### 3. Linear Martingale for Expected Exit Time
Construct linear martingale $N_t = X_t - \mu t$. By OST $\mathbb{E}[N_\tau] = 0$:

$$
\boxed{\mathbb{E}[\tau] = \frac{\mathbb{E}[X_\tau]}{\mu} = \frac{b p_b - a(1 - p_b)}{\mu}}
$$

---

### Problem 5: 2-Urn Ball Placement Game & Symmetric Random Walk Martingale

> **General Problem Statement (Jane Street / Two Sigma Capstone Game)**:
> 
> Two urns start empty. The game lasts $N = 2m$ rounds (even). In each round, an urn is chosen with probability $1/2$. The player must choose either to **deposit a ball** or **retrieve a ball**. Successfully retrieving a ball from a non-empty urn scores $+1$.
> **Find the optimal policy and the maximum expected balls retrieved.**

#### 1. Martingale Classification
This problem belongs to **Pathwise Exchange Argument Strategy Reduction + Binomial Convolution + Symmetric Random Walk Absolute First Moment**.

#### 2. Strategy Reduction via Exchange Argument
"Deposit $k$ balls first, then retrieve $N-k$ balls" pointwise dominates any interleaved sequence on every sample path.

#### 3. Binomial Convolution & Symmetric Random Walk
$X \sim \text{Bin}(k, 1/2), R \sim \text{Bin}(N-k, 1/2)$ are independent. Total retrieved $Y = \min(X, R) + \min(k-X, N-k-R)$.
Using $\min(a, b) = \frac{a+b-|a-b|}{2}$ and convolution $Z = X + (N-k-R) \sim \text{Bin}(N, 1/2)$:

$$
Y = \frac{N}{2} - \frac{|Z - (N-k)| + |Z - k|}{2} \le \frac{N}{2} - \left| Z - \frac{N}{2}  ight|
$$

Optimal allocation is $k^* = N/2 = m$. Mapping $2Z - N \stackrel{d}{=} S_N$ (an $N$-step symmetric random walk):

$$
\mathbb{E}[Y] = m - \frac{1}{2} \mathbb{E}[|S_{2m}|] = \boxed{m - m \binom{2m}{m} 2^{-2m} \approx m - \sqrt{\frac{m}{\pi}}}
$$

**Special Case ($N = 100, m = 50$)**:
- Optimal policy: **Deposit in first 50 rounds, retrieve in last 50 rounds**;
- $\mathbb{E}[Y] = 50 - 50 \binom{100}{50} 2^{-100} \approx 50 - \sqrt{50/\pi} \approx \mathbf{46.011}$.

---

### Problem 6: Card Drawing Without Replacement Symmetry

> **General Problem Statement (Jane Street / SIG High-Frequency Interview)**:
> 
> A shuffled deck has $R$ red cards and $B$ black cards.
> 1. Find expected cards until the 1st red card: $\mathbb{E}[T_1]$;
> 2. Find expected black cards remaining when the last red card is drawn: $\mathbb{E}[\text{Remaining Black}]$.

#### 1. Martingale Classification
This problem belongs to **Indicator Symmetry & Exchangeable Interval Partitioning**.

#### 2. Indicator Symmetry & Equal Bin Expectation
$R$ red cards partition $B$ black cards into $R+1$ identically distributed bins $I_0, I_1, \dots, I_R$.
By exchangeability: $\mathbb{E}[I_k] = \frac{B}{R+1}$.
1. $\mathbb{E}[T_1] = \mathbb{E}[I_0] + 1 = \boxed{\frac{B + R + 1}{R + 1}}$.
2. $\mathbb{E}[\text{Remaining Black}] = \mathbb{E}[I_R] = \boxed{\frac{B}{R + 1}}$.

For a standard deck ($R=26, B=26$):
- $\mathbb{E}[T_1] = \frac{53}{27} \approx \mathbf{1.963}$;
- $\mathbb{E}[\text{Remaining Black}] = \frac{26}{27} \approx \mathbf{0.963}$.

---

### Problem 7: Square Numbers Hazard Roll-or-Stop Decision & Absorbing Submartingale

> **General Problem Statement (Citadel / Jump Trading Quant Interview)**:
> 
> A player accumulates sum $S_n = S_0 + \sum_{k=1}^n D_k$ where step increments $D_k \in \{1, \dots, K\}$ are i.i.d. with mean $\mu_D > 0$.
> There exist hazard points $\mathcal{H} = \{H_1, H_2, \dots\}$ (such as perfect squares $1^2, 2^2, 3^2, \dots$).
> If the cumulative sum lands exactly on a hazard point ($S_n = H$), the score drops to $0$ and is permanently absorbed at $0$.
> **Should a player at $S_0$ (distance $d = H - S_0 \le K$) stop immediately or continue rolling?**

#### 1. Martingale Classification
This problem belongs to **Absorbing Submartingales with Safe-Zone Drift & Multi-Step Strategy Lower Bounds**.

#### 2. Submartingale Formulation
Let $T = \inf\{n \ge 1 : S_n = H\}$ and $Y_n = S_n \mathbf{1}_{\{T > n\}}$.
Once $S_n$ safely clears $H$, over the safe interval $[H+1, H_{\text{next}} - K]$:

$$
\mathbb{E}[Y_{n+1} - Y_n \mid \mathcal{F}_n] = \mu_D > 0
$$

Thus $Y_n$ is a strict submartingale in the safe zone.

#### 3. Two-Step Strategy Lower Bound
Construct stopping rule $\tau$: roll once ($n=1$); if $D_1 \ne d$, roll a second time ($n=2$) and stop.

$$
\mathbb{E}[Y_\tau] = \frac{K - 1}{K} S_0 + \frac{1}{K} \left( \sum_{k \ne d} k \right) + \frac{K - 1}{K} \mu_D
$$

**Special Case ($S_0 = 35, H = 36 = 6^2, K = 6, \mu_D = 3.5$)**:
- $\mathbb{P}(D_1 \ne 1) = 5/6$, $\mathbb{E}[S_1 \mid D_1 \ge 2] = 39$;
- $\mathbb{E}[Y_2 \mid D_1 \ge 2] = 39 + 3.5 = 42.5$;
- $\mathbb{E}[Y_\tau] = \frac{5}{6} \times 42.5 = \frac{212.5}{6} \approx \mathbf{35.42} > 35$.
- **Conclusion**: The player should **continue rolling**!

---

### Problem 8: Pólya's Urn & Proportion Limit Theorems

> **General Problem Statement (Jane Street / SIG Classical Probability Question)**:
> 
> Urn starts with $R$ red and $B$ black balls. Draw a ball, replace it, and add $c$ balls of the same color. Prove proportion $M_n = \frac{R_n}{R+B+nc}$ is a martingale and find its limit distribution.

#### 1. Martingale Classification
This problem belongs to **Bounded Proportion Martingales & Doob's Martingale Convergence Theorem**.

#### 2. Martingale Verification
$\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = M_n \frac{R_n+c}{T_n+c} + (1-M_n)\frac{R_n}{T_n+c} = \frac{M_n c + M_n T_n}{T_n+c} = M_n$.
$M_n \in [0, 1]$ is a bounded martingale.

#### 3. Limit Distribution
By Doob's Martingale Convergence Theorem, $M_n \to M_\infty$ a.s. with:

$$
\boxed{M_\infty \sim \text{Beta}\left( \frac{R}{c}, \frac{B}{c} \right)}
$$

For $R=1, B=1, c=1$, $M_\infty \sim \text{Uniform}(0, 1)$.

---

### Problem 9: Galton-Watson Branching Processes & Extinction

> **General Problem Statement (Citadel / Two Sigma Stochastic Processes Question)**:
> 
> Let offspring distribution have pgf $G(s) = \mathbb{E}[s^K]$ and mean $m = \mathbb{E}[K]$. Prove extinction probability $\pi$ is the smallest non-negative root of $G(s) = s$.

#### 1. Martingale Classification
This problem belongs to **Normalized Branching Martingales & Probability Generating Function Bounded Martingales**.

#### 2. Pgf Martingale Formulation
For any root $s \in [0, 1]$ of $G(s) = s$, $Y_n = s^{Z_n}$ is a bounded martingale ($0 \le Y_n \le 1$).
$\mathbb{E}[Y_{n+1} \mid \mathcal{F}_n] = (G(s))^{Z_n} = s^{Z_n} = Y_n$.

#### 3. Extinction Fixed Point
Applying dominated convergence: $\mathbb{E}[Y_n] = s = \lim_{n \to \infty} \mathbb{E}[s^{Z_n}] = \pi \cdot s^0 + (1 - \pi) \cdot 0 = \pi \implies \boxed{\pi = G(\pi)}$.


