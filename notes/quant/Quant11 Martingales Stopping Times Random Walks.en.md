# Quant 11 · Martingales, Stopping Times & Random Walks: Wald's Identity, Martingale Construction & Optimal Stopping

In the core mathematical and probability interviews at top quantitative hedge funds and proprietary trading firms (Jane Street, Optiver, Citadel, SIG, Jump Trading, Two Sigma), **Martingales and the Optional Stopping Theorem (OST)** are among the most powerful dimensional-reduction mathematical weapons.

This tutorial completes the foundational theory behind the martingale tools introduced in Quant 10: rigorous definitions of martingales and stopping times, the three sufficient conditions of OST and classic counterexamples, the Wald equation family, the four master martingale construction templates, the complete analytical solution for one-dimensional random walks (Gambler's Ruin), and optimal stopping theory applied to classic quantitative interview problems (the Secretary Problem 37% rule, sequential die rolling games, American options early exercise boundaries, and pattern waiting times with Li's casino bankroll martingale).

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
5. Optimal Stopping via Backward Induction: For multi-stage decision problems, construct the Snell Envelope and compute the optimal thresholds via backward dynamic induction.
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

$$\mathbb{E}\left[(S_T - T\mu)^2\right] = \sigma^2 \mathbb{E}[T]$$

For zero-drift walks ($\mu = 0$): $\mathbb{E}[S_T^2] = \sigma^2 \mathbb{E}[T]$.

---

### 3. Wald's Exponential Identity

Let $M(\theta) = \mathbb{E}[e^{\theta X_1}]$. The geometric process $M_n(\theta) = \frac{e^{\theta S_n}}{(M(\theta))^n}$ is a martingale. By OST:

$$\mathbb{E}\left[\frac{e^{\theta S_T}}{(M(\theta))^T}\right] = 1$$

---

## Module 5: Martingale Construction & 4 Master Templates

| Martingale Template | Mathematical Form $M_n$ | Applicable Problem Type | Resulting Algebraic Identity |
| :--- | :--- | :--- | :--- |
| **1. Linear Drift Cancellation** | $S_n - n\mu$ | Expected exit time for drifted random walks $\mathbb{E}[T]$ | $\mathbb{E}[T] = \frac{\mathbb{E}[S_T]}{\mu}$ |
| **2. Quadratic Variance Correction** | $(S_n - n\mu)^2 - n\sigma^2$ | Expected exit time for zero-drift walks $\mathbb{E}[T]$ | $\mathbb{E}[T] = \frac{\mathbb{E}[S_T^2]}{\sigma^2}$ |
| **3. Exponential Geometric** | $\left(\frac{q}{p}\right)^{S_n}$ or $\frac{e^{\theta S_n}}{(M(\theta))^n}$ | Hitting probability $\mathbb{P}(\text{Hit } a)$ for asymmetric walks | $\mathbb{P}_a \left(\frac{q}{p}\right)^a + (1 - \mathbb{P}_a)\left(\frac{q}{p}\right)^{-b} = 1$ |
| **4. Harmonic Function** | $f(X_n)$, with $(P - I)f = 0$ | Hitting probabilities for general Markov chains | $\mathbb{E}[f(X_T)] = f(X_0)$ |

---

## Module 6: One-Dimensional Random Walks (Gambler's Ruin)

Let $S_0 = 0$. Upward probability is $p$, downward is $q = 1 - p$.
Stopping time: $T = \min\{n \ge 0 : S_n = a \text{ or } S_n = -b\}$.

### 1. Symmetric Walk ($p = 1/2$)

- **Hitting Probability**: $M_n = S_n \implies \mathbb{P}_a = \frac{b}{a + b}, \quad \mathbb{P}_{-b} = \frac{a}{a + b}$.
- **Expected Exit Time**: $M_n = S_n^2 - n \implies \mathbb{E}[T] = a \cdot b$.

### 2. Asymmetric Walk ($p \ne 1/2$)

- **Hitting Probability**: $M_n = (q/p)^{S_n} \implies \mathbb{P}_a = \frac{(q/p)^b - 1}{(q/p)^{a+b} - 1}$.
- **Expected Exit Time**: $M_n = S_n - n(p - q) \implies \mathbb{E}[T] = \frac{a \mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$.

---

## Module 7: Optimal Stopping Theory & 4 Classic Interview Problems

### 1. Secretary Problem ($37\%$ Rule)

- Discrete Probability: $\mathbb{P}(\text{Success} \mid k) = \frac{k-1}{n} \sum_{j=k}^n \frac{1}{j-1}$.
- Continuous Limit: $f(x) = -x \ln x \implies x^* = 1/e \approx 36.8\%$, achieving a $36.8\%$ peak success probability.

### 2. Sequential Die Rolling Game

- Backward induction values: $v_1 = 3.5 \to v_2 = 4.25 \to v_3 \approx 4.67 \to v_4 \approx 4.94$.

### 3. American Options Early Exercise

- **American Call (No Dividends)**: $C(S_t, t) > S_t - K$ by Jensen's inequality and discounted asset martingale property; never exercise early.
- **American Put**: Bounded by interest earnings on strike $K$ vs time value, producing boundary $S^*(t)$.

### 4. Coin Pattern Waiting Times & Li's Martingale

- Casino net profit martingale gives: $\mathbb{E}[T_A] = (A * A)_2 = \sum_{k=1}^m 2^k \cdot \mathbf{1}_{\{\text{Prefix}(A, k) = \text{Suffix}(A, k)\}}$.
- $\mathbb{E}[T_{\text{HTTH}}] = 16 + 2 = 18$.
- $\mathbb{E}[T_{\text{HTHT}}] = 16 + 4 = 20$.

---

## Module 8: Interview Quick-Reference Matrix

| Scenario | Recommended Martingale | Analytic Formula / Theorem | Verification Points & Traps |
| :--- | :--- | :--- | :--- |
| **Symmetric Walk Ruin Probability** | $M_n = S_n$ | $\mathbb{P}_a = \frac{b}{a + b}$ | Bounded trajectory guarantees OST holds |
| **Symmetric Walk Expected Time** | $M_n = S_n^2 - n$ | $\mathbb{E}[T] = a \cdot b$ | 2nd Wald identity with $\mu = 0, \sigma^2 = 1$ |
| **Asymmetric Walk Ruin Probability** | $M_n = (q/p)^{S_n}$ | $\mathbb{P}_a = \frac{(q/p)^b - 1}{(q/p)^{a+b} - 1}$ | Derived from $p(q/p) + q(p/q) = 1$ |
| **Asymmetric Walk Expected Time** | $M_n = S_n - n(p - q)$ | $\mathbb{E}[T] = \frac{a\mathbb{P}_a - b(1 - \mathbb{P}_a)}{p - q}$ | Non-zero denominator $p - q \ne 0$ |
| **One-Sided Hitting Time** | Wald Exponential Martingale | $\mathbb{E}[s^{T_a}] = \left(\frac{1 - \sqrt{1 - 4pqs^2}}{2ps}\right)^a$ | $\mathbb{P}(T < \infty) = 1$ but $\mathbb{E}[T] = \infty$ |
| **Pattern Waiting Times** | Casino bankroll net profit | $\mathbb{E}[T_A] = (A * A)_2 = \sum 2^k \mathbf{1}_{\{\text{Prefix=Suffix}\}}$ | Penney's game non-transitivity |
| **Optimal Stopping Decisions** | Snell Envelope | $U_n = \max(Z_n, \mathbb{E}[U_{n+1} \mid \mathcal{F}_n])$ | Backward dynamic programming induction |

---

## Module 9: Quick Practice Quizzes

```quiz
title: Quick Quiz 1
question: For a stochastic process X_n to be a martingale, which of the following is the defining core condition?
answer: C
A. The variance of X_n is constant over time
B. X_n is strictly monotonic
C. E[X_{n+1} | F_n] = X_n (fair game property)
D. The marginal distribution of X_n is invariant over time
explanation: The defining property of a martingale is the conditional expectation condition: given the current information, the best prediction of the next step is the current value itself.
```

```quiz
title: Quick Quiz 2
question: Regarding the formal definition of a stopping time T, which statement is true?
answer: B
A. A stopping time can depend on future outcomes as long as it is finite
B. The event {T <= n} must be measurable with respect to F_n (no peeking into the future)
C. A stopping time must be a deterministic constant
D. The last return to zero in a random walk is a valid stopping time
explanation: A stopping time requires that the decision to stop at step n is made solely on information up to step n. The last return to zero requires looking infinitely into the future.
```

```quiz
title: Quick Quiz 3
question: For a simple symmetric random walk S_n, why does applying E[S_T] = E[S_0] fail for the stopping time T = min{n : S_n = 1}?
answer: D
A. T is not a valid stopping time
B. S_n is not a martingale
C. T is bounded
D. T is almost surely finite, but E[T] = ∞ and the stopped process is unbounded
explanation: Recurrence ensures P(T < ∞) = 1, but the expected stopping time is infinite and the path can drift arbitrarily far to the negative side, violating all three OST conditions.
```

```quiz
title: Quick Quiz 4
question: Under Doob's Optional Stopping Theorem, Condition (C) requires:
answer: A
A. E[T] < ∞ and bounded conditional increments |X_{n+1} - X_n| <= c
B. T must be a deterministic constant
C. The entire martingale must be bounded for all time
D. The stopping time must be zero almost surely
explanation: Condition C allows unbounded trajectories provided the expected stopping time is finite and individual step increments have a uniform upper bound.
```

```quiz
title: Quick Quiz 5
question: In a symmetric random walk starting at 0 with absorbing barriers at -a and b, the probability of hitting b before -a is:
answer: B
A. 1/2
B. a / (a + b)
C. b / (a + b)
D. ab / (a + b)
explanation: Applying OST on S_n gives -a(1 - P_b) + b P_b = 0, which yields P_b = a / (a + b).
```

```quiz
title: Quick Quiz 6
question: In a symmetric random walk starting at 0 with absorbing barriers at -a and b, the expected exit time E[T] is:
answer: C
A. a + b
B. (a + b)^2
C. a · b
D. Undetermined without variance bounds
explanation: Applying OST to the variance martingale M_n = S_n^2 - n yields E[T] = E[S_T^2] = a^2(b/(a+b)) + b^2(a/(a+b)) = a · b.
```

```quiz
title: Quick Quiz 7
question: Why can't we directly apply OST to S_n in an asymmetric random walk (p ≠ q)?
answer: B
A. Stopping times do not exist for asymmetric walks
B. S_n has a systematic drift E[X_{n+1} | F_n] = p - q ≠ 0, so it is not a martingale
C. Asymmetric walks never hit boundaries
D. E[T] is always infinite
explanation: When p ≠ q, S_n has non-zero drift and is not a martingale. One must use the exponential martingale (q/p)^{S_n} instead.
```

```quiz
title: Quick Quiz 8
question: In the exponential martingale M_n = r^{S_n}, how is r = q/p uniquely determined?
answer: A
A. As the non-trivial root of E[r^{X_1}] = p·r + q/r = 1
B. As an arbitrary constant > 1
C. It must be p/q rather than q/p
D. It is determined by the boundary values a and b
explanation: Setting E[r^{X_1}] = 1 yields pr^2 - r + q = 0 => (r - 1)(pr - q) = 0, giving the non-trivial solution r = q/p.
```

```quiz
title: Quick Quiz 9
question: Which statement correctly describes the recurrence/transience of 1D random walks?
answer: C
A. Both symmetric and asymmetric walks are recurrent
B. Both symmetric and asymmetric walks are transient
C. Symmetric random walks are recurrent (with E[T] = ∞), while asymmetric walks are transient
D. Recurrence depends only on the barrier locations
explanation: Symmetric 1D random walks return to any level with probability 1 (recurrent), but expected return time is infinite. Asymmetric walks drift to ±∞ almost surely (transient).
```

```quiz
title: Quick Quiz 10
question: Why does OST hold for bounded Gambler's Ruin but fail for one-sided hitting time T = min{n : S_n = 1}?
answer: D
A. OST fails in both cases
B. OST holds in both cases
C. OST holds for one-sided hitting times but fails with barriers
D. Barriers restrict the process to a bounded interval and ensure E[T] < ∞, satisfying OST conditions
explanation: Finite dual absorbing barriers ensure both a bounded state space and finite expected stopping time E[T] = ab < ∞, fully satisfying OST conditions.
```
