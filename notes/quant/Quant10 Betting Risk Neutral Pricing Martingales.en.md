# Quant 10 · Risk-Neutral Pricing and Optimal Betting Strategies: Martingales, Problem of Points & Utility Theory

This chapter addresses a recurring archetype in quantitative trading and research interviews: given a stochastic betting process over time (such as a sports playoff series or sequential coin-flip game), deduce how much capital to bet at any intermediate step, or what the contingent claim on the final outcome is fairly worth at any intermediate score. While presented as a probability brainteaser, the true mathematical foundation is derived from financial derivative pricing: **transforming dynamic hedging problems into static terminal expectations under a risk-neutral measure via no-arbitrage arguments**.

```text
Mental Model for Betting & Martingale Problems:
1. Identify the Martingale: "Is the conditional expectation of my portfolio at the next step equal to its current value?" Under fair betting odds, wealth is a martingale under the risk-neutral measure Q.
2. Characterize the Terminal Boundary: The terminal payoff is binary (achieve target WT or lose all WL). The martingale property gives: Current Wealth = WL + (WT - WL) * P_Q(Win).
3. Compute the Risk-Neutral Probability: Use the classic Problem of Points (Pascal recurrence or Fermat binomial tail) to find P(a, b).
4. Apply the Optional Stopping Theorem: Verify that the stopping time is bounded (e.g., a Best-of-(2N-1) series ends in <= 2N-1 games), ensuring E[W_tau] = W_0 without extra technical conditions.
5. Sizing Bets under Long-Term Growth: When the objective is maximizing long-term geometric compounding rather than replicating a fixed payout, switch to the Kelly Criterion by optimizing E[ln(W)].
6. St. Petersburg Divergence: When raw expectation diverges to infinity, introduce concave utility functions U(W) to evaluate the certainty-equivalent entry price.
```

---

## Classic Interview Problem: Best-of-7 Series Replication & Betting Strategy

> **Problem Statement (Jane Street / SIG / Optiver / Citadel High-Frequency Interview Question)**:
> 
> Team A and Team B are competing in a Best-of-7 championship series (the first team to win 4 games wins the title).
> 
> You start with an initial bankroll of **\$1,000**. Your goal is to design a per-game betting strategy such that:
> 1. If **Team A wins the series**, your total wealth ends up at exactly **\$2,000**;
> 2. If **Team A loses the series** (Team B wins), your total wealth ends up at exactly **\$0**.
> 
> **Market Constraints**:
> - Before each game, a betting market offers fixed **$1{:}1$ fair odds** on either team (a bet of $x$ returns a net profit of $+x$ if the chosen team wins, and a loss of $-x$ if they lose);
> - The strategy must be **self-financing**: no capital injections or withdrawals are allowed during the series;
> - The target payouts must be replicated with $100\%$ precision across **all** possible match paths (whether a 4-0 sweep or a 4-3 comeback).
> 
> **Key Questions**:
> 1. **How much money must you bet on Team A in Game 1?**
> 2. At an intermediate series score $(a, b)$, what is the fair value of your portfolio $W(a, b)$, and what is the required bet $X(a, b)$ on the next game?
> 3. If you believe Team A's **true physical win probability** is $p = 70\%$ (rather than $50\%$), does your required bet or portfolio valuation change? Why or why not?

---

## Module 1: Mathematical Foundations — Martingales, No-Arbitrage & Optional Stopping Theorem

### 1. Definition and Intuition of Martingales

**Core Principle**: A discrete-time stochastic process $\{X_n\}$ adapted to a filtration $\{\mathcal F_n\}$ is a martingale if:

$$
\mathbb E[X_{n+1} \mid \mathcal F_n] = X_n
$$

Intuitively, a martingale formalizes a "fair game": given all history up to step $n$, the best prediction of future wealth is the current wealth. By the tower property of conditional expectation, $\mathbb E[X_m \mid \mathcal F_n] = X_n$ for any $m > n$.

### 2. No-Arbitrage Principle and Risk-Neutral Measure

**Core Principle**: In a market offering $1{:}1$ odds (bet $x$, payoff $+x$ or $-x$), there exists a unique **risk-neutral probability** $q$ under which the expected net return of each bet is zero. For $1{:}1$ odds, $q = \tfrac12$.

Let physical win probability be $p \in (0,1)$. A bet of $x$ on Team A yields:

$$
\text{Net Profit} =
\begin{cases}
+x, & \text{with physical probability } p \\
-x, & \text{with physical probability } 1-p
\end{cases}
$$

Under no-arbitrage pricing, the value of this contract must satisfy:

$$
\mathbb E_{\mathbb Q}[\text{Net Profit}] = q(+x) + (1-q)(-x) = 0 \implies q = \frac12
$$

This risk-neutral probability $q = \tfrac12$ depends **strictly on the payout structure (odds)** and is completely independent of the physical win probability $p$ (even if Team A has $p = 0.90$). Replicating strategies must hedge under $q = \tfrac12$ to ensure self-financing.

### 3. Theorem: Wealth is a Martingale under $\mathbb{Q}$ and Series Ending is a Bounded Stopping Time

**Core Theorem**: If at each score $(a, b)$ your wealth is set to $V(a, b) = W_L + (W_T - W_L) P(a, b)$, where $P(a, b)$ is the risk-neutral probability of Team A winning the series, then $\{W_n\}$ is a martingale under $\mathbb{Q}$. Since the series must terminate in at most $2N-1$ games, the stopping time $\tau \le 2N-1$ is bounded, and by Doob's Optional Stopping Theorem:

$$
W_0 = V(0,0) = \mathbb E_{\mathbb Q}[W_\tau] = W_L + (W_T - W_L) P(0,0)
$$

### 4. Problem of Points: Pascal's Recurrence and Fermat's Binomial Tail

Let $r = N - a$ be games needed by Team A, and $s = N - b$ be games needed by Team B.

**Approach 1 (Pascal's Backward Induction)**:

$$
P(a,b) = q\, P(a+1,b) + (1-q)\, P(a,b+1), \qquad P(N,b)=1 \ (b<N), \quad P(a,N)=0 \ (a<N)
$$

**Approach 2 (Fermat's Combinatorial Extension)**:
Imagine playing out all remaining $r+s-1$ hypothetical games. Team A wins the series if and only if they win at least $r$ of these $r+s-1$ games:

$$
\boxed{P(a,b) = \sum_{k=r}^{r+s-1} \binom{r+s-1}{k} q^k (1-q)^{r+s-1-k}}
$$

For Best-of-7 ($N=4$) at score $(1,0)$: $r=3, s=4, q=\tfrac12$:

$$
P(1,0) = \sum_{k=3}^6 \binom{6}{k} \left( \frac12 \r\right)^6 = \frac{20 + 15 + 6 + 1}{64} = \frac{42}{64} = \frac{21}{32} = 0.65625
$$

---

## Module 2: General Framework & Problem Solution

### 5. General Parametric Framework

Given:
- Initial wealth $W_0$, Best-of-$(2N-1)$ format (first to $N$ wins);
- $1{:}1$ fair odds ($q = \tfrac12$);
- Target wealth $W_T$ if A wins, $W_L$ if A loses.

#### State Fair Value:
$$
W(a,b) = W_L + (W_T - W_L)\, P(a,b)
$$

#### Dynamic Bet Sizing Formula:
To maintain self-financing across both branches:
- If A wins: $W(a, b) + X(a, b) = W(a+1, b)$
- If A loses: $W(a, b) - X(a, b) = W(a, b+1)$

Subtracting yields the exact bet size:

$$
\boxed{X(a,b) = W(a+1,b) - W(a,b) = \frac{W(a+1,b) - W(a,b+1)}{2} = \frac{W_T - W_L}{2}\big[P(a+1,b) - P(a,b+1)\big]}
$$

For Game 1 at $(0,0)$:

$$
X_1 = X(0,0) = \frac{W(1,0) - W(0,1)}{2} = \frac{W_T - W_L}{2}\big[P(1,0) - P(0,1)\big]
$$

### 6. Numerical Substitution for Best-of-7 Original Problem

Parameters: $N=4,\ W_0=1000,\ W_T=2000,\ W_L=0,\ q=\tfrac12$.

#### Risk-Neutral Win Probability Table $P(a,b)$:

| $P(a,b)$ | $b=0$ | $b=1$ | $b=2$ | $b=3$ | $b=4$ (B Wins) |
|---|---|---|---|---|---|
| $a=0$ | 0.50000 | 0.34375 | 0.18750 | 0.06250 | 0 |
| $a=1$ | 0.65625 | 0.50000 | 0.31250 | 0.12500 | 0 |
| $a=2$ | 0.81250 | 0.68750 | 0.50000 | 0.25000 | 0 |
| $a=3$ | 0.93750 | 0.87500 | 0.75000 | 0.50000 | 0 |
| $a=4$ (A Wins) | 1 | 1 | 1 | 1 | — |

#### Fair Value Table $W(a,b) = 2000 \times P(a,b)$ (\$):

| $W(a,b)$ (\$) | $b=0$ | $b=1$ | $b=2$ | $b=3$ | $b=4$ |
|---|---|---|---|---|---|
| $a=0$ | **1000** | 687.5 | 375 | 125 | 0 |
| $a=1$ | 1312.5 | 1000 | 625 | 250 | 0 |
| $a=2$ | 1625 | 1375 | 1000 | 500 | 0 |
| $a=3$ | 1875 | 1750 | 1500 | **1000** | 0 |
| $a=4$ | **2000** | **2000** | **2000** | **2000** | — |

#### Game 1 Bet Size:
$P(1,0) = 21/32 = 0.65625$, $P(0,1) = 11/32 = 0.34375$:

$$
W(1,0) = 2000 \times \frac{21}{32} = \$1312.5, \qquad W(0,1) = 2000 \times \frac{11}{32} = \$687.5
$$

$$
X_1 = \frac{1312.5 - 687.5}{2} = \$312.5
$$

**Original Problem Answers**:
1. **Bet \$312.5 on Team A in Game 1**. If A wins Game 1, wealth becomes $\$1000 + \$312.5 = \$1312.5$; if A loses, wealth becomes $\$1000 - \$312.5 = \$687.5$.
2. In subsequent games, dynamically size bets via $X(a,b) = W(a+1,b) - W(a,b)$.
3. **Physical win probability does not alter replication bets**: The replication strategy hedges market exposure based on the traded odds ($1{:}1 \implies q=0.5$). Sizing bets using $p=0.7$ breaks self-financing and fails to guarantee the exact $\$2000 / \$0$ targets.

### 7. Self-Test with Fresh Numbers: Nonzero Downside Payoff + Physical vs. Risk-Neutral Probability

This is the fullest version of the question as it tends to appear in interviews: swap out the numbers from the template above and add the "does the true win probability change anything" question, bundled into one self-test problem.

**Problem**: Teams A and B play a Best-of-7 final. You start with $W_0=\$7{,}000$ and want a self-financing betting strategy such that your wealth becomes exactly $\$12{,}000$ if A wins the series and exactly $\$2{,}000$ if A loses (note the losing payoff is no longer $\$0$ — the first difference from the template example). Every game trades at fixed $1{:}1$ odds. (a) How much should you bet on A in Game 1? (b) At a score of $(2,1)$ (A ahead), what is your fair wealth $W(2,1)$, and how much should you bet on the next game? (c) If you believe A's true probability of winning the series is $p=70\%$, does that change any of the bet sizes or prices above?

**Derivation (a): check the initial capital for consistency, then solve Game 1**: Before computing anything, there is a check that is easy to skip but matters — given $W_T=12000,\ W_L=2000$, if the strategy must be strictly self-financing starting from $(0,0)$, the fair value of $(0,0)$ is not a free choice. Under fair odds $q=\tfrac12$, $(0,0)$ is symmetric, so $P(0,0)=\tfrac12$, which forces

$$
W(0,0) = W_L + (W_T-W_L)\times\frac12 = 2000 + 10000\times\frac12 = 7000
$$

This exactly matches the stated $W_0=\$7{,}000$ — not a coincidence, but the necessary condition for the problem to be internally consistent (if the given $W_0$ did not equal $(W_T+W_L)/2$, an exact self-financing replication from that $W_0$ would be mathematically impossible, and the right move in an interview is to point that out rather than push ahead). With consistency confirmed, substitute into the template to get $W(1,0)$ and $W(0,1)$:

$$
W(1,0) = 2000 + 10000\times\frac{21}{32} = 8562.5, \qquad W(0,1) = 2000+10000\times\frac{11}{32}=5437.5
$$

$$
X_1 = \frac{W(1,0)-W(0,1)}{2} = \frac{8562.5-5437.5}{2} = \boxed{\$1562.5}
$$

Check: $7000+1562.5=8562.5=W(1,0)$ and $7000-1562.5=5437.5=W(0,1)$, both consistent.

**Derivation (b): holdings and the next bet at score $(2,1)$**: Looking up (or recursing) $P(2,1)=\tfrac{11}{16}$, $P(3,1)=\tfrac78$, $P(2,2)=\tfrac12$, and substituting into the template:

$$
W(2,1) = 2000+10000\times\frac{11}{16} = \boxed{\$8875}
$$

$$
W(3,1) = 2000+10000\times\frac78 = 10750, \qquad W(2,2) = 2000+10000\times\frac12=7000
$$

$$
X_{(2,1)} = \frac{W(3,1)-W(2,2)}{2} = \frac{10750-7000}{2} = \boxed{\$1875}
$$

Check: $8875+1875=10750=W(3,1)$ (fair wealth after A extends the lead to 3–1) and $8875-1875=7000=W(2,2)$ (fair wealth after B ties it at 2–2), both consistent.

**Derivation (c): does knowing the true probability $p=70\%$ change anything?**: None of the numbers above change. This traces back to Module 1, Section 2 — as long as the odds stay locked at $1{:}1$, building a strategy that is strictly self-financing and pays out exactly $W_T$/$W_L$ at the end requires using the risk-neutral probability implied by the odds, $q=\tfrac12$, not anyone's subjective estimate of the true win rate. $p=70\%$ is information about how strong Team A actually is; "how to replicate a fixed terminal payoff exactly using $1{:}1$ bets" is a separate question. What $p=70\%$ does tell you is that **this replication strategy itself is a positive-expectation trade**: if you genuinely believe A's true win probability is $70\%$, then the strategy priced at $q=\tfrac12$ is buying a contingent claim you believe is worth more than the market price — a long-run version of that judgment is exactly the "how much leverage should I take on an edge" question that Example 1 (Kelly criterion) addresses. But as long as the goal remains "replicate $W_T/W_L$ exactly," the bet sizes must be computed with $q=\tfrac12$ only — betting more because you favor A turns the exact-replication strategy into something that no longer hits $W_T$ or $W_L$ precisely; it adds a directional side bet with its own extra variance.

**Common follow-up / interview trap**

> "Is the true probability $p=70\%$ completely useless here, so why does the question even give it?": It is not useless — it answers a different question. Given $p=70\%$, you can evaluate whether this exact-replication trade is worth doing (its true expected profit is positive, since you are paying the $q=50\%$ market price for a claim you believe is worth $70\%$), or how aggressively to size a position if you are willing to tolerate some terminal uncertainty in exchange for a better expected return (the Kelly-criterion family of questions). It cannot be used to answer "how much should I bet to replicate $W_T/W_L$ exactly" — plugging $p=70\%$ into the $P(a,b)$ recursion and using that instead is the single most common way candidates lose points on this question, since the resulting numbers are neither self-financing nor executable as an exact hedge in the market.

---

## Module 3: Three Advanced Examples

### Example 1: Kelly Criterion Variant (Log Wealth Maximization)

**Problem**: Starting with wealth $W_0$, you play repeated coin flips with win probability $p > \tfrac12$. In each round you may wager a fraction $f \in [0,1]$ of your current bankroll. A win multiplies your wealth by $(1+f)$, and a loss multiplies it by $(1-f)$. Find the fraction $f^*$ that maximizes the long-run geometric growth rate.

**Derivation**:
By the Strong Law of Large Numbers, the asymptotic growth rate is:

$$
g(f) = \lim_{n\to\infty} \frac{1}{n} \ln\left( \frac{W_n}{W_0} \r\right) = p \ln(1+f) + (1-p) \ln(1-f)
$$

Differentiating and setting to zero:

$$
g'(f) = \frac{p}{1+f} - \frac{1-p}{1-f} = 0 \implies p(1-f) = (1-p)(1+f) \implies \boxed{f^* = 2p - 1}
$$

Since $g''(f) = -\frac{p}{(1+f)^2} - \frac{1-p}{(1-f)^2} < 0$, $f^* = 2p - 1$ is a strictly concave global maximum.

### Example 2: St. Petersburg Paradox & Utility Functions

**Problem**: A fair coin is tossed until the first Tails appears. If Tails appears on flip $n$, the payout is $\$2^n$.
1. What is the expected raw payoff?
2. If an investor has logarithmic utility $U(W) = \ln W$, what is the maximum entry price $C$ they should pay?

**Derivation**:
1. Raw expectation:
$$
\mathbb E[\text{Payoff}] = \sum_{n=1}^\infty \left( \frac12 \r\right)^n 2^n = \sum_{n=1}^\infty 1 = \infty
$$
2. Expected log utility:
$$
\mathbb E[U(\text{Payoff})] = \sum_{n=1}^\infty \left( \frac12 \r\right)^n \ln(2^n) = \ln 2 \sum_{n=1}^\infty \frac{n}{2^n}
$$
Using the identity $\sum_{n=1}^\infty n x^n = \frac{x}{(1-x)^2}$ for $x = \tfrac12$, $\sum \frac{n}{2^n} = 2$.
Thus $\mathbb E[U] = 2\ln 2 = \ln 4$.
Setting $U(C) = \ln C = \ln 4$ yields $\boxed{C = \$4}$.

### Example 3: Replicating Portfolio Strategy in Best-of-5

**Problem**: In a Best-of-5 series ($N=3$, first to 3 wins), initial capital is $\$1000$. A win by Team A pays $\$3000$, and a loss pays $\$0$. At series score $(1,1)$, what is the fair value of your holding?

**Derivation**:
At $(1,1)$, both teams need 2 more wins ($r=s=2$). Under $q=\tfrac12$, by symmetry $P(1,1) = \tfrac12$:

$$
W(1,1) = 0 + (3000 - 0) \times \frac12 = \boxed{\$1500}
$$

If the score were $(2,1)$ (A needs 1 win, B needs 2 wins), $P(2,1) = \sum_{k=1}^2 \binom{2}{k}(\tfrac12)^2 = \tfrac34$, giving $W(2,1) = 3000 \times \tfrac34 = \$2250$.

---

## Practice Multiple Choice Quizzes

```quiz
title: Quick Quiz 1
question: In risk-neutral pricing of a sports betting series with 1:1 odds, what determines the risk-neutral probability q?
answer: B
A. Subjective belief about which team is stronger
B. The betting odds structure under the no-arbitrage condition (q = 1/2)
C. Historical win-loss records of the two teams
D. The bankroll size of the trader
explanation: The risk-neutral probability q is uniquely determined by the market betting odds to ensure zero expected arbitrage profit. For 1:1 odds, q is always 1/2 regardless of true team strength.
```

```quiz
title: Quick Quiz 2
question: A discrete-time stochastic process X_n is a martingale with respect to F_n if:
answer: C
A. X_n is strictly increasing
B. Var(X_n) is constant
C. E[X_{n+1} | F_n] = X_n
D. The distribution of X_n is stationary
explanation: The defining property of a martingale is that the conditional expectation of the next step equals the current value: E[X_{n+1}|F_n] = X_n.
```

```quiz
title: Quick Quiz 3
question: Why does the formula W_0 = E_Q[W_tau] apply directly via the Optional Stopping Theorem in a Best-of-(2N-1) series?
answer: B
A. Because wealth is non-negative
B. Because the stopping time is strictly bounded (tau <= 2N - 1)
C. Because the game odds are 1:1
D. Because the Optional Stopping Theorem holds unconditionally for all martingales
explanation: The Optional Stopping Theorem holds whenever the stopping time is bounded. A Best-of-(2N-1) playoff series must conclude within at most 2N-1 games, satisfying this condition automatically.
```

```quiz
title: Quick Quiz 4
question: In Fermat's combinatorial approach to the Problem of Points, what is the core equivalence?
answer: A
A. Team A winning the series is identical to Team A winning at least r games in a hypothetical extension of r + s - 1 games
B. The probability is an approximation scaled by a constant factor
C. Both teams are forced to play exactly 2N - 1 games
D. It relies on Poisson approximation
explanation: In r + s - 1 games, Team B can win at most s - 1 games. Thus Team A winning >= r games in the extended sequence is logically equivalent to Team A reaching r wins before Team B reaches s wins in real life.
```

```quiz
title: Quick Quiz 5
question: What objective function does the Kelly Criterion maximize?
answer: C
A. Next-period expected wealth E[W_1]
B. Inverse of the probability of ruin
C. Long-term expected logarithmic growth rate E[ln(W)]
D. Variance-adjusted Sharpe ratio
explanation: The Kelly Criterion maximizes expected log wealth E[ln(W_n)], which by the Law of Large Numbers corresponds to the asymptotic geometric growth rate. Maximizing expected wealth directly would result in reckless 100% bets that lead to certain ruin.
```

```quiz
title: Quick Quiz 6
question: For an even-money bet (1:1 odds) with win probability p > 0.5, what is the optimal Kelly fraction f*?
answer: B
A. f* = p
B. f* = 2p - 1
C. f* = p / (1 - p)
D. f* = ln(p / (1 - p))
explanation: Differentiating g(f) = p ln(1+f) + (1-p) ln(1-f) and setting to zero yields f* = 2p - 1.
```

```quiz
title: Quick Quiz 7
question: What is the raw mathematical expectation of the St. Petersburg Paradox game?
answer: D
A. $2
B. $4
C. $8
D. Infinity
explanation: Each flip n contributes (1/2)^n * 2^n = $1, and summing infinitely many $1 terms yields infinity.
```

```quiz
title: Quick Quiz 8
question: Why does logarithmic utility U(W) = ln(W) resolve the St. Petersburg Paradox with a finite entry fee?
answer: A
A. Logarithm compresses exponential payoffs into linear growth, which converges when multiplied by exponentially decaying probabilities
B. It changes the physical coin flip probabilities
C. It caps the maximum possible payout
D. It eliminates uncertainty
explanation: U(2^n) = n ln(2) grows linearly in n, which is overwhelmed by the exponential decay of (1/2)^n, causing the expected utility sum to converge to ln(4), yielding C = $4.
```

```quiz
title: Quick Quiz 9
question: In a Best-of-5 series tied at (1,1) with 1:1 odds, the risk-neutral win probability P(1,1) equals:
answer: B
A. Dependent on Team A's true win rate p
B. Exactly 1/2 due to symmetry
C. 3/4
D. Indeterminate without additional data
explanation: Both teams need exactly 2 more wins under symmetric 1:1 odds (q = 1/2), making the situation completely symmetric and P(1,1) = 1/2.
```

```quiz
title: Quick Quiz 10
question: In the Best-of-7 series replication problem with W_0 = $1000 and target payouts $2000 / $0, what is the first game bet on Team A?
answer: C
A. $1000
B. $500
C. $312.50
D. $250
explanation: X_1 = (W(1,0) - W(0,1)) / 2 = ($1312.50 - $687.50) / 2 = $312.50.
```

```quiz
title: Quick Quiz 11
question: For the Super St. Petersburg Paradox with payoffs of 2^(2^n), what happens under log utility?
answer: C
A. Log utility still yields a finite certainty equivalent
B. Expected utility is negative
C. Expected utility diverges to infinity again, showing concave utility only shifts the divergence threshold
D. Payoffs become bounded
explanation: When payoffs grow doubly exponentially, even logarithmic utility U(2^(2^n)) = 2^n ln(2) fails to compress the series sufficiently, and the sum diverges again.
```

```quiz
title: Quick Quiz 12
question: Which statement regarding the dynamic replication formula W(a, b) = WL + (WT - WL) * P(a, b) is FALSE?
answer: D
A. It is an exact mathematical consequence of martingale theory and optional stopping, not an empirical approximation
B. It requires the betting strategy to be self-financing
C. It relies on the risk-neutral probability q rather than physical win rate p
D. It holds unconditionally for arbitrary, possibly unbounded stopping times without any regularity requirements
explanation: Optional stopping requires regularity conditions (such as bounded stopping times or uniform integrability). It cannot be applied blindly to unbounded stopping times without verification.
```

```quiz
title: Quick Quiz 13
question: A series pays WT if A wins and WL if A loses (not necessarily 0). With 1:1 odds and a start at (0,0), what must the initial capital W0 satisfy for the strategy to be exactly self-financing?
answer: B
A. W0 can be any value, independent of WT and WL
B. W0 must equal (WT + WL) / 2, because P(0,0) = 1/2 by symmetry
C. W0 must equal WT, to cover the worst case
D. W0 must equal WL, to guarantee no loss
explanation: Under fair odds, (0,0) is symmetric, so P(0,0) = 1/2, giving W(0,0) = WL + (WT-WL) x 1/2 = (WT+WL)/2. If the stated initial capital does not equal this value, the "strictly self-financing, exact terminal payoff" setup is mathematically inconsistent, and that should be flagged before computing anything.
```

```quiz
title: Quick Quiz 14
question: In the general version where the losing payoff WL need not be 0, given the two next-game wealth states W(a+1,b) and W(a,b+1), the fair-odds bet size at (a,b) is:
answer: A
A. (W(a+1,b) - W(a,b+1)) / 2
B. W(a+1,b) - W(a,b+1)
C. (W(a+1,b) + W(a,b+1)) / 2
D. W(a+1,b) directly
explanation: The bet must satisfy both W(a,b)+X=W(a+1,b) (A wins) and W(a,b)-X=W(a,b+1) (A loses) simultaneously; subtracting the two equations gives X=(W(a+1,b)-W(a,b+1))/2, the general form of the template formula at any intermediate score, not just game 1.
```

```quiz
title: Quick Quiz 15
question: Suppose Team A's true probability of winning the series is p=70%, but the replication strategy is still sized using the risk-neutral probability q=50%. Which statement is most accurate?
answer: C
A. This is a mistake; all bet sizes should be recomputed using p=70%
B. p=70% carries no useful information at all and can be ignored
C. Sizing bets with q=50% guarantees the exact WT/WL payout; p=70% tells you this trade has positive expected value from your own point of view, which is a separate question
D. p and q must be equal, or the strategy cannot be executed
explanation: Exact replication of a fixed terminal payoff must use the odds-implied risk-neutral probability q, independent of the true win rate p. p=70% answers a different question: whether this market-priced replication trade is worth doing at all — if the true win probability really is higher, the trade has positive expected value, but that does not change the q used to size the exact-replication bets.
```
