# Quant 07 · Martingales, stopping times, and betting

Location: [[Quant09 Hypothesis Testing Maximum Likelihood|06 Hypothesis Testing]] → This Note → [[Quant12 Brownian Motion Ito Calculus Stopping Times and Options|08 Brownian Motion]]

A martingale is the strict mathematical abstraction of a memoryless fair game. Combined with the Optional Stopping Theorem (OST), it reduces complex path-dependent waiting problems into simple algebraic equations. This note starts from core definitions and derives Gambler's Ruin, risk-neutral pricing, utility optimization, and pattern waiting times.

---

## 1 · Martingales and Stopping Times

### Algebraic Definition

A stochastic process $X_n$ adapted to the filtration $\{\mathcal{F}_n\}$ is a martingale if and only if:

1. **Integrability**: $\mathbb{E}[|X_n|] < \infty$ for all $n$.
2. **Zero Drift**: $\mathbb{E}[X_{n+1} \mid \mathcal{F}_n] = X_n$.

If $\mathbb{E}[X_{n+1} \mid \mathcal{F}_n] \ge X_n$, it is a **submartingale** (expected growth, favorable).
If $\mathbb{E}[X_{n+1} \mid \mathcal{F}_n] \le X_n$, it is a **supermartingale** (expected decay, unfavorable).

### Stopping Time

A random variable $T$ is a stopping time if the occurrence of the event $\{T = n\}$ depends entirely on the information available up to step $n$ ($\mathcal{F}_n$), without requiring knowledge of the future.

```text
Stopping time: The first time two consecutive Heads appear.
Stopping time: The first time an asset price drops below $100.
Not a stopping time: The last time a Head appears.
Not a stopping time: The day the stock price hits its yearly maximum.
```

### Optional Stopping Theorem (OST)

For a martingale $X_n$ and a stopping time $T$, the expectation is conserved at the stopping time ($\mathbb{E}[X_T] = \mathbb{E}[X_0]$) if **any one** of the following three conditions holds:

| Condition | Mathematical Formulation | Physical Meaning |
|---|---|---|
| **1. Bounded Time** | There exists $N$ such that $\mathbb{P}(T \le N) = 1$ | The game has a hard deadline and cannot continue indefinitely. |
| **2. Bounded Process & Finite Time** | $|X_n| \le M \quad (\forall n \le T)$ and $\mathbb{P}(T < \infty) = 1$ | The player has absolute limits on bankroll or debt; no infinite leverage. |
| **3. Bounded Increments & Finite Expected Time** | $|X_{n+1} - X_n| \le c$ and $\mathbb{E}[T] < \infty$ | Maximum single-step win/loss is capped, and the game ends quickly on average. |

### The Counterexample: Martingale Betting Strategy (+1)

In a fair coin toss game, double the bet after every loss (1, 2, 4, 8...). Stop as soon as you win.
Let $T$ be the step of the first win. The number of tosses follows a geometric distribution:

$$
\mathbb{P}(T = n) = \left(\frac{1}{2}\right)^n
$$

This guarantees $\mathbb{P}(T < \infty) = 1$. The player will eventually win.
At step $T$, the accumulated losses from the previous $T-1$ steps are:

$$
\sum_{i=1}^{T-1} 2^{i-1} = 2^{T-1} - 1
$$

Winning $2^{T-1}$ on the $T$-th step yields a net profit:

$$
X_T = 2^{T-1} - (2^{T-1} - 1) = +1
$$

Thus $\mathbb{E}[X_T] = 1 \neq \mathbb{E}[X_0] = 0$. OST fails here because the strategy evades all three conditions:

- **Condition 1 fails**: $T$ can theoretically be arbitrarily large.
- **Condition 2 fails**: Before stopping, $X_n$ can plunge arbitrarily deep into the negative (no lower bound $M$).
- **Condition 3 fails**: Although $\mathbb{E}[T] = 2 < \infty$, the step increment $|X_{n+1} - X_n| = 2^n$ grows exponentially and is not bounded by a constant $c$.

---

## 2 · Random Walks and Wald's Equation

Let $Y_1, Y_2, \dots$ be i.i.d. random variables with mean $\mu$ and variance $\sigma^2$. Define the partial sum $S_n = \sum_{i=1}^n Y_i$.

### Wald's First Identity (Expectation)

If $T$ is a stopping time with respect to $Y_i$ and $\mathbb{E}[T] < \infty$, then:

$$
\mathbb{E}[S_T] = \mu \mathbb{E}[T]
$$

**Derivation**:
Construct the process $M_n = S_n - n\mu$.
$$
\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = \mathbb{E}[S_n + Y_{n+1} - (n+1)\mu \mid \mathcal{F}_n] = S_n + \mu - n\mu - \mu = M_n
$$
Thus $M_n$ is a martingale. Assuming bounded increments $|Y_{n+1} - \mu|$, since $\mathbb{E}[T] < \infty$, it perfectly satisfies OST Condition 3.
Applying OST yields $\mathbb{E}[M_T] = \mathbb{E}[M_0] = 0$, which expands to $\mathbb{E}[S_T] - \mu \mathbb{E}[T] = 0$.

### Wald's Second Identity (Variance)

If $\mu = 0$ and $\mathbb{E}[T] < \infty$, then:

$$
\mathbb{E}[S_T^2] = \sigma^2 \mathbb{E}[T]
$$

**Derivation**:
Construct the process $N_n = S_n^2 - n\sigma^2$.
$$
\begin{aligned}
\mathbb{E}[N_{n+1} \mid \mathcal{F}_n] &= \mathbb{E}[(S_n + Y_{n+1})^2 - (n+1)\sigma^2 \mid \mathcal{F}_n] \\
&= \mathbb{E}[S_n^2 + 2S_n Y_{n+1} + Y_{n+1}^2 \mid \mathcal{F}_n] - (n+1)\sigma^2 \\
&= S_n^2 + 2S_n\mathbb{E}[Y_{n+1}] + \mathbb{E}[Y_{n+1}^2] - n\sigma^2 - \sigma^2 \\
&= S_n^2 + 0 + \sigma^2 - n\sigma^2 - \sigma^2 = N_n
\end{aligned}
$$
$N_n$ is a martingale. Applying OST yields $\mathbb{E}[N_T] = 0$.

---

## 3 · Gambler's Ruin

Start with capital $k$ ($0 < k < N$). In each round, win \$1 with probability $p$ and lose \$1 with probability $q = 1-p$. The game stops upon reaching $N$ (victory) or $0$ (ruin). The stopping time is $T = \inf\{n : S_n = N \text{ or } 0\}$.

### Scenario 1: Fair Game ($p = 1/2$)

**Hitting Probability**: $S_n$ is a martingale, strictly bounded in $[0, N]$ before $T$, satisfying OST Condition 2.

$$
\mathbb{E}[S_T] = \mathbb{E}[S_0] \implies P_N \cdot N + (1 - P_N) \cdot 0 = k
$$

The probability of reaching $N$ is:

$$
P_N = \frac{k}{N}
$$

**Expected Steps**: Use Wald's Second Identity. The step variance is $\sigma^2 = 1$. Since $S_n$ tracks absolute position, the centered process is $S_n - k$.
From $\mathbb{E}[(S_T - k)^2] = 1 \cdot \mathbb{E}[T]$:

$$
\mathbb{E}[T] = P_N(N-k)^2 + (1-P_N)(0-k)^2
$$

Substitute $P_N = k/N$:

$$
\mathbb{E}[T] = \frac{k}{N}(N-k)^2 + \frac{N-k}{N}k^2 = \frac{k(N-k)}{N}(N - k + k) = k(N-k)
$$

### Scenario 2: Unfair Game ($p \neq 1/2$)

**Hitting Probability**: The step expectation is non-zero, so $S_n$ is not a martingale. Construct the **exponential martingale** $M_n = (q/p)^{S_n}$.
Verify the martingale property:

$$
\mathbb{E}\left[\left(\frac{q}{p}\right)^{Y_i}\right] = p \cdot \left(\frac{q}{p}\right)^1 + q \cdot \left(\frac{q}{p}\right)^{-1} = q + p = 1
$$

Thus $\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = M_n \cdot 1 = M_n$.
Apply OST to $M_n$:

$$
\mathbb{E}[M_T] = \mathbb{E}[M_0] \implies P_N \left(\frac{q}{p}\right)^N + (1 - P_N) \left(\frac{q}{p}\right)^0 = \left(\frac{q}{p}\right)^k
$$

Solve for the win rate $P_N$:

$$
P_N = \frac{(q/p)^k - 1}{(q/p)^N - 1}
$$

**Limit Behavior**:
If $p < 1/2$ (disadvantageous game), $q/p > 1$. As the target $N \to \infty$, the denominator grows unbounded:
$$ \lim_{N \to \infty} P_N = 0 $$
Ruin is inevitable against an infinitely capitalized casino in a negative-edge game.
If $p > 1/2$ (advantageous game), $q/p < 1$. As $N \to \infty$, $(q/p)^N \to 0$:
$$ \lim_{N \to \infty} P_N = 1 - \left(\frac{q}{p}\right)^k > 0 $$
With a positive edge, there is a non-zero probability of never going bankrupt, even when aiming for infinite wealth.

**Expected Steps**: Use Wald's First Identity $\mathbb{E}[S_T - k] = (p-q)\mathbb{E}[T]$.

$$
P_N(N - k) + (1 - P_N)(0 - k) = (p-q)\mathbb{E}[T]
$$

Rearranging gives:

$$
\mathbb{E}[T] = \frac{P_N N - k}{p - q}
$$

---

## 4 · Risk-Neutral Pricing and the Problem of Points

In derivative pricing, the focus is not on forecasting the true physical probability $p$, but on constructing a **no-arbitrage replicating portfolio**.

### Best-of-7 Replication Pricing

Teams A and B play a Best-of-7 series. A derivative contract pays $W_T = \$12000$ if A wins the series, and $W_L = \$2000$ if B wins. Betting odds for every single game are fixed at 1:1.

**Core Principle: Regardless of Team A's true win rate, 1:1 odds imply a risk-neutral probability $q = 1/2$.**
If market odds are 1:1, pricing under any other measure introduces an arbitrage opportunity. All no-arbitrage pricing must be calculated using $q=1/2$.

Assume the current series score is $(2,1)$ (A leads).
A needs 2 more wins; B needs 3 more. A maximum of 4 games remain.
A wins the series if they win at least 2 of the next 4 games. We compute the risk-neutral probability $P(2,1)$ using the Problem of Points:

$$
P(2,1) = \sum_{k=2}^4 \binom{4}{k} \left(\frac{1}{2}\right)^4 = \frac{\binom{4}{2} + \binom{4}{3} + \binom{4}{4}}{16} = \frac{6 + 4 + 1}{16} = \frac{11}{16}
$$

The fair value of the derivative at this state is:

$$
W(2,1) = P(2,1) \times 12000 + (1 - P(2,1)) \times 2000 = \frac{11}{16}(12000) + \frac{5}{16}(2000) = 8250 + 625 = 8875
$$

### Delta Hedging

The current portfolio value is $W(2,1) = \$8875$. Consider the next game:

- If A wins, the score becomes $(3,1)$. A needs 1 win, B needs 3. Max 3 games left.
  $P(3,1) = 1 - (1/2)^3 = 7/8$.
  Contract value $W(3,1) = \frac{7}{8}(12000) + \frac{1}{8}(2000) = 10500 + 250 = 10750$.
- If A loses, the score becomes $(2,2)$. Both teams need 2 wins.
  $P(2,2) = 1/2$.
  Contract value $W(2,2) = \frac{1}{2}(12000) + \frac{1}{2}(2000) = 7000$.

To perfectly replicate this derivative (self-financing portfolio), the hedge size $\Delta$ to bet on A in the next game must be:

$$
\Delta = \frac{W(3,1) - W(2,2)}{2} = \frac{10750 - 7000}{2} = 1875
$$

Verification:
If A wins, bankroll becomes $8875 + 1875 = 10750$, exactly matching $W(3,1)$.
If A loses, bankroll becomes $8875 - 1875 = 7000$, exactly matching $W(2,2)$.
Risk is point-wise neutralized.

**The Subjective Probability Trap**:
If someone firmly believes A's true win rate is $p=70\%$, they can use this to evaluate expected profits. However, they must **never** substitute $p=0.7$ for $q=1/2$ in the hedging calculation. Since the replication trades against the market's 1:1 odds, using physical probabilities breaks the self-financing property and fails to replicate the terminal payoffs precisely.

---

## 5 · Utility Optimization and the Kelly Criterion

When the true win rate $p > 1/2$, the game carries a positive expectation. However, maximizing the single-step expectation means betting the entire bankroll, which guarantees ruin upon the first loss.
The Kelly Criterion optimizes the **long-term expected logarithmic wealth (geometric growth rate)**, balancing compounding returns with strict ruin avoidance.

### 1:1 Odds Derivation

Start with capital $W_0$. Wager a fraction $f \in (0, 1)$ of the current bankroll in each round. A win updates wealth by a factor of $(1+f)$; a loss by $(1-f)$.
After $n$ rounds, the wealth $W_n$ is:

$$
W_n = W_0 \prod_{i=1}^n (1+f)^{\mathbf{1}\{\text{Win}\}} (1-f)^{\mathbf{1}\{\text{Loss}\}}
$$

Taking the logarithm and applying the Strong Law of Large Numbers yields the asymptotic geometric growth rate $g(f)$:

$$
g(f) = \lim_{n \to \infty} \frac{1}{n} \ln\left(\frac{W_n}{W_0}\right) = p \ln(1+f) + (1-p) \ln(1-f)
$$

Set the derivative with respect to $f$ (First-Order Condition) to zero:

$$
g'(f) = \frac{p}{1+f} - \frac{1-p}{1-f} = 0
$$

Cross-multiply and expand:

$$
p(1-f) = (1-p)(1+f) \implies p - pf = 1 + f - p - pf \implies 2p - 1 = f
$$

This is the standard Kelly formula:

$$
f^* = 2p - 1
$$

For example, with $p = 0.6$, the optimal strategy is to bet $20\%$ of the bankroll on every flip.

### General Odds $b:1$

If the payout is $b:1$ (bet \$1: win returns original \$1 plus \$b profit; loss costs \$1).
The wealth multipliers become $(1 + bf)$ and $(1 - f)$. The objective function is:

$$
g(f) = p \ln(1 + bf) + (1-p) \ln(1-f)
$$

Differentiate and set to zero:

$$
\frac{pb}{1+bf} - \frac{1-p}{1-f} = 0 \implies pb(1-f) = (1-p)(1+bf)
$$

Expand:

$$
pb - pbf = 1 + bf - p - pbf \implies pb = 1 + bf - p \implies bf = pb + p - 1
$$

This yields the generalized Kelly formula:

$$
f^* = \frac{pb + p - 1}{b} = p - \frac{1-p}{b}
$$

The intuition is: **Win Probability - (Loss Probability / Odds)**.

---

## 6 · Optimal Stopping: The Secretary Problem (37% Rule)

You evaluate $N$ candidates in a random sequence. After each evaluate, you must immediately decide whether to hire them or reject them permanently. The goal is to **maximize the probability of selecting the single best candidate**.

**Strategy Definition**:
Use the first $r-1$ candidates as an observation phase. Evaluate them, reject them, but record the highest score seen.
Starting from candidate $r$, immediately hire the first person whose score strictly exceeds the observation phase maximum.

**Probability Derivation**:
Assume the absolute best candidate is located at position $k$. To successfully hire this candidate, two independent conditions must be met:
1. The candidate must appear after the observation phase: $k \ge r$.
2. Among all $k-1$ candidates appearing before position $k$, the best one must have appeared within the first $r-1$ observation slots. Otherwise, the strategy would trigger prematurely and hire a suboptimal candidate before reaching $k$. Since all permutations are equally likely, the probability of the maximum of the first $k-1$ elements falling within the first $r-1$ positions is $\frac{r-1}{k-1}$.

The prior probability of the best candidate being at any position is $\frac{1}{N}$. Using the Law of Total Probability, the overall success probability $\mathbb{P}(r)$ is:

$$
\mathbb{P}(r) = \sum_{k=r}^N \frac{1}{N} \cdot \frac{r-1}{k-1} = \frac{r-1}{N} \sum_{k=r}^N \frac{1}{k-1}
$$

As $N \to \infty$, define the continuous ratio $x = \lim \frac{r}{N}$. The Riemann sum transforms into an integral:

$$
\mathbb{P}(x) = x \int_x^1 \frac{1}{t} dt = -x \ln x
$$

To find the maximum, set the derivative to zero:

$$
\frac{d}{dx}(-x \ln x) = -1 \cdot \ln x - x \cdot \frac{1}{x} = -\ln x - 1 = 0
$$

Solving for $x$:

$$
x = \frac{1}{e}
$$

Thus, the optimal observation phase should cover $\mathbf{1/e \approx 37\%}$ of the applicant pool. The resulting probability of hiring the absolute best candidate is exactly $1/e \approx 37\%$.

---

## 7 · Pattern Waiting Times & Li's Martingale

Consider tossing a fair coin repeatedly. Find the expected number of tosses until a specific sequence (like `HHT`) appears for the first time. The overlapping nature of string patterns makes standard Markov chain equations cumbersome.

```martingale-rw-demo
```

### The Casino Martingale Setup

Introduce a fictitious casino framework:
1. Before every coin toss, a new gambler enters the casino bringing exactly \$1.
2. The gambler bets their entire bankroll on the first character of the target pattern. If they win, the casino pays 1:1 (bankroll doubles to \$2), and they bet everything on the next character. If they lose any toss, they are wiped out and leave with \$0.
3. The game halts the moment the full pattern appears.

Because every bet is fair (1:1 odds on a 50/50 coin), the casino's net profit $M_n$ is a martingale. Since the game stops in finite expected time, OST applies.
At stopping time $T$, the casino's expected net profit must be 0:
$$ \mathbb{E}[\text{Total Entry Fees} - \text{Total Payouts to Winners}] = 0 $$
Since exactly one gambler entered per round paying \$1, the total entry fee collected is $T$. Thus:
$$ \mathbb{E}[T] = \mathbb{E}[\text{Total Payouts to Winners}] $$

### Comparing HHT, HTH, and HHH

**Scenario 1: Waiting for HHT**
When `HHT` appears (stopping time $T$):
- **Gambler arriving at $T-2$**: Bets `H`, `H`, `T`, winning all three. Leaves with $2^3 = \$8$.
- **Gambler arriving at $T-1$**: Bets `H` and wins (since round $T-1$ was `H`). In round $T$, bets `H` but a `T` was flipped. Loses everything. Leaves with \$0.
- **Gambler arriving at $T$**: Bets `H`, but a `T` was flipped. Leaves with \$0.

Only the $T-2$ gambler gets paid. Total casino payout is \$8:
$$ \mathbb{E}[T_{HHT}] = 8 $$

**Scenario 2: Waiting for HTH**
When `HTH` appears (stopping time $T$):
- **Gambler arriving at $T-2$**: Wins three consecutive bets (`H`, `T`, `H`). Leaves with $2^3 = \$8$.
- **Gambler arriving at $T-1$**: Bets `H` on a `T` toss. Ruined. Leaves with \$0.
- **Gambler arriving at $T$**: Bets `H` on the final toss, which happens to be `H`! Wins the first bet and leaves with $2^1 = \$2$.

Two gamblers get paid. Total casino payout is $8 + 2 = 10$:
$$ \mathbb{E}[T_{HTH}] = 10 $$

**Scenario 3: Waiting for HHH**
When `HHH` appears (stopping time $T$):
- **Gambler arriving at $T-2$**: Wins all three. Leaves with $2^3 = \$8$.
- **Gambler arriving at $T-1$**: Wins their first two bets. Leaves with $2^2 = \$4$.
- **Gambler arriving at $T$**: Wins their first bet. Leaves with $2^1 = \$2$.

All three active gamblers get paid:
$$ \mathbb{E}[T_{HHH}] = 8 + 4 + 2 = 14 $$

**The Rule of Autocorrelation (Penney's Game)**:
Let the pattern length be $k$. Gamblers arriving late can only survive and extract payouts if the pattern's **prefix overlaps with its suffix** (Autocorrelation).
If the pattern overlaps with itself shifted by $j$ positions, add $2^{k-j}$ to the expected waiting time.
This overlapping structural advantage also creates the non-transitive paradox in Penney's Game: for any 3-coin sequence chosen by Player 1, Player 2 can always select a dominating sequence with a higher probability of appearing first (e.g., THH beats HHT).

---

## References

- [Optional Stopping Theorem](https://en.wikipedia.org/wiki/Optional_stopping_theorem)
- [Wald's Equation](https://en.wikipedia.org/wiki/Wald%27s_equation)
- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Secretary Problem](https://en.wikipedia.org/wiki/Secretary_problem)
- [Penney's Game](https://en.wikipedia.org/wiki/Penney%27s_game)
