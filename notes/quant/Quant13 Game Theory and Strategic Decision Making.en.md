# Quant 09 · Game theory

Course: [[Quant12 Brownian Motion Ito Calculus Stopping Times and Options|08 Brownian Motion]] → This note → [[Quant14 Financial Markets Asset Classes and Portfolio Theory|10 Markets]]

Game theory analyzes strategic interactions, backward induction, and asymmetric information.

## 1 · Core concepts and equilibria

| Concept | Definition |
|---|---|
| Normal-form game | Defined by players, strategy spaces, and payoffs. Used for static simultaneous games. |
| Extensive-form game | Represented by a game tree with decision nodes, information sets, and action sequences. |
| Dominant and dominated strategies | A strictly dominant strategy yields a strictly higher payoff regardless of opponents' actions. Rational players never play strictly dominated strategies. |
| Nash equilibrium | No player has an incentive to unilaterally deviate from their chosen strategy (mutual best responses). |
| Mixed strategy and indifference | Every pure strategy in a mixed strategy support must yield the exact same expected payoff. |
| Minimax theorem | In zero-sum games, maximizing the lower bound of one's payoff equates to minimizing the upper bound of the opponent's payoff, driving expected payoff equalization. |
| Subgame perfect equilibrium (SPE) | A strategy profile that induces a Nash equilibrium in every subgame. Typically solved via backward induction to eliminate non-credible threats. |

---

## 2 · Dynamic games and backward induction

### Pirate gold

5 pirates (1 is most senior, 5 is most junior) divide 100 gold coins. 1 proposes an allocation. It requires $\ge 50\%$ approval to pass, otherwise 1 is thrown to sharks. Preferences: Survival > Gold > Bloodlust.
Backward induction:
- 2 remaining (4, 5): 4 proposes `(100, 0)` and votes for it.
- 3 remaining (3, 4, 5): 3 needs 2 votes. Giving 5 one coin secures their vote ($1 > 0$), so `(99, 0, 1)`.
- 4 remaining (2, 3, 4, 5): 2 needs 2 votes. Giving 4 one coin is enough, so `(98, 0, 1, 0)`.
- 5 remaining: 1 needs 3 votes. Buying 3 and 5 with one coin each yields `(96, 0, 1, 0, 1)`.

### Tigers and sheep

$N$ perfectly rational tigers and 1 sheep. A tiger that eats the sheep turns into a sheep. Preferences: Survival > Eating sheep.
- $N=1$: Eats (safe).
- $N=2$: Does not eat (eating turns it into a sheep to be eaten by the remaining tiger).
- $N=3$: Eats (after eating, 2 tigers remain and dare not eat).
Parity recursion: An odd number guarantees the first tiger eats; an even number guarantees no tiger eats.

### The truel

A hits with $1/3$, B with $2/3$, C with $1$. Sequential shooting (A $\to$ B $\to$ C). Players can intentionally shoot into the air.
A's win probability in 2-player subgames:
- A shoots first against B: $P_{AB} = 1/3 + (2/3)(1/3)P_{AB} \implies P_{AB} = 3/7$.
- A shoots first against C: $P_{AC} = 1/3 + (2/3) \times 0 = 1/3$.
With all 3 alive, B must target C, and C must target B. A's first shot options:
- Hit C: Leads to A vs B with B shooting first. A's survival is $(1 - 2/3) \times 3/7 = 1/7$. Total probability: $1/3 \times 1/7 + 2/3 \times 3/7 = 1/3$.
- Hit B: A is killed by C. Total probability: $2/7$.
- Shoot in the air: B shoots C. If B kills C (prob $2/3$), A faces B shooting first (win prob $3/7$). If B misses (prob $1/3$), C kills B, and A faces C shooting first (win prob $1/3$). A's total probability: $(2/3)(3/7) + (1/3)(1/3) = 25/63 \approx 39.7\%$.
The optimal decision is to shoot into the air.

---

## 3 · Static and combinatorial games

### Guess 2/3 of the average

$N$ players pick a number in $[0, 100]$. The closest to $2/3$ of the group average wins.
- Strategy space $S_0 = [0, 100]$. The average is at most 100, so the target is at most $66.67$. Bidding over $66.67$ is strictly dominated. Space contracts to $S_1 = [0, 66.67]$.
- Round $k$ contraction: $S_k = [0, 100 \times (2/3)^k]$.
The limit is $S_\infty = \{0\}$. The unique Nash equilibrium is for everyone to bid $0$.

### Auctions and revenue equivalence

Bidders have independent private valuations $v \sim U[0, 1]$:
- **Second-price auction**: If bidding $b > v$, winning when the second price $P_2 \in (v, b)$ causes a net loss $v - P_2 < 0$. If $b < v$, losing when $P_2 \in (b, v)$ forfeits a $>0$ profit. Thus, truthful bidding $b(v) = v$ is weakly dominant.
- **First-price auction**: Expected profit $E[\Pi] = (v - b) \beta(b)^{n-1}$. The FOC is $-( \beta(b) )^{n-1} + (v-b)(n-1)\beta(b)^{n-2}\beta'(b) = 0$. Substituting the symmetric equilibrium $\beta(b)=v$ yields $b(v) = \frac{n-1}{n} v$.
- **All-pay auction**: Expected profit $E[\Pi] = v \beta(b)^{n-1} - b$. FOC yields $b(v) = \frac{n-1}{n} v^n$.
The revenue equivalence theorem states all three mechanisms yield an identical expected seller revenue of $\frac{n-1}{n+1}$.

### Corporate acquisition

Target company T's true value $V \sim U[0, 100]$. Under company A, value rises to $1.5V$. A makes a flat bid $B$, and T accepts if $B \ge V$.
Due to adverse selection, the acquisition succeeds only if $V \le B$. The expected true value given success is the truncated expectation $E[V \mid V \le B] = B/2$.
A's expected value post-acquisition is $1.5 \times (B/2) = 0.75B$.
Expected profit is $P(V \le B) \times (0.75B - B) = -0.25B^2 / 100 \le 0$. The optimal bid is $B^* = 0$.

### Coins in a line

$2n$ coins of known values are placed in a line. Two players alternate taking one coin from either end.
Odd/even partition: odd-indexed sum $S_{odd}$ and even-indexed sum $S_{even}$.
The first player can take an odd or even coin to continually force the second player into exposing coins of the opposite parity. This guarantees the first player can collect all odd or all even coins, securing at least $\max(S_{odd}, S_{even}) \ge 50\%$.

### Chomp grid and hat parity

**Chomp**: $R \times C$ grid with a poisoned cell at $(1, 1)$. Picking a cell removes it and everything to its top and right. The player forced to eat $(1, 1)$ loses.
Strategy-stealing argument: Suppose the second player has a winning strategy. The first player takes only the top-right cell $(R, C)$, entering the second player's winning state, which implies a winning response $A$. However, the first player could have directly played $A$ on turn 1. Contradiction. The first player must have a winning strategy.

**100 prisoners and hats**: Prisoners face forward. Prisoner 100 counts the red hats ahead and calls out the parity bit $S_{99} \pmod 2$.
Prisoner 99 uses this global parity and the 98 visible hats to deduce their own color, surviving and updating the parity. The 99 prisoners ahead survive with $100\%$ certainty.

### Russian roulette

A 6-chamber revolver has 2 live bullets. The opponent pulls the trigger and survives.
- **Adjacent bullets**: The cylinder is at one of 4 empty chambers. Pulling directly hits a bullet $1/4 = 25\%$ of the time. Spinning yields $2/6 = 33.3\%$. Pull directly.
- **Non-adjacent bullets**: Of the 4 empty chambers, 2 are immediately followed by a bullet. Pulling directly yields $2/4 = 50\%$. Spinning yields $33.3\%$. Spin.

---

## 4 · Game model summary

| Model | Characteristics | Core conclusion |
|---|---|---|
| Pirate gold / Tigers and sheep | Dynamic complete info | Backward induction. Odd tigers eat, even do not. |
| The truel | State transitions | Subgame backward reasoning. Weakest player passes. |
| Guess 2/3 of average | Continuous coordination | Iterated elimination of dominated strategies. Equilibrium is 0. |
| Sealed-bid auctions | Incomplete info game | FPA bids $v(n-1)/n$, SPA bids $v$. Revenue equivalent. |
| Corporate acquisition | Adverse selection | Truncated conditional expectation $E[V \mid V \le B]$. Never bid. |
| Coins in a line / Chomp | Combinatorial game | Parity invariant. Strategy-stealing proves first-player win. |
| 100 prisoners' hats | Cooperative information | Last person sacrifices to broadcast global parity. |

---

## 5 · Interactive simulator

```game-theory-interactive-demo
```

---

## References

- Osborne, M. J., & Rubinstein, A. *A Course in Game Theory*. MIT Press.
- Zhou, J. *A Practical Guide to Quantitative Finance Interviews*.
- Crack, T. F. *Heard on the Street*.
