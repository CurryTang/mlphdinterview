# Quant 09 · Game Theory and Strategic Decision Making

Course track: [[Quant12 Brownian Motion Ito Calculus Stopping Times and Options|08 Brownian Motion]] → This Note → [[Quant14 Financial Markets Asset Classes and Portfolio Theory|10 Markets and Assets]]

Game theory studies optimal decision-making when outcomes depend on the interactive strategies of multiple agents. In quantitative trading, market making, and algorithmic order execution, game theory provides the structural foundation for analyzing counterparty behavior, adverse selection in order books, auction design, and information asymmetry.

This note is tailored specifically to the level expected in quantitative finance interviews, stripping away esoteric topology and dense abstract notations to focus on **core intuition, backward induction, the indifference principle, classic interview puzzle derivations, and market-maker microeconomics**.

---

## 1 · Core Concepts & Equilibrium Intuition

A game consists of players, available actions/strategy spaces, and payoff functions.

### 1.1 Pure Strategy Nash Equilibrium (PNE) & Best Response

The essence of a **Nash Equilibrium** is **mutual best responses**:
> A profile of strategies where no single player can unilaterally deviate and strictly increase their payoff, holding all other players' strategies fixed.

#### The Prisoner's Dilemma
Two suspects are interrogated separately. The payoff matrix (utilities representing sentence reductions, negative numbers denote prison years):

| Player A \ Player B | Defect (Confess) | Cooperate (Silent) |
|---|---|---|
| **Defect (Confess)** | $(-5, -5)$ | $(0, -10)$ |
| **Cooperate (Silent)** | $(-10, 0)$ | $(-1, -1)$ |

- **Strictly Dominant Strategy**: A strategy that strictly yields a higher payoff regardless of the counterparty's action.
  - If B defects, A gets $-5$ by defecting vs. $-10$ by staying silent;
  - If B cooperates, A gets $0$ by defecting vs. $-1$ by staying silent;
  - Defecting is strictly dominant for A, and symmetrically for B.
- **Pareto Inefficiency**: The unique Nash Equilibrium is `(Defect, Defect)` yielding $(-5, -5)$, even though `(Cooperate, Cooperate)` $(-1, -1)$ is strictly Pareto superior. Individual rationality produces collective suboptimality.

---

### 1.2 Mixed Strategy Nash Equilibrium (MSNE) & The Indifference Principle

When a game has no pure strategy saddle point (e.g., matching pennies, penalty kicks, poker bluffing), players must randomize across pure actions.

> **Fundamental Principle: The Indifference Principle**
> In a mixed strategy Nash equilibrium, a player randomizes their own actions to make the counterparty **completely indifferent between the pure strategies in their active support**.
> If the counterparty had an action yielding strictly higher expected utility, they would exploit it deterministically, breaking the equilibrium.

#### Example: The Penalty Kick
A striker shoots Left or Right; the goalkeeper dives Left or Right. Goal conversion probabilities (striker's payoff in a zero-sum game):

| Striker \ Goalkeeper | Dive Left (L) | Dive Right (R) |
|---|---|---|
| **Shoot Left (L)** | $0.60$ | $0.90$ |
| **Shoot Right (R)** | $0.95$ | $0.70$ |

Let the striker shoot Left with probability $p$ and Right with probability $1-p$.
The striker chooses $p$ such that the goalkeeper's expected conceded goals are identical whether diving Left or Right:
$$E[\text{Goal} \mid \text{Keeper L}] = 0.6p + 0.95(1-p)$$
$$E[\text{Goal} \mid \text{Keeper R}] = 0.9p + 0.7(1-p)$$
Setting them equal:
$$0.6p + 0.95 - 0.95p = 0.9p + 0.7 - 0.7p \implies 0.95 - 0.35p = 0.7 + 0.2p$$
$$0.55p = 0.25 \implies p^* = \frac{25}{55} = \frac{5}{11} \approx 45.5\%$$
The striker shoots Left with probability $5/11$ and Right with probability $6/11$.

---

### 1.3 Zero-Sum Games & The Minimax Theorem

In a two-player zero-sum game, Player 1's gain is Player 2's loss. Von Neumann's Minimax Theorem establishes:
$$\max_{\sigma_1} \min_{\sigma_2} E[u_1(\sigma_1, \sigma_2)] = \min_{\sigma_2} \max_{\sigma_1} E[u_1(\sigma_1, \sigma_2)] = V^*$$
$V^*$ is the **Value of the Game**.

- **Intuition: "Leveling Opponent Payoffs"**:
  If Player 1 adopts a distribution where Player 2 gets payoff 3 by choosing Action 1 and payoff 7 by choosing Action 2, a rational Player 2 will pick Action 1 to minimize Player 1's payoff.
  To maximize the worst-case floor, Player 1 must level the opponent's payoff profile until the peaks and valleys are completely flat.

---

### 1.4 Dynamic Games, Backward Induction & Subgame Perfection (SPE)

- **Extensive-form Game**: Games played sequentially, represented as a tree with decision nodes and information sets.
- **Non-credible Threat**: A strategy specifying an irrational, self-destructive action at an unreached node (e.g., "if you enter my market, I will price below cost forever"). If that subgame is reached, executing the threat is suboptimal.
- **Backward Induction**: Solving from the terminal subgame leaves backward to the root. The resulting equilibrium is a **Subgame Perfect Equilibrium (SPE)**, eliminating non-credible threats.

---

## 2 · Classic Quant Interview Dynamic Games

---

### 2.1 The Pirate Game (Pirate Loot Division)

**Problem**: 5 strictly rational pirates (ranked 1 to 5, where 1 is the most senior/fierce, 5 is the weakest) must divide 100 gold coins.
Rules:
1. Pirate 1 proposes a division. All living pirates vote (including the proposer).
2. If $\ge 50\%$ vote in favor, the proposal passes and the game ends. Otherwise, the proposer is thrown overboard to the sharks, and the next pirate proposes.
3. Preferences: **Survival first > Number of coins second > Bloodthirst third** (if coin payout is identical, vote to kill).

**Backward Induction Derivation**:
- **Base Case: Only Pirates 4 and 5 remain**
  - Pirate 4 proposes `4: 100, 5: 0`. Pirate 4 votes for himself ($1/2 = 50\%$), passing the proposal. Pirate 5 gets 0.
- **3 Pirates remain: 3, 4, 5**
  - Pirate 3 needs 2 votes ($2/3 > 50\%$). Needs to buy 1 vote.
  - Pirate 5 gets 0 in the next stage; Pirate 3 gives Pirate 5 **1 coin** ($1 > 0$), securing Pirate 5's vote.
  - Proposal: `3: 99, 4: 0, 5: 1`. Passed by 3 and 5.
- **4 Pirates remain: 2, 3, 4, 5**
  - Pirate 2 needs 2 votes ($2/4 = 50\%$). Needs to buy 1 vote.
  - If Pirate 2 dies, distribution is `3: 99, 4: 0, 5: 1`.
  - Pirate 4 gets 0 in that round. Pirate 2 buys Pirate 4 with **1 coin** (buying 5 would cost 2 coins).
  - Proposal: `2: 98, 3: 0, 4: 1, 5: 0`. Passed by 2 and 4.
- **Full Game: 1, 2, 3, 4, 5**
  - Pirate 1 needs 3 votes ($3/5 = 60\% \ge 50\%$). Needs to buy 2 votes.
  - If Pirate 1 dies, distribution is `2: 98, 3: 0, 4: 1, 5: 0`.
  - Opportunity cost: Pirates 3 and 5 get 0 coins in the next stage.
  - Pirate 1 offers **1 coin to Pirate 3** and **1 coin to Pirate 5**.
  - Proposal: `1: 98, 2: 0, 3: 1, 4: 0, 5: 1`. Passed by 1, 3, and 5.

$$\boxed{\text{Pirate 1 Optimal Proposal: }(98, 0, 1, 0, 1) \quad \text{Passed with 3 votes}}$$

#### Scaling to Large $N$
With $M=100$ coins:
- For $N \le 2M = 200$, the leader survives by bribing half the pirates with 1 coin;
- For $N > 200$, 100 coins cannot buy $\lceil N/2 \rceil$ votes. The proposer must give away all 100 coins and gets 0 coins to survive;
- When coins are exhausted, survival depends on voting coalitions of pirates who know they will die if earlier leaders fail. Proposers only survive when $N$ takes the form $2M + 2^k$ (e.g., 201, 202, 204, 208, 216...).

---

### 2.2 Tigers and Sheep (Island Parity Game)

**Problem**: 1 sheep and $N$ rational tigers on an island.
Rules:
1. Tigers prefer mutton over grass.
2. If a tiger eats the sheep, that tiger **turns into a sheep**.
3. Preferences: **Survival > Eating mutton**.
4. Does the first tiger eat the sheep?

**Parity Backward Induction**:
- $N = 1$: Tiger eats the sheep, becomes a sheep, no other tigers exist. Safe. **Eats**.
- $N = 2$: If Tiger A eats the sheep, it becomes a sheep with 1 tiger remaining. By $N=1$, Tiger B will eat it. Tiger A **does not eat**.
- $N = 3$: If Tiger A eats the sheep, 2 tigers remain. By $N=2$, neither remaining tiger dares to eat. Tiger A is safe. **Eats**.
- $N = 4$: By induction, eating reduces the game to $N=3$ where it will be eaten. **Does not eat**.

$$\boxed{\begin{cases} N \text{ is Odd} & \implies \text{The first tiger eats the sheep immediately} \\ N \text{ is Even} & \implies \text{No tiger eats; the sheep survives safely} \end{cases}}$$

---

### 2.3 The Truel (3-Player Duel with Unequal Accuracies)

**Problem**: Three players A, B, and C take turns shooting in cyclic order (A $\to$ B $\to$ C $\to$ A $\dots$). Their individual single-shot accuracies are:
$$p_A = 1/3, \quad p_B = 2/3, \quad p_C = 1 \text{ (Never misses / Perfect marksman)}$$
Rules:
1. The last surviving player wins the entire game;
2. On each turn, a player may shoot at any living opponent, or **deliberately shoot into the air (Pass / miss on purpose)**;
3. A hit eliminates the victim immediately, skipping them in all subsequent rounds.
**Question**: What is A's strictly optimal strategy on turn 1?

#### 1. Core Notation: 2-Player Endgame Win Rates $P_{AB}$ and $P_{AC}$
To analyze the dynamic 3-player game via **backward induction**, we first solve the terminal 1v1 duels:

- **Strict Definition of $P_{AB}$**:
  **The probability that A wins (survives) in a 1-on-1 duel between A and B, given that it is currently A's turn to shoot first.**
- **Markov Recursive Derivation of $P_{AB}$**:
  Conditioning on the first exchange of shots:
  - **Branch 1 (A hits B)**: Probability $p_A = 1/3$. B is eliminated immediately. A wins (payoff 1);
  - **Branch 2 (A misses B)**: Probability $1 - p_A = 2/3$. Turn passes to B:
    - If B hits A (probability $p_B = 2/3$): A is eliminated. A loses (payoff 0);
    - If B misses A (probability $1 - p_B = 1/3$): Both missed. The game **resets to the exact identical state** (A and B alive, A to shoot first). By the memoryless Markov property, A's forward win probability is again $P_{AB}$.
  
  Applying the law of total probability (equivalent to summing the infinite geometric series):
  $$
  P_{AB} = \underbrace{p_A \times 1}_{\text{A hits on turn 1}} + \underbrace{(1 - p_A)}_{\text{A misses}} \times \left[ \underbrace{p_B \times 0}_{\text{B hits A}} + \underbrace{(1 - p_B) \times P_{AB}}_{\text{B misses; state resets}} \right]
  $$
  Substituting the parameters:
  $$
  P_{AB} = \frac{1}{3} + \frac{2}{3} \times \left(1 - \frac{2}{3}\right) P_{AB} = \frac{1}{3} + \frac{2}{9} P_{AB}
  $$
  Solving for $P_{AB}$:
  $$
  \left(1 - \frac{2}{9}\right) P_{AB} = \frac{1}{3} \implies \frac{7}{9} P_{AB} = \frac{1}{3} \implies P_{AB} = \frac{3}{7} \approx 42.86\%
  $$
  *(Note: If the 1v1 duel starts with **B shooting first**, A survives only if B misses on turn 1, after which A gets the first shot. Thus A's win probability drops to $(1 - p_B) P_{AB} = \frac{1}{3} \times \frac{3}{7} = \frac{1}{7} \approx 14.29\%$.)*

- **Strict Definition and Value of $P_{AC}$**:
  The probability that A wins a 1-on-1 duel against C when **A shoots first**.
  Since $p_C = 1$, if A misses, C shoots back with 100% precision:
  $$
  P_{AC} = p_A \times 1 + (1 - p_A) \times (1 - p_C) \times P_{AC} = \frac{1}{3} + \frac{2}{3} \times 0 = \frac{1}{3} \approx 33.33\%
  $$

#### 2. Cross-Targeting Incentives with All 3 Alive
When all three players are alive:
- **B's Target**: B will never shoot A. If B kills A, next is C, who kills B with 100% certainty. If B kills C, B faces A with a dominant win rate of $1 - 3/7 = 4/7$. Hence, **B must target C**.
- **C's Target**: C will never shoot A. B poses a much higher threat ($p_B = 2/3$ vs $p_A = 1/3$) and shoots immediately after A. Hence, **C must target B**.

#### 3. Complete Comparison of A's Three Actions on Turn 1

| Strategy | Hits Target (Prob 1/3) | Misses Target (Prob 2/3) | A's Overall Win Probability |
|---|---|---|---|
| **Strategy 1: Shoot B** | **Kills B**. Leaves A vs C with **C shooting next**! C kills A with 100% certainty ($P = 0$). | **Misses**. All 3 alive; enters common miss branch ($P = \frac{25}{63}$). | $P = \frac{1}{3} \times 0 + \frac{2}{3} \times \frac{25}{63} = \frac{50}{189} \approx \mathbf{26.46\%}$ |
| **Strategy 2: Shoot C** | **Kills C**. Leaves A vs B with **B shooting next**! A's survival is $(1 - p_B)P_{AB} = \frac{1}{7}$. | **Misses**. All 3 alive; enters common miss branch ($P = \frac{25}{63}$). | $P = \frac{1}{3} \times \frac{1}{7} + \frac{2}{3} \times \frac{25}{63} = \frac{59}{189} \approx \mathbf{31.22\%}$ |
| **Strategy 3: Pass (Shoot into air)** | **Deterministic Miss (100%)**. Leaves all 3 alive; B must shoot C! | See breakdown below. | $P = \frac{2}{3} \times P_{AB} + \frac{1}{3} \times P_{AC} = \frac{75}{189} = \frac{\mathbf{25}}{\mathbf{63}} \approx \mathbf{39.68\%}$ |

> **Breakdown Following a Pass**:
> If A passes, turn goes to B, who fires at C:
> - **Case 2.1 (B hits C, prob $p_B = 2/3$)**: C is eliminated. Skipping dead C, **A shoots next**! A faces B with first-shot advantage: win rate is $P_{AB} = 3/7$;
> - **Case 2.2 (B misses C, prob $1 - p_B = 1/3$)**: Turn passes to C, who eliminates B with 100% probability. **A shoots next**! A faces C with first-shot advantage: win rate is $P_{AC} = 1/3$.
> 
> $$P_A(\text{Pass}) = \frac{2}{3} \times \frac{3}{7} + \frac{1}{3} \times \frac{1}{3} = \frac{2}{7} + \frac{1}{9} = \frac{25}{63} \approx 39.68\%$$

$$\boxed{\text{A's strictly optimal strategy on turn 1 is to intentionally Pass (shoot into the air), yielding } \frac{25}{63} \approx 39.7\%}$$

#### Strategic Takeaways
- **Elimination Backfire**: Successfully hitting any rival immediately gives the remaining surviving rival the first shot, dramatically increasing A's risk of death.
- **First-Mover Preservation**: Intentionally missing lets the two strongest competitors weaken or destroy each other while **guaranteeing that A holds the decisive first shot** in the final two-player showdown.

---

## 3 · Market Mechanisms, Adverse Selection & Auctions

---

### 3.1 Auction Theory: First-Price vs. Second-Price (Vickrey) Auctions

$n$ bidders compete for an item with private valuations independent and uniformly distributed: $v_i \sim U[0, 1]$.

#### Second-Price Sealed-Bid Auction (Vickrey Auction)
- **Rule**: The highest bidder wins, but pays the **second-highest bid ($P_2$)**.
- **Weakly Dominant Strategy**: **Truthful bidding $b(v) = v$ is weakly dominant**.
  - Bidding higher $b > v$: Changes the outcome only if $v < P_2 < b$. You win but pay $P_2 > v$, suffering a net loss $v - P_2 < 0$.
  - Bidding lower $b < v$: Changes the outcome only if $b < P_2 < v$. You forfeit a profitable purchase.
  - Bidding exact valuation dominates all deviations.

#### First-Price Sealed-Bid Auction
- **Rule**: Highest bidder wins and pays **their own bid**.
- **Bid Shading**: Bidding $b = v$ yields 0 profit. Bidders shade bids to balance winning probability against margin:
  $$b(v) = \frac{n-1}{n} v$$
  For $n=2$, bid $v/2$. As $n \to \infty$, competition forces $b(v) \to v$.

#### Revenue Equivalence Theorem
> Under standard benchmark conditions (neutral risk, independent private values), any auction mechanism that allocates the item to the highest-valuation bidder and yields zero surplus to the lowest-valuation bidder generates the exact same expected revenue for the seller:
> $$\mathbb{E}[\text{Revenue}] = \frac{n-1}{n+1}$$

---

### 3.2 Target Acquisition, Adverse Selection & The Winner's Curse

**Problem**: A target company's true value is $V \sim U[0, 100]$. An acquirer can manage the company better, increasing its value to $1.5 V$. The acquirer makes a take-it-or-leave-it tender offer $B$. The target accepts if $B \ge V$. What is the optimal bid $B^*$?

**Derivation**:
1. **Adverse Selection**: The target accepts only if $V \le B$.
2. **Conditional Expectation**: Conditional on the bid being accepted, the target's expected value is truncated:
   $$\mathbb{E}[V \mid V \le B] = \frac{B}{2}$$
3. **Expected Value After Acquisition**:
   $$1.5 \times \mathbb{E}[V \mid V \le B] = 1.5 \times \frac{B}{2} = 0.75 B$$
4. **Acquirer's Expected Profit**:
   $$\mathbb{E}[\Pi] = \mathbb{P}(V \le B) \times (0.75 B - B) = \frac{B}{100} \times (-0.25 B) = -\frac{0.25 B^2}{100} \le 0$$

$$\boxed{\text{Optimal Bid } B^* = 0\text{. Any positive bid leads to guaranteed expected loss.}}$$

---

### 3.3 Market Making Adverse Selection (Glosten-Milgrom Framework)

In electronic market making, the bid-ask spread compensates for **adverse selection risk** against informed counterparties.

- Asset true value $V \in \{V_L, V_H\}$ with prior expectation $V_0$.
- Order flow consists of two types of traders:
  - **Informed Traders (fraction $\alpha$)**: Possess inside knowledge. Buy if $V = V_H$, sell if $V = V_L$.
  - **Noise Traders (fraction $1 - \alpha$)**: Trade for liquidity (equal 50% probability of buying or selling).
- **Bayesian Update on an Incoming Buy Order**:
  $$\mathbb{P}(V = V_H \mid \text{Buy}) = \frac{\alpha \cdot 1 + (1-\alpha) \cdot 0.5}{1} = \frac{1+\alpha}{2} > 0.5$$
- **Zero-Profit Competitive Quotes**:
  $$\text{Ask} = \mathbb{E}[V \mid \text{Buy}] = V_0 + \frac{\alpha}{2}(V_H - V_L)$$
  $$\text{Bid} = \mathbb{E}[V \mid \text{Sell}] = V_0 - \frac{\alpha}{2}(V_H - V_L)$$
- **Optimal Spread**:
  $$\text{Spread} = \text{Ask} - \text{Bid} = \alpha (V_H - V_L)$$

**Quant Takeaway**:
The spread scales directly with $\alpha$. When informed trading volume surges, market makers widen spreads to prevent toxic order flow from depleting capital.

---

## 4 · Combinatorial Games & Quant Brainteasers

---

### 4.1 Guess 2/3 of the Average (The Keynesian Beauty Contest)

**Problem**: $N$ participants each pick a real number in $[0, 100]$. The winner is whoever is closest to $2/3$ of the group average.

**Iterated Elimination of Strictly Dominated Strategies (IESDS)**:
1. Max possible average is 100 $\implies$ max target is $66.67$. Numbers $> 66.67$ are strictly dominated.
2. In $[0, 66.67]$, max possible average is $66.67 \implies$ max target is $66.67 \times 2/3 = 44.44$.
3. At round $k$, the space contracts to $[0, 100 \times (2/3)^k]$.
4. As $k \to \infty$, the unique Nash equilibrium collapses to:
   $$s^* = 0$$

**Trading Reality (Level-$k$ Thinking)**:
Real market participants are not infinite-depth logicians. In practice:
- Level-0: Random pick, avg 50;
- Level-1: Expects Level-0, picks $50 \times 2/3 \approx 33$;
- Level-2: Expects Level-1, picks $33 \times 2/3 \approx 22$.
Being three steps ahead of the market is indistinguishable from being wrong; quant strategy design requires estimating counterparty sophistication depth.

---

### 4.2 Coins in a Line

**Problem**: An even number ($2n$) of coins with arbitrary values are placed in a row. Players alternate taking one coin from either the left or right end. Can the first player always guarantee at least half the total value?

**Parity Coloring Strategy**:
Label coin positions $1, 2, 3, \dots, 2n$:
- Odd positions sum: $S_{\text{odd}} = c_1 + c_3 + \dots + c_{2n-1}$
- Even positions sum: $S_{\text{even}} = c_2 + c_4 + \dots + c_{2n}$

**First Player's Control**:
- If $S_{\text{odd}} \ge S_{\text{even}}$, Player 1 takes $c_1$ (odd position).
- Both ends exposed to Player 2 are now $c_2$ and $c_{2n}$ (both even positions).
- Player 2 is forced to take an even coin. Player 1 can then take another odd coin.
- Player 1 guarantees at least $\max(S_{\text{odd}}, S_{\text{even}}) \ge 50\%$ of the total value.

---

### 4.3 Nim Game & The Sprague-Grundy Theorem

**Problem**: $k$ heaps of stones with sizes $x_1, x_2, \dots, x_k$. Players alternate taking any positive number of stones from a single heap. Last player to move wins.

**Bouton's Theorem**:
Compute the XOR sum (Nim-Sum):
$$S = x_1 \oplus x_2 \oplus \dots \oplus x_k$$
- **$S = 0$**: Losing position (P-position, second player wins);
- **$S \ne 0$**: Winning position (N-position, first player wins).

**Winning Move Execution**:
If $S \ne 0$, identify the most significant bit $d$ of $S$. Pick a heap $x_i$ whose $d$-th bit is 1. Reduce that heap to:
$$x_i' = x_i \oplus S < x_i$$
The new XOR sum becomes $S' = 0$. The first player systematically maintains $S=0$ for the opponent, guaranteeing victory.

---

### 4.4 Chomp & The Strategy-Stealing Argument

**Problem**: An $R \times C$ grid of chocolate. Bottom-left square $(1, 1)$ is poisoned. Players alternate choosing a square and eating it along with all squares above and to its right. Whoever eats $(1, 1)$ loses. Does Player 1 have a winning strategy?

**Strategy-Stealing Proof (Non-Constructive)**:
1. Suppose Player 2 has a winning strategy.
2. Player 1 takes only the top-right single square $(R, C)$.
3. This transitions the board to state $S_1$. By assumption, Player 2 has a winning response, move $A$.
4. However, move $A$ was an available legal first move from the initial board $S_0$! Player 1 could have played move $A$ on move 1.
5. Player 1 steals the winning strategy, a contradiction.
6. Hence Player 1 must have a winning strategy.

---

### 4.5 100 Prisoners Hat Puzzle

**Problem**: 100 prisoners lined up single file. Each wears a red or blue hat. Each sees all hats in front, but neither their own nor those behind. Starting from Prisoner 100 at the back, each must guess their own hat color out loud. How many prisoners can be guaranteed to survive?

**Parity Protocol**:
- Encode Red $= 1$, Blue $= 0$.
- **Prisoner 100**: Sums the red hats seen among the 99 prisoners ahead ($R_{99}$). Calls "Red" if $R_{99}$ is odd, "Blue" if even. Prisoner 100 survives with 50% probability.
- **Prisoner 99**: Counts red hats ahead ($R_{98}$). If $R_{98}$ parity matches Prisoner 100's call, Prisoner 99's hat must be Blue; otherwise Red. Prisoner 99 survives with 100% certainty.
- **Subsequent Prisoners (98 down to 1)**: Each deducts the known colors called behind them, surviving with 100% certainty.
- **Outcome: 99 prisoners guaranteed to survive; expected survival is 99.5%**.

---

### 4.6 Russian Roulette Conditional Decision

**Problem**: A 6-chamber revolver has 2 bullets. Player 1 points the gun at their own head and pulls the trigger: click, empty chamber. It is now your turn. You can: (1) Pull the trigger immediately; (2) Spin the cylinder before pulling. Which gives higher survival?

- **Scenario 1: 2 Bullets are Adjacent**
  - Player 1 survived an empty chamber, so we are at one of the 4 empty chambers.
  - Of the 4 empty chambers, only 1 is immediately followed by a bullet.
  - Shooting directly: Death probability is $1/4 = 25\%$.
  - Spinning cylinder: Death probability is $2/6 = 33.3\%$.
  - $\implies$ **Pull the trigger directly without spinning**.
- **Scenario 2: 2 Bullets are Non-Adjacent**
  - Of the 4 empty chambers, 2 are followed by a bullet.
  - Shooting directly: Death probability is $2/4 = 50\%$.
  - Spinning cylinder: Death probability is $2/6 = 33.3\%$.
  - $\implies$ **Spin the cylinder first**.

---

## 5 · High-Frequency Quant Interview Tricks & Core Templates

In quantitative finance interviews (Jane Street, Citadel, SIG, Optiver, etc.), game theory problems must typically be solved cleanly on a whiteboard in 5–10 minutes. Below are the 8 most essential problem-solving tricks, templates, and intuitive shortcuts:

---

### Trick 1: Backward Induction & The "0-to-1" Marginal Bribe Template
- **Applicable Problems**: Sequential, finite-stage dynamic games with voting, elimination, or loot division (Pirate Game, partner voting).
- **3-Step Execution Template**:
  1. **Lock in the Terminal Base Case ($k=1, 2$)**: Determine the outcome when the game collapses to the final 1–2 survivors.
  2. **Track Counterparty Opportunity Costs**: Specifically identify **who receives 0 coins in the subsequent stage**.
  3. **Buy Votes with Minimal Marginal Cost**: The proposer never seeks to please everyone; the goal is strictly to buy the minimum coalition required for $\ge 50\%$ approval. Always offer **1 coin** to those who would otherwise receive 0 (since $1 > 0$). Never waste budget trying to buy participants who already expect large payouts in the next round.

---

### Trick 2: Minimax Payoff Leveling Principle (Equalizing Opponent Payoffs)

Classic Quant Interview Problem Statement:
> **"From a game-theoretic perspective, if A can find a mixed strategy that makes B's expected score the same regardless of which number B guesses (or which action B takes), then B's expected score is minimized."**

---

#### 1. Why Does Leveling B's Expectation Guarantee Minimizing B's Score? (First-Principles Proof)

Let Player A adopt a mixed strategy probability distribution $\mathbf{p} = (p_1, \dots, p_n)$.
1. **The Counterparty's Rational Best Response**:
   A rational opponent B, upon observing or inferring A's probability vector $\mathbf{p}$, will not guess at random. B computes the expected payoff for every candidate action $g$, $E_B(g; \mathbf{p})$, and **deterministically chooses the action with the single highest expected payout**:
   $$\text{Score}_B(\mathbf{p}) = \max_{g} E_B(g; \mathbf{p})$$
2. **A's Minimax Objective**:
   In zero-sum/adversarial games, Player A's objective is to minimize the counterparty's maximum achievable return:
   $$\min_{\mathbf{p}} \max_g E_B(g; \mathbf{p})$$
3. **Peak-Shaving Dynamic Proof (Contradiction)**:
   - Suppose under A's strategy $\mathbf{p}$, the expected payouts across B's options are **unequal**. For instance, across 3 actions: $E_B(1) = 3$, $E_B(2) = 6$, and $E_B(3) = 2$;
   - B will ruthlessly exploit this by picking Action 2, reaping the peak score $\max(3, 6, 2) = 6$;
   - Observing this peak at 6, Player A can shift probability mass away from outcomes that reward Action 2 and allocate it toward suppressing options;
   - This adjustment lowers Action 2's return from 6 to 5, while slightly elevating Action 1 from 3 to 3.5. B's resulting maximum score drops from 6 to 5!
   - **Key Invariant**: **As long as peaks and valleys exist across B's options, B will exploit the peak, and A can always lower the maximum score by shaving the peak and filling the valley.**
   - **Terminal Steady State**: A cannot lower the peak any further **only when all candidate options for B are leveled to a uniform horizontal constant $V^*$**:
     $$E_B(1; \mathbf{p}^*) = E_B(2; \mathbf{p}^*) = \dots = E_B(n; \mathbf{p}^*) = V^*$$
     B is left with zero exploitability: $\max_g E_B(g; \mathbf{p}^*) = V^*$ reaches the global minimax optimum.

#### 2. Geometric Interpretation: Upper Envelope Minimum at the Intersection
For each choice $g$, $E_B(g; \mathbf{p})$ is an affine linear hyper-plane over $\mathbf{p}$.
The composite function $M(\mathbf{p}) = \max_g E_B(g; \mathbf{p})$ represents the **upper envelope** of these hyper-planes, forming a convex, V-shaped piecewise surface.
The global infimum (minimum) of a convex V-shaped envelope occurs precisely at the **intersection point** of the opposing planes. At this intersection, the expected heights across all active options are algebraically equal.

---

#### 3. Quant Interview Case Studies

##### Case A: Weighted Number Guessing Game ($1$ to $n$ with Payout $k$)
> **Problem**: Player A chooses $X \in \{1, 2, \dots, n\}$. Player B guesses once. If B guesses $k$ correctly, A pays B $k$ dollars; otherwise 0. What is A's optimal mixed strategy and what is B's expected score?

- **Applying the Leveling Template**:
  Let A pick number $k$ with probability $p_k$. If B guesses $k$, B's expected return is $k \cdot p_k$.
  To minimize B's return, A levels B's expected score to a constant $C$ across all guesses:
  $$1 \cdot p_1 = 2 \cdot p_2 = 3 \cdot p_3 = \dots = n \cdot p_n = C$$
  Hence $p_k = C/k$. Substituting into normalization $\sum_{k=1}^n p_k = 1$:
  $$C \sum_{k=1}^n \frac{1}{k} = 1 \implies C = \frac{1}{H_n}, \quad p_k^* = \frac{1/k}{H_n}$$
  where $H_n = 1 + \frac{1}{2} + \dots + \frac{1}{n}$ is the harmonic number. B's expected score is minimized to $\frac{1}{H_n}$.

##### Case B: The Green Book 1-to-4 Game with High/Low Clues
> **Problem**: A chooses $X \in \{1, 2, 3, 4\}$. B guesses: if correct on attempt 1, B gets 4 points. If incorrect, A reveals whether the guess was "too high" or "too low". B gets a second guess: if correct on attempt 2, B gets 2 points; otherwise 0. What is A's optimal strategy and B's expected score?

- **Applying the Leveling Template**:
  By symmetry, let A choose probabilities $(p_1, p_2, p_2, p_1)$ with $2p_1 + 2p_2 = 1$.
  - If B starts by guessing 2:
    - If $X=2$ (prob $p_2$), score is 4;
    - If $X < 2 \implies X=1$ (prob $p_1$), A says "too high", B guesses 1 on attempt 2, score is 2;
    - If $X > 2 \implies X \in \{3, 4\}$, A says "too low", B guesses 3 on attempt 2, score is 2 if $X=3$ (prob $p_2$);
    - Expected score: $E[\text{Guess } 2] = 2p_1 + 4p_2 + 2p_2 = 2p_1 + 6p_2$.
  - If B starts by guessing 1:
    - If $X=1$ (prob $p_1$), score is 4;
    - If $X > 1 \implies X \in \{2, 3, 4\}$, A says "too low", B optimally guesses 3 on attempt 2, getting 2 if $X=3$ (prob $p_2$);
    - Expected score: $E[\text{Guess } 1] = 4p_1 + 2p_2$.
  - **By the Payoff Leveling Principle, equate B's candidate scores**:
    $$E[\text{Guess } 1] = E[\text{Guess } 2] \implies 4p_1 + 2p_2 = 2p_1 + 6p_2 \implies 2p_1 = 4p_2 \implies p_1 = 2p_2$$
  - Combining with $2p_1 + 2p_2 = 1 \implies 4p_2 + 2p_2 = 1 \implies p_2^* = \frac{1}{6}, \; p_1^* = \frac{1}{3}$.
  - A's optimal mixed strategy is:
    $$\mathbf{p}^* = \left(\frac{1}{3}, \frac{1}{6}, \frac{1}{6}, \frac{1}{3}\right)$$
  - B's minimized expected score is:
    $$V^* = 4\left(\frac{1}{3}\right) + 2\left(\frac{1}{6}\right) = \frac{4}{3} + \frac{1}{3} = \frac{5}{3} \approx 1.67$$

---

### Trick 3: Adverse Selection & Truncated Expectation Template
- **Applicable Problems**: Informational asymmetry games where transactions only settle upon mutual agreement (target acquisitions, lemons markets, market-maker adverse selection).
- **Core Pitfall**: The unconditional expectation $\mathbb{E}[V]$ is fundamentally different from the **conditional expectation given trade $\mathbb{E}[V \mid \text{Trade}]$**!
- **3-Step Execution Template**:
  1. **Identify the Acceptance Condition**: The seller only agrees if their private valuation is below your bid ($V \le B$).
  2. **Compute Truncated Conditional Expectation**:
     $$\mathbb{E}[V \mid V \le B] = \frac{B}{2} \quad (\text{assuming } V \sim U[0, 100])$$
  3. **Evaluate Expected Profit**:
     $$\mathbb{E}[\Pi] = \mathbb{P}(V \le B) \times (\text{Synergy} \times \mathbb{E}[V \mid V \le B] - B)$$
  *Intuition*: Any asset the counterparty is willing to sell has its quality halved on average. If synergies cannot overcome this adverse selection penalty, the optimal choice is a **corner solution (bid $B^* = 0$, walk away)**.

---

### Trick 4: Parity Partitioning & Mirror Symmetry Template
- **Applicable Problems**: Coins in a line, circular table coin placement, symmetric grid games.
- **Two Core Patterns**:
  1. **Parity Partitioning**:
     Partition a 1D sequence into odd and even indexed sets. By taking an odd-indexed coin on move 1, Player 1 forces Player 2 to expose only even-indexed ends on every subsequent turn, locking in $\ge \max(S_{\text{odd}}, S_{\text{even}}) \ge 50\%$.
  2. **Mirror Strategy**:
     For centrally symmetric boards (e.g., circular tables), Player 1 takes the exact center on move 1. On all subsequent moves, Player 1 mirrors Player 2's placement directly across the center point, guaranteeing victory.

---

### Trick 5: The Strategy-Stealing Argument Template
- **Applicable Problems**: Full-information, finite, symmetric games with no draws where adding pieces never disadvantages the player (Chomp, Hex).
- **Standard Proof Structure**:
  1. Assume by contradiction that Player 2 has a winning strategy.
  2. Player 1 plays a benign, minimal move on move 1 (e.g., the isolated top-right corner square).
  3. The board transitions to state $S_1$, where Player 2 supposedly has a winning response, Move $A$.
  4. However, Move $A$ was an entirely legal first move from the original board $S_0$! Player 1 could have executed Move $A$ on turn 1.
  5. Player 1 steals the winning strategy, yielding a contradiction. Hence **Player 1 must have a winning strategy**.

---

### Trick 6: Nim-Sum XOR Invariant Template
- **Applicable Problems**: Multi-heap token subtraction, impartial DAG games.
- **Quick Rules**:
  1. Compute the bitwise XOR sum: $S = x_1 \oplus x_2 \oplus \dots \oplus x_k$.
  2. **$S = 0 \iff$ Losing State (P-position)**; **$S \ne 0 \iff$ Winning State (N-position)**.
  3. **Finding the Winning Move**: Locate the most significant bit $d$ of $S$. Choose any heap $x_i$ where the $d$-th bit is 1, and reduce it to $x_i' = x_i \oplus S$.

---

### Trick 7: Auction Bid Shading Template
- **Applicable Problems**: First-price and second-price sealed-bid auctions.
- **Formulas**:
  - **Second-Price (Vickrey)**: Truthful bidding $b(v) = v$ is weakly dominant.
  - **First-Price Auction ($n$ bidders, $v \sim U[0, 1]$)**:
    $$b(v) = \frac{n-1}{n} v$$
    *Intuition*: With 2 bidders, shade by half ($v/2$); with 3 bidders, bid $2/3 v$; with 100 bidders, shade to $99/100 v$. More competition forces bids closer to true value.

---

### Trick 8: The Truel / Weakest Player "Intentional Pass" Trick
- **Applicable Problems**: 3-player duels, truels, multi-firm market wars.
- **Core Rule**:
  - The two stronger players treat each other as the primary lethal threat.
  - If the weakest player attacks and eliminates one of the strong players, they immediately face the surviving powerhouse's lethal retaliation.
  - **Intentionally shooting into the air (Pass)** is optimal: it lets the two giants eliminate each other while guaranteeing that the weakest player holds the decisive first shot in the final two-player showdown!

---

## 6 · Quick Reference Cheatsheet

| Game Model | Category | Key Mechanism | Optimal Result / Equilibrium |
|---|---|---|---|
| **Prisoner's Dilemma** | Static Non-Zero Sum | Dominant Strategy | `(Defect, Defect)`, Pareto suboptimal |
| **Penalty Kick** | Static Zero Sum | Indifference Principle | Match expectations to remove counterparty edge |
| **Pirate Game** | Dynamic Finite Game | Backward Induction | 5-player solution: `(98, 0, 1, 0, 1)` |
| **Tigers and Sheep** | Dynamic Full Information | Parity Recurrence | Odd tigers eat; even tigers starve |
| **The Truel** | Dynamic Stochastic Duel | State Machine Reverse | Weakest player shoots into the air ($\approx 39.7\%$) |
| **Vickrey Auction** | Incomplete Info | Weakly Dominant | Truthful bidding $b = v$ |
| **First-Price Auction** | Incomplete Info | Bid Shading | $b(v) = \frac{n-1}{n} v$ |
| **Target Acquisition** | Adverse Selection | Truncated Mean | $\mathbb{E}[V \mid V \le B] = B/2 \implies B^* = 0$ |
| **Market Making** | Microstructure Spread | Adverse Selection Spread | $\text{Spread} = \alpha(V_H - V_L)$ |
| **Beauty Contest** | Coordination Game | IESDS Contraction | Fixed point is 0 |
| **Coins in a Line** | Combinatorial Game | Parity Coloring | First player guarantees $\ge \max(S_{\text{odd}}, S_{\text{even}})$ |
| **Nim Game** | Impartial Game | XOR Sum Invariant | $\bigoplus x_i = 0$ is losing; $\ne 0$ is winning |
| **Chomp** | Symmetric Finite Game | Strategy Stealing | Player 1 always has a winning strategy |
| **100 Prisoners Hats** | Collaborative Signaling | Parity Bit Broadcast | 99 prisoners survive with 100% certainty |

---

## 7 · Interactive Visualizer

```game-theory-interactive-demo
```

---

## Primary References

- Osborne, M. J., & Rubinstein, A. *A Course in Game Theory*. MIT Press.
- Zhou, J. *A Practical Guide to Quantitative Finance Interviews*.
- Crack, T. F. *Heard on the Street: Quantitative Questions from Wall Street Job Interviews*.
- Glosten, L. R., & Milgrom, P. R. (1985). *Bid, ask and transaction prices in a specialist market with heterogeneously informed traders*. Journal of Financial Economics.
