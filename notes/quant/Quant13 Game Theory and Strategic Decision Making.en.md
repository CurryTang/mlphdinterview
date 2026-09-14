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

#### 1. Classical Model Setup
- **Target Company**:
  True intrinsic asset value is a continuous random variable $V \sim \mathrm{Uniform}[0, 100]$. The target's founders/insiders **know the exact true value $V$**.
- **Acquirer (Buyer)**:
  **Cannot directly observe $V$**, only knowing its prior distribution $V \sim U[0, 100]$ with prior mean $\mathbb{E}[V] = 50$.
- **Synergy Multiplier**:
  Due to superior management and operational synergies, the company is worth $1.5 V$ under the acquirer's ownership (synergy factor $k = 1.5$).
- **Game Protocol**:
  The acquirer makes a single take-it-or-leave-it tender offer $B \ge 0$.
  The target is strictly rational: it accepts the offer if and only if the offer meets or exceeds its true intrinsic value ($B \ge V$); otherwise, it rejects ($B < V$).
- **Objective**: What bid $B^*$ should the acquirer choose to maximize expected net profit?

#### 2. The Naive Fallacy
A common pitfall in quantitative interviews:
> "The company's average value is $\mathbb{E}[V] = 50$. Under our management, it becomes worth $1.5 \times 50 = 75$. If we offer $B = 60$, it's higher than their average value ($60 > 50$), so they will likely sell, and lower than our post-acquisition value ($60 < 75$), locking in a net profit of $75 - 60 = 15$."

**Why is this completely wrong?**
Because it treats transaction acceptance as an unconditioned event, ignoring the **target's adverse self-selection**! Bidding $B = 60$ does not buy an average \$50 company.

#### 3. Adverse Selection Mechanism & Truncated Expectation
The target's management acts rationally on their private information:
- If $V > B$: The offer is inadequate; valuable targets reject the deal immediately.
- If $V \le B$: The offer is overly generous; low-value targets gladly cash out.

Therefore, **"the offer is accepted" is a conditioning event that filters out all high-quality assets**:
Once the deal closes, the asset's true value is conditionally truncated to the sub-interval $[0, B]$:
$$
\mathbb{E}[V \mid \text{Acquisition Succeeds}] = \mathbb{E}[V \mid V \le B] = \frac{0 + B}{2} = \frac{B}{2}
$$
- **Expected Post-Acquisition Asset Value**:
  $$1.5 \times \mathbb{E}[V \mid V \le B] = 1.5 \times \frac{B}{2} = 0.75 B$$
- **Conditional Net Profit per Successful Deal**:
  $$\mathbb{E}[\text{Profit} \mid V \le B] = 0.75 B - B = -0.25 B$$
  **The acquirer loses an expected 25% of the bid price on every single transaction it wins!**

#### 4. Global Expected Profit and Optimal Bid
The probability of transaction acceptance is $\mathbb{P}(V \le B) = \frac{B}{100}$ (for $0 \le B \le 100$). The acquirer's unconditional expected net profit is:
$$
\mathbb{E}[\Pi(B)] = \mathbb{P}(V \le B) \cdot \mathbb{E}[\text{Profit} \mid V \le B] = \left(\frac{B}{100}\right) \times (-0.25 B) = -\frac{B^2}{400} \le 0
$$
- For any $B > 0$, the expected profit is strictly negative.
- The global maximum occurs uniquely at $B = 0$ where expected profit is 0.

$$\boxed{\text{Optimal Bid } B^* = 0\text{. Any positive bid leads to guaranteed expected loss.}}$$

#### 5. Extension: How Strong Must Synergies Be to Overcome the Winner's Curse?
Let the synergy factor be $k > 1$ (asset worth $k V$ to the acquirer):
- Conditional net profit is $k \frac{B}{2} - B = \left(\frac{k}{2} - 1\right) B$.
- For expected profit to be positive:
  $$\frac{k}{2} - 1 > 0 \implies k > 2$$
**Quant Takeaway**:
In the presence of one-sided asymmetric information, the acquirer must **more than double the target's value ($k > 200\%$)** to overcome the lemon discounting caused by adverse selection. With only a 1.5x synergy, the market collapses into Akerlof's lemon trap.

---

### 3.3 Market Making Adverse Selection (Glosten-Milgrom Framework)

In high-frequency market making and limit order book (LOB) dynamics, the **bid-ask spread primarily compensates for adverse selection against informed counterparties**, rather than broker exchange fees or inventory holding costs.

#### 1. Microstructure Model & Player Roles
- **Terminal Asset Value $V$**:
  The asset will ultimately realize either a high value $V_H$ or a low value $V_L$ (defining $\Delta V = V_H - V_L > 0$), each with equal prior probability:
  $$\mathbb{P}(V = V_H) = \mathbb{P}(V = V_L) = \frac{1}{2}, \quad V_0 = \mathbb{E}[V] = \frac{V_H + V_L}{2}$$
- **Competitive Market Maker (MM)**:
  Posts two-sided quotes: an Ask (selling price) and a Bid (buying price). The MM **does not know** the true realization of $V$.
- **Order Flow Composition**:
  1. **Informed Traders (fraction $\alpha \in [0, 1]$)**:
     Possess insider or alpha signals and know true $V$.
     - If $V = V_H$: Informed traders always submit market **Buy** orders to lift the MM's Ask.
     - If $V = V_L$: Informed traders always submit market **Sell** orders to hit the MM's Bid.
     - *The market maker always loses money against informed traders.*
  2. **Noise / Uninformed Traders (fraction $1 - \alpha$)**:
     Trade purely for liquidity or external hedging. Their order directions are independent of fundamental value, buying or selling with equal 50% probability:
     $$\mathbb{P}(\text{Buy} \mid \text{Noise}) = \frac{1}{2}, \quad \mathbb{P}(\text{Sell} \mid \text{Noise}) = \frac{1}{2}$$
     - *The market maker earns the spread from noise traders to cross-subsidize losses against informed traders.*

#### 2. Bayesian Updating from the Order Flow
When the MM receives an incoming market buy order ($\text{Order} = \text{Buy}$), the order arrival itself is an informative signal.

##### (1) Likelihood of an Incoming Buy Order:
- Under the high state ($V = V_H$):
  $$\mathbb{P}(\text{Buy} \mid V_H) = \underbrace{\alpha \times 1}_{\text{Informed always buys}} + \underbrace{(1 - \alpha) \times \frac{1}{2}}_{\text{Noise buys 50\%}} = \frac{1 + \alpha}{2}$$
- Under the low state ($V = V_L$):
  $$\mathbb{P}(\text{Buy} \mid V_L) = \underbrace{\alpha \times 0}_{\text{Informed never buys}} + \underbrace{(1 - \alpha) \times \frac{1}{2}}_{\text{Noise buys 50\%}} = \frac{1 - \alpha}{2}$$

##### (2) Total Probability of an Incoming Buy:
$$
\mathbb{P}(\text{Buy}) = \mathbb{P}(\text{Buy} \mid V_H)\mathbb{P}(V_H) + \mathbb{P}(\text{Buy} \mid V_L)\mathbb{P}(V_L) = \frac{1+\alpha}{2} \cdot \frac{1}{2} + \frac{1-\alpha}{2} \cdot \frac{1}{2} = \frac{1}{2}
$$

##### (3) Bayesian Posterior:
By Bayes' Theorem, after observing a Buy order:
$$
\mathbb{P}(V = V_H \mid \text{Buy}) = \frac{\mathbb{P}(\text{Buy} \mid V_H)\mathbb{P}(V_H)}{\mathbb{P}(\text{Buy})} = \frac{\frac{1+\alpha}{2} \cdot \frac{1}{2}}{\frac{1}{2}} = \frac{1 + \alpha}{2}
$$
Similarly, $\mathbb{P}(V = V_L \mid \text{Buy}) = 1 - \frac{1+\alpha}{2} = \frac{1 - \alpha}{2}$.

> **Core Intuition**:
> The prior probability was $0.5$. Receiving a market buy jumps the belief of high value to $\frac{1+\alpha}{2} > 0.5$. Order flow conveys toxic, directionally informed signals!

#### 3. Competitive Pricing (Bertrand Zero-Profit Condition)
In a competitive market making environment, Bertrand competition drives expected economic profit to zero. Quotes must match the conditional expected asset value:

##### (1) Ask Price Formulation:
$$
\begin{aligned}
\text{Ask} &= \mathbb{E}[V \mid \text{Buy}] \\
&= V_H \cdot \mathbb{P}(V = V_H \mid \text{Buy}) + V_L \cdot \mathbb{P}(V = V_L \mid \text{Buy}) \\
&= V_H \left(\frac{1+\alpha}{2}\right) + V_L \left(\frac{1-\alpha}{2}\right) \\
&= \frac{V_H + V_L}{2} + \frac{\alpha}{2}(V_H - V_L) \\
&= V_0 + \frac{\alpha}{2}\Delta V
\end{aligned}
$$

##### (2) Bid Price Formulation:
Symmetrically, observing an incoming market Sell order shifts the posterior towards $V_L$:
$$
\mathbb{P}(V = V_L \mid \text{Sell}) = \frac{1+\alpha}{2}
$$
$$
\text{Bid} = \mathbb{E}[V \mid \text{Sell}] = V_0 - \frac{\alpha}{2}\Delta V
$$

#### 4. The Equilibrium Bid-Ask Spread
$$
\boxed{\text{Spread} = \text{Ask} - \text{Bid} = \alpha (V_H - V_L) = \alpha \cdot \Delta V}
$$

#### 5. Quant Microstructure Insights
1. **Spread Scales with Informed Flow $\alpha$ (Toxic Order Flow)**:
   - If the market is purely noise ($\alpha = 0$): The MM faces zero adverse selection, and competition forces the spread to 0.
   - If the market is dominated by informed traders ($\alpha \to 1$): Spreads widen to the full fundamental gap $V_H - V_L$. Metrics like **VPIN (Volume-Synchronized Probability of Toxicity)** measure this parameter in real-time.
2. **Spread Scales with Fundamental Volatility $\Delta V$**:
   Ahead of high-impact macroeconomic announcements (e.g., CPI, FOMC) or corporate earnings, fundamental uncertainty $\Delta V$ surges, forcing automated market makers to widen spreads or cancel passive quotes.
3. **Price Impact (Mid-Price Drift)**:
   Every executed market order updates the MM's Bayesian expectation of fair value, explaining why successive buy orders create persistent upward price impact.

---

## 4 · Combinatorial Games & Quant Brainteasers

---

### 4.1 Guess 2/3 of the Average (The Keynesian Beauty Contest)

#### 1. Problem Formulation
$N$ rational players each choose a real number $x_i \in [0, 100]$. Let $\mu = \frac{1}{N}\sum_{i=1}^N x_i$ denote the arithmetic mean of all chosen numbers. The prize is awarded to the player whose guess is closest to the target $T = \frac{2}{3}\mu$ (shared equally in case of a tie).
**Question**: What is the unique theoretical Nash equilibrium? How should a quantitative trader play this game in real markets?

#### 2. Rigorous Proof via Iterated Elimination of Strictly Dominated Strategies (IESDS)
Let $S_0 = [0, 100]$ be the initial strategy set.
- **Iteration 1**:
  Even if every single player irrationally chooses the maximum possible value $100$, the sample average cannot exceed $\mu \le 100$. Hence, the target value is bounded from above by:
  $$T \le \frac{2}{3} \times 100 \approx 66.67$$
  If a player chooses any number $x_i > 66.67$, then choosing $66.67$ is strictly closer to the true target $T$ than $x_i$, regardless of what other players pick. Thus, all strategies in $(66.67, 100]$ are **strictly dominated** and eliminated under first-order rationality. The viable strategy set contracts to $S_1 = [0, 66.67]$.
- **Iteration 2**:
  Since it is second-order common knowledge that no rational player chooses $> 66.67$, the maximum possible average in $S_1$ drops to $66.67$. The target ceiling shrinks to:
  $$T \le \frac{2}{3} \times 66.67 \approx 44.44$$
  Strategies in $(44.44, 66.67]$ are strictly dominated and eliminated. The viable set becomes $S_2 = [0, 44.44]$.
- **Iteration $k$**:
  After $k$ levels of iterative reasoning, the viable strategy space contracts according to:
  $$S_k = \left[0, 100 \times \left(\frac{2}{3}\right)^k\right]$$
- **Infinite-Horizon Limit (Fixed Point)**:
  As $k \to \infty$, the contraction mapping collapses to its unique fixed point:
  $$x^* = \frac{2}{3} x^* \implies \frac{1}{3} x^* = 0 \implies x^* = 0$$

$$\boxed{\text{The unique symmetric Nash equilibrium in continuous space is for all players to choose } x^* = 0}$$

> **Discrete Integer Trap in Quant Interviews**:
> If the rules mandate picking **positive integers** from $\{1, 2, \dots, 100\}$:
> - When the viable set contracts to $\{1, 2, 3\}$, if everyone chooses 1, the average is 1, and the target is $2/3 \approx 0.67$.
> - The closest positive integer to $0.67$ is $1$.
> - Thus, on positive integers $\{1, \dots, 100\}$, **the unique weak Nash equilibrium is for everyone to choose 1**.

#### 3. Behavioral Game Theory & Level-$k$ Reasoning (Why Theory Fails in Practice)
John Maynard Keynes famously described stock market speculation through a newspaper beauty contest: professional investors do not pick who they believe is genuinely prettiest, but rather predict what average opinion expects average opinion to be. In experimental economics (e.g., Nagel 1995, Financial Times reader competitions), the actual winning number is consistently around **21.6**!

The Cognitive Hierarchy distribution:
- **Level-0 (Zero-order noise)**: Guesses randomly without strategic deliberation, yielding an empirical mean around $50$;
- **Level-1 (First-order reasoning)**: Assumes all opponents are Level-0, best-responding to 50: $50 \times \frac{2}{3} \approx 33.3$;
- **Level-2 (Second-order reasoning)**: Assumes opponents are Level-1, choosing $33.3 \times \frac{2}{3} \approx \mathbf{22.2}$ (the modal cluster of smart traders in the real world);
- **Level-3 (Third-order reasoning)**: Assumes opponents are Level-2, choosing $22.2 \times \frac{2}{3} \approx 14.8$;
- **Level-$\infty$ (Pure game theorists)**: Chooses 0, and almost always finishes dead last because the market has not iterated to infinity.

**Hedge Fund Trading Implication**:
Generating Alpha is not about proving where the asymptotic equilibrium lies, but rather diagnosing the prevailing cognitive level of market counterparties. **"Being half a step ahead of the herd captures Alpha; being three steps ahead gets you crushed by short-term liquidity."**

---

### 4.2 Coins in a Line

#### 1. Problem Formulation
An even number of coins ($2n$ coins) with arbitrary non-negative values $c_1, c_2, \dots, c_{2n} \ge 0$ are arranged in a straight line. Two players alternate turns. On each turn, a player may take either the **leftmost or rightmost** remaining coin. Both players seek to maximize their total collected coin value.
**Question**: Does the first player have a guaranteed winning strategy? What is the lower bound on their payoff? What happens if $N$ is odd?

#### 2. Parity Coloring Invariant (Qualitative Lower Bound Proof)
Partition the $2n$ positions by index parity:
- **Sum of odd-indexed coins**: $S_{\text{odd}} = \sum_{k=1}^n c_{2k-1} = c_1 + c_3 + \dots + c_{2n-1}$
- **Sum of even-indexed coins**: $S_{\text{even}} = \sum_{k=1}^n c_{2k} = c_2 + c_4 + \dots + c_{2n}$
- **Total value**: $S_{\text{total}} = S_{\text{odd}} + S_{\text{even}}$

**The First Player's Parity Monopoly**:
Without loss of generality, assume $S_{\text{odd}} \ge S_{\text{even}}$ (if opposite, swap roles):
1. **Turn 1 (Player 1 takes an odd coin)**:
   Player 1 picks the leftmost coin $c_1$ (odd index). The remaining line has $2n-1$ coins, with the leftmost at $c_2$ (even) and the rightmost at $c_{2n}$ (even). **Both endpoints are even-indexed!**
2. **Turn 2 (Player 2 is forced to take an even coin)**:
   Player 2 is forced to choose between $c_2$ and $c_{2n}$. Regardless of choice, Player 2 **must take an even-indexed coin**.
3. **Turn 3 (Player 1 regains parity control)**:
   - If Player 2 took left ($c_2$): endpoints become $c_3$ (odd) and $c_{2n}$ (even);
   - If Player 2 took right ($c_{2n}$): endpoints become $c_2$ (even) and $c_{2n-1}$ (odd).
   - In either case, **one odd-indexed and one even-indexed coin are exposed**. Player 1 immediately takes the odd-indexed coin.
4. **Inductive Invariant**:
   Player 1 can unilaterally force the game so that they collect **every single odd-indexed coin**, restricting Player 2 exclusively to even-indexed coins.

$$\boxed{\text{Player 1 has a guaranteed win strategy, securing at least } \max(S_{\text{odd}}, S_{\text{even}}) \ge \frac{1}{2} S_{\text{total}}}$$

#### 3. Algorithmic Game Theory: Minimax Dynamic Programming
Parity coloring guarantees $\ge 50\%$, but does not necessarily find the global maximum score. In algorithmic quant interviews (LeetCode 486 / Predict the Winner):

- **State Definition**:
  Let $dp[i][j]$ be the maximum **net relative score difference** (current player's score minus opponent's score) achievable from the sub-array $[i, j]$.
- **Recurrence Relation**:
  The current player chooses between taking $c_i$ or $c_j$, knowing the opponent will play optimally on the remainder:
  $$dp[i][j] = \max\Big( c_i - dp[i+1][j], \quad c_j - dp[i][j-1] \Big)$$
- **Base Cases**:
  For single coins ($j = i$): $dp[i][i] = c_i$.
- **Final Exact Payoff**:
  The net advantage of Player 1 over Player 2 is $dp[1][2n]$. Player 1's exact total collected value is:
  $$\text{Player 1 Total} = \frac{S_{\text{total}} + dp[1][2n]}{2}$$

#### 4. Critical Interview Follow-up: What if the Number of Coins is Odd ($2n+1$)?
**Conclusion: Player 1 is NOT guaranteed to win, and can suffer a devastating loss!**
- **Decisive Counterexample**: Consider 3 coins $[1, 100, 1]$ (odd length $N = 3$).
  - Player 1 takes either the left 1 or the right 1, leaving $[1, 100]$ or $[100, 1]$.
  - Player 2 immediately takes the center prize $100$!
  - Player 1 ends with $1 + 1 = 2$, while Player 2 wins $100$.
- **Why Parity Coloring Breaks**:
  With an odd number of coins, the first index (1, odd) and the last index ($2n+1$, odd) are **both odd**! After Player 1 takes one odd coin, the remaining endpoints are one odd and one even, allowing Player 2 to also claim odd-indexed coins. Player 1 loses the ability to monopolize a parity class.

---

### 4.3 Nim Game & Sprague-Grundy Theorem

#### 1. Problem Formulation
There are $k$ piles of stones with sizes $x_1, x_2, \dots, x_k \ge 0$. Two players alternate turns. On their turn, a player chooses any non-empty pile and removes at least 1 stone (up to the entire pile). Under standard Normal Play convention, **the player who takes the last stone wins**.
**Question**: How do we evaluate whether a position is a winning or losing state? What is the explicit winning move algorithm?

#### 2. Bouton's Theorem & The Nim-Sum
Define the binary bitwise XOR sum (the Nim-sum) of the game state:
$$S = x_1 \oplus x_2 \oplus \dots \oplus x_k$$
- **If $S = 0$**: The position is a **P-position (Previous-player-winning / Losing state)**. The player whose turn it is faces certain defeat under optimal counterplay.
- **If $S \ne 0$**: The position is an **N-position (Next-player-winning / Winning state)**. The player whose turn it is has an explicit winning move.

#### 3. Rigorous Proof of Bouton's Three Axiomatic Properties
In impartial combinatorial game theory, the characterization of P and N positions rests upon three necessary and sufficient conditions:

##### (1) Terminal State is a P-position:
When all stones have been removed, the state is $(0, 0, \dots, 0)$ with Nim-sum $S = 0$. By definition, the player facing this empty board has no legal moves and loses. Hence, the terminal $S = 0$ state is a P-position.

##### (2) From any $S \ne 0$ state, there exists at least one legal move leading to $S' = 0$:
- Let $S \ne 0$, and let the **most significant set bit (highest power of 2)** in $S$ be bit $d$ ($2^d \le S < 2^{d+1}$).
- By the definition of XOR, among the piles $x_1, \dots, x_k$, **there must exist at least one pile $x_i$ whose $d$-th bit is also 1** (if all were 0, the $d$-th bit of $S$ could not be 1).
- We reduce this pile $x_i$ to:
  $$x_i' = x_i \oplus S$$
- **Proof that $x_i' < x_i$ (ensuring legality)**:
  At the most significant differing bit $d$, both $x_i$ and $S$ have a 1. In $x_i \oplus S$, bit $d$ becomes 0. For all bits higher than $d$, $S$ has 0s, leaving $x_i$'s higher bits unchanged. Therefore, $x_i' = x_i \oplus S < x_i$ strictly holds!
- **Proof that the new Nim-sum $S' = 0$**:
  $$S' = S \oplus x_i \oplus x_i' = S \oplus x_i \oplus (x_i \oplus S) = S \oplus S \oplus x_i \oplus x_i = 0$$

##### (3) From any $S = 0$ state, EVERY legal move results in $S' \ne 0$:
- Suppose $S = 0$. A player alters pile $x_i$ to $x_i' < x_i$.
- The new Nim-sum is:
  $$S' = S \oplus x_i \oplus x_i' = 0 \oplus x_i \oplus x_i' = x_i \oplus x_i'$$
- Since a legal move requires $x_i' \ne x_i$, their binary representations differ in at least one bit, so $x_i \oplus x_i' \ne 0 \implies S' \ne 0$.

**Inductive Conclusion**: An N-player always transitions $S \ne 0 \to S = 0$. The opponent is trapped into transitioning $S = 0 \to S \ne 0$. The N-player steadily drives the game to the terminal $(0, \dots, 0)$, winning the game.

#### 4. Step-by-Step Calculation: 3 Piles of Sizes (3, 4, 5)
Let $x_1 = 3, x_2 = 4, x_3 = 5$:
1. **Binary representations and XOR sum**:
   $$
   \begin{aligned}
   x_1 &= 3 = (011)_2 \\
   x_2 &= 4 = (100)_2 \\
   x_3 &= 5 = (101)_2 \\
   \hline
   S &= 3 \oplus 4 \oplus 5 = (010)_2 = 2 \ne 0 \quad (\text{Winning state / N-position!})
   \end{aligned}
   $$
2. **Find the target pile**:
   The highest bit of $S = 2 = (010)_2$ is bit 1 (value 2).
   Checking which pile has bit 1 set: only $x_1 = 3 = (011)_2$ has a 1 in bit 1.
3. **Compute replacement value**:
   $$x_1' = x_1 \oplus S = 3 \oplus 2 = 1 < 3$$
4. **Optimal Move**:
   Remove 2 stones from pile 1, reducing it from 3 to 1!
   The new state is $(1, 4, 5)$. Check new Nim-sum: $1 \oplus 4 \oplus 5 = 001_2 \oplus 100_2 \oplus 101_2 = 0$. Opponent is doomed.

#### 5. Misère Nim (The Last-Stone-Loser Variation)
If rules invert: **the player who takes the last stone LOSES**.
- **Golden Rule of Misère Play**:
  1. While **at least two piles have size $> 1$**: Play exactly according to standard Nim rules (keep $S = 0$ after your move);
  2. The turning point arrives when your move can leave **exactly one pile of size $> 1$ and all other piles of size 1 (state $x, 1, 1, \dots$)**:
     - Do NOT make the XOR sum 0! Instead, reduce the large pile to 0 or 1 such that **an ODD number of piles of size 1 remain**!
     - The opponent is forced to take from singleton piles one by one, ultimately taking the final stone and losing.

---

### 4.4 Chomp & The Strategy-Stealing Argument

#### 1. Problem Formulation
An $R \times C$ rectangular chocolate grid has a poisoned bottom-left square at $(1, 1)$. Two players alternate picking a remaining square $(r, c)$. Choosing $(r, c)$ eats that square along with all squares located to its **upper-right** (all $(r', c')$ such that $r' \ge r$ and $c' \ge c$). The player forced to eat the poisoned square $(1, 1)$ loses.
**Question**: Does Player 1 have a guaranteed winning strategy?

#### 2. Strategy-Stealing Argument (Non-Constructive Proof)
Chomp is a finite, deterministic, perfect-information game with no passing and no draws. By Zermelo's Theorem, one player must have a winning strategy.

**Proof by Contradiction**:
1. Assume that **Player 2 (the second player) has a winning strategy**.
2. Player 1 opens the game by eating only the single top-right corner square $(R, C)$.
   - The board transitions from initial state $S_0$ to new state $S_1$.
3. By our assumption that Player 2 has a winning strategy, Player 2 has a designated winning response to $S_1$. Let this move be eating square $(r^*, c^*)$ and its upper-right shadow.
4. **The Logical Contradiction**:
   - Notice that $(R, C)$ satisfies $R \ge r^*$ and $C \ge c^*$. Therefore, square $(R, C)$ is **already contained within the upper-right region dominated by $(r^*, c^*)$**!
   - Consequently, the board state resulting from "Player 1 takes $(R, C)$, then Player 2 takes $(r^*, c^*)$" is **physically identical** to the board state created if someone had simply taken $(r^*, c^*)$ from the initial full board $S_0$ in one single stroke!
   - But $(r^*, c^*)$ was an available legal move for Player 1 on Turn 1!
   - Thus, Player 1 could have bypassed $(R, C)$ entirely and played $(r^*, c^*)$ on the very first turn, thereby "stealing" Player 2's winning position!
5. This contradicts the premise that Player 2 has a winning strategy.

$$\boxed{\text{Player 2 cannot have a winning strategy. Hence, Player 1 must have a winning strategy.}}$$

#### 3. Explicit Winning Constructions for Specific Grids
While the strategy-stealing proof is non-constructive (finding winning moves for general Chomp is EXPTIME-complete), interviewers often ask for explicit constructions on symmetric boards:

- **Square Board $N \times N$**:
  - **Winning Move**: Player 1 eats square $(2, 2)$ and its upper-right quadrant on Turn 1!
  - The remaining board forms a symmetric "L-shape" with two arms of length $N$ (row 1 and column 1).
  - **Mirroring Strategy**: If Player 2 bites row 1 at $(1, k)$, Player 1 mirrors by biting column 1 at $(k, 1)$. Player 1 preserves equal arm lengths until Player 2 is forced to eat the poison $(1, 1)$.
- **$2 \times N$ Rectangular Board**:
  - Player 1 bites $(2, N)$ on Turn 1, making row 2 one square shorter than row 1.
  - Invariant: Player 1 always keeps row 2 exactly one unit shorter than row 1, forcing Player 2 into the poisoned corner.

---

### 4.5 The 100 Prisoners Hat Puzzle

#### 1. Problem Formulation
100 rational prisoners are queued in a line ($P_1$ to $P_{100}$). Each wears a Red hat (1) or Blue hat (0).
- **Vision**: Each prisoner can see the hat colors of all prisoners standing in front of them, but cannot see their own hat or anyone standing behind them ($P_{100}$ sees 99 hats; $P_1$ sees 0 hats);
- **Protocol**: Starting from $P_{100}$ at the back and proceeding sequentially forward to $P_1$, each prisoner speaks either "Red" or "Blue";
- **Survival**: If a prisoner speaks their own hat color, they survive; otherwise they are executed. Everyone hears all prior public declarations.
**Question**: What pre-arranged coordination protocol maximizes the expected number of survivors?

#### 2. The Parity Check Protocol
Let hat colors be $c_i \in \{0, 1\}$ (Red $= 1$, Blue $= 0$).

```
  [P1 (Front)]    [P2]        ...        [P98]        [P99]        [P100 (Back)]
   Sees: 0 hats   Sees: 1 hat            Sees: 97     Sees: 98     Sees: All 99 hats
        ↑                                                ↑              ↑
   Speaks last                                      Speaks 2nd     Speaks first (Broadcasts Parity)
```

##### (1) Prisoner 100 (The Martyr & Parity Beacon):
- $P_{100}$ has no information about their own hat, but counts the total number of red hats among the 99 prisoners in front:
  $$S_{99} = \sum_{k=1}^{99} c_k$$
- $P_{100}$ computes the modulo-2 parity check:
  $$b_{100} = S_{99} \bmod 2 = \left(\sum_{k=1}^{99} c_k\right) \bmod 2$$
- Protocol: $P_{100}$ announces "Red" if $b_{100} = 1$, and "Blue" if $b_{100} = 0$.
- $P_{100}$ has a $50\%$ chance of personal survival, but broadcasts the global parity of the front 99 hats to the entire group!

##### (2) Prisoner 99 (First Guaranteed Survivor):
- $P_{99}$ hears parity $b_{100}$.
- $P_{99}$ directly sees the 98 hats ahead, summing their red hats: $S_{98} = \sum_{k=1}^{98} c_k$.
- Since $S_{99} = c_{99} + S_{98}$, taking modulo 2:
  $$c_{99} = (b_{100} - S_{98}) \bmod 2$$
- Both $b_{100}$ and $S_{98}$ are known with 100% certainty! $P_{99}$ announces $c_{99}$ and survives with $100\%$ certainty.

##### (3) Prisoner $i$ ($i$ descending from 98 to 1):
- At turn $i$, prisoner $P_i$ knows:
  1. The global base parity $b_{100}$;
  2. The deduced colors $c_{99}, \dots, c_{i+1}$ from behind;
  3. The visible colors $c_{i-1}, \dots, c_1$ seen ahead.
- $P_i$ computes their own hat color deterministically:
  $$c_i = \left( b_{100} - \sum_{j=i+1}^{99} c_j - \sum_{k=1}^{i-1} c_k \right) \bmod 2$$
- All terms are known. Every prisoner from 99 down to 1 answers correctly!

$$\boxed{\text{Prisoners 1 through 99 survive with } 100\% \text{ certainty. Expected survivors: } 99 + 0.5 = 99.5 \text{ prisoners (99.5\%)}}$$

#### 3. Generalization to $K$ Hat Colors
If hats have $K \ge 2$ distinct colors, encode colors as elements of $\mathbb{Z}_K = \{0, 1, \dots, K-1\}$. $P_{100}$ announces $\left(\sum_{k=1}^{99} c_k\right) \bmod K$. Each subsequent prisoner performs modular subtraction in $\mathbb{Z}_K$, guaranteeing $100\%$ survival for all 99 remaining prisoners.

---

### 4.6 Russian Roulette Conditional Decision

#### 1. Problem Formulation
A standard 6-chamber revolver contains 2 live bullets (4 empty chambers). Two players play Russian roulette.
- Opponent goes first, points the gun at their head, pulls the trigger: **Click! An empty chamber. Opponent survives**.
- The gun is handed to you. You have two mutually exclusive options:
  - **Option 1 (Stay)**: Do not spin the cylinder; pull the trigger on the next adjacent chamber;
  - **Option 2 (Spin)**: Spin the cylinder vigorously, randomizing the chamber before pulling the trigger.
**Question**: Which action maximizes your survival probability under different bullet configurations?

#### 2. Configuration 1: The Two Bullets are Adjacent
Label the 6 chambers clockwise: $1 \to 2 \to 3 \to 4 \to 5 \to 6 \to 1$.
With adjacent bullets at chambers 1 and 2, chambers 3, 4, 5, 6 are consecutive empties:
$$\text{Chambers: } [B_1, \ B_2, \ E_3, \ E_4, \ E_5, \ E_6] \quad (B = \text{Bullet}, \ E = \text{Empty})$$

- **Conditioning on Opponent's Survival**:
  The opponent's shot was empty, meaning the hammer landed on one of the 4 empty chambers: $\{E_3, E_4, E_5, E_6\}$.
- **Tracing the Next Chamber**:
  - Opponent hit $E_3 \implies$ Your chamber is $E_4$ (Empty, Survive);
  - Opponent hit $E_4 \implies$ Your chamber is $E_5$ (Empty, Survive);
  - Opponent hit $E_5 \implies$ Your chamber is $E_6$ (Empty, Survive);
  - Opponent hit $E_6 \implies$ Your chamber is $B_1$ (**Bullet, Fatal!**).
- **Probability Comparison**:
  - **Stay**: Only 1 of the 4 conditioned outcomes is a bullet:
    $$\mathbb{P}(\text{Bullet} \mid \text{Stay, Adjacent}) = \frac{1}{4} = \mathbf{25\%}$$
  - **Spin**: Randomizes over all 6 chambers with 2 bullets:
    $$\mathbb{P}(\text{Bullet} \mid \text{Spin}) = \frac{2}{6} = \frac{1}{3} \approx \mathbf{33.33\%}$$
- **Optimal Choice**: $25\% < 33.33\% \implies$ **Stay! Do not spin the cylinder.**
- **Physical Intuition**: Clustering bullets creates a wide continuous "oasis" of 4 empty chambers. An empty shot indicates you are inside this oasis, giving a $75\%$ conditional survival rate. Spinning breaks this protective cluster.

#### 3. Configuration 2: The Two Bullets are Separated (e.g., Separated by 1 Chamber)
Place bullets at chambers 1 and 3, leaving 2, 4, 5, 6 empty:
$$\text{Chambers: } [B_1, \ E_2, \ B_3, \ E_4, \ E_5, \ E_6]$$

- Conditioned on empty chamber: $\{E_2, E_4, E_5, E_6\}$.
- Tracing the next chamber:
  - Opponent hit $E_2 \implies$ Next is $B_3$ (**Bullet, Fatal!**);
  - Opponent hit $E_4 \implies$ Next is $E_5$ (Empty, Survive);
  - Opponent hit $E_5 \implies$ Next is $E_6$ (Empty, Survive);
  - Opponent hit $E_6 \implies$ Next is $B_1$ (**Bullet, Fatal!**).
- **Probability Comparison**:
  - **Stay**: 2 out of 4 outcomes lead to death:
    $$\mathbb{P}(\text{Bullet} \mid \text{Stay, Separated}) = \frac{2}{4} = \mathbf{50\%}$$
  - **Spin**:
    $$\mathbb{P}(\text{Bullet} \mid \text{Spin}) = \frac{2}{6} \approx \mathbf{33.33\%}$$
- **Optimal Choice**: $33.33\% < 50\% \implies$ **Spin! Always re-spin when bullets are separated.**

#### 4. Configuration 3 (The Ultimate Interview Twist): Uniformly Random Bullets
What if the dealer loaded 2 bullets uniformly at random among the 6 chambers ($\binom{6}{2} = 15$ equally likely placements)?
- **Adjacent configurations**: Exactly 6 pairs: $(1,2), (2,3), (3,4), (4,5), (5,6), (6,1)$. Prior probability is $\frac{6}{15} = 40\%$;
- **Separated configurations**: The remaining $15 - 6 = 9$ pairs. Prior probability is $\frac{9}{15} = 60\%$.

**Posterior Likelihood**:
Observing that the opponent hit an empty chamber does not change the ratio of adjacent to separated configurations (each has 4 empty chambers, so the likelihood ratio is $4/6 : 4/6 = 1:1$).
- Expected death rate under **Stay**:
  $$\mathbb{P}(\text{Bullet} \mid \text{Stay}) = 0.40 \times \frac{1}{4} + 0.60 \times \frac{2}{4} = 0.10 + 0.30 = \mathbf{40\%}$$
- Death rate under **Spin**:
  $$\mathbb{P}(\text{Bullet} \mid \text{Spin}) = \frac{2}{6} = \frac{1}{3} \approx \mathbf{33.33\%}$$

$$\boxed{33.33\% < 40\% \implies \text{Under uniformly random loading, you should strictly choose to SPIN}}$$

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
