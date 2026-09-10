# Quant 13 · Game Theory & Strategic Decision Making: Nash Equilibrium, Minimax Theorem, and Dynamic Induction

Game Theory is the mathematical framework for analyzing multi-agent strategic interactions, backward induction, and decision-making under asymmetric information. In market making, auction mechanism design, dark pool liquidity competition, and algorithmic execution, game theory provides the bedrock for strategic and microstructural modeling.

This note systematically breaks down the core theoretical foundations of non-cooperative and combinatorial games, extracts **5 Core Analytical Frameworks**, and presents rigorous mathematical derivations and generalized modeling for **10 classic game theory models** from *A Practical Guide To Quantitative Finance Interviews* and *Heard on the Street*.

---

## Module 1: Foundations of Non-Cooperative Game Theory

### 1. Mathematical Formulation of Games

#### Normal-Form (Strategic-Form) Games
An $n$-player finite normal-form game is defined by a tuple $\mathcal{G} = \left( \mathcal{N}, (S_i)_{i \in \mathcal{N}}, (u_i)_{i \in \mathcal{N}} \right)$:
- Player set $\mathcal{N} = \{1, 2, \dots, n\}$;
- Pure strategy space $S_i$ for each player $i$; profile space $S = S_1 \times S_2 \times \dots \times S_n$;
- Payoff function $u_i: S \to \mathbb{R}$.

#### Extensive-Form Games
A game tree representing sequential decision making: includes decision nodes, information sets, player actions, nature moves, and terminal payoffs.

---

### 2. Dominance & Iterated Elimination of Strictly Dominated Strategies (IESDS)

- **Strictly Dominant Strategy**: A strategy $s_i^* \in S_i$ such that against all opponent strategy profiles $s_{-i}$:
  $$u_i(s_i^*, s_{-i}) > u_i(s_i, s_{-i}), \quad \forall s_i \ne s_i^*$$
- **Strictly Dominated Strategy**: If there exists $s_i'$ (or mixed strategy $\sigma_i'$) such that for all $s_{-i}$, $u_i(s_i', s_{-i}) > u_i(s_i, s_{-i})$, then $s_i$ is strictly dominated.
- **IESDS Principle**: Rational players never play strictly dominated strategies. Recursively eliminating dominated strategies leads to the **Rationalizable Solution Set**.

---

### 3. Nash Equilibrium (NE)

#### Pure Strategy Nash Equilibrium (PNE)
A strategy profile $s^* = (s_1^*, s_2^*, \dots, s_n^*) \in S$ is a Nash Equilibrium if for all players $i \in \mathcal{N}$ and all unilateral deviations $s_i \in S_i$:
$$u_i(s_i^*, s_{-i}^*) \ge u_i(s_i, s_{-i}^*)$$
No player has an incentive to unilaterally deviate (best responses form mutual fixed points).

#### Mixed Strategy Nash Equilibrium (MSNE) & The Indifference Principle
Let player $i$ choose mixed strategy $\sigma_i \in \Delta(S_i)$.
> **The Indifference Principle**: In any MSNE, every pure strategy in player $i$'s support $\operatorname{supp}(\sigma_i)$ must yield the **exact same expected payoff**, which is at least as high as any strategy outside the support:
> 
> $$\mathbb{E}_{\sigma_{-i}}[u_i(s_{i, 1}, \sigma_{-i})] = \mathbb{E}_{\sigma_{-i}}[u_i(s_{i, 2}, \sigma_{-i})] = \dots = \max_{s_i \in S_i} \mathbb{E}_{\sigma_{-i}}[u_i(s_i, \sigma_{-i})]$$

---

### 4. Zero-Sum Games & von Neumann's Minimax Theorem

In two-player zero-sum games ($u_1(s_1, s_2) + u_2(s_1, s_2) = 0$), Player 1's payoff maximization is equivalent to Player 2's minimization of Player 1's payoff. With Player 1 payoff $u(s_1, s_2)$, von Neumann's Minimax Theorem guarantees:

$$\max_{\sigma_1 \in \Delta_1} \min_{\sigma_2 \in \Delta_2} \mathbb{E}[u(\sigma_1, \sigma_2)] = \min_{\sigma_2 \in \Delta_2} \max_{\sigma_1 \in \Delta_1} \mathbb{E}[u(\sigma_1, \sigma_2)] = V^*$$

$V^*$ is termed the **Value of the Game**.

#### Minimax Leveling and the Duality with the Indifference Principle ("Shaving Peaks and Filling Valleys")

Why must the expected payoffs of the opponent's candidate pure actions be precisely leveled in a mixed strategy Nash equilibrium?

**Core Mechanism**:
The underlying principle stems from the **Minimax Theorem**: if the expected payoffs of different options are unequal, Player B will rationally exploit the situation by choosing whichever option offers the highest payoff. To suppress B's maximum possible payoff to its lowest possible level, Player A must "shave the peaks and fill the valleys" (level the payoffs) until all candidate options in B's support are completely equalized.

The formal mathematical formulation is as follows:
Let Player A choose mixed strategy $p$. Facing a rational opponent B with $K$ candidate pure strategy options $\{s_{B,1}, s_{B,2}, \dots, s_{B,K}\}$, under A's strategy $p$, the expected payoffs for opponent B's choices are:
$$E_B(s_{B,1}; p), \; E_B(s_{B,2}; p), \; \dots, \; E_B(s_{B,K}; p)$$

1. **Opponent Exploitation Tendency**:
   Opponent B, as a rational payoff maximizer, upon deducing or observing A's probability distribution $p$, **will invariably choose the option with the maximum expected payoff**:
   $$\text{Best Response of B} = \arg\max_{j} E_B(s_{B,j}; p)$$
   B's actual payoff is bounded below by the maximum of these options: $\max_j E_B(s_{B,j}; p)$.

2. **A's Necessary Minimax Leveling ("Shaving Peaks and Filling Valleys")**:
   Player A's objective is to minimize B's maximum attainable payoff ($\min_p \max_j E_B(s_{B,j}; p)$):
   - If the expected payoffs across candidate options differ (e.g., Option 1 yields 2, while Option 2 yields 5), B will target Option 2, capturing a high payoff of 5;
   - This difference creates an exploitable "payoff peak". To drive B's score down, A must shift probability mass to suppress Option 2;
   - Shifting probability mass reduces the higher expected payoff while elevating lower ones—a process of **shaving peaks and filling valleys**;
   - As long as candidate payoffs remain unequal, B can always target the highest peak, giving A continuous incentive to further level the distribution;
   - **Steady State**: A must continue this leveling process **until all candidate options in B's support yield the exact same expected payoff**:
     $$E_B(s_{B,1}; p^*) = E_B(s_{B,2}; p^*) = \dots = E_B(s_{B,K}; p^*) = V^*$$

3. **Geometric Duality (Upper Envelope Minimization)**:
   The function $M(p) = \max_j E_B(s_{B,j}; p)$ is the **upper envelope** of several affine linear functions of $p$, forming a convex, piecewise-linear V-shape.
   Because the affine lines have slopes of opposing signs, the minimum of the upper envelope ($\min_p M(p)$) occurs precisely at the **intersection point (vertex)** where the lines cross. At this vertex, the candidate payoffs are strictly equal. Thus, the minimax optimality condition is geometrically equivalent to rendering the opponent completely indifferent across candidate actions.

---

### 5. Dynamic Games & Subgame Perfect Nash Equilibrium (SPE)

- **Subgame**: A self-contained subtree starting from a singleton information set.
- **Subgame Perfect Equilibrium (SPE)**: A strategy profile that induces a Nash equilibrium in **every subgame**.
- **Kuhn's Theorem**: Every finite extensive-form game of perfect information can be solved via **Backward Induction** to find a pure-strategy SPE.

---

## Module 2: 5 Core Analytical Frameworks

Based on the information structure and action sequence, classic game models can be classified into 5 core analytical frameworks:

| Framework | Applicable Game Characteristics | Core Mathematical Mechanism | Canonical Models |
| :--- | :--- | :--- | :--- |
| **1. Backward Induction & Parity Recursion** | Complete information, finite dynamic games | Subgame Perfect Equilibrium (SPE), Kuhn's Theorem, backward recurrence | Pirate Gold, Tigers & Sheep, The Truel |
| **2. Contraction Mapping & IESDS** | Complete information, continuous/measurable strategy spaces | Contraction Mapping, Banach Fixed-Point Theorem | Guess 2/3 of the Average, Traveler's Dilemma |
| **3. Mixed Strategies & Minimax Leveling** | Complete information, discrete adversarial simultaneous games | von Neumann Minimax Theorem, Indifference Principle, linear algebra systems | Inspection games, Bluffing, Russian Roulette |
| **4. Bayesian Nash & Auction ODEs** | Incomplete information, private values continuous bidding | Bayesian Nash Equilibrium (BNE), First-Order Condition ODEs, Envelope Theorem | First/Second-Price, All-Pay Auction, Winner's Curse |
| **5. Combinatorial Games & Invariant Arguments** | Deterministic 2-player discrete complete information games | Sprague-Grundy Theorem, Nim-Sum parity, Strategy-Stealing argument | Coins in a Line, Chomp Grid, Hat Riddle |

---

### Framework 1: Backward Induction & Parity Recursion
- **Applicability**: Pirate gold sharing, Tigers & Sheep, The Truel (3-way duel), sequential turn-based decisions.
- **Standard 3-Step Procedure**:
  1. **Identify the Base Case ($k=1, 2$)**: Directly solve the terminal state with minimal remaining players;
  2. **Inductive Step & Marginal Incentives ($k \to k+1$)**: Determine the minimal cost required for a new decision-maker to buy the needed votes (turning a 0 into a 1) or exploit survival threats;
  3. **Parity Collapse & Invariants**: If state transitions exhibit periodic cycles or absorbing states, extract parity or modular arithmetic patterns.

---

### Framework 2: Contraction Mapping & IESDS
- **Applicability**: Guess $2/3$ of the Average (Beauty Contest), Traveler's Dilemma.
- **Standard 3-Step Procedure**:
  1. **Bound the Feasible Strategy Space $[a_0, b_0]$**;
  2. **Prove Strict Dominance on Boundary Slices**: For any player, bidding $s > \alpha b_0$ is strictly dominated by bidding $\alpha b_0$;
  3. **Construct a Contracting Sequence**: $b_{k+1} = \mathcal{T}(b_k)$, and invoke the Banach Fixed-Point Theorem to show the interval collapses to the unique fixed point $s^* = 0$.

---

### Framework 3: Mixed Strategies, Indifference Principle & Minimax Leveling
- **Applicability**: Rock-Paper-Scissors variants, Inspection games, Poker bluffing, Russian roulette.
- **Standard 3-Step Procedure**:
  1. **Parameterize Player Strategy Distribution**: Let Player A play mixed strategy vector $p = (p_1, \dots, p_m)$;
  2. **Formulate Opponent's Expected Payoffs**: Compute opponent B's payoff $E_B(j; p)$ for each pure action $j$;
  3. **Minimax Leveling System of Equations**: Using the Minimax principle, equate all candidate actions in B's support: $E_B(1; p^*) = E_B(2; p^*) = \dots = V^*$, and solve for the optimal mixed strategy $p^*$.

---

### Framework 4: Bayesian Symmetric Bidding & ODEs
- **Applicability**: First-Price Auction (FPA), All-Pay Auction, Winner's Curse corporate acquisition.
- **Standard 3-Step Procedure**:
  1. **Assume Symmetric Monotonic Bidding**: Hypothesize a strictly monotonic bid function $b(v)$ with inverse $v = \beta(b)$;
  2. **Construct Expected Profit Objective**:
     $$\mathbb{E}[\Pi(b \mid v)] = (v - b) \cdot \mathbb{P}\left(\max_{j \ne i} v_j < \beta(b) \right) = (v - b) \cdot [F(\beta(b))]^{n-1}$$
  3. **Transform FOC into an ODE with Boundary Conditions**: Set $\frac{\partial \mathbb{E}[\Pi]}{\partial b} = 0$, substitute $\beta(b) = v$ to derive a linear ODE, and integrate using $b(0) = 0$.

---

### Framework 5: Combinatorial Games, Nim-Sum & Strategy-Stealing
- **Applicability**: Nim variants, Chomp chocolate grid, Coins in a line, Hat riddles.
- **Standard 3-Step Procedure**:
  1. **Define $P$-positions (Previous player wins / losing state) and $N$-positions (Next player wins / winning state)**;
  2. **Compute Invariants**: Calculate the binary Nim-sum $\bigoplus x_i$, where $0$ corresponds to a $P$-position;
  3. **Strategy-Stealing Proof**: Show that if the second player had a winning strategy, the first player could take a minimal non-disruptive move on Turn 1 to steal it, establishing a contradiction.

---

## Module 3: 10 Classic Quant Game Theory Models & Rigorous Solutions

---

### 1. Pirate Gold Sharing ($N$ Pirates & $M$ Gold Coins Generalization)

> **Problem Definition (Pirate Gold Allocation / Backward Induction)**:
> 5 hyper-rational pirates (ranked 1 to 5 by seniority/cruelty, Pirate 1 most senior) must divide 100 indivisible gold coins.
> Rules:
> 1. Pirate 1 proposes an allocation. All pirates vote (including the proposer);
> 2. If $\ge 50\%$ of remaining pirates vote YES, the proposal passes and the game ends;
> 3. If rejected, the proposer is thrown overboard to sharks, and Pirate 2 proposes next;
> 4. Pirate preferences: Survival > Gold > Bloodlust (if payoff is tied, vote NO to throw proposer overboard).
> 
> **Question**: What allocation should Pirate 1 propose? What is the general rule for $N$ pirates and $M$ coins?

#### Analytical Method: Backward Induction

#### Rigorous Derivation:
- **Subgame 1 (2 Pirates remaining: 4, 5)**:
  - Pirate 4 proposes: voting for themselves guarantees $\frac{1}{2} = 50\% \ge 50\%$;
  - Allocation: `(100, 0)`. Pirate 5 receives 0 coins.
- **Subgame 2 (3 Pirates remaining: 3, 4, 5)**:
  - Pirate 3 needs 1 additional vote ($2/3 \ge 50\%$);
  - In the next round, Pirate 5 would receive 0. Offering Pirate 5 **1 coin** guarantees their vote ($1 > 0$ strictly dominates bloodlust);
  - Allocation: `Pirate 3: 99, Pirate 4: 0, Pirate 5: 1`. Votes: Pirates 3, 5 vote YES (passes with 2 votes).
- **Subgame 3 (4 Pirates remaining: 2, 3, 4, 5)**:
  - Pirate 2 needs 1 additional vote ($2/4 = 50\%$);
  - Opportunity cost check: if Pirate 2 dies, Pirate 4 gets 0, Pirate 5 gets 1;
  - Pirate 2 buys Pirate 4 with 1 coin:
  - Allocation: `Pirate 2: 98, Pirate 3: 0, Pirate 4: 1, Pirate 5: 0`. Votes: Pirates 2, 4 vote YES (passes with 2 votes).
- **Subgame 4 (Full Game with 5 Pirates: 1 to 5)**:
  - Pirate 1 needs 2 additional votes ($3/5 = 60\% \ge 50\%$);
  - Opportunity cost check if Pirate 1 dies: `Pirate 2: 98, Pirate 3: 0, Pirate 4: 1, Pirate 5: 0`;
  - Pirate 1 buys the cheapest votes: Pirate 3 (1 coin) and Pirate 5 (1 coin):
  - Allocation: `Pirate 1: 96, Pirate 2: 0, Pirate 3: 1, Pirate 4: 0, Pirate 5: 1`.

$$\boxed{\text{Pirate 1 proposes: }(96, 0, 1, 0, 1) \quad \text{Votes: Pirates 1, 3, 5 (Passes with 3 votes)}}$$

#### Generalization to $N$ Pirates:
When $N > 2M$, coins are exhausted and proposers cannot buy enough votes. Proposers must offer all coins to survive (receiving 0 coins). For very large $N$, only pirates where the number of surviving pirates is of the form $2M + 2^k$ can survive; all others are voted off.

---

### 2. Tigers and Sheep Island ($N$ Tigers, 1 Sheep)

> **Problem Definition (Tigers and Sheep / Parity Invariant & Absorbing Threats)**:
> An island has 1 sheep and $N$ hyper-rational tigers.
> Rules:
> 1. Tigers can eat grass to survive, but prefer eating sheep;
> 2. If a tiger eats the sheep, that tiger **turns into a sheep** (and can subsequently be eaten by others);
> 3. Tiger preferences: Survival > Eating sheep;
> 4. All tigers are perfectly rational and possess common knowledge of rationality.
> 
> **Question**: For $N$ tigers, will the first tiger eat the sheep?

#### Analytical Method: Parity Backward Induction

#### Rigorous Derivation:
- **$N = 1$ Tiger**: The tiger eats the sheep, turns into a sheep. With no other tigers on the island, it survives. **Eats!**
- **$N = 2$ Tigers**: If Tiger A eats the sheep, it turns into a sheep with 1 tiger remaining. By the $N=1$ result, Tiger B will eat it. Tiger A would die. By survival prioritization: **Does NOT eat!**
- **$N = 3$ Tigers**: If Tiger A eats the sheep, it turns into a sheep with 2 tigers remaining. By the $N=2$ result, neither tiger dares eat it. Tiger A is safe. By food preference: **Eats!**
- **$N = 4$ Tigers**: If Tiger A eats the sheep, the game reduces to $N=3$ where the next tiger will eat it. Hence: **Does NOT eat!**

#### General Parity Rule:
$$\boxed{\begin{cases} N \text{ is Odd} & \implies \text{The first tiger immediately eats the sheep} \\ N \text{ is Even} & \implies \text{No tiger eats the sheep; the sheep survives safely} \end{cases}}$$

---

### 3. The Truel: 3-Player Duel with Different Accuracies

> **Problem Definition (The Truel / State Transition Probabilities & Strategic Pass)**:
> Three gunfighters A, B, C duel, taking turns shooting in sequential order (A $\to$ B $\to$ C $\to$ A $\dots$).
> Fixed hit probabilities:
> - Gunfighter A: $p_A = 1/3$;
> - Gunfighter B: $p_B = 2/3$;
> - Gunfighter C: $p_C = 1$ (perfect marksman).
> 
> The last survivor wins. When it is a player's turn, they may shoot at any opponent or **deliberately shoot into the air (Pass)**.
> **Question**: What is Gunfighter A's optimal first-round strategy? What are the ultimate survival probabilities for each player?

#### Analytical Method: State Transition Probabilities & Subgame Backward Analysis

#### Rigorous Derivation:
1. **Analyze 2-Player Subgames**:
   - **A vs B (A shoots first)**: Let A's win probability be $P_{AB}$.
     $$P_{AB} = p_A + (1 - p_A)(1 - p_B) P_{AB} = \frac{1}{3} + \frac{2}{3} \cdot \frac{1}{3} P_{AB} = \frac{1}{3} + \frac{2}{9} P_{AB} \implies \boxed{P_{AB} = \frac{3}{7}}$$
     B's win probability is $1 - \frac{3}{7} = \frac{4}{7}$.
   - **A vs C (A shoots first)**:
     $$P_{AC} = p_A + (1 - p_A) \cdot 0 = \boxed{\frac{1}{3}}$$
   - **B vs C (B shoots first)**:
     $$P_{BC} = p_B + (1 - p_B) \cdot 0 = \boxed{\frac{2}{3}}$$

2. **Targeting Logic with 3 Players Alive**:
   - **B's optimal target**: Must shoot at the most lethal threat **C** (killing A leaves C to kill B with certainty; killing C leaves a duel with A where B has higher accuracy);
   - **C's optimal target**: Must shoot at the more accurate threat **B** (killing A leaves B with a $2/3$ chance to kill C).

3. **A's First-Round Decision**:
   - **Strategy 1: A attempts to shoot C**:
     - If hits C (probability $1/3$), enters duel with B, but **B shoots first**! A's survival probability is $(1 - p_B) P_{AB} = \frac{1}{3} \times \frac{3}{7} = \frac{1}{7}$;
     - If misses C (probability $2/3$), B takes turn, shoots C, then enters duel with B where A shoots first (win probability $\frac{3}{7}$);
     - Total survival prob $= \frac{1}{3} \times \frac{1}{7} + \frac{2}{3} \times \frac{3}{7} = \frac{1 + 6}{21} = \frac{7}{21} = \frac{1}{3} \approx 33.3\%$.
   - **Strategy 2: A attempts to shoot B**:
     - If hits B (probability $1/3$), C shoots next and kills A with 100% certainty (payoff 0);
     - If misses B (probability $2/3$), B shoots C, then A shoots first against B (win prob $\frac{3}{7}$);
     - Total survival prob $= \frac{1}{3} \times 0 + \frac{2}{3} \times \frac{3}{7} = \frac{6}{21} \approx 28.6\%$.
   - **Strategy 3: A deliberately shoots into the air (Pass)**:
     - A misses intentionally; turn passes to B;
     - B must target C (kills C with probability $2/3$);
     - If B kills C ($2/3$), A enters duel with B with **first-mover advantage**: win prob $P_{AB} = \frac{3}{7}$;
     - If B misses C ($1/3$), C kills B ($1$), and A enters duel with C with **first-mover advantage**: win prob $P_{AC} = \frac{1}{3}$;
     - Total survival prob:
       $$\mathbb{P}(\text{A Wins}) = \frac{2}{3} \times \frac{3}{7} + \frac{1}{3} \times \frac{1}{3} = \frac{2}{7} + \frac{1}{9} = \frac{18 + 7}{63} = \boxed{\frac{25}{63} \approx 39.7\%}$$

#### Conclusion:
$$\boxed{\text{Gunfighter A's optimal strategy: Deliberately shoot into the air (Pass)}}$$
- Final survival probabilities: $\mathbb{P}(A) = \frac{25}{63} \approx 39.7\%$, $\mathbb{P}(B) = \frac{2}{3} \times \frac{4}{7} = \frac{8}{21} = \frac{24}{63} \approx 38.1\%$, $\mathbb{P}(C) = \frac{1}{3} \times \frac{2}{3} = \frac{14}{63} \approx 22.2\%$. The gunfighter with the worst accuracy attains the highest survival probability.

---

### 4. Guess 2/3 of the Average (Keynesian Beauty Contest)

> **Problem Definition (Beauty Contest / Guess 2/3 of the Average)**:
> $N$ players ($N \ge 3$) independently submit a real number $x_i \in [0, 100]$.
> Let the group average be $\mu = \frac{1}{N} \sum_{i=1}^N x_i$.
> The player whose choice is **closest to $\frac{2}{3}$ of the average ($\frac{2}{3}\mu$)** wins the entire prize.
> **Question**: Under common knowledge of rationality, what is the unique Nash equilibrium?

#### Analytical Method: Contraction Mapping & IESDS

#### Rigorous Derivation:
1. **Initial Feasible Domain**: $S_0 = [0, 100]$. Since $x_i \le 100$, the maximum possible average is $\mu \le 100$, so the target is bounded by $\frac{2}{3}\mu \le \frac{200}{3} \approx 66.67$.
   - Any bid $> 66.67$ is strictly dominated (regardless of others' bids, $66.67$ is strictly closer to $\frac{2}{3}\mu$).
   - **Round 1 Elimination**: Strategy space contracts to $S_1 = [0, 66.67]$.
2. **Round 2 Elimination**: Rational players know all bids are in $[0, 66.67]$. Hence $\mu \le 66.67$, giving an upper bound $\frac{2}{3} \times 66.67 = \frac{400}{9} \approx 44.44$.
   - Strategy space contracts to $S_2 = [0, 44.44]$.
3. **Round $k$ Recurrence**:
   $$S_k = \left[ 0, 100 \times \left( \frac{2}{3} \right)^k \right]$$
4. **Limiting Behavior**:
   $$\lim_{k \to \infty} 100 \times \left( \frac{2}{3} \right)^k = 0 \implies S_\infty = \{0\}$$

$$\boxed{\text{Unique Nash Equilibrium: Every player bids } 0 \quad (x_1^* = x_2^* = \dots = x_N^* = 0)}$$

---

### 5. Russian Roulette Decision (Bayesian Filtering & Conditional Odds)

> **Problem Definition (Russian Roulette / Bayesian Conditioning & Survival Probability)**:
> A 6-chamber revolver is loaded with 2 live bullets. The cylinder is given a single random spin.
> Rules:
> 1. The opponent takes the revolver, aims at themselves, and pulls the trigger—**Click (Empty chamber, survives)**;
> 2. It is now your turn. You have two options:
>    - **Option A**: Pull the trigger directly without spinning;
>    - **Option B**: Require the cylinder to be re-spun randomly before pulling the trigger.
> 
> **Question**: Which option should you choose under the following two scenarios?
> - **Scenario 1**: The two live bullets were loaded in **adjacent** chambers;
> - **Scenario 2**: The two live bullets were loaded in **non-adjacent** chambers.

#### Analytical Method: Conditional Probability & Bayesian Filtering

#### Rigorous Derivation:
- **Scenario 1 (Adjacent Loading, $(1, 1, 0, 0, 0, 0)$)**:
  - Six consecutive cylinder positions: $(B_1, B_2, E_3, E_4, E_5, E_6)$ ($B$ = Bullet, $E$ = Empty);
  - Opponent pulled trigger on an empty chamber, so their position was one of $\{E_3, E_4, E_5, E_6\}$ (4 equally likely outcomes);
  - **If pulling directly (No Re-spin)**: The cylinder advances to the next position:
    - If opponent was at $E_3 \to$ next is $E_4$ (Safe);
    - If opponent was at $E_4 \to$ next is $E_5$ (Safe);
    - If opponent was at $E_5 \to$ next is $E_6$ (Safe);
    - If opponent was at $E_6 \to$ next is $B_1$ (**Shot!**);
    - Probability of getting shot: $\mathbb{P}(\text{Shot} \mid \text{No Re-spin}) = \frac{1}{4} = \mathbf{25\%}$.
  - **If requesting a Re-spin**: Resets to prior distribution over 6 chambers with 2 bullets:
    - $\mathbb{P}(\text{Shot} \mid \text{Re-spin}) = \frac{2}{6} = \frac{1}{3} \approx \mathbf{33.33\%}$.
  - **Conclusion 1**: For adjacent bullets, **Option A (Pull directly without re-spinning) has lower hazard (25% < 33.3%)**!

- **Scenario 2 (Non-adjacent Loading)**:
  - Opponent was at one of the 4 empty chambers;
  - Since bullets are non-adjacent, each bullet is preceded by an empty chamber (2 of the 4 empty chambers lead directly to a bullet);
  - **If pulling directly**: $\mathbb{P}(\text{Shot} \mid \text{No Re-spin}) = \frac{2}{4} = \mathbf{50\%}$;
  - **If requesting a Re-spin**: $\mathbb{P}(\text{Shot} \mid \text{Re-spin}) = \frac{2}{6} = \mathbf{33.33\%}$;
  - **Conclusion 2**: For non-adjacent bullets, **Option B (Must re-spin) has lower hazard (33.3% < 50%)**!

---

### 6. Classical Auction Theory (First-Price, Second-Price, All-Pay & Revenue Equivalence)

> **Problem Definition (Auction Theory / Sealed-Bid BNE & Revenue Equivalence)**:
> $n$ risk-neutral bidders compete for a single indivisible good. Each bidder's private valuation $v_i$ is drawn independently from $U[0, 1]$.
> 1. **Second-Price Auction (Vickrey)**: Highest bidder wins and pays the **second-highest bid**;
> 2. **First-Price Auction (FPA)**: Highest bidder wins and pays **their own bid**;
> 3. **All-Pay Auction**: Highest bidder wins, but **every bidder pays their own bid regardless of winning**.
> 
> **Question**: Derive the symmetric Bayesian Nash equilibrium bidding strategies $b^*(v)$ for each auction format and compare the seller's expected revenue.

#### Analytical Method: Symmetric Bayesian Nash Equilibrium & ODEs

#### Rigorous Derivation:

#### 1. Second-Price Auction (Vickrey)
- Bidding true valuation $b^*(v) = v$ is a weakly dominant strategy:
  - If bidding $b > v$: the outcome changes only if the second-highest bid $P_2 \in (v, b)$. You win but pay $P_2 > v$, resulting in net payoff $v - P_2 < 0$;
  - If bidding $b < v$: the outcome changes only if $P_2 \in (b, v)$. You lose a profitable transaction ($v - P_2 > 0$), resulting in payoff 0;
  - Hence, truthful bidding $\boxed{b^*(v) = v}$ is weakly dominant.

#### 2. First-Price Auction
- Postulate a strictly increasing symmetric bidding function $b(v)$ with inverse $\beta(b)$. Expected profit:
  $$\mathbb{E}[\Pi(b \mid v)] = (v - b) \cdot \mathbb{P}\left(\max_{j \ne i} v_j < \beta(b) \right) = (v - b) \cdot [\beta(b)]^{n-1}$$
- First-Order Condition (FOC) with respect to $b$:
  $$- [\beta(b)]^{n-1} + (v - b) (n - 1) [\beta(b)]^{n-2} \beta'(b) = 0$$
- In symmetric equilibrium, $b = b(v) \implies \beta(b) = v$ and $\beta'(b) = \frac{1}{b'(v)}$:
  $$-v^{n-1} + (v - b(v)) (n - 1) v^{n-2} \frac{1}{b'(v)} = 0 \implies b'(v) v + (n - 1) b(v) = (n - 1) v$$
- Multiply both sides by the integrating factor $v^{n-2}$:
  $$\frac{d}{dv} \left[ b(v) v^{n-1} \right] = (n - 1) v^{n-1} \implies b(v) v^{n-1} = \frac{n - 1}{n} v^n + C$$
- Boundary condition $b(0) = 0 \implies C = 0$:
  $$\boxed{b^*(v) = \frac{n - 1}{n} v}$$
  *(For two bidders $n=2$, $b^*(v) = \frac{1}{2}v$, i.e., bidding half valuation).*

#### 3. All-Pay Auction
- Expected profit: $\mathbb{E}[\Pi(b \mid v)] = v \cdot [\beta(b)]^{n-1} - b$.
- FOC: $v (n - 1) [\beta(b)]^{n-2} \beta'(b) - 1 = 0 \implies b'(v) = (n - 1) v^{n-1}$.
- Integrating with $b(0) = 0$:
  $$\boxed{b^*(v) = \frac{n - 1}{n} v^n}$$

#### 4. Revenue Equivalence Theorem
Under the four standard benchmark assumptions (independent private values, risk neutrality, highest valuation wins, zero surplus for lowest valuation), the **expected seller revenue across all three auction formats is strictly identical**:
$$\mathbb{E}[\text{Revenue}] = \frac{n - 1}{n + 1}$$

---

### 7. Winner's Curse in Corporate Acquisition

> **Problem Definition (Winner's Curse / Adverse Selection & Truncated Expectation)**:
> Company A evaluates an acquisition of Target Company T.
> Rules:
> - Target T's true underlying value $V$ is unknown to A, distributed uniformly $V \sim U[0, 100]$;
> - Target T's management knows its exact value $V$;
> - Under A's superior management, T's value will increase by $50\%$ (becoming $1.5 V$);
> - A makes a take-it-or-leave-it cash tender offer $B$: if $B \ge V$, T accepts, giving A payoff $1.5 V - B$; if $B < V$, T rejects, giving payoff 0.
> 
> **Question**: What bid $B^*$ should Company A make to maximize expected profit?

#### Analytical Method: Truncated Conditional Expectation

#### Rigorous Derivation:
- **Common Cognitive Trap**: Unconditional expectation $\mathbb{E}[V] = 50$, post-acquisition expected value $1.5 \times 50 = 75$. Bidding $B = 50$ naively appears to generate expected profit $75 - 50 = 25$.
- **Rigorous Conditional Derivation**:
  - The acquisition succeeds only if $V \le B$ (adverse selection: target accepts only when undervalued by the bid);
  - **Conditioned on success, target valuation is not 50, but the truncated expectation**:
    $$\mathbb{E}[V \mid V \le B] = \frac{B}{2}$$
  - Post-acquisition expected value conditioned on acceptance is:
    $$\mathbb{E}[1.5 V \mid V \le B] = 1.5 \times \frac{B}{2} = 0.75 B$$
  - Company A's expected profit is:
    $$\mathbb{E}[\text{Profit}] = \mathbb{P}(V \le B) \cdot \left( \mathbb{E}[1.5 V \mid V \le B] - B \right) = \frac{B}{100} \cdot (0.75 B - B) = -\frac{0.25 B^2}{100} \le 0$$

$$\boxed{\text{Optimal Bid is: } B^* = 0 \quad (\text{Any positive bid } B > 0 \text{ yields an expected loss; never acquire})}$$

---

### 8. Coins in a Line Game ($2n$ Coins)

> **Problem Definition (Coins in a Line / Parity Invariant & First-Player Advantage)**:
> $2n$ coins of known values $v_1, v_2, \dots, v_{2n}$ are arranged in a line.
> Rules: Two players alternate taking one coin from either the far-left or far-right end. Both players are hyper-rational.
> **Question**: Prove that the first player can always secure at least $50\%$ of the total sum (i.e., never loses). What is the optimal parity strategy?

#### Analytical Method: Parity Position Invariant

#### Rigorous Derivation:
1. **Odd/Even Index Partition**:
   Partition all coins into two sets based on their original positions:
   - **Odd Set**: $O = \{v_1, v_3, v_5, \dots, v_{2n-1}\}$, with sum $S_{\text{odd}} = \sum v_{2k-1}$;
   - **Even Set**: $E = \{v_2, v_4, v_6, \dots, v_{2n}\}$, with sum $S_{\text{even}} = \sum v_{2k}$.
2. **First-Player Parity Control**:
   - Initially, the leftmost coin is $v_1$ (odd) and the rightmost is $v_{2n}$ (even);
   - **If Player 1 wants all odd-indexed coins**:
     - Player 1 takes $v_1$ (odd) on Turn 1;
     - The exposed ends are now $v_2$ (even) and $v_{2n}$ (even);
     - **Player 2 is forced to choose an even-indexed coin**;
     - Whichever even coin Player 2 takes, an odd-indexed coin is immediately exposed at that end for Player 1;
   - Similarly, if Player 1 chooses $v_{2n}$ (even) on Turn 1, Player 2 is forced to choose odd-indexed coins, allowing Player 1 to capture all even-indexed coins.
3. **First-Player Decision**:
   - Compare $S_{\text{odd}}$ and $S_{\text{even}}$:
     - If $S_{\text{odd}} \ge S_{\text{even}}$, take $v_1$ on Turn 1 and exclusively odd coins thereafter;
     - If $S_{\text{even}} > S_{\text{odd}}$, take $v_{2n}$ on Turn 1 and exclusively even coins thereafter.

$$\boxed{\text{First player guaranteed payoff lower bound: } \max(S_{\text{odd}}, S_{\text{even}}) \ge \frac{1}{2} \sum_{i=1}^{2n} v_i}$$

---

### 9. Chomp Chocolate Grid & Strategy-Stealing Argument

> **Problem Definition (Chomp Grid Game / Strategy-Stealing Contradiction)**:
> An $R \times C$ ($R, C \ge 2$) chocolate grid has a poisoned piece at $(1, 1)$ (bottom-left).
> Two players take turns selecting an available cell $(r, c)$ and removing that cell along with all cells to its top and right ($\{(x, y) \mid x \ge r, y \ge c\}$).
> The player forced to eat the poisoned piece $(1, 1)$ loses.
> **Question**: Prove that for any $R, C \ge 2$, the first player always has a winning strategy.

#### Analytical Method: Strategy-Stealing Argument

#### Rigorous Derivation:
Chomp is a finite, deterministic 2-player game of perfect information with no ties. By Zermelo's Theorem, one player must possess a winning strategy.
We prove by contradiction using a **Strategy-Stealing argument** that Player 2 cannot possess a winning strategy:
1. Suppose Player 2 has a winning strategy $\mathcal{S}_2$;
2. On Move 1, Player 1 eats only the single top-right square $(R, C)$. Let the resulting board state be $K$;
3. By hypothesis, Player 2 facing state $K$ has a winning response, say removing sub-rectangle $A$ (rooted at $(r_0, c_0)$), leading to winning state $K'$;
4. However, Player 1 could have chosen move $A$ **directly on Move 1** (since removing $A$ inherently removes $(R, C)$ as well), transitioning the board immediately to state $K'$ and adopting Player 2's putative winning strategy $\mathcal{S}_2$;
5. This contradicts the hypothesis that Player 2 has a winning strategy. Thus, Player 2 cannot possess a winning strategy.

$$\boxed{\text{The First Player must possess a winning strategy (Non-constructive proof)}}$$

---

### 10. 100 Prisoners and Red/Blue Hats (Parity Code Invariant)

> **Problem Definition (100 Prisoners Hat Riddle / Parity Check Broadcast)**:
> 100 prisoners stand in a single line facing forward. Each wears a red or blue hat chosen at random.
> Rules:
> 1. Each prisoner sees all hats ahead of them, but cannot see their own hat or hats behind them;
> 2. Starting from the back (Prisoner 100, who sees 99 hats) forward to Prisoner 1, each prisoner must state their own hat color;
> 3. No communication other than speaking the single word "Red" or "Blue" is permitted;
> 4. Guessing correctly saves the prisoner; guessing incorrectly results in execution.
> 
> **Question**: If prisoners strategize beforehand, what is the maximum number guaranteed to survive?

#### Analytical Method: Parity Invariant Coding

#### Rigorous Derivation:
- **Color Mapping**: Let Red $= 1$, Blue $= 0$;
- **Prisoner 100's Broadcast**:
  - Prisoner 100 counts the total number of red hats ahead: $S_{99} = \sum_{i=1}^{99} c_i$;
  - Prisoner 100 calls out the **parity bit** of these 99 hats:
    $$\text{Color} = S_{99} \pmod 2 \quad (\text{Odd} \implies \text{Red}, \text{Even} \implies \text{Blue})$$
  - Prisoner 100 has a $50\%$ chance of surviving.
- **Deterministic Survival for Prisoners 99 down to 1**:
  - Prisoner 99 knows the total parity $P = S_{99} \pmod 2$;
  - Prisoner 99 counts the red hats among the 98 prisoners ahead: $S_{98}$;
  - Prisoner 99 deduces their own hat color:
    $$c_{99} = (P - S_{98}) \pmod 2$$
  - Prisoner 99 calls out $c_{99}$ with $100\%$ accuracy and survives!
  - Prisoner 98 hears $c_{99}$, updates the known parity sum, and deduces $c_{98}$ with $100\%$ accuracy;
  - Proceeding inductively, all prisoners 99 down to 1 survive with $100\%$ certainty.

$$\boxed{\text{Guarantees at least 99 prisoners survive (Expected survival: } 99.5 \text{ prisoners)}}$$

---

## Module 4: Summary Matrix of Classic Game Models and Equilibria

| Classic Game Model | Core Characteristics & Game Type | Analytical Method / Induction Logic | Optimal Policy / Core Conclusion |
| :--- | :--- | :--- | :--- |
| **Pirate Gold Sharing ($N$ Pirates, $M$ Coins)** | Finite dynamic game of complete information | Backward Induction | 5 Pirates 100 Coins: $(96, 0, 1, 0, 1)$ |
| **Tigers & Sheep ($N$ Tigers, 1 Sheep)** | Parity absorbing threat game | Parity Induction | Odd $N \implies$ Eat; Even $N \implies$ Never eat |
| **The Truel (Accuracies $1/3, 2/3, 1$)** | Discrete Markov sequential dynamic duel | Subgame Backward Reasoning | Weakest gunfighter intentionally **passes (shoots in air)** $\implies 39.7\%$ win rate |
| **Guess $2/3$ of Average (Beauty Contest)** | Continuous coordination game | Contraction Mapping & IESDS | Unique Nash Equilibrium: All bid $0$ |
| **Russian Roulette Decision** | Bayesian screening under incomplete info | Conditional Probability | Adjacent: **No re-spin** ($25\%$); Non-adjacent: **Must re-spin** ($33\%$) |
| **First-Price Auction (FPA)** | Independent private values symmetric static game | Symmetric ODE | $b^*(v) = \frac{n-1}{n} v$ (For $n=2$, bid $50\%$ of valuation) |
| **Second-Price Auction (SPA)** | Explicit weakly dominant strategy mechanism | Dominant Strategy Verification | $b^*(v) = v$ (Truthful valuation bidding) |
| **Corporate Acquisition (Winner's Curse)** | Adverse selection & conditional expectation | Truncated Conditional Expectation | $B^* = 0$ (Winner's curse guarantees expected loss; never bid) |
| **Coins in a Line ($2n$ Coins)** | Deterministic zero-sum combinatorial game | Odd/Even Index Parity | First player selects larger parity group $\implies \ge 50\%$ total |
| **Chomp Chocolate Grid** | Impartial game with no ties | Strategy-Stealing Argument | First player winning strategy exists (Non-constructive) |
| **100 Prisoners Hat Riddle** | Distributed noiseless cooperative game | Global Parity Invariant Code | Tail sacrifices to broadcast parity $\implies$ 99 survive with $100\%$ certainty |

---

## Module 5: Interactive Game Theory Simulator

```game-theory-interactive-demo
```
