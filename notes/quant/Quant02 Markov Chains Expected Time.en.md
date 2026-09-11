# Quant 02 · Markov chains
Course: [[Quant01 Expectation Counting Multinomial|01 Expectation]] → This note → [[Quant03 Continuous Distribution Geometry Transform|03 Continuous Distribution]]


## 1 · Transition probability and first-step analysis

A discrete-time Markov chain consists of a state space and transition probabilities. Its core is the Markov property: the next step depends only on the current state, not on earlier history.


$$

  P(X_{n+1}=j \mid X_n=i, X_{n-1}, \ldots, X_0) 
  = P(X_{n+1}=j \mid X_n=i)


$$


If the state space is finite, the transition probabilities form a matrix:


$$

  P_{ij} = P(X_{n+1}=j \mid X_n=i)


$$


The Chapman-Kolmogorov equation splits an $m+n$-step transition into two legs: $m$ steps to some intermediate state $k$, then $n$ steps from $k$:


$$

  P_{ij}^{(m+n)} = \sum_k P_{ik}^{(m)} P_{kj}^{(n)}


$$


In matrix form, the $n$-step transition matrix is the $n$-th power of the transition matrix:


$$

  P^{(n)} = P^n


$$


### First-step analysis

For computing expected times, the primary tool is first-step analysis.

Let $E_i$ be the expected number of steps to reach a target state from state $i$. If $i$ is already the target state:


$$

  E_i = 0


$$


Otherwise, taking one step consumes 1 unit of time, and the process continues from the new state:


$$

  E_i = 1 + \sum_j P_{ij} E_j


$$


The intuition: total time equals the time for the first step plus the weighted average of the expected time from the next state.

**Example: Two-state weather**

The weather is Sunny (S) or Rainy (R). The transition matrix is:

```text
S -> S: 0.8
S -> R: 0.2
R -> S: 0.4
R -> R: 0.6
```

Find the expected number of days to reach R starting from S. The target is R, so:


$$

  E_R = 0


$$


From S:


$$

  E_S = 1 + 0.8 E_S + 0.2 E_R


$$


Substituting $E_R = 0$:


$$

  E_S = 1 + 0.8 E_S


$$


Solving gives $0.2 E_S = 1$, so:


$$

  E_S = 5


$$


---


## 2 · Hitting time linear systems and absorbing states

A state $i$ is an absorbing state if, once entered, the process never leaves: $P_{ii} = 1$. If the process eventually enters an absorbing state with probability 1 from any non-absorbing state, it is an absorbing chain.

Absorption probability and expected time to absorption can both be written as linear systems using first-step analysis:

| Problem | Variable | Recurrence | Boundary Condition |
|---|---|---|---|
| Absorption probability | $h_i$ | $h_i = \sum_j P_{ij} h_j$ | Target absorbing state $h_i = 1$, others $h_i = 0$ |
| Expected time | $E_i$ | $E_i = 1 + \sum_j P_{ij} E_j$ | Any absorbing state $E_i = 0$ |

As long as absorption is certain, these systems have unique solutions.


### Gambler's ruin

A gambler has $i$ dollars. Each round, they win 1 dollar with probability $p$ and lose 1 dollar with probability $q = 1 - p$. The game ends at 0 or $N$. This is a random walk with two absorbing barriers.

**Absorption probability**

Let $h_i$ be the probability of reaching $N$ from $i$. Boundaries are:


$$

  h_0 = 0 \\

  h_N = 1


$$


Intermediate states:


$$

  h_i = p h_{i+1} + q h_{i-1}


$$


When $p = q = 1/2$, the solution is linear:


$$

  h_i = \frac{i}{N}


$$


**Expected time to absorption**

Let $E_i$ be the expected steps to absorption from $i$. Boundaries are:


$$

  E_0 = 0 \\

  E_N = 0


$$


When $p = q = 1/2$, intermediate states satisfy:


$$

  E_i = 1 + \frac{1}{2} E_{i+1} + \frac{1}{2} E_{i-1}


$$


Rearranged into a second-order recurrence:


$$

  E_{i+1} - 2E_i + E_{i-1} = -2


$$


The general solution is $A+Bi$ plus a particular solution $-i^2$, yielding:


$$

  E_i = -i^2 + A + Bi


$$


Substituting $E_0=0$ gives $A=0$; $E_N=0$ gives $B=N$. The result is:


$$

  E_i = i(N-i)


$$


### Random walk with a reflecting barrier

A reflecting barrier deterministically pushes the particle back into the interior, while an absorbing barrier traps it. Both fit the first-step template.

State space $\{0, 1, 2, 3\}$. State 0 deterministically transitions to 1 (reflecting). States 1 and 2 are a symmetric random walk. State 3 is absorbing. Find the expected steps to absorption from 0, $E_0$:

Equations are:


$$

\begin{aligned}
  E_0 &= 1 + E_1 \\

  E_1 &= 1 + \frac{1}{2} E_0 + \frac{1}{2} E_2 \\

  E_2 &= 1 + \frac{1}{2} E_1 + \frac{1}{2} E_3
\end{aligned}


$$


Substituting $E_3 = 0$ and $E_0 = 1 + E_1$ into the second equation gives:


$$

  E_1 = 1 + \frac{1}{2}(1+E_1) + \frac{1}{2} E_2


$$


Simplifying:


$$

  E_1 = 3 + E_2


$$


Substituting into the third equation gives:


$$

  E_2 = 1 + \frac{1}{2}(3+E_2)


$$


Solving:


$$

  E_2 = 5


$$


Back-substituting yields $E_1 = 8, E_0 = 9$.

---


## 3 · State compression (Lumpability)

According to the Kemeny-Snell criterion: if states are partitioned into equivalence classes such that any two states in the same class have the same total transition probability into any other class, the compressed process remains a Markov chain.


### Random walk on a complete graph

An ant randomly walks on a regular tetrahedron (complete graph $K_4$). Since any two vertices are adjacent, the state can be defined as "number of distinct vertices visited, $i$".

The 3 possible next steps always consist of $i-1$ already-visited vertices and $4-i$ new ones. Transition probabilities depend only on $i$.

The compressed chain has 4 states:

```text
S1(1) --1--> S2(2) --2/3--> S3(3) --1/3--> S4(4)
```

Let $T_i$ be the steps to go from $i$ visited to $i+1$ visited, which is geometric with success probability $(4-i)/3$. The total expected steps are:


$$

  \mathbb E[T] = \sum_{i=1}^3 \frac{3}{4-i} = 1 + 1.5 + 3 = 5.5


$$


### Random walk on a cube

On a cube ($Q_3$), "number of vertices visited" is not lumpable. But the cube is vertex-transitive, so the state can be compressed to "graph distance from the start, $d \in \{0, 1, 2, 3\}$".

| Current distance $d$ | Split of 3 neighbors | Transition probabilities |
|---|---|---|
| 0 (start) | all 3 at $d=1$ | $P(0\to 1)=1$ |
| 1 | 1 at $d=0$, 2 at $d=2$ | $P(1\to 0)=1/3,\ P(1\to 2)=2/3$ |
| 2 | 2 at $d=1$, 1 at $d=3$ | $P(2\to 1)=2/3,\ P(2\to 3)=1/3$ |
| 3 (antipode) | all 3 at $d=2$ | $P(3\to 2)=1$ |

Let $h_d$ be the expected steps from distance $d$ to $d=3$:


$$

  h_3=0,\quad h_2 = 1+\frac{2}{3} h_1,\quad h_1 = 1+\frac{1}{3} h_0+\frac{2}{3} h_2,\quad h_0=1+h_1


$$


Solving yields $h_2=7, h_1=9, h_0=10$.


### Five-light-bulb toggling

There are 5 lights, initially all off. Each second, one light is chosen uniformly and toggled. Find the expected time to first return to the "all off" state.

Compress the state to "number of lights currently on, $k$". The probability of choosing an on light and turning it off is $k/5$; choosing an off light and turning it on is $(5-k)/5$.

```text
S0 --1--> S1 --4/5--> S2 --3/5--> S3 --2/5--> S4 --1/5--> S5
       <--1/5--    <--2/5--    <--3/5--    <--4/5--    <--1--
```

Let $E_k$ be the expected time to reach 0 from $k$ lights on. Boundary condition $E_0 = 0$.


$$

\begin{aligned}
  E_1 &= 1 + \frac{1}{5}E_0 + \frac{4}{5}E_2 \\

  E_2 &= 1 + \frac{2}{5}E_1 + \frac{3}{5}E_3 \\

  E_3 &= 1 + \frac{3}{5}E_2 + \frac{2}{5}E_4 \\

  E_4 &= 1 + \frac{4}{5}E_3 + \frac{1}{5}E_5 \\

  E_5 &= 1 + E_4
\end{aligned}


$$


Solving gives:


$$

  E_1 = 31


$$


Since the system starts all off, the first second must turn one light on. The total expected time to return is:


$$

  1 + E_1 = 32


$$


---


## 4 · Stationary return time

A stationary distribution is a stable distribution $\pi$ satisfying $\pi P = \pi$ and $\sum \pi_i = 1$. For a finite, irreducible chain, $P^n$ converges to $\pi$.

The expected time to return to state $i$ for the first time, starting from $i$ (mean recurrence time), is:


$$

  \mathbb{E}_i[T_i^+] = \frac{1}{\pi_i}


$$


This provides a shortcut for return-time problems. Note it computes return time ($i \to i$), not first passage time ($i \to j$).

For the five-light-bulb problem, the full state space is the on/off configuration of every light, e.g., `00000`, `00001`, giving $2^5 = 32$ states. Toggling one bit per step is equivalent to a random walk on a 5-dimensional hypercube.

Since the walk is symmetric, the stationary distribution is uniform:


$$

  \pi(x) = \frac{1}{32}


$$


The stationary probability of the all-off state `00000` is $1/32$. By the mean recurrence time formula, the expected time to return to `00000` is:


$$

  \mathbb{E}_{00000}[T_{00000}^+] = \frac{1}{1/32} = 32


$$


This matches the result from state compression.

---


## 一手资料

- Zhou, *A Practical Guide to Quantitative Finance Interviews*
- Crack, *Heard on the Street*
