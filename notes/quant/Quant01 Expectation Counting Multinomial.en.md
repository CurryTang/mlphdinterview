# Quant 01 · Expectation, counting, and recursion

Course location: this note → [[Quant02 Markov Chains Expected Time|02 Markov Chains]]

Direct calculation of joint distributions is often prohibitively complex when dealing with expectations of counting variables, joint products of frequencies, or steady-state features of dynamic processes.

This note introduces three structured dimension-reduction techniques:

- Indicator decomposition: bypassing dependencies using linearity of expectation.
- Falling factorial moments: handling joint expectations of frequency products in multinomial distributions.
- Recursion and state compression: identifying isomorphic subproblems of smaller size in random processes.

---

## 1 · Indicator variables and linear expectation

A counting variable $X$ can often be decomposed into a sum of 0/1 indicator variables:

$$
X = \sum_{i} I_i
$$

Due to the linearity of expectation, even if the $I_i$ are highly correlated, we have:

$$
\mathbb{E}[X] = \sum_{i} \mathbb{E}[I_i]
$$

Because $I_i$ is a binary 0/1 indicator, its expectation is simply the probability of the event:

$$
\mathbb{E}[I_i] = \mathbb{P}(I_i = 1)
$$

Thus, we can evaluate the overall expectation as a sum of probabilities:

$$
\mathbb{E}[X] = \sum_{i} \mathbb{P}(I_i = 1)
$$

This avoids the need to enumerate complex joint distributions.

However, when calculating the expectation of a product like $\mathbb{E}[X Y]$ or $\mathbb{E}[\prod X_i]$, the expectation is not multiplicative.

Unless the variables are mutually independent, $\mathbb{E}[X Y] \neq \mathbb{E}[X] \mathbb{E}[Y]$.

The expected product depends strictly on the joint structure between the variables.

---

## 2 · Multinomial distribution and falling factorial moments

A fair six-sided die is rolled independently 10 times.

For each face $i \in \{1,2,3,4,5,6\}$, let $N_i$ denote the number of times face $i$ appears in these 10 rolls.

Find the joint expectation of the product of all six frequencies:

$$
\mathbb{E}[N_1 N_2 N_3 N_4 N_5 N_6]
$$

### Why independent factorization fails

Each frequency $N_i$ follows a binomial distribution $N_i \sim \text{Binomial}(10, 1/6)$.

The marginal expectation is:

$$
\mathbb{E}[N_i] = 10 \times \frac16 = \frac53
$$

Multiplying these marginally yields $(5/3)^6 \approx 21.43$.

This is incorrect.

There is a strong negative correlation among the frequencies: the sum of all appearances is strictly constrained to $\sum N_i = 10$.

If $N_1$ is large, fewer rolls remain for the other faces.

Furthermore, if any face fails to appear even once, the entire product immediately collapses to zero.

### The falling factorial moment formula

The results of the 10 rolls follow a multinomial distribution:

$$
(N_1, \ldots, N_6) \sim \text{Multinomial}\left( 10;\, \frac16, \dots, \frac16 \right)
$$

For a multinomial distribution $(N_1, \dots, N_k) \sim \text{Multinomial}(n; p_1, \dots, p_k)$, let $(x)_a = x(x-1)\cdots(x-a+1)$ denote the falling factorial.

For any non-negative integers $a_1, \dots, a_k$, the falling factorial moment is:

$$
\mathbb{E}\left[ \prod_{j=1}^k (N_j)_{a_j} \right] = (n)_{a_1 + \dots + a_k} \prod_{j=1}^k p_j^{a_j}
$$

In this problem, the power of each frequency is 1.

Since $(N_i)_1 = N_i$, we have $a_1 = \dots = a_6 = 1$.

The sum of the orders is 6.

Substituting into the formula gives:

$$
\mathbb{E}[N_1 N_2 N_3 N_4 N_5 N_6] = (10)_6 \left( \frac16 \right)^6
$$

Expanding this calculation:

$$
\mathbb{E}[N_1 N_2 \dots N_6] = \frac{10 \times 9 \times 8 \times 7 \times 6 \times 5}{6^6}
$$

$$
\mathbb{E}[N_1 N_2 \dots N_6] = \frac{151{,}200}{46{,}656} = \frac{175}{54}
$$

The true expectation is roughly $3.24$, which is far less than $21.43$.

### Combinatorial insight using indicator variables

We can also understand this using an indicator variable expansion.

Define $X_{t, i} = \mathbf{1}\{\text{roll } t \text{ yields face } i\}$, where $t \in \{1, \dots, 10\}$.

$$
N_i = \sum_{t=1}^{10} X_{t, i}
$$

Multiplying these sums and applying linearity of expectation:

$$
\mathbb{E}[N_1 \dots N_6] = \sum_{t_1=1}^{10} \dots \sum_{t_6=1}^{10} \mathbb{E}[X_{t_1, 1} \dots X_{t_6, 6}]
$$

Within a single roll, outcomes are mutually exclusive.

If any indices repeat (e.g., $t_1 = t_2$), it means the same roll produced both face 1 and face 2, which has a probability of 0.

A product term can be 1 if and only if $t_1, \dots, t_6$ are all distinct.

The number of ways to orderedly select 6 distinct roll indices out of 10 is $(10)_6$.

For any specific sequence of indices, the probability of obtaining the exact specified faces is $(1/6)^6$.

Therefore:

$$
\mathbb{E}[N_1 \dots N_6] = (10)_6 \left( \frac16 \right)^6
$$

### Parameter variations

If the number of rolls $n < 6$:

At least one face will never appear. Thus, some $N_i = 0$.

The formula $(n)_6 = 0$ holds, and the expectation is strictly 0.

If the number of rolls $n = 6$:

Each face must appear exactly once. The expectation equals the probability of rolling a full permutation:

$$
\mathbb{E}[N_1 \dots N_6] = \frac{6!}{6^6} = \frac{5}{324}
$$

Moments with higher powers (e.g., $\mathbb{E}[N_1^2 N_2 \dots N_6]$):

Use algebraic identities to convert standard powers into falling factorials, such as $N_1^2 = (N_1)_2 + (N_1)_1$.

$$
\mathbb{E}[N_1^2 N_2 \dots N_6] = \mathbb{E}[(N_1)_2 N_2 \dots N_6] + \mathbb{E}[(N_1)_1 N_2 \dots N_6]
$$

Apply the formula to each term separately:

- First term: sum of orders is $2+1+1+1+1+1 = 7$, expectation is $(10)_7 (1/6)^7$.
- Second term: sum of orders is $1+1+1+1+1+1 = 6$, expectation is $(10)_6 (1/6)^6$.

---

## 3 · Prefix extrema and records

Consider $n$ hikers walking along a single-lane mountain trail in the same direction.

Their initial positions from front to back are numbered $1, \ldots, n$.

Each hiker's walking speed $V_i$ is drawn independently from the same continuous probability distribution.

The trail does not permit passing; when a faster hiker catches up to a slower one, they must join the group and walk at the leader's slower speed.

After a sufficiently long time, what is the expected number of distinct hiking groups remaining?

### Dimensional reduction to static features

Tracing the timeline of collisions is tedious. Instead, look at the final steady state.

In the limit of infinite time, person $i$ becomes the leader of an independent group if and only if their speed is strictly slower than everyone ahead of them.

$$
V_i < \min(V_1, V_2, \ldots, V_{i-1})
$$

This means $V_i$ is a strict prefix minimum (or record low) in the sequence $(V_1, \ldots, V_i)$.

If $V_i$ is slower than everyone ahead, they will never catch up to the groups in front.

Conversely, if someone ahead is slower, person $i$ will eventually catch them and merge.

```record-minimum-demo
```

### Expected number of groups

Define the indicator variable for person $i$ becoming a leader:

$$
I_i = \mathbf{1}\{V_i = \min(V_1, \ldots, V_i)\}
$$

The total number of surviving groups is $K_n = \sum_{i=1}^n I_i$.

Because the speeds $V_i$ are continuous i.i.d. random variables, the probability of any two being exactly equal is 0.

By exchangeability symmetry, the minimum among the first $i$ values $\{V_1, \dots, V_i\}$ is equally likely to be located at any specific position.

Therefore:

$$
\mathbb{E}[I_i] = \mathbb{P}(I_i = 1) = \frac{1}{i}
$$

The expected total number of groups is:

$$
\mathbb{E}[K_n] = \sum_{i=1}^n \frac{1}{i} = H_n
$$

$H_n$ is the $n$-th harmonic number. For large $n$:

$$
H_n \approx \ln n + \gamma
$$

Here $\gamma \approx 0.577$ is the Euler–Mascheroni constant.

### Rényi's theorem and variance

The variance of $K_n$ depends on the covariances between the indicator variables.

Rényi's independence theorem states that for a sequence of continuous i.i.d. variables, the prefix extremum indicators $I_1, \ldots, I_n$ are mutually independent.

Given independence, the variance is the sum of the individual variances:

$$
\operatorname{Var}(K_n) = \sum_{i=1}^n \operatorname{Var}(I_i) = \sum_{i=1}^n \frac{1}{i}\left( 1 - \frac{1}{i} \right) = H_n - H_n^{(2)}
$$

Where $H_n^{(2)} = \sum_{i=1}^n 1/i^2$.

As $n \to \infty$, $H_n^{(2)} \to \pi^2/6 \approx 1.645$.

### Isomorphic problems

The prefix extremum model appears in many scenarios with identical mathematical structures:

| Problem scenario | Prefix extremum definition | Result |
|---|---|---|
| Visible buildings in a line | Building height $H_i > \max(H_1, \dots, H_{i-1})$ | Expected visible count $= H_n$ |
| Global max updates | Record highs while scanning an array | Expected updates $= H_n$ |
| Disjoint cycles in a random permutation | Number of cycles in $\pi \in S_n$ | Expected cycles $= H_n$ |

---

## 4 · Recursion and state compression

Recursion is useful for problems where a current random choice leaves behind a smaller problem with the identical structure.

The key steps are: defining the target metric for size $n$, categorizing by the first step, identifying the remaining isomorphic subproblem, and writing the recurrence relation.

### The absent-minded passenger model

An airplane has $n$ seats, numbered $1$ to $n$, and $n$ passengers board in order.

Each passenger is supposed to sit in the seat with their own number.

If a passenger finds their assigned seat already occupied, they choose uniformly at random from all currently empty seats.

Passenger 1 is absent-minded and initially chooses a seat uniformly at random from all $n$ seats. All other passengers follow the standard rule.

What is the probability that passenger $n$ ultimately sits in seat $n$? ($n \ge 2$)

Define the target state:

$$
p_n = \mathbb{P}(\text{with } n \text{ passengers, passenger } n \text{ sits in seat } n)
$$

When $n=2$, passenger 1 chooses seat 1 or 2 with equal probability.

If they choose 1, passenger 2 succeeds. If they choose 2, passenger 2 fails. The boundary condition is $p_2 = 1/2$.

### Categorizing the first choice

Passenger 1 chooses each seat with probability $1/n$.

1. Choosing seat 1: All subsequent passengers can sit in their assigned seats. Passenger $n$ definitely succeeds, contributing $1/n \cdot 1$.
2. Choosing seat $n$: Passenger $n$'s seat is occupied, so they definitely fail, contributing $1/n \cdot 0$.
3. Choosing an intermediate seat $k$ ($2 \le k \le n-1$):
   Passengers $2, \dots, k-1$ sit normally.
   When passenger $k$ boards, they find their seat taken by passenger 1, and must choose randomly among the remaining empty seats.

The state at this point is:

```text
Passengers 2 to k-1: no conflict, all sat normally.
Passenger k: assigned seat taken, becomes the next random chooser.
Seats 2 to k-1: correctly occupied, no longer affect the process.
```

The historical process can be compressed away.

After removing the fixed passengers and seats, the passengers who still need to choose are $k, \ldots, n$.

The available empty seats are $1, k+1, \ldots, n$.

Treating passenger $k$ as a new "absent-minded passenger 1", this is perfectly equivalent to an isomorphic original problem of size $n-k+1$.

Under this condition, the success probability is $p_{n-k+1}$.

### Solving the recurrence

Using the law of total probability, expand $p_n$:

$$
p_n = \frac{1}{n} + \frac{1}{n}\sum_{k=2}^{n-1}p_{n-k+1}
$$

Let $m = n-k+1$, and change the summation index:

$$
n p_n = 1 + \sum_{m=2}^{n-1} p_m
$$

The same equation holds for a problem of size $n-1$:

$$
(n-1) p_{n-1} = 1 + \sum_{m=2}^{n-2} p_m
$$

Subtracting the two equations yields $n p_n - (n-1) p_{n-1} = p_{n-1}$, which simplifies to:

$$
p_n = p_{n-1}
$$

We already established the boundary condition:

$$
p_2 = \frac{1}{2}
$$

Since $p_n = p_{n-1}$, we can unfold the recurrence:

$$
p_n = p_{n-1} = \dots = p_2 = \frac{1}{2}
$$

By mathematical induction, we conclude that for all $n \ge 2$:

$$
p_n = \frac{1}{2}
$$

### Symmetry argument

When the conflict chain hits an intermediate seat, it merely transfers the random choice to a later passenger.

Throughout the entire chain, only seat 1 (terminates with success) and seat $n$ (terminates with failure) directly decide the final outcome.

During every random selection, as long as seats 1 and $n$ are still empty, their relative probabilities of being chosen are always perfectly equal.

Therefore, the conflict chain is equally likely to eventually land on seat 1 or seat $n$, giving a success probability of $1/2$.

---

## 5 · References

- William Feller, *An Introduction to Probability Theory and Its Applications, Volume 1*
- Sheldon Ross, *A First Course in Probability*
