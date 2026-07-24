# Quant 1 · Expectation and Counting: Indicators, Records, and the Multinomial Distribution

This note opens the probability section of Quant. It reviews three tools that often appear together:

```text
1. Split a counting variable into indicators, then use linearity of expectation.
2. Recognize record highs and record lows in a random ordering.
3. For products of multinomial counts, use falling factorial moments.
```

The first half starts with dice counts to explain multinomial moments. The second half uses a "single-file hiking group" to explain record minima. The two problems look different, but both solutions begin by decomposing the target count into a sequence of 0/1 events.

---

## 1. Problem

A fair six-sided die is rolled 10 times. For each face $i \in \{1,2,\ldots,6\}$, let $N_i$ denote the number of times face $i$ appears.

Find:

$$
\mathbb{E}[N_1N_2N_3N_4N_5N_6]
$$

---

## 2. Core idea

The random variable in this problem is not a single $N_i$, but the product of six counts. Because $N_1,\ldots,N_6$ come from the same 10 rolls, they are not independent:

```text
If one face appears more often,
the total number of appearances available to the other faces decreases.
```

Therefore, we cannot write:

$$
\mathbb{E}[N_1]\mathbb{E}[N_2]\cdots\mathbb{E}[N_6]
$$

The right tool is the factorial moment of the multinomial distribution, or equivalently, a combinatorial argument using an indicator expansion.

---

## 3. Method 1: Multinomial falling factorial moment

After 10 rolls of a fair die:

$$
(N_1,\ldots,N_6) \sim \text{Multinomial}\left(10;\frac16,\ldots,\frac16\right)
$$

The multinomial distribution has an important identity:

$$
\mathbb{E}\left[\prod_{j=1}^k (N_j)_{a_j}\right]
=
(n)_{a_1+\cdots+a_k}\prod_{j=1}^k p_j^{a_j}
$$

where:

$$
(x)_a = x(x-1)\cdots(x-a+1)
$$

is the falling factorial.

Here, each face is selected only once, so $a_1=\cdots=a_6=1$. Since $(N_i)_1=N_i$:

$$
\mathbb{E}[N_1N_2N_3N_4N_5N_6]
=
(10)_6\left(\frac16\right)^6
$$

Expanding:

$$
(10)_6 = 10\cdot 9\cdot 8\cdot 7\cdot 6\cdot 5
$$

Therefore, the answer is:

$$
\boxed{\frac{10\cdot9\cdot8\cdot7\cdot6\cdot5}{6^6}}
$$

It can also be written as:

$$
\boxed{\frac{151200}{46656}=\frac{175}{54}}
$$

---

## 4. Method 2: Indicator expansion

This method is better suited to explaining why the formula works in an interview.

Write $N_i$ as a sum of indicators. Let the indicator for whether roll $t$ produces face $i$ be:

$$
X_{t,i} =
\mathbf{1}\{\text{roll }t\text{ produces }i\}
$$

Then:

$$
N_i = \sum_{t=1}^{10} X_{t,i}
$$

Therefore:

$$
N_1N_2N_3N_4N_5N_6
=
\sum_{t_1,\ldots,t_6}
X_{t_1,1}X_{t_2,2}X_{t_3,3}X_{t_4,4}X_{t_5,5}X_{t_6,6}
$$

The key point is that if two $t$ values are equal, such as $t_1=t_2$, the same roll cannot produce both face 1 and face 2, so that term must be 0.

A term can be nonzero only when:

```text
t_1, t_2, t_3, t_4, t_5, t_6
are all distinct
```

How many ordered selections of distinct indices are there?

$$
10\cdot9\cdot8\cdot7\cdot6\cdot5 = (10)_6
$$

For each selection, the probability that the six specified rolls produce $1,2,3,4,5,6$, respectively, is:

$$
\left(\frac16\right)^6
$$

Therefore:

$$
\mathbb{E}[N_1N_2N_3N_4N_5N_6]
=
(10)_6\left(\frac16\right)^6
=
\frac{175}{54}
$$

---

## 5. Why we cannot multiply expectations

Each $N_i$ has expectation:

$$
\mathbb{E}[N_i] = \frac{10}{6}
$$

But this does not give:

$$
\left(\frac{10}{6}\right)^6
$$

The reason is that the $N_i$ are negatively correlated. The total number of rolls is fixed at 10:

$$
N_1+\cdots+N_6=10
$$

If $N_1$ is large, less capacity remains for the other five counts. The expectation of the product must account for these dependencies.

---

## 6. Summary of the first problem

For a multinomial distribution, a product of ordinary counts can often be converted into a falling factorial moment:

$$
\mathbb{E}[N_1N_2\cdots N_k]
=
(n)_k p_1p_2\cdots p_k
$$

provided each count appears only once. The intuition is:

```text
First select k distinct trials,
then require them to fall into the specified k categories, respectively.
```

---

## 7. Second problem: A single-file hiking group

$n$ hikers travel in the same direction on an infinitely long, narrow mountain path. From front to back, their starting positions are numbered $1,2,\ldots,n$. Each person's speed is independently drawn from the same continuous distribution, and no passing is allowed.

If a faster hiker behind catches a slower group ahead, the hiker joins the back of that group and thereafter moves at the group's speed. After enough time has passed, what is the expected number of groups remaining?

The continuity assumption rules out equal speeds. The answer does not depend on whether speeds follow a uniform, exponential, lognormal, or any other continuous distribution.

---

## 8. Reduce the dynamic process to a static condition

Let $V_i$ be the speed of hiker $i$, still ordered from front to back. Hiker $i$ ultimately leads a group if and only if:

$$
V_i < \min(V_1,\ldots,V_{i-1})
$$

In other words, the hiker's speed is a new low among all speeds seen so far, called a **prefix minimum** or **record low**.

- If $V_i$ is slower than everyone ahead, the hiker can never catch any group ahead and therefore remains the leader of a new group.
- If someone ahead is already slower, the hiker eventually catches that slower group. The number of intermediate mergers does not affect the final result.

Distance and catch-up times therefore drop out of the problem, leaving only the relative ranking of speeds.

```record-minimum-demo
```

The speeds in the figure are just one particular sample. It may produce 4 groups, but the problem asks for the average over all random orderings.

---

## 9. Indicators and symmetry

Let:

$$
I_i
=
\mathbf{1}\{V_i \text{ is the minimum among }V_1,\ldots,V_i\}
$$

The final number of groups is:

$$
K_n=\sum_{i=1}^n I_i
$$

Consider the first $i$ speeds. Because they are i.i.d. with no ties, the minimum is equally likely to occur in any of the $i$ positions. Therefore:

$$
\mathbb{P}(I_i=1)=\frac{1}{i}
$$

By linearity of expectation:

$$
\mathbb{E}[K_n]
=
\sum_{i=1}^n \mathbb{E}[I_i]
=
\sum_{i=1}^n \frac1i
=
\boxed{H_n}
$$

where $H_n$ is the $n$th harmonic number. For large $n$:

$$
H_n
=
\log n+\gamma+\frac{1}{2n}+O(n^{-2})
$$

Thus, multiplying the number of hikers by ten does not multiply the number of groups by ten. It increases the expected number of groups by only about $\log 10$.

---

## 10. Common wrong turns in interviews

### Comparing only adjacent hikers

Writing $\mathbb{P}(V_i<V_{i-1})=1/2$ gives roughly $n/2$ groups, but a leader must be slower than **everyone** ahead, not just the immediately preceding hiker.

### Trying to simulate every catch-up

Initial distances change who catches whom first, but not the set of leaders after infinite time. A static characterization of the final state is usually simpler than writing a recurrence for every collision.

### Studying the distribution's density first

This problem uses only relative ranks. The i.i.d. assumption, continuity, and exchangeability already give $1/i$, so no integral is needed.

### Forgetting the model boundaries

If the path is finite, the observation time is finite, passing is allowed, or speeds have many ties, the prefix-minimum conclusion must be checked again.

---

## 11. Related interview problems

| Problem type | Hidden indicator | Result or connection |
|---|---|---|
| Number of randomly sized buildings visible from the left | Building $i$ is taller than all preceding buildings | Expectation is $H_n$; replace record low with record high |
| Replace an employee whenever a historically best candidate appears | Candidate $i$ is the best so far | Expected number of replacements is $H_n$ |
| Number of updates to the running maximum while scanning a random array | Item $i$ sets a new prefix maximum | Expectation is $H_n$ |
| Number of cycles in a random permutation | Each cycle contributes one representative element | Expectation is also $H_n$; the cycle count and record count even have the same distribution |
| Depth of a node in a random BST | Which interval elements become its ancestors | The decomposition produces two harmonic sums |
| Secretary problem | Only record candidates are worth hiring | Records still apply, but the objective becomes maximizing the probability of selecting the global best candidate |

The coupon collector problem also contains $H_n$, but for a different reason: it comes from the decreasing number of categories that remain uncollected, not from prefix records. When a harmonic number appears, explain where the denominator $1/i$ comes from.

---

## 12. Possible follow-up questions

### Probability of exactly $k$ groups

The record count has the same distribution as the number of cycles in a random permutation. Therefore:

$$
\mathbb{P}(K_n=k)
=
\frac{\left[{n\atop k}\right]}{n!}
$$

$\left[{n\atop k}\right]$ is an unsigned Stirling number of the first kind.

### Variance

For continuous i.i.d. samples, the record indicators are mutually independent. Therefore:

$$
\operatorname{Var}(K_n)
=
\sum_{i=1}^n \frac1i\left(1-\frac1i\right)
=
H_n-H_n^{(2)}
$$

where $H_n^{(2)}=\sum_{i=1}^n 1/i^2$.

---

## 13. One-minute answer sequence

```text
Number the hikers from front to back
→ final leaders are exactly the prefix minima
→ K_n = Σ I_i
→ position i is a prefix minimum with probability 1/i
→ E[K_n] = H_n ≈ log n + γ
```

This problem does not require a recurrence for the pursuit process. The only difficult step is recognizing that what looks like continuous merging is actually a count of records in a random sequence.
