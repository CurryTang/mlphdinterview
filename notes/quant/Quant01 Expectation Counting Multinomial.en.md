# Quant 1 · Expectation and Counting: Indicators, Records, and the Multinomial Distribution

This note opens the probability and expectation curriculum in quantitative interview preparation. In quantitative trading and research interviews (Jane Street, Citadel, Optiver, SIG, Two Sigma), a prominent archetype requires evaluating **expectations of dependent counting variables, joint products of frequencies, or steady-state properties of particle/agent collision processes**.

Rather than enumerating complex joint distributions or simulating collisions over time, master three fundamental dimension-reduction tools:

```text
Mental Model for Expectation & Counting Brainteasers:
1. Indicator Decomposition: Decompose a complicated counting variable X into binary indicators X = Σ I_i. Linearity of expectation E[X] = Σ E[I_i] = Σ P(I_i = 1) bypasses intricate mutual dependencies without needing independence.
2. Dynamic Collisions to Static Prefix Records: In single-lane merging (hikers, traffic jams) or streaming extrema, the surviving steady-state leaders map bijectively to prefix extrema (Record Lows / Highs). By exchangeability of continuous i.i.d. random variables, P(I_i = 1) = 1/i, and the total count converges to the Harmonic Number H_n.
3. Multinomial Falling Factorial Moments: For products of frequencies E[N_1 N_2 ... N_k] in multinomial trials, avoid assuming independence; apply the falling factorial moment formula E[Π (N_j)_{a_j}] = (n)_{Σ a_j} Π p_j^{a_j}, which counts ordered allocations of trials.
```

---

## Module 1: Multinomial Moments & Falling Factorials

### 1. Classic Interview Problem: Expected Product of Dice Frequencies

> **Problem Statement (Jane Street / Citadel / Optiver High-Frequency Question)**:
> 
> A fair six-sided die is rolled independently $10$ times. For each face $i \in \{1,2,3,4,5,6\}$, let $N_i$ denote the number of times face $i$ appears across these $10$ rolls.
> 
> **Find the joint expectation of the product of all six frequencies**:
> 
> $$
> \mathbb{E}[N_1 N_2 N_3 N_4 N_5 N_6]
> $$

---

### 2. Intuition Pitfall: Why You Cannot Factor into $\prod \mathbb{E}[N_i]$

Each individual count marginally follows a binomial distribution $N_i \sim \text{Binomial}(10, \tfrac16)$, giving marginal expectations:

$$
\mathbb{E}[N_i] = 10 \times \frac16 = \frac{5}{3}
$$

**Fatal Error**: Multiplying marginal expectations directly yields $\left( \frac{5}{3} \right)^6 \approx 21.43$. This is **completely incorrect**.

**Why Factorization Fails**:
1. **Fixed Sum Constraint (Negative Correlation)**: The sum is strictly constrained by $\sum_{i=1}^6 N_i = 10$. If face 1 appears frequently (e.g., $N_1 = 5$), only $5$ rolls remain for the other 5 faces combined, drastically reducing their counts. This inherent **negative covariance** suppresses the joint product far below the independent baseline.
2. **Zero-Product Annihilation**: If any face fails to appear at least once ($N_k = 0$), the entire product $N_1 \dots N_6$ collapses to zero.

---

### 3. Method 1: Multinomial Falling Factorial Moment Formula

The joint counts follow a **Multinomial Distribution**:

$$
(N_1, N_2, \ldots, N_6) \sim \text{Multinomial}\left( 10;\, \frac16, \frac16, \frac16, \frac16, \frac16, \frac16 \right)
$$

**Core Theorem (Multinomial Falling Factorial Moments)**:
Let $(N_1, \dots, N_k) \sim \text{Multinomial}(n; p_1, \dots, p_k)$. With $(x)_a = x(x-1)\cdots(x-a+1)$ denoting the falling factorial, for any non-negative integers $a_1, \dots, a_k$:

$$
\boxed{\mathbb{E}\left[ \prod_{j=1}^k (N_j)_{a_j} \right] = (n)_{a_1 + a_2 + \cdots + a_k} \prod_{j=1}^k p_j^{a_j}}
$$

**Application to the Dice Problem**:
Each face appears with power $1$. Since $(N_i)_1 = N_i$, we have $a_1 = \dots = a_6 = 1$, and total order $\sum a_j = 6$:

$$
\mathbb{E}[N_1 N_2 N_3 N_4 N_5 N_6] = (10)_6 \left( \frac16 \right)^6
$$

Expanding $(10)_6 = 10 \times 9 \times 8 \times 7 \times 6 \times 5 = 151{,}200$:

$$
\mathbb{E}[N_1 N_2 N_3 N_4 N_5 N_6] = \frac{151{,}200}{46{,}656} = \boxed{\frac{175}{54} \approx 3.2407}
$$

The true expected value is $\approx 3.24$, far below the naive $21.43$.

---

### 4. Method 2: Indicator Expansion & Combinatorial Selection

In an interview setting, the most intuitive and robust derivation uses **indicator decomposition**:

Let $X_{t, i} = \mathbf{1}\{\text{roll } t \text{ yields face } i\}$ for $t \in \{1,\dots,10\}$ and $i \in \{1,\dots,6\}$. Then:

$$
N_i = \sum_{t=1}^{10} X_{t, i}
$$

Multiplying all six summations and taking expectations:

$$
\mathbb{E}[N_1 N_2 \dots N_6] = \sum_{t_1=1}^{10} \sum_{t_2=1}^{10} \dots \sum_{t_6=1}^{10} \mathbb{E}[X_{t_1, 1} X_{t_2, 2} \dots X_{t_6, 6}]
$$

**Combinatorial Insight**:
- Within a single roll $t$, outcomes are mutually exclusive. If indices repeat (e.g., $t_1 = t_2$), roll $t_1$ cannot simultaneously show face 1 and face 2; the product is identically $0$.
- A term is non-zero ($= 1$) if and only if $t_1, t_2, t_3, t_4, t_5, t_6$ are **all distinct**.

**Counting & Probability**:
1. Number of distinct ordered selections of 6 trials out of 10 is:
   $$
   10 \times 9 \times 8 \times 7 \times 6 \times 5 = (10)_6
   $$
2. For any such distinct sequence $(t_1, \dots, t_6)$, the probability of these specific outcomes on these 6 independent trials is:
   $$
   \left( \frac16 \right)^6
   $$

Summing all non-zero terms gives:

$$
\mathbb{E}[N_1 N_2 \dots N_6] = (10)_6 \left( \frac16 \right)^6 = \frac{175}{54}
$$

---

### 5. Extensions and Variations

#### Case A: Fewer Rolls $n < 6$ or $n = 6$
- If $n < 6$ (e.g., 5 rolls): By the pigeonhole principle, not all 6 faces can appear; at least one $N_i = 0$, so $\mathbb{E}[N_1 \dots N_6] = 0$. In the formula, $(5)_6 = 0$.
- If $n = 6$: All 6 faces must appear exactly once. The expectation equals the probability of rolling a full permutation:
  $$
  \mathbb{E}[N_1 \dots N_6] = \mathbb{P}(N_1=1, \dots, N_6=1) = \frac{6!}{6^6} = \frac{720}{46656} = \frac{5}{324}
  $$

#### Case B: Higher Powers (e.g., $\mathbb{E}[N_1^2 N_2 N_3 N_4 N_5 N_6]$)
Convert powers to falling factorials: $N_1^2 = (N_1)_2 + (N_1)_1$.
$$
\mathbb{E}[N_1^2 N_2 \dots N_6] = \mathbb{E}[(N_1)_2 N_2 \dots N_6] + \mathbb{E}[(N_1)_1 N_2 \dots N_6] = (10)_7 \left( \frac16 \right)^7 + (10)_6 \left( \frac16 \right)^6
$$

---

## Module 2: Prefix Extremes (Record High/Low) & Harmonic Numbers

### 6. Classic Interview Problem: Single-File Hiking Merge

> **Problem Statement (Two Sigma / Akuna / SIG High-Frequency Question)**:
> 
> $n$ hikers walk along a narrow, infinite single-lane mountain trail in the same direction, starting at initial positions indexed $1, 2, \ldots, n$ from front to back.
> 
> Each hiker's walking speed $V_i$ is drawn independently from the same continuous probability distribution. The trail is too narrow to permit passing.
> 
> When a faster hiker catches up to a slower hiker/group ahead, they cannot overtake and must match the slower leader's speed at the rear of that group.
> 
> **Question**: After a sufficiently long time, what is the **expected number of distinct hiking groups** remaining?

---

### 7. Dimensional Reduction: Equivalence to Prefix Minima

Tracing chronological collisions is tedious and depends on starting distances. However, the asymptotic steady state depends strictly on relative speeds:

**Leader Criterion**:
In the infinite-time limit, hiker $i$ (at initial position $i$ from the front) remains the leader of an independent group **if and only if their speed is strictly slower than all hikers ahead**:

$$
V_i < \min(V_1, V_2, \ldots, V_{i-1})
$$

That is, $V_i$ is a **Prefix Minimum (Record Low)** in the sequence $(V_1, \ldots, V_i)$.

**Proof**:
1. **Sufficiency**: If $V_i < \min(V_1, \dots, V_{i-1})$, every group ahead travels strictly faster than hiker $i$. Hiker $i$ never catches anyone ahead, forming a persistent group.
2. **Necessity**: If there exists some $j < i$ with $V_j \le V_i$, given infinite time hiker $i$ will eventually merge into hiker $j$'s cluster.

```record-minimum-demo
```

---

### 8. Indicator Variables & Symmetry Argument: Why $1/i$

Define the indicator for hiker $i$ being a leader:

$$
I_i = \mathbf{1}\{V_i = \min(V_1, V_2, \ldots, V_i)\}
$$

The total number of groups is:

$$
K_n = \sum_{i=1}^n I_i
$$

**Symmetry**:
Since $V_1, \ldots, V_n$ are i.i.d. continuous random variables:
1. Continuous distributions ensure zero probability of ties ($\mathbb{P}(V_i = V_j) = 0$).
2. By exchangeability (permutation symmetry), among the first $i$ values $\{V_1, \dots, V_i\}$, the minimum is equally likely to be at any of the $i$ positions:

$$
\mathbb{P}(I_i = 1) = \mathbb{E}[I_i] = \frac{1}{i}
$$

---

### 9. Expected Number of Groups: The Harmonic Number $H_n$

Applying linearity of expectation:

$$
\mathbb{E}[K_n] = \sum_{i=1}^n \mathbb{E}[I_i] = \sum_{i=1}^n \frac{1}{i} = \boxed{H_n}
$$

where $H_n = 1 + \frac12 + \dots + \frac1n$ is the $n$-th **Harmonic Number**.

**Asymptotic Expansion**:
For large $n$, by Euler's asymptotic formula:

$$
H_n = \ln n + \gamma + \frac{1}{2n} - \frac{1}{12n^2} + O\left( \frac{1}{n^4} \right)
$$

where $\gamma \approx 0.57721566$ is the Euler–Mascheroni constant.

> **Intuition**: For $n = 1{,}000$ hikers, the expected number of surviving groups is merely $\ln(1000) + 0.577 \approx 6.91 + 0.58 \approx 7.49$. Multiplying group size by 10 increases the number of groups by only $\ln(10) \approx 2.3$.

---

### 10. Rényi's Independence Theorem & Variance

**Follow-Up Question**: What is the variance of the group count $K_n$? Are the indicator variables $I_1, \dots, I_n$ independent?

**Rényi's Theorem**:
For continuous i.i.d. random variables, the record indicators $I_1, I_2, \ldots, I_n$ are **mutually independent**.

**Reason**: The rank of $V_i$ among $\{V_1, \dots, V_i\}$ is uniformly distributed on $\{1, \dots, i\}$ and is statistically independent of the internal ranking of the first $i-1$ elements.

Because of independence, variances add linearly:

$$
\operatorname{Var}(K_n) = \sum_{i=1}^n \operatorname{Var}(I_i) = \sum_{i=1}^n \frac{1}{i}\left( 1 - \frac{1}{i} \right) = \sum_{i=1}^n \frac{1}{i} - \sum_{i=1}^n \frac{1}{i^2}
$$

$$
\boxed{\operatorname{Var}(K_n) = H_n - H_n^{(2)}}
$$

As $n \to \infty$, $\sum_{i=1}^\infty \frac{1}{i^2} = \frac{\pi^2}{6} \approx 1.6449$, so $\operatorname{Var}(K_n) = \ln n + \gamma - \frac{\pi^2}{6} + o(1)$.

---

## Module 3: Related Interview Archetypes

| Interview Problem | Latent Indicator Structure | Core Conclusion |
| :--- | :--- | :--- |
| **Visible Buildings** | Building $i$ is visible from ground zero $\iff H_i > \max(H_1, \dots, H_{i-1})$ (Record High) | Expected visible buildings $= H_n$ |
| **Running Max Updates** | Stream of $n$ numbers; count times the maximum variable updates | Expected updates $= H_n$ |
| **Secretary Candidate Hires** | Interview $n$ candidates; count how many times a candidate is "best-so-far" | Expected hires $= H_n$ |
| **Permutation Cycles** | Number of disjoint cycles in a uniformly random permutation $\pi \in S_n$ | Expected cycles $= H_n$; cycle count and record count are identically distributed! |
| **Random BST Depth** | Expected ancestors of a node in a random binary search tree | Decomposes into harmonic sums $O(\log n)$ |

---

## Module 4: Interview Traps & Structured Answer Strategy

### Common Pitfalls
1. **Pairwise Comparison Trap**: Assuming $V_i < V_{i-1}$ suffices gives $\mathbb{P} = 1/2$ and $n/2$ groups. Correct: a leader must be slower than **all** preceding hikers ($1/i$).
2. **Unnecessary Integration**: Attempting to integrate over continuous density $f(v)$ instead of exploiting permutation exchangeability.
3. **Harmonic Confusion**: Confusing the geometric waiting time origin of $H_n$ in the Coupon Collector problem with the symmetry origin of $1/i$ in Record problems.

### 1-Minute Structuring Template
```text
1. Dimensional Reduction: Hiker i leads a group iff V_i < min(V_1...V_{i-1}) (Prefix Minimum).
2. Indicator Definition: Let I_i = 1{V_i is prefix min}, total groups K_n = Σ I_i.
3. Symmetry: By continuous i.i.d. exchangeability, P(I_i = 1) = E[I_i] = 1/i.
4. Linearity: E[K_n] = Σ_{i=1}^n (1/i) = H_n ≈ ln(n) + γ.
5. Variance: By Rényi's theorem, I_i are independent, Var(K_n) = H_n - Σ(1/i^2) ≈ ln(n) + γ - π^2/6.
```

---

## Module 5: Practice Multiple Choice Quizzes

```quiz
title: Quick Quiz 1
question: For 10 rolls of a fair 6-sided die with face counts N_1 to N_6, why is E[N_1 N_2 ... N_6] NOT equal to (10/6)^6?
answer: B
A. Because the marginal expectation of each face is not 10/6
B. Because the sum is constrained to 10, creating negative correlation, and any face with count 0 zeroes the product
C. Because die rolls are continuous random variables
D. Because the multinomial distribution lacks a well-defined expectation
explanation: Counts come from the same 10 rolls and sum to 10. Large counts in one face suppress other counts. Any zero count collapses the entire product.
```

```quiz
title: Quick Quiz 2
question: In the multinomial falling factorial moment formula for (N_1...N_k) ~ Multinomial(n, p_1...p_k), what is E[N_1 N_2 ... N_k]?
answer: A
A. (n)_k * p_1 * p_2 * ... * p_k
B. n^k * p_1 * p_2 * ... * p_k
C. (n)_k / (p_1 * ... * p_k)
D. k! * n
explanation: With power 1 for each count, the falling factorial (N_i)_1 = N_i gives total order k, yielding E[N_1...N_k] = (n)_k * p_1 * ... * p_k.
```

```quiz
title: Quick Quiz 3
question: If a fair 6-sided die is rolled only 5 times (n = 5), what is E[N_1 N_2 N_3 N_4 N_5 N_6]?
answer: D
A. (5/6)^6
B. 5! / 6^6
C. 1
D. 0, because 5 rolls cannot cover all 6 distinct faces
explanation: By the pigeonhole principle, 5 rolls can produce at most 5 distinct faces, ensuring at least one N_i = 0, so the product is always zero. In the formula, (5)_6 = 5*4*3*2*1*0 = 0.
```

```quiz
title: Quick Quiz 4
question: In the single-file hiking model, what is the necessary and sufficient condition for hiker i to remain a group leader?
answer: C
A. V_i < V_{i-1} (slower than the immediate predecessor)
B. V_i > V_1 (faster than the first hiker)
C. V_i < min(V_1, ..., V_{i-1}) (speed is the prefix minimum)
D. V_i is the global minimum among all n hikers
explanation: If V_i is slower than all hikers ahead, they never catch any preceding group and maintain their own cluster. If anyone ahead is slower, hiker i eventually merges into that group.
```

```quiz
title: Quick Quiz 5
question: Why is the probability that hiker i's speed is the minimum among the first i hikers exactly equal to 1/i?
answer: B
A. Because speeds follow a normal distribution
B. Because speeds are continuous i.i.d. variables, making all i relative rank positions equally likely by permutation symmetry
C. It is an asymptotic approximation from the Law of Large Numbers
D. Because speeds decay with distance
explanation: Under continuous i.i.d. assumptions, all i! rank permutations of the first i elements are equally probable, so the minimum is at position i with probability (i-1)!/i! = 1/i.
```

```quiz
title: Quick Quiz 6
question: What is the expected number of groups remaining from n hikers?
answer: A
A. Harmonic number H_n = 1 + 1/2 + ... + 1/n ≈ ln(n) + γ
B. n / 2
C. sqrt(n)
D. log_2(n!)
explanation: By linearity of expectation on indicators: E[K_n] = Σ E[I_i] = Σ (1/i) = H_n.
```

```quiz
title: Quick Quiz 7
question: Are the prefix minimum indicator variables I_1, I_2, ..., I_n statistically independent?
answer: C
A. No, they are positively correlated
B. No, they are negatively correlated
C. Yes, they are mutually independent by Rényi's Theorem, allowing variances to add linearly: Var(K_n) = Σ Var(I_i)
D. Only asymptotically independent as n approaches infinity
explanation: By Rényi's Theorem, whether element i is the minimum among the first i elements is completely independent of the internal ranking among the first i-1 elements.
```

```quiz
title: Quick Quiz 8
question: What is the expected number of visible buildings when viewing n random-height skyscrapers from the ground?
answer: B
A. n / e
B. H_n ≈ ln(n) + γ
C. n / 2
D. 2 * sqrt(n)
explanation: Building i is visible iff its height is a prefix maximum (Record High), which occurs with probability 1/i. Summing gives H_n.
```
