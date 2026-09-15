# Quant 12 · Number Theory and Invariants

---

## 1 · Pebble Piles Division and Prime Invariance

### Problem Statement
You are given three piles of pebbles containing $5$, $49$, and $51$ pebbles, respectively. You are allowed to repeatedly perform two types of operations:
1. **Operation (a) Merge**: Choose any two piles and merge them into a single pile ($a, b \to a + b$);
2. **Operation (b) Split**: Choose any pile containing an even number of pebbles and divide it into two equal piles ($2k \to k, k$).

**Question**: Is it possible to perform a finite sequence of these operations to achieve a total of $105$ piles, each containing exactly $1$ pebble?

---

### Step-by-Step Derivation: The Invariant Principle

The key insight lies in identifying an **absorbing property of the state space**: once the system enters a specific algebraic divisibility state, all future transitions remain strictly trapped within it.

#### Step 1: Initial Move Constraints and the First Merge
Observe the three starting pile sizes:
$$x_1 = 5, \quad x_2 = 49, \quad x_3 = 51$$
The total pebble count is $5 + 49 + 51 = 105$.

Since $5, 49, 51$ are all **odd integers**, none of the piles can be split via Operation (b).  
Thus, the very first move is forced: **we must merge two of the piles (Operation a)**.

There are exactly 3 possible pairs to merge. Let us examine the resulting configuration in each case:

1. **Merge $5$ and $49$**:
   - The remaining piles have sizes: $\{54, 51\}$
   - Prime factorizations:
     $$54 = 3 \times 18, \quad 51 = 3 \times 17$$
   - **Both pile sizes are divisible by 3 ($3 \mid 54$ and $3 \mid 51$).**

2. **Merge $49$ and $51$**:
   - The remaining piles have sizes: $\{100, 5\}$
   - Prime factorizations:
     $$100 = 5 \times 20, \quad 5 = 5 \times 1$$
   - **Both pile sizes are divisible by 5 ($5 \mid 100$ and $5 \mid 5$).**

3. **Merge $5$ and $51$**:
   - The remaining piles have sizes: $\{56, 49\}$
   - Prime factorizations:
     $$56 = 7 \times 8, \quad 49 = 7 \times 7$$
   - **Both pile sizes are divisible by 7 ($7 \mid 56$ and $7 \mid 49$).**

**Core Observation from Step 1**:  
No matter which merge is chosen on Turn 1, every pile on the board immediately becomes a multiple of a chosen **odd prime $p \in \{3, 5, 7\}$**.

---

#### Step 2: Algebraic Closure of Divisibility (The Invariant)
Suppose that after the initial merge, all pile sizes on the board are multiples of some odd prime $p \in \{3, 5, 7\}$. We now prove that neither permitted operation can ever break this property:

1. **Applying Operation (a) (Merging two piles)**:
   Suppose two piles of sizes $A$ and $B$ satisfy $p \mid A$ and $p \mid B$. By the linearity of divisibility:
   $$A = p \cdot u, \quad B = p \cdot v \implies A + B = p \cdot (u + v)$$
   The merged pile $A + B$ **remains a multiple of $p$**.

2. **Applying Operation (b) (Splitting an even pile)**:
   Suppose an even pile of size $2k$ satisfies $p \mid 2k$.
   Because $p \in \{3, 5, 7\}$ is an **odd prime**, $p$ is coprime to $2$: $\gcd(p, 2) = 1$.
   By **Euclid's Lemma**, if a prime $p$ divides a product $2k$ and $p \nmid 2$, then:
   $$p \mid k$$
   Therefore, each of the resulting two sub-piles of size $k$ **is also strictly divisible by $p$**!

**The Invariant Theorem**:  
If all piles in the system belong to the ideal $p\mathbb{Z}$ (multiples of prime $p$), any subsequent state reached via Operations (a) or (b) will consist entirely of piles in $p\mathbb{Z}$. The set of states whose pile sizes are all multiples of $p$ is **closed and absorbing**.

---

#### Step 3: Reachability and Final Conclusion
The stated goal is:
> To reach a state with $105$ piles, each containing exactly $1$ pebble.

- Reaching this terminal state requires producing **at least one pile of size $1$** at some step;
- However, our derivation shows:
  - Turn 1 unconditionally forces all pile sizes into $3\mathbb{Z}$, $5\mathbb{Z}$, or $7\mathbb{Z}$;
  - Step 2 guarantees that no subsequent operation can ever escape that ideal;
  - Every pebble pile $S$ on the board must satisfy $S \ge p$ and $p \mid S$ for some $p \in \{3, 5, 7\}$.
- Clearly, the integer $1$ is not divisible by $3$, $5$, or $7$:
  $$1 \notin 3\mathbb{Z}, \quad 1 \notin 5\mathbb{Z}, \quad 1 \notin 7\mathbb{Z}$$

Therefore, producing a pile of size $1$ is mathematically impossible.

$$\boxed{\text{Conclusion: Impossible. It is impossible to achieve 105 piles of 1 pebble.}}$$
