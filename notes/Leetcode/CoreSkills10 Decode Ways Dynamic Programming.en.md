# Dynamic Programming: From Recurrence to Space Optimization

## Module 1: Core DP Mental Model & 6 Skeletons Taxonomy

### 1. Interview Goal & Mindset Refactoring

Dynamic Programming (DP) is never about "memorizing templates." Its fundamental essence is: **decomposing a complex, large-scale decision problem into overlapping isomorphic subproblems, and computing/caching answers along their topological dependency order**.

A robust, reusable 3-step DP formulation for whiteboard interviews consists of:

```text
Step 1: Mathematically Define State ➔ Clarify the physical meaning of dp[...] & Base Cases
Step 2: Enumerate Decision Branches (Transition) ➔ Formulate recurrence & determine Iteration Order
Step 3: Analyze Dependency Radius (Space Optimization) ➔ Apply rolling variables / array space reduction
```

> [!IMPORTANT]
> **Golden Interview Rule**: In whiteboard coding interviews, clearly communicating the State Definition, Base Cases, and Transitions to the interviewer *before* writing code is far more critical than jumping straight to implementation. Once the recurrence is rigorous, writing code is mere mechanical translation.

---

### 2. Identifying DP Signals

Consider DP whenever a problem exhibits the following signals:

- **Optimization/Counting Goal**: Asks for "number of ways / minimum cost / maximum profit / feasibility (True/False)".
- **Staged Decisions**: Progresses in stages (e.g., prefix of `i` characters, first `i` items, day `i`, interval `[i, j]`).
- **Overlapping Subproblems**: Naive recursive search repeatedly explores identical subproblems with identical arguments.
- **Optimal Substructure & No Aftereffect (Markov Property)**: The optimal solution to the current state depends strictly on optimal solutions to smaller subproblems, completely independent of the historical path taken to achieve them.

The four pillars of Dynamic Programming:

$$\text{DP Formula} = \mathbf{State} + \mathbf{Transition} + \mathbf{Base\ Case} + \mathbf{Iteration\ Order}$$

---

### 3. Deep Dive into the 3-Step Workflow

#### Step 1: Formulate Recurrence (State & Transition)
Before writing any loops, define `dp[...]` with one unambiguous sentence:
- `dp[i]`: Answer for prefix `arr[:i]` or suffix starting at index `i`.
- `dp[i][j]`: Answer for double sequence prefixes `(i, j)` or closed interval `[i, j]`.
- `dp[i][state]`: Answer at stage `i` under a discrete named state (e.g., holding stock / cooldown).

Combine previous sub-states according to the problem's objective:
- **Counting (Ways)**: $\text{dp}[i] = \sum \text{dp}[\text{prev}]$
- **Optimization (Min/Max)**: $\text{dp}[i] = \min / \max(\text{dp}[\text{prev}] + \text{cost})$
- **Feasibility (Boolean)**: $\text{dp}[i] = \bigvee \text{dp}[\text{prev}]$

#### Step 2: Determine Iteration Order from Dependencies
Loop directions are dictated solely by **topological dependencies**:
- If $dp[i]$ depends on $dp[i-1]:$ Traverse forward ($i: 1 \to n$).
- If $dp[i]$ depends on $dp[i+1]:$ Traverse backward ($i: n-1 \to 0$).
- If $dp[i][j]$ depends on $dp[i+1][j-1]$ (Interval DP): Traverse by increasing length ($L: 1 \to n$) or bottom-up ($i: n-1 \to 0$).

#### Step 3: Space Optimization
- Depends only on $dp[i-1]:$ Reduce to a single variable ($O(1)$ space).
- Depends on $dp[i-1], dp[i-2]:$ Reduce to two rolling variables (e.g., Fibonacci / House Robber / Decode Ways).
- 2D table $dp[i][j]$ depends only on row $dp[i-1][\dots]:$ Reduce to a 1D rolling array ($O(nm) \to O(m)$).

---

### 4. 6 DP Skeletons Taxonomy

```text
What is the structural dimension of the state?
│
├── 1. Single index i (prefix / suffix / linear progression)
│     └── Skeleton 1: 1D Linear DP
│         Key problems: Climbing Stairs, Min Cost Climbing Stairs, House Robber I & II,
│                       Decode Ways, Word Break, LIS, Maximum Product Subarray
│
├── 2. Two prefix pointers (i, j) across two sequences
│     └── Skeleton 2: Two Sequences & String Matching DP
│         Key problems: Longest Common Subsequence (LCS), Edit Distance,
│                       Interleaving String, Distinct Subsequences, Regular Expression Matching
│
├── 3. Closed interval [i, j] (substring / peeling onion / length expansion)
│     └── Skeleton 3: Interval DP
│         Key problems: Longest Palindromic Substring, Palindromic Substrings, Burst Balloons
│
├── 4. Items + Capacity (0/1 vs Complete, Combinations vs Permutations)
│     └── Skeleton 4: Knapsack Family
│         Key problems: Partition Equal Subset Sum (0/1 Feasibility), Target Sum (0/1 Count),
│                       Coin Change (Complete Min Coins), Coin Change II (Complete Combinations)
│
├── 5. 2D grid coordinates (r, c)
│     └── Skeleton 5: Grid & DAG DP
│         Key problems: Unique Paths (deterministic sweep),
│                       Longest Increasing Path in a Matrix (arbitrary directions ➔ Memoized DFS)
│
└── 6. Stage + discrete named states (holding / sold cooldown / free rest)
      └── Skeleton 6: State Machine DP
          Key problems: Best Time to Buy and Sell Stock with Cooldown
```

---

## Module 2: Skeleton 1 · 1D Linear DP & Prefix States

1D Linear DP represents the foundational cornerstone of dynamic programming, where states are defined over prefixes or suffixes of an array or string.

### 1. Deep Dive: Decode Ways

**Problem Statement**: A message containing letters from `A-Z` is encoded as `'A' -> 1, 'B' -> 2, ..., 'Z' -> 26`. Given a digit string `s`, compute the total number of valid decoding configurations.

#### Recurrence & Boundary Design
Define $dp[i]$ as the number of decoding ways for suffix string $s[i:]$.
- **Base Case**: $dp[n] = 1$ (empty suffix represents 1 valid completed decoding path).
- **Leading `'0'` Trap**: If $s[i] == '0'$, digit `'0'` cannot map to any letter on its own and leading zeros are invalid $\implies dp[i] = 0$.
- **1-Digit Choice**: Single digit $s[i] \in ['1'..'9'] \implies$ contributes $dp[i+1]$.
- **2-Digit Choice**: Two digits $s[i..i+1] \in [10..26] \implies$ additionally contributes $dp[i+2]$.

$$dp[i] = \begin{cases} 0 & s[i] = '0' \\ dp[i+1] + [10 \le s[i..i+1] \le 26] \cdot dp[i+2] & s[i] \neq '0' \end{cases}$$

#### Full Array to Space-Optimized Implementation

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':
            return 0
            
        n = len(s)
        # Space Optimization: dp[i] only depends on dp[i+1] (one) and dp[i+2] (two)
        one = 1  # Corresponds to dp[i+1], initialized as dp[n] = 1
        two = 0  # Corresponds to dp[i+2]
        
        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                cur = 0
            else:
                cur = one
                if i + 1 < n and (s[i] == '1' or (s[i] == '2' and s[i + 1] in '0123456')):
                    cur += two
            two = one
            one = cur
            
        return one
```

- **Complexity**: Time $O(n)$, Space $O(1)$.
- **Pitfalls**: Leading `'0'` cannot decode; boundary check requires `i + 1 < n`.

---

### 2. Stair Climbing Family: Climbing Stairs & Min Cost Climbing Stairs

#### Climbing Stairs
- **State**: $dp[i]$ = Number of ways to reach step $i$.
- **Transition**: $dp[i] = dp[i-1] + dp[i-2]$ (Fibonacci isomorphism).
- **Implementation**:

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        a, b = 1, 2
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b
```

#### Min Cost Climbing Stairs
- **State**: $dp[i]$ = Minimum cumulative cost to reach step $i$ and pay `cost[i]` to step off.
- **Top Floor Semantic**: The top floor has no cost, so the final answer is $\min(dp[n-1], dp[n-2])$.
- **Transition**: $dp[i] = cost[i] + \min(dp[i-1], dp[i-2])$.

```python
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        a, b = cost[0], cost[1]
        for i in range(2, len(cost)):
            a, b = b, cost[i] + min(a, b)
        return min(a, b)
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

---

### 3. Robbery Family: House Robber & House Robber II

#### House Robber I (Linear Street)
- **Constraint**: Cannot rob two adjacent houses on the same night.
- **State**: $dp[i]$ = Max money robbed considering first $i$ houses.
- **Transition**: $dp[i] = \max(dp[i-1], dp[i-2] + nums[i-1])$.

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        prev2, prev1 = 0, 0
        for x in nums:
            prev2, prev1 = prev1, max(prev1, prev2 + x)
        return prev1
```

#### House Robber II (Circular Street)
- **Circular Constraint**: House $0$ and House $n-1$ are adjacent and cannot both be robbed.
- **Dual Linear Subsegment Decomposition**:
  1. Case A: Skip last house $\implies$ Run linear Robber on `nums[0..n-2]`;
  2. Case B: Skip first house $\implies$ Run linear Robber on `nums[1..n-1]`;
  3. Global Max: $\max(\text{rob}(nums[0..n-2]), \text{rob}(nums[1..n-1]))$.

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return nums[0]
            
        def rob_linear(arr: list[int]) -> int:
            a = b = 0
            for x in arr:
                a, b = b, max(b, a + x)
            return b
            
        return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

- **Complexity**: Time $O(n)$, Space $O(1)$.
- **Pitfall**: Do not forget the $n=1$ edge case.

---

### 4. String Prefix Partition: Word Break

- **State**: $dp[i]$ = Boolean flag indicating if prefix $s[:i]$ can be segmented into dictionary words.
- **Transition**: Enumerate the split point $j \in [0, i)$ for the last word:
  $$dp[i] = \bigvee_{j=0}^{i-1} \bigl(dp[j] \land (s[j:i] \in \text{wordDict})\bigr)$$

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        words = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Empty prefix is trivially segmentable
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in words:
                    dp[i] = True
                    break
        return dp[n]
```

- **Complexity**: Time $O(n^2 \cdot L)$ ($L$ is substring slice cost, optimizable via Trie), Space $O(n)$.

---

### 5. Historical Sweep: Longest Increasing Subsequence (LIS)

- **State**: $dp[i]$ = Length of longest strictly increasing subsequence **ending strictly at $nums[i]$**.
- **Transition**: Scan all smaller predecessors $j < i$:
  $$dp[i] = 1 + \max_{j < i, nums[j] < nums[i]} dp[j]$$
- **Global Answer**: $\max_{0 \le i < n} dp[i]$ (not necessarily at the last cell).

```python
class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

- **Complexity**: Time $O(n^2)$, Space $O(n)$. (Note: Binary search / Patience sorting achieves $O(n \log n)$).

---

### 6. Sign Flipping & Dual Extrema: Maximum Product Subarray

- **Core Challenge**: Multiplying negative numbers flips signs; a local minimum (most negative) flips into a global maximum upon meeting another negative number!
- **Dual Extremum Tracking**: Maintain both `max_here` and `min_here` ending at index $i$:
  $$\begin{aligned}
  \text{max\_here}' &= \max(x, \text{max\_here} \cdot x, \text{min\_here} \cdot x) \\
  \text{min\_here}' &= \min(x, \text{max\_here} \cdot x, \text{min\_here} \cdot x)
  \end{aligned}$$

```python
class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        ans = max_here = min_here = nums[0]
        for x in nums[1:]:
            candidates = (x, max_here * x, min_here * x)
            max_here, min_here = max(candidates), min(candidates)
            ans = max(ans, max_here)
        return ans
```

## Stock Cooldown: Technique 1, Record Three States

Many people write recursion with `buying=True/False`, and that is certainly fine. But for bottom-up DP, a more intuitive method is to record which state you are in at the end of each day.

Define three states:

```text
hold[i] = the maximum profit when holding one share at the end of day i
sold[i] = the maximum profit when having sold the stock today at the end of day i
rest[i] = the maximum profit when holding no stock and not being in the just-sold state at the end of day i
```

These three states answer:

- `hold`: I currently hold a stock, and I can sell in the future.
- `sold`: I sold today, so tomorrow I must cooldown and cannot buy.
- `rest`: I currently hold no stock, and I am not restricted by a just-sold state, so I can buy in the future.

State transitions:

```text
hold[i] = max(
  hold[i - 1],              # I was already holding yesterday, so I continue doing nothing today
  rest[i - 1] - prices[i]   # yesterday I was freely out of the market, so I buy today
)

sold[i] = hold[i - 1] + prices[i]
  # I must have held yesterday in order to sell today

rest[i] = max(
  rest[i - 1],              # yesterday I was freely out of the market, so I continue resting today
  sold[i - 1]               # yesterday I just sold, so after cooldown today I become freely out of the market
)
```

Notice that the cooldown appears here:

```text
hold[i] can only buy from rest[i - 1], not from sold[i - 1]
```

because `sold[i - 1]` means you sold yesterday, so today is still the cooldown day and you cannot buy.

The base case can be understood like this:

```text
hold = -infinity  # before processing any day, it is impossible to already hold stock
sold = -infinity  # before processing any day, it is impossible to already have sold
rest = 0          # do nothing, profit is 0
```

The final answer cannot be `hold`, because still holding a stock does not count as realized profit:

```text
answer = max(sold[n - 1], rest[n - 1])
```

## Stock Cooldown: Full DP Table

First write the full DP to make the state meanings easy to confirm:

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        n = len(prices)
        neg_inf = float("-inf")
        hold = [neg_inf] * n
        sold = [neg_inf] * n
        rest = [0] * n

        hold[0] = -prices[0]
        sold[0] = neg_inf
        rest[0] = 0

        for i in range(1, n):
            hold[i] = max(hold[i - 1], rest[i - 1] - prices[i])
            sold[i] = hold[i - 1] + prices[i]
            rest[i] = max(rest[i - 1], sold[i - 1])

        return max(sold[n - 1], rest[n - 1])
```

Walk through `[1, 2, 3, 0, 2]` once:

```text
day  price  hold  sold  rest
0    1      -1    -inf  0
1    2      -1     1    0
2    3      -1     2    1
3    0       1    -1    2
4    2       1     3    2

answer = max(sold, rest) = max(3, 2) = 3
```

Look at day 3 where `price = 0` and `hold = 1`:

```text
hold[3] = max(hold[2], rest[2] - 0)
        = max(-1, 1 - 0)
        = 1
```

This means: you made `1` by selling on day 1, entered `rest = 1` after the cooldown on day 2, and on day 3 you can buy the stock priced at `0`, so the profit of the holding state is still `1`.

## Stock Cooldown: Technique 2, Space Optimization

Observe the transitions:

```text
hold[i] only depends on hold[i - 1], rest[i - 1]
sold[i] only depends on hold[i - 1]
rest[i] only depends on rest[i - 1], sold[i - 1]
```

That means day `i` uses only the three states from day `i - 1`. So you do not need three arrays, only three variables:

```text
hold = previous hold
sold = previous sold
rest = previous rest
```

Each day, compute the new states first, then replace them all at once:

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        hold = float("-inf")
        sold = float("-inf")
        rest = 0

        for price in prices:
            next_hold = max(hold, rest - price)
            next_sold = hold + price
            next_rest = max(rest, sold)

            hold, sold, rest = next_hold, next_sold, next_rest

        return max(sold, rest)
```

The most important detail here is: do not overwrite the old variables while computing.

Incorrect version:

```python
hold = max(hold, rest - price)
sold = hold + price  # wrong: this uses today's updated hold
```

`sold` must come from "held yesterday, sold today," so it must use the old `hold`. Therefore you should first store the new states in `next_*`, and assign them only at the end.

## Stock Cooldown: Relationship to the Recursive Version

The recursive version often uses two states:

```text
dfs(i, buying)
```

- `buying = True`: you do not hold stock and can buy.
- `buying = False`: you hold stock and can sell.

When selling, it jumps to `i + 2`:

```text
sell = prices[i] + dfs(i + 2, True)
```

The three-state bottom-up version explicitly splits this `i + 2` cooldown into `sold -> rest`:

```text
hold --sell--> sold --cooldown one day--> rest --buy--> hold
```

So the two approaches are essentially the same; they just record information differently. In interviews, if the problem has process constraints like "holding / just sold / out of the market and allowed to buy," the three-state machine is less error-prone.

## Interval DP: substrings and interval problems

The state is a closed interval `[i, j]` (substring, balloon interval). Transitions depend on shorter subintervals, so you must fill by increasing interval length, not by rows or columns.

## Longest Palindromic Substring

Use interval boolean DP: `$dp[i][j]$` means whether `$s[i..j]$` is a palindrome.

$$dp[i][j] = (s[i]=s[j]) \land (j-i<2 \lor dp[i+1][j-1])$$

Fill by increasing length while recording the start and length of the longest palindromic interval. Center expansion also works; here the point is the interval-DP shape.

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        start = length = 0
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    if j - i + 1 > length:
                        start, length = i, j - i + 1
        return s[start:start + length]
```

(Filling right-to-left and top-to-bottom is equivalent to increasing length; you can also write an explicit outer `length` loop.)

Complexity: Time `$O(n^2)$`, Space `$O(n^2)$`.

Pitfall: the transition depends on `$dp[i+1][j-1]$`, so shorter intervals must already be done; single characters are palindromes and should count toward the initial longest length.

## Palindromic Substrings

Use the same boolean table; only change the answer from "record the longest one" to "count every `$dp[i][j]=True$`".

```palindrome-dp-demo
```

### 2D DP Matrix Filling & Southwest Dependency Principles

1. **State Definition**:
   - `$dp[i][j]$` is a boolean indicating whether substring `$s[i..j]$` is a palindrome.
   - Because only `$i \le j$` substrings are valid, only the **main diagonal and upper triangle** of the 2D matrix are evaluated (lower triangle `$i > j$` is invalid).

2. **The 3-Step State Transition**:
   - **Step 1: Boundary Match Check**
     If `$s[i] \neq s[j]$`, the substring cannot be a palindrome ➔ `$dp[i][j] = False$`;
   - **Step 2: Base Cases (Length $\le 2$)**
     If `$s[i] == s[j]$`:
     - If `$j - i = 0$` (Length 1, main diagonal `$i = j$`): Single characters are always palindromes ➔ `$dp[i][i] = True$`;
     - If `$j - i = 1$` (Length 2, e.g. `"aa"`): Identical adjacent pair ➔ `$dp[i][i+1] = True$`;
   - **Step 3: Peeling the Onion (Length $\ge 3$)**
     If `$s[i] == s[j]$` and `$j - i \ge 2$`:
     The inner substring after stripping the two endpoints is `$s[i+1..j-1]$`. Query the matrix cell located **one row down ($i+1$) and one column left ($j-1$)** — the **southwest diagonal neighbor**:
     $$dp[i][j] = dp[i+1][j-1]$$

3. **Loop Ordering Principle**:
   Because computing cell `(i, j)` depends on its southwest neighbor `(i+1, j-1)`, shorter intervals must be computed before longer ones. We can iterate either:
   - Outer: `$i$` from `$n-1$` down to `0` (bottom-up); Inner: `$j$` from `$i$` up to `$n-1$` (left-to-right).
   - Or outer: length `$L = 1..n$`; Inner: start `$i = 0..n-L$`.

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        ans = 0
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    ans += 1
        return ans
```

Complexity: Time `$O(n^2)$`, Space `$O(n^2)$`.

Pitfall: do not only count the "longest" palindromes from center expansion; every valid `(i, j)` counts once, including all length-1 single characters.

## Burst Balloons

Pad a virtual balloon `1` on both ends. Define an open interval: `$dp[i][j]$` = maximum coins from bursting every balloon between `$i$` and `$j$`.

Key trick: enumerate the balloon `$k$` burst **last** inside the interval, not first. When `$k$` is last, both sides are already empty, so `$k$`'s neighbors at that moment are exactly the virtual boundaries `$a[i]$` and `$a[j]$`:

$$dp[i][j] = \max_{i<k<j}\bigl(dp[i][k] + a[i]\cdot a[k]\cdot a[j] + dp[k][j]\bigr)$$

Fill in increasing `$j-i$` order. The answer is `$dp[0][n+1]$` (`$n$` is the original length; after two virtual pads there are `$n+2$` positions).

```python
class Solution:
    def maxCoins(self, nums: list[int]) -> int:
        a = [1] + nums + [1]
        n = len(a)
        dp = [[0] * n for _ in range(n)]
        for length in range(2, n):
            for i in range(0, n - length):
                j = i + length
                for k in range(i + 1, j):
                    dp[i][j] = max(
                        dp[i][j],
                        dp[i][k] + a[i] * a[k] * a[j] + dp[k][j],
                    )
        return dp[0][n - 1]
```

Complexity: Time `$O(n^3)$`, Space `$O(n^2)$`.

Pitfall: thinking "which balloon to burst first" makes subproblem boundaries depend on "which balloons remain," so the transition is unclear; after switching to "which balloon is last in this interval," the left and right subintervals no longer interfere and can recurse independently.

#---

## Module 3: Skeleton 2 · Two Sequences & String Matching DP

Two-sequence DP handles alignment, edit operations, interleaving, and subsequence matching between two strings `s1` and `s2`. States are indexed by $(i, j)$ representing prefixes `s1[:i]` and `s2[:j]`.

```text
2D Two-Sequence Dependency Topology:
         dp[i-1][j-1] (Diagonal: match / substitute) ────> dp[i-1][j] (Top: delete / skip s1)
              │                                                │
              ▼                                                ▼
         dp[i][j-1] (Left: insert / skip s2)         ────> dp[i][j] (Current State)
```

---

### 1. Alignment & Common Subsequence: Longest Common Subsequence (LCS)

- **State**: $dp[i][j]$ = Length of LCS between `text1[:i]` and `text2[:j]`.
- **Transition**:
  $$dp[i][j] = \begin{cases} dp[i-1][j-1] + 1 & \text{text1}[i-1] == \text{text2}[j-1] \\ \max(dp[i-1][j], dp[i][j-1]) & \text{text1}[i-1] \neq \text{text2}[j-1] \end{cases}$$

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$ (rolling array reduces to $O(\min(m, n))$).
- **Pitfall**: When characters match, you must take the diagonal $+ 1$, not $\max(\text{top}, \text{left}) + 1$.

---

### 2. Tri-Directional Decision: Edit Distance

- **State**: $dp[i][j]$ = Minimum operations (insert, delete, replace) to convert `word1[:i]` to `word2[:j]`.
- **Transition**:
  $$dp[i][j] = \begin{cases} dp[i-1][j-1] & \text{word1}[i-1] == \text{word2}[j-1] \\ 1 + \min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) & \text{otherwise} \end{cases}$$

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$.

---

### 3. Dual-Source Feasibility: Interleaving String

- **State**: $dp[i][j]$ = Whether `s1[:i]` and `s2[:j]` interleave to form `s3[:i+j]`.
- **Transition**:
  $$dp[i][j] = (dp[i-1][j] \land s1[i-1] == s3[i+j-1]) \lor (dp[i][j-1] \land s2[j-1] == s3[i+j-1])$$

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n = len(s1), len(s2)
        if m + n != len(s3):
            return False
            
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = (
                    (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1])
                    or (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
                )
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$.

---

### 4. Subsequence Counting: Distinct Subsequences

- **State**: $dp[i][j]$ = Number of subsequences of `s[:i]` that equal `t[:j]`.
- **Transition**:
  $$dp[i][j] = \begin{cases} dp[i-1][j-1] + dp[i-1][j] & s[i-1] == t[j-1] \\ dp[i-1][j] & s[i-1] \neq t[j-1] \end{cases}$$

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 1  # Empty target t can always be matched once (by picking nothing)
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$.

---

### 5. Wildcard Branches: Regular Expression Matching

- **State**: $dp[i][j]$ = Whether `s[:i]` matches pattern `p[:j]`.
- **Transition**:
  1. $p[j-1] \neq '*'$: If $p[j-1] == s[i-1] \lor p[j-1] == '.' \implies dp[i][j] = dp[i-1][j-1]$.
  2. $p[j-1] == '*'$:
     - **Match 0 times**: Discard the `x*` token completely $\implies dp[i][j] = dp[i][j-2]$;
     - **Match 1 or more times**: If preceding char matches ($p[j-2] == s[i-1] \lor p[j-2] == '.'$) $\implies$ consume char from $s$ and keep pattern $\implies dp[i-1][j]$.

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        # Base Cases for patterns like a*, a*b*, a*b*c* matching empty string
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
                
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2]  # Match 0 times
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]  # Match 1+ times
                elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$.

---

## Module 4: Skeleton 4 · Interval DP & Matrix Evolution

Interval DP states are defined over a closed interval $[i, j]$. Because larger intervals strictly depend on their shorter sub-intervals, iteration order must proceed **by increasing interval length ($L: 1 \to n$)** or **bottom-up ($i: n-1 \to 0$)**.

---

### 1. Palindrome Family: Longest Palindromic Substring & Palindromic Substrings

#### Recurrence & Onion Peeling Principle
Define $dp[i][j]$ as whether substring $s[i..j]$ is a palindrome:

$$dp[i][j] = (s[i] == s[j]) \land (j - i < 2 \lor dp[i+1][j-1])$$

1. **Boundary Mismatch** ($s[i] \neq s[j]$): Cannot be a palindrome $\implies dp[i][j] = False$;
2. **Length $\le 2$** ($j - i < 2$): If endpoints match, single char ($i=j$) or adjacent pair ($j=i+1$) are base cases $\implies dp[i][j] = True$;
3. **Length $\ge 3$** ($j - i \ge 2$): Check the **southwest neighbor $\swarrow$** $dp[i+1][j-1]$ (inner peeled core).

#### 2D State Transition Matrix Interactive Visualization

```palindrome-dp-demo
```

#### Code Implementation

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        ans = 0
        
        # Bottom-up sweep guarantees dp[i+1][j-1] is computed before dp[i][j]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    ans += 1
        return ans

    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        start, max_len = 0, 1
        
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    if j - i + 1 > max_len:
                        start, max_len = i, j - i + 1
        return s[start:start + max_len]
```

- **Complexity**: Time $O(n^2)$, Space $O(n^2)$.

---

### 2. Reverse Thinking of "Last Burst": Burst Balloons

- **Why "first burst" fails**: Popping $k$ first merges its left and right neighbors, dynamically entangling subproblem boundaries!
- **Reverse Formulation**: Enumerate the **last balloon $k$ to burst** in open interval $(i, j)$. When $k$ is burst last, all balloons in $(i, k)$ and $(k, j)$ are already cleared, leaving $k$'s neighbors fixed as the boundary elements $a[i]$ and $a[j]$!

$$dp[i][j] = \max_{i < k < j} \bigl(dp[i][k] + a[i] \cdot a[k] \cdot a[j] + dp[k][j]\bigr)$$

```python
class Solution:
    def maxCoins(self, nums: list[int]) -> int:
        a = [1] + nums + [1]
        n = len(a)
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for i in range(0, n - length):
                j = i + length
                for k in range(i + 1, j):
                    dp[i][j] = max(
                        dp[i][j],
                        dp[i][k] + a[i] * a[k] * a[j] + dp[k][j]
                    )
        return dp[0][n - 1]
```

- **Complexity**: Time $O(n^3)$, Space $O(n^2)$.

---

## Module 5: Skeleton 4 · Knapsack Family (0/1, Complete & Counting)

Knapsack problems handle constrained optimization and counting. The defining difference in 1D space compression is the inner loop direction:

```text
Knapsack Traversal Direction:
┌─────────────────┬───────────────────┬───────────────────────────────────────┐
│ Knapsack Type   │ Item Usage Limit  │ Inner Capacity Loop Direction         │
├─────────────────┼───────────────────┼───────────────────────────────────────┤
│ 0/1 Knapsack    │ At most 1 per item│ Backward (Capacity ➔ Weight) [Prevent]│
│ Complete        │ Unlimited per item│ Forward (Weight ➔ Capacity) [Allow]   │
└─────────────────┴───────────────────┴───────────────────────────────────────┘
```

---

### 1. 0/1 Knapsack: Partition Equal Subset Sum (Feasibility)

- **Reduction**: Determine if a subset can be chosen (each element at most once) summing to $target = \text{total} / 2$.
- **Transition**: $dp[j] = dp[j] \lor dp[j - x]$ (traversed backward).

```python
class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False
        target = total // 2
        
        dp = [False] * (target + 1)
        dp[0] = True
        
        for x in nums:
            for j in range(target, x - 1, -1):  # 0/1 Knapsack MUST traverse backward!
                dp[j] = dp[j] or dp[j - x]
        return dp[target]
```

- **Complexity**: Time $O(n \cdot target)$, Space $O(target)$.

---

### 2. 0/1 Knapsack: Target Sum (Algebraic Reduction & Counting)

- **Algebraic Derivation**:
  $$P - N = target,\quad P + N = total \implies 2P = target + total \implies P = \frac{target + total}{2}$$
- **Reduction**: Count subset combinations summing to $bag = (target + total) // 2$.
- **Transition**: $dp[s] += dp[s - num]$ (backward).

```python
class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        total = sum(nums)
        if abs(target) > total or (target + total) % 2 != 0:
            return 0
            
        bag = (target + total) // 2
        dp = [0] * (bag + 1)
        dp[0] = 1
        
        for num in nums:
            for s in range(bag, num - 1, -1):  # Backward
                dp[s] += dp[s - num]
        return dp[bag]
```

- **Complexity**: Time $O(n \cdot bag)$, Space $O(bag)$.

---

### 3. Complete Knapsack: Coin Change (Min Coins)

- **Semantic**: Unlimited coins, find minimum coin count to make `amount`.
- **Transition**: $dp[s] = \min(dp[s], dp[s - coin] + 1)$ (**forward traversal**).

```python
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        inf = amount + 1
        dp = [inf] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for s in range(coin, amount + 1):  # Forward
                dp[s] = min(dp[s], dp[s - coin] + 1)
                
        return -1 if dp[amount] == inf else dp[amount]
```

- **Complexity**: Time $O(n \cdot amount)$, Space $O(amount)$.

---

### 4. Complete Knapsack: Coin Change II (Combinations vs Permutations)

- **Combinations vs Permutations Loop Order**:
  - **Combinations (`{1, 2}` same as `{2, 1}`)**: **Items in outer loop, Capacity in inner loop**. Each coin type is processed once in fixed order.
  - **Permutations (`{1, 2}` different from `{2, 1}`)**: **Capacity in outer loop, Items in inner loop**.

```python
class Solution:
    def change(self, amount: int, coins: list[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        # Combinations: Coins in outer loop
        for c in coins:
            for a in range(c, amount + 1):
                dp[a] += dp[a - c]
        return dp[amount]
```

- **Complexity**: Time $O(n \cdot amount)$, Space $O(amount)$.

---

### 5. Knapsack Initialization Cheat Sheet

| Objective | `dp[0]` Base | Rest `dp[1..W]` | Transition Operator |
|---|---|---|---|
| Max Value (Capacity $\le W$) | `0` | `0` | $\max(dp[j], dp[j-w] + v)$ |
| Max Value (Capacity $= W$ exact) | `0` | `-inf` | $\max(dp[j], dp[j-w] + v)$ |
| Min Items (Capacity $= W$ exact) | `0` | `+inf` (or `amount + 1`) | $\min(dp[j], dp[j-c] + 1)$ |
| Count Ways | `1` | `0` | $dp[j] += dp[j-w]$ |
| Feasibility | `True` | `False` | $dp[j] = dp[j] \lor dp[j-w]$ |

---

## Module 6: Skeleton 5 · Grid & DAG Memoization

### 1. Deterministic Grid Sweep: Unique Paths

- **Rule**: Only move down or right $\implies$ Dependency is purely top and left.
- **Transition**: $dp[j] = dp[j] + dp[j-1]$ (single rolling row).

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
```

- **Complexity**: Time $O(mn)$, Space $O(n)$.

### 2. Arbitrary Grid & Implicit DAG: Longest Increasing Path in a Matrix

- **Why nested loops fail**: Values are arbitrarily distributed without a static geometric sweep order.
- **DAG Property**: Strictly increasing paths form a Directed Acyclic Graph (DAG).
- **Solution**: Memoized DFS computes topological DP over the DAG.

```python
class Solution:
    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        memo = {}
        
        def dfs(r: int, c: int) -> int:
            if (r, c) in memo:
                return memo[(r, c)]
            best = 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and matrix[nr][nc] > matrix[r][c]:
                    best = max(best, 1 + dfs(nr, nc))
            memo[(r, c)] = best
            return best
            
        return max(dfs(r, c) for r in range(m) for c in range(n))
```

- **Complexity**: Time $O(mn)$ (each cell computed exactly once), Space $O(mn)$.

---

## Module 7: Skeleton 6 · State Machine DP

When a stage can occupy one of several mutually exclusive named discrete states, State Machine DP decouples transition logic cleanly.

### Classic: Best Time to Buy and Sell Stock with Cooldown

```text
3-State Finite State Machine (FSM):
           ┌─────────────────────────── buy ───────────────────────────┐
           │                                                           │
           ▼                                                           │
      ┌─────────┐                  sell                    ┌───────────────┐
      │  Hold   │ ───────────────────────────────────────> │     Sold      │
      │ (owned) │                                          │(just sold/cool)│
      └─────────┘                                          └───────────────┘
           │                                                           │
          rest                                                    cooldown
           │                                                           │
           ▼                                                           ▼
      ┌─────────┐                                          ┌───────────────┐
      │  Hold   │ <──────────────────── rest ───────────── │     Rest      │
      └─────────┘                                          │  (free cash)  │
                                                           └───────────────┘
```

#### States & Transitions
- `hold`: Max profit holding stock at end of day $\implies \text{hold} = \max(\text{hold}, \text{rest} - price)$
- `sold`: Max profit selling stock today (cooldown next day) $\implies \text{sold} = \text{hold} + price$
- `rest`: Max profit holding no stock & ready to buy $\implies \text{rest} = \max(\text{rest}, \text{sold})$

#### Implementation (Avoiding Variable Dirty-Read Overwrite)

```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        hold = float("-inf")
        sold = float("-inf")
        rest = 0
        
        for price in prices:
            next_hold = max(hold, rest - price)
            next_sold = hold + price
            next_rest = max(rest, sold)
            # Synchronous update prevents new hold from corrupting sold calculation
            hold, sold, rest = next_hold, next_sold, next_rest
            
        return max(sold, rest)
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

---

## Module 8: Advanced Insight · DP to Greedy Compression

Many greedy algorithms are mathematically derived from **compressing full DP tables via monotonicity and dominance relations**.

---

### 1. Kadane's Algorithm: Maximum Subarray

- **DP Formulation**: $dp[i] = \max(nums[i], dp[i-1] + nums[i])$.
- **Greedy Compression**: If $curSum < 0$, keeping it drags down all future extensions; thus reset $curSum = 0$ immediately!

```python
class Solution:
    def maxSubArray(self, nums: list[int]) -> int:
        cur_sum = 0
        max_sum = nums[0]
        for x in nums:
            if cur_sum < 0:
                cur_sum = 0
            cur_sum += x
            max_sum = max(max_sum, cur_sum)
        return max_sum
```

- **Complexity**: Time $O(n)$, Space $O(1)$.
- **Pitfall**: In an all-negative array (e.g. `[-3, -1, -2]`), `max_sum` must initialize to `nums[0]`, not `0`.

### 2. Jump Game: Leftmost Goal Compression

- **Full DP Formulation**: $dp[i] = \bigvee_{j=i+1}^{i+nums[i]} dp[j]$ in $O(n^2)$ time.
- **Greedy Reduction**:
  - The left-most good position dominates all positions to its right (easier to reach from earlier indices).
  - Hence, compress the entire boolean array into a single scalar `goal` (leftmost reachable position).
  - If $i + nums[i] \ge goal$, update $goal = i$.

```python
class Solution:
    def canJump(self, nums: list[int]) -> bool:
        goal = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= goal:
                goal = i
        return goal == 0
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

#### Criteria for DP ➔ Greedy Compression:
1. **Monotonic Boundary**: State set can be represented by a single extremum (e.g. leftmost `goal` or furthest `reach`).
2. **State Dominance**: The chosen representative state strictly outperforms all dominated states across all future branches.
3. **Choice Safety**: Local greedy decisions never eliminate any globally optimal possibilities.

---

## Module 9: Interview Pitfalls, Complexity Matrix & Quizzes

### 1. Complexity & Space Optimization Summary

| Skeleton | Problem | Standard Time | Space (Standard $\to$ Optimized) | Key Transitions & Order |
|---|---|---|---|---|
| **1D Linear** | Decode Ways | $O(n)$ | $O(n) \to O(1)$ | Backward sweep, handle `'0'`, 2 rolling variables |
| **1D Linear** | Climbing Stairs | $O(n)$ | $O(n) \to O(1)$ | Fibonacci structure, 2 rolling variables |
| **1D Linear** | Min Cost Climbing Stairs | $O(n)$ | $O(n) \to O(1)$ | Top floor has no cost, take $\min(dp[n-1], dp[n-2])$ |
| **1D Linear** | House Robber I / II | $O(n)$ | $O(n) \to O(1)$ | Pick vs skip; II splits into 2 linear subsegments |
| **1D Linear** | Word Break | $O(n^2 \cdot L)$ | $O(n)$ | Set lookup, enumerate last split point |
| **1D Linear** | LIS | $O(n^2)$ | $O(n)$ | Global predecessor sweep; answer is $\max(dp)$ |
| **1D Linear** | Maximum Product Subarray | $O(n)$ | $O(n) \to O(1)$ | Track `max_here` and `min_here` simultaneously |
| **Two Sequences** | LCS | $O(mn)$ | $O(mn) \to O(\min(m, n))$ | Match $\to$ diagonal $+1$; else $\max(\text{top}, \text{left})$ |
| **Two Sequences** | Edit Distance | $O(mn)$ | $O(mn)$ | Match $\to$ diagonal; else $1 + \min(\text{insert, delete, replace})$ |
| **Two Sequences** | Interleaving String | $O(mn)$ | $O(mn)$ | Top or left matching `s3[i+j-1]` |
| **Two Sequences** | Distinct Subsequences | $O(mn)$ | $O(mn)$ | Match $\to$ sum of pick and skip; $dp[i][0]=1$ |
| **Two Sequences** | Regex Matching | $O(mn)$ | $O(mn)$ | `*` branches into $dp[i][j-2]$ (0x) and $dp[i-1][j]$ (1+x) |
| **Interval DP** | Longest Palindrome / Count | $O(n^2)$ | $O(n^2)$ | Southwest neighbor $dp[i+1][j-1]$, bottom-up sweep |
| **Interval DP** | Burst Balloons | $O(n^3)$ | $O(n^2)$ | Reverse formulation: last balloon $k$ to burst |
| **0/1 Knapsack** | Partition Equal Subset Sum | $O(n \cdot \frac{\text{sum}}{2})$ | $O(\frac{\text{sum}}{2})$ | Feasibility knapsack, capacity must loop **backward** |
| **0/1 Knapsack** | Target Sum | $O(n \cdot bag)$ | $O(bag)$ | Algebraic reduction to subset count, **backward** |
| **Complete** | Coin Change (Min Coins) | $O(n \cdot amount)$ | $O(amount)$ | Complete knapsack, capacity loops **forward**, $\min$ |
| **Complete** | Coin Change II (Combinations)| $O(n \cdot amount)$ | $O(amount)$ | Combinations: **Coins in outer loop, capacity forward in inner** |
| **Grid DP** | Unique Paths | $O(mn)$ | $O(n)$ | Top + Left, single rolling row |
| **Grid DP** | Longest Increasing Path | $O(mn)$ | $O(mn)$ | Implicit DAG over grid, Memoized DFS |
| **State Machine**| Stock with Cooldown | $O(n)$ | $O(n) \to O(1)$ | Hold / Sold / Rest 3-state synchronous rotation |
| **DP ➔ Greedy**| Kadane Max Subarray | $O(n)$ | $O(1)$ | Negative prefix reset, 1 rolling variable |
| **DP ➔ Greedy**| Jump Game | $O(n)$ | $O(1)$ | Maintain leftmost `goal` or furthest `reach` |

---

### 2. High-Frequency Interview Pitfalls

1. **Writing code without clarifying Base Cases**: E.g. $dp[0]=1$, empty string match, top floor having no cost.
2. **Iteration order violating topological dependencies**: In Interval DP, sweeping without length-increasing/bottom-up order reads uninitialized $dp[i+1][j-1]$.
3. **Reversing knapsack loop direction**:
   - 0/1 Knapsack in forward order allows the same item to be reused multiple times within the same step, degenerating into complete knapsack.
   - Permutations vs Combinations: Coins in outer loop yields combinations (Coin Change II); capacity in outer loop yields permutations (Combination Sum IV).
4. **State Machine dirty-read overwrites during space optimization**: Modifying `hold` before computing `sold = hold + price` incorrectly reads the updated `hold`. Always use temporary variables for synchronous updates!
5. **Wrong answer location in extremum DP**: Assuming the final answer is always at $dp[n-1]$ (in LIS and Maximum Product Subarray, the answer is the global maximum over all cells).

---

### 3. Quick Quizzes (12 Classic Questions)

```quiz
title: Quick quiz 1
question: In House Robber II compared to House Robber I, what is the most critical modification?
answer: B
A. Change to 2D DP dp[i][j]
B. Split the circular street into two linear subproblems and take the max
C. Must use interval DP
D. Change to complete knapsack
explanation: The first and last houses are adjacent; solve linear robbery on nums[0..n-2] and nums[1..n-1] separately, then take the maximum.
```

```quiz
title: Quick quiz 2
question: For 1D space-optimized DP, what is the key difference between Coin Change (min coins) and Coin Change II (combinations)?
answer: C
A. One uses backward loop and the other uses forward loop
B. One must be 2D and the other 1D
C. Different objective functions (min vs additive counting), and II requires coins in the outer loop to enforce combination semantics
D. Both transitions are identical except for return value
explanation: Both are complete knapsack shapes; II counts combinations so coins must be the outer loop; transitions are min vs addition.
```

```quiz
title: Quick quiz 3
question: In Burst Balloons interval transition, what does k represent?
answer: A
A. The last balloon to burst in open interval (i, j)
B. The first balloon to burst in the open interval
C. The interval length
D. The index of the virtual boundary 1
explanation: When k bursts last, balloons between i and k, and between k and j are already cleared, isolating the remaining cost to a[i]*a[k]*a[j] + subproblem DPs.
```

```quiz
title: Quick quiz 4
question: When reducing Target Sum to subset sum, what is the target subset sum P?
answer: B
A. (sum - target) / 2
B. (sum + target) / 2
C. sum - target
D. target
explanation: P + N = sum and P - N = target yield P = (sum + target) / 2; requires (sum + target) to be non-negative and even.
```

```quiz
title: Quick quiz 5
question: Why can Longest Increasing Path in a Matrix not be solved by a simple row-by-row nested loop like Unique Paths?
answer: D
A. Because you can only move right and down
B. Because space must be O(1)
C. Because it is not a DP problem
D. Increasing neighbors can point in any direction with no fixed sweep order; Memoized DFS on the implicit DAG is required
explanation: Strictly increasing edges form a DAG; memoized DFS computes topological DP correctly.
```

```quiz
title: Quick quiz 6
question: In Edit Distance when word1[i-1] == word2[j-1], what is the correct transition?
answer: A
A. dp[i][j] = dp[i-1][j-1]
B. dp[i][j] = dp[i-1][j-1] + 1
C. dp[i][j] = max(dp[i-1][j], dp[i][j-1])
D. dp[i][j] = dp[i][j-1] + 1
explanation: Characters match, requiring 0 edits; inherit diagonal directly. Mismatches take 1 + min(insert, delete, replace).
```

```quiz
title: Quick quiz 7
question: Why must Maximum Product Subarray maintain min_here alongside max_here?
answer: C
A. To handle zeros
B. To achieve O(1) space
C. A negative number flips a minimum product into a maximum; tracking only max_here loses optimal solutions
D. The problem asks for minimum product
explanation: In inputs like [-2, 3, -4], negative times negative flips the minimum (-6) to maximum (+24).
```

```quiz
title: Quick quiz 8
question: In Distinct Subsequences when s[i-1] == t[j-1], what is the transition?
answer: B
A. Only dp[i-1][j-1]
B. dp[i-1][j-1] + dp[i-1][j] (choose to match or skip current char)
C. dp[i][j-1] + dp[i-1][j]
D. max(dp[i-1][j-1], dp[i-1][j])
explanation: Sum the ways from matching s[i-1] with t[j-1] and skipping s[i-1].
```

```quiz
title: Quick quiz 9
question: In Regular Expression Matching when p[j-1] == '*', what represents matching 0 times?
answer: A
A. dp[i][j-2]
B. dp[i-1][j]
C. dp[i-1][j-1]
D. dp[i][j-1]
explanation: Discard the entire x* pattern token and check if p[:j-2] matches s[:i].
```

```quiz
title: Quick quiz 10
question: Why must the capacity loop run backward in 1D space-optimized 0/1 Knapsack?
answer: C
A. It is faster
B. To turn combinations into permutations
C. To prevent the same item from being reused multiple times in the same round
D. Backward traversal is required for complete knapsack
explanation: Backward loop ensures dp[j-x] references the state from before considering the current item.
```

```quiz
title: Quick quiz 11
question: In Stock with Cooldown, what is the precise meaning of the sold state?
answer: B
A. Any empty holding state
B. Stock was sold today, mandating cooldown tomorrow
C. Currently holding stock
D. Cumulative sell count
explanation: sold specifically marks "sold today"; the next day can only enter rest, not buy directly.
```

```quiz
title: Quick quiz 12
question: In House Robber II, why can you not directly reuse House Robber I's linear DP?
answer: B
A. Different data ranges require a different algorithm
B. First and last houses are adjacent and form a ring, so you must split into two linear subproblems and take the larger answer
C. A circular array must use interval DP
D. The dp transition equation itself must change on a circular array
explanation: The only change from the ring is "cannot rob first and last together"; splitting into [0..n-2] and [1..n-1] linear Robber bypasses that constraint, and the transition inside each segment is unchanged.
```

---

### 4. Structured Interview Communication Template

When presenting your solution in a whiteboard interview, follow these 6 structured steps:

1. **State Definition**: "I define $dp[i][j]$ as the optimal value / number of ways for prefix/interval/stage..."
2. **Decision Branches & Transition**: "At the current position, there are $k$ choices, which map to sub-states..."
3. **Base Cases**: "The base cases are $dp[0]$ / the main diagonal, representing..."
4. **Iteration Order**: "Because the current state depends on earlier states, we iterate in forward/backward/length-increasing order."
5. **Implementation & Dry Run**: "Let's implement the 2D/1D DP and dry-run with a small example."
6. **Complexity & Space Optimization**: "The time complexity is $O(\dots)$ and space is $O(\dots)$. Observing that dependencies only reach the previous row/two variables, we can compress space to $O(1)$ / $O(m)$."
