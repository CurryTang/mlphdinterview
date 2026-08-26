# Dynamic Programming Patterns: Six Skeletons for 23 Problems

DP problems look numerous, but there are only six skeletons. First ask: is the state a 1D index, a range, a knapsack capacity, two sequence prefixes, a grid cell, or a small set of named states?

This note covers the 23 problems below. For each one it only states the DP definition, transition, base case, iteration order, how to read the answer, and one concrete pitfall. Decode Ways is already fully developed in CoreSkills10; here it is only classified with a cross-reference.

| Skeleton | Representative problems |
|---|---|
| 1D linear DP | Climbing Stairs, House Robber, Coin Change, LIS, ... |
| Interval DP | Longest Palindromic Substring, Burst Balloons |
| Knapsack family | Partition Equal Subset Sum, Coin Change II, Target Sum |
| Two-sequence DP | LCS, Edit Distance, Regex Matching, ... |
| Grid DP | Unique Paths (exception: Longest Increasing Path uses memoized DFS) |
| State-machine DP | Best Time to Buy and Sell Stock with Cooldown |

## Universal solving steps

For every problem above, walk these five steps before writing loops:

1. **Define the state** `dp[...]`: one sentence for what it means exactly (prefix / range / capacity / two prefixes / cell / day i in some named state).
2. **Clarify the last step**: which smaller states combine into the current one; write the transition (`min` / `max` / `+` / `or`).
3. **Fix the base case**: answers for the smallest subproblems; many bugs live here, not in the transition.
4. **Fix the computation order**: bottom-up, ensure dependencies are already filled; memoized DFS, recurse toward smaller scale on a DAG with no cycles.
5. **Extract the final answer from `dp`**: a fixed `dp[n]` / `dp[m][n]`, a `max` over all `i`, or a field of rolling variables. Missed boundaries (empty string, `n=1`, all zeros) usually fail here.

In one line:

```text
state + transition + base case + iteration order + answer extraction
```

## The universal template: start here

```text
What is the state dimension?

├── A single index i (first i elements / climb to step i)
│     -> 1D linear DP
│       Examples: Climbing Stairs, House Robber, Decode Ways, Word Break, LIS, Coin Change, Max Product
│
├── A closed range [i, j] (substring / balloon range), fill by increasing length
│     -> Interval DP
│       Examples: Palindrome series, Burst Balloons
│
├── Items + capacity (take or skip / unbounded / counting)
│     -> 0/1 knapsack / unbounded knapsack / counting knapsack
│       Examples: Partition Equal Subset Sum, Coin Change II, Target Sum
│
├── Prefixes of two sequences (i, j)
│     -> Two-sequence DP
│       Examples: LCS, Edit Distance, Interleaving String, Distinct Subsequences, Regex Matching
│
├── Grid cell (r, c) with a fixed sweep direction
│     -> Grid DP
│       Example: Unique Paths
│       Exception: Longest Increasing Path has no fixed direction -> memoized DFS
│
└── A few named states per step (hold / cooldown / free)
      -> State-machine DP
        Example: Stock with Cooldown
```

### Skeleton A: 1D linear DP

```python
def solve_1d(arr):
    n = len(arr)
    dp = [INIT] * (n + 1)   # or n, depending on indexing
    # BASE: fill dp[0] / dp[1] / ...
    for i in range(START, n):
        dp[i] = TRANSITION(dp, i, arr)   # depends on O(1)~O(k) earlier states
    return ANSWER(dp)                    # dp[n] or max(dp), etc.
```

| Problem | State | Transition | Base | Order | Answer |
|---|---|---|---|---|---|
| Climbing Stairs | `dp[i]` = ways to reach step i | `dp[i]=dp[i-1]+dp[i-2]` | `dp[0]=1,dp[1]=1` | `i=2..n` | `dp[n]` |
| Min Cost Climbing Stairs | `dp[i]` = min cost to reach i | `dp[i]=cost[i]+min(dp[i-1],dp[i-2])` | `dp[0]=cost[0],dp[1]=cost[1]` | `i=2..n-1` | `min(dp[n-1],dp[n-2])` |
| House Robber | `dp[i]` = best over first i houses | `dp[i]=max(dp[i-1],dp[i-2]+nums[i-1])` | `dp[0]=0,dp[1]=nums[0]` | `i=2..n` | `dp[n]` |
| House Robber II | same, ring split into two linear runs | run Robber on `[0..n-2]` and `[1..n-1]` | same as Robber per run | two linear passes | `max` of both answers |
| Decode Ways | see CoreSkills10 | add if 1-digit / 2-digit valid | `dp[n]=1`; leading `0` -> 0 | prefix or suffix | `dp[0]` or `dp[n]` |
| Word Break | `dp[i]` = whether `s[:i]` can break | `dp[j] and s[j:i] in dict` | `dp[0]=True` | `i=1..n`, inner `j` | `dp[n]` |
| LIS | `dp[i]` = LIS ending at `i` | `dp[i]=max(dp[j])+1` (`nums[j]<nums[i]`) | all `1` | nested `i,j<i` | `max(dp)` |
| Coin Change | `dp[a]` = fewest coins for amount `a` | `dp[a]=min(dp[a-c])+1` | `dp[0]=0`, else `inf` | amounts ascending | `dp[amount]` (`inf` -> `-1`) |
| Max Product Subarray | max/min product ending at `i` | roll both `max_here,min_here` | first element | one scan | global `max` |

### Skeleton B: interval DP

```python
def solve_interval(s):
    n = len(s)
    dp = [[INIT] * n for _ in range(n)]
    # BASE: length 1 (sometimes also length 0 / 2)
    for length in range(2, n + 1):       # increasing interval length
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i][j] = TRANSITION(dp, i, j, s)
    return ANSWER(dp)
```

| Problem | State | Transition | Base | Order | Answer |
|---|---|---|---|---|---|
| Longest Palindromic Substring | `dp[i][j]` = whether `s[i..j]` is palindrome | ends equal and (len<=2 or `dp[i+1][j-1]`) | singles `True` | by length | record longest `(i,j)` |
| Palindromic Substrings | same boolean table | same | same | same | count `True` cells |
| Burst Balloons | `dp[i][j]` = max coins bursting open `(i,j)` | `max_k a[i]*a[k]*a[j]+dp[i][k]+dp[k][j]` (`k` last) | adjacent `i,j` = 0 | length up; pad `1`s | `dp[0][n+1]` |

### Skeleton C: knapsack family

```python
def solve_knapsack(items, capacity):
    dp = [INIT] * (capacity + 1)
    dp[0] = BASE0
    for x in items:                      # items outer -> combinations
        for j in range(...):             # 0/1: descending; unbounded: ascending
            dp[j] = COMBINE(dp[j], dp[j - x], x)
    return ANSWER(dp)
```

| Problem | State | Transition | Base | Order | Answer |
|---|---|---|---|---|---|
| Partition Equal Subset Sum | `dp[j]` = can we make sum `j` | `dp[j]\|=dp[j-x]` (0/1) | `dp[0]=True` | items outer, capacity descending | `dp[sum/2]` |
| Coin Change II | `dp[a]` = number of combinations for `a` | `dp[a]+=dp[a-c]` (unbounded) | `dp[0]=1` | **coins outer, amount inner ascending** | `dp[amount]` |
| Target Sum | reduce to subset-sum counting | same as 0/1 counting knapsack | `dp[0]=1` | items outer, capacity descending | `dp[(sum+target)/2]` |

Coin Change (fewest coins) is also unbounded-knapsack shaped, but the objective is `min`, not counting; it sits in Skeleton A because it is usually written as a 1D `dp[amount]` table. Versus Coin Change II: same unbounded shape; items outer / amount inner yields combinations; amount outer / items inner would count permutations (II wants combinations).

### Skeleton D: two-sequence DP

```python
def solve_two_seq(a, b):
    m, n = len(a), len(b)
    dp = [[INIT] * (n + 1) for _ in range(m + 1)]
    # BASE: fill row 0 / column 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = TRANSITION(dp, i, j, a, b)
    return dp[m][n]
```

| Problem | State | Transition key | Base |
|---|---|---|---|
| LCS | `dp[i][j]` = LCS length of `a[:i]`, `b[:j]` | equal -> `+1`, else `max(up,left)` | row/col 0 = 0 |
| Edit Distance | min ops to turn `a[:i]` into `b[:j]` | equal copy diagonal; else insert/delete/replace +1 | `dp[i][0]=i`,`dp[0][j]=j` |
| Interleaving String | can `s1[:i]` and `s2[:j]` form `s3[:i+j]` | take from `s1` above or `s2` left | empty matches empty |
| Distinct Subsequences | ways `s[:i]` subsequences equal `t[:j]` | chars equal: use or skip; else skip | `dp[i][0]=1` |
| Regex Matching | does `p[:j]` match `s[:i]` | `.` any char; `*` zero times (`j-2`) or more (if match, `i-1`) | empty pattern; `x*` can match empty |

### Skeleton E: grid DP (with memoization exception)

```python
def unique_paths(m, n):
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]
```

Longest Increasing Path in a Matrix **cannot** be filled with a single row/column sweep: matrix values are arbitrary, so there is no one valid topological scan order. The right approach is memoized DFS from each cell, only moving to a strictly larger neighbor. That forms a DAG (no cycles), so overlapping subproblems can be cached. This is still DP (optimal substructure + overlapping subproblems + memo), just without an explicit tabulation loop.

### Skeleton F: state-machine DP

```python
# hold / sold(cooldown) / free
for price in prices:
    hold, sold, free = TRANSITIONS(...)
return max(sold, free)
```

| Problem | States | Transitions |
|---|---|---|
| Stock with Cooldown | `hold` / `sold` / `free` | see the three-state equations below |

---

## 1D linear DP

### Climbing Stairs

State: `dp[i]` = number of ways to reach step `i`.

$$dp[i] = dp[i-1] + dp[i-2]$$

Base: $dp[0]=1$, $dp[1]=1$ (or start from $dp[1]=1,dp[2]=2$).

Order: $i=2..n$. Answer: $dp[n]$.

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

Complexity: Time $O(n)$, Space $O(1)$.

Pitfall: for $n=1$ do not read $dp[2]$; in interviews it is enough to note the Fibonacci isomorphism.

### Min Cost Climbing Stairs

You may start at index `0` or `1`. `dp[i]` = min total cost after reaching step `i` and paying `cost[i]`.

$$dp[i] = cost[i] + \min(dp[i-1], dp[i-2])$$

The answer is not $dp[n]$ (the top has no cost); it is $\min(dp[n-1], dp[n-2])$: the last step can leave from either of the last two stairs.

```python
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        n = len(cost)
        a, b = cost[0], cost[1]
        for i in range(2, n):
            a, b = b, cost[i] + min(a, b)
        return min(a, b)
```

Complexity: Time $O(n)$, Space $O(1)$.

Pitfall: forgetting that the top floor has no cost, and returning only $dp[n-1]$.

### House Robber

Cannot rob adjacent houses. `dp[i]` = max money considering only the first `i` houses.

$$dp[i] = \max(dp[i-1], dp[i-2] + nums[i-1])$$

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        prev2, prev1 = 0, 0
        for x in nums:
            prev2, prev1 = prev1, max(prev1, prev2 + x)
        return prev1
```

Complexity: Time $O(n)$, Space $O(1)$.

Pitfall: empty / single-element arrays; keep `nums[i-1]` vs `nums[i]` consistent with the `dp` length convention.

### House Robber II

Houses form a ring: first and last are adjacent. Split into two linear House Robber runs:

```text
max( rob(nums[0..n-2]), rob(nums[1..n-1]) )
```

When $n=1$, return $nums[0]$ directly.

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        def rob_linear(arr):
            a = b = 0
            for x in arr:
                a, b = b, max(b, a + x)
            return b
        return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

Complexity: Time $O(n)$, Space $O(1)$.

Pitfall: missing $n=1$; wrong slices that drop the only middle house.

### Decode Ways

Classification: 1D linear DP (counting ways). Add $dp[i-1]$ if a one-digit code is valid; add $dp[i-2]$ if a two-digit code is in `10..26`.

Critical boundaries: leading / embedded `'0'`. A lone `'0'` cannot decode; `'10'`/`'20'` are valid, `'30'`-like pairs are not. Full derivation, space optimization, and alternate writeups live in [CoreSkills10 Decode Ways Dynamic Programming](./CoreSkills10%20Decode%20Ways%20Dynamic%20Programming.en.md).

### Coin Change

Unbounded knapsack minimization: `dp[a]` = fewest coins to make amount `a`.

$$dp[a] = \min_{c\le a}(dp[a-c] + 1)$$

Base: $dp[0]=0$, others $+\infty$. Return `-1` if unreachable.

```python
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        INF = amount + 1
        dp = [INF] * (amount + 1)
        dp[0] = 0
        for a in range(1, amount + 1):
            for c in coins:
                if c <= a:
                    dp[a] = min(dp[a], dp[a - c] + 1)
        return dp[amount] if dp[amount] != INF else -1
```

Complexity: Time $O(amount\cdot|coins|)$, Space $O(amount)$.

Pitfall: using `0` as "undefined" collides with "zero coins make 0"; initialize with `inf`.

### Maximum Product Subarray

Negatives flip ordering, so maintain both the max and min products ending here:

```text
max_here' = max(x, max_here*x, min_here*x)
min_here' = min(x, max_here*x, min_here*x)
```

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

Complexity: Time $O(n)$, Space $O(1)$.

Pitfall: tracking only max fails on `[-2, 3, -4]` (gets 3 instead of 24); a `0` naturally restarts from the current element.

### Word Break

`dp[i]` = whether prefix `s[:i]` can be segmented into dictionary words.

$$dp[i] = \bigvee_{j<i}\bigl(dp[j] \land s[j:i]\in dict\bigr)$$

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        words = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in words:
                    dp[i] = True
                    break
        return dp[n]
```

Complexity: Time $O(n^2\cdot L)$ ($L$ = slice compare cost; a trie helps), Space $O(n)$.

Pitfall: $dp[0]=True$ (empty prefix); put `wordDict` in a `set` when the dictionary is large.

### Longest Increasing Subsequence

`dp[i]` = LIS length ending at `nums[i]`.

$$dp[i] = 1 + \max_{j<i,\,nums[j]<nums[i]} dp[j]$$

(If no smaller predecessor, stay at `1`.) Answer $\max_i dp[i]$.

```python
class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

Complexity: Time $O(n^2)$, Space $O(n)$.

Footnote: patience sorting + binary search reaches $O(n\log n)$; that is a separate optimization line. This note focuses on the $O(n^2)$ DP table.

Pitfall: the answer is `max(dp)`, not `dp[-1]`; strict increase uses `<`, not `<=`.

---

## Interval DP

### Longest Palindromic Substring

Boolean interval DP: $dp[i][j]$ means whether $s[i..j]$ is a palindrome.

$$dp[i][j] = (s[i]=s[j]) \land (j-i<2 \lor dp[i+1][j-1])$$

Fill by increasing length, and record the longest interval start and length. Center expansion also works; here the point is the interval-DP shape.

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

(Equivalent: outer loop over `length` ascending.)

Complexity: Time $O(n^2)$, Space $O(n^2)$.

Pitfall: the transition needs $dp[i+1][j-1]$, so shorter intervals must be done first; single characters count toward the initial longest length.

### Palindromic Substrings

Same boolean table; the answer is the count of cells with $dp[i][j]=True$.

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

Complexity: Time $O(n^2)$, Space $O(n^2)$.

Pitfall: do not only count the "longest" palindrome from center expansion; every palindromic substring counts once.

### Burst Balloons

Pad a virtual balloon `1` on both ends. Define an open interval: $dp[i][j]$ = max coins from bursting every balloon **strictly between** $i$ and $j$.

Key: enumerate the balloon $k$ burst **last** inside the interval (not first). When $k$ is last, left and right are already empty, so the neighbors are exactly $a[i]$ and $a[j]$:

$$dp[i][j] = \max_{i<k<j}\bigl(dp[i][k] + a[i]\cdot a[k]\cdot a[j] + dp[k][j]\bigr)$$

Fill by increasing $j-i$. Answer $dp[0][n+1]$.

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

Complexity: Time $O(n^3)$, Space $O(n^2)$.

Pitfall: thinking "who to burst first" makes subproblem boundaries depend on unburst balloons; "who is last" splits the range into independent halves.

---

## Knapsack family

### Partition Equal Subset Sum

Check that the total is even; target $target=sum/2$. 0/1 feasibility: `dp[j]` = whether some subset sums to `j`.

```python
class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False
        target = total // 2
        dp = [False] * (target + 1)
        dp[0] = True
        for x in nums:
            for j in range(target, x - 1, -1):
                dp[j] = dp[j] or dp[j - x]
        return dp[target]
```

Complexity: Time $O(n\cdot target)$, Space $O(target)$.

Pitfall: capacity must go **descending**; ascending reuses the same `x` in one round and becomes unbounded knapsack.

### Coin Change II

Unbounded knapsack **combination count**: `dp[a]` = number of ways to make `a`.

```python
class Solution:
    def change(self, amount: int, coins: list[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for c in coins:                 # items outer -> combinations
            for a in range(c, amount + 1):
                dp[a] += dp[a - c]
        return dp[amount]
```

Versus Coin Change (fewest coins): different objective (`min` vs `+`). Versus the wrong loop (amount outer, coins inner): `(1,2)` and `(2,1)` would both count, i.e. permutations.

Complexity: Time $O(amount\cdot|coins|)$, Space $O(amount)$.

Pitfall: swapped loop order; `dp[0]=1` means "one way to make 0".

### Target Sum

Assign `+` or `-` to each number so the expression equals `target`. Let $P$ be the sum of the positive group and $N$ the absolute sum of the negative group:

$$P+N=\mathrm{sum},\quad P-N=\mathrm{target} \implies P=\frac{\mathrm{sum}+\mathrm{target}}{2}$$

So the problem becomes: how many subsets sum to $P$ (0/1 counting knapsack). If $(sum+target)$ is odd or $|target|>sum$, the answer is 0.

```python
class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        total = sum(nums)
        if (total + target) % 2 or abs(target) > total:
            return 0
        subset = (total + target) // 2
        dp = [0] * (subset + 1)
        dp[0] = 1
        for x in nums:
            for j in range(subset, x - 1, -1):
                dp[j] += dp[j - x]
        return dp[subset]
```

Complexity: Time $O(n\cdot sum)$, Space $O(sum)$.

Pitfall: with zeros, `+/-` on `0` are two contributions; descending 0/1 counting handles this (`dp[j]+=dp[j-0]` doubles); ascending would be wrong.

---

## Two-sequence DP

### Longest Common Subsequence

$$dp[i][j]=\begin{cases}dp[i-1][j-1]+1 & text1[i-1]=text2[j-1]\\ \max(dp[i-1][j],dp[i][j-1]) & \text{otherwise}\end{cases}$$

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

Complexity: Time $O(mn)$, Space $O(mn)$ (can roll to $O(\min(m,n))$).

Pitfall: on equality you must take the diagonal `+1`, not `max(up,left)+1` (that double-counts).

### Interleaving String

`dp[i][j]` = whether `s1[:i]` and `s2[:j]` can interleave into `s3[:i+j]`.

$$dp[i][j] = (dp[i-1][j]\land s1[i-1]=s3[i+j-1]) \lor (dp[i][j-1]\land s2[j-1]=s3[i+j-1])$$

First check $len(s1)+len(s2)=len(s3)$.

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

Complexity: Time $O(mn)$, Space $O(mn)$.

Pitfall: skipping the length check; the `s3` index is $i+j-1$, not $i+j$.

### Distinct Subsequences

`dp[i][j]` = number of subsequences of `s[:i]` equal to `t[:j]`.

$$dp[i][j]=\begin{cases}dp[i-1][j-1]+dp[i-1][j] & s[i-1]=t[j-1]\\ dp[i-1][j] & \text{otherwise}\end{cases}$$

Base: $dp[i][0]=1$ (one way to match empty `t`), $dp[0][j]=0$ for $j>0$.

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[m][n]
```

Complexity: Time $O(mn)$, Space $O(mn)$.

Pitfall: on equality add both "use this char" and "skip this char"; omitting `dp[i-1][j]` undercounts.

### Edit Distance

$$dp[i][j]=\begin{cases}dp[i-1][j-1] & word1[i-1]=word2[j-1]\\ 1+\min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]) & \text{otherwise}\end{cases}$$

The three branches are delete / insert / replace.

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

Complexity: Time $O(mn)$, Space $O(mn)$.

Pitfall: base cases are the cost of deleting / inserting a whole prefix ($i$ / $j$); on equality do not add `1`.

### Regular Expression Matching

`dp[i][j]` = whether `s[:i]` is matched by `p[:j]`.

- Ordinary char or `.`: `dp[i][j]=dp[i-1][j-1]` when the char matches.
- `p[j-1]=='*'`: `x*` used zero times -> `dp[i][j-2]`; or `x` matches `s[i-1]` and `dp[i-1][j]` (consume one `s` char, stay on `x*`).

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2]
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]
                elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
        return dp[m][n]
```

Complexity: Time $O(mn)$, Space $O(mn)$.

Pitfall: `*` applies to the **previous** element; use $j-2$ / $p[j-2]$. Prefill row 0 for empty `s` against `a*`, `a*b*`, and similar.

---

## Grid DP

### Unique Paths

Only right or down. $dp[i][j]=dp[i-1][j]+dp[i][j-1]$; first row/column all 1.

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
```

Complexity: Time $O(mn)$, Space $O(n)$.

Pitfall: when $m=1$ or $n=1$ the answer is 1; the rolling-array init already covers that.

### Longest Increasing Path in a Matrix

For each cell, $dfs(r,c)$ = longest increasing path length starting there:

$$dfs(r,c)=1+\max_{\text{strictly larger neighbor}} dfs(nr,nc)$$

(If no legal neighbor, 1.) Cache with memo.

```python
class Solution:
    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        memo = {}
        def dfs(r, c):
            if (r, c) in memo:
                return memo[(r, c)]
            best = 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and matrix[nr][nc] > matrix[r][c]:
                    best = max(best, 1 + dfs(nr, nc))
            memo[(r, c)] = best
            return best
        return max(dfs(i, j) for i in range(m) for j in range(n))
```

Complexity: Time $O(mn)$ (each cell once), Space $O(mn)$.

Why this is still DP: strictly ascending edges form a DAG; subproblems $dfs(r,c)$ overlap and are acyclic, so memo is the DP table. A plain double loop fails because a "larger neighbor" can sit in any direction; there is no single legal fill order.

Pitfall: using `>=` creates cycles; forgetting memo explodes exponentially.

---

## State-machine DP

### Best Time to Buy and Sell Stock with Cooldown

After a sell, the next day cannot buy. Three states (end of day $i$):

- `hold`: holding a share
- `sold`: just sold today (entering cooldown)
- `free`: empty-handed and allowed to buy (cooldown already finished)

Transitions:

$$
\begin{aligned}
hold' &= \max(hold,\ free - price)\\
sold' &= hold + price\\
free' &= \max(free,\ sold)
\end{aligned}
$$

Answer $\max(sold, free)$ (ending while still holding is never better than having sold earlier in an optimal plan you already tracked).

```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        hold, sold, free = float("-inf"), 0, 0
        for price in prices:
            hold, sold, free = (
                max(hold, free - price),
                hold + price,
                max(free, sold),
            )
        return max(sold, free)
```

Complexity: Time $O(n)$, Space $O(1)$.

Pitfall: feeding the same day's new `sold` into `free` mixes states; the tuple update above reads only old values.

---

## When to use which skeleton

```text
One index, depends on a few prior cells     -> 1D linear
Substring / open interval, fill by length   -> interval DP
Choose items to hit a capacity              -> knapsack family (0/1 vs unbounded; feasibility / opt / count)
Two string prefixes                         -> two-sequence
Grid with fixed move directions             -> grid tabulation
Grid whose "better neighbor" has no order   -> memoized DFS
A few mutually exclusive states per day     -> state machine
```

## Common pitfalls

- Mixing up `dp[n]` vs `max(dp)` (LIS, Max Product, Min Cost Climbing).
- Updating 0/1 knapsack ascending so an item is reused.
- Coin Change II with amount outer / coins inner, counting permutations.
- Interval DP not filled by length, reading uncomputed `dp[i+1][j-1]`.
- Burst Balloons thinking "burst first".
- Regex `*` forgetting it binds the previous char, or missing empty-string bases for `a*`.
- House Robber II forgetting `n=1`.
- Longest Increasing Path filled with a naive double loop.

## Quick self-check

```quiz
title: Quick quiz 1
question: Relative to House Robber, what is the key extra step in House Robber II?
answer: B
A. Switch to a 2D dp[i][j]
B. Split the ring into two linear Robber runs and take max
C. Must use interval DP
D. Switch to unbounded knapsack
explanation: First and last are adjacent, so they cannot both be robbed; run linear Robber on nums[0..n-2] and nums[1..n-1], then take max.
```

```quiz
title: Quick quiz 2
question: For Coin Change (fewest coins) vs Coin Change II (combinations) on 1D DP, the most important difference is?
answer: C
A. One must descend, the other ascend (when both are unbounded)
B. One must be 2D and the other 1D
C. Different objectives (min vs additive count), and II needs items-outer for combination semantics
D. Transitions are identical; only the return value differs
explanation: Both are unbounded-knapsack shaped; II needs combinations not permutations, so coins loop outer; transitions are min vs additive counting.
```

```quiz
title: Quick quiz 3
question: In the Burst Balloons interval transition, what does k mean?
answer: A
A. The balloon burst last inside open interval (i,j)
B. The balloon burst first inside the open interval
C. The interval length
D. The index of a padded boundary 1
explanation: When k is last, left and right are empty, so the gain splits into a[i]*a[k]*a[j] plus the two sub-interval DPs.
```

```quiz
title: Quick quiz 4
question: When Target Sum reduces to subset sum, the subset target P equals?
answer: B
A. (sum - target) / 2
B. (sum + target) / 2
C. sum - target
D. target
explanation: P+N=sum and P-N=target imply P=(sum+target)/2; need divisibility and |target|<=sum.
```

```quiz
title: Quick quiz 5
question: Why can Longest Increasing Path in a Matrix not be filled like Unique Paths by row sweep?
answer: D
A. Because moves are only right and down
B. Because it must use O(1) space
C. Because it is not DP
D. Larger neighbors can face any direction, so there is no single legal fill order; use memoized DFS on the DAG
explanation: Strictly ascending edges form a DAG; memoized DFS is the correct DP form.
```

```quiz
title: Quick quiz 6
question: In Edit Distance, when word1[i-1]==word2[j-1], the correct transition is?
answer: A
A. dp[i][j] = dp[i-1][j-1]
B. dp[i][j] = dp[i-1][j-1] + 1
C. dp[i][j] = max(dp[i-1][j], dp[i][j-1])
D. dp[i][j] = dp[i][j-1] + 1
explanation: Characters already match, so no op; copy the diagonal. Only on mismatch take min of insert/delete/replace plus one.
```

```quiz
title: Quick quiz 7
question: Why does Maximum Product Subarray also keep min_here?
answer: C
A. To handle zeros
B. To get O(1) space
C. A sign flip can turn the smallest product into the largest; max alone loses solutions
D. The problem asks for the minimum product
explanation: Example [-2,3,-4]; a negative times a previous most-negative product yields the global max.
```

```quiz
title: Quick quiz 8
question: For Distinct Subsequences, when s[i-1]==t[j-1], the transition should be?
answer: B
A. Only add dp[i-1][j-1]
B. dp[i-1][j-1] + dp[i-1][j] (use or skip the current char)
C. dp[i][j-1] + dp[i-1][j]
D. max(dp[i-1][j-1], dp[i-1][j])
explanation: Either use the current s char to match t's last char, or skip it; add both way counts.
```

```quiz
title: Quick quiz 9
question: In Regex Matching, when p[j-1]=='*', "match zero times" corresponds to?
answer: A
A. dp[i][j-2]
B. dp[i-1][j]
C. dp[i-1][j-1]
D. dp[i][j-1]
explanation: Drop the whole x* token and ask whether p[:j-2] already matches s[:i]; multiple matches OR in dp[i-1][j].
```

```quiz
title: Quick quiz 10
question: Why must the capacity loop go descending in 1D 0/1 knapsack?
answer: C
A. It is faster
B. To turn combinations into permutations
C. To avoid reusing the same item in the current round
D. Descending is required for unbounded knapsack
explanation: Descending keeps dp[j-x] as the "without current item" old value; ascending becomes unbounded knapsack.
```

```quiz
title: Quick quiz 11
question: Which meaning of the sold state in Stock with Cooldown is most accurate?
answer: B
A. Any empty-handed day
B. Just sold today, so tomorrow is in cooldown
C. Holding a share
D. Cumulative number of sells
explanation: sold marks "sold today"; the next day can only enter free, not buy immediately.
```

```quiz
title: Quick quiz 12
question: Which skeleton does Decode Ways belong to here, and where are the full details?
answer: A
A. 1D linear DP; full derivation in CoreSkills10
B. Interval DP; see Burst Balloons in this note
C. Two-sequence DP; see LCS
D. State-machine DP; see Stock Cooldown
explanation: Prefix/suffix way counts are classic 1D; leading-zero edges are covered in CoreSkills10.
```
