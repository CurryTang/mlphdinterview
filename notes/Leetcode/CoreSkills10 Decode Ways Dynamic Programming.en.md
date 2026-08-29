# Dynamic Programming: From Recurrence to Space Optimization

## Interview Goal

Dynamic Programming is not about "memorizing templates." It is about breaking a problem into stable subproblems and then computing the answers in dependency order.

A reusable DP workflow usually has three steps:

1. Write the recurrence.
2. Convert the recurrence into a loop pattern.
3. Optimize space based on the dependency relationships.

These three steps matter more than writing code directly. As long as the recurrence is clear, the code is usually just a mechanical translation.

## When to Think of DP

When a problem has the following signals, you should consider DP first:

- It asks for "number of ways / minimum cost / maximum profit / feasibility."
- The decision process can be split into stages, such as processing up to the `i`-th character, the `i`-th item, or the `i`-th day.
- The same subproblem is encountered repeatedly.
- The current answer can be derived from smaller-scale answers.

The essence of DP is:

```text
state + transition + base case + iteration order
```

That is:

- State: what does `dp[...]` represent?
- Transition: which previous states does the current state come from?
- Initial value: what is the answer for the smallest subproblem?
- Order: what do you compute first, and what do you compute later?

## Step 1: Write the Recurrence

Do not write loops yet. First define the state in one sentence.

Common state forms:

```text
dp[i]       = the answer for the first i elements / the suffix starting at i
dp[i][j]    = the answer determined jointly by two dimensions
dp[i][state]= the answer when being in some state at step i
```

When defining the state, ask two questions:

1. Can it directly express the final answer?
2. Can it transition from smaller states?

Then write the choice. The recurrence in every DP problem is basically:

```text
dp[current] = combine(dp[previous states])
```

If you are counting the number of ways, `combine` is usually addition:

```text
dp[i] = dp[a] + dp[b]
```

If you are finding an optimal value, `combine` is usually `min` or `max`:

```text
dp[i] = min(dp[a] + costA, dp[b] + costB)
```

If you are checking feasibility, `combine` is usually `or`:

```text
dp[i] = dp[a] or dp[b]
```

Finally, add the base case. The base case is the starting point of the recurrence, not a minor detail. Many DP bugs come from not thinking the base case through clearly.

## Step 2: Convert It into a Loop Pattern

Once the recurrence is written, the loop direction is determined by the dependency relationships.

If:

```text
dp[i] depends on dp[i - 1]
```

you usually go from left to right.

If:

```text
dp[i] depends on dp[i + 1]
```

you usually go from right to left.

If:

```text
dp[i][j] depends on dp[i - 1][j] and dp[i][j - 1]
```

you usually use nested loops and fill the table from small to large.

When converting the recurrence into loops, you can follow this template:

```text
initialize dp
set base cases
for state in valid order:
  compute dp[state] from previous states
return answer
```

Note: loop order is not a style issue, but a correctness issue. You must ensure that the previous states required by the current state have already been computed.

## Step 3: Space Optimization

Do not optimize space at the beginning. First write the full DP clearly, then observe the dependency range.

If:

```text
dp[i] only depends on dp[i - 1]
```

you can keep only one variable.

If:

```text
dp[i] only depends on dp[i + 1] and dp[i + 2]
```

you can keep only two variables.

If a 2D DP depends only on the previous row:

```text
dp[i][j] depends on dp[i - 1][...]
```

you can use a rolling array and reduce space from `O(nm)` to `O(m)`.

The principle of space optimization is:

> Keep only the states that will still be used in the future.

Do not destroy the semantics of the states just to save space. In interviews, it is better to explain the full DP first, then explain how to optimize it.

## Six skeletons at a glance: classify before coding

The three steps above explain how to turn a recurrence you already understand into code. An earlier question is: what should the state look like? DP problems look numerous, but state shapes really come in only six forms. When you get a problem, ask first: is the state an index, a closed interval, a knapsack capacity, prefixes of two sequences, a grid cell, or a small set of named states?

```text
What is the state dimension?

├── one index i (first i elements / climb to step i)
│     → Skeleton A: 1D linear DP
│       Examples: Climbing Stairs, House Robber, Decode Ways, Word Break, LIS, Coin Change, Max Product Subarray
│
├── one closed interval [i, j] (substring / balloon interval), fill by increasing length
│     → Skeleton B: interval DP
│       Examples: Palindrome family, Burst Balloons
│
├── items + capacity (take or not / reusable / counting)
│     → Skeleton C: 0/1 knapsack / complete knapsack / counting knapsack
│       Examples: Partition Equal Subset Sum, Coin Change II, Target Sum
│
├── prefixes of two sequences (i, j)
│     → Skeleton D: two-sequence DP
│       Examples: LCS, Edit Distance, Interleaving String, Distinct Subsequences, Regex Matching
│
├── grid cell (r, c) with a fixed sweep direction
│     → Skeleton E: grid DP
│       Examples: Unique Paths
│       Exception: Longest Increasing Path has no fixed direction → memoized DFS
│
└── a few named states at each step (hold / cooldown / free)
      → Skeleton F: state-machine DP
        Examples: Stock with Cooldown
```

(Kadane's maximum subarray sum and Jump Game also appear later in this note, but their focus is how DP compresses into Greedy, so they do not get their own skeleton.)

## Skeleton A: 1D linear DP

```python
def solve_1d(arr):
    n = len(arr)
    dp = [INIT] * (n + 1)   # or n, depending on index convention
    # BASE: fill dp[0] / dp[1] / ...
    for i in range(START, n):
        dp[i] = TRANSITION(dp, i, arr)   # depends only on O(1)~O(k) earlier states
    return ANSWER(dp)                    # dp[n] or max(dp), etc.
```

| Problem | State | Transition | base | Iteration | Answer |
|---|---|---|---|---|---|
| Climbing Stairs | `dp[i]` = ways to reach step i | `dp[i]=dp[i-1]+dp[i-2]` | `dp[0]=1,dp[1]=1` | `i=2..n` | `dp[n]` |
| Min Cost Climbing Stairs | `dp[i]` = min cost to reach i | `dp[i]=cost[i]+min(dp[i-1],dp[i-2])` | `dp[0]=cost[0],dp[1]=cost[1]` | `i=2..n-1` | `min(dp[n-1],dp[n-2])` |
| House Robber | `dp[i]` = max money considering first i houses | `dp[i]=max(dp[i-1],dp[i-2]+nums[i-1])` | `dp[0]=0,dp[1]=nums[0]` | `i=2..n` | `dp[n]` |
| House Robber II | same, split the ring into two linear runs | run Robber on `[0..n-2]` and `[1..n-1]` | same as linear Robber | two linear passes | `max` of the two answers |
| Decode Ways | `dp[i]` = ways for suffix `s[i:]` | valid 1-digit `+dp[i+1]`; valid 2-digit (`10..26`) `+dp[i+2]` | `dp[n]=1` | right to left | `dp[0]` |
| Word Break | `dp[i]` = whether `s[:i]` can be segmented | `dp[j] and s[j:i] in dict` | `dp[0]=True` | `i=1..n`, inner `j` | `dp[n]` |
| Longest Increasing Subsequence | `dp[i]` = LIS length ending at `i` | `dp[i]=max(dp[j])+1` (`nums[j]<nums[i]`) | all `1` | nested `i,j<i` | `max(dp)` |
| Coin Change | `dp[a]` = fewest coins to make amount `a` | `dp[a]=min(dp[a-c])+1` | `dp[0]=0`, others `inf` | amounts ascending | `dp[amount]` (`inf`→`-1`) |
| Maximum Product Subarray | max/min product ending at `i` | roll `max_here,min_here` together | first element | one pass | global `max` |

Below we first walk through Decode Ways end to end, then fill in the remaining problems from this table.

## Skeleton B: interval DP

```python
def solve_interval(s):
    n = len(s)
    dp = [[INIT] * n for _ in range(n)]
    # BASE: length 1 (sometimes also length 0 / 2)
    for length in range(2, n + 1):       # increase by interval length
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i][j] = TRANSITION(dp, i, j, s)
    return ANSWER(dp)
```

| Problem | State | Transition | base | Iteration | Answer |
|---|---|---|---|---|---|
| Longest Palindromic Substring | `dp[i][j]` = whether `s[i..j]` is a palindrome | ends equal and (length≤2 or `dp[i+1][j-1]`) | single chars `True` | by length | record longest `(i,j)` |
| Palindromic Substrings | same boolean table | same | same | same | count `True` cells |
| Burst Balloons | `dp[i][j]` = max coins bursting open interval `(i,j)` | `max_k a[i]*a[k]*a[j]+dp[i][k]+dp[k][j]` (`k` last) | adjacent `i,j` are 0 | length↑; pad ends with `1` | `dp[0][n+1]` |

## Skeleton C: knapsack family

```python
def solve_knapsack(items, capacity):
    dp = [INIT] * (capacity + 1)
    dp[0] = BASE0
    for x in items:                      # items outside → combinations; amount outside → permutations
        for j in range(...):             # 0/1: descending; complete: ascending
            dp[j] = COMBINE(dp[j], dp[j - x], x)
    return ANSWER(dp)
```

| Problem | State | Transition | base | Iteration | Answer |
|---|---|---|---|---|---|
| Partition Equal Subset Sum | `dp[j]` = whether sum `j` is reachable | `dp[j]\|=dp[j-x]` (0/1) | `dp[0]=True` | items outside, capacity descending | `dp[sum/2]` |
| Coin Change II | `dp[a]` = number of combinations for `a` | `dp[a]+=dp[a-c]` (complete) | `dp[0]=1` | coins outside, amounts ascending | `dp[amount]` |
| Target Sum | reduce to subset-sum counting | same as 0/1 counting knapsack | `dp[0]=1` | items outside, capacity descending | `dp[(sum+target)/2]` |

The three problems in this table, plus Coin Change later, are derived in the "0/1 knapsack" and "complete knapsack" sections below.

## Skeleton D: two-sequence DP

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

| Problem | State | Transition highlights | base |
|---|---|---|---|
| LCS | `dp[i][j]` = LCS length of `a[:i]` and `b[:j]` | equal → `+1`, else `max(up,left)` | row/col 0 = 0 |
| Edit Distance | min ops to turn `a[:i]` into `b[:j]` | equal copy diagonal; else insert/delete/replace +1 | `dp[i][0]=i`,`dp[0][j]=j` |
| Interleaving String | whether `s1[:i]+s2[:j]` can form `s3[:i+j]` | take from above via `s1` or from left via `s2` | empty matches empty prefix |
| Distinct Subsequences | how many subsequences of `s[:i]` equal `t[:j]` | equal chars: use or skip; else only skip | `dp[i][0]=1` |
| Regex Matching | whether `s[:i]` matches `p[:j]` | `.` any one char; `*` zero times (`j-2`) or more (if match, `i-1`) | empty pattern; `x*` can match empty |

## Skeleton E: grid DP (with memoization exception)

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

Longest Increasing Path in a Matrix **cannot** be filled with a single row/column sweep: matrix values are arbitrary, so there is no single topological sweep order. The correct approach is memoized DFS from each cell: only walk to strictly larger neighbors, which implicitly forms a DAG (acyclic, overlapping subproblems cacheable). This is still DP (optimal substructure + overlapping subproblems + memo); you just do not write an explicit fill loop.

## Skeleton F: state-machine DP

```python
# hold / sold(cooldown) / free
for price in prices:
    hold, sold, free = TRANSITIONS(...)
return max(sold, free)
```

See the three-state equations in the Stock Cooldown section below.

## Example: Decode Ways

Now let us go through the full process with Decode Ways.

Problem: given a string `s` containing only digits, with the mapping:

```text
1 -> A
2 -> B
...
26 -> Z
```

ask how many valid decoding ways there are.

For example:

```text
s = "226"

2 | 2 | 6   -> B B F
22 | 6      -> V F
2 | 26      -> B Z

answer = 3
```

## Decode Ways: Recurrence

Define the state:

```text
dp[i] = the number of decoding ways for the suffix string s[i:]
```

Final answer:

```text
dp[0]
```

Base case:

```text
dp[n] = 1
```

The meaning is: if you have already reached the end of the string, then the earlier choices have formed one complete decoding.

If the current position is `'0'`:

```text
dp[i] = 0
```

because `0` cannot map to any letter by itself.

Otherwise there are two types of choices at the current position:

```text
take one digit  -> dp[i + 1]
take two digits -> dp[i + 2], if 10 <= s[i:i+2] <= 26
```

Recurrence:

```text
if s[i] == '0':
  dp[i] = 0
else:
  dp[i] = dp[i + 1]
  if i + 1 < n and 10 <= int(s[i:i+2]) <= 26:
    dp[i] += dp[i + 2]
```

This step is the most important one. The loop and the space optimization that follow are just implementations of this recurrence.

## Decode Ways: Loop Pattern

Because `dp[i]` depends on `dp[i + 1]` and `dp[i + 2]`, you must compute from right to left.

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        dp[n] = 1

        for i in range(n - 1, -1, -1):
            if s[i] == "0":
                dp[i] = 0
                continue

            dp[i] = dp[i + 1]

            if i + 1 < n and (
                s[i] == "1" or
                (s[i] == "2" and s[i + 1] in "0123456")
            ):
                dp[i] += dp[i + 2]

        return dp[0]
```

Walk through `"226"` once:

```text
dp[3] = 1

i = 2, "6":
  dp[2] = dp[3] = 1

i = 1, "26":
  dp[1] = dp[2] + dp[3] = 2

i = 0, "22":
  dp[0] = dp[1] + dp[2] = 3
```

## Decode Ways: Space Optimization

Observe the dependencies:

```text
dp[i] only depends on dp[i + 1] and dp[i + 2]
```

So you only need two variables:

```text
one = dp[i + 1]
two = dp[i + 2]
```

Each time you compute:

```text
cur = dp[i]
```

Then shift the window left:

```text
two = one
one = cur
```

Code:

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        one = 1
        two = 0

        for i in range(len(s) - 1, -1, -1):
            if s[i] == "0":
                cur = 0
            else:
                cur = one

                if i + 1 < len(s) and (
                    s[i] == "1" or
                    (s[i] == "2" and s[i + 1] in "0123456")
                ):
                    cur += two

            two = one
            one = cur

        return one
```

## Climbing Stairs

State: `dp[i]` = number of ways to climb to step `i`.

$$dp[i] = dp[i-1] + dp[i-2]$$

Base: `$dp[0]=1$`, `$dp[1]=1$` (or directly `$dp[1]=1,dp[2]=2$`).

Iteration: `$i=2..n$`. Answer: `$dp[n]$`.

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

Complexity: Time `$O(n)$`, Space `$O(1)$`.

Pitfall: do not touch `$dp[2]$` when `$n=1$`; this is isomorphic to Fibonacci, which is enough to say in an interview.

## Min Cost Climbing Stairs

You may start at index `0` or `1`. `dp[i]` = minimum total cost after reaching step `i` and paying `cost[i]`.

$$dp[i] = cost[i] + \min(dp[i-1], dp[i-2])$$

The answer is not `$dp[n]$` (the top has no step and thus no cost), but `$\min(dp[n-1], dp[n-2])$`: the last step can come from the penultimate or antepenultimate stair.

```python
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        n = len(cost)
        a, b = cost[0], cost[1]
        for i in range(2, n):
            a, b = b, cost[i] + min(a, b)
        return min(a, b)
```

Complexity: Time `$O(n)$`, Space `$O(1)$`.

Pitfall: forgetting "the top has no cost" and returning `$dp[n-1]$`.

## House Robber

You cannot rob adjacent houses. `dp[i]` = maximum money considering only the first `i` houses.

$$dp[i] = \max(dp[i-1], dp[i-2] + nums[i-1])$$

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        prev2, prev1 = 0, 0
        for x in nums:
            prev2, prev1 = prev1, max(prev1, prev2 + x)
        return prev1
```

Complexity: Time `$O(n)$`, Space `$O(1)$`.

Pitfall: empty / single-element arrays need care; whether the index is `nums[i-1]` or `nums[i]` must stay consistent with the `dp` length convention.

## House Robber II

Houses form a ring: the first and last are adjacent and cannot both be robbed. Split into two linear House Robber runs and take the larger:

```text
max( rob(nums[0..n-2]), rob(nums[1..n-1]) )
```

When `$n=1$`, return `$nums[0]$` directly.

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

Complexity: Time `$O(n)$`, Space `$O(1)$`.

Pitfall: missing the `$n=1$` special case; or slicing the two ranges wrong and skipping or double-counting a middle house.

## Word Break

`$dp[i]$` = whether prefix `$s[:i]$` can be segmented into dictionary words.

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

Complexity: Time `$O(n^2\cdot L)$` (`$L$` is the cost of slice comparison; a trie can help), Space `$O(n)$`.

Pitfall: `$dp[0]=True$` means the empty prefix is always segmented; convert `wordDict` to a `set` when the dictionary is large, otherwise `in` is a linear scan.

## Longest Increasing Subsequence

`$dp[i]$` = length of the longest strictly increasing subsequence ending at `$nums[i]$`.

$$dp[i] = 1 + \max_{j<i,\,nums[j]<nums[i]} dp[j]$$

(Use `1` when no smaller predecessor exists.) The answer is `$\max_i dp[i]$`, not `$dp[n-1]$`.

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

Complexity: Time `$O(n^2)$`, Space `$O(n)$`.

Footnote: patience sorting + binary search reaches `$O(n\log n)$`; that is a separate optimization track. Here we give the standard `$O(n^2)$` DP table so the state and transition are clear first.

Pitfall: the answer is `max(dp)`, not `dp[-1]`; "strictly increasing" uses `<`, not `<=`.

## Maximum Product Subarray

It looks like a variant of Kadane's maximum subarray **sum**, but multiplication can flip signs on negatives, so you must maintain both the maximum and minimum products ending at the current position:

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

Complexity: Time `$O(n)$`, Space `$O(1)$`.

Pitfall: keeping only `max_here` misses the answer `24` on inputs like `[-2, 3, -4]` (two negatives multiply to a positive); when you hit `0`, `max_here`/`min_here` naturally reset to restart from the current element, so no extra special case is needed.

## Example: Best Time to Buy and Sell Stock with Cooldown

This problem is even better than Decode Ways for practicing "state machine DP."

Problem: given daily stock prices `prices`, you may buy and sell multiple times, but you may hold at most one share at any time. After selling, you cannot buy on the next day because there is a one-day cooldown. Find the maximum profit.

For example:

```text
prices = [1, 2, 3, 0, 2]
answer = 3

One optimal sequence of operations:
day 0 buy at 1
day 1 sell at 2
day 2 cooldown
day 3 buy at 0
day 4 sell at 2
profit = 1 + 2 = 3
```

This problem mainly practices two recording techniques:

1. First write down the states clearly. It is best to draw them as a state machine, then write the state equations.
2. Then notice that each day depends only on the previous day, so you can compress the DP table into constant space.

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

## Knapsack Problems: 0/1 Knapsack vs Complete Knapsack

The core question in knapsack DP is:

```text
There are some items, and each item has a cost / weight / value.
Under a capacity limit, find feasibility, maximum value, minimum count, or number of ways.
```

In interviews, the most important distinction is:

```text
0/1 knapsack: each item can be used at most once
complete knapsack: each item can be used infinitely many times
```

The code for these two problems often differs by only one loop direction, but the semantics are completely different.

## 0/1 Knapsack Example: Partition Equal Subset Sum

Before Target Sum, look at a more direct 0/1 knapsack feasibility problem: given an array, can you split it into two subsets with equal sums?

If the total is odd, there is no solution. Otherwise the goal is to pick some numbers (each at most once) that sum to `$target = sum / 2$`:

$$dp[j] = dp[j] \lor dp[j - x]$$

`$dp[j]$` means "whether sum `$j$` is reachable"; this is a boolean 0/1 knapsack.

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

Complexity: Time `$O(n\cdot target)$`, Space `$O(target)$`.

Pitfall: as with Target Sum below, the capacity loop must be descending; otherwise the same `x` can be used multiple times in one round and it degrades into a complete knapsack.

## 0/1 Knapsack Example: Target Sum

Problem: given an integer array `nums` and an integer `target`. You must choose either a plus or minus sign for every number. Ask how many expressions can evaluate to `target`.

For example:

```text
nums = [2, 2, 2], target = 2

+2 +2 -2 = 2
+2 -2 +2 = 2
-2 +2 +2 = 2

answer = 3
```

On the surface this is about plus and minus signs, but it can actually be converted into a 0/1 knapsack problem.

Put all numbers chosen with a plus sign into set `P`, and all numbers chosen with a minus sign into set `N`:

```text
sum(P) - sum(N) = target
sum(P) + sum(N) = total
```

Add the two equations:

```text
2 * sum(P) = target + total
sum(P) = (target + total) / 2
```

So the problem becomes:

```text
Choose some numbers from nums, using each number at most once, so that their sum equals bag = (target + total) / 2.
Ask how many ways there are to choose them.
```

This is the "count the number of ways" version of 0/1 knapsack.

When is there no solution?

```text
abs(target) > total
target + total is odd
```

because `sum(P)` must be a nonnegative integer.

## Target Sum: 2D 0/1 Knapsack

Define the state:

```text
dp[i][s] = the number of ways to make sum s using only the first i numbers
```

Base case:

```text
dp[0][0] = 1
```

The meaning is: using no numbers, there is one way to make 0, which is to choose nothing.

Transition:

```text
do not choose nums[i - 1]: dp[i - 1][s]
choose nums[i - 1]: dp[i - 1][s - nums[i - 1]]

dp[i][s] = dp[i - 1][s] + dp[i - 1][s - nums[i - 1]]
```

Code:

```python
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if abs(target) > total or (target + total) % 2 == 1:
            return 0

        bag = (target + total) // 2
        n = len(nums)
        dp = [[0] * (bag + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            num = nums[i - 1]
            for s in range(bag + 1):
                dp[i][s] = dp[i - 1][s]
                if s >= num:
                    dp[i][s] += dp[i - 1][s - num]

        return dp[n][bag]
```

Example:

```text
nums = [2, 2, 2], target = 2
total = 6
bag = (2 + 6) / 2 = 4

The problem becomes: from [2, 2, 2], choose some numbers whose sum is 4. How many ways are there?

Choose the 1st and 2nd 2
Choose the 1st and 3rd 2
Choose the 2nd and 3rd 2

answer = 3
```

Note: these three `2`s are three numbers at different positions, so choosing the 1st and 2nd is a different solution from choosing the 1st and 3rd.

## Target Sum: 1D Space Optimization

The 2D transition depends only on the previous row:

```text
dp[i][s] depends on dp[i - 1][s] and dp[i - 1][s - num]
```

So it can be compressed to 1D:

```text
dp[s] = the number of ways to make sum s using the numbers processed so far
```

The 1D loop for 0/1 knapsack must go in reverse:

```text
for s from bag down to num:
```

Reason: each number can be used only once. Reverse order guarantees that `dp[s - num]` is still the result from the "previous item layer," not a result just updated by the current `num`.

Code:

```python
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if abs(target) > total or (target + total) % 2 == 1:
            return 0

        bag = (target + total) // 2
        dp = [0] * (bag + 1)
        dp[0] = 1

        for num in nums:
            for s in range(bag, num - 1, -1):
                dp[s] += dp[s - num]

        return dp[bag]
```

Why is reverse order so important?

Suppose there is only one number `2`, and `bag = 4`.

If you go forward:

```text
dp[2] += dp[0]  # used the 2 once
dp[4] += dp[2]  # immediately uses the just-updated dp[2], meaning the same 2 is used twice
```

Then it turns into complete knapsack, so the semantics are wrong.

## What Is Complete Knapsack

The difference in complete knapsack is:

```text
each item can be used infinitely many times
```

For example, coin change:

```text
coins = [1, 2, 5], amount = 11
```

Each type of coin can be used many times, so this is complete knapsack.

The 1D loop for complete knapsack usually goes forward:

```text
for coin in coins:
  for s from coin to amount:
    dp[s] = combine(dp[s], dp[s - coin])
```

The meaning of forward order is: the current `coin` is allowed to be used repeatedly. Because when you compute `dp[s]`, `dp[s - coin]` may already include results that used the current coin.

## Complete Knapsack: Minimum Number of Coins

If the task is to find the minimum number of coins:

```text
dp[s] = the minimum number of coins needed to make amount s
```

Transition:

```text
dp[s] = min(dp[s], dp[s - coin] + 1)
```

Code:

```python
from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        inf = amount + 1
        dp = [inf] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for s in range(coin, amount + 1):
                dp[s] = min(dp[s], dp[s - coin] + 1)

        return -1 if dp[amount] == inf else dp[amount]
```

Here `s` goes forward because one `coin` can be reused repeatedly.

## Complete Knapsack Example: Coin Change II

Coin Change asks for the fewest coins; Coin Change II is the counting version of complete knapsack: `$dp[a]$` = number of combinations that make amount `$a$` (not permutations).

```python
class Solution:
    def change(self, amount: int, coins: list[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for c in coins:                 # coins outside → combinations
            for a in range(c, amount + 1):
                dp[a] += dp[a - c]
        return dp[amount]
```

Complexity: Time `$O(amount\cdot|coins|)$`, Space `$O(amount)$`.

Pitfall: putting coins inside and amounts outside treats `(1, 2)` and `(2, 1)` as different plans and counts permutations instead of combinations. That is exactly the pitfall from the earlier section on "counting problems: combination vs permutation loop order," made concrete here. `dp[0]=1` means there is one way to make 0: choose nothing.

## How to Choose Between 0/1 Knapsack and Complete Knapsack

First ask whether items can be reused:

```text
each element can be chosen only once     -> 0/1 knapsack
each type of item can be chosen infinitely many times -> complete knapsack
each item can be chosen at most k times  -> bounded knapsack
```

Then ask what the goal is:

```text
whether a capacity can be made           -> boolean dp
how many ways there are                  -> count dp
maximum value / minimum count            -> max / min dp
```

Finally determine the loop direction:

```text
1D optimization for 0/1 knapsack:
for item in items:
  for capacity from target down to item:
    use previous item layer

1D optimization for complete knapsack:
for item in items:
  for capacity from item up to target:
    allow using the current item again
```

One-sentence memory aid:

> Reverse order for 0/1 knapsack to prevent reusing the same item; forward order for complete knapsack to explicitly allow reusing the same item.

## Knapsack Quick Reference: 1D Template, Initialization, Loop Order

Both worked examples above come wrapped in a problem statement. Strip the wrapper away and the 1D knapsack is only these two blocks:

```python
# 0/1 knapsack: each item used at most once, maximize value
dp = [0] * (W + 1)
for weight, value in items:
    for c in range(W, weight - 1, -1):     # descending
        dp[c] = max(dp[c], dp[c - weight] + value)
return dp[W]

# Complete knapsack: each item usable unlimited times, maximize value
dp = [0] * (W + 1)
for weight, value in items:
    for c in range(weight, W + 1):         # ascending
        dp[c] = max(dp[c], dp[c - weight] + value)
return dp[W]
```

The only difference is the direction of the inner loop. Going down, `dp[c - weight]` still holds the state from after the previous item was processed, so the current item is used at most once. Going up, `dp[c - weight]` may already include the current item, so the same item stacks.

The lower bound in `range(W, weight - 1, -1)` is `weight - 1` because `range` excludes its endpoint, which makes `weight` the last capacity actually visited. Capacities below `weight` cannot hold the item at all, so skipping them is correct.

Both versions run in $O(nW)$ time and $O(W)$ space. Note that `W` appears in the complexity: once the capacity reaches something like $10^9$, knapsack DP is off the table and the problem wants greedy, number theory, or search instead.

### Initialization Depends on "Exactly Full" vs "At Most"

The initial values of `dp` are not automatically 0. They encode whether capacities other than 0 start out legal:

| Goal | `dp[0]` | Other entries | Meaning |
|---|---|---|---|
| Max value, partial fill allowed | `0` | `0` | Any capacity can simply hold nothing |
| Max value, must fill exactly | `0` | `-inf` | This capacity is not reachable yet, so it is illegal |
| Min count, must fill exactly | `0` | `+inf` (or `amount + 1` as a sentinel) | Same idea, opposite direction |
| Number of ways | `1` | `0` | Capacity 0 is reached in exactly one way: pick nothing |
| Feasibility | `True` | `False` | Capacity 0 is always reachable |

The coin change code above uses `inf = amount + 1` rather than a true infinity because reaching `amount` takes at most `amount` coins of denomination 1. `amount + 1` already exceeds every legal answer, and it sidesteps any discussion of `inf + 1` overflow.

### Counting Problems: Combinations and Permutations Need Different Loop Orders

The counting variant hides one more trap: whether items or capacities sit in the outer loop decides whether you are counting combinations or permutations.

```python
# Combinations: {1,2} and {2,1} count once  -  items outside
for coin in coins:
    for s in range(coin, amount + 1):
        dp[s] += dp[s - coin]

# Permutations: {1,2} and {2,1} count twice  -  capacities outside
for s in range(1, amount + 1):
    for coin in coins:
        if coin <= s:
            dp[s] += dp[s - coin]
```

With items outside, each item is considered only during its own pass, which pins the selection order to the item enumeration order, so a given set is counted exactly once. With capacities outside, every coin gets a chance to be the "last coin" at the same capacity, which makes order significant. Coin Change II asks for combinations and Combination Sum IV asks for permutations; the code is nearly identical, and swapping the loop order produces the wrong answer.

### Common Pitfalls

- Wrong loop direction: writing a 0/1 knapsack in ascending order turns it into a complete knapsack, and items get picked repeatedly.
- Initialization that ignores the problem statement: filling the whole array with 0 when the problem demands an exact fill makes unreachable capacities look like valid solutions.
- Reversed loop order in a counting problem: combinations written as permutations passes small samples and overshoots on larger ones.
- Forgetting to skip cells where `weight > c`, or writing the descending lower bound as `-1`, which lets `dp[c - weight]` index negatively.
- Trying to recover which items were chosen after collapsing to 1D: the single array has already overwritten the intermediate layers, so reporting the actual selection requires keeping the 2D table.

## Two-sequence DP: advance two strings together

The state becomes two indices `$(i, j)$`, the prefix lengths of the two strings. Transitions usually come from three directions: `$dp[i-1][j-1]$` (both pointers move), `$dp[i-1][j]$` (only the first string), `$dp[i][j-1]$` (only the second string).

## Longest Common Subsequence

`$dp[i][j]$` = LCS length of `text1[:i]` and `text2[:j]`.

$$dp[i][j]=\begin{cases}dp[i-1][j-1]+1 & \text{text1}[i-1]=\text{text2}[j-1]\\ \max(dp[i-1][j],dp[i][j-1]) & \text{otherwise}\end{cases}$$

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

Complexity: Time `$O(mn)$`, Space `$O(mn)$` (can roll down to `$O(\min(m,n))$`).

Pitfall: when the two characters match you must take the diagonal `+1`; writing `max(up, left) + 1` double-counts the same character.

## Edit Distance

`$dp[i][j]$` = minimum operations to turn `word1[:i]` into `word2[:j]`.

$$dp[i][j]=\begin{cases}dp[i-1][j-1] & \text{word1}[i-1]=\text{word2}[j-1]\\ 1+\min(dp[i-1][j],\ dp[i][j-1],\ dp[i-1][j-1]) & \text{otherwise}\end{cases}$$

The three candidates are delete, insert, and replace.

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

Complexity: Time `$O(mn)$`, Space `$O(mn)$`.

Pitfall: the base case is the cost of deleting a prefix empty or inserting a prefix from empty, i.e. `$i$` and `$j$`; when characters match, copy the diagonal and do not add `1`.

## Interleaving String

`$dp[i][j]$` = whether `s1[:i]` and `s2[:j]` can interleave to form `s3[:i+j]`.

$$dp[i][j] = \bigl(dp[i-1][j]\land s1[i-1]=s3[i+j-1]\bigr) \lor \bigl(dp[i][j-1]\land s2[j-1]=s3[i+j-1]\bigr)$$

First check `$\text{len}(s1)+\text{len}(s2)=\text{len}(s3)$`; wrong lengths return `False` immediately.

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

Complexity: Time `$O(mn)$`, Space `$O(mn)$`.

Pitfall: forgetting the length check; the index into `s3` is `$i+j-1$`, not `$i+j$` (easy off-by-one with 0-indexing).

## Distinct Subsequences

`$dp[i][j]$` = how many subsequences of `s[:i]` equal `t[:j]`.

$$dp[i][j]=\begin{cases}dp[i-1][j-1]+dp[i-1][j] & s[i-1]=t[j-1]\\ dp[i-1][j] & \text{otherwise}\end{cases}$$

Base: `$dp[i][0]=1$` (empty `t` always matches once, i.e. "take nothing"), `$dp[0][j]=0$` (when `$j>0$`, nonempty `t` cannot match empty `s`).

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

Complexity: Time `$O(mn)$`, Space `$O(mn)$`.

Pitfall: when characters match, add both "use this char to match the end of t" and "skip this char of s"; dropping `dp[i-1][j]` undercounts.

## Regular Expression Matching

`$dp[i][j]$` = whether `s[:i]` is matched by `p[:j]`.

- Ordinary char or `.`: `$dp[i][j]=dp[i-1][j-1]$`, and the current characters must match.
- `$p[j-1]='*'$`: `$x*$` matching zero times looks at `$dp[i][j-2]$` (drop the whole `x*`); matching the current character, if `$x$` matches `$s[i-1]$`, also OR `$dp[i-1][j]$` (consume one `s` char and keep the pattern on `x*` for reuse).

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

Complexity: Time `$O(mn)$`, Space `$O(mn)$`.

Pitfall: `*` applies to the previous character, so transitions use `$j-2$` / `$p[j-2]$`, not `$j-1$`; for empty-string matches against patterns like `a*` and `a*b*`, fill row 0 correctly first.

## Kadane's Algorithm: Maximum Subarray Sum

Kadane's algorithm solves Maximum Subarray:

```text
Given an array nums, find a contiguous subarray with the largest sum.
```

For example:

```text
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
answer = 6

The optimal subarray is [4, -1, 2, 1]
sum = 6
```

This problem is often explained as greedy, but in essence it can also be viewed as a very concise 1D DP.

## Kadane: State Definition

Define the state:

```text
dp[i] = the maximum sum of a contiguous subarray that must end at nums[i]
```

Notice that it is "must end at `i`," not "the maximum answer among the first `i` elements." This definition makes the transition very stable.

For `nums[i]`, there are only two choices:

```text
1. Extend the previous subarray: dp[i - 1] + nums[i]
2. Start over from the current position: nums[i]
```

So the state equation is:

```text
dp[i] = max(dp[i - 1] + nums[i], nums[i])
```

Final answer:

```text
answer = max(dp[i] for all i)
```

Full DP version:

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        ans = nums[0]

        for i in range(1, n):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
            ans = max(ans, dp[i])

        return ans
```

## Kadane: Why a Negative Prefix Can Be Dropped

If the current running sum is negative:

```text
curSum < 0
```

then attaching it in front of any future subarray only makes the future sum smaller.

For example, if the future starts at `x`:

```text
curSum + x < x
```

So a negative prefix has no value to keep, and you can restart directly from the next position.

That is what this line means in Kadane:

```python
if curSum < 0:
    curSum = 0
```

## Kadane: Space-Optimized Version

Because `dp[i]` depends only on `dp[i - 1]`, you do not need the full array, only one variable:

```text
curSum = the sum of the subarray currently being extended
maxSub = the maximum subarray sum seen so far
```

Code:

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSub = nums[0]
        curSum = 0

        for num in nums:
            if curSum < 0:
                curSum = 0
            curSum += num
            maxSub = max(maxSub, curSum)

        return maxSub
```

Why `maxSub = nums[0]` instead of `0`?

Because the array may contain all negative numbers:

```text
nums = [-5, -2, -7]
answer = -2
```

If you initialize the answer to `0`, you will incorrectly return the empty subarray. But the problem requires a non-empty subarray, so you must initialize the answer with the first element.

## Kadane: Manual Walkthrough

Use the classic example:

```text
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
```

Process:

```text
num   curSum after update   maxSub
-2    -2                    -2
 1     1                     1      # curSum before this was < 0, so drop -2
-3    -2                     1
 4     4                     4      # curSum before this was < 0, so restart from 4
-1     3                     4
 2     5                     5
 1     6                     6
-5     1                     6
 4     5                     6
```

The final answer is `6`.

## Kadane: Relationship to Ordinary DP

Kadane can be understood like this:

```text
full DP:
dp[i] = max(dp[i - 1] + nums[i], nums[i])
ans = max(ans, dp[i])

space optimization:
curSum = dp[i - 1]
after the update curSum = dp[i]
```

So it is not magic. In essence it is:

```text
define the state as "ending at the current position," then compress the dp array into one variable.
```

Common variants of this type of problem:

- Maximum product subarray: you need to track both the maximum and the minimum because a negative number flips the sign.
- Circular maximum subarray: you need to compare the ordinary maximum subarray with `total_sum - minimum subarray sum`.
- Maximum profit from one stock trade: this can be seen as maintaining the historical minimum buy price, or converted into a maximum difference subarray.

## Jump Game: When DP Can Become Greedy

Jump Game asks:

```text
Given nums, start from index 0.
nums[i] means you can jump at most nums[i] steps to the right from i.
Ask whether you can reach the last index.
```

Example:

```text
nums = [2, 3, 1, 1, 4]
answer = true

0 -> 1 -> 4
```

Counterexample:

```text
nums = [3, 2, 1, 0, 4]
answer = false

No matter how you jump, you get stuck at the 0 at index 3.
```

This problem is very suitable for understanding a more important question:

> What kind of DP can be further compressed into Greedy?

Not every DP can become greedy. A DP that can become greedy usually has a shared feature:

```text
The answers for a large set of states can be fully summarized by one monotonic boundary, one optimal representative, or one extreme-value variable.
```

In Jump Game, that representative variable is `goal`.

## Jump Game: Recursive View

Do not rush to write greedy yet. Start with the most direct search.

Define the recursive function:

```text
dfs(i) = whether you can reach the last index starting from index i
```

From `i`, you can jump to:

```text
i + 1, i + 2, ..., i + nums[i]
```

So the recursion is:

```text
dfs(i) = dfs(i + 1) or dfs(i + 2) or ... or dfs(i + nums[i])
```

Code:

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        def dfs(i):
            if i >= len(nums) - 1:
                return True

            end = min(len(nums) - 1, i + nums[i])
            for j in range(i + 1, end + 1):
                if dfs(j):
                    return True

            return False

        return dfs(0)
```

This version is logically clear, but it repeatedly recomputes many states. For example, many paths ask the same question:

```text
Can you reach the destination starting from index 3?
```

So it times out.

## Jump Game: Top-Down DP

After adding memoization, the state stays the same:

```text
dfs(i) = whether you can reach the end from i
```

You simply cache the computed `dfs(i)`.

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        memo = {}

        def dfs(i):
            if i >= len(nums) - 1:
                return True

            if i in memo:
                return memo[i]

            end = min(len(nums) - 1, i + nums[i])
            for j in range(i + 1, end + 1):
                if dfs(j):
                    memo[i] = True
                    return True

            memo[i] = False
            return False

        return dfs(0)
```

Complexity:

```text
Time:  O(n^2)
Space: O(n)
```

Why is it `O(n^2)`?

Because each `i` is computed at most once, but inside each `i` you may scan many `j`s.

## Jump Game: Bottom-Up DP

Reverse the recursion and fill the table from right to left.

Define:

```text
dp[i] = whether you can reach the last index starting from index i
```

base case:

```text
dp[n - 1] = true
```

Transition:

```text
dp[i] = any(dp[j] == true for j in [i + 1, i + nums[i]])
```

Code:

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [False] * n
        dp[n - 1] = True

        for i in range(n - 2, -1, -1):
            end = min(n - 1, i + nums[i])
            for j in range(i + 1, end + 1):
                if dp[j]:
                    dp[i] = True
                    break

        return dp[0]
```

Up to this point, it is still standard DP.

Now the key question appears:

```text
Does dp[i] really need to know every dp[j] on its right?
```

If yes, then it cannot become greedy.

If not, and all it needs is one "most critical position," then it has a chance to become greedy.

## Jump Game: Observe the DP Table

Look at the example:

```text
nums = [2, 3, 1, 1, 4]
index  0  1  2  3  4
```

Look from right to left.

The last position is definitely a good position:

```text
index  0  1  2  3  4
good              T
goal = 4
```

`i = 3`:

```text
3 + nums[3] = 3 + 1 = 4
```

It can jump to `goal = 4`, so index 3 is also a good position:

```text
index  0  1  2  3  4
good           T  T
goal = 3
```

`i = 2`:

```text
2 + nums[2] = 3
```

It can jump to `goal = 3`, so index 2 is also a good position:

```text
index  0  1  2  3  4
good        T  T  T
goal = 2
```

`i = 1`:

```text
1 + nums[1] = 4
```

It can jump past `goal = 2`, and of course it can also reach a good position, so:

```text
goal = 1
```

`i = 0`:

```text
0 + nums[0] = 2
```

It can reach `goal = 1`, so:

```text
goal = 0
```

In the end, `goal == 0`, so the answer is `true`.

Something very important happened here:

```text
DP originally stores all good positions.
Greedy stores only the leftmost good position.
```

Why is it enough to store only the leftmost good position?

Because if some index `i` can jump to the leftmost good position `goal`, then it can definitely reach the end.

If `i` cannot even reach the leftmost good position, then reaching a good position farther to the right is only harder.

So the entire block of DP information on the right can be compressed into one boundary variable:

```text
goal = the leftmost position currently known to be able to reach the end
```

## Jump Game: Greedy Version

Scan from right to left:

```text
If i can jump to goal or beyond goal:
    i becomes the new goal
```

Code:

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) - 1

        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= goal:
                goal = i

        return goal == 0
```

Complexity:

```text
Time:  O(n)
Space: O(1)
```

This is not "greedy by intuition." It is greedy compressed from DP:

```text
dp array of good positions
        ↓
leftmost good position
        ↓
goal
```

## Jump Game: Why This Greedy Is Correct

We maintain an invariant:

```text
When scanning at i, goal is the leftmost good position in [i + 1, n - 1].
```

A good position means:

```text
Starting from this position, you can reach the last index.
```

When we check index `i`:

```text
i + nums[i] >= goal
```

this means `i` can jump in one step to `goal`, and `goal` can already reach the end.

So `i` can also reach the end. And because `i` is farther left than the old `goal`, we update:

```text
goal = i
```

If:

```text
i + nums[i] < goal
```

then `i` cannot even reach the leftmost good position.

Any good position farther to the right has an even larger index, so it is even less reachable. Therefore `i` is not a good position.

After the scan ends:

```text
goal == 0
```

means index 0 is a good position.

## When DP Can Be Converted into Greedy

You can use the following questions to judge.

### 1. Does the DP state maintain only a "feasible set"?

Jump Game's DP is:

```text
dp[i] = whether i can reach the end
```

It maintains a set of good positions:

```text
good positions = { i | dp[i] == true }
```

If a DP maintains complex numeric values, such as different costs, different paths, or different choice histories for each state, then it is usually not easy to turn directly into greedy.

### 2. Can this set be represented by one boundary?

In Jump Game, we do not need to know all good positions. We only need to know:

```text
the leftmost good position goal
```

The reason is that the target is only:

```text
whether you can reach some good position
```

and `goal` is the easiest good position among all good positions to reach.

This is a monotonic boundary.

### 3. Does a local update destroy future choices?

When `i` can reach `goal`, we update `goal` to `i`.

This does not make the future worse. It makes the future easier:

```text
old goal is farther right
new goal is farther left
```

After that, for indices farther left, reaching a good position only becomes easier, never harder.

This property is called greedy-choice safety:

```text
local choice does not remove any globally optimal possibility
```

### 4. Is there a dominance relationship?

In Jump Game:

```text
a good position farther left dominates a good position farther right
```

because for some index `i` on the left:

```text
being able to jump to a farther-left good position
=> is definitely easier than jumping to a farther-right good position
```

So we can discard all farther-right good positions and keep only the leftmost one.

This is the most important signal for converting DP to greedy:

> There is a dominance relationship among multiple states, and the dominated states can never be more useful than the representative state.

## The General Pattern for Converting DP to Greedy

Many greedy algorithms are not invented out of thin air. They are derived like this:

```text
1. Write the full DP.
2. Observe what information the DP stores.
3. Check whether that information has monotonicity.
4. Find a variable that can represent a whole group of states.
5. Prove that keeping only that variable does not lose the answer.
```

For Jump Game:

```text
full DP: dp[i] means whether i is good
stored information: all good positions
monotonicity: a farther-left good position is easier for indices on the left to reach
representative variable: leftmost good position, which is goal
safety: if you can reach goal, then you can definitely reach the end
```

So:

```text
O(n^2) DP
  -> store all good positions
  -> keep only the leftmost good position
  -> O(n) Greedy
```

## Another Greedy Angle: Maintain the Farthest Reachable Position from Left to Right

Jump Game also has another common solution that scans from left to right:

```text
reach = the farthest position reachable so far
```

If when scanning to `i`:

```text
i > reach
```

that means you cannot even get to the current position, so you definitely cannot continue farther, and should return `False`.

Otherwise update:

```text
reach = max(reach, i + nums[i])
```

Code:

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        reach = 0

        for i, jump in enumerate(nums):
            if i > reach:
                return False
            reach = max(reach, i + jump)

        return True
```

This version maintains another monotonic boundary:

```text
the segment [0, reach] is already reachable in some way
```

As long as `reach` keeps expanding to the right, it is feasible once it passes `n - 1`.

The two greedy solutions are essentially the same:

```text
from right to left: maintain the leftmost good position
from left to right: maintain the farthest reachable position
```

Both compress an entire set of DP states into one boundary variable.

## Jump Game: How to Explain It in an Interview

A good answer order is:

1. I first define `dp[i]`: whether you can reach the end from `i`.
2. The recurrence checks all `j` that `i` can jump to. As long as some `dp[j]` is true, `dp[i]` is true.
3. This bottom-up DP is `O(n^2)`.
4. I observe that we do not need to store all positions whose value is true. We only need to store the leftmost true position, `goal`.
5. If `i + nums[i] >= goal`, then `i` can reach a known good position, so `i` is also a good position.
6. Because a farther-left good position is more helpful for the future, updating `goal = i` is safe.
7. Finally, check whether `goal == 0`.

One-sentence summary:

> This problem can change from DP to greedy because "all positions that can reach the end" can be fully represented by "the leftmost position that can reach the end."

## Grid DP: two cases on coordinates (r, c)

## Unique Paths

A robot may only move right or down. How many distinct paths from the top-left to the bottom-right? `$dp[i][j]=dp[i-1][j]+dp[i][j-1]$`. The first row and first column each have only one way, so initialize them all to 1.

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
```

Complexity: Time `$O(mn)$`, Space `$O(n)$` (rolled into a 1D array).

Pitfall: when `$m=1$` or `$n=1$` the answer should be 1; the rolling-array version above already covers that without an extra special case.

## Longest Increasing Path in a Matrix

This looks like grid DP, but you cannot sweep by rows and columns the way Unique Paths does: values are arbitrarily placed, so no single direction guarantees that "when you compute a cell, every cell it depends on is already done."

The correct approach is memoized DFS from each cell: `$dfs(r,c)$` = length of the longest strictly increasing path starting at that cell, walking only to strictly larger neighbors:

$$dfs(r,c)=1+\max_{\text{strictly larger neighbor}} dfs(nr,nc)$$

(If there is no legal neighbor, the length is 1.) Cache results with memo.

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

Complexity: Time `$O(mn)$` (each cell is truly computed once), Space `$O(mn)$` (memo plus recursion stack).

Why this is still DP: strictly increasing edges form a directed acyclic graph (DAG). Subproblem `$dfs(r,c)$` is reached by many paths; caching with memo is the standard "optimal substructure + overlapping subproblems." The only difference is that you do not write an explicit double for-loop fill; memoized DFS computes along the DAG's topological order implicitly.

Pitfall: writing `>=` instead of `>` creates cycles on equal values and recursion never ends; forgetting memo expands the same cell repeatedly and degrades to exponential time.

## Common Pitfalls

- Starting to write code before defining what `dp[i]` means.
- Reversing the loop direction relative to the dependency direction.
- Forgetting the base case, such as `dp[n] = 1`.
- Treating an invalid state as `1`, especially `'0'` in Decode Ways.
- Doing space optimization too early and making the variable meanings unclear.
- In the stock cooldown problem, buying directly from `sold` after selling and missing the cooldown.
- Overwriting old variables during space optimization, causing `sold` to use the `hold` updated on the same day.
- Using forward order for 0/1 knapsack 1D optimization, causing the same item to be reused.
- Forgetting to check whether `(target + total)` is even before solving Target Sum.
- Forgetting that `nums[i] = 0` in Target Sum also creates different sign assignments; the knapsack counting formulation handles this naturally.
- Initializing `maxSub` to `0` in Kadane, causing the wrong result on all-negative arrays.
- Forgetting that the subarray in Kadane must be contiguous and cannot skip elements in the middle like a subsequence.
- Treating `nums[i]` in Jump Game as the only destination; it is actually the maximum jump length, so you may jump to any position in the interval `[i + 1, i + nums[i]]`.
- Saying "always jump the farthest" in Jump Game without a proof; in an interview you should explain why the state set can be represented by the boundary `goal` or `reach`.
- Forgetting that the condition in the right-to-left greedy for Jump Game is `i + nums[i] >= goal`, not `nums[i] >= goal`.

- Confusing whether the answer is `dp[n]` or `max(dp)`, especially on LIS, Maximum Product Subarray, Min Cost Climbing Stairs, and other problems where the answer is not the last cell.
- Writing 0/1 knapsack 1D optimization in ascending order, so the same item is reused in one round and it becomes a complete knapsack.
- Putting amounts outside and coins inside for Coin Change II, turning combinations into permutations.
- Not filling interval DP by increasing length, so you read an unfinished `dp[i+1][j-1]`.
- Thinking "which balloon to burst first" in Burst Balloons instead of "which balloon is last in this interval," so subproblem boundaries stay unclear.
- Forgetting that `*` in Regex Matching applies to the previous character, or missing the empty-string base case for patterns like `a*`.
- Forgetting the separate `n=1` case in House Robber II.
- Filling Longest Increasing Path in a Matrix with an ordinary double loop by rows and columns instead of memoized DFS.

## Complexity

Decode Ways array version:

- Time: `O(n)`
- Space: `O(n)`

Space-optimized version:

- Time: `O(n)`
- Space: `O(1)`

Stock Cooldown three-state array version:

- Time: `O(n)`
- Space: `O(n)`

Stock Cooldown space-optimized version:

- Time: `O(n)`
- Space: `O(1)`

Target Sum 0/1 knapsack:

- Time: `O(n * bag)`, where `bag = (target + sum(nums)) / 2`
- Space: `O(n * bag)` for the 2D version, `O(bag)` for the 1D version

Complete knapsack:

- Time: usually `O(number_of_items * capacity)`
- Space: `O(capacity)` after 1D optimization

Kadane maximum subarray sum:

- Time: `O(n)`
- Space: `O(n)` for the full DP, `O(1)` after space optimization

Jump Game:

- Recursion: exponential time, space `O(n)`
- Top-down / bottom-up DP: time `O(n^2)`, space `O(n)`
- Greedy: time `O(n)`, space `O(1)`

Climbing Stairs / Min Cost Climbing Stairs / House Robber / House Robber II:

- Time: `O(n)`
- Space: `O(1)` (rolling variables)

Word Break:

- Time: `O(n^2 · L)`, where `L` is the substring comparison cost
- Space: `O(n)`

Longest Increasing Subsequence (the O(n²) DP version in this note):

- Time: `O(n^2)`
- Space: `O(n)`

Maximum Product Subarray:

- Time: `O(n)`
- Space: `O(1)`

Longest Palindromic Substring / Palindromic Substrings:

- Time: `O(n^2)`
- Space: `O(n^2)`

Burst Balloons:

- Time: `O(n^3)`
- Space: `O(n^2)`

Partition Equal Subset Sum:

- Time: `O(n · sum/2)`
- Space: `O(sum/2)`

Coin Change II:

- Time: `O(amount · |coins|)`
- Space: `O(amount)`

Longest Common Subsequence / Edit Distance / Interleaving String / Distinct Subsequences / Regular Expression Matching:

- Time: `O(mn)`
- Space: `O(mn)` (LCS and Edit Distance can roll down to `O(min(m,n))`)

Unique Paths:

- Time: `O(mn)`
- Space: `O(n)` (rolling array)

Longest Increasing Path in a Matrix:

- Time: `O(mn)`
- Space: `O(mn)` (memo + recursion stack)

## Quick self-check

```quiz
title: Quick quiz 1
question: Relative to House Robber, what is the most critical extra handling in House Robber II?
answer: B
A. Switch to 2D dp[i][j]
B. Split the ring into two linear Robber runs and take max
C. Must use interval DP
D. Switch to complete knapsack
explanation: First and last are adjacent and cannot both be robbed; run linear Robber on nums[0..n-2] and nums[1..n-1], then take max.
```

```quiz
title: Quick quiz 2
question: On 1D DP, what is the most important difference between Coin Change (fewest coins) and Coin Change II (combinations)?
answer: C
A. One uses descending and one ascending (when both are complete knapsack)
B. One must be 2D and one must be 1D
C. Different objectives (min vs additive counting), and II needs "items outside" for combination semantics
D. Transitions are identical; only the return value differs
explanation: Both are complete-knapsack shaped; II needs combinations not permutations, so the coin loop is outside; transitions are min vs additive counting.
```

```quiz
title: Quick quiz 3
question: In the Burst Balloons interval transition, what does k mean?
answer: A
A. The balloon burst last inside open interval (i,j)
B. The balloon burst first inside the open interval
C. The interval length
D. The index of a virtual boundary 1
explanation: When k is last, both sides are empty; the payoff splits into a[i]*a[k]*a[j] plus the two subinterval DPs.
```

```quiz
title: Quick quiz 4
question: When Target Sum reduces to subset sum, the subset target P equals?
answer: B
A. (sum - target) / 2
B. (sum + target) / 2
C. sum - target
D. target
explanation: P+N=sum and P-N=target, so P=(sum+target)/2; it must divide evenly and |target|<=sum.
```

```quiz
title: Quick quiz 5
question: Why can Longest Increasing Path in a Matrix not fill dp by rows like Unique Paths?
answer: D
A. Because you may only go right and down
B. Because you must use O(1) space
C. Because it is not DP
D. Larger neighbors point in arbitrary directions, so there is no single valid fill order; use memoized DFS on the DAG
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
explanation: Characters already match, so no operation; inherit the diagonal. Only when they differ do you take min among insert/delete/replace plus one.
```

```quiz
title: Quick quiz 7
question: Why must Maximum Product Subarray also maintain min_here?
answer: C
A. To handle 0
B. For O(1) space
C. A negative sign can flip the minimum product into the maximum; keeping only max loses solutions
D. The problem asks for the minimum product
explanation: For example [-2,3,-4]; negative times negative depends on the prior minimum (most negative) product.
```

```quiz
title: Quick quiz 8
question: In Distinct Subsequences, when s[i-1]==t[j-1], the transition should be?
answer: B
A. Only add dp[i-1][j-1]
B. dp[i-1][j-1] + dp[i-1][j] (use or skip the current character)
C. dp[i][j-1] + dp[i-1][j]
D. max(dp[i-1][j-1], dp[i-1][j])
explanation: Match t's last character with the current s character, or skip the current s character; add both path counts.
```

```quiz
title: Quick quiz 9
question: In Regex Matching, when p[j-1]=='*', "match zero times" corresponds to?
answer: A
A. dp[i][j-2]
B. dp[i-1][j]
C. dp[i-1][j-1]
D. dp[i][j-1]
explanation: Drop the whole x* and ask whether p[:j-2] already matches s[:i]; multiple matches OR in dp[i-1][j].
```

```quiz
title: Quick quiz 10
question: Why must the capacity loop be descending in 0/1 knapsack 1D optimization?
answer: C
A. It is faster
B. To turn combinations into permutations
C. To avoid reusing the same item in the current round
D. Descending is required for complete knapsack
explanation: Descending keeps dp[j-x] as the old value from "before taking the current item"; ascending becomes complete knapsack.
```

```quiz
title: Quick quiz 11
question: What is the most accurate meaning of the sold state in Stock with Cooldown?
answer: B
A. Any empty position
B. Just sold today; tomorrow is in cooldown
C. Holding stock
D. Cumulative sell count
explanation: sold specially marks "sold today"; the next day can only enter free, not buy directly.
```

```quiz
title: Quick quiz 12
question: In House Robber II, why can you not directly reuse House Robber's linear DP?
answer: B
A. Different data ranges require a different algorithm
B. First and last houses are adjacent and form a ring, so you must split into two linear subproblems and take the larger answer
C. A circular array must use interval DP
D. The dp transition equation itself must change on a circular array
explanation: The only change from the ring is "cannot rob first and last together"; splitting into [0..n-2] and [1..n-1] linear Robber bypasses that constraint, and the transition inside each segment is unchanged.
```

## Interview Answer Template

You can explain a DP problem like this:

1. I first define what the state `dp[...]` represents.
2. Then I list the choices available at each step.
3. I convert each choice into a reference to previous states and obtain the recurrence.
4. I determine the loop order from the recurrence dependencies.
5. I write the full DP first.
6. Finally, I observe which states are actually depended on and then do space optimization.
