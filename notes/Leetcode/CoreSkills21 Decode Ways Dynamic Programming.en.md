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

## Interview Answer Template

You can explain a DP problem like this:

1. I first define what the state `dp[...]` represents.
2. Then I list the choices available at each step.
3. I convert each choice into a reference to previous states and obtain the recurrence.
4. I determine the loop order from the recurrence dependencies.
5. I write the full DP first.
6. Finally, I observe which states are actually depended on and then do space optimization.
