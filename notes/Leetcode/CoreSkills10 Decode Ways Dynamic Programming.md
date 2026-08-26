# Dynamic Programming：从递推到空间优化

## 面试目标

Dynamic Programming 不是“背模板”，而是把一个问题拆成稳定的子问题，然后按依赖顺序把答案算出来。

一套可复用的 DP 写法通常分三步：

1. 写出递推表达式。
2. 把递推表达式改成循环模式。
3. 根据依赖关系做空间优化。

这三步比直接写代码重要。只要递推表达式清楚，代码通常只是机械翻译。

## 什么时候想到 DP

题目有下面几个信号时，可以优先考虑 DP：

- 问“方案数 / 最小代价 / 最大收益 / 是否可行”。
- 决策过程可以拆成阶段，例如处理到第 `i` 个字符、第 `i` 个物品、第 `i` 天。
- 同一个子问题会被重复遇到。
- 当前答案可以由更小规模的答案推出。

DP 的本质是：

```text
state + transition + base case + iteration order
```

也就是：

- 状态：`dp[...]` 表示什么？
- 转移：当前状态从哪些旧状态来？
- 初始值：最小问题的答案是什么？
- 顺序：先算谁，后算谁？

## 第一步：写出递推表达式

先不要写循环。先用一句话定义状态。

常见状态形式：

```text
dp[i]       = 前 i 个元素 / 从 i 开始的后缀 的答案
dp[i][j]    = 两个维度共同决定的答案
dp[i][state]= 第 i 步处于某种状态时的答案
```

定义状态时要问两个问题：

1. 它能不能直接表达最终答案？
2. 它能不能从更小的状态转移过来？

然后写 choice。每道 DP 题的递推基本都是：

```text
dp[current] = combine(dp[previous states])
```

如果是求方案数，`combine` 通常是加法：

```text
dp[i] = dp[a] + dp[b]
```

如果是求最优值，`combine` 通常是 `min` 或 `max`：

```text
dp[i] = min(dp[a] + costA, dp[b] + costB)
```

如果是求可行性，`combine` 通常是 `or`：

```text
dp[i] = dp[a] or dp[b]
```

最后补 base case。base case 是递推的起点，不是细节。很多 DP bug 都来自 base case 没有想清楚。

## 第二步：写成循环模式

递推表达式写出来后，循环方向由依赖关系决定。

如果：

```text
dp[i] depends on dp[i - 1]
```

通常从左往右。

如果：

```text
dp[i] depends on dp[i + 1]
```

通常从右往左。

如果：

```text
dp[i][j] depends on dp[i - 1][j] and dp[i][j - 1]
```

通常双层循环，从小到大填表。

把递推写成循环时，可以按这个模板：

```text
initialize dp
set base cases
for state in valid order:
  compute dp[state] from previous states
return answer
```

注意：循环顺序不是风格问题，而是正确性问题。你必须保证当前状态需要的旧状态已经算好。

## 第三步：空间优化

空间优化不要一开始就做。先写清楚完整 DP，再观察依赖范围。

如果：

```text
dp[i] only depends on dp[i - 1]
```

可以只保留一个变量。

如果：

```text
dp[i] only depends on dp[i + 1] and dp[i + 2]
```

可以只保留两个变量。

如果二维 DP 只依赖上一行：

```text
dp[i][j] depends on dp[i - 1][...]
```

可以滚动数组，把 `O(nm)` 空间降到 `O(m)`。

空间优化的原则是：

> 只保留未来还会被用到的状态。

不要为了省空间破坏状态语义。面试里更推荐先讲完整 DP，再讲如何优化。

## 六种骨架速查：拿到题先分类

上面三步讲的是"怎么把一个已经想清楚的递推写成代码"，但更早的问题是"这题的状态应该长什么样"。DP 题看起来很多，状态的形状其实只有六种。拿到题先问自己：状态是一个下标、一个闭区间、背包容量、两个序列的前缀、网格坐标，还是有限个命名状态？

```text
这题的状态维度是什么？

├── 一个下标 i（前 i 个元素 / 爬到第 i 阶）
│     → 骨架 A：1D 线性 DP
│       代表题：Climbing Stairs, House Robber, Decode Ways, Word Break, LIS, Coin Change, Max Product Subarray
│
├── 一个闭区间 [i, j]（子串 / 气球区间），按长度递增填
│     → 骨架 B：区间 DP
│       代表题：Palindrome 系列, Burst Balloons
│
├── 物品 + 容量（选或不选 / 可重复选 / 计数）
│     → 骨架 C：0/1 背包 / 完全背包 / 计数背包
│       代表题：Partition Equal Subset Sum, Coin Change II, Target Sum
│
├── 两个序列的前缀 (i, j)
│     → 骨架 D：双序列 DP
│       代表题：LCS, Edit Distance, Interleaving String, Distinct Subsequences, Regex Matching
│
├── 网格格子 (r, c)，有固定扫掠方向
│     → 骨架 E：网格 DP
│       代表题：Unique Paths
│       例外：Longest Increasing Path 无固定方向 → 记忆化 DFS
│
└── 每一步有少量命名状态（持有 / 冷却 / 空仓）
      → 骨架 F：状态机 DP
        代表题：Stock with Cooldown
```

（Kadane 最大子数组和与 Jump Game 也会在本文后半段出现，但它们的重点是"DP 怎么压缩成 Greedy"，不单独占一种骨架。）

## 骨架 A：1D 线性 DP

```python
def solve_1d(arr):
    n = len(arr)
    dp = [INIT] * (n + 1)   # 或 n，看下标约定
    # BASE: 填好 dp[0] / dp[1] / ...
    for i in range(START, n):
        dp[i] = TRANSITION(dp, i, arr)   # 只依赖 O(1)~O(k) 个更早状态
    return ANSWER(dp)                    # dp[n] 或 max(dp) 等
```

| 题 | 状态定义 | 转移 | base | 遍历 | 答案 |
|---|---|---|---|---|---|
| Climbing Stairs | `dp[i]` = 爬到 i 阶方案数 | `dp[i]=dp[i-1]+dp[i-2]` | `dp[0]=1,dp[1]=1` | `i=2..n` | `dp[n]` |
| Min Cost Climbing Stairs | `dp[i]` = 到达 i 的最小花费 | `dp[i]=cost[i]+min(dp[i-1],dp[i-2])` | `dp[0]=cost[0],dp[1]=cost[1]` | `i=2..n-1` | `min(dp[n-1],dp[n-2])` |
| House Robber | `dp[i]` = 考虑前 i 家最大金额 | `dp[i]=max(dp[i-1],dp[i-2]+nums[i-1])` | `dp[0]=0,dp[1]=nums[0]` | `i=2..n` | `dp[n]` |
| House Robber II | 同上，环拆成两段线性 | 对 `[0..n-2]` 与 `[1..n-1]` 各跑一次 Robber | 单段同 Robber | 两次线性 | 两段答案取 `max` |
| Decode Ways | `dp[i]` = 后缀 `s[i:]` 的方案数 | 一位合法 `+dp[i+1]`；两位合法(`10..26`) `+dp[i+2]` | `dp[n]=1` | 从右往左 | `dp[0]` |
| Word Break | `dp[i]` = `s[:i]` 可否被拆 | `dp[j] and s[j:i] in dict` | `dp[0]=True` | `i=1..n`，内层 `j` | `dp[n]` |
| Longest Increasing Subsequence | `dp[i]` = 以 `i` 结尾的 LIS 长 | `dp[i]=max(dp[j])+1`（`nums[j]<nums[i]`） | 全 `1` | 双重 `i,j<i` | `max(dp)` |
| Coin Change | `dp[a]` = 凑出金额 `a` 最少硬币 | `dp[a]=min(dp[a-c])+1` | `dp[0]=0`，其余 `inf` | 金额从小到大 | `dp[amount]`（`inf`→`-1`） |
| Maximum Product Subarray | 以 `i` 结尾的最大/最小乘积 | 同时滚 `max_here,min_here` | 首元素 | 一次扫描 | 全局 `max` |

下面先走完 Decode Ways 的完整流程，再补齐这张表里剩下的题。

## 骨架 B：区间 DP

```python
def solve_interval(s):
    n = len(s)
    dp = [[INIT] * n for _ in range(n)]
    # BASE: 长度 1（有时也要长度 0 / 2）
    for length in range(2, n + 1):       # 按区间长度递增
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i][j] = TRANSITION(dp, i, j, s)
    return ANSWER(dp)
```

| 题 | 状态定义 | 转移 | base | 遍历 | 答案 |
|---|---|---|---|---|---|
| Longest Palindromic Substring | `dp[i][j]` = `s[i..j]` 是否回文 | 两端相等且（长度≤2 或 `dp[i+1][j-1]`） | 单字符 `True` | 按长度 | 记下最长的 `(i,j)` |
| Palindromic Substrings | 同上布尔表 | 同上 | 同上 | 同上 | 统计 `True` 个数 |
| Burst Balloons | `dp[i][j]` = 戳开区间 `(i,j)` 最大收益 | `max_k a[i]*a[k]*a[j]+dp[i][k]+dp[k][j]`（`k` 最后戳） | 相邻 `i,j` 为 0 | 长度↑；两端补 `1` | `dp[0][n+1]` |

## 骨架 C：背包族

```python
def solve_knapsack(items, capacity):
    dp = [INIT] * (capacity + 1)
    dp[0] = BASE0
    for x in items:                      # 物品在外 → 组合；金额在外 → 排列语义不同
        for j in range(...):             # 0/1：倒序；完全：正序
            dp[j] = COMBINE(dp[j], dp[j - x], x)
    return ANSWER(dp)
```

| 题 | 状态定义 | 转移 | base | 遍历 | 答案 |
|---|---|---|---|---|---|
| Partition Equal Subset Sum | `dp[j]` = 能否凑出和 `j` | `dp[j]\|=dp[j-x]`（0/1） | `dp[0]=True` | 物品外，容量倒序 | `dp[sum/2]` |
| Coin Change II | `dp[a]` = 凑出 `a` 的组合数 | `dp[a]+=dp[a-c]`（完全） | `dp[0]=1` | 硬币外、金额内正序 | `dp[amount]` |
| Target Sum | 化为子集和计数 | 同 0/1 计数背包 | `dp[0]=1` | 物品外，容量倒序 | `dp[(sum+target)/2]` |

这张表里的三道题、以及后面的 Coin Change，具体推导见下文"0/1 背包"与"完全背包"两节。

## 骨架 D：双序列 DP

```python
def solve_two_seq(a, b):
    m, n = len(a), len(b)
    dp = [[INIT] * (n + 1) for _ in range(m + 1)]
    # BASE: 填第 0 行 / 第 0 列
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = TRANSITION(dp, i, j, a, b)
    return dp[m][n]
```

| 题 | 状态定义 | 转移要点 | base |
|---|---|---|---|
| LCS | `dp[i][j]` = `a[:i]` 与 `b[:j]` 的 LCS 长 | 相等则 `+1`，否则 `max(上,左)` | 0 行/列 = 0 |
| Edit Distance | 把 `a[:i]` 变成 `b[:j]` 最少操作 | 相等抄对角；否则 insert/delete/replace +1 | `dp[i][0]=i`,`dp[0][j]=j` |
| Interleaving String | `s1[:i]+s2[:j]` 能否交错成 `s3[:i+j]` | 从上方吃 `s1` 或从左方吃 `s2` | 空串匹配空前缀 |
| Distinct Subsequences | `s[:i]` 中有多少子序列等于 `t[:j]` | 字符相等：用或不用；否则只用跳过 | `dp[i][0]=1` |
| Regex Matching | `s[:i]` 能否被 `p[:j]` 匹配 | `.` 任意单字符；`*` 零次（`j-2`）或多次（匹配则 `i-1`） | 空 pattern；`x*` 可匹配空 |

## 骨架 E：网格 DP（含记忆化例外）

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

Longest Increasing Path in a Matrix **不能**按行列扫一遍：矩阵值任意，没有单一拓扑扫掠方向。正确做法是对每个格子做记忆化 DFS：只走向严格更大的邻居，隐式构成 DAG，无环，重叠子问题可缓存。这仍是 DP（最优子结构 + 重叠子问题 + memo），只是不写显式填表循环。

## 骨架 F：状态机 DP

```python
# hold / sold(cooldown) / free
for price in prices:
    hold, sold, free = TRANSITIONS(...)
return max(sold, free)
```

具体的三态方程见下文 Stock Cooldown 一节。

## Example：Decode Ways

现在用 Decode Ways 走一遍完整流程。

题目：给定只包含数字的字符串 `s`，按照：

```text
1 -> A
2 -> B
...
26 -> Z
```

问有多少种合法解码方式。

例如：

```text
s = "226"

2 | 2 | 6   -> B B F
22 | 6      -> V F
2 | 26      -> B Z

answer = 3
```

## Decode Ways：递推表达式

定义状态：

```text
dp[i] = s[i:] 这个后缀字符串的解码方案数
```

最终答案：

```text
dp[0]
```

Base case：

```text
dp[n] = 1
```

含义是：如果已经走到字符串末尾，说明前面形成了一种完整解码。

如果当前位置是 `'0'`：

```text
dp[i] = 0
```

因为 `0` 不能单独映射成任何字母。

否则当前位置有两类选择：

```text
take one digit  -> dp[i + 1]
take two digits -> dp[i + 2], if 10 <= s[i:i+2] <= 26
```

递推式：

```text
if s[i] == '0':
  dp[i] = 0
else:
  dp[i] = dp[i + 1]
  if i + 1 < n and 10 <= int(s[i:i+2]) <= 26:
    dp[i] += dp[i + 2]
```

这一步最重要。后面的循环和空间优化都只是对这个递推式的实现。

## Decode Ways：循环模式

因为 `dp[i]` 依赖 `dp[i + 1]` 和 `dp[i + 2]`，所以要从右往左算。

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

用 `"226"` 走一遍：

```text
dp[3] = 1

i = 2, "6":
  dp[2] = dp[3] = 1

i = 1, "26":
  dp[1] = dp[2] + dp[3] = 2

i = 0, "22":
  dp[0] = dp[1] + dp[2] = 3
```

## Decode Ways：空间优化

观察依赖：

```text
dp[i] only depends on dp[i + 1] and dp[i + 2]
```

所以只需要两个变量：

```text
one = dp[i + 1]
two = dp[i + 2]
```

每次算出：

```text
cur = dp[i]
```

然后窗口左移：

```text
two = one
one = cur
```

代码：

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

状态：`dp[i]` = 爬到第 `i` 阶的方案数。

$$dp[i] = dp[i-1] + dp[i-2]$$

Base：`$dp[0]=1$`，`$dp[1]=1$`（或直接 `$dp[1]=1,dp[2]=2$`）。

遍历：`$i=2..n$`。答案：`$dp[n]$`。

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

复杂度：Time `$O(n)$`，Space `$O(1)$`。

坑：`$n=1$` 时不要访问 `$dp[2]$`；这题和斐波那契同构，面试里说明这一点即可。

## Min Cost Climbing Stairs

可以从下标 `0` 或 `1` 起步。`dp[i]` = 到达台阶 `i` 并支付 `cost[i]` 后的最小总花费。

$$dp[i] = cost[i] + \min(dp[i-1], dp[i-2])$$

答案不是 `$dp[n]$`（顶楼没有台阶，也就没有 cost），而是 `$\min(dp[n-1], dp[n-2])$`：最后一步可以从倒数第一或倒数第二级跨上去。

```python
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        n = len(cost)
        a, b = cost[0], cost[1]
        for i in range(2, n):
            a, b = b, cost[i] + min(a, b)
        return min(a, b)
```

复杂度：Time `$O(n)$`，Space `$O(1)$`。

坑：忘记"顶楼无 cost"，把答案写成 `$dp[n-1]$`。

## House Robber

不能抢相邻房屋。`dp[i]` = 只考虑前 `i` 家时的最大金额。

$$dp[i] = \max(dp[i-1], dp[i-2] + nums[i-1])$$

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        prev2, prev1 = 0, 0
        for x in nums:
            prev2, prev1 = prev1, max(prev1, prev2 + x)
        return prev1
```

复杂度：Time `$O(n)$`，Space `$O(1)$`。

坑：空数组 / 单元素要单独处理；下标是 `nums[i-1]` 还是 `nums[i]`，要和 `dp` 的长度约定保持一致。

## House Robber II

房屋围成环：第一家和最后一家相邻，不能同时抢。拆成两次线性 House Robber，取较大值：

```text
max( rob(nums[0..n-2]), rob(nums[1..n-1]) )
```

`$n=1$` 时直接返回 `$nums[0]$`。

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

复杂度：Time `$O(n)$`，Space `$O(1)$`。

坑：漏掉 `$n=1$` 这个特判；或者两段切片写错，漏抢或多抢中间那家。

## Word Break

`$dp[i]$` = 前缀 `$s[:i]$` 能否拆成字典单词。

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

复杂度：Time `$O(n^2\cdot L)$`（`$L$` 是切片比较的成本，可用字典树优化），Space `$O(n)$`。

坑：`$dp[0]=True$` 表示空前缀总能被拆；字典很大时先把 `wordDict` 转成 `set`，否则 `in` 是线性扫描。

## Longest Increasing Subsequence

`$dp[i]$` = 以 `$nums[i]$` 结尾的最长严格递增子序列长度。

$$dp[i] = 1 + \max_{j<i,\,nums[j]<nums[i]} dp[j]$$

（找不到更小的前驱时为 `1`。）答案是 `$\max_i dp[i]$`，不是 `$dp[n-1]$`。

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

复杂度：Time `$O(n^2)$`，Space `$O(n)$`。

脚注：可以用耐心排序 + 二分做到 `$O(n\log n)$`，那是另一条优化线；这里给出的是标准的 `$O(n^2)$` DP 表，先把状态和转移吃透。

坑：答案是 `max(dp)`，不是 `dp[-1]`；"严格递增"用 `<`，不是 `<=`。

## Maximum Product Subarray

和 Kadane 的最大子数组**和**看起来像同一题的变体，但乘法会被负号翻转符号，所以必须同时维护以当前位置结尾的最大乘积和最小乘积：

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

复杂度：Time `$O(n)$`，Space `$O(1)$`。

坑：只维护 `max_here` 会在 `[-2, 3, -4]` 这类输入上漏掉答案 `24`（两个负数相乘变正）；遇到 `0` 时 `max_here`/`min_here` 会自然被重置为从当前元素重新开始，不需要额外特判。

## Example：Best Time to Buy and Sell Stock with Cooldown

这题比 Decode Ways 更适合练“状态机 DP”。

题目：给定每天股价 `prices`，可以买卖多次，但同一时间最多持有一股。卖出后，下一天不能买入，要 cooldown 一天。求最大利润。

例如：

```text
prices = [1, 2, 3, 0, 2]
answer = 3

一种最优操作：
day 0 buy at 1
day 1 sell at 2
day 2 cooldown
day 3 buy at 0
day 4 sell at 2
profit = 1 + 2 = 3
```

这题主要练两个记录技巧：

1. 先把状态记录清楚，最好画成状态机，再写状态方程。
2. 再看每一天只依赖前一天，于是把 DP 表压缩成常数空间。

## Stock Cooldown：技巧一，记录三个状态

很多人会用 `buying=True/False` 写递归，这当然可以。但 bottom-up 时，更直观的做法是记录每天结束后处于哪种状态。

定义三个状态：

```text
hold[i] = 第 i 天结束后，手里持有一股时的最大利润
sold[i] = 第 i 天结束后，今天刚卖出股票时的最大利润
rest[i] = 第 i 天结束后，手里没有股票，且不处于刚卖出的状态时的最大利润
```

这三个状态分别回答：

- `hold`：我现在手里有股票，未来可以卖。
- `sold`：我今天刚卖，明天必须 cooldown，不能买。
- `rest`：我现在手里没股票，也没有刚卖出的限制，未来可以买。

状态转移：

```text
hold[i] = max(
  hold[i - 1],              # 昨天就持有，今天继续不动
  rest[i - 1] - prices[i]   # 昨天是自由空仓，今天买入
)

sold[i] = hold[i - 1] + prices[i]
  # 昨天必须持有，今天才能卖出

rest[i] = max(
  rest[i - 1],              # 昨天自由空仓，今天继续休息
  sold[i - 1]               # 昨天刚卖出，今天 cooldown 后变成自由空仓
)
```

注意 cooldown 体现在这里：

```text
hold[i] 只能从 rest[i - 1] 买入，不能从 sold[i - 1] 买入
```

因为 `sold[i - 1]` 表示昨天刚卖，今天还在 cooldown，不能买。

Base case 可以这样理解：

```text
hold = -infinity  # 还没处理任何一天，不可能已经持股
sold = -infinity  # 还没处理任何一天，不可能已经卖出
rest = 0          # 什么都不做，利润为 0
```

最后答案不能是 `hold`，因为手里还拿着股票不算落袋利润：

```text
answer = max(sold[n - 1], rest[n - 1])
```

## Stock Cooldown：完整 DP 表

先写完整 DP，便于确认状态含义：

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

用 `[1, 2, 3, 0, 2]` 走一遍：

```text
day  price  hold  sold  rest
0    1      -1    -inf  0
1    2      -1     1    0
2    3      -1     2    1
3    0       1    -1    2
4    2       1     3    2

answer = max(sold, rest) = max(3, 2) = 3
```

看第 3 天 `price = 0` 的 `hold = 1`：

```text
hold[3] = max(hold[2], rest[2] - 0)
        = max(-1, 1 - 0)
        = 1
```

这表示：第 1 天卖出赚了 `1`，第 2 天 cooldown 后进入 `rest = 1`，第 3 天可以买入价格 `0` 的股票，所以持股状态的利润仍是 `1`。

## Stock Cooldown：技巧二，空间优化

观察转移式：

```text
hold[i] only depends on hold[i - 1], rest[i - 1]
sold[i] only depends on hold[i - 1]
rest[i] only depends on rest[i - 1], sold[i - 1]
```

也就是说，第 `i` 天只用到第 `i - 1` 天的三个状态。因此不需要三条数组，只需要三个变量：

```text
hold = previous hold
sold = previous sold
rest = previous rest
```

每一天先算新状态，再整体替换：

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

这里最重要的细节是：不要边算边覆盖旧变量。

错误写法：

```python
hold = max(hold, rest - price)
sold = hold + price  # wrong: 这里用到的是今天更新后的 hold
```

`sold` 必须来自“昨天持股，今天卖出”，所以要用旧的 `hold`。因此要先用 `next_*` 保存新状态，最后统一赋值。

## Stock Cooldown：和递归写法的关系

递归写法常用两个状态：

```text
dfs(i, buying)
```

- `buying = True`：手里没有股票，可以买。
- `buying = False`：手里有股票，可以卖。

卖出时跳到 `i + 2`：

```text
sell = prices[i] + dfs(i + 2, True)
```

三状态 bottom-up 写法把这个 `i + 2` 的 cooldown 显式拆成了 `sold -> rest`：

```text
hold --sell--> sold --cooldown one day--> rest --buy--> hold
```

所以两个写法本质相同，只是记录方式不同。面试里如果题目有“持有 / 刚卖 / 空仓可买”这种过程限制，三状态状态机会更不容易写错。

## 区间 DP：子串与区间问题

状态是一个闭区间 `[i, j]`（子串、气球区间），转移依赖更短的子区间，所以必须按区间长度递增来填表，而不是按行或按列扫。

## Longest Palindromic Substring

用区间布尔 DP：`$dp[i][j]$` 表示 `$s[i..j]$` 是否回文。

$$dp[i][j] = (s[i]=s[j]) \land (j-i<2 \lor dp[i+1][j-1])$$

按长度递增填表，同时记录最长回文区间的起点与长度。中心扩展法也能做这题，这里强调的是区间 DP 这种形态。

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

（这里从右往左、从上往下填，效果上等价于按长度递增；也可以显式写成外层按 `length` 递增。）

复杂度：Time `$O(n^2)$`，Space `$O(n^2)$`。

坑：转移依赖 `$dp[i+1][j-1]$`，必须保证更短的区间已经算完；单字符自身也是回文，要计入初始的最长长度。

## Palindromic Substrings

用同一张布尔表，只是把答案从"记录最长的一个"换成"统计所有 `$dp[i][j]=True$` 的个数"。

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

复杂度：Time `$O(n^2)$`，Space `$O(n^2)$`。

坑：不要只统计中心扩展法找到的"最长"回文；每一个满足条件的 `(i, j)` 都要算一次，包括所有长度为 1 的单字符。

## Burst Balloons

在数组两端各补一个虚拟气球 `1`。定义开区间：`$dp[i][j]$` = 戳破 `$i$` 与 `$j$` 之间所有气球能得到的最大硬币数。

关键技巧：枚举区间内最后戳破的气球 `$k$`，而不是最先戳破的。最后戳 `$k$` 时，它左右两边的气球都已经戳完，`$k$` 此刻的邻居正好是虚拟边界 `$a[i]$` 与 `$a[j]$`：

$$dp[i][j] = \max_{i<k<j}\bigl(dp[i][k] + a[i]\cdot a[k]\cdot a[j] + dp[k][j]\bigr)$$

按 `$j-i$` 递增顺序填表。答案是 `$dp[0][n+1]$`（`$n$` 为原数组长度，补了两个虚拟边界后共 `$n+2$` 个位置）。

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

复杂度：Time `$O(n^3)$`，Space `$O(n^2)$`。

坑：如果去想"先戳哪个气球"，子问题的边界会依赖"还有哪些气球没戳"，转移写不清楚；换成"这个区间里最后戳哪个"之后，左右两段子区间互不影响，才能独立递归。

## 背包问题：0/1 Knapsack vs Complete Knapsack

背包 DP 的核心问题是：

```text
有一些物品，每个物品有 cost / weight / value。
在容量限制内，求是否可行、最大价值、最小数量、方案数。
```

面试里最重要的区别是：

```text
0/1 背包：每个物品最多用一次
完全背包：每个物品可以用无限次
```

这两个问题的代码经常只差一行循环方向，但语义完全不同。

## 0/1 背包例题：Partition Equal Subset Sum

在讲 Target Sum 之前，先看一道更直接的 0/1 背包可行性问题：给定数组，能否把它分成两个子集，使两个子集的和相等。

如果总和是奇数，直接无解。否则目标是从数组里选出若干个数（每个数最多选一次），凑出 `$target = sum / 2$`：

$$dp[j] = dp[j] \lor dp[j - x]$$

`$dp[j]$` 表示"能否凑出和 `$j$`"，这是一个布尔版本的 0/1 背包。

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

复杂度：Time `$O(n\cdot target)$`，Space `$O(target)$`。

坑：和下面的 Target Sum 一样，容量循环必须倒序，否则同一个数 `x` 会在同一轮里被用多次，退化成完全背包。

## 0/1 背包例题：Target Sum

题目：给定整数数组 `nums` 和整数 `target`。对每个数都必须选择加号或减号，问有多少种表达式能得到 `target`。

例如：

```text
nums = [2, 2, 2], target = 2

+2 +2 -2 = 2
+2 -2 +2 = 2
-2 +2 +2 = 2

answer = 3
```

这题表面是加减号，实际可以转成 0/1 背包。

把所有被加号选中的数放进集合 `P`，被减号选中的数放进集合 `N`：

```text
sum(P) - sum(N) = target
sum(P) + sum(N) = total
```

两式相加：

```text
2 * sum(P) = target + total
sum(P) = (target + total) / 2
```

所以问题变成：

```text
从 nums 里选一些数，每个数最多选一次，使它们的和等于 bag = (target + total) / 2。
问有多少种选法。
```

这就是 0/1 背包的“方案数”版本。

什么时候无解？

```text
abs(target) > total
target + total 是奇数
```

因为 `sum(P)` 必须是一个非负整数。

## Target Sum：二维 0/1 背包

定义状态：

```text
dp[i][s] = 只看前 i 个数，凑出和 s 的方案数
```

Base case：

```text
dp[0][0] = 1
```

含义是：不用任何数，凑出 0 有一种方法，就是什么都不选。

转移：

```text
不选 nums[i - 1]：dp[i - 1][s]
选 nums[i - 1]：dp[i - 1][s - nums[i - 1]]

dp[i][s] = dp[i - 1][s] + dp[i - 1][s - nums[i - 1]]
```

代码：

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

用例：

```text
nums = [2, 2, 2], target = 2
total = 6
bag = (2 + 6) / 2 = 4

问题变成：从 [2, 2, 2] 中选一些数，和为 4，有几种选法？

选第 1、2 个 2
选第 1、3 个 2
选第 2、3 个 2

answer = 3
```

注意：这里三个 `2` 是三个不同位置的数，所以选第 1、2 个和选第 1、3 个是不同方案。

## Target Sum：一维空间优化

二维转移只依赖上一行：

```text
dp[i][s] depends on dp[i - 1][s] and dp[i - 1][s - num]
```

所以可以压成一维：

```text
dp[s] = 当前已经处理过的数里，凑出和 s 的方案数
```

0/1 背包的一维循环必须倒序：

```text
for s from bag down to num:
```

原因：每个数只能用一次。倒序可以保证 `dp[s - num]` 还是“上一轮物品”的结果，而不是当前这个 `num` 刚更新出来的结果。

代码：

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

为什么倒序这么重要？

假设只有一个数 `2`，`bag = 4`。

如果正序：

```text
dp[2] += dp[0]  # 用了一次 2
dp[4] += dp[2]  # 又立刻用刚更新的 dp[2]，等于同一个 2 用了两次
```

这就变成完全背包了，语义错了。

## 完全背包是什么

完全背包的区别是：

```text
每个物品可以用无限次
```

例如 coin change：

```text
coins = [1, 2, 5], amount = 11
```

每种硬币可以用很多枚，这就是完全背包。

完全背包的一维循环通常正序：

```text
for coin in coins:
  for s from coin to amount:
    dp[s] = combine(dp[s], dp[s - coin])
```

正序的含义是：允许当前 `coin` 被重复使用。因为当你计算 `dp[s]` 时，`dp[s - coin]` 可能已经包含当前 coin 的结果。

## 完全背包：最少硬币数

如果要求最少硬币数：

```text
dp[s] = 凑出金额 s 的最少硬币数
```

转移：

```text
dp[s] = min(dp[s], dp[s - coin] + 1)
```

代码：

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

这里 `s` 正序，因为一枚 `coin` 可以被反复使用。

## 完全背包例题：Coin Change II

Coin Change 求的是最少硬币数，Coin Change II 换成完全背包的计数版本：`$dp[a]$` = 凑出金额 `$a$` 的组合数（不是排列数）。

```python
class Solution:
    def change(self, amount: int, coins: list[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for c in coins:                 # 硬币在外层 → 组合
            for a in range(c, amount + 1):
                dp[a] += dp[a - c]
        return dp[amount]
```

复杂度：Time `$O(amount\cdot|coins|)$`，Space `$O(amount)$`。

坑：如果把硬币放在内层、金额放在外层，会把 `(1, 2)` 和 `(2, 1)` 当成两种不同方案，算成排列数而不是组合数，这正是前面"计数问题：组合数和排列数的循环顺序不同"一节讲的那个坑，这里是它的具体例子。`dp[0]=1` 表示"凑出 0"只有"什么都不选"这一种方式。

## 0/1 背包和完全背包怎么选

先问物品能不能重复用：

```text
每个元素只能选一次          -> 0/1 背包
每种物品可以选无限次        -> 完全背包
每个物品最多选 k 次         -> 多重背包
```

再问目标是什么：

```text
是否能凑出容量              -> boolean dp
有多少种凑法                -> count dp
最大价值 / 最小数量         -> max / min dp
```

最后决定循环方向：

```text
0/1 背包一维优化：
for item in items:
  for capacity from target down to item:
    use previous item layer

完全背包一维优化：
for item in items:
  for capacity from item up to target:
    allow using current item again
```

一句话记忆：

> 0/1 背包倒序，防止同一个物品重复使用；完全背包正序，主动允许同一个物品重复使用。

## 背包速查：一维模板、初始化与循环顺序

上面两道例题都套着题目外壳。把外壳拆掉，一维背包只剩下这两段代码：

```python
# 0/1 背包：每个物品最多用一次，求最大价值
dp = [0] * (W + 1)
for weight, value in items:
    for c in range(W, weight - 1, -1):     # 倒序
        dp[c] = max(dp[c], dp[c - weight] + value)
return dp[W]

# 完全背包：每个物品可以用无限次，求最大价值
dp = [0] * (W + 1)
for weight, value in items:
    for c in range(weight, W + 1):         # 正序
        dp[c] = max(dp[c], dp[c - weight] + value)
return dp[W]
```

差别只有内层循环方向。倒序时 `dp[c - weight]` 还停在上一个物品处理完的状态，当前物品最多被用一次；正序时 `dp[c - weight]` 可能已经含有当前物品，同一个物品就能叠加多次。

`range(W, weight - 1, -1)` 的下界写成 `weight - 1` 是因为 `range` 不包含终点，这样最后一个取到的容量正好是 `weight`。容量小于 `weight` 的格子放不下这个物品，跳过即可。

两种写法的复杂度都是时间 $O(nW)$、空间 $O(W)$。注意 `W` 出现在复杂度里：容量到 $10^9$ 这种量级时背包 DP 直接不可用，要换成贪心、数论或搜索。

### 初始化取决于“恰好装满”还是“不超过”

`dp` 数组的初值不是固定的 0，它编码的是“除了容量 0 以外，其他容量一开始合不合法”：

| 问题目标 | `dp[0]` | 其余初值 | 语义 |
|---|---|---|---|
| 最大价值，允许不装满 | `0` | `0` | 任何容量都可以什么都不放 |
| 最大价值，必须恰好装满 | `0` | `-inf` | 还没凑出这个容量，非法 |
| 最小数量，必须恰好装满 | `0` | `+inf`（或 `amount + 1` 当哨兵） | 同上，方向相反 |
| 方案数 | `1` | `0` | 凑出容量 0 只有“什么都不选”一种方式 |
| 可行性 | `True` | `False` | 容量 0 总是可达 |

上面 coin change 用 `inf = amount + 1` 而不是真正的无穷大，是因为凑出 `amount` 最多用 `amount` 枚面值为 1 的硬币，`amount + 1` 已经超过任何合法答案，同时避免了 `inf + 1` 的溢出讨论。

### 计数问题：组合数和排列数的循环顺序不同

方案数版本还有一个坑：外层循环放物品还是放容量，决定了你数的是组合还是排列。

```python
# 组合数：{1,2} 和 {2,1} 算一种  -  物品在外层
for coin in coins:
    for s in range(coin, amount + 1):
        dp[s] += dp[s - coin]

# 排列数：{1,2} 和 {2,1} 算两种  -  容量在外层
for s in range(1, amount + 1):
    for coin in coins:
        if coin <= s:
            dp[s] += dp[s - coin]
```

物品在外层时，每个物品只在自己那一轮被考虑，选择顺序被钉死成物品的枚举顺序，所以同一个集合只会被数一次。容量在外层时，同一个容量下每种硬币都有机会当“最后一枚”，顺序就被区分开了。Coin Change II 问的是组合数，Combination Sum IV 问的是排列数，两题的代码几乎一样，循环顺序反了答案就错。

### 常见坑

- 循环方向写反：0/1 背包写成正序会变成完全背包，同一个物品被反复选。
- 初始化没跟着题目走：题目要求恰好装满却把整个数组初始化成 0，会把不可达的容量当成合法解。
- 计数题的循环顺序反了：组合数写成排列数，样例能过、大样例数值偏大。
- 忘记跳过 `weight > c` 的格子，或者倒序的下界写成 `-1`，导致 `dp[c - weight]` 用到负下标。
- 二维转一维之后还想回溯具体选了哪些物品：一维数组已经把中间层覆盖掉了，要输出方案就得保留二维表。

## 双序列 DP：两个字符串一起推进

状态变成两个下标 `$(i, j)$`，分别是两个字符串的前缀长度。转移通常来自三个方向：`$dp[i-1][j-1]$`（两个指针一起走一步）、`$dp[i-1][j]$`（只走第一个串）、`$dp[i][j-1]$`（只走第二个串）。

## Longest Common Subsequence

`$dp[i][j]$` = `text1[:i]` 与 `text2[:j]` 的最长公共子序列长度。

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

复杂度：Time `$O(mn)$`，Space `$O(mn)$`（可以滚动压到 `$O(\min(m,n))$`）。

坑：两个字符相等时必须走对角线 `+1`，不要写成 `max(上, 左) + 1`，那样会把同一个字符重复计入。

## Edit Distance

`$dp[i][j]$` = 把 `word1[:i]` 变成 `word2[:j]` 所需的最少操作数。

$$dp[i][j]=\begin{cases}dp[i-1][j-1] & \text{word1}[i-1]=\text{word2}[j-1]\\ 1+\min(dp[i-1][j],\ dp[i][j-1],\ dp[i-1][j-1]) & \text{otherwise}\end{cases}$$

三个候选分别对应删除、插入、替换。

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

复杂度：Time `$O(mn)$`，Space `$O(mn)$`。

坑：base case 是把前缀删空或从空串插入成前缀所需的代价，也就是 `$i$` 和 `$j$`；两个字符相等时直接抄对角线，不要再 `+1`。

## Interleaving String

`$dp[i][j]$` = `s1[:i]` 和 `s2[:j]` 能否交错组成 `s3[:i+j]`。

$$dp[i][j] = \bigl(dp[i-1][j]\land s1[i-1]=s3[i+j-1]\bigr) \lor \bigl(dp[i][j-1]\land s2[j-1]=s3[i+j-1]\bigr)$$

先检查 `$\text{len}(s1)+\text{len}(s2)=\text{len}(s3)$`，长度不对直接返回 `False`。

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

复杂度：Time `$O(mn)$`，Space `$O(mn)$`。

坑：忘记先检查长度；`s3` 的下标是 `$i+j-1$`，不是 `$i+j$`（0-indexed 的偏移很容易错一位）。

## Distinct Subsequences

`$dp[i][j]$` = `s[:i]` 中有多少个子序列等于 `t[:j]`。

$$dp[i][j]=\begin{cases}dp[i-1][j-1]+dp[i-1][j] & s[i-1]=t[j-1]\\ dp[i-1][j] & \text{otherwise}\end{cases}$$

Base：`$dp[i][0]=1$`（空 `t` 总能被匹配一次，即"什么都不取"），`$dp[0][j]=0$`（`$j>0$` 时非空 `t` 不可能被空 `s` 匹配）。

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

复杂度：Time `$O(mn)$`，Space `$O(mn)$`。

坑：字符相等时是"用这个字符去匹配 t 的末位"和"跳过 s 的这个字符"两条路径相加，漏掉 `dp[i-1][j]` 会少算方案。

## Regular Expression Matching

`$dp[i][j]$` = `s[:i]` 是否被 `p[:j]` 匹配。

- 普通字符或 `.`：`$dp[i][j]=dp[i-1][j-1]$`，且要求当前字符匹配。
- `$p[j-1]='*'$`：`$x*$` 匹配零次时看 `$dp[i][j-2]$`（整段 `x*` 丢掉）；匹配当前字符时，如果 `$x$` 能匹配 `$s[i-1]$`，还要或上 `$dp[i-1][j]$`（消耗一个 `s` 字符，pattern 停在 `x*` 上继续复用）。

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

复杂度：Time `$O(mn)$`，Space `$O(mn)$`。

坑：`*` 管的是前一个字符，写转移时要用 `$j-2$` / `$p[j-2]$`，不是 `$j-1$`；空 pattern 对 `a*`、`a*b*` 这类可以匹配空串的情况，要先把第 0 行填对。

## Kadane's Algorithm：最大子数组和

Kadane 算法解决的是 Maximum Subarray：

```text
给定数组 nums，找到一个连续子数组，使它的和最大。
```

例如：

```text
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
answer = 6

最优子数组是 [4, -1, 2, 1]
sum = 6
```

这题经常被当成贪心讲，但本质也可以看成非常简洁的一维 DP。

## Kadane：状态定义

定义状态：

```text
dp[i] = 必须以 nums[i] 结尾的最大连续子数组和
```

注意是“必须以 `i` 结尾”，不是“前 `i` 个元素里的最大答案”。这个定义让转移非常稳定。

对于 `nums[i]`，只有两种选择：

```text
1. 接在前面的子数组后面：dp[i - 1] + nums[i]
2. 从当前位置重新开始：nums[i]
```

所以状态方程：

```text
dp[i] = max(dp[i - 1] + nums[i], nums[i])
```

最终答案：

```text
answer = max(dp[i] for all i)
```

完整 DP 写法：

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

## Kadane：为什么负数前缀可以丢掉

如果当前累积和是负数：

```text
curSum < 0
```

那么它接到任何未来子数组前面，只会让未来的和变小。

例如未来从 `x` 开始：

```text
curSum + x < x
```

所以负数前缀没有保留价值，可以直接从下一个位置重新开始。

这就是 Kadane 里这句的含义：

```python
if curSum < 0:
    curSum = 0
```

## Kadane：空间优化写法

因为 `dp[i]` 只依赖 `dp[i - 1]`，所以不需要完整数组，只保留一个变量：

```text
curSum = 当前正在延续的子数组和
maxSub = 到目前为止见过的最大子数组和
```

代码：

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

为什么 `maxSub = nums[0]`，而不是 `0`？

因为数组可能全是负数：

```text
nums = [-5, -2, -7]
answer = -2
```

如果把答案初始化成 `0`，就会错误地返回空子数组。但题目要求子数组非空，所以必须用第一个元素初始化答案。

## Kadane：手动走一遍

用经典例子：

```text
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
```

过程：

```text
num   curSum after update   maxSub
-2    -2                    -2
 1     1                     1      # 前面 curSum < 0，丢掉 -2
-3    -2                     1
 4     4                     4      # 前面 curSum < 0，从 4 重新开始
-1     3                     4
 2     5                     5
 1     6                     6
-5     1                     6
 4     5                     6
```

最终答案是 `6`。

## Kadane：和普通 DP 的关系

Kadane 可以这样理解：

```text
完整 DP：
dp[i] = max(dp[i - 1] + nums[i], nums[i])
ans = max(ans, dp[i])

空间优化：
curSum = dp[i - 1]
更新后 curSum = dp[i]
```

所以它不是魔法，本质就是：

```text
定义“以当前位置结尾”的状态，然后把 dp 数组压成一个变量。
```

这类题常见变形：

- 最大乘积子数组：需要同时记录最大值和最小值，因为负数会翻转符号。
- 环形最大子数组：需要比较普通最大子数组和 `total_sum - 最小子数组和`。
- 买卖股票一次最大利润：可以看成维护历史最低买入价，也可以转成最大差分子数组。

## Jump Game：DP 什么时候可以变成 Greedy

Jump Game 问的是：

```text
给定 nums，从 index 0 出发。
nums[i] 表示从 i 最多可以向右跳 nums[i] 步。
问能不能到达最后一个 index。
```

例子：

```text
nums = [2, 3, 1, 1, 4]
answer = true

0 -> 1 -> 4
```

反例：

```text
nums = [3, 2, 1, 0, 4]
answer = false

无论怎么跳，都会被 index 3 的 0 卡住。
```

这题很适合用来理解一个更重要的问题：

> 什么样的 DP 可以进一步压成 Greedy？

不是所有 DP 都能贪心。能贪心的 DP，通常有一个共同特征：

```text
一大批状态的答案，可以被一个单调边界、最优代表、或极值变量完整概括。
```

Jump Game 里的这个代表变量就是 `goal`。

## Jump Game：递归视角

先不要急着写 greedy。先从最直接的搜索开始。

定义递归函数：

```text
dfs(i) = 从 index i 出发，能不能到达最后一个 index
```

从 `i` 可以跳到：

```text
i + 1, i + 2, ..., i + nums[i]
```

所以递归关系是：

```text
dfs(i) = dfs(i + 1) or dfs(i + 2) or ... or dfs(i + nums[i])
```

代码：

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

这个写法逻辑清楚，但会重复计算大量状态。比如很多路径都会问同一个问题：

```text
从 index 3 出发能不能到终点？
```

所以它会超时。

## Jump Game：Top-Down DP

加 memo 后，状态不变：

```text
dfs(i) = 从 i 出发能不能到最后
```

只是把算过的 `dfs(i)` 缓存起来。

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

复杂度：

```text
Time:  O(n^2)
Space: O(n)
```

为什么是 `O(n^2)`？

因为每个 `i` 最多算一次，但每个 `i` 内部可能扫描很多个 `j`。

## Jump Game：Bottom-Up DP

把递归反过来，从右往左填表。

定义：

```text
dp[i] = 从 index i 出发，能不能到达最后一个 index
```

base case：

```text
dp[n - 1] = true
```

转移：

```text
dp[i] = any(dp[j] == true for j in [i + 1, i + nums[i]])
```

代码：

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

到这里还是标准 DP。

关键问题来了：

```text
dp[i] 真的需要知道右边每一个 dp[j] 吗？
```

如果需要，那就不能贪心。

如果不需要，而只需要知道一个“最关键的位置”，那就有机会变成 greedy。

## Jump Game：观察 DP 表

看例子：

```text
nums = [2, 3, 1, 1, 4]
index  0  1  2  3  4
```

从右往左看。

最后一个位置一定是好位置：

```text
index  0  1  2  3  4
good              T
goal = 4
```

`i = 3`：

```text
3 + nums[3] = 3 + 1 = 4
```

它能跳到 `goal = 4`，所以 index 3 也是好位置：

```text
index  0  1  2  3  4
good           T  T
goal = 3
```

`i = 2`：

```text
2 + nums[2] = 3
```

它能跳到 `goal = 3`，所以 index 2 也是好位置：

```text
index  0  1  2  3  4
good        T  T  T
goal = 2
```

`i = 1`：

```text
1 + nums[1] = 4
```

它能跳过 `goal = 2`，当然也能到达好位置，所以：

```text
goal = 1
```

`i = 0`：

```text
0 + nums[0] = 2
```

它能到达 `goal = 1`，所以：

```text
goal = 0
```

最终 `goal == 0`，答案是 `true`。

这里发生了一件很重要的事情：

```text
DP 原本保存所有 good positions。
Greedy 只保存最左边的 good position。
```

为什么只保存最左边的好位置就够了？

因为如果某个 index `i` 能跳到最左边的好位置 `goal`，那么它一定能到达终点。

如果 `i` 连最左边的好位置都跳不到，那么它跳到更右边的好位置只会更难。

所以右侧一整段 DP 信息，可以被一个边界变量压缩：

```text
goal = 当前已经知道的、最靠左的可达终点的位置
```

## Jump Game：Greedy 写法

从右往左扫描：

```text
如果 i 能跳到 goal 或超过 goal：
    i 变成新的 goal
```

代码：

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

复杂度：

```text
Time:  O(n)
Space: O(1)
```

这不是“凭感觉贪心”，而是从 DP 压缩出来的 greedy：

```text
dp array of good positions
        ↓
leftmost good position
        ↓
goal
```

## Jump Game：为什么这个 Greedy 是正确的

我们维护一个不变量：

```text
在扫描到 i 的时候，goal 是 [i + 1, n - 1] 中最靠左的好位置。
```

好位置的意思是：

```text
从这个位置出发，可以到达最后一个 index。
```

当我们检查 index `i`：

```text
i + nums[i] >= goal
```

说明 `i` 可以一步跳到 `goal`，而 `goal` 已经能到终点。

所以 `i` 也能到终点。并且 `i` 比旧的 `goal` 更靠左，于是更新：

```text
goal = i
```

如果：

```text
i + nums[i] < goal
```

说明 `i` 连最靠左的好位置都够不到。

更右边的好位置位置更大，更不可能够到，所以 `i` 不是好位置。

扫描结束后：

```text
goal == 0
```

就表示 index 0 是好位置。

## DP 什么时候可以转换为 Greedy

可以用下面这组问题判断。

### 1. DP 状态是否只在维护一个“可行集合”

Jump Game 的 DP 是：

```text
dp[i] = i 是否能到终点
```

它维护的是一组好位置：

```text
good positions = { i | dp[i] == true }
```

如果 DP 维护的是复杂数值，例如每个状态都有不同代价、不同路径、不同选择历史，通常不容易直接贪心。

### 2. 这个集合能不能被一个边界表示

Jump Game 里，我们不需要知道所有好位置，只需要知道：

```text
最左边的好位置 goal
```

原因是目标只有一个：

```text
能不能到达某个好位置
```

而 `goal` 是所有好位置里最容易够到的那个。

这就是单调边界。

### 3. 局部更新会不会破坏未来选择

当 `i` 能到 `goal` 时，我们把 `goal` 更新成 `i`。

这不会让未来更差，反而让未来更容易：

```text
旧 goal 更靠右
新 goal 更靠左
```

之后更左边的 index 想要够到一个好位置，只会更容易，不会更难。

这种性质叫贪心选择安全：

```text
local choice does not remove any globally optimal possibility
```

### 4. 是否存在“支配关系”

在 Jump Game 里：

```text
更靠左的 good position 支配更靠右的 good position
```

因为对于左侧的某个 `i` 来说：

```text
能跳到更左的 good position
=> 一定比跳到更右的 good position 更容易
```

所以我们可以丢掉所有更右的 good positions，只保留最左边那个。

这是 DP 转 greedy 最重要的信号：

> 多个状态之间存在支配关系，被支配的状态永远不可能比代表状态更有用。

## DP 转 Greedy 的通用模式

很多 greedy 不是凭空想出来的，而是这样推出来的：

```text
1. 写出完整 DP。
2. 观察 DP 保存了哪些信息。
3. 判断这些信息是否有单调性。
4. 找到能代表一整批状态的变量。
5. 证明只保留这个变量不会丢答案。
```

Jump Game 对应：

```text
完整 DP：dp[i] 表示 i 是否 good
保存信息：所有 good positions
单调性：越靠左的 good position 越容易被左侧 index reach
代表变量：leftmost good position，也就是 goal
安全性：能 reach goal 就一定能 reach end
```

所以：

```text
O(n^2) DP
  -> 保存所有 good positions
  -> 只保存 leftmost good position
  -> O(n) Greedy
```

## 另一个 Greedy 角度：从左往右维护最远可达

Jump Game 还有一个常见写法，从左往右扫：

```text
reach = 当前为止最远能到达的位置
```

如果扫描到 `i` 时：

```text
i > reach
```

说明当前位置都到不了，后面也没法继续，返回 `False`。

否则更新：

```text
reach = max(reach, i + nums[i])
```

代码：

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

这个写法维护的是另一个单调边界：

```text
[0, reach] 这一段都已经可以被某种方式到达
```

只要 `reach` 不断向右扩张，最后超过 `n - 1` 就可行。

两个 greedy 写法本质一样：

```text
从右往左：维护最左 good position
从左往右：维护最远 reachable position
```

它们都把一整组 DP 状态压成了一个边界变量。

## Jump Game：面试里怎么讲

比较好的回答顺序是：

1. 我先定义 `dp[i]`：从 `i` 出发能不能到终点。
2. 递推是检查 `i` 能跳到的所有 `j`，只要某个 `dp[j]` 为 true，`dp[i]` 就为 true。
3. 这个 bottom-up DP 是 `O(n^2)`。
4. 观察到我们不需要保存所有 true 的位置，只需要保存最左边的 true 位置 `goal`。
5. 如果 `i + nums[i] >= goal`，那 `i` 可以到达一个已知好位置，所以 `i` 也是好位置。
6. 因为更左的好位置对未来更有利，所以更新 `goal = i` 是安全的。
7. 最后判断 `goal == 0`。

一句话总结：

> 这题能从 DP 变成 greedy，是因为“所有能到终点的位置”可以被“最左边的能到终点的位置”完整代表。

## 网格 DP：坐标 (r, c) 上的两种情况

## Unique Paths

机器人只能向右或向下走，从左上角走到右下角有多少条不同路径。`$dp[i][j]=dp[i-1][j]+dp[i][j-1]$`，第一行和第一列都只有一条走法，全部初始化为 1。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
```

复杂度：Time `$O(mn)$`，Space `$O(n)$`（滚动成一维数组）。

坑：`$m=1$` 或 `$n=1$` 时答案应该是 1，上面的滚动数组写法已经自动覆盖了这个情况，不需要额外特判。

## Longest Increasing Path in a Matrix

这题看起来是网格 DP，但不能像 Unique Paths 那样按行列扫一遍：矩阵里的数值任意分布，不存在一个方向能保证"算某个格子时，它依赖的格子都已经算完"。

正确做法是对每个格子做记忆化 DFS：`$dfs(r,c)$` = 从这个格子出发的最长严格递增路径长度，只往严格更大的邻居走：

$$dfs(r,c)=1+\max_{\text{邻格严格更大}} dfs(nr,nc)$$

（如果没有合法邻居，长度为 1。）用 memo 缓存结果。

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

复杂度：Time `$O(mn)$`（每个格子只会被真正计算一次），Space `$O(mn)$`（memo 加递归栈）。

这为什么仍然算 DP：严格递增的边构成一个有向无环图（DAG），子问题 `$dfs(r,c)$` 会被多条路径重复访问，用 memo 缓存就是标准的"最优子结构 + 重叠子问题"。区别只是这里没有写显式的双重 for 循环填表，而是用记忆化 DFS 沿着 DAG 的拓扑顺序隐式地算。

坑：写成 `>=` 而不是 `>` 会在有相等值时出现环，递归无法终止；忘记 memo 会导致同一个格子被重复展开，退化成指数级。

## 常见坑

- 没有先定义 `dp[i]` 的含义就开始写代码。
- 循环方向和依赖方向反了。
- 忘记 base case，例如 `dp[n] = 1`。
- 把非法状态当成 `1`，尤其是 Decode Ways 里的 `'0'`。
- 过早做空间优化，导致变量含义说不清楚。
- 股票 cooldown 题里，卖出后直接从 `sold` 买入，漏掉 cooldown。
- 空间优化时覆盖旧变量，导致 `sold` 使用了当天刚更新的 `hold`。
- 0/1 背包一维优化时用了正序，导致同一个物品被重复使用。
- Target Sum 忘记先判断 `(target + total)` 是否为偶数。
- Target Sum 忘记 `nums[i] = 0` 也会产生不同符号方案；背包计数写法会自然处理这个情况。
- Kadane 把 `maxSub` 初始化成 `0`，导致全负数数组返回错误。
- Kadane 忘记子数组必须连续，不能像子序列一样随意跳过中间元素。
- Jump Game 把 `nums[i]` 当成唯一目的地；它其实是最大跳跃长度，可以跳到区间 `[i + 1, i + nums[i]]` 里的任意位置。
- Jump Game 直接说“每次跳最远”但不给证明；面试里要解释为什么状态集合可以被 `goal` 或 `reach` 这个边界代表。
- Jump Game 从右往左 greedy 时忘记条件是 `i + nums[i] >= goal`，不是 `nums[i] >= goal`。

- 答案取 `dp[n]` 还是 `max(dp)` 搞混，尤其是 LIS、Maximum Product Subarray、Min Cost Climbing Stairs 这类"答案不等于最后一格"的题。
- 0/1 背包一维优化写成正序，导致同一个物品在一轮里被重复使用，变成完全背包。
- Coin Change II 外层放金额、内层放硬币，把组合数写成了排列数。
- 区间 DP 不按长度递增填表，读到还没算出来的 `dp[i+1][j-1]`。
- Burst Balloons 想着"先戳哪个气球"，而不是"这个区间里最后戳哪个"，导致子问题边界说不清楚。
- Regex Matching 的 `*` 忘记对应"前一个字符"，或者空串对 `a*` 这类 pattern 的 base case 没处理。
- House Robber II 忘了单独处理 `n=1` 的情况。
- Longest Increasing Path in a Matrix 用普通的双重循环按行列填表，而不是记忆化 DFS。

## 复杂度

Decode Ways 数组版：

- 时间：`O(n)`
- 空间：`O(n)`

空间优化版：

- 时间：`O(n)`
- 空间：`O(1)`

Stock Cooldown 三状态数组版：

- 时间：`O(n)`
- 空间：`O(n)`

Stock Cooldown 空间优化版：

- 时间：`O(n)`
- 空间：`O(1)`

Target Sum 0/1 背包：

- 时间：`O(n * bag)`，其中 `bag = (target + sum(nums)) / 2`
- 空间：二维版 `O(n * bag)`，一维版 `O(bag)`

完全背包：

- 时间：通常 `O(number_of_items * capacity)`
- 空间：一维优化后 `O(capacity)`

Kadane 最大子数组和：

- 时间：`O(n)`
- 空间：完整 DP 为 `O(n)`，空间优化后为 `O(1)`

Jump Game：

- 递归：时间指数级，空间 `O(n)`
- Top-down / bottom-up DP：时间 `O(n^2)`，空间 `O(n)`
- Greedy：时间 `O(n)`，空间 `O(1)`

Climbing Stairs / Min Cost Climbing Stairs / House Robber / House Robber II：

- 时间：`O(n)`
- 空间：`O(1)`（滚动变量）

Word Break：

- 时间：`O(n^2 · L)`，`L` 为子串比较成本
- 空间：`O(n)`

Longest Increasing Subsequence（本文的 O(n²) DP 版本）：

- 时间：`O(n^2)`
- 空间：`O(n)`

Maximum Product Subarray：

- 时间：`O(n)`
- 空间：`O(1)`

Longest Palindromic Substring / Palindromic Substrings：

- 时间：`O(n^2)`
- 空间：`O(n^2)`

Burst Balloons：

- 时间：`O(n^3)`
- 空间：`O(n^2)`

Partition Equal Subset Sum：

- 时间：`O(n · sum/2)`
- 空间：`O(sum/2)`

Coin Change II：

- 时间：`O(amount · |coins|)`
- 空间：`O(amount)`

Longest Common Subsequence / Edit Distance / Interleaving String / Distinct Subsequences / Regular Expression Matching：

- 时间：`O(mn)`
- 空间：`O(mn)`（LCS 和 Edit Distance 都可以滚动压缩到 `O(min(m,n))`）

Unique Paths：

- 时间：`O(mn)`
- 空间：`O(n)`（滚动数组）

Longest Increasing Path in a Matrix：

- 时间：`O(mn)`
- 空间：`O(mn)`（memo + 递归栈）

## 快速自测

```quiz
title: 快速选择题 1
question: House Robber II 相对 House Robber，最关键的额外处理是？
answer: B
A. 改成二维 dp[i][j]
B. 环拆成两段线性 Robber，取 max
C. 必须用区间 DP
D. 改成完全背包
explanation: 首尾相邻，不能同时抢；对 nums[0..n-2] 与 nums[1..n-1] 各跑一次线性 Robber 再取 max。
```

```quiz
title: 快速选择题 2
question: Coin Change（最少枚数）与 Coin Change II（组合数）在一维 DP 上，最重要的差别是？
answer: C
A. 一个用倒序一个用正序（都是完全背包时）
B. 一个必须二维一个必须一维
C. 目标函数不同（min vs 累加计数），且 II 要用"物品在外"保证组合语义
D. 二者转移完全相同只是返回值不同
explanation: 二者都是完全背包形状；II 需要组合而非排列，故硬币循环在外；转移分别是 min 与加法计数。
```

```quiz
title: 快速选择题 3
question: Burst Balloons 区间转移里，k 表示什么？
answer: A
A. 开区间 (i,j) 里最后戳破的气球
B. 开区间里最先戳破的气球
C. 区间长度
D. 虚拟边界 1 的下标
explanation: 最后戳 k 时左右已空，收益拆成 a[i]*a[k]*a[j] 加两段子区间 DP。
```

```quiz
title: 快速选择题 4
question: Target Sum 化成子集和时，子集目标和 P 等于？
answer: B
A. (sum - target) / 2
B. (sum + target) / 2
C. sum - target
D. target
explanation: P+N=sum，P-N=target，故 P=(sum+target)/2；需整除且 |target|<=sum。
```

```quiz
title: 快速选择题 5
question: Longest Increasing Path in a Matrix 为什么不能像 Unique Paths 那样按行扫 dp？
answer: D
A. 因为只能右和下走
B. 因为必须 O(1) 空间
C. 因为不是 DP
D. 更大邻居方向任意，没有单一合法填表顺序；用记忆化 DFS 走 DAG
explanation: 严格上升边构成 DAG；memo 化的 DFS 才是正确的 DP 形态。
```

```quiz
title: 快速选择题 6
question: Edit Distance 中 word1[i-1]==word2[j-1] 时，正确转移是？
answer: A
A. dp[i][j] = dp[i-1][j-1]
B. dp[i][j] = dp[i-1][j-1] + 1
C. dp[i][j] = max(dp[i-1][j], dp[i][j-1])
D. dp[i][j] = dp[i][j-1] + 1
explanation: 字符已相等，无需操作，直接继承对角；不相等才在 insert/delete/replace 中取 min 加一。
```

```quiz
title: 快速选择题 7
question: Maximum Product Subarray 为什么要同时维护 min_here？
answer: C
A. 为了处理 0
B. 为了 O(1) 空间
C. 负号会把最小乘积翻成最大，只维护 max 会丢解
D. 题目要求返回最小乘积
explanation: 例如 [-2,3,-4]；负负得正依赖此前的最小（最负）乘积。
```

```quiz
title: 快速选择题 8
question: Distinct Subsequences 在 s[i-1]==t[j-1] 时，转移应为？
answer: B
A. 只加 dp[i-1][j-1]
B. dp[i-1][j-1] + dp[i-1][j]（用或不用当前字符）
C. dp[i][j-1] + dp[i-1][j]
D. max(dp[i-1][j-1], dp[i-1][j])
explanation: 用当前字符匹配 t 的末位，或跳过 s 的当前字符，两路方案数相加。
```

```quiz
title: 快速选择题 9
question: Regex Matching 中 p[j-1]=='*' 时，"匹配零次"对应？
answer: A
A. dp[i][j-2]
B. dp[i-1][j]
C. dp[i-1][j-1]
D. dp[i][j-1]
explanation: x* 整段丢掉，看 p[:j-2] 是否已匹配 s[:i]；多次匹配才或上 dp[i-1][j]。
```

```quiz
title: 快速选择题 10
question: 0/1 背包一维优化时容量循环必须倒序，原因是？
answer: C
A. 更快
B. 为了组合数变排列数
C. 避免同一个物品在本轮被重复使用
D. 倒序才能处理完全背包
explanation: 倒序保证 dp[j-x] 仍是"未选当前物品"的旧值；正序会变成完全背包。
```

```quiz
title: 快速选择题 11
question: Stock with Cooldown 的 sold 状态含义最准确的是？
answer: B
A. 任意空仓
B. 今天刚卖出，明天处于冷却
C. 手持股票
D. 累计卖出次数
explanation: sold 专门标记"今日卖出"，次日只能进入 free，不能直接 buy。
```

```quiz
title: 快速选择题 12
question: House Robber II 中，为什么不能直接套用 House Robber 的线性 DP？
answer: B
A. 数据范围不同，需要换算法
B. 首尾房屋相邻，形成环，必须拆成两段线性子问题分别求解再取较大值
C. 环形数组必须用区间 DP
D. 环形数组下 dp 转移方程本身要变
explanation: 环带来的唯一变化是"不能同时抢第一家和最后一家"，拆成 [0..n-2] 与 [1..n-1] 两段线性 Robber 就绕开了这个限制，单段内部的转移方程完全不变。
```

## 面试回答模板

DP 题可以这样讲：

1. 我先定义状态 `dp[...]` 表示什么。
2. 然后列出每一步有哪些选择。
3. 把每个选择转成对旧状态的引用，得到递推式。
4. 根据递推依赖决定循环顺序。
5. 先写完整 DP。
6. 最后观察只依赖哪些状态，再做空间优化。
