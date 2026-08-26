# Dynamic Programming Patterns：23 题六套骨架

DP 题看起来很多，但骨架只有六种。拿到题先问：状态是一维下标、区间、背包容量、双序列、网格坐标，还是有限个命名状态？

这份笔记覆盖下面 23 道题，每道题只讲状态 / 转移 / base / 遍历顺序 / 答案怎么取，以及一个具体坑。Decode Ways 已在 CoreSkills10 完整展开，这里只做归类与交叉引用。

| 骨架 | 代表题 |
|---|---|
| 1D 线性 DP | Climbing Stairs, House Robber, Coin Change, LIS, ... |
| 区间 DP | Longest Palindromic Substring, Burst Balloons |
| 背包族 | Partition Equal Subset Sum, Coin Change II, Target Sum |
| 双序列 DP | LCS, Edit Distance, Regex Matching, ... |
| 网格 DP | Unique Paths（例外：Longest Increasing Path 用记忆化 DFS） |
| 状态机 DP | Best Time to Buy and Sell Stock with Cooldown |

## 万能解题步骤

对上面每一道题，都按这五步走，不要直接跳到写循环：

1. **定义状态** `dp[...]`：一句话说清楚它精确表示什么（前缀 / 区间 / 容量 / 两个前缀 / 格子 / 第 i 天处于某状态）。
2. **想清楚最后一步**：当前状态是由哪些更小状态怎么组合出来的，写出转移方程（`min` / `max` / `+` / `or`）。
3. **确定 base case**：最小规模子问题的答案；很多 bug 出在这里，不是出在转移。
4. **确定计算顺序**：自底向上时保证被依赖的状态已算完；记忆化 DFS 时保证递归方向朝向更小规模，且 DAG 无环。
5. **从 dp 取最终答案**：是固定的 `dp[n]` / `dp[m][n]`，还是要对所有 `i` 取 `max`，还是滚动变量里的某个字段。边界漏了（空串、`n=1`、全零）通常就挂在这一步。

一句话：

```text
state + transition + base case + iteration order + answer extraction
```

## 万能模板：先套这一个

```text
这题的状态维度是什么？

├── 一个下标 i（前 i 个元素 / 爬到第 i 阶）
│     → 1D 线性 DP
│       代表题：Climbing Stairs, House Robber, Decode Ways, Word Break, LIS, Coin Change, Max Product
│
├── 一个闭区间 [i, j]（子串 / 气球区间），按长度递增填
│     → 区间 DP
│       代表题：Palindrome 系列, Burst Balloons
│
├── 物品 + 容量（选或不选 / 可重复选 / 计数）
│     → 0/1 背包 / 完全背包 / 计数背包
│       代表题：Partition Equal Subset Sum, Coin Change II, Target Sum
│
├── 两个序列的前缀 (i, j)
│     → 双序列 DP
│       代表题：LCS, Edit Distance, Interleaving String, Distinct Subsequences, Regex Matching
│
├── 网格格子 (r, c)，有固定扫掠方向
│     → 网格 DP
│       代表题：Unique Paths
│       例外：Longest Increasing Path 无固定方向 → 记忆化 DFS
│
└── 每一步有少量命名状态（持有 / 冷却 / 空仓）
      → 状态机 DP
        代表题：Stock with Cooldown
```

### 骨架 A：1D 线性 DP

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
| Decode Ways | 见 CoreSkills10 | 一位 / 两位合法则累加 | `dp[n]=1`；前导 `0` 为 0 | 后缀或前缀皆可 | `dp[0]` 或 `dp[n]` |
| Word Break | `dp[i]` = `s[:i]` 可否被拆 | `dp[j] and s[j:i] in dict` | `dp[0]=True` | `i=1..n`，内层 `j` | `dp[n]` |
| LIS | `dp[i]` = 以 `i` 结尾的 LIS 长 | `dp[i]=max(dp[j])+1` (`nums[j]<nums[i]`) | 全 `1` | 双重 `i,j<i` | `max(dp)` |
| Coin Change | `dp[a]` = 凑出金额 `a` 最少硬币 | `dp[a]=min(dp[a-c])+1` | `dp[0]=0`，其余 `inf` | 金额从小到大 | `dp[amount]`（`inf`→`-1`） |
| Max Product Subarray | 以 `i` 结尾的最大/最小乘积 | 同时滚 `max_here,min_here` | 首元素 | 一次扫描 | 全局 `max` |

### 骨架 B：区间 DP

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

### 骨架 C：背包族

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
| Coin Change II | `dp[a]` = 凑出 `a` 的组合数 | `dp[a]+=dp[a-c]`（完全） | `dp[0]=1` | **硬币外、金额内正序** | `dp[amount]` |
| Target Sum | 化为子集和计数 | 同 0/1 计数背包 | `dp[0]=1` | 物品外，容量倒序 | `dp[(sum+target)/2]` |

Coin Change（最少枚数）也是完全背包形状，但目标是 `min` 不是计数；放在骨架 A 是因为它常写成一维 `dp[amount]` 表。和 Coin Change II 对比：同样是完全背包，外层循环物品、内层金额 → 组合；若外层金额、内层物品 → 会把排列也算进去（本笔记 II 要的是组合）。

### 骨架 D：双序列 DP

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

### 骨架 E：网格 DP（含记忆化例外）

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

### 骨架 F：状态机 DP

```python
# hold / sold(cooldown) / free
for price in prices:
    hold, sold, free = TRANSITIONS(...)
return max(sold, free)
```

| 题 | 状态 | 转移 |
|---|---|---|
| Stock with Cooldown | `hold` / `sold` / `free` | 见下文三态方程 |

---

## 1D 线性 DP

### Climbing Stairs

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

坑：`$n=1$` 时不要访问 `$dp[2]$`；面试里说明这和斐波那契同构即可。

### Min Cost Climbing Stairs

可以从下标 `0` 或 `1` 起步。`dp[i]` = 到达台阶 `i` 并支付 `cost[i]` 后的最小总花费。

$$dp[i] = cost[i] + \min(dp[i-1], dp[i-2])$$

答案不是 `$dp[n]$`（顶楼无台阶），而是 `$\min(dp[n-1], dp[n-2])$`：最后一步可以从倒数第一或倒数第二跨上去。

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

坑：忘记“顶楼无 cost”，把答案写成 `$dp[n-1]$`。

### House Robber

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

坑：空数组 / 单元素要单独处理；下标是 `nums[i-1]` 还是 `nums[i]` 要和 `dp` 长度约定一致。

### House Robber II

房屋围成环：第一家和最后一家相邻。拆成两次线性 House Robber：

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

坑：漏掉 `$n=1$`；或者两次切片写成开区间错误导致漏抢中间唯一一家。

### Decode Ways

归类：1D 线性 DP（方案数）。一位合法加 `$dp[i-1]$`，两位在 `10..26` 加 `$dp[i-2]$`。

关键边界：前导 / 嵌入的 `'0'`。单独的 `'0'` 无法解码；`'10'`/`'20'` 合法，`'30'` 之类不合法。完整推导、空间优化与多种写法见 [CoreSkills10 Decode Ways Dynamic Programming](./CoreSkills10%20Decode%20Ways%20Dynamic%20Programming.md)。

### Coin Change

完全背包最小化：`$dp[a]$` = 凑出金额 `$a$` 的最少硬币数。

$$dp[a] = \min_{c\le a}(dp[a-c] + 1)$$

Base：`$dp[0]=0$`，其余初始化为 `$+\infty$`。凑不出返回 `-1`。

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

复杂度：Time `$O(amount\cdot|coins|)$`，Space `$O(amount)$`。

坑：用 `0` 当“未定义”会和“零枚凑出 0”混淆；初始化要用 `inf`。

### Maximum Product Subarray

负数会翻转大小关系，所以同时维护以当前位置结尾的最大乘积与最小乘积：

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

坑：只维护 max 会在 `[-2, 3, -4]` 上得到 3 而不是 24；遇到 `0` 会自然重置为从当前元素重开。

### Word Break

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

复杂度：Time `$O(n^2\cdot L)$`（`$L$` 为切片比较成本，可用字典树优化），Space `$O(n)$`。

坑：`$dp[0]=True$`（空前缀）；字典很大时先把 `wordDict` 变 `set`。

### Longest Increasing Subsequence

`$dp[i]$` = 以 `$nums[i]$` 结尾的 LIS 长度。

$$dp[i] = 1 + \max_{j<i,\,nums[j]<nums[i]} dp[j]$$

（无更小前驱则为 `1`。）答案 `$\max_i dp[i]$`。

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

脚注：可用耐心排序 + 二分做到 `$O(n\log n)$`，那是另一条优化线；本笔记的点是这张 `$O(n^2)$` DP 表。

坑：答案是 `max(dp)` 不是 `dp[-1]`；严格递增用 `<`，不是 `<=`。

---

## 区间 DP

### Longest Palindromic Substring

用区间布尔 DP：`$dp[i][j]$` 表示 `$s[i..j]$` 是否回文。

$$dp[i][j] = (s[i]=s[j]) \land (j-i<2 \lor dp[i+1][j-1])$$

按长度递增填表，同时记录最长区间起点与长度。中心扩展也能做，这里强调区间 DP 形态。

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

（等价写法：外层按 `length` 递增。）

复杂度：Time `$O(n^2)$`，Space `$O(n^2)$`。

坑：转移依赖 `$dp[i+1][j-1]$`，必须先算更短区间；单字符长度要计入初始最长。

### Palindromic Substrings

同一张布尔表，答案改为统计所有 `$dp[i][j]=True$` 的个数。

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

坑：不要只数中心扩展的“最大”回文；每个长度的回文子串都算一次。

### Burst Balloons

数组两端各补一个虚拟气球 `1`。定义开区间：`$dp[i][j]$` = 戳破 `$i$` 与 `$j$` **之间**所有气球能得到的最大硬币。

关键：枚举区间内**最后**戳破的气球 `$k$`（不是最先）。最后戳 `$k$` 时，左右已空，两边邻居正好是 `$a[i]$` 与 `$a[j]$`：

$$dp[i][j] = \max_{i<k<j}\bigl(dp[i][k] + a[i]\cdot a[k]\cdot a[j] + dp[k][j]\bigr)$$

按 `$j-i$` 递增填。答案 `$dp[0][n+1]$`。

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

坑：想“先戳谁”会让子问题边界依赖未戳气球，转移写不清；改成“最后戳谁”后左右子区间独立。

---

## 背包族

### Partition Equal Subset Sum

先判断总和是否偶数，目标 `$target=sum/2$`。0/1 背包可行性：`$dp[j]$` = 能否选出若干数凑出 `$j$`。

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

坑：容量必须**倒序**，否则同一个 `x` 在一层里被用多次，退化成完全背包。

### Coin Change II

完全背包**组合数**：`$dp[a]$` = 凑出 `$a$` 的方案数。

```python
class Solution:
    def change(self, amount: int, coins: list[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for c in coins:                 # 物品在外 → 组合
            for a in range(c, amount + 1):
                dp[a] += dp[a - c]
        return dp[amount]
```

对比 Coin Change（最少枚数）：目标函数不同（`min` vs `+`）。对比错误循环（金额在外、硬币在内）：会把 `(1,2)` 与 `(2,1)` 算成两种，变成排列数。

复杂度：Time `$O(amount\cdot|coins|)$`，Space `$O(amount)$`。

坑：循环顺序搞反；`dp[0]=1` 表示“一种方式凑出 0”。

### Target Sum

给每个数加 `+` 或 `-`，使表达式等于 `target`。令正数组之和为 `$P$`，负数组绝对值之和为 `$N$`：

$$P+N=\mathrm{sum},\quad P-N=\mathrm{target} \implies P=\frac{\mathrm{sum}+\mathrm{target}}{2}$$

于是变成：有多少个子集和为 `$P$`（0/1 计数背包）。`$(sum+target)$` 为奇或 `$|target|>sum$` 时答案为 0。

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

复杂度：Time `$O(n\cdot sum)$`，Space `$O(sum)$`。

坑：数组含 `0` 时，`0` 的 `+/-` 是两种贡献，倒序 0/1 计数会正确处理（同一容量上 `dp[j]+=dp[j-0]` 相当于乘 2）；正序会错。

---

## 双序列 DP

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

复杂度：Time `$O(mn)$`，Space `$O(mn)$`（可滚成 `$O(\min(m,n))$`）。

坑：相等时必须走对角 `+1`，不要写成 `max(上,左)+1`（会重复计）。

### Interleaving String

`$dp[i][j]$` = `$s1[:i]$` 与 `$s2[:j]$` 能否交错组成 `$s3[:i+j]$`。

$$dp[i][j] = (dp[i-1][j]\land s1[i-1]=s3[i+j-1]) \lor (dp[i][j-1]\land s2[j-1]=s3[i+j-1])$$

先检查 `$len(s1)+len(s2)=len(s3)$`。

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

坑：漏长度检查；`$s3$` 下标是 `$i+j-1$` 不是 `$i+j$`。

### Distinct Subsequences

`$dp[i][j]$` = `$s[:i]$` 中等于 `$t[:j]$` 的子序列个数。

$$dp[i][j]=\begin{cases}dp[i-1][j-1]+dp[i-1][j] & s[i-1]=t[j-1]\\ dp[i-1][j] & \text{otherwise}\end{cases}$$

Base：`$dp[i][0]=1$`（空 `t` 有一种匹配），`$dp[0][j]=0$`（`$j>0$`）。

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

坑：相等时是“用这个字符”+“不用这个字符”两路相加，漏加 `dp[i-1][j]` 会少算。

### Edit Distance

$$dp[i][j]=\begin{cases}dp[i-1][j-1] & word1[i-1]=word2[j-1]\\ 1+\min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]) & \text{otherwise}\end{cases}$$

三路分别对应 delete / insert / replace。

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

坑：base 是把前缀删空 / 插入成前缀的代价 `$i$` / `$j$`；相等时不要再 `+1`。

### Regular Expression Matching

`$dp[i][j]$` = `$s[:i]$` 是否被 `$p[:j]$` 匹配。

- 普通字符或 `.`：`$dp[i][j]=dp[i-1][j-1]$` 且字符匹配。
- `$p[j-1]='*'$`：`$x*$` 匹配零次 → `$dp[i][j-2]$`；或 `$x$` 匹配当前 `$s[i-1]$` 且 `$dp[i-1][j]$`（消耗一个 `$s$` 字符，pattern 停在 `$x*$`）。

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

坑：`*` 管的是**前一个**元素；写转移时用 `$j-2$` / `$p[j-2]$`。空串对 `a*`、`a*b*` 等的 base 要先填好第 0 行。

---

## 网格 DP

### Unique Paths

只能向右或向下。`$dp[i][j]=dp[i-1][j]+dp[i][j-1]$`，第一行/列全为 1。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
```

复杂度：Time `$O(mn)$`，Space `$O(n)$`。

坑：`$m=1$` 或 `$n=1$` 时答案是 1，滚动数组初始化已覆盖。

### Longest Increasing Path in a Matrix

对每个格子 `$dfs(r,c)$` = 从该格出发的最长递增路径长度：

$$dfs(r,c)=1+\max_{\text{邻格严格更大}} dfs(nr,nc)$$

（无合法邻居则为 1。）用 memo 缓存。

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

复杂度：Time `$O(mn)$`（每格算一次），Space `$O(mn)$`。

为什么仍是 DP：严格上升边构成 DAG，子问题 `$dfs(r,c)$` 重叠且无环，memo 就是 DP 表。不能按行扫是因为“更大邻居”可能在任意方向，不存在单一合法填表顺序。

坑：写成 `>=` 会出现环；忘记 memo 会指数爆炸。

---

## 状态机 DP

### Best Time to Buy and Sell Stock with Cooldown

卖出后次日不能买。三个状态（第 `$i$` 天结束时）：

- `$hold$`：手持股票
- `$sold$`：今天刚卖出（进入冷却）
- `$free$`：空仓且可买（含冷却已结束）

转移：

$$
\begin{aligned}
hold' &= \max(hold,\ free - price)\\
sold' &= hold + price\\
free' &= \max(free,\ sold)
\end{aligned}
$$

答案 `$\max(sold, free)$`（结束时不应还持仓更优地考虑已卖）。

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

复杂度：Time `$O(n)$`，Space `$O(1)$`。

坑：用同一天的新 `sold` 去更新 `free` 会串态；上面用元组同时更新，读的都是旧值。

---

## 什么时候用哪种骨架

```text
一个下标、只靠前几项     → 1D 线性
子串 / 开区间、按长度填   → 区间 DP
选物品凑容量             → 背包族（先分 0/1 还是完全，再分可行性 / 最值 / 计数）
两个字符串前缀           → 双序列
网格且方向固定           → 网格填表
网格但依赖“更优邻居”无序 → 记忆化 DFS
每天少量互斥状态         → 状态机
```

## 常见坑

- 答案取 `dp[n]` 还是 `max(dp)` 搞混（LIS、Max Product、Min Cost Climbing）。
- 0/1 背包正序更新，物品被重复使用。
- Coin Change II 外层金额、内层硬币，变成排列数。
- 区间 DP 不按长度填，读到未计算的 `dp[i+1][j-1]`。
- Burst Balloons 想“先戳谁”。
- Regex `*` 忘记对应前一个字符，或空串 base 没处理 `a*`。
- House Robber II 忘了 `n=1`。
- Longest Increasing Path 用普通双重循环填表。

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
C. 目标函数不同（min vs 累加计数），且 II 要用“物品在外”保证组合语义
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
question: Regex Matching 中 p[j-1]=='*' 时，“匹配零次”对应？
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
explanation: 倒序保证 dp[j-x] 仍是“未选当前物品”的旧值；正序会变成完全背包。
```

```quiz
title: 快速选择题 11
question: Stock with Cooldown 的 sold 状态含义最准确的是？
answer: B
A. 任意空仓
B. 今天刚卖出，明天处于冷却
C. 手持股票
D. 累计卖出次数
explanation: sold 专门标记“今日卖出”，次日只能进入 free，不能直接 buy。
```

```quiz
title: 快速选择题 12
question: Decode Ways 应归入本笔记的哪类骨架，细节应去哪看？
answer: A
A. 1D 线性 DP；完整推导见 CoreSkills10
B. 区间 DP；见本文 Burst Balloons
C. 双序列 DP；见 LCS
D. 状态机 DP；见 Stock Cooldown
explanation: 按前缀/后缀方案数递推，是经典 1D；前导 0 等边界在 CoreSkills10。
```
