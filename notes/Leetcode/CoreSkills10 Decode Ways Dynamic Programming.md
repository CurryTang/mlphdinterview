# Dynamic Programming：从递推到空间优化

## 模块一：DP 核心心法与六大骨架分类体系（The Mental Model & 6 Skeletons Taxonomy）

### 1. 面试目标与认知重塑

Dynamic Programming（动态规划，简称 DP）绝不是“死背模板”，其核心本质是：**将一个复杂的大规模决策问题，拆解为具有重叠性的同构子问题，并按照依赖拓扑顺序将解缓存与递推出来**。

一套标准、稳健且在白板面试中不易翻车的 DP 求解流程分为严谨的三步：

```text
Step 1: 数学定义状态 (State) ➔ 明确 dp[...] 的物理含义与 Base Case
Step 2: 列出决策分支 (Transition) ➔ 建立状态转移方程并确定循环遍历拓扑序 (Order)
Step 3: 观察依赖半径 (Space Optimization) ➔ 按需执行滚动变量 / 滚动数组空间压缩
```

> [!IMPORTANT]
> **面试黄金法则**：在白板面试中，先写出状态定义、Base Case 和转移方程并向面试官口述验证，远比一上来直接写代码重要得多。只要递推逻辑严密，写出代码往往只是机械翻译。

---

### 2. 什么时候识别出 DP 信号

当一道算法题呈现以下特征时，应立即激活 DP 思考路径：

- **求解目标**：问“方案总数 / 最小代价 / 最大收益 / 是否可行（True/False）”。
- **阶段性决策**：问题可按步骤推进（如处理到第 `i` 个字符、前 `i` 个物品、第 `i` 天、区间 `[i, j]`）。
- **重叠子问题（Overlapping Subproblems）**：朴素递归搜索过程中，同一个参数配置的子问题被反复展开计算。
- **无后效性与最优子结构（Optimal Substructure）**：当前状态的最优解，仅依赖于此前较小子问题的最优解，而与那些解是如何达成的具体历史路径无关。

动态规划的四要素闭环：

$$\text{DP Formula} = \mathbf{State} + \mathbf{Transition} + \mathbf{Base\ Case} + \mathbf{Iteration\ Order}$$

---

### 3. 三步落地法详解

#### 第一步：写出递推表达式（定义状态与转移）
先不要写任何循环代码。用清晰的一句话定义 `dp[...]` 的状态语义：
- `dp[i]`：前 `i` 个元素 / 从 `i` 开始的后缀的答案。
- `dp[i][j]`：双序列前缀 `(i, j)` / 闭区间子串 `[i, j]` 的答案。
- `dp[i][state]`：阶段 `i` 且处于特定状态（如持有股票/空仓冷冻）的答案。

状态确定后，根据 Choice 组合旧状态：
- **方案数（Count）**：$\text{dp}[i] = \sum \text{dp}[\text{prev}]$
- **最值（Min/Max）**：$\text{dp}[i] = \min / \max(\text{dp}[\text{prev}] + \text{cost})$
- **可行性（Boolean）**：$\text{dp}[i] = \bigvee \text{dp}[\text{prev}]$

#### 第二步：根据依赖确定循环模式（Iteration Order）
循环方向必须严格遵循**拓扑计算依赖**：
- 若 $dp[i]$ 依赖 $dp[i-1]$：从左往右正序遍历（$i: 1 \to n$）。
- 若 $dp[i]$ 依赖 $dp[i+1]$：从右往左倒序遍历（$i: n-1 \to 0$）。
- 若 $dp[i][j]$ 依赖 $dp[i+1][j-1]$（如区间 DP）：按区间长度递增（$L: 1 \to n$）或行从下往上遍历。

#### 第三步：空间优化（Space Optimization）
- 若只依赖前一个状态 $dp[i-1]$：压缩为单变量（$O(1)$ 空间）。
- 若依赖前两个状态 $dp[i-1], dp[i-2]$：压缩为两个滚动变量（如 Fibonacci / House Robber / Decode Ways）。
- 若二维表 $dp[i][j]$ 仅依赖上一行 $dp[i-1][\dots]$：压缩为单行滚动数组（$O(nm) \to O(m)$）。

---

### 4. 六大 DP 骨架全景分类图谱

```text
这题的状态维度与结构是什么？
│
├── 1. 一个下标 i（前 i 项 / 后缀 / 线性推进）
│     └── 骨架一：1D 线性 DP
│         代表题：Climbing Stairs, Min Cost Climbing Stairs, House Robber I & II,
│                 Decode Ways, Word Break, LIS, Maximum Product Subarray
│
├── 2. 两个序列的前缀双指针 (i, j)
│     └── 骨架二：双序列与字符串匹配 DP
│         代表题：Longest Common Subsequence (LCS), Edit Distance,
│                 Interleaving String, Distinct Subsequences, Regular Expression Matching
│
├── 3. 一个闭区间 [i, j]（子串 / 剥洋葱 / 长度扩展）
│     └── 骨架三：区间 DP
│         代表题：Longest Palindromic Substring, Palindromic Substrings, Burst Balloons
│
├── 4. 物品 + 容量（选/不选、单次/无限次、组合/排列）
│     └── 骨架四：背包全家桶（Knapsack Family）
│         代表题：Partition Equal Subset Sum (0/1 可行性), Target Sum (0/1 计数),
│                 Coin Change (完全背包最值), Coin Change II (完全背包组合数)
│
├── 5. 二维网格坐标 (r, c)
│     └── 骨架五：网格与图 DAG DP
│         代表题：Unique Paths（固定扫掠方向）,
│                 Longest Increasing Path in a Matrix（任意方向 ➔ 记忆化 DFS）
│
└── 6. 阶段 + 离散命名状态（持有 / 卖出冷冻 / 自由空仓）
      └── 骨架六：状态机 DP
          代表题：Best Time to Buy and Sell Stock with Cooldown
```

---

## 模块二：骨架一 · 一维线性 DP 与前缀状态（1D Linear DP）

一维线性 DP 是动态规划最基础也是最重要的形态，状态通常定义为以下标 `i` 结尾或从 `i` 开始的前缀/后缀解。

### 1. 经典精讲：Decode Ways（解码方法）

**题目描述**：一条包含字母 `A-Z` 的消息通过 `'A' -> 1, 'B' -> 2, ..., 'Z' -> 26` 进行编码。给定只含数字的字符串 `s`，计算其合法的解码方案总数。

#### 递推与边界设计
定义 $dp[i]$ 为后缀子串 $s[i:]$ 的解码方案数。
- **Base Case**：$dp[n] = 1$（空后缀视为 1 种成功解码方案）。
- **遇 `'0'` 陷阱**：若 $s[i] == '0'$，由于 `'0'` 无法单独解码且前导零非法，直接 $dp[i] = 0$。
- **一位决策**：单字符 $s[i] \in ['1'..'9']$，贡献 $dp[i+1]$。
- **两位决策**：双字符 $s[i..i+1] \in [10..26]$，额外贡献 $dp[i+2]$。

$$dp[i] = \begin{cases} 0 & s[i] = '0' \\ dp[i+1] + [10 \le s[i..i+1] \le 26] \cdot dp[i+2] & s[i] \neq '0' \end{cases}$$

#### 完整数组实现到空间优化

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':
            return 0
            
        n = len(s)
        # 空间优化：dp[i] 只依赖 dp[i+1] (one) 和 dp[i+2] (two)
        one = 1  # 对应 dp[i+1]，初始为 dp[n] = 1
        two = 0  # 对应 dp[i+2]
        
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

- **复杂度**：时间 $O(n)$，空间 $O(1)$。
- **易错点**：前导 `'0'` 不能解码；跨步检查必须确保 `i + 1 < n`。

---

### 2. 爬楼梯族：Climbing Stairs & Min Cost Climbing Stairs

#### Climbing Stairs（爬楼梯）
- **状态定义**：$dp[i]$ 表示爬到第 $i$ 阶的方案数。
- **状态转移**：$dp[i] = dp[i-1] + dp[i-2]$（斐波那契同构）。
- **代码实现**：

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

#### Min Cost Climbing Stairs（使用最小花费爬楼梯）
- **状态定义**：$dp[i]$ 表示到达台阶 $i$ 并在离开该台阶时支付 `cost[i]` 后的累计最小花费。
- **终点语义**：顶楼在楼梯顶端（无 cost），故答案为 $\min(dp[n-1], dp[n-2])$。
- **转移方程**：$dp[i] = cost[i] + \min(dp[i-1], dp[i-2])$。

```python
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        a, b = cost[0], cost[1]
        for i in range(2, len(cost)):
            a, b = b, cost[i] + min(a, b)
        return min(a, b)
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$。

---

### 3. 选与不选族：House Robber & House Robber II

#### House Robber（打家劫舍 I · 线性房屋）
- **约束**：两间相邻的房屋不能在同一晚上被盗。
- **状态定义**：$dp[i]$ 表示考虑前 $i$ 间房屋所能盗得的最大金额。
- **转移方程**：$dp[i] = \max(dp[i-1], dp[i-2] + nums[i-1])$。

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        prev2, prev1 = 0, 0
        for x in nums:
            prev2, prev1 = prev1, max(prev1, prev2 + x)
        return prev1
```

#### House Robber II（打家劫舍 II · 环形房屋）
- **环形本质**：第一间房与最后一间房相邻，不可同时被抢。
- **双段线性拆分法**：
  1. 场景 A：不抢最后一间 $\implies$ 求解区间 `nums[0..n-2]` 的线性 Robber；
  2. 场景 B：不抢第一间 $\implies$ 求解区间 `nums[1..n-1]` 的线性 Robber；
  3. 全局最优：$\max(\text{rob}(nums[0..n-2]), \text{rob}(nums[1..n-1]))$。

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

- **复杂度**：时间 $O(n)$，空间 $O(1)$。
- **易错点**：千万别漏掉 $n=1$ 的特判。

---

### 4. 字符串前缀切分：Word Break（单词拆分）

#### 方案一：标准 Set 哈希切片 DP（基础解法）

- **状态定义**：$dp[i]$ 表示前缀子串 $s[:i]$ 是否能够被字典中的单词空格拆分。
- **状态转移**：枚举最后一个单词的分割点 $j \in [0, i)$：

$$dp[i] = \bigvee_{j=0}^{i-1} \bigl(dp[j] \land (s[j:i] \in \text{wordDict})\bigr)$$

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        words = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Base case: 空串天然可被拆分
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in words:
                    dp[i] = True
                    break
        return dp[n]
```

- **复杂度**：时间 $O(n^2 \cdot L)$（$L$ 为子串切片长度），空间 $O(n + \sum \text{len}(\text{words}))$。
- **性能瓶颈**：
  1. **字符串切片开销**：每次循环计算 `s[j:i]` 都需要在堆上拷贝字符分配新字符串对象（$O(i - j)$ 开销）；
  2. **无法做前缀级即时剪枝**：即使 `s[j:j+2]` 在字典中没有任何单词以此为前缀，传统 DP 仍会盲目向后枚举尝试 `s[j:i]`。

---

#### 方案二：前缀树 (Trie) + DP 极致优化（前向匹配与即时剪枝）

在工业级文本分词或大规模字典场景下，**将字典构建为前缀树（Trie）并结合 DP 前向匹配**是理论与实测性能最优解：

- **核心优化逻辑**：
  1. **字典树构建**：将 `wordDict` 插入一棵 Trie 树中，同时记录单词最大长度 `max_len`；
  2. **前向驱动推进**：当且仅当 $dp[i] == \text{True}$ 时，以位置 $i$ 为起点，在 Trie 树上**顺向推进**扫描字符 $s[j]$；
  3. **分支即时剪枝（Pruning）**：一旦当前字符在 Trie 中没有对应的子节点，**立刻 `break` 截断内层循环**（字典中没有任何单词包含此前缀，后续更长子串绝不可能匹配！）；
  4. **零切片开销**：完全基于单个字符指针下移，不产生任何子字符串对象切片分配。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        # 1. 构建前缀树 (Trie)
        root = TrieNode()
        max_len = 0
        for word in wordDict:
            node = root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.is_word = True
            max_len = max(max_len, len(word))

        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Base case: 空前缀有效

        # 2. 前向 Trie 遍历与即时剪枝 DP
        for i in range(n):
            if not dp[i]:
                continue  # 前驱前缀不可拆分，跳过
            
            node = root
            # 从 i 出发顺向在 Trie 树上匹配字符 s[j]
            for j in range(i, min(n, i + max_len)):
                ch = s[j]
                if ch not in node.children:
                    break  # 核心剪枝：Trie 树失配，立即终结后续更长子串尝试！
                node = node.children[ch]
                if node.is_word:
                    dp[j + 1] = True

        return dp[n]
```

- **复杂度与架构对比**：

| 架构对比维度 | 方案一：标准 Set + 切片 DP | 方案二：Trie 前向匹配 + DP 优化 |
|---|---|---|
| **字符串切片开销** | 每次枚举产生 `s[j:i]` 新对象 ($O(L)$ 内存分配与拷贝) | **零切片开销**（仅指针字符级遍历） |
| **前缀匹配剪枝** | 无法剪枝（盲目枚举全部 $j \in [0, i)$） | **即时剪枝**（分支失配立即 `break`） |
| **最坏时间复杂度** | $O(n^2 \cdot L)$ | $O(n \cdot \min(n, L_{\max}) + \sum \text{len})$ |
| **空间复杂度** | $O(n + \sum \text{len})$ | $O(n + \Sigma \cdot \text{Nodes})$ |
| **工程适用场景** | 字典较小、短文本 | 大规模词典、NLP 文本分词、高吞吐场景 |

---

### 5. 历史扫描：Longest Increasing Subsequence（LIS 最长递增子序列）

- **状态定义**：$dp[i]$ 表示**必须以 $nums[i]$ 结尾**的最长严格递增子序列长度。
- **状态转移**：扫描所有在 $i$ 之前且数值更小的前驱 $j < i$：

$$dp[i] = 1 + \max_{j < i, nums[j] < nums[i]} dp[j]$$
- **全局答案**：$\max_{0 \le i < n} dp[i]$（注意答案不一定在最后一格）。

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

- **复杂度**：时间 $O(n^2)$，空间 $O(n)$。（注：二分贪心/耐人寻味的耐心排序可优化至 $O(n \log n)$，本章聚焦 DP 基础状态设计）。

---

### 6. 符号翻转与极值双追踪：Maximum Product Subarray（乘积最大子数组）

- **本质难点**：负数乘以负数会翻转为正数，因此局部的极小值（最负的数）在遇到下一个负数时会一跃成为全局极大值！
- **双状态追踪**：必须同时维护以 $i$ 结尾的 `max_here` 与 `min_here`：

$$\text{max\_here}' = \max(x, \text{max\_here} \cdot x, \text{min\_here} \cdot x)$$

$$\text{min\_here}' = \min(x, \text{max\_here} \cdot x, \text{min\_here} \cdot x)$$

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

- **复杂度**：时间 $O(n)$，空间 $O(1)$。

---

## 模块三：骨架二 · 双序列与字符串匹配 DP（Two Sequences DP）

双序列 DP 解决两个字符串/序列 `s1` 与 `s2` 的对齐、编辑、交错与子序列匹配问题。状态通常定义为网格点 $(i, j)$，分别代表 `s1[:i]` 与 `s2[:j]` 的前缀。

```text
二维双序列状态依赖拓扑：
         dp[i-1][j-1] (对角线: 同时匹配/替换) ────> dp[i-1][j] (上方: 删除/跳过 s1)
              │                                      │
              ▼                                      ▼
         dp[i][j-1] (左方: 插入/跳过 s2)     ────> dp[i][j] (当前状态)
```

---

### 1. 对齐与公共子序列：Longest Common Subsequence（LCS）

- **状态定义**：$dp[i][j]$ 表示 `text1[:i]` 与 `text2[:j]` 的最长公共子序列长度。
- **状态转移**：

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

- **复杂度**：时间 $O(mn)$，空间 $O(mn)$（滚动数组可压缩至 $O(\min(m, n))$）。
- **易错点**：字符匹配时必须走对角线 $dp[i-1][j-1] + 1$，不可写成 $\max(dp[i-1][j], dp[i][j-1]) + 1$。

---

### 2. 三向决策：Edit Distance（编辑距离）

- **状态定义**：$dp[i][j]$ 表示将 `word1[:i]` 转换为 `word2[:j]` 所需的最少操作步数（插入、删除、替换）。
- **状态转移**：

$$dp[i][j] = \begin{cases} dp[i-1][j-1] & \text{word1}[i-1] == \text{word2}[j-1] \\ 1 + \min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) & \text{otherwise} \end{cases}$$
  - $dp[i-1][j] + 1$：删除 `word1[i-1]`；
  - $dp[i][j-1] + 1$：插入 `word2[j-1]`；
  - $dp[i-1][j-1] + 1$：将 `word1[i-1]` 替换为 `word2[j-1]`。

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

- **复杂度**：时间 $O(mn)$，空间 $O(mn)$.

---

### 3. 双源推进可行性：Interleaving String（交错字符串）

- **状态定义**：$dp[i][j]$ 表示 `s1[:i]` 与 `s2[:j]` 能否交错组成 `s3[:i+j]`。
- **状态转移**：当前字符 `s3[i+j-1]` 可以来自 `s1[i-1]`（从上方来）或 `s2[j-1]`（从左方来）：

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

- **复杂度**：时间 $O(mn)$，空间 $O(mn)$。

---

### 4. 字符复用与子序列计数：Distinct Subsequences（不同的子序列）

- **状态定义**：$dp[i][j]$ 表示 `s[:i]` 中有多少个子序列与 `t[:j]` 完全相同。
- **状态转移**：

$$dp[i][j] = \begin{cases} dp[i-1][j-1] + dp[i-1][j] & s[i-1] == t[j-1] \\ dp[i-1][j] & s[i-1] \neq t[j-1] \end{cases}$$
  - $dp[i-1][j-1]$：选择用当前字符 `s[i-1]` 匹配 `t[j-1]`；
  - $dp[i-1][j]$：舍弃/跳过当前字符 `s[i-1]`。

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 1  # 空字符串 t 总是可以被匹配 1 次 (什么都不选)
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[m][n]
```

- **复杂度**：时间 $O(mn)$，空间 $O(mn)$。

---

### 5. 通配分支：Regular Expression Matching（正则表达式匹配）

- **状态定义**：$dp[i][j]$ 表示 `s[:i]` 是否能被正则 `p[:j]` 匹配。
- **状态转移**：
  1. $p[j-1] \neq '*'$：若 $p[j-1] == s[i-1] \lor p[j-1] == '.'$，则 $dp[i][j] = dp[i-1][j-1]$。
  2. $p[j-1] == '*'$：
     - **匹配 0 次**：完全抛弃 `x*` 组合 $\implies dp[i][j] = dp[i][j-2]$；
     - **匹配 1 次或多次**：当前字符匹配（$p[j-2] == s[i-1] \lor p[j-2] == '.'$）时，消耗一个字符 $s[i-1]$ 并继续保持 pattern $\implies dp[i-1][j]$。

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        # 处理类似 a*, a*b*, a*b*c* 匹配空字符串的 Base Cases
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
                
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2]  # 匹配 0 次
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]  # 匹配多次
                elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    
        return dp[m][n]
```

- **复杂度**：时间 $O(mn)$，空间 $O(mn)$。

---

## 模块四：骨架三 · 区间 DP 与矩阵几何（Interval DP & 2D Matrix Evolution）

区间 DP 的状态由一个闭区间 $[i, j]$ 决定。其核心特征是**大区间的解依赖于其严格较短的子区间**，因此遍历拓扑必须**按区间长度递增（Length-increasing）**或**从底向上（$i: n-1 \to 0$）**填表。

---

### 1. 区间回文族：Longest Palindromic Substring & Palindromic Substrings

#### 递推表达式与洋葱剥皮原理
定义 $dp[i][j]$ 表示子串 $s[i..j]$ 是否为回文串：

$$dp[i][j] = (s[i] == s[j]) \land (j - i < 2 \lor dp[i+1][j-1])$$

1. **两端不同**（$s[i] \neq s[j]$）：必不是回文 $\implies dp[i][j] = False$；
2. **长度 $\le 2$**（$j - i < 2$）：若两端相同，单字符（$i=j$）或双字符（$j=i+1$）天然回文 $\implies dp[i][j] = True$；
3. **长度 $\ge 3$**（$j - i \ge 2$）：查询其**左下方/西南角邻居** $dp[i+1][j-1]$（去皮后的内部核心）。

#### 2D 状态转移矩阵动态可视化交互演示

```palindrome-dp-demo
```

#### 代码实现与答案统计

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        ans = 0
        
        # 从底向上、从左往右扫描，保证 dp[i+1][j-1] 总是先被计算
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

- **复杂度**：时间 $O(n^2)$，空间 $O(n^2)$。

---

### 2. 逆向思维之“最后戳破”：Burst Balloons（戳气球）

**题目关键技巧**：在原数组两端补虚拟气球 `1`，变成开区间 $(i, j)$。
- **为什么“先戳哪个”行不通**：若先戳 $k$，左右两边气球因 $k$ 消失而相连，子区间边界动态粘连，无法独立！
- **逆向枚举“区间 $(i, j)$ 里最后一个戳破的气球 $k$”**：当 $k$ 最后被戳破时，$(i, k)$ 与 $(k, j)$ 内的所有气球已经被戳光，此时 $k$ 的邻居恰好是确定不变的固定外边界 $a[i]$ 与 $a[j]$！

$$dp[i][j] = \max_{i < k < j} \bigl(dp[i][k] + a[i] \cdot a[k] \cdot a[j] + dp[k][j]\bigr)$$

```python
class Solution:
    def maxCoins(self, nums: list[int]) -> int:
        a = [1] + nums + [1]
        n = len(a)
        dp = [[0] * n for _ in range(n)]
        
        # 按开区间跨度递增填表
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

- **复杂度**：时间 $O(n^3)$，空间 $O(n^2)$。

---

## 模块五：骨架四 · 背包全家桶（Knapsack Family: 0/1, Complete & Counting）

背包问题是一类经典的带容量约束离散选择与组合优化问题。无论是求最大价值、最少硬币数、能否凑齐目标值，还是计算组合数/排列数，其底层均遵循一套严密统一的数学模型。

---

### 1. 背包全家桶通用解题通关攻略（General Strategy Guide）

面对面试中千变万化的背包变体，严格遵循以下 **“4 步定型判定法”** 与 **“代数降维套路”**，可将一切背包题快速拆解为标准模板：

#### 第一步：背包 4 步定型判定法（The 4-Step Identification Framework）

```text
背包问题 4 步定型流程：
┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ 1. 识别背包与物品     │ ➔   │ 2. 判断物品复用性     │ ➔   │ 3. 确定求解目标与算子 │ ➔   │ 4. 决定循环层次与方向 │
│ 容量 W 与开销 weight  │     │ 0/1 背包 vs 完全背包  │     │ 最值 vs 可行性 vs 计数│     │ 组合数 vs 排列数      │
└───────────────────────┘     └───────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

1. **识别资源约束与物品成本（Capacity & Cost）**：
   - 什么是“背包容量” $W$？（目标总和、最大金额、字符上限、体积等）；
   - 什么是“物品”？每个物品消耗的成本是 $weight_i$，带来的收益是 $value_i$。
2. **判断物品复用性（0/1 or Complete or Bounded）**：
   - **每个物品最多选 1 次** $\implies$ **0/1 背包**：一维压缩后，内层容量循环**必须倒序**（$W \to weight_i$），防止同一物品在同一轮被多次累加；
   - **每个物品可无限复选** $\implies$ **完全背包**：一维压缩后，内层容量循环**必须正序**（$weight_i \to W$），主动利用本轮已更新的状态实现多次复选；
   - **每个物品有限定次数 $k_i$** $\implies$ **多重背包**：通过二进制拆分（将 $k_i$ 拆为 $1, 2, 4, \dots, k_i - 2^p + 1$）将其等价转化为多件独立的 0/1 背包物品。
3. **确定求解目标与转移算子（Target Metric & Operator）**：
   - **最值问题（Optimization）**：使用 $\max$（最大价值）或 $\min$（如 Coin Change 最少硬币数）；
   - **可行性问题（Feasibility / Boolean）**：使用 $\lor$（逻辑或，如能否凑出目标和）；
   - **方案计数问题（Counting）**：使用 $+$（加法原理累加各独立分支方案数）。
4. **判定循环嵌套层级（Outer vs Inner Hierarchy）**：
   - **物品在外层，容量在内层** $\implies$ **组合数（Combinations）**：`[1, 2]` 与 `[2, 1]` 视为同一种选法（每种物品只在固定轮次被处理一次，杜绝了排列重数，如 Coin Change II）；
   - **容量在外层，物品在内层** $\implies$ **排列数（Permutations）**：`[1, 2]` 与 `[2, 1]` 视为不同方案（在每个容量节点，任何物品都可作为最后一步加入，如 Combination Sum IV）。

---

#### 第二步：高频代数降维与问题等价转化套路（Algebraic Reduction Patterns）

绝大多数高频面试题不会直接告诉你“这是一个背包”，而是包装成各种现实问题，需要先进行代数降维：

```text
代数转化 4 大经典模型：
1. 等和子集两等分 (Partition Sum):
   判断子集和是否为 total / 2 ➔ 0/1 背包可行性 (target = total / 2)

2. 正负符号代数拆解 (Target Sum):
   P - N = target 且 P + N = total ➔ P = (target + total) / 2 ➔ 0/1 背包方案计数

3. 最小差值两等分子集 (Last Stone Weight II):
   寻找不超过 floor(total / 2) 的最大子集和 ➔ 最小差值为 total - 2 * dp[floor(total / 2)]

4. 多维费用背包 (Ones and Zeroes):
   同时受最多 M 个 '0' 和 N 个 '1' 限制 ➔ 二维状态 dp[j][k]，双重倒序循环
```

---

#### 第三步：初值与哨兵基石配置矩阵（Initialization & Sentinel Matrix）

背包问题最容易失分的往往是边界与初始值配置：

| 求解目标 | 必须恰好装满 | `dp[0]` 初值 | 其余 `dp[1..W]` 初值 | 核心逻辑说明 |
|---|---|---|---|---|
| **最大价值（$\max$）** | 否（$\le W$ 均可） | `0` | `0` | 容量为 0 收益为 0，空背包合法且价值为 0 |
| **最大价值（$\max$）** | 是（恰好 $= W$） | `0` | `-inf` | 仅 $dp[0]$ 合法，其余初始状态非法（不可达） |
| **最小数量（$\min$）** | 是（恰好 $= W$） | `0` | `+inf` (或 `amount + 1`) | $dp[0]=0$（0 元需 0 枚），其余未达成前均为 $\infty$ |
| **方案计数（$+$）** | 是（恰好 $= W$） | `1` | `0` | $dp[0]=1$（选空集是凑成容量 0 的唯一 1 种方案） |
| **可行性判定（$\lor$）** | 是（恰好 $= W$） | `True` | `False` | 仅容量 0 天然可行，其余初始不可行 |

---

#### 第四步：万能解题模板与统一代码骨架（Universal Knapsack Blueprint）

所有背包问题都可以抽象为由 **3 个核心开关（Knobs）** 调控的统一参数化代码骨架：

```text
背包万能 3 旋钮体系：
┌───────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 控制开关              │ 选项分支与工程含义                                                     │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. 循环层次 (Hierarchy)│ 物品在外、容量在内 ➔ 组合数 / 最值 / 可行性                            │
│                       │ 容量在外、物品在内 ➔ 排列数 (Permutations，如 Combination Sum IV)       │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. 容量方向 (Direction)│ 倒序 range(W, w-1, -1) ➔ 0/1 背包 (防同一物品复选)                     │
│                       │ 正序 range(w, W+1)     ➔ 完全背包 (允许同一物品无限复选)               │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. 转移算子 (Operator) │ 最值优化 ➔ dp[j] = min/max(...)                                        │
│                       │ 方案计数 ➔ dp[j] += dp[j - w]                                          │
│                       │ 可行性   ➔ dp[j] = dp[j] or dp[j - w]                                  │
└───────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

##### 1. 万能统一参数化函数（Universal Knapsack Function）

```python
def universal_knapsack(
    items: list[tuple[int, int]],  # [(weight, value), ...]
    capacity: int,
    problem_type: str = "01_max",
    # 可选类型: "01_max" | "01_min" | "01_feas" | "01_count" | "complete_min" | "complete_combo" | "complete_perm"
) -> int | bool:
    """
    万能背包统领框架 (Universal Knapsack Master Framework)
    """
    # 1. 初始化基石
    if problem_type in ("01_max", "complete_max"):
        dp = [0] * (capacity + 1)
    elif problem_type == "01_feas":
        dp = [True] + [False] * capacity
    elif problem_type in ("01_count", "complete_combo", "complete_perm"):
        dp = [1] + [0] * capacity
    elif problem_type in ("01_min", "complete_min"):
        INF = capacity + 1
        dp = [0] + [INF] * capacity

    # 2. 状态转移执行
    if problem_type == "complete_perm":
        # 排列数特殊结构：容量在外层，物品在内层
        for j in range(1, capacity + 1):
            for weight, val in items:
                if j >= weight:
                    dp[j] += dp[j - we\right]
    else:
        # 标准结构：物品在外层，容量在内层
        for weight, val in items:
            # 0/1 背包倒序 vs 完全背包正序
            step_range = (
                range(capacity, weight - 1, -1)
                if problem_type.startswith("01")
                else range(weight, capacity + 1)
            )
            for j in step_range:
                if problem_type in ("01_max", "complete_max"):
                    dp[j] = max(dp[j], dp[j - we\right] + val)
                elif problem_type in ("01_min", "complete_min"):
                    dp[j] = min(dp[j], dp[j - we\right] + 1)
                elif problem_type in ("01_count", "complete_combo"):
                    dp[j] += dp[j - we\right]
                elif problem_type == "01_feas":
                    dp[j] = dp[j] or dp[j - we\right]

    return dp[capacity]
```

##### 2. 面试高频五大场景 10 行秒杀速写卡（Quick Snippets）

```python
# 1. 0/1 背包 · 可行性判定 (Partition Equal Subset Sum)
dp = [True] + [False] * target
for x in nums:
    for j in range(target, x - 1, -1):  # 倒序
        dp[j] = dp[j] or dp[j - x]

# 2. 0/1 背包 · 方案计数 (Target Sum)
dp = [1] + [0] * bag
for x in nums:
    for j in range(bag, x - 1, -1):     # 倒序
        dp[j] += dp[j - x]

# 3. 完全背包 · 最小物品数 (Coin Change)
dp = [0] + [amount + 1] * amount
for c in coins:
    for j in range(c, amount + 1):      # 正序
        dp[j] = min(dp[j], dp[j - c] + 1)

# 4. 完全背包 · 组合数计数 (Coin Change II)
dp = [1] + [0] * amount
for c in coins:                         # 物品在外
    for j in range(c, amount + 1):      # 正序在内
        dp[j] += dp[j - c]

# 5. 完全背包 · 排列数计数 (Combination Sum IV)
dp = [1] + [0] * target
for j in range(1, target + 1):          # 容量在外
    for x in nums:                      # 物品在内
        if j >= x:
            dp[j] += dp[j - x]
```

##### 3. 面试 3 秒速记口诀（Mental Mnemonics）

> 💡 **背包三秒速记口诀**：
> - **“零一倒序防复选，完全正序连环算。”**
> - **“物品在外组合现，容量在外排列见。”**
> - **“可行用或最值限，计数用加初始一点（$dp[0]=1$）。”**

---

### 2. 0/1 背包例题一：Partition Equal Subset Sum（分割等和子集 · 可行性）

- **问题转化**：判断能否从数组中选出若干数（每个数最多选 1 次），使其和恰好为 $target = \text{sum} / 2$。
- **状态与转移**：$dp[j]$ 表示能否凑出容量 $j$：

$$dp[j] = dp[j] \lor dp[j - x]$$

#### 0/1 背包一维状态演化、倒序防复选与可达性回溯可视化

```subset-sum-demo
```

#### 代码实现

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
            for j in range(target, x - 1, -1):  # 0/1 背包必须倒序！
                dp[j] = dp[j] or dp[j - x]
        return dp[target]
```

- **复杂度**：时间 $O(n \cdot target)$，空间 $O(target)$。

---

### 3. 0/1 背包例题二：Target Sum（目标和 · 代数转化与计数）

- **代数推导**：设正数子集为 $P$，负数绝对值子集为 $N$：

$$P - N = target,\quad P + N = total \implies 2P = target + total \implies P = \frac{target + total}{2}$$
- **问题等价于**：从 `nums` 中选择子集使其和恰好为 $bag = (target + total) // 2$ 的方案数！
- **二维到一维压缩**：$dp[s] = dp[s] + dp[s - num]$（倒序遍历）。

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
            for s in range(bag, num - 1, -1):  # 倒序
                dp[s] += dp[s - num]
        return dp[bag]
```

- **复杂度**：时间 $O(n \cdot bag)$，空间 $O(bag)$。

---

### 4. 完全背包例题一：Coin Change（零钱兑换 · 最少枚数）

- **语义**：每种面值的硬币有无限枚可用，求凑齐总金额 `amount` 所需的**最少硬币总数**。
- **状态定义**：$dp[a]$ 表示凑出金额 $a$ 所需的最少硬币枚数。
- **状态转移**：枚举当前硬币 $coin$：

$$dp[a] = \min(dp[a], dp[a - coin] + 1) \quad (a \ge coin)$$

- **为什么完全背包必须正序遍历？**
  - 在一维滚动数组中，正序遍历使得当我们计算 $dp[a]$ 时，$dp[a - coin]$ 已经在**本轮循环中更新过了**，从而天然包含了“当前硬币已被复选多次”的状态（例如可以用三个 1 元凑出 3 元）；
  - 若为 0/1 背包（每种硬币至多选 1 枚），则必须倒序遍历以强制读取上一轮未选该硬币的旧状态。

#### 完全背包一维 DP 状态演化与找零路径回溯可视化

```coin-change-demo
```

#### 代码实现

```python
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        inf = amount + 1  # 哨兵上限 (最多需要 amount 枚 1 元，amount+1 即代表不可达)
        dp = [inf] * (amount + 1)
        dp[0] = 0  # Base case: 凑出金额 0 需 0 枚硬币
        
        for coin in coins:
            for s in range(coin, amount + 1):  # 正序遍历 ➔ 允许无限复选
                dp[s] = min(dp[s], dp[s - coin] + 1)
                
        return -1 if dp[amount] == inf else dp[amount]
```

- **复杂度**：时间 $O(n \cdot amount)$，空间 $O(amount)$。
- **面试避坑点**：为什么不能用贪心？
  - 例如 `coins = [1, 3, 4], amount = 6`：
    - **贪心策略**优先拿最大面值：$4 + 1 + 1 \implies 3$ 枚；
    - **DP 最优解**全局统筹：$3 + 3 \implies 2$ 枚！
  - 只有在硬币面值呈特定倍数体系（如美元/人民币标准面额）时贪心才成立；通用面额必须使用 DP。

---

### 5. 完全背包例题二：Coin Change II（零钱兑换 II · 组合数 vs 排列数）

- **核心陷阱：组合数 vs 排列数**：
  - **组合数（Combinations，`{1, 2}` 与 `{2, 1}` 视为相同）**：**物品在外层，容量在内层**。每种硬币只在固定轮次被处理，杜绝了顺序产生的重数。
  - **排列数（Permutations，`{1, 2}` 与 `{2, 1}` 视为不同）**：**容量在外层，物品在内层**。在每个金额下，任何硬币都可作为最后一枚加入。

```python
class Solution:
    def change(self, amount: int, coins: list[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        # 组合数：硬币在外层
        for c in coins:
            for a in range(c, amount + 1):
                dp[a] += dp[a - c]
        return dp[amount]
```

- **复杂度**：时间 $O(n \cdot amount)$，空间 $O(amount)$。

---

### 6. 多维费用 0/1 背包：Ones and Zeroes（一和零 · 二维容量约束）

- **问题特征**：每个字符串消耗 $zeros$ 个 `'0'` 和 $ones$ 个 `'1'`，目标是在容量最多 $m$ 个 `'0'` 和 $n$ 个 `'1'` 限制下，选出最多数量的字符串。
- **降维策略**：本质是**双重容量约束的 0/1 背包**。一维压缩为二维矩阵 $dp[j][k]$，双重内层循环均**倒序遍历**！

```python
class Solution:
    def findMaxForm(self, strs: list[str], m: int, n: int) -> int:
        # dp[j][k] 表示最多有 j 个 0 和 k 个 1 时能选出的最大字符串子集大小
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for s in strs:
            zeros = s.count('0')
            ones = len(s) - zeros
            
            # 0/1 背包：双重容量必须均倒序遍历！
            for j in range(m, zeros - 1, -1):
                for k in range(n, ones - 1, -1):
                    dp[j][k] = max(dp[j][k], dp[j - zeros][k - ones] + 1)
                    
        return dp[m][n]
```

- **复杂度**：时间 $O(L \cdot m \cdot n)$（$L$ 为字符串数组长度），空间 $O(m \cdot n)$。

---

### 7. 背包全家桶终极通关决策速查总表

| 背包场景类别 | 代表问题 | 循环层次 | 遍历方向 | 转移核心方程 |
|---|---|---|---|---|
| **0/1 背包 · 可行性** | Partition Equal Subset Sum | 物品在外，容量在内 | 容量**倒序** ($W \to w$) | $dp[j] = dp[j] \lor dp[j - w]$ |
| **0/1 背包 · 方案计数** | Target Sum | 物品在外，容量在内 | 容量**倒序** ($W \to w$) | $dp[j] += dp[j - w]$ |
| **0/1 背包 · 多维约束** | Ones and Zeroes | 物品在外，多维容量在内 | 各维容量**倒序** | $dp[j][k] = \max(dp[j][k], dp[j-z][k-o] + 1)$ |
| **完全背包 · 最值优化** | Coin Change (最少枚数) | 物品在外，容量在内 | 容量**正序** ($w \to W$) | $dp[j] = \min(dp[j], dp[j - c] + 1)$ |
| **完全背包 · 组合数计数** | Coin Change II | **物品在外，容量在内** | 容量**正序** ($w \to W$) | $dp[j] += dp[j - c]$ |
| **完全背包 · 排列数计数** | Combination Sum IV | **容量在外，物品在内** | 容量**正序** ($1 \to W$) | $dp[j] += dp[j - num]$ |

---

## 模块六：骨架五 · 网格与图 DAG 记忆化（Grid DP & DAG Memoization）

### 1. 有序网格扫掠：Unique Paths（不同路径）

- **规则**：只能向下或向右走 $\implies$ 拓扑依赖天然有序（只依赖上方与左方）。
- **转移方程**：$dp[j] = dp[j] + dp[j-1]$（空间优化为单行滚动）。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
```

- **复杂度**：时间 $O(mn)$，空间 $O(n)$。

---

### 2. 无序网格与隐式 DAG：Longest Increasing Path in a Matrix（矩阵中的最长递增路径）

- **为什么不能用双重 for 循环**：矩阵数值任意分布，没有统一的几何扫掠方向（可能向上下左右任意严格递增方向走）。
- **本质**：严格递增关系天然构成一个**有向无环图（DAG）**。
- **解法**：记忆化搜索（Memoized DFS）即是 DAG 上的 DP！

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

- **复杂度**：时间 $O(mn)$（每个格子仅被计算 1 次），空间 $O(mn)$。

---

## 模块七：骨架六 · 状态机 DP（State Machine DP）

当问题在每个阶段存在有限个互相转换的具名离散状态时，使用状态机 DP 能够清晰解耦转移逻辑。

### 经典实战：Best Time to Buy and Sell Stock with Cooldown（买卖股票含冷冻期）

```text
三状态有限状态机（FSM）：
           ┌─────────────────────── 买入 (buy) ──────────────────────┐
           │                                                         │
           ▼                                                         │
      ┌─────────┐                卖出 (sell)                 ┌────────────┐
      │  Hold   │ ─────────────────────────────────────────> │    Sold    │
      │ (持股)  │                                            │ (刚卖/冷冻)│
      └─────────┘                                            └────────────┘
           │                                                         │
       保持不动                                                  次日解冻
           │                                                         │
           ▼                                                         ▼
      ┌─────────┐                                            ┌────────────┐
      │  Hold   │ <──────────────── 不动 (rest) ──────────── │    Rest    │
      └─────────┘                                            │ (自由空仓) │
                                                             └────────────┘
```

#### 状态定义与转移方程
- `hold`: 当天结束后持有一股的最大利润 $\implies \text{hold} = \max(\text{hold}, \text{rest} - price)$
- `sold`: 当天刚卖出股票的最大利润（次日必须冷冻） $\implies \text{sold} = \text{hold} + price$
- `rest`: 当天自由空仓（可随时买入）的最大利润 $\implies \text{rest} = \max(\text{rest}, \text{sold})$

#### 空间优化实现（避免变量覆盖陷阱）

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
            # 同步更新，切忌用新 hold 覆盖旧 hold 去算 sold
            hold, sold, rest = next_hold, next_sold, next_rest
            
        return max(sold, rest)
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$。

---

## 模块八：进阶心法 · 从 DP 到 Greedy 的降维与支配分析（DP to Greedy Compression）

并不是所有贪心算法都是凭空猜测的，很多高频贪心题的底层逻辑是**由完整 DP 表格通过“单调性与支配关系”压缩而来的极速解法**！

---

### 1. Kadane's Algorithm（最大子数组和 · 负资产切除）

- **DP 视角**：定义 $dp[i]$ 为**必须以 $nums[i]$ 结尾**的最大连续子数组和：

$$dp[i] = \max(nums[i], dp[i-1] + nums[i])$$
- **贪心压缩本质**：若 $curSum < 0$，带入未来任何元素只会产生负向拖累，因此立即将 $curSum$ 归零重置！

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

- **复杂度**：时间 $O(n)$，空间 $O(1)$。
- **易错点**：全负数数组（如 `[-3, -1, -2]`）时，`max_sum` 必须初始化为 `nums[0]` 而非 `0`。

---

### 2. Jump Game（跳跃游戏 · 最左好位置压缩）

- **完整 DP 视角**：$dp[i]$ 表示从 $i$ 出发能否到达终点，$dp[i] = \bigvee_{j=i+1}^{i+nums[i]} dp[j]$，时间复杂度 $O(n^2)$。
- **贪心降维契机**：我们真的需要维护所有能到终点的 `good` 集合吗？
  - 观察发现：越靠左的好位置，越容易被更左边的起点够到（**支配关系**）！
  - 于是，整个布尔数组被压缩为一个单一标量 `goal`（已知的最左好位置）。
  - 若 $i + nums[i] \ge goal$，则 $i$ 自身成为新的最左好位置：$goal = i$。

```python
class Solution:
    def canJump(self, nums: list[int]) -> bool:
        goal = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= goal:
                goal = i
        return goal == 0
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$。

#### 什么样的 DP 可以被贪心降维？
1. **单调边界**：集合信息可被一个极值代表（如最左 `goal` 或最远 `reach`）；
2. **状态支配**：优选状态在任何未来分支下均严格优于被支配状态；
3. **选择安全**：局部最优决策绝不破坏全局最优解的存在性。

---

## 模块九：面试避坑指南、复杂度总表与白板表达模板

### 1. 全问题时空复杂度与空间优化速查表

| 骨架类别 | 代表问题 | 标准时间复杂度 | 空间复杂度（标准 $\to$ 优化后） | 核心转移与依赖要点 |
|---|---|---|---|---|
| **1D 线性** | Decode Ways | $O(n)$ | $O(n) \to O(1)$ | 倒序遍历，处理 `'0'`，双变量滚动 |
| **1D 线性** | Climbing Stairs | $O(n)$ | $O(n) \to O(1)$ | 斐波那契结构，双变量滚动 |
| **1D 线性** | Min Cost Climbing Stairs | $O(n)$ | $O(n) \to O(1)$ | 顶楼无 cost，取 $\min(dp[n-1], dp[n-2])$ |
| **1D 线性** | House Robber I / II | $O(n)$ | $O(n) \to O(1)$ | 选或不选；II 拆分为两段线性子区间 |
| **1D 线性** | Word Break | $O(n^2 \cdot L) \xrightarrow{\text{Trie}} O(n \cdot L_{\max})$ | $O(n)$ | 字典转 Set / Trie 前缀树前向即时剪枝 |
| **1D 线性** | LIS (最长递增子序列) | $O(n^2)$ | $O(n)$ | 全局扫描前驱；答案为 $\max(dp)$ |
| **1D 线性** | Maximum Product Subarray | $O(n)$ | $O(n) \to O(1)$ | 同步维护 `max_here` 与 `min_here` |
| **双序列** | LCS (最长公共子序列) | $O(mn)$ | $O(mn) \to O(\min(m, n))$ | 相等走对角 $+1$，不等走 $\max(\text{上}, \text{左})$ |
| **双序列** | Edit Distance (编辑距离) | $O(mn)$ | $O(mn)$ | 相等抄对角，不等取 $1 + \min(\text{增, 删, 改})$ |
| **双序列** | Interleaving String | $O(mn)$ | $O(mn)$ | 上方或左方匹配 `s3[i+j-1]` |
| **双序列** | Distinct Subsequences | $O(mn)$ | $O(mn)$ | 匹配时选与不选相加；$dp[i][0]=1$ |
| **双序列** | Regex Matching | $O(mn)$ | $O(mn)$ | `*` 对应 $dp[i][j-2]$ (0次) 与 $dp[i-1][j]$ (多次) |
| **区间 DP** | Longest Palindrome / Count | $O(n^2)$ | $O(n^2)$ | 西南角依赖 $dp[i+1][j-1]$，底向上填表 |
| **区间 DP** | Burst Balloons (戳气球) | $O(n^3)$ | $O(n^2)$ | 逆向枚举最后戳破气球 $k$ |
| **0/1 背包** | Partition Equal Subset Sum | $O(n \cdot \frac{\text{sum}}{2})$ | $O(\frac{\text{sum}}{2})$ | 可行性背包，容量循环必须**倒序** |
| **0/1 背包** | Target Sum | $O(n \cdot bag)$ | $O(bag)$ | 代数转换为子集和计数，容量循环**倒序** |
| **完全背包** | Coin Change (最少枚数) | $O(n \cdot amount)$ | $O(amount)$ | 完全背包，容量循环**正序**，取 $\min$ |
| **完全背包** | Coin Change II (组合数) | $O(n \cdot amount)$ | $O(amount)$ | 组合数计数：**硬币在外，容量正序在内** |
| **网格 DP** | Unique Paths | $O(mn)$ | $O(n)$ | 上左相加，单行滚动 |
| **网格 DP** | Longest Increasing Path | $O(mn)$ | $O(mn)$ | 无序网格转隐式 DAG 记忆化搜索 |
| **状态机** | Stock with Cooldown | $O(n)$ | $O(n) \to O(1)$ | Hold / Sold / Rest 三状态同步轮转 |
| **DP ➔ 贪心** | Kadane Max Subarray | $O(n)$ | $O(1)$ | 负数前缀归零，单变量滚动 |
| **DP ➔ 贪心** | Jump Game | $O(n)$ | $O(1)$ | 维护最左好位置 `goal` 或最远 `reach` |

---

### 2. 高频易错点避坑清单

1. **未想清楚 Base Case 就写循环**：例如 $dp[0]=1$、空串匹配、顶楼无花费等。
2. **遍历顺序与依赖拓扑相悖**：区间 DP 没有按长度或底向上扫，读到未初始化的 $dp[i+1][j-1]$。
3. **背包循环方向搞反**：
   - 0/1 背包正序会导致同一个物品在本轮被重复选用，退化为完全背包；
   - 组合数与排列数弄混：硬币在外是组合数（Coin Change II），容量在外是排列数（Combination Sum IV）。
4. **状态机空间优化时发生变量脏读覆盖**：在同一循环内先修改了 `hold`，紧接着计算 `sold = hold + price` 时误读了新 `hold`。必须使用临时变量统一赋值！
5. **极值 DP 混淆答案位置**：误以为答案永远在最后一格 $dp[n-1]$（LIS、乘积最大子数组的答案为全局 $\max$）。

---

### 3. 面试结构化表达模板

在面对任何 DP 面试题时，请严格按照以下 6 步推进沟通：

1. **状态定义（State）**：“我定义 $dp[i][j]$ 为前缀/区间/阶段下的某种最优值/方案数。”
2. **决策分支（Choices & Transition）**：“在当前位置，我有 $k$ 种选择，分别对应之前的某些子状态……”
3. **边界条件（Base Cases）**：“最小的边界是 $dp[0]$ / 对角线，其物理含义为……”
4. **计算顺序（Order）**：“由于当前状态依赖于旧状态的位置，我们需要以正序/倒序/按长度递增填表。”
5. **编码实现与干跑用例（Dry Run）**：“我们先写出清晰的二维/一维完整表代码，并用一个小样例干跑验证。”
6. **时空复杂度与空间压缩（Optimization）**：“当前时空复杂度为 $O(\dots)$；观察到依赖半径仅为上一行/前两个变量，我们可进一步将空间压缩至 $O(1)$ / $O(m)$。”

