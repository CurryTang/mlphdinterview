# Core Skills 12 · 贪心算法：从 Kadane 原理、强制选择到最远包络

## 1. 贪心算法的核心心智模型

在刷题和面试中，很多候选人对贪心算法（Greedy）的印象停留在“凭直觉每一步选当前看起来最好的”。然而在真正的算法面试中，**单凭直觉写贪心极易掉入局部最优无法收敛至全局最优的陷阱**。

可靠的贪心策略背后必然对应严格的数学性质：
1. **贪心选择性质（Greedy Choice Property）**：全局最优解可以通过一系列局部最优选择来达到。
2. **最优子结构（Optimal Substructure）**：做出局部贪心选择后，原问题缩减为一个规模更小但结构完全相同的独立子问题。

在面试中证明贪心正确性，最标准的两大论证武器是：
- **强制选择（Forced Move / Unique Placement）**：某个最极端（最小、最早结束、最紧迫）的元素在任何合法解中都**没有其他放置可能**，因此优先处理它不会丢失任何合法解。
- **替换论证（Exchange Argument / Dominance）**：假设存在一个不包含当前贪心选择的最优解 $OPT$，我们可以将其中的某个决策替换为当前贪心决策，而得到的解 $OPT'$ 质量绝不会变差（甚至更优）。

```text
看到贪心问题该检查的 4 步思考框架：
1. 观察极限与边界：是否存在某个“必须最先解决”的元素（如最小的数、最早截止的时间、最右的边界）？
2. 检验无后效性：当前局部选择是否会对未来的其他选择施加不可逆的负面束缚？若有束缚，往往需要回溯或动态规划。
3. 维护单调不变量（Invariant）：例如“当前能到达的最远右边界 max_reach”、“当前子数组的非负前缀和”、“未匹配左括号的最小/最大可能数量 [cmin, cmax]”。
4. 排除负向累赘：一旦某个局部前缀的净收益变为负数，继续携带它只会拖累未来，必须果断重置起点。
```

---

## 2. 深入解构 Kadane 算法：原理、动态规划与贪心双重视角

### 2.1 什么是 Kadane 算法？它解决了什么问题？

**Kadane 算法（Kadane's Algorithm）** 由卡内基梅隆大学（CMU）统计学家 **Joseph Born Kadane** 于 1984 年提出（并在计算机大师 Jon Bentley 的名著《编程珠玑》*Programming Pearls* 中被广为传阅）。

它所解决的是计算机科学中最经典的基石问题之一——**最大子数组和问题（Maximum Subarray Problem）**：
> 给定一个整数数组 `nums`（包含正数、负数与零），找到一个具有最大和的**连续非空子数组** $\max_{0 \le i \le j < n} \sum_{k=i}^j nums[k]$，并返回其最大和。

在 Kadane 算法出现之前，该问题的求解复杂度经历了一系列演进：
1. **暴力枚举（Brute Force）**：枚举所有起点 $i$ 与终点 $j$，再用循环求和，时间复杂度 $O(n^3)$；
2. **前缀和优化（Prefix Sums）**：预计算前缀和数组 $P$，则任意区间和可在 $O(1)$ 计算，但枚举 $(i, j)$ 仍需 $O(n^2)$；
3. **分治法（Divide and Conquer）**：将数组一分为二，递归求解左半、右半和跨越中点的最大子数组，时间复杂度 $O(n \log n)$；
4. **Kadane 算法**：仅需**单次线性扫描**，将时间复杂度彻底压至理论下界 **$O(n)$**，空间复杂度仅需 **$O(1)$**！

---

### 2.2 核心原理：动态规划视角 vs 贪心前缀重置视角

Kadane 算法之所以精妙，在于它既可以被严谨地解释为**一维动态规划的状态压缩**，也可以被直观地理解为**贪心前缀止损重置机制**。

#### 视角 1：动态规划（DP）的状态设计与无后效性
如果简单地将状态定义为“前 $i$ 个元素中的最大子数组和”，我们会立刻遇到一个致命困境：这个最大子数组可能出现在数组前面的任意位置，**它不一定以 $nums[i]$ 结尾**！当处理第 $i+1$ 个元素时，我们无法将其与之前的子数组连续拼接，状态失去了递推的“连续性连接点”（违背无后效性）。

Kadane 的核心突破在于定义了一个**具有局部连续约束的状态**：
- **定义 $dp[i]$ 为：以第 $i$ 个元素 $nums[i]$ 结尾的最大连续子数组和**（强制包含 $nums[i]$）。

此时，对于第 $i$ 个元素 $nums[i]$，只有两种抉择：
1. **接在前面**：将 $nums[i]$ 拼接到以 $nums[i-1]$ 结尾的最优子数组后面，收益为 $dp[i-1] + nums[i]$；
2. **另起炉灶**：抛弃前面的所有元素，单由 $nums[i]$ 作为新子数组的起点，收益为 $nums[i]$。

因此状态转移方程为：
$$dp[i] = \max(nums[i], dp[i-1] + nums[i]) = \max(0, dp[i-1]) + nums[i]$$

全局最大和即为所有局部结尾状态的最大值：
$$\text{GlobalMax} = \max_{0 \le i < n} dp[i]$$

由于 $dp[i]$ 的计算只依赖于紧邻的前一个状态 $dp[i-1]$，我们可以直接将整个 DP 数组压缩为一个标量变量 `cur_sum`，从而实现 **$O(1)$ 常数额外空间**。

---

#### 视角 2：贪心前缀重置（Momentum vs Liability / 资产与负债）
从贪心角度思考，遍历到元素 $x$ 时，历史累积的前缀和 `cur_sum` 对当前决策的作用只有两种：
- **正向资产（Positive Momentum）**：若 `cur_sum > 0`，说明历史前缀具有正收益。将正数带入当前元素 $x$，无论 $x$ 本身是正是负，整体和必定大于单拿 $x$（$x + \text{cur\_sum} > x$）。
- **纯负债拖累（Pure Liability / Drag）**：若 `cur_sum \le 0`，说明历史前缀已经沦为“负资产”。如果把这部分负资产加到 $x$ 上，只会让以 $x$ 开头的任何子数组变小！因此**必须果断止损归零（`cur_sum = 0`），抛弃全部历史拖累，让新子数组直接从 $x$ 重新起步**。

```kadane-demo
```

---

### 2.3 数学严密证明（反证法 / Proof by Contradiction）

为什么“一旦前缀和变负就直接归零重置”绝不会漏掉全局最优解？

**定理**：设全局最大连续子数组为 $nums[L \dots R]$（其和为 $S^*$）。那么对于该最优子数组内部的任意前缀 $nums[L \dots k]$（其中 $L \le k < R$），其前缀和必须满足：
$$\sum_{j=L}^k nums[j] \ge 0$$

**反证证明**：
1. 假设存在某个位置 $k$（$L \le k < R$），使得该前缀和为负：
   $$S_{\text{prefix}} = \sum_{j=L}^k nums[j] < 0$$
2. 现在我们从原最优子数组 $nums[L \dots R]$ 中切除这段负前缀，观察剩余的后缀子数组 $nums[k+1 \dots R]$ 之和 $S_{\text{suffix}}$：
   $$S_{\text{suffix}} = \sum_{j=k+1}^R nums[j] = S^* - S_{\text{prefix}}$$
3. 因为 $S_{\text{prefix}} < 0$，所以 $-S_{\text{prefix}} > 0$，可得：
   $$S_{\text{suffix}} = S^* + (-S_{\text{prefix}}) > S^*$$
4. 也就是说，切除负前缀后的剩余子数组之和 $S_{\text{suffix}}$ 竟然严格大于原全局最大和 $S^*$！
5. 这与 $S^*$ 是全局最大子数组和的前提产生直接矛盾！

**结论**：**全局最优子数组内部的任何一个真前缀，其和都绝对不可能为负数**。因此，Kadane 算法一旦检测到累积前缀和 $< 0$ 就立即归零重置，**剔除的全是数学上证明不可能产生全局最优解的无效路径**，故算法具备 100% 的完备性与正确性。

---

### 2.4 Kadane 算法的核心代码骨架与思维迁移

```python
def kadane(nums: list[int]) -> int:
    max_sum = nums[0]
    cur_sum = 0
    
    for x in nums:
        # 核心转移：若 cur_sum 为负则归零，再加上当前值 x
        cur_sum = max(x, cur_sum + x)
        max_sum = max(max_sum, cur_sum)
        
    return max_sum
```

```cpp
#include <vector>
#include <algorithm>

int kadane(const std::vector<int>& nums) {
    int max_sum = nums[0];
    int cur_sum = 0;
    for (int x : nums) {
        cur_sum = std::max(x, cur_sum + x);
        max_sum = std::max(max_sum, cur_sum);
    }
    return max_sum;
}
```

#### 思维迁移：Kadane 的前缀重置思想在其他高频题中的应用
1. **Gas Station（加油站 · LC 134）**：将各站的净收益定义为 $net[i] = gas[i] - cost[i]$。当油箱剩余油量 $tank < 0$ 时，说明从起点到当前站构成了净负债前缀，其间任何加油站都无法作为有效起点，候选起点直接跳跃至 $i + 1$。
2. **Best Time to Buy and Sell Stock（买卖股票最佳时机 · LC 121）**：若将每日价格差 $\Delta p_i = price[i] - price[i-1]$ 视为数组，求单次最大利润等价于求 $\Delta p$ 的最大子数组和（Kadane 算法的直接变体）。

---

## 3. 经典 4 大贪心万能模板

| 模板分类 | 核心不变量与操作机制 | 典型代表题目 |
| :--- | :--- | :--- |
| **1. 前缀收益归零与断点重置**<br>*(Prefix Reset)* | 历史累加和 `< 0` 时对未来纯拉低效益，立即抛弃并重置起点。 | **Maximum Subarray** (LC 53)<br>**Gas Station** (LC 134) |
| **2. 覆盖包络线与隐式 BFS 窗口**<br>*(Envelope & BFS Window)* | 维护最远可达边界 `max_reach`；逐层寻找下一跳最大覆盖范围。 | **Jump Game** (LC 55)<br>**Jump Game II** (LC 45) |
| **3. 极端约束强制固定与频次切片**<br>*(Forced Choice & Sorting)* | 最小元素无前驱，被迫开序列；超标分量一票否决。 | **Hand of Straights** (LC 846)<br>**Merge Triplets** (LC 1899) |
| **4. 边界合并与状态区间追踪**<br>*(Interval & Range Bounds)* | 字符最后出现位置闭合切断；通配符追踪 `[min, max]` 容许域。 | **Partition Labels** (LC 763)<br>**Valid Parenthesis String** (LC 678) |

```greedy-patterns
```

---

## 4. NeetCode 150 经典贪心八题深度解构

---

### 题 1：Maximum Subarray（最大子数组和 · LC 53）

#### 问题描述
给定一个整数数组 `nums`，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

#### 贪心决策与不变量证明
设当前连续子数组的和为 `cur_sum`。遍历元素 `x` 时：
- 如果 `cur_sum > 0`，则将 `x` 加入当前子数组必然对 `x` 有增益（即使 `x` 为负，当前整体依然保留了之前的正收益）；
- 如果 `cur_sum <= 0`，说明历史前缀已经变成“负资产”，把负资产加到 `x` 上只会让以 `x` 开头的子数组变小！因此**必须果断抛弃历史前缀，让当前子数组从 `x` 重新开始**。

```text
Kadane 贪心状态转移：
    cur_sum = max(x, cur_sum + x)
    max_sum = max(max_sum, cur_sum)
```

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        cur_sum = 0
        
        for num in nums:
            cur_sum = max(num, cur_sum + num)
            max_sum = max(max_sum, cur_sum)
            
        return max_sum
```

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxSubArray(const std::vector<int>& nums) {
        int max_sum = nums[0];
        int cur_sum = 0;
        for (int num : nums) {
            cur_sum = std::max(num, cur_sum + num);
            max_sum = std::max(max_sum, cur_sum);
        }
        return max_sum;
    }
};
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$。
- **易错陷阱**：数组全为负数时（如 `[-3, -2, -1]`），`max_sum` 不能初始化为 `0`，必须初始化为 `nums[0]`。

---

### 题 2：Jump Game（跳跃游戏 · LC 55）

#### 问题描述
给定一个非负整数数组 `nums`，最初位于第一个下标。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标。

#### 贪心决策与不变量证明
不需要搜索每一种跳跃步数（那会退化成 $O(2^n)$ 回溯）。**我们只需要维护一个全局不变量：当前能够到达的最远下标 `max_reach`**。
- 遍历下标 `i`：如果 `i > max_reach`，说明当前点在之前的任何跳跃范围内都不可达，直接返回 `False`。
- 否则，当前点 `i` 是可达的，可以用 `i + nums[i]` 尝试扩展最远覆盖包络线：`max_reach = max(max_reach, i + nums[i])`。
- 如果 `max_reach >= n - 1`，说明已经可以到达终点，可提前返回 `True`。

```jump-game-demo
```

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reach = 0
        n = len(nums)
        
        for i, jump in enumerate(nums):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i + jump)
            if max_reach >= n - 1:
                return True
                
        return True
```

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    bool canJump(const std::vector<int>& nums) {
        int max_reach = 0;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            if (i > max_reach) return false;
            max_reach = std::max(max_reach, i + nums[i]);
            if (max_reach >= n - 1) return true;
        }
        return true;
    }
};
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$。

---

### 题 3：Jump Game II（跳跃游戏 II · LC 45）

#### 问题描述
给定一个长度为 `n` 的非负整数数组 `nums`，生成到达最后一个下标所需的最少跳跃次数（假设总是可以到达终点）。

#### 贪心决策：隐式 BFS 分层窗口
求最少步数本质是图的最短路。如果将跳跃看作图的边，第 $k$ 次跳跃能到达的全部节点构成一个连续窗口 `[cur_start, cur_end]`。
- 我们不需要用显式队列维护 BFS，只需在遍历当前窗口 `[cur_start, cur_end]` 时，记录所有点能跳到的**最远边界 `farthest = max(farthest, i + nums[i])`**。
- 当指针 `i` 走到当前跳跃窗口的终点 `cur_end` 时，说明当前这一跳的潜力已被全部榨干，必须发起下一次跳跃：
  - `steps += 1`
  - 将窗口右端点推进到 `cur_end = farthest`。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
            
        steps = 0
        cur_end = 0
        farthest = 0
        
        # 注意：遍历到 n - 2 即可！如果遍历到 n - 1，刚好到达终点时会多触发一次无意义的 steps += 1
        for i in range(n - 1):
            farthest = max(farthest, i + nums[i])
            if i == cur_end:
                steps += 1
                cur_end = farthest
                if cur_end >= n - 1:
                    break
                    
        return steps
```

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int jump(const std::vector<int>& nums) {
        int n = nums.size();
        if (n <= 1) return 0;
        
        int steps = 0;
        int cur_end = 0;
        int farthest = 0;
        
        for (int i = 0; i < n - 1; ++i) {
            farthest = std::max(farthest, i + nums[i]);
            if (i == cur_end) {
                ++steps;
                cur_end = farthest;
                if (cur_end >= n - 1) break;
            }
        }
        return steps;
    }
};
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$。
- **核心避坑点**：循环上限必须是 `n - 2`（即 `range(n - 1)`）。因为一旦当前覆盖范围已经到达或超过 `n - 1`，我们已经在终点，不需要再跳第 `steps + 1` 步。

---

### 题 4：Gas Station（加油站 · LC 134）

#### 问题描述
在一条环路上有 `n` 个加油站，其中第 `i` 个加油站有汽油 `gas[i]` 升。从第 `i` 个加油站开往第 `i+1` 个加油站需要消耗汽油 `cost[i]` 升。求从哪个加油站出发可以绕环路行驶一周。若不存在则返回 `-1`。

#### 贪心定理与严密数学证明
1. **全局可达性定理**：如果 $\sum gas[i] < \sum cost[i]$，总油量小于总消耗，必然无解返回 `-1`；反之若 $\sum gas[i] \ge \sum cost[i]$，**环路上必定存在唯一一个可行起点**。
2. **断点跳跃定理（核心贪心）**：
   - 假设我们从起点 `start` 出发，一路顺畅走到 `j - 1`，但在到达 `j` 时累积油量首次出现 `tank < 0`（断油）。
   - **结论：在区间 `[start, j]` 之间的任何一个加油站 $k$（$start \le k \le j$），都绝对不可能作为环路的有效起点！**
   - **证明**：因为从 `start` 出发能顺利走到 $k$，说明到达 $k$ 时油箱里的剩余油量 $\ge 0$。连带着从前面带过来的非负油量都没能跨过 `j`，如果直接以 $k$ 为起点（初始油量为 0），到达 `j` 时油量只会更少，必然更早断油！
   - **贪心操作**：直接将下一个候选起点跃迁至 `start = j + 1`，油箱重置 `tank = 0`。

```gas-station-demo
```

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_surplus = 0
        cur_tank = 0
        start = 0
        
        for i in range(len(gas)):
            net = gas[i] - cost[i]
            total_surplus += net
            cur_tank += net
            
            # 从当前 start 出发在 i 处断油，[start, i] 内全军覆没，跳到 i + 1
            if cur_tank < 0:
                start = i + 1
                cur_tank = 0
                
        return start if total_surplus >= 0 else -1
```

```cpp
#include <vector>

class Solution {
public:
    int canCompleteCircuit(const std::vector<int>& gas, const std::vector<int>& cost) {
        int total_surplus = 0;
        int cur_tank = 0;
        int start = 0;
        
        for (int i = 0; i < gas.size(); ++i) {
            int net = gas[i] - cost[i];
            total_surplus += net;
            cur_tank += net;
            if (cur_tank < 0) {
                start = i + 1;
                cur_tank = 0;
            }
        }
        
        return total_surplus >= 0 ? start : -1;
    }
};
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$。

---

### 题 5：Hand of Straights（一手顺子 / 划分连续组 · LC 846）

#### 问题描述
给定整数数组 `hand` 和整数 `groupSize`。判断能否将全部牌重新排列分成若干组，使得每组大小均为 `groupSize` 且由连续递增的整数组成。

#### 贪心决策：全局最小值无前驱强制开顺子
如果当前剩余的最小牌面是 `x`，请问 `x` 能放在哪个顺子里？
- `x` 能不能作为某个顺子的第二张、第三张？**绝对不能！** 因为那需要存在比 `x` 更小的牌 `x - 1`，但 `x` 已经是全局剩余牌中最小的一张，`x - 1` 已经不存在！
- 因此：**当前剩余的最小牌面 `x` 具有 100% 的强制性，它必须作为以 `x` 为首的顺子 `[x, x+1, ..., x + groupSize - 1]` 的起点！**
- 如果当前 `x` 有 `need = count[x]` 张，那么必须一次性扣除 `count[x + k] -= need`（$0 \le k < groupSize$）。只要其中任何一张牌数量不足 `need`，说明无法配平，立即返回 `False`。

```python
from collections import Counter
from typing import List

class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0:
            return False
            
        count = Counter(hand)
        
        # 按照牌面从小到大强制结算
        for first in sorted(count):
            need = count[first]
            if need == 0:
                continue
                
            for card in range(first, first + groupSize):
                if count[card] < need:
                    return False
                count[card] -= need
                
        return True
```

```cpp
#include <vector>
#include <map>

class Solution {
public:
    bool isNStraightHand(const std::vector<int>& hand, int groupSize) {
        if (hand.size() % groupSize != 0) return false;
        
        std::map<int, int> count;
        for (int card : hand) {
            count[card]++;
        }
        
        for (auto [first, freq] : count) {
            if (freq == 0) continue;
            
            for (int k = 0; k < groupSize; ++k) {
                int card = first + k;
                if (count[card] < freq) return false;
                count[card] -= freq;
            }
        }
        return true;
    }
};
```

- **复杂度**：时间 $O(u \log u + u \cdot groupSize)$（其中 $u \le n$ 是不同牌面的数量），空间 $O(u)$。

---

### 题 6：Merge Triplets to Form Target Triplet（合并三元组达到目标 · LC 1899）

#### 问题描述
给定一个二维整数数组 `triplets`，其中 `triplets[i] = [ai, bi, ci]`。同时给定目标三元组 `target = [x, y, z]`。你可以任意次选择两个三元组取各分量的最大值 `[max(a1, a2), max(b1, b2), max(c1, c2)]` 进行合并。问最终能否得到 `target`？

#### 贪心决策：超标候选一票否决与坐标独立性
`max` 操作具有**单调不减性**（一旦某个坐标的值超过了目标值，就永远无法通过后续合并降低）。
1. **一票否决安全过滤**：如果一个三元组 `t` 的任何一个分量超过了目标分量（即 `t[0] > target[0] or t[1] > target[1] or t[2] > target[2]`），这个三元组**绝对不能参与任何合并**，必须直接忽略！
2. **贪心全合并**：对于剩下所有安全的三元组（三个分量均 $\le target$），我们将它们全部合并起来，各个分量也绝不会超过 `target`！
3. **达标判定**：只要在这些安全三元组中，能分别找到 $t[0] == target[0]$、$t[1] == target[1]$ 和 $t[2] == target[2]$ 的候选者，那么全部合并后必然精确得到 `target`。

```python
class Solution:
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        tx, ty, tz = target
        has_x = has_y = has_z = False
        
        for a, b, c in triplets:
            # 只要有一个分量超标，该三元组废弃
            if a > tx or b > ty or c > tz:
                continue
                
            if a == tx: has_x = True
            if b == ty: has_y = True
            if c == tz: has_z = True
            
            if has_x and has_y and has_z:
                return True
                
        return False
```

```cpp
#include <vector>

class Solution {
public:
    bool mergeTriplets(const std::vector<std::vector<int>>& triplets, const std::vector<int>& target) {
        int tx = target[0], ty = target[1], tz = target[2];
        bool has_x = false, has_y = false, has_z = false;
        
        for (const auto& t : triplets) {
            if (t[0] > tx || t[1] > ty || t[2] > tz) continue;
            
            if (t[0] == tx) has_x = true;
            if (t[1] == ty) has_y = true;
            if (t[2] == tz) has_z = true;
            
            if (has_x && has_y && has_z) return true;
        }
        return false;
    }
};
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$。

---

### 题 7：Partition Labels（划分字母区间 · LC 763）

#### 问题描述
给你一个字符串 `s`。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

#### 贪心决策：最远出现位置包络与即时切断
1. **预处理**：扫描一遍字符串，记录每个字符最后一次出现的下标 `last[c]`。
2. **贪心扫描**：维护当前片段的起始点 `start` 和最远必需延伸边界 `end`：
   - 遍历到字符 `s[i]` 时，当前片段必须至少延伸到 `last[s[i]]`，因此 `end = max(end, last[s[i]])`。
   - 当指针 `i` 走到 `end` 时（`i == end`），说明**当前片段内包含的所有字符在后面都不会再出现**！
   - 此时可以贪心地立刻在此处切断，记录片段长度 `i - start + 1`，并开启新片段 `start = i + 1`。这样能保证切出的片段数最多、各片段长度最短。

```partition-labels-demo
```

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = {c: i for i, c in enumerate(s)}
        
        partitions = []
        start = 0
        end = 0
        
        for i, c in enumerate(s):
            end = max(end, last[c])
            if i == end:
                partitions.append(i - start + 1)
                start = i + 1
                
        return partitions
```

```cpp
#include <vector>
#include <string>
#include <algorithm>

class Solution {
public:
    std::vector<int> partitionLabels(const std::string& s) {
        int last[26] = {0};
        for (int i = 0; i < s.size(); ++i) {
            last[s[i] - 'a'] = i;
        }
        
        std::vector<int> partitions;
        int start = 0;
        int end = 0;
        
        for (int i = 0; i < s.size(); ++i) {
            end = std::max(end, last[s[i] - 'a']);
            if (i == end) {
                partitions.push_back(i - start + 1);
                start = i + 1;
            }
        }
        return partitions;
    }
};
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$（字符集大小固定为 26）。

---

### 题 8：Valid Parenthesis String（有效的括号字符串 · LC 678）

#### 问题描述
给定一个只包含 `'('`、`')'` 和 `'*'` 的字符串。`'*'` 可以被视为 `'('`、`')'` 或一个空字符串 `""`。判断该字符串是否有效。

#### 贪心决策：状态区间范围追踪 `[cmin, cmax]`
如果使用回溯或 DP，每个 `*` 有 3 种分支，最坏复杂度会达到 $O(3^n)$ 或 $O(n^2)$。
贪心的精髓在于：**我们不需要记录所有分支，只需要追踪当前未匹配左括号数量的闭区间范围 `[cmin, cmax]`**：
- `cmax`：将所有通配符 `*` 都贪心地当作 `'('` 时的未匹配左括号数（上限）；
- `cmin`：将所有通配符 `*` 都贪心地当作 `')'` 时的未匹配左括号数（下限，遇到 0 时不能为负，截断为 0）。

状态转移规则：
1. 遇到 `'('`：`cmin += 1, cmax += 1`
2. 遇到 `')'`：`cmin = max(0, cmin - 1), cmax -= 1`
3. 遇到 `'*'`：`cmin = max(0, cmin - 1)`（当 `)` 或 `""`）, `cmax += 1`（当 `(`）
4. 剪枝合法性检验：
   - 若 `cmax < 0`，说明把所有 `*` 全部当成 `'('` 都无法抵消多余的 `')'`，立即返回 `False`；
   - 遍历结束后，只要 `cmin == 0`，说明在容许范围内存在一种抵消方案使得最终未匹配左括号恰好为 0，返回 `True`。

```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        cmin = 0 # 尽可能将 * 视作 ) 时，最少未闭合的 '(' 数量
        cmax = 0 # 尽可能将 * 视作 ( 时，最多未闭合的 '(' 数量
        
        for ch in s:
            if ch == '(':
                cmin += 1
                cmax += 1
            elif ch == ')':
                cmin -= 1
                cmax -= 1
            else: # '*'
                cmin -= 1
                cmax += 1
                
            if cmax < 0:
                return False
                
            cmin = max(cmin, 0)
            
        return cmin == 0
```

```cpp
#include <string>
#include <algorithm>

class Solution {
public:
    bool checkValidString(const std::vector<char>& s) {
        int cmin = 0;
        int cmax = 0;
        
        for (char ch : s) {
            if (ch == '(') {
                ++cmin;
                ++cmax;
            } else if (ch == ')') {
                --cmin;
                --cmax;
            } else {
                --cmin;
                ++cmax;
            }
            
            if (cmax < 0) return false;
            cmin = std::max(cmin, 0);
        }
        
        return cmin == 0;
    }
};
```

- **复杂度**：时间 $O(n)$，空间 $O(1)$。

---

## 5. 贪心算法面试避坑与对比速查

| 序号 / 题目 | 核心贪心判定 | 致命避坑陷阱 | 复杂度 |
| :--- | :--- | :--- | :--- |
| **53. Maximum Subarray** | 前缀和 `< 0` 立即归零并重新开始 | 全负数数组时 `max_sum` 绝不能初始化为 0 | 时间 $O(n)$<br>空间 $O(1)$ |
| **55. Jump Game** | 维护 `max_reach` 最远可达包络线 | 遇到 0 盲目回溯（只需 `max_reach` 越过即可） | 时间 $O(n)$<br>空间 $O(1)$ |
| **45. Jump Game II** | 隐式 BFS 窗口推进（`i == cur_end` 时 `steps += 1`） | 循环遍历至 `n - 1` 导致在终点多跳一步 | 时间 $O(n)$<br>空间 $O(1)$ |
| **134. Gas Station** | 断油点前全部排除，候选起点直接跳至 `i + 1` | 忘记判断全局总净油量 `total_surplus >= 0` | 时间 $O(n)$<br>空间 $O(1)$ |
| **846. Hand of Straights** | 全局最小剩余牌无前驱，100% 强制作为顺子起点 | 未提前检查 `len(hand) % groupSize == 0` | 时间 $O(n \log n)$<br>空间 $O(n)$ |
| **1899. Merge Triplets** | 任何分量超标的三元组一票否决，其余安全项全合并 | 误以为要单个三元组全匹配（实际各坐标独立） | 时间 $O(n)$<br>空间 $O(1)$ |
| **763. Partition Labels** | 预处理 `last[c]`，当扫描指针到达 `i == end` 时即时切断 | 边扫描边切断而未提前预处理最后下标 | 时间 $O(n)$<br>空间 $O(1)$ |
| **678. Valid Parenthesis String** | 维护未配对左括号连续范围 `[cmin, cmax]` | 忘记将 `cmin` 下限截断至 0 | 时间 $O(n)$<br>空间 $O(1)$ |

**要记住**
- 面试时先讲“不变量”和“为什么这个局部选择不会漏解”，切忌直接甩代码。
- 凡是遇到“前缀累计收益可能拖累后续”的序列问题，优先联想 Kadane 和 Gas Station 的断点重置。
- 凡是遇到“覆盖范围最少步数”问题，优先使用 Jump Game II 的隐式 BFS 窗口模型。
- 凡是多维 `max`/`min` 目标合并问题，优先考虑超标元素的**一票否决预过滤**。
- 遇到带有通配符的括号匹配，不要写指数级回溯，追踪未匹配括号的 `[min, max]` 范围是唯一的 $O(n)$ 最优解。
