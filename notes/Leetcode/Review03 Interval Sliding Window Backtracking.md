# 复习卡片：区间、滑窗与回溯 (Review Flashcards · Intervals, Sliding Window & Backtracking)

本篇为算法面试高频复习卡片第三辑：系统梳理**区间重叠与扫描线 (Intervals & Sweep Line)**、**双指针与滑动窗口 (Two Pointers & Sliding Window)** 以及**回溯与组合搜索 (Backtracking & Combinatorial Search)** 的核心高频考题、拓扑变体、生产级实现与时空复杂度全景。

---

## 模块一：区间重叠与扫描线 (Intervals & Sweep Line)

### 1. 合并区间与其工业级拓扑变体全家桶 (Merge Intervals & Interval Topology Variants)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">区间 01</span>
  <span class="review-card-title">合并区间与其工业级拓扑变体全家桶 (Merge Intervals & Interval Topology Variants)</span>
  <span class="review-card-tag">闭区间语义 · 预排序加速 · 嵌套覆盖 · 插入区间 · 无重叠区间贪心 · 会议室扫描线</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与核心变体全景矩阵</div>

给定一个区间的集合 `intervals`，其中若干区间可能发生重叠。请合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间：

```python
def merge(intervals: List[List[int]]) -> List[List[int]]: ...
```

| 变体编号 | 核心变体名称 | 核心特征 / 变异条件 | 算法架构与破局关键 |
|---|---|---|---|
| **变体 1** | **经典闭区间合并 (LC 56)** | 输入为无序闭区间 $[s, e]$，端点相碰（如 $[1, 4]$ 与 $[4, 5]$）算作重叠 | **起始端点升序排序** + 维护 `merged[-1][1] = max(merged[-1][1], cur[1])`。 |
| **变体 2** | **预排序输入加速 (Presorted)** | 上游数据流已知按起始时间 `start` 严格递增送达 | **免去 $O(N \log N)$ 排序**，直接单趟 $O(N)$ 线性扫描，零额外时间损耗。 |
| **变体 3** | **完全嵌套覆盖 (Nested)** | 存在完全被外层区间包裹的子区间（如 $[1, 10]$ 与 $[2, 5]$） | 由右边界 `max` 函数自动覆盖吸收，结果维持 $[1, 10]$。 |
| **变体 4** | **插入新区间 (LC 57)** | 给定已升序且无重叠的区间列表，插入单个新区间 `newInterval` | **三阶段线性归并**：严格在左侧 $\to$ 重叠区融合扩圈 $\to$ 严格在右侧，达成 $O(N)$ 时间与 $O(1)$ 空间。 |
| **变体 5** | **无重叠区间最小剔除 (LC 435)** | 计算最少需要移除多少个区间，才能使剩余区间互不重叠 | **按结束时间 `end` 贪心排序**：优先保留结束时间最早的区间，为后续留出最大空间。 |
| **变体 6** | **会议室需求峰值 (LC 253)** | 求解最多同时进行的会议数（即所需最少会议室数量） | **扫描线 / 差分事件法**（`start` 产生 $+1$，`end` 产生 $-1$ 排序）或**小根堆维护当前会议结束时间**。 |
| **变体 7** | **开闭区间语义判据 (Open vs Closed)** | 业务场景为半开区间 $[s, e)$（如网络端口、时间片） | 相碰端点不再重叠（$[1, 4)$ 与 $[4, 5)$ 无交集），重叠判定条件由 `cur[0] <= prev[1]` 调整为严格小于 `cur[0] < prev[1]`。 |

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心机制深度剖析</div>

- **排序保证单调性**：按 `start` 升序排序后，只需比对 `cur[0] <= merged[-1][1]`。若满足直接合并；否则另立门户。
- **插入区间三阶段**：左侧无交集直接加 $\to$ 交集扩展范围 $\to$ 右侧无交集直接追加，全程严格单趟 $O(N)$。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List
import heapq

class IntervalSolution:
    @staticmethod
    def merge(intervals: List[List[int]]) -> List[List[int]]:
        if not intervals: return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for i in range(1, len(intervals)):
            cur = intervals[i]
            if cur[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], cur[1])
            else:
                merged.append(cur)
        return merged

    @staticmethod
    def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        i, n = 0, len(intervals)
        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        res.append(newInterval)
        while i < n:
            res.append(intervals[i])
            i += 1
        return res

    @staticmethod
    def minMeetingRooms(intervals: List[List[int]]) -> int:
        if not intervals: return 0
        intervals.sort(key=lambda x: x[0])
        rooms = []
        heapq.heappush(rooms, intervals[0][1])
        for start, end in intervals[1:]:
            if rooms[0] <= start:
                heapq.heappop(rooms)
            heapq.heappush(rooms, end)
        return len(rooms)
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：`merge` 为 $\mathcal{O}(N \log N)$（预排序时为 $\mathcal{O}(N)$）；`insert` 为严格 $\mathcal{O}(N)$。
- **高频避坑**：闭区间判断重叠必须包含等号 `cur[0] <= merged[-1][1]`。

</div>

</div>
</details>

---

## 模块二：双指针与滑动窗口 (Two Pointers & Sliding Window)

### 2. 勾股数三元组判定与多重集保留 (Pythagorean Triplet & Multiplicity 2-Pointer Search)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">双指针 02</span>
  <span class="review-card-title">勾股数三元组判定与多重集保留 (Pythagorean Triplet & Multiplicity 2-Pointer Search)</span>
  <span class="review-card-tag">平方映射 · 元素多重集保留 · 3SUM 规约 · 双指针相向夹逼 · O(N^2) 最优性证明</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与典型示例矩阵</div>

给定一个整数数组 `nums`，判断是否存在三个元素 $a, b, c$ 满足勾股定理：
$$a^2 + b^2 = c^2$$

**核心条件约束与面试切入点**：
1. **允许负数**：输入可以包含负整数（因为公式针对平方项，如 $(-3)^2 + 4^2 = 5^2$）。
2. **多重集重数约束 (Element Multiplicity)**：同一个数值只有在原数组中拥有足够多的物理副本时，才允许重复使用。原数组中的单个元素绝对不能同时充当多个位置。
3. **最优性理论辩护**：在无附加值域范围假设下，证明通用算法无法突破至低于 $\mathcal{O}(N^2)$ 的亚二次复杂度。

**经典测试用例判题矩阵**：

| 测试用例 | 预期返回值 | 核心判题考点与成因剖析 |
|---|---|---|
| `[0, 1, -2, 3, 4, 5]` | `True` | 存在 $3^2 + 4^2 = 5^2$ ($9 + 16 = 25$) |
| `[0, 0, 0]` | `True` | 数组拥有 3 个独立的 0，满足 $0^2 + 0^2 = 0^2$ |
| `[0, 1, -1]` | `True` | 满足 $0^2 + (-1)^2 = 1^2$ ($0 + 1 = 1$) |
| `[0]` | `False` | 单个 0 无法同时供给 3 个位置，**严禁判为 True** |
| `[0, 2]` | `False` | 元素不足 3 个，无法构成三元组 |

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心机制深度剖析</div>

#### 1. 为什么纯哈希表 (Set) 方案是致命错误的？
- 若先计算所有平方存入哈希集合 `squares_set = {x**2 for x in nums}`，然后双重循环枚举 $a, b$ 检查 $a^2 + b^2 \in squares\_set$：
  - 当输入仅为 `[0, 5]` 时，$0^2 + 0^2 = 0^2$ 会在集合中命中 $0$，导致错误返回 `True`！
  - 纯哈希表彻底抹杀了**元素多重集（Multiplicity）** 的物理索引独立性。

#### 2. 平方映射 + 升序排序 + 双指针夹逼黄金解法
1. **就地平方映射**：遍历数组将每个元素替换为其平方 $x \gets x^2$。此时所有元素自动转为非负整数，天然消除负号影响。
2. **升序排序**：对平方数组进行快速排序，耗时 $\mathcal{O}(N \log N)$。
3. **固定最长斜边，双指针夹逼两直角边**：
   - 从最大下标 $k = N - 1$ 逆序遍历至 $2$，将 $nums[k]$ 视为潜在的斜边平方 $c^2$。
   - 在左侧子区间 $[0, k - 1]$ 设置双指针 $i = 0, j = k - 1$：
     - 若 $nums[i] + nums[j] == nums[k]$：由于三者下标严格满足 $i < j < k$，代表三者来自原数组中**互不相同的物理位置**，多重集重数得到天然保证，直接返回 `True`！
     - 若 $nums[i] + nums[j] < nums[k]$：两数和过小，左指针右移 $i \gets i + 1$；
     - 若 $nums[i] + nums[j] > nums[k]$：两数和过大，右指针左移 $j \gets j - 1$。
4. 若遍历完所有 $k$ 仍未找到匹配，返回 `False`。

#### 3. 为什么在无值域约束下无法超越 $O(N^2)$？（面试理论答辩）
- 本质上，勾股三元组判定是 **3SUM 问题** 的严格同构变体（将 $a + b + (-c) = 0$ 替换为非负实数域上的 $a' + b' - c' = 0$）。
- 在理论计算机科学中，3SUM 问题是 **3SUM 猜想 (3SUM Conjecture)** 的核心基石：在比较模型（Real RAM Model）下，不存在时间复杂度为 $\mathcal{O}(N^{2 - \varepsilon})$（$\varepsilon > 0$）的亚二次算法。
- **何时能突破？** 唯有在引入了强值域约束（例如限定元素值域 $|nums[i]| \le U$）时，才可以通过布尔卷积或快速傅里叶变换 (FFT) 在 $\mathcal{O}(U^2 \log U)$ 或位图切分下突破，但在通用任意整数输入下，$\mathcal{O}(N^2)$ 已是渐进最优。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List

class PythagoreanTripletSolution:
    """
    勾股三元组判定系统级实现：
    严格保证元素多重集索引独立性，完美应对负数与重复 0
    时间复杂度 O(N^2)，额外空间复杂度 O(1)
    """

    @staticmethod
    def judgePythagoreanTriplet(nums: List[int]) -> bool:
        n = len(nums)
        if n < 3:
            return False

        # 1. 就地平方映射
        squares = [x * x for x in nums]

        # 2. 升序排序
        squares.sort()

        # 3. 逆序枚举斜边 c^2 的下标 k
        for k in range(n - 1, 1, -1):
            target = squares[k]
            i = 0
            j = k - 1

            # 双指针相向夹逼直角边 a^2 + b^2
            while i < j:
                cur_sum = squares[i] + squares[j]
                if cur_sum == target:
                    return True
                elif cur_sum < target:
                    i += 1
                else:
                    j -= 1

        return False
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：
  - 平方映射耗时 $\mathcal{O}(N)$。
  - 排序耗时 $\mathcal{O}(N \log N)$。
  - 外层循环遍历 $k$（$N$ 次），内层双指针扫描区间 $[0, k-1]$ 最多走 $k$ 步，总比较步数为 $\sum_{k=2}^{N-1} k = \frac{(N-1)(N-2)}{2} = \mathcal{O}(N^2)$。
  - 整体运行时间为严格 $\mathcal{O}(N^2)$。
- **空间复杂度**：
  - 若允许修改输入，空间为 $\mathcal{O}(1)$；若保留原数组，平方列表占用 $\mathcal{O}(N)$ 空间。
- **高频避坑清单**：
  1. **单零判真陷阱**：使用哈希表未记录频次，输入 `[0]` 或 `[0, 1]` 被误判为 True。
  2. **双指针越界相撞**：内层循环终止条件必须是 `while i < j:`，不可包含等号，防止同一位置被自加两次。

</div>

</div>
</details>

---

### 3. 变长与定长滑动窗口经典范式 (Variable & Fixed Sliding Window Patterns)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">滑窗 03</span>
  <span class="review-card-title">变长与定长滑动窗口经典范式 (Variable & Fixed Sliding Window Patterns)</span>
  <span class="review-card-tag">变长双指针 · 哈希跳跃加速 · 字符频次差分机 · 状态满足度计数器</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与经典模型</div>

- **无重复字符的最长子串 (LC 3)**：找出无重复字符的最长子串长度。
- **最小覆盖子串 (LC 76)**：涵盖目标集所有字符的最小子串。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
class SlidingWindowSolution:
    @staticmethod
    def lengthOfLongestSubstring(s: str) -> int:
        last_seen = {}
        max_len = left = 0
        for right, ch in enumerate(s):
            if ch in last_seen:
                left = max(left, last_seen[ch] + 1)
            last_seen[ch] = right
            max_len = max(max_len, right - left + 1)
        return max_len

    @staticmethod
    def minWindow(s: str, t: str) -> str:
        from collections import Counter
        target_counts = Counter(t)
        window_counts = {}
        required = len(target_counts)
        formed = left = 0
        min_len, best_left = float('inf'), 0

        for right, ch in enumerate(s):
            window_counts[ch] = window_counts.get(ch, 0) + 1
            if ch in target_counts and window_counts[ch] == target_counts[ch]:
                formed += 1
            while left <= right and formed == required:
                if right - left + 1 < min_len:
                    min_len, best_left = right - left + 1, left
                left_ch = s[left]
                window_counts[left_ch] -= 1
                if left_ch in target_counts and window_counts[left_ch] < target_counts[left_ch]:
                    formed -= 1
                left += 1

        return "" if min_len == float('inf') else s[best_left : best_left + min_len]
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：$\mathcal{O}(N)$。
- **空间复杂度**：$\mathcal{O}(|\Sigma|)$。

</div>

</div>
</details>

---

## 模块三：回溯与组合搜索 (Backtracking & Combinatorial Search)

### 4. 子集、排列与组合通用回溯范式 (Subsets, Permutations & Combinations)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">回溯 04</span>
  <span class="review-card-title">子集、排列与组合通用回溯范式 (Subsets, Permutations & Combinations)</span>
  <span class="review-card-tag">树形状态空间 · 剪枝去重 · 元素复用控制 · 使用标记数组</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心变体全景矩阵</div>

| 题型类型 | 典型代表 | 核心决策点 | 核心去重 / 状态传递方式 |
|---|---|---|---|
| **子集型 (Subsets)** | LC 78, LC 90 | 选或不选当前元素，收集所有节点 | 传 `start` 下标；若有重复，先排序，且同层判断 `i > start and nums[i] == nums[i-1]` 剪枝。 |
| **组合型 (Combinations)** | LC 77, LC 39, LC 40 | 选满指定个数或凑满特定和 | 传 `start` 下标控制元素不逆序选取；目标和超出直接提前返回剪枝。 |
| **排列型 (Permutations)** | LC 46, LC 47 | 元素顺序敏感，每次从 0 开始挑 | 维护布尔数组 `used`；同层去重：`nums[i] == nums[i-1] and not used[i-1]` 剪枝。 |

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List

class BacktrackSolution:
    @staticmethod
    def subsetsWithDup(nums: List[int]) -> List[List[int]]:
        nums.sort()
        res, path = [], []
        def backtrack(start: int):
            res.append(list(path))
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return res

    @staticmethod
    def combinationSum2(candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res, path = [], []
        def backtrack(start: int, remain: int):
            if remain == 0:
                res.append(list(path))
                return
            for i in range(start, len(candidates)):
                if candidates[i] > remain:
                    break
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                backtrack(i + 1, remain - candidates[i])
                path.pop()
        backtrack(0, target)
        return res
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：$\mathcal{O}(N \cdot 2^N)$。
- **空间复杂度**：$\mathcal{O}(N)$。

</div>

</div>
</details>
