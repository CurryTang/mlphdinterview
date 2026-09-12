# 复习卡片：区间、滑窗与回溯 (Review Flashcards · Intervals, Sliding Window & Backtracking)

本篇为算法面试高频复习卡片第三辑：系统梳理**区间重叠与扫描线 (Intervals & Sweep Line)**、**双指针与滑动窗口 (Two Pointers & Sliding Window)** 以及**回溯与数位贪心 (Backtracking & Digit Greedy)** 的核心高频考题、拓扑变体、生产级实现与时空复杂度全景。

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

给定一个区间的集合 `intervals`，合并所有重叠的闭区间，返回不重叠且覆盖全集的区间列表：

```python
def merge(intervals: List[List[int]]) -> List[List[int]]: ...
```

| 变体编号 | 核心变体名称 | 核心特征 / 变异条件 | 算法架构与破局关键 |
|---|---|---|---|
| **变体 1** | **经典闭区间合并 (LC 56)** | 输入为无序闭区间 $[s, e]$，端点相碰（如 $[1, 4]$ 与 $[4, 5]$）算作重叠 | **起始端点升序排序** + 维护 `merged[-1][1] = max(merged[-1][1], cur[1])`。 |
| **变体 2** | **预排序输入加速 (Presorted)** | 上游数据流已知按起始时间 `start` 递增送达 | **免去排序**，直接单趟 $O(N)$ 线性扫描，零额外时间损耗。 |
| **变体 3** | **完全嵌套覆盖 (Nested)** | 存在完全被外层包裹的子区间（如 $[1, 10]$ 与 $[2, 5]$） | 由右边界 `max` 函数自动覆盖吸收，结果维持 $[1, 10]$。 |
| **变体 4** | **插入新区间 (LC 57)** | 给定已升序且无重叠的区间列表，插入单个新区间 `newInterval` | **三阶段线性归并**：严格在左侧 $\to$ 重叠区融合扩圈 $\to$ 严格在右侧，达成 $O(N)$ 时间与 $O(1)$ 空间。 |
| **变体 5** | **无重叠区间最小剔除 (LC 435)** | 计算最少需要移除多少个区间使剩余不重叠 | **按结束时间 `end` 贪心排序**：优先保留结束时间最早的区间。 |
| **变体 6** | **会议室需求峰值 (LC 253)** | 求解最多同时进行的会议数（所需最少会议室数量） | **扫描线 / 差分事件法**（`start` 为 $+1$，`end` 为 $-1$）或**小根堆维护结束时间**。 |

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
        res, i, n = [], 0, len(intervals)
        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i]); i += 1
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        res.append(newInterval)
        while i < n:
            res.append(intervals[i]); i += 1
        return res
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- `merge`: $\mathcal{O}(N \log N)$（已排序为 $\mathcal{O}(N)$）；`insert`: $\mathcal{O}(N)$；空间 $\mathcal{O}(1)$。

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
<div class="review-block-label">📌 题目定义与测试矩阵</div>

给定一个整数数组 `nums`，判断是否存在三个元素 $a, b, c$ 满足 $a^2 + b^2 = c^2$：
- 允许负数；同一个数值只有在原数组中拥有足够多的物理副本时才允许复用；证明无值域约束下无法超越 $\mathcal{O}(N^2)$。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List

class PythagoreanTripletSolution:
    @staticmethod
    def judgePythagoreanTriplet(nums: List[int]) -> bool:
        n = len(nums)
        if n < 3: return False
        squares = sorted([x * x for x in nums])
        for k in range(n - 1, 1, -1):
            target = squares[k]
            i, j = 0, k - 1
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
<div class="review-block-label">⏱️ 复杂度与理论依据</div>

- 时间复杂度严格为 $\mathcal{O}(N^2)$。3SUM 猜想证明比较模型下不存在 $\mathcal{O}(N^{2-\varepsilon})$ 亚二次算法。空间 $\mathcal{O}(N)$。

</div>

</div>
</details>

---

### 3. 最小覆盖子串与多最优解候选集 (Minimum Window Substring & Multi-Candidate Expansion)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">滑窗 03</span>
  <span class="review-card-title">最小覆盖子串与多最优解候选集 (Minimum Window Substring & Multi-Candidate Expansion)</span>
  <span class="review-card-tag">变长滑窗 · O(|S| + |T|) 复杂度推导 · k 倍频次扩展 · 全量平局最短子串 · 定长数组常数优化</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与工业面试核心追问</div>

给定源字符串 $S$ 和目标字符串 $T$，在 $S$ 中寻找包含 $T$ 中所有必需字符（严格保留重复字符频次）的最短子串：

```python
def minWindow(s: str, t: str) -> str: ...
```

在系统大厂（如 Apple / Google）多轮代码考核中，该题常引申出以下四大核心追问：
1. **复杂度严格推导**：给出并严密证明时间复杂度为 $\mathcal{O}(|S| + |T|)$ 的数学论据。
2. **频次 $k$ 倍扩展追问**：若要求变更——目标集中的每个字符必须在窗口中至少出现 $k$ 次（或原始频次的 $k$ 倍），状态机如何迁移？
3. **全量平局最短子串返回 (All Tied Minimum Windows)**：若存在多个长度相同且均满足条件的最小窗口，要求**返回所有平局的最短子串列表**，而非仅单个答案。
4. **哈希表 vs 定长数组常数边界挑战**：哈希表操作是否为真正的 $O(1)$？若面临面试官挑战，如何用 ASCII 128 定长数组实现严格无哈希碰撞的常数时间界。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心状态机机制剖析</div>

#### 1. 复杂度为何严格是 $O(|S| + |T|)$？
- 预统计阶段：扫描目标串 $T$ 构建频次表耗时 $\mathcal{O}(|T|)$。
- 双指针滑窗阶段：右指针 `right` 从 0 到 $|S|-1$ 单调递增，执行 $|S|$ 次扩展；左指针 `left` 同样只向右移动，单调递增至多 $|S|$ 次收缩。每个字符最多进窗 1 次、出窗 1 次。
- 窗口内状态维护：利用定长数组或哈希表配合计数器 `formed`，单次进出窗更新耗时 $\mathcal{O}(1)$。
- 总体时间复杂度严格为 $\mathcal{O}(|S| + |T|)$。

#### 2. 状态机不变量：满足度计数器 `formed`
- 设 $T$ 中包含 $U$ 种互不相同的字符。维护 `required: Dict[char, int]`。
- 维护变量 `formed = 0`：表示当前窗口内出现频次**已经达到或超过**目标要求的字符种类数。
- **神圣更新规则**：
  - 当字符 $ch$ 进窗且计数值**恰好等于** `required[ch]` 的瞬间，执行 `formed += 1`；后续如果该字符继续冗余出现，**严禁重复递增**！
  - 当字符 $ch$ 出窗且计数值**从合格跌落为不合格（即由等于变小于）** 的瞬间，执行 `formed -= 1`。

#### 3. 收缩前先行归档与全平局结果集收集
- **收缩前先行记录**：当 `formed == required` 时，当前区间 $[left, right]$ 必定合法。**必须在执行 `left += 1` 移出字符之前先行记录当前解**！因为移出字符后窗口可能瞬间失效。
- **全平局最短子串收集算法**：
  - 维护全局最短长度 `min_len = float('inf')` 与结果列表 `ans_list = []`。
  - 发现更短的合法窗口：若 `right - left + 1 < min_len`，重置 `min_len = right - left + 1`，并**清空列表** `ans_list = [s[left : right + 1]]`；
  - 发现长度相等的平局合法窗口：若 `right - left + 1 == min_len`，直接追加 `ans_list.append(s[left : right + 1])`。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码（含全平局输出与 k 倍频次支持）</div>

```python
from typing import List, Dict
from collections import Counter

class MinWindowSolution:

    @classmethod
    def minWindowAll(cls, s: str, t: str, k_scale: int = 1) -> List[str]:
        """
        全量平局最小覆盖子串收集器：
        1. 支持 k_scale 频次倍数扩展
        2. 返回所有长度并列最小的有效子串列表
        3. 采用 ASCII 定长数组实现严格常数时间界
        时间复杂度 O(|S| + |T|)，空间复杂度 O(|Σ|)
        """
        if not s or not t or k_scale <= 0:
            return []

        # 统计目标需求（支持 k 倍扩展）
        target_counts = Counter(t)
        # 用定长数组替代哈希表消除碰撞开销
        req = [0] * 128
        for ch, count in target_counts.items():
            req[ord(ch)] = count * k_scale

        required_kinds = len(target_counts)
        formed_kinds = 0

        win = [0] * 128
        min_len = float('inf')
        ans_list: List[str] = []

        left = 0
        s_len = len(s)

        for right in range(s_len):
            r_code = ord(s[right])
            win[r_code] += 1

            # 仅当频次恰好达到目标门槛时，有效种类计数加一
            if req[r_code] > 0 and win[r_code] == req[r_code]:
                formed_kinds += 1

            # 窗口处于完全合法状态，尝试收缩左边界以寻找局部最小
            while left <= right and formed_kinds == required_kinds:
                cur_len = right - left + 1

                # 必须在弹出左字符之前归档答案！
                if cur_len < min_len:
                    min_len = cur_len
                    ans_list = [s[left : right + 1]]
                elif cur_len == min_len:
                    ans_list.append(s[left : right + 1])

                l_code = ord(s[left])
                win[l_code] -= 1
                # 仅当频次跌破目标门槛时，有效种类计数减一
                if req[l_code] > 0 and win[l_code] < req[l_code]:
                    formed_kinds -= 1

                left += 1

        return ans_list
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：左右指针均只进不退，字符比较操作借助定长数组为严格 $\mathcal{O}(1)$，总体耗时严格为 $\mathcal{O}(|S| + |T|)$。
- **空间复杂度**：字符集大小固化为 $\mathcal{O}(128) = \mathcal{O}(1)$。
- **高频避坑清单**：
  1. **出窗后才记录答案**：千万不可将结果更新写在 `left += 1` 之后，否则记录的子串已被破坏截断。
  2. **冗余字符重复增加 `formed`**：若目标需要 2 个 `'A'`，窗口内出现第 3 个 `'A'` 时绝不能再次递增 `formed`。

</div>

</div>
</details>

---

### 4. 替换后的最长重复字符与滑窗最频计数维持 (Longest Repeating Character Replacement & Max-Frequency Invariant)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">滑窗 04</span>
  <span class="review-card-title">替换后的最长重复字符与滑窗最频计数维持 (Longest Repeating Character Replacement & Max-Frequency Invariant)</span>
  <span class="review-card-tag">变长滑动窗口 · 最频字符不变式 · 窗口非递减贪心 · O(N) 单趟线性</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与经典模型</div>

给你一个只包含大写英文字母的字符串 `s`，你可以将其中至多 `k` 个字符替换为任意其他大写英文字母。返回在执行至多 `k` 次替换后，能够获得的最长由相同字符构成的子串长度（LC 424）：

```python
def characterReplacement(s: str, k: int) -> int: ...
```

**输入输出示例**：
- 输入：`s = "ABAB", k = 2` $\implies$ 输出：`4`
- 输入：`s = "AABABBA", k = 1` $\implies$ 输出：`4`

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与窗口最频字符维持精髓</div>

#### 1. 窗口合法性充要条件
对于任何窗口 $[left, right]$，其长度为 $L = right - left + 1$。
- 设该窗口内出现频次最高的字符频次为 $\operatorname{max\_freq}$。
- 为了将窗口全部变为该字符，需要替换的非主导字符数量为：
  $$\text{需替换数} = L - \operatorname{max\_freq}$$
- 只要满足 $L - \operatorname{max\_freq} \le k$，当前窗口即为合法窗口！

#### 2. 为什么收缩时不需全局重新扫描 `max_freq`？（高级面试关键点）
- 朴素思考：当窗口左端字符移出（`counts[s[left]] -= 1`）时，似乎需要重新遍历 26 个字母计算新的最高频次。
- **突破性洞察**：**我们根本不需要重算，保持 `max_freq` 为历史峰值即可！**
  - 因为我们追求的是**全局最长长度**。
  - 如果左移后窗口内的实际最大频次变小了，该缩小的频次绝对不可能帮助我们刷新全局最大长度记录。
  - 唯有在未来的某个时刻，进窗字符让某个字符的频次**超越了历史最高值 `max_freq`** 时，窗口才可能被进一步拓宽。
  - 因此，`max_freq` 具有单调不减性，省去了每步 $\mathcal{O}(26)$ 的重算，代码运行速度达到极致！

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
class CharacterReplacementSolution:

    @staticmethod
    def characterReplacement(s: str, k: int) -> int:
        """LC 424: 替换后的最长重复字符，单趟 O(N)"""
        counts = [0] * 26
        left = 0
        max_freq = 0
        max_len = 0

        for right, ch in enumerate(s):
            idx = ord(ch) - ord('A')
            counts[idx] += 1
            max_freq = max(max_freq, counts[idx])

            # 当需要替换的字符数 > k 时，窗口向右滑动一格收缩
            while (right - left + 1) - max_freq > k:
                counts[ord(s[left]) - ord('A')] -= 1
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：左右指针均只递增，时间复杂度严格为 $\mathcal{O}(N)$。
- **空间复杂度**：存储 26 个字母计数，空间复杂度为 $\mathcal{O}(1)$。

</div>

</div>
</details>

---

### 5. 变长与定长滑动窗口经典范式 (Variable & Fixed Sliding Window Patterns)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">滑窗 05</span>
  <span class="review-card-title">变长与定长滑动窗口经典范式 (Variable & Fixed Sliding Window Patterns)</span>
  <span class="review-card-tag">无重复字符最长子串 · 哈希跳跃加速 · 字符频次差分机</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码与跳跃优化</div>

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
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- 时间 $\mathcal{O}(N)$，空间 $\mathcal{O}(|\Sigma|)$。

</div>

</div>
</details>

---

## 模块三：回溯与数位贪心 (Backtracking & Digit Greedy)

### 6. 子集、排列与组合通用回溯范式 (Subsets, Permutations & Combinations)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">回溯 06</span>
  <span class="review-card-title">子集、排列与组合通用回溯范式 (Subsets, Permutations & Combinations)</span>
  <span class="review-card-tag">树形状态空间 · 树层剪枝去重 · 元素复用控制</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心模板代码</div>

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
                if i > start and nums[i] == nums[i - 1]: continue
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return res
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- 时间 $\mathcal{O}(N \cdot 2^N)$，空间 $\mathcal{O}(N)$。

</div>

</div>
</details>

---

### 7. 用指定数字集拼出严格小于 N 的最大数 (Largest Number Smaller than N from Digits A / Digit Greedy Backtracking)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">数位 07</span>
  <span class="review-card-title">用指定数字集拼出严格小于 N 的最大数 (Largest Number Smaller than N from Digits A / Digit Greedy Backtracking)</span>
  <span class="review-card-tag">数位回溯 · 贪心前缀匹配 · 降级后缀全最大填充 · 长度回退退化</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与工业场景需求</div>

给定一个正整数 $N$，以及一个包含十进制数字的候选列表 $A \subseteq \{0, 1, \dots, 9\}$。集合 $A$ 中的数字可以被**无限次重复使用**。

要求仅使用 $A$ 中的数字，构造出一个**严格小于 $N$ 的最大正整数**（若不存在返回 `-1` 或约定的空值）：

```python
def findLargestSmaller(N: int, A: List[int]) -> int: ...
```

**输入输出示例**：
- 输入：`N = 23415, A = [2, 4, 9]` $\implies$ 输出：`22999`
- 输入：`N = 222, A = [2]` $\implies$ 输出：`22`（同长度无法构造小于 $N$，退化为长度少一位的最大数）
- 输入：`N = 20, A = [2]` $\implies$ 输出：`2`

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与数位贪心回溯架构剖析</div>

#### 1. 数位贪心分支状态机
设 $N$ 的十进制字符串形式长度为 $L$。将数字集 $A$ 排序去重为有序数组。
从最高位（从左到右）逐位考察：
1. **分支 1（前缀尝试相等，贪心试探）**：
   - 若 $A$ 中存在与当前位 $N[i]$ 严格相等的数字，我们可以假定当前位取该数字，并递归搜索后续数位 $[i+1, L-1]$。
   - 若后续递归成功返回了一个合法解，则直接采纳该分支。
2. **分支 2（前缀降级，后缀全填最大值）**：
   - 若相等分支无法产生合法解，我们在当前位选取 $A$ 中**严格小于 $N[i]$ 的最大数字**。
   - **关键贪心不变式**：一旦在当前位选了严格小于 $N[i]$ 的数，无论后续各位怎么填，生成的整个数值都已**注定严格小于 $N$**！为了让总值最大，**后续所有剩余的数位必须无脑全部填充 $A$ 中的最大数字 $\max(A)$**！
3. **分支 3（无法在当前位构造，高位回溯）**：
   - 若当前位连比 $N[i]$ 小的数字都没有，说明此前的前缀相等决策错误，必须回溯到前面某一位，将其降级为更小的数字，其余后缀全填 $\max(A)$。
4. **分支 4（同长度完全无解，降维减少一位）**：
   - 若尝试了所有可能依然无法拼出长度为 $L$ 且严格小于 $N$ 的数（例如 $N=222, A=[3, 9]$），此时最优解必然是**长度为 $L - 1$ 且全部数位都填 $\max(A)$ 的数**（例如 $99$）。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List, Optional

class DigitConstructionSolution:

    @classmethod
    def findLargestSmaller(cls, N: int, A: List[int]) -> int:
        """
        利用候选数字集 A 拼出严格小于 N 的最大整数
        时间复杂度 O(L * |A|)，空间复杂度 O(L)
        """
        if N <= 0 or not A:
            return -1

        # 升序排序并去重
        digits = sorted(list(set(A)))
        max_d = digits[-1]
        s_N = str(N)
        L = len(s_N)

        res_digits: List[int] = []

        def backtrack(idx: int, is_less: bool) -> bool:
            """
            idx: 当前考察的数位索引
            is_less: 当前前缀是否已经严格小于 N 的前缀
            """
            if idx == L:
                # 只有在前缀已严格小于 N 时，同长度候选才是合法的
                return is_less

            cur_target = int(s_N[idx])

            if is_less:
                # 既然已经严格小于，后续所有数位全选最大数字
                res_digits.append(max_d)
                if backtrack(idx + 1, True):
                    return True
                res_digits.pop()
                return False

            # 尚未小于 N，优先尝试逆序可选数字（贪心追求最大）
            for d in reversed(digits):
                if d == cur_target:
                    res_digits.append(d)
                    if backtrack(idx + 1, False):
                        return True
                    res_digits.pop()
                elif d < cur_target:
                    # 选严格小于当前的数字，之后直接转为 is_less = True
                    res_digits.append(d)
                    if backtrack(idx + 1, True):
                        return True
                    res_digits.pop()

            return False

        # 尝试构造同长度严格小于 N 的数
        if backtrack(0, False):
            return int("".join(map(str, res_digits)))

        # 若同长度无解，退化构造长度为 L - 1 的最大数
        if L > 1:
            # 排除全 0 前导
            if max_d == 0:
                return -1
            return int(str(max_d) * (L - 1))

        return -1
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：数位长度为 $L$（对 64 位整数 $L \le 19$），每位最多尝试 $|A|$ 个数字（$|A| \le 10$），回溯调用树极度扁平，时间复杂度为严格 $\mathcal{O}(L \cdot |A|)$，耗时 $< 1\text{ms}$。
- **空间复杂度**：递归栈与临时数组占用 $\mathcal{O}(L)$。
- **高频避坑清单**：
  1. **遗漏长度退化分支**：当输入 $N = 222, A = [3]$ 时，长度为 3 的数最小也是 333（大于 222）。必须自动退化为长度 2 的最大数 33。
  2. **贪心全最大填充时机**：只要某一位选择了 $d < N[i]$，后续必须一律填 $\max(A)$，切忌继续与原数比对。

</div>

</div>
</details>

---

### 06. 无重叠区间贪心调度与最少移除数 (Non-overlapping Intervals via Earliest Deadline First)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GREEDY 06</span>
  <span class="review-card-title">无重叠区间贪心调度与最少移除数 (Non-overlapping Intervals via Earliest Deadline First)</span>
  <span class="review-card-tag">贪心区间调度 · 最早截止时间优先 · 端点排序 · O(N log N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class NonOverlappingIntervalsSolution:
    @classmethod
    def eraseOverlapIntervals(cls, intervals: List[List[int]]) -> int:
        """
        计算移除重叠区间所需的最少区间数，使得剩余区间互不重叠。
        接触端点 [a, b] 与 [b, c] 视为不重叠兼容。
        """
        if not intervals:
            return 0

        # 按右端点 (结束时间) 升序排序
        intervals.sort(key=lambda x: x[1])

        kept_count = 1
        prev_end = intervals[0][1]

        for i in range(1, len(intervals)):
            # 若当前区间起始时间 >= 上一个保留区间的结束时间，说明无冲突，果断保留
            if intervals[i][0] >= prev_end:
                kept_count += 1
                prev_end = intervals[i][1]

        # 最少移除数 = 总区间数 - 最大可保留不重叠区间数
        return len(intervals) - kept_count
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **问题对偶转化（Dual Problem）**：
  “移除最少重叠区间”等价于“在给定区间集合中选择**尽可能多且互不重叠的区间**”。设最多可保留 $K$ 个区间，则最小移除数为 $N - K$。
- **最早结束时间贪心律（Earliest Deadline First, EDF）**：
  优先选择结束时间最早的区间，能够为后续容纳更多区间留出最大的剩余时间跨度。任何结束时间更晚的同位替代选择，只会挤压后续空间，不可能优于 EDF 选择。
- **边界兼容语义**：
  根据题目约定，$start == end$ 属于可兼容相邻（如 $[1, 2]$ 与 $[2, 3]$ 不冲突），因此判断准则严格为 `intervals[i][0] >= prev_end`。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N \log N)$，主要耗费在区间按右端点排序；后续线性扫描为 $\mathcal{O}(N)$。
- **空间复杂度**：$\mathcal{O}(\log N)$（Timsort 排序所需栈空间）。

</div>

</div>
</details>

---

### 07. 时间戳键值存储与有序版本二分检索 (Time-Based Key-Value Store via Binary Search)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">BS 07</span>
  <span class="review-card-title">时间戳键值存储与有序版本二分检索 (Time-Based Key-Value Store via Binary Search)</span>
  <span class="review-card-tag">二分查找 (bisect) · 时间序列多版本存储 · 有序数组 · O(log N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from collections import defaultdict
import bisect
from typing import List, Tuple

class TimeMap:
    def __init__(self):
        # key -> list of (timestamp, value)
        self.store = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        """记录指定 key 在特定 timestamp 下的 value。假定 timestamp 严格递增到达。"""
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        """返回 timestamp_prev <= timestamp 的最大时间戳对应的值，若无则返回空串。"""
        if key not in self.store:
            return ""

        records = self.store[key]
        
        # 二分寻找首个时间戳 > timestamp 的位置
        # 由于记录元组为 (ts, val)，传入 (timestamp, chr(127)) 可保证严格右侧边界判定
        idx = bisect.bisect_right(records, (timestamp, chr(127)))

        # 若 idx == 0，说明所有记录的时间戳都严格大于目标 timestamp
        if idx == 0:
            return ""

        return records[idx - 1][1]
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **二分查找前驱节点（Predecessor Bisection）**：
  由于写入时间戳严格单调递增，`records` 数组天然具备保序性。查询 $\le timestamp$ 的最大版本等价于利用 `bisect_right` 找到第一个 $> timestamp$ 的位置，其前驱下标 `idx - 1` 即为所求。
- **乱序写入追问（Out-of-Order Writes）**：
  若写入时间戳并非严格递增，可在 `set` 时利用 `bisect.insort` 插入保证有序（插入 $\mathcal{O}(M)$），或在内部采用自平衡二叉搜索树（红黑树 / `SortedDict`），使插入与查询均保持在 $\mathcal{O}(\log M)$。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：`set` 操作 $\mathcal{O}(1)$（列表末尾追加）；`get` 操作 $\mathcal{O}(\log M)$，其中 $M$ 为该 key 下的历史版本数量。
- **空间复杂度**：$\mathcal{O}(N)$，存储全量键值历史版本。

</div>

</div>
</details>

---

### 08. 区间列表相交两指针交集扫描 (Interval List Intersections via Two-Pointer Scan)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">TP 08</span>
  <span class="review-card-title">区间列表相交两指针交集扫描 (Interval List Intersections via Two-Pointer Scan)</span>
  <span class="review-card-tag">双指针 · 闭区间相交判准 · 较早结束者平移 · O(M + N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class IntervalIntersectionSolution:
    @classmethod
    def intervalIntersection(
        cls, firstList: List[List[int]], secondList: List[List[int]]
    ) -> List[List[int]]:
        """
        求解两个互不重叠且已排序闭区间列表的公共交集列表。
        """
        i, j = 0, 0
        m, n = len(firstList), len(secondList)
        result = []

        while i < m and j < n:
            # 1. 计算当前双区间的可能交集区间 [start, end]
            start = max(firstList[i][0], secondList[j][0])
            end = min(firstList[i][1], secondList[j][1])

            # 2. 闭区间相交判准: start <= end
            if start <= end:
                result.append([start, end])

            # 3. 推进谁？淘汰结束时间较早的区间（因为后续区间绝不可能再与它产生重叠）
            if firstList[i][1] < secondList[j][1]:
                i += 1
            else:
                j += 1

        return result
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **双指针交集数学闭环**：
  任意两个闭区间 $[A_s, A_e]$ 与 $[B_s, B_e]$ 的交集必为 $[\max(A_s, B_s), \min(A_e, B_e)]$。若 $\max(A_s, B_s) \le \min(A_e, B_e)$，该闭区间非空且为有效交集。
- **单向淘汰推进机制**：
  若 $A_e < B_e$，由于列表中区间互不相交且单调递增，下一个区间 $A_{i+1}$ 的起始时间必然满足 $A_{i+1, s} > A_e$。因此区间 $A_i$ 绝无可能再与后续任何区间相交，可放心淘汰并推进指针 $i$。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(M + N)$，每个步骤至少有一个指针右移一位。
- **空间复杂度**：$\mathcal{O}(1)$ 额外辅助空间（不计输出结果）。

</div>

</div>
</details>

---

### 09. 至多 K 个不同字符的最长子串滑动窗口 (Longest Substring with At Most K Distinct Characters)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">SLIDE 09</span>
  <span class="review-card-title">至多 K 个不同字符的最长子串滑动窗口 (Longest Substring with At Most K Distinct Characters)</span>
  <span class="review-card-tag">可变滑动窗口 · 字符频次哈希 · 零频物理剔除 · O(N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from collections import defaultdict

class LongestSubstringKDistinctSolution:
    @classmethod
    def lengthOfLongestSubstringKDistinct(cls, s: str, k: int) -> int:
        """
        求解至多包含 k 个不同字符的最长子串长度。
        """
        if not s or k <= 0:
            return 0

        counts = defaultdict(int)
        left = 0
        max_len = 0

        for right, ch in enumerate(s):
            counts[ch] += 1

            # 若不同字符数超过 k，收缩左边界
            while len(counts) > k:
                left_ch = s[left]
                counts[left_ch] -= 1
                if counts[left_ch] == 0:
                    del counts[left_ch]  # 必须物理删除 key，否则 len(counts) 无法减少
                left += 1

            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len

        return max_len
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **哈希表物理键剔除（Key Eviction Invariant）**：
  判断当前窗口不同字符数的依据是 `len(counts)`。当某字符计数减为 0 时，必须执行 `del counts[ch]`。若仅留存计数值为 0 的键，`len(counts)` 不会减小，导致死循环或错误判断。
- **滑动窗口平摊线性度**：
  右指针 $right$ 遍历 $N$ 次，左指针 $left$ 最多前进 $N$ 次。每个字符进入与离开窗口各一次，双指针整体严格平摊 $\mathcal{O}(N)$。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$。
- **空间复杂度**：$\mathcal{O}(K)$，哈希表中最多保留 $K + 1$ 个字符映射。

</div>

</div>
</details>

---

### 10. 旋转有序数组二分查找与重复元素退化 (Search in Rotated Sorted Array: Distinct vs Duplicates)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">BS 10</span>
  <span class="review-card-title">旋转有序数组二分查找与重复元素退化 (Search in Rotated Sorted Array: Distinct vs Duplicates)</span>
  <span class="review-card-tag">对偶半区保序性 · 重复元素二义性 · 边界线性收缩 · O(log N) -> O(N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class SearchRotatedArraySolution:
    @classmethod
    def searchDistinct(cls, nums: List[int], target: int) -> int:
        """
        在无重复元素的旋转有序数组中检索目标值下标。
        严格 O(log N) 时间。
        """
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid

            # 判定前半区 [left, mid] 是否严格单调递增
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # 后半区 [mid, right] 严格单调递增
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1

    @classmethod
    def searchDuplicates(cls, nums: List[int], target: int) -> bool:
        """
        包含重复元素的旋转数组检索目标值是否存在。
        当三端相等时触发 O(1) 边界向内收缩，最坏退化至 O(N)。
        """
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True

            # 核心歧义处理：当 nums[left] == nums[mid] == nums[right] 时无法判定哪侧有序
            if nums[left] == nums[mid] == nums[right]:
                left += 1
                right -= 1
            elif nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return False
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **局部保序对偶性（Half-Sorted Partitioning）**：
  旋转后的数组被中点 $mid$ 切分为两半后，**至少有一半是严格单调有序的**。
  若 $nums[left] \le nums[mid]$，则左半区必定连续有序；否则右半区必连续有序。只需检查 $target$ 是否落在该有序半区的端点区间内，即可果断排除另一半。
- **重复元素的三态歧义与退化**：
  若允许重复元素，如 $[1, 0, 1, 1, 1]$，此时 $nums[left] == nums[mid] == nums[right] == 1$，无法分辨断崖拐点到底在左侧还是右侧。此时二分剪枝失效，只能安全地双向收缩边界 `left += 1, right -= 1`。在全相等数组（如 $[1, 1, 1, \dots, 1]$ 查 0）中最坏退化为 $\mathcal{O}(N)$。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **无重复版本**：时间复杂度严格为 $\mathcal{O}(\log N)$，空间复杂度 $\mathcal{O}(1)$。
- **含重复版本**：平均时间复杂度 $\mathcal{O}(\log N)$，最坏退化时间复杂度 $\mathcal{O}(N)$，空间复杂度 $\mathcal{O}(1)$。

</div>

</div>
</details>

---

### 11. 无人机中继贪心跳跃与步行距离最小化 (Drone Relay to Target via Greedy Forward Progression)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GREEDY 11</span>
  <span class="review-card-title">无人机中继贪心跳跃与步行距离最小化 (Drone Relay to Target via Greedy Forward Progression)</span>
  <span class="review-card-tag">贪心模拟 · 中继站二分/双指针 · 前向跳跃更新 · O(M log M + M)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class DroneRelaySolution:
    @classmethod
    def minWalkingDistance(cls, target: int, relay_points: List[int]) -> int:
        """
        从原点 0 出发前往目标点 target。
        到达中继站 r (需步行支付代价) 后，可将包裹放入无人机瞬间向前跳跃 10 单位。
        计算抵达或越过 target 时所累积支付的最小步行距离总和。
        """
        if target <= 0:
            return 0

        # 过滤负坐标并升序去重排序
        relays = sorted(set(p for p in relay_points if p >= 0))
        
        curr_pos = 0
        total_walk_cost = 0
        idx = 0
        m = len(relays)

        while curr_pos < target:
            # 1. 寻找当前位置之后最近的可选中继站
            while idx < m and relays[idx] < curr_pos:
                idx += 1

            # 2. 若前方已无可用中继站，或者当前直达 target 距离更近
            if idx >= m or relays[idx] >= target:
                total_walk_cost += (target - curr_pos)
                break

            next_relay = relays[idx]
            walk_to_relay = next_relay - curr_pos

            # 3. 边界对比：若直接走到 target 距离比走到中继站还要短，直接走完全程
            if (target - curr_pos) <= walk_to_relay:
                total_walk_cost += (target - curr_pos)
                break

            # 4. 贪心决策：步行至该中继站，并由无人机运载向前飞跃 10 个单位
            total_walk_cost += walk_to_relay
            curr_pos = next_relay + 10  # 无人机将位置向前传送 10 单位
            idx += 1

        return total_walk_cost
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **无后效性贪心决策律**：
  无人机跳跃步长为固定正值（10 单位），且不产生任何步行花费。对于任何中继站 $r$，只要其位于当前坐标前方且在 $target$ 左侧，步行至该站并借助无人机向前位移 10 单位，总是比徒步相同距离获得更远的有效位移（净赚至多 10 单位免费位移）。
- **终点与中继站位置相对性校验**：
  若下一中继站 $relays[idx] \ge target$，显然不可前往该中继站（否则会白白走过头，增加无效步行开销），此时最优策略是直接步行走完剩余的 $target - curr\_pos$ 距离。
- **目标点即达与越过判定**：
  若经无人机传送后 $curr\_pos \ge target$，循环立即终止，无需再走多余距离。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(M \log M)$，主要开销在中继站排序（若输入已排序则为严格 $\mathcal{O}(M)$ 双指针推进）。
- **空间复杂度**：$\mathcal{O}(M)$（排序去重存储）。

</div>

</div>
</details>

