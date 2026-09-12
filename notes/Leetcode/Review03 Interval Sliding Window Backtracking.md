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
