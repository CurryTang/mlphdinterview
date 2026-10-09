# Interval Problems：排序、前沿演进与扫描线

区间问题的核心本质，是将一维数轴上的离散线段投影到单向推进的时间轴上。表面上看题目五花八门（合并、去重、插入、会议室、气球、离线查询），但若剥离表象，**所有区间问题底层仅存在 3 大核心模式**。

---

## 一、 区间题的底层共同点 (Universal Foundations)

无论是哪一类区间题，解题的第一步与判定基石完全一致：

### 1. 为什么永远统一按 `start` 升序排序？
- **降维打击**：区间 $[s, e]$ 是二维属性。如果不排序，任意两区间在数轴上的相对位置有 $3^2 = 9$ 种可能；
- **状态不变量**：一旦按 `start` 升序排序（$s_0 \le s_1 \le s_2 \dots$），时间轴从左向右单向推进。当我们处理到第 $i$ 个区间时，**所有 $j < i$ 的区间起点必然已在其左侧**。此时只需与维护的“活跃前沿”比较，复杂度直接从 $O(n^2)$ 降为 $O(n \log n)$。

### 2. 重叠与断开的充要条件
在 $s_1 \le s_2$ 的前提下：
- **发生重叠**：$s_2 \le e_1$（若端点相碰不算重叠，则为 $s_2 < e_1$）。两区间存在交集 $[s_2, \min(e_1, e_2)]$。
- **完全断开**：$s_2 > e_1$。说明区间 1 已不可能再与后续任何区间发生交集，前沿安全闭合。

---

## 二、 模式一：单指针前沿演进 (Single Frontier Scan)

> **统领题目**：
> - LC 56 合并区间 (Merge Intervals)
> - LC 57 插入区间 (Insert Interval)
> - LC 435 无重叠区间 (Non-overlapping Intervals)
> - LC 452 用最少数量的箭引爆气球 (Burst Balloons)
> - LC 252 会议室 (Meeting Rooms)

### 1. 核心共同点与高度对称性
所有“前沿演进”题目都维护一个正在推进的右端点 `prev_end`（或 `merged[-1]`）。
遍历后续新区间 `[start, end]` 时，重叠判定**恒为 `start <= prev_end`**。

不同题目的区别**仅仅在于重叠时对 `prev_end` 的一个动作**：

| 业务意图 | 对应题目 | 重叠动作 (`start <= prev_end`) | 物理含义 |
|---|---|---|---|
| **扩张合并** | LC 56, LC 57 | `prev_end = max(prev_end, end)` | 区间融合成大段，右边界向外延展 |
| **贪心去重** | LC 435, LC 452 | `prev_end = min(prev_end, end)` | 冲突必须弃一，保留结束早的以腾出右侧空间 |
| **冲突检测** | LC 252 | `return False` | 出现重叠即违章，直接退出 |

> **核心顿悟（消除按 start 还是按 end 排序的记忆撕裂）**：
> 传统题解常认为“合并区间按 start 排序，去重区间按 end 排序”。
> 实际上，**统一按 start 排序一套骨架通杀**：
> 当 $start_i < prev\_end$ 发生冲突时，两者不可兼得；保留 $end$ 更小的那一个（`min(prev_end, end)`）能为未来右侧的区间留下最大容纳空间，与经典按 end 排序在数学决策序列上**完全等价**！

### 2. 统一前沿骨架代码

```python
from typing import List, Union

class IntervalFrontierTemplate:
    @staticmethod
    def solve(intervals: List[List[int]], mode: str = "merge") -> Union[List[List[int]], int, bool]:
        """统一按 start 升序排序，单指针推进前沿"""
        if not intervals:
            return [] if mode != "detect" else True

        # 1. 统一按 start 升序排序
        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]
        removed = 0

        for start, end in intervals[1:]:
            prev_end = merged[-1][1]

            # 2. 统一重叠判定 (LC 56 端点接触算重叠 <=；LC 435/252 端点接触不算重叠 <)
            is_overlap = (start <= prev_end) if mode == "merge" else (start < prev_end)

            if is_overlap:
                if mode == "merge":
                    # 扩张：向右延展
                    merged[-1][1] = max(prev_end, end)
                elif mode == "erase":
                    # 去重：淘汰右延展更长的区间，保留结束早者
                    removed += 1
                    merged[-1][1] = min(prev_end, end)
                elif mode == "detect":
                    # 检测：重叠即不可行
                    return False
            else:
                # 安全闭合，推进新前沿
                merged.append([start, end])

        if mode == "merge":
            return merged
        if mode == "erase":
            return removed
        return True
```

### 3. 代表题实战

#### 变体 A：LC 56 合并区间 (Merge Intervals)

```interval-merge-demo
```

直接应用前沿扩张：

```python
from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        merged = []

        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        return merged
```

复杂度：
- 时间复杂度：$O(n \log n)$
- 空间复杂度：$O(n)$

#### 变体 B：LC 57 插入区间 (Insert Interval)

```interval-insert-demo
```

**共同点升华**：由于输入已排好序且互不重叠，不需要 $O(n \log n)$ 重新排序，直接用三段扫描法单遍 $O(n)$ 完成：

```python
from typing import List

class Solution:
    def insert(
        self,
        intervals: List[List[int]],
        newInterval: List[int],
    ) -> List[List[int]]:
        result = []
        i = 0
        n = len(intervals)

        # 1. 完全在 newInterval 左侧：直接追加
        while i < n and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1

        # 2. 与 newInterval 重叠：连续取 min/max 扩张
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        result.append(newInterval)

        # 3. 完全在 newInterval 右侧：直接追加
        while i < n:
            result.append(intervals[i])
            i += 1

        return result
```

复杂度：
- 时间复杂度：$O(n)$
- 空间复杂度：$O(n)$

#### 变体 C：LC 435 无重叠区间 (Non-overlapping Intervals)

删除最少区间使剩余互不重叠：

```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        # 解法一：统一按 start 排序，冲突时保留 end 更小者
        intervals.sort(key=lambda x: x[0])
        removed = 0
        prev_end = intervals[0][1]

        for start, end in intervals[1:]:
            if start < prev_end:
                removed += 1
                prev_end = min(prev_end, end)  # 贪心淘汰 end 更长者
            else:
                prev_end = end

        return removed
```

（经典按 `end` 排序亦可：`intervals.sort(key=lambda x: x[1])`，两者数学等价。）

#### 变体 D：LC 252 会议室 (Meeting Rooms)

判断一个人能否参加所有会议：

```python
from typing import List

class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort(key=lambda x: x[0])
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i - 1][1]:
                return False
        return True
```

---

## 三、 模式二：差分事件并发峰值 (Sweep Line & Concurrency Peak)

> **统领题目**：
> - LC 253 会议室 II (Meeting Rooms II)
> - LC 1094 拼车 (Car Pooling)
> - LC 732 我的日程安排 III (My Calendar III)

### 1. 核心共同点：区间的“借与还”
当题目询问“**同一时刻的最大重叠数量**”或“**最少需要多少并发资源**”时，我们不再关注区间的连续形态，而是将其解耦为时间轴上的离散事件：
- **进入事件**：`start` 时刻借出资源，标记为 `+val`（或 `+1`）；
- **离开事件**：`end` 时刻归还资源，标记为 `-val`（或 `-1`）。

**必须牢记的共同排序规则**：
- 按时间点升序排序；
- **同一时刻，释放事件（`-1`）必须排在占用事件（`+1`）之前**。这在数学上保证了端点相碰（如会议 `[1, 2]` 与 `[2, 3]`）可以立即复用同一个房间，无需任何条件特判。

### 2. 代表题与双视角实现：LC 253 会议室 II

```interval-rooms-demo
```

#### 视角 A：差分事件扫描线 (纯计数最优解)

```python
from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        events = []
        for start, end in intervals:
            events.append((start, 1))   # 占用
            events.append((end, -1))   # 释放

        # 核心：时间相同时，-1 排在 1 前面
        events.sort(key=lambda x: (x[0], x[1]))

        active = 0
        peak = 0
        for _, delta in events:
            active += delta
            peak = max(peak, active)

        return peak
```

复杂度：
- 时间复杂度：$O(n \log n)$
- 空间复杂度：$O(n)$

#### 视角 B：小顶堆维护活跃房间 (状态跟踪解)

当题目不仅需要知道最大房间数，还要求“给每个会议具体分配房间号”时，采用最小堆维护正在占用的房间结束时间：

```python
from heapq import heappop, heappush
from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[0])
        rooms = []  # 堆中存放正在进行的会议结束时间

        for start, end in intervals:
            # 若最早结束的会议已结束，房间可直接复用
            if rooms and rooms[0] <= start:
                heappop(rooms)
            heappush(rooms, end)

        return len(rooms)
```

---

## 四、 模式三：离线候选池小顶堆 (Offline Query & Active Heap)

> **统领题目**：
> - LC 1851 包含每个查询的最小区间 (Minimum Interval to Include Each Query)

### 1. 核心共同点：离线推进与双向过滤
当输入除区间外还包含一组查询点 $queries = [q_1, q_2, \dots]$，要求对每个 $q$ 找出覆盖它的“最优区间（如长度最小）”时：
- 暴力扫描为 $O(N \times Q)$，必然超时；
- **共同优化策略**：
  1. **离线双排序**：将 `intervals` 按 `start` 升序，`queries` 也按值升序排（保留原始索引）；
  2. **双指针推进候选池**：当查询推进到 $q$ 时，所有 $start \le q$ 的区间全部加入小顶堆（以区间长度为 key）；
  3. **堆顶惰性淘汰**：堆顶中所有 $end < q$ 的区间已经不可能覆盖 $q$ 及后续更大的查询，直接弹出；
  4. **堆顶即为最优解**：此时堆顶满足 $start \le q \le end$ 且长度最短。

### 2. 代表题实战：LC 1851

```interval-query-demo
```

```python
from heapq import heappop, heappush
from typing import List

class Solution:
    def minInterval(
        self,
        intervals: List[List[int]],
        queries: List[int],
    ) -> List[int]:
        intervals.sort(key=lambda x: x[0])
        sorted_queries = sorted((q, idx) for idx, q in enumerate(queries))

        ans = [-1] * len(queries)
        heap = []  # 元素为 (length, end)
        i, n = 0, len(intervals)

        for q, orig_idx in sorted_queries:
            # 1. 把所有覆盖 q 左侧的区间推进堆
            while i < n and intervals[i][0] <= q:
                start, end = intervals[i]
                heappush(heap, (end - start + 1, end))
                i += 1

            # 2. 惰性淘汰所有已经越过 q 右侧的失效区间
            while heap and heap[0][1] < q:
                heappop(heap)

            # 3. 堆顶即是覆盖 q 的最短区间
            if heap:
                ans[orig_idx] = heap[0][0]

        return ans
```

复杂度：
- 时间复杂度：$O((n + q) \log n + q \log q)$
- 空间复杂度：$O(n + q)$

---

## 五、 三大模式极简决策表与高频细节

### 1. 终极决策矩阵

| 核心维度 | 模式一：前沿演进 (Frontier) | 模式二：差分并发 (Sweep Line) | 模式三：离线候选池 (Active Heap) |
|---|---|---|---|
| **提问方式** | “合并重叠部分”、“移除最少区间使其不冲突”、“能否参加所有会议” | “最少需要多少房间”、“车上最高乘客数”、“最大并发事件数” | “对每个查询点，找出包含它的最短/最优区间” |
| **区间视角** | **连续线段**：关注区间整体与其覆盖边界 | **离散借还点**：拆解为 `+1` 与 `-1` 离散事件 | **动态候选集**：多约束下的最优堆选 |
| **排序依据** | 统一按 `start` 升序 | 按时间戳排序，同一时间 `-1` 先于 `+1` | `intervals` 与 `queries` 双双升序 |
| **核心操作** | 比较 `start <= prev_end`：<br>• 合并：`max(prev_end, end)`<br>• 去重：`min(prev_end, end)`<br>• 违规：`return False` | 前缀累加：<br>`active += delta`<br>`peak = max(peak, active)` | 堆顶双向夹逼：<br>• `start <= q` 入堆<br>• `end < q` 弹堆 |
| **维护结构** | 单个滚动变量 `prev_end` | 计数器 `active` 或 Min-Heap | Min-Heap `(length, end)` |
| **时空复杂度**| 时间 $O(N \log N)$，空间 $O(1)$ 或 $O(N)$ | 时间 $O(N \log N)$，空间 $O(N)$ | 时间 $O((N + Q) \log N)$，空间 $O(N + Q)$ |

### 2. 四大高频易错细节

1. **端点相交的严格性定义**：
   - 绝大多数题目（如 Merge Intervals），$[1, 2]$ 与 $[2, 3]$ 视为重叠（$s_2 \le e_1$）；
   - 但在会议室与排程中，前一场 2 点结束，后一场 2 点开始**不算冲突**（判定条件为严格小于 $s_2 < e_1$）。
2. **LC 57 Insert Interval 避免二次排序**：
   - 输入已然有序，务必利用左、中（合并）、右三段式遍历达到 $O(n)$，不要做多余的 $O(n \log n)$ 重新排序。
3. **差分事件的排序优先级**：
   - 当 `start == end` 时，必须确保 `-1`（释放）排在 `+1`（占用）之前。写成 tuple `(time, delta)` 时，`-1 < 1` 天然保证了释放优先。
4. **离线查询堆的淘汰条件**：
   - 堆顶检查必须写在 `while heap and heap[0][1] < q:` 循环中，而不是 `if` 单次判断，彻底清空所有历史过期区间。


## 六、 模式四：双指针区间交集扫描 (Two-Pointer Intersection)

### LC 986. 区间列表的交集 (Interval List Intersections)

#### 核心心智
两个排好序的区间列表 `firstList` 和 `secondList`：
- 当前两区间 `[s1, e1]` 和 `[s2, e2]` 的重叠部分为：
  $$start = \max(s1, s2), \quad end = \min(e1, e2)$$
- 若 $start \le end$，则存在有效交集 `[start, end]`；
- **指针移动法则**：谁的终点更小（`e1 < e2` 还是 `e2 < e1`），谁就不可能再与后续区间产生重叠，因此谁就向后移动一步！

```python
from typing import List

class Solution:
    def intervalIntersection(
        self,
        firstList: List[List[int]],
        secondList: List[List[int]]
    ) -> List[List[int]]:
        i = j = 0
        ans = []

        while i < len(firstList) and j < len(secondList):
            s1, e1 = firstList[i]
            s2, e2 = secondList[j]

            start = max(s1, s2)
            end = min(e1, e2)
            if start <= end:
                ans.append([start, end])

            # 淘汰终点较早者
            if e1 < e2:
                i += 1
            else:
                j += 1

        return ans
```
