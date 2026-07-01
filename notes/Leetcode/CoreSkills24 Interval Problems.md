# Interval Problems：排序、扫描线、堆

区间题看起来很多，但核心只有几种模式。

如果你看到输入长这样：

```text
intervals = [[start, end], ...]
```

第一反应应该是：

```text
先按 start 排序。
然后问：我要维护的是一个 merged interval、一个 active count，还是一个 candidate heap？
```

## 这类题怎么分类

| 题型 | 代表题 | 核心动作 | 常用结构 |
|---|---|---|---|
| 合并覆盖范围 | Merge Intervals | 排序后维护当前合并段 | sort + one pass |
| 插入新区间 | Insert Interval | 左边直接放，中间合并，右边追加 | three zones |
| 删除最少重叠 | Non-overlapping Intervals | 保留结束最早的区间 | greedy by end |
| 判断会议冲突 | Meeting Rooms | 相邻区间是否重叠 | sort by start |
| 会议室数量 | Meeting Rooms II | 最大同时在线区间数 | heap / sweep line / two pointers |
| query 找最小覆盖区间 | Minimum Interval to Include Each Query | 按 query 推进候选区间 | sort + min heap |

区间题的共同点：

```text
排序后，时间轴从左往右变成可控的。
```

排序不是为了“好看”，而是为了让你知道：

```text
当前区间之前的东西，要么已经输出，要么正在和 current 合并，要么已经不可能再影响未来。
```

## 区间关系先背熟

给两个区间：

```text
a = [s1, e1]
b = [s2, e2]
```

如果按 start 排序，通常有：

```text
s1 <= s2
```

那么只需要看：

```text
s2 <= e1
```

如果成立，两个区间重叠。

合并结果：

```text
[s1, max(e1, e2)]
```

如果不成立：

```text
s2 > e1
```

说明它们断开，前一个区间可以安全输出。

## Pattern 1：Merge Intervals

题目：

```text
给一堆区间，合并所有重叠区间。
```

例子：

```text
intervals = [[1,3], [2,6], [8,10], [15,18]]
answer = [[1,6], [8,10], [15,18]]
```

可视化：

```interval-merge-demo
```

核心逻辑：

```text
先按 start 排序。
维护当前正在合并的区间 current。

如果 next.start <= current.end:
    current.end = max(current.end, next.end)
否则:
    output current
    current = next
```

代码：

```python
from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda interval: interval[0])
        merged = []

        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        return merged
```

这里 `merged[-1]` 就是 current interval。

复杂度：

```text
Time:  O(n log n)
Space: O(n)  # output 不算额外空间时可以说 O(1)
```

## Pattern 2：Insert Interval

题目：

```text
intervals 已经按 start 排序，并且互不重叠。
插入 newInterval，保持结果仍然有序且互不重叠。
```

例子：

```text
intervals = [[1,2], [3,5], [6,7], [8,10], [12,16]]
newInterval = [4,8]
answer = [[1,2], [3,10], [12,16]]
```

可视化：

```interval-insert-demo
```

Insert Interval 不是普通 merge 的复杂版，它其实只有三段：

```text
1. 完全在 newInterval 左边：直接输出
2. 和 newInterval 重叠：不断扩张 newInterval
3. 完全在 newInterval 右边：先输出 newInterval，再输出剩下区间
```

代码：

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

        while i < n and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1

        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1

        result.append(newInterval)

        while i < n:
            result.append(intervals[i])
            i += 1

        return result
```

判断关系时注意：

```text
interval.end < new.start     -> 在左边
interval.start > new.end     -> 在右边
否则                         -> 重叠
```

复杂度：

```text
Time:  O(n)
Space: O(n)
```

因为输入已经有序，而且互不重叠，不需要重新排序。

## Pattern 3：Non-overlapping Intervals

题目：

```text
给一堆区间，删除最少区间，使剩下的互不重叠。
```

例子：

```text
intervals = [[1,2], [2,3], [3,4], [1,3]]
answer = 1
```

这题的贪心点是：

> 如果两个区间冲突，保留 end 更小的那个。

为什么？

因为结束越早，留给后面区间的空间越大。

代码：

```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda interval: interval[1])

        removed = 0
        prev_end = float("-inf")

        for start, end in intervals:
            if start >= prev_end:
                prev_end = end
            else:
                removed += 1

        return removed
```

这个写法等价于：

```text
最多能保留多少个不重叠区间？
答案 = n - keep
```

也可以按 start 排序写，但冲突时要更新：

```python
prev_end = min(prev_end, end)
```

面试里更推荐按 end 排序，因为贪心意图更清楚。

复杂度：

```text
Time:  O(n log n)
Space: O(1) or O(n) depending on sort implementation
```

## Pattern 4：Meeting Rooms

题目：

```text
给一堆会议时间，判断一个人能不能参加所有会议。
```

只需要检查是否存在重叠。

```python
from typing import List

class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort(key=lambda interval: interval[0])

        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i - 1][1]:
                return False

        return True
```

注意边界：

```text
[1, 2] 和 [2, 3] 不重叠
```

所以判断是：

```text
next.start < prev.end
```

而不是：

```text
next.start <= prev.end
```

## Pattern 5：Meeting Rooms II

题目：

```text
给一堆会议时间，求最少需要多少会议室。
```

本质：

```text
最大同时进行的会议数量
```

可视化：

```interval-rooms-demo
```

### 解法 A：min heap 存结束时间

按 start 排序。

heap 里存当前正在占用会议室的会议结束时间。

```text
如果最早结束的会议 end <= 当前 start:
    这个房间可以复用，pop
把当前会议 end push 进去
heap size 就是当前占用房间数
```

代码：

```python
from heapq import heappop, heappush
from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda interval: interval[0])
        rooms = []

        for start, end in intervals:
            if rooms and rooms[0] <= start:
                heappop(rooms)
            heappush(rooms, end)

        return len(rooms)
```

复杂度：

```text
Time:  O(n log n)
Space: O(n)
```

### 解法 B：start / end 双指针

把开始时间和结束时间拆开：

```python
from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        starts = sorted(interval[0] for interval in intervals)
        ends = sorted(interval[1] for interval in intervals)

        rooms = 0
        max_rooms = 0
        end_ptr = 0

        for start in starts:
            if start >= ends[end_ptr]:
                rooms -= 1
                end_ptr += 1

            rooms += 1
            max_rooms = max(max_rooms, rooms)

        return max_rooms
```

这版的含义：

```text
start < earliest_end  -> 新会议开始时没有房间释放，需要新房间
start >= earliest_end -> 有会议结束，房间可以复用
```

## Pattern 6：Sweep Line

会议室也可以写成扫描线。

每个区间 `[start, end]` 变成两个事件：

```text
start: +1
end:   -1
```

按时间排序后累加 active，最大 active 就是答案。

```python
from collections import defaultdict
from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        events = defaultdict(int)

        for start, end in intervals:
            events[start] += 1
            events[end] -= 1

        active = 0
        answer = 0

        for time in sorted(events):
            active += events[time]
            answer = max(answer, active)

        return answer
```

这个写法尤其适合：

- Car Pooling
- My Calendar III
- 统计最大重叠数量

边界细节：

```text
如果 end == start，通常要先处理 end，再处理 start。
```

用 `events[end] -= 1` 和同一时间点合并累加，可以自然处理会议室复用。

## Pattern 7：Minimum Interval to Include Each Query

题目：

```text
给 intervals 和 queries。
对每个 query，返回包含它的最短区间长度。
如果没有区间包含它，返回 -1。
```

例子：

```text
intervals = [[1,4], [2,4], [3,6], [4,4]]
queries = [2, 3, 4, 5]
answer = [3, 3, 1, 4]
```

可视化：

```interval-query-demo
```

这题的关键是不要对每个 query 暴力扫所有 interval。

正确做法：

```text
1. intervals 按 start 排序
2. queries 按值排序，但保留原始 query
3. 对当前 query q：
   - 把所有 start <= q 的区间加入 heap
   - heap 按区间长度排序
   - 弹出所有 end < q 的过期区间
   - heap 顶就是包含 q 的最短区间
```

代码：

```python
from heapq import heappop, heappush
from typing import List

class Solution:
    def minInterval(
        self,
        intervals: List[List[int]],
        queries: List[int],
    ) -> List[int]:
        intervals.sort(key=lambda interval: interval[0])
        result = {}
        heap = []
        i = 0

        for query in sorted(queries):
            while i < len(intervals) and intervals[i][0] <= query:
                start, end = intervals[i]
                length = end - start + 1
                heappush(heap, (length, end))
                i += 1

            while heap and heap[0][1] < query:
                heappop(heap)

            result[query] = heap[0][0] if heap else -1

        return [result[query] for query in queries]
```

为什么 heap 里只放 `(length, end)`？

因为加入 heap 时已经保证：

```text
start <= query
```

之后只需要检查：

```text
end >= query
```

如果 `end < query`，这个区间对当前和未来更大的 query 都没用了，可以弹掉。

复杂度：

```text
Time:  O((n + q) log n)
Space: O(n + q)
```

## 什么时候用哪种模板

### 只需要合并结果

用：

```text
sort by start + current merged interval
```

代表题：

- Merge Intervals
- Insert Interval

### 要删除最少冲突

用：

```text
sort by end + keep earliest ending interval
```

代表题：

- Non-overlapping Intervals

### 要判断有没有冲突

用：

```text
sort by start + compare adjacent
```

代表题：

- Meeting Rooms

### 要最少资源数 / 最大重叠数

用：

```text
heap of end times
or
start/end two pointers
or
sweep line events
```

代表题：

- Meeting Rooms II
- Car Pooling
- My Calendar III

### 每个 query 要找覆盖它的最优区间

用：

```text
sort intervals by start
sort queries
heap stores active candidate intervals
```

代表题：

- Minimum Interval to Include Each Query

## 常见坑

- 把 `[1,2]` 和 `[2,3]` 是否重叠搞混。Merge Intervals 通常视为重叠；Meeting Rooms 通常不冲突。
- Insert Interval 里忘记输入已经有序且互不重叠，写成重新排序的 `O(n log n)`。
- Non-overlapping Intervals 按 start 排序后，冲突时没有保留更小的 end。
- Meeting Rooms II 里复用房间条件写成 `rooms[0] < start`，导致 `[1,2]` 和 `[2,3]` 错误地需要两个房间。
- Sweep Line 同一时间点 start/end 顺序处理错误。
- Minimum Interval Query 忘记按原 query 顺序返回答案。
- Heap 里过期区间只弹一次；应该 `while heap and heap[0].end < query` 一直弹。

## 面试回答模板

区间题可以这样开场：

1. 我先把区间放到一条时间线上。
2. 如果题目关心覆盖范围，我按 start 排序并维护 current interval。
3. 如果题目关心最多不重叠，我按 end 排序，因为结束越早越不影响后面。
4. 如果题目关心同时存在多少区间，我用 heap / sweep line 维护 active intervals。
5. 如果题目有 query，我把 query 也排序，用 heap 维护当前 query 的候选区间。

一句话总结：

> 区间题不是背很多题，而是判断你在时间轴上维护的是 merged range、earliest end、active count，还是 candidate heap。
