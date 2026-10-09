# Interval Problems: Sorting, Frontier Evolution, and Sweep Line

At their mathematical core, all interval problems project discrete line segments onto a unidirectionally advancing timeline. Although the problem statements vary widely—merging, greedy pruning, insertion, meeting room allocation, bursting balloons, or offline queries—**all interval problems boil down to three foundational patterns**.

---

## 1. Universal Foundations of Interval Problems

Regardless of the problem variation, the initial premise and overlap mechanics remain identical:

### 1. Why Universally Sort by `start` Ascending?
- **Dimensionality Reduction**: An interval $[s, e]$ is a 2D entity. Without sorting, the relative spatial positioning between any two intervals has $3^2 = 9$ possibilities.
- **State Invariant**: Once sorted by `start` ascending ($s_0 \le s_1 \le s_2 \dots$), the timeline advances unidirectionally from left to right. When processing the $i$-th interval, **all previous intervals $j < i$ are guaranteed to have their starts at or to the left of $s_i$**. Comparing solely against an active frontier collapses complexity from $O(n^2)$ to $O(n \log n)$.

### 2. Necessary and Sufficient Conditions for Overlap
Given $s_1 \le s_2$:
- **Overlap**: $s_2 \le e_1$ (or $s_2 < e_1$ if touching endpoints do not conflict). Their intersection is $[s_2, \min(e_1, e_2)]$.
- **Disjoint**: $s_2 > e_1$. Interval 1 can never intersect with any subsequent interval; its frontier is safely finalized.

---

## 2. Pattern 1: Single Frontier Scan

> **Governs**:
> - LC 56 Merge Intervals
> - LC 57 Insert Interval
> - LC 435 Non-overlapping Intervals
> - LC 452 Minimum Number of Arrows to Burst Balloons
> - LC 252 Meeting Rooms

### 1. Core Commonalities and Symmetrical Operations
All frontier problems maintain a rolling right boundary `prev_end` (or `merged[-1]`).
When traversing each incoming interval `[start, end]`, the overlap test is **always `start <= prev_end`**.

The only variation is **how `prev_end` is updated during an overlap**:

| Objective | Representative Problems | Overlap Action (`start <= prev_end`) | Physical Meaning |
|---|---|---|---|
| **Expand / Merge** | LC 56, LC 57 | `prev_end = max(prev_end, end)` | Fuse intervals; right boundary extends outward |
| **Greedy Pruning** | LC 435, LC 452 | `prev_end = min(prev_end, end)` | Conflict requires dropping one; retain the one ending earlier to maximize remaining room |
| **Conflict Check** | LC 252 | `return False` | Any overlap is a violation; terminate immediately |

> **Key Insight (Eliminating the sort-by-start vs. sort-by-end split)**:
> Common lore suggests sorting by `start` for merging and sorting by `end` for non-overlapping intervals.
> In reality, **sorting universally by `start` handles both symmetrically**:
> When $start_i < prev\_end$, the two intervals collide and one must be removed. Retaining the one with the smaller end (`min(prev_end, end)`) leaves maximum room for all future intervals, producing a decision sequence mathematically identical to sorting by `end`.

### 2. Unified Frontier Skeleton Code

```python
from typing import List, Union

class IntervalFrontierTemplate:
    @staticmethod
    def solve(intervals: List[List[int]], mode: str = "merge") -> Union[List[List[int]], int, bool]:
        """Universally sort by start ascending and advance the frontier."""
        if not intervals:
            return [] if mode != "detect" else True

        # 1. Sort by start ascending
        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]
        removed = 0

        for start, end in intervals[1:]:
            prev_end = merged[-1][1]

            # 2. Unified overlap condition (touching counts for LC 56 <=; does not count for LC 435/252 <)
            is_overlap = (start <= prev_end) if mode == "merge" else (start < prev_end)

            if is_overlap:
                if mode == "merge":
                    # Expand: stretch frontier rightward
                    merged[-1][1] = max(prev_end, end)
                elif mode == "erase":
                    # Prune: greedily keep the one ending earlier
                    removed += 1
                    merged[-1][1] = min(prev_end, end)
                elif mode == "detect":
                    # Detect: overlap is invalid
                    return False
            else:
                # Safe: seal frontier and advance
                merged.append([start, end])

        if mode == "merge":
            return merged
        if mode == "erase":
            return removed
        return True
```

### 3. Representative Problem Walkthroughs

#### Variant A: LC 56 Merge Intervals

```interval-merge-demo
```

Direct application of frontier expansion:

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

Complexity:
- Time: $O(n \log n)$
- Space: $O(n)$

#### Variant B: LC 57 Insert Interval

```interval-insert-demo
```

**Commonality Insight**: Because the input is already sorted and disjoint, an $O(n \log n)$ full re-sort is unnecessary. A 3-zone scan achieves $O(n)$ in one pass:

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

        # 1. Completely to the left of newInterval
        while i < n and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1

        # 2. Overlapping with newInterval: expand bounds
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        result.append(newInterval)

        # 3. Completely to the right of newInterval
        while i < n:
            result.append(intervals[i])
            i += 1

        return result
```

Complexity:
- Time: $O(n)$
- Space: $O(n)$

#### Variant C: LC 435 Non-overlapping Intervals

Remove minimum intervals to make the remainder disjoint:

```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        # Approach 1: Unified sort by start; keep smaller end on collision
        intervals.sort(key=lambda x: x[0])
        removed = 0
        prev_end = intervals[0][1]

        for start, end in intervals[1:]:
            if start < prev_end:
                removed += 1
                prev_end = min(prev_end, end)  # greedily drop interval extending further right
            else:
                prev_end = end

        return removed
```

(The classic sort by end `intervals.sort(key=lambda x: x[1])` is mathematically equivalent.)

#### Variant D: LC 252 Meeting Rooms

Determine whether a person can attend all meetings:

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

## 3. Pattern 2: Sweep Line & Concurrency Peak

> **Governs**:
> - LC 253 Meeting Rooms II
> - LC 1094 Car Pooling
> - LC 732 My Calendar III

### 1. Core Commonality: Resource Borrowing and Returning
When asked for the **maximum concurrent overlap** or **minimum resource capacity** at any single moment, intervals are decomposed into discrete timestamped events:
- **Arrival**: `start` timestamp borrows a resource (`+val` or `+1`);
- **Departure**: `end` timestamp releases a resource (`-val` or `-1`).

**Critical Event Ordering Rule**:
- Sort primarily by timestamp ascending;
- **At the exact same timestamp, release events (`-1`) must precede arrival events (`+1`)**. This mathematically guarantees that abutting intervals (e.g., `[1, 2]` and `[2, 3]`) seamlessly reuse the same resource without edge-case branches.

### 2. Representative Problem & Dual Perspectives: LC 253 Meeting Rooms II

```interval-rooms-demo
```

#### Perspective A: Difference Event Sweep Line (Pure Counting)

```python
from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        events = []
        for start, end in intervals:
            events.append((start, 1))   # acquire
            events.append((end, -1))   # release

        # Crucial: -1 precedes 1 at the same timestamp
        events.sort(key=lambda x: (x[0], x[1]))

        active = 0
        peak = 0
        for _, delta in events:
            active += delta
            peak = max(peak, active)

        return peak
```

Complexity:
- Time: $O(n \log n)$
- Space: $O(n)$

#### Perspective B: Min-Heap for Active Rooms (State Tracking)

When the problem requires assigning specific room IDs or tracking active sessions dynamically, a min-heap tracks ongoing meeting end times:

```python
from heapq import heappop, heappush
from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[0])
        rooms = []  # stores end times of active meetings

        for start, end in intervals:
            # If the earliest ending meeting has finished, reuse the room
            if rooms and rooms[0] <= start:
                heappop(rooms)
            heappush(rooms, end)

        return len(rooms)
```

---

## 4. Pattern 3: Offline Query & Active Heap

> **Governs**:
> - LC 1851 Minimum Interval to Include Each Query

### 1. Core Commonality: Offline Progression & Dual Filtering
When the problem provides intervals alongside an external set of query points $queries = [q_1, q_2, \dots]$:
- Brute-force scanning takes $O(N \times Q)$;
- **Common Optimization Strategy**:
  1. **Dual Offline Sorting**: Sort `intervals` by `start` ascending, and sort `queries` ascending (retaining original indices).
  2. **Advance Candidates into Heap**: As query $q$ advances rightward, push all intervals with $start \le q$ into a min-heap keyed by interval length.
  3. **Lazy Eviction of Stale Intervals**: Pop any interval from the heap top where $end < q$, as it can never cover $q$ or any future query.
  4. **Heap Top is Optimal**: The heap top now satisfies $start \le q \le end$ with minimal length.

### 2. Representative Problem: LC 1851

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
        heap = []  # element: (length, end)
        i, n = 0, len(intervals)

        for q, orig_idx in sorted_queries:
            # 1. Push all candidate intervals whose start <= q into heap
            while i < n and intervals[i][0] <= q:
                start, end = intervals[i]
                heappush(heap, (end - start + 1, end))
                i += 1

            # 2. Lazily evict expired intervals whose end < q
            while heap and heap[0][1] < q:
                heappop(heap)

            # 3. Heap top is the shortest valid interval covering q
            if heap:
                ans[orig_idx] = heap[0][0]

        return ans
```

Complexity:
- Time: $O((n + q) \log n + q \log q)$
- Space: $O(n + q)$

---

## 5. Decision Matrix and High-Frequency Pitfalls

### 1. Unified Decision Matrix

| Dimension | Pattern 1: Frontier Scan | Pattern 2: Sweep Line & Concurrency | Pattern 3: Offline Query & Heap |
|---|---|---|---|
| **Query Goal** | Merge overlaps, remove fewest to eliminate conflict, check schedule feasibility | Minimum rooms required, peak passengers, max concurrent events | Find shortest / optimal interval covering each query point |
| **Interval View** | **Continuous segments**: focus on range coverage | **Discrete delta events**: decomposed into `+1` and `-1` | **Dynamic candidates**: heap-ranked under dual constraints |
| **Sort Key** | Universally by `start` ascending | By timestamp; `-1` precedes `+1` on tie | Both `intervals` and `queries` ascending |
| **Core Action** | Compare `start <= prev_end`:<br>• Merge: `max(prev_end, end)`<br>• Prune: `min(prev_end, end)`<br>• Feasibility: `return False` | Prefix sum:<br>`active += delta`<br>`peak = max(peak, active)` | Dual clamp:<br>• `start <= q` push<br>• `end < q` pop |
| **State Structure** | Rolling variable `prev_end` | Counter `active` or Min-Heap | Min-Heap `(length, end)` |
| **Complexity** | Time $O(N \log N)$, Space $O(1)$ or $O(N)$ | Time $O(N \log N)$, Space $O(N)$ | Time $O((N + Q) \log N)$, Space $O(N + Q)$ |

### 2. Four Critical Pitfalls

1. **Strictness of Endpoint Overlap**:
   - For Merge Intervals, $[1, 2]$ and $[2, 3]$ are overlapping ($s_2 \le e_1$).
   - For Meeting Rooms, ending at 2 and starting at 2 is **not a conflict** ($s_2 < e_1$).
2. **Avoid Re-sorting in LC 57 Insert Interval**:
   - The input is already sorted and disjoint. Use the 3-zone scan for $O(n)$ time.
3. **Event Tie-Breaking Order**:
   - When `start == end`, release events (`-1`) must precede arrival events (`+1`). Storing tuples as `(time, delta)` automatically ensures this because `-1 < 1`.
4. **While Loop for Heap Eviction**:
   - Always pop stale intervals with `while heap and heap[0][1] < q:`, never with a single `if`.


## VI. Pattern 4: Two-Pointer Interval Intersections

### LC 986. Interval List Intersections

#### Core Mental Model
Two sorted, disjoint interval lists.
- Overlap of `[s1, e1]` and `[s2, e2]`:
  $$start = \max(s1, s2), \quad end = \min(e1, e2)$$
- If $start \le end$, record `[start, end]`.
- Advance pointer of the interval with the smaller end point (`e1 < e2` advances `i`, otherwise `j`).

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
            start, end = max(s1, s2), min(e1, e2)
            if start <= end:
                ans.append([start, end])
            if e1 < e2:
                i += 1
            else:
                j += 1
        return ans
```
