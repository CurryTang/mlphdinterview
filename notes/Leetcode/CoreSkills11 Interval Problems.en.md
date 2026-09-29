# Interval Problems: Sorting, Sweep Line, Heaps

Interval problems look varied, but there are only a few core patterns.

If you see input like this:

```text
intervals = [[start, end], ...]
```

Your first reaction should be:

```text
Sort by start first.
Then ask: am I maintaining a merged interval, an active count, or a candidate heap?
```

## How to categorize this type of problem

| Problem type | Representative problem | Core action | Common structure |
|---|---|---|---|
| Merge covered ranges | Merge Intervals | After sorting, maintain the current merged segment | sort + one pass |
| Insert a new interval | Insert Interval | Append the left side directly, merge the middle, append the right side | three zones |
| Remove the fewest overlaps | Non-overlapping Intervals | Keep the interval that ends earliest | greedy by end |
| Check meeting conflicts | Meeting Rooms | Check whether adjacent intervals overlap | sort by start |
| Number of meeting rooms | Meeting Rooms II | Maximum number of simultaneously active intervals | heap / sweep line / two pointers |
| For each query, find the minimum covering interval | Minimum Interval to Include Each Query | Advance candidate intervals along with the query | sort + min heap |

What interval problems have in common:

```text
After sorting, the timeline becomes manageable from left to right.
```

The purpose of sorting is not to make things "look nicer", but to let you know:

```text
Everything before the current interval has either already been output, is currently being merged with current, or can no longer affect the future.
```

## Memorize interval relationships first

Given two intervals:

```text
a = [s1, e1]
b = [s2, e2]
```

If they are sorted by start, we usually have:

```text
s1 <= s2
```

Then you only need to check:

```text
s2 <= e1
```

If this holds, the two intervals overlap.

Merged result:

```text
[s1, max(e1, e2)]
```

If it does not hold:

```text
s2 > e1
```

That means they are disjoint, so the previous interval can be safely output.

## The Unified Interval Blueprint

Interval problems project 1D segments onto an ordered timeline.
All seemingly disparate interval variants converge mathematically to **one unified ordering + two core models**:

```text
                               ┌───────────────────────────┐
                               │ Universally sort by start │
                               │ intervals.sort(key=...)   │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
         [Model 1: Frontier Evolution]              [Model 2: Concurrency Peak]
         (LC 56, 57, 435, 452, 252)                  (LC 253, 1094, 732)
                       │                                           │
                       ▼                                           ▼
       Maintain active frontier prev = intervals[0]   Decompose into (time, delta) events
       Traverse subsequent [start, end]:              Sort by timestamp (-1 before +1)
                       │                                           │
         ┌─────────────┴─────────────┐                             ▼
         ▼                           ▼              Prefix accumulate: active += delta
      Overlap (start <= prev.end)   Disjoint (start > prev.end)  max_peak = max(max_peak, active)
         │                           │
         ├─ Merge (LC 56/57)         └─ Frontier safely sealed; output prev
         │  prev.end = max(prev.end, end)   prev = current interval
         ├─ Greedy Pruning (LC 435/452)
         │  prev.end = min(prev.end, end)
         └─ Conflict Check (LC 252)
            return False
```

### 1. Unified Frontier Evolution Template

This template governs all "range merging", "greedy pruning", and "overlap detection" problems. **Always sort universally by `start`**. When an overlap occurs, a single character difference (`max` vs. `min`) seamlessly switches between merging and pruning:

```python
from typing import List


class IntervalFrontierTemplate:
    @staticmethod
    def solve_frontier(intervals: List[List[int]], mode: str = "merge"):
        """Universally sort by start; maintain rolling frontier [prev_start, prev_end]

        mode supports:
        - "merge": LC 56 Merge Intervals / LC 57 Insert Interval
        - "erase": LC 435 Non-overlapping Intervals / LC 452 Burst Balloons
        - "detect": LC 252 Meeting Rooms (check for any collision)
        """
        if not intervals:
            return [] if mode != "detect" else True

        # 1. Always sort by start ascending
        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]  # merged[-1] acts as the active rolling frontier
        removed = 0

        for start, end in intervals[1:]:
            prev_end = merged[-1][1]

            # 2. Overlap condition: current start <= previous end
            if start <= prev_end:
                if mode == "merge":
                    # Mode A: Expand frontier to subsume overlap (LC 56)
                    merged[-1][1] = max(prev_end, end)
                elif mode == "erase":
                    # Mode B: Greedy pruning (LC 435); keep the one with earlier end to preserve future room
                    removed += 1
                    merged[-1][1] = min(prev_end, end)
                elif mode == "detect":
                    # Mode C: Collision detected (LC 252)
                    return False
            else:
                # 3. Disjoint: seal current frontier and advance to new interval
                merged.append([start, end])

        if mode == "merge":
            return merged
        if mode == "erase":
            return removed
        return True
```

> **First Principles Epiphany (Why sorting by start also solves LC 435):**
> A widespread misconception claims that "Merge Intervals must be sorted by start, but Non-overlapping Intervals must be sorted by end."
> In reality, **sorting universally by start**:
> - When two intervals collide, exactly one must be removed. To maximize the remaining room for future intervals, we **must keep the one that ends earlier**;
> - Thus, executing `merged[-1][1] = min(prev_end, end)` drops the interval with the later end;
> - This is mathematically **identical** to sorting by end, completely eliminating cognitive dissonance about switching sort keys.

---

### 2. Unified Sweep Line / Concurrency Template

This template governs all "maximum simultaneous occupancy", "minimum meeting rooms", and "passenger capacity peak" problems:

```python
from typing import List


class IntervalSweepLineTemplate:
    @staticmethod
    def min_meeting_rooms(intervals: List[List[int]]) -> int:
        """Event Delta Stream: finds maximum peak concurrency (LC 253 / LC 1094 / LC 732)"""
        events = []
        for start, end in intervals:
            # Start event (+1 capacity); End event (-1 capacity)
            events.append((start, 1))
            events.append((end, -1))

        # Tie-breaker rule: departure (-1) sorts before arrival (+1) at identical timestamps.
        # This naturally allows meetings [1, 2] and [2, 3] to reuse a room without special branching.
        events.sort(key=lambda x: (x[0], x[1]))

        active = 0
        peak = 0
        for _, delta in events:
            active += delta
            peak = max(peak, active)

        return peak
```

---

### 3. Unified Active Query Heap Template

This template governs range queries with point evaluations (LC 1851 Minimum Interval to Include Each Query):
* **Dual-pointer advance**: Sort both `intervals` (by start) and `queries` ascending;
* **Min-heap candidate pool**: Push all intervals with `start <= q` into the min-heap as `(length, end)`;
* **Eviction of stale candidates**: Pop all intervals from heap top whose `end < q`;
* **Heap top is optimal**: The remaining heap top is guaranteed to be the shortest interval covering query $q$.

## Pattern 1: Merge Intervals

Problem:

```text
Given a list of intervals, merge all overlapping intervals.
```

Example:

```text
intervals = [[1,3], [2,6], [8,10], [15,18]]
answer = [[1,6], [8,10], [15,18]]
```

Visualization:

```interval-merge-demo
```

Core logic:

```text
Sort by start first.
Maintain the current interval being merged, current.

If next.start <= current.end:
    current.end = max(current.end, next.end)
Otherwise:
    output current
    current = next
```

Code:

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

Here `merged[-1]` is the current interval.

Complexity:

```text
Time:  O(n log n)
Space: O(n)  # if output does not count as extra space, you can say O(1)
```

## Pattern 2: Insert Interval

Problem:

```text
intervals is already sorted by start and has no overlaps.
Insert newInterval while keeping the result sorted and non-overlapping.
```

Example:

```text
intervals = [[1,2], [3,5], [6,7], [8,10], [12,16]]
newInterval = [4,8]
answer = [[1,2], [3,10], [12,16]]
```

Visualization:

```interval-insert-demo
```

Insert Interval is not a more complicated version of ordinary merge. It really has only three sections:

```text
1. Completely to the left of newInterval: output directly
2. Overlapping with newInterval: keep expanding newInterval
3. Completely to the right of newInterval: output newInterval first, then output the remaining intervals
```

Code:

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

When checking the relationship, note:

```text
interval.end < new.start     -> on the left
interval.start > new.end     -> on the right
otherwise                    -> overlapping
```

Complexity:

```text
Time:  O(n)
Space: O(n)
```

Because the input is already sorted and non-overlapping, there is no need to sort again.

## Pattern 3: Non-overlapping Intervals

Problem:

```text
Given a list of intervals, remove the minimum number of intervals so the rest are non-overlapping.
```

Example:

```text
intervals = [[1,2], [2,3], [3,4], [1,3]]
answer = 1
```

The greedy insight in this problem is:

> If two intervals conflict, keep the one with the smaller end.

Why?

Because the earlier it ends, the more space it leaves for later intervals.

### Approach 1: Unified Frontier Evolution (Sort by Start, Symmetrical to Merge Intervals)

Sorting key remains identical to interval merging. When an overlap conflict occurs, merging uses `max` to expand, while non-overlapping uses `min` to greedily retain the interval ending earlier and prune the one ending later:

```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        # Universally sort by start
        intervals.sort(key=lambda x: x[0])

        removed = 0
        prev_end = intervals[0][1]

        for start, end in intervals[1:]:
            if start < prev_end:
                # Overlap conflict: must remove one
                # Greedy choice: retain interval with smaller end to maximize remaining room
                removed += 1
                prev_end = min(prev_end, end)
            else:
                # Safe: advance frontier
                prev_end = end

        return removed
```

### Approach 2: Classic Greedy (Sort by End)

Count the maximum number of non-overlapping intervals kept, where the answer is $n - \text{keep}$:

```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # Sort by end ascending
        intervals.sort(key=lambda x: x[1])

        removed = 0
        prev_end = float("-inf")

        for start, end in intervals:
            if start >= prev_end:
                prev_end = end
            else:
                removed += 1

        return removed
```

Both approaches are mathematically and asymptotically equivalent ($O(n \log n)$). Approach 1 has the distinct advantage of sharing the exact same `start` sorting key and frontier comparison as Merge Intervals, eliminating context switching.

Complexity:

```text
Time:  O(n log n)
Space: O(1) or O(n) depending on sort implementation
```

## Pattern 4: Meeting Rooms

Problem:

```text
Given a list of meeting times, determine whether one person can attend all meetings.
```

You only need to check whether any overlap exists.

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

Note the boundary:

```text
[1, 2] and [2, 3] do not overlap
```

So the condition is:

```text
next.start < prev.end
```

not:

```text
next.start <= prev.end
```

## Pattern 5: Meeting Rooms II

Problem:

```text
Given a list of meeting times, return the minimum number of meeting rooms required.
```

In essence:

```text
the maximum number of meetings happening at the same time
```

Visualization:

```interval-rooms-demo
```

### Solution A: min heap storing end times

Sort by start.

The heap stores the end times of meetings currently occupying rooms.

```text
If the earliest-ending meeting has end <= current start:
    that room can be reused, so pop
Push the current meeting's end
heap size is the number of rooms currently in use
```

Code:

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

Complexity:

```text
Time:  O(n log n)
Space: O(n)
```

### Solution B: start / end two pointers

Split start times and end times apart:

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

What this version means:

```text
start < earliest_end  -> no room has been freed when the new meeting starts, so a new room is needed
start >= earliest_end -> a meeting has ended, so a room can be reused
```

## Pattern 6: Sweep Line

Meeting rooms can also be written as a sweep line problem.

Each interval `[start, end]` becomes two events:

```text
start: +1
end:   -1
```

After sorting by time, accumulate active, and the maximum active is the answer.

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

This approach is especially suitable for:

- Car Pooling
- My Calendar III
- counting the maximum number of overlaps

Boundary detail:

```text
If end == start, you usually need to process end before start.
```

Using `events[end] -= 1` and combining counts at the same time point naturally handles room reuse.

## Pattern 7: Minimum Interval to Include Each Query

Problem:

```text
Given intervals and queries.
For each query, return the length of the shortest interval that contains it.
If no interval contains it, return -1.
```

Example:

```text
intervals = [[1,4], [2,4], [3,6], [4,4]]
queries = [2, 3, 4, 5]
answer = [3, 3, 1, 4]
```

Visualization:

```interval-query-demo
```

The key to this problem is not to brute-force scan all intervals for every query.

The correct approach:

```text
1. Sort intervals by start
2. Sort queries by value, but keep the original query
3. For the current query q:
   - add all intervals with start <= q into the heap
   - sort the heap by interval length
   - pop all expired intervals with end < q
   - the top of the heap is the shortest interval containing q
```

Code:

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

Why does the heap only store `(length, end)`?

Because when an interval is added to the heap, we have already guaranteed:

```text
start <= query
```

After that, we only need to check:

```text
end >= query
```

If `end < query`, that interval is no longer useful for the current query or for any larger future query, so it can be popped.

Complexity:

```text
Time:  O((n + q) log n)
Space: O(n + q)
```

## Which template to use when

### You only need the merged result

Use:

```text
sort by start + current merged interval
```

Representative problems:

- Merge Intervals
- Insert Interval

### You need to remove the fewest conflicts / greedy pruning

Use:

```text
sort by start + on conflict min(prev_end, end) (Unified Frontier)
or
sort by end + keep earliest ending interval (Classic Greedy)
```

Representative problems:

- Non-overlapping Intervals
- Minimum Number of Arrows to Burst Balloons

### You need to determine whether there is a conflict

Use:

```text
sort by start + compare adjacent
```

Representative problems:

- Meeting Rooms

### You need the minimum number of resources / maximum number of overlaps

Use:

```text
heap of end times
or
start/end two pointers
or
sweep line events
```

Representative problems:

- Meeting Rooms II
- Car Pooling
- My Calendar III

### For each query, you need the best interval covering it

Use:

```text
sort intervals by start
sort queries
heap stores active candidate intervals
```

Representative problems:

- Minimum Interval to Include Each Query

## Common pitfalls

- Confusing whether `[1,2]` and `[2,3]` overlap. In Merge Intervals they are usually treated as overlapping; in Meeting Rooms they usually are not a conflict.
- In Insert Interval, forgetting that the input is already sorted and non-overlapping, and rewriting it as an `O(n log n)` re-sort.
- In Non-overlapping Intervals, sorting by start but not keeping the smaller end when there is a conflict.
- In Meeting Rooms II, writing the room reuse condition as `rooms[0] < start`, which incorrectly makes `[1,2]` and `[2,3]` require two rooms.
- In Sweep Line, processing start/end in the wrong order at the same time point.
- In Minimum Interval Query, forgetting to return answers in the original query order.
- In the heap, popping an expired interval only once; it should keep popping with `while heap and heap[0].end < query`.

## Interview answer template

You can start answering interval problems like this:

1. First, I place the intervals on a timeline.
2. If the problem cares about covered ranges, I sort by start and maintain the current interval.
3. If the problem cares about maximizing non-overlap / removing fewest intervals, I sort by start and on conflict take min(prev_end, end) to retain the earliest ending interval (or sort by end with classic greedy).
4. If the problem cares about how many intervals exist at the same time, I use a heap / sweep line to maintain active intervals.
5. If the problem has queries, I sort the queries too and use a heap to maintain the candidate intervals for the current query.

One-sentence summary:

> Interval problems are not about memorizing many problems, but about deciding whether you are maintaining a merged range, earliest end, active count, or candidate heap on the timeline.
