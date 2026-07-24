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

Code:

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

This formulation is equivalent to:

```text
What is the maximum number of non-overlapping intervals we can keep?
answer = n - keep
```

You can also write it by sorting on start, but when there is a conflict you need to update:

```python
prev_end = min(prev_end, end)
```

In interviews, sorting by end is usually preferred because the greedy intention is clearer.

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

### You need to remove the fewest conflicts

Use:

```text
sort by end + keep earliest ending interval
```

Representative problems:

- Non-overlapping Intervals

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
3. If the problem cares about maximizing non-overlap, I sort by end, because the earlier an interval ends, the less it affects what comes after.
4. If the problem cares about how many intervals exist at the same time, I use a heap / sweep line to maintain active intervals.
5. If the problem has queries, I sort the queries too and use a heap to maintain the candidate intervals for the current query.

One-sentence summary:

> Interval problems are not about memorizing many problems, but about deciding whether you are maintaining a merged range, earliest end, active count, or candidate heap on the timeline.
