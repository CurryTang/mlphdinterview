# Review Flashcards: Intervals, Sliding Window & Backtracking

This note is the third volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Intervals & Sweep Line**, **Two Pointers & Sliding Window**, and **Backtracking & Combinatorial Search** with production-grade implementations and asymptotic complexity breakdowns.

---

## Module 1: Intervals & Sweep Line

### 1. Merge Intervals & Interval Topology Variants

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Interval 01</span>
  <span class="review-card-title">Merge Intervals & Interval Topology Variants</span>
  <span class="review-card-tag">Closed Interval Semantics · Presorted Acceleration · Nested Absorption · Insert Interval · Non-Overlapping Greedy · Meeting Rooms Sweep Line</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Master Variant Matrix</div>

Given an array of intervals where each interval is $[s, e]$, merge all overlapping intervals and return an array of the non-overlapping intervals:

```python
def merge(intervals: List[List[int]]) -> List[List[int]]: ...
```

| Variant | Variant Name | Core Tokens / Features | Algorithmic Strategy |
|---|---|---|---|
| **Variant 1** | **Classic Closed Interval Merge (LC 56)** | Input consists of unsorted closed intervals $[s, e]$; touching endpoints (e.g. $[1, 4]$ and $[4, 5]$) overlap | **Sort by start time** + dynamically maintain `merged[-1][1] = max(merged[-1][1], cur[1])`. |
| **Variant 2** | **Presorted Acceleration** | Upstream data stream guarantees intervals arrive sorted by `start` time | **Bypasses $O(N \log N)$ sorting**, enabling a strictly linear $O(N)$ single-pass sweep with zero added latency. |
| **Variant 3** | **Nested Absorption** | Completely contained intervals (e.g. $[1, 10]$ and $[2, 5]$) | Handled seamlessly by `max` on the right boundary, preserving $[1, 10]$. |
| **Variant 4** | **Insert Interval (LC 57)** | Insert a new interval into an already sorted, non-overlapping list | **Three-phase linear sweep**: strictly left $\to$ expand merged overlapping cluster $\to$ strictly right, achieving $O(N)$ time and $O(1)$ auxiliary space. |
| **Variant 5** | **Non-Overlapping Intervals (LC 435)** | Minimum number of intervals to remove to make the rest non-overlapping | **Greedy sorting by end time**: retain the interval that finishes earliest to leave maximum headroom for subsequent intervals. |
| **Variant 6** | **Meeting Rooms II (LC 253)** | Minimum conference rooms required to hold all scheduled meetings | **Sweep Line / Difference events** ($+1$ at `start`, $-1$ at `end`) or a **min-heap storing active meeting end times**. |
| **Variant 7** | **Open vs Closed Semantics** | Half-open intervals $[s, e)$ (e.g. socket port ranges or timeslices) | Touching endpoints no longer overlap ($[1, 4)$ and $[4, 5)$ are disjoint); overlap condition switches from `cur[0] <= prev[1]` to strict `cur[0] < prev[1]`. |

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

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
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: `merge` takes $\mathcal{O}(N \log N)$ (or $\mathcal{O}(N)$ if presorted), `insert` takes $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(1)$ auxiliary space beyond output.

</div>

</div>
</details>

---

## Module 2: Two Pointers & Sliding Window

### 2. Pythagorean Triplet & Multiplicity 2-Pointer Search

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Two Pointers 02</span>
  <span class="review-card-title">Pythagorean Triplet & Multiplicity 2-Pointer Search</span>
  <span class="review-card-tag">Square Mapping · Multiplicity Preservation · 3SUM Reduction · Two Pointers Convergence · O(N^2) Optimality Proof</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Multiplicity Test Matrix</div>

Given an integer array `nums`, determine whether it contains three elements $a, b, c$ satisfying:
$$a^2 + b^2 = c^2$$

**Interview Constraints & Deep Invariants**:
1. **Negative inputs allowed**: Squaring naturally normalizes signs (e.g. $(-3)^2 + 4^2 = 5^2$).
2. **Respect element multiplicity**: Repeated use of the same numeric value is valid **only when the array contains enough copies**. A single array position cannot supply multiple slots.
3. **Optimality discussion**: Can the general solution improve below $\mathcal{O}(N^2)$ without an additional value-domain bound?

**Test Case Matrix**:

| Test Case | Return Value | Critical Verification Rationale |
|---|---|---|
| `[0, 1, -2, 3, 4, 5]` | `True` | $3^2 + 4^2 = 5^2$ ($9 + 16 = 25$) |
| `[0, 0, 0]` | `True` | 3 distinct zero positions satisfy $0^2 + 0^2 = 0^2$ |
| `[0, 1, -1]` | `True` | Satisfies $0^2 + (-1)^2 = 1^2$ ($0 + 1 = 1$) |
| `[0]` | `False` | A single element cannot fill all three positions |
| `[0, 2]` | `False` | Fewer than 3 elements |

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

#### 1. Why a Hash Set Fails
A simple set of squared values loses multiplicity information. For input `[0, 5]`, checking $0^2 + 0^2 = 0 \in \text{set}$ produces a false positive `True`, because it reuses the single `0` three times.

#### 2. Square Mapping + Sort + Two-Pointer Squeeze
1. **Map to squares**: $x \gets x^2$, transforming all entries into non-negative values.
2. **Sort**: Ascending sort in $\mathcal{O}(N \log N)$.
3. **Iterate candidate hypotenuse $c^2$ from back**:
   - For $k = N - 1$ down to $2$:
   - Run two pointers $i = 0, j = k - 1$:
     - If $squares[i] + squares[j] == squares[k]$: Since $i < j < k$, they correspond to three distinct original array positions, preserving multiplicity! Return `True`.
     - If sum $< target$: $i \gets i + 1$.
     - If sum $> target$: $j \gets j - 1$.
4. Return `False` if no triplet matches.

#### 3. Why Sub-Quadratic $O(N^{2-\varepsilon})$ Is Impossible Unbounded
This problem is an algebraic equivalent of **3SUM**. In the Real RAM comparison model with unbounded integers, the 3SUM conjecture states that no $\mathcal{O}(N^{2-\varepsilon})$ algorithm exists. Breaking the quadratic bound is only possible under bounded integer domains (e.g. $|nums[i]| \le U$) via FFT convolution.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List

class PythagoreanTripletSolution:
    """
    Production-grade Pythagorean Triplet verifier:
    Strictly preserves element multiplicity across zero and duplicate values.
    Time Complexity: O(N^2), Space Complexity: O(1) auxiliary beyond square array.
    """

    @staticmethod
    def judgePythagoreanTriplet(nums: List[int]) -> bool:
        n = len(nums)
        if n < 3:
            return False

        # 1. Map to squares
        squares = [x * x for x in nums]

        # 2. Sort
        squares.sort()

        # 3. Two-pointer convergence per candidate hypotenuse
        for k in range(n - 1, 1, -1):
            target = squares[k]
            i = 0
            j = k - 1

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
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: Squaring $\mathcal{O}(N)$, Sorting $\mathcal{O}(N \log N)$, Two-pointer outer loop $N$ steps with inner sweep $\mathcal{O}(k)$, summing to $\mathcal{O}(N^2)$. Total is strictly $\mathcal{O}(N^2)$.
- **Space Complexity**: $\mathcal{O}(N)$ for squared array (or $\mathcal{O}(1)$ in-place).
- **Critical Pitfall**: Allowing $i = j$ in the two-pointer loop, which would reuse a single element twice.

</div>

</div>
</details>

---

### 3. Variable & Fixed Sliding Window Patterns

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Window 03</span>
  <span class="review-card-title">Variable & Fixed Sliding Window Patterns</span>
  <span class="review-card-tag">Two Pointers · Hash Jump Acceleration · Frequency State Machine · Satisfaction Counter</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Archetypes</div>

- **Longest Substring Without Repeating Characters (LC 3)**: Longest substring with unique characters.
- **Minimum Window Substring (LC 76)**: Minimum window containing all target characters.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

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
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(|\Sigma|)$.

</div>

</div>
</details>

---

## Module 3: Backtracking & Combinatorial Search

### 4. Subsets, Permutations & Combinations

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Backtrack 04</span>
  <span class="review-card-title">Subsets, Permutations & Combinations</span>
  <span class="review-card-tag">State-Space Tree · Pruning Deduplication · Element Reuse · Used Tracking Array</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Paradigm Matrix</div>

| Pattern | Archetypes | State Decision | Pruning / Deduplication Strategy |
|---|---|---|---|
| **Subsets** | LC 78, LC 90 | Include or exclude element; collect all nodes | Pass `start` index; if duplicates exist, sort first, prune via `i > start and nums[i] == nums[i-1]`. |
| **Combinations** | LC 77, LC 39, LC 40 | Select fixed count or target sum | Pass `start` index to prevent backward picks; prune when sum exceeds target. |
| **Permutations** | LC 46, LC 47 | Ordering matters, pick from index 0 | Track boolean `used` array; prune via `nums[i] == nums[i-1] and not used[i-1]`. |

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

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
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N \cdot 2^N)$.
- **Space Complexity**: $\mathcal{O}(N)$.

</div>

</div>
</details>
