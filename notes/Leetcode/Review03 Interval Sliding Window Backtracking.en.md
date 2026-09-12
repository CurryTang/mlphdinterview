# Review Flashcards: Intervals, Sliding Window & Backtracking

This note is the third volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Intervals & Sweep Line**, **Two Pointers & Sliding Window**, and **Backtracking & Digit Greedy** with production-grade implementations and asymptotic complexity breakdowns.

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
<div class="review-block-label">📌 Implementation</div>

```python
from typing import List

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
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity</div>

- $\mathcal{O}(N \log N)$ sorting, $\mathcal{O}(1)$ space.

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
<div class="review-block-label">📌 Implementation</div>

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
<div class="review-block-label">⏱️ Complexity</div>

- $\mathcal{O}(N^2)$ time. In the comparison model, no $\mathcal{O}(N^{2-\varepsilon})$ algorithm exists by 3SUM conjecture.

</div>

</div>
</details>

---

### 3. Minimum Window Substring & Multi-Candidate Expansion

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Window 03</span>
  <span class="review-card-title">Minimum Window Substring & Multi-Candidate Expansion</span>
  <span class="review-card-tag">Variable Window · O(|S| + |T|) Rigorous Proof · k-Factor Frequency · All Tied Minimum Windows · Fixed-Size Array Constant Bound</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Follow-ups</div>

Given strings $S$ and $T$, find the shortest substring in $S$ containing every character in $T$, respecting multiplicities:
- **Linear complexity proof**: Explain $\mathcal{O}(|S| + |T|)$.
- **k-factor follow-up**: Every character in $T$ must appear at least $k$ times (or $k$ times its required frequency).
- **All tied minimum windows**: Return **every valid substring tied for the minimum length**, rather than only one.
- **Fixed array bound**: Fixed-size counter array (ASCII 128) guarantees strict constant lookup without hash collision overhead.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Invariants</div>

- `formed_kinds` increments strictly when a character count first matches target requirement; extra copies do not re-increment.
- **Contract and record before removal**: When valid, update the answer list before incrementing `left`, because removing the leftmost character can immediately invalidate the window.
- When finding a strictly smaller window, clear the answer list; append on ties.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List
from collections import Counter

class MinWindowSolution:

    @classmethod
    def minWindowAll(cls, s: str, t: str, k_scale: int = 1) -> List[str]:
        """
        Collects all tied shortest substrings containing T (scaled by k_scale).
        Time: O(|S| + |T|), Space: O(1) with 128-byte array.
        """
        if not s or not t or k_scale <= 0:
            return []

        target_counts = Counter(t)
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

            if req[r_code] > 0 and win[r_code] == req[r_code]:
                formed_kinds += 1

            while left <= right and formed_kinds == required_kinds:
                cur_len = right - left + 1

                if cur_len < min_len:
                    min_len = cur_len
                    ans_list = [s[left : right + 1]]
                elif cur_len == min_len:
                    ans_list.append(s[left : right + 1])

                l_code = ord(s[left])
                win[l_code] -= 1
                if req[l_code] > 0 and win[l_code] < req[l_code]:
                    formed_kinds -= 1

                left += 1

        return ans_list
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(|S| + |T|)$. Each character enters and leaves the window at most once.
- **Space Complexity**: $\mathcal{O}(1)$ with fixed-size ASCII array.
- **Critical Pitfalls**: Recording answers after incrementing `left` loses valid boundary data.

</div>

</div>
</details>

---

### 4. Longest Repeating Character Replacement & Max-Frequency Invariant

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Window 04</span>
  <span class="review-card-title">Longest Repeating Character Replacement & Max-Frequency Invariant</span>
  <span class="review-card-tag">Sliding Window · Max-Frequency Invariant · Non-Decreasing Window Size · O(N) Single-Pass</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Examples</div>

Given an uppercase string `s` and integer `k`, replace at most `k` characters to form the longest substring with identical characters (LC 424):
- `s = "ABAB", k = 2` $\implies 4$
- `s = "AABABBA", k = 1` $\implies 4$

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Max-Frequency Monotonicity</div>

- Window condition: `(length - max_freq) <= k`.
- **Key Insight**: `max_freq` does NOT need to be recomputed when shrinking `left`! A smaller frequency cannot yield a larger valid window than the historical best.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
class CharacterReplacementSolution:

    @staticmethod
    def characterReplacement(s: str, k: int) -> int:
        counts = [0] * 26
        left = max_freq = max_len = 0

        for right, ch in enumerate(s):
            idx = ord(ch) - ord('A')
            counts[idx] += 1
            max_freq = max(max_freq, counts[idx])

            while (right - left + 1) - max_freq > k:
                counts[ord(s[left]) - ord('A')] -= 1
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: Strictly $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(1)$ for 26 letters.

</div>

</div>
</details>

---

### 5. Variable & Fixed Sliding Window Patterns

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Window 05</span>
  <span class="review-card-title">Variable & Fixed Sliding Window Patterns</span>
  <span class="review-card-tag">Longest Substring Without Repeating Characters · Hash Jump Optimization</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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

</div>
</details>

---

## Module 3: Backtracking & Digit Greedy

### 6. Subsets, Permutations & Combinations

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Backtrack 06</span>
  <span class="review-card-title">Subsets, Permutations & Combinations</span>
  <span class="review-card-tag">State-Space Tree · Pruning Deduplication</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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

</div>
</details>

---

### 7. Largest Number Smaller than N from Digits A (Digit Greedy Backtracking)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Digit 07</span>
  <span class="review-card-title">Largest Number Smaller than N from Digits A (Digit Greedy Backtracking)</span>
  <span class="review-card-tag">Digit Backtracking · Greedy Prefix Match · Downgrade Suffix Max Fill · Shorter Length Fallback</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Examples</div>

Given a positive integer $N$ and a set of decimal digits $A \subseteq \{0, \dots, 9\}$ reusable indefinitely:
- Construct the largest integer strictly smaller than $N$ using only digits from $A$.
- Example: $N = 23415, A = [2, 4, 9] \implies 22999$.
- Example: $N = 222, A = [2] \implies 22$.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & State Machine</div>

- Sort $A$ ascending.
- At position $i$:
  1. Try $d == N[i]$, recurse.
  2. If recursion fails, pick largest $d < N[i]$, and immediately fill all remaining suffix positions with $\max(A)$.
  3. If no same-length candidate exists, fallback to length $L - 1$ filled entirely with $\max(A)$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List

class DigitConstructionSolution:

    @classmethod
    def findLargestSmaller(cls, N: int, A: List[int]) -> int:
        if N <= 0 or not A:
            return -1

        digits = sorted(list(set(A)))
        max_d = digits[-1]
        s_N = str(N)
        L = len(s_N)
        res_digits: List[int] = []

        def backtrack(idx: int, is_less: bool) -> bool:
            if idx == L:
                return is_less

            cur_target = int(s_N[idx])

            if is_less:
                res_digits.append(max_d)
                if backtrack(idx + 1, True):
                    return True
                res_digits.pop()
                return False

            for d in reversed(digits):
                if d == cur_target:
                    res_digits.append(d)
                    if backtrack(idx + 1, False):
                        return True
                    res_digits.pop()
                elif d < cur_target:
                    res_digits.append(d)
                    if backtrack(idx + 1, True):
                        return True
                    res_digits.pop()

            return False

        if backtrack(0, False):
            return int("".join(map(str, res_digits)))

        if L > 1:
            if max_d == 0:
                return -1
            return int(str(max_d) * (L - 1))

        return -1
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(L \cdot |A|)$ where $L = \operatorname{len}(str(N))$.
- **Space Complexity**: $\mathcal{O}(L)$ recursion depth.
- **Critical Pitfalls**: Forgetting shorter-length fallback (e.g. $N=222, A=[3] \implies 33$).

</div>

</div>
</details>

---

### 06. Non-overlapping Intervals via Earliest Deadline First

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GREEDY 06</span>
  <span class="review-card-title">Non-overlapping Intervals via Earliest Deadline First</span>
  <span class="review-card-tag">Greedy Interval Scheduling · Earliest Deadline First · O(N log N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class NonOverlappingIntervalsSolution:
    @classmethod
    def eraseOverlapIntervals(cls, intervals: List[List[int]]) -> int:
        """
        Computes minimum removals to make remaining intervals non-overlapping.
        Touching endpoints [a, b] and [b, c] are compatible.
        """
        if not intervals:
            return 0

        # Sort by end time ascending
        intervals.sort(key=lambda x: x[1])

        kept_count = 1
        prev_end = intervals[0][1]

        for i in range(1, len(intervals)):
            if intervals[i][0] >= prev_end:
                kept_count += 1
                prev_end = intervals[i][1]

        return len(intervals) - kept_count
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Dual Formulation**:
  Minimizing removals is mathematically equivalent to maximizing the cardinality of a mutually disjoint subset of intervals.
- **Earliest Deadline First (EDF) Optimality**:
  Picking intervals that finish earliest leaves the largest possible remaining window for future candidates.
- **Touching Boundary Invariant**:
  Per standard interval convention, $start == end$ is compatible, requiring `intervals[i][0] >= prev_end`.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N \log N)$ for sorting, followed by an $\mathcal{O}(N)$ linear pass.
- **Space Complexity**: $\mathcal{O}(\log N)$ auxiliary space for sorting.

</div>

</div>
</details>

---

### 07. Time-Based Key-Value Store via Binary Search

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">BS 07</span>
  <span class="review-card-title">Time-Based Key-Value Store via Binary Search</span>
  <span class="review-card-tag">Binary Search (bisect) · Time Series Multiversion Storage · O(log N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from collections import defaultdict
import bisect
from typing import List, Tuple

class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        """Stores key-value at timestamp. Assumes monotonically increasing timestamps."""
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        """Returns value with largest timestamp_prev <= timestamp, or empty string."""
        if key not in self.store:
            return ""

        records = self.store[key]
        idx = bisect.bisect_right(records, (timestamp, chr(127)))

        if idx == 0:
            return ""

        return records[idx - 1][1]
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Predecessor Bisection**:
  Because write timestamps arrive strictly in ascending order, each key's list is ordered. `bisect_right` finds the first entry $> timestamp$, whose immediate predecessor `idx - 1` gives the latest valid revision.
- **Out-of-Order Writes Follow-up**:
  If timestamps arrive out of order, use `bisect.insort` or maintain a balanced tree / `SortedDict` to preserve $\mathcal{O}(\log M)$ operations.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: `set` is $\mathcal{O}(1)$ amortized; `get` is $\mathcal{O}(\log M)$ where $M$ is the revision count for that key.
- **Space Complexity**: $\mathcal{O}(N)$ overall storage across all entries.

</div>

</div>
</details>

---

### 08. Interval List Intersections via Two-Pointer Scan

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">TP 08</span>
  <span class="review-card-title">Interval List Intersections via Two-Pointer Scan</span>
  <span class="review-card-tag">Two Pointers · Closed Interval Intersection · Earliest End Advance · O(M + N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class IntervalIntersectionSolution:
    @classmethod
    def intervalIntersection(
        cls, firstList: List[List[int]], secondList: List[List[int]]
    ) -> List[List[int]]:
        """
        Computes pairwise intersection intervals between two sorted disjoint interval lists.
        """
        i, j = 0, 0
        m, n = len(firstList), len(secondList)
        result = []

        while i < m and j < n:
            start = max(firstList[i][0], secondList[j][0])
            end = min(firstList[i][1], secondList[j][1])

            if start <= end:
                result.append([start, end])

            # Advance the interval that finishes earlier
            if firstList[i][1] < secondList[j][1]:
                i += 1
            else:
                j += 1

        return result
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Overlap Formula**:
  The intersection of $[A_s, A_e]$ and $[B_s, B_e]$ is $[\max(A_s, B_s), \min(A_e, B_e)]$, which is non-empty iff $\max \le \min$.
- **Pointer Advancement Invariant**:
  If $A_e < B_e$, no subsequent interval in list $B$ can ever intersect $A_i$ because list $B$ is disjoint and sorted. Thus, advancing $i$ safely discards $A_i$ without missing potential overlaps.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(M + N)$, each step advances at least one pointer.
- **Space Complexity**: $\mathcal{O}(1)$ auxiliary space.

</div>

</div>
</details>

---

### 09. Longest Substring with At Most K Distinct Characters

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">SLIDE 09</span>
  <span class="review-card-title">Longest Substring with At Most K Distinct Characters</span>
  <span class="review-card-tag">Sliding Window · Frequency Map · Key Eviction · O(N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from collections import defaultdict

class LongestSubstringKDistinctSolution:
    @classmethod
    def lengthOfLongestSubstringKDistinct(cls, s: str, k: int) -> int:
        """
        Finds length of the longest substring with at most k distinct characters.
        """
        if not s or k <= 0:
            return 0

        counts = defaultdict(int)
        left = 0
        max_len = 0

        for right, ch in enumerate(s):
            counts[ch] += 1

            while len(counts) > k:
                left_ch = s[left]
                counts[left_ch] -= 1
                if counts[left_ch] == 0:
                    del counts[left_ch]  # Physical eviction ensures correct key count
                left += 1

            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len

        return max_len
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Physical Eviction Invariant**:
  `len(counts)` tracks distinct keys. Keys with zero count must be deleted via `del counts[ch]`; otherwise, zero-frequency characters falsely inflate the distinct count.
- **Amortized Sliding Window**:
  Right pointer expands, and left pointer shrinks monotonically. Every character enters and leaves at most once.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(K)$ for the frequency map.

</div>

</div>
</details>

---

### 10. Search in Rotated Sorted Array: Distinct vs Duplicates

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">BS 10</span>
  <span class="review-card-title">Search in Rotated Sorted Array: Distinct vs Duplicates</span>
  <span class="review-card-tag">Half-Sorted Partitioning · Duplicate Ambiguity · Boundary Shrinkage · O(log N) -> O(N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class SearchRotatedArraySolution:
    @classmethod
    def searchDistinct(cls, nums: List[int], target: int) -> int:
        """Search in rotated array with distinct values in strict O(log N)."""
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid

            # Check if left half is sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # Right half is sorted
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1

    @classmethod
    def searchDuplicates(cls, nums: List[int], target: int) -> bool:
        """Search in rotated array with duplicates; worst case degrades to O(N)."""
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True

            # Ambiguity when ends match mid
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
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Half-Sorted Invariant**:
  Dividing a rotated array always yields at least one monotone half. If target lies within that monotone range, search that half; otherwise discard it.
- **Duplicate Ambiguity & Linear Degeneration**:
  When $nums[left] == nums[mid] == nums[right]$, the inflection point cannot be determined. Bisection safely contracts boundaries by $1$, degrading to $\mathcal{O}(N)$ in the worst case (e.g. all equal elements).

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Distinct**: $\mathcal{O}(\log N)$ time, $\mathcal{O}(1)$ space.
- **Duplicates**: $\mathcal{O}(\log N)$ average, $\mathcal{O}(N)$ worst-case, $\mathcal{O}(1)$ space.

</div>

</div>
</details>

---

### 11. Drone Relay to Target via Greedy Forward Progression

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GREEDY 11</span>
  <span class="review-card-title">Drone Relay to Target via Greedy Forward Progression</span>
  <span class="review-card-tag">Greedy Simulation · Relay Pointer · Forward Leap Update · O(M log M + M)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class DroneRelaySolution:
    @classmethod
    def minWalkingDistance(cls, target: int, relay_points: List[int]) -> int:
        """
        Calculates minimum total walking distance from 0 to reach or pass target.
        Walking to a relay point incurs cost equal to distance; drone leaps forward 10 units.
        """
        if target <= 0:
            return 0

        relays = sorted(set(p for p in relay_points if p >= 0))
        
        curr_pos = 0
        total_walk_cost = 0
        idx = 0
        m = len(relays)

        while curr_pos < target:
            while idx < m and relays[idx] < curr_pos:
                idx += 1

            # No relays ahead, or next relay is at/past target
            if idx >= m or relays[idx] >= target:
                total_walk_cost += (target - curr_pos)
                break

            next_relay = relays[idx]
            walk_to_relay = next_relay - curr_pos

            # If walking directly to target is shorter than walking to relay
            if (target - curr_pos) <= walk_to_relay:
                total_walk_cost += (target - curr_pos)
                break

            # Walk to relay and take 10-unit drone jump
            total_walk_cost += walk_to_relay
            curr_pos = next_relay + 10
            idx += 1

        return total_walk_cost
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Greedy Invariant**:
  A 10-unit drone jump provides free forward displacement. Any forward relay located before the target provides non-negative displacement gain over walking.
- **Terminal Overshoot Guard**:
  If the next relay lies beyond target ($relays[idx] \ge target$), walking to it overshoots and wastes energy; walking directly to target completes the journey optimally.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(M \log M)$ to sort relays, plus $\mathcal{O}(M)$ two-pointer scan.
- **Space Complexity**: $\mathcal{O}(M)$ auxiliary space for sorted relays.

</div>

</div>
</details>

