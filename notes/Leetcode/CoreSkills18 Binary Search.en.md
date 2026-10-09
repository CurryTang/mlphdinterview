# Binary Search: A Unified Template

The fundamental essence of binary search is simple: **narrowing the search interval monotonically and recording the optimal candidate solution**.

Regardless of problem variations (exact match, answer range, rotated array, median partition), we consistently use closed-interval probing (`while left <= right`), recording `ans = mid` when feasible and shifting inward. Problem solving only requires clarifying the **search space**, the **feasibility predicate**, and the **exploration direction**.

## Learning Order

Problems selected from the Binary Search module of [NeetCode 150](https://neetcode.io/practice/practice/neetcode150):

| Order | Problem | Core Pattern | Closed-Interval Key Points |
|---:|---|---|---|
| 1 | [704. Binary Search](https://neetcode.io/problems/binary-search/question?list=neetcode150) | Basic Exact Match | Range `[0, n-1]`, return on match, shrink bounds otherwise |
| 2 | [74. Search a 2D Matrix](https://neetcode.io/problems/search-a-2d-matrix/question?list=neetcode150) | 2D Matrix Flattening | Flatten index `[0, m*n - 1]`, map via `(mid // n, mid % n)` |
| 3 | [875. Koko Eating Bananas](https://neetcode.io/problems/koko-eating-bananas/question?list=neetcode150) | Answer Range Minimum | Range `[1, max]`, record `ans` and probe left (`right = mid - 1`) |
| 4 | [153. Find Minimum in Rotated Sorted Array](https://neetcode.io/problems/find-minimum-in-rotated-sorted-array/question?list=neetcode150) | Rotated Array Minimum | Test `nums[mid] <= nums[-1]`, record `ans` and probe left |
| 5 | [33. Search in Rotated Sorted Array](https://neetcode.io/problems/find-target-in-rotated-sorted-array/question?list=neetcode150) | Rotated Array Search | Partitioned search (identify sorted half and shrink bounds) |
| 6 | [981. Time Based Key-Value Store](https://neetcode.io/problems/time-based-key-value-store/question?list=neetcode150) | Max Feasible Timestamp | Test `time <= query`, record `ans` and probe right (`left = mid + 1`) |
| 7 | [4. Median of Two Sorted Arrays](https://neetcode.io/problems/median-of-two-sorted-arrays/question?list=neetcode150) | Median of Two Sorted Arrays | Partition shorter array, record `ans` and probe left |

## Module 1: General Binary Search Template (`ans`-Recording Method)

### 1. The Issue with Traditional Templates: Why Finding Maximum Is Error-Prone

When using `while lo < hi:` (half-open interval):
- **Finding Minimum (e.g., LC 875 Min Speed)**: Conditions matching the target move left (`hi = mid`), returning `lo` at termination.
- **Finding Maximum (e.g., LC 2226 Max Candies, rope cutting, max capacity)**:
  - The problem asks for the *last feasible value*, but the template forces finding the *first infeasible value*, followed by `lo - 1`.
  - When no valid answer exists (e.g., total candies less than $k$), additional edge-case handling is needed for `lo - 1` out-of-bounds.

---

### 2. Closed-Interval `ans` Recording Template (`while left <= right`)

Maintain a variable `ans` to store the latest valid result, and let `check(mid)` directly assess whether `mid` is feasible:

```python
left, right = min_possible, max_possible
ans = default_value  # Fallback if no solution exists, e.g., 0 for LC 2226

while left <= right:
    mid = (left + right) // 2
    if check(mid):  # Current mid is feasible
        ans = mid   # Record feasible candidate
        # Finding maximum -> explore larger values: left = mid + 1
        # Finding minimum -> explore smaller values: right = mid - 1
    else:
        # Current mid is infeasible
        # Finding maximum -> value too large, shrink: right = mid - 1
        # Finding minimum -> value too small, increase: left = mid + 1

return ans
```

Key points:
- `check(mid)` tests the direct feasibility condition without negation.
- Store feasible answers via `ans = mid`.
- For maximum, advance right (`left = mid + 1`); for minimum, advance left (`right = mid - 1`).
- Return `ans` directly upon loop termination, without `lo - 1` or index offsets.

---

### 3. Implementation Comparison: Finding Maximum (LC 2226) vs. Minimum (LC 875)

#### 1. Finding Maximum Feasible Value: LC 2226. Maximum Candies Allocated to K Children
> Rule: Each pile can be split but not merged. Allocate equal positive integer candies to $k$ children. Return max candies per child, or 0 if impossible.

```python
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        left, right = 1, max(candies)
        ans = 0  # Default 0 if cannot allocate even 1 candy each

        while left <= right:
            mid = (left + right) // 2
            # Can we allocate at least k piles of size mid?
            if sum(c // mid for c in candies) >= k:
                ans = mid       # Feasible, record value
                left = mid + 1  # Finding max, try larger values
            else:
                right = mid - 1 # Too large, reduce value

        return ans
```

#### 2. Finding Minimum Feasible Value: LC 875. Koko Eating Bananas
> Rule: Finish all piles in $h$ hours. Return minimum speed.

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        ans = right  # Default to max pile size, always feasible

        while left <= right:
            mid = (left + right) // 2
            # Can Koko finish in <= h hours at speed mid?
            if sum((p + mid - 1) // mid for p in piles) <= h:
                ans = mid        # Feasible, record value
                right = mid - 1  # Finding min, try smaller values
            else:
                left = mid + 1   # Too slow, increase speed

        return ans
```

---

### 4. Comparison of the Two Approaches

| Aspect | Traditional `while lo < hi` | `ans` Template (`left <= right`) |
|---|---|---|
| **Condition for Maximum** | Needs inverted `not check` for first invalid | Directly tests positive `check`, assigns `ans = mid` |
| **Return Value** | Requires choosing between `lo` and `lo - 1` | Always returns `ans` |
| **No-Solution Handling** | Needs manual check on `lo == 0` | Naturally returns default `ans` (e.g., 0 or -1) |
| **Convergence** | Potential infinite loop on `lo = mid` without rounding | Always shifts by `mid + 1` or `mid - 1` |

---

### 5. Common Problem Mapping Table

| Problem Scenario | Range `[left, right]` | `check(mid)` Condition | Movement when Feasible | Default `ans` |
|---|---|---|---|---|
| **LC 2226 Candies (Find Max)** | `1, max(candies)` | `sum(c // mid) >= k` | `left = mid + 1` | `0` |
| **LC 875 Bananas (Find Min)** | `1, max(piles)` | `hours_needed(mid) <= h` | `right = mid - 1` | `max(piles)` |
| **LC 704 Basic Search** | `0, len(nums) - 1` | `nums[mid] == target` | Return `mid` directly | `-1` |
| **First $\ge target$** | `0, len(nums) - 1` | `nums[mid] >= target` | `right = mid - 1` | `-1` or `len(nums)` |
| **Last $\le target$** | `0, len(nums) - 1` | `nums[mid] <= target` | `left = mid + 1` | `-1` |
| **LC 153 Min in Rotated Array** | `0, len(nums) - 1` | `nums[mid] <= nums[-1]` | `right = mid - 1` | `nums[-1]` |
| **LC 981 Time-Based KV Store** | `0, len(entries) - 1` | `entries[mid].time <= query` | `left = mid + 1` | `""` |

---

## Module 2: Mapping Each of the Seven Problems

### Binary Search: Basic Exact Match

The search space is the closed interval `[0, len(nums) - 1]`. Directly check `mid` against the target. Return immediately on a match; adjust the left or right bound according to value comparison. No sentinels required.

| Item | Value |
|---|---|
| Search space | Closed index interval `[0, len(nums) - 1]` |
| `check(mid)` | `nums[mid] == target` |
| Pointer movement | Return `mid` on hit; if smaller, `left = mid + 1`; if larger, `right = mid - 1` |
| Default return | Return `-1` if loop terminates without finding target |

#### Quick Coding: Binary Search

```python
def search(nums, target):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1
```

```python
# Standard library bisect approach:
from bisect import bisect_left
from typing import List


class SolutionBisect:
    def search(self, nums: List[int], target: int) -> int:
        idx = bisect_left(nums, target)
        return idx if idx < len(nums) and nums[idx] == target else -1
```

</details>

### Search a 2D Matrix: Mapping a 2D Index to 1D

The flattened 2D matrix forms a strictly sorted sequence with $m \times n$ total elements. The search space is the closed interval `[0, m * n - 1]`. Retrieve elements via `(mid // n, mid % n)` and apply standard closed-interval binary search.

| Item | Value |
|---|---|
| Search space | Closed flattened interval `[0, m * n - 1]` |
| Coordinate map | `row, col = mid // n, mid % n` |
| Pointer movement | Return `True` on match; if smaller, `left = mid + 1`; if larger, `right = mid - 1` |
| Default return | Return `False` after loop termination |

#### Quick Coding: Search a 2D Matrix

```python
def searchMatrix(matrix, target):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1

        while left <= right:
            mid = (left + right) // 2
            val = matrix[mid // n][mid % n]
            if val == target:
                return True
            elif val < target:
                left = mid + 1
            else:
                right = mid - 1

        return False
```

</details>

### Koko Eating Bananas: Answer Range Minimization

Rather than indexing an array, find the **minimum speed** in the range `[1, max(piles)]` such that total eating time is $\le h$.
- `check(mid)`: Whether eating at speed `mid` finishes within $h$ hours (`sum((p + mid - 1) // mid for p in piles) <= h`).
- When viable: Record `ans = mid` and probe left for smaller valid speeds (`right = mid - 1`).
- When overtime: Speed is insufficient, increase speed (`left = mid + 1`).

$$
\text{hours\_needed(speed)} = \sum_{\text{pile}} \left\lceil \frac{\text{pile}}{\text{speed}} \right\rceil = \sum_{\text{pile}} \lfloor \frac{\text{pile} + \text{speed} - 1}{\text{speed}} \rfloor
$$

| Item | Value |
|---|---|
| Search space | Speed closed interval `[1, max(piles)]` |
| `check(mid)` | `hours_needed(mid) <= h` |
| Movement rule | Feasible: record `ans = mid`, probe left (`right = mid - 1`); Overtime: probe right (`left = mid + 1`) |
| Default return | `ans = max(piles)` (maximum pile size is always viable) |

#### Quick Coding: Koko Eating Bananas

```python
def minEatingSpeed(piles, h):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        ans = right

        while left <= right:
            mid = (left + right) // 2
            # Ceiling division: (p + mid - 1) // mid
            if sum((p + mid - 1) // mid for p in piles) <= h:
                ans = mid        # Feasible, record speed
                right = mid - 1  # Probe left for smaller speed
            else:
                left = mid + 1   # Overtime, increase speed

        return ans
```

```python
# Standard library bisect one-liner:
from bisect import bisect_left
from typing import List


class SolutionBisect:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        r = range(1, max(piles) + 1)
        idx = bisect_left(r, True, key=lambda s: sum((p + s - 1) // s for p in piles) <= h)
        return r[idx]
```

</details>

### Find Minimum in Rotated Sorted Array: Rotated Array Minimum

The rotated array consists of two ascending segments: every element in the left segment is $> nums[-1]$, and every element in the right segment is $\le nums[-1]$.
- Condition: `nums[mid] <= nums[-1]`.
- When satisfied: `mid` is in the right segment and is a valid candidate. Record `ans = nums[mid]` and probe left to find the earlier transition boundary (`right = mid - 1`).
- When unsatisfied: `mid` is still in the left segment; the minimum must lie strictly to the right (`left = mid + 1`).
- Naturally handles unrotated arrays because the first element already satisfies `<= nums[-1]`.

| Item | Value |
|---|---|
| Search space | Closed index interval `[0, len(nums) - 1]` |
| `check(mid)` | `nums[mid] <= nums[-1]` |
| Movement rule | Feasible: `ans = nums[mid]`, probe left `right = mid - 1`; Infeasible: probe right `left = mid + 1` |
| Default return | `ans = nums[-1]` (fallback to tail element) |

#### Quick Coding: Find Minimum in Rotated Sorted Array

```python
def findMin(nums):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        ans = nums[-1]

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= nums[-1]:
                ans = nums[mid]  # Right segment candidate
                right = mid - 1  # Probe left for earlier boundary
            else:
                left = mid + 1   # Left segment, minimum is to the right

        return ans
```

</details>

### Search in Rotated Sorted Array: Rotated Array Search

Perform binary search in `[0, len(nums) - 1]`. Although the array as a whole is not monotonically sorted, dividing at `mid` guarantees that **at least one of the two halves is strictly sorted**:
- If `nums[left] <= nums[mid]`: The left half is sorted. If `nums[left] <= target < nums[mid]`, shrink right bound (`right = mid - 1`); otherwise search right half (`left = mid + 1`).
- Otherwise: The right half is sorted. If `nums[mid] < target <= nums[right]`, shrink left bound (`left = mid + 1`); otherwise search left half (`right = mid - 1`).

| Item | Value |
|---|---|
| Search space | Closed index interval `[0, len(nums) - 1]` |
| Core logic | Identify sorted half, then test whether `target` falls inside |
| Pointer movement | Return `mid` on hit; adjust `left` / `right` based on interval |
| Default return | Return `-1` if loop terminates without finding target |

#### Quick Coding: Search in Rotated Sorted Array

```python
def search(nums, target):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


# Approach 1: Classic Partitioned Search (Recommended)
class Solution:
    def search(self, nums: List[int], target: int) -> int:
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
            else:  # Right half is sorted
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1


# Approach 2: Key Transformation Linearization (Closed Interval)
class SolutionKeyTransform:
    def search(self, nums: List[int], target: int) -> int:
        pivot = nums[-1]
        def key(x: int): return (x <= pivot, x)

        target_key = key(target)
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            k = key(nums[mid])
            if k == target_key:
                return mid
            elif k < target_key:
                left = mid + 1
            else:
                right = mid - 1

        return -1
```

</details>

### Time Based Key-Value Store: Find Maximum Feasible Timestamp

`set` operations append records in strictly ascending timestamps, so records for each `key` are naturally sorted.
`get` queries the **latest (maximum)** record with timestamp $\le query$:
- Feasibility condition: `entries[mid][0] <= timestamp`.
- When feasible: Record `ans = entries[mid][1]` and probe right for newer timestamps (`left = mid + 1`).
- When infeasible: Timestamp is too new, shrink left (`right = mid - 1`).
- Default return: `ans = ""` (if all records exceed query time, returns empty string cleanly with no index special cases).

| Item | Value |
|---|---|
| Search space | Closed index interval `[0, len(entries) - 1]` |
| `check(mid)` | `entries[mid][0] <= timestamp` |
| Movement rule | Feasible: record `ans = entries[mid][1]`, probe right (`left = mid + 1`); Infeasible: probe left (`right = mid - 1`) |
| Default return | `ans = ""` (returned directly if not found) |

#### Quick Coding: Time Based Key-Value Store

```python
class TimeMap:
    def __init__(self):
        ...

    def set(self, key, value, timestamp):
        ...

    def get(self, key, timestamp):
        ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import defaultdict


class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)  # key -> [(timestamp, value), ...]

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        entries = self.store[key]
        left, right = 0, len(entries) - 1
        ans = ""

        while left <= right:
            mid = (left + right) // 2
            if entries[mid][0] <= timestamp:
                ans = entries[mid][1]  # Feasible, record value
                left = mid + 1         # Probe right for newer record
            else:
                right = mid - 1        # Infeasible, shrink left

        return ans
```

```python
# Standard library bisect approach:
from bisect import bisect_right
from collections import defaultdict


class TimeMapBisect:
    def __init__(self):
        self.store = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        entries = self.store[key]
        idx = bisect_right(entries, timestamp, key=lambda x: x[0]) - 1
        return entries[idx][1] if idx >= 0 else ""
```

</details>

### Median of Two Sorted Arrays: Partitioning the Shorter Array

The search space is the cut position $i \in [0, m]$ on the shorter array $A$ (length $m$). Both left partitions together must have `half = (m + n + 1) // 2` elements, leaving cut position $j = \text{half} - i$ on $B$.
- Condition: Whether the cut satisfies $A$'s right element $\ge B$'s left element (`A[mid] >= B[half - mid - 1]`).
- When feasible: Record `ans = mid` and probe left for the minimum valid partition cut (`right = mid - 1`).
- When infeasible: Cut on $A$ is too small, probe right (`left = mid + 1`).
- Compute the median using `ans` upon loop exit.

$$
\text{check}(i) = A[i] \ge B[j-1], \quad j = \text{half} - i
$$

| Item | Value |
|---|---|
| Search space | Closed partition interval `[0, m]` ($m \le n$) |
| `check(mid)` | `a_right >= b_left` (out-of-bounds padded with $\pm\infty$) |
| Movement rule | Feasible: record `ans = mid`, probe left `right = mid - 1`; Infeasible: probe right `left = mid + 1` |
| Default return | `ans = m` |

#### Quick Coding: Median of Two Sorted Arrays

```python
def findMedianSortedArrays(nums1, nums2):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import math
from typing import List


# Approach 1: Closed-Interval ans Recording (Manual Template)
class Solution:
    def findMedianSortedArrays(
        self,
        nums1: List[int],
        nums2: List[int],
    ) -> float:
        A, B = nums1, nums2
        if len(A) > len(B):
            A, B = B, A

        m, n = len(A), len(B)
        half = (m + n + 1) // 2

        def a_right_big_enough(i: int) -> bool:
            j = half - i
            a_right = A[i] if i < m else math.inf
            b_left = B[j - 1] if j > 0 else -math.inf
            return a_right >= b_left

        left, right = 0, m
        ans = m

        while left <= right:
            mid = (left + right) // 2
            if a_right_big_enough(mid):
                ans = mid        # Feasible, record cut
                right = mid - 1  # Probe left for minimum valid cut
            else:
                left = mid + 1   # Cut too small, advance right

        i = ans
        j = half - i
        a_left = A[i - 1] if i > 0 else -math.inf
        a_right = A[i] if i < m else math.inf
        b_left = B[j - 1] if j > 0 else -math.inf
        b_right = B[j] if j < n else math.inf

        max_left = max(a_left, b_left)
        if (m + n) % 2 == 1:
            return float(max_left)

        min_right = min(a_right, b_right)
        return (max_left + min_right) / 2


# Approach 2: Standard Library bisect (Concise & Easy to Memorize)
from bisect import bisect_left


class SolutionBisect:
    def findMedianSortedArrays(
        self,
        nums1: List[int],
        nums2: List[int],
    ) -> float:
        A, B = (nums1, nums2) if len(nums1) <= len(nums2) else (nums2, nums1)
        m, n = len(A), len(B)
        half = (m + n + 1) // 2

        def check(i: int) -> bool:
            j = half - i
            a_right = A[i] if i < m else math.inf
            b_left = B[j - 1] if j > 0 else -math.inf
            return a_right >= b_left

        # Binary search for the first cut in [0, m] where check(i) is True
        i = bisect_left(range(m + 1), True, key=check)
        j = half - i

        a_left = A[i - 1] if i > 0 else -math.inf
        a_right = A[i] if i < m else math.inf
        b_left = B[j - 1] if j > 0 else -math.inf
        b_right = B[j] if j < n else math.inf

        max_left = max(a_left, b_left)
        if (m + n) % 2 == 1:
            return float(max_left)

        min_right = min(a_right, b_right)
        return (max_left + min_right) / 2
```

</details>

## Module 3: Pre-Interview Checklist

3-Step Checklist:
1. **Interval `[left, right]`**: Index range is `[0, len(nums) - 1]`; answer domain is `[min_val, max_val]`.
2. **Predicate `check(mid)`**: Directly code "is current mid viable/valid?".
3. **Move & Record**:
   - When viable, record `ans = mid`;
   - To maximize, probe right (`left = mid + 1`); to minimize, probe left (`right = mid - 1`);
   - Return `ans` directly after exiting loop.


## Module 4: Binary Search High-Frequency Extensions

### Majority Element in Sorted Array via Sublinear Binary Search Probe

#### Core Mental Model (Probe Sampling + Binary Search Verification)
Elements with frequency $> \lfloor n/3 \rfloor$ must span across quantile indices `n // 3` or `2 * n // 3`. Sample at most 2 candidates at these probe points, then count exact frequency via `bisect_right - bisect_left` in $O(\log n)$ time. Total time is strictly $O(\log n)$!

```python
from bisect import bisect_left, bisect_right
from typing import List

def findMajorityElementsSorted(nums: List[int]) -> List[int]:
    n = len(nums)
    if not nums: return []
    threshold = n // 3
    res = []
    for cand in {nums[n // 3], nums[(2 * n) // 3]}:
        if bisect_right(nums, cand) - bisect_left(nums, cand) > threshold:
            res.append(cand)
    return sorted(res)
```

---

## Module 5: Python Standard Library bisect Guide

Python provides the built-in `bisect` module implemented in C for high performance. In interviews, when binary search is merely an auxiliary step of a broader problem (e.g., LIS, greedy interval scheduling, timestamp-based key-value lookups), using `bisect` saves time and eliminates edge-case boundary errors.

### 1. Function Signatures & Parameter Details

```python
bisect_left(a, x, lo=0, hi=len(a), *, key=None)
bisect_right(a, x, lo=0, hi=len(a), *, key=None)  # Alias: bisect
insort_left(a, x, lo=0, hi=len(a), *, key=None)
insort_right(a, x, lo=0, hi=len(a), *, key=None)  # Alias: insort
```

Parameters and considerations:

| Parameter | Type & Default | Description | Key Notes & Edge Cases |
|---|---|---|---|
| `a` | Sequence (Required) | Sorted sequence to search | Supports random access (`list`, `tuple`, `range`); must be sorted ascending by key |
| `x` | Any (Required) | Target value to search / insert | If `key` is specified, `x` is compared directly with `key(element)` |
| `lo` | `int = 0` | Lower bound of slice (inclusive) | Bounds search to `[lo, hi)`; **avoids costly slice copy `a[lo:hi]`** |
| `hi` | `int = len(a)` | Upper bound of slice (exclusive) | Half-open interval; returned index is still global index relative to `a` |
| `key` | 1-arg function (Python 3.10+) | Key extraction function | **Only applied to elements in `a[i]`, NOT to `x`**; used for projections |

#### Parameter Usage Details:
1. **Advantages of `lo` and `hi` (Avoiding Slice Copies)**:
   - Slicing `bisect_left(nums[10:50], target)` creates a new 40-element list, incurring $O(k)$ time/memory overhead and requiring manual index offsets.
   - Passing `lo=10, hi=50` searches in-place with zero memory allocation, and returns the absolute index in `nums`.
2. **One-Way Application of `key`**:
   - `key` is only evaluated on elements of `a` (`key(a[mid])` compared to `x`).
   - Therefore, `x` must match the type of the extracted key. For instance, searching timestamps in `[(1, 'a'), (3, 'b')]` passes integer `3` for `x` with `key=lambda item: item[0]`.
3. **`insort_left` / `insort_right`**:
   - Locates insertion point and executes `a.insert(idx, x)` in-place, keeping `a` sorted.
   - Note: `list.insert` takes $O(n)$ time due to array shifting. For repeated insertions and queries, use `heapq` or a balanced tree.

---

### 2. Four Classic Array Queries

`bisect_left` finds the first insertion index where elements are $\ge target$; `bisect_right` (alias `bisect`) finds the first insertion index where elements are $> target$:

| Target Query | Code Expression | Note & Boundaries |
|---|---|---|
| **First $\ge target$** | `idx = bisect_left(nums, target)` | Returns `len(nums)` if none satisfy |
| **First $> target$** | `idx = bisect_right(nums, target)` | Returns `len(nums)` if none satisfy |
| **Last $\le target$** | `idx = bisect_right(nums, target) - 1` | Returns `-1` (i.e. `idx < 0`) if none satisfy |
| **Last $< target$** | `idx = bisect_left(nums, target) - 1` | Returns `-1` (i.e. `idx < 0`) if none satisfy |
| **Exact match lookup** | `i = bisect_left(nums, target)`<br>`found = (i < len(nums) and nums[i] == target)` | LC 704 standard exact match verification |

---

### 3. Python 3.10+ `key=` Parameter

Starting with Python 3.10, `bisect` supports a `key=` parameter for projecting elements or compound tuples.

#### Example: LC 981 Time-Based Key-Value Store
```python
from bisect import bisect_right

# entries stores sorted records: [(timestamp_1, val_1), (timestamp_2, val_2), ...]
def get(entries, query_time):
    # Find the last record with timestamp <= query_time
    idx = bisect_right(entries, query_time, key=lambda x: x[0]) - 1
    return entries[idx][1] if idx >= 0 else ""
```

---

### 4. Binary Search on Answer Range via `range` + `key`

In Python, `range()` is a virtual sequence with $O(1)$ random access that requires $O(1)$ auxiliary space. Combined with `key=`, it can perform binary search directly over an answer range.

Since boolean values in Python satisfy `False < True` ($0 < 1$), if a predicate sequence is monotonically `[False, False, ..., True, True]`, `bisect_left(..., True)` directly locates the first `True`:

#### Example A: Minimization · LC 875 Koko Eating Bananas
```python
from bisect import bisect_left
from typing import List

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        r = range(1, max(piles) + 1)
        # Speeds that are too slow exceed time limit (False); becomes True when valid.
        # Find first True:
        idx = bisect_left(r, True, key=lambda s: sum((p + s - 1) // s for p in piles) <= h)
        return r[idx]
```

#### Example B: Maximization · LC 2226 Maximum Candies Allocated to K Children
```python
from bisect import bisect_left
from typing import List

class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        # Condition becomes True when candy count is too large to distribute k piles (inverted predicate).
        # Find first invalid point; its 0-based index corresponds to the maximum valid candies (starting from 1).
        return bisect_left(
            range(1, max(candies) + 1),
            True,
            key=lambda s: sum(c // s for c in candies) < k
        )
```

#### Example C: Partition Point Search · LC 4 Median of Two Sorted Arrays
```python
from bisect import bisect_left
import math
from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        A, B = (nums1, nums2) if len(nums1) <= len(nums2) else (nums2, nums1)
        m, n = len(A), len(B)
        half = (m + n + 1) // 2

        # Check whether A's right element >= B's left element
        def check(i: int) -> bool:
            j = half - i
            a_right = A[i] if i < m else math.inf
            b_left = B[j - 1] if j > 0 else -math.inf
            return a_right >= b_left

        # Binary search for the first cut in [0, m] where check(i) is True
        i = bisect_left(range(m + 1), True, key=check)
        j = half - i

        a_left = A[i - 1] if i > 0 else -math.inf
        a_right = A[i] if i < m else math.inf
        b_left = B[j - 1] if j > 0 else -math.inf
        b_right = B[j] if j < n else math.inf

        max_left = max(a_left, b_left)
        if (m + n) % 2 == 1:
            return float(max_left)
        return (max_left + min(a_right, b_right)) / 2.0
```

---

### 5. Interview Strategy

- **Binary search is the main focus**: (e.g., the interviewer asks you to write binary search manually or discuss rotated sorted array edge cases): **Write out the manual closed-interval `while left <= right` template** to demonstrate mastery of loop invariants and boundaries.
- **Binary search is a secondary utility step**: (e.g., part of a Hard problem, LIS subproblem, greedy scheduling): **Prefer `bisect`**, and tell the interviewer: "The array is sorted, so I'm using the standard library `bisect` for $O(\log n)$ lookup to avoid boundary edge cases."

