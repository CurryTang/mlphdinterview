# Binary Search: A Unified Template

The fundamental essence of binary search is simple: **finding a transition boundary on a monotonic predicate (First True)**.

Regardless of problem variations (exact match, answer range, rotated array, median partition), the template code is always 6 lines. **You only ever need to fill in 3 blanks**, taking under 10 seconds with zero off-by-one errors.

## Learning Order

Problems selected from the Binary Search module of [NeetCode 150](https://neetcode.io/practice/practice/neetcode150):

| Order | Problem | Core Pattern | 3-Step Fill-in Key Point |
|---:|---|---|---|
| 1 | [704. Binary Search](https://neetcode.io/problems/binary-search/question?list=neetcode150) | Basic Exact Match | First $\ge target$, then verify equality |
| 2 | [74. Search a 2D Matrix](https://neetcode.io/problems/search-a-2d-matrix/question?list=neetcode150) | 2D Matrix Flattening | Flatten index `[0, m*n]`, map via `(mid // n, mid % n)` |
| 3 | [875. Koko Eating Bananas](https://neetcode.io/problems/koko-eating-bananas/question?list=neetcode150) | Answer Range Binary Search | Search space is speed `[1, max]`, check total hours $\le h$ |
| 4 | [153. Find Minimum in Rotated Sorted Array](https://neetcode.io/problems/find-minimum-in-rotated-sorted-array/question?list=neetcode150) | Rotated Array Minimum | Compare with tail `nums[-1]`, return `nums[lo]` |
| 5 | [33. Search in Rotated Sorted Array](https://neetcode.io/problems/find-target-in-rotated-sorted-array/question?list=neetcode150) | Rotated Array Search | Classic sorted-half branch / key linearization |
| 6 | [981. Time Based Key-Value Store](https://neetcode.io/problems/time-based-key-value-store/question?list=neetcode150) | Find Last True | Find first exceeding timestamp, return `lo - 1` |
| 7 | [4. Median of Two Sorted Arrays](https://neetcode.io/problems/median-of-two-sorted-arrays/question?list=neetcode150) | Median of Two Sorted Arrays | Binary search partition point on shorter array |

## Module 1: The Zero-Mental-Burden Template (`ans`-Recording Greedy Probe)

### 1. Why "Find Last via Right Neighbor Minus 1 (lo - 1)" Is Unnatural

Traditional tutorials force the `while lo < hi:` half-open template, which creates an asymmetric cognitive burden between finding minimums vs. maximums:
- **Finding Minimum (e.g., LC 875 Koko Bananas)**: Looking for the *first valid value* returns `lo` naturally.
- **Finding Maximum (e.g., LC 2226 Maximum Candies, rope cutting, truck capacity)**:
  - You want the *last feasible value*. The template forces you to negate your thinking into "find the first *infeasible* value", then do `lo - 1`!
  - If no solution is possible, you also have to guard against `lo == 0`. This goes against natural human reasoning.

---

### 2. The Definitive Solution: Closed-Interval `ans` Probe Template (`while left <= right`)

**Zero mental inversions. Zero guess-work between `lo` and `lo - 1`. The rule is just one sentence:**

> **`check(mid)` always asks directly: "Is `mid` feasible/valid?"**  
> **If feasible, record `ans = mid` immediately;**  
> **To find maximum, greedily probe right (`left = mid + 1`); to find minimum, greedily probe left (`right = mid - 1`)!**

```python
left, right = min_possible, max_possible
ans = default_fallback  # e.g., 0 if allocating candies is impossible, -1 if array search fails

while left <= right:
    mid = (left + right) // 2
    if check(mid):  # Direct intuition: is mid feasible?
        ans = mid   # Feasible! Safely record current best
        # Greedily probe for an even better solution:
        # If finding MAXIMUM (LC 2226) -> try larger: left = mid + 1
        # If finding MINIMUM (LC 875)  -> try smaller: right = mid - 1
    else:
        # Infeasible; shrink in the opposite direction:
        # If finding MAXIMUM -> too large to satisfy; shrink: right = mid - 1
        # If finding MINIMUM -> too small / timed out; speed up: left = mid + 1

return ans  # Return ans directly; zero +1 / -1 offsets!
```

---

### 3. Symmetrical Duality: Maximum (LC 2226) vs. Minimum (LC 875)

These two problems are the canonical dual templates for value-range binary search. Their code structure is 100% symmetrical:

#### Duality A: Find Maximum · LC 2226. Maximum Candies Allocated to K Children
> Problem: Distribute candies to $k$ children such that each child gets the same number of positive integer candies. Piles can be split but not merged. Find the **maximum candy count per child**; return 0 if impossible.

```python
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        left, right = 1, max(candies)
        ans = 0  # Fallback: if not even 1 candy per child is possible, return 0

        while left <= right:
            mid = (left + right) // 2
            # Direct question: can we allocate at least k piles of size mid?
            if sum(c // mid for c in candies) >= k:
                ans = mid       # Feasible! Record current best maximum
                left = mid + 1  # Greedy probe: try larger candy counts
            else:
                right = mid - 1 # Cannot allocate k piles; reduce candy count

        return ans
```

#### Duality B: Find Minimum · LC 875. Koko Eating Bananas
> Problem: Finish all piles of bananas within $h$ hours. Find the **minimum eating speed**.

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        ans = right  # Fallback: eating at max pile speed is always feasible

        while left <= right:
            mid = (left + right) // 2
            # Direct question: can Koko finish all piles in <= h hours at speed mid?
            if sum((p + mid - 1) // mid for p in piles) <= h:
                ans = mid        # Feasible! Record current best minimum
                right = mid - 1  # Greedy probe: try smaller eating speed
            else:
                left = mid + 1   # Too slow; must increase speed

        return ans
```

---

### 4. Why This Template Eliminates All Bugs (Three Core Advantages)

| Pain Point | Traditional `while lo < hi` | `ans`-Recording Greedy Probe (`left <= right`) |
|---|---|---|
| **Finding Maximum** | Must invert to `not check` to find first failure, then `lo - 1` | **Zero Inversion**: Write positive `check(mid)`; record `ans = mid` |
| **Return Value Offset** | Prone to confusing `lo` vs. `lo - 1` vs. `lo + 1` | **Zero Offsets**: Always `return ans` |
| **No-Solution Defense** | Requires manual guards like `lo == 0` or `lo == n` | **Zero Extra Guards**: Naturally returns initialized `ans` (e.g., 0 or -1) |
| **Infinite Loop Risk** | May loop forever when assigning `lo = mid` without ceil division | **Zero Risk**: Always strictly shrinks by `mid + 1` or `mid - 1` |

---

### 5. Universal Problem Fill-in Cheat Sheet

| Problem Scenario | Range `[left, right]` | Positive `check(mid)` | Probe Direction when Feasible | Default `ans` |
|---|---|---|---|---|
| **LC 2226 Candies (Find Max)** | `1, max(candies)` | `sum(c // mid) >= k` | `left = mid + 1` (probe larger) | `0` |
| **LC 875 Bananas (Find Min)** | `1, max(piles)` | `hours_needed(mid) <= h` | `right = mid - 1` (probe smaller) | `max(piles)` |
| **LC 704 Basic Exact Match** | `0, len(nums) - 1` | `nums[mid] == target` | Direct hit: `return mid` | `-1` |
| **First $\ge target$ (lower_bound)** | `0, len(nums) - 1` | `nums[mid] >= target` | `right = mid - 1` (probe left) | `-1` or `n` |
| **Last $\le target$ (upper_bound)** | `0, len(nums) - 1` | `nums[mid] <= target` | `left = mid + 1` (probe right) | `-1` |
| **LC 153 Min in Rotated Array** | `0, len(nums) - 1` | `nums[mid] <= nums[-1]` | `right = mid - 1` (probe left) | `nums[-1]` |
| **LC 981 Time-Based KV Store** | `0, len(entries) - 1` | `entries[mid].time <= query` | `left = mid + 1` (probe right) | `""` |

---

## Module 2: Mapping Each of the Seven Problems

### Binary Search: The Template's Basic Form

The search space is index `[0, n]`, with one position past `n` acting as the sentinel. The predicate `check(mid) = nums[mid] >= target` is monotonic on a sorted array. After finding boundary `lo`, verify `nums[lo] == target`; otherwise the target does not exist.

| Item | Value |
|---|---|
| Search space | index `[0, n]` |
| `check(mid)` | `nums[mid] >= target` |
| Sentinel setup | One past the end |
| Boundary handling | Verify `nums[lo] == target` |

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
        lo, hi = 0, len(nums)

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] >= target:
                hi = mid
            else:
                lo = mid + 1

        if lo < len(nums) and nums[lo] == target:
            return lo
        return -1
```

</details>

### Search a 2D Matrix: Mapping a 2D Index to 1D

Each row is sorted ascending, and the first element of each row is greater than the last element of the previous row, so the flattened matrix is ascending overall. Map index `k` to `(k // n, k % n)`; everything else matches the classic binary search.

| Item | Value |
|---|---|
| Search space | flattened index `[0, m*n]` |
| `check(mid)` | `matrix[mid // n][mid % n] >= target` |
| Sentinel setup | One past the end |
| Boundary handling | Verify the value at the flattened position equals `target` |

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
        lo, hi = 0, m * n

        while lo < hi:
            mid = lo + (hi - lo) // 2
            row, col = divmod(mid, n)
            if matrix[row][col] >= target:
                hi = mid
            else:
                lo = mid + 1

        if lo < m * n:
            row, col = divmod(lo, n)
            return matrix[row][col] == target
        return False
```

</details>

### Koko Eating Bananas: The Search Space Is the Answer Range

This problem does not ask for a position inside an array. It asks for a boundary in the range of possible eating speeds. As speed increases, the number of hours needed to finish all piles does not increase, so the predicate "hours needed `<= h`" is monotonic in speed and the template applies directly.

$$
\text{hours\_needed(speed)} = \sum_{\text{pile}} \left\lceil \frac{\text{pile}}{\text{speed}} \right\rceil
$$

| Item | Value |
|---|---|
| Search space | speed `[1, max(piles)]` |
| `check(mid)` | `hours_needed(mid) <= h` |
| Sentinel setup | Always-true boundary |
| Boundary handling | Return boundary `b` directly |

#### Quick Coding: Koko Eating Bananas

```python
def minEatingSpeed(piles, h):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import math
from typing import List


class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def hours_needed(speed: int) -> int:
            return sum(math.ceil(pile / speed) for pile in piles)

        lo, hi = 1, max(piles)

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if hours_needed(mid) <= h:
                hi = mid
            else:
                lo = mid + 1

        return lo
```

</details>

The problem constraints guarantee `h >= len(piles)`. At speed `max(piles)`, each pile takes at most 1 hour, so the total is at most `h`. The always-true boundary holds without an extra case for it.

### Find Minimum in Rotated Sorted Array: A Predicate Without a Target

This problem has no `target`; the predicate has to come from the structure of the array itself. A rotated array is two ascending runs joined together: every value in the first run is greater than `nums[-1]`, and every value in the second run is less than or equal to `nums[-1]`. The predicate `check(mid) = nums[mid] <= nums[-1]` switches from `False` to `True` exactly at the join, and that boundary is the index of the minimum.

When the array is not rotated, `nums[0] <= nums[-1]` already holds, so the boundary lands at index `0` — no separate branch is needed for the unrotated case.

| Item | Value |
|---|---|
| Search space | index `[0, n-1]` |
| `check(mid)` | `nums[mid] <= nums[-1]` |
| Sentinel setup | Always-true boundary: `nums[n-1] <= nums[n-1]` |
| Boundary handling | Return `nums[b]` |

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
        lo, hi = 0, len(nums) - 1

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] <= nums[-1]:
                hi = mid
            else:
                lo = mid + 1

        return nums[lo]
```

</details>

### Search in Rotated Sorted Array: Linearizing with a Key Transform

This problem has both a rotation and a target value. Comparing `nums[mid]` directly against `target` is no longer monotonic, because the array is not sorted overall. The fix is to assign every value a key:

```text
key(x) = (x <= nums[-1], x)
```

The first run (values greater than `nums[-1]`) gets a key whose first component is `False`; the second run (values less than or equal to `nums[-1]`) gets `True`. Under tuple comparison, the entire first run sorts before the entire second run, and within each run the comparison falls back to the value itself. So `key(nums[i])` is strictly increasing in index `i`, matching the behavior of an unrotated array. `target` is assigned a key with the same rule, and the predicate becomes a key comparison.

| Item | Value |
|---|---|
| Search space | index `[0, n]` |
| `check(mid)` | `key(nums[mid]) >= key(target)` |
| Sentinel setup | One past the end |
| Boundary handling | Verify `nums[b] == target` |

#### Quick Coding: Search in Rotated Sorted Array

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
        pivot_value = nums[-1]

        def key(value: int):
            return (value <= pivot_value, value)

        target_key = key(target)
        lo, hi = 0, len(nums)

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if key(nums[mid]) >= target_key:
                hi = mid
            else:
                lo = mid + 1

        if lo < len(nums) and nums[lo] == target:
            return lo
        return -1
```

</details>

The problem guarantees all elements are distinct, so key comparisons never tie.

### Time Based Key-Value Store: The Last-False Reading

`set` writes with strictly increasing timestamps, so the records for a given `key` are already ordered. `get` asks for the last record whose timestamp does not exceed the query — the "last False" reading. The predicate `check(mid) = timestamps[mid] > query` finds the first position whose timestamp exceeds the query; the answer index is one position before that.

| Item | Value |
|---|---|
| Search space | index `[0, len(entries)]` |
| `check(mid)` | `entries[mid].timestamp > query` |
| Sentinel setup | One past the end |
| Boundary handling | Take `b - 1`; if `b == 0`, no record satisfies the condition |

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
        lo, hi = 0, len(entries)

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if entries[mid][0] > timestamp:
                hi = mid
            else:
                lo = mid + 1

        if lo == 0:
            return ""
        return entries[lo - 1][1]
```

</details>

### Median of Two Sorted Arrays: The Search Space Is a Partition Point

Here the search space is neither an array index nor an answer range; it is the position of a cut through the shorter array. Cutting both arrays so the left side holds `half = (m + n + 1) // 2` elements total, the predicate checks whether the right side of `A` is large enough:

$$
\text{check}(i) = A[i] \ge B[j-1], \quad j = \text{half} - i
$$

As `i` increases, `A[i]` (or `+inf` when out of range) does not decrease; as `j` decreases, `B[j-1]` (or `-inf` when out of range) does not increase. The predicate is monotonic in `i`, so it can be searched directly. Using `±inf` for out-of-range values means `i == m` or `j == 0` need no separate case.

Once boundary `i` is found, the maximum of the left side (`max_left`) and the minimum of the right side (`min_right`) are the two quantities the median is built from: when the total length is odd, the median is `max_left`; when even, it is the average of `max_left` and `min_right`.

| Item | Value |
|---|---|
| Search space | partition point `i ∈ [0, m]` (`m` is the length of the shorter array) |
| `check(mid)` | `A[mid] >= B[half - mid - 1]` (out-of-range values use `±inf`) |
| Sentinel setup | Always-true boundary: at `i == m`, the right side of `A` is `+inf` |
| Boundary handling | Compute the median from `max_left` and `min_right` at boundary `i` |

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

        lo, hi = 0, m
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if a_right_big_enough(mid):
                hi = mid
            else:
                lo = mid + 1

        i = lo
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

3-Step Fill-in Checklist:
1. **Blank 1 (Search Space)**: Array index `[0, n]` or answer range `[min, max]`?
2. **Blank 2 (Predicate)**: For "First" write directly; for "Last" find right neighbor and subtract 1; for answer ranges ask "is it viable"?
3. **Blank 3 (Return)**: Verify `nums[lo] == target` for exact matches; take `lo - 1` for last element.

Finally, remember:

> Binary search does not look for a value, but for a transition boundary; simply fill in Search Space, Predicate, and Return Value.


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
