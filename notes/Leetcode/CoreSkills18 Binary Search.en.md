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

## Module 1: Unified Template & Minimal 3-Step Fill-in Method

### 1. The Core 6-Line Template

Binary search fundamentally searches for **the first point satisfying a predicate (First True)**. The skeleton code is always these 6 lines:

```python
def find_first_true(lo, hi, check):
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if check(mid):
            hi = mid      # Predicate satisfied; shrink left to find earlier candidates
        else:
            lo = mid + 1  # Not satisfied; eliminate left half
    return lo             # Loop exits with lo == hi: the exact First True boundary
```

```binary-search-template-demo
```

> **Loop Invariant**: Upon termination, `lo == hi` is strictly guaranteed, pointing precisely to **the first index where `check(mid)` is True**.

---

### 2. How to Fill the 3 Blanks (Minimal Mental Model)

In an interview, forget confusing mirror tables and sentinel categorizations. **Simply fill in 3 blanks in order**:

#### Step 1: Blank 1 — How to fill `[lo, hi]`?

- **Standard Array Index (finding a position)**: `lo = 0, hi = len(nums)`
  - *Why `hi = len(nums)`?* Because the target may not exist; `len(nums)` acts as a natural out-of-bounds sentinel meaning "not found".
- **Answer Value Range (finding min speed, capacity, etc.)**: `lo = min_possible, hi = max_possible`
  - *Why no +1?* Because the maximum possible value is guaranteed to satisfy the condition (a known True upper bound).

---

#### Step 2: Blank 2 — How to fill `check(mid)`? (The 4 Boundary Cheat Codes)

Candidates often struggle with `>` vs. `>=`. The core question to ask is always:
> **"Does the current `mid` already satisfy (or reach) the goal?"**
> - If yes (`True`), we found a valid candidate, but want to check if an even smaller/earlier candidate exists on the left: `hi = mid`.
> - If no (`False`), we must search to the right: `lo = mid + 1`.

For searching in a sorted array, remember this **foolproof rule**:

| Target to Find | How to write `check(mid)` | Final Answer | Rule of Thumb |
|---|---|---|---|
| **First $\ge target$** | `nums[mid] >= target` | `lo` | **For "First", write condition directly** |
| **First $> target$** | `nums[mid] > target` | `lo` | **For "First", write condition directly** |
| **Last $\le target$** | `nums[mid] > target` | `lo - 1` | **For "Last", find its right neighbor (First $>$), then subtract 1** |
| **Last $< target$** | `nums[mid] >= target` | `lo - 1` | **For "Last", find its right neighbor (First $\ge$), then subtract 1** |

> 💡 **Golden Rule**:
> 1. **To find "First"**: write the exact condition desired; return `lo`.
> 2. **To find "Last"**: find the "First" of the opposing right neighbor, then return `lo - 1`.

**What about inverse monotonic relationships (e.g., Koko Bananas, higher speed = less hours)?**
No mirror rule gymnastics needed! Just follow plain intuition:
- Ask: "Does speed `mid` finish within allowed hours?"
- Write: `hours_needed(mid) <= h`.
- If True, `hi = mid` to try smaller speeds; if False, `lo = mid + 1`. Done!

---

#### Step 3: Blank 3 — How to handle the return value?

When loop terminates (`lo == hi`), wrap up based on problem goal:
1. **Exact match query (LC 704, LC 74)**:
   - Check in-bounds and equality: `if lo < len(nums) and nums[lo] == target: return lo`, else return `-1`.
2. **Answer range / extreme value (LC 875, LC 153)**:
   - Directly return `lo` (or `nums[lo]`), which is the minimal viable solution.
3. **Last element matching query (LC 981)**:
   - Return `lo - 1` (if `lo == 0`, no valid record exists).

---

### 3. Seven Canonical Problems Fill-in Matrix

| Problem | Blank 1: `[lo, hi]` | Blank 2: `check(mid)` Predicate | Blank 3: Return Handling |
|---|---|---|---|
| **LC 704. Binary Search** | `0, len(nums)` | `nums[mid] >= target` | Verify `lo < n and nums[lo] == target`, else `-1` |
| **LC 74. Search a 2D Matrix** | `0, m * n` | `matrix[mid // n][mid % n] >= target` | Verify equality, else `False` |
| **LC 875. Koko Eating Bananas** | `1, max(piles)` | `hours_needed(mid) <= h` | Return `lo` directly |
| **LC 153. Find Min in Rotated Array** | `0, len(nums) - 1` | `nums[mid] <= nums[-1]` | Return `nums[lo]` directly |
| **LC 33. Search in Rotated Array** | `0, len(nums)` | Key transform / classic sorted-half | Verify `nums[lo] == target`, else `-1` |
| **LC 981. Time Based KV Store** | `0, len(entries)` | `entry[0] > timestamp` | Return `entries[lo - 1][1]` (or empty if `lo == 0`) |
| **LC 4. Median of Two Sorted Arrays** | `0, m` (shorter array) | `A[mid] >= B[half - mid - 1]` | Compute median from partition extremes |

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
