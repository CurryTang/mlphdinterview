# Binary Search: A Unified Template

The classic binary search is not the hard part. The variants are: exact match, searching over an answer range, rotated arrays, and interval matching each handle the boundary differently. This note reduces seven problems to one template: find the boundary of a monotonic predicate. Only three pieces change between problems.

## Learning Order

Problems are drawn from the Binary Search module of [NeetCode 150](https://neetcode.io/practice/practice/neetcode150).

| Order | Problem | What to Master |
|---:|---|---|
| 1 | [704. Binary Search](https://neetcode.io/problems/binary-search/question?list=neetcode150) | The template's most basic form: exact match |
| 2 | [74. Search a 2D Matrix](https://neetcode.io/problems/search-a-2d-matrix/question?list=neetcode150) | Mapping a 2D index to a 1D index |
| 3 | [875. Koko Eating Bananas](https://neetcode.io/problems/koko-eating-bananas/question?list=neetcode150) | The search space is the answer range, not an array index |
| 4 | [153. Find Minimum in Rotated Sorted Array](https://neetcode.io/problems/find-minimum-in-rotated-sorted-array/question?list=neetcode150) | Defining a predicate without a target value |
| 5 | [33. Search in Rotated Sorted Array](https://neetcode.io/problems/find-target-in-rotated-sorted-array/question?list=neetcode150) | Linearizing a rotated array with a key transform |
| 6 | [981. Time Based Key-Value Store](https://neetcode.io/problems/time-based-key-value-store/question?list=neetcode150) | The "last False" reading |
| 7 | [4. Median of Two Sorted Arrays](https://neetcode.io/problems/median-of-two-sorted-arrays/question?list=neetcode150) | The search space is a partition point, using `±inf` as the sentinel |

## Module 1: The Unified Template

### Search Space and Predicate

Binary search locates a boundary in an interval `[lo, hi]`, together with a boolean predicate `check(x)` defined on that interval, not necessarily an array itself. The template requires `check` to be monotonic on the interval: there exists a boundary `b` such that `check(x)` is `False` for `x < b` and `True` for `x >= b`. The template returns `b`.

The interval does not have to be an array index; it can be a range of candidate answers, or the position of a partition point. The predicate does not have to compare against a target value; it only has to be monotonic.

### Template Code

```python
def find_first_true(lo, hi, check):
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if check(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

Loop invariant: the boundary `b` always lies within `[lo, hi]`. Each iteration shrinks `hi - lo` by at least half. The loop ends when `lo == hi`, and the return value is `b`.

If no position in `[lo, hi)` satisfies `check`, the function returns `hi`. This is not because `check(hi)` was called and returned `True` — `mid` is always strictly less than `hi`, so `hi` itself is never passed to `check`. In this case `hi` acts as a sentinel, standing for "not found" or "the answer lies outside the search space."

Only three things change across the seven problems:

```text
lo, hi: the two ends of the search space
check(mid): the definition of the monotonic predicate
what happens after finding boundary b: use it directly, verify equality, or take b - 1
```

```binary-search-template-demo
```

### Concrete Traces

#### Exact Match: `nums = [1, 3, 5, 7, 9, 11, 13]`

| Example | Iteration | `lo` | `hi` | `mid` | `check(mid)` | Action |
|---|---:|---:|---:|---:|---|---|
| `target = 9` | 1 | 0 | 7 | 3 | `nums[3] = 7 >= 9` → False | `lo = 4` |
| `target = 9` | 2 | 4 | 7 | 5 | `nums[5] = 11 >= 9` → True | `hi = 5` |
| `target = 9` | 3 | 4 | 5 | 4 | `nums[4] = 9 >= 9` → True | `hi = 4` |
| `target = 9` | Finish | 4 | 4 | - | boundary `b = 4` | verify `nums[4] == 9`, return `4` |
| `target = 6` | 1 | 0 | 7 | 3 | `nums[3] = 7 >= 6` → True | `hi = 3` |
| `target = 6` | 2 | 0 | 3 | 1 | `nums[1] = 3 >= 6` → False | `lo = 2` |
| `target = 6` | 3 | 2 | 3 | 2 | `nums[2] = 5 >= 6` → False | `lo = 3` |
| `target = 6` | Finish | 3 | 3 | - | boundary `b = 3` | verify `nums[3] = 7 != 6`, return `-1` |

#### Answer Range: `piles = [3, 6, 7, 11]`, `h = 8`

| Iteration | `lo` | `hi` | `mid` | `check(mid)` | Action |
|---|---:|---:|---:|---|---|
| 1 | 1 | 11 | 6 | `1 + 1 + 2 + 2 = 6 <= 8` → True | `hi = 6` |
| 2 | 1 | 6 | 3 | `1 + 2 + 3 + 4 = 10 <= 8` → False | `lo = 4` |
| 3 | 4 | 6 | 5 | `1 + 2 + 2 + 3 = 8 <= 8` → True | `hi = 5` |
| 4 | 4 | 5 | 4 | `1 + 2 + 2 + 3 = 8 <= 8` → True | `hi = 4` |
| Finish | 4 | 4 | - | boundary `b = 4` | return the minimum feasible speed `4` |

#### Last False: `timestamps = [1, 4, 7, 10]`, `query = 8`

| Iteration | `lo` | `hi` | `mid` | `check(mid)` | Action |
|---|---:|---:|---:|---|---|
| 1 | 0 | 4 | 2 | `timestamps[2] = 7 > 8` → False | `lo = 3` |
| 2 | 3 | 4 | 3 | `timestamps[3] = 10 > 8` → True | `hi = 3` |
| Finish | 3 | 3 | - | boundary `b = 3` | return `b - 1 = 2`, whose timestamp is `7` |

### Two Readings

| Answer needed | How to obtain it |
|---|---|
| First position that satisfies the condition | Return `b` directly |
| Last position that does not satisfy the condition | Return `b - 1`, and check `b > 0` |

### Four Standard Forms of `check(mid)`

The question to answer is "what is the first position satisfying which condition," not "should this be `>` or `>=`." Rephrasing the problem this way fixes the comparison operator.

Four boundaries map to four forms of `check(mid)`:

| Answer needed | `check(mid)` | Final answer |
|---|---|---|
| First `>= target` | `nums[mid] >= target` | `lo` |
| First `> target` | `nums[mid] > target` | `lo` |
| Last `<= target` | `nums[mid] > target` | `lo - 1` |
| Last `< target` | `nums[mid] >= target` | `lo - 1` |

Rule of thumb:

> For "first," write the condition directly.
> For "last," find the first position on its right, then subtract one.

Example with `nums = [1, 3, 3, 3, 5, 7]`:

First `>= 3`:

```text
check(mid) = nums[mid] >= 3
F T T T T T
  ↑ answer (index 1)
```

First `> 3`:

```text
check(mid) = nums[mid] > 3
F F F F T T
        ↑ answer (index 4)
```

Last `<= 3`: find first `> 3`, then subtract one.

```text
check(mid) = nums[mid] > 3
answer = lo - 1 = 3
```

Last `< 3`: find first `>= 3`, then subtract one.

```text
check(mid) = nums[mid] >= 3
answer = lo - 1 = 0
```

Exact-match search, the Binary Search problem itself, reads "find the first `>= target`, then verify":

```python
check(mid) = nums[mid] >= target

if lo < len(nums) and nums[lo] == target:
    return lo
return -1
```

`nums[mid] == target` cannot be used as `check` directly, because it is not monotonic over the array:

```text
False False True False False
```

The structure the template requires is:

```text
False False False | True True True
                  ↑ boundary
```

Rule of thumb: `>=` finds the left boundary, `>` finds the right boundary past any duplicates; subtract one from the corresponding boundary whenever "last" is needed.

### When the Direction Flips, the Comparisons Mirror

The table above assumes the compared quantity `f(mid)` (e.g. `nums[mid]`) is non-decreasing as `mid` grows. That holds for a sorted array, but not every problem satisfies it — in Koko Eating Bananas, `hours_needed(speed)` is non-increasing as `speed` grows (a higher speed takes no more time). When the direction reverses, the comparisons mirror as a whole:

| Direction of `f(mid)` | Answer needed | `check(mid)` | Final answer |
|---|---|---|---|
| Non-decreasing | First `>= target` | `f(mid) >= target` | `lo` |
| Non-decreasing | Last `<= target` | `f(mid) > target` | `lo - 1` |
| Non-increasing | First `<= target` | `f(mid) <= target` | `lo` |
| Non-increasing | Last `>= target` | `f(mid) < target` | `lo - 1` |

Koko falls in the "non-increasing, first `<= target`" row: `check(mid) = hours_needed(mid) <= h`, returned directly as `lo` — matching the form already used in the Koko subsection of Module 2.

The seven problems sorted into this table:

| Category | Problems |
|---|---|
| Non-decreasing + `>=`/`>` | Binary Search, Search a 2D Matrix, Time Based Key-Value Store |
| Non-increasing + `<=`/`<` | Koko Eating Bananas |
| Structural exception, not a value-vs-threshold comparison | Find Minimum in Rotated Sorted Array (compares against `nums[-1]`, not an external threshold), Search in Rotated Sorted Array (key transform), Median of Two Sorted Arrays (partition balance condition) |

The last category has no fixed comparison operator to plug in — `check(mid)` has to be constructed individually, following the derivation in each problem's Module 2 subsection.

### Two Ways to Set the Boundary Sentinel

| Setup | Meaning | When to use |
|---|---|---|
| One past the end | `hi` is set one position outside the search space, e.g. `len(nums)` | The target may not exist; a "not found" return value is needed |
| Always-true boundary | `hi` is set to a position inside the search space that is known to satisfy `check` | The predicate's truth at that position can be shown directly, with no extra case |

Setup for each of the seven problems:

| Problem | Search space | Sentinel setup |
|---|---|---|
| Binary Search | index `[0, n]` | One past the end |
| Search a 2D Matrix | flattened index `[0, m*n]` | One past the end |
| Koko Eating Bananas | speed `[1, max(piles)]` | Always-true boundary: at speed `max(piles)`, each pile takes at most 1 hour |
| Find Minimum in Rotated Sorted Array | index `[0, n-1]` | Always-true boundary: `nums[n-1] <= nums[n-1]` always holds |
| Search in Rotated Sorted Array | index `[0, n]` | One past the end |
| Time Based Key-Value Store | index `[0, len(timestamps)]` | One past the end |
| Median of Two Sorted Arrays | partition point `[0, m]` | Always-true boundary: the out-of-range side is replaced with `+inf` |

### Common Mistakes

| Issue | Effect |
|---|---|
| `check` is not monotonic over the search space | Binary search fails; the template's correctness argument no longer holds |
| `mid` computed as `lo + (hi - lo) // 2`, but the `False` branch omits `+ 1` | `lo` stops advancing; infinite loop |
| No equality check after finding the boundary | An exact-match problem treats "nearest position" as "position exists" |
| Sentinel set to one past the end, but the return value is used without a bounds check | The return value equals `hi`, which points to a position that does not exist |

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
\text{hours\_needed(speed)} = \sum_{\text{pile}} \left\lceil \frac{\text{pile}}{\text{speed}}  ightceil
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

        min_right = min(a_right, b_\right)
        return (max_left + min_\right) / 2
```

</details>

## Module 3: Final Checks Before an Interview

1. Is the search space an array index, an answer range, or a partition point?
2. What is `check(mid)`? Is it monotonic in `mid`?
3. Is the answer "the first True" or "the last False"?
4. Does `hi` use one-past-the-end, or an always-true boundary?
5. Does the boundary need verification after it is found, or can it be used directly?

One sentence to keep:

> Binary search does not locate a target value. It locates the boundary of a monotonic predicate.
