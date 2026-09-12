# Review Flashcards: Core Fundamentals (Review 1)

This module provides high-yield algorithm interview review flashcards: distilled **Problem Definitions**, **Core Mental Models**, **Minimal Core Implementations**, **Complexity Invariants**. Click any card title to expand.

---

## Module 1: Core Fundamental Algorithms

### 1. Merge Sort

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 01</span>
  <span class="review-card-title">Merge Sort</span>
  <span class="review-card-tag">[LeetCode 912 · Sort an Array](https://leetcode.com/problems/sort-an-array/) · Divide &amp; Conquer · Stable</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Sort an unsorted array of $n$ integers in non-decreasing order. Worst-case time complexity must be strictly guaranteed to be $O(n \log n)$, preserving the relative order of duplicate elements (stability).

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Canonical Divide & Conquer three-step pipeline:
1. **Divide**: Compute midpoint $mid = \lfloor (l + r) / 2 \rfloor$ to split into equal halves.
2. **Conquer**: Recursively sort left and right halves until subsegments reach base case length $\le 1$.
3. **Combine**: Linearly merge using two pointers; on ties, prefer left elements to guarantee stability.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def merge_sort(nums: list[int]) -> list[int]:
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left, right = merge_sort(nums[:mid]), merge_sort(nums[mid:])
    
    # Core two-pointer merge (<= ensures stability)
    res, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    return res + left[i:] + right[j:]

if __name__ == "__main__":
    assert merge_sort([5, 2, 3, 1]) == [1, 2, 3, 5]
    assert merge_sort([5, 1, 1, 2, 0, 0]) == [0, 0, 1, 1, 2, 5]
    assert merge_sort([]) == []
    assert merge_sort([42]) == [42]
    print("✅ Card 01 (Merge Sort) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: Best $O(n \log n)$ / Worst $O(n \log n)$ / Average $O(n \log n)$ (Tree height $\log n$, level merge work fixed at $O(n)$)
- **Auxiliary Space**: $O(n)$ (merge buffer) + $O(\log n)$ (call stack frames)
- **Stability**: **Stable** (left-half precedence on ties)

</div>

</div>
</details>

---

### 2. Quick Sort

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 02</span>
  <span class="review-card-title">Quick Sort</span>
  <span class="review-card-tag">[LeetCode 912 · Sort an Array](https://leetcode.com/problems/sort-an-array/) / [LeetCode 215](https://leetcode.com/problems/kth-largest-element-in-an-array/) · Partitioning · In-Place · Unstable</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Sort an unsorted array of $n$ integers in non-decreasing order in-place. Average time complexity must achieve $O(n \log n)$, requiring no auxiliary data structures beyond recursive stack frames.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Partitioning before Recursion**:
1. **Randomized Pivot**: Use `random.randint(l, r)` to pick a random element and swap it with the end $r$. This completely breaks adversarial inputs (e.g., sorted or reverse-sorted arrays), eliminating the $O(n^2)$ worst-case skew.
2. **Dedicated Partition Function**: Lomuto partitioning maintains pointer $i$ as the right boundary of elements $\le pivot$. Iterate across $[l, r - 1]$, swapping elements $\le pivot$ into place; finally swap pivot with $nums[i]$ and return split index $p = i$.
3. **Divide & Conquer Recursion**: Recursively sort subarrays around the pivot: `[l, p - 1]` and `[p + 1, r]`.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
import random

def partition(nums: list[int], l: int, r: int) -> int:
    # 1. Random pivot selection to prevent worst-case O(n^2) degeneration
    rand_idx = random.randint(l, r)
    nums[rand_idx], nums[r] = nums[r], nums[rand_idx]

    # 2. Core Lomuto partition: i maintains boundary of elements <= pivot
    pivot, i = nums[r], l
    for j in range(l, r):
        if nums[j] <= pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[r] = nums[r], nums[i]
    return i

def quick_sort(nums: list[int], l: int, r: int) -> None:
    if l >= r:
        return
    p = partition(nums, l, r)
    quick_sort(nums, l, p - 1)
    quick_sort(nums, p + 1, r)

if __name__ == "__main__":
    arr1 = [5, 2, 3, 1]
    quick_sort(arr1, 0, len(arr1) - 1)
    assert arr1 == [1, 2, 3, 5]
    arr2 = [5, 1, 1, 2, 0, 0]
    quick_sort(arr2, 0, len(arr2) - 1)
    assert arr2 == [0, 0, 1, 1, 2, 5]
    print("✅ Card 02 (Quick Sort) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: Best $O(n \log n)$ / Worst $O(n^2)$ (skewed partitions) / Average $O(n \log n)$
- **Auxiliary Space**: $O(\log n)$ (stack frames, degrades to $O(n)$ in worst case)
- **Stability**: **Unstable** (long-distance swaps disrupt relative order)

</div>

</div>
</details>

---

### 3. Dynamic Array Implementation

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 03</span>
  <span class="review-card-title">Dynamic Array Implementation</span>
  <span class="review-card-tag">[LeetCode 1929 · Concatenation of Array](https://leetcode.com/problems/concatenation-of-array/) / Design Vector · Contiguous Memory · Geometric Doubling · Amortized</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Implement a resizable dynamic array from scratch backed by a fixed-size contiguous buffer (analogous to Python `list` or C++ `std::vector`), supporting $O(1)$ random indexing, tail append `push_back`, tail pop `pop_back`, and automatic doubling expansion.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Mental model and amortized constant time rationale:
1. **Contiguous Buffer**: Maintain fixed capacity `cap` with active item count `size`.
2. **Geometric Doubling**: When `size == cap`, allocate a new contiguous chunk of $2 \times cap$, copy elements across, and discard old buffer.
3. **Amortized Analysis ($O(1)$)**: A single expansion copies $O(n)$ elements, but occurs exponentially less often. Sum of all copies $1 + 2 + 4 + \dots + n \le 2n$. Amortized over $n$ appends, cost is strictly $O(1)$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
class DynamicArray:
    def __init__(self, capacity: int = 2):
        self.cap, self.size = capacity, 0
        self.arr = [None] * self.cap

    def push_back(self, val: int) -> None:
        # Core: Geometric doubling when full, amortized O(1)
        if self.size == self.cap:
            self.cap *= 2
            new_arr = [None] * self.cap
            for i in range(self.size):
                new_arr[i] = self.arr[i]
            self.arr = new_arr
        self.arr[self.size] = val
        self.size += 1

    def pop_back(self) -> int:
        self.size -= 1
        return self.arr[self.size]

    def get(self, i: int) -> int:
        return self.arr[i]

if __name__ == "__main__":
    arr = DynamicArray(capacity=2)
    arr.push_back(10)
    arr.push_back(20)
    assert arr.get(0) == 10 and arr.get(1) == 20
    assert arr.cap == 2
    arr.push_back(30)
    assert arr.cap == 4
    assert arr.pop_back() == 30
    assert arr.size == 2
    print("✅ Card 03 (Dynamic Array) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Random Indexing `get`/`set`**: $O(1)$ (direct memory address calculation $base + i \times size$)
- **Tail Append `push_back`**: **Amortized $O(1)$** (Worst $O(n)$ during expansion)
- **Tail Pop `pop_back`**: $O(1)$
- **Space Utilization**: $\ge 50\%$

</div>

</div>
</details>

---

### 4. Binary Search Boundary Template

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 04</span>
  <span class="review-card-title">Binary Search Boundary Template</span>
  <span class="review-card-tag">[LeetCode 704 · Binary Search](https://leetcode.com/problems/binary-search/) / [LeetCode 34](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) · Monotonic Search · Interval Invariant</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given a non-decreasing sorted integer array, locate the **first occurrence (Lower Bound)** of `target`. If absent, return the index where it should be inserted. Must run in $O(\log n)$ with zero danger of infinite loops.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Rigid **Loop Invariant maintenance**:
1. **Closed Interval**: Maintain a **closed range $[l, r]$**, initialized with $l = 0, r = len(nums) - 1$.
2. **Overflow-safe Midpoint**: $mid = l + \lfloor (r - l) / 2 \rfloor$.
3. **Shrinking Decision**:
   - If $nums[mid] \ge target$: Target is at $mid$ or left; shrink right bound: $r = mid - 1$.
   - If $nums[mid] < target$: Target is strictly right; shrink left bound: $l = mid + 1$.
4. **Convergence**: Loop while $l \le r$. Terminates strictly when $l = r + 1$, where $l$ lands on the first item $\ge target$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def search_lower_bound(nums: list[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    # Strictly maintain closed interval [l, r]
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] >= target:
            r = mid - 1  # Seek lower index to the left
        else:
            l = mid + 1
    return l  # Terminates with l as first index >= target

if __name__ == "__main__":
    nums = [1, 2, 2, 2, 3, 5]
    assert search_lower_bound(nums, 2) == 1
    assert search_lower_bound(nums, 3) == 4
    assert search_lower_bound(nums, 4) == 5
    assert search_lower_bound(nums, 0) == 0
    assert search_lower_bound(nums, 6) == 6
    print("✅ Card 04 (Binary Search Bounds) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(\log n)$ (halves search space every iteration)
- **Auxiliary Space**: $O(1)$ (iterative without stack frames)
- **Termination Invariant**: Loop always terminates with $l = r + 1$

</div>

</div>
</details>

---

### 5. Rejection Sampling: Implement Rand10 Using Rand7

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 05</span>
  <span class="review-card-title">Rejection Sampling (Rand7 to Rand10)</span>
  <span class="review-card-tag">[LeetCode 470 · Implement Rand10() Using Rand7()](https://leetcode.com/problems/implement-rand10-using-rand7/) · Grid Flattening · Divisible Prefix · Expected 2.45 Calls</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given an API `rand7()` that returns a uniform random integer in $1 \dots 7$, implement `rand10()` to return a uniform random integer in $1 \dots 10$. Using external random libraries is strictly forbidden, and every integer from 1 to 10 must be generated with exact probability $1/10$.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Multidimensional Grid Flattening + Rejection Sampling**:
1. **Space Construction (Grid Flattening)**: A single `rand7()` call produces only 7 outcomes, insufficient for 10. Invoking `rand7()` twice constructs a $7 \times 7 = 49$ independent, uniformly distributed 2D grid:
   $$x = (rand7() - 1) \times 7 + rand7() \in [1, 49]$$
   Each cell occurs with exact probability $\frac{1}{7} \times \frac{1}{7} = \frac{1}{49}$.
2. **Why Direct Modulo Fails**: 49 is not divisible by 10. Direct modulo would map outcomes $1 \dots 9$ five times ($5/49$), but outcome 10 only four times ($4/49$), violating equiprobability.
3. **Accept Largest Divisible Prefix (Rejection Sampling)**:
   - Accept only the prefix $1 \dots 40$ ($40$ is the largest multiple of 10 $\le 49$). Each outcome from 1 to 10 is mapped to exactly 4 cells, giving acceptance probability $\frac{40}{49}$, mapped uniformly via `(x - 1) % 10 + 1`.
   - If $x \in [41, 49]$ (9 states), discard (reject) and retry in the next loop iteration.
4. **General Template ($randM \to randN$)**: Pick the smallest $k$ such that $M^k \ge N$, generate $1 \dots M^k$, set $limit = \lfloor M^k / N \rfloor \times N$, accept if $\le limit$, otherwise reject.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def rand10() -> int:
    while True:
        # 1. Two rand7() calls construct uniform discrete space [1, 49]
        x = (rand7() - 1) * 7 + rand7()
        # 2. Accept largest prefix divisible by 10: [1, 40] (4 cells each, strictly uniform)
        if x <= 40:
            return (x - 1) % 10 + 1
        # 41..49 rejected, retry

if __name__ == "__main__":
    import random
    def rand7():
        return random.randint(1, 7)
    samples = [rand10() for _ in range(20000)]
    for num in range(1, 11):
        freq = samples.count(num)
        assert 1600 <= freq <= 2400
    print("✅ Card 05 (Rejection Sampling Rand7->Rand10) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Expected Time**: $O(1)$, average calls to `rand7()` = $2 \times \frac{49}{40} = 2.45$ (Geometric distribution expectation $1/p$)
- **Worst-Case Time**: $O(\infty)$ (theoretically infinite rejection path with probability 0)
- **Auxiliary Space**: $O(1)$ (in-place scalar computation, no extra memory)
- **Core Mental Model**: Flatten into a grid; discard the non-divisible remainder

</div>

</div>
</details>

---

## Module 2: Arrays & Hashing Core Problem Group

### 6. Contains Duplicate

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Array 01</span>
  <span class="review-card-title">Contains Duplicate</span>
  <span class="review-card-tag">[LeetCode 217 · Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) · Hash Set · Early Exit · Single Pass</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given an integer array `nums`, return `True` if any value appears at least twice in the array, and return `False` if every element is distinct.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Real-Time Set Probing & Early Exit**:
1. **State Initialization**: Maintain an empty hash set `seen = set()`.
2. **Streaming Probing**: Iterate over each element $x$ in `nums`:
   - If $x \in seen$, a duplicate is detected; immediately return `True` to terminate execution early;
   - If $x \notin seen$, insert $x$ into `seen`.
3. **Fallthrough**: If the loop terminates without triggering early exit, all elements are unique; return `False`.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def contains_duplicate(nums: list[int]) -> bool:
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False

if __name__ == "__main__":
    assert contains_duplicate([1, 2, 3, 1]) is True
    assert contains_duplicate([1, 2, 3, 4]) is False
    assert contains_duplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]) is True
    print("✅ Card 06 (Contains Duplicate) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(n)$ (Hash set lookups and insertions average $O(1)$; worst case $O(n)$ under pathological hash collisions)
- **Auxiliary Space**: $O(n)$ (In the worst case with no duplicates, set holds $n$ elements)
- **Key Mental Model**: Space-for-time trade-off; stream probing with immediate early termination

</div>

</div>
</details>

---

### 7. Valid Anagram

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Array 02</span>
  <span class="review-card-title">Valid Anagram</span>
  <span class="review-card-tag">[LeetCode 242 · Valid Anagram](https://leetcode.com/problems/valid-anagram/) · Frequency Array · ASCII Delta · Length Pruning</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given two strings `s` and `t`, return `True` if `t` is an anagram of `s`, and `False` otherwise (composed of identical characters with identical frequencies, permuted in any order).

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Character Count Zero-Sum Balancing**:
1. **Length Pruning**: If `len(s) != len(t)`, total characters do not match; immediately return `False`.
2. **Fixed Frequency Array**: For lowercase English letters, allocate a fixed-size integer array `counts = [0] * 26`.
3. **Dual Counter Balance**: Iterate simultaneously over both strings using `zip(s, t)`. Increment `counts[ord(c1) - ord('a')]` for $s$ and decrement `counts[ord(c2) - ord('a')]` for $t$.
4. **Zero-Residual Invariant**: Inspect `counts`. If every frequency bin equals 0, return `True`; any non-zero bin indicates a frequency mismatch, returning `False`.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def is_anagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    counts = [0] * 26
    for c1, c2 in zip(s, t):
        counts[ord(c1) - ord('a')] += 1
        counts[ord(c2) - ord('a')] -= 1
    return all(c == 0 for c in counts)

if __name__ == "__main__":
    assert is_anagram("anagram", "nagaram") is True
    assert is_anagram("rat", "car") is False
    assert is_anagram("a", "ab") is False
    print("✅ Card 07 (Valid Anagram) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(n)$ ($n$ is string length; single synchronous pass)
- **Auxiliary Space**: $O(1)$ (Fixed 26-element array; for arbitrary Unicode characters, a hash map takes $O(k)$ space where $k$ is the alphabet size)
- **Key Mental Model**: Add-subtract zero-sum balancing; anagram equivalence requires zero residual count

</div>

</div>
</details>

---

### 8. Two Sum

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Array 03</span>
  <span class="review-card-title">Two Sum</span>
  <span class="review-card-tag">[LeetCode 1 · Two Sum](https://leetcode.com/problems/two-sum/) · Hash Map · Complement Matching · Single Pass</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`. Assume exactly one solution exists, and each element may not be used twice.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Prefix Inverted Index & Complement Matching**:
1. **Complement Inverted Index**: For current element $x = nums[i]$, its required matching pair value is $\text{complement} = target - x$.
2. **Single-Pass Hash Map**: Maintain dictionary `lookup` mapping `value -> historical index`.
   - Before inserting the current number, check if `complement` is already in `lookup`:
     - If hit: return `[lookup[complement], i]`;
     - If missed: record `lookup[x] = i` and proceed.
3. **Self-Pairing Prevention**: Probing strictly against historical prefixes ensures an element cannot match with itself.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    lookup = {}
    for i, x in enumerate(nums):
        complement = target - x
        if complement in lookup:
            return [lookup[complement], i]
        lookup[x] = i
    return []

if __name__ == "__main__":
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]
    print("✅ Card 08 (Two Sum) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(n)$ (Single linear pass; hash map insertions and lookups average $O(1)$)
- **Auxiliary Space**: $O(n)$ (Stores up to $n - 1$ historical elements and indices)
- **Key Mental Model**: Store historical prefixes; future elements close the complement equation

</div>

</div>
</details>

---

### 9. Group Anagrams

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Array 04</span>
  <span class="review-card-title">Group Anagrams</span>
  <span class="review-card-tag">[LeetCode 49 · Group Anagrams](https://leetcode.com/problems/group-anagrams/) · Frequency Tuple · Canonical Hash Key · Grouping</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given an array of strings `strs`, group the anagrams together in any order and return as a list of lists.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Multiset Canonical Key Vectorization**:
1. **Invariant Feature Key**: Anagrams share identical character counts. Map every string to an invariant, hashable canonical key:
   - Sorted key: `"".join(sorted(s))` costs $O(k \log k)$;
   - **Frequency Tuple Key (Optimal)**: Count frequencies across 26 letters and convert into an immutable tuple `tuple(counts)`. Per-string cost drops to strictly linear $O(k)$.
2. **Multi-to-One Hash Grouping**:
   - Use `collections.defaultdict(list)` with the 26-element tuple as the dictionary key, appending matching original strings to the corresponding bucket.
   - Return all values of the dictionary.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        # Build 26-element character count tuple as immutable hash key
        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord('a')] += 1
        groups[tuple(counts)].append(s)
    return list(groups.values())

if __name__ == "__main__":
    res = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    sorted_res = sorted([sorted(g) for g in res])
    assert sorted_res == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert group_anagrams([""]) == [[""]]
    assert group_anagrams(["a"]) == [["a"]]
    print("✅ Card 09 (Group Anagrams) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(n \cdot k)$ ($n$ is the number of strings, $k$ is maximum string length; counting takes $O(k)$)
- **Auxiliary Space**: $O(n \cdot k)$ (Stores all grouped strings and tuple keys inside the hash map)
- **Key Mental Model**: Canonical count tuple standardization; multi-to-one aggregation

</div>

</div>
</details>

---

### 10. Top K Frequent Elements

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Array 05</span>
  <span class="review-card-title">Top K Frequent Elements</span>
  <span class="review-card-tag">[LeetCode 347 · Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) · Bucket Sort · Inverted Index · Linear Time</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given an integer array `nums` and an integer `k`, return the `k` most frequent elements in any order. Algorithm time complexity must be strictly better than $O(n \log n)$.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Inverted Frequency Indexing via Bucket Sort**:
1. **Frequency Counting**: Compute frequencies using `Counter(nums)`. Let distinct elements count be $m \le n$.
2. **Complexity Analysis Across Paradigms**:
   - Full sorting: $O(m \log m)$;
   - Min-Heap of size $k$: $O(n + m \log k)$;
   - **Bucket Sort / Inverted Index (Strict Linear Optimal)**:
     - Frequencies are strictly bounded by array length $n$;
     - Allocate $n + 1$ buckets `buckets = [[] for _ in range(n + 1)]`, where index $i$ stores all elements appearing exactly $i$ times;
     - Append each `(num, freq)` into `buckets[freq]`.
3. **Reverse Greedy Collection**:
   - Iterate buckets descending from index $n$ down to 1. Append elements to result until $k$ elements are gathered.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    counts = Counter(nums)
    # Bucket index represents frequency, range 0..len(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in counts.items():
        buckets[freq].append(num)
        
    res = []
    # Collect top-k elements descending from highest possible frequency
    for freq in range(len(nums), 0, -1):
        for num in buckets[freq]:
            res.append(num)
            if len(res) == k:
                return res
    return res

if __name__ == "__main__":
    res1 = top_k_frequent([1, 1, 1, 2, 2, 3], 2)
    assert set(res1) == {1, 2}
    assert top_k_frequent([1], 1) == [1]
    print("✅ Card 10 (Top K Frequent Elements) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(n)$ (Counting $O(n)$, bucket population $O(m)$, reverse scan bounded by $O(n)$; strictly linear)
- **Auxiliary Space**: $O(n)$ (Counter hash map and $n + 1$ frequency buckets)
- **Key Mental Model**: Bounded frequencies enable inverted indexing, bypassing comparison sort limits

</div>

</div>
</details>

---

### 11. Encode and Decode Strings

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Array 06</span>
  <span class="review-card-title">Encode and Decode Strings</span>
  <span class="review-card-tag">[LeetCode 271 · Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/) · Length Prefix Protocol · Chunked Stream · Delimiter Independence</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Design an algorithm to encode a list of strings to a single composite string and decode that string back to the original list of strings. Strings may contain any valid ASCII/Unicode characters (including delimiters, punctuation, and newlines).

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Length-Prefix Framing Protocol**:
1. **Delimiter Failure Trap**: Using fixed delimiters (e.g., `,`, `#`) causes parsing failure when original strings contain those characters; escaping schemes add fragile edge cases.
2. **Length-Prefix Protocol Framing**:
   - Follow chunked streaming protocols (e.g., HTTP chunked transfer or Redis RESP protocol):
   - **Encoding Format**: `"{length}#{payload}"`. E.g., `["neet", "code"] -> "4#neet4#code"`.
   - **Decoding Process**:
     - Maintain cursor $i$. Find the next delimiter `'#'` at index $j$;
     - Parse slice `s[i:j]` as integer $L$, defining exact payload length;
     - Slice out payload `s[j + 1 : j + 1 + L]`;
     - Advance cursor $i = j + 1 + L$ and repeat.
3. **Delimiter Independence**: Any `'#'` inside payloads is ignored because the parser jumps strictly by length $L$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
class Codec:
    def encode(self, strs: list[str]) -> str:
        # Protocol framing: length + '#' + payload
        res = []
        for s in strs:
            res.append(f"{len(s)}#{s}")
        return "".join(res)

    def decode(self, s: str) -> list[str]:
        res, i = [], 0
        while i < len(s):
            # Locate delimiter immediately following length prefix
            j = s.find('#', i)
            length = int(s[i:j])
            # Extract exact payload slice based on length
            res.append(s[j + 1 : j + 1 + length])
            i = j + 1 + length
        return res

if __name__ == "__main__":
    codec = Codec()
    test_cases = [
        ["lint", "code", "love", "you"],
        [""],
        [],
        ["hello#world", "123#456", "##"],
        ["a" * 100, "b" * 50]
    ]
    for tc in test_cases:
        encoded = codec.encode(tc)
        decoded = codec.decode(encoded)
        assert decoded == tc, f"Failed on {tc}: got {decoded}"
    print("✅ Card 11 (Encode and Decode Strings) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: Encode $O(N)$, Decode $O(N)$ ($N$ is total character count; single linear parse)
- **Auxiliary Space**: $O(1)$ (Beyond input and output buffers, only pointers are tracked)
- **Key Mental Model**: Metadata framing precedes payload; length contracts guarantee delimiter immunity

</div>

</div>
</details>

---

### 12. Product of Array Except Self

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Array 07</span>
  <span class="review-card-title">Product of Array Except Self</span>
  <span class="review-card-tag">[LeetCode 238 · Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) · Prefix &amp; Suffix Decomposition · Two-Pass In-Place · O(1) Auxiliary Space</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given an integer array `nums`, return an array `res` such that `res[i]` equals the product of all elements of `nums` except `nums[i]`. **Division operations are strictly forbidden**, and time complexity must be $O(n)$. Follow-up: achieve $O(1)$ auxiliary space (excluding the output array).

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Prefix-Suffix Decomposition & Two-Pass In-Place Traversal**:
1. **Orthogonal Decomposition**: The product of all elements excluding index $i$ equals the product of all elements to its left multiplied by the product of all elements to its right:
   $$res[i] = \left(\prod_{j=0}^{i-1} nums[j]\right) \times \left(\prod_{j=i+1}^{n-1} nums[j]\right)$$
2. **Two-Pass In-Place Traversal ($O(1)$ Auxiliary Space)**:
   - **Forward Pass (Prefix Accumulation)**:
     - Initialize `res = [1] * n` and scalar `prefix = 1`.
     - Iterate $i$ from 0 to $n - 1$: assign `res[i] = prefix`, then update `prefix *= nums[i]`.
   - **Backward Pass (Suffix Rolling Product)**:
     - Maintain scalar `postfix = 1`.
     - Iterate $i$ from $n - 1$ down to 0: multiply `res[i] *= postfix`, then update `postfix *= nums[i]`.
3. **Space Optimization**: Storing prefixes in `res` and streaming suffixes through a single scalar variable avoids allocating auxiliary prefix/suffix arrays.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    res = [1] * n
    
    # 1. Forward pass: compute left prefixes into res[i]
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
        
    # 2. Backward pass: accumulate right postfixes via scalar
    postfix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= postfix
        postfix *= nums[i]
        
    return res

if __name__ == "__main__":
    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
    print("✅ Card 12 (Product of Array Except Self) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(n)$ (Two linear passes; exactly $2n$ multiplications)
- **Auxiliary Space**: $O(1)$ (Output array excluded; uses only `prefix` and `postfix` scalar accumulators)
- **Key Mental Model**: Orthogonal left-right prefix separation; write forward, roll backward

</div>

</div>
</details>

---

### 13. Valid Sudoku

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Array 08</span>
  <span class="review-card-title">Valid Sudoku</span>
  <span class="review-card-tag">[LeetCode 36 · Valid Sudoku](https://leetcode.com/problems/valid-sudoku/) · Row/Col/Box Validation · Grid Coordinate Flattening · O(1) Bound</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Determine if a $9 \times 9$ Sudoku board is valid according to the following rules (empty cells are marked `'.'`):
1. Each row must contain digits 1-9 without repetition.
2. Each column must contain digits 1-9 without repetition.
3. Each of the nine $3 \times 3$ sub-boxes must contain digits 1-9 without repetition.
*Note: Only existing filled digits need validation; board solvability is not required.*

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Sub-Box Coordinate Mapping & Three-Way Set Validation**:
1. **Sub-Box Coordinate Mapping**:
   - The $9 \times 9$ grid contains 9 sub-boxes.
   - For cell $(r, c)$, its sub-box coordinate is $(r // 3, c // 3)$, flattened into 1D index:
     $$box\_idx = (r // 3) \times 3 + (c // 3) \in [0, 8]$$
2. **Three-Way Set Validation**:
   - Maintain 9 row sets, 9 column sets, and 9 sub-box sets.
   - Iterate through all 81 cells once. Skip `'.'`.
   - For digit `val`, check if it already exists in `rows[r]`, `cols[c]`, or `boxes[box_idx]`. If any condition matches, collision detected; return `False`.
   - Otherwise, insert `val` into all three sets.
3. Return `True` after full grid scan.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def is_valid_sudoku(board: list[list[str]]) -> bool:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == '.':
                continue
            box_idx = (r // 3) * 3 + (c // 3)
            # Parallel check across row, column, and 3x3 sub-box
            if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                return False
            rows[r].add(val)
            cols[c].add(val)
            boxes[box_idx].add(val)
    return True

if __name__ == "__main__":
    board = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    assert is_valid_sudoku(board) is True
    board[0][0] = "8"
    assert is_valid_sudoku(board) is False
    print("✅ Card 13 (Valid Sudoku) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(1)$ (Fixed $9 \times 9 = 81$ cells scanned)
- **Auxiliary Space**: $O(1)$ ($3 \times 9 = 27$ sets with at most 9 elements each)
- **Key Mental Model**: $(r//3) \times 3 + (c//3)$ mapping flattens 2D sub-boxes; three sets check orthogonal constraints concurrently

</div>

</div>
</details>

---

### 14. Longest Consecutive Sequence

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Array 09</span>
  <span class="review-card-title">Longest Consecutive Sequence</span>
  <span class="review-card-tag">[LeetCode 128 · Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) · Hash Set · Predecessor Probing · Strict O(n)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence. The algorithm must run in strictly $O(n)$ time.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Predecessor Probing & Unidirectional Chain Expansion**:
1. **Sorting Overhead Incompatible**: Sorting takes $O(n \log n)$, which violates the strict $O(n)$ constraint.
2. **Hash Set for $O(1)$ Lookups**:
   - Convert `nums` to `num_set = set(nums)` to deduplicate and allow constant-time existence queries.
3. **Predecessor Probing Pruning (Sequence Start Detection)**:
   - Iterate through each number $x \in num\_set$:
   - **Pruning Invariant**: Check if $x - 1$ exists in `num_set`:
     - If $x - 1 \in num\_set$: $x$ is not the sequence head (a smaller consecutive predecessor exists); **skip immediately**.
     - If $x - 1 \notin num\_set$: $x$ is guaranteed to be the **unique starting anchor** of a consecutive chain.
  4. **Unidirectional Chain Expansion**:
     - Only when $x$ is an anchor, launch a `while` loop probing $x + 1, x + 2, \dots$ until the chain breaks, updating `max_len`.
  5. **Strict $O(n)$ Runtime Guarantee**:
     - Each number is checked once in the outer loop for predecessor existence, and touched at most once within a `while` loop across the entire run. Total operations $\le 2n$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def longest_consecutive(nums: list[int]) -> int:
    num_set = set(nums)
    max_len = 0
    
    for x in num_set:
        # Pruning invariant: only probe when x is the start of a sequence
        if x - 1 not in num_set:
            current_num = x
            current_len = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_len += 1
                
            max_len = max(max_len, current_len)
            
    return max_len

if __name__ == "__main__":
    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
    assert longest_consecutive([]) == 0
    print("✅ Card 14 (Longest Consecutive Sequence) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(n)$ (Set construction $O(n)$; sequence head pruning ensures the inner loop runs at most $n$ times total)
- **Auxiliary Space**: $O(n)$ (Hash set storing unique numbers)
- **Key Mental Model**: Only initiate exploration from predecessor-less anchors, avoiding redundant linear sweeps

</div>

</div>
</details>



### 15. Majority Element in Sorted Array via Sublinear Binary Search Probe

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Card 15</span>
  <span class="review-card-title">Majority Element in Sorted Array via Sublinear Binary Search Probe</span>
  <span class="review-card-tag">[LeetCode 229 · Majority Element II](https://leetcode.com/problems/majority-element-ii/) / [LeetCode 1150](https://leetcode.com/problems/check-if-a-number-is-majority-element-in-a-sorted-array/) · Sorted Array · Pigeonhole Principle · Probe Sampling · Binary Search Boundaries · O(log N) Sublinear</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Requirements</div>

Given a **sorted array** `nums` with length $n \ge 3$, return all numbers that appear strictly more than $\lfloor n / 3 \rfloor$ times:

```python
def findMajorityElementsSorted(nums: List[int]) -> List[int]: ...
```

- **Linear baseline**: Scan adjacent runs in $\mathcal{O}(n)$.
- **Sublinear follow-up**: Exploit the sorted property to achieve strictly sublinear **$\mathcal{O}(\log n)$ time complexity**.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Logarithmic Probe Invariants</div>

#### 1. Pigeonhole Principle & Candidate Reduction
- By the Pigeonhole Principle, at most 2 distinct elements can appear $> n/3$ times in any array.
- In a sorted array, any run of length $> n/3$ must span across at least one of the two probe coordinates:
  $$idx_1 = \lfloor n / 3 \rfloor, \qquad idx_2 = \lfloor 2n / 3 \rfloor$$
- Candidate set is reduced to $\{nums[idx_1], nums[idx_2]\}$ (at most 2 candidates).

#### 2. Exact Frequency via Binary Search
- For each unique candidate $v$:
  - Find first and last occurrences via `bisect_left` and `bisect_right`.
  - $\operatorname{freq}(v) = \operatorname{bisect\_right} - \operatorname{bisect\_left}$.
  - Verification takes $2 \times \mathcal{O}(\log n)$. Total time is strictly $\mathcal{O}(\log n)$!

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from bisect import bisect_left, bisect_right
from typing import List

class SortedMajoritySolution:

    @staticmethod
    def findMajorityElementsSorted(nums: List[int]) -> List[int]:
        n = len(nums)
        if n < 3:
            threshold = n // 3
            return [x for x in set(nums) if nums.count(x) > threshold]

        threshold = n // 3
        probe_indices = [n // 3, (2 * n) // 3]
        candidates = set(nums[i] for i in probe_indices)

        res = []
        for cand in sorted(list(candidates)):
            left = bisect_left(nums, cand)
            right = bisect_right(nums, cand)
            if right - left > threshold:
                res.append(cand)

        return res

if __name__ == "__main__":
    assert SortedMajoritySolution.findMajorityElementsSorted([1, 1, 2, 3, 4]) == [1]
    assert SortedMajoritySolution.findMajorityElementsSorted([1, 1, 2, 4, 4]) == [1, 4]
    assert SortedMajoritySolution.findMajorityElementsSorted([1, 2, 3]) == []
    assert SortedMajoritySolution.findMajorityElementsSorted([1, 2, 3, 4, 5, 6, 7]) == []
    print("✅ Card 15 (Sorted Array Majority Probe) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(\log n)$.
- **Space Complexity**: $\mathcal{O}(1)$.
- **Critical Pitfalls**: Forgetting to deduplicate candidate set when both probe points hit the same long sequence.

</div>

</div>
</details>

---

### 16. Reverse Words with Exact Spacing Preservation & In-Place Semantics

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Card 16</span>
  <span class="review-card-title">Reverse Words with Exact Spacing Preservation & In-Place Semantics</span>
  <span class="review-card-tag">[LeetCode 151 · Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/) / [LeetCode 186](https://leetcode.com/problems/reverse-words-in-a-string-ii/) · Two Pointers · In-Place Dual Reversal · Exact Whitespace Preservation · O(1) Auxiliary</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Follow-ups</div>

Given a string `s`, reverse the order of words:
- **Variant 1 (LC 151)**: Trim multiple spaces down to single space.
- **Variant 2 (Preserve Spacing)**: **Exact spacing preservation**: Reverses word order while keeping the original space runs intact between word positions.
- **Variant 3 (In-place C++)**: Strictly $\mathcal{O}(1)$ extra space on mutable character arrays.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Two-Pointer Mechanics</div>

- **Preserve Spacing**: Separate string into alternating tokens of space-blocks and word-blocks. Reverse the word list, then interleave back into the static space positions.
- **In-Place Dual Reversal**: Reverse entire string array in-place, then reverse each word in-place.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List

class ReverseWordsSolution:

    @staticmethod
    def reverseWordsPreserveSpacing(s: str) -> str:
        words = []
        tokens = []
        i, n = 0, len(s)

        while i < n:
            if s[i] == ' ':
                j = i
                while j < n and s[j] == ' ': j += 1
                tokens.append(s[i:j])
                i = j
            else:
                j = i
                while j < n and s[j] != ' ': j += 1
                word = s[i:j]
                tokens.append(word)
                words.append(word)
                i = j

        words.reverse()
        word_idx = 0
        res = []
        for token in tokens:
            if token.startswith(' '):
                res.append(token)
            else:
                res.append(words[word_idx])
                word_idx += 1

        return "".join(res)

    @staticmethod
    def reverseWordsInPlace(chars: List[str]) -> None:
        def reverse_sub(l: int, r: int):
            while l < r:
                chars[l], chars[r] = chars[r], chars[l]
                l += 1
                r -= 1

        n = len(chars)
        reverse_sub(0, n - 1)
        start = 0
        while start < n:
            if chars[start] == ' ':
                start += 1
                continue
            end = start
            while end < n and chars[end] != ' ':
                end += 1
            reverse_sub(start, end - 1)
            start = end

if __name__ == "__main__":
    assert ReverseWordsSolution.reverseWordsPreserveSpacing("  hello   world  ") == "  world   hello  "
    assert ReverseWordsSolution.reverseWordsPreserveSpacing("the sky is blue") == "blue is sky the"
    chars = list("the sky is blue")
    ReverseWordsSolution.reverseWordsInPlace(chars)
    assert "".join(chars) == "blue is sky the"
    print("✅ Card 16 (Reverse Words Exact Spacing) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(N)$ for string preservation; $\mathcal{O}(1)$ for mutable array in-place reversal.

</div>

</div>
</details>

---

### 17. Local Maximum on a 1-D Stream with Boundary Degradation

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">ARRAY 17</span>
  <span class="review-card-title">Local Maximum on a 1-D Stream with Boundary Degradation</span>
  <span class="review-card-tag">Stream Signal Processing / Local Peak Detection · Stream Signal Processing / Local Peak Detection · Bidirectional Monotonicity · Boundary Fallback · Sliding Neighborhood · O(N * K)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class LocalMaximaStreamSolution:
    @classmethod
    def findLocalMaxima(cls, rawData: List[float], localArea: int) -> List[int]:
        """
        Returns all indices i in a 1-D stream where localArea neighbors on each flank
        form strictly decreasing subsequences moving away from i.
        
        Boundary fallback: if fewer than localArea neighbors exist on a side,
        validate against all available neighbors on that side.
        """
        n = len(rawData)
        if n == 0:
            return []

        result = []

        for i in range(n):
            is_peak = True

            # 1. Check left flank: strictly increasing toward i
            left_len = min(i, localArea)
            for j in range(1, left_len + 1):
                if rawData[i - j + 1] <= rawData[i - j]:
                    is_peak = False
                    break

            if not is_peak:
                continue

            # 2. Check right flank: strictly decreasing away from i
            right_len = min(n - 1 - i, localArea)
            for j in range(1, right_len + 1):
                if rawData[i + j - 1] <= rawData[i + j]:
                    is_peak = False
                    break

            if is_peak:
                result.append(i)

        return result

if __name__ == "__main__":
    assert LocalMaximaStreamSolution.findLocalMaxima([1, 3, 5, 4, 2, 6, 2, 1], 2) == [2]
    assert LocalMaximaStreamSolution.findLocalMaxima([10], 3) == [0]
    assert LocalMaximaStreamSolution.findLocalMaxima([2, 4, 4, 1], 1) == []
    print("✅ Card 17 (Local Maximum 1-D Stream) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Bidirectional Monotonicity**:
  A local maximum requires strict descent when moving outward in both directions. The left subarray $rawData[i-L \dots i]$ must strictly increase toward $i$, and the right subarray $rawData[i \dots i+R]$ must strictly decrease away from $i$. Equal plateau values violate strict inequality and are rejected.
- **Boundary Fallback**:
  At $i=0$, the left neighbor count is 0, satisfying the left flank vacuously; only the right flank is verified. At $i=n-1$, only the left flank is checked. A single-element array returns `[0]`.
- **Streaming Context**:
  Widely deployed in financial tick analysis and sensor stream signal processing for real-time peak identification and technical indicator feature generation.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N \cdot K)$ where $K = 	ext{localArea}$. For small fixed $K$, runtime scales linearly with stream length $N$.
- **Space Complexity**: $\mathcal{O}(1)$ auxiliary space excluding output list.

</div>

</div>
</details>

---

### 18. Largest Min+Max in Subarray via Adjacent Pair Reduction

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">ARRAY 18</span>
  <span class="review-card-title">Largest Min+Max in Subarray via Adjacent Pair Reduction</span>
  <span class="review-card-tag">Contiguous Subarray Mathematical Reduction · Contiguous Subarray Mathematical Reduction · Mathematical Reduction · Local Dominance · Adjacent Pair Scan · O(N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class LargestMinMaxSumSolution:
    @classmethod
    def largestMinMaxSum(cls, nums: List[int]) -> int:
        """
        Finds the maximum possible value of min(sub) + max(sub) for any
        contiguous subarray of length >= 2.
        
        Mathematical Reduction:
        For any contiguous subarray nums[i..j] (j - i >= 1), its min + max
        is bounded above by the adjacent pair containing its maximum element.
        Thus the global optimum reduces to max(nums[i] + nums[i+1]).
        """
        n = len(nums)
        if n < 2:
            raise ValueError("Array length must be at least 2")

        max_sum = nums[0] + nums[1]
        for i in range(1, n - 1):
            pair_sum = nums[i] + nums[i + 1]
            if pair_sum > max_sum:
                max_sum = pair_sum

        return max_sum

if __name__ == "__main__":
    assert LargestMinMaxSumSolution.largestMinMaxSum([5, 12, 9, 6, 4]) == 21
    assert LargestMinMaxSumSolution.largestMinMaxSum([1, 2]) == 3
    assert LargestMinMaxSumSolution.largestMinMaxSum([10, 1, 10]) == 11
    print("✅ Card 18 (Largest Min+Max Subarray) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Mathematical Reduction Proof**:
  Let $[i, j]$ be a subarray with $j - i \ge 1$, minimum $m$, and maximum $M$.
  Consider the adjacent pair containing $M$ inside this subarray (e.g. $(nums[p-1], M)$ or $(M, nums[p+1])$).
  Every element in the subarray is $\ge m$, so the neighbor element is $\ge m$.
  Hence $M + 	ext{neighbor} \ge M + m$.
  Therefore, every valid subarray's objective is dominated by an adjacent pair within it. No sliding window or segment tree is needed; a linear $\mathcal{O}(N)$ scan over adjacent pairs suffices.
- **Length Constraint Guard**:
  Clarifying "length $\ge 2$" is essential: a single-element subarray would trivially give $2 	imes nums[i]$, completely altering the problem.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$ single linear pass.
- **Space Complexity**: $\mathcal{O}(1)$ auxiliary space.

</div>

</div>
</details>

---

### 19. Subarray Sum Equals K via Prefix Sum Hash Map

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">ARRAY 19</span>
  <span class="review-card-title">Subarray Sum Equals K via Prefix Sum Hash Map</span>
  <span class="review-card-tag">[LeetCode 560 · Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) · Prefix Sum Difference · Frequency Hash Map · Negative-Value Robustness · O(N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List
from collections import defaultdict

class SubarraySumEqualsKSolution:
    @classmethod
    def subarraySum(cls, nums: List[int], k: int) -> int:
        """
        Counts contiguous subarrays whose sum equals k.
        Robust to positive, negative, and zero values.
        """
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1  # Base prefix for exact matches from index 0
        
        current_sum = 0
        total_valid_subarrays = 0

        for num in nums:
            current_sum += num
            
            # Check for prefix sum satisfying: current_sum - prefix_sum = k
            if (current_sum - k) in prefix_counts:
                total_valid_subarrays += prefix_counts[current_sum - k]

            prefix_counts[current_sum] += 1

        return total_valid_subarrays

if __name__ == "__main__":
    assert SubarraySumEqualsKSolution.subarraySum([1, 1, 1], 2) == 2
    assert SubarraySumEqualsKSolution.subarraySum([1, 2, 3], 3) == 2
    assert SubarraySumEqualsKSolution.subarraySum([1, -1, 0], 0) == 3
    print("✅ Card 19 (Subarray Sum Equals K) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Prefix Sum Difference Property**:
  $$\sum_{p=i}^j nums[p] = S_j - S_{i-1} = k \iff S_{i-1} = S_j - k$$
  Tracking historical prefix frequencies in a hash map allows counting valid starting indices for ending index $j$ in $\mathcal{O}(1)$ amortized time.
- **Why Sliding Window Fails**:
  Negative values destroy monotonicity of cumulative sums; contracting the left pointer no longer guarantees shrinking the window sum.
- **Base Case `{0: 1}`**:
  When $S_j == k$, the entire subarray from index 0 to $j$ is valid. Seeding `{0: 1}` guarantees this match is counted.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$ single pass with $\mathcal{O}(1)$ hash map lookups.
- **Space Complexity**: $\mathcal{O}(N)$ storing at most $N+1$ prefix values.

</div>

</div>
</details>

---

### 20. Longest Substring Without Repeating Characters

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STRING 20</span>
  <span class="review-card-title">Longest Substring Without Repeating Characters</span>
  <span class="review-card-tag">[LeetCode 3 powers · Longest Substring Without Repeating](https://leetcode.com/problems/longest-substring-without-repeating-characters/) · Sliding Window · Last Seen Index Map · Monotonic Left Jump · O(N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
class LongestSubstringWithoutRepeatingSolution:
    @classmethod
    def lengthOfLongestSubstring(cls, s: str) -> int:
        """
        Computes length of longest contiguous substring without repeating characters.
        Uses a last-seen index map for O(1) left pointer jumps.
        """
        char_last_seen = {}
        left = 0
        max_length = 0

        for right, ch in enumerate(s):
            if ch in char_last_seen and char_last_seen[ch] >= left:
                left = char_last_seen[ch] + 1
            
            char_last_seen[ch] = right
            current_window = right - left + 1
            if current_window > max_length:
                max_length = current_window

        return max_length

if __name__ == "__main__":
    assert LongestSubstringWithoutRepeatingSolution.lengthOfLongestSubstring("abcabcbb") == 3
    assert LongestSubstringWithoutRepeatingSolution.lengthOfLongestSubstring("bbbbb") == 1
    assert LongestSubstringWithoutRepeatingSolution.lengthOfLongestSubstring("pwwkew") == 3
    assert LongestSubstringWithoutRepeatingSolution.lengthOfLongestSubstring("") == 0
    print("✅ Card 20 (Longest Substring Without Repeating) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **O(1) Boundary Jump**:
  Tracking `char_last_seen[ch]` allows the left pointer to jump directly to `last_seen + 1`, avoiding iterative single-character shrinkage.
- **Monotonicity Guard**:
  The condition `char_last_seen[ch] >= left` prevents the left boundary from moving backward to stale indices outside the active window.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$, single pass.
- **Space Complexity**: $\mathcal{O}(\min(N, |\Sigma|))$, bounded by the character alphabet size.

</div>

</div>
</details>

---

### 21. 8-Byte Aligned Memory Allocator Simulation

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">DESIGN 21</span>
  <span class="review-card-title">8-Byte Aligned Memory Allocator Simulation</span>
  <span class="review-card-tag">[LeetCode 2502 · Design Memory Allocator](https://leetcode.com/problems/design-memory-allocator/) · Systems Emulation · 8-Byte Alignment Stride · Unique Block ID · O(N/8 * X)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class AlignedMemoryAllocator:
    """
    Simulates memory allocation under an 8-byte aligned start-index constraint.
    
    - alloc(x): finds leftmost slot starting at index = 0 (mod 8) with >= x
                consecutive free cells (0), tags them with unique autoincrement ID,
                and returns start index. Returns -1 if no space fits.
    - erase(id): frees all cells tagged with id, returning total cells cleared.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[int] = [0] * capacity
        self.next_alloc_id: int = 1

    def alloc(self, x: int) -> int:
        if x <= 0:
            return -1

        current_id = self.next_alloc_id

        # Scan multiples of 8 exclusively: 0, 8, 16, 24, ...
        for start in range(0, self.capacity, 8):
            if start + x <= self.capacity:
                can_fit = True
                for offset in range(x):
                    if self.memory[start + offset] != 0:
                        can_fit = False
                        break

                if can_fit:
                    for offset in range(x):
                        self.memory[start + offset] = current_id
                    self.next_alloc_id += 1
                    return start

        return -1

    def erase(self, req_id: int) -> int:
        if req_id <= 0:
            return 0

        cleared_count = 0
        for i in range(self.capacity):
            if self.memory[i] == req_id:
                self.memory[i] = 0
                cleared_count += 1

        return cleared_count

if __name__ == "__main__":
    alloc = AlignedMemoryAllocator(24)
    assert alloc.alloc(5) == 0
    assert alloc.alloc(10) == 8
    assert alloc.alloc(5) == -1
    assert alloc.erase(1) == 5
    assert alloc.alloc(6) == 0
    print("✅ Card 21 (8-Byte Aligned Memory Allocator) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Alignment Stride Constraint**:
  Enforcing $start \pmod 8 == 0$ restricts candidate search indices to step size 8, reducing candidate locations to $\lceil 	ext{capacity} / 8 
ceil$.
- **Monotonic ID Tagging**:
  Auto-incrementing `next_alloc_id` ensures each allocation retains distinct identities even across allocations of identical size.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**:
  - `alloc(x)`: $\mathcal{O}(rac{N}{8} \cdot X)$ worst-case.
  - `erase(id)`: $\mathcal{O}(N)$ single pass over memory array.
- **Space Complexity**: $\mathcal{O}(N)$ for memory state array.

</div>

</div>
</details>

---

### 22. Zigzag Alternating-Parity Subarrays

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">ARRAY 22</span>
  <span class="review-card-title">Zigzag Alternating-Parity Subarrays</span>
  <span class="review-card-tag">[LeetCode 2765 · Longest Alternating Subarray](https://leetcode.com/problems/longest-alternating-subarray/) / [LeetCode 978](https://leetcode.com/problems/longest-turbulent-subarray/) · Running Streak · Parity Disparity · Single Pass · O(N) Time</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class AlternatingParitySubarraysSolution:
    @classmethod
    def countAlternatingSubarrays(cls, nums: List[int]) -> int:
        """
        Counts contiguous subarrays whose adjacent elements alternate in parity.
        Single-element subarrays count as length-1 alternating sequences.
        """
        if not nums:
            return 0

        total_subarrays = 1
        current_streak = 1

        for i in range(1, len(nums)):
            if (nums[i] % 2) != (nums[i - 1] % 2):
                current_streak += 1
            else:
                current_streak = 1

            # Number of alternating subarrays ending at index i equals current_streak
            total_subarrays += current_streak

        return total_subarrays

if __name__ == "__main__":
    assert AlternatingParitySubarraysSolution.countAlternatingSubarrays([1, 2, 3, 4]) == 10
    assert AlternatingParitySubarraysSolution.countAlternatingSubarrays([2, 4, 6]) == 3
    assert AlternatingParitySubarraysSolution.countAlternatingSubarrays([1]) == 1
    print("✅ Card 22 (Zigzag Alternating-Parity Subarrays) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Running Streak Counting Principle**:
  If the longest alternating contiguous subarray ending at $i-1$ has length $k$, and $nums[i]$ has opposite parity from $nums[i-1]$, then extending all $k$ previous subarrays plus the length-1 subarray $[nums[i]]$ yields exactly $k+1$ valid subarrays ending at $i$.
- **Avoiding $\mathcal{O}(N^2)$ Trap**:
  Maintaining `current_streak` removes the need to enumerate endpoints or verify subsegments.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$ linear scan.
- **Space Complexity**: $\mathcal{O}(1)$ auxiliary space.

</div>

</div>
</details>

---

### 23. Two-Direction Justified Newspaper Layout

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STRING 23</span>
  <span class="review-card-title">Two-Direction Justified Newspaper Layout</span>
  <span class="review-card-tag">[LeetCode 68 · Text Justification](https://leetcode.com/problems/text-justification/) · Greedy Word Packing · Dual Alignment Padding · Asterisk Framing · O(Total Words)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class NewspaperLayoutSolution:
    @classmethod
    def layoutNewspaper(
        cls,
        paragraphs: List[List[str]],
        alignments: List[str],
        width: int
    ) -> List[str]:
        """
        Lays out words greedily per paragraph with LEFT or RIGHT justification up to width,
        and wraps the rendered output in a '*' border.
        """
        content_lines: List[str] = []

        for words, align in zip(paragraphs, alignments):
            current_line_words: List[str] = []
            current_line_len = 0

            for word in words:
                needed_len = len(word) if not current_line_words else len(word) + 1

                if current_line_len + needed_len <= width:
                    current_line_words.append(word)
                    current_line_len += needed_len
                else:
                    line_text = " ".join(current_line_words)
                    pad_spaces = " " * (width - len(line_text))
                    
                    if align == "LEFT":
                        formatted_line = line_text + pad_spaces
                    else:
                        formatted_line = pad_spaces + line_text

                    content_lines.append(f"*{formatted_line}*")
                    current_line_words = [word]
                    current_line_len = len(word)

            if current_line_words:
                line_text = " ".join(current_line_words)
                pad_spaces = " " * (width - len(line_text))
                if align == "LEFT":
                    formatted_line = line_text + pad_spaces
                else:
                    formatted_line = pad_spaces + line_text
                content_lines.append(f"*{formatted_line}*")

        horizontal_border = "*" * (width + 2)
        return [horizontal_border] + content_lines + [horizontal_border]

if __name__ == "__main__":
    paras = [["Hello", "world"], ["Antigravity", "AI", "news"]]
    aligns = ["LEFT", "RIGHT"]
    rendered = NewspaperLayoutSolution.layoutNewspaper(paras, aligns, 16)
    assert rendered[0] == "******************"
    assert rendered[1] == "*Hello world     *"
    assert rendered[2] == "*  Antigravity AI*"
    assert rendered[3] == "*            news*"
    assert rendered[4] == "******************"
    print("✅ Card 23 (Newspaper Layout) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Greedy Word Packing**:
  Each line packs maximal words separated by single spaces until adding the next exceeds `width`.
- **Directional Alignment**:
  - `LEFT`: text padded with trailing spaces;
  - `RIGHT`: text padded with leading spaces.
- **Border Enclosure**:
  Top and bottom lines are solid `*` strings of width `width + 2`, enclosing left and right frame stars.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(L)$ where $L$ is total length of all words and padded spaces.
- **Space Complexity**: $\mathcal{O}(L)$ storing rendered strings.

</div>

</div>
</details>

---

### 24. Longest Palindromic Substring: Center vs Manacher

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STRING 24</span>
  <span class="review-card-title">Longest Palindromic Substring: Center vs Manacher</span>
  <span class="review-card-tag">[LeetCode 5 · Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) · Center Expansion · Manacher's Algorithm · Symmetry Radius Mapping · Strict O(N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
class LongestPalindromeSolution:
    @classmethod
    def longestPalindromeCenterExpand(cls, s: str) -> str:
        """Baseline Center Expansion: O(N^2) time, O(1) space."""
        if not s:
            return ""

        start, max_len = 0, 1

        def expand(left: int, right: int) -> int:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1

        for i in range(len(s)):
            len1 = expand(i, i)
            len2 = expand(i, i + 1)
            cur_max = max(len1, len2)
            if cur_max > max_len:
                max_len = cur_max
                start = i - (cur_max - 1) // 2

        return s[start : start + max_len]

    @classmethod
    def longestPalindromeManacher(cls, s: str) -> str:
        """Manacher's Algorithm: strict O(N) time and O(N) space."""
        if not s:
            return ""

        transformed = "^#" + "#".join(s) + "#$"
        m = len(transformed)
        radius = [0] * m
        center = 0
        right = 0

        for i in range(1, m - 1):
            i_mirror = 2 * center - i

            if right > i:
                radius[i] = min(right - i, radius[i_mirror])
            else:
                radius[i] = 0

            while transformed[i + 1 + radius[i]] == transformed[i - 1 - radius[i]]:
                radius[i] += 1

            if i + radius[i] > right:
                center = i
                right = i + radius[i]

        best_radius = 0
        best_center = 0
        for i in range(1, m - 1):
            if radius[i] > best_radius:
                best_radius = radius[i]
                best_center = i

        start_orig = (best_center - best_radius) // 2
        return s[start_orig : start_orig + best_radius]

if __name__ == "__main__":
    assert LongestPalindromeSolution.longestPalindromeCenterExpand("babad") in ("bab", "aba")
    assert LongestPalindromeSolution.longestPalindromeCenterExpand("cbbd") == "bb"
    assert LongestPalindromeSolution.longestPalindromeManacher("babad") in ("bab", "aba")
    assert LongestPalindromeSolution.longestPalindromeManacher("cbbd") == "bb"
    assert LongestPalindromeSolution.longestPalindromeManacher("a") == "a"
    print("✅ Card 24 (Longest Palindrome Manacher) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Center Expansion ($\mathcal{O}(N^2)$ Baseline)**:
  Examines $2N-1$ possible centers, expanding symmetrically. Degenerates to $\mathcal{O}(N^2)$ on repetitive strings (e.g. `"aaaa"`).
- **Manacher's $\mathcal{O}(N)$ Symmetry Reuse**:
  1. **Even/Odd Unification**: Inserting `#` transforms all palindromes into odd-length ones centered on a character or `#`.
  2. **Mirror Seeding**: When $i < right$, the palindrome radius around $i$ is seeded from its mirror $i_{mirror} = 2 \cdot center - i$, bounded by `right - i`.
  3. **Amortized Linearity**: Character comparisons only occur when expanding beyond the current `right` boundary. Because `right` advances monotonically at most $2N$ times, total runtime is strictly $\mathcal{O}(N)$.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: Center expansion is $\mathcal{O}(N^2)$; Manacher's algorithm is $\mathcal{O}(N)$.
- **Space Complexity**: Center expansion is $\mathcal{O}(1)$; Manacher's algorithm is $\mathcal{O}(N)$ for transformed string and radius array.

</div>

</div>
</details>

