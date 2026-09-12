# Review Flashcards: Core Fundamentals (Review 1)

This module provides high-yield algorithm interview review flashcards: distilled **Problem Definitions**, **Core Mental Models**, **Minimal Core Implementations**, **Complexity Invariants**. Click any card title to expand.

---

## Module 1: Core Fundamental Algorithms

### 1. Merge Sort

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 01</span>
  <span class="review-card-title">Merge Sort</span>
  <span class="review-card-tag">Divide &amp; Conquer · Stable</span>
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
  <span class="review-card-tag">Partitioning · In-Place · Unstable</span>
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
  <span class="review-card-tag">Contiguous Memory · Geometric Doubling · Amortized</span>
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
  <span class="review-card-tag">Monotonic Search · Interval Invariant</span>
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
  <span class="review-card-tag">Grid Flattening · Divisible Prefix · Expected 2.45 Calls</span>
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
  <span class="review-card-tag">Hash Set · Early Exit · Single Pass</span>
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
  <span class="review-card-tag">Frequency Array · ASCII Delta · Length Pruning</span>
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
  <span class="review-card-tag">Hash Map · Complement Matching · Single Pass</span>
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
  <span class="review-card-tag">Frequency Tuple · Canonical Hash Key · Grouping</span>
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
  <span class="review-card-tag">Bucket Sort · Inverted Index · Linear Time</span>
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
  <span class="review-card-tag">Length Prefix Protocol · Chunked Stream · Delimiter Independence</span>
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
  <span class="review-card-tag">Prefix &amp; Suffix Decomposition · Two-Pass In-Place · O(1) Auxiliary Space</span>
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
  <span class="review-card-tag">Row/Col/Box Validation · Grid Coordinate Flattening · O(1) Bound</span>
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
  <span class="review-card-tag">Hash Set · Predecessor Probing · Strict O(n)</span>
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



