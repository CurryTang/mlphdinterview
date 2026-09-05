# Review Flashcards: Core Fundamentals (Review 1)

This module provides high-yield algorithm interview review flashcards: distilled **Problem Definitions**, **Core Mental Models**, **Minimal Core Implementations**, **Complexity Invariants**, and **Interactive Self-Check MCQs**. Click any card title to expand.

---

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

<div class="review-block">
<div class="review-block-label">🎯 Interactive Self-Check MCQ (Click to reveal explanation)</div>

```quiz
title: Self Check · Merge Sort Space Complexity
question: When performing merge sort on a singly linked list of n nodes using a bottom-up iterative approach, what is the optimal auxiliary space complexity?
A. O(1)
B. O(log n)
C. O(n)
D. O(n log n)
Answer: A
Explanation: Array merge sort requires an O(n) temporary buffer to avoid expensive element shifting; however, a linked list can be merged entirely in-place by rewiring the 'next' pointers. Using iterative bottom-up step sizes (1, 2, 4, 8...), it avoids recursion stack frames entirely, achieving strict O(1) auxiliary space.
```

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
1. **Pivot**: Choose a pivot element (randomized or median-of-three to break adversarial inputs).
2. **Partition**: Rearrange array elements such that all elements $\le pivot$ move left and $\ge pivot$ move right, placing the pivot at final index $i$.
3. **Recurse**: Recursively sort left subarray $[l, i - 1]$ and right subarray $[i + 1, r]$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def quick_sort(nums: list[int], l: int, r: int) -> None:
    if l >= r:
        return
    # Core Lomuto partition: i maintains boundary of elements <= pivot
    pivot, i = nums[r], l
    for j in range(l, r):
        if nums[j] <= pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[r] = nums[r], nums[i]

    quick_sort(nums, l, i - 1)
    quick_sort(nums, i + 1, r)
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: Best $O(n \log n)$ / Worst $O(n^2)$ (skewed partitions) / Average $O(n \log n)$
- **Auxiliary Space**: $O(\log n)$ (stack frames, degrades to $O(n)$ in worst case)
- **Stability**: **Unstable** (long-distance swaps disrupt relative order)

</div>

<div class="review-block">
<div class="review-block-label">🎯 Interactive Self-Check MCQ (Click to reveal explanation)</div>

```quiz
title: Self Check · Quick Sort Duplicate Degradation
question: When sorting an array where all elements are identical (e.g. 10000 copies of value 7), what happens to standard Lomuto partition quick sort?
A. Degenerates severely to O(n^2) runtime
B. Runs in optimal O(n log n) time
C. Runs in linear O(n) time
D. Triggers an index out of bounds error
Answer: A
Explanation: In standard Lomuto partition, condition nums[j] <= pivot evaluates to true for every single element. Every element is repeatedly swapped into the left partition, advancing the pivot by only 1 index per round and yielding recursion depth n with total time O(n^2). The canonical fix is three-way partitioning (Dutch National Flag), which collects equal elements in the center and removes them from subsequent recursive subproblems in linear O(n) total time.
```

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

<div class="review-block">
<div class="review-block-label">🎯 Interactive Self-Check MCQ (Click to reveal explanation)</div>

```quiz
title: Self Check · Dynamic Array Growth Strategy
question: If a dynamic array resizes by adding a constant increment (e.g. capacity += 1000 whenever full), what is the total copy time and amortized complexity per push_back across N sequential appends?
A. Total time O(N^2), amortized O(N)
B. Total time O(N log N), amortized O(log N)
C. Total time O(N), amortized O(1)
D. Total time O(N), amortized O(N)
Answer: A
Explanation: With constant step C = 1000, appending N items triggers N/C expansions. The copy counts across resizes are C, 2C, 3C, ..., N, giving total copies C * (1 + 2 + ... + N/C) ≈ O(N^2). Amortized across N insertions, each append degrades to O(N) average time. This quadratic blowout explains why production runtimes must employ geometric doubling (multiplicative scaling) rather than linear growth.
```

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

<div class="review-block">
<div class="review-block-label">🎯 Interactive Self-Check MCQ (Click to reveal explanation)</div>

```quiz
title: Self Check · Binary Search Infinite Loop Trap
question: When using a closed-range binary search template, if the update logic has a branch l = mid alongside default floor mid = l + (r - l) // 2, what happens when the search space narrows to exactly 2 elements (r = l + 1)?
A. Enters an infinite loop
B. Converges normally
C. Triggers out of bounds error
D. Incorrectly reports target absent
Answer: A
Explanation: When r = l + 1, integer division floors mid = l + (1) // 2 = l. If the l = mid branch is chosen, l is assigned its existing value, leaving the interval [l, r] completely unreduced. The loop runs forever. The fundamental rule to prevent this: whenever an l = mid branch exists, the midpoint must round up using mid = l + (r - l + 1) // 2.
```

</div>

</div>
</details>
