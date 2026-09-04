# Review Flashcards: Core Fundamentals (Review 1)

This module represents the first part of our algorithm interview review series: **Core Fundamentals (常考基础题)**.  
Each flashcard is an isolated, collapsible card containing: **Problem Definition**, **Core Thought Process**, **Standard Python Implementation**, **Complexity & Invariants Card**, and an **Interactive Review MCQ** (answers and explanations remain fully hidden until the user selects an option).

---

## 🛠️ Reusable Flashcard HTML Block Specification

Future flashcards can be constructed by copying the following template directly:

```html
<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Card ID</span>
  <span class="review-card-title">Algorithm / Problem Title</span>
  <span class="review-card-tag">Category Tag</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Constraints</div>
<div class="review-block-body">Problem summary, inputs, invariants...</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach & Algorithm</div>
<div class="review-block-body">Mental model, step-by-step reasoning...</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation</div>
<div class="review-block-body">

```python
# Standard Python code
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity & Invariants</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">Best Time</span><span class="complexity-item-value">O(...)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Worst Time</span><span class="complexity-item-value">O(...)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Auxiliary Space</span><span class="complexity-item-value">O(...)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Stability</span><span class="complexity-item-value">Stable / Unstable</span></div>
</div>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 Interactive Review MCQ (Click an option to reveal answer)</div>
<div class="review-block-body">

```quiz
title: Self Check · Key Concept
question: Question description here?
A. Option 1
B. Option 2
C. Option 3
D. Option 4
Answer: A
Explanation: In-depth explanation of why option A is correct...
```

</div>
</div>

</div>
</details>
```

---

## 📚 Core Fundamentals Review Flashcards

### 1. Merge Sort

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Core 01</span>
  <span class="review-card-title">Merge Sort</span>
  <span class="review-card-tag">Divide &amp; Conquer · Recursion · Stable</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Constraints</div>
<div class="review-block-body">
Given an unsorted integer array of length $n$, sort it in non-decreasing order. The worst-case time complexity must be strictly guaranteed to be $O(n \log n)$, and the relative order of duplicate elements must be preserved (stability).
</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Algorithm</div>
<div class="review-block-body">
Merge Sort is a canonical realization of the **Divide and Conquer** paradigm:
<ol>
  <li><strong>Divide</strong>: Find the midpoint <code>mid = len(nums) // 2</code> to divide the array evenly into left and right halves.</li>
  <li><strong>Conquer</strong>: Recursively invoke merge sort on both halves until subproblems reach base cases of length $\le 1$ (inherently sorted).</li>
  <li><strong>Combine</strong>: Linearly merge the two sorted halves using two pointers, picking the smaller current item into the result; on ties, prefer the left half to preserve stability.</li>
</ol>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation</div>
<div class="review-block-body">

```python
class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        if len(nums) <= 1:
            return nums

        mid = len(nums) // 2
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:])

        return self._merge(left, right)

    def _merge(self, left: list[int], right: list[int]) -> list[int]:
        res = []
        i = j = 0

        # Two pointers merge two sorted sublists
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:  # <= ensures stable sorting
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1

        res.extend(left[i:])
        res.extend(right[j:])
        return res
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Invariants</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">Best Time</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Worst Time</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Average Time</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Auxiliary Space</span><span class="complexity-item-value">O(n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Stability</span><span class="complexity-item-value">Stable</span></div>
</div>
<p style="margin-top: 0.5rem; font-size: 0.88rem; color: #466370;">
<strong>Key Invariant</strong>: The recursion tree has strict height $\lceil \log_2 n \rceil$, and total work at each level across all merges is bounded by $O(n)$. Auxiliary space is $O(n)$ for buffer allocation plus $O(\log n)$ recursive stack frames.
</p>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 Interactive Review MCQ (Click an option to reveal answer)</div>
<div class="review-block-body">

```quiz
title: Self Check · Merge Sort Space Complexity
question: When performing merge sort on a singly linked list of n nodes using a bottom-up iterative approach, what is the optimal auxiliary space complexity (excluding constant pointers)?
A. O(1)
B. O(log n)
C. O(n)
D. O(n log n)
Answer: A
Explanation: Array merge sort requires an O(n) temporary buffer to avoid expensive element shifting; however, a linked list can be merged entirely in-place by rewiring the 'next' pointers. Using iterative bottom-up step sizes (1, 2, 4, 8...), it avoids recursion stack frames entirely, achieving strict O(1) auxiliary space.
```

</div>
</div>

</div>
</details>

---

### 2. Quick Sort

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Core 02</span>
  <span class="review-card-title">Quick Sort</span>
  <span class="review-card-tag">Divide &amp; Conquer · In-place Partition · Unstable</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Constraints</div>
<div class="review-block-body">
Given an array of $n$ integers, sort it in non-decreasing order in-place. The average time complexity must achieve $O(n \log n)$, requiring minimal auxiliary space (no extra heap or dynamic data structures beyond call stack frames).
</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Algorithm</div>
<div class="review-block-body">
Quick Sort works by performing **Partitioning before Recursion**:
<ol>
  <li><strong>Pivot Selection</strong>: Pick a pivot randomly (or median-of-three) to break adversarial pre-sorted order and eliminate $O(n^2)$ degradations.</li>
  <li><strong>Partition</strong>: Rearrange the array such that all elements $\le pivot$ lie to the left, and elements $\ge pivot$ lie to the right, landing the pivot at its exact final sorted index <code>p_idx</code>.</li>
  <li><strong>Recurse</strong>: Recursively sort the subsegments <code>[l, p_idx - 1]</code> and <code>[p_idx + 1, r]</code> independently.</li>
</ol>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation</div>
<div class="review-block-body">

```python
import random

class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        self._quick_sort(nums, 0, len(nums) - 1)
        return nums

    def _quick_sort(self, nums: list[int], l: int, r: int) -> None:
        if l >= r:
            return

        # Randomize pivot to eliminate worst-case pre-sorted inputs
        pivot_idx = random.randint(l, r)
        nums[pivot_idx], nums[r] = nums[r], nums[pivot_idx]

        p_idx = self._partition(nums, l, r)
        self._quick_sort(nums, l, p_idx - 1)
        self._quick_sort(nums, p_idx + 1, r)

    def _partition(self, nums: list[int], l: int, r: int) -> int:
        pivot = nums[r]
        i = l
        for j in range(l, r):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Invariants</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">Best Time</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Worst Time</span><span class="complexity-item-value">O(n²)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Average Time</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Auxiliary Space</span><span class="complexity-item-value">O(log n) ~ O(n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Stability</span><span class="complexity-item-value">Unstable</span></div>
</div>
<p style="margin-top: 0.5rem; font-size: 0.88rem; color: #466370;">
<strong>Key Invariant</strong>: If partitioning repeatedly creates highly skewed splits (e.g. 1 and $n-1$), the recursion tree reaches depth $n$, degenerating to $O(n^2)$. Space is strictly call stack depth ($O(\log n)$ expected, $O(n)$ worst-case). Long-range swaps can jump duplicate elements across one another, making it unstable.
</p>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 Interactive Review MCQ (Click an option to reveal answer)</div>
<div class="review-block-body">

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

</div>
</details>

---

### 3. Dynamic Array Implementation

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Core 03</span>
  <span class="review-card-title">Dynamic Array Implementation</span>
  <span class="review-card-tag">Data Structure Design · Amortized Analysis</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Constraints</div>
<div class="review-block-body">
Implement a resizable dynamic array from scratch (analogous to Python's <code>list</code> or C++'s <code>std::vector</code>) backed by a fixed-size contiguous buffer. Implement the following APIs:
<ul>
  <li><code>get(i)</code>: Read value at index $i$; raise exception on out-of-bounds.</li>
  <li><code>set(i, val)</code>: Overwrite value at index $i$.</li>
  <li><code>push_back(val)</code>: Append to end; automatically double capacity when buffer is exhausted.</li>
  <li><code>pop_back()</code>: Remove and return terminal element.</li>
  <li><code>resize()</code>: Double internal buffer capacity and migrate existing elements.</li>
</ul>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Algorithm</div>
<div class="review-block-body">
The core mental model of a dynamic array:
<ol>
  <li><strong>Contiguous Memory &amp; Twin Counters</strong>: Maintain fixed allocation <code>capacity</code> alongside active element count <code>size</code>.</li>
  <li><strong>Geometric Doubling</strong>: When <code>size == capacity</code>, allocate a new contiguous chunk with double the capacity (<code>2 * capacity</code>), copy elements across, and release the old buffer.</li>
  <li><strong>Amortized $O(1)$ Proof</strong>: A single resize requires $O(n)$ copies, but resize events occur exponentially less frequently. Expanding from 1 to $n$ copies $1 + 2 + 4 + \dots + n \le 2n$ elements total. Amortized across $n$ operations, the cost per append is strictly $O(1)$.</li>
</ol>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation</div>
<div class="review-block-body">

```python
class DynamicArray:
    def __init__(self, capacity: int = 2):
        self.capacity = max(1, capacity)
        self.size = 0
        self.array = [None] * self.capacity

    def get(self, i: int) -> int:
        if not (0 <= i < self.size):
            raise IndexError("Index out of bounds")
        return self.array[i]

    def set(self, i: int, val: int) -> None:
        if not (0 <= i < self.size):
            raise IndexError("Index out of bounds")
        self.array[i] = val

    def push_back(self, val: int) -> None:
        # Trigger geometric doubling when buffer full
        if self.size == self.capacity:
            self._resize(self.capacity * 2)

        self.array[self.size] = val
        self.size += 1

    def pop_back(self) -> int:
        if self.size == 0:
            raise IndexError("Cannot pop from empty array")
        self.size -= 1
        val = self.array[self.size]
        self.array[self.size] = None  # Prevent memory leak
        return val

    def _resize(self, new_capacity: int) -> None:
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity

    def get_size(self) -> int:
        return self.size

    def get_capacity(self) -> int:
        return self.capacity
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Invariants</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">Random Access get/set</span><span class="complexity-item-value">O(1)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Append push_back</span><span class="complexity-item-value">Amortized O(1) (Worst O(n))</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Pop pop_back</span><span class="complexity-item-value">O(1)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Mid Insert/Delete</span><span class="complexity-item-value">O(n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Space Utilization</span><span class="complexity-item-value">≥ 50%</span></div>
</div>
<p style="margin-top: 0.5rem; font-size: 0.88rem; color: #466370;">
<strong>Key Invariant</strong>: Direct indexing leverages pointer arithmetic <code>base + i * element_size</code> for $O(1)$ read/write. Append costs $O(n)$ only on the rare occasions resize triggers, amortizing to constant time. Arbitrary index insertion requires shifting subsequent elements, incurring $O(n)$ worst-case time.
</p>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 Interactive Review MCQ (Click an option to reveal answer)</div>
<div class="review-block-body">

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

</div>
</details>

---

### 4. Binary Search Boundary Template

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Core 04</span>
  <span class="review-card-title">Binary Search Boundary Template</span>
  <span class="review-card-tag">Monotonic Search · Interval Invariant</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Constraints</div>
<div class="review-block-body">
Given a non-decreasing sorted integer array, locate the **first occurrence (Lower Bound / leftmost index)** of <code>target</code>. If the target does not exist, return the index where it should be inserted to maintain order. Runtime must be $O(\log n)$ with absolute zero risk of infinite loops.
</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Algorithm</div>
<div class="review-block-body">
Binary search depends entirely on **Loop Invariant maintenance**:
<ol>
  <li><strong>Interval Definition</strong>: Fixate on a **closed interval <code>[l, r]</code>**, initialized at <code>l = 0, r = len(nums) - 1</code>.</li>
  <li><strong>Overflow-safe Midpoint</strong>: Use <code>mid = l + (r - l) // 2</code> to eliminate potential integer overflow.</li>
  <li><strong>Left-boundary Shrinkage</strong>:
    <ul>
      <li>If <code>nums[mid] >= target</code>: Target could be <code>mid</code> or to the left; shrink right bound: <code>r = mid - 1</code>.</li>
      <li>If <code>nums[mid] < target</code>: Target is strictly to the right; shrink left bound: <code>l = mid + 1</code>.</li>
    </ul>
  </li>
  <li><strong>Convergence Guarantee</strong>: Loop runs while <code>l <= r</code>, terminating strictly at <code>l = r + 1</code>, with <code>l</code> landing on the first element $\ge target$.</li>
</ol>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation</div>
<div class="review-block-body">

```python
class Solution:
    def searchLowerBound(self, nums: list[int], target: int) -> int:
        l = 0
        r = len(nums) - 1

        # Maintain invariant over closed range [l, r]
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] >= target:
                r = mid - 1  # Seek lower matching bound to the left
            else:
                l = mid + 1  # Seek rightward

        # Upon termination l == r + 1; l is the first index >= target
        return l
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Invariants</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">Time Complexity</span><span class="complexity-item-value">O(log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Auxiliary Space</span><span class="complexity-item-value">O(1)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">Search Space Reduction</span><span class="complexity-item-value">Halved each step (1/2)</span></div>
</div>
<p style="margin-top: 0.5rem; font-size: 0.88rem; color: #466370;">
<strong>Key Invariant</strong>: Each iteration halves the candidate search space, guaranteeing at most $\lfloor \log_2 n \rfloor + 1$ comparisons. The iterative approach consumes strictly $O(1)$ auxiliary space.
</p>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 Interactive Review MCQ (Click an option to reveal answer)</div>
<div class="review-block-body">

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

</div>
</details>
