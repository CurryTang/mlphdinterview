# Sliding Window

The sliding window technique fundamentally operates as a **two-pointer model maintaining a dynamic closed interval $[\text{left}, \text{right}]$ over a 1D sequence**. All sliding window problems share an identical loop skeleton. Any sliding window problem can be solved systematically by answering **"Three Decision Questions"** and filling **"Three Code Slots"**:

```text
[Three Decision Questions & Code Slots]
1. state & add_right: What incremental state is maintained? How does right entering update it?
2. shrink condition : When should left advance to restore the invariant? (while for variable, if for fixed)
3. record answer    : When should the answer be recorded? (after shrinking for longest, inside shrinking for shortest, when full for fixed)
```

---

## Why Sliding Window? (Foundational Motivation)

For a 1D sequence (array or string) of length $N$, there are $\frac{N(N+1)}{2} = \mathcal{O}(N^2)$ possible contiguous subsegments (subarrays or substrings).

Understanding the core value of the sliding window technique lies in analyzing how it fundamentally diverges from the **Vanilla Brute-Force Solution** in terms of **state reuse** and **pointer movement topology**.

### 1. Bottlenecks of the Vanilla Brute-Force Solution

A naive solution typically employs nested loops iterating over all candidate interval boundaries $(i, j)$:
- The outer loop fixes the left boundary $i \in [0, N-1]$;
- The inner loop enumerates the right boundary $j \in [i, N-1]$;
- For each candidate window $[i, j]$, the elements within the window are scanned from scratch to compute the aggregate metric (such as running sum, duplicate character check, or extremum lookup).

**The Root Inefficiency: Complete Loss of State & Redundant Recomputation**.
Consider two adjacent candidate windows $W_1 = [i, j]$ and $W_2 = [i+1, j+1]$: they share $j - i$ identical elements (an overlap ratio approaching $100\%$). However, the vanilla solution evaluates $W_2$ by **rescanning every single element in the overlapping intersection**. This repetitive work inflates the time complexity to $\mathcal{O}(N^2)$ or even $\mathcal{O}(N^3)$.

---

### 2. Concrete Comparative Example: Fixed-Size Subarray Sum (Vanilla vs. Sliding Window)

Consider the canonical problem underlying **LeetCode 643 (Maximum Average Subarray I)**: Given `nums = [1, 12, -5, -6, 50, 3]` and window size $k = 4$, find the maximum sum among all contiguous subarrays of length $k$.

#### Approach A: Vanilla Brute-Force Solution

For every valid starting index $i$, recompute the sum of the $k$ elements in the slice from scratch:

```python
def max_sum_vanilla(nums: list[int], k: int) -> int:
    n = len(nums)
    max_val = float("-inf")
    for i in range(n - k + 1):
        # Full slice summation takes O(k) time per window
        curr_sum = sum(nums[i : i + k])
        max_val = max(max_val, curr_sum)
    return max_val
```

**Execution Trace:**
- Window 0 `[1, 12, -5, -6]`: 3 additions $\implies 1 + 12 + (-5) + (-6) = 2$;
- Window 1 `[12, -5, -6, 50]`: 3 additions $\implies 12 + (-5) + (-6) + 50 = 51$;
  - **Severe Redundancy**: The common subsegment `[12, -5, -6]` is fully summed again from scratch!
- Window 2 `[-5, -6, 50, 3]`: 3 additions $\implies (-5) + (-6) + 50 + 3 = 42$;
  - **Severe Redundancy**: The subsegment `[-5, -6, 50]` is recalculated yet again!

- **Time Complexity**: There are $(N - k + 1)$ windows, each taking $\mathcal{O}(k)$ time, yielding $\mathcal{O}((N - k + 1) \cdot k) = \mathcal{O}(N \cdot k)$. If $k = N/2$, runtime reaches $\mathcal{O}(N^2)$.
- **Space Complexity**: If producing subarray slices `nums[i:i+k]`, allocation overhead is $\mathcal{O}(k)$; if indexing in place, it is $\mathcal{O}(1)$.

#### Approach B: Sliding Window Solution

The core design principle of the sliding window is **Incremental State Maintenance**:
Adjacent windows differ by only two boundary elements: **one element entering at `nums[right]`, and one element departing at `nums[left]`**.

$$\text{curr\_sum}_{\text{new}} = \text{curr\_sum}_{\text{old}} + \text{nums}[\text{right}] - \text{nums}[\text{left}]$$

```python
def max_sum_sliding_window(nums: list[int], k: int) -> int:
    n = len(nums)
    # Compute full sum only once for the very first window: O(k)
    curr_sum = sum(nums[:k])
    max_val = curr_sum

    for right in range(k, n):
        # Strict O(1) incremental update: add entering element, subtract leaving element
        curr_sum += nums[right] - nums[right - k]
        max_val = max(max_val, curr_sum)
    return max_val
```

**Execution Trace:**
- Initial window sum: $S_0 = 1 + 12 + (-5) + (-6) = 2$;
- Window 1: $S_1 = S_0 + \text{nums}[4] - \text{nums}[0] = 2 + 50 - 1 = 51$ (only 1 addition and 1 subtraction);
- Window 2: $S_2 = S_1 + \text{nums}[5] - \text{nums}[1] = 51 + 3 - 12 = 42$ (only 1 addition and 1 subtraction).

- **Time Complexity**: Initializing the first window requires $\mathcal{O}(k)$, followed by $N - k$ sliding steps, each executing strict $\mathcal{O}(1)$ arithmetic. Total time drops to **$\mathcal{O}(N)$**!
- **Space Complexity**: Only a single scalar accumulator `curr_sum` is maintained, strictly **$\mathcal{O}(1)$**.

---

### 3. Dimensionality Reduction in Variable-Length Windows: Monotonicity & Non-Backtracking

In variable-length window problems (e.g. LC 3 Longest Substring Without Repeating Characters, LC 209 Minimum Size Subarray Sum), brute force forces the right pointer $j$ to **backtrack** to $i+1$ whenever $i$ advances, producing $\mathcal{O}(N^2)$ overhead.

Sliding window reduces variable-length problems to $\mathcal{O}(N)$ by exploiting **Interval Monotonicity**:
1. **Invalidity Pruning**: In LC 3, if the window $[left, right]$ contains a duplicate character, any larger interval expanding rightwards $[left, right+1], [left, right+2], \dots$ will unconditionally contain that duplicate and remain invalid. Thus, inner exploration can terminate immediately without evaluating further right extensions!
2. **Non-Backtracking Pointers (Amortized $\mathcal{O}(1)$)**:
   By incrementing `left` until the duplicate is evicted and validity is restored, **the right pointer `right` never needs to backtrack to `left`—it simply continues advancing rightwards**.
   - `right` advances monotonically in the outer loop: at most $N$ steps;
   - `left` advances monotonically in the inner loop: at most $N$ steps;
   - The total number of pointer movements across the entire execution is strictly bounded by:
     $$\text{Total Steps} \le 2N$$
   Amortized cost per element is $\mathcal{O}(1)$, reducing nested loop complexity from multiplicative $\mathcal{O}(N^2)$ to additive $\mathcal{O}(N)$.

---

### 4. Complexity Comparison Matrix (Time & Space Complexity)

| Metric | Vanilla Brute-Force Solution | Sliding Window Solution | Core Reason for Acceleration |
|---|---|---|---|
| **Fixed-Size Window ($k$) Time** | $\mathcal{O}(N \cdot k)$ (worst-case $\mathcal{O}(N^2)$) | **Strict $\mathcal{O}(N)$** | State Reuse: Replaces $\mathcal{O}(k)$ recomputations with $\mathcal{O}(1)$ boundary deltas |
| **Variable Window Time** | $\mathcal{O}(N^2) \sim \mathcal{O}(N^3)$ | **Amortized $\mathcal{O}(N)$** | Monotonicity: `left` and `right` advance unidirectionally; total pointer movements $\le 2N$ |
| **Space Complexity** | $\mathcal{O}(1) \sim \mathcal{O}(k)$ (slice allocations) | **Strict $\mathcal{O}(1)$ or $\mathcal{O}(\|\Sigma\|)$** | In-place incremental updates without intermediate slice allocations |
| **Pointer Trajectory** | Right pointer continually resets and backtracks | Both pointers move forward monotonically with zero backtracking | Eliminates search space backtracking; ensures forward topological progression |
| **Runtime for $N = 10^5$** | $\approx 10^{10}$ operations $\implies$ **Time Limit Exceeded (TLE)** | $\approx 2 \times 10^5$ operations $\implies$ **$< 10\text{ ms}$ (instant response)** | Collapses polynomial complexity into a single linear pass |

---

## Original Problems and Learning Order

This note selects 8 high-frequency problems categorized into four archetypes: **Longest Valid Window**, **Shortest Satisfying Window**, **Fixed-Size Window**, and **Monotonic Deque Window**:

| Order | Original Problem | Window Archetype | Core Maintained State | Shrink Mechanism | Record Timing |
|---:|---|---|---|---|---|
| 1 | [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/) | Longest Valid Window | Character frequency map | `while` duplicate exists | Update `max` after shrink |
| 2 | [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/description/) | Longest Valid Window | Zero counter `zeros` | `while zeros > k` | Update `max` after shrink |
| 3 | [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/description/) | Longest Valid Window | Frequency map + `max_freq` | `while len - max_freq > k` | Update `max` after shrink |
| 4 | [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/) | Shortest Satisfying Window | Running sum `sum` | `while sum >= target` | Update `min` inside shrink |
| 5 | [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/) | Shortest Satisfying Window | `need/window` + `have` | `while have == required` | Update `min` inside shrink |
| 6 | [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/description/) | Fixed-Size Window | Two 26-element frequency arrays | `if len > k` | Compare at full size and return bool |
| 7 | [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/description/) | Fixed-Size Window | Two 26-element frequency arrays | `if len > k` | Collect matching `left` indices at full size |
| 8 | [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/) | Fixed-Size Window | Decreasing monotonic index deque | `if len > k` | Read front element maximum at full size |

Compare how the core patterns map to the unified skeleton:

```sliding-window-patterns
```

---

## First, Fix the Window Invariant

The window interval is mathematically modeled as a closed interval:

$$
[\text{left}, \text{right}], \qquad \text{length} = \text{right} - \text{left} + 1.
$$

### Why the Closed Interval Length Requires `+ 1`

Both `left` and `right` point to valid elements currently included inside the window. The difference $\text{right} - \text{left}$ measures the **index step distance** between the two pointers. The number of elements contained in the closed interval must also count the starting index `left` itself, requiring the additional $+ 1$:

```text
Array Indices:  2   3   4
Window Items:  [A   B   C]
Pointers:      left = 2, right = 4
Index Gap:     4 - 2 = 2
Element Count: 4 - 2 + 1 = 3
```

Boundary check: When a single-element window has `left == right`, the closed interval length is $\text{right} - \text{left} + 1 = 1$; omitting $+ 1$ would incorrectly yield a length of 0.

| Window Convention | Includes Boundary `right`? | Length Formula | Usage |
|---|---|---|---|
| **Closed Interval $[\text{left}, \text{right}]$** | Yes (both ends inclusive) | $\text{right} - \text{left} + 1$ | **Standard algorithmic convention (used throughout this note)** |
| Half-open Interval $[\text{left}, \text{right})$ | No (left inclusive, right exclusive) | $\text{right} - \text{left}$ | C++ STL iterators / slicing `s[left:right]` |

### Strict Timing Between State Mutation and Pointer Advance

In sliding window algorithms, state updates and pointer movements must be atomically synchronized. When evicting an element from the left, you must **decrement/remove the element from the state first, before advancing the pointer**:

```python
# Correct order: revert state first, then increment pointer
remove_left(state, items[left])
left += 1

# Fatal bug: advancing pointer first deletes the wrong subsequent element!
# left += 1
# remove_left(state, items[left])
```

---

## Core Architecture: Data Structure Selection for Window & State

In sliding window algorithms, the **window boundary itself** only requires two integer pointers `left` and `right` maintaining a dynamic closed interval $[\text{left}, \text{right}]$, requiring strictly $O(1)$ spatial overhead.

What truly governs algorithmic complexity, viability, and systems efficiency is the **data structure selected for the internal state (`state`)**.

### First Principles of State Selection: Incremental Progress and Symmetric Reversal

Whether a sliding window problem can be solved in $O(n)$ time depends directly on whether state mutations satisfy **low-cost reversibility**:
- **`right` Enters**: Can the state be updated incrementally in $O(1)$ or $O(\log k)$ time?
- **`left` Evicts**: Can the leftmost item be symmetrically undone/subtracted in $O(1)$ or $O(\log k)$ time?
- **Validity Check**: Can the predicate for `while` or `if` be answered in $O(1)$ scalar time without traversing the entire state?

### Five Archetypes of State Data Structures

```text
                     What must the window maintain and query?
                                │
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
  [Scalar Sum / Count]     [Frequencies / Keys]      [Dynamic Extremum / Order]
        │                       │                        │
  Algebraically Reversible? Small / Bounded Alphabet?   Only Max / Min Needed?
  ┌─────┴─────┐           ┌─────┴─────┐           ┌──────┴──────┐
  │ Yes       │ No        │ Yes       │ No        │ Yes         │ No (Median/Kth)
  ▼           ▼           ▼           ▼           ▼             ▼
Scalar Var  Prefix Sum  Fixed Array HashMap+have  Monotonic Deque Dual Heaps/BST
 (int)      (Not Window) (int[26])  (Map+scalar)   (Index Deque)   (Multiset)
```

#### 1. Scalar Variables (`int` / `float`)
* **Applicability:** When the window state is an aggregatable scalar whose operations satisfy **strict algebraic reversibility** via addition and subtraction.
* **Canonical Metrics:**
  - **Running Interval Sum:** `curr_sum += nums[right]`, evict `curr_sum -= nums[left]` (LC 209).
  - **Specific Element Counter:** Zero flip budget `zeros += 1`, evict `zeros -= 1` (LC 1004).
* **Complexity:** $O(1)$ update, $O(1)$ space.
* **Boundary Pitfall:** If the array contains **negative numbers**, the running sum loses monotonicity with respect to pointer advances; sliding window collapses and requires Prefix Sums + Monotonic Deque (LC 862).

#### 2. Direct-Mapped Fixed-Size Array (`int[26]` / `int[128]`)
* **Applicability:** Key universe is small, contiguous, and bounded (lowercase `a-z`, uppercase `A-Z`, standard ASCII `128`).
* **Why Strictly Superior to Hash Maps:**
  - **L1 Cache Locality:** `int count[26]` consumes only $104$ bytes, fitting entirely within a single CPU L1 cache line. Direct indexing `ord(c) - ord('a')` is a base address offset with 0 hash overhead, 0 dynamic allocation, and 0 pointer chasing.
  - **$O(1)$ Array Equality Comparison:** In fixed windows, comparing `window == need` requires only 26 integer comparisons, vectorized automatically by modern compilers (SIMD) with near-zero latency.
* **Representative Problems:** LC 567 (Permutation), LC 438 (Anagrams), LC 424 (Character Replacement).

#### 3. Dynamic Hash Map + Scalar Counter (`defaultdict(int)` + `have`)
* **Applicability:** Keys belong to an unbounded, arbitrary domain (arbitrary integers, Unicode characters, sparse strings).
* **Scalar Dimension-Reduction Design:**
  - If a hash map is used in isolation, verifying whether the window satisfies requirements on each step takes $O(|\Sigma|)$ dictionary comparison.
  - **Scalar Anchoring:** Maintain a scalar counter alongside the hash map (e.g., `have: int` counting how many distinct characters satisfy target frequency; or `distinct: int` counting unique active keys).
  - Mutate `have += 1` or `have -= 1` only at the exact instant a key reaches or falls below its threshold. The `while` loop condition is compressed into an $O(1)$ scalar check (`have == required`).
* **Representative Problems:** LC 3 (Unique Substring), LC 76 (Minimum Window Substring), LC 992 (K Distinct Integers).

#### 4. Monotonic Double-Ended Queue (Monotonic Deque)
* **Applicability:** When the window dynamically queries **Extremum (Maximum / Minimum)**.
* **Why Scalars and Hash Maps Fail for Extremum:**
  - **Extremum operations lack algebraic reversibility**: When the maximum element leaves the window, arithmetic cannot reveal what the second-largest element was; without a deque, rescanning takes $O(k)$.
  - Priority queues support fast queries, but arbitrary deletion in sliding windows takes $O(k)$ or heavy heap rebalancing.
* **Core Principles:**
  - **Must Store Indices, Never Values Alone:** Storing indices enables $O(1)$ checks on whether the extremum has slid past the left boundary (`q[0] < left`).
  - Strict monotonic decrease from head to tail. Tail elements dominated by incoming items are permanently discarded (no future relevance); expired head elements are popped.
* **Representative Problems:** LC 239 (Sliding Window Maximum), LC 1438 (Absolute Diff Limit Subarray).

#### 5. Dual Heaps / Balanced BST (`std::multiset` / Two Heaps + Lazy Deletion)
* **Applicability:** When the window queries **Advanced Order Statistics (Dynamic Median, Kth Largest)**.
* **Why Monotonic Deque Fails:** A deque only tracks dominated extremes; the median sits in the middle of the sorted order, requiring all intermediate values to be precisely tracked.
* **Implementation:**
  - **Dual Heaps with Lazy Deletion:** Max-heap for lower half, min-heap for upper half; evicted elements are tracked in a hash map and pruned lazily upon reaching the heap top.
  - **Balanced BST (`std::multiset`):** Native $O(\log k)$ insertion, iterator-based deletion, and median querying.
* **Representative Problems:** LC 480 (Sliding Window Median).

---

### State Data Structure Decision Matrix

| Query Objective | Recommended Structure (`state`) | Add Operation (`add`) | Evict Operation (`remove`) | Check Complexity | Representative Problems |
|---|---|---|---|---|---|
| **Interval Sum / Average** | Scalar `curr_sum: int` | `+= nums[right]` | `-= nums[left]` | $O(1)$ | LC 209, LC 643 |
| **Filtered Element Count** | Scalar `count: int` | Match: `+ 1` | Match: `- 1` | $O(1)$ | LC 1004, LC 1248 |
| **Fixed Alphabet Frequency** | Fixed array `int[26]` | `arr[c] += 1` | `arr[c] -= 1` | $O(1)$ array comparison | LC 567, LC 438, LC 424 |
| **Unbounded Key Frequency** | Hash map `defaultdict(int)` | `map[x] += 1` | `map[x] -= 1` (prune on 0) | $O(1)$ hash hit | LC 3, LC 340 |
| **Full Set Covering Predicate** | Hash map + scalar `have: int` | Threshold reached: `have += 1` | Below threshold: `have -= 1` | $O(1)$ scalar check | LC 76 |
| **Distinct Count (Distinct $K$)** | Hash map + scalar `distinct: int` | `0 -> 1`: `+= 1` | `1 -> 0`: `-= 1` | $O(1)$ scalar check | LC 992 |
| **Dynamic Maximum / Minimum** | Monotonic deque `deque` (indices) | Pop dominated tail elements | Expired front: `popleft()` | $O(1)$ read front | LC 239, LC 1438 |
| **Dynamic Median / Kth Largest** | Dual heaps + lazy deletion map | Insert & rebalance heaps | Mark in lazy deletion map | $O(1)$ read, $O(\log k)$ update | LC 480 |

### Engineering Principles

1. **Lightweight Degradation Principle: Prefer scalars over collections, and fixed arrays over hash maps.**
   If counting zeros, declare `int zeros = 0`, never a `set`. When alphabet is `a-z`, allocate `[0] * 26`, which executes 3x to 5x faster than `defaultdict`.
2. **Scalar Anchoring Principle: Always pair composite state with a scalar; never perform full scans inside `while`.**
   When verifying multi-character requirements, tie an integer counter `have` to the hash table. Loop entry conditions must be $O(1)$ scalar predicates (`have == required`), never an $O(|\Sigma|)$ full iteration.
3. **Index Fidelity Principle: Extremum queues must store indices, never raw values.**
   If a deque stores only values, it is impossible to determine whether the front element expired past the left boundary or remains valid inside the window. Store indices and map to values `nums[q[0]]`; expiration checks `q[0] < left` become instantaneous.

---

## Template: Three Questions & Three Slots Framework

What all sliding window problems truly share is not an arbitrary line of code, but a structured mental decision pipeline:

```text
                    ┌────────────────────────────┐
                    │  for right, item in ...:   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 1. Enter: add_right(item)  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 2. Is window size fixed k? │
                    └──────┬──────────────┬──────┘
                           │ Yes          │ No
                           ▼              ▼
         ┌────────────────────────┐  ┌─────────────────────────────────┐
         │ if len > k:            │  │ while shrink condition met:     │
         │   remove_left(left)    │  │   [Shortest: record min ans]    │
         │   left += 1            │  │   remove_left(left)             │
         └─────────────┬──────────┘  │   left += 1                     │
                       │             └─────────────┬───────────────────┘
                       │                           │
                       ▼                           ▼
         ┌────────────────────────┐  ┌─────────────────────────────────┐
         │ if len == k:           │  │ [Longest: update after shrink]  │
         │   record_answer(...)   │  │ ans = max(ans, right-left+1)    │
         └────────────────────────┘  └─────────────────────────────────┘
```

### 1. Universal Skeleton Code

```python
def universal_sliding_window(items, k=None):
    left = 0
    state = initialize_state()
    answer = initialize_answer()

    for right, item in enumerate(items):
        # Slot 1: Add new element and update state (add_right)
        add_right(state, item)

        # Slot 2: Window shrinkage control (shrink)
        if k is not None:
            # Branch A: Fixed-Size Window
            if right - left + 1 > k:
                remove_left(state, items[left])
                left += 1

            # Slot 3: Record answer when window reaches fixed size k
            if right - left + 1 == k:
                record_fixed(answer, state, left, right)
        else:
            # Branch B: Variable-Size Window
            # Pattern I: Shortest Satisfying Window (e.g., LC 209, LC 76)
            while is_satisfied(state):
                record_min(answer, left, right)   # Record min before evicting
                remove_left(state, items[left])
                left += 1

            # Pattern II: Longest Valid Window (e.g., LC 3, LC 1004, LC 424)
            # while is_invalid(state):
            #     remove_left(state, items[left]) # Evict until restored to valid
            #     left += 1
            # record_max(answer, left, right)     # Update max once valid
    return answer
```

### 2. Decision Matrix for the Three Archetypes

| Window Archetype | Classic Problems | Shrink Condition (`shrink`) | Control Statement | Record Answer Timing (`record`) | Core Principle |
|---|---|---|---|---|---|
| **Longest Valid Window** | LC 3, LC 1004, LC 424 | Window violates rules (**invalid**) | `while invalid:` | **After `while` loop completes** (when restored to valid) | Expand while valid, shrink when invalid, record max outside loop |
| **Shortest Satisfying Window** | LC 209, LC 76 | Window **satisfies** target goal | `while satisfied:` | **Inside `while` loop** (before each eviction) | Shrink when satisfied to minimize, record min inside loop |
| **Fixed-Size Window** | LC 567, LC 438, LC 239 | Window length **exceeds $k$** | `if len > k:` | **When window length equals $k$** | At most 1 element evicted per turn, evaluate at size $k$ |

---

## 1. Longest Substring Without Repeating Characters

### Problem Description

Given a string `s`, find the length of the **longest continuous substring** without duplicate characters. $s$ length is up to $5\times10^4$.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `count[char]`, hash map tracking character frequencies in the current window |
| `add_right` | `count[char] += 1` |
| `shrink` Condition | `while count[char] > 1` (current incoming character creates a duplicate) |
| `remove_left` | `count[s[left]] -= 1`; `left += 1` |
| `record` Timing | After `while` finishes (window is valid without duplicates), `ans = max(ans, right - left + 1)` |

```python
from collections import defaultdict


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        count = defaultdict(int)
        left = 0
        answer = 0

        for right, char in enumerate(s):
            count[char] += 1

            # Slot 2: When duplicate occurs, shrink until valid
            while count[char] > 1:
                count[s[left]] -= 1
                left += 1

            # Slot 3: Record max length once window is restored to valid
            answer = max(answer, right - left + 1)

        return answer
```

```longest-substring-demo
```

### Complexity Analysis

The outer pointer `right` advances $n$ times. The inner pointer `left` only advances monotonically to the right, incrementing at most $n$ times in total. Each character enters the window once and exits at most once. Amortized time complexity is $O(n)$, and space complexity is $O(|\Sigma|) \le O(128) = O(1)$.

---

## 2. Longest Repeating Character Replacement

### Problem Description

Given a string `s` consisting of uppercase English letters and an integer `k`. You can choose at most `k` characters and replace them with any other uppercase character. Return the length of the longest substring containing the same letter you can achieve. $s$ length is up to $10^5$.

Whether a window can be converted into a uniform character string with at most $k$ replacements depends solely on the **window length** and the **frequency of the most frequent character**:

$$
\text{replacements} = \text{window\_length} - \text{max\_frequency} \le k.
$$

We simply preserve the highest-frequency character and replace all other characters.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `count[char]` and historical peak frequency `max_freq` |
| `add_right` | `count[char] += 1`; `max_freq = max(max_freq, count[char])` |
| `shrink` Condition | `while right - left + 1 - max_freq > k` (required replacements exceed budget $k$) |
| `remove_left` | `count[s[left]] -= 1`; `left += 1` |
| `record` Timing | Once valid, update `ans = max(ans, right - left + 1)` |

```python
from collections import defaultdict


class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = defaultdict(int)
        left = 0
        max_freq = 0
        answer = 0

        for right, char in enumerate(s):
            count[char] += 1
            max_freq = max(max_freq, count[char])

            # Slot 2: Shrink window when replacements exceed k
            while right - left + 1 - max_freq > k:
                count[s[left]] -= 1
                left += 1

            # Slot 3: Record max window length
            answer = max(answer, right - left + 1)

        return answer
```

### Why `max_freq` Does Not Need to Decrement During Shrink

This is a classic **historical proof-of-work optimization**:
We only care whether a window can beat the historical maximum length. If after shrinking the actual highest character frequency in the current window decreases, keeping `max_freq` at its historical peak causes no false positives:
1. If no incoming character ever breaks this historical peak, the window can never exceed our already discovered optimum anyway.
2. Only when an incoming character's frequency strictly surpasses `max_freq` will the window legitimately expand to record a larger answer.

Thus, allowing `max_freq` to remain monotonic non-decreasing preserves mathematical correctness while eliminating the $O(26)$ search on each shrink.

---

## 3. Permutation in String

### Problem Description

Given two strings `s1` and `s2`, return `True` if `s2` contains a permutation of `s1`. A permutation requires: **identical length and identical character frequencies across all 26 lowercase letters**. This is a canonical **Fixed-Size Window** problem where window size is fixed at $k = \text{len}(s1)$.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | Two 26-element arrays: `need[26]` and `window[26]` |
| `add_right` | `window[ord(char) - ord('a')] += 1` |
| `shrink` Condition | `if right - left + 1 > k:` (exceeds length $k$, single eviction) |
| `remove_left` | `window[ord(s2[left]) - ord('a')] -= 1`; `left += 1` |
| `record` Timing | `if right - left + 1 == k and window == need: return True` |

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False

        need = [0] * 26
        window = [0] * 26

        for char in s1:
            need[ord(char) - ord('a')] += 1

        k = len(s1)
        left = 0

        for right, char in enumerate(s2):
            window[ord(char) - ord('a')] += 1

            # Slot 2: Fixed-size window eviction (at most 1 element)
            if right - left + 1 > k:
                window[ord(s2[left]) - ord('a')] -= 1
                left += 1

            # Slot 3: Evaluate comparison when window reaches full size k
            if right - left + 1 == k and window == need:
                return True

        return False
```

Comparing two 26-element arrays takes $O(26)$. Total time complexity is strictly $O(26 \cdot n) = O(n)$.

---

## 4. Minimum Window Substring

### Problem Description

Given strings `s` and `t`, return the **minimum window substring** of `s` that covers all characters in `t` (including duplicate frequencies). If no such substring exists, return `""`.

### Type Count Compression: Achieving $O(1)$ Validity via `have`

Directly comparing frequency dictionaries on each step takes $O(|\Sigma|)$. Instead, introduce a match counter `have`:
- `required = len(need)`: total number of **distinct characters** in `t`.
- `have`: number of distinct characters in the current window whose frequency has reached or exceeded `need[c]`.
- The window is valid if and only if:

$$
\text{have} == \text{required}.
$$

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `need`, `window`, `have`, `required` |
| `add_right` | `window[c] += 1`; if `window[c] == need[c]` then `have += 1` |
| `shrink` Condition | `while have == required:` (target satisfied, shrink to minimize length) |
| `record` Timing | **Inside `while` loop before evicting**: record candidate minimum first |
| `remove_left` | If `window[old] == need[old]` then `have -= 1`; `window[old] -= 1`; `left += 1` |

```python
from collections import Counter, defaultdict


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ""

        need = Counter(t)
        window = defaultdict(int)
        required = len(need)
        have = 0

        left = 0
        best_start = 0
        best_len = float('inf')

        for right, char in enumerate(s):
            window[char] += 1
            if char in need and window[char] == need[char]:
                have += 1

            # Slot 2: Window satisfies target, shrink to find minimum
            while have == required:
                length = right - left + 1
                # Slot 3: Record optimal candidate before evicting left
                if length < best_len:
                    best_start = left
                    best_len = length

                old = s[left]
                if old in need and window[old] == need[old]:
                    have -= 1
                window[old] -= 1
                left += 1

        if best_len == float('inf'):
            return ""
        return s[best_start:best_start + best_len]
```

---

## 5. Sliding Window Maximum

### Problem Description

Given an integer array `nums` and a sliding window size `k`. The window slides from left to right one position at a time. Return the max sliding window. $n \le 10^5$.

Naive search takes $O(nk)$. A max heap takes $O(n\log k)$. Using a **Double-Ended Monotonic Queue (Monotonic Deque)** achieves optimal $O(n)$ time complexity.

### Monotonic Deque Mathematical Invariants

The deque **stores array indices only** (indices are required to determine whether an entry has expired beyond the left boundary), maintaining two strict mathematical invariants:

1. **Index Monotonicity**: Indices in the deque strictly increase from head to tail:
   $$q[0] < q[1] < \dots < q[m-1]$$
2. **Value Monotonicity**: Corresponding values strictly decrease from head to tail:
   $$\text{nums}[q[0]] > \text{nums}[q[1]] > \dots > \text{nums}[q[m-1]]$$
3. **Extremum Property**: **The front element $q[0]$ is unconditionally the maximum of the current window**, accessible in $O(1)$ time.

### Queue Elimination Mechanics (Why Tail and Head Evictions are Safe)

- **Tail Elimination (Domination Principle)**:
  When examining incoming element $\text{nums}[\text{right}]$, if deque tail $\text{tail} = q[-1]$ satisfies $\text{nums}[\text{tail}] \le \text{nums}[\text{right}]$, pop $\text{tail}$ from the back (`pop()`).
  *Proof*: For any current or future window containing $\text{right}$, index $\text{tail}$ will expire earlier, and its value is no greater than $\text{nums}[\text{right}]$. Thus $\text{tail}$ can **never again become a window maximum** (it is both older and weaker). It is permanently dominated and can be discarded.
- **Head Expiration (Expiration Principle)**:
  When the window advances its left boundary to $\text{left}$, if the front element $q[0] < \text{left}$ (or in fixed window logic $q[0] == \text{left}$), the old maximum has fallen outside the window boundary. It must be popped from the front (`popleft()`).

```sliding-window-max-demo
```

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | Double-ended queue storing indices with strictly decreasing values: `candidates = deque()` |
| `add_right` | Pop all dominated smaller tail elements, then append `right` |
| `shrink` Condition | `if right - left + 1 > k:` (fixed-size window exceeds $k$) |
| `remove_left` | If the front maximum expired (`candidates[0] == left`), pop it (`popleft()`); `left += 1` |
| `record` Timing | `if right - left + 1 == k:` append `nums[candidates[0]]` to answer |

```python
from collections import deque
from typing import List


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        candidates = deque()
        answer = []
        left = 0

        for right, value in enumerate(nums):
            # Slot 1: Add right. Pop weaker tail candidates to preserve strict monotonic decrease
            while candidates and nums[candidates[-1]] <= value:
                candidates.pop()
            candidates.append(right)

            # Slot 2: Fixed window shrink. Evict front if it expired
            if right - left + 1 > k:
                if candidates[0] == left:
                    candidates.popleft()
                left += 1

            # Slot 3: When window size reaches k, read front maximum
            if right - left + 1 == k:
                answer.append(nums[candidates[0]])

        return answer
```

### Why the Nested `while` is Strictly Amortized $O(n)$

Analyzing the macro lifetime of each index $i \in [0, n-1]$:
- Push: `candidates.append(i)` executes exactly once per index.
- Pop: Either evicted from tail by a larger element (at most once) or evicted from head due to expiration (at most once).

Total push and pop operations across the entire execution cannot exceed $2n$. Thus time complexity is strictly $O(n)$, with auxiliary space $O(k)$.

### 5.1 Canonical Variant: Fixed Window (LC 239) vs Streaming Trailing Window Max (Warm-up)

In quantitative finance high-frequency signal extraction and online streaming risk systems (e.g., maximum drawdown over the past 15 minutes, peak concurrency), a variant closely aligned with production realities often appears: **Trailing Window Maximum with Warm-up**.

#### Core Mechanism Comparison

| Dimension | Standard LC 239 (Fixed Window, Batch) | Trailing / Rolling Window Max (Streaming, Online) |
|---|---|---|
| **Window Span** | Strictly fixed at size $k$ | First $n-1$ points undergo warm-up (size $t+1 \le n$), then stabilize at $n$ |
| **Output Timing** | Must wait until window accumulates $k$ items (`right >= k - 1`) | **Emits output immediately at every timestamp $t$** |
| **Output Length** | $N - k + 1$ | Strictly equals input length $N$ |
| **Head Expiry** | `candidates[0] == left` (or `<= right - k`) | `q[0] <= t - n` (active lookback interval is $[t - n + 1, t]$) |
| **Recording Logic**| Guarded by `if right >= k - 1: ans.append(nums[q[0]])` | Unguarded: append `res.append(xs[q[0]])` unconditionally at every step |

#### Streaming Rolling Max Implementation

```python
from collections import deque
from typing import List

def rolling_max(xs: List[float], n: int) -> List[float]:
    """
    Trailing window maximum (window size <= n).
    
    Computes the maximum within lookback interval [max(0, t - n + 1), t] at each step t.
    Produces output immediately starting from t = 0; output length strictly equals len(xs).
    """
    if not xs or n <= 0:
        return []

    q = deque()  # stores indices; maintains strictly decreasing values
    res = []

    for t, val in enumerate(xs):
        # 1. Head expiry: active window left bound is t - n + 1; pop indices <= t - n
        while q and q[0] <= t - n:
            q.popleft()

        # 2. Tail domination: older and smaller/equal elements are strictly dominated; pop them
        while q and xs[q[-1]] <= val:
            q.pop()

        # 3. Push current timestamp index
        q.append(t)

        # 4. Deque front is the maximum for current window; emit immediately
        res.append(float(xs[q[0]]))

    return res
```

The monotonic eviction logic between the two variants is **mathematically identical**. The distinguishing feature of the streaming variant is its **warm-up phase**, which eliminates cold-start voids while preserving online causality without lookahead bias.

---

## 6. Minimum Size Subarray Sum

### Problem Description

Given an array of $n$ **positive integers** `nums` and a positive integer `target`. Return the minimal length of a contiguous subarray whose sum is greater than or equal to `target`. If there is no such subarray, return 0.

This is the quintessential baseline for mastering the **Shortest Satisfying Window** archetype, sharing the exact same shrink and record timing as LC 76.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `curr_sum`, running sum of elements in the window |
| `add_right` | `curr_sum += nums[right]` |
| `shrink` Condition | `while curr_sum >= target:` (target reached, shrink left to minimize length) |
| `record` Timing | **Inside `while` loop before evicting**: `ans = min(ans, right - left + 1)` |
| `remove_left` | `curr_sum -= nums[left]`; `left += 1` |

```python
from typing import List


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        curr_sum = 0
        ans = float('inf')

        for right, val in enumerate(nums):
            # Slot 1: Add element to window sum
            curr_sum += val

            # Slot 2: Target met, shrink left to find minimal length
            while curr_sum >= target:
                # Slot 3: Record min length inside loop before evicting
                ans = min(ans, right - left + 1)
                curr_sum -= nums[left]
                left += 1

        return 0 if ans == float('inf') else ans
```

### In-Depth Theory: Why Must Elements be Strictly Positive?

Sliding window algorithms require **Monotonicity**:
- When all elements are positive, advancing `right` strictly increases `curr_sum`, and advancing `left` strictly decreases `curr_sum`.
- If the array contains **negative numbers**, expanding the window might decrease the sum, and shrinking the window might increase the sum. Monotonicity collapses, causing standard two-pointer sliding window to fail!
- **Solution for Negative Numbers**: To find the shortest subarray with sum $\ge target$ on general arrays with negatives (such as [LC 862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)), one must use **Prefix Sums + Monotonic Deque** in $O(n)$ time.

---

## 7. Max Consecutive Ones III

### Problem Description

Given a binary array `nums` and an integer `k`, return the maximum number of consecutive `1`'s in the array if you can flip at most `k` `0`'s.

### Problem Abstraction: Translating to Longest Valid Window

"Flipping at most $k$ zeros" is equivalent to: **Find the longest contiguous subarray containing at most $k$ zeros**.
- Valid Condition: $\text{zeros} \le k$.
- Invalid Condition: $\text{zeros} > k$.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `zeros`, count of zero elements within the window |
| `add_right` | `if nums[right] == 0: zeros += 1` |
| `shrink` Condition | `while zeros > k:` (zero count exceeds flip budget $k$) |
| `remove_left` | `if nums[left] == 0: zeros -= 1`; `left += 1` |
| `record` Timing | After `while` loop (restored to valid), `ans = max(ans, right - left + 1)` |

```python
from typing import List


class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        zeros = 0
        ans = 0

        for right, val in enumerate(nums):
            # Slot 1: Update zero count
            if val == 0:
                zeros += 1

            # Slot 2: Shrink when zero count exceeds budget k
            while zeros > k:
                if nums[left] == 0:
                    zeros -= 1
                left += 1

            # Slot 3: Record max length once restored to valid
            ans = max(ans, right - left + 1)

        return ans
```

---

## 8. Find All Anagrams in a String

### Problem Description

Given two strings `s` and `p`, return an array of all the **start indices** of `p`'s anagrams in `s`.

This is isomorphic to LC 567 (Fixed-Size Window of length $k = \text{len}(p)$). The only difference is that LC 567 returns boolean `True` on the first match, whereas LC 438 appends `left` to a results list on every match.

### Filling the Template Slots

```python
from typing import List


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(p) > len(s):
            return []

        need = [0] * 26
        window = [0] * 26

        for char in p:
            need[ord(char) - ord('a')] += 1

        k = len(p)
        left = 0
        result = []

        for right, char in enumerate(s):
            # Slot 1: Increment frequency of incoming character
            window[ord(char) - ord('a')] += 1

            # Slot 2: Fixed-size window eviction
            if right - left + 1 > k:
                window[ord(s[left]) - ord('a')] -= 1
                left += 1

            # Slot 3: Collect match index at size k
            if right - left + 1 == k and window == need:
                result.append(left)

        return result
```

---

## 9. Exact K via Dual Sliding Window

When confronting problems like [LC 992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/) or [LC 1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/), a standard sliding window gets stuck because **"exactly $k$" is non-monotonic**: expanding the window can enter or leave the exact $k$ state intermittently.

### Dimension Reduction: The Identity Decomposition

Convert the non-monotonic "exactly $k$" problem into the difference of two **strictly monotonic "at most $k$" problems**:

$$
\text{Exact}(k) = \text{atMost}(k) - \text{atMost}(k - 1).
$$

For an "at most $k$ distinct elements" window, if $[\text{left}, \text{right}]$ is valid, then **all subarrays ending at $\text{right}$ with start $\ge \text{left}$ (exactly $\text{right} - \text{left} + 1$ subarrays) are unconditionally valid**!

```python
from collections import defaultdict
from typing import List


class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def atMost(k_max: int) -> int:
            if k_max == 0:
                return 0
            count = defaultdict(int)
            left = 0
            distinct = 0
            total = 0

            for right, num in enumerate(nums):
                if count[num] == 0:
                    distinct += 1
                count[num] += 1

                while distinct > k_max:
                    count[nums[left]] -= 1
                    if count[nums[left]] == 0:
                        distinct -= 1
                    left += 1

                # Number of valid subarrays ending at right with <= k_max distinct elements
                total += right - left + 1

            return total

        return atMost(k) - atMost(k - 1)
```

---

## Complete Problem Matrix & Slot Comparison

| Problem | LC # | Window Archetype | `state` Tracked | Shrink Condition & Control | Record Timing & Formula |
|---|---|---|---|---|---|
| **Longest Substring** | LC 3 | Longest Valid | Char frequencies | `while count[c] > 1` | Outside loop: `ans = max(ans, len)` |
| **Max Consecutive Ones III** | LC 1004 | Longest Valid | Zero count `zeros` | `while zeros > k` | Outside loop: `ans = max(ans, len)` |
| **Character Replacement** | LC 424 | Longest Valid | Frequencies + `max_freq` | `while len - max_freq > k` | Outside loop: `ans = max(ans, len)` |
| **Min Size Subarray Sum** | LC 209 | Shortest Satisfying | Running sum `sum` | `while sum >= target` | Inside loop: `ans = min(ans, len)` |
| **Minimum Window Substring** | LC 76 | Shortest Satisfying | `need/window` + `have` | `while have == required` | Inside loop: `ans = min(ans, len)` |
| **Permutation in String** | LC 567 | Fixed-Size | 26-elem frequency array | `if len > k` (evict 1) | At size $k$: compare and return `True` |
| **Find All Anagrams** | LC 438 | Fixed-Size | 26-elem frequency array | `if len > k` (evict 1) | At size $k$: append matching `left` |
| **Sliding Window Maximum** | LC 239 | Fixed-Size | Monotonic index deque | `if len > k` (evict head) | At size $k$: read `nums[deque[0]]` |

---

## Common Pitfalls & Debugging Checklist

| Common Mistake | Root Cause & Failure Mechanism | Correct Practice |
|---|---|---|
| **Using `if` instead of `while` in variable window** | An incoming element can cause multiple violations, requiring continuous eviction | Always use `while` for variable window invariant restoration |
| **Using `while` blindly in fixed window** | A fixed window adds 1 element per step, exceeding by at most 1 | Use `if right - left + 1 > k` to clearly express fixed-window mechanics |
| **Omitting `+ 1` in window length** | Confuses interval distance with element cardinality | Closed interval element count is strictly `right - left + 1` |
| **Recording longest window before shrinking** | Window is currently in an invalid violation state, polluting the optimal answer | For longest valid windows, update `max` **after** `while` loop completes |
| **Recording shortest window after shrinking** | The window may no longer satisfy the target after eviction, missing the minimal valid state | For shortest satisfying windows, update `min` **inside** `while` before eviction |
| **Advancing `left` before mutating state** | Pointer increment shifts `left` to the subsequent element, corrupting state synchronization | Always subtract `items[left]` from state **first**, then `left += 1` |
| **Storing values instead of indices in deque** | Values alone cannot indicate when an element has fallen outside window bounds | Monotonic queues must store **indices**, referencing values as needed |

---

## How to Apply the Template to New Problems

Follow this 4-step deduction checklist on any new sliding window problem:

```text
Step 1: Is the window length fixed?
        -> Fixed length k: Control with `if len > k`, record at size k.
        -> Variable length: Proceed to Step 2 and 3.

Step 2: What is the optimization goal?
        -> "Longest valid": while invalid shrink, update max outside loop.
        -> "Shortest satisfying": while satisfied shrink, update min inside loop.
        -> "Exact K count": Decompose into `atMost(K) - atMost(K-1)`.

Step 3: How is state incrementally updated and reverted?
        -> Entering items[right]: What state variables can be maintained in O(1)?
        -> Evicting items[left]: How can the state be reversed symmetrically?

Step 4: Verify the monotonicity assumption!
        -> Does moving pointers monotonically change the validity metrics?
        -> If negative values exist (e.g. subarray sum with negatives), sliding window fails;
           switch to Prefix Sums + Monotonic Deque / Hash Map.
```

---

## Template Summary

```text
[Universal 3-Slot Summary]
Add right to state upon entry,
Shrink left to maintain the boundary,
Record longest after loop is done,
Record shortest before eviction run,
Record fixed window when size is won!
```


## Module 3: Sliding Window High-Frequency Extensions

### LC 340. Longest Substring with At Most K Distinct Characters

#### Core Mental Model (Variable Window + Frequency Map)
Expand right pointer into window and increment frequency. While distinct character count exceeds $k$, shrink left pointer, deleting entries that reach frequency 0. Record maximum length $r - l + 1$. Time: $O(n)$, Space: $O(k)$.

```python
from collections import defaultdict

class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        counts = defaultdict(int)
        l, ans = 0, 0
        for r, ch in enumerate(s):
            counts[ch] += 1
            while len(counts) > k:
                counts[s[l]] -= 1
                if counts[s[l]] == 0:
                    del counts[s[l]]
                l += 1
            ans = max(ans, r - l + 1)
        return ans
```
