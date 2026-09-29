# Stack · MinStack and Monotonic Stack

The stack API is simple. The two patterns that matter in interviews are:

```text
MinStack: preserve history snapshots during push so that extreme-value queries never rescan
Monotonic Stack: maintain an active candidate set of unresolved elements, waiting for right-side elements to trigger eviction and establish boundaries
```

MinStack is a standalone state-snapshot design; Monotonic Stack is a unified family of problems resolving "1D nearest local extreme boundaries". Mastering Monotonic Stack requires establishing the **"Three-Question Four-Slot" Universal Blueprint**, mapping any problem to this framework by merely swapping eviction comparators and settlement slots.

## Learning Order

Selected from high-frequency core interview problems to build progressive mastery from state snapshots to full dual-boundary monotonic stacks:

| Order | Original Problem | Core Pattern | Key Engineering Problem Solved |
|---:|---|---|---|
| 1 | [155. Min Stack](https://neetcode.io/problems/minimum-stack/question?list=neetcode150) | State-Snapshot Pattern | Bind `min_so_far` to each stack frame for $O(1)$ query and rollback |
| 2 | [739. Daily Temperatures](https://neetcode.io/problems/daily-temperatures/question?list=neetcode150) | Monotonic Stack · Right Boundary (Next Greater) | Slot 3A: Settle waiting distance via index delta upon eviction |
| 3 | [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/) | Monotonic Stack · Circular Array | Slot 1: Virtual $2n$ modulo doubling, seamlessly reusing the template |
| 4 | [84. Largest Rectangle in Histogram](https://neetcode.io/problems/largest-rectangle-in-histogram/question?list=neetcode150) | Monotonic Stack · Dual Boundaries | Slot 1 sentinel padding; Slot 3A resolves both left and right boundaries simultaneously |
| 5 | [42. Trapping Rain Water](https://neetcode.io/problems/trapping-rain-water/question?list=neetcode150) | Monotonic Stack · Trough Fill | Eviction identifies trough floor; stack top and current bar form horizontal bounding box |

---

## Module 1: MinStack

### 1.1 · State-Snapshot Pattern: Storing History at Every Stack Level

If we only track a single global variable `minimum`, `push` is straightforward, but once the current minimum is popped, the system loses the historical previous minimum. Rescanning the stack costs $O(n)$.

The most robust engineering pattern is the **State-Snapshot Pattern**: store a pair in each stack frame:

```text
(current value, min_so_far after pushing this value)
```

The stack top then carries dual information in $O(1)$:

```text
top()    = stack[-1][0]
getMin() = stack[-1][1]
```

Every `push` takes an immutable snapshot of the current state. When `pop` removes a snapshot, the historical minimum stored in the frame below is instantly restored without needing rollback logic or secondary synchronization.

### 1.2 · Why Duplicate Minimum Values Do Not Desynchronize

Push `[2, 1, 1]` in sequence:

```text
1. push(2) -> stack: [(2, 2)]
2. push(1) -> stack: [(2, 2), (1, 1)]
3. push(1) -> stack: [(2, 2), (1, 1), (1, 1)]
```

When one `1` is popped, the frame below still holds `(1, 1)`, keeping the minimum at `1`. Auxiliary two-stack approaches that only push strictly smaller values require delicate conditional checks during `pop`, making synchronization error-prone. The tuple snapshot approach is algebraically self-contained with minimal branching.

### 1.3 · Quick Coding: Implement MinStack

Implement `push`, `pop`, `top`, and `getMin`, all in $O(1)$ time.

<details>
<summary>Reference Implementation (State-Snapshot)</summary>

```python
class MinStack:
    def __init__(self):
        # Stack frames store (val, min_so_far)
        self.stack = []

    def push(self, val: int) -> None:
        current_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

- **Time Complexity**: Strictly $O(1)$ for all operations.
- **Space Complexity**: $O(n)$ space to store snapshot tuples.

</details>

### 1.4 · Advanced Extension: Value-Difference Encoding for Optimal Space

In embedded or ultra-low-memory environments where tuple allocation overhead is unacceptable, the **Value-Difference Encoding** pattern uses a single stack of integer deltas and a scalar `min_val`:

1. Store `diff = val - min_val` in the stack.
2. If `val < min_val`, `diff < 0` is pushed, followed by updating `min_val = val`.
3. During `pop()`, observing `diff < 0` indicates the popped element established the current minimum; restore the previous minimum via `min_val = min_val - diff`.

This eliminates all tuple overhead, with the caveat of guarding against integer overflow in typed languages (e.g., C++ `int64_t`).

---

## Module 2: Monotonic Stack

### 2.1 · Core Mechanics & What Problems It Solves

A Monotonic Stack fundamentally answers:

> **In a 1D sequence, for each position, find the nearest greater or smaller element to its left or right.**

Diverse problem statements map to this identical underlying query:

| Problem Statement | Mathematical Essence | Monotonic Stack Mechanics |
|---|---|---|
| Days until warmer temperature? | First strictly greater element on the right (Next Greater) | Record index difference $i - j$ |
| Next greater element in array? | First greater element in circular array (Next Greater) | Record actual value $nums[i]$ |
| Maximum width a histogram bar can extend? | Nearest smaller elements on both sides (Dual Smaller Boundaries) | Eviction locks both left and right boundaries |
| Trapping rain water volume? | Bounding bars on left and right forming a trough | Eviction determines trough floor; stack top and current bar set water height |
| Sum of subarray minimums? | Maximal bounding interval $[L+1, R-1]$ where current element is minimum | Dual boundaries with tie-breaking multiplication |

#### Why Brute Force is $O(n^2)$ While Monotonic Stack is $O(n)$?
Brute force scans linearly left or right for every element, yielding $\sum_{i=1}^n i = O(n^2)$ worst-case time on monotonic inputs.
A monotonic stack maintains an **active waiting set of unresolved indices**:
- Each index is pushed **exactly once**;
- Each index is popped **at most once**;
- Total `push` and `pop` operations across the entire algorithm are bounded by $\le 2n$, guaranteeing strict $O(n)$ amortized time.

#### Why Store Indices Instead of Values?
Indices provide three complete dimensions of information simultaneously:
1. **Value Lookup**: $nums[j]$ recovers the value in $O(1)$;
2. **Span Metric**: Calculate geometric distance $\Delta = i - j$ or rectangle width $W = R - L - 1$;
3. **Direct Writeback**: Directly index the result array $ans[j]$ without hash table lookups or collision risks.

---

### 2.2 · Universal Monotonic Stack Blueprint: The Three-Question Four-Slot Model

All monotonic stack problems share a unified mental model and code skeleton:

```text
                           ┌────────────────────────────┐
                           │   for i, current in arr:   │
                           └─────────────┬──────────────┘
                                         │
                                         ▼
                           ┌────────────────────────────┐
                           │ Slot 1 [Sentinel & Init]   │
                           │ arr = nums + [0] (Optional)│
                           └─────────────┬──────────────┘
                                         │
                                         ▼
                           ┌────────────────────────────┐
        ┌─────────────────►│ Slot 2 [Eviction Predicate]│
        │                  │ while stack and pop_cond:  │
        │                  └──────┬──────────────┬──────┘
        │                         │ True         │ False
        │                         ▼              ▼
        │             ┌───────────────────────┐  ┌────────────────────────┐
        │             │ mid = stack.pop()     │  │ Slot 3B [Prev Boundary]│
        │             │                       │  │ ans[i] = stack[-1]     │
        │             │ Slot 3A [Next/Dual]   │  └───────────┬────────────┘
        │             │ ans[mid] = i / Area   │              │
        │             └───────────┬───────────┘              │
        │                         │                          │
        └─────────────────────────┘                          ▼
                                                 ┌────────────────────────┐
                                                 │ Slot 4 [Push to Wait]  │
                                                 │ stack.append(i)        │
                                                 └────────────────────────┘
```

#### 1. Three Clarification Questions

- **Q1: Direction & Attribution**
  - **Right-Side Boundary (Next-X)**: The current element $i$ acts as the "resolver". When it violates monotonicity, it evicts top index $j$ and answers $j$ with current position $i$ (**settled inside `while` loop at Slot 3A**).
  - **Left-Side Boundary (Prev-X)**: The current element $i$ is the "target". The `while` loop cleans away invalid candidates; the surviving stack top is the nearest valid left boundary for $i$ (**settled after `while` loop at Slot 3B**).
  - **Dual Boundaries**: When $mid = stack.pop()$ occurs:
    - Current index $i$ is the **first smaller/greater on the right** ($R = i$);
    - The newly exposed stack top $stack[-1]$ is the **nearest smaller/greater on the left** ($L = stack[-1]$);
    - A single eviction locks both boundaries, establishing the maximal span interval $[L + 1, R - 1]$.

- **Q2: Strictness & Tie-Breaking Principle**
  - **Finding Greater Elements**: Pop when current is greater; stack maintains monotonic decreasing order.
  - **Finding Smaller Elements**: Pop when current is smaller; stack maintains monotonic increasing order.
  - **Exact Partitioning Theorem**:
    - When duplicate values exist and the problem counts subarray contributions (e.g., LC 907, LC 84):
      - Strict inequality on both sides (`<` and `>`) **under-counts** subarrays between duplicates;
      - Non-strict inequality on both sides (`<=` and `>=`) **over-counts** duplicates;
      - **Golden Rule**: You MUST set **one side strict (e.g. left strictly smaller `<`) and the other non-strict (e.g. right smaller or equal `<=`)**, establishing mutually disjoint and exhaustive half-open intervals.

- **Q3: Storage Carrier & Answer Form**
  - Always store indices in the stack. Populate answers as index values, waiting distances ($i - j$), spans ($R - L - 1$), or geometric areas.

---

#### 2. Universal Template Code

```python
from typing import List, Optional


def universal_monotonic_stack(
    nums: List[int],
    mode: str = "next_greater",  # "next_greater" | "next_smaller" | "prev_greater" | "prev_smaller" | "dual_smaller"
    with_sentinel: bool = False,
    sentinel_val: int = 0,
) -> List[int]:
    """Universal Monotonic Stack Blueprint

    4 Parameterized Slots:
    [Slot 1] Sentinel & Initialization: Initialize answer container and optional sentinel
    [Slot 2] Eviction Predicate: Evaluates whether current value breaks monotonicity
    [Slot 3] Settlement Actions:
             - Slot 3A: Settle upon eviction (Next or Dual boundary problems)
             - Slot 3B: Settle after eviction (Prev boundary problems)
    [Slot 4] Push to Wait: Enqueue current index to await future resolvers
    """
    n = len(nums)

    # ──────────────────────────────────────────────────────
    # Slot 1: Sentinel Padding & Container Init
    # ──────────────────────────────────────────────────────
    arr = nums + [sentinel_val] if with_sentinel else nums
    limit = len(arr)
    ans = [-1] * n
    stack = []  # Strictly stores indices

    # ──────────────────────────────────────────────────────
    # Slot 2: Eviction Predicate
    # ──────────────────────────────────────────────────────
    def should_pop(top_val: int, curr_val: int) -> bool:
        if mode in ("next_greater", "prev_greater"):
            return curr_val > top_val
        elif mode in ("next_smaller", "prev_smaller", "dual_smaller"):
            return curr_val < top_val
        return False

    for i in range(limit):
        curr_val = arr[i]

        while stack and should_pop(arr[stack[-1]], curr_val):
            mid = stack.pop()

            # ──────────────────────────────────────────────────
            # Slot 3A: Settle upon Eviction (Next / Dual Boundaries)
            # ──────────────────────────────────────────────────
            if mode.startswith("next") and mid < n:
                ans[mid] = i  # Or distance: i - mid
            elif mode == "dual_smaller":
                left = stack[-1] if stack else -1
                right = i
                width = right - left - 1
                # Aggregate geometric area: ans = max(ans, arr[mid] * width)

        # ──────────────────────────────────────────────────────
        # Slot 3B: Settle after Eviction (Prev Boundary for Current i)
        # ──────────────────────────────────────────────────────
        if mode.startswith("prev") and i < n:
            ans[i] = stack[-1] if stack else -1

        # ──────────────────────────────────────────────────────
        # Slot 4: Push Current Index to Wait
        # ──────────────────────────────────────────────────────
        stack.append(i)

    return ans
```

---

### 2.3 · Comparator and Monotonicity Reference Table

Let `top = nums[stack[-1]]` be the historical top value, and `current = nums[i]` be the scanning value:

| Target Query | Eviction Condition (`while`) | Stack Order After Eviction (Bottom → Top) | Typical Use Cases |
|---|---|---|---|
| **First strictly greater on right** | `top < current` | Monotonic non-increasing | Daily Temperatures, Next Greater Element |
| **First greater or equal on right** | `top <= current` | Strictly decreasing | Pruning duplicates on the right |
| **First strictly smaller on right** | `top > current` | Monotonic non-decreasing | Largest Rectangle in Histogram, Subarray Min |
| **First smaller or equal on right** | `top >= current` | Strictly increasing | Exact partitioning for tie-breaking |

> **Universal Memory Rule**:
> Never memorize whether to use an increasing or decreasing stack. Simply ask: **"Does the current scanned element satisfy what the stack-top element is waiting for?"**
> If yes, evict and settle immediately.

The interactive visualizer below steps through "Next Greater" and "Next Smaller" on the same array:

```monotonic-stack-demo
```

---

### 2.4 · Universal Slot-Filling Matrix

Every problem maps directly into the 4 slots:

| Classic Problem | Mode | Slot 1 (Sentinel) | Slot 2 (Eviction Predicate) | Slot 3 (Settlement Calculation) | Slot 4 (Push) |
|---|---|---|---|---|---|
| **LC 739. Daily Temperatures** | Next Greater | None | `top < current` | Slot 3A: `ans[mid] = i - mid` | `stack.append(i)` |
| **LC 496. Next Greater Element I** | Next Greater | None | `top < current` | Slot 3A: `ans[mid] = current` | `stack.append(i)` |
| **LC 503. Next Greater Element II** | Circular Next Greater | Virtual $2n$ loop | `top < nums[i % n]` | Slot 3A: `ans[mid] = nums[i % n]` (when `mid < n`) | `if i < n: stack.append(i)` |
| **LC 84. Largest Rectangle** | Dual Smaller | Append `0` | `top > current` | Slot 3A: `w = i - stack[-1] - 1`<br>`ans = max(ans, heights[mid] * w)` | `stack.append(i)` |
| **LC 42. Trapping Rain Water** | Dual Greater | None | `top < current` | Slot 3A: Bounded water trough<br>`h = min(top, current) - mid_h` | `stack.append(i)` |
| **LC 907. Subarray Minimums** | Dual Smaller (Tie-break) | Append `0` | Left `<` strict, Right `<=` non-strict | Slot 3A: Product rule for subarrays<br>`count = (mid - left) * (right - mid)` | `stack.append(i)` |

---

### 2.5 · Canonical Problems & Blueprint Instantiation

#### Practice 1: Daily Temperatures
Given daily temperatures, return the number of days to wait until a warmer temperature.
- **Mapping**: Canonical **Next Greater** problem where answer format is the waiting span $i - mid$.

```python
from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []  # Stores indices

        for i, temp in enumerate(temperatures):
            # Slot 2: Current temperature higher than stack top
            while stack and temperatures[stack[-1]] < temp:
                mid = stack.pop()
                # Slot 3A: Settle distance delta upon eviction
                ans[mid] = i - mid
            # Slot 4: Enqueue current day
            stack.append(i)

        return ans
```

#### Practice 2: Next Greater Element II (Circular Array via Modulo)
Find the next greater element in a circular array.
- **Mapping**: Extend iteration to $2n$ in **Slot 1** using `i % n`, pushing to stack only during the first cycle ($i < n$).

```python
from typing import List


class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [-1] * n
        stack = []

        # Slot 1: Virtual doubling via 2*n iteration
        for i in range(2 * n):
            val = nums[i % n]
            # Slot 2: Eviction comparison
            while stack and nums[stack[-1]] < val:
                mid = stack.pop()
                # Slot 3A: Record next greater value
                ans[mid] = val
            # Slot 4: Only push during first cycle
            if i < n:
                stack.append(i)

        return ans
```

#### Practice 3: Largest Rectangle in Histogram (Dual Boundaries & Sentinel)
Find the area of the largest rectangle in the histogram.
- **Mathematical Derivation**: With bar $heights[mid]$ as height, the maximal span width is bounded by the **nearest strictly shorter bars on left and right**:
  $$W = \text{right} - \text{left} - 1$$
- **Sentinel Optimization**: Appending an artificial bar of height `0` at index $n$ forces eviction of all residual bars, eliminating post-loop cleanup code.

```largest-rectangle-demo
```

```python
from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        max_area = 0
        stack = []

        # Slot 1: Inject tail sentinel of height 0
        for right in range(len(heights) + 1):
            curr_h = 0 if right == len(heights) else heights[right]

            # Slot 2: Shorter bar breaks increasing monotonicity
            while stack and heights[stack[-1]] > curr_h:
                mid = stack.pop()
                mid_h = heights[mid]
                # Slot 3A: Left and right boundaries locked simultaneously
                left = stack[-1] if stack else -1
                width = right - left - 1
                max_area = max(max_area, mid_h * width)

            # Slot 4: Enqueue current index
            stack.append(right)

        return max_area
```

#### Practice 4: Trapping Rain Water (Horizontal Trough Fill)
- **Model**: Monotonic decreasing stack. When a taller bar is encountered, the evicted bar serves as the trough floor $mid$. The new stack top is the left boundary $left$, and current bar is the right boundary $right$.
- **Formulas**:
  $$H = \min(height[left], height[right]) - height[mid]$$
  $$W = right - left - 1$$
  $$\text{Water} = H \times W$$

```python
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        water = 0
        stack = []

        for right, curr_h in enumerate(height):
            # Slot 2: Taller bar encountered, forming bounding trough
            while stack and height[stack[-1]] < curr_h:
                mid = stack.pop()
                if not stack:
                    break  # No left boundary to trap water

                left = stack[-1]
                # Slot 3A: Horizontal trough slice accumulation
                h = min(height[left], curr_h) - height[mid]
                w = right - left - 1
                water += h * w

            # Slot 4: Enqueue current bar
            stack.append(right)

        return water
```

---

### 2.6 · Complexity Proof: Why Nested while Loops Run in Strict O(n) Time

A common pitfall is mistaking nested `while` inside `for` as $O(n^2)$.

The formal **Aggregate Analysis** proof is straightforward:
1. In an array of size $n$, each index enters the `for` loop and is pushed via `stack.append()` **at most once**;
2. An element can only be popped via `stack.pop()` if it currently resides in the stack, so each element is popped **at most once**;
3. Once an index is popped, it is permanently discarded and never re-enters the stack;
4. Therefore, across all $n$ outer iterations, the inner `while` condition evaluates to true and executes `pop()` at most $n$ times total.

$$\sum_{i=1}^n (\text{Push Count} + \text{Pop Count}) \le n + n = 2n = O(n)$$

Thus, the monotonic stack algorithm runs in strict $O(n)$ time and $O(n)$ auxiliary space in the worst case.

---

### 2.7 · Whiteboard Interview Checklist

When presenting a monotonic stack solution in technical interviews, structure your explanation across 5 clear milestones:

1. **Classification**: "This problem requires finding nearest extreme-value boundaries for each position in 1D. Brute force is $O(n^2)$; a monotonic stack optimizes this to linear $O(n)$ time."
2. **Data Structure**: "The stack stores indices rather than values because calculating spans $i - j$ and rectangle widths requires physical distances."
3. **Monotonicity**: "We maintain a monotonic decreasing stack because we are seeking the next greater element; any value violating this triggers immediate eviction."
4. **Attribution**: "Answers are settled upon eviction (Slot 3A) because the current scanned element serves as the active resolver for waiting elements."
5. **Sentinel**: "For histogram area or multi-interval aggregation, append a `0` sentinel to flush residual frames automatically without duplicate cleanup code."
