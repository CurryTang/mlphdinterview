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

### 2.2 · Intuitive Mental Models & The Epiphany of Eviction

Many engineers struggle with monotonic stacks because LIFO eviction initially feels unnatural. Three physical analogies explain the underlying invariants:

#### Model 1: The "Skyline Sightline Shadow" Model (Why Evicted Elements are Safely Forgotten)

Consider finding the next greater element, where array values represent building heights:
- As the scan pointer advances rightward, a tall building $B_{\text{current}}$ arrives, exceeding previous shorter buildings $B_{\text{old}}$.
- Why can we permanently pop and discard $B_{\text{old}}$ without ever regretting it later?
  - **For $B_{\text{old}}$ itself**: Looking rightward, the very first taller structure it encounters is $B_{\text{current}}$. Its question is completely resolved.
  - **For any future building $B_{\text{future}}$ further to the right**: Looking backward to the left, the towering $B_{\text{current}}$ stands directly in its line of sight. Shorter buildings like $B_{\text{old}}$ are cast entirely into $B_{\text{current}}$'s shadow (Eclipsed). Any sightline from the right will hit $B_{\text{current}}$ long before it could ever reach $B_{\text{old}}$!
  - Therefore, $B_{\text{old}}$ can **never** serve as the "nearest greater element to the left" for any future element. Its lifecycle is permanently complete, and popping it is algebraically sound and lossless.

```text
[The Skyline Eclipse Principle]
Height:
  6 │                █ (current: 6)
  5 │        █       █
  4 │        █   █   █ ─── Height 6 towers over everything to its left;
  2 │  █     █   █   █     any future building looking left will hit 6
  1 │  █  █  █   █   █     and NEVER see the shadowed bars 1 and 2!
────┼───────────────────────► Index progression
Idx:   0  1  2   3   4
```

#### Model 2: The "Bounty Board & Weakest Barrier" Model (Why Specifically a LIFO Stack?)

Why must a monotonic stack be a Last-In, First-Out (LIFO) stack rather than a queue or table?
- Each index in the stack represents an **unresolved bounty**: *"I am waiting for the first element taller than me on the right."*
- From bottom to top, the stack maintains strictly decreasing values: the bottom holds the tallest, strongest barriers; the top holds the shortest, most fragile **"weakest barrier"**.
- When a newcomer arrives, it only needs to challenge the **top of the stack (weakest barrier)**:
  - If the newcomer cannot even defeat the shortest element at the top, it has **zero chance** of defeating the taller elements deeper down! There is no need to search further: the check terminates in $O(1)$, and the newcomer posts its own bounty (`push`).
  - If the newcomer defeats the top, that bounty is collected (`pop` and answer recorded). The next-weakest barrier is exposed, and the challenge repeats until the newcomer hits an insurmountable wall.
- **Key Insight**: A LIFO stack ensures every search attacks the point of least resistance first, eliminating redundant comparisons.

#### Model 3: The "Two-Coin Amortized" Model (Why Nested while is Strictly O(n))

Nested loops (`while` inside `for`) naturally trigger fears of $O(n^2)$. The clearest proof of linearity is the **Two-Coin Token Model**:
- Give every element in the array exactly **two gold coins**:
  - **Coin 1**: Pays for its invocation of `push` into the stack;
  - **Coin 2**: Pre-pays for its eventual `pop` out of the stack.
- Once an element is popped, it is permanently discarded. There is no third coin to bring it back.
- If an iteration pops 5 elements in a single `while` loop, it is merely cashing in the second coins pre-paid by those 5 elements earlier.
- The total coins spent across the entire execution cannot exceed $2n$. Thus, total operations are strictly bounded by $O(n)$.

#### The Core Epiphany: The Spatiotemporal Collapse of Eviction (Locking Both Boundaries)

In Largest Rectangle in Histogram (LC 84) and Trapping Rain Water (LC 42), the most profound question is: *"How does a single pop simultaneously establish both the left and right boundaries of an element?"*

At the exact microsecond `mid = stack.pop()` executes:
1. **Right Boundary ($R$)**: It is the current scanning element $i$. Because $arr[i]$ broke the monotonic property, $i$ is guaranteed to be the **first smaller/greater barrier to the right** of $mid$.
2. **Left Boundary ($L$)**: After popping $mid$, the surviving stack top $stack[-1]$ is guaranteed to be the **nearest smaller/greater barrier to the left** of $mid$!
   - *Why are there no smaller bars in between?* Because anything between $stack[-1]$ and $mid$ was already evicted by $mid$ when $mid$ originally arrived!
3. **The Maximal Span is Instantly Known**: The maximal open interval where $arr[mid]$ can expand is strictly $(Left, Right)$, with physical width:
   $$W = \text{Right} - \text{Left} - 1$$
   A single eviction locks both bounding walls simultaneously.

---

### 2.3 · Universal Monotonic Stack Blueprint: The Three-Question Four-Slot Model

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

### 2.4 · Comparator and Monotonicity Reference Table

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

### 2.5 · Universal Slot-Filling Matrix

Every problem maps directly into the 4 slots:

| Classic Problem | Mode | Slot 1 (Sentinel) | Slot 2 (Eviction Predicate) | Slot 3 (Settlement Calculation) | Slot 4 (Push) |
|---|---|---|---|---|---|
| **LC 739. Daily Temperatures** | Next Greater | None | `top < current` | Slot 3A: `ans[mid] = i - mid` | `stack.append(i)` |
| **LC 496. Next Greater Element I** | Next Greater | None | `top < current` | Slot 3A: `ans[mid] = current` | `stack.append(i)` |
| **LC 503. Next Greater Element II** | Circular Next Greater | Virtual $2n$ loop | `top < nums[i % n]` | Slot 3A: `ans[mid] = nums[i % n]` (when `mid < n`) | `if i < n: stack.append(i)` |
| **LC 84. Largest Rectangle** | Dual Smaller | Append `0` | `top > current` | Slot 3A: `w = i - stack[-1] - 1`<br>`ans = max(ans, heights[mid] * w)` | `stack.append(i)` |
| **LC 42. Trapping Rain Water** | Dual Greater | None | `top < current` | Slot 3A: Bounded water trough<br>`h = min(top, current) - mid_h` | `stack.append(i)` |
| **LC 907. Subarray Minimums** | Dual Smaller (Tie-break) | Append `0` | Left `<` strict, Right `<=` non-strict | Slot 3A: Product rule for subarrays<br>`count = (mid - left) * (right - mid)` | `stack.append(i)` |
| **LC 1063. Valid Subarrays** | Next Strictly Smaller | Append `-inf` (or loop to $n$) | `top > current` | Slot 3A: `ans += i - mid` (Span contribution) | `stack.append(i)` |

---

### 2.6 · Sentinel Mechanics Demystified: When and Why Are Sentinels Necessary?

A sentinel element in a monotonic stack is not merely syntactic sugar; it is a **mathematical and operational barrier designed to eliminate branch checks, prevent illegal index underflows, and force unresolved computations to terminate deterministically**.

#### 1. Dual Failure Modes Solved by Sentinels

Without sentinels, monotonic stack implementations face two chronic boundary pathologies:

- **Pathology 1: Head Stack Underflow & Left Boundary Fallback**
  - **Manifestation**: After executing `mid = stack.pop()`, the algorithm needs to look at its left boundary `left = stack[-1]`. If the stack is now completely empty (meaning no element to the left of $mid$ is smaller/larger, making $mid$ the global extreme of the prefix), accessing `stack[-1]` raises an `IndexError`.
  - **Non-Sentinel Workaround**: Developers are forced to clutter code with ternaries: `left = stack[-1] if stack else -1`.
- **Pathology 2: Tail Stranding & Flush Omission**
  - **Manifestation**: When array traversal finishes, elements often remain stranded inside the stack (e.g., if the array or its suffix is strictly monotonic). When **business logic is executed upon eviction in Slot 3A** (such as LC 84 Largest Rectangle, LC 85 Maximal Rectangle, or LC 907 Subarray Minimums), no further array elements arrive to disrupt monotonicity. The stranded elements **never get popped, causing severe computation omissions**!
  - **Non-Sentinel Workaround**: A duplicate, error-prone `while stack:` loop must be appended after the main loop to flush out remaining items.

```text
[Without Sentinel vs. With Sentinels Architecture]

Without Sentinel:
Iterate nums ─────► Stranded unevicted elements ─────► Must duplicate while stack flush loop
                       │
                       └─► Every left lookup needs: stack[-1] if stack else -1

Dual Sentinels:
[-∞ / 0] + nums + [-∞ / 0] ───────────────► All elements guaranteed flushed by tail sentinel
   │                       │
   │                       └─► Tail Sentinel: 100% evictions triggered in Slot 3A (Zero-leak)
   └─────────────────────────► Head Sentinel: Stack never empty, stack[-1] always valid (No underflow)
```

#### 2. Sentinel Taxonomy & Decision Matrix

| Sentinel Strategy | Canonical Form | Primary Purpose | Problem Characteristics | Canonical Problems |
|---|---|---|---|---|
| **Tail Sentinel** | `arr = nums + [0]`<br>or `range(len(nums) + 1)` | Injects a **global disrupter** to force-flush all stranded stack elements in the final step. | **Settlement occurs upon eviction (Slot 3A)**, requiring exact full-span aggregation. | **LC 84** (Largest Rectangle)<br>**LC 907** (Subarray Minimums)<br>**LC 85** (Maximal Rectangle) |
| **Head Sentinel** | Preset `stack = [-1]`<br>or `arr = [0] + nums` | Serves as a **natural open-interval left pivot**, keeping stack non-empty at all times. | Left boundary lookup is required without risk of stack underflow. | **LC 32** (Longest Valid Parentheses)<br>**LC 84** (Monotonic increasing left pivot) |
| **Dual Sentinels** | `arr = [0] + heights + [0]`<br>or `[-inf] + nums + [-inf]` | **Simultaneously eliminates both head underflows and tail flush loops**. Maximum symmetry and cleanest logic. | Bilateral span queries, geometric rectangle calculations. | **LC 84** (Largest Rectangle optimal)<br>**LC 85** (Maximal Rectangle) |
| **No Sentinel Needed** | Keep raw `nums`<br>Pre-fill result array | **Natural default semantics**; stranded elements naturally represent "no answer exists". | 1. Pre-filled array (e.g., `-1` or `0` in LC 739);<br>2. Settlement occurs **before push during left queries (Slot 3B)**. | **LC 739** (Daily Temperatures)<br>**LC 496** (Next Greater Element I)<br>**LC 503** (Modulo virtual doubling) |

#### 3. Extreme Value Selection Principle

The sentinel value must strictly adhere to the **domain bound theorem**:
- **Monotonically Increasing Stack (Finding Smaller Elements, e.g., LC 84, LC 907)**:
  - Tail sentinel must be **strictly smaller** than any valid input element.
  - If inputs are non-negative ($heights[i] \ge 0$), use `0`.
  - If inputs contain arbitrary or negative integers, use negative infinity: `float('-inf')`.
- **Monotonically Decreasing Stack (Finding Greater Elements, e.g., Trapping Rain Water)**:
  - Tail sentinel must be **strictly larger** than any valid input element, typically `float('inf')`.

#### 4. Implementation Evolution (LC 84 Case Study)

```python
# Approach A: No Sentinels (Verbose, requires underflow ternaries and duplicate cleanup loop)
class SolutionNoSentinel:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack, max_area = [], 0
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                mid = stack.pop()
                left = stack[-1] if stack else -1
                max_area = max(max_area, heights[mid] * (i - left - 1))
            stack.append(i)
        # Duplicate post-loop flush required!
        while stack:
            mid = stack.pop()
            left = stack[-1] if stack else -1
            max_area = max(max_area, heights[mid] * (len(heights) - left - 1))
        return max_area


# Approach B: Dual Sentinels (Zero branches, perfectly symmetric, bulletproof)
class SolutionDualSentinels:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # Pad both ends with height 0 sentinels
        arr = [0] + heights + [0]
        stack, max_area = [], 0

        for i, h in enumerate(arr):
            # Left sentinel prevents empty stack; right sentinel guarantees total flush
            while stack and arr[stack[-1]] > h:
                mid = stack.pop()
                left = stack[-1]  # Invariant: stack is never empty here!
                width = i - left - 1
                max_area = max(max_area, arr[mid] * width)
            stack.append(i)

        return max_area
```

---

### 2.7 · Canonical Problems & Blueprint Instantiation

#### Practice 1: Daily Temperatures
Given daily temperatures, return the number of days to wait until a warmer temperature.
- **Mapping**: Canonical **Next Greater** problem where answer format is the waiting span $i - mid$.

```daily-temperatures-demo
```

##### Dynamic Execution Trace & Multi-Level Eviction Flow

Tracing `temperatures = [73, 74, 75, 71, 69, 72, 76, 73]`:

```text
[Monotonic Decreasing Stack Execution Flow]

Day 0 (73°): Empty stack ──► Push 0(73°)
             Stack: [ 0(73°) ]

Day 1 (74°): 74° > 73° ──► Pop 0(73°), ans[0] = 1 - 0 = 1 ──► Push 1(74°)
             Stack: [ 1(74°) ]

Day 2 (75°): 75° > 74° ──► Pop 1(74°), ans[1] = 2 - 1 = 1 ──► Push 2(75°)
             Stack: [ 2(75°) ]

Day 3 (71°): 71° < 75° ──► Push 3(71°)
             Stack: [ 2(75°), 3(71°) ]

Day 4 (69°): 69° < 71° ──► Push 4(69°) (cooling streak builds decreasing bounty chain)
             Stack: [ 2(75°), 3(71°), 4(69°) ] ◄── Top 4(69°) is the weakest barrier

Day 5 (72°): [Multi-Level Eviction Event 1]
             ├─ 72° > 69° ──► Pop 4(69°), ans[4] = 5 - 4 = 1
             ├─ 72° > 71° ──► Pop 3(71°), ans[3] = 5 - 3 = 2
             └─ 72° < 75° ──► Encounters barrier 75°, stop! Push 5(72°)
             Stack: [ 2(75°), 5(72°) ]

Day 6 (76°): [Multi-Level Eviction Event 2 · Global Breakthrough]
             ├─ 76° > 72° ──► Pop 5(72°), ans[5] = 6 - 5 = 1
             └─ 76° > 75° ──► Pop 2(75°), ans[2] = 6 - 2 = 4 (resolves 4-day wait!)
             Push 6(76°)
             Stack: [ 6(76°) ]

Day 7 (73°): 73° < 76° ──► Push 7(73°)
             Stack: [ 6(76°), 7(73°) ]

End of Scan:
             Residual days [6, 7] never see a warmer future; default 0 is retained.
Output:      ans = [1, 1, 4, 2, 1, 1, 0, 0]
```

```python
from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []  # Stores indices; temperatures strictly decrease from bottom to top

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

#### Practice 5: Number of Valid Subarrays (LC 1063, Monotonic Span Contribution)

Given an integer array `nums`, return the number of non-empty continuous subarrays where the **leftmost element is less than or equal to every other element in that subarray** (i.e., the leftmost element equals the subarray's minimum).

##### 1. Mathematical Formulation & Boundary Semantics
- **Condition Definition**:
  A subarray $nums[i..j]$ is valid iff $\min(nums[i..j]) = nums[i]$, which requires:
  $$nums[k] \ge nums[i], \quad \forall k \in [i, j]$$
- **Follow-up: Do equal elements truncate the window?**
  - **Conclusion: Absolutely not!**
  - **Proof**: If $nums[k] == nums[i]$, the minimum of $nums[i..k]$ remains $nums[i]$ (equality is explicitly permitted). Thus, equal elements never break subarray validity. The window expands unimpeded until encountering a strictly smaller value ($nums[k] < nums[i]$).

##### 2. Left Endpoint, Right Endpoint, and Contribution Formula
- **Left Endpoint ($i$)**: Fix each index $i$ as the designated starting position of the subarray.
- **Right Boundary ($R_i$)**: Search to the right for the **first strictly smaller element**:
  $$R_i = \min \{ k \mid k > i \text{ and } nums[k] < nums[i] \}$$
  If no element to the right is strictly smaller than $nums[i]$, the right boundary extends past the array boundary: $R_i = n$.
- **Valid Right Endpoints**: Since all elements in $[i, R_i - 1]$ satisfy $nums[k] \ge nums[i]$, every index $j \in [i, R_i - 1]$ forms a distinct valid subarray $nums[i..j]$.
- **Single Element Contribution Formula**:
  The number of valid subarrays originating at $i$ strictly equals the number of choices for $j$:
  $$\text{Count}(i) = (R_i - 1) - i + 1 = R_i - i$$
- **Global Answer**:
  $$\text{Total} = \sum_{i=0}^{n-1} (R_i - i)$$

##### 3. Monotonic Stack Blueprint Instantiation

Maintain a **monotonically non-decreasing stack** (from bottom to top $nums[stack[k]] \le nums[stack[k+1]]$, permitting equal values):
- **Slot 1 (Sentinel)**: Virtual $-\infty$ at index $n$ to guarantee complete flushing;
- **Slot 2 (Eviction Predicate)**: `while stack and nums[stack[-1]] > curr_val:` (strictly smaller breaks monotonicity);
- **Slot 3A (Settlement)**: For popped element $mid$, current index $i$ is its first strictly smaller right boundary $R_{mid} = i$. Add $i - mid$ to answer;
- **Slot 4 (Push)**: `stack.append(i)`.

```python
from typing import List


class Solution:
    def validSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        stack = []  # Invariant: nums[stack[k]] <= nums[stack[k+1]]

        # Slot 1: Iterate up to n with virtual -inf sentinel
        for i in range(n + 1):
            curr_val = float("-inf") if i == n else nums[i]

            # Slot 2: Evict on strictly smaller element
            while stack and nums[stack[-1]] > curr_val:
                mid = stack.pop()
                # Slot 3A: Settle span contribution for mid
                ans += i - mid

            # Slot 4: Push current index
            stack.append(i)

        return ans
```

##### 4. Mathematical Duality: Active Stack Depth Perspective

Beyond the left-endpoint span perspective ($R_{mid} - mid$ on eviction), there exists a dual viewpoint:
- **Right Endpoint Perspective**:
  After popping all elements strictly greater than $nums[i]$, every remaining index $k$ in the stack satisfies $nums[k] \le nums[i]$, with no intervening elements smaller than $nums[k]$.
- **Duality Invariant**:
  Every surviving index in `stack` can serve as a valid left endpoint for a subarray ending at $i$. Pushing $i$ makes the number of valid subarrays ending at $i$ **identically equal to `len(stack)`**!

```python
class SolutionStackDepth:
    def validSubarrays(self, nums: List[int]) -> int:
        ans = 0
        stack = []

        for num in nums:
            while stack and stack[-1] > num:
                stack.pop()
            stack.append(num)
            # Every element currently in stack forms a valid subarray ending at num
            ans += len(stack)

        return ans
```

Both formulations are mathematically isomorphic:
$$\sum_{i=0}^{n-1} (R_i - i) \equiv \sum_{j=0}^{n-1} \text{StackDepth}(j)$$

---

### 2.8 · Complexity Proof: Why Nested while Loops Run in Strict O(n) Time

A common pitfall is mistaking nested `while` inside `for` as $O(n^2)$.

The formal **Aggregate Analysis** proof is straightforward:
1. In an array of size $n$, each index enters the `for` loop and is pushed via `stack.append()` **at most once**;
2. An element can only be popped via `stack.pop()` if it currently resides in the stack, so each element is popped **at most once**;
3. Once an index is popped, it is permanently discarded and never re-enters the stack;
4. Therefore, across all $n$ outer iterations, the inner `while` condition evaluates to true and executes `pop()` at most $n$ times total.

$$\sum_{i=1}^n (\text{Push Count} + \text{Pop Count}) \le n + n = 2n = O(n)$$

Thus, the monotonic stack algorithm runs in strict $O(n)$ time and $O(n)$ auxiliary space in the worst case.

---

### 2.9 · Whiteboard Interview Checklist

When presenting a monotonic stack solution in technical interviews, structure your explanation across 5 clear milestones:

1. **Classification**: "This problem requires finding nearest extreme-value boundaries for each position in 1D. Brute force is $O(n^2)$; a monotonic stack optimizes this to linear $O(n)$ time."
2. **Data Structure**: "The stack stores indices rather than values because calculating spans $i - j$ and rectangle widths requires physical distances."
3. **Monotonicity**: "We maintain a monotonic decreasing stack because we are seeking the next greater element; any value violating this triggers immediate eviction."
4. **Attribution**: "Answers are settled upon eviction (Slot 3A) because the current scanned element serves as the active resolver for waiting elements."
5. **Sentinel**: "For histogram area or multi-interval aggregation, append a `0` sentinel to flush residual frames automatically without duplicate cleanup code."
