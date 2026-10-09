# Stack · MinStack and Monotonic Stack

The stack API is simple. The two patterns that matter in interviews are:

```text
MinStack: preserve history snapshots during push so that extreme-value queries never rescan
Monotonic Stack: maintain an active candidate set of unresolved elements, waiting for right-side elements to trigger eviction and establish boundaries
```

MinStack is a standalone state-snapshot design; Monotonic Stack is a high-frequency interview pattern for resolving "1D nearest extreme boundaries". Mastering it requires only **one core intuition: "Whoever breaks the monotonicity is the right boundary responsible for evicting and settling the stack top"**. Regardless of the problem, the core skeleton is only 5 lines of code.

## Learning Order

Selected from high-frequency core interview problems to build progressive mastery from state snapshots to full dual-boundary monotonic stacks:

| Order | Original Problem | Core Pattern | Key Engineering Problem Solved |
|---:|---|---|---|
| 1 | [155. Min Stack](https://neetcode.io/problems/minimum-stack/question?list=neetcode150) | State-Snapshot Pattern | Bind `min_so_far` to each stack frame for $O(1)$ query and rollback |
| 2 | [739. Daily Temperatures](https://neetcode.io/problems/daily-temperatures/question?list=neetcode150) | Monotonic Stack · Right Boundary (Next Greater) | Settle waiting days via index delta upon eviction |
| 3 | [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/) | Monotonic Stack · Circular Array | Traverse $2n$ with modulo, seamlessly reusing monotonic stack |
| 4 | [84. Largest Rectangle in Histogram](https://neetcode.io/problems/largest-rectangle-in-histogram/question?list=neetcode150) | Monotonic Stack · Dual Boundaries | Pad `0` sentinels at both ends, resolving left and right boundaries simultaneously |
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

### 2.3 · The Minimal Mental Model and 5-Line Universal Skeleton

There is no need to memorize complex "slots" or multi-question frameworks. In an interview, internalize this **single core rule**:

> **"Always store indices in the stack. Whoever breaks monotonicity is the right boundary, responsible for popping the stack top and settling its answer."**

#### 1. Core Rule of Thumb (Deciding Increasing vs. Decreasing in 10s)
- **Find Next Greater** $	o$ Maintain a **monotonically decreasing stack** (top is smallest):
  - When a smaller element arrives: push to stack and wait;
  - When a **larger element** arrives: it breaks the decreasing order — the stack top has finally met someone larger! **Pop and record the answer immediately**.
- **Find Next Smaller** $	o$ Maintain a **monotonically increasing stack** (top is largest):
  - When a larger element arrives: push to stack and wait;
  - When a **smaller element** arrives: it breaks the increasing order — the stack top has finally met someone smaller! **Pop and record the answer immediately**.

#### 2. The 5-Line Universal Skeleton

```python
stack = []  # Store indices
ans = [0] * n  # Default values (-1 or 0 for elements with no greater/smaller element)

for i, x in enumerate(nums):
    # Find next greater: x is greater than stack top, satisfying and evicting it!
    # (To find next smaller, simply flip < to >)
    while stack and nums[stack[-1]] < x:
        top = stack.pop()
        ans[top] = i - top  # or ans[top] = x, according to problem needs
    stack.append(i)
```

Interactive demonstration comparing Next Greater and Next Smaller on the same array:

```monotonic-stack-demo
```

---

### 2.4 · The Only 3 Real-World Variations

All variations are tiny tweaks on top of the 5-line skeleton:

1. **Variation 1: Standard Single-Sided Boundary (e.g., LC 739 Daily Temperatures, LC 496)**
   - Initialize answer array with default values (`-1` or `0`);
   - Any index remaining in the stack has no greater/smaller neighbor to its right, naturally retaining the default. **No post-loop cleanup required**.

2. **Variation 2: Circular Array (e.g., LC 503)**
   - Loop twice: `for i in range(2 * n): x = nums[i % n]`;
   - Pop and settle as usual, but only push during the first pass: `if i < n: stack.append(i)`.

3. **Variation 3: Dual Boundaries / Histogram Rectangles (e.g., LC 84)**
   - **Pad `0` at both ends (the cleanest sentinels)**: `heights = [0] + heights + [0]`;
   - **Left `0`**: ensures stack is never empty (accessing `stack[-1]` never errors);
   - **Right `0`**: guarantees all remaining bars in the stack are flushed and calculated at the end (zero omission);
   - Upon popping, both boundaries are captured at once:
     - Height: `h = heights[stack.pop()]`
     - Left boundary: new stack top `left = stack[-1]`
     - Right boundary: current `right = i`
     - Width: `w = right - left - 1`, Area: `h * w`.

---

### 2.5 · Core Problem Lookup Matrix

| Problem | Target | Stack Order | Eviction Condition (`while`) | Settled Value | Interview Technique |
|---|---|---|---|---|---|
| **LC 739. Daily Temperatures** | Next Greater | Decreasing (Large $	o$ Small) | `nums[stack[-1]] < x` | `ans[top] = i - top` (waiting days) | Default array filled with 0 |
| **LC 503. Next Greater II** | Circular Next Greater | Decreasing (Large $	o$ Small) | `nums[stack[-1]] < x` | `ans[top] = x` (value) | Loop $2n$, index `i % n` |
| **LC 84. Largest Rectangle** | Dual Smaller (width) | Increasing (Small $	o$ Large) | `arr[stack[-1]] > x` | `h * (i - stack[-1] - 1)` | Pad `0` at both ends |
| **LC 42. Trapping Rain Water** | Dual Greater (trough) | Decreasing (Large $	o$ Small) | `height[stack[-1]] < x` | `(min(L, R) - bottom) * (R - L - 1)` | Popped item is floor; stack top is left wall |
| **LC 1063. Valid Subarrays** | Next Strictly Smaller | Increasing (Small $	o$ Large) | `nums[stack[-1]] > x` | Contribution: `ans += i - top` | Virtual `-inf` flush at end |

---

### 2.6 · The Minimal Sentinel Rule: When to Use Sentinels?

Remember this single practical rule:
- **Single-sided queries (Daily Temperatures, Next Greater)**: **No sentinels needed**. Pre-fill default values `-1` or `0`; remaining items naturally represent unresolved elements.
- **Dual-sided geometric areas (LC 84 Largest Rectangle)**: **Pad 0 at both ends** (`[0] + heights + [0]`). Left 0 avoids underflow; right 0 flushes remaining bars cleanly without duplicate post-loops.

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
            # Eviction check: Current temperature higher than stack top
            while stack and temperatures[stack[-1]] < temp:
                mid = stack.pop()
                # Eviction settlement: Settle distance delta upon eviction
                ans[mid] = i - mid
            # Push to stack: Enqueue current day
            stack.append(i)

        return ans
```

##### Right View Implementation (Reverse Traversal with Instant Settlement · Immediate settlement)

If we traverse in **reverse order ($i = n-1 \to 0$)**, the current element $i$ acts directly as the **protagonist**. The stack maintains all viable candidates to its right. Any right-hand candidate with $temperatures[\text{top}] \le temp$ is permanently shadowed by Day $i$ (since Day $i$ is both closer to any left-side day and warmer or equal), and is immediately evicted. Once evictions finish, the stack top is guaranteed to be the first strictly warmer day, allowing **instant settlement (Immediate settlement)**:

```python
class SolutionRightView:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []  # Temperatures strictly decrease from bottom to top

        # Initialization: Reverse traversal
        for i in range(n - 1, -1, -1):
            temp = temperatures[i]
            # Eviction check: Evict shadowed right-hand candidates
            while stack and temperatures[stack[-1]] <= temp:
                stack.pop()
            # Immediate settlement: Instant settlement for Day i
            ans[i] = stack[-1] - i if stack else 0
            # Push to stack: Enqueue Day i as candidate for earlier days
            stack.append(i)

        return ans
```

##### Left View vs. Right View: Structural Comparison & Selection Criteria

| Dimension | Left View (Forward + Pop-Driven · Eviction settlement) | Right View (Reverse + Push-Driven · Immediate settlement) |
| :--- | :--- | :--- |
| **Protagonist** | Evicted element `mid` is the protagonist; scanning element $i$ is the right terminator | Scanning element $i$ is the protagonist; stack stores the candidate skeleton |
| **Settlement Timing** | **Asynchronous / Delayed**: Element enqueues and waits until a warmer future day pops it | **Instant / Online**: Answer for $i$ is finalized immediately on the spot |
| **Residual Stack Items** | Unresolved days; retain default 0 or require trailing sentinels to flush | Global candidates; all answers already finalized, completely eliminating sentinels |
| **When Right View is Simpler** | Bilateral boundary problems (e.g., Largest Rectangle, Rain Water) | **Online streaming** (LC 901), **Subarray counting** (LC 1063), **DP state transitions** (LC 907) |

##### Aggregate Analysis of While Loop Comparisons: Formal Upper Bound Proof

Let $n$ denote array length. The outer `for` loop executes exactly $n$ iterations. For each execution of `while stack and condition:`:

1. **Short-Circuit on Empty Stack**:
   When `stack` is empty, evaluation terminates immediately without performing any numeric comparison.
2. **Condition Evaluates to True**:
   Every True evaluation triggers exactly one `stack.pop()`. Since each index is pushed at most once, at most $n$ successful pops can occur:
   $$\text{Comparisons}_{\text{True}} \le n$$
3. **Condition Evaluates to False**:
   Every False evaluation terminates the `while` loop immediately. Since the loop can terminate at most once per outer iteration:
   $$\text{Comparisons}_{\text{False}} \le n$$
4. **Tight Amortized Upper Bound**:
   $$\text{Total Comparisons} = \text{Comparisons}_{\text{True}} + \text{Comparisons}_{\text{False}} \le n + n = 2n$$
   Hence, monotonic stack value comparisons never exceed $2n$, giving strict amortized $\Theta(n)$ execution time.

###### Concrete Execution Trace for `temperatures = [73, 74, 75, 71, 69, 72, 76, 73]` ($n = 8$)

On this canonical input, both views execute **exactly 10 numeric comparisons** (well below the $2n = 16$ ceiling):

- **Left View (Forward) Trace**:
  - Day 0 (73°): Stack empty $\to$ **0 comparisons**. Push 0.
  - Day 1 (74°): $73° < 74°$ True (pop 0, settle); stack empty $\to$ **1 True**. Push 1.
  - Day 2 (75°): $74° < 75°$ True (pop 1, settle); stack empty $\to$ **1 True**. Push 2.
  - Day 3 (71°): $75° < 71°$ False (terminate) $\to$ **1 False**. Push 3.
  - Day 4 (69°): $71° < 69°$ False (terminate) $\to$ **1 False**. Push 4.
  - Day 5 (72°): $69° < 72°$ True (pop 4); $71° < 72°$ True (pop 3); $75° < 72°$ False (terminate) $\to$ **2 True + 1 False = 3 comparisons**. Push 5.
  - Day 6 (76°): $72° < 76°$ True (pop 5); $75° < 76°$ True (pop 2); stack empty $\to$ **2 True**. Push 6.
  - Day 7 (73°): $76° < 73°$ False (terminate) $\to$ **1 False**. Push 7.
  - **Total**: $6 \text{ True} + 4 \text{ False} = 10 \text{ numeric comparisons}$.

- **Right View (Reverse) Trace**:
  - Day 7 (73°): Stack empty $\to$ **0 comparisons**. Settle $ans[7]=0$. Push 7.
  - Day 6 (76°): $73° \le 76°$ True (evict 7); stack empty $\to$ **1 True**. Settle $ans[6]=0$. Push 6.
  - Day 5 (72°): $76° \le 72°$ False (terminate) $\to$ **1 False**. Top is 6, settle $ans[5] = 6 - 5 = 1$. Push 5.
  - Day 4 (69°): $72° \le 69°$ False (terminate) $\to$ **1 False**. Top is 5, settle $ans[4] = 5 - 4 = 1$. Push 4.
  - Day 3 (71°): $69° \le 71°$ True (evict 4); $72° \le 71°$ False (terminate) $\to$ **1 True + 1 False = 2 comparisons**. Top is 5, settle $ans[3] = 5 - 3 = 2$. Push 3.
  - Day 2 (75°): $71° \le 75°$ True (evict 3); $72° \le 75°$ True (evict 5); $76° \le 75°$ False (terminate) $\to$ **2 True + 1 False = 3 comparisons**. Top is 6, settle $ans[2] = 6 - 2 = 4$. Push 2.
  - Day 1 (74°): $75° \le 74°$ False (terminate) $\to$ **1 False**. Top is 2, settle $ans[1] = 2 - 1 = 1$. Push 1.
  - Day 0 (73°): $74° \le 73°$ False (terminate) $\to$ **1 False**. Top is 1, settle $ans[0] = 1 - 0 = 1$. Push 0.
  - **Total**: $4 \text{ True} + 6 \text{ False} = 10 \text{ numeric comparisons}$.

#### Practice 2: Next Greater Element II (Circular Array via Modulo)
Find the next greater element in a circular array.
- **Mapping**: Extend iteration to $2n$ in **Initialization** using `i % n`, pushing to stack only during the first cycle ($i < n$).

```python
from typing import List


class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [-1] * n
        stack = []

        # Initialization: Virtual doubling via 2*n iteration
        for i in range(2 * n):
            val = nums[i % n]
            # Eviction check: Eviction comparison
            while stack and nums[stack[-1]] < val:
                mid = stack.pop()
                # Eviction settlement: Record next greater value
                ans[mid] = val
            # Push to stack: Only push during first cycle
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

        # Initialization: Inject tail sentinel of height 0
        for right in range(len(heights) + 1):
            curr_h = 0 if right == len(heights) else heights[right]

            # Eviction check: Shorter bar breaks increasing monotonicity
            while stack and heights[stack[-1]] > curr_h:
                mid = stack.pop()
                mid_h = heights[mid]
                # Eviction settlement: Left and right boundaries locked simultaneously
                left = stack[-1] if stack else -1
                width = right - left - 1
                max_area = max(max_area, mid_h * width)

            # Push to stack: Enqueue current index
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
            # Eviction check: Taller bar encountered, forming bounding trough
            while stack and height[stack[-1]] < curr_h:
                mid = stack.pop()
                if not stack:
                    break  # No left boundary to trap water

                left = stack[-1]
                # Eviction settlement: Horizontal trough slice accumulation
                h = min(height[left], curr_h) - height[mid]
                w = right - left - 1
                water += h * w

            # Push to stack: Enqueue current bar
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
- **Initialization (Sentinel)**: Virtual $-\infty$ at index $n$ to guarantee complete flushing;
- **Eviction check (Eviction Predicate)**: `while stack and nums[stack[-1]] > curr_val:` (strictly smaller breaks monotonicity);
- **Eviction settlement (Settlement)**: For popped element $mid$, current index $i$ is its first strictly smaller right boundary $R_{mid} = i$. Add $i - mid$ to answer;
- **Push to stack (Push)**: `stack.append(i)`.

```python
from typing import List


class Solution:
    def validSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        stack = []  # Invariant: nums[stack[k]] <= nums[stack[k+1]]

        # Initialization: Iterate up to n with virtual -inf sentinel
        for i in range(n + 1):
            curr_val = float("-inf") if i == n else nums[i]

            # Eviction check: Evict on strictly smaller element
            while stack and nums[stack[-1]] > curr_val:
                mid = stack.pop()
                # Eviction settlement: Settle span contribution for mid
                ans += i - mid

            # Push to stack: Push current index
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

### 2.9 · Structured Interview Delivery Checklist

Follow this clean 3-step delivery during a coding interview:

1. **State the Pattern**: "This problem searches for the nearest extreme boundary in a 1D sequence. Brute force is $O(n^2)$; a monotonic stack optimizes this to amortized $O(n)$."
2. **Clarify Stack Invariant**: "Store indices in the stack to compute distances and widths. To find the next greater element, maintain a decreasing stack and evict as soon as a strictly greater element arrives."
3. **Handle Boundaries**: "For one-sided queries, initialize defaults with -1 or 0. For dual-boundary geometric areas (like histograms), pad 0 sentinels at both ends to eliminate stack underflow and flush omissions."


## Module 3: Stack & Monotonic Stack High-Frequency Extensions

### 1. LC 224 / 227 Universal Basic Calculator via Operator Precedence

#### Core Mental Model
Maintain `nums` and `ops` stacks with a precedence map (`+,-`=1, `*,/`=2). When encountering an operator, pop and evaluate any stack operators with $\ge$ precedence. Handle parentheses by pushing `(` and popping until matching `(`. Insert `0` before unary operators.

---

### 2. LC 316 / 1081 Remove Duplicate Letters via Monotonic Stack

#### Core Mental Model
Maintain a monotonic increasing stack:
- If character is already in `in_stack`, skip.
- While `stack[-1] > ch` and `stack[-1]` occurs again later (`last_pos[stack[-1]] > i`), pop it to achieve a smaller lexicographical order.
- Push current character and mark in `in_stack`. Time: $O(n)$, Space: $O(|\Sigma|)$.

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        last_pos = {ch: i for i, ch in enumerate(s)}
        stack, in_stack = [], set()
        for i, ch in enumerate(s):
            if ch in in_stack: continue
            while stack and stack[-1] > ch and last_pos[stack[-1]] > i:
                in_stack.remove(stack.pop())
            stack.append(ch)
            in_stack.add(ch)
        return "".join(stack)
```
