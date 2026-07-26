# Stack · MinStack and Monotonic Stack

The stack API is simple. The two patterns that matter in interviews are:

```text
MinStack: preserve history during push so a query never has to rescan
Monotonic stack: let indices wait until the first qualifying value on the right resolves them
```

MinStack is a fairly isolated design problem; memorizing one reliable implementation is enough. Monotonic stack is a family of problems. Instead of memorizing each solution, keep one template and replace only its comparator and answer format.

## Learning Order

The problems come from [NeetCode 150](https://neetcode.io/practice/practice/neetcode150), but this chapter keeps only the three that directly support its two modules.

| Order | Original Problem | What to Learn |
|---:|---|---|
| 1 | [155. Min Stack](https://neetcode.io/problems/minimum-stack/question?list=neetcode150) | Store `min_so_far` in every stack frame |
| 2 | [739. Daily Temperatures](https://neetcode.io/problems/daily-temperatures/question?list=neetcode150) | First greater value on the right |
| 3 | [84. Largest Rectangle in Histogram](https://neetcode.io/problems/largest-rectangle-in-histogram/question?list=neetcode150) | Use a monotonic stack to determine boundaries |

## Module 1: MinStack

### Save the Minimum at Every Level

If we keep only one global variable named `minimum`, `push` and `getMin` are easy. Once the current minimum is popped, however, we no longer know the previous minimum. Rescanning the entire stack would cost $O(n)$.

A more reliable representation stores a pair for every element:

```text
(current value, min_so_far after pushing this value)
```

The top stack frame then carries both answers:

```text
top()    = stack[-1][0]
getMin() = stack[-1][1]
```

Every `push` takes a snapshot of the current state. A `pop` removes that snapshot, automatically revealing the previous minimum stored one level below. No special rollback logic is needed.

### Why Duplicate Minimum Values Work

Push `2, 1, 1` in order:

```text
[(2, 2), (1, 1), (1, 1)]
```

After one `1` is popped, the frame below still stores `(1, 1)`, so the minimum remains `1`. Recording only values that establish a new minimum makes duplicate handling unnecessarily delicate.

### Quick Coding: Implement MinStack

Implement `push`, `pop`, `top`, and `getMin`, all in $O(1)$ time.

<details>
<summary>Reference answer</summary>

```python
class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, value: int) -> None:
        minimum = value if not self.stack else min(
            value,
            self.stack[-1][1],
        )
        self.stack.append((value, minimum))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

Every operation is $O(1)$. Storing one additional minimum per element uses $O(n)$ space.

</details>

For this problem, it is worth memorizing `(value, min_so_far)` directly. A two-stack implementation is also correct, but it adds a synchronization rule without making the interview solution clearer.

## Module 2: Monotonic Stack

### What Kind of Problems Does It Solve?

A monotonic stack most often handles this question:

> For every position in an array, find the first (nearest) greater or smaller element to its left or right.

The wording may change while the underlying query stays the same:

| Problem Wording | What It Is Really Asking For |
|---|---|
| How many days until it gets warmer? | First strictly greater value on the right; return the index difference |
| What is the next greater element? | First strictly greater value on the right; return its value |
| How far can a histogram bar extend? | First smaller value on both sides |
| How many previous prices does today's price dominate? | The nearest greater value on the left forms the boundary |

There are two strong signals:

1. The input is usually a one-dimensional sequence, and the problem needs a boundary for **every position**.
2. The boundary is not the global maximum or minimum. It is the **first position in one direction that satisfies a comparison**.

A brute-force solution scans left or right from every position and can take $O(n^2)$. A monotonic stack keeps only positions whose boundaries are still unknown. Since each index is pushed and popped at most once, all of these queries can be answered in $O(n)$.

For a maximum over every fixed-size window, a monotonic deque is usually the right tool. For one maximum over the entire array, a linear scan is enough. A monotonic stack is useful when many positions each need their own nearest boundary.

### What Does the Stack Actually Store?

For "the first strictly greater value to the right," the stack does not hold answers. It holds:

> Indices that have already been scanned but whose first greater value on the right has not appeared.

Store **indices**, rather than values alone, because an index gives all three pieces of information:

```text
nums[j]: the value used in comparisons
j: the position used to compute a distance such as i - j
answer[j]: the slot where the result belongs
```

Storing only values loses their positions and cannot distinguish duplicate values.

For example, scan `nums = [5, 2, 4, 6]`:

```text
Read 5: stack = [0]       index 0 is waiting for something greater
Read 2: stack = [0, 1]    2 cannot resolve 5, so index 1 also waits
Read 4: pop 1             4 is the first greater value to the right of index 1
        stack = [0, 2]    indices 0 and 2 are still unresolved
Read 6: pop 2, then 0     6 resolves both remaining indices
```

The stack is therefore not "everything seen so far." It contains only unresolved candidate indices that survive earlier comparisons. At index `i`, the current value `nums[i]` repeatedly resolves the top:

```python
while stack and nums[i] > nums[stack[-1]]:
    j = stack.pop()
    answer[j] = i
```

After the popping stops, push `i`: the current position now begins waiting for its own answer on the right. The values represented by the stack are non-increasing from bottom to top, but that monotonic order is a consequence of the waiting-and-removal process, not the goal by itself.

Why is the current `i` the **first** answer for a popped index `j`? Since `j` was pushed, the scan has visited `j + 1, j + 2, ...` in order. If an earlier value had satisfied the condition, it would already have popped `j`.

### One Unified Template

Start with "the first qualifying position on the right":

```python
def first_match_on_right(nums):
    answer = [-1] * len(nums)
    stack = []  # indices whose answer has not appeared

    for i, value in enumerate(nums):
        while stack and value_satisfies(nums[stack[-1]], value):
            j = stack.pop()
            answer[j] = i

        stack.append(i)

    return answer
```

Only two slots change:

```text
value_satisfies: when the current value answers the stack top
answer[j]: whether the problem asks for an index, value, or distance
```

### Comparator Table

Let `old = nums[stack[-1]]` and `current = nums[i]`:

| Desired Answer | Pop Condition in `while` | Values Left in Stack: Bottom → Top |
|---|---|---|
| First strictly greater on the right | `old < current` | Non-increasing |
| First greater-or-equal on the right | `old <= current` | Strictly decreasing |
| First strictly smaller on the right | `old > current` | Non-decreasing |
| First smaller-or-equal on the right | `old >= current` | Strictly increasing |

The safest way to avoid reversing the comparator is:

> Do not begin by memorizing "increasing stack" or "decreasing stack." Ask whether the current value satisfies what the top is waiting for. If it does, pop.

The walkthrough below runs "next greater" and "next smaller" on the same array. Switching the target changes one comparator in the template.

```monotonic-stack-demo
```

### Daily Temperatures: Change an Index into a Distance

Given daily temperatures, return how many days each day must wait for a warmer temperature; return `0` if no warmer day follows.

This is exactly "the first strictly greater value on the right," except the required answer is a distance:

```text
answer[j] = i - j
```

### Quick Coding: Daily Temperatures

```python
def dailyTemperatures(temperatures):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def dailyTemperatures(
        self,
        temperatures: List[int],
    ) -> List[int]:
        answer = [0] * len(temperatures)
        stack = []

        for i, temperature in enumerate(temperatures):
            while (
                stack
                and temperatures[stack[-1]] < temperature
            ):
                j = stack.pop()
                answer[j] = i - j

            stack.append(i)

        return answer
```

Temperatures represented by the stack indices are non-increasing from bottom to top. An equal temperature does not answer "strictly warmer," so the comparator must be `<`, not `<=`.

</details>

### Answers on the Right and Left Have Different Recording Times

The two common monotonic-stack questions differ mainly in who owns the answer.

### First Answer on the Right: Resolve an Old, Popped Index

```python
for i, value in enumerate(nums):
    while stack and current_answers_top(...):
        j = stack.pop()
        answer[j] = i
    stack.append(i)
```

One current value may resolve several old indices, so answers are written inside the `while` loop.

### Nearest Answer on the Left: Remove Invalid Candidates, Then Read the Top

For the nearest strictly smaller value on the left:

```python
answer = [-1] * len(nums)
stack = []

for i, value in enumerate(nums):
    while stack and nums[stack[-1]] >= value:
        stack.pop()

    if stack:
        answer[i] = stack[-1]

    stack.append(i)
```

Here the answer belongs to the current index `i`. The `while` loop removes candidates that cannot answer the current index. Once it finishes, the top is the nearest strictly smaller value.

| Question | What the `while` Loop Does | Where to Write the Answer |
|---|---|---|
| First qualifying value on the right | The current value resolves old indices | Write `answer[j]` while popping |
| Nearest qualifying value on the left | Remove candidates invalid for the current index | Read the top after `while`, then write `answer[i]` |

### Largest Rectangle: Determine Both Boundaries When Popping

In [84. Largest Rectangle in Histogram](https://neetcode.io/problems/largest-rectangle-in-histogram/question?list=neetcode150), when a shorter bar at `right` pops index `j`:

```text
right = first strictly shorter position on the right
stack[-1] after the pop = nearest shorter position on the left
```

The width covered by `heights[j]` is therefore:

$$
\text{width}
=
\text{right}
-
\text{left}
-
1.
$$

A final sentinel bar of height `0` pops all remaining bars, avoiding a duplicated cleanup loop.

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        answer = 0
        stack = []

        for right in range(len(heights) + 1):
            current = 0 if right == len(heights) else heights[right]

            while stack and heights[stack[-1]] > current:
                j = stack.pop()
                left = stack[-1] if stack else -1
                width = right - left - 1
                answer = max(answer, heights[j] * width)

            stack.append(right)

        return answer
```

The sentinel index is pushed during the last iteration, but the loop ends immediately afterward, so `heights[len(heights)]` is never read.

</details>

### Why the Nested `while` Is Still O(n)

Do not multiply the outer `for` and inner `while` mechanically. Each index is:

```text
pushed at most once
popped at most once
```

There are at most $n$ pushes and $n$ pops, so the total number of stack operations is $O(n)$. The stack holds at most $n$ indices, giving $O(n)$ space.

This is the same amortized argument used for two pointers and sliding windows: a local loop may repeat, but an element never returns after it is removed.

### Final Interview Checklist

1. Should the stack store values or indices? Prefer indices when you need distances, boundaries, or access to the original array.
2. Is the current element resolving old indices, or looking for its own answer on the left?
3. Does the problem ask for a strict or non-strict comparison? That decides `<` versus `<=`.
4. Should `answer[j]` store an index, value, or `i - j`?
5. In a histogram, does a sentinel empty the remaining stack?

Keep one sentence in memory:

> The index at the top still has no answer. Once the current value satisfies what it is waiting for, pop it and write its answer.
