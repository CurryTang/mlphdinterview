# Stack · MinStack and Monotonic Stack

The stack API is simple. The two patterns that matter in interviews are:

```text
MinStack: preserve history during push so a query never has to rescan
Monotonic stack: let indices wait until the first qualifying value on the right resolves them
```

MinStack is a fairly isolated design problem; memorizing one reliable implementation is enough. Monotonic stack is a family of problems. Instead of memorizing each solution, keep one template and replace only its comparator and answer format.

## Learning Order

| Order | Original Problem | What to Learn |
|---:|---|---|
| 1 | [155. Min Stack](https://leetcode.com/problems/min-stack/description/) | Store `min_so_far` in every stack frame |
| 2 | [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/description/) | Standard next-greater-on-the-right template |
| 3 | [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/description/) | A circular array adds only a modulo scan |
| 4 | [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/description/) | Use a monotonic stack to determine boundaries |

## 1. MinStack: Save the Minimum at Every Level

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

## 2. What Does a Monotonic Stack Actually Store?

Consider "the first strictly greater value to the right." When the scan reaches index `i`, the current value `nums[i]` can resolve only the indices at the top of the stack that are still waiting for an answer:

```python
while stack and nums[i] > nums[stack[-1]]:
    j = stack.pop()
    answer[j] = i
```

The most useful invariant is more precise than "the stack is decreasing":

> The stack stores indices that have been seen but whose answer on the right has not appeared yet.

Why is the current `i` the **first** answer for a popped index `j`? Since `j` was pushed, the scan has visited `j + 1, j + 2, ...` in order. If an earlier value had satisfied the condition, it would already have popped `j`.

## 3. One Unified Template

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

## 4. Daily Temperatures: Change an Index into a Distance

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

## 5. Answers on the Right and Left Have Different Recording Times

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

## 6. Circular Arrays: Keep the Template and Scan Twice

The right side of a circular array may wrap to the beginning. Virtually duplicate indices `0 ... n - 1` by scanning `2n` positions:

```python
for scan in range(2 * n):
    i = scan % n
```

The second pass exists only to resolve questions left by the first pass. Do not push each index a second time.

### Quick Coding: Next Greater Element II

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        answer = [-1] * n
        stack = []

        for scan in range(2 * n):
            i = scan % n

            while stack and nums[stack[-1]] < nums[i]:
                j = stack.pop()
                answer[j] = nums[i]

            if scan < n:
                stack.append(i)

        return answer
```

This problem asks for values, so the assignment is `answer[j] = nums[i]`. Every index is pushed only during the first pass and popped at most once, keeping the total complexity at $O(n)$.

</details>

## 7. Largest Rectangle: Determine Both Boundaries When Popping

In [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/description/), when a shorter bar at `right` pops index `j`:

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

## 8. Why the Nested `while` Is Still O(n)

Do not multiply the outer `for` and inner `while` mechanically. Each index is:

```text
pushed at most once
popped at most once
```

There are at most $n$ pushes and $n$ pops, so the total number of stack operations is $O(n)$. The stack holds at most $n$ indices, giving $O(n)$ space.

This is the same amortized argument used for two pointers and sliding windows: a local loop may repeat, but an element never returns after it is removed.

## Final Interview Checklist

1. Should the stack store values or indices? Prefer indices when you need distances, boundaries, or access to the original array.
2. Is the current element resolving old indices, or looking for its own answer on the left?
3. Does the problem ask for a strict or non-strict comparison? That decides `<` versus `<=`.
4. Should `answer[j]` store an index, value, or `i - j`?
5. In a circular array, are indices pushed only during the first pass?
6. In a histogram, does a sentinel empty the remaining stack?

Keep one sentence in memory:

> The index at the top still has no answer. Once the current value satisfies what it is waiting for, pop it and write its answer.
