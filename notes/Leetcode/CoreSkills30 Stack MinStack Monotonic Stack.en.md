# Stack · MinStack and Monotonic Stack

The stack API is simple. The two patterns that matter in interviews are:

```text
MinStack: preserve history during push so a query never has to rescan
Monotonic stack: let indices wait until the first qualifying value on the right resolves them
```

MinStack is a fairly isolated design problem; memorizing one reliable implementation is enough. Monotonic stack is a family of problems. Instead of memorizing each solution, keep one template and replace only its comparator and answer format.

## Learning Order

The problem set follows [NeetCode 150](https://neetcode.io/practice/practice/neetcode150). `Generate Parentheses` is fundamentally a backtracking problem; its stack only stores the current path and does not use the monotonic-stack template below.

| Order | Original Problem | What to Learn |
|---:|---|---|
| 1 | [20. Valid Parentheses](https://neetcode.io/problems/validate-parentheses/question?list=neetcode150) | The last opening bracket must close first |
| 2 | [155. Min Stack](https://neetcode.io/problems/minimum-stack/question?list=neetcode150) | Store `min_so_far` in every stack frame |
| 3 | [150. Evaluate Reverse Polish Notation](https://neetcode.io/problems/evaluate-reverse-polish-notation/question?list=neetcode150) | An operator consumes the two most recent operands |
| 4 | [22. Generate Parentheses](https://neetcode.io/problems/generate-parentheses/question?list=neetcode150) | Use a stack as the backtracking path |
| 5 | [739. Daily Temperatures](https://neetcode.io/problems/daily-temperatures/question?list=neetcode150) | First greater value on the right |
| 6 | [853. Car Fleet](https://neetcode.io/problems/car-fleet/question?list=neetcode150) | Maintain fleet arrival times after sorting |
| 7 | [84. Largest Rectangle in Histogram](https://neetcode.io/problems/largest-rectangle-in-histogram/question?list=neetcode150) | Use a monotonic stack to determine boundaries |

## 1. Ordinary Stack: Start with LIFO

### Quick Coding: Valid Parentheses

Push an opening bracket. A closing bracket can match only the most recent unmatched opening bracket. Return `False` if the stack is empty, the types differ, or unmatched openings remain at the end.

<details>
<summary>Reference answer</summary>

```python
class Solution:
    def isValid(self, s: str) -> bool:
        opening = {
            ")": "(",
            "]": "[",
            "}": "{",
        }
        stack = []

        for bracket in s:
            if bracket not in opening:
                stack.append(bracket)
                continue

            if not stack or stack.pop() != opening[bracket]:
                return False

        return not stack
```

Both time and space complexity are $O(n)$.

</details>

### Quick Coding: Evaluate Reverse Polish Notation

Push numbers. When an operator appears, pop the two most recent values, evaluate them, and push the result. Subtraction and division require the original operand order:

```text
right = stack.pop()
left = stack.pop()
result = left op right
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []

        for token in tokens:
            if token not in {"+", "-", "*", "/"}:
                stack.append(int(token))
                continue

            right = stack.pop()
            left = stack.pop()

            if token == "+":
                stack.append(left + right)
            elif token == "-":
                stack.append(left - right)
            elif token == "*":
                stack.append(left * right)
            else:
                stack.append(int(left / right))

        return stack[-1]
```

Python's `//` rounds toward negative infinity, while the problem requires truncation toward zero. `int(left / right)` provides the required behavior here. Both time and space complexity are $O(n)$.

</details>

## 2. MinStack: Save the Minimum at Every Level

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

## 3. Generate Parentheses: The Stack Is Only the Current Path

This problem is easy to misclassify simply because it appears in a Stack study list. Its actual pruning rules are:

```text
add "(" only when opened < n
add ")" only when closed < opened
```

The `stack` is an efficient mutable path buffer; the core algorithm is still backtracking.

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        answer = []
        stack = []

        def backtrack(opened, closed):
            if len(stack) == 2 * n:
                answer.append("".join(stack))
                return

            if opened < n:
                stack.append("(")
                backtrack(opened + 1, closed)
                stack.pop()

            if closed < opened:
                stack.append(")")
                backtrack(opened, closed + 1)
                stack.pop()

        backtrack(0, 0)
        return answer
```

The output count is the $n$th Catalan number. Time is commonly written as $O(C_n n)$, proportional to the total length of all valid outputs, and the recursion path uses $O(n)$ space.

</details>

## 4. What Does a Monotonic Stack Actually Store?

Consider "the first strictly greater value to the right." When the scan reaches index `i`, the current value `nums[i]` can resolve only the indices at the top of the stack that are still waiting for an answer:

```python
while stack and nums[i] > nums[stack[-1]]:
    j = stack.pop()
    answer[j] = i
```

The most useful invariant is more precise than "the stack is decreasing":

> The stack stores indices that have been seen but whose answer on the right has not appeared yet.

Why is the current `i` the **first** answer for a popped index `j`? Since `j` was pushed, the scan has visited `j + 1, j + 2, ...` in order. If an earlier value had satisfied the condition, it would already have popped `j`.

## 5. One Unified Template

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

## 6. Daily Temperatures: Change an Index into a Distance

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

## 7. Answers on the Right and Left Have Different Recording Times

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

## 8. Car Fleet: Sort First, Then Keep New Slowest Arrival Times

Cars cannot pass, so first sort them from closest to the target to farthest away. A car's solo arrival time is:

$$
t_i=\frac{target-position_i}{speed_i}.
$$

Scan in that order:

```text
current time <= fleet time ahead: it catches that fleet by the target
current time > fleet time ahead: it cannot catch up and forms a new fleet
```

The stack stores arrival times of fleets that remain distinct, in strictly increasing order. This is not the "current value pops unresolved indices" template. Sorting first establishes which cars can interact; the stack then keeps only fleets that cannot merge.

### Quick Coding: Car Fleet

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def carFleet(
        self,
        target: int,
        position: List[int],
        speed: List[int],
    ) -> int:
        cars = sorted(zip(position, speed), reverse=True)
        fleet_times = []

        for start, velocity in cars:
            arrival = (target - start) / velocity

            if not fleet_times or arrival > fleet_times[-1]:
                fleet_times.append(arrival)

        return len(fleet_times)
```

Sorting costs $O(n\log n)$, followed by an $O(n)$ scan. The arrival-time stack holds at most $n$ values.

</details>

## 9. Largest Rectangle: Determine Both Boundaries When Popping

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

## 10. Why the Nested `while` Is Still O(n)

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
5. For Car Fleet, did you sort positions from closest to the target to farthest?
6. In a histogram, does a sentinel empty the remaining stack?

Keep one sentence in memory:

> The index at the top still has no answer. Once the current value satisfies what it is waiting for, pop it and write its answer.
