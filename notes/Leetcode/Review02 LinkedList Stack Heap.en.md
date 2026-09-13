# Review Flashcards: Linked List, Stack & Heap

This note is the second volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Linked List & Hash-Linked Structures**, **Stack & Monotonic Stack / Deque**, and **Heap & Priority Queue** with production-grade implementations and rigorous complexity breakdowns.

---

## Module 1: Linked List & Hash-Linked Structures

### 1. LRU Cache & System-Level Extensions

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Linked List 01</span>
  <span class="review-card-title">LRU Cache & System-Level Extensions</span>
  <span class="review-card-tag">Hash Map + Doubly Linked List · Frequency Buckets · O(1) Eviction</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 146 · LRU Cache](https://leetcode.com/problems/lru-cache/) — `https://leetcode.com/problems/lru-cache/`
> - [LeetCode 460 · LFU Cache](https://leetcode.com/problems/lfu-cache/) — `https://leetcode.com/problems/lfu-cache/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **LRU Cache & System-Level Extensions (LeetCode 146 / 460)**:
> Design a data structure that follows the constraints of a Least Recently Used (LRU) cache with strict $\mathcal{O}(1)$ average time complexity for both `get` and `put`.
>
> **Reported Follow-ups**:
> 1. **Add TTL**: Each entry has an expiry time. `get` returns -1 if expired; expired entries must not occupy capacity.
> 2. **Add LFU**: A secondary eviction round where ties on usage frequency are broken by recency.
> 3. **4-End List**: Support `rpush`, `rpop`, `lpush`, `lpop` plus indexed access in $\mathcal{O}(1)$.
> 4. **Print / Iterate**: Walk cache from least-recent to most-recent.
> 5. **High Miss-Rate Tuning**: Capacity growth, ARC/2Q policies, prefetching, and consistent hash sharding.

**Interface Definition**:
```python
class LRUCache:
    def __init__(self, capacity: int): ...
    def get(self, key: int) -> int: ...
    def put(self, key: int, value: int) -> None: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
import time
from typing import Dict, Optional

class DLLNode:
    __slots__ = ('key', 'val', 'prev', 'next', 'expiry')
    def __init__(self, key: int = 0, val: int = 0, expiry: float = float('inf')):
        self.key, self.val, self.expiry = key, val, expiry
        self.prev = self.next = None

class LRUCacheWithTTL:
    def __init__(self, capacity: int):
        if capacity <= 0: raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.map: Dict[int, DLLNode] = {}
        self.head, self.tail = DLLNode(), DLLNode()
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node: DLLNode) -> None:
        node.prev.next, node.next.prev = node.next, node.prev

    def _append_to_tail(self, node: DLLNode) -> None:
        node.prev, node.next = self.tail.prev, self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def _evict_node(self, node: DLLNode) -> None:
        self._remove(node)
        self.map.pop(node.key, None)

    def get(self, key: int) -> int:
        if key not in self.map: return -1
        node = self.map[key]
        if time.time() > node.expiry:
            self._evict_node(node)
            return -1
        self._remove(node)
        self._append_to_tail(node)
        return node.val

    def put(self, key: int, value: int, ttl: Optional[float] = None) -> None:
        expiry = time.time() + ttl if ttl is not None else float('inf')
        if key in self.map:
            node = self.map[key]
            node.val, node.expiry = value, expiry
            self._remove(node)
            self._append_to_tail(node)
            return
        if len(self.map) >= self.capacity:
            self._evict_node(self.head.next)
        new_node = DLLNode(key, value, expiry)
        self.map[key] = new_node
        self._append_to_tail(new_node)

if __name__ == "__main__":
    lru = LRUCacheWithTTL(2)
    lru.put(1, 10, ttl=100)
    lru.put(2, 20, ttl=100)
    assert lru.get(1) == 10
    lru.put(3, 30, ttl=100)  # 淘汰 key 2
    assert lru.get(2) == -1
    assert lru.get(3) == 30
    assert lru.get(1) == 10
    print("✅ Card 01 (LRU Cache with TTL) all tests passed!")
```

</div>

</div>
</details>

---

### 2. Reverse Nodes in k-Group & Structural Group Inversion

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Linked List 02</span>
  <span class="review-card-title">Reverse Nodes in k-Group & Structural Group Inversion</span>
  <span class="review-card-tag">Dummy Head · Lookahead K-Check · Sub-List Inversion · O(1) Space</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 25 · Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) — `https://leetcode.com/problems/reverse-nodes-in-k-group/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Reverse Nodes in k-Group (LeetCode 25)**:
> Given the head of a linked list, reverse the nodes of the list $k$ at a time, and return the modified list.
> $k$ is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of $k$ then left-out nodes, in the end, should remain as it is.
>
> **Industrial Follow-ups**:
> 1. What if remaining nodes (< $k$) at the tail should also be reversed?
> 2. What if groups of $k$ themselves are reversed in order while preserving relative order within each group?

**Function Signature**:
```python
class KGroupReverser:
    @staticmethod
    def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]: ...
```

**Examples**:
- `head = [1, 2, 3, 4, 5], k = 2` $\implies$ `[2, 1, 4, 3, 5]`
- `head = [1, 2, 3, 4, 5], k = 3` $\implies$ `[3, 2, 1, 4, 5]`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import Optional, List, Tuple

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val, self.next = val, next

class KGroupReverser:
    @staticmethod
    def _reverse_segment(head: Optional[ListNode]) -> Tuple[Optional[ListNode], Optional[ListNode]]:
        prev, cur, tail = None, head, head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev, cur = cur, nxt
        return prev, tail

    @classmethod
    def reverseKGroup(cls, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or k <= 1: return head
        dummy = ListNode(0, head)
        group_prev = dummy
        while True:
            group_end = group_prev
            for _ in range(k):
                group_end = group_end.next
                if not group_end: return dummy.next
            group_start = group_prev.next
            next_group_head = group_end.next
            group_end.next = None
            new_head, new_tail = cls._reverse_segment(group_start)
            group_prev.next = new_head
            new_tail.next = next_group_head
            group_prev = new_tail

if __name__ == "__main__":
    def build_list(vals):
        d = ListNode(0); c = d
        for v in vals: c.next = ListNode(v); c = c.next
        return d.next

    def to_list(head):
        r = []
        while head: r.append(head.val); head = head.next
        return r

    h1 = build_list([1, 2, 3, 4, 5])
    assert to_list(KGroupReverser.reverseKGroup(h1, 2)) == [2, 1, 4, 3, 5]
    h2 = build_list([1, 2, 3, 4, 5])
    assert to_list(KGroupReverser.reverseKGroup(h2, 3)) == [3, 2, 1, 4, 5]
    print("✅ Card 02 (Reverse Nodes in k-Group) all tests passed!")
```

</div>

</div>
</details>

---

### 3. Flatten Multilevel Doubly Linked List with Empty-Node Filtering

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Linked List 03</span>
  <span class="review-card-title">Flatten Multilevel Doubly Linked List with Empty-Node Filtering</span>
  <span class="review-card-tag">Pointer Relinking · DFS Traversal · Tail Backtracking · In-Place Flattening</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 430 · Flatten a Multilevel Doubly Linked List](https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/) — `https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Flatten a Multilevel Doubly Linked List (LeetCode 430)**:
> You are given a doubly linked list, which contains nodes that have a next pointer, a previous pointer, and an additional child pointer. This child pointer may or may not point to a separate doubly linked list, also containing these special nodes.
> Flatten the list so that all the nodes appear in a single-level, doubly linked list. Nodes in the child list should appear between the parent node and the parent node's next node.
>
> **Industrial Extension**: Filter out empty or null nodes (`val is None`) during flattening and maintain bidirectional pointer integrity.

**Class & Method Signature**:
```python
class MultiLevelNode:
    def __init__(self, val=None, prev=None, next=None, child=None): ...

class MultiLevelListFlattenSolution:
    @classmethod
    def flattenAndFilterEmpty(cls, head: Optional[MultiLevelNode]) -> Optional[MultiLevelNode]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import Optional

class MultiLevelNode:
    def __init__(self, val: Optional[int] = None, prev=None, next=None, child=None):
        self.val, self.prev, self.next, self.child = val, prev, next, child

class MultiLevelListFlattenSolution:
    @classmethod
    def flattenAndFilterEmpty(cls, head: Optional[MultiLevelNode]) -> Optional[MultiLevelNode]:
        if not head: return None
        cur, stack = head, []
        while cur:
            if cur.child:
                if cur.next: stack.append(cur.next)
                cur.next = cur.child
                cur.child.prev = cur
                cur.child = None
            if not cur.next and stack:
                nxt = stack.pop()
                cur.next = nxt
                nxt.prev = cur
            cur = cur.next

        dummy = MultiLevelNode(0, next=head)
        head.prev = dummy
        cur = head
        while cur:
            nxt = cur.next
            if cur.val is None:
                cur.prev.next = nxt
                if nxt: nxt.prev = cur.prev
                cur.prev = cur.next = None
            cur = nxt

        new_head = dummy.next
        if new_head: new_head.prev = None
        return new_head

if __name__ == "__main__":
    n1 = MultiLevelNode(1)
    n2 = MultiLevelNode(2)
    n_empty = MultiLevelNode(None)
    n3 = MultiLevelNode(3)
    n1.next = n2; n2.prev = n1
    n2.child = n_empty
    n_empty.child = n3
    res_head = MultiLevelListFlattenSolution.flattenAndFilterEmpty(n1)
    vals = []
    c = res_head
    while c:
        vals.append(c.val)
        c = c.next
    assert vals == [1, 2, 3]
    print("✅ Card 03 (Flatten Multilevel Doubly Linked List) all tests passed!")
```

</div>

</div>
</details>

---

## Module 2: Stack & Monotonic Stack / Deque

### 4. Basic Calculator & Operator Precedence Hierarchy

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Stack 04</span>
  <span class="review-card-title">Basic Calculator & Operator Precedence Hierarchy</span>
  <span class="review-card-tag">Operator Precedence · Recursive Descent · Stack Evaluation · Parentheses Scoping</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 224 · Basic Calculator](https://leetcode.com/problems/basic-calculator/) — `https://leetcode.com/problems/basic-calculator/`
> - [LeetCode 227 · Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/) — `https://leetcode.com/problems/basic-calculator-ii/`
> - [LeetCode 772 · Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/) — `https://leetcode.com/problems/basic-calculator-iii/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Basic Calculator Hierarchy (LeetCode 224, 227, 772)**:
> Given a string expression representing an arithmetic expression containing non-negative integers, operators `+`, `-`, `*`, `/`, and parentheses `(`, `)`, evaluate the expression and return its integer value.
> Division must truncate toward zero (e.g. $-3 // 2 = -1$).
> Must handle nested parentheses, arbitrary whitespace, and multi-digit integers in strict $\mathcal{O}(n)$ time.

**Function Signature**:
```python
class ExpressionCalculator:
    @classmethod
    def calculate(cls, s: str) -> int: ...
```

**Examples**:
- `s = "3+2*2"` $\implies$ `7`
- `s = " 3/2 "` $\implies$ `1`
- `s = " 3+5 / 2 "` $\implies$ `5`
- `s = "2*(5+5*2)/3+(6/2+8)"` $\implies$ `21`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class ExpressionCalculator:
    PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    IS_RIGHT_ASSOCIATIVE = {'^': True, '+': False, '-': False, '*': False, '/': False}

    @staticmethod
    def _trunc_div(a: int, b: int) -> int:
        if b == 0: raise ZeroDivisionError("Division by zero")
        return int(a / b)

    @classmethod
    def _apply_op(cls, op: str, b: int, a: int) -> int:
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return cls._trunc_div(a, b)
        if op == '^': return a ** b
        raise ValueError(f"Unknown operator: {op}")

    @classmethod
    def calculate(cls, s: str) -> int:
        nums, ops = [], []
        i, n, expect_operand = 0, len(s), True

        def evaluate_top_op():
            op = ops.pop()
            b = nums.pop()
            a = nums.pop()
            nums.append(cls._apply_op(op, b, a))

        while i < n:
            ch = s[i]
            if ch == ' ': i += 1; continue
            if ch.isdigit():
                val = 0
                while i < n and s[i].isdigit():
                    val = val * 10 + int(s[i]); i += 1
                nums.append(val); expect_operand = False; continue
            if ch == '(':
                ops.append('('); expect_operand = True; i += 1; continue
            if ch == ')':
                while ops and ops[-1] != '(': evaluate_top_op()
                ops.pop(); expect_operand = False; i += 1; continue
            if ch in cls.PRECEDENCE:
                if expect_operand:
                    if ch in ('+', '-'): nums.append(0)
                    else: raise ValueError(f"Syntax error: {ch}")
                cur_prec, is_right = cls.PRECEDENCE[ch], cls.IS_RIGHT_ASSOCIATIVE[ch]
                while ops and ops[-1] != '(':
                    top_prec = cls.PRECEDENCE.get(ops[-1], 0)
                    if (not is_right and top_prec >= cur_prec) or (is_right and top_prec > cur_prec):
                        evaluate_top_op()
                    else: break
                ops.append(ch); expect_operand = True; i += 1; continue
            raise ValueError(f"Invalid character: {ch}")

        while ops: evaluate_top_op()
        return nums[0] if nums else 0

if __name__ == "__main__":
    assert ExpressionCalculator.calculate("1 + 1") == 2
    assert ExpressionCalculator.calculate(" 2-1 + 2 ") == 3
    assert ExpressionCalculator.calculate("(1+(4+5+2)-3)+(6+8)") == 23
    assert ExpressionCalculator.calculate("3+2*2") == 7
    assert ExpressionCalculator.calculate(" 3/2 ") == 1
    assert ExpressionCalculator.calculate(" 3+5 / 2 ") == 5
    assert ExpressionCalculator.calculate("2^3^2") == 512
    assert ExpressionCalculator.calculate("-3 + 5") == 2
    print("✅ Card 04 (Basic Calculator Hierarchy) all tests passed!")
```

</div>

</div>
</details>

---

### 5. Sliding Window Maximum & Monotonic Deque Pattern

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Queue 05</span>
  <span class="review-card-title">Sliding Window Maximum & Monotonic Deque Pattern</span>
  <span class="review-card-tag">Monotonic Deque · Index Expiry Eviction · Amortized O(1) Transition</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 239 · Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) — `https://leetcode.com/problems/sliding-window-maximum/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Sliding Window Maximum (LeetCode 239)**:
> You are given an array of integers `nums`, there is a sliding window of size $k$ which is moving from the very left of the array to the very right. You can only see the $k$ numbers in the window. Each time the sliding window moves right by one position.
> Return the max sliding window.

**Function Signature**:
```python
class SlidingWindowMaxSolution:
    @classmethod
    def maxSlidingWindow(cls, nums: List[int], k: int) -> List[int]: ...
```

**Examples**:
- `nums = [1,3,-1,-3,5,3,6,7], k = 3` $\implies$ `[3, 3, 5, 5, 6, 7]`
- `nums = [1], k = 1` $\implies$ `[1]`

**Key Invariant**:
- Maintain a monotonic deque of indices with values strictly decreasing from front to back. Pop front when index falls out of $i - k$, and pop back values smaller than current element for amortized $\mathcal{O}(1)$ step time.

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from collections import deque
from typing import List

class SlidingWindowMaxSolution:
    @staticmethod
    def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
        if not nums or k <= 0: return []
        q = deque()
        res = []
        for i, val in enumerate(nums):
            while q and nums[q[-1]] <= val: q.pop()
            q.append(i)
            if q[0] <= i - k: q.popleft()
            if i >= k - 1: res.append(nums[q[0]])
        return res

if __name__ == "__main__":
    assert SlidingWindowMaxSolution.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert SlidingWindowMaxSolution.maxSlidingWindow([1], 1) == [1]
    assert SlidingWindowMaxSolution.maxSlidingWindow([1, -1], 1) == [1, -1]
    assert SlidingWindowMaxSolution.maxSlidingWindow([9, 11], 2) == [11]
    assert SlidingWindowMaxSolution.maxSlidingWindow([4, -2], 2) == [4]
    print("✅ Card 05 (Sliding Window Maximum) all tests passed!")
```

</div>

</div>
</details>

---

### 6. Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Stack 06</span>
  <span class="review-card-title">Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern</span>
  <span class="review-card-tag">Monotonic Stack · Dual Sentinels · Width Calculation · 2D Reduction</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 84 · Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) — `https://leetcode.com/problems/largest-rectangle-in-histogram/`
> - [LeetCode 85 · Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/) — `https://leetcode.com/problems/maximal-rectangle/`

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Requirements</div>

Given an array of integers `heights` representing histogram bar heights of width 1, return the area of the largest rectangle inside the histogram (LC 84):

```python
def largestRectangleArea(heights: List[int]) -> int: ...
```

Examples:
- `heights = [2, 1, 5, 6, 2, 3]` $\implies 10$
- `heights = [2, 1, 2]` $\implies 3$

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Two-Sentinel Technique</div>

- **Monotonic Increasing Stack**: A bar popped at index $mid$ finds its right smaller boundary at current index $i$ and its left smaller boundary at new stack top $stack[-1]$.
- Width: $i - stack[-1] - 1$.
- **Two-Sentinel Trick**: Prepend and append $0$ (`[0] + heights + [0]`). The first $0$ prevents empty-stack edge cases; the last $0$ flushes the entire stack without post-loop logic.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List

class HistogramSolution:

    @staticmethod
    def largestRectangleArea(heights: List[int]) -> int:
        if not heights:
            return 0

        padded = [0] + heights + [0]
        stack = []
        max_area = 0

        for i, h in enumerate(padded):
            while stack and padded[stack[-1]] > h:
                mid = stack.pop()
                height = padded[mid]
                width = i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)

        return max_area

if __name__ == "__main__":
    assert HistogramSolution.largestRectangleArea([2, 1, 5, 6, 2, 3]) == 10
    assert HistogramSolution.largestRectangleArea([2, 4]) == 4
    assert HistogramSolution.largestRectangleArea([]) == 0
    assert HistogramSolution.largestRectangleArea([2, 1, 2]) == 3
    print("✅ Card 06 (Largest Rectangle in Histogram) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: Strictly $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(N)$.
- **Critical Pitfalls**: Off-by-one errors on width; must be `i - stack[-1] - 1`.

</div>

</div>
</details>

---

### 7. Valid Parenthesis String with Wildcard & Concrete String Enumeration

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Stack 07</span>
  <span class="review-card-title">Valid Parenthesis String with Wildcard & Concrete String Enumeration</span>
  <span class="review-card-tag">Interval Greedy · O(N) Dual Bounds · Branch-and-Bound DFS</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 678 · Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/) — `https://leetcode.com/problems/valid-parenthesis-string/`

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Dual Tasks</div>

Given string `s` containing `(`, `)`, and `*` (which can act as `(`, `)`, or `""`):
1. `checkValidString(s: str) -> bool`: Validate in strict $\mathcal{O}(N)$ time.
2. `allValidStrings(s: str) -> List[str]`: Return all distinct valid concrete strings produced by DFS expanding each `*`.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Range Greedy Mechanics</div>

- Maintain possible open count range $[low, high]$:
  - `(`: $low+1, high+1$; `)`: $low-1, high-1$; `*`: $low-1, high+1$.
  - Clamp $low = \max(0, low)$.
  - If $high < 0$, fail immediately. At the end, $low == 0 \implies \text{Valid}$.
- All-strings DFS: Branch `*` into 3 possibilities, prune on $balance < 0$, collect into set when $balance == 0$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List

class WildcardParenthesesSolution:

    @staticmethod
    def checkValidString(s: str) -> bool:
        low = 0
        high = 0
        for ch in s:
            if ch == '(':
                low += 1; high += 1
            elif ch == ')':
                low -= 1; high -= 1
            elif ch == '*':
                low -= 1; high += 1
            if high < 0:
                return False
            if low < 0:
                low = 0
        return low == 0

    @classmethod
    def allValidStrings(cls, s: str) -> List[str]:
        n = len(s)
        results = set()
        path = []

        def dfs(i: int, balance: int):
            if balance < 0: return
            if i == n:
                if balance == 0:
                    results.add("".join(path))
                return

            ch = s[i]
            if ch == '(':
                path.append('('); dfs(i + 1, balance + 1); path.pop()
            elif ch == ')':
                path.append(')'); dfs(i + 1, balance - 1); path.pop()
            else:
                path.append('('); dfs(i + 1, balance + 1); path.pop()
                path.append(')'); dfs(i + 1, balance - 1); path.pop()
                dfs(i + 1, balance)

        dfs(0, 0)
        return sorted(list(results))

if __name__ == "__main__":
    assert WildcardParenthesesSolution.checkValidString("()") is True
    assert WildcardParenthesesSolution.checkValidString("(*)") is True
    assert WildcardParenthesesSolution.checkValidString("(*))") is True
    assert WildcardParenthesesSolution.checkValidString(")(") is False
    assert WildcardParenthesesSolution.allValidStrings("(*)") == ["()"]
    assert WildcardParenthesesSolution.allValidStrings("(*))") == ["(())"]
    print("✅ Card 07 (Valid Parenthesis String) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$ for boolean validity; $\mathcal{O}(3^K)$ for enumeration with $K$ wildcards.
- **Space Complexity**: $\mathcal{O}(1)$ for validity; $\mathcal{O}(N)$ recursion depth.

</div>

</div>
</details>

---

## Module 3: Heap & Priority Queue

### 8. Find Median from Data Stream & K-Way Merge

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 08</span>
  <span class="review-card-title">Find Median from Data Stream & K-Way Merge</span>
  <span class="review-card-tag">Dual Heaps · Balance Invariant · Lazy Eviction</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 295 · Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) — `https://leetcode.com/problems/find-median-from-data-stream/`
> - [LeetCode 23 · Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) — `https://leetcode.com/problems/merge-k-sorted-lists/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Find Median from Data Stream (LeetCode 295)**:
> The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.
> Implement the `MedianFinder` class:
> - `MedianFinder()` initializes the MedianFinder object.
> - `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
> - `double findMedian()` returns the median of all elements so far.

**Interface Definition**:
```python
class MedianFinder:
    def __init__(self): ...
    def addNum(self, num: int) -> None: ...
    def findMedian(self) -> float: ...
```

**Examples**:
- `mf = MedianFinder()`
- `mf.addNum(1); mf.addNum(2); mf.findMedian()` $\implies$ `1.5`
- `mf.addNum(3); mf.findMedian()` $\implies$ `2.0`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
import heapq
from typing import List

class MedianFinder:
    def __init__(self):
        self.lo: List[int] = []
        self.hi: List[int] = []

    def addNum(self, num: int) -> None:
        heapq.heappush(self.lo, -num)
        max_lo = -heapq.heappop(self.lo)
        heapq.heappush(self.hi, max_lo)
        if len(self.hi) > len(self.lo):
            min_hi = heapq.heappop(self.hi)
            heapq.heappush(self.lo, -min_hi)

    def findMedian(self) -> float:
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (-self.lo[0] + self.hi[0]) / 2.0

if __name__ == "__main__":
    mf = MedianFinder()
    mf.addNum(1)
    mf.addNum(2)
    assert mf.findMedian() == 1.5
    mf.addNum(3)
    assert mf.findMedian() == 2.0
    print("✅ Card 08 (Find Median from Data Stream) all tests passed!")
```

</div>

</div>
</details>

---

### 9. Tiered Priority Task Scheduler

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 09</span>
  <span class="review-card-title">Tiered Priority Task Scheduler</span>
  <span class="review-card-tag">Merchant Min-Heap · FIFO Timestamp Queues · Active Round-Robin · VIP Weighted Allocation</span>
</summary>
<div class="review-card-content">

> 💡 **Problem Type**: Standalone Algorithm Implementation (No direct LeetCode equivalent; logic and test specifications are standalone, please run the self-contained test suite below for local verification).

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Tiered Priority Task Scheduler**:
> Design a multi-merchant fair round-robin task scheduler. Merchants have priority tiers (e.g., VIP vs Regular). Tasks arrive with timestamps.
> - `receive_task(merchant_id, task_id, is_vip, arrival_time)`
> - `process_next_task() -> tuple`
> Must dispatch tasks fairly: round-robin across active merchants, giving VIP merchants proportional quota, preserving FIFO arrival order within each merchant.

**Interface Definition**:
```python
class TieredTaskScheduler:
    def __init__(self, vip_ratio: int = 2): ...
    def receive_task(self, merchant_id: str, task_id: str, is_vip: bool, arrival_time: int) -> None: ...
    def process_next_task(self) -> Optional[Tuple[str, str, bool, int]]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional, Set

class TieredTaskScheduler:
    TIER_WEIGHTS = {'VIP': 2, 'STANDARD': 1}

    def __init__(self):
        self.seq = 0
        self.seller_heaps: Dict[str, List[Tuple[int, int, str]]] = {}
        self.seller_tiers: Dict[str, str] = {}
        self.active_sellers: deque[str] = deque()
        self.in_active_set: Set[str] = set()
        self.current_seller: Optional[str] = None
        self.remaining_quota: int = 0

    def receive_task(self, task: dict) -> None:
        task_id = task['task_id']
        seller_id = task['seller_id']
        tier = task.get('tier', 'STANDARD')
        priority = task['priority']
        self.seq += 1
        self.seller_tiers[seller_id] = tier
        if seller_id not in self.seller_heaps:
            self.seller_heaps[seller_id] = []
        heapq.heappush(self.seller_heaps[seller_id], (priority, self.seq, task_id))
        if seller_id != self.current_seller and seller_id not in self.in_active_set:
            self.active_sellers.append(seller_id)
            self.in_active_set.add(seller_id)

    def process_next_task(self) -> Optional[Tuple[str, str]]:
        while True:
            if not self.current_seller or self.remaining_quota <= 0:
                if not self.active_sellers:
                    self.current_seller = None
                    self.remaining_quota = 0
                    return None
                self.current_seller = self.active_sellers.popleft()
                self.in_active_set.remove(self.current_seller)
                tier = self.seller_tiers.get(self.current_seller, 'STANDARD')
                self.remaining_quota = self.TIER_WEIGHTS.get(tier, 1)

            heap = self.seller_heaps.get(self.current_seller, [])
            if not heap:
                self.current_seller = None
                self.remaining_quota = 0
                continue

            priority, seq, task_id = heapq.heappop(heap)
            self.remaining_quota -= 1
            seller = self.current_seller

            if self.remaining_quota <= 0 or not heap:
                if heap:
                    self.active_sellers.append(self.current_seller)
                    self.in_active_set.add(self.current_seller)
                self.current_seller = None
                self.remaining_quota = 0
            return (seller, task_id)

if __name__ == "__main__":
    ts = TieredTaskScheduler()
    ts.receive_task({'task_id': 'T1', 'seller_id': 'S1', 'priority': 2, 'tier': 'VIP'})
    ts.receive_task({'task_id': 'T2', 'seller_id': 'S1', 'priority': 1, 'tier': 'VIP'})
    ts.receive_task({'task_id': 'T3', 'seller_id': 'S1', 'priority': 1, 'tier': 'VIP'})
    ts.receive_task({'task_id': 'T4', 'seller_id': 'S2', 'priority': 1, 'tier': 'STANDARD'})
    assert ts.process_next_task() == ('S1', 'T2')
    assert ts.process_next_task() == ('S1', 'T3')
    assert ts.process_next_task() == ('S2', 'T4')
    assert ts.process_next_task() == ('S1', 'T1')
    assert ts.process_next_task() is None
    print("✅ Card 09 (Tiered Task Scheduler) all tests passed!")
```

</div>

</div>
</details>

---

### 10. Timestamp Task Scheduler with Direct ID Removal

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 13</span>
  <span class="review-card-title">Timestamp Task Scheduler with Direct ID Removal</span>
  <span class="review-card-tag">Composite Min-Heap · Lazy Eviction · Hash Version Validation</span>
</summary>
<div class="review-card-content">

> 💡 **Problem Type**: Standalone Algorithm Implementation (No direct LeetCode equivalent; logic and test specifications are standalone, please run the self-contained test suite below for local verification).

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Timestamp Task Scheduler with Direct ID Removal**:
> Design a priority task scheduler managing tasks with timestamps, priorities, and unique task IDs:
> - `addTask(task_id, priority, timestamp)`: schedules a task.
> - `removeTask(task_id)`: cancels/evicts a task directly before execution.
> - `pollNextTask(current_time)`: dispatches highest-priority task whose timestamp $\le \text{current\_time}$.
> Must use a min-heap with lazy eviction and hash map version validation for $\mathcal{O}(\log N)$ amortized operations.

**Interface Definition**:
```python
class TimestampTaskScheduler:
    def __init__(self): ...
    def addTask(self, task_id: str, priority: int, timestamp: int) -> None: ...
    def removeTask(self, task_id: str) -> bool: ...
    def pollNextTask(self, current_time: int) -> Optional[Tuple[str, int, int]]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
import heapq
from typing import Dict, List

class TimestampTaskScheduler:
    def __init__(self):
        self.heap: List[tuple] = []
        self.task_map: Dict[str, int] = {}
        self.seq = 0

    def addTask(self, taskID: str, timestamp: int) -> None:
        self.seq += 1
        self.task_map[taskID] = timestamp
        heapq.heappush(self.heap, (timestamp, self.seq, taskID))

    def removeTask(self, taskID: str) -> bool:
        if taskID in self.task_map:
            del self.task_map[taskID]
            return True
        return False

    def popTask(self, num: int) -> List[str]:
        result = []
        while self.heap and len(result) < num:
            timestamp, _, taskID = heapq.heappop(self.heap)
            if self.task_map.get(taskID) == timestamp:
                result.append(taskID)
                del self.task_map[taskID]
        return result

if __name__ == "__main__":
    tts = TimestampTaskScheduler()
    tts.addTask("taskA", 100)
    tts.addTask("taskB", 200)
    tts.addTask("taskC", 150)
    tts.removeTask("taskA")
    assert tts.popTask(2) == ["taskC", "taskB"]
    assert tts.popTask(1) == []
    print("✅ Card 10 (Timestamp Task Scheduler) all tests passed!")
```

</div>

</div>
</details>

---

### 11. Reverse Linked List In-Place via Three Pointers

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LIST 08</span>
  <span class="review-card-title">Reverse Linked List In-Place via Three Pointers</span>
  <span class="review-card-tag">Three Pointers · In-Place Reversal · Subsegment Inversion · O(1) Space</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 206 · Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) — `https://leetcode.com/problems/reverse-linked-list/`
> - [LeetCode 92 · Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/) — `https://leetcode.com/problems/reverse-linked-list-ii/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Reverse Linked List (LeetCode 206 / 92)**:
> Given the head of a singly linked list, reverse the list in-place, and return the reversed list.
> Follow-up: implement both iterative three-pointer sliding and clean recursion, maintaining $\mathcal{O}(1)$ auxiliary space in iterative mode.

**Function Signature**:
```python
class ReverseListSolution:
    @staticmethod
    def reverseListIterative(head: Optional[ListNode]) -> Optional[ListNode]: ...
    @staticmethod
    def reverseListRecursive(head: Optional[ListNode]) -> Optional[ListNode]: ...
```

**Examples**:
- `head = [1, 2, 3, 4, 5]` $\implies$ `[5, 4, 3, 2, 1]`
- `head = [1, 2]` $\implies$ `[2, 1]`
- `head = []` $\implies$ `[]`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class ReverseListSolution:
    @classmethod
    def reverseListIterative(cls, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverses a singly linked list iteratively in-place.
        Time O(N), auxiliary space O(1).
        """
        prev: Optional[ListNode] = None
        curr = head

        while curr is not None:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt

        return prev

    @classmethod
    def reverseListRecursive(cls, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Reverses a singly linked list recursively.
        Time O(N), call stack space O(N).
        """
        if head is None or head.next is None:
            return head

        new_head = cls.reverseListRecursive(head.next)
        head.next.next = head
        head.next = None
        return new_head

if __name__ == "__main__":
    def build_list(vals):
        d = ListNode(); c = d
        for v in vals: c.next = ListNode(v); c = c.next
        return d.next

    def to_list(h):
        r = []
        while h: r.append(h.val); h = h.next
        return r

    rl_h = build_list([1, 2, 3, 4, 5])
    assert to_list(ReverseListSolution.reverseListIterative(rl_h)) == [5, 4, 3, 2, 1]
    rl_h2 = build_list([1, 2])
    assert to_list(ReverseListSolution.reverseListRecursive(rl_h2)) == [2, 1]
    print("✅ Card 11 (Reverse Linked List) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Three-Pointer Invariant**:
  At any step, `prev` anchors the reversed prefix, `curr` points to the active node being re-pointed, and `nxt` preserves the unvisited suffix. Each reassignment `curr.next = prev` operates with zero heap allocations.
- **Recursive Tail Clearance**:
  `head.next.next = head` reverses the pointer of the immediate child. Clearing `head.next = None` prevents circular deadlocks at the original list head.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$, visiting each node once.
- **Space Complexity**: $\mathcal{O}(1)$ for iterative; $\mathcal{O}(N)$ call stack for recursive.

</div>

</div>
</details>

---

### 12. Remove Duplicate Letters via Monotonic Stack

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STACK 09</span>
  <span class="review-card-title">Remove Duplicate Letters via Monotonic Stack</span>
  <span class="review-card-tag">Monotonic Stack · Last Occurrence Map · Visited Bitset · O(N)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 316 · Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/) — `https://leetcode.com/problems/remove-duplicate-letters/`
> - [LeetCode 1081 · Smallest Subsequence of Distinct Characters](https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/) — `https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Remove Duplicate Letters (LeetCode 316 / 1081)**:
> Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.

**Function Signature**:
```python
class RemoveDuplicateLettersSolution:
    @classmethod
    def removeDuplicateLetters(cls, s: str) -> str: ...
```

**Examples**:
- `s = "bcabc"` $\implies$ `"abc"`
- `s = "cbacdcbc"` $\implies$ `"acdb"`

**Monotonic Stack Invariant**:
- Maintain a monotonic increasing stack. Pop the top element if it is lexicographically larger than the current character AND it will appear again later in the string, maintaining $\mathcal{O}(n)$ time.

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
class RemoveDuplicateLettersSolution:
    @classmethod
    def removeDuplicateLetters(cls, s: str) -> str:
        """
        Removes duplicate letters to ensure each appears once, minimizing lexicographical order.
        """
        last_occurrence = {ch: i for i, ch in enumerate(s)}
        stack = []
        in_stack = set()

        for i, ch in enumerate(s):
            if ch in in_stack:
                continue

            # Greedy monotonic stack maintenance:
            # Pop top character if it is larger than ch AND appears again later
            while stack and stack[-1] > ch and last_occurrence[stack[-1]] > i:
                popped = stack.pop()
                in_stack.remove(popped)

            stack.append(ch)
            in_stack.add(ch)

        return "".join(stack)

if __name__ == "__main__":
    assert RemoveDuplicateLettersSolution.removeDuplicateLetters("bcabc") == "abc"
    assert RemoveDuplicateLettersSolution.removeDuplicateLetters("cbacdcbc") == "acdb"
    print("✅ Card 12 (Remove Duplicate Letters) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Monotonic Greedy Rule**:
  Smaller characters should precede larger ones. If stack top `top > ch` and `last_occurrence[top] > i`, discarding `top` now is safe because it will appear downstream, allowing smaller `ch` to claim a higher significance index.
- **Last-Appearance Shield**:
  If `last_occurrence[top] <= i`, `top` cannot be discarded; doing so would fail the requirement that every unique character be retained.
- **Deduplication Guard**:
  If `ch` is already in the stack, its existing position is optimal under prior choices; re-adding it would only increase lexicographical weight.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$, each character pushed and popped at most once.
- **Space Complexity**: $\mathcal{O}(|\Sigma|)$ where $|\Sigma| \le 26$ for the alphabet.

</div>

</div>
</details>

---

### 13. MinStack, MaxStack, Streaming Median & System Extensions

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">HEAP 13</span>
  <span class="review-card-title">MinStack, MaxStack, Streaming Median & System Extensions</span>
  <span class="review-card-tag">Dual Heaps · O(1) Extremum Stack · Lazy Eviction · Streaming Extensions</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 155 · Min Stack](https://leetcode.com/problems/min-stack/) — `https://leetcode.com/problems/min-stack/`
> - [LeetCode 716 · Max Stack](https://leetcode.com/problems/max-stack/) — `https://leetcode.com/problems/max-stack/`
> - [LeetCode 295 · Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) — `https://leetcode.com/problems/find-median-from-data-stream/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **MinStack, MaxStack & Streaming Median System (LeetCode 155 / 716 / 295)**:
> Design stack structures supporting $\mathcal{O}(1)$ push, pop, top, and retrieval of the minimum (or maximum) element:
> - `MinStack`: tracks running minimum using parallel min stack in $\mathcal{O}(1)$.
> - `Streaming Median`: maintains running median over an infinite stream via dual balancing heaps in $\mathcal{O}(\log n)$ per insertion.

**Interface Definition**:
```python
class MinStack:
    def __init__(self): ...
    def push(self, val: int) -> None: ...
    def pop(self) -> None: ...
    def top(self) -> int: ...
    def getMin(self) -> int: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
import heapq
from typing import Optional

class MinStack:
    """O(1) auxiliary stack tracking prefix minimums."""
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


class MedianFinder:
    """Two-heap balance structure maintaining a streaming dynamic median."""
    def __init__(self):
        self.small = []  # Max-heap (storing negated values): lower half
        self.large = []  # Min-heap: upper half

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))

        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        return (-self.small[0] + self.large[0]) / 2.0

if __name__ == "__main__":
    ms = MinStack()
    ms.push(-2)
    ms.push(0)
    ms.push(-3)
    assert ms.getMin() == -3
    ms.pop()
    assert ms.top() == 0
    assert ms.getMin() == -2
    print("✅ Card 13 (MinStack & Streaming Median) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Two-Heap Invariant**:
  All elements in `small` are $\le$ all elements in `large`. Sizes are kept balanced with $0 \le |small| - |large| \le 1$. Median extraction is $\mathcal{O}(1)$; insertion is $\mathcal{O}(\log N)$.
- **System Extensions & Follow-ups**:
  1. **MaxStack + Running Median**:
     To support `popMax` alongside median tracking, combine a Doubly-Linked List with an ordered balanced BST (`TreeMap`) for $\mathcal{O}(\log N)$ arbitrary deletion, coupled with hash-indexed lazy deletion on the two heaps.
  2. **Bounded Domain (0..100)**:
     If inputs are restricted to $[0, 100]$, heaps are completely replaced by a fixed frequency array `count[101]`. Insertion is $\mathcal{O}(1)$; finding the median takes a bounded $\le 101$-step prefix scan ($\mathcal{O}(1)$ time).
  3. **Memory-Constrained Streams**:
     When the stream exceeds RAM capacity, exact medians are intractable. Deploy streaming quantile estimators: **t-digest** (clustering near values into centroids) or **Reservoir Sampling**.
  4. **High-Concurrency Access**:
     Protect heaps via a **Read-Write Lock** (`findMedian` takes shared read locks; `addNum` acquires an exclusive write lock). Note: Medians are holistic rank statistics and cannot be partitioned across independent worker shards without coordinate bisection.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **MinStack**: $\mathcal{O}(1)$ time for all operations, $\mathcal{O}(N)$ auxiliary space.
- **MedianFinder**: `addNum` in $\mathcal{O}(\log N)$, `findMedian` in $\mathcal{O}(1)$, $\mathcal{O}(N)$ space.

</div>

</div>
</details>

