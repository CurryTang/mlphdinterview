# Review Flashcards: Linked List, Stack & Heap

This note is the second volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Linked List & Hash-Linked Structures**, **Stack & Monotonic Stack / Deque**, and **Heap & Priority Queue** with production-grade implementations and rigorous complexity breakdowns.

---

## Module 1: Linked List & Hash-Linked Structures

### 1. LRU Cache & System-Level Extensions

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Linked List 01</span>
  <span class="review-card-title">LRU Cache & System-Level Extensions</span>
  <span class="review-card-tag">Doubly Linked List · Hash Map · TTL Expiration · LFU Replacement · Miss-Rate Tuning</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>

---

### 2. Reverse Nodes in k-Group & Structural Group Inversion

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Linked List 02</span>
  <span class="review-card-title">Reverse Nodes in k-Group & Structural Group Inversion</span>
  <span class="review-card-tag">Dummy Sentinel · Local Pointer Reversal · Invert Group Order</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>

---

### 3. Flatten Multilevel Doubly Linked List with Empty-Node Filtering

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Linked List 03</span>
  <span class="review-card-title">Flatten Multilevel Doubly Linked List with Empty-Node Filtering</span>
  <span class="review-card-tag">Multilevel DLL · DFS Flattening · Empty Node Cleansing · Pointer Healing</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>

---

## Module 2: Stack & Monotonic Stack / Deque

### 4. Basic Calculator & Operator Precedence Hierarchy

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Stack 04</span>
  <span class="review-card-title">Basic Calculator & Operator Precedence Hierarchy</span>
  <span class="review-card-tag">Single-Stack Accumulation · Parentheses State Stashing · Right-Associative Exponent</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>

---

### 5. Sliding Window Maximum & Monotonic Deque Pattern

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Queue 05</span>
  <span class="review-card-title">Sliding Window Maximum & Monotonic Deque Pattern</span>
  <span class="review-card-tag">Monotonic Deque · Index Expiration Eviction · Amortized O(1) Transition</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>

---

### 6. Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Stack 06</span>
  <span class="review-card-title">Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern</span>
  <span class="review-card-tag">Monotonic Increasing Stack · Two-Sentinel Pattern · Dynamic Width · Maximal Rectangle Matrix</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Stack 07</span>
  <span class="review-card-title">Valid Parenthesis String with Wildcard & Concrete String Enumeration</span>
  <span class="review-card-tag">Interval Greedy · O(N) Two-Counter · Low Bound Clamping · DFS Branch Enumeration</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 08</span>
  <span class="review-card-title">Find Median from Data Stream & K-Way Merge</span>
  <span class="review-card-tag">Dual-Heap Balance · Invariant Maintenance</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>

---

### 9. Tiered Priority Task Scheduler

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 09</span>
  <span class="review-card-title">Tiered Priority Task Scheduler</span>
  <span class="review-card-tag">Per-Seller Min-Heap · Active Deque Round-Robin</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

```python
import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional, Set

class TieredTaskScheduler:
    def __init__(self):
        self.seq = 0
        self.seller_heaps: Dict[str, List[Tuple[int, int, str]]] = {}
        self.seller_tiers: Dict[str, str] = {}
        self.active_sellers: deque[str] = deque()
        self.in_active_set: Set[str] = set()
        self.current_seller: Optional[str] = None
        self.remaining_quota: int = 0

    def receive_task(self, task: dict) -> None:
        task_id, seller_id, priority = task['task_id'], task['seller_id'], task['priority']
        self.seq += 1
        self.seller_tiers[seller_id] = task.get('tier', 'STANDARD')
        if seller_id not in self.seller_heaps:
            self.seller_heaps[seller_id] = []
        heapq.heappush(self.seller_heaps[seller_id], (priority, self.seq, task_id))
        if seller_id != self.current_seller and seller_id not in self.in_active_set:
            self.active_sellers.append(seller_id)
            self.in_active_set.add(seller_id)
```

</div>

</div>
</details>

---

### 10. Timestamp Task Scheduler with Direct ID Removal

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 10</span>
  <span class="review-card-title">Timestamp Task Scheduler with Direct ID Removal</span>
  <span class="review-card-tag">Composite Min-Heap · Lazy Deletion · Hash Timestamp Verification</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>
