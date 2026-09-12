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

---

### 08. Reverse Linked List In-Place via Three Pointers

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">LIST 08</span>
  <span class="review-card-title">Reverse Linked List In-Place via Three Pointers</span>
  <span class="review-card-tag">Three-Pointer Iteration · Predecessor Shielding · In-Place · O(1) Space</span>
</summary>
<div class="review-card-content">

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

### 09. Remove Duplicate Letters via Monotonic Stack

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">STACK 09</span>
  <span class="review-card-title">Remove Duplicate Letters via Monotonic Stack</span>
  <span class="review-card-tag">Monotonic Increasing Stack · Last Seen Index Map · In-Stack Set · O(N)</span>
</summary>
<div class="review-card-content">

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

### 10. MinStack, MaxStack, Streaming Median & System Extensions

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">HEAP 10</span>
  <span class="review-card-title">MinStack, MaxStack, Streaming Median & System Extensions</span>
  <span class="review-card-tag">Two-Heap Dynamic Median · O(1) Extremum Stack · Lazy Deletion · Concurrency</span>
</summary>
<div class="review-card-content">

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

