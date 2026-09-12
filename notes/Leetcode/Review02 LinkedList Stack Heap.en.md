# Review Flashcards: Linked List, Stack & Heap

This note is the second volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Linked List & Hash-Linked Structures**, **Stack & Monotonic Deque**, and **Heap & Priority Queue** with production-grade implementations and rigorous complexity breakdowns.

---

## Module 1: Linked List & Hash-Linked Structures

### 1. LRU Cache & System-Level Extensions

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Linked List 01</span>
  <span class="review-card-title">LRU Cache & System-Level Extensions</span>
  <span class="review-card-tag">Doubly Linked List · Hash Map · TTL Expiration · LFU Replacement · 4-End Deque · Miss-Rate Tuning</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Master Extension Matrix</div>

Design and implement a data structure following the constraints of a **Least Recently Used (LRU)** cache, supporting strictly $O(1)$ time complexity for both `get(key)` and `put(key, value)` operations:

```python
class LRUCache:
    def __init__(self, capacity: int): ...
    def get(self, key: int) -> int: ...
    def put(self, key: int, value: int) -> None: ...
```

| Variant | Variant Topic | Core Variation | Architectural Strategy |
|---|---|---|---|
| **Follow-up 1** | **Add TTL (Time-To-Live)** | Expired entries return -1 and must not consume capacity | **Lazy Eviction** on access paired with **Active Sweeper / Min-Heap**. |
| **Follow-up 2** | **Add LFU (Frequency Eviction)** | Evict lowest usage frequency; ties broken by recency | **Two-Tier Hash Mapping** + track global scalar `min_freq`. |
| **Follow-up 3** | **4-End Deque with Indexing** | Support `lpush`/`rpush`/`lpop`/`rpop` + indexed access | **Chunked Deque (Quicklist)** or dynamic circular ring buffers. |
| **Follow-up 4** | **Iterate in LRU Order** | Traverse active entries from LRU to MRU | Sequential walk from sentinel `head.next` to `tail.prev`. |
| **Follow-up 5** | **High Cache-Miss-Rate Tuning** | Cache miss rate spiked (> 30%) | Working set re-sizing, **W-TinyLFU admission filter**, multi-tier caching (L1 local + L2 Redis), striped locking. |

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
import time
from typing import Dict, Optional, List, Tuple

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

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: Strictly $\mathcal{O}(1)$.
- **Space Complexity**: $\mathcal{O}(C)$.

</div>

</div>
</details>

---

### 2. Reverse Nodes in k-Group & Structural Group Inversion

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Linked List 02</span>
  <span class="review-card-title">Reverse Nodes in k-Group & Structural Group Inversion</span>
  <span class="review-card-tag">Dummy Sentinel · Local Pointer Reversal · Reverse Partial Tail · Invert Group Order · Scaffolding</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Variants</div>

```python
from typing import Optional, List, Tuple

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val, self.next = val, next

def build_list(values: List[int]) -> Optional[ListNode]:
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def to_list(head: Optional[ListNode]) -> List[int]:
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res

class KGroupReverser:
    @staticmethod
    def _reverse_segment(head: Optional[ListNode]) -> Tuple[Optional[ListNode], Optional[ListNode]]:
        prev, cur, tail = None, head, head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
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

    @classmethod
    def reverseGroupOrderNotWithin(cls, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or k <= 1: return head
        groups = []
        cur = head
        while cur:
            g_head = g_tail = cur
            count = 1
            while count < k and g_tail.next:
                g_tail = g_tail.next
                count += 1
            nxt = g_tail.next
            g_tail.next = None
            groups.append((g_head, g_tail))
            cur = nxt
        dummy = ListNode(0)
        tail = dummy
        for g_h, g_t in reversed(groups):
            tail.next = g_h
            tail = g_t
        return dummy.next
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(1)$ in-place; $\mathcal{O}(N/k)$ for group order inversion.

</div>

</div>
</details>

---

### 3. Flatten Multilevel Doubly Linked List with Empty-Node Filtering

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Linked List 03</span>
  <span class="review-card-title">Flatten Multilevel Doubly Linked List with Empty-Node Filtering</span>
  <span class="review-card-tag">Multilevel DLL · Depth-First Flattening · Empty Node Cleansing · Bidirectional Pointer Healing</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Core Variant Twist</div>

Given a doubly linked list where nodes have `prev`, `next`, and a `child` pointer to a separate sub-doubly-linked list:
- **Base Task (LC 430)**: Flatten the list such that all nodes appear in a single-level doubly linked list via preorder depth-first traversal.
- **Variant Twist**: Nested nodes carry empty values (`val is None` or empty placeholder).
  - **Requirement**: In the flattened result, **completely filter out and exclude all empty-valued nodes**, while preserving the entire structure and valid nodes reachable through them.
  - Splicing must strictly maintain bidirectional consistency (`prev` and `next` match symmetrically, and all `child` pointers are cleared to `None`).

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Two-Stage Decoupled Pipeline</div>

Attempting to filter empty nodes concurrently while splicing nested children introduces severe pointer dangling risks.
- **Two-Stage Architecture**:
  1. **Stage 1 (Canonical Flattening)**: Use a stack to hold pending `next` branches, elevating `child` chains into the main list and clearing `child = None`.
  2. **Stage 2 (In-Place Empty Node Purging)**: Perform a linear pass over the flattened single-level list:
     - For any node with `cur.val is None`: bypass it via `cur.prev.next = cur.next` and `cur.next.prev = cur.prev`.
- **Complexity**: Time strictly $\mathcal{O}(N)$, auxiliary stack space $\mathcal{O}(D)$ where $D$ is the maximum nesting depth.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import Optional

class MultiLevelNode:
    def __init__(self, val: Optional[int] = None, prev=None, next=None, child=None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

class MultiLevelListFlattenSolution:

    @classmethod
    def flattenAndFilterEmpty(cls, head: Optional[MultiLevelNode]) -> Optional[MultiLevelNode]:
        """
        Flatten multilevel DLL and purge empty-value nodes.
        Time: O(N), Space: O(D)
        """
        if not head:
            return None

        # Stage 1: DFS Flattening
        cur = head
        stack = []

        while cur:
            if cur.child:
                if cur.next:
                    stack.append(cur.next)
                cur.next = cur.child
                cur.child.prev = cur
                cur.child = None

            if not cur.next and stack:
                nxt = stack.pop()
                cur.next = nxt
                nxt.prev = cur

            cur = cur.next

        # Stage 2: In-place empty node removal
        dummy = MultiLevelNode(0, next=head)
        head.prev = dummy

        cur = head
        while cur:
            nxt = cur.next
            if cur.val is None:
                cur.prev.next = nxt
                if nxt:
                    nxt.prev = cur.prev
                cur.prev = cur.next = None
            cur = nxt

        new_head = dummy.next
        if new_head:
            new_head.prev = None
        return new_head
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: Strictly $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(D)$ stack frames ($D \le N$).
- **Critical Pitfalls**: Forgetting to update `child = None`; missing `nxt.prev = cur.prev` during deletion causing broken backward traversals.

</div>

</div>
</details>

---

## Module 2: Stack & Monotonic Deque

### 4. Basic Calculator & Operator Precedence Hierarchy

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Stack 04</span>
  <span class="review-card-title">Basic Calculator & Operator Precedence Hierarchy</span>
  <span class="review-card-tag">Single-Stack Accumulation · Parentheses State Stashing · Unary Minus Normalization · Right-Associative Exponent · Dijkstra Shunting-Yard</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Implementation</div>

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
        nums: List[int] = []
        ops: List[str] = []
        i, n = 0, len(s)
        expect_operand = True

        def evaluate_top_op():
            op = ops.pop()
            b = nums.pop()
            a = nums.pop()
            nums.append(cls._apply_op(op, b, a))

        while i < n:
            ch = s[i]
            if ch == ' ':
                i += 1; continue
            if ch.isdigit():
                val = 0
                while i < n and s[i].isdigit():
                    val = val * 10 + int(s[i])
                    i += 1
                nums.append(val)
                expect_operand = False
                continue
            if ch == '(':
                ops.append('(')
                expect_operand = True
                i += 1; continue
            if ch == ')':
                while ops and ops[-1] != '(': evaluate_top_op()
                ops.pop()
                expect_operand = False
                i += 1; continue
            if ch in cls.PRECEDENCE:
                if expect_operand:
                    if ch in ('+', '-'): nums.append(0)
                    else: raise ValueError(f"Syntax error: {ch}")
                cur_prec = cls.PRECEDENCE[ch]
                is_right = cls.IS_RIGHT_ASSOCIATIVE[ch]
                while ops and ops[-1] != '(':
                    top_prec = cls.PRECEDENCE.get(ops[-1], 0)
                    if (not is_right and top_prec >= cur_prec) or (is_right and top_prec > cur_prec):
                        evaluate_top_op()
                    else:
                        break
                ops.append(ch)
                expect_operand = True
                i += 1; continue
            raise ValueError(f"Invalid character: {ch}")

        while ops: evaluate_top_op()
        return nums[0] if nums else 0
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(N)$.

</div>

</div>
</details>

---

### 5. Sliding Window Maximum & Monotonic Deque Pattern

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Queue 05</span>
  <span class="review-card-title">Sliding Window Maximum & Monotonic Deque Pattern</span>
  <span class="review-card-tag">Monotonic Deque · Index Expiration Eviction · Amortized O(1) Per-Element Transition</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Canonical Scenarios</div>

Given an integer array `nums` and a sliding window of size `k` moving from left to right, return the maximum value for each window position (LC 239):

```python
def maxSlidingWindow(nums: List[int], k: int) -> List[int]: ...
```

Example: `nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3` $\implies$ `[3, 3, 5, 5, 6, 7]`.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Monotonic Deque Invariants</div>

- **The Dominance Rule**: When an element $nums[i]$ arrives, any preceding element in the window that is $\le nums[i]$ will **never become the maximum**, because it is both smaller and expires sooner.
- **Deque Invariants**:
  1. Stores **array indices**; values at indices are strictly monotonically decreasing.
  2. Pop tail while $nums[q[-1]] \le val$.
  3. Pop left if $q[0] \le i - k$ (out of sliding window).
  4. Max value is always at $nums[q[0]]$ taking $O(1)$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from collections import deque
from typing import List

class SlidingWindowMaxSolution:
    @staticmethod
    def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
        if not nums or k <= 0:
            return []

        q: deque[int] = deque()
        res: List[int] = []

        for i, val in enumerate(nums):
            while q and nums[q[-1]] <= val:
                q.pop()
            q.append(i)

            if q[0] <= i - k:
                q.popleft()

            if i >= k - 1:
                res.append(nums[q[0]])

        return res
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: Each index is pushed and popped at most once, strictly $\mathcal{O}(N)$.
- **Space Complexity**: Deque stores at most $k$ indices, strictly $\mathcal{O}(k)$.
- **Critical Pitfalls**: Storing values instead of indices in deque prevents expiring nodes.

</div>

</div>
</details>

---

## Module 3: Heap & Priority Queue

### 6. Find Median from Data Stream & K-Way Merge

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 06</span>
  <span class="review-card-title">Find Median from Data Stream & K-Way Merge</span>
  <span class="review-card-tag">Dual-Heap Balance · Invariant Maintenance · Lazy Deletion · K-Way Merge · Top-K Bucket Sort</span>
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

<div class="review-block">
<div class="review-block-label">⏱️ Complexity</div>

- Time: $\mathcal{O}(\log N)$ per add, $\mathcal{O}(1)$ median query. Space: $\mathcal{O}(N)$.

</div>

</div>
</details>

---

### 7. Tiered Priority Task Scheduler

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 07</span>
  <span class="review-card-title">Tiered Priority Task Scheduler</span>
  <span class="review-card-tag">Per-Seller Min-Heap · FIFO Arrival Sequence · Active Deque Round-Robin · Weighted Quota Dispatch</span>
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
        while self.current_seller is None:
            if not self.active_sellers: return None
            cand = self.active_sellers.popleft()
            self.in_active_set.remove(cand)
            if self.seller_heaps.get(cand):
                self.current_seller = cand
                tier = self.seller_tiers.get(cand, 'STANDARD')
                self.remaining_quota = 2 if tier == 'VIP' else 1
                break
        seller = self.current_seller
        _, _, task_id = heapq.heappop(self.seller_heaps[seller])
        self.remaining_quota -= 1
        has_more = len(self.seller_heaps[seller]) > 0
        quota_out = (self.remaining_quota <= 0)
        if not has_more:
            self.current_seller = None
            self.remaining_quota = 0
        elif quota_out:
            self.active_sellers.append(seller)
            self.in_active_set.add(seller)
            self.current_seller = None
            self.remaining_quota = 0
        return (seller, task_id)
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity</div>

- $\mathcal{O}(\log M)$ per task, $\mathcal{O}(T + S)$ space.

</div>

</div>
</details>

---

### 8. Timestamp Task Scheduler with Direct ID Removal

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 08</span>
  <span class="review-card-title">Timestamp Task Scheduler with Direct ID Removal</span>
  <span class="review-card-tag">Composite Min-Heap · Lazy Deletion · Hash Timestamp Verification · Destructive Pop</span>
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

<div class="review-block">
<div class="review-block-label">⏱️ Complexity</div>

- `addTask` $\mathcal{O}(\log N)$, `removeTask` $\mathcal{O}(1)$, `popTask` $\mathcal{O}(K \log N)$.

</div>

</div>
</details>
