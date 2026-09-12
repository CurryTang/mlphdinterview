# Review Flashcards: Linked List, Stack & Heap

This note is the second volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Linked List & Hash-Linked Structures**, **Stack & Expression Parsing**, and **Heap & Priority Queue** with production-grade implementations and rigorous complexity breakdowns.

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
    def get(self, key: int) -> int: ...   # returns -1 if non-existent or expired
    def put(self, key: int, value: int) -> None: ...
```

In systems architecture and distributed engineering interviews, this canonical problem universally branches into five key follow-up extensions:

| Variant | Variant Topic | Core Variation / System Requirement | Architectural Strategy & Resolution |
|---|---|---|---|
| **Follow-up 1** | **Add TTL (Time-To-Live)** | Entries have an expiration timestamp; `get` returns `-1` if expired, and expired keys must not indefinitely occupy memory capacity. | **Lazy Eviction**: Validate timestamp on access and unlink immediately; paired with **Active Sweeper / Min-Heap** to proactively reclaim cold expired keys. |
| **Follow-up 2** | **Add LFU (Frequency Eviction)** | Evict keys with the lowest usage frequency; ties are broken by recency (least recently used). | **Two-Tier Hash Mapping**: `key_node_map` + `freq_dll_map` + track global scalar `min_freq`, keeping all state transitions strictly $O(1)$. |
| **Follow-up 3** | **4-End Deque with Indexing** | Maintain $O(1)$ bounds while supporting `lpush`, `rpush`, `lpop`, `rpop`, plus indexed random access `get_by_index(i)`. | Pure DLL provides $O(1)$ for 4 ends but degrades to $O(N)$ for index; production solutions use **Chunked Deque (Quicklist)** or **Dynamic Circular Ring Buffers** to achieve $O(1)$ ends with amortized $O(1)$ indexing. |
| **Follow-up 4** | **Iterate in LRU Order (Print Path)** | Traverse all active cache entries in order from least-recently-used to most-recently-used. | Traverse DLL topology order: from sentinel `head.next` sequentially to sentinel `tail.prev`, yielding a clean generator iterator. |
| **Follow-up 5** | **High Cache-Miss-Rate Tuning** | Production alarms indicate cache miss rate spiked (> 30%). How do you systematically improve cache hit ratio? | Capacity re-sizing (working set analysis), **Eviction policy upgrades (W-TinyLFU admission / ARC)**, **Multi-tier caching (L1 local + L2 Redis)**, **Spatial locality prefetching**, and **Striped locking to eliminate concurrency bottlenecks**. |

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

#### 1. Canonical LRU: Doubly Linked List (DLL) + Hash Map Synergy
- **Why a Singly Linked List Fails**:
  - Unlinking a known node from a singly linked list requires scanning from the head to locate its predecessor `prev`, taking $O(N)$ time.
  - A **Doubly Linked List (DLL)** maintains explicit `prev` and `next` pointers on every node. Given a direct pointer, it unlinks in strictly $O(1)$ time:
    ```python
    node.prev.next = node.next
    node.next.prev = node.prev
    ```
- **Sentinel Dummy Head & Tail Invariant**:
  - Initialize pseudo-nodes `head` and `tail`, permanently connected: `head.next = tail`, `tail.prev = head`.
  - All valid data nodes reside strictly between `head` and `tail`. This invariant eliminates edge-case checks for empty lists or head/tail insertions.
- **Recency Convention**:
  - Node adjacent to `tail.prev` is designated the **Most Recently Used (MRU)**.
  - Node adjacent to `head.next` is designated the **Least Recently Used (LRU)**.
  - On `get(key)` or `put(key, val)`, lookup the node in $O(1)$ via hash map, unlink via `_remove(node)`, and splice onto the tail via `_append_to_tail(node)`. If capacity overflows, pop `head.next`.

#### 2. TTL (Time-To-Live) Reclamation Strategy
1. **Passive / Lazy Eviction**: Inside `get(key)`, check if `time.time() > node.expiry`. If expired, unlink from DLL, remove from hash map, and return `-1`.
2. **Proactive Sweeper**: A background min-heap or timing wheel periodically evicts dead entries to prevent cold memory leaks.

#### 3. System Architecture for High Cache Miss Rates
1. **Working Set Sizing**: Expand capacity if active working set footprint exceeds cache size.
2. **Admission Control (W-TinyLFU)**: Prevent one-hit wonders from flushing hot data using a Count-Min Sketch admission filter.
3. **Multi-Tier Caching**: L1 In-process memory + L2 Distributed Redis cluster.
4. **Spatial Locality & Prefetching**: Prefetch associated records on misses; pre-warm cache before traffic spikes.
5. **Striped Locking (Sharding)**: Partition cache into hash slots to eliminate mutex contention.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
import time
from typing import Dict, Optional, List, Tuple

class DLLNode:
    """Doubly Linked List node holding key, value, and TTL metadata."""
    __slots__ = ('key', 'val', 'prev', 'next', 'expiry')
    
    def __init__(self, key: int = 0, val: int = 0, expiry: float = float('inf')):
        self.key = key
        self.val = val
        self.expiry = expiry
        self.prev: Optional['DLLNode'] = None
        self.next: Optional['DLLNode'] = None


class LRUCacheWithTTL:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.map: Dict[int, DLLNode] = {}
        
        self.head = DLLNode()
        self.tail = DLLNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: DLLNode) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _append_to_tail(self, node: DLLNode) -> None:
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def _is_expired(self, node: DLLNode) -> bool:
        return time.time() > node.expiry

    def _evict_node(self, node: DLLNode) -> None:
        self._remove(node)
        self.map.pop(node.key, None)

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        node = self.map[key]
        if self._is_expired(node):
            self._evict_node(node)
            return -1
        self._remove(node)
        self._append_to_tail(node)
        return node.val

    def put(self, key: int, value: int, ttl: Optional[float] = None) -> None:
        expiry = time.time() + ttl if ttl is not None else float('inf')
        if key in self.map:
            node = self.map[key]
            node.val = value
            node.expiry = expiry
            self._remove(node)
            self._append_to_tail(node)
            return

        if len(self.map) >= self.capacity:
            lru_victim = self.head.next
            self._evict_node(lru_victim)

        new_node = DLLNode(key, value, expiry)
        self.map[key] = new_node
        self._append_to_tail(new_node)

    def get_lru_order(self) -> List[Tuple[int, int]]:
        result = []
        cur = self.head.next
        now = time.time()
        while cur != self.tail:
            next_node = cur.next
            if cur.expiry <= now:
                self._evict_node(cur)
            else:
                result.append((cur.key, cur.val))
            cur = next_node
        return result
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: `get` and `put` are strictly $\mathcal{O}(1)$.
- **Space Complexity**: Auxiliary space strictly $\mathcal{O}(C)$ where $C = \text{capacity}$.
- **Critical Pitfalls**: Forgetting to remove victim from hash map `self.map.pop(victim.key)`.

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
<div class="review-block-label">📌 Problem Definition & Master Variant Matrix</div>

Given the head of a linked list, reverse the nodes of the list $k$ at a time, and return the modified list:

```python
def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]: ...
```

In technical interviews (Lark / HackerRank / MLE screens), this canonical problem universally branches into five key follow-up extensions:

| Variant | Variant Name | Core Requirement | Structural Strategy |
|---|---|---|---|
| **Variant 1** | **Classic k-Group Reversal (LC 25)** | Nodes fewer than $k$ at the tail remain untouched. | **Lookahead Probe**: Advance $k$ steps; if group is complete, decouple, reverse, and splice; otherwise terminate. |
| **Variant 2** | **Reverse Partial Tail Group** | Non-standard prompt: even if fewer than $k$ nodes remain at the tail, **reverse them anyway**. | **Drop Lookahead Guard**: Unconditionally reverse whatever remains (up to $k$ nodes) until end of list. |
| **Variant 3** | **Two-Parter Warmup** | Round starts with reversing a plain single list, then generalizes into grouped reversal. | Implement clean helper `reverse_single_list(head)` and reuse as modular primitive. |
| **Variant 4** | **Custom ListNode Scaffolding** | Platform provides no prebuilt linked list helper; candidate must build `ListNode` and test drivers from scratch. | Implement `class ListNode`, plus `build_list(values)` and `to_list(head)` testing scaffold. |
| **Variant 5** | **Reverse Group Order, Not Within Groups** | Keep each $k$-group's internal order intact, but **reverse the sequence of the groups themselves** (e.g. for $k=3$, $1\to2\to3\to4\to5\to6$ becomes $4\to5\to6\to1\to2\to3$). | **Group Detachment + Reverse Re-linking**: Do NOT touch internal links; collect `(head, tail)` pairs for each group and re-link from back to front. |
| **Variant 6** | **Edge Probes** | $k=1$ is a no-op; $k > \text{length}$; empty lists. | When $k=1$, return `head` immediately. When $k > N$, Variant 1 returns untouched, Variant 2 reverses entire list. |

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

#### 1. Canonical k-Group: Four-Pointer Splicing Model
Anchor four critical pointers:
1. `group_prev`: Predecessor before the group (must link to new group head).
2. `group_start`: Initial head of the group (becomes new group tail).
3. `group_end`: Initial tail of the group (becomes new group head).
4. `next_group_head`: Successor head after the group (`group_end.next`).

#### 2. Reverse Group Order, Not Within Groups
- **Do NOT mutate internal pointers**:
  - Segment list into chunks of size $k$ by setting each group's `tail.next = None`.
  - Store group boundaries `[(head_1, tail_1), (head_2, tail_2), ...]`.
  - Traverse `groups` in reverse, chaining `tail_i.next = head_{i-1}`.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations (with Scaffolding)</div>

```python
from typing import Optional, List, Tuple

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

def build_list(values: List[int]) -> Optional[ListNode]:
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def to_list(head: Optional[ListNode]) -> List[int]:
    res = []
    cur = head
    while cur:
        res.append(cur.val)
        cur = cur.next
    return res


class KGroupReverser:

    @staticmethod
    def _reverse_single_list(head: Optional[ListNode]) -> Tuple[Optional[ListNode], Optional[ListNode]]:
        prev = None
        cur = head
        tail = head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev, tail

    @classmethod
    def reverseKGroup(cls, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """Variant 1: Classic (leave trailing < k untouched). O(N) time, O(1) space."""
        if not head or k <= 1:
            return head

        dummy = ListNode(0, head)
        group_prev = dummy

        while True:
            group_end = group_prev
            for _ in range(k):
                group_end = group_end.next
                if not group_end:
                    return dummy.next

            group_start = group_prev.next
            next_group_head = group_end.next

            group_end.next = None
            new_head, new_tail = cls._reverse_single_list(group_start)

            group_prev.next = new_head
            new_tail.next = next_group_head
            group_prev = new_tail

    @classmethod
    def reverseKGroupAll(cls, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """Variant 2: Reverse trailing partial group too."""
        if not head or k <= 1:
            return head

        dummy = ListNode(0, head)
        group_prev = dummy

        while group_prev.next:
            count = 0
            group_end = group_prev
            while count < k and group_end.next:
                group_end = group_end.next
                count += 1

            group_start = group_prev.next
            next_group_head = group_end.next

            group_end.next = None
            new_head, new_tail = cls._reverse_single_list(group_start)

            group_prev.next = new_head
            new_tail.next = next_group_head
            group_prev = new_tail

        return dummy.next

    @classmethod
    def reverseGroupOrderNotWithin(cls, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """Variant 5: Reverse sequence of groups while preserving intra-group order."""
        if not head or k <= 1:
            return head

        groups: List[Tuple[ListNode, ListNode]] = []
        cur = head

        while cur:
            g_head = cur
            g_tail = cur
            count = 1
            while count < k and g_tail.next:
                g_tail = g_tail.next
                count += 1
            
            nxt = g_tail.next
            g_tail.next = None
            groups.append((g_head, g_tail))
            cur = nxt

        if not groups:
            return None

        dummy = ListNode(0)
        curr_tail = dummy
        for g_head, g_tail in reversed(groups):
            curr_tail.next = g_head
            curr_tail = g_tail

        return dummy.next
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: Strictly $\mathcal{O}(N)$.
- **Space Complexity**: In-place reversal is $\mathcal{O}(1)$; group order reversal takes $\mathcal{O}(N/k)$ auxiliary list memory.
- **Critical Pitfalls**: Forgetting to set `group_end.next = None` before reversal causes circular references and hangs test runners.

</div>

</div>
</details>

---

## Module 2: Stack & Expression Parsing

### 3. Basic Calculator & Operator Precedence Hierarchy

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Stack 03</span>
  <span class="review-card-title">Basic Calculator & Operator Precedence Hierarchy</span>
  <span class="review-card-tag">Single-Stack Accumulation · Parentheses State Stashing · Unary Minus Normalization · Right-Associative Exponent · Dijkstra Shunting-Yard</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Master Variant Matrix</div>

Implement a calculator engine evaluating a mathematical expression string `s` containing non-negative integers, arithmetic operators, and nested parentheses:

```python
def calculate(s: str) -> int: ...
```

| Variant | Variant Name | Core Tokens / Features | Algorithmic Strategy |
|---|---|---|---|
| **Variant 1** | **No Parentheses Arithmetic (LC 227)** | Contains spaces, multi-digits, `+`, `-`, `*`, `/`, no parentheses | **Single-stack term accumulation**: `+`/`-` pushes values onto stack, `*`/`/` pops top to evaluate immediately, final sum. |
| **Variant 2** | **Parentheses Addition/Subtraction (LC 224)** | Only `+`, `-`, and nested `( )` | **Context stack**: on `(`, push outer `(res, sign)` and reset state; on `)`, pop context and combine. |
| **Variant 3** | **Full Infix with Parentheses (LC 772)** | Complete `+`, `-`, `*`, `/` and arbitrary parentheses | **Recursive sub-expression evaluation** or **Dijkstra Two-Stack Shunting-Yard Algorithm**. |
| **Variant 4** | **Unary Minus / Leading Signs** | Expressions like `-5`, `1 - (-2)`, `(-3 + 4)` | **State flag**: if preceding non-space token is `(` or string start, mark operator as unary and push synthetic `0` to operand stack. |
| **Variant 5** | **Right-Associative Exponentiation (`^`)** | Exponent `^` with higher precedence than `*`/`/`, and **right-associative** | $2\text{\textasciicircum}3\text{\textasciicircum}2 = 2^{(3^2)} = 512$; Shunting-Yard pops only when top operator has **strictly greater** precedence. |
| **Variant 6** | **Truncation toward Zero Division** | Arithmetic expressions with negative intermediate division | **Language semantics pitfall**: Python `//` floors towards $-\infty$ (`-3 // 2 == -2`), whereas C++/Java/LeetCode truncates toward zero (`int(-3 / 2) == -1`). |

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

- **Shunting-Yard Two-Stack Model**: Operand stack `nums`, Operator stack `ops`.
- Precedence: `+`, `-` (1); `*`, `/` (2); `^` (3).
- Left-associative ops pop while $\ge$ top precedence; right-associative `^` pops only when top is strictly $>$.
- Unary minus converted to $0 - x$ when encountering sign under `expect_operand == True`.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

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
                i += 1
                continue
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
                i += 1
                continue
            if ch == ')':
                while ops and ops[-1] != '(':
                    evaluate_top_op()
                ops.pop()
                expect_operand = False
                i += 1
                continue
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
                i += 1
                continue
            raise ValueError(f"Invalid character: {ch}")

        while ops:
            evaluate_top_op()
        return nums[0] if nums else 0
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(N)$.
- **Critical Pitfalls**: Inverted operand order on subtraction and division; using Python `//` on negative numbers.

</div>

</div>
</details>

---

## Module 3: Heap & Priority Queue

### 4. Find Median from Data Stream & K-Way Merge

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 04</span>
  <span class="review-card-title">Find Median from Data Stream & K-Way Merge</span>
  <span class="review-card-tag">Dual-Heap Balance · Invariant Maintenance · Lazy Deletion · K-Way Merge · Top-K Bucket Sort</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Canonical Scenarios</div>

Design a data structure receiving a real-time continuous stream of integers and dynamically querying the **median** of all observed elements:

```python
class MedianFinder:
    def __init__(self): ...
    def addNum(self, num: int) -> None: ...
    def findMedian(self) -> float: ...
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

- Balance dual heaps: Max-heap `lo` for lower half, Min-heap `hi` for upper half.
- Invariants: $\max(\text{lo}) \le \min(\text{hi})$ and $\operatorname{len}(\text{lo}) \in \{\operatorname{len}(\text{hi}), \;\operatorname{len}(\text{hi}) + 1\}$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

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
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: `addNum` takes $\mathcal{O}(\log N)$, `findMedian` takes $\mathcal{O}(1)$.
- **Space Complexity**: $\mathcal{O}(N)$.

</div>

</div>
</details>

---

### 5. Tiered Priority Task Scheduler

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 05</span>
  <span class="review-card-title">Tiered Priority Task Scheduler</span>
  <span class="review-card-tag">Per-Seller Min-Heap · FIFO Arrival Sequence · Active Deque Round-Robin · Weighted Quota Dispatch</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Scheduling Invariants</div>

Each task contains: `task_id` (str), `seller_id` (str), `tier` (`'VIP'` or `'STANDARD'`), and `priority` (int, 1 is high, 3 is low).

Implement:
```python
def receive_task(task: dict) -> None: ...
def process_next_task() -> Optional[Tuple[str, str]]: ...  # returns (seller_id, task_id)
```

**Constraints**:
1. Within each seller: smaller priority numbers processed first; equal priorities preserve arrival order (FIFO).
2. Rotate across sellers: VIP seller processes at most 2 consecutive tasks per turn; STANDARD seller processes at most 1 before the next seller gets a turn.
3. Early turn relinquishment if seller heap empties. Requeue active seller once turn ends; never duplicate an active seller in the deque.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

- Per-seller priority heap ordered by `(priority, arrival_sequence, task_id)` ensuring strict FIFO tie-breaking.
- Deque of active sellers `active_sellers` paired with `in_active_set` to prevent duplicate enqueuing.
- Turn state tracking: `current_seller` and `remaining_quota`.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

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
            if not self.active_sellers:
                return None
            candidate = self.active_sellers.popleft()
            self.in_active_set.remove(candidate)
            if self.seller_heaps.get(candidate):
                self.current_seller = candidate
                tier = self.seller_tiers.get(candidate, 'STANDARD')
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
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(\log M)$ per task push and pop ($M$ is queued tasks for that seller); $\mathcal{O}(1)$ deque rotations.
- **Space Complexity**: $\mathcal{O}(T + S)$ for $T$ total tasks and $S$ tracked sellers.
- **Critical Pitfalls**: Duplicate seller enqueuing without `in_active_set` checks causes quota corruption.

</div>

</div>
</details>

---

### 6. Timestamp Task Scheduler with Direct ID Removal

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Heap 06</span>
  <span class="review-card-title">Timestamp Task Scheduler with Direct ID Removal</span>
  <span class="review-card-tag">Composite Min-Heap · Lazy Deletion · Hash Timestamp Verification · Destructive Pop</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Contract</div>

Implement a scheduler container supporting ordered timestamp retrieval and direct cancellation:

```python
class TaskScheduler:
    def addTask(self, taskID: str, timestamp: int) -> None: ...
    def removeTask(self, taskID: str) -> bool: ...
    def popTask(self, num: int) -> List[str]: ...
```

- `addTask(taskID, timestamp)`: Insert task with its execution timestamp.
- `removeTask(taskID)`: Directly remove identified task if present, returning `True`.
- `popTask(num)`: **Destructively** retrieve up to `num` earliest tasks ordered by timestamp.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

- **Lazy Deletion Pattern**:
  - `removeTask(taskID)` simply deletes `taskID` from `task_map: Dict[str, int]` in $\mathcal{O}(1)$.
  - `popTask(num)` pops from min-heap and checks `task_map.get(taskID) == timestamp`. If stale/cancelled, silently drop it.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

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
        result: List[str] = []
        while self.heap and len(result) < num:
            timestamp, _, taskID = heapq.heappop(self.heap)
            if self.task_map.get(taskID) == timestamp:
                result.append(taskID)
                del self.task_map[taskID]
        return result
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: `addTask` $\mathcal{O}(\log N)$, `removeTask` $\mathcal{O}(1)$, `popTask` amortized $\mathcal{O}(K \log N)$.
- **Space Complexity**: $\mathcal{O}(N)$.
- **Critical Pitfalls**: Treating `popTask` as non-destructive and failing to delete from `task_map`.

</div>

</div>
</details>
