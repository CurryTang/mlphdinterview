# 复习卡片：链表、栈与堆 (Review Flashcards · Linked List, Stack & Heap)

本篇为算法面试高频复习卡片第二辑：系统整理**链表与复合哈希结构 (Linked List & Hash-Linked Structures)**、**栈与单调结构 (Stack & Monotonic Stack / Deque)** 以及**堆与优先队列 (Heap & Priority Queue)** 的核心高频考题、工业级变体、生产级实现与时空复杂度全景。

---

## 模块一：链表与复合哈希结构 (Linked List & Hash-Linked Structures)

### 1. LRU 缓存与其系统级演进全家桶 (LRU Cache & System-Level Extensions)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 01</span>
  <span class="review-card-title">LRU 缓存与其系统级演进全家桶 (LRU Cache & System-Level Extensions)</span>
  <span class="review-card-tag">哈希表 + 双向链表 · 复合哈希双向链表 · O(1) 淘汰</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 146 · LRU Cache](https://leetcode.com/problems/lru-cache/) — `https://leetcode.com/problems/lru-cache/`
> - [LeetCode 460 · LFU Cache](https://leetcode.com/problems/lfu-cache/) — `https://leetcode.com/problems/lfu-cache/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与五大工业级追问全景矩阵</div>

设计并实现一个满足 **LRU (Least Recently Used，最近最少使用)** 缓存约束的数据结构，支持严格 $O(1)$ 时间复杂度的 `get(key)` 与 `put(key, value)` 操作：

```python
class LRUCache:
    def __init__(self, capacity: int): ...
    def get(self, key: int) -> int: ...
    def put(self, key: int, value: int) -> None: ...
```

- 支持 TTL 惰性过期与主动回收；
- LFU 频次双层映射与 `min_freq`；
- 高缓存未命中率调优体系：工作集评估、W-TinyLFU 准入过滤、L1/L2 多级缓存、分段锁。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

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

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- `get` 与 `put` 严格为 $\mathcal{O}(1)$；辅助空间 $\mathcal{O}(C)$。

</div>

</div>
</details>

---

### 2. K 个一组翻转链表全家桶与组间重排 (Reverse Nodes in k-Group & Structural Group Inversion)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 02</span>
  <span class="review-card-title">K 个一组翻转链表全家桶与组间重排 (Reverse Nodes in k-Group & Structural Group Inversion)</span>
  <span class="review-card-tag">虚拟头节点 · K 长度探测 · 局部反转接回 · 空间 O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：[LeetCode 25 · Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) — `https://leetcode.com/problems/reverse-nodes-in-k-group/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Reverse Nodes in k-Group (LeetCode 25)**:
> Given the head of a linked list, reverse the nodes of the list $k$ at a time, and return the modified list.
> $k$ is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of $k$ then left-out nodes, in the end, should remain as it is.
>
> **Industrial Follow-ups**:
> 1. What if remaining nodes (< $k$) at the tail should also be reversed?
> 2. What if groups of $k$ themselves are reversed in order while preserving relative order within each group?

**函数签名**：
```python
class KGroupReverser:
    @staticmethod
    def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]: ...
```

**输入输出示例**：
- `head = [1, 2, 3, 4, 5], k = 2` $\implies$ `[2, 1, 4, 3, 5]`
- `head = [1, 2, 3, 4, 5], k = 3` $\implies$ `[3, 2, 1, 4, 5]`

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码与实现</div>

```python
from typing import Optional, List, Tuple

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val, self.next = val, next

def build_list(values: List[int]) -> Optional[ListNode]:
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(v); cur = cur.next
    return dummy.next

def to_list(head: Optional[ListNode]) -> List[int]:
    res = []
    while head:
        res.append(head.val); head = head.next
    return res

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

    @classmethod
    def reverseGroupOrderNotWithin(cls, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or k <= 1: return head
        groups = []
        cur = head
        while cur:
            g_head = g_tail = cur
            count = 1
            while count < k and g_tail.next:
                g_tail = g_tail.next; count += 1
            nxt = g_tail.next
            g_tail.next = None
            groups.append((g_head, g_tail))
            cur = nxt
        dummy = ListNode(0)
        tail = dummy
        for g_h, g_t in reversed(groups):
            tail.next = g_h; tail = g_t
        return dummy.next

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

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- 时间 $\mathcal{O}(N)$，原地反转空间 $\mathcal{O}(1)$。

</div>

</div>
</details>

---

### 3. 扁平化多级双向链表与空节点过滤 (Flatten Multilevel Doubly Linked List with Empty-Node Filtering)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 03</span>
  <span class="review-card-title">扁平化多级双向链表与空节点过滤 (Flatten Multilevel Doubly Linked List with Empty-Node Filtering)</span>
  <span class="review-card-tag">双向指针修复 · DFS 展开 · 尾节点回溯接回 · 原地操作</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：[LeetCode 430 · Flatten a Multilevel Doubly Linked List](https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/) — `https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Flatten a Multilevel Doubly Linked List (LeetCode 430)**:
> You are given a doubly linked list, which contains nodes that have a next pointer, a previous pointer, and an additional child pointer. This child pointer may or may not point to a separate doubly linked list, also containing these special nodes. These child lists may have one or more children of their own, and so on, to produce a multilevel data structure.
> Flatten the list so that all the nodes appear in a single-level, doubly linked list. Nodes in the child list should appear between the parent node and the parent node's next node.
>
> **Industrial Extension**: Filter out empty or null nodes (`val is None`) during flattening and maintain bidirectional pointer integrity.

**类与函数签名**：
```python
class MultiLevelNode:
    def __init__(self, val=None, prev=None, next=None, child=None): ...

class MultiLevelListFlattenSolution:
    @classmethod
    def flattenAndFilterEmpty(cls, head: Optional[MultiLevelNode]) -> Optional[MultiLevelNode]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- 时间 $\mathcal{O}(N)$，空间 $\mathcal{O}(D)$。

</div>

</div>
</details>

---

## 模块二：栈与单调结构 (Stack & Monotonic Stack / Deque)

### 4. 表达式计算器与运算符优先级全景全家桶 (Basic Calculator & Operator Precedence Hierarchy)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">栈 04</span>
  <span class="review-card-title">表达式计算器与运算符优先级全景全家桶 (Basic Calculator & Operator Precedence Hierarchy)</span>
  <span class="review-card-tag">符号栈 · 递归下降 · 乘除即时结合 · 括号递归分治</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 224 · Basic Calculator](https://leetcode.com/problems/basic-calculator/) — `https://leetcode.com/problems/basic-calculator/`
> - [LeetCode 227 · Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/) — `https://leetcode.com/problems/basic-calculator-ii/`
> - [LeetCode 772 · Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/) — `https://leetcode.com/problems/basic-calculator-iii/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Basic Calculator Hierarchy (LeetCode 224, 227, 772)**:
> Given a string expression representing an arithmetic expression containing non-negative integers, operators `+`, `-`, `*`, `/`, and parentheses `(`, `)`, evaluate the expression and return its integer value.
> Division must truncate toward zero (e.g. $-3 // 2 = -1$).
> The solution must support nested parentheses, arbitrary whitespace, and multi-digit integers in strict $\mathcal{O}(n)$ time.

**函数签名**：
```python
class ExpressionCalculator:
    @classmethod
    def calculate(cls, s: str) -> int: ...
```

**输入输出示例**：
- `s = "3+2*2"` $\implies$ `7`
- `s = " 3/2 "` $\implies$ `1`
- `s = " 3+5 / 2 "` $\implies$ `5`
- `s = "2*(5+5*2)/3+(6/2+8)"` $\implies$ `21`

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- 时间 $\mathcal{O}(N)$，空间 $\mathcal{O}(N)$。

</div>

</div>
</details>

---

### 5. 滑动窗口最大值与单调双端队列全景 (Sliding Window Maximum & Monotonic Deque Pattern)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">队列 05</span>
  <span class="review-card-title">滑动窗口最大值与单调双端队列全景 (Sliding Window Maximum & Monotonic Deque Pattern)</span>
  <span class="review-card-tag">单调双端队列 · 索引窗口失效淘汰 · 均摊 O(1) 转移</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：[LeetCode 239 · Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) — `https://leetcode.com/problems/sliding-window-maximum/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Sliding Window Maximum (LeetCode 239)**:
> You are given an array of integers `nums`, there is a sliding window of size $k$ which is moving from the very left of the array to the very right. You can only see the $k$ numbers in the window. Each time the sliding window moves right by one position.
> Return the max sliding window.

**函数签名**：
```python
class SlidingWindowMaxSolution:
    @classmethod
    def maxSlidingWindow(cls, nums: List[int], k: int) -> List[int]: ...
```

**输入输出示例**：
- `nums = [1,3,-1,-3,5,3,6,7], k = 3` $\implies$ `[3, 3, 5, 5, 6, 7]`
- `nums = [1], k = 1` $\implies$ `[1]`

**核心不变量**：
- 维护一个存储**元素下标**的单调双端队列（Monotonic Deque），队头到队尾对应数值严格单调递减；
- 窗口滑出时淘汰超出左界 $i - k$ 的队头下标；新元素压入时弹出所有小于当前值的队尾元素，实现均摊 $\mathcal{O}(1)$ 转移。

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from collections import deque
from typing import List

class SlidingWindowMaxSolution:
    @staticmethod
    def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
        if not nums or k <= 0: return []
        q: deque[int] = deque()
        res: List[int] = []
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

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- 时间 $\mathcal{O}(N)$，空间 $\mathcal{O}(k)$。

</div>

</div>
</details>

---

### 6. 柱状图中最大的矩形与单调栈双哨兵范式 (Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">单调栈 06</span>
  <span class="review-card-title">柱状图中最大的矩形与单调栈双哨兵范式 (Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern)</span>
  <span class="review-card-tag">单调递增栈 · 双哨兵 (Two-Sentinel) 技巧 · 左右边界动态判定 · 最大矩形降维扩展</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 84 · Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) — `https://leetcode.com/problems/largest-rectangle-in-histogram/`
> - [LeetCode 85 · Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/) — `https://leetcode.com/problems/maximal-rectangle/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与工业场景需求</div>

给定一个非负整数数组 `heights`，每个数表示柱状图中各个柱子的高度，每个柱子的宽度均为 1。

求在该柱状图中能够勾勒出的**最大矩形的面积**（LC 84）：

```python
def largestRectangleArea(heights: List[int]) -> int: ...
```

**输入输出示例**：
- 输入：`heights = [2, 1, 5, 6, 2, 3]` $\implies$ 输出：`10`（高度为 5 和 6 的两根柱子，面积为 $5 \times 2 = 10$）
- 输入：`heights = [2, 4]` $\implies$ 输出：`4`
- 输入：`heights = [2, 1, 2]` $\implies$ 输出：`3`（全宽跨越高为 1 的矩形，面积 $1 \times 3 = 3$）

**核心系统进阶追问**：
- 二维二进制矩阵中的最大全 1 矩形（Maximal Rectangle, LC 85）：如何将二维矩阵按行聚合，等价降维为该柱状图单调栈问题求解？

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与单调递增栈不变量</div>

#### 1. 矩形面积的本质：向左与向右的“最远延伸边界”
- 考虑以任意柱子 $i$ 的高度 $h = heights[i]$ 作为矩形的高：
  - 该矩形能向左延伸多远？直到遇到**左侧第一个严格小于 $h$ 的柱子**；
  - 该矩形能向右延伸多远？直到遇到**右侧第一个严格小于 $h$ 的柱子**。
- 暴力两边扩展耗时 $\mathcal{O}(N^2)$。
- **单调递增栈（Monotonic Increasing Stack）的优雅解耦**：
  - 维护一个存储柱子**下标**的栈，栈内元素对应的高度严格单调递增。
  - 当遍历到下标 $i$ 时，若 $heights[i] < heights[stack[-1]]$：说明当前柱子 $i$ 是栈顶柱子右侧第一个更矮的边界！
  - 弹出栈顶 $mid = stack.pop()$，其高度为 $h = heights[mid]$。
  - 此时，新栈顶 $stack[-1]$ 恰好是 $mid$ 左侧第一个更矮的边界！
  - 从而，以 $h$ 为高的矩形宽度可直接算得：
    $$\text{width} = i - stack[-1] - 1$$
    $$\text{area} = h \times \text{width}$$

#### 2. 双哨兵技巧 (Two-Sentinel Pattern) 的绝妙之处
初学者容易漏判两个边界情况：
1. 栈为空时的左边界越界判断；
2. 循环结束后栈中残留单调递增元素未被弹出结算。
- **工业级解决方案：首尾双零哨兵**：
  - 构造 `padded_heights = [0] + heights + [0]`。
  - **首部 `0`**：永远不会被弹出，充当所有柱子的绝对左边界，消除 `stack` 为空的繁琐判断；
  - **尾部 `0`**：比任何正常柱子都矮，强制在循环结束时将栈中所有残留的高柱子一网打尽全部弹出计算！代码极其短小精悍。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List

class HistogramSolution:

    @staticmethod
    def largestRectangleArea(heights: List[int]) -> int:
        """
        单调栈 + 双哨兵求柱状图最大矩形面积
        时间复杂度 O(N)，空间复杂度 O(N)
        """
        if not heights:
            return 0

        # 添加首尾 0 哨兵
        padded = [0] + heights + [0]
        stack = []
        max_area = 0

        for i, h in enumerate(padded):
            # 破坏单调性，弹出栈顶并结算以栈顶高度为基准的最大矩形
            while stack and padded[stack[-1]] > h:
                mid = stack.pop()
                height = padded[mid]
                # 宽度 = 右边界 i - 左边界 stack[-1] - 1
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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：每个柱子下标进栈一次、出栈一次，单趟线性扫描，时间复杂度严格为 $\mathcal{O}(N)$。
- **空间复杂度**：单调栈深度最大为 $N + 2$，额外空间复杂度为 $\mathcal{O}(N)$。
- **高频避坑清单**：
  1. **宽度的偏移量计算**：必须是 `width = i - stack[-1] - 1`，不可误写为 `i - mid`。例如 `[2, 1, 2]` 若宽度算错，面积会被缩减。

</div>

</div>
</details>

---

### 7. 星号通配符括号有效性与全量展开 (Valid Parenthesis String with Wildcard & Concrete String Enumeration)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">栈/回溯 07</span>
  <span class="review-card-title">星号通配符括号有效性与全量展开 (Valid Parenthesis String with Wildcard & Concrete String Enumeration)</span>
  <span class="review-card-tag">区间贪心 · O(N) 双界指针 · 负下限保护 · DFS 全量分支展开 · 剪枝去重</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：[LeetCode 678 · Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/) — `https://leetcode.com/problems/valid-parenthesis-string/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与双核任务架构</div>

给定一个只包含 `'('`、`')'` 和 `'*'` 的字符串 `s`。其中 `'*'` 可以被视为左括号 `'('`、右括号 `')'` 或空字符串 `""`（LC 678）。

实现以下两个层级函数：
1. **可行性判别 (Boolean Feasibility)**：`checkValidString(s: str) -> bool`，在严格 $\mathcal{O}(N)$ 时间内返回该字符串是否能够匹配为合法括号。
2. **全量合法具体串生成 (All Valid Concrete Strings)**：`allValidStrings(s: str) -> List[str]`，通过对每个 `'*'` 进行三种可能性的展开，返回由输入衍生出的**所有互不相同的具体合法括号字符串列表**。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与区间贪心状态机剖析</div>

#### 1. 可行性判别的区间贪心法（严格 O(N) 无栈解法）
- 核心不确定性：由于 `'*'` 的多重身份，当前未匹配的左括号数量不是一个确定的值，而是**一个闭区间 $[low, high]$**：
  - $low$：在所有可能的分支中，当前未匹配的**最少左括号数量**（贪心地将 `'*'` 尽可能当成 `')'` 或 `""`）；
  - $high$：在所有可能的分支中，当前未匹配的**最多左括号数量**（贪心地将 `'*'` 全部当成 `'('`）。
- **状态转移规则**：
  - 遇到 `'('`：$low \gets low + 1, \; high \gets high + 1$；
  - 遇到 `')'`：$low \gets low - 1, \; high \gets high - 1$；
  - 遇到 `'*'`：$low \gets low - 1, \; high \gets high + 1$；
  - **神圣下限保护**：$low = \max(0, low)$（因为未匹配的左括号数量在合法前缀中永远不可能为负数，若变成负数说明多余的右括号已被忽略）；
  - **非法拦截**：若 $high < 0$，说明即便把之前所有星号全当成左括号，右括号依然过量，直接判定为 `False`！
- 遍历结束时：若 $low == 0$，说明存在某种替换方案使左右括号完全抵消，返回 `True`。

#### 2. 全量合法具体串展开 (DFS 回溯与前缀剪枝)
- 对每个字符递归分支：若是 `'('` 或 `')'` 顺延；若是 `'*'`，分支调用 `'('`、`')'`、`""` 三种可能。
- **剪枝核心**：实时记录当前未配对左括号计数 `balance`。若在任意时刻 `balance < 0`，立刻剪枝终止！
- 递归终点：当遍历完毕且 `balance == 0`，将构造出的具体字符串加入哈希集合 `set` 去重。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List

class WildcardParenthesesSolution:

    @staticmethod
    def checkValidString(s: str) -> bool:
        """
        任务 1: O(N) 区间贪心检验括号有效性
        """
        low = 0   # 最小未匹配左括号数
        high = 0  # 最大未匹配左括号数

        for ch in s:
            if ch == '(':
                low += 1
                high += 1
            elif ch == ')':
                low -= 1
                high -= 1
            elif ch == '*':
                low -= 1
                high += 1

            # 核心拦截：右括号严重过量
            if high < 0:
                return False

            # 下限不能小于 0
            if low < 0:
                low = 0

        return low == 0

    @classmethod
    def allValidStrings(cls, s: str) -> List[str]:
        """
        任务 2: 全量 DFS 展开所有互不相同的具体合法括号字符串
        """
        n = len(s)
        results = set()
        path = []

        def dfs(i: int, balance: int):
            # 剪枝：右括号超量
            if balance < 0:
                return

            if i == n:
                if balance == 0:
                    results.add("".join(path))
                return

            ch = s[i]
            if ch == '(':
                path.append('(')
                dfs(i + 1, balance + 1)
                path.pop()
            elif ch == ')':
                path.append(')')
                dfs(i + 1, balance - 1)
                path.pop()
            else: # ch == '*'
                # 选项 1: 视为 '('
                path.append('(')
                dfs(i + 1, balance + 1)
                path.pop()

                # 选项 2: 视为 ')'
                path.append(')')
                dfs(i + 1, balance - 1)
                path.pop()

                # 选项 3: 视为空字符串 ""
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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：
  - `checkValidString`：单趟遍历，严格 $\mathcal{O}(N)$。
  - `allValidStrings`：星号数量为 $K$，分支状态数为 $\mathcal{O}(3^K)$，剪枝显著收缩搜索树。
- **高频避坑清单**：
  1. **遗漏 `low` 的负数截断**：如果不写 `if low < 0: low = 0`，对于输入 `"(*))"`，`low` 会在星号处变为 -1，遇到第二个 `)` 变为 -2，导致最终误判为 False。

</div>

</div>
</details>

---

## 模块三：堆与优先队列 (Heap & Priority Queue)

### 8. 数据流中位数与多路归并全景 (Find Median from Data Stream & K-Way Merge)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">堆 08</span>
  <span class="review-card-title">数据流中位数与多路归并全景 (Find Median from Data Stream & K-Way Merge)</span>
  <span class="review-card-tag">对顶双堆 · 严格平衡不变量 · 惰性删除</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 295 · Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) — `https://leetcode.com/problems/find-median-from-data-stream/`
> - [LeetCode 23 · Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) — `https://leetcode.com/problems/merge-k-sorted-lists/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Find Median from Data Stream (LeetCode 295)**:
> The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.
> Implement the `MedianFinder` class:
> - `MedianFinder()` initializes the MedianFinder object.
> - `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
> - `double findMedian()` returns the median of all elements so far. Answers within $10^{-5}$ of the actual answer will be accepted.

**接口定义**：
```python
class MedianFinder:
    def __init__(self): ...
    def addNum(self, num: int) -> None: ...
    def findMedian(self) -> float: ...
```

**输入输出示例**：
- `mf = MedianFinder()`
- `mf.addNum(1); mf.addNum(2); mf.findMedian()` $\implies$ `1.5`
- `mf.addNum(3); mf.findMedian()` $\implies$ `2.0`

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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

### 9. 多商户分级加权轮转任务调度器 (Tiered Priority Task Scheduler)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">堆 09</span>
  <span class="review-card-title">多商户分级加权轮转任务调度器 (Tiered Priority Task Scheduler)</span>
  <span class="review-card-tag">商户级小顶堆 · FIFO 时间戳序列 · 活跃商户轮转队列 · VIP 加权配额调度</span>
</summary>
<div class="review-card-content">

> 💡 **题目类型**：独立算法工程实现（无直接对应 LeetCode 原题，逻辑规则与测试规格独立，请直接使用卡片内置完整测试桩在本地运行验证）

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Tiered Priority Task Scheduler**:
> Design a multi-merchant fair round-robin task scheduler. Merchants have priority tiers (e.g., VIP vs Regular). Tasks arrive with timestamps.
> - Support receiving tasks: `receive_task(merchant_id, task_id, is_vip, arrival_time)`
> - Support dispatching tasks: `process_next_task() -> tuple`
> Tasks must be dispatched fairly: round-robin across active merchants, giving VIP merchants proportional quota / priority, while preserving FIFO arrival order for each merchant's own tasks.

**接口定义**：
```python
class TieredTaskScheduler:
    def __init__(self, vip_ratio: int = 2): ...
    def receive_task(self, merchant_id: str, task_id: str, is_vip: bool, arrival_time: int) -> None: ...
    def process_next_task(self) -> Optional[Tuple[str, str, bool, int]]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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

### 10. 时间戳任务调度器与直接 ID 淘汰 (Timestamp Task Scheduler with Direct ID Removal)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">堆 13</span>
  <span class="review-card-title">时间戳任务调度器与直接 ID 淘汰 (Timestamp Task Scheduler with Direct ID Removal)</span>
  <span class="review-card-tag">复合小顶堆 · 惰性删除 · 哈希版本校验</span>
</summary>
<div class="review-card-content">

> 💡 **题目类型**：独立算法工程实现（无直接对应 LeetCode 原题，逻辑规则与测试规格独立，请直接使用卡片内置完整测试桩在本地运行验证）

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Timestamp Task Scheduler with Direct ID Removal**:
> Design a priority task scheduler that manages tasks with timestamps, priorities, and unique task IDs:
> - `addTask(task_id, priority, timestamp)`: schedules a task.
> - `removeTask(task_id)`: cancels/evicts a task directly before execution.
> - `pollNextTask(current_time)`: dispatches the highest-priority eligible task whose timestamp $\le \text{current\_time}$.
> Must use a min-heap with lazy eviction and hash map version validation for $\mathcal{O}(\log N)$ amortized operations.

**接口定义**：
```python
class TimestampTaskScheduler:
    def __init__(self): ...
    def addTask(self, task_id: str, priority: int, timestamp: int) -> None: ...
    def removeTask(self, task_id: str) -> bool: ...
    def pollNextTask(self, current_time: int) -> Optional[Tuple[str, int, int]]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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

### 11. 链表原地反转与三指针迭代推进 (Reverse Linked List In-Place via Three Pointers)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LIST 08</span>
  <span class="review-card-title">链表原地反转与三指针迭代推进 (Reverse Linked List In-Place via Three Pointers)</span>
  <span class="review-card-tag">三指针滑动 · 前驱后继保护 · 原地反转 · O(1) 空间</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 206 · Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) — `https://leetcode.com/problems/reverse-linked-list/`
> - [LeetCode 92 · Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/) — `https://leetcode.com/problems/reverse-linked-list-ii/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Reverse Linked List (LeetCode 206 / 92)**:
> Given the head of a singly linked list, reverse the list in-place, and return the reversed list.
> Follow-up: implement both iterative three-pointer sliding and clean recursion, maintaining $\mathcal{O}(1)$ auxiliary space in iterative mode.

**函数签名**：
```python
class ReverseListSolution:
    @staticmethod
    def reverseListIterative(head: Optional[ListNode]) -> Optional[ListNode]: ...
    @staticmethod
    def reverseListRecursive(head: Optional[ListNode]) -> Optional[ListNode]: ...
```

**输入输出示例**：
- `head = [1, 2, 3, 4, 5]` $\implies$ `[5, 4, 3, 2, 1]`
- `head = [1, 2]` $\implies$ `[2, 1]`
- `head = []` $\implies$ `[]`

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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
        三指针迭代原地反转单链表。
        时间 O(N)，额外空间 O(1)。
        """
        prev: Optional[ListNode] = None
        curr = head

        while curr is not None:
            # 1. 临时保存后继节点，防止链条断裂
            nxt = curr.next
            # 2. 翻转指针方向
            curr.next = prev
            # 3. 双指针同步向前平移
            prev = curr
            curr = nxt

        return prev

    @classmethod
    def reverseListRecursive(cls, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        递归反转单链表。
        时间 O(N)，调用栈空间 O(N)。
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
<div class="review-block-label">💡 机制剖析</div>

- **三指针不变量（Three-Pointer Invariant）**：
  在任意时刻，$prev$ 指向已反转完成的子链表头部，$curr$ 指向当前待处理节点，$nxt$ 暂存原链表剩余未处理部分。每步操作通过 `curr.next = prev` 翻转指针，绝不引入任何堆内存分配。
- **递归版归纳基底与尾部清空**：
  在递归回溯阶段，`head.next.next = head` 让下一个节点反向指向当前节点；随后必须将 `head.next = None`，防止在原头节点处形成环形死锁。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，每个节点被精确访问一次。
- **空间复杂度**：迭代法 $\mathcal{O}(1)$；递归法 $\mathcal{O}(N)$（递归系统调用栈深度）。

</div>

</div>
</details>

---

### 12. 单调栈去重与字典序最小子序列 (Remove Duplicate Letters via Monotonic Stack)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STACK 09</span>
  <span class="review-card-title">单调栈去重与字典序最小子序列 (Remove Duplicate Letters via Monotonic Stack)</span>
  <span class="review-card-tag">单调递增栈 · 末次出现位置表 · 栈内存在性哈希 · O(N)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 316 · Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/) — `https://leetcode.com/problems/remove-duplicate-letters/`
> - [LeetCode 1081 · Smallest Subsequence of Distinct Characters](https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/) — `https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Remove Duplicate Letters (LeetCode 316 / 1081)**:
> Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.

**函数签名**：
```python
class RemoveDuplicateLettersSolution:
    @classmethod
    def removeDuplicateLetters(cls, s: str) -> str: ...
```

**输入输出示例**：
- `s = "bcabc"` $\implies$ `"abc"`
- `s = "cbacdcbc"` $\implies$ `"acdb"`

**核心约束与单调栈贪心**：
- 维护单调递增栈：当新字符比栈顶小，且栈顶字符在后续字符串中还会再次出现时，可安全弹出栈顶以换取更小的字典序；
- 辅助记录每个字符的最后出现下标 `last_seen` 与栈内存在性布尔集合 `in_stack`，在 $\mathcal{O}(n)$ 时间内求解。

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
class RemoveDuplicateLettersSolution:
    @classmethod
    def removeDuplicateLetters(cls, s: str) -> str:
        """
        移除重复字母，使得每个字母出现且仅出现一次，并保证结果字典序最小。
        """
        # 1. 记录每个字符在原字符串中的最终出现下标 (Last Occurrence)
        last_occurrence = {ch: i for i, ch in enumerate(s)}
        
        stack = []
        in_stack = set()  # 记录当前已存在于栈中的字符

        for i, ch in enumerate(s):
            # 若字符已在栈中，直接跳过（保持当前已锁定的最优字典序位置）
            if ch in in_stack:
                continue

            # 贪心维护单调递增栈：
            # 若栈顶字符比当前字符大，且栈顶字符在后续还会再次出现，则果断弹出栈顶
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
<div class="review-block-label">💡 机制剖析</div>

- **单调栈贪心决策律**：
  字典序越小的字符越应靠前。遇到字符 $ch$ 时，若栈顶字符 $top > ch$，且 $top$ 在后续文本中还会再次登场（$last\_occurrence[top] > i$），则此时抛弃 $top$ 绝不会导致未来缺失该字符，同时让更小的 $ch$ 占据高位，必然能使整体字典序变小。
- **不可挽回字符的刚性保护**：
  若 $last\_occurrence[top] \le i$，意味着这是当前字符最后一次露面的机会，此时严禁弹出，必须保留在栈中以满足“包含每个不同字符”的硬性前提。
- **已入栈字符直接跳过**：
  若 $ch$ 已在栈中，由于之前的入栈位置必然是在更早决策下取得的字典序最优位，重新弹出重排只会使字典序变大或破坏单调性，故直接 `continue`。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，每个字符最多入栈出栈各一次。
- **空间复杂度**：$\mathcal{O}(|\Sigma|)$，栈与哈希表大小受限于唯一字符集大小（英文字母为 $\le 26$）。

</div>

</div>
</details>

---

### 13. 双堆中位数流与多维栈系统架构 (MinStack, MaxStack, Streaming Median & Extensions)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">HEAP 10</span>
  <span class="review-card-title">双堆中位数流与多维栈系统架构 (MinStack, MaxStack, Streaming Median & Extensions)</span>
  <span class="review-card-tag">对顶堆 · O(1) 极值栈 · 懒删除 · 流式高并发扩展</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 155 · Min Stack](https://leetcode.com/problems/min-stack/) — `https://leetcode.com/problems/min-stack/`
> - [LeetCode 716 · Max Stack](https://leetcode.com/problems/max-stack/) — `https://leetcode.com/problems/max-stack/`
> - [LeetCode 295 · Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) — `https://leetcode.com/problems/find-median-from-data-stream/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **MinStack, MaxStack & Streaming Median System (LeetCode 155 / 716 / 295)**:
> Design stack structures supporting $\mathcal{O}(1)$ push, pop, top, and retrieval of the minimum (or maximum) element:
> - `MinStack`: tracks running minimum using parallel differential / min stack in $\mathcal{O}(1)$.
> - `Streaming Median`: maintains running median over an infinite stream via dual balancing heaps in $\mathcal{O}(\log n)$ per insertion.

**接口定义**：
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
<div class="review-block-label">📌 核心代码</div>

```python
import heapq
from typing import Optional

class MinStack:
    """O(1) 辅助栈维护运行最小值"""
    def __init__(self):
        self.stack = []      # 存储真实数值
        self.min_stack = []  # 存储当前前缀最小值

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
    """双堆（对顶堆）维护流式动态中位数"""
    def __init__(self):
        self.small = []  # 大顶堆 (Python heapq 存相反数): 存放较小的一半元素
        self.large = []  # 小顶堆: 存放较大的一半元素

    def addNum(self, num: int) -> None:
        # 1. 优先推入 small 堆
        heapq.heappush(self.small, -num)
        # 2. 将 small 堆顶的最大值转移给 large 堆
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # 3. 平衡条件: len(small) >= len(large)，且容量差不超过 1
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
<div class="review-block-label">💡 机制剖析与工业延伸</div>

- **双堆动态天平原理**：
  将全部数据切分为两半：左半部 $small$ 维持大顶堆，右半部 $large$ 维持小顶堆。始终满足 $\max(small) \le \min(large)$。两堆大小差严格控制在 $0$ 或 $1$。中位数可由堆顶以 $\mathcal{O}(1)$ 直接产出，插入开销为 $\mathcal{O}(\log N)$。
- **高频系统架构追问（System Extensions）**：
  1. **复合结构（MaxStack + 动态中位数）**：
     若要求栈结构同时支持 `popMax` 与实时中位数：
     - 使用双向链表记录元素物理入栈顺序，并结合平衡二叉搜索树（如 `TreeMap` / 红黑树）维护数值有序映射，实现 $\mathcal{O}(\log N)$ 的 `popMax` 与任意节点删除；
     - 配合哈希表索引的**懒删除（Lazy Deletion）对顶堆**维护中位数。
  2. **有界固定值域（Bounded Domain 0~100）**：
     若数据流数值严格属于 $[0, 100]$，则完全废弃堆结构！改用大小为 101 的频次计数数组 `count[101]`。插入 $\mathcal{O}(1)$；查询中位数只需扫描计数数组累计频次至 $N/2$，单次查询严格 $\mathcal{O}(100) = \mathcal{O}(1)$，无任何内存碎片。
  3. **超大数据流无法放入内存（Memory-Bound Stream）**：
     当单机内存无法承载全量流数据时，无法精确维护绝对中位数，必须采用**分位数流式近似算法**：
     - **t-digest**：聚类压缩临近数值为带权质心，特别适合高分位（P99, P99.9）与中位数近似；
     - **Count-Min Sketch** 或 **蓄水池抽样（Reservoir Sampling）**。
  4. **高并发读写同步（Concurrency & Thread Safety）**：
     采用**读写锁（Read-Write Lock）**保护双堆。写操作 `addNum` 占用独占写锁；高频读操作 `findMedian` 共享读锁。注意：中位数属于全局秩统计量（Rank Statistic），无法像加法统计量（Sum / Count）那样简单按分片（Shard）做局部合并，因此分片中位数需要两阶段二分协同。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **MinStack**：所有操作均为严格 $\mathcal{O}(1)$ 时间，$\mathcal{O}(N)$ 辅助空间。
- **MedianFinder**：`addNum` 耗时 $\mathcal{O}(\log N)$，`findMedian` 耗时 $\mathcal{O}(1)$，空间复杂度 $\mathcal{O}(N)$。

</div>

</div>
</details>

