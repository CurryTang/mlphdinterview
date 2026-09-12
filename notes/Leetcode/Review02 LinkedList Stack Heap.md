# 复习卡片：链表、栈与堆 (Review Flashcards · Linked List, Stack & Heap)

本篇为算法面试高频复习卡片第二辑：系统整理**链表与复合哈希结构 (Linked List & Hash-Linked Structures)**、**栈与单调结构 (Stack & Monotonic Stack / Deque)** 以及**堆与优先队列 (Heap & Priority Queue)** 的核心高频考题、工业级变体、生产级实现与时空复杂度全景。

---

## 模块一：链表与复合哈希结构 (Linked List & Hash-Linked Structures)

### 1. LRU 缓存与其系统级演进全家桶 (LRU Cache & System-Level Extensions)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">链表 01</span>
  <span class="review-card-title">LRU 缓存与其系统级演进全家桶 (LRU Cache & System-Level Extensions)</span>
  <span class="review-card-tag">双向链表 · 哈希映射 · TTL 过期 · LFU 频次置换 · 4 端双端队列 · 缓存未命中调优</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">链表 02</span>
  <span class="review-card-title">K 个一组翻转链表全家桶与组间重排 (Reverse Nodes in k-Group & Structural Group Inversion)</span>
  <span class="review-card-tag">哨兵虚拟节点 · 局部反转双指针 · 尾部不足也翻转 · 组间反转组内保序 · 自定义脚手架</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">链表 03</span>
  <span class="review-card-title">扁平化多级双向链表与空节点过滤 (Flatten Multilevel Doubly Linked List with Empty-Node Filtering)</span>
  <span class="review-card-tag">多级双向链表 · 子链优先递归/栈 · 空值节点清洗 · 双向指针自愈缝合</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与实现</div>

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

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- 时间 $\mathcal{O}(N)$，空间 $\mathcal{O}(D)$。

</div>

</div>
</details>

---

## 模块二：栈与单调结构 (Stack & Monotonic Stack / Deque)

### 4. 表达式计算器与运算符优先级全景全家桶 (Basic Calculator & Operator Precedence Hierarchy)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">栈 04</span>
  <span class="review-card-title">表达式计算器与运算符优先级全景全家桶 (Basic Calculator & Operator Precedence Hierarchy)</span>
  <span class="review-card-tag">单栈即时归约 · 括号状态暂存 · 负号/一元负数预处理 · 幂运算右结合 · Dijkstra 双栈调度场</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">队列 05</span>
  <span class="review-card-title">滑动窗口最大值与单调双端队列全景 (Sliding Window Maximum & Monotonic Deque Pattern)</span>
  <span class="review-card-tag">单调双端队列 · 索引窗口失效淘汰 · 均摊 O(1) 转移</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">单调栈 06</span>
  <span class="review-card-title">柱状图中最大的矩形与单调栈双哨兵范式 (Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern)</span>
  <span class="review-card-tag">单调递增栈 · 双哨兵 (Two-Sentinel) 技巧 · 左右边界动态判定 · 最大矩形降维扩展</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">栈/回溯 07</span>
  <span class="review-card-title">星号通配符括号有效性与全量展开 (Valid Parenthesis String with Wildcard & Concrete String Enumeration)</span>
  <span class="review-card-tag">区间贪心 · O(N) 双界指针 · 负下限保护 · DFS 全量分支展开 · 剪枝去重</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">堆 08</span>
  <span class="review-card-title">数据流中位数与多路归并全景 (Find Median from Data Stream & K-Way Merge)</span>
  <span class="review-card-tag">对顶双堆 · 严格平衡不变量 · 惰性删除</span>
</summary>
<div class="review-card-content">

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
```

</div>

</div>
</details>

---

### 9. 多商户分级加权轮转任务调度器 (Tiered Priority Task Scheduler)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">堆 09</span>
  <span class="review-card-title">多商户分级加权轮转任务调度器 (Tiered Priority Task Scheduler)</span>
  <span class="review-card-tag">商户级小顶堆 · FIFO 时间戳序列 · 活跃商户轮转队列 · VIP 加权配额调度</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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

    def process_next_task() -> Optional[Tuple[str, str]]:
        # 调度状态机代码见 Review02 生产实现
        pass
```

</div>

</div>
</details>

---

### 10. 时间戳任务调度器与直接 ID 淘汰 (Timestamp Task Scheduler with Direct ID Removal)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">堆 10</span>
  <span class="review-card-title">时间戳任务调度器与直接 ID 淘汰 (Timestamp Task Scheduler with Direct ID Removal)</span>
  <span class="review-card-tag">复合小顶堆 · 惰性删除 · 哈希版本校验</span>
</summary>
<div class="review-card-content">

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
```

</div>

</div>
</details>
