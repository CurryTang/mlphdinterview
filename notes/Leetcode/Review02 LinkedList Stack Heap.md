# 复习卡片：链表、栈与堆 (Review Flashcards · Linked List, Stack & Heap)

本篇为算法面试高频复习卡片第二辑：系统整理**链表与复合哈希结构 (Linked List & Hash-Linked Structures)**、**栈与单调双端队列 (Stack & Monotonic Deque)** 以及**堆与优先队列 (Heap & Priority Queue)** 的核心高频考题、工业级变体、生产级实现与时空复杂度全景。

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
    def get(self, key: int) -> int: ...   # 若不存在或已失效返回 -1
    def put(self, key: int, value: int) -> None: ...
```

| 追问编号 | 追问主题 | 核心变异条件 / 工业需求 | 架构突破口与实现策略 |
|---|---|---|---|
| **追问 1** | **TTL 键过期淘汰 (Add TTL)** | 写入时附带存活时间 `ttl`；读取已过期的键返回 `-1`，且过期键不可无谓占用物理容量。 | **惰性淘汰 (Lazy Eviction)**：访问时检查时间戳并即时摘除；配合**主动定时轮询/小根堆 (Proactive Sweeper)** 回收冷过期数据。 |
| **追问 2** | **LFU 频次淘汰 (Add LFU)** | 置换时首先淘汰使用频率最小的键；频次平局（Tie）时按最近最少使用淘汰。 | **双层映射哈希**：`key_node_map` + `freq_dll_map` + 维护全局最小频次标尺 `min_freq`，严控各状态转移全为 $O(1)$。 |
| **追问 3** | **四端队列与随机索引 (4-End Deque)** | 保持 $O(1)$ 约束，同时支持 `lpush`, `rpush`, `lpop`, `rpop` 以及基于下标的随机访问 `get_by_index(i)`。 | 双向链表原生支持四端 $O(1)$ 弹入弹出；工业级解法采用**分块双向链表 (Chunked Deque / Quicklist)** 或**动态环形数组 (Circular Ring Buffer)** 实现端点 $O(1)$、索引均摊 $O(1)$。 |
| **追问 4** | **LRU 访问路径迭代 (Print Path)** | 按从“最久未被访问”到“最新被访问”的时序遍历全量有效缓存条目。 | 遍历双向链表的拓扑序：从哨兵头节点 `head.next` 逐级推进至哨兵尾节点 `tail.prev`，输出迭代器生成器。 |
| **追问 5** | **高缓存未命中率调优 (High Miss Rate)** | 生产监控告警缓存未命中率飙升（> 30%），如何在前端/网关及服务端综合提升缓存命中率？ | 容量重估（工作集分析）、**置换策略跃迁 (TinyLFU 门禁准入/ARC)**、**多级缓存架构 (L1 本地内存 + L2 分布式 Redis)**、**空间局部性预取 (Prefetching)** 与**分段锁并发降低争用**。 |

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心机制深度剖析</div>

- **双向链表 + 字典**：字典存 `key -> node` 达成 $O(1)$ 寻址；双向链表节点持 `prev` 与 `next`，给定节点指针在 $O(1)$ 时间内完成局部断链并挪移至尾部。
- **哨兵头尾防空指针**：`head.next = tail`, `tail.prev = head`，彻底消除边界分支判断。
- **高未命中率治理体系**：容量重估（Amdahl 定律/工作集模型）、准入控制（W-TinyLFU Count-Min Sketch 过滤一次性冷数据防污染）、多级缓存拓扑（L1 本地进程内存 + L2 集中式 Redis）、分段锁（Striped Locking）削减并发排队延迟。

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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时空复杂度**：`get` 与 `put` 严格为 $\mathcal{O}(1)$；辅助空间为 $\mathcal{O}(C)$（$C$ 为容量）。
- **核心避坑**：从链表删除节点时必须同步调用 `del self.map[node.key]`，否则字典虚胖导致容量失真。

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
<div class="review-block-label">📌 题目定义与五大面试演进变体全景矩阵</div>

给你链表的头节点 `head`，每 `k` 个节点一组进行翻转，请你返回修改后的链表：

```python
def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]: ...
```

| 变体编号 | 核心变体名称 | 核心特征 / 变异条件 | 破局关键与指针重组策略 |
|---|---|---|---|
| **变体 1** | **经典 K 组翻转 (LC 25)** | 节点总数不是 $k$ 的整数倍时，最后剩余少于 $k$ 个的节点保持原有顺序。 | **先探测后反转**：用前向探针走 $k$ 步确认完整组存在；若不足 $k$ 步则终止保持原样。 |
| **变体 2** | **尾部不足亦翻转 (Reverse Partial Tail)** | 非标准变体：若链表末端剩余节点不足 $k$ 个，**依然无条件翻转**该尾部段落。 | **取消前置截断**：主循环只要剩余节点数 $\ge 1$，均截取最多 $k$ 个节点无条件翻转回接。 |
| **变体 3** | **两阶段分步热身 (Two-Parter Warmup)** | 一面热身题：第一问先手写基础单链表翻转；第二问直接以此为子模块组合成 K 组翻转。 | 将单链表翻转提取为通用子函数 `reverse_single_list(head)`。 |
| **变体 4** | **自定义 ListNode 与测试脚手架** | 面试平台无内置链表支持，现场手写 `ListNode` 类及数组与链表转换工具。 | 实现 `ListNode`，以及 `build_list` 和 `to_list` 断言测试套件。 |
| **变体 5** | **组间逆序而组内保序 (Reverse Group Order, Not Within)** | 保持每个 $k$ 组内部顺序不变，但将**各个组本身的拓扑顺序逆序拼接**。例如 $k=3$，$1\to2\to3\to4\to5\to6$ 变换为 $4\to5\to6\to1\to2\to3$。 | **分段收集 + 组级倒接**：不改变组内指针，按 $k$ 步切断收集各组 `(head, tail)`，逆向将各组串接。 |

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：严格 $\mathcal{O}(N)$。
- **空间复杂度**：原地指针翻转为 $\mathcal{O}(1)$；组间逆序记录段表为 $\mathcal{O}(N/k)$。
- **核心避坑**：翻转前必须将 `group_end.next = None`，否则反转后链表出现环形引用。

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
<div class="review-block-label">📌 题目定义与工业级变异条件</div>

给定一个多级双向链表，链表中的节点除了拥有 `prev` 和 `next` 指针外，还可能持有一个指向单独子双向链表的 `child` 指针。

**标准任务 (LC 430)**：将该多级链表展平，使所有节点出现在单层双向链表中，展开顺序遵循**先序深度优先遍历（深度子链优先于同级后继节点）**。

**面试高阶变体 (Core Twist)**：
- 某些嵌套节点携带空值（`val is None` 或特定占位符）。
- **要求**：在扁平化的最终结果中，**彻底剔除所有空值节点**，但必须完好保留通过这些空节点才能抵达的全部有效子链和有效后继节点！
- 展平并过滤后，必须严格恢复双向链表的不变式（所有相邻有效节点的 `prev` 与 `next` 严格双向对称互指，且所有 `child` 指针置空）。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与解耦治理架构</div>

#### 1. 为什么“边展平边跳过空节点”极其危险？
- 在同时处理 `child` 展平与 `next` 回接的复杂指针操作中，若再耦合就地空节点剔除逻辑，极易引发悬挂指针（Dangling Pointer）与 `prev` 错位断流。
- **工业级最佳实践：两阶段高内聚解耦流水线**：
  1. **阶段一：经典 DFS 展平 (Canonical Flattening)**：
     - 利用显式工作栈 `stack` 暂存同级后继节点 `cur.next`。
     - 遇到 `child` 时，将 `child` 节点作为当前节点的直接 `next`，将其 `prev` 绑定回当前节点，并将 `child` 指针清空。
     - 当遍历到当前子链尾部时，从栈中弹出暂存的父级 `next` 节点，完美回接。
  2. **阶段二：原地空节点摘除与双向指针对称自愈 (In-place Node Purging)**：
     - 单趟线性扫描已展平的单层链表。
     - 若当前节点 `cur.val` 为空：
       - 若存在前驱 `cur.prev`，令 `cur.prev.next = cur.next`；
       - 若存在后继 `cur.next`，令 `cur.next.prev = cur.prev`；
       - 若当前节点恰好是链表头，更新头指针向后移动。
- **收益**：将复杂的拓扑递归与数据清洗彻底解耦，时间复杂度保持严格 $O(N)$，额外空间 $O(D)$（$D$ 为嵌套层数），彻底杜绝指针悬挂 Bug！

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import Optional

class MultiLevelNode:
    """多级双向链表节点定义"""
    def __init__(self, val: Optional[int] = None, prev=None, next=None, child=None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

class MultiLevelListFlattenSolution:

    @classmethod
    def flattenAndFilterEmpty(cls, head: Optional[MultiLevelNode]) -> Optional[MultiLevelNode]:
        """
        多级双向链表展平并清洗空节点：
        1. 深度优先展平所有子链
        2. 剔除 val is None 的空节点并自愈前后指针
        时间复杂度 O(N)，空间复杂度 O(D)
        """
        if not head:
            return None

        # ---------------------------------------------------------
        # 阶段 1: 经典 DFS 展平（先序遍历，显式工作栈）
        # ---------------------------------------------------------
        cur = head
        stack = []

        while cur:
            if cur.child:
                # 若存在同级后继，压栈暂存
                if cur.next:
                    stack.append(cur.next)
                # 子链提升为直接后继
                cur.next = cur.child
                cur.child.prev = cur
                cur.child = None  # 置空 child 满足规范

            # 抵达当前分支末梢且栈中有挂起的后继分支
            if not cur.next and stack:
                nxt = stack.pop()
                cur.next = nxt
                nxt.prev = cur

            cur = cur.next

        # ---------------------------------------------------------
        # 阶段 2: 单趟原地清洗空值节点，自愈双向对称性
        # ---------------------------------------------------------
        dummy = MultiLevelNode(0, next=head)
        head.prev = dummy

        cur = head
        while cur:
            nxt = cur.next
            if cur.val is None:
                # 剔除空节点：前驱接后继，后继接前驱
                cur.prev.next = nxt
                if nxt:
                    nxt.prev = cur.prev
                cur.prev = cur.next = None  # 协助 GC
            cur = nxt

        new_head = dummy.next
        if new_head:
            new_head.prev = None
        return new_head
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：展平单趟 $\mathcal{O}(N)$，清洗扫描单趟 $\mathcal{O}(N)$，总体时间复杂度严格为 $\mathcal{O}(N)$。
- **空间复杂度**：工作栈深度等于多级子链最大嵌套深度 $\mathcal{O}(D)$，最坏退化为 $\mathcal{O}(N)$。
- **高频避坑清单**：
  1. **遗漏清空 `child` 指针**：展平后所有节点的 `child` 必须显式赋为 `None`。
  2. **双向指针单边断裂**：清洗空节点时，必须同时维护 `prev.next` 和 `next.prev`。若漏改 `next.prev`，反向遍历时将触发链表断流。

</div>

</div>
</details>

---

## 模块二：栈与单调双端队列 (Stack & Monotonic Deque)

### 4. 表达式计算器与运算符优先级全景全家桶 (Basic Calculator & Operator Precedence Hierarchy)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">栈 04</span>
  <span class="review-card-title">表达式计算器与运算符优先级全景全家桶 (Basic Calculator & Operator Precedence Hierarchy)</span>
  <span class="review-card-tag">单栈即时归约 · 括号状态暂存 · 负号/一元负数预处理 · 幂运算右结合 · Dijkstra 双栈调度场</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与核心变体全景矩阵</div>

实现一个支持对包含非负整数、加减乘除运算符以及嵌套括号的数学表达式字符串 `s` 求值的计算器引擎：

```python
def calculate(s: str) -> int: ...
```

- 支持 `+`, `-`, `*`, `/` 及右结合乘方 `^`。
- 支持前导负号 `-5`、`(-3 + 4)`。
- 除法遵循严格的向零截断整数语义 `int(a / b)`。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：严格 $\mathcal{O}(N)$。
- **空间复杂度**：$\mathcal{O}(N)$。
- **核心避坑**：弹出左右操作数次序颠倒（先弹出右操作数 $b$，后弹出左操作数 $a$）。

</div>

</div>
</details>

---

### 5. 滑动窗口最大值与单调双端队列全景 (Sliding Window Maximum & Monotonic Deque Pattern)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">队列 05</span>
  <span class="review-card-title">滑动窗口最大值与单调双端队列全景 (Sliding Window Maximum & Monotonic Deque Pattern)</span>
  <span class="review-card-tag">单调队列 · 双端队列 (Deque) · 索引窗口失效淘汰 · 均摊 O(1) 状态转移</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与经典应用场景</div>

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值序列（LC 239）：

```python
def maxSlidingWindow(nums: List[int], k: int) -> List[int]: ...
```

**输入输出示例**：
- 输入：`nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3`
- 输出：`[3, 3, 5, 5, 6, 7]`

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与单调队列心智模型</div>

#### 1. 为什么优先队列（大顶堆）不能做到最优？
- 大顶堆提取最大值是 $\mathcal{O}(1)$，但插入和维持是 $\mathcal{O}(\log k)$。总时间复杂度为 $\mathcal{O}(N \log k)$。
- 此外，堆中无法快速检索并删除因滑出窗口左侧而失效的元素（必须配合哈希延迟删除）。

#### 2. 单调双端队列 (Monotonic Deque) 的数学原理
- **核心淘汰洞察**：
  - 如果一个新元素进入窗口时，其数值比窗口内更早到来的元素还要大，那么**那些既比它小、又比它先过期的老旧元素，在有生之年绝无可能成为窗口的最大值**！
  - 因此，它们可以直接被永远逐出队列。
- **双端队列两头操作契约**：
  1. **队列内部严格单调递减**：队列中存储元素的**数组下标**，对应数值从队头到队尾严格递减。
  2. **队尾入队前清洗弱者**：当扫描到下标 $i$ 对应的数 $nums[i]$ 时，只要队尾下标元素 $\le nums[i]$，循环弹出队尾（`pop()`）。
  3. **队头检查过期**：检查队头下标是否超出窗口有效范围（即 $\le i - k$），若超限则从队头弹出（`popleft()`）。
  4. **队头直接读取最大值**：当前窗口的最大值永远静止位于队头 `deque[0]`，耗时严格 $O(1)$！

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from collections import deque
from typing import List

class SlidingWindowMaxSolution:
    """
    滑动窗口最大值单调双端队列实现：
    时间复杂度 O(N)，额外空间复杂度 O(k)
    """

    @staticmethod
    def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
        if not nums or k <= 0:
            return []

        # q 中存储下标，维护对应的数值严格单调递减
        q: deque[int] = deque()
        res: List[int] = []

        for i, val in enumerate(nums):
            # 1. 清洗队尾所有小于等于当前值的弱势元素
            while q and nums[q[-1]] <= val:
                q.pop()

            # 2. 将当前元素下标压入队尾
            q.append(i)

            # 3. 淘汰超出滑动窗口左边界的过期元素
            if q[0] <= i - k:
                q.popleft()

            # 4. 当窗口形成后（i >= k - 1），收集队头最大值
            if i >= k - 1:
                res.append(nums[q[0]])

        return res
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：每个元素的下标最多进队 1 次、出队 1 次，总体循环执行次数严格为 $\mathcal{O}(N)$。
- **空间复杂度**：双端队列在任意时刻最多存储 $k$ 个下标，空间复杂度为严格 $\mathcal{O}(k)$。
- **高频避坑清单**：
  1. **队列中存数值还是存下标**：**必须存下标**！若存数值，无法精确判定队头元素是否已滑出窗口边界。
  2. **窗口形成前的提前收集**：只有在 $i \ge k - 1$ 时窗口才初次填满，前 $k-1$ 步只需维护队列，不可向结果集追加元素。

</div>

</div>
</details>

---

## 模块三：堆与优先队列 (Heap & Priority Queue)

### 6. 数据流中位数与多路归并全景 (Find Median from Data Stream & K-Way Merge)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">堆 06</span>
  <span class="review-card-title">数据流中位数与多路归并全景 (Find Median from Data Stream & K-Way Merge)</span>
  <span class="review-card-tag">对顶双堆 · 严格平衡不变量 · 惰性删除 · 多路归并 · Top-K 桶排序</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与对顶堆模型</div>

```python
import heapq
from typing import List

class MedianFinder:
    def __init__(self):
        self.lo: List[int] = []  # 大顶堆（存相反数）
        self.hi: List[int] = []  # 小顶堆

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
<div class="review-block-label">⏱️ 复杂度分析</div>

- `addNum`: $\mathcal{O}(\log N)$；`findMedian`: $\mathcal{O}(1)$；空间: $\mathcal{O}(N)$。

</div>

</div>
</details>

---

### 7. 多商户分级加权轮转任务调度器 (Tiered Priority Task Scheduler)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">堆 07</span>
  <span class="review-card-title">多商户分级加权轮转任务调度器 (Tiered Priority Task Scheduler)</span>
  <span class="review-card-tag">商户级小顶堆 · FIFO 时间戳序列 · 活跃商户轮转队列 · VIP 加权配额调度</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与调度机制</div>

每个任务包含 `task_id`, `seller_id`, `tier` (`VIP`/`STANDARD`), `priority` (1 最高, 3 最低)。
- 商户内按 `priority` 升序，同优先级严格 FIFO；
- 跨商户轮转：VIP 每次最多处理 2 个任务，STANDARD 每次最多处理 1 个任务；堆空提前让渡。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- 每次堆操作 $\mathcal{O}(\log M)$，轮转 $\mathcal{O}(1)$，空间 $\mathcal{O}(T + S)$。

</div>

</div>
</details>

---

### 8. 时间戳任务调度器与直接 ID 淘汰 (Timestamp Task Scheduler with Direct ID Removal)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">堆 08</span>
  <span class="review-card-title">时间戳任务调度器与直接 ID 淘汰 (Timestamp Task Scheduler with Direct ID Removal)</span>
  <span class="review-card-tag">复合小顶堆 · 惰性删除 (Lazy Deletion) · 哈希版本校验 · 破坏性出堆</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与实现代码</div>

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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- `addTask` $\mathcal{O}(\log N)$, `removeTask` $\mathcal{O}(1)$, `popTask` 均摊 $\mathcal{O}(K \log N)$。

</div>

</div>
</details>
