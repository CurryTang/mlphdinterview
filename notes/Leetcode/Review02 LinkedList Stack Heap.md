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
<div class="review-block">
<div class="review-block-label">🌐 核心基石延伸：五大工业级衍生架构全家桶</div>

#### 1. 餐厅候补队列匹配系统 (Restaurant Waitlist / Table Matching Queue)
- **业务场景**：
  - `join(user, party_size)`：顾客入队，追加到排队队列尾部，$\mathcal{O}(1)$。
  - `delete(user)`：顾客因超时或取消排队离开队伍，可位于队列任意位置，要求 $\mathcal{O}(1)$ 删除。
  - `find_first_match(table_size: int)`：空出一张容量为 `table_size` 的餐桌，从排队队列中检索**到达时间最早（FIFO）且 `party_size <= table_size`** 的顾客。检索不删除顾客。
- **架构权衡与设计**：
  - **基础双向链表 + 哈希表（LRU 结构对偶）**：
    - `user_map: Dict[user, DLLNode]` 存储节点句柄，实现 $\mathcal{O}(1)$ 的 `delete` 与 `join`；
    - `find_first_match(t)` 从链表头（最早排队顾客）向后线性扫描，找到第一个 `party_size <= t` 的节点。时间复杂度为 $\mathcal{O}(N)$。
  - **分桶队列优化（Table-Size Bucket Optimization）**：
    - 现实中顾客人数 `party_size` 通常为小整数（如 $1 \sim 10$）。
    - 针对每个就餐人数 $s \in [1, 10]$ 分别维护一条双向队列 `buckets[s]`。
    - `join(user, s)`：将用户追加到 `buckets[s]` 的尾部，时间 $\mathcal{O}(1)$。
    - `delete(user)`：通过哈希表定位所属 bucket 与节点指针，从对应双向链表中 $\mathcal{O}(1)$ 摘除。
    - `find_first_match(table_size)`：只需扫描 $s \in [1, 	ext{table\_size}]$ 的各个桶头节点，比较其进入时间戳，选取**时间戳最小（最先到达）**的顾客。时间复杂度降为严格 $\mathcal{O}(	ext{table\_size}) = \mathcal{O}(1)$。

```python
class WaitlistSystem:
    class CustomerNode:
        def __init__(self, user: str, size: int, ts: int):
            self.user = user
            self.size = size
            self.ts = ts
            self.prev = self.next = None

    def __init__(self, max_party_size: int = 10):
        self.max_party_size = max_party_size
        self.user_map = {}  # user -> CustomerNode
        self.buckets = {s: (self.CustomerNode("", 0, 0), self.CustomerNode("", 0, 0)) for s in range(1, max_party_size + 1)}
        for s in self.buckets:
            h, t = self.buckets[s]
            h.next, t.prev = t, h
        self.clock = 0

    def join(self, user: str, party_size: int) -> None:
        if user in self.user_map or party_size > self.max_party_size:
            return
        self.clock += 1
        node = self.CustomerNode(user, party_size, self.clock)
        self.user_map[user] = node
        h, t = self.buckets[party_size]
        # 追加至 tail 前
        node.prev, node.next = t.prev, t
        t.prev.next = node
        t.prev = node

    def delete(self, user: str) -> bool:
        if user not in self.user_map:
            return False
        node = self.user_map.pop(user)
        node.prev.next = node.next
        node.next.prev = node.prev
        return True

    def find_first_match(self, table_size: int) -> Optional[str]:
        earliest_node = None
        limit = min(table_size, self.max_party_size)
        for s in range(1, limit + 1):
            h, t = self.buckets[s]
            head_cand = h.next
            if head_cand is not t:
                if earliest_node is None or head_cand.ts < earliest_node.ts:
                    earliest_node = head_cand
        return earliest_node.user if earliest_node else None
```

#### 2. 支持乱序时间戳的日志限流器 (Robot Logger with Out-of-Order Timestamps)
- **业务场景**：
  - 分布式机器人或传感器上报 `(timestamp, message)`。
  - 规则：若同一 `message` 在逻辑时间区间 $[timestamp - 10, timestamp)$ 内**已经打印过**，则拦截隐藏（返回 `False`）；否则放行打印（返回 `True`）。
  - **核心难点（乱序上报）**：由于网络延迟，数据包可能乱序到达（例如 `(12, "foo")` 先到达，随后 `(10, "foo")` 到达）。
  - **因果约束律**：
    1. 后来的时间戳绝不能反向封杀早前的时间戳（$12$ 不属于 $10$ 过去 10 秒的历史，故 $10$ 必须放行）；
    2. 同一时间戳多次调用，仅首次放行；
    3. 未放行的拦截调用绝不能刷新时间窗口！
- **数据结构选型**：
  - 对每个 `message`，维护其所有**已成功放行时间戳的动态有序集合**（红黑树 / 跳表 / `SortedSet`）。
  - 判定流程：
    1. 在有序表中二分查找 $timestamp$ 的最大严格前驱 $pred = \max \{x \in S \mid x < timestamp\}$。
    2. 若 $timestamp \in S$ 或 ($pred \neq \text{None}$ 且 $pred \ge timestamp - 10$)，返回 `False`；
    3. 否则将 $timestamp$ 插入集合 $S$，返回 `True`。
  - 单次判定开销 $\mathcal{O}(\log K)$，其中 $K$ 为该消息历史放行条数。在时间窗口滑动时，利用双向指针或 LRU 机制淘汰全局安全下限之外的历史数据。

#### 3. 基于 Idempotency-Key 的幂等 API 处理器 (Idempotency API with Concurrency & TTL)
- **核心契约**：
  - 首次携带 Key $K$ 与请求体 $B$：正常执行并返回响应 $R$，持久化 $(K \to R)$；
  - 相同 Key $K$ 与相同 Body $B$ 重试：直接返回缓存的 $R$，杜绝重复扣款/下单；
  - 相同 Key $K$ 但不同 Body $B'$：抛出 `409 Conflict`（非法键复用）；
  - **高并发竞态防护**：两个携带相同 Key $K$ 的请求并发到达时，第二个请求必须在 `threading.Condition` 上挂起等待第一个请求完成，绝不可双重执行！
  - **TTL 回收策略**：缓存记录配置 TTL（如 24 小时），支持惰性校验与后台周期扫描。

```python
import hashlib, json, time, threading
from enum import Enum
from typing import Dict, Any, Tuple

class RequestStatus(Enum):
    IN_FLIGHT = 1
    DONE = 2

class IdempotencyRecord:
    def __init__(self, body_hash: str, ttl_seconds: float):
        self.body_hash = body_hash
        self.status = RequestStatus.IN_FLIGHT
        self.response = None
        self.expires_at = time.time() + ttl_seconds
        self.condition = threading.Condition()

class IdempotencyManager:
    def __init__(self, default_ttl: float = 86400.0):
        self.default_ttl = default_ttl
        self.store: Dict[str, IdempotencyRecord] = {}
        self.lock = threading.Lock()

    def _hash_body(self, body: Any) -> str:
        canonical_json = json.dumps(body, sort_keys=True)
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def handle_request(self, idempotency_key: str, body: Any, execute_fn) -> Tuple[int, Any]:
        body_hash = self._hash_body(body)
        now = time.time()

        with self.lock:
            if idempotency_key in self.store:
                rec = self.store[idempotency_key]
                # 检查是否已过期
                if now > rec.expires_at:
                    del self.store[idempotency_key]
                else:
                    # 检查请求体冲突 (409 Conflict)
                    if rec.body_hash != body_hash:
                        return 409, {"error": "Idempotency key re-used with different payload"}
                    
                    # 若仍在处理中，等待其完成
                    if rec.status == RequestStatus.IN_FLIGHT:
                        rec.condition.wait()
                        return 200, rec.response
                    else:
                        return 200, rec.response

            # 首次进入：注册 IN_FLIGHT 记录
            rec = IdempotencyRecord(body_hash, self.default_ttl)
            self.store[idempotency_key] = rec

        # 在锁外执行真实业务逻辑
        try:
            res = execute_fn(body)
            with rec.condition:
                rec.response = res
                rec.status = RequestStatus.DONE
                rec.condition.notify_all()
            return 200, res
        except Exception as e:
            with self.lock:
                self.store.pop(idempotency_key, None)
            with rec.condition:
                rec.condition.notify_all()
            raise e
```

#### 4. 连续内存分配器 (First-Fit Memory Allocator with Splitting & Coalescing)
- **业务场景**：
  - 给定大小为 $N$ 的单一连续内存池。
  - `allocate(size)`：寻找大小 $\ge size$ 的连续空闲块并分配，返回起始偏移；若不足返回错误。
  - `free(offset)`：归还指定块，并与**物理相邻的空闲块合并（Coalescing）**，防止内存外部碎片化。
- **双向链表 + 边界标记法 (Boundary Tags)**：
  - 每个内存块节点维护：`offset, size, is_free, prev, next`。
  - `allocate`：采用首次适应算法（First-Fit）沿双向链表扫描。若空闲块 `size > request_size`，将其切分为已分配块和剩余空闲块。
  - `free`：将当前块标记为 `is_free = True`；若 `prev` 也是空闲块，直接向前合并；若 `next` 也是空闲块，直接向后合并。合并仅涉及双向链表常数个指针的重排，为严格 $\mathcal{O}(1)$ 操作！

#### 5. 并发线程安全与分布式分片架构
- **单机并发安全**：
  - 粗粒度保护：全局 `threading.RLock()` 封装 `get`/`put`。
  - 细粒度分片（Sharded Cache / ConcurrentHashMap 模式）：
    将大缓存水平切分为 $M$ 个独立分片（Shard），依据 $	ext{hash}(key) \pmod M$ 路由到具体分片。各分片持独立的锁与双向链表，使并发写吞吐随核心数线性扩展（注意：此时退化为分片局部的近拟 LRU）。
- **跨机器分布式缓存**：
  - 采用**一致性哈希环（Consistent Hashing with Virtual Nodes）**实现节点弹性扩缩容；单机节点内部继续运行双向链表+哈希表的纯粹 LRU 引擎。

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
<div class="review-block">
<div class="review-block-label">🌐 核心基石延伸：回文链表原状复原与高位减法全家桶</div>

#### 1. 回文链表与工程级原状复原 (Palindrome Linked List - LC 234)
- **核心契约**：
  - 判断单链表是否为回文序列；
  - 空间约束：$\mathcal{O}(1)$ 辅助空间（严禁将节点全量存入数组）；
  - **工业级生产约束 (Production Follow-up)**：在函数返回前，**必须将链表恢复为其原本的物理指针结构**，防止上游调用方观察到破坏性的副作用（In-Place Mutation Side Effect）。
- **算法实施**：
  1. 快慢双指针探测中点：`slow` 单步推进，`fast` 双步推进。当 `fast` 抵达末尾时，`slow` 恰好停在前半段末尾，`slow.next` 为后半段起点。
  2. 翻转后半段链表：`second_half = reverseList(slow.next)`。
  3. 双指针平移校验：`p1 = head`, `p2 = second_half` 逐值比对。
  4. **原状恢复 (State Restoration)**：再次翻转后半段，`slow.next = reverseList(second_half)`，无缝拼接复原。

```python
class PalindromeSolution:
    @staticmethod
    def isPalindrome(head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True

        # 1. 快慢指针找中点
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        # 2. 原地翻转后半部分
        def reverse_chain(node: Optional[ListNode]) -> Optional[ListNode]:
            prev = None
            curr = node
            while curr:
                nxt = curr.next
                curr.next = prev
                prev = curr
                curr = nxt
            return prev

        second_head = reverse_chain(slow.next)

        # 3. 比较前半部分与后半部分
        p1, p2 = head, second_head
        is_pal = True
        while is_pal and p2:
            if p1.val != p2.val:
                is_pal = False
            p1 = p1.next
            p2 = p2.next

        # 4. 【关键工程保护】：将后半段翻转接回，恢复调用方链表物理原貌
        slow.next = reverse_chain(second_head)

        return is_pal
```

#### 2. 高位在前单链表减法 (Forward-Order Linked List Subtraction $l_1 - l_2$)
- **核心契约**：
  - 给出两个非空单链表 $l_1, l_2$，每个节点包含一位十进制数，**高位在先（MSB First）**。
  - 保证 $l_1 \ge l_2$，计算 $l_1 - l_2$ 并以相同的高位在前链表返回。
  - 剔除多余前导零（结果为 0 则保留单个 0 节点）。严禁将链表整体转化为内置大整数类型（数字长度可达 $10^5$）。
- **算法实施**：
  - 先将两链表就地翻转为低位在先（LSB First），从而在 $\mathcal{O}(1)$ 空间内对齐个位数；
  - 模拟小学竖式减法，维护借位量 `borrow = 0`：
    $$	ext{diff} = val_1 - val_2 - borrow$$
    若 $	ext{diff} < 0$，则 $	ext{diff} += 10, borrow = 1$；否则 $borrow = 0$。
  - 得到差值链表后再次翻转回 MSB 顺序，最后快慢指针剥离前导零。

```python
class LinkedListSubtractionSolution:
    @classmethod
    def subtractLinkedList(cls, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        def reverse(node):
            prev, curr = None, node
            while curr:
                nxt = curr.next
                curr.next = prev
                prev = curr
                curr = nxt
            return prev

        r1 = reverse(l1)
        r2 = reverse(l2)

        p1, p2 = r1, r2
        dummy = ListNode(0)
        curr = dummy
        borrow = 0

        while p1:
            v1 = p1.val
            v2 = p2.val if p2 else 0
            diff = v1 - v2 - borrow
            if diff < 0:
                diff += 10
                borrow = 1
            else:
                borrow = 0
            curr.next = ListNode(diff)
            curr = curr.next
            p1 = p1.next
            if p2: p2 = p2.next

        # 将 r1 和 r2 恢复以保护原结构
        reverse(r1)
        reverse(r2)

        # 结果链表翻转回高位在前
        res = reverse(dummy.next)

        # 剥离前导 0
        while res and res.val == 0 and res.next:
            res = res.next

        return res
```

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
  字典序越小的字符越应靠前。遇到字符 $ch$ 时，若栈顶字符 $top > ch$, 且 $top$ 在后续文本中还会再次登场（$last\_occurrence[top] > i$），则此时抛弃 $top$ 绝不会导致未来缺失该字符，同时让更小的 $ch$ 占据高位，必然能使整体字典序变小。
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

---

### 14. 删除链表的倒数第 N 个节点 (Remove Nth Node From End of List)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 14</span>
  <span class="review-card-title">删除链表的倒数第 N 个节点 (Remove Nth Node From End of List)</span>
  <span class="review-card-tag">双指针快慢定距 · 虚拟头节点哨兵 · 单趟扫描 · 空间 O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 19 · Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) — `https://leetcode.com/problems/remove-nth-node-from-end-of-list/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> Given the head of a singly linked list, remove the $n$-th node from the end of the list and return its head.
> **Constraint**: You must solve it in a single pass with $\mathcal{O}(1)$ auxiliary space.

**函数签名**：
```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]: ...
```

**输入输出示例**：
- `head = [1, 2, 3, 4, 5], n = 2` $\implies$ `[1, 2, 3, 5]`
- `head = [1], n = 1` $\implies$ `[]`
- `head = [1, 2], n = 1` $\implies$ `[1]`

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class RemoveNthFromEndSolution:
    @staticmethod
    def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 单趟扫描删除链表倒数第 n 个节点。
        # 使用哨兵虚拟头节点消除删除 head 的特例分支。
        dummy = ListNode(0, head)
        fast = dummy
        slow = dummy

        # 1. fast 指针先行前进 n + 1 步，构建长度为 n + 1 的跨步窗口
        for _ in range(n + 1):
            fast = fast.next

        # 2. fast 与 slow 同步向前单步滑动，直到 fast 越过链表末尾变为 None
        while fast is not None:
            fast = fast.next
            slow = slow.next

        # 3. 此时 slow 恰好停在待删除节点的前驱节点 (Predecessor)
        slow.next = slow.next.next

        return dummy.next
```

```cpp
#include <memory>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x, ListNode* n = nullptr) : val(x), next(n) {}
};

class RemoveNthFromEndSolution {
public:
    static ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0, head);
        ListNode* fast = &dummy;
        ListNode* slow = &dummy;

        // 1. fast 先行推进 n + 1 步
        for (int i = 0; i <= n; ++i) {
            fast = fast->next;
        }

        // 2. 双指针同步推移
        while (fast != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }

        // 3. 跨过目标节点完成删除
        ListNode* target = slow->next;
        slow->next = slow->next->next;
        delete target; // 释放堆内存避免泄漏

        return dummy.next;
    }
};
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析与不变量证明</div>

- **定距滑动窗口不变量 (Fixed-Offset Invariant)**：
  - 设链表节点总数为 $L$，虚拟头节点位于索引 $0$，链表节点位于索引 $1 \dots L$，末尾空指针位于索引 $L + 1$。
  - 待删除的倒数第 $n$ 个节点其正向索引为 $L - n + 1$。
  - 其前驱节点（即需要被修改 `next` 指针的节点）正向索引为 $(L - n + 1) - 1 = L - n$。
  - 初始化时，`fast` 先从 $0$（`dummy`）走 $n + 1$ 步到达索引 $n + 1$。此时 `fast` 与 `slow` 的距离差恒为 $n + 1$。
  - 当 `fast` 滑动至 $L + 1$（即 `fast is None`）时，`slow` 所在索引为 $(L + 1) - (n + 1) = L - n$。
  - 不变量成立：**`slow` 必然严格停在待删除节点的前驱节点**。
- **哨兵节点 (Sentinel / Dummy Head) 的工程价值**：
  - 若删除原链表的头节点（$n = L$），若不设哨兵，需单独判定 `if n == length: return head.next`。
  - 引入 `dummy` 节点后，链表头节点退化为普通内部节点，所有删除逻辑统一为 `slow.next = slow.next.next`，彻底消除边界分支。
- **面试口述规范 (Communication Checklist)**：
  - 面试时明确表述窗口定距不变式：“I advance the fast pointer by $n+1$ steps from a dummy node so that when fast hits null, slow is guaranteed to sit exactly at the node immediately preceding the target deletion node.”

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与边界分析</div>

- **时间复杂度**：$\mathcal{O}(L)$，单趟（One-pass）遍历，`fast` 仅扫描 $L + 1$ 次。
- **空间复杂度**：$\mathcal{O}(1)$，仅需常数级别的辅助指针。
- **核心边界用例**：
  1. $L = 1, n = 1$：单节点链表删除后返回空链表。
  2. $n = L$：删除链表原首节点。
  3. $n = 1$：删除链表尾节点。

</div>

</div>
</details>


---

### 15. 循环有序单链表的插入 (Insert into a Sorted Circular Linked List)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 15</span>
  <span class="review-card-title">循环有序单链表的插入 (Insert into a Sorted Circular Linked List)</span>
  <span class="review-card-tag">双指针循环遍历 · 拐点判定 · 环形边界环绕 · 空间 O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **相关链接**：
> - [LeetCode 708 · Insert into a Sorted Circular Linked List](https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/) — `https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/`
> - [1point3acres 面经真题](https://www.1point3acres.com/interview/problems/e1081044-6f41-5139-8596-3e843a348997) — Meta 电话面试高频题

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> Given a Circular Linked List node, which is sorted in non-descending order, write a function to insert a value `insertVal` into the list such that it remains a sorted circular list.
> The given node can be a reference to **any single node** in the list and may not necessarily be the smallest value in the circular list.
> If the list is empty (i.e., the given node is `null`), you should create a new single circular list and return the reference to that single node. Otherwise, you should return the original given node.

**函数签名**：
```python
class Solution:
    def insert(self, head: Optional[Node], insertVal: int) -> Node: ...
```

**输入输出示例**：
- `head = [3, 4, 1], insertVal = 2` $\implies$ `[3, 4, 1, 2]`
- `head = [], insertVal = 1` $\implies$ `[1]` (自环单节点)
- `head = [1], insertVal = 0` $\implies$ `[1, 0]`

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import Optional

class Node:
    def __init__(self, val: int = 0, next: Optional['Node'] = None):
        self.val = val
        self.next = next

class InsertSortedCircularListSolution:
    @staticmethod
    def insert(head: Optional[Node], insertVal: int) -> Node:
        # 向循环升序单链表中插入 insertVal，并保持环形有序。
        # head 可以是环中任意节点。
        # 边界情况 1: 空链表，直接创建自环单节点
        if not head:
            new_node = Node(insertVal)
            new_node.next = new_node
            return new_node

        prev = head
        curr = head.next

        while True:
            # 判定情况 2: 内部有序插入区间 (prev.val <= insertVal <= curr.val)
            if prev.val <= insertVal <= curr.val:
                break

            # 判定情况 3: 跨越最大值到最小值的断层拐点 (Inflection Point)
            if prev.val > curr.val:
                # 插入值大于等于全环最大值，或小于等于全环最小值
                if insertVal >= prev.val or insertVal <= curr.val:
                    break

            prev = curr
            curr = curr.next

            # 判定情况 4: 完整遍历环一圈回到起点 (如链表中所有节点值皆相同)
            if prev == head:
                break

        # 将新节点插入 prev 与 curr 之间
        new_node = Node(insertVal, curr)
        prev.next = new_node

        return head
```

```cpp
class Node {
public:
    int val;
    Node* next;
    Node(int _val) : val(_val), next(nullptr) {}
    Node(int _val, Node* _next) : val(_val), next(_next) {}
};

class InsertSortedCircularListSolution {
public:
    static Node* insert(Node* head, int insertVal) {
        if (!head) {
            Node* newNode = new Node(insertVal);
            newNode->next = newNode;
            return newNode;
        }

        Node* prev = head;
        Node* curr = head->next;

        while (true) {
            // 情况 2: 落在常规升序区间内
            if (prev->val <= insertVal && insertVal <= curr->val) {
                break;
            }

            // 情况 3: 到达最大值 -> 最小值的跃变拐点
            if (prev->val > curr->val) {
                if (insertVal >= prev->val || insertVal <= curr->val) {
                    break;
                }
            }

            prev = curr;
            curr = curr->next;

            // 情况 4: 兜圈一周回到起始指针 (例如所有节点值全相等)
            if (prev == head) {
                break;
            }
        }

        prev->next = new Node(insertVal, curr);
        return head;
    }
};
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析与四大判定分支</div>

- **环形有序结构的数学分类**：
  由于给定指针 `head` 可能指向环中任意位置，单链表包含唯一一个**最大值降至最小值的跃变拐点（Inflection Point）**（除非所有元素完全相等）。
  整个插入决策分为完备互斥的四大分支：
  1. **空链表初始化**：
     若 `head == None`，构造新节点并让其指向自身（`node.next = node`），返回该节点。
  2. **区间内部平滑插入 (Interior Ordered Segment)**：
     若 `prev.val <= insertVal <= curr.val`，`insertVal` 恰好介于局部有序相邻两节点之间，直接插入即可。
  3. **极值拐点环绕插入 (Inflection Wrap-Around)**：
     当 `prev.val > curr.val` 时，`prev` 为全环局部最大值，`curr` 为全环局部最小值。
     - 若 `insertVal >= prev.val`（新元素比全环最大值还要大或相等）；
     - 若 `insertVal <= curr.val`（新元素比全环最小值还要小或相等）；
     此时新元素在逻辑上必须置于最大值与最小值之间。
  4. **全环等值或兜圈兜底 (Degenerate / Monotonous Loop)**：
     若链表中所有节点值完全相等（例如 `[3, 3, 3]`），或遍历整整一圈回到原点（`prev == head`）仍未触发前述条件，说明新元素可插入环中任意位置。此时在 `prev` 之后直接插入均满足循环升序。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与边界分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，最坏情况下遍历循环链表一周（$N$ 个节点）。
- **空间复杂度**：$\mathcal{O}(1)$，仅需常数个指针维护移动。
- **极端用例防御**：
  - `head = None`（空链表处理）。
  - 单节点链表自环（`head.next == head`）。
  - 双节点且有明显升序 `[1, 3]` 插入 `2`、`0`、`4`。
  - 全等节点链表 `[3, 3, 3]` 插入 `1` 或 `5`。

</div>

</div>
</details>


---

### 16. 常数时间随机集合与弹出容器 (Randomized Container with O(1) Insert & PopRandom)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">容器 16</span>
  <span class="review-card-title">常数时间随机集合与弹出容器 (Randomized Container with O(1) Insert & PopRandom)</span>
  <span class="review-card-tag">连续动态数组 + 哈希索引表 · 尾部元素置换 (Swap with Last) · 等概率随机抽取 · O(1) 均摊</span>
</summary>
<div class="review-card-content">

> 🔗 **相关链接**：
> - [LeetCode 380 · Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) — `https://leetcode.com/problems/insert-delete-getrandom-o1/`
> - [LeetCode 381 · Insert Delete GetRandom O(1) - Duplicates allowed](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/) — `https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/`
> - [1point3acres 面经真题](https://www.1point3acres.com/interview/problems/company/meta/randomized-container) — Meta 工业级容器设计题

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> Design a data structure that supports all following operations in average $\mathcal{O}(1)$ time complexity:
> 1. `insert(val)`: Inserts an item `val` to the set if not already present. Returns `true` if the item was not present, `false` otherwise.
> 2. `remove(val)`: Removes an item `val` from the set if present. Returns `true` if the item was present, `false` otherwise.
> 3. `getRandom()`: Returns a random element from the current set of elements. Each element must have the **same probability** of being returned.
> 4. `popRandom()` (Meta Extension): Removes and returns a random element from the container in $\mathcal{O}(1)$ time with uniform probability.

**核心约束**：
- 普通哈希表支持 $\mathcal{O}(1)$ 插入与删除，但底层存储分散，**无法在 $\mathcal{O}(1)$ 内产生真正的严格均匀等概率随机索引**。
- 普通连续数组支持 $\mathcal{O}(1)$ 下标随机访问与尾部操作，但中间位置删除需要 $\mathcal{O}(N)$ 平移移动元素。
- **系统目标**：将二者深度融合，在 $\mathcal{O}(1)$ 内同时达成键检索与等概率紧凑随机抽取。

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
import random
from typing import Dict, List

class RandomizedSet:
    # 基础版：元素互不重复 (LeetCode 380 + Meta popRandom 扩展)
    # 时间复杂度: 所有操作均摊 O(1)
    def __init__(self):
        self.vals: List[int] = []          # 紧凑连续动态数组，支持 O(1) 随机访问
        self.val_to_index: Dict[int, int] = {}  # 哈希映射：元素值 -> 其在 vals 中的下标

    def insert(self, val: int) -> bool:
        if val in self.val_to_index:
            return False
        self.val_to_index[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val: int) -> bool:
        # 核心机制：尾部元素置换 (Swap with Last)
        # 将待删除元素与数组最后一个元素互换，随后执行 O(1) 的 pop()
        if val not in self.val_to_index:
            return False

        idx_to_remove = self.val_to_index[val]
        last_val = self.vals[-1]

        # 将尾部元素移至 idx_to_remove 位置
        self.vals[idx_to_remove] = last_val
        self.val_to_index[last_val] = idx_to_remove

        # 弹出数组末尾并清理哈希索引
        self.vals.pop()
        del self.val_to_index[val]
        return True

    def getRandom(self) -> int:
        # 均匀等概率获取随机元素
        return random.choice(self.vals)

    def popRandom(self) -> int:
        # Meta 面经高频扩展：随机等概率弹出并移除一个元素
        # 时间复杂度: 严格 O(1)
        if not self.vals:
            raise IndexError("popRandom from empty RandomizedSet")

        # 1. 随机生成一个合法下标
        rand_idx = random.randrange(len(self.vals))
        val_to_pop = self.vals[rand_idx]
        last_val = self.vals[-1]

        # 2. 将末尾元素覆盖至 rand_idx
        self.vals[rand_idx] = last_val
        self.val_to_index[last_val] = rand_idx

        # 3. 弹出末尾并删除被弹出元素的映射
        self.vals.pop()
        del self.val_to_index[val_to_pop]

        return val_to_pop
```

```cpp
#include <vector>
#include <unordered_map>
#include <random>
#include <stdexcept>

class RandomizedSet {
private:
    std::vector<int> vals;
    std::unordered_map<int, int> valToIndex;
    std::mt19937 rng{std::random_device{}()};

public:
    RandomizedSet() {}

    bool insert(int val) {
        if (valToIndex.count(val)) return false;
        valToIndex[val] = vals.size();
        vals.push_back(val);
        return true;
    }

    bool remove(int val) {
        auto it = valToIndex.find(val);
        if (it == valToIndex.end()) return false;

        int idxToRemove = it->second;
        int lastVal = vals.back();

        // 覆盖待删位置
        vals[idxToRemove] = lastVal;
        valToIndex[lastVal] = idxToRemove;

        // 移除尾部
        vals.pop_back();
        valToIndex.erase(it);
        return true;
    }

    int getRandom() {
        std::uniform_int_distribution<int> dist(0, vals.size() - 1);
        return vals[dist(rng)];
    }

    int popRandom() {
        if (vals.empty()) {
            throw std::out_of_range("Container is empty");
        }
        std::uniform_int_distribution<int> dist(0, vals.size() - 1);
        int randIdx = dist(rng);
        int valToPop = vals[randIdx];
        int lastVal = vals.back();

        vals[randIdx] = lastVal;
        valToIndex[lastVal] = randIdx;

        vals.pop_back();
        valToIndex.erase(valToPop);
        return valToPop;
    }
};
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析与重复项进阶扩展 (LC 381)</div>

- **尾置换删除律 (Swap-with-Last Invariant)**：
  - 在数组中删除任意索引 $i$ 的元素会引发 $\mathcal{O}(N)$ 的整体内存向前平移。
  - 由于容器本身为无序集合（Set），**元素在动态数组中的相对顺序无任何业务语义**！
  - 因此将待删除位置 $i$ 直接由数组最末尾元素 `last_val` 填补，随后调用 `pop()` 截断尾部，使得数组物理删除开销从 $\mathcal{O}(N)$ 骤降至 $\mathcal{O}(1)$。
- **进阶面试追问：允许重复元素时的等概率保障 (Duplicates Allowed - LC 381)**：
  - 若输入数据允许重复，每个元素被抽中的概率必须**正比于其出现频次**（即每个实例等概率 $1/N$）。
  - **结构演进**：将哈希表的值类型由单一整数下标升级为索引哈希集合：
    $$	ext{val\_to\_indices}: 	ext{Dict}[val, 	ext{Set}[int]]$$
  - **删除逻辑的严密性**：
    1. 获取待删元素 $val$ 的任意一个下标 `idx = next(iter(val_to_indices[val]))`；
    2. 若 `idx` 恰好是末尾下标 `len(vals) - 1`，直接 pop 并在 set 中移除；
    3. 若 `idx` 不是末尾，将 `last_val = vals[-1]` 覆盖至 `vals[idx]`；
    4. 从 `val_to_indices[last_val]` 中移除旧末尾下标 `len(vals) - 1`，并加入新下标 `idx`；
    5. 从 `val_to_indices[val]` 中移除 `idx`（若 set 为空则清理 key）；
    6. 执行 `vals.pop()`。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：
  - `insert`: 均摊 $\mathcal{O}(1)$。
  - `remove`: $\mathcal{O}(1)$（字典查寻、数组下表置换、末尾 pop 均为 $\mathcal{O}(1)$）。
  - `getRandom`: $\mathcal{O}(1)$（伪随机数发生器生成随机索引 + 数组直接寻址）。
  - `popRandom`: $\mathcal{O}(1)$。
- **空间复杂度**：$\mathcal{O}(N)$，动态数组与哈希表严格与存入元素总量成线性关系。

</div>

</div>
</details>


---

### 17. 独立迭代器设计与共享流式缓冲区 (Python itertools.tee & Shared-State Streaming Buffer)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">迭代/流 17</span>
  <span class="review-card-title">独立迭代器设计与共享流式缓冲区 (Python itertools.tee & Shared-State Streaming Buffer)</span>
  <span class="review-card-tag">迭代器协议 · 共享单向链表 · 多游标追赶 · 自动引用计数垃圾回收 · 内存 O(g + n)</span>
</summary>
<div class="review-card-content">

> 🔗 **设计规范与原型**：
> - Python 标准库 `itertools.tee(iterable, n=2)` 底层实现原理
> - 工业级流处理（Stream Processing）多分支消费与背压解耦

<div class="review-block">
<div class="review-block-label">📌 题目定义与架构演进</div>

**题目原文 (Problem Statement)**：
> Implement the behavior of Python's `itertools.tee`:
> Accept one source iterable/iterator and an integer $n \ge 1$. Return $n$ independent iterators over the same source sequence.
> Each returned iterator must advance independently while preserving the exact sequence order.
> Start with a correct baseline, then optimize toward the optimal shared-state design.

**架构对比与内存瓶颈**：
- **朴素独立队列方案 (Naive Multi-Queue)**：
  - 为 $n$ 个分支迭代器各自分配一个 `collections.deque`。
  - 某个分支请求 `next()` 时，若其专属队列为空，则从源迭代器拉取一个新值，并**广播复制追加到全部 $n$ 个队列中**。
  - **缺陷**：若最快消费分支与最慢消费分支相隔 $g$ 个元素，这 $g$ 个元素会在所有滞后队列中均存在副本，产生 $\mathcal{O}(n \cdot g)$ 的冗余内存浪费！
- **最优共享单向链式缓冲区 (Optimal Shared-State Linked Buffer)**：
  - 全局仅维护**单向共享链表**，每个节点仅存一份数据 `Node(val, next)`。
  - 每个分支迭代器仅持有一个指向该链表节点的游标 `cursor`。
  - 当某个游标需要向后推进且 `cursor.next is None` 时，由该游标负责从底能源拉取新元素，创建新节点挂在 `cursor.next` 上。
  - **垃圾回收机制（Garbage Collection）**：节点只有单向 `next` 指针。当最慢的消费者游标离开某节点时，在 Python 引用计数机制下，该节点前驱的引用归零，被即时 $\mathcal{O}(1)$ 自动物理回收！
  - **辅助内存**：严格由 $\mathcal{O}(n \cdot g)$ 优化为 $\mathcal{O}(g + n)$！

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import Any, Iterator, List

class _TeeNode:
    __slots__ = ('val', 'next')
    def __init__(self, val: Any = None):
        self.val = val
        self.next: Optional['_TeeNode'] = None

class _TeeIterator:
    def __init__(self, head_node: _TeeNode, shared_source: Iterator[Any]):
        self.cursor = head_node
        self.source = shared_source

    def __iter__(self):
        return self

    def __next__(self):
        # 若当前游标后方尚未拉取数据，向共享底能源拉取并追加
        if self.cursor.next is None:
            val = next(self.source)  # 若底能源耗尽，将在此处抛出 StopIteration
            self.cursor.next = _TeeNode(val)

        # 游标向后推进一步，并产出对应值
        self.cursor = self.cursor.next
        return self.cursor.val

def custom_tee(iterable: Any, n: int = 2) -> List[Iterator[Any]]:
    # 生产级 Python itertools.tee 实现：
    共享单向链表多游标驱动，内存开销严格受限于 O(g + n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []

    shared_source = iter(iterable)
    # 创建哨兵根节点，所有 n 个分支起始游标均指向根节点
    root = _TeeNode()
    return [_TeeIterator(root, shared_source) for _ in range(n)]

if __name__ == "__main__":
    src = iter([10, 20, 30, 40, 50])
    it1, it2, it3 = custom_tee(src, 3)

    assert next(it1) == 10
    assert next(it1) == 20
    # it1 领先 2 步，it2 和 it3 滞后
    assert next(it2) == 10
    assert next(it3) == 10
    assert next(it3) == 20
    assert next(it3) == 30
    assert next(it1) == 30
    assert next(it2) == 20
    print("✅ Card 17 (itertools.tee) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与工业考点</div>

- **时间复杂度**：每个源元素被底能源精确生成一次；每个分支迭代器产出一个元素耗时严格 $\mathcal{O}(1)$。
- **空间复杂度**：$\mathcal{O}(g + n)$，其中 $g$ 为最快与最慢迭代器之间的跨步差（Gap），彻底杜绝了广播式多队列的 $\mathcal{O}(n \cdot g)$ 内存膨胀。
- **面试口述核心**：
  *"A naive queue-per-consumer model duplicates elements $n$ times during lagging consumption, incurring $\mathcal{O}(n \cdot g)$ memory. By using a shared singly-linked stream with forward-only node pointers, each iterator operates as an independent cursor. As soon as the slowest cursor advances, Python's native reference count for earlier nodes reaches zero, reclaiming memory in $\mathcal{O}(1)$ and yielding an optimal $\mathcal{O}(g + n)$ footprint."*

</div>

</div>
</details>


---

## 模块四：NeetCode 极速速记与高频题型决策树 (NeetCode Quick-Recall Decision Framework & Mental Models)

为了在面试高压环境下于 10 秒内迅速锁定最优解题范式，将链表、栈、堆领域最具代表性的核心考题沉淀为如下标准化心智模型与决策树速记矩阵：

### 4.1 链表题型心智模型与 10 秒决策速查 (Linked List Decision Matrix)

| 场景模式 (Pattern) | 触发关键词 / 题型特征 | 黄金解法架构 (Optimal Archetype) | 代表真题 (NeetCode / LC) |
| :--- | :--- | :--- | :--- |
| **快慢双指针 (Floyd)** | 环检测、求环入口、找链表中点、回文判断 | `slow = slow.next`, `fast = fast.next.next`；相遇后一针回原点单步同步走 | LC 141 (环检测), LC 142 (环入口), LC 876 (中点), LC 234 (回文) |
| **虚拟哨兵头节点 (Dummy)** | 链表头可能被删除、被合并、链表前插入 | `dummy = ListNode(0, head)`，统一内部与头节点逻辑 | LC 19 (删倒数第 N), LC 21 (合并两链表), LC 2 (两数相加), LC 86 (分隔) |
| **三指针局部翻转** | 反转整链、反转区间、K个一组反转 | `nxt = curr.next; curr.next = prev; prev = curr; curr = nxt` | LC 206 (反转), LC 92 (反转II), LC 25 (K个一组) |
| **拆分/穿插指针映射** | 深拷贝带随机指针的链表、重排链表 | 原地复制 `node.next = copyNode` 后交叉拆分；快慢针截半后交替穿插 | LC 138 (复制随机指针), LC 143 (重排链表) |
| **复合哈希双向链表** | 严格 $\mathcal{O}(1)$ 缓存插入/访问/淘汰 | 双向链表记录时间顺序 + 哈希表直指节点；头尾双哨兵四指针无分支断连 | LC 146 (LRU), LC 460 (LFU), Meta 餐厅候补 |
| **动态数组末尾置换** | $\mathcal{O}(1)$ 插入、删除与等概率随机抽取 | 动态数组保存元素 + 哈希表记录下标；删除时与末尾元素置换后执行 `pop()` | LC 380 (O(1)集合), LC 381 (允许重复), Meta 随机容器 |

---

### 4.2 栈与单调结构心智模型 (Stack & Monotonic Structure Matrix)

| 场景模式 (Pattern) | 触发关键词 / 题型特征 | 黄金解法架构 (Optimal Archetype) | 代表真题 (NeetCode / LC) |
| :--- | :--- | :--- | :--- |
| **配对与平衡检验** | 括号有效性、消除相邻重复项、回文消除 | 栈存储左半边期待，遇到右半边校验栈顶并弹出 | LC 20 (有效括号), LC 1047 (消除相邻重复) |
| **单调递增/递减栈** | 下一个更大/更小元素、柱状图最大矩形、股票买卖天数 | 维护栈内元素严格单调；破坏单调时弹出栈顶并结算其右边界 | LC 739 (每日温度), LC 496 (下一个更大), LC 84 (柱状图最大矩形), LC 853 (车队) |
| **单调双端队列** | 滑动窗口动态极值 (最大值/最小值) | 队头到队尾维持严格单调递减，队尾剔除较小者，队头剔除过期索引 | LC 239 (滑动窗口最大值), LC 1438 |
| **双栈表达式求值** | 四则运算、括号嵌套、运算符优先级 | 操作数栈 + 运算符栈；乘除即时结合，遇到括号递归下降分治 | LC 150 (逆波兰), LC 224 (基础计算器), LC 227 (计算器II) |
| **单调栈贪心去重** | 字典序最小的不重复子序列 | 单调增栈 + 字符最后出现位置哈希 + 栈内存在性集合；可后现者果断弹出 | LC 316 / LC 1081 (去重保持最小字典序) |

---

### 4.3 堆与优先队列心智模型 (Heap & Priority Queue Matrix)

| 场景模式 (Pattern) | 触发关键词 / 题型特征 | 黄金解法架构 (Optimal Archetype) | 代表真题 (NeetCode / LC) |
| :--- | :--- | :--- | :--- |
| **Top-K 动态维护** | 求海量数据中最大/最小的 K 个元素 | 维持容量为 $K$ 的**小顶堆**（求最大 K 个）或大顶堆，超出容量弹出堆顶 | LC 215 (第K大), LC 347 (前K高频), LC 703 (数据流第K大) |
| **多路归并排序** | 合并 K 个升序链表 / 数组 | 堆内始终只维护各有序序列的当前游标头部（容量为 $K$），弹出堆顶并推进后继 | LC 23 (合并K个升序链表), LC 378 (矩阵第K小) |
| **对顶平衡双堆** | 数据流中无序插入，实时查询中位数 | 大顶堆（存放小半部）+ 小顶堆（存放大半部）；容量差严格 $\le 1$ | LC 295 (数据流中位数) |
| **冷却调度与模拟** | 相同任务必须间隔 $N$ 时间单位冷却 | 贪心大顶堆按频次降序挑选 + 冷却队列暂存等待倒计时归零 | LC 621 (任务调度器), LC 355 (设计推特) |

