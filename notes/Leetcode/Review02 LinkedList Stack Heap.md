# 复习卡片：链表、栈与堆 (Review Flashcards · Linked List, Stack & Heap)

本篇为算法面试高频复习卡片第二辑：系统整理**链表与复合哈希结构 (Linked List & Hash-Linked Structures)**、**栈与单调结构 (Stack & Monotonic Stack / Deque)** 以及**堆与优先队列 (Heap & Priority Queue)** 的核心高频考题、工业级变体、生产级实现与时空复杂度全景。

---

---

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
if __name__ == "__main__":
    rw = RestaurantWaitlist(max_party_size=10)
    rw.join("Alice", 2)
    rw.join("Bob", 4)
    rw.join("Carol", 6)
    assert rw.find_first_match(3) == "Alice"
    assert rw.find_first_match(5) == "Alice"
    assert rw.delete("Alice") is True
    assert rw.find_first_match(5) == "Bob"
    assert rw.find_first_match(2) is None
    print("✅ RestaurantWaitlist tests passed!")
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

```python
import bisect
import collections
from typing import Dict, List

class RobotLogger:
    """
    乱序时间戳日志限流器
    时间复杂度: 单次判定 O(log K) (K 为历史放行条数)
    空间复杂度: O(M * K) (M 为独立 message 数量)
    """
    def __init__(self, window_seconds: int = 10):
        self.window = window_seconds
        self.approved: Dict[str, List[int]] = collections.defaultdict(list)

    def should_print_message(self, timestamp: int, message: str) -> bool:
        ts_list = self.approved[message]
        idx = bisect.bisect_left(ts_list, timestamp)

        # 同一时间戳重复调用: 拦截
        if idx < len(ts_list) and ts_list[idx] == timestamp:
            return False

        # 因果前驱判定: 检查严格前驱 pred 是否在 [timestamp - 10, timestamp) 区间
        if idx > 0:
            pred = ts_list[idx - 1]
            if timestamp - pred < self.window:
                return False

        # 放行并维持有序集合
        ts_list.insert(idx, timestamp)
        return True
if __name__ == "__main__":
    logger = RobotLogger(10)
    assert logger.should_print_message(12, "foo") is True
    assert logger.should_print_message(10, "foo") is True # 乱序 10 判定前驱，不被后来的 12 封杀
    assert logger.should_print_message(15, "foo") is False # 15 - 12 < 10 拦截
    assert logger.should_print_message(25, "foo") is True
    print("✅ RobotLogger tests passed!")
```



#### 3. 基于 Idempotency-Key 的幂等 API 处理器 (Idempotency API with Concurrency & TTL)
- **核心业务契约与防重约束**：
  - **首次执行**：携带唯一 Key $K$ 与请求载荷 $B$，正常执行耗时业务函数并返回响应 $R$，持久化映射 $(K \to R)$；
  - **重试命中**：相同 Key $K$ 与相同 Payload $B$ 再次请求时，直接返回已持久化的缓存响应 $R$，杜绝重复下单或二次扣款；
  - **Payload 篡改拦截**：相同 Key $K$ 但携带不同 Body $B'$，必须严格拦截并抛出 `409 Conflict`（防止重放与键非法复用）；
  - **TTL 自动驱逐**：记录配置生存时间（如 24 小时），过期后自动失效，支持惰性校验与周期清理。
- **并发控制的两阶段锁架构 (Two-Phase Lock Protocol)**：
  - **常见致命缺陷**：若在持全局锁 `with self.lock:` 内部直接调用 `rec.condition.wait()`，会引发两大灾难：
    1. **未获底层锁异常**：Python `threading.Condition` 默认持独立的递归锁，未进入 `with rec.condition:` 直接调用 `wait()` 将抛出 `RuntimeError: cannot wait on un-acquired lock`；
    2. **全局锁饥饿与死锁 (Deadlock)**：持全局锁挂起会导致所有其他完全不相干 Key 的并发请求被全量阻塞；若执行线程抛出异常尝试获取 `self.lock` 清理缓存，将发生死锁！
  - **工业级两阶段协作设计**：
    - **阶段 1（全局状态登记，持有全局锁）**：在 `self.lock` 保护下只做轻量状态查询、过期清理与 `IN_FLIGHT` 占位登记，完成状态交接后**立即释放全局锁**；
    - **阶段 2（记录级细粒度等待与执行，全局锁外）**：
      - 领头者线程（Leader）：在无锁状态下执行耗时业务逻辑 `execute_fn(body)`，完成后在记录专属的 `rec.condition` 上置为 `DONE` 并唤醒所有等待者；
      - 追随者线程（Follower）：在记录专属的 `with rec.condition:` 上挂起等待（`while status == IN_FLIGHT: wait()`），绝不占用全局锁。

```python
import hashlib, json, time, threading
from enum import Enum
from typing import Dict, Any, Tuple

class RequestStatus(Enum):
    IN_FLIGHT = 1
    DONE = 2
    FAILED = 3

class IdempotencyRecord:
    def __init__(self, body_hash: str, ttl_seconds: float):
        self.body_hash = body_hash
        self.status = RequestStatus.IN_FLIGHT
        self.response = None
        self.expires_at = time.time() + ttl_seconds
        self.condition = threading.Condition() # 每条记录独立的条件变量

class IdempotencyManager:
    """
    两阶段细粒度并发安全幂等管理器
    """
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

        # 阶段 1: 在全局锁保护下原子化登记/读取状态，极速释放全局锁
        with self.lock:
            if idempotency_key in self.store:
                rec = self.store[idempotency_key]
                if now > rec.expires_at:
                    del self.store[idempotency_key]
                elif rec.body_hash != body_hash:
                    return 409, {"error": "Idempotency key re-used with different payload"}
                elif rec.status == RequestStatus.DONE:
                    return 200, rec.response
                else:
                    # 正在处理中 (IN_FLIGHT)：记录引用，移至全局锁外等待
                    wait_rec = rec
                    is_leader = False

            if idempotency_key not in self.store:
                rec = IdempotencyRecord(body_hash, self.default_ttl)
                self.store[idempotency_key] = rec
                wait_rec = None
                is_leader = True

        # 阶段 2: 全局锁外协作
        if not is_leader:
            # 追随者线程：释放全局锁后，仅在专属记录条件变量上等待，绝不阻塞其他不相干 Key
            with wait_rec.condition:
                while wait_rec.status == RequestStatus.IN_FLIGHT:
                    wait_rec.condition.wait()
                if wait_rec.status == RequestStatus.DONE:
                    return 200, wait_rec.response
                else:
                    return 500, {"error": "Upstream request failed, please retry"}

        # 领头者线程：在无锁状态下执行耗时业务逻辑
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
                rec.status = RequestStatus.FAILED
                rec.condition.notify_all()
            raise e

if __name__ == "__main__":
    mgr = IdempotencyManager(default_ttl=10.0)
    exec_count = 0
    test_lock = threading.Lock()

    def slow_business(payload):
        nonlocal exec_count
        with test_lock:
            exec_count += 1
        time.sleep(0.05) # 模拟 I/O 耗时
        return {"status": "success", "order_id": 999}

    # 测试并发 5 个相同 Key 请求
    results = []
    def worker():
        code, res = mgr.handle_request("order_key_1", {"amount": 100}, slow_business)
        results.append((code, res))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    # 验证 5 个并发请求仅执行 1 次业务函数，其余全部安全等待并获取缓存结果
    assert exec_count == 1
    assert len(results) == 5
    for code, res in results:
        assert code == 200
        assert res["order_id"] == 999

    # 测试 Payload 篡改防御 (409 Conflict)
    c_conflict, _ = mgr.handle_request("order_key_1", {"amount": 200}, slow_business)
    assert c_conflict == 409
    print("✅ IdempotencyManager concurrent tests passed!")
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

```python
from typing import Dict, Optional

class MemBlock:
    def __init__(self, offset: int, size: int, is_free: bool = True):
        self.offset = offset
        self.size = size
        self.is_free = is_free
        self.prev: Optional['MemBlock'] = None
        self.next: Optional['MemBlock'] = None

class FirstFitMemoryAllocator:
    """
    首次适应连续内存分配器 (First-Fit Allocator)
    时间复杂度: allocate O(N), free 严格 O(1) 双向链表相邻合并
    空间复杂度: O(N) 块元数据
    """
    def __init__(self, total_size: int):
        self.head = MemBlock(0, total_size, is_free=True)
        self.allocated: Dict[int, MemBlock] = {}

    def allocate(self, size: int) -> int:
        if size <= 0:
            return -1
        curr = self.head
        while curr:
            if curr.is_free and curr.size >= size:
                remainder = curr.size - size
                curr.size = size
                curr.is_free = False
                self.allocated[curr.offset] = curr

                # 切分空闲块
                if remainder > 0:
                    split_block = MemBlock(curr.offset + size, remainder, is_free=True)
                    split_block.next = curr.next
                    split_block.prev = curr
                    if curr.next:
                        curr.next.prev = split_block
                    curr.next = split_block
                return curr.offset
            curr = curr.next
        return -1 # 空间不足 (OOM)

    def free(self, offset: int) -> bool:
        if offset not in self.allocated:
            return False
        node = self.allocated.pop(offset)
        node.is_free = True

        # 向右合并空闲块
        if node.next and node.next.is_free:
            right = node.next
            node.size += right.size
            node.next = right.next
            if right.next:
                right.next.prev = node

        # 向左合并空闲块
        if node.prev and node.prev.is_free:
            left = node.prev
            left.size += node.size
            left.next = node.next
            if node.next:
                node.next.prev = left
        return True

if __name__ == "__main__":
    alloc = FirstFitMemoryAllocator(100)
    p1 = alloc.allocate(30)
    p2 = alloc.allocate(40)
    assert p1 == 0 and p2 == 30
    assert alloc.free(p1) is True # 释放 [0, 30]
    assert alloc.free(p2) is True # 释放 [30, 70]，向左合并为 [0, 100]
    p3 = alloc.allocate(90)
    assert p3 == 0
    print("✅ FirstFitMemoryAllocator tests passed!")
```


#### 5. 并发线程安全与分布式分片架构
- **单机并发安全**：
  - 粗粒度保护：全局 `threading.RLock()` 封装 `get`/`put`。
  - 细粒度分片（Sharded Cache / ConcurrentHashMap 模式）：
    将大缓存水平切分为 $M$ 个独立分片（Shard），依据 $	ext{hash}(key) \pmod M$ 路由到具体分片。各分片持独立的锁与双向链表，使并发写吞吐随核心数线性扩展（注意：此时退化为分片局部的近拟 LRU）。
- **跨机器分布式缓存**：
  - 采用**一致性哈希环（Consistent Hashing with Virtual Nodes）**实现节点弹性扩缩容；单机节点内部继续运行双向链表+哈希表的纯粹 LRU 引擎。

```python
import hashlib
import threading
import collections
from typing import Any, Optional

class ShardedLRUCache:
    """
    细粒度分片并发安全 LRU 缓存 (Striped Locks)
    """
    def __init__(self, total_capacity: int, num_shards: int = 16):
        self.num_shards = num_shards
        self.shard_cap = max(1, total_capacity // num_shards)
        self.shards = [collections.OrderedDict() for _ in range(num_shards)]
        self.locks = [threading.Lock() for _ in range(num_shards)]

    def _shard_index(self, key: str) -> int:
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % self.num_shards

    def get(self, key: str) -> Optional[Any]:
        idx = self._shard_index(key)
        with self.locks[idx]:
            cache = self.shards[idx]
            if key not in cache:
                return None
            cache.move_to_end(key)
            return cache[key]

    def put(self, key: str, val: Any) -> None:
        idx = self._shard_index(key)
        with self.locks[idx]:
            cache = self.shards[idx]
            if key in cache:
                cache.move_to_end(key)
            cache[key] = val
            if len(cache) > self.shard_cap:
                cache.popitem(last=False)
if __name__ == "__main__":
    sc = ShardedLRUCache(total_capacity=16, num_shards=4)
    sc.put("k1", 100)
    sc.put("k2", 200)
    assert sc.get("k1") == 100
    assert sc.get("k2") == 200
    assert sc.get("nonexistent") is None
    print("✅ ShardedLRUCache tests passed!")
```



#### 6. 内存分配器 O(log m) 进阶与四向相邻合并 (Memory Allocator with O(log m) Size Indexing)
- **从 O(N) 线性扫描到 O(log m) 工业级跃迁**：
  - 在高碎片化场景下，对空闲块链表执行 $\mathcal{O}(N)$ 首次适应扫描无法满足高频分配要求。
  - **核心双索引架构**：
    1. **空闲块大小索引 (Size-Keyed Balanced Tree / Sorted Sizes)**：以块大小作为键，可在 $\mathcal{O}(\log m)$ 内二分定位首个满足 $\text{block.size} \ge size$ 的最佳空闲块（Best-Fit）；
    2. **全量物理地址链表 (Address-Ordered Doubly Linked List)**：所有块（无论空闲或已分配）按物理内存地址严格升序串联在一条双向链表中。
- **释放内存时的四大完备合并分支 (Four Coalescing Scenarios on Free)**：
  调用 `free(offset)` 时，检查物理相邻节点：
  1. **左右均已占用 (No Neighbour Free)**：直接将当前块标记为 `is_free = True`，插入大小索引；
  2. **仅左邻居空闲 (Left Neighbour Free)**：左块吞并当前块，从大小索引更新左块大小；
  3. **仅右邻居空闲 (Right Neighbour Free)**：当前块吞并右块，从大小索引注销右块；
  4. **左右邻居均空闲 (Both Neighbours Free)**：左块、当前块、右块三合一！左块大小累加三者总和，彻底注销右块。

```python
import bisect
from typing import Dict, List, Optional, Set

class SizeIndexedMemoryAllocator:
    """
    带大小索引的工业级内存分配器 (O(log m) 最佳适应查找 + O(1) 双向链表物理合并)
    """
    def __init__(self, total_size: int):
        self.head = MemBlock(0, total_size, is_free=True)
        self.allocated: Dict[int, MemBlock] = {}
        self.free_by_size: Dict[int, Set[MemBlock]] = {total_size: {self.head}}
        self.sorted_sizes: List[int] = [total_size]

    def _add_free_index(self, block: MemBlock) -> None:
        s = block.size
        if s not in self.free_by_size:
            self.free_by_size[s] = set()
            bisect.insort(self.sorted_sizes, s)
        self.free_by_size[s].add(block)

    def _remove_free_index(self, block: MemBlock) -> None:
        s = block.size
        if s in self.free_by_size and block in self.free_by_size[s]:
            self.free_by_size[s].remove(block)
            if not self.free_by_size[s]:
                del self.free_by_size[s]
                idx = bisect.bisect_left(self.sorted_sizes, s)
                if idx < len(self.sorted_sizes) and self.sorted_sizes[idx] == s:
                    self.sorted_sizes.pop(idx)

    def allocate(self, size: int) -> int:
        if size <= 0:
            return -1
        # 二分查找最佳适应块 (Best-Fit)
        idx = bisect.bisect_left(self.sorted_sizes, size)
        if idx >= len(self.sorted_sizes):
            return -1

        target_size = self.sorted_sizes[idx]
        block = next(iter(self.free_by_size[target_size]))
        self._remove_free_index(block)

        remainder = block.size - size
        block.size = size
        block.is_free = False
        self.allocated[block.offset] = block

        if remainder > 0:
            split = MemBlock(block.offset + size, remainder, is_free=True)
            split.next = block.next
            split.prev = block
            if block.next:
                block.next.prev = split
            block.next = split
            self._add_free_index(split)

        return block.offset

    def free(self, offset: int) -> bool:
        if offset not in self.allocated:
            return False
        node = self.allocated.pop(offset)
        node.is_free = True

        # 合并右侧
        if node.next and node.next.is_free:
            right = node.next
            self._remove_free_index(right)
            node.size += right.size
            node.next = right.next
            if right.next:
                right.next.prev = node

        # 合并左侧
        if node.prev and node.prev.is_free:
            left = node.prev
            self._remove_free_index(left)
            left.size += node.size
            left.next = node.next
            if node.next:
                node.next.prev = left
            node = left

        self._add_free_index(node)
        return True

if __name__ == "__main__":
    sia = SizeIndexedMemoryAllocator(100)
    b1 = sia.allocate(20)
    b2 = sia.allocate(50)
    assert b1 == 0 and b2 == 20
    assert sia.free(b1) is True
    assert sia.free(b2) is True
    assert sia.allocate(95) == 0
    print("✅ SizeIndexedMemoryAllocator tests passed!")
```


#### 7. LRU + LFU + 策略模式可插拔淘汰引擎 (Pluggable Eviction Strategy Pattern)
- **架构解耦核心**：将底层键值容器存储与上层淘汰策略彻底剥离。
- **策略接口规范**：
  - `on_access(key)`：缓存读写命中时回调，由策略更新对应的时间戳、频次桶或权重；
  - `pick_victim() -> key`：当容器容量溢出时，由策略提名下一个被淘汰的牺牲品键；
  - `on_evict(key)`：执行物理淘汰后同步清理策略内部的辅助结构。
- **策略实现在同一框架下的统一**：
  - `LRUPolicy`：维护单条按访问时间倒序的双向链表，`on_access` 移动至表头，`pick_victim` 选表尾；
  - `LFUPolicy`：维护 `freq_buckets: Dict[int, DLL]` 与 `min_freq` 指针。`on_access` 将节点自旧频次桶移入新频次桶并更新 `min_freq`；`pick_victim` 从 `freq_buckets[min_freq]` 的尾部选出 LRU 节点，严格保证同频次下的 LRU 二级平局打破；
  - `TTLWeightedPolicy` / `SizeWeightedPolicy`：策略对象内部持小顶堆或按权重索引的跳表，无缝注入主缓存。

```python
import collections
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class EvictionPolicy(ABC):
    @abstractmethod
    def on_access(self, key: str) -> None: ...
    @abstractmethod
    def pick_victim(self) -> Optional[str]: ...
    @abstractmethod
    def on_evict(self, key: str) -> None: ...

class LRUPolicy(EvictionPolicy):
    def __init__(self):
        self.od = collections.OrderedDict()
    def on_access(self, key: str) -> None:
        self.od[key] = None; self.od.move_to_end(key)
    def pick_victim(self) -> Optional[str]:
        return next(iter(self.od)) if self.od else None
    def on_evict(self, key: str) -> None:
        self.od.pop(key, None)

class LFUPolicy(EvictionPolicy):
    """O(1) 频次桶双向链表 LFU 策略，二级平局采用 LRU 打破"""
    def __init__(self):
        self.key_to_freq: Dict[str, int] = {}
        self.freq_to_keys: Dict[int, collections.OrderedDict] = collections.defaultdict(collections.OrderedDict)
        self.min_freq = 0

    def on_access(self, key: str) -> None:
        if key in self.key_to_freq:
            f = self.key_to_freq[key]
            del self.freq_to_keys[f][key]
            if not self.freq_to_keys[f]:
                del self.freq_to_keys[f]
                if self.min_freq == f:
                    self.min_freq += 1
            new_f = f + 1
        else:
            new_f = 1
            self.min_freq = 1
        self.key_to_freq[key] = new_f
        self.freq_to_keys[new_f][key] = None

    def pick_victim(self) -> Optional[str]:
        if not self.key_to_freq:
            return None
        return next(iter(self.freq_to_keys[self.min_freq]))

    def on_evict(self, key: str) -> None:
        if key in self.key_to_freq:
            f = self.key_to_freq.pop(key)
            del self.freq_to_keys[f][key]
            if not self.freq_to_keys[f]:
                del self.freq_to_keys[f]

class PluggableCache:
    def __init__(self, capacity: int, policy: EvictionPolicy):
        self.capacity = capacity
        self.policy = policy
        self.store: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        if key not in self.store: return None
        self.policy.on_access(key)
        return self.store[key]

    def put(self, key: str, val: Any) -> None:
        if key in self.store:
            self.store[key] = val
            self.policy.on_access(key)
            return
        if len(self.store) >= self.capacity:
            victim = self.policy.pick_victim()
            if victim:
                del self.store[victim]
                self.policy.on_evict(victim)
        self.store[key] = val
        self.policy.on_access(key)
if __name__ == "__main__":
    # 1. 测试 LRU 策略
    lru_cache = PluggableCache(capacity=2, policy=LRUPolicy())
    lru_cache.put("a", 1); lru_cache.put("b", 2)
    assert lru_cache.get("a") == 1 # a 变为 MRU
    lru_cache.put("c", 3) # 驱逐 b
    assert lru_cache.get("b") is None
    assert lru_cache.get("a") == 1

    # 2. 测试 LFU 策略 (二级平局用 LRU 打破)
    lfu_cache = PluggableCache(capacity=2, policy=LFUPolicy())
    lfu_cache.put("x", 10); lfu_cache.put("y", 20)
    lfu_cache.get("x"); lfu_cache.get("x") # freq(x)=3, freq(y)=1
    lfu_cache.put("z", 30) # 驱逐最低频次 y
    assert lfu_cache.get("y") is None
    assert lfu_cache.get("x") == 10
    print("✅ PluggableCache tests passed!")
```



#### 8. 带预写日志与崩溃恢复的持久化缓存 (Durable In-Memory Cache with WAL & Replay)
- **函数参数规范化哈希 Bug 修复**：
  - 针对通用装饰器 `generate_key(*args, **kwargs)`，直接 `hash((args, kwargs))` 会因 `kwargs` 是字典而抛出 `TypeError: unhashable type: 'dict'`，且字典键值对遍历顺序可能导致相同参数产生不同键。
  - **规范化防线**：递归将参数中的 `list/dict` 转换为不可变元组，或采用标准 JSON 规范化：`json.dumps(args) + json.dumps(kwargs, sort_keys=True)`，保证入参键严格确定。
- **预写日志 (Write-Ahead Logging, WAL) 与崩溃重放恢复**：
  - 每次写操作或读命中，以追加写（Append-Only）方式向磁盘文件写入日志行：`{"op": "PUT", "key": k, "val": v, "ts": now}`。
  - **重放核心防坑点**：重启恢复时，仅仅将最新值写入字典会导致 LRU 淘汰序退化为 FIFO！**在重放回放日志时，对于已存在的 key，必须显式调用 `move_to_end`**，确保按日志中最后一次出现的先后顺序完全复现崩溃前的物理 LRU 队列。
  - **I/O 吞吐权衡**：单次写入同步 `fsync` 确保零数据丢失（金融级） vs. Group Commit 批量刷盘（吞吐优先）；周期性内存快照（Snapshot）截断并压缩（Truncate）历史 WAL。

```python
import json, os, collections
from typing import Any, Optional

class DurableLRUCache:
    """
    带预写日志 (WAL) 与崩溃恢复的持久化 LRU 缓存
    """
    def __init__(self, capacity: int, wal_path: str):
        self.capacity = capacity
        self.wal_path = wal_path
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self._recover()

    def _recover(self) -> None:
        if not os.path.exists(self.wal_path):
            return
        with open(self.wal_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                op, key = rec.get("op"), rec.get("key")
                if op == "PUT":
                    self.cache[key] = rec["val"]
                    self.cache.move_to_end(key) # 恢复精确物理时序!
                    if len(self.cache) > self.capacity:
                        self.cache.popitem(last=False)
                elif op == "ACCESS":
                    if key in self.cache:
                        self.cache.move_to_end(key)

    def put(self, key: str, val: Any) -> None:
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"op": "PUT", "key": key, "val": val}) + "\n")
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = val
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"op": "ACCESS", "key": key}) + "\n")
        self.cache.move_to_end(key)
        return self.cache[key]
if __name__ == "__main__":
    import tempfile
    wal_f = tempfile.mktemp()
    w1 = DurableLRUCache(capacity=2, wal_path=wal_f)
    w1.put("k1", "v1")
    w1.put("k2", "v2")
    w1.get("k1") # k1 变为 MRU
    w1.put("k3", "v3") # 空间不足驱逐 k2
    # 崩溃重启回放
    w2 = DurableLRUCache(capacity=2, wal_path=wal_f)
    assert w2.get("k2") is None # 确认 k2 已被物理淘汰
    assert w2.get("k1") == "v1"
    assert w2.get("k3") == "v3"
    if os.path.exists(wal_f): os.remove(wal_f)
    print("✅ DurableLRUCache tests passed!")
```



#### 9. 四层级内存数据库系统实现 (Multi-Level In-Memory Database Engine)
- **核心业务需求与四层级渐进演进**：
  - **Level 1 (基础键值对存储)**：每个顶层 `key` 映射多个 `field -> value` 键值对（均为字符串）。提供基础 CRUD 接口：`set(key, field, value)`，`get(key, field)`，`delete(key, field) -> bool`。
  - **Level 2 (字典序与前缀扫描)**：`scan(key)` 按 `field` 的字典序升序返回所有有效字段及值 `["field(value)", ...]`；`scan_by_prefix(key, prefix)` 在字典序基础上增加前缀匹配过滤。
  - **Level 3 (时序与 TTL 字段失效)**：所有操作扩展为 `_at` 时序变体。`set_at_with_ttl(key, field, value, timestamp, ttl)` 定义有效生命周期闭开区间 $[timestamp, timestamp + ttl)$。过期字段在读取时自动对外部不可见并惰性清理。
  - **Level 4 (快照备份与时钟重定位恢复)**：
    - `backup(timestamp)`：对数据库当前所有未过期的活跃字段创建深拷贝（Deep Copy）快照，保存每个字段相对于当前时刻的**剩余生存时间 (Remaining TTL)**：$\Delta = (ts_{set} + ttl) - timestamp$；
    - `restore(timestamp, timestamp_to_restore)`：查找满足 $ts_{backup} \le timestamp\_to\_restore$ 的最近一份历史快照，在当前时刻 $timestamp$ 恢复该快照。**生命周期重定位**：所有带 TTL 字段的过期时间被重新计算为 $new\_expiry = timestamp + \Delta$！
- **关键架构陷阱与不变量设计**：
  1. **禁止持久化绝对过期时间戳**：若在快照中直接存储绝对过期时间戳 $ts_{expiry}$，则在跨越较长时间恢复快照时，原本未过期的记录会被错误判定为已过期。通过存储相对剩余时长 $\Delta$，恢复时以新时间轴重新起算，严格保证数据恢复的时钟隔离性。
  2. **深拷贝防污染**：快照必须完全复制值与元数据，切忌引用原字典，防止后续的实时写操作脏污历史备份。

```python
import collections
from typing import Dict, List, Optional, Tuple

class FieldRecord:
    __slots__ = ('val', 'remaining_ttl', 'absolute_expiry')
    def __init__(self, val: str, remaining_ttl: Optional[int] = None, absolute_expiry: Optional[int] = None):
        self.val = val
        self.remaining_ttl = remaining_ttl
        self.absolute_expiry = absolute_expiry

class InMemoryDatabase:
    """
    四层级内存数据库完整参考实现
    Level 1: set, get, delete
    Level 2: scan, scan_by_prefix (字典序排列)
    Level 3: _at 时序接口与 TTL 失效 [ts, ts + ttl)
    Level 4: backup(ts) 与 restore(ts, ts_to_restore) 相对 TTL 恢复
    """
    def __init__(self):
        self.store: Dict[str, Dict[str, FieldRecord]] = collections.defaultdict(dict)
        self.backups: List[Tuple[int, Dict[str, Dict[str, Tuple[str, Optional[int]]]]]] = []

    def _is_alive(self, rec: FieldRecord, ts: Optional[int]) -> bool:
        if rec.absolute_expiry is None or ts is None:
            return True
        return ts < rec.absolute_expiry

    # --- Level 1 ---
    def set(self, key: str, field: str, value: str) -> None:
        self.store[key][field] = FieldRecord(value)

    def get(self, key: str, field: str) -> Optional[str]:
        if key in self.store and field in self.store[key]:
            return self.store[key][field].val
        return None

    def delete(self, key: str, field: str) -> bool:
        if key in self.store and field in self.store[key]:
            del self.store[key][field]
            if not self.store[key]:
                del self.store[key]
            return True
        return False

    # --- Level 2 ---
    def scan(self, key: str) -> List[str]:
        if key not in self.store:
            return []
        return [f"{f}({self.store[key][f].val})" for f in sorted(self.store[key].keys())]

    def scan_by_prefix(self, key: str, prefix: str) -> List[str]:
        if key not in self.store:
            return []
        return [f"{f}({self.store[key][f].val})" for f in sorted(self.store[key].keys()) if f.startswith(prefix)]

    # --- Level 3 ---
    def set_at(self, key: str, field: str, value: str, timestamp: int) -> None:
        self.store[key][field] = FieldRecord(value)

    def set_at_with_ttl(self, key: str, field: str, value: str, timestamp: int, ttl: int) -> None:
        self.store[key][field] = FieldRecord(value, remaining_ttl=ttl, absolute_expiry=timestamp + ttl)

    def get_at(self, key: str, field: str, timestamp: int) -> Optional[str]:
        if key in self.store and field in self.store[key]:
            rec = self.store[key][field]
            if self._is_alive(rec, timestamp):
                return rec.val
            del self.store[key][field]
            if not self.store[key]:
                del self.store[key]
        return None

    def delete_at(self, key: str, field: str, timestamp: int) -> bool:
        if key in self.store and field in self.store[key]:
            alive = self._is_alive(self.store[key][field], timestamp)
            del self.store[key][field]
            if not self.store[key]:
                del self.store[key]
            return alive
        return False

    def scan_at(self, key: str, timestamp: int) -> List[str]:
        if key not in self.store:
            return []
        items = []
        expired_fields = []
        for f in sorted(self.store[key].keys()):
            rec = self.store[key][f]
            if self._is_alive(rec, timestamp):
                items.append(f"{f}({rec.val})")
            else:
                expired_fields.append(f)
        for f in expired_fields:
            del self.store[key][f]
        if not self.store[key]:
            del self.store[key]
        return items

    def scan_by_prefix_at(self, key: str, prefix: str, timestamp: int) -> List[str]:
        if key not in self.store:
            return []
        items = []
        expired_fields = []
        for f in sorted(self.store[key].keys()):
            rec = self.store[key][f]
            if self._is_alive(rec, timestamp):
                if f.startswith(prefix):
                    items.append(f"{f}({rec.val})")
            else:
                expired_fields.append(f)
        for f in expired_fields:
            del self.store[key][f]
        if not self.store[key]:
            del self.store[key]
        return items

    # --- Level 4 ---
    def backup(self, timestamp: int) -> int:
        snapshot: Dict[str, Dict[str, Tuple[str, Optional[int]]]] = collections.defaultdict(dict)
        saved_count = 0
        for key, fields in list(self.store.items()):
            for field, rec in list(fields.items()):
                if self._is_alive(rec, timestamp):
                    rem = (rec.absolute_expiry - timestamp) if rec.absolute_expiry is not None else None
                    snapshot[key][field] = (rec.val, rem)
                    saved_count += 1
                else:
                    del fields[field]
            if not fields:
                del self.store[key]
        self.backups.append((timestamp, snapshot))
        return saved_count

    def restore(self, timestamp: int, timestamp_to_restore: int) -> None:
        idx = -1
        for i in range(len(self.backups) - 1, -1, -1):
            if self.backups[i][0] <= timestamp_to_restore:
                idx = i
                break
        if idx == -1:
            return

        target_backup = self.backups[idx][1]
        self.store.clear()
        for key, fields in target_backup.items():
            for field, (val, rem) in fields.items():
                abs_exp = (timestamp + rem) if rem is not None else None
                self.store[key][field] = FieldRecord(val=val, remaining_ttl=rem, absolute_expiry=abs_exp)
if __name__ == "__main__":
    db = InMemoryDatabase()
    # Level 1
    db.set("user1", "name", "Alice"); db.set("user1", "age", "30")
    assert db.get("user1", "name") == "Alice"
    # Level 2 字典序扫描
    assert db.scan("user1") == ["age(30)", "name(Alice)"]
    # Level 3 TTL
    db.set_at_with_ttl("u2", "token", "abc", timestamp=100, ttl=50) # 有效期 [100, 150)
    assert db.get_at("u2", "token", 120) == "abc"
    assert db.get_at("u2", "token", 150) is None
    # Level 4 相对 TTL 快照恢复
    db.set_at_with_ttl("u3", "session", "s1", timestamp=200, ttl=100) # 300 过期，在 240 时剩余 60
    db.backup(240)
    db.restore(timestamp=500, timestamp_to_restore=240) # 恢复到新时钟 500: 新过期时间为 560
    assert db.get_at("u3", "session", 550) == "s1"
    assert db.get_at("u3", "session", 560) is None
    print("✅ InMemoryDatabase tests passed!")
```


```cpp
#include <iostream>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <optional>

class InMemoryDatabase {
private:
    struct FieldRecord {
        std::string val;
        std::optional<long long> remaining_ttl;
        std::optional<long long> absolute_expiry;
    };
    struct SnapshotField {
        std::string val;
        std::optional<long long> remaining_ttl;
    };

    std::unordered_map<std::string, std::map<std::string, FieldRecord>> store_;
    std::vector<std::pair<long long, std::unordered_map<std::string, std::map<std::string, SnapshotField>>>> backups_;

    bool isAlive(const FieldRecord& rec, std::optional<long long> ts) const {
        if (!rec.absolute_expiry.has_value() || !ts.has_value()) return true;
        return ts.value() < rec.absolute_expiry.value();
    }

public:
    void set(const std::string& key, const std::string& field, const std::string& val) {
        store_[key][field] = FieldRecord{val, std::nullopt, std::nullopt};
    }

    std::optional<std::string> get(const std::string& key, const std::string& field) {
        auto kit = store_.find(key);
        if (kit == store_.end()) return std::nullopt;
        auto fit = kit->second.find(field);
        if (fit == kit->second.end()) return std::nullopt;
        return fit->second.val;
    }

    bool del(const std::string& key, const std::string& field) {
        auto kit = store_.find(key);
        if (kit == store_.end()) return false;
        auto fit = kit->second.find(field);
        if (fit == kit->second.end()) return false;
        kit->second.erase(fit);
        if (kit->second.empty()) store_.erase(kit);
        return true;
    }

    std::vector<std::string> scan(const std::string& key) {
        std::vector<std::string> res;
        auto kit = store_.find(key);
        if (kit == store_.end()) return res;
        for (const auto& [f, rec] : kit->second) {
            res.push_back(f + "(" + rec.val + ")");
        }
        return res;
    }

    std::vector<std::string> scanByPrefix(const std::string& key, const std::string& prefix) {
        std::vector<std::string> res;
        auto kit = store_.find(key);
        if (kit == store_.end()) return res;
        for (const auto& [f, rec] : kit->second) {
            if (f.rfind(prefix, 0) == 0) res.push_back(f + "(" + rec.val + ")");
        }
        return res;
    }

    void setAtWithTtl(const std::string& key, const std::string& field, const std::string& val, long long ts, long long ttl) {
        store_[key][field] = FieldRecord{val, ttl, ts + ttl};
    }

    std::optional<std::string> getAt(const std::string& key, const std::string& field, long long ts) {
        auto kit = store_.find(key);
        if (kit == store_.end()) return std::nullopt;
        auto fit = kit->second.find(field);
        if (fit == kit->second.end()) return std::nullopt;
        if (isAlive(fit->second, ts)) return fit->second.val;
        kit->second.erase(fit);
        if (kit->second.empty()) store_.erase(kit);
        return std::nullopt;
    }

    int backup(long long ts) {
        std::unordered_map<std::string, std::map<std::string, SnapshotField>> snapshot;
        int count = 0;
        for (auto kit = store_.begin(); kit != store_.end(); ) {
            for (auto fit = kit->second.begin(); fit != kit->second.end(); ) {
                if (isAlive(fit->second, ts)) {
                    std::optional<long long> rem = fit->second.absolute_expiry.has_value() ?
                        std::make_optional(fit->second.absolute_expiry.value() - ts) : std::nullopt;
                    snapshot[kit->first][fit->first] = SnapshotField{fit->second.val, rem};
                    count++;
                    ++fit;
                } else {
                    fit = kit->second.erase(fit);
                }
            }
            if (kit->second.empty()) kit = store_.erase(kit);
            else ++kit;
        }
        backups_.push_back({ts, std::move(snapshot)});
        return count;
    }

    void restore(long long ts, long long ts_to_restore) {
        int idx = -1;
        for (int i = (int)backups_.size() - 1; i >= 0; --i) {
            if (backups_[i].first <= ts_to_restore) {
                idx = i;
                break;
            }
        }
        if (idx == -1) return;
        store_.clear();
        for (const auto& [key, fields] : backups_[idx].second) {
            for (const auto& [field, item] : fields) {
                FieldRecord rec;
                rec.val = item.val;
                rec.remaining_ttl = item.remaining_ttl;
                rec.absolute_expiry = item.remaining_ttl.has_value() ? std::make_optional(ts + item.remaining_ttl.value()) : std::nullopt;
                store_[key][field] = rec;
            }
        }
    }
};
```

#### 10. 待办事项面向对象设计与职责分离 (Todo List OOP Design)
- **接口契约**：`add(entry) -> id`，`delete(id) -> bool`，`get_todo() -> List[str]`，`get_all() -> List[str]`。外部提供 $\mathcal{O}(1)$ 判定器 `check_todo(id) -> bool` 查询任务完成状态。
- **职责分离原则 (Separation of Concerns)**：
  - `TodoList` 仅负责管理时序存储、ID 分配与增删，完成状态由外部源权威维护，类内部绝不缓存完成布尔值，杜绝脏数据。
  - 结构采用哈希表 `id_to_node` + 双向链表（保留插入顺序）。`id` 采用单调递增计数器分配，严禁复用已删除 ID。
  - `get_todo` 遍历链表并在生成时由 `check_todo` 惰性过滤，达到严格 $\mathcal{O}(1)$ 增删与 $\mathcal{O}(k)$ 输出大小遍历。

```python
from typing import Dict, List, Optional, Callable

class TodoNode:
    def __init__(self, tid: int, entry: str):
        self.id = tid
        self.entry = entry
        self.prev: Optional['TodoNode'] = None
        self.next: Optional['TodoNode'] = None

class TodoList:
    """
    面向对象职责分离的待办事项管理系统
    """
    def __init__(self):
        self.id_to_node: Dict[int, TodoNode] = {}
        self.head = TodoNode(0, "")
        self.tail = TodoNode(0, "")
        self.head.next = self.tail
        self.tail.prev = self.head
        self._next_id = 1

    def add(self, entry: str) -> int:
        tid = self._next_id
        self._next_id += 1
        node = TodoNode(tid, entry)
        self.id_to_node[tid] = node
        node.prev, node.next = self.tail.prev, self.tail
        self.tail.prev.next = node
        self.tail.prev = node
        return tid

    def delete(self, tid: int) -> bool:
        if tid not in self.id_to_node:
            return False
        node = self.id_to_node.pop(tid)
        node.prev.next = node.next
        node.next.prev = node.prev
        return True

    def get_all(self) -> List[str]:
        res = []
        curr = self.head.next
        while curr is not self.tail:
            res.append(curr.entry)
            curr = curr.next
        return res

    def get_todo(self, is_completed_fn: Callable[[int], bool]) -> List[str]:
        res = []
        curr = self.head.next
        while curr is not self.tail:
            if not is_completed_fn(curr.id):
                res.append(curr.entry)
            curr = curr.next
        return res
if __name__ == "__main__":
    tl = TodoList()
    t1 = tl.add("Review code")
    t2 = tl.add("Write tests")
    assert tl.get_all() == ["Review code", "Write tests"]
    completed = {t1: True} # 外部维护权威状态
    assert tl.get_todo(lambda tid: completed.get(tid, False)) == ["Write tests"]
    assert tl.delete(t2) is True
    assert tl.get_all() == ["Review code"]
    print("✅ TodoList tests passed!")
```



#### 11. 带权重与变长尺寸限制的 LRU 缓存 (Weighted LRU Cache with Size-Bounded Eviction)
- **业务场景与不变量**：
  - 经典 LRU 仅统计条目数量（Count），每个键值对等价计为 1。
  - 在工业级对象缓存与显存张量缓存中，条目具有**异构尺寸（Size / Weight）**。总容量限制 `capacity` 表示驻留对象的总字节数或总权重，而非条目数量。
  - **核心不变量**：$\sum_{x \in Cache} x.size \le capacity$。
- **级联驱逐循环 (Multi-Item Cascade Eviction)**：
  - 在写入新元素 `put(key, value, size)` 时，若当前总占用 `current_size + size > capacity`，可能需要**连续淘汰多个 LRU 尾部元素**才能腾出充足空间；
  - **键值更新的不变量调整**：当更新已存在的键时，必须先自 `current_size` 扣减旧尺寸 `current_size -= old_size`，再执行空间富余度检查与驱逐循环，最后累加新尺寸并移至 MRU 表头，杜绝统计漂移。
- **严密边界防御 (Edge Cases)**：
  1. **单对象尺寸超出全局总容量 (`size > capacity`)**：即便将缓存完全清空亦无法容纳。标准契约：若键已存在先将其彻底剔除，随后直接放弃插入（或抛出异常），保证不变量不被击穿；
  2. **非正数非法尺寸 (`size <= 0`)**：校验并抛出参数异常；
  3. **精确贴合容积 (`current_size + size == capacity`)**：循环直接终止，无多余淘汰。

```python
from typing import Dict, Optional

class DLinkedNode:
    def __init__(self, key: str = "", val: int = 0, size: int = 0):
        self.key = key
        self.val = val
        self.size = size
        self.prev: Optional['DLinkedNode'] = None
        self.next: Optional['DLinkedNode'] = None

class WeightedLRUCache:
    """
    带权重/尺寸限制的 LRU 缓存 (Weighted LRU Cache)
    时间复杂度: get O(1), put 均摊 O(1)
    空间复杂度: O(N) (N 为常驻键总数)
    """
    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("Capacity must be non-negative")
        self.capacity = capacity
        self.current_size = 0
        self.cache: Dict[str, DLinkedNode] = {}
        # 哨兵头尾节点 (Head 为 MRU 端，Tail 为 LRU 端)
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: DLinkedNode) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DLinkedNode) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: DLinkedNode) -> None:
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[DLinkedNode]:
        res = self.tail.prev
        if res is self.head:
            return None
        self._remove_node(res)
        return res

    def get(self, key: str) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_head(node)
        return node.val

    def put(self, key: str, value: int, size: int) -> None:
        if size <= 0:
            raise ValueError("Item size must be strictly positive")
        
        # 边界 1: 单个元素尺寸直接超过总容量，绝不可能驻留
        if size > self.capacity:
            if key in self.cache:
                old = self.cache.pop(key)
                self._remove_node(old)
                self.current_size -= old.size
            return

        if key in self.cache:
            node = self.cache[key]
            # 扣减旧尺寸
            self.current_size -= node.size
            node.val = value
            node.size = size
            self._move_to_head(node)
        else:
            node = DLinkedNode(key, value, size)
            self.cache[key] = node
            self._add_to_head(node)

        self.current_size += size

        # 核心级联驱逐循环: 只要空间溢出且链表非空，持续驱逐 LRU 尾部
        while self.current_size > self.capacity and self.tail.prev is not self.head:
            victim = self._pop_tail()
            if victim:
                del self.cache[victim.key]
                self.current_size -= victim.size
if __name__ == "__main__":
    cache = WeightedLRUCache(capacity=10)
    cache.put("a", 1, 3)     # total = 3
    cache.put("b", 2, 4)     # total = 7
    cache.put("c", 3, 5)     # 7+5 > 10 -> 驱逐 "a" (3) -> total = 4+5 = 9
    assert cache.get("a") == -1
    assert cache.get("b") == 2
    assert cache.current_size == 9
    cache.put("d", 4, 3)     # 9+3 > 10 -> 驱逐 "c" (LRU, b刚被访问) -> total = 4+3 = 7
    assert cache.get("c") == -1
    assert cache.get("b") == 2
    assert cache.get("d") == 4
    # 更新已有 key 并缩放尺寸
    cache.put("b", 20, 6)    # 旧尺寸 4 -> 新尺寸 6, total = 7 - 4 + 6 = 9 <= 10
    assert cache.get("b") == 20
    assert cache.current_size == 9
    # 超额尺寸直接拦截
    cache.put("oversized", 99, 15)
    assert cache.get("oversized") == -1
    assert cache.current_size == 9
    print("✅ WeightedLRUCache tests passed!")
```


```cpp
#include <string>
#include <unordered_map>

class WeightedLRUCache {
private:
    struct Node {
        std::string key;
        int val;
        int size;
        Node* prev{nullptr};
        Node* next{nullptr};
        Node(std::string k = "", int v = 0, int s = 0) : key(std::move(k)), val(v), size(s) {}
    };

    int capacity_;
    int current_size_{0};
    std::unordered_map<std::string, Node*> cache_;
    Node* head_;
    Node* tail_;

    void addToHead(Node* node) {
        node->prev = head_;
        node->next = head_->next;
        head_->next->prev = node;
        head_->next = node;
    }

    void removeNode(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void moveToHead(Node* node) {
        removeNode(node);
        addToHead(node);
    }

    Node* popTail() {
        Node* res = tail_->prev;
        if (res == head_) return nullptr;
        removeNode(res);
        return res;
    }

public:
    explicit WeightedLRUCache(int capacity) : capacity_(capacity) {
        head_ = new Node();
        tail_ = new Node();
        head_->next = tail_;
        tail_->prev = head_;
    }

    ~WeightedLRUCache() {
        Node* curr = head_;
        while (curr) {
            Node* next = curr->next;
            delete curr;
            curr = next;
        }
    }

    int get(const std::string& key) {
        auto it = cache_.find(key);
        if (it == cache_.end()) return -1;
        moveToHead(it->second);
        return it->second->val;
    }

    void put(const std::string& key, int value, int size) {
        if (size <= 0) return;
        if (size > capacity_) {
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                current_size_ -= it->second->size;
                removeNode(it->second);
                delete it->second;
                cache_.erase(it);
            }
            return;
        }

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            Node* node = it->second;
            current_size_ -= node->size;
            node->val = value;
            node->size = size;
            moveToHead(node);
        } else {
            Node* node = new Node(key, value, size);
            cache_[key] = node;
            addToHead(node);
        }
        current_size_ += size;

        while (current_size_ > capacity_ && tail_->prev != head_) {
            Node* victim = popTail();
            if (victim) {
                cache_.erase(victim->key);
                current_size_ -= victim->size;
                delete victim;
            }
        }
    }
};
```

</div>

</div>
</details>

---

### 2. 链表原地反转与三指针迭代推进 (Reverse Linked List In-Place via Three Pointers)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 02</span>
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
#### 3. 高位在前单链表加法 (Add Two Numbers - Forward Order MSB First, LC 445)
- **核心约束**：给出两个高位在前链表 $l_1, l_2$，在**严禁翻转输入链表**的前提下求和并以高位在前的单链表返回。
- **双栈解法 (Two-Stack Archetype)**：
  - 将 $l_1, l_2$ 节点值依次压入 `s1, s2`，栈顶自然对齐个位数；
  - 循环弹出栈顶求和：$\text{val} = v_1 + v_2 + carry$，更新 $carry = \text{val} // 10$，当前位为 $\text{val} \% 10$；
  - **头插法 (Head Insertion)**：每次将新生成的节点插入结果链表的最前端：`new_node.next = head; head = new_node`，天然生成高位在前链表，无需二次反转。

#### 4. N 叉树权值求和与叶子节点后继链表编织 (N-ary Tree Sum + Leaf Next Pointer)
- **三层递进考点**：
  1. **N 叉树求和**：后序/先序 DFS 累加所有节点权值；
  2. **叶子节点串联**：先序 DFS 遍历树，维护 `prev_leaf` 指针。每当检测到当前节点为叶子节点（`not node.children`），执行 `prev_leaf.next = curr; prev_leaf = curr`，将所有叶子节点串接为单向链表；
  3. **O(1) 额外空间优化 (Follow-up)**：利用节点闲置的 `next` 指针作为遍历线索（Morris-like Threading），在遍历过程中复用结构内部指针完成叶子链表组装，彻底省去递归栈或辅助收集数组。

</div>
</details>

---

### 3. 合并有序链表与低位求和进位链 (Merge Two Sorted Lists & Add Two Numbers)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 03</span>
  <span class="review-card-title">合并有序链表与低位求和进位链 (Merge Two Sorted Lists & Add Two Numbers)</span>
  <span class="review-card-tag">虚拟哨兵头节点 · 双指针归并 · 低位向高位进位链 · O(1) 辅助空间</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 21 · Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) — `https://leetcode.com/problems/merge-two-sorted-lists/`
> - [LeetCode 2 · Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) — `https://leetcode.com/problems/add-two-numbers/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与双题合璧要求</div>

**题目原文 (Problem Statement)**：
1. **Merge Two Sorted Lists (LC 21)**: Merge two sorted singly linked lists into one sorted list by splicing together the nodes of the first two lists.
2. **Add Two Numbers (LC 2)**: You are given two non-empty linked lists representing two non-negative integers. The digits are stored in **reverse order** (units place at head), and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

**核心约束**：
- 均需在线性时间 $\mathcal{O}(N + M)$ 与常数额外空间 $\mathcal{O}(1)$ 内完成；
- 利用虚拟哨兵头节点（Dummy Sentinel）消除首节点分支判断。

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class MergeAndAddSolution:
    @staticmethod
    def mergeTwoLists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # 合并两个升序单链表 (LC 21)
        dummy = ListNode(0)
        curr = dummy

        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next

        # 直接将非空剩余段整体拼接，O(1) 完成
        curr.next = l1 if l1 else l2
        return dummy.next

    @staticmethod
    def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # 两数相加：逆序存储/低位在先 (LC 2)
        dummy = ListNode(0)
        curr = dummy
        carry = 0

        p1, p2 = l1, l2
        while p1 or p2 or carry:
            val1 = p1.val if p1 else 0
            val2 = p2.val if p2 else 0
            total = val1 + val2 + carry

            carry = total // 10
            curr.next = ListNode(total % 10)
            curr = curr.next

            if p1: p1 = p1.next
            if p2: p2 = p2.next

        return dummy.next
```

```cpp
class MergeAndAddSolution {
public:
    static ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* curr = &dummy;
        while (l1 && l2) {
            if (l1->val <= l2->val) {
                curr->next = l1;
                l1 = l1->next;
            } else {
                curr->next = l2;
                l2 = l2->next;
            }
            curr = curr->next;
        }
        curr->next = l1 ? l1 : l2;
        return dummy.next;
    }

    static ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* curr = &dummy;
        int carry = 0;
        while (l1 || l2 || carry) {
            int v1 = l1 ? l1->val : 0;
            int v2 = l2 ? l2->val : 0;
            int sum = v1 + v2 + carry;
            carry = sum / 10;
            curr->next = new ListNode(sum % 10);
            curr = curr->next;
            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }
        return dummy.next;
    }
};
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **哨兵节点的统一指针语义**：
  `dummy` 使得结果链表的首节点与后续追加节点享有完全相同的修改语法（`curr.next = ...`），无需在循环外部特判 `head` 初始化。
- **剩余链表 $\mathcal{O}(1)$ 直挂**：
  链表归并不同于数组归并！当某一条链表遍历完毕时，无需逐个拷贝剩余节点，只需将 `curr.next` 直接指向未耗尽链表的头节点指针，耗时严格 $\mathcal{O}(1)$。
- **进位闭包循环条件**：
  `while p1 or p2 or carry` 将链表遍历与最高位产生的进位溢出完全合并进单个循环体，杜绝在循环外部漏写 `if carry: curr.next = ListNode(1)` 的常见漏洞。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：
  - 合并升序链表：$\mathcal{O}(N + M)$，其中 $N, M$ 分别为两链表长度；
  - 两数相加：$\mathcal{O}(\max(N, M))$。
- **空间复杂度**：合并为 $\mathcal{O}(1)$ 原地重排；相加为 $\mathcal{O}(1)$ 额外辅助空间（不计新链表节点）。

</div>

</div>
</details>

---

### 4. 快慢双指针环检测与数组链表化 (Linked List Cycle & Find Duplicate Number)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 04</span>
  <span class="review-card-title">快慢双指针环检测与数组链表化 (Linked List Cycle & Find Duplicate Number)</span>
  <span class="review-card-tag">Floyd 判圈算法 · 环入口数学推导 · 数组下标隐式图转化 · O(1) 空间</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 141 · Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) — `https://leetcode.com/problems/linked-list-cycle/`
> - [LeetCode 142 · Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/) — `https://leetcode.com/problems/linked-list-cycle-ii/`
> - [LeetCode 287 · Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) — `https://leetcode.com/problems/find-the-duplicate-number/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
1. **Linked List Cycle I & II (LC 141 / 142)**: Given `head`, determine if the linked list has a cycle in it. If there is a cycle, return the node where the cycle begins. Solve it using $\mathcal{O}(1)$ memory.
2. **Find the Duplicate Number (LC 287)**: Given an array of integers `nums` containing $n + 1$ integers where each integer is in the range $[1, n]$ inclusive. There is only one repeated number in `nums`, find this repeated number **without modifying the array** and using only $\mathcal{O}(1)$ extra space.

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import Optional, List

class CycleAndDuplicateSolution:
    @staticmethod
    def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:
        # 寻找链表环入口 (LC 142)
        if not head or not head.next:
            return None

        slow = head
        fast = head

        # 阶段一：判定是否存在环（快慢指针同起点推进）
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None  # fast 抵达链表末尾，无环

        # 阶段二：寻找环入口节点
        # 一针归位至 head，双针等速单步向前
        ptr1 = head
        ptr2 = slow
        while ptr1 != ptr2:
            ptr1 = ptr1.next
            ptr2 = ptr2.next

        return ptr1

    @staticmethod
    def findDuplicate(nums: List[int]) -> int:
        # 寻找数组中唯一重复数：下标隐式链表判圈 (LC 287)
        # 数组下标作为节点地址，nums[i] 作为 next 指针
        slow = nums[0]
        fast = nums[0]

        # 阶段一：快慢指针寻找相遇点
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        # 阶段二：定位环入口（即入度大于 1 的重复数值）
        ptr1 = nums[0]
        ptr2 = slow
        while ptr1 != ptr2:
            ptr1 = nums[ptr1]
            ptr2 = nums[ptr2]

        return ptr1
```

```cpp
class CycleAndDuplicateSolution {
public:
    static ListNode* detectCycle(ListNode* head) {
        if (!head || !head->next) return nullptr;
        ListNode* slow = head;
        ListNode* fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) break;
        }
        if (!fast || !fast->next) return nullptr;
        ListNode* ptr1 = head;
        ListNode* ptr2 = slow;
        while (ptr1 != ptr2) {
            ptr1 = ptr1->next;
            ptr2 = ptr2->next;
        }
        return ptr1;
    }

    static int findDuplicate(const std::vector<int>& nums) {
        int slow = nums[0];
        int fast = nums[0];
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);

        int ptr1 = nums[0];
        int ptr2 = slow;
        while (ptr1 != ptr2) {
            ptr1 = nums[ptr1];
            ptr2 = nums[ptr2];
        }
        return ptr1;
    }
};
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析与数学推导</div>

- **Floyd 判圈算法数学证明 (Mathematical Derivation)**：
  - 设链表头部到环入口的距离为 $a$；
  - 环入口到快慢指针首次相遇点的距离为 $b$；
  - 环的周长为 $C$，相遇点走完剩余环到达入口的距离为 $C - b$。
  - 在首次相遇时：
    $$\text{slow 走过的距离} = a + b$$
    $$\text{fast 走过的距离} = a + b + k \cdot C \quad (k \ge 1)$$
  - 因为 fast 速度是 slow 的 2 倍：
    $$2(a + b) = a + b + k \cdot C \implies a + b = k \cdot C \implies a = k \cdot C - b = (k - 1)C + (C - b)$$
  - **结论**：从链表头部出发一个指针 $ptr_1$，同时从相遇点出发一个指针 $ptr_2$，两者均以单步速度同步推移。当 $ptr_1$ 走完距离 $a$ 到达环入口时，$ptr_2$ 刚好走完 $(k-1)$ 整圈并加上剩余的 $(C - b)$ 距离，**两指针必精准在环入口相遇**！
- **数组向链表的降维映射 (LC 287 Invariant)**：
  - 数组长度为 $n + 1$，元素范围为 $[1, n]$。根据鸽巢原理（Pigeonhole Principle），必然存在至少一个重复数。
  - 建立有向图：节点 $i \to nums[i]$。
  - 因为 $nums[i] \ge 1$，所以**节点 $0$ 绝对不可能有任何入边**（入度为 0，绝对不在环内，必然是链表的起点）。
  - 若存在重复数 $target$，意味着有多个不同的下标 $i, j$ 满足 $nums[i] = nums[j] = target$。在有向图中表现为**节点 $target$ 的入度 $\ge 2$**。
  - 在每个节点出度为 1 的函数图中，入度大于 1 的节点正是**环的入口节点**！直接应用 Floyd 判圈算法即可在 $\mathcal{O}(1)$ 空间且不修改原数组的前提下求出重复值。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，快慢指针在环内至多循环一圈即可相遇，找入口至多 $N$ 步。
- **空间复杂度**：$\mathcal{O}(1)$，仅需常数级别的指针变量。

</div>

</div>
</details>

---

### 5. 链表中点截断、后半反转与交叉穿插重排 (Reorder List)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 05</span>
  <span class="review-card-title">链表中点截断、后半反转与交叉穿插重排 (Reorder List)</span>
  <span class="review-card-tag">快慢指针定中点 · 原地链表反转 · 双链交替穿插 · 空间 O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 143 · Reorder List](https://leetcode.com/problems/reorder-list/) — `https://leetcode.com/problems/reorder-list/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> You are given the head of a singly linked-list: $L_0 \to L_1 \to \dots \to L_{n-1} \to L_n$.
> Reorder the list to be on the following form: $L_0 \to L_n \to L_1 \to L_{n-1} \to L_2 \to L_{n-2} \to \dots$
> You may not modify the values in the list's nodes. Only nodes themselves may be changed. Solve in-place in $\mathcal{O}(1)$ auxiliary space.

**输入输出示例**：
- `head = [1, 2, 3, 4]` $\implies$ `[1, 4, 2, 3]`
- `head = [1, 2, 3, 4, 5]` $\implies$ `[1, 5, 2, 4, 3]`

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import Optional

class ReorderListSolution:
    @staticmethod
    def reorderList(head: Optional[ListNode]) -> None:
        # 原地重排链表：三步经典组合拳。
        # 时间复杂度 O(N)，额外空间复杂度 O(1)。
        if not head or not head.next:
            return

        # 步骤 1: 快慢指针寻找中点，并将链表从中点截断为两条独立链表
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        # 此时 slow 为前半部分末尾，slow.next 为后半部分起点
        second_head = slow.next
        slow.next = None  # 截断链表，解除环形依赖

        # 步骤 2: 原地翻转后半部分单链表
        prev = None
        curr = second_head
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        p2 = prev  # 翻转后后半部分的新头节点

        # 步骤 3: 交叉穿插缝合前半段 (p1) 与后半段 (p2)
        p1 = head
        while p2:
            t1 = p1.next
            t2 = p2.next

            p1.next = p2
            p2.next = t1

            p1 = t1
            p2 = t2
```

```cpp
class ReorderListSolution {
public:
    static void reorderList(ListNode* head) {
        if (!head || !head->next) return;

        // 1. 快慢指针找中点
        ListNode* slow = head;
        ListNode* fast = head;
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        ListNode* second = slow->next;
        slow->next = nullptr;

        // 2. 原地反转后半部分
        ListNode* prev = nullptr;
        ListNode* curr = second;
        while (curr) {
            ListNode* nxt = curr->next;
            curr->next = prev;
            prev = curr;
            curr = nxt;
        }

        // 3. 交叉缝合
        ListNode* p1 = head;
        ListNode* p2 = prev;
        while (p2) {
            ListNode* t1 = p1->next;
            ListNode* t2 = p2->next;

            p1->next = p2;
            p2->next = t1;

            p1 = t1;
            p2 = t2;
        }
    }
};
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析与三步合成律</div>

- **三步合成范式 (Three-Step Composition)**：
  本题为链表三大基本操作的经典组合，是检验指针操作基本功的试金石：
  1. **中点探测与严密截断**：
     条件 `fast.next and fast.next.next` 确保无论是偶数长度（如 4 个节点停在索引 1）还是奇数长度（如 5 个节点停在索引 2），前半段长度均大于等于后半段（$len(p_1) \ge len(p_2)$），截断 `slow.next = None` 保证后续遍历有清晰的终止边界；
  2. **原地反转后半段**：
     标准三指针原地反转，不引入数组或调用栈；
  3. **交替穿插缝合 (Interleaving Merge)**：
     因为前半段长度必然大于等于后半段，所以循环条件仅需 `while p2` 即可自然收尾，绝不会发生空指针异常。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，探测中点 $N/2$ 步，反转 $N/2$ 步，合并 $N/2$ 步，整体单趟线性。
- **空间复杂度**：$\mathcal{O}(1)$，严格无任何堆/栈空间开销。

</div>

</div>
</details>

---

### 6. 删除链表的倒数第 N 个节点 (Remove Nth Node From End of List)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 06</span>
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

### 7. 深拷贝带随机指针的链表与原地穿插拆分 (Copy List with Random Pointer)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 07</span>
  <span class="review-card-title">深拷贝带随机指针的链表与原地穿插拆分 (Copy List with Random Pointer)</span>
  <span class="review-card-tag">原地交织插入 · 随机指针投影映射 · 链表解耦拆分 · 空间 O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 138 · Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/) — `https://leetcode.com/problems/copy-list-with-random-pointer/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> A linked list of length $n$ is given such that each node contains an additional random pointer, which could point to any node in the list, or `null`.
> Construct a **deep copy** of the list. The deep copy should consist of exactly $n$ brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the `next` and `random` pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state.
> Return the head of the copied linked list.

**核心矛盾与解法跃迁**：
- **哈希表基准解法**：`map[old_node] = new_node`，两趟扫描。时空均为 $\mathcal{O}(N)$。
- **工业级最优：三趟扫描原地交织拆分法 (In-Place Interleaving)**：彻底抛弃哈希表，利用原链表节点的 `next` 字段暂存对应副本节点，将辅助空间优化至极致的严格 $\mathcal{O}(1)$！

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import Optional

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class CopyRandomListSolution:
    @staticmethod
    def copyRandomList(head: Optional[Node]) -> Optional[Node]:
        # 三趟扫描原地交织复制法。
        # 额外辅助空间复杂度严格 O(1)。
        if not head:
            return None

        # 第一趟: 原地克隆节点并就地穿插: A -> A' -> B -> B' -> C -> C'
        curr = head
        while curr:
            copy = Node(curr.val, curr.next)
            curr.next = copy
            curr = copy.next

        # 第二趟: 构建新克隆节点的 random 指针
        # 因为原节点 curr 的副本就是 curr.next，所以其 random 目标对应的副本必然是 curr.random.next
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next

        # 第三趟: 拆分交织链表，恢复原链表并提取出独立深拷贝链表
        curr = head
        copy_head = head.next
        while curr:
            copy = curr.next
            curr.next = copy.next
            if copy.next:
                copy.next = copy.next.next
            curr = curr.next

        return copy_head
```

```cpp
class CopyRandomListSolution {
public:
    static Node* copyRandomList(Node* head) {
        if (!head) return nullptr;

        // 1. 原地复制并交织
        Node* curr = head;
        while (curr) {
            Node* copy = new Node(curr->val);
            copy->next = curr->next;
            curr->next = copy;
            curr = copy->next;
        }

        // 2. 映射 random 指针
        curr = head;
        while (curr) {
            if (curr->random) {
                curr->next->random = curr->random->next;
            }
            curr = curr->next->next;
        }

        // 3. 拆分解耦
        curr = head;
        Node* copyHead = head->next;
        while (curr) {
            Node* copy = curr->next;
            curr->next = copy->next;
            if (copy->next) {
                copy->next = copy->next->next;
            }
            curr = curr->next;
        }

        return copyHead;
    }
};
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析与映射不变量</div>

- **原位穿插不变量 (In-Place Interleaving Invariant)**：
  通过将每个新节点 $curr'$ 插入到原节点 $curr$ 与 $curr.next$ 之间，我们建立了一个完全物理绑定的映射关系：
  $$\text{copy}(curr) \equiv curr.next$$
  由此，当需要查找 $curr.random$ 的深拷贝节点时，无需哈希查表，直接通过指针寻址即可得到：
  $$\text{copy}(curr.random) \equiv curr.random.next$$
- **解耦分离的干净度**：
  在第三趟扫描中，必须严密恢复原链表 `curr.next = copy.next`，防止修改原链表引发调用方不可预期的指针破损副作用。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，三趟线性扫描，每趟耗时严格 $\mathcal{O}(N)$。
- **空间复杂度**：$\mathcal{O}(1)$，除返回值新节点外，额外辅助空间严格为常数。

</div>

</div>
</details>

---

### 8. K 个一组翻转链表全家桶与组间重排 (Reverse Nodes in k-Group & Structural Group Inversion)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 08</span>
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
<div class="review-block">
<div class="review-block-label">🌐 核心基石延伸：多线程并发安全任务队列变换 (Thread-Safe Task Queue Transformation)</div>

#### 并发安全重排架构权衡 (Multi-Threading Trade-off Matrix)
在多线程生产者-消费者模型中，对链表任务队列执行批处理分组翻转 `reverseKGroup(head, k)` 时，必须防范数据竞争与死锁：
1. **全局互斥锁 (Global Mutex)**：
   - 机制：单个互斥锁保护整个链表，翻转期间拒绝一切读写。
   - 优劣：实现零心智负担，但读操作被完全阻塞，吞吐量断崖式下跌。
2. **读写锁 (Reader-Writer Lock / shared_mutex)**：
   - 机制：并发只读扫描共享读锁；分组翻转获取独占写锁。
   - 优劣：读者无冲突，但长时间的 $K$ 翻转会导致大量读者饥饿等待。
3. **分段锁 (Segmented / Fine-Grained Locking)**：
   - 机制：为每 $K$ 个节点的子链表分配独立锁。各工作线程并发翻转各自独立的 $K$ 节点区间。
   - 关键防死锁纪律：在跨组重连（`group_prev.next = new_sub_head`）时，涉及相邻组两把锁的获取，**必须严格按照物理内存地址或节点自然序号升序加锁**，杜绝循环等待死锁。
4. **写时复制 (Copy-On-Write / COW) 与原子指针替换**：
   - 机制：翻转线程在私有内存中构建新翻转子链，完成后通过原子比较交换（CAS / `atomic_store`）将父节点的 `next` 指针一步切换为新头节点。
   - 优劣：读线程完全无锁运行，零等待延迟，是工业级高频任务队列的首选方案。

</div>

</div>
</details>

---

### 9. 循环有序单链表的插入 (Insert into a Sorted Circular Linked List)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 09</span>
  <span class="review-card-title">循环有序单链表的插入 (Insert into a Sorted Circular Linked List)</span>
  <span class="review-card-tag">双指针循环遍历 · 拐点判定 · 环形边界环绕 · 空间 O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **相关链接**：
> - [LeetCode 708 · Insert into a Sorted Circular Linked List](https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/) — `https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/`

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

### 10. 扁平化多级双向链表与空节点过滤 (Flatten Multilevel Doubly Linked List with Empty-Node Filtering)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表 10</span>
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
<div class="review-block">
<div class="review-block-label">🌐 核心基石延伸：递归 DFS 展开 vs 显式栈迭代工程对比</div>

- **递归深度优先下探 (Recursive DFS with Tail Return)**：
  - 定义辅助函数 `dfs(node) -> tail`，展开子树并返回该分支的最后一个节点；
  - 遇到 `curr.child` 时：
    1. 暂存 `nxt = curr.next`；
    2. 递归获取子链表尾部 `child_tail = dfs(curr.child)`；
    3. 双向接入：`curr.next = curr.child; curr.child.prev = curr`；
    4. 尾部回接：若 `nxt` 存在，`child_tail.next = nxt; nxt.prev = child_tail`；
    5. 清空 `curr.child = None`；
    6. 从 `nxt` 或 `child_tail` 继续推进。
- **显式栈迭代解法 (Explicit Stack Iteration)**：
  - 工业级生产环境中为规避系统调用栈溢出（Stack Overflow），改用显式数据结构 `stack`：
  - 遍历主链表，当 `curr` 拥有 `child` 时：
    - 若 `curr.next` 存在，将其压入 `stack`（后续等待拼接）；
    - 将 `curr.next` 重定向至 `curr.child`，修复 `curr.child.prev = curr`；
    - 清空 `curr.child = None`；
  - 当 `curr.next` 为空且 `stack` 非空时，弹出待处理节点接在 `curr.next` 并修复 `prev` 指针。空间复杂度严格为 $\mathcal{O}(\text{depth})$。

</div>

</div>
</details>

---

### 11. 常数时间随机集合与弹出容器 (Randomized Container with O(1) Insert & PopRandom)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">容器 11</span>
  <span class="review-card-title">常数时间随机集合与弹出容器 (Randomized Container with O(1) Insert & PopRandom)</span>
  <span class="review-card-tag">连续动态数组 + 哈希索引表 · 尾部元素置换 (Swap with Last) · 等概率随机抽取 · O(1) 均摊</span>
</summary>
<div class="review-card-content">

> 🔗 **相关链接**：
> - [LeetCode 380 · Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) — `https://leetcode.com/problems/insert-delete-getrandom-o1/`
> - [LeetCode 381 · Insert Delete GetRandom O(1) - Duplicates allowed](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/) — `https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> Design a data structure that supports all following operations in average $\mathcal{O}(1)$ time complexity:
> 1. `insert(val)`: Inserts an item `val` to the set if not already present. Returns `true` if the item was not present, `false` otherwise.
> 2. `remove(val)`: Removes an item `val` from the set if present. Returns `true` if the item was present, `false` otherwise.
> 3. `getRandom()`: Returns a random element from the current set of elements. Each element must have the **same probability** of being returned.
> 4. `popRandom()` (Random Pop Extension): Removes and returns a random element from the container in $\mathcal{O}(1)$ time with uniform probability.

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
    # 基础版：元素互不重复 (LeetCode 380 + popRandom 扩展)
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
        # 核心扩展：随机等概率弹出并移除一个元素
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

### 12. 独立迭代器设计与共享流式缓冲区 (Python itertools.tee & Shared-State Streaming Buffer)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">迭代/流 12</span>
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

## 模块二：栈与单调结构 (Stack & Monotonic Stack / Deque)

### 13. 表达式计算器与运算符优先级全景全家桶 (Basic Calculator & Operator Precedence Hierarchy)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">栈 13</span>
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

### 14. 滑动窗口最大值与单调双端队列全景 (Sliding Window Maximum & Monotonic Deque Pattern)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">双端队列 14</span>
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
<div class="review-block">
<div class="review-block-label">🌐 核心基石延伸：时间窗口流式事件聚合与有界内存去重</div>

#### 1. 消息事件双向聚合器 (Message Event Aggregator with 5-Minute Window)
- **业务场景**：
  - 接收聊天流事件 `(timestamp, user_id, chat_id, event_type)`，`event_type \in {message, react, end_chat}`。
  - 滑动窗口为闭区间 $[t - 300, t]$（5 分钟）。输出保持原始输入顺序：
    1. `message_count`：同 `chat_id` 在当前窗口内的 message 事件总数；
    2. `active_chat_count`：当前 `user_id` 在当前窗口内处于活跃状态的 chat 总数。
  - **活跃状态精确判定法则**：
    - 在窗口 $[t - 300, t]$ 内该 `(user_id, chat_id)` 必须至少存在一次 `react` 或 `end_chat`；
    - **且窗口内该对的最后一次状态事件必须是 `react`**（若最后为 `end_chat` 则非活跃）。
- **乱序与在线有界内存清理 (Bounded Memory Eviction)**：
  - 对每个 `chat_id` 维护消息时间戳双端队列；
  - 对每个 `user_id` 维护活跃 chat 状态字典 `chat_state[chat_id] -> deque[(timestamp, type)]`；
  - 当窗口前进时，主动从队头弹出 $< t - 300$ 的过期条目；若队列排空则从状态字典中移除对应键，确保内存严格有界。

#### 2. 时间窗口重复日志检测器 (Sliding-Window Duplicate Records Detection)
- **业务场景**：流式日志包含 `(id, text, title, timestamp)`，时间戳单调递增。若同一内容在过去 60 秒内（$[t - 60, t]$）重复出现，立即输出该重复记录。
- **空间优化要求**：内存不得随全量流线性膨胀，超过 60 秒的历史条目必须及时驱逐。
- **双端队列 + 哈希表联动淘汰机制**：
  - 维护哈希表 `content_to_ts: Dict[content, int]` 记录各内容在窗口内的最后出现时间；
  - 维护时间序双端队列 `order_queue: deque[(timestamp, content)]`；
  - 每次处理新日志时，循环检查 `order_queue` 队头：若 `head.timestamp < t - 60`，弹出队头；若 `content_to_ts[head.content] == head.timestamp`，从哈希表中安全移除该键；
  - 内存占用严格受限于 60 秒窗口内的独立内容数。

</div>

</div>
</details>

---

### 15. 柱状图中最大的矩形与单调栈双哨兵范式 (Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">单调栈 15</span>
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

### 16. 单调栈去重与字典序最小子序列 (Remove Duplicate Letters via Monotonic Stack)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">单调栈 16</span>
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

### 17. 星号通配符括号有效性与全量展开 (Valid Parenthesis String with Wildcard & Concrete String Enumeration)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">栈/回溯 17</span>
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

### 18. 数据流中位数与多路归并全景 (Find Median from Data Stream & K-Way Merge)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">堆 18</span>
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

<div class="review-block">
<div class="review-block-label">💡 机制剖析与工业延伸</div>

- **双堆动态天平不变量证明 (Dual-Heap Invariant)**：
  - 数据流元素被动态划分为左右两个等势区间：较小半区为大顶堆 $lo$（存储相反数），较大半区为小顶堆 $hi$。
  - **排序关系不变量**：$\max(lo) \le \min(hi)$。
  - **容量平衡不变量**：$|lo| \ge |hi|$ 且 $|lo| - |hi| \le 1$。
  - 插入新元素时，新元素先进入 $lo$，弹出最大值注入 $hi$；若此时 $|hi| > |lo|$，将 $hi$ 的最小值回流至 $lo$。中位数始终可在 $\mathcal{O}(1)$ 提取：奇数时直接取 $\max(lo)$，偶数时取 $(\max(lo) + \min(hi)) / 2$。

#### 1. 合并 K 个升序链表 (LeetCode 23: Merge k Sorted Lists)
- **多路归并双解法**：
  1. **最小堆优先队列法**：维护大小为 $k$ 的小顶堆。首轮将 $k$ 条链表的头节点推入堆中；每次弹出最小节点接入输出链表，并将其 `next` 节点推入堆。总时间复杂度 $\mathcal{O}(N \log k)$，空间复杂度 $\mathcal{O}(k)$；
  2. **分治两两归并法 (Divide and Conquer)**：将 $k$ 条链表两两配对合并，每轮规模减半，总轮数为 $\lceil \log_2 k \rceil$。空间复杂度 $\mathcal{O}(1)$（迭代版），在无堆开销下更具缓存局部性。

```python
import heapq
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    dummy = ListNode(0)
    curr = dummy
    heap = []
    for i, l in enumerate(lists):
        if l:
            heapq.heappush(heap, (l.val, i, l))
    
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
            
    return dummy.next
if __name__ == "__main__":
    def build_list(vals):
        dummy = ListNode(0)
        curr = dummy
        for v in vals:
            curr.next = ListNode(v)
            curr = curr.next
        return dummy.next

    def to_list(node):
        res = []
        while node:
            res.append(node.val)
            node = node.next
        return res

    l1 = build_list([1, 4, 5])
    l2 = build_list([1, 3, 4])
    l3 = build_list([2, 6])
    merged = mergeKLists([l1, l2, l3])
    assert to_list(merged) == [1, 1, 2, 3, 4, 4, 5, 6]
    print("✅ mergeKLists tests passed!")
```


#### 2. 内存受限的滑动时间窗口 K 阶元素统计器 (K-th Element on a Streaming Time Window under Hard Memory Bound)
- **业务场景与硬性限制**：
  - 数据流以时序元组 $(timestamp, value)$ 持续高速涌入。
  - **查询目标**：在任意时刻 $now$，实时返回最近时间窗口 $[now - W, now]$ 内的所有活跃数值中的**第 $K$ 小（或第 $K$ 大）元素**。
  - **内存硬性约束 (Hard Memory Bound)**：严禁无上限存储历史数据；内存占用必须严格受限于窗口长度 $W$ 与事件发生率 $R$ 的物理上限。
- **架构方案 1：有界整数值域 $[V_{min}, V_{max}]$ (FIFO 窗口队列 + 频次桶数组)**：
  - 若数值在固定离散值域内（如网络延迟 $0 \sim 1000\text{ ms}$，量化微秒级跳价值）：
  - **数据结构**：
    1. 时间窗口双端队列 `queue: Deque[Tuple[ts, val]]` 记录窗口内的活跃事件物理时序；
    2. 频次计数桶数组 `buckets: List[int]`，长度 $B = V_{max} - V_{min} + 1$。
  - **操作复杂度**：
    - 入队 `add(ts, val)`：剔除 $ts' < ts - W$ 的过期元素（从对应桶中扣减计数），将新事件压入队列，`buckets[val - min_val] += 1`，均摊 $\mathcal{O}(1)$；
    - 查询 `find_kth(now, k)`：线性扫描桶数组累加前缀和，首个累积频次 $\ge k$ 的桶索引即为答案，时间复杂度严格 $\mathcal{O}(B)$。
  - **内存空间上限**：$\le R \cdot W \times 16\text{ 字节} + B \times 4\text{ 字节}$，严格有界且完全规避堆重平衡。

- **架构方案 2：无界任意实数域 (FIFO 窗口队列 + 树状数组 / 动态平衡树)**：
  - 当数值范围巨大或为浮点数时，采用树状数组 (Fenwick Tree) 对离散化坐标进行动态秩统计，或结合 `SortedList` / 双堆延迟删除。
  - 入队与过期剔除均为 $\mathcal{O}(\log M)$，查询第 $K$ 阶元素耗时 $\mathcal{O}(\log M)$（倍增二分），其中 $M = R \cdot W$ 为窗口活跃元素总数。

```python
import collections
from typing import Optional

class StreamingWindowKthBounded:
    """
    有界值域时间窗口 K 阶元素统计器
    时间复杂度: add 均摊 O(1), find_kth O(B) (B = max_val - min_val + 1)
    空间复杂度: O(R * W + B) 内存严格有界
    """
    def __init__(self, window_seconds: int, min_val: int, max_val: int):
        self.w = window_seconds
        self.min_val = min_val
        self.max_val = max_val
        self.num_buckets = max_val - min_val + 1
        self.buckets = [0] * self.num_buckets
        self.queue = collections.deque() # (timestamp, val)
        self.total_count = 0

    def _evict_expired(self, current_time: int) -> None:
        threshold = current_time - self.w
        while self.queue and self.queue[0][0] < threshold:
            _, val = self.queue.popleft()
            self.buckets[val - self.min_val] -= 1
            self.total_count -= 1

    def add(self, timestamp: int, value: int) -> None:
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(f"Value {value} out of bounded range [{self.min_val}, {self.max_val}]")
        self._evict_expired(timestamp)
        self.queue.append((timestamp, value))
        self.buckets[value - self.min_val] += 1
        self.total_count += 1

    def find_kth_smallest(self, current_time: int, k: int) -> Optional[int]:
        """返回 [current_time - W, current_time] 窗口内的第 k 小元素 (1-indexed)"""
        self._evict_expired(current_time)
        if k < 1 or k > self.total_count:
            return None
        cum = 0
        for i in range(self.num_buckets):
            cum += self.buckets[i]
            if cum >= k:
                return self.min_val + i
        return None

    def find_kth_largest(self, current_time: int, k: int) -> Optional[int]:
        """返回 [current_time - W, current_time] 窗口内的第 k 大元素 (1-indexed)"""
        self._evict_expired(current_time)
        if k < 1 or k > self.total_count:
            return None
        cum = 0
        for i in range(self.num_buckets - 1, -1, -1):
            cum += self.buckets[i]
            if cum >= k:
                return self.min_val + i
        return None
if __name__ == "__main__":
    sk = StreamingWindowKthBounded(window_seconds=5, min_val=0, max_val=100)
    sk.add(1, 10)
    sk.add(2, 30)
    sk.add(3, 20)
    sk.add(4, 50)
    sk.add(5, 40)
    # 当前窗口 [0, 5]: 包含 [10, 20, 30, 40, 50]
    assert sk.find_kth_smallest(5, 1) == 10
    assert sk.find_kth_smallest(5, 3) == 30
    assert sk.find_kth_smallest(5, 5) == 50
    assert sk.find_kth_largest(5, 1) == 50
    # 前进到时刻 7，窗口 [2, 7]: 时刻 1 (10) 过期淘汰，剩余 [20, 30, 40, 50]
    assert sk.find_kth_smallest(7, 1) == 20
    assert sk.find_kth_smallest(7, 2) == 30
    print("✅ StreamingWindowKthBounded tests passed!")
```


```cpp
#include <vector>
#include <deque>
#include <optional>
#include <cstdint>

class StreamingWindowKthBounded {
private:
    int64_t w_;
    int min_val_;
    int max_val_;
    std::vector<int> buckets_;
    std::deque<std::pair<int64_t, int>> queue_;
    int total_count_{0};

    void evictExpired(int64_t current_time) {
        int64_t threshold = current_time - w_;
        while (!queue_.empty() && queue_.front().first < threshold) {
            int v = queue_.front().second;
            queue_.pop_front();
            buckets_[v - min_val_]--;
            total_count_--;
        }
    }

public:
    StreamingWindowKthBounded(int64_t window_seconds, int min_val, int max_val)
        : w_(window_seconds), min_val_(min_val), max_val_(max_val),
          buckets_(max_val - min_val + 1, 0) {}

    void add(int64_t ts, int value) {
        evictExpired(ts);
        queue_.push_back({ts, value});
        buckets_[value - min_val_]++;
        total_count_++;
    }

    std::optional<int> findKthSmallest(int64_t current_time, int k) {
        evictExpired(current_time);
        if (k < 1 || k > total_count_) return std::nullopt;
        int cum = 0;
        for (size_t i = 0; i < buckets_.size(); ++i) {
            cum += buckets_[i];
            if (cum >= k) return min_val_ + static_cast<int>(i);
        }
        return std::nullopt;
    }
};
```

</div>

</div>

</div>
</details>

---

### 19. 双堆中位数流与多维栈系统架构 (MinStack, MaxStack, Streaming Median & Extensions)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">堆/系统 19</span>
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
<div class="review-block">
<div class="review-block-label">🌐 关联工程卡片</div>

限价委托与时间优先的复杂复合数据结构实现，详见 [第 22 题：极速限价订单簿系统](#22-极速限价订单簿系统-limit-order-book-system-with-price-time-priority)。

</div>

</div>
</details>

---

### 20. 多商户分级加权轮转任务调度器 (Tiered Priority Task Scheduler)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">堆/调度 20</span>
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
<div class="review-block">
<div class="review-block-label">🌐 核心基石延伸：高可用容错工作队列、时间片轮转与分级任务管理</div>

#### 1. 高可用容错工作队列系统 (Fault-Tolerant Work Queue with Leases, Retries & DLQ)
- **核心契约**：
  - `reserve() -> Optional[(task_id, token)]`：将任务状态由 `READY` 转为 `RESERVED`，授予租约时限 `lease_deadline = now + timeout` 并生成唯一防护版本 `version_token`；
  - `complete(task_id, token)`：只有持有当前版本有效租约的 Worker 才能将其转为 `COMPLETED`；
  - `fail(task_id, token)`：增加重试计数 `attempts`。若 `attempts < max_attempts` 重新投递回 `READY`，否则打入死信队列（Dead-Letter Queue, DLQ）。
- **Stale Lease 防护不变量 (Fencing Token Invariant)**：
  - 若 Worker A 遭遇长 GC 导致租约过期，调度器在检测到租约超时后，将任务版本递增为 `token + 1` 并重新投递给 Worker B；
  - 随后 Worker A 苏醒并上报 `complete(task_id, token)`，因其携带的旧 Token 失效，请求被拒绝，杜绝重复结算！
- **三维数据结构组合**：
  - `ready_queue`: `deque` 维护就绪任务 ID，$\mathcal{O}(1)$ 派发；
  - `task_store`: `Dict[task_id, TaskRecord]` 记录权威状态、重试次数与当前 Token；
  - `lease_heap`: `heapq` 存储 `(lease_deadline, task_id, version_token)`，$\mathcal{O}(1)$ 查看最早超期任务，超时检查时惰性比对 Token 剔除已完成或已变更的幽灵节点。

#### 2. 时间片轮转任务调度器 (Round-Robin Task Scheduler)
- **核心架构**：就绪任务队列 + 固定时间配额（Time Slice Quantum）。
- **操作循环**：
  - 每个 Tick 弹出队头任务，赋予一个时间片；若任务执行完毕则将其销毁；若未完结则重新追加至队尾；
  - **协作式 vs 抢占式 (Cooperative vs Preemptive)**：协作式依赖任务显式 yield，抢占式由调度器通过定时器中断强制挂起任务。
  - **加权赤字轮转 (Deficit Round Robin / DRR)**：为不同权重任务维护赤字计数器（Credit），解决不同任务包大小/耗时不均的问题。

#### 3. 分级任务管理器 (Task Manager with TTL & Quota)
- **四层能力递进**：
  - Level 1-2：按用户增删查改任务，维护优先级与创建时间排序；
  - Level 3：任务带 TTL 自动失效，以及基于用户配额（Quota）的任务分配（每个用户同时处于活动期的任务数严格受限）；
  - Level 4：时光旅行（Time-Travel Look-back）与逾期未完成任务报告（Overdue Reporting）。

</div>
</details>

---

### 21. 时间戳任务调度器与直接 ID 淘汰 (Timestamp Task Scheduler with Direct ID Removal)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">堆/调度 21</span>
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

### 22. 极速限价订单簿系统 (Limit Order Book System with Price-Time Priority)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">链表/系统 22</span>
  <span class="review-card-title">极速限价订单簿系统 (Limit Order Book System with Price-Time Priority)</span>
  <span class="review-card-tag">分层有序映射 · 双向链表 FIFO 队列 · 哈希句柄直指 · 价格优先-时间优先 · O(1) 撤单与深度聚合</span>
</summary>
<div class="review-card-content">

> 💡 **题目类型**：独立工业级系统与复合数据结构设计（无直接对应单一 LeetCode 原题，为金融交易与订单匹配引擎的核心基础组件，请直接使用卡片内置完整测试桩在本地运行验证）

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

**题目原文 (Problem Statement)**：
> **Limit Order Book (LOB) System**:
> Design a low-latency Limit Order Book maintaining resting limit orders with **Price-Time Priority (FIFO)**:
> - `add_order(side: str, price: float, qty: int, order_id: str) -> bool`: Inserts a resting limit order. Rejects invalid quantity ($\le 0$), invalid price ($\le 0$), or duplicate `order_id`.
> - `cancel_order(order_id: str) -> bool`: Cancels and evicts an order by `order_id` in amortized $\mathcal{O}(1)$ time. Returns `False` if the order does not exist.
> - `best_bid() -> Optional[float]`: Returns the current highest buy price, or `None` if the bid book is empty ($\mathcal{O}(1)$ time).
> - `best_ask() -> Optional[float]`: Returns the current lowest sell price, or `None` if the ask book is empty ($\mathcal{O}(1)$ time).
> - `top_of_book_volume() -> Tuple[int, int]`: Returns the total aggregate resting volume at the top-of-book `(best_bid_volume, best_ask_volume)` in $\mathcal{O}(1)$ time.
>
> **Execution & Priority Rules**:
> 1. **Price Priority**: Higher buy orders precede lower buy orders; lower sell orders precede higher sell orders.
> 2. **Time Priority (FIFO)**: Orders placed at the same price level must be queued and filled strictly in order of arrival.
> 3. **Fast Cancellation**: Order eviction must unlink the target order directly without scanning other orders at that price level.

**接口定义**：
```python
class LimitOrderBook:
    def __init__(self): ...
    def add_order(self, side: str, price: float, qty: int, order_id: str) -> bool: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def best_bid(self) -> Optional[float]: ...
    def best_ask(self) -> Optional[float]: ...
    def top_of_book_volume(self) -> Tuple[int, int]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">🎯 架构设计与核心机制</div>

订单簿的核心难点在于兼顾**全局价格极值快速查询**、**单价位内部严格时间先后排队**与**任意订单的即时注销**：

1. **两层分级存储结构 (Two-Level Map + FIFO Doubly Linked List)**：
   - **第一层：价格有序索引 (Sorted Price Map)**：
     - 买盘（Bids）：维护价格从高到低的有序表。最优买价始终位于首位；
     - 卖盘（Asks）：维护价格从低到高的有序表。最优卖价始终位于首位；
     - 在 C++ 中使用 `std::map<double, PriceLevel, std::greater<double>>`（买盘）与 `std::map<double, PriceLevel, std::less<double>>`（卖盘）；在 Python 中通过二分维护有序价格列表 `sorted_bid_prices` 与 `sorted_ask_prices`。
   - **第二层：价位内部双向链表 (PriceLevel FIFO Queue)**：
     - 每个价位包含一个带有虚拟头尾哨兵的双向链表（Doubly Linked List），新订单追加至链表尾部，满足严格 FIFO 时间优先；
     - 价位对象维护一个标量字段 `total_volume`，在订单添加与撤销时增减，使 `top_of_book_volume()` 聚合查询能在 $\mathcal{O}(1)$ 内返回，无需遍历链表。

2. **订单哈希索引表实现 $\mathcal{O}(1)$ 快速撤单 (Order Location Map)**：
   - 维护全局哈希表 `order_map: Dict[order_id, OrderNode]`（C++ 中存储节点迭代器 `std::list<Order>::iterator`）；
   - 撤单时通过 `order_id` 直接定位到该节点所属的 `PriceLevel` 与链表节点，执行双向链表常数时间摘除：
     $$\text{node.prev.next} = \text{node.next}, \quad \text{node.next.prev} = \text{node.prev}$$
   - 若该价位链表为空（`total_volume == 0` 或 `is_empty()`），从第一层的有序价格表中同步移除该价位，保证盘口查询不命中空价位。

</div>

<div class="review-block">
<div class="review-block-label">📌 核心代码 (Python 3)</div>

```python
import bisect
from typing import Dict, List, Optional, Tuple

class OrderNode:
    """双向链表节点：维护单笔委托订单的生命周期与位置句柄"""
    def __init__(self, order_id: str, side: str, price: float, qty: int):
        self.order_id = order_id
        self.side = side.upper()
        self.price = price
        self.qty = qty
        self.prev: Optional['OrderNode'] = None
        self.next: Optional['OrderNode'] = None
        self.level: Optional['PriceLevel'] = None

class PriceLevel:
    """双向链表维护同一价位上的订单，严格保证时间优先 (FIFO)"""
    def __init__(self, price: float):
        self.price = price
        self.total_volume = 0
        self.head = OrderNode("", "", 0.0, 0)
        self.tail = OrderNode("", "", 0.0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def append(self, order: OrderNode) -> None:
        order.prev = self.tail.prev
        order.next = self.tail
        self.tail.prev.next = order
        self.tail.prev = order
        order.level = self
        self.total_volume += order.qty

    def remove(self, order: OrderNode) -> None:
        order.prev.next = order.next
        order.next.prev = order.prev
        self.total_volume -= order.qty
        order.prev = None
        order.next = None
        order.level = None

    def is_empty(self) -> bool:
        return self.head.next is self.tail

class LimitOrderBook:
    """
    极速限价订单簿系统 (Limit Order Book)
    结构组合: 有序价格表 (Bids 降序 / Asks 升序) + 内部 FIFO 双向链表 + 订单哈希句柄表
    """
    def __init__(self):
        self.bids: Dict[float, PriceLevel] = {}
        self.sorted_bid_prices: List[float] = []  # 降序维护: 存相反数或按降序二分

        self.asks: Dict[float, PriceLevel] = {}
        self.sorted_ask_prices: List[float] = []  # 升序维护

        self.order_map: Dict[str, OrderNode] = {}

    def _add_price_level(self, side: str, price: float) -> PriceLevel:
        level = PriceLevel(price)
        if side == "BUY":
            self.bids[price] = level
            idx = bisect.bisect_left([-p for p in self.sorted_bid_prices], -price)
            self.sorted_bid_prices.insert(idx, price)
        else:
            self.asks[price] = level
            idx = bisect.bisect_left(self.sorted_ask_prices, price)
            self.sorted_ask_prices.insert(idx, price)
        return level

    def _remove_price_level(self, side: str, price: float) -> None:
        if side == "BUY":
            if price in self.bids:
                del self.bids[price]
                idx = bisect.bisect_left([-p for p in self.sorted_bid_prices], -price)
                if idx < len(self.sorted_bid_prices) and self.sorted_bid_prices[idx] == price:
                    self.sorted_bid_prices.pop(idx)
        else:
            if price in self.asks:
                del self.asks[price]
                idx = bisect.bisect_left(self.sorted_ask_prices, price)
                if idx < len(self.sorted_ask_prices) and self.sorted_ask_prices[idx] == price:
                    self.sorted_ask_prices.pop(idx)

    def add_order(self, side: str, price: float, qty: int, order_id: str) -> bool:
        if order_id in self.order_map or qty <= 0 or price <= 0:
            return False
        side = side.upper()
        if side not in ("BUY", "SELL"):
            return False

        price_map = self.bids if side == "BUY" else self.asks
        if price not in price_map:
            level = self._add_price_level(side, price)
        else:
            level = price_map[price]

        node = OrderNode(order_id, side, price, qty)
        level.append(node)
        self.order_map[order_id] = node
        return True

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.order_map:
            return False
        node = self.order_map.pop(order_id)
        level = node.level
        level.remove(node)

        # 若当前价位订单清空，从价格索引表中注销该价位
        if level.is_empty():
            self._remove_price_level(node.side, node.price)
        return True

    def best_bid(self) -> Optional[float]:
        return self.sorted_bid_prices[0] if self.sorted_bid_prices else None

    def best_ask(self) -> Optional[float]:
        return self.sorted_ask_prices[0] if self.sorted_ask_prices else None

    def top_of_book_volume(self) -> Tuple[int, int]:
        bid_vol = self.bids[self.sorted_bid_prices[0]].total_volume if self.sorted_bid_prices else 0
        ask_vol = self.asks[self.sorted_ask_prices[0]].total_volume if self.sorted_ask_prices else 0
        return bid_vol, ask_vol

if __name__ == "__main__":
    lob = LimitOrderBook()
    # 1. 挂入多笔委托，包含相同价位不同时间戳的排队
    assert lob.add_order("BUY", 100.5, 10, "ord_1") is True
    assert lob.add_order("BUY", 100.5, 20, "ord_2") is True  # 同价位排在 ord_1 之后
    assert lob.add_order("BUY", 100.0, 50, "ord_3") is True
    assert lob.add_order("SELL", 101.0, 15, "ord_4") is True
    assert lob.add_order("SELL", 102.0, 30, "ord_5") is True

    # 2. 校验盘口最优价与挂单量
    assert lob.best_bid() == 100.5
    assert lob.best_ask() == 101.0
    bid_v, ask_v = lob.top_of_book_volume()
    assert bid_v == 30  # 10 + 20
    assert ask_v == 15

    # 3. 撤销排在队首的 ord_1 (最优价不变，挂单量缩减为 20)
    assert lob.cancel_order("ord_1") is True
    bid_v, _ = lob.top_of_book_volume()
    assert bid_v == 20
    assert lob.best_bid() == 100.5

    # 4. 撤销同价位剩余的 ord_2 (该价位清空，最优买价自动下移至次优价 100.0)
    assert lob.cancel_order("ord_2") is True
    assert lob.best_bid() == 100.0
    bid_v, _ = lob.top_of_book_volume()
    assert bid_v == 50

    # 5. 重复撤单校验幂等与不存在处理
    assert lob.cancel_order("ord_1") is False
    print("✅ LimitOrderBook Python all test cases passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">📌 生产级 C++ 实现 (C++17 / C++20)</div>

```cpp
#include <iostream>
#include <string>
#include <unordered_map>
#include <map>
#include <list>
#include <optional>
#include <cassert>

struct Order {
    std::string order_id;
    std::string side;
    double price;
    int qty;
};

struct PriceLevel {
    double price{0.0};
    int total_volume{0};
    std::list<Order> orders;
};

class LimitOrderBook {
private:
    struct OrderLocation {
        std::string side;
        double price;
        std::list<Order>::iterator it;
    };

    // 买盘降序排列（最高买价优先），卖盘升序排列（最低卖价优先）
    std::map<double, PriceLevel, std::greater<double>> bids_;
    std::map<double, PriceLevel, std::less<double>> asks_;
    std::unordered_map<std::string, OrderLocation> order_map_;

public:
    bool addOrder(const std::string& side, double price, int qty, const std::string& order_id) {
        if (order_map_.count(order_id) || qty <= 0 || price <= 0.0) return false;

        if (side == "BUY") {
            auto& level = bids_[price];
            level.price = price;
            level.total_volume += qty;
            level.orders.push_back({order_id, side, price, qty});
            auto it = std::prev(level.orders.end());
            order_map_[order_id] = {side, price, it};
        } else if (side == "SELL") {
            auto& level = asks_[price];
            level.price = price;
            level.total_volume += qty;
            level.orders.push_back({order_id, side, price, qty});
            auto it = std::prev(level.orders.end());
            order_map_[order_id] = {side, price, it};
        } else {
            return false;
        }
        return true;
    }

    bool cancelOrder(const std::string& order_id) {
        auto it = order_map_.find(order_id);
        if (it == order_map_.end()) return false;

        const auto& loc = it->second;
        if (loc.side == "BUY") {
            auto bit = bids_.find(loc.price);
            if (bit != bids_.end()) {
                bit->second.total_volume -= loc.it->qty;
                bit->second.orders.erase(loc.it);
                if (bit->second.orders.empty()) {
                    bids_.erase(bit);
                }
            }
        } else {
            auto ait = asks_.find(loc.price);
            if (ait != asks_.end()) {
                ait->second.total_volume -= loc.it->qty;
                ait->second.orders.erase(loc.it);
                if (ait->second.orders.empty()) {
                    asks_.erase(ait);
                }
            }
        }
        order_map_.erase(it);
        return true;
    }

    std::optional<double> bestBid() const {
        if (bids_.empty()) return std::nullopt;
        return bids_.begin()->first;
    }

    std::optional<double> bestAsk() const {
        if (asks_.empty()) return std::nullopt;
        return asks_.begin()->first;
    }

    std::pair<int, int> topOfBookVolume() const {
        int bid_vol = bids_.empty() ? 0 : bids_.begin()->second.total_volume;
        int ask_vol = asks_.empty() ? 0 : asks_.begin()->second.total_volume;
        return {bid_vol, ask_vol};
    }
};

int main() {
    LimitOrderBook lob;
    assert(lob.addOrder("BUY", 100.5, 10, "ord_1"));
    assert(lob.addOrder("BUY", 100.5, 20, "ord_2"));
    assert(lob.addOrder("BUY", 100.0, 50, "ord_3"));
    assert(lob.addOrder("SELL", 101.0, 15, "ord_4"));
    assert(lob.addOrder("SELL", 102.0, 30, "ord_5"));

    assert(lob.bestBid().value() == 100.5);
    assert(lob.bestAsk().value() == 101.0);
    auto [bv1, av1] = lob.topOfBookVolume();
    assert(bv1 == 30);
    assert(av1 == 15);

    assert(lob.cancelOrder("ord_1"));
    auto [bv2, av2] = lob.topOfBookVolume();
    assert(bv2 == 20);
    assert(lob.bestBid().value() == 100.5);

    assert(lob.cancelOrder("ord_2"));
    assert(lob.bestBid().value() == 100.0);
    auto [bv3, av3] = lob.topOfBookVolume();
    assert(bv3 == 50);

    assert(!lob.cancelOrder("ord_1"));
    std::cout << "✅ LimitOrderBook C++ all test cases passed!\n";
    return 0;
}
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与系统权衡分析</div>

| 操作方法 | 时间复杂度 | 空间复杂度 | 关键路径开销来源 |
| :--- | :--- | :--- | :--- |
| `best_bid()` / `best_ask()` | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | 直接读取有序价格表根节点/首元素指针 |
| `top_of_book_volume()` | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | 增量维护标量字段，直接返回无需遍历订单链表 |
| `cancel_order(order_id)` | 均摊 $\mathcal{O}(1)$ 或 $\mathcal{O}(\log P)$ | $\mathcal{O}(1)$ | 哈希索引定位 $\mathcal{O}(1)$ + 双向链表节点解引用摘除 $\mathcal{O}(1)$；当价位清空注销时需自平衡树删除 $\mathcal{O}(\log P)$ |
| `add_order(side, ...)` | $\mathcal{O}(\log P)$ | $\mathcal{O}(1)$ | 有序红黑树价位检索/插入 $\mathcal{O}(\log P)$ + 双向链表尾部追加 $\mathcal{O}(1)$（$P$ 为活跃价位总数） |

- **高频低延迟工程演进 (High-Frequency Engineering Considerations)**：
  1. **固定步长离散化与直接数组寻址 (Tick Discretization via Flat Array)**：
     - 若标的资产报价区间有界且最小跳动单位（Tick Size）固定（例如价格在 $[1.00, 1000.00]$ 且 step 为 $0.01$），生产系统会废除红黑树，改用预分配的扁平连续数组 `PriceLevel levels[100000]`；
     - 价格通过公式 $\text{index} = \lfloor \frac{\text{price} - \text{min\_price}}{\text{tick\_size}} \rfloor$ 直接寻址，将挂单开销从 $\mathcal{O}(\log P)$ 骤降至严格 $\mathcal{O}(1)$，彻底消除红黑树指针跳转导致的 CPU Cache Miss。
  2. **内存池与零堆分配 (Memory Arena / Zero-Allocation Pool)**：
     - 在关键交易链路中，禁止使用系统级动态分配器（`malloc` / `new`）；
     - 采用预分配固定容量的循环对象池（Object Pool）或环形缓冲区（Ring Buffer），订单节点的分配与归还仅涉及数组下标的原子递增，避免内存碎片与缺页中断。
  3. **并发模型与单线程撮合 (Single-Threaded Actor Model)**：
     - 现代撮合引擎核心通常绑定独立物理 CPU 核（CPU Pinning），采用单线程纯内存无锁循环，输入网络事件通过无锁单生产者单消费者队列（SPSC Ring Buffer）排队注入，消除互斥锁上下文切换与缓存一致性风暴。

</div>

</div>
</details>

---

---

## 模块四：NeetCode 极速速记与高频题型决策树 (NeetCode Quick-Recall Decision Framework & Mental Models)

为了在面试高压环境下于 10 秒内迅速锁定最优解题范式，将链表、栈、堆领域最具代表性的核心考题沉淀为如下标准化心智模型与决策树速记矩阵：

### 4.1 链表题型心智模型与 10 秒决策速查 (Linked List Decision Matrix)

| 场景模式 (Pattern) | 触发关键词 / 题型特征 | 黄金解法架构 (Optimal Archetype) | 代表题型 (NeetCode / LC) |
| :--- | :--- | :--- | :--- |
| **快慢双指针 (Floyd)** | 环检测、求环入口、找链表中点、回文判断 | `slow = slow.next`, `fast = fast.next.next`；相遇后一针回原点单步同步走 | LC 141 (环检测), LC 142 (环入口), LC 876 (中点), LC 234 (回文) |
| **虚拟哨兵头节点 (Dummy)** | 链表头可能被删除、被合并、链表前插入 | `dummy = ListNode(0, head)`，统一内部与头节点逻辑 | LC 19 (删倒数第 N), LC 21 (合并两链表), LC 2 (两数相加), LC 86 (分隔) |
| **三指针局部翻转** | 反转整链、反转区间、K个一组反转 | `nxt = curr.next; curr.next = prev; prev = curr; curr = nxt` | LC 206 (反转), LC 92 (反转II), LC 25 (K个一组) |
| **拆分/穿插指针映射** | 深拷贝带随机指针的链表、重排链表 | 原地复制 `node.next = copyNode` 后交叉拆分；快慢针截半后交替穿插 | LC 138 (复制随机指针), LC 143 (重排链表) |
| **复合哈希双向链表** | 严格 $\mathcal{O}(1)$ 缓存插入/访问/淘汰 | 双向链表记录时间顺序 + 哈希表直指节点；头尾双哨兵四指针无分支断连 | LC 146 (LRU), LC 460 (LFU), 餐厅候补队列 |
| **动态数组末尾置换** | $\mathcal{O}(1)$ 插入、删除与等概率随机抽取 | 动态数组保存元素 + 哈希表记录下标；删除时与末尾元素置换后执行 `pop()` | LC 380 (O(1)集合), LC 381 (允许重复), O(1) 随机容器 |
| **分层有序映射 + 双向链表** | 价格优先与时间优先、$\mathcal{O}(1)$ 撤单与深度查询 | 有序价格表 (`std::map`/二分) 维护盘口极值 + 双向链表维护时间优先队列 + 哈希表记录节点指针 | 卡片 22 (极速限价订单簿系统) |

---

### 4.2 栈与单调结构心智模型 (Stack & Monotonic Structure Matrix)

| 场景模式 (Pattern) | 触发关键词 / 题型特征 | 黄金解法架构 (Optimal Archetype) | 代表题型 (NeetCode / LC) |
| :--- | :--- | :--- | :--- |
| **配对与平衡检验** | 括号有效性、消除相邻重复项、回文消除 | 栈存储左半边期待，遇到右半边校验栈顶并弹出 | LC 20 (有效括号), LC 1047 (消除相邻重复) |
| **单调递增/递减栈** | 下一个更大/更小元素、柱状图最大矩形、股票买卖天数 | 维护栈内元素严格单调；破坏单调时弹出栈顶并结算其右边界 | LC 739 (每日温度), LC 496 (下一个更大), LC 84 (柱状图最大矩形), LC 853 (车队) |
| **单调双端队列** | 滑动窗口动态极值 (最大值/最小值) | 队头到队尾维持严格单调递减，队尾剔除较小者，队头剔除过期索引 | LC 239 (滑动窗口最大值), LC 1438 |
| **双栈表达式求值** | 四则运算、括号嵌套、运算符优先级 | 操作数栈 + 运算符栈；乘除即时结合，遇到括号递归下降分治 | LC 150 (逆波兰), LC 224 (基础计算器), LC 227 (计算器II) |
| **单调栈贪心去重** | 字典序最小的不重复子序列 | 单调增栈 + 字符最后出现位置哈希 + 栈内存在性集合；可后现者果断弹出 | LC 316 / LC 1081 (去重保持最小字典序) |

---

### 4.3 堆与优先队列心智模型 (Heap & Priority Queue Matrix)

| 场景模式 (Pattern) | 触发关键词 / 题型特征 | 黄金解法架构 (Optimal Archetype) | 代表题型 (NeetCode / LC) |
| :--- | :--- | :--- | :--- |
| **Top-K 动态维护** | 求海量数据中最大/最小的 K 个元素 | 维持容量为 $K$ 的**小顶堆**（求最大 K 个）或大顶堆，超出容量弹出堆顶 | LC 215 (第K大), LC 347 (前K高频), LC 703 (数据流第K大) |
| **多路归并排序** | 合并 K 个升序链表 / 数组 | 堆内始终只维护各有序序列的当前游标头部（容量为 $K$），弹出堆顶并推进后继 | LC 23 (合并K个升序链表), LC 378 (矩阵第K小) |
| **对顶平衡双堆** | 数据流中无序插入，实时查询中位数 | 大顶堆（存放小半部）+ 小顶堆（存放大半部）；容量差严格 $\le 1$ | LC 295 (数据流中位数) |
| **冷却调度与模拟** | 相同任务必须间隔 $N$ 时间单位冷却 | 贪心大顶堆按频次降序挑选 + 冷却队列暂存等待倒计时归零 | LC 621 (任务调度器), LC 355 (设计推特) |
