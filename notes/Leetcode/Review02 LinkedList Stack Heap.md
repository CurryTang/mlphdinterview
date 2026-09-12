# 复习卡片：链表、栈与堆 (Review Flashcards · Linked List, Stack & Heap)

本篇为算法面试高频复习卡片第二辑：系统整理**链表与复合哈希结构 (Linked List & Hash-Linked Structures)**、**栈与表达式计算 (Stack & Expression Parsing)** 以及**堆与优先队列 (Heap & Priority Queue)** 的核心高频考题、工业级变体、生产级实现与时空复杂度全景。

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

在系统架构与分布式计算面试中，该题常引申出以下五大高频工业级系统演进追问：

| 追问编号 | 追问主题 | 核心变异条件 / 工业需求 | 架构突破口与实现策略 |
|---|---|---|---|
| **追问 1** | **TTL 键过期淘汰 (Add TTL)** | 写入时附带存活时间 `ttl`；读取已过期的键返回 `-1`，且过期键不可无谓占用物理容量。 | **惰性淘汰 (Lazy Eviction)**：访问时检查时间戳并即时摘除；配合**主动定时轮询/小根堆 (Proactive Sweeper)** 回收冷过期数据。 |
| **追问 2** | **LFU 频次淘汰 (Add LFU)** | 置换时首先淘汰使用频率最小的键；频次平局（Tie）时按最近最少使用淘汰。 | **双层映射哈希**：`key_node_map` + `freq_dll_map` + 维护全局最小频次标尺 `min_freq`，严控各状态转移全为 $O(1)$。 |
| **追问 3** | **四端队列与随机索引 (4-End Deque)** | 保持 $O(1)$ 约束，同时支持 `lpush`, `rpush`, `lpop`, `rpop` 以及基于下标的随机访问 `get_by_index(i)`。 | 双向链表原生支持四端 $O(1)$ 弹入弹出，但下标访问退化为 $O(N)$；工业级解法采用**分块双向链表 (Chunked Deque / Quicklist)** 或**动态环形数组 (Circular Ring Buffer)** 实现端点 $O(1)$、索引均摊 $O(1)$。 |
| **追问 4** | **LRU 访问路径迭代 (Print Path)** | 按从“最久未被访问”到“最新被访问”的时序遍历全量有效缓存条目。 | 遍历双向链表的拓扑序：从哨兵头节点 `head.next` 逐级推进至哨兵尾节点 `tail.prev`，输出迭代器生成器。 |
| **追问 5** | **高缓存未命中率调优 (High Miss Rate)** | 生产监控告警缓存未命中率飙升（> 30%），如何在前端/网关及服务端综合提升缓存命中率？ | 容量重估（工作集分析）、**置换策略跃迁 (TinyLFU 门禁准入/ARC)**、**多级缓存架构 (L1 本地内存 + L2 分布式 Redis)**、**空间局部性预取 (Prefetching)** 与**分段锁并发降低争用**。 |

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心机制深度剖析</div>

#### 1. 经典 LRU：双向链表 (DLL) + 哈希表 (HashMap) 黄金解耦
- **为何单链表不行？**
  - 单向链表删除已知节点必须从头开始遍历定位其 `prev` 前驱节点，单次删除耗时 $O(N)$。
  - **双向链表 (Doubly Linked List, DLL)** 内部节点持有 `prev` 与 `next`，给定节点指针时，可在 $O(1)$ 时间内完成局部断链拼接：
    ```python
    node.prev.next = node.next
    node.next.prev = node.prev
    ```
- **伪头伪尾哨兵 (Dummy Head & Tail) 的防空指针不变式**：
  - 初始化虚拟节点 `head` 和 `tail`，并预先相连：`head.next = tail`, `tail.prev = head`。
  - 所有真实数据节点严格存在于 `head` 和 `tail` 之间。在执行插入、删除、迁移等指针重定向操作时，彻底消除对“当前是否为头节点”、“链表是否为空”的繁琐分支判断。
- **最新与最久时序约定**：
  - 约定紧靠 `tail.prev` 的节点为“最新使用 (Most Recently Used, MRU)”。
  - 约定紧靠 `head.next` 的节点为“最久未使用 (Least Recently Used, LRU)”。
  - 每次读取 `get(key)` 或更新 `put(key, val)` 时，通过哈希表在 $O(1)$ 时间定位节点，将其从原位置摘除（`_remove(node)`），并重新追加至尾部（`_append_to_tail(node)`）。若容量超限，则直接将 `head.next` 弹出淘汰。

#### 2. TTL (Time-To-Live) 过期键清理机制
在缓存系统（如 Redis、Memcached）中，TTL 清理遵循双轨制：
1. **被动/惰性淘汰 (Passive / Lazy Eviction)**：
   - 当客户端调用 `get(key)` 时，首先比对 `time.time() > node.expiry`。若已过期，立即在内部调用链表删除逻辑将其连根拔起并从哈希表中移除，返还容量配额，直接返回 `-1`。
2. **主动扫描淘汰 (Proactive Sweeper)**：
   - 若某过期键长期无人问津，纯惰性淘汰会导致内存泄露（冷数据死锁占用物理容量）。
   - 生产解法：在外部维护一个小顶堆 `heapq` 记录 `(expiry, key)`，或挂载后台周期性定时线程（时间轮 Timing Wheel / 采样扫描），定时弹出已过期条目。

#### 3. 应对缓存高未命中率（High Cache Miss Rate）的系统级治理体系
当生产系统中监控到 Cache Miss Rate 异常居高不下时，面试标准答题体系应覆盖以下五个系统层级：
1. **容量重估与工作集拟合 (Working Set Sizing)**：
   - 使用 Amdahl 定律与活跃工作集（Working Set）模型评估：当前缓存容量是否远小于热点数据集基数。若缓存容量小于高频读写数据集总和，将发生剧烈的**缓存颠簸 (Cache Thrashing)**，唯有垂直扩容。
2. **击破 LRU 原生缺陷：准入控制与 TinyLFU / ARC**：
   - **LRU 的致命软肋——突发扫描污染 (Scan Resistance)**：当发生周期性全表扫描或一次性冷数据暴增（One-hit Wonders）时，海量冷数据涌入会瞬间将真正的长期热点数据从 LRU 队列中全部挤出淘汰。
   - **跃迁方案 A (2Q / ARC)**：2Q 采用两个队列，新数据先进入 FIFO 试用期队列，只有在此期间再次命中才提升至 LRU 队列；ARC（自适应替换缓存）通过四组链表自适应动态调节频次与新鲜度的置换权重。
   - **跃迁方案 B (W-TinyLFU)**：现代高性能缓存库（如 Java Caffeine）标配。在 LRU 头部前置基于 Count-Min Sketch 的准入过滤器（Admission Filter）。新元素只有在估计历史访问频次严格大于待淘汰牺牲品（Victim）的频次时，才允许换入主缓存区，否则直接拒绝准入，彻底杜绝缓存污染。
3. **多级缓存拓扑 (Multi-Tier Caching Architecture)**：
   - 建立 **L1 本地内存缓存**（In-process Memory，如 Caffeine / Python local dict，延迟 $< 100\text{ns}$，无网络序列化开销）+ **L2 分布式集中缓存**（Redis / Memcached 集群，延迟 $1\sim 3\text{ms}$，跨机共享全局一致）+ **L3 底层持久化存储 (DB)**。
4. **空间局部性预取 (Prefetching & Cache Warming)**：
   - 针对有时序连续性或关联关系的访问模式，在发生单点未命中回源时，异步批量预取关联扇出节点（Prefetch Next $K$ Items）。
   - 结合系统重启或发布时的离线热点统计，进行提前**缓存预热 (Cache Warming)**。
5. **并发争用与分段锁优化 (Striped Locking / Sharding)**：
   - 单体 LRU 链表上的所有操作都需要加互斥锁，高并发读写下锁竞争会造成线程排队与超时未命中。采用一致性哈希分片为多槽独立 LRU（如划分为 16 或 64 个并发 Segment），各分片持独占读写锁，显著削减排队延迟。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
import time
from typing import Dict, Optional, List, Tuple

class DLLNode:
    """双向链表数据节点"""
    __slots__ = ('key', 'val', 'prev', 'next', 'expiry')
    
    def __init__(self, key: int = 0, val: int = 0, expiry: float = float('inf')):
        self.key = key
        self.val = val
        self.expiry = expiry
        self.prev: Optional['DLLNode'] = None
        self.next: Optional['DLLNode'] = None


class LRUCacheWithTTL:
    """
    生产级 LRU 缓存：
    1. 基础 get/put 操作 O(1)
    2. 支持 TTL 键生命周期（惰性失效淘汰）
    3. 支持按访问时序 (LRU -> MRU) 全量遍历
    """
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.map: Dict[int, DLLNode] = {}
        
        # 伪头伪尾哨兵节点
        self.head = DLLNode()
        self.tail = DLLNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: DLLNode) -> None:
        """从双向链表中原子摘除指定节点"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _append_to_tail(self, node: DLLNode) -> None:
        """将节点压入链表尾部（标记为最新访问 MRU）"""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def _is_expired(self, node: DLLNode) -> bool:
        """判断节点是否已超时过期"""
        return time.time() > node.expiry

    def _evict_node(self, node: DLLNode) -> None:
        """物理销毁并释放节点空间"""
        self._remove(node)
        self.map.pop(node.key, None)

    def get(self, key: int) -> int:
        """读取指定键的值，自动触发惰性过期与时序翻新，O(1)"""
        if key not in self.map:
            return -1
        
        node = self.map[key]
        # 检查 TTL 惰性过期
        if self._is_expired(node):
            self._evict_node(node)
            return -1
        
        # 翻新至尾部 (MRU)
        self._remove(node)
        self._append_to_tail(node)
        return node.val

    def put(self, key: int, value: int, ttl: Optional[float] = None) -> None:
        """写入/覆写指定键，可选附带秒级 TTL，O(1)"""
        expiry = time.time() + ttl if ttl is not None else float('inf')
        
        if key in self.map:
            node = self.map[key]
            node.val = value
            node.expiry = expiry
            self._remove(node)
            self._append_to_tail(node)
            return

        # 若已达容量上限，弹出最久未使用的合法项（从 head.next 开始）
        if len(self.map) >= self.capacity:
            lru_victim = self.head.next
            self._evict_node(lru_victim)

        # 插入新节点
        new_node = DLLNode(key, value, expiry)
        self.map[key] = new_node
        self._append_to_tail(new_node)

    def get_lru_order(self) -> List[Tuple[int, int]]:
        """
        按从最久未访问 (LRU) 到最新访问 (MRU) 遍历全量活跃节点
        返回: [(key, val), ...]
        """
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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：
  - `get(key)`：哈希表寻址 $O(1)$，双向链表断链与尾部拼接 $O(1)$，总耗时严格为 $\mathcal{O}(1)$。
  - `put(key, value)`：哈希表更新/插入 $O(1)$，淘汰旧项与压入尾部均为 $O(1)$，总耗时严格为 $\mathcal{O}(1)$。
  - `get_lru_order()`：一次性链表拓扑遍历，耗时 $\mathcal{O}(K)$，其中 $K \le \text{capacity}$。
- **空间复杂度**：
  - 维护双向链表与哈希映射，额外空间开销严格为 $\mathcal{O}(C)$，其中 $C = \text{capacity}$。
- **高频边界致命陷阱**：
  1. **哨兵指针未互指**：初始化 `head` 和 `tail` 时切记绑定 `head.next = tail` 与 `tail.prev = head`。
  2. **淘汰项字典不同步**：在从链表头部摘除最老节点 `victim = head.next` 后，**必须同步调用 `del self.map[victim.key]`**。
  3. **覆写键时的容量误判**：当调用 `put(key, new_val)` 且 `key` 已存在时，仅做原地修改与尾部挪移，**绝对不能触发容量溢出淘汰判断**！

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

在各平台（如 Lark / HackerRank）及高阶算法考核中，该题常派生出以下**五大高频变体与考察深度**：

| 变体编号 | 核心变体名称 | 核心特征 / 变异条件 | 破局关键与指针重组策略 |
|---|---|---|---|
| **变体 1** | **经典 K 组翻转 (LC 25)** | 节点总数不是 $k$ 的整数倍时，最后剩余少于 $k$ 个的节点保持原有顺序。 | **先探测后反转**：用前向探针走 $k$ 步确认完整组存在；若存在则断开局部反转并两头回接；若不足 $k$ 步则保持原样终止。 |
| **变体 2** | **尾部不足亦翻转 (Reverse Partial Tail)** | 非标准变体：若链表末端剩余节点不足 $k$ 个，**依然无条件翻转**该尾部段落。 | **取消前置长度截断**：主循环不再提前探针终止，只要当前剩余节点数 $\ge 1$，均截取最多 $k$ 个节点无条件翻转回接。 |
| **变体 3** | **两阶段分步热身 (Two-Parter Warmup)** | 一面热身题：第一问先实现基础单链表就地翻转；第二问在此基础上拓展为 K 组翻转。 | **高内聚子函数设计**：将单链表反转提取为通用原语 `reverse_segment(head, tail)`，在 K 组翻转外层做指针状态机调用。 |
| **变体 4** | **自定义 ListNode 与测试脚手架 (Test Scaffolding)** | 面试平台无内置链表支持，需候选人现场手写 `ListNode` 类及数组与链表互相转换工具。 | 手写结构体 `class ListNode: def __init__(self, val=0, next=None)`，并提供 `build_list(arr)` 与 `to_list(head)` 测试驱动。 |
| **变体 5** | **组间逆序而组内保序 (Reverse Group Order, Not Within)** | 保持每个 $k$ 组内部节点的相对顺序不变，但将**各个组本身的拓扑顺序逆序拼接**。例如 $k=3$，$1\to2\to3\to4\to5\to6$ 变换为 $4\to5\to6\to1\to2\to3$。 | **切断分段 + 组级逆序缝合**：不改变组内指针，按 $k$ 步切断并收集各组的 `(group_head, group_tail)`，利用栈或列表逆序将各组的 `tail.next` 串接至上一组的 `head`。 |
| **变体 6** | **边界探针分析 (Edge Probes)** | $k=1$ 时的退化性；$k > \text{length}$ 时的表现；空链表防护。 | 当 $k=1$ 时直接返回原头节点（空翻转无操作）；当 $k > N$ 时，变体 1 原样返回，变体 2 等价于全链表反转。 |

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心机制深度剖析</div>

#### 1. 经典 K 组翻转：四指针锚定与局部缝合模型
翻转一个长度为 $k$ 的链表子区间并完美缝合回主链表，需要精准锁定四个指针：
1. `group_prev`：当前组的前一个节点（前驱锚点，翻转后需指向新组头）；
2. `group_start`：当前组的起始节点（翻转后变成新组尾）；
3. `group_end`：当前组的末尾节点（翻转后变成新组头）；
4. `next_group_head`：紧随当前组之后的下一组起始节点（`group_end.next`）。

**标准执行步骤**：
- 步骤 1（探针）：从 `group_prev` 出发向前走 $k$ 步。若中途触及 `None`，说明剩余节点不足 $k$ 个，跳出循环；
- 步骤 2（截断）：记录 `next_group_head = group_end.next`，令 `group_end.next = None` 暂时将当前组物理独立；
- 步骤 3（局部翻转）：调用局部翻转子例程 `reverse(group_start)`，返回反转后的新头（即原 `group_end`）；
- 步骤 4（回接）：`group_prev.next = group_end`，并将新尾部 `group_start.next = next_group_head`；
- 步骤 5（步进）：`group_prev = group_start`，继续下一轮迭代。

#### 2. 组间逆序组内保序（Reverse Group Order, Not Within Groups）算法设计
- 该变体属于高频变形。核心是**绝对不能动组内的 `next` 指针**！
- 算法策略：
  1. 遍历链表，每走 $k$ 步将其断开（`tail.next = None`），并将该组的 `(head, tail)` 作为一个整体存储进列表 `groups = []`。若最后一段不足 $k$ 个，视面试官约定亦存入。
  2. 逆向遍历 `groups` 列表（从最后一个组到第一个组），将 `groups[i].tail.next = groups[i-1].head`。
  3. 首组的 `head` 即为新链表的头节点，原第一组的 `tail.next = None`。时间 $O(N)$，空间 $O(N/k)$（或就地链表逆序头插法做到 $O(1)$ 额外空间）。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码（含测试脚手架与全变体）</div>

```python
from typing import Optional, List, Tuple

# -------------------------------------------------------------
# 面试现场手写脚手架 (Test Scaffolding)
# -------------------------------------------------------------
class ListNode:
    """链表基础数据节点"""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

def build_list(values: List[int]) -> Optional[ListNode]:
    """从数组构建单链表"""
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def to_list(head: Optional[ListNode]) -> List[int]:
    """将单链表转为数组便于比对断言"""
    res = []
    cur = head
    while cur:
        res.append(cur.val)
        cur = cur.next
    return res


# -------------------------------------------------------------
# 变体实现
# -------------------------------------------------------------
class KGroupReverser:

    @staticmethod
    def _reverse_single_list(head: Optional[ListNode]) -> Tuple[Optional[ListNode], Optional[ListNode]]:
        """
        翻转以 head 开头的单链表
        返回: (new_head, new_tail)
        """
        prev: Optional[ListNode] = None
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
        """
        变体 1: 经典 K 组翻转 (不足 k 个保持原样)
        时间复杂度 O(N)，空间复杂度 O(1)
        """
        if not head or k <= 1:
            return head

        dummy = ListNode(0, head)
        group_prev = dummy

        while True:
            # 1. 探针检查是否满 k 个节点
            group_end = group_prev
            for _ in range(k):
                group_end = group_end.next
                if not group_end:
                    return dummy.next

            group_start = group_prev.next
            next_group_head = group_end.next

            # 2. 隔离断开
            group_end.next = None

            # 3. 翻转局部
            new_head, new_tail = cls._reverse_single_list(group_start)

            # 4. 回接
            group_prev.next = new_head
            new_tail.next = next_group_head

            # 5. 步进
            group_prev = new_tail

    @classmethod
    def reverseKGroupAll(cls, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        变体 2: 尾部不足 k 个亦翻转
        时间复杂度 O(N)，空间复杂度 O(1)
        """
        if not head or k <= 1:
            return head

        dummy = ListNode(0, head)
        group_prev = dummy

        while group_prev.next:
            # 向前走最多 k 个
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
        """
        变体 5: 组间逆序而组内保序
        例如 k=3, 1->2->3->4->5->6 变为 4->5->6->1->2->3
        """
        if not head or k <= 1:
            return head

        # 收集每组的 (head, tail)
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
            g_tail.next = None  # 断开当前组
            groups.append((g_head, g_tail))
            cur = nxt

        if not groups:
            return None

        # 组间逆序串联
        dummy = ListNode(0)
        curr_tail = dummy
        for g_head, g_tail in reversed(groups):
            curr_tail.next = g_head
            curr_tail = g_tail

        return dummy.next
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：
  - 经典翻转与尾部翻转：每个节点至多被探针和反转各访问一次，时间复杂度严格为 $\mathcal{O}(N)$。
  - 组间逆序：切分耗时 $\mathcal{O}(N)$，逆序缝合耗时 $\mathcal{O}(N/k)$，总时间复杂度为 $\mathcal{O}(N)$。
- **空间复杂度**：
  - 原地双指针局部翻转空间复杂度为严格 $\mathcal{O}(1)$。
  - 组间逆序显式记录段表占用 $\mathcal{O}(N/k)$ 空间。
- **高频避坑清单**：
  1. **拼接后遗漏断环**：局部翻转前若未将 `group_end.next` 断开并置为 `None`，反转后尾部会和原链表形成环形死锁，导致遍历陷入死循环！
  2. **混淆翻转组内与翻转组间**：面试开头必须主动确认题目需求——是“组内翻转、组间顺序不变”（LeetCode 25 原题），还是“组内保序、组间拓扑逆序”（变体 5）。

</div>

</div>
</details>

---

## 模块二：栈与表达式计算 (Stack & Expression Parsing)

### 3. 表达式计算器与运算符优先级全景全家桶 (Basic Calculator & Operator Precedence Hierarchy)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">栈 03</span>
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

该算法派生出的**六大核心变体与考点梯队**如下：

| 变体编号 | 核心变体名称 | 核心特征 / 算符集合 | 核心算法架构与突破口 |
|---|---|---|---|
| **变体 1** | **无括号四则运算 (LC 227)** | 包含空格、多位数、`+`, `-`, `*`, `/`，无括号 | **单栈项积累法**：利用运算符后置触发机制，`+`/`-` 压入当前项，`*`/`/` 弹出栈顶立即结算，最后统一求和。 |
| **变体 2** | **带括号简单加减 (LC 224)** | 仅含 `+`, `-` 和多层嵌套括号 `( )` | **栈暂存符号与累计值**：遇到 `(` 将外层 `(res, sign)` 压栈并重置；遇到 `)` 弹出外层上下文并合并。 |
| **变体 3** | **全四则运算带括号 (LC 772)** | 结合 `+`, `-`, `*`, `/` 和深度括号嵌套 | **分治递归求解括号子问题** 或 **Dijkstra 双栈调度场算法 (Shunting-Yard)**。 |
| **变体 4** | **一元前导负号 (Unary Minus)** | 表达式存在 `-5`, `1 - (-2)`, `(-3 + 4)` 等一元负号 | **前置状态位识别**：若前一个非空格字符是 `(` 或处于开头，则当前的 `-` 判定为一元算符，在操作数栈隐式补 `0`。 |
| **变体 5** | **右结合幂运算 (Exponentiation `^`)** | 引入乘方 `^`，优先级高于乘除，且为**右结合** | $2\text{\textasciicircum}3\text{\textasciicircum}2 = 2^{(3^2)} = 512$（非 $8^2=64$）；调度场弹栈条件变为：栈顶优先级**严格大于**当前算符方可出栈。 |
| **变体 6** | **向零取整除法语义 (Division Semantics)** | 表达式计算存在负数除法截断 | **语言底层陷阱**：Python 默认向下取整 (`-3 // 2 == -2`)，而 C++/Java/LeetCode 规则要求向零截断 (`int(-3 / 2) == -1`)。 |

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心机制深度剖析</div>

#### 1. 架构方案对决：单栈状态暂存法 vs Dijkstra 双栈调度场算法 (Shunting-Yard)
- **单栈项积累法 (适用于变体 1 与变体 2)**：
  - 数学表达式本质是多项式求和 $\sum \text{term}_i$。维护数值栈 `stack` 与上一个符号 `pre_op`。
- **工业通用解法：Dijkstra 调度场双栈模型 (Shunting-Yard Algorithm)**：
  - 构建两个显式栈：**操作数栈 `nums`** 与 **运算符栈 `ops`**。
  - 优先级：`+`, `-` 为 1；`*`, `/` 为 2；`^` 为 3。
  - 左结合算符同级需出栈归约；右结合算符 `^` 栈顶严格大于才出栈。

#### 2. 一元负号 (Unary Minus) 规范化判据
- 维护标志位 `expect_operand = True`（在表达式首个字符、二元算符或 `(` 之后重置为 True）。
- 若在 `expect_operand == True` 时遇到 `+` 或 `-`，判定为一元符号，向操作数栈压入 `0` 将其转化为 $0 - x$。

#### 3. Python 整数除法向零取整的底层陷阱
- 必须使用 `int(a / b)` 替代 `a // b`，防止负数除法向负无穷偏移。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List

class ExpressionCalculator:
    """通用工业级表达式求值引擎"""
    PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    IS_RIGHT_ASSOCIATIVE = {'^': True, '+': False, '-': False, '*': False, '/': False}

    @staticmethod
    def _trunc_div(a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("Division by zero")
        return int(a / b)

    @classmethod
    def _apply_op(cls, op: str, b: int, a: int) -> int:
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return cls._trunc_div(a, b)
        if op == '^':
            if b < 0: raise ValueError("Negative exponent not supported")
            return a ** b
        raise ValueError(f"Unknown op: {op}")

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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：每个 Token 进出栈各一次，严格 $\mathcal{O}(N)$。
- **空间复杂度**：双栈深度最坏情况下为 $\mathcal{O}(N)$。
- **高频避坑**：先出栈的是右操作数 $b$，后出栈的是左操作数 $a$，减法和除法绝不能倒置。

</div>

</div>
</details>

---

## 模块三：堆与优先队列 (Heap & Priority Queue)

### 4. 数据流中位数与多路归并全景 (Find Median from Data Stream & K-Way Merge)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">堆 04</span>
  <span class="review-card-title">数据流中位数与多路归并全景 (Find Median from Data Stream & K-Way Merge)</span>
  <span class="review-card-tag">对顶双堆 · 严格平衡不变量 · 惰性删除 · 多路归并 · Top-K 桶排序</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与经典应用场景</div>

设计一个支持实时接收连续整数数据流并动态输出当前已观察序列**中位数 (Median)** 的系统容器：

```python
class MedianFinder:
    def __init__(self): ...
    def addNum(self, num: int) -> None: ...
    def findMedian(self) -> float: ...
```

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心机制深度剖析</div>

- **结构划分**：大顶堆 `lo`（存负数）存放前半部较小数据；小顶堆 `hi` 存放后半部较大数据。
- **不变式维持**：$\max(\text{lo}) \le \min(\text{hi})$，且 $\operatorname{len}(\text{lo}) \in \{\operatorname{len}(\text{hi}), \;\operatorname{len}(\text{hi}) + 1\}$。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：`addNum` 为 $\mathcal{O}(\log N)$，`findMedian` 为 $\mathcal{O}(1)$。
- **空间复杂度**：$\mathcal{O}(N)$。

</div>

</div>
</details>

---

### 5. 多商户分级加权轮转任务调度器 (Tiered Priority Task Scheduler)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">堆 05</span>
  <span class="review-card-title">多商户分级加权轮转任务调度器 (Tiered Priority Task Scheduler)</span>
  <span class="review-card-tag">商户级小顶堆 · FIFO 时间戳序列 · 活跃商户轮转队列 · VIP 加权配额调度</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与工业调度规则</div>

设计一个多租户任务调度中心，接收带有商户等级和优先级的任务流，并按加权轮转策略有序分发执行：

每个任务数据载荷包含：`task_id` (str), `seller_id` (str), `tier` (`'VIP'` 或 `'STANDARD'`), `priority` (int, 1 为最高, 3 为最低)。

实现以下两个核心接口：
```python
def receive_task(task: dict) -> None: ...
def process_next_task() -> Optional[Tuple[str, str]]: ...  # 返回 (seller_id, task_id)
```

**核心调度规则约束**：
1. **商户内部偏序**：相同商户内，`priority` 较小的任务优先处理；若优先级相同，严格按照到达先后顺序执行（**FIFO 先来先出**）。
2. **跨商户轮转加权 (Weighted Round-Robin)**：在不同商户间轮流调度。
   - `VIP` 商户每轮最多连续处理 **2** 个任务；
   - `STANDARD` 商户每轮最多连续处理 **1** 个任务。
3. **提前让渡与重新入队**：
   - 若商户的当轮配额未耗尽但其任务堆已空，则立即结束当轮，切至下一商户；
   - 当商户用尽当轮配额且内部仍有待处理任务时，将其放回活跃轮转队列末尾；若已无剩余任务，则从活跃队列移出，**严禁在活跃队列中存在重复商户条目**。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心机制深度剖析</div>

- **商户内部堆设计**：
  - 维护字典 `seller_heaps: Dict[str, List[Tuple[int, int, str]]]`，堆元素元组为 `(priority, arrival_seq, task_id)`。
  - 利用全局自增计数器 `arrival_seq`，彻底解决 Python 堆在相同优先级下的 Tie-breaking 问题，原生保持严格的 FIFO 顺序。
- **活跃商户轮转拓扑**：
  - 维护双端队列 `active_sellers = deque[str]`。
  - 配合集合 `in_active_set: Set[str]`，确保任意时刻同一 `seller_id` 在 `active_sellers` 中至多出现一次。
  - 记录当前正在服务的商户 `current_seller` 及当前轮次剩余配额 `remaining_quota`。
- **状态机流转逻辑**：
  - 当调用 `process_next_task()` 时，若无正在服务的商户，从 `active_sellers` 左侧弹出一个商户，赋予其配额（VIP 为 2，STANDARD 为 1）；
  - 从该商户的小顶堆中弹出一个最高优先级任务；配额减 1；
  - 检查该商户堆是否为空：若已空，立即结束当轮；若未空且配额耗尽，将该商户重新压回 `active_sellers` 尾部，重置 `current_seller = None`。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional, Set

class TieredTaskScheduler:
    """
    多商户分级加权轮转任务调度器
    1. receive_task: O(log M)
    2. process_next_task: O(log M)
    """
    def __init__(self):
        # 全局递增序列号，维持相同 priority 下的绝对 FIFO
        self.seq = 0
        
        # seller_id -> 小顶堆 [(priority, seq, task_id), ...]
        self.seller_heaps: Dict[str, List[Tuple[int, int, str]]] = {}
        # seller_id -> tier ('VIP' or 'STANDARD')
        self.seller_tiers: Dict[str, str] = {}
        
        # 活跃商户双端轮转队列及查重集合
        self.active_sellers: deque[str] = deque()
        self.in_active_set: Set[str] = set()
        
        # 当前调度轮次状态上下文
        self.current_seller: Optional[str] = None
        self.remaining_quota: int = 0

    def receive_task(self, task: dict) -> None:
        """接收并入队任务，task 格式: {'task_id': str, 'seller_id': str, 'tier': str, 'priority': int}"""
        task_id = task['task_id']
        seller_id = task['seller_id']
        tier = task.get('tier', 'STANDARD')
        priority = task['priority']
        
        self.seq += 1
        self.seller_tiers[seller_id] = tier
        
        if seller_id not in self.seller_heaps:
            self.seller_heaps[seller_id] = []
        
        heapq.heappush(self.seller_heaps[seller_id], (priority, self.seq, task_id))
        
        # 若商户不在活跃轮转队列且当前未在调度中，激活并加入轮转队列
        if seller_id != self.current_seller and seller_id not in self.in_active_set:
            self.active_sellers.append(seller_id)
            self.in_active_set.add(seller_id)

    def process_next_task(self) -> Optional[Tuple[str, str]]:
        """提取并执行下一个任务，返回 (seller_id, task_id)"""
        # 若当前无聚焦商户，尝试从轮转队列选拔下一个候选者
        while self.current_seller is None:
            if not self.active_sellers:
                return None  # 系统全空，无任何任务
            
            candidate = self.active_sellers.popleft()
            self.in_active_set.remove(candidate)
            
            # 若该商户确实有未完成任务
            if self.seller_heaps.get(candidate):
                self.current_seller = candidate
                tier = self.seller_tiers.get(candidate, 'STANDARD')
                self.remaining_quota = 2 if tier == 'VIP' else 1
                break

        # 从当前商户中弹出最高优先级任务
        seller = self.current_seller
        _, _, task_id = heapq.heappop(self.seller_heaps[seller])
        self.remaining_quota -= 1

        # 检查是否需要结束当轮调度
        has_more_tasks = len(self.seller_heaps[seller]) > 0
        quota_exhausted = (self.remaining_quota <= 0)

        if not has_more_tasks:
            # 堆空，提前让渡轮次
            self.current_seller = None
            self.remaining_quota = 0
        elif quota_exhausted:
            # 配额耗尽但仍有任务，重入活跃队列尾部
            self.active_sellers.append(seller)
            self.in_active_set.add(seller)
            self.current_seller = None
            self.remaining_quota = 0

        return (seller, task_id)
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：
  - `receive_task`：商户堆推入元素耗时 $\mathcal{O}(\log M)$（$M$ 为该商户积压任务数），集合与双端队列操作 $\mathcal{O}(1)$。
  - `process_next_task`：商户堆弹出耗时 $\mathcal{O}(\log M)$，商户轮转调度 $\mathcal{O}(1)$。
- **空间复杂度**：
  - 全量存储活跃任务与元数据，空间复杂度为 $\mathcal{O}(T + S)$（$T$ 为待处理任务总数，$S$ 为不同商户数量）。
- **高频避坑清单**：
  1. **队列重复入队导致的饥饿死锁**：若未维护 `in_active_set`，当商户在队列中排队期间收到新任务时，会被重复加入 `active_sellers`，导致后续配额计算失真。
  2. **空堆残留与幽灵轮转**：商户任务全被取走后必须清理或跳过，防止空商户空耗轮转配额。

</div>

</div>
</details>

---

### 6. 时间戳任务调度器与直接 ID 淘汰 (Timestamp Task Scheduler with Direct ID Removal)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">堆 06</span>
  <span class="review-card-title">时间戳任务调度器与直接 ID 淘汰 (Timestamp Task Scheduler with Direct ID Removal)</span>
  <span class="review-card-tag">复合小顶堆 · 惰性删除 (Lazy Deletion) · 哈希版本校验 · 破坏性出堆</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与接口契约</div>

实现一个按开始时间戳排序并支持实时撤销的任务调度器容器：

```python
class TaskScheduler:
    def addTask(self, taskID: str, timestamp: int) -> None: ...
    def removeTask(self, taskID: str) -> bool: ...
    def popTask(self, num: int) -> List[str]: ...
```

**操作契约规范**：
1. `addTask(taskID, timestamp)`：向调度器中登记一个任务及其触发时间戳；
2. `removeTask(taskID)`：若调度器中存在该任务，立即撤销并返回 `True`；若不存在返回 `False`；
3. `popTask(num)`：**破坏性 (Destructive)** 提取并移除时间戳最早的最多 `num` 个任务 ID（若总任务数不足 `num`，全部弹出即可）。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与惰性删除心智模型</div>

- **核心矛盾**：
  - 最小堆支持高效提取最小值（`popTask` 是堆的强项，单次弹出 $\mathcal{O}(\log N)$）；
  - 但最小堆不支持按 `taskID` 在 $\mathcal{O}(\log N)$ 内直接删除任意元素（堆内部搜寻目标需要 $\mathcal{O}(N)$）。
- **生产级突破口：哈希映射版本号 + 堆顶惰性删除 (Lazy Eviction)**：
  - 维护哈希表 `task_map: Dict[str, int]`，记录 `taskID -> valid_timestamp`。
  - `removeTask(taskID)`：直接调用 `task_map.pop(taskID, None)`，耗时严格 $\mathcal{O}(1)$！并不立刻在物理堆内部执行昂贵检索。
  - `popTask(num)`：从堆顶持续 `heappop`。每次取出 `(timestamp, seq, taskID)` 时，检验 `taskID in task_map and task_map[taskID] == timestamp`：
    - 若检验失败：说明该任务此前已被 `removeTask` 撤回或时间戳已被更新覆写，属于**陈旧脏数据 (Stale Entry)**，直接丢弃；
    - 若检验成功：该任务有效，收集入结果集，并同步从 `task_map` 中移除，直到收集满 `num` 个或堆彻底排空。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
import heapq
from typing import Dict, List

class TimestampTaskScheduler:
    """
    支持按 ID 撤回的时间戳任务调度器：
    1. addTask: O(log N)
    2. removeTask: O(1) 惰性标记
    3. popTask: 均摊 O(K log N)
    """
    def __init__(self):
        # 堆元素: (timestamp, seq, taskID)
        self.heap: List[tuple] = []
        # taskID -> current_valid_timestamp
        self.task_map: Dict[str, int] = {}
        self.seq = 0

    def addTask(self, taskID: str, timestamp: int) -> None:
        """登记新任务或更新任务时间戳"""
        self.seq += 1
        self.task_map[taskID] = timestamp
        heapq.heappush(self.heap, (timestamp, self.seq, taskID))

    def removeTask(self, taskID: str) -> bool:
        """根据 taskID 撤回任务，O(1) 惰性注销"""
        if taskID in self.task_map:
            del self.task_map[taskID]
            return True
        return False

    def popTask(self, num: int) -> List[str]:
        """破坏性提取时间戳最早的至多 num 个有效任务"""
        result: List[str] = []

        while self.heap and len(result) < num:
            timestamp, _, taskID = heapq.heappop(self.heap)
            # 校验堆顶条目是否依然合法（未被撤销且时间戳未被篡改）
            if self.task_map.get(taskID) == timestamp:
                result.append(taskID)
                del self.task_map[taskID]  # 物理出队并销毁注册
            # 若校验不通过，则该条目为已被撤销的脏数据，由循环静默吞噬

        return result
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：
  - `addTask`：堆推入 $\mathcal{O}(\log N)$，字典赋值 $\mathcal{O}(1)$。
  - `removeTask`：字典删除严格 $\mathcal{O}(1)$。
  - `popTask`：每次提取有效项均摊耗时 $\mathcal{O}(\log N)$，提取 $K$ 个任务均摊复杂度为 $\mathcal{O}(K \log N)$。
- **空间复杂度**：
  - 最坏情况下堆中包含全部历史待清理条目，空间复杂度为 $\mathcal{O}(N)$。
- **高频避坑**：
  1. **误将 `popTask` 视为只读检索**：必须在提取时同步从 `task_map` 中删除，避免后续再次被误读。

</div>

</div>
</details>
