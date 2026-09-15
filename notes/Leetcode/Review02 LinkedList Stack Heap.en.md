# Review Flashcards: Linked List, Stack & Heap

This note is the second volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Linked List & Hash-Linked Structures**, **Stack & Monotonic Stack / Deque**, and **Heap & Priority Queue** with production-grade implementations and rigorous complexity breakdowns.

---

---

---

## Module 1: Linked List & Hash-Linked Structures

### 1. LRU Cache & System-Level Extensions

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 01</span>
  <span class="review-card-title">LRU Cache & System-Level Extensions</span>
  <span class="review-card-tag">Hash Map + Doubly Linked List · Frequency Buckets · O(1) Eviction</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 146 · LRU Cache](https://leetcode.com/problems/lru-cache/) — `https://leetcode.com/problems/lru-cache/`
> - [LeetCode 460 · LFU Cache](https://leetcode.com/problems/lfu-cache/) — `https://leetcode.com/problems/lfu-cache/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **LRU Cache & System-Level Extensions (LeetCode 146 / 460)**:
> Design a data structure that follows the constraints of a Least Recently Used (LRU) cache with strict $\mathcal{O}(1)$ average time complexity for both `get` and `put`.
>
> **Reported Follow-ups**:
> 1. **Add TTL**: Each entry has an expiry time. `get` returns -1 if expired; expired entries must not occupy capacity.
> 2. **Add LFU**: A secondary eviction round where ties on usage frequency are broken by recency.
> 3. **4-End List**: Support `rpush`, `rpop`, `lpush`, `lpop` plus indexed access in $\mathcal{O}(1)$.
> 4. **Print / Iterate**: Walk cache from least-recent to most-recent.
> 5. **High Miss-Rate Tuning**: Capacity growth, ARC/2Q policies, prefetching, and consistent hash sharding.

**Interface Definition**:
```python
class LRUCache:
    def __init__(self, capacity: int): ...
    def get(self, key: int) -> int: ...
    def put(self, key: int, value: int) -> None: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

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
<div class="review-block-label">⏱️ Complexity Analysis</div>

- `get` and `put` run in strict $\mathcal{O}(1)$ time; auxiliary space $\mathcal{O}(C)$.

</div>
<div class="review-block">
<div class="review-block-label">🌐 Foundational Extensions: 5 Production-Grade Architectural Archetypes</div>

#### 1. Restaurant Waitlist / Table Matching Queue
- **System Requirements**:
  - `join(user, party_size)`: Append user to waitlist tail in $\mathcal{O}(1)$.
  - `delete(user)`: Remove user from anywhere in the queue in $\mathcal{O}(1)$ upon cancellation or timeout.
  - `find_first_match(table_size: int)`: When a table of size `table_size` becomes vacant, return the earliest customer (FIFO) whose `party_size <= table_size` without removing them.
- **Architectural Trade-offs**:
  - **Baseline LRU-Style (DLL + Hash Map)**:
    - `user_map: Dict[user, DLLNode]` provides $\mathcal{O}(1)$ `delete` and `join`;
    - `find_first_match` performs a linear walk from the head (earliest customer), taking $\mathcal{O}(N)$.
  - **Table-Size Bucketing Optimization**:
    - In reality, `party_size` is bounded by small integers ($1 \sim 10$).
    - Maintain one dedicated doubly-linked queue `buckets[s]` for each size $s \in [1, 10]$.
    - `join(user, s)`: Appends to `buckets[s]` tail in $\mathcal{O}(1)$.
    - `delete(user)`: Unlinks from the matching bucket in $\mathcal{O}(1)$ via hash map.
    - `find_first_match(table_size)`: Inspects only the heads of `buckets[1 \dots table_size]` ($s \le 	ext{table\_size}$) and selects the candidate with the earliest arrival timestamp. Time drops to strict $\mathcal{O}(	ext{table\_size}) = \mathcal{O}(1)$.

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
        # Append before tail sentinel
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

#### 2. Robot Logger Rate Limiter with Out-of-Order Timestamps
- **System Requirements**:
  - Events stream in as `(timestamp, message)`.
  - Rule: Print and return `True` if no previously printed event for the same message has timestamp in $[timestamp - 10, timestamp)$; otherwise suppress and return `False`.
  - **Out-of-Order Semantics**: Network delays may deliver `(12, "foo")` before `(10, "foo")`.
  - **Causal Invariants**:
    1. A future event must never retroactively suppress an earlier event (timestamp 12 is in the future of timestamp 10, so 10 is permitted);
    2. Identical timestamps allow only the first call;
    3. Suppressed messages must never refresh or alter the printed window!
- **Data Structure**:
  - Maintain an **ordered dynamic set (Balanced BST / Skip List / `SortedSet`)** of approved timestamps for each message.
  - Decision steps:
    1. Search greatest strict predecessor $pred = \max \{x \in S \mid x < timestamp\}$.
    2. If $timestamp \in S$ or ($pred \neq \text{None}$ and $pred \ge timestamp - 10$), return `False`.
    3. Otherwise, insert $timestamp$ into $S$ and return `True`.
  - Time complexity: $\mathcal{O}(\log K)$ per query where $K$ is the number of approved occurrences. Stale timestamps older than the maximum timestamp minus the 10-second window can be evicted via sliding LRU deque.

```python
import bisect
import collections
from typing import Dict, List

class RobotLogger:
    """
    Robot Logger with Out-of-Order Timestamps
    Time: O(log K) per check (K = approved timestamp count)
    Space: O(M * K) (M = unique messages)
    """
    def __init__(self, window_seconds: int = 10):
        self.window = window_seconds
        self.approved: Dict[str, List[int]] = collections.defaultdict(list)

    def should_print_message(self, timestamp: int, message: str) -> bool:
        ts_list = self.approved[message]
        idx = bisect.bisect_left(ts_list, timestamp)

        # Duplicate timestamp check
        if idx < len(ts_list) and ts_list[idx] == timestamp:
            return False

        # Predecessor causality check
        if idx > 0:
            pred = ts_list[idx - 1]
            if timestamp - pred < self.window:
                return False

        # Approve and maintain sorted order
        ts_list.insert(idx, timestamp)
        return True
```


#### 3. Idempotency-Key API Endpoint with Concurrency & TTL
- **Contract**:
  - First call with key $K$ and body $B$: Processes normally, stores $(K \to R)$, returns $R$;
  - Replay with same $K$ and same $B$: Returns cached $R$ without re-execution;
  - Replay with same $K$ but different body $B'$: Raises `409 Conflict`;
  - **Concurrency Guard**: Concurrent in-flight requests with identical $K$ synchronize on a per-key `threading.Condition`. The follower blocks on `IN_FLIGHT` and awakens when the leader transitions to `DONE`.
  - **TTL Eviction**: Records carry a TTL (e.g. 24h); stale entries are purged via lazy checks on access and periodic background sweeps.

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
                if now > rec.expires_at:
                    del self.store[idempotency_key]
                else:
                    if rec.body_hash != body_hash:
                        return 409, {"error": "Idempotency key re-used with different payload"}
                    
                    if rec.status == RequestStatus.IN_FLIGHT:
                        rec.condition.wait()
                        return 200, rec.response
                    else:
                        return 200, rec.response

            rec = IdempotencyRecord(body_hash, self.default_ttl)
            self.store[idempotency_key] = rec

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

#### 4. Contiguous Memory Allocator (First-Fit with Splitting & Coalescing)
- **Problem Statement**:
  - Given a single contiguous pool of $N$ memory units.
  - `allocate(size)`: Reserve a contiguous chunk $\ge size$, return handle/offset; signal OOM if no suitable block exists.
  - `free(offset)`: Return previously allocated block, **coalescing with physically adjacent free blocks** in $\mathcal{O}(1)$ to eliminate external fragmentation.
- **Doubly-Linked Boundary Tags Architecture**:
  - Each block tracks: `offset, size, is_free, prev, next`.
  - `allocate`: Walks free list using First-Fit. If `block.size > size`, splits into allocated block and remaining free block.
  - `free`: Sets `is_free = True`. If `prev.is_free`, merge with predecessor. If `next.is_free`, merge with successor. Coalescing involves purely constant-time pointer rewiring.

#### 5. Concurrency & Distributed Sharding Architecture
- **Thread-Safety Trade-off**:
  - Coarse-grained locking: Single `threading.RLock()` guarding all methods.
  - Sharded Cache (`ConcurrentHashMap` pattern): Partition keys across $M$ distinct stripes via $	ext{hash}(key) \pmod M$. Each shard possesses its own lock and DLL, achieving lock-free concurrency across keys while yielding per-shard approximate LRU.
- **Distributed Cache**:
  - Consistent hashing ring with virtual nodes balances partitions across physical nodes; each node executes the standard DLL+hashmap engine locally.

```python
import hashlib
import threading
import collections
from typing import Any, Optional

class ShardedLRUCache:
    """
    Sharded Concurrent LRU Cache with Striped Locks
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
```


#### 6. Memory Allocator with O(log m) Free Gap Indexing & Coalescing
- **Transition from O(N) Linear Scan to O(log m) Production Tier**:
  - Linear scanning over free gaps fails under heavy fragmentation.
  - **Dual-Indexing Architecture**:
    1. **Size-Keyed Balanced Tree / SortedDict**: `free_by_size: SortedDict[int, Set[BlockNode]]`. Binary search locates the leftmost block satisfying $\text{block.size} \ge size$ in $\mathcal{O}(\log m)$ time;
    2. **Address-Ordered Doubly Linked List**: All blocks (allocated and free) are chained in ascending physical address order (`start, size, is_free, prev, next`).
- **Four Coalescing Scenarios on Free**:
  When `free(address, size)` executes, validation guards against double-free via `{address: size}` active allocations map, followed by physical neighbor inspection:
  1. **No Neighbour Free**: Mark block as free and insert into `free_by_size`;
  2. **Left Neighbour Free**: Merge left (`left.size += size`), update left block entry in `free_by_size`;
  3. **Right Neighbour Free**: Merge right into current, remove right node from list and size index;
  4. **Both Neighbours Free**: Collapse three blocks into one. Left block absorbs current and right, right node is deleted, left node size updated in `free_by_size`.

```python
from typing import Dict, Optional

class MemBlock:
    def __init__(self, offset: int, size: int, is_free: bool = True):
        self.offset = offset
        self.size = size
        self.is_free = is_free
        self.prev: Optional['MemBlock'] = None
        self.next: Optional['MemBlock'] = None

class MemoryAllocator:
    """
    Memory Allocator with Boundary Tags and O(1) Coalescing
    """
    def __init__(self, total_size: int):
        self.head = MemBlock(0, total_size, is_free=True)
        self.allocated: Dict[int, MemBlock] = {} # offset -> Block

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

                # Split remaining free block
                if remainder > 0:
                    split_block = MemBlock(curr.offset + size, remainder, is_free=True)
                    split_block.next = curr.next
                    split_block.prev = curr
                    if curr.next:
                        curr.next.prev = split_block
                    curr.next = split_block
                return curr.offset
            curr = curr.next
        return -1 # Out of memory

    def free(self, offset: int) -> bool:
        if offset not in self.allocated:
            return False
        node = self.allocated.pop(offset)
        node.is_free = True

        # Coalescing branches:
        # Branch 1: Coalesce right
        if node.next and node.next.is_free:
            right = node.next
            node.size += right.size
            node.next = right.next
            if right.next:
                right.next.prev = node

        # Branch 2: Coalesce left
        if node.prev and node.prev.is_free:
            left = node.prev
            left.size += node.size
            left.next = node.next
            if node.next:
                node.next.prev = left
        return True
```


#### 7. LRU + LFU + Pluggable Eviction Strategy Pattern
- **Architectural Decoupling**: Separate backing storage from eviction logic.
- **Strategy Interface**:
  - `on_access(key)`: Hook invoked on hit to update recency, frequency buckets, or weight;
  - `pick_victim() -> key`: Nominates next key to evict when capacity is exceeded;
  - `on_evict(key)`: Cleans auxiliary tracking metadata.
- **Unified Policy Implementations**:
  - `LRUPolicy`: Single recency DLL;
  - `LFUPolicy`: `freq_buckets: Dict[int, DLL]` + `min_freq` cursor. Ties at `min_freq` are broken by LRU within the lowest frequency bucket;
  - `TTLWeightedPolicy` / `SizeWeightedPolicy`: Pluggable heap or skip-list indexed by weight.

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
    """O(1) LFU Policy with frequency buckets, tie-broken by LRU"""
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
```


#### 8. Durable In-Memory Cache with WAL & Crash Recovery
- **Deterministic Key Hashing**:
  - `generate_key(*args, **kwargs)` fails if `kwargs` is hashed directly (`TypeError: unhashable type: 'dict'`).
  - Canonical solution: Recursive transformation to immutable tuples, or `json.dumps(args) + json.dumps(kwargs, sort_keys=True)`.
- **Write-Ahead Log (WAL) & Replay Ordering**:
  - Append every mutation/access to disk: `{"op": "PUT", "key": k, "val": v, "ts": now}`.
  - **Replay Invariant**: Simple dict assignment during recovery degrades LRU ordering to FIFO. On replay, **explicitly call `move_to_end` for previously seen keys** to rebuild exact recency positions.
  - Trade-off: Synchronous `fsync` (zero data loss) vs Group Commit (high throughput); periodic checkpoint snapshots truncate stale WAL.

```python
import json, os, collections
from typing import Any, Optional

class DurableLRUCache:
    """
    Durable In-Memory LRU Cache with WAL & Replay
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
                    self.cache.move_to_end(key) # Preserve exact replay recency order
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
```


#### 9. 4-Level In-Memory Database Implementation (Anthropic CodeSignal OA)
- **Functional Requirements Across 4 Levels**:
  - **Level 1 (Core Key-Field Storage)**: Each top-level `key` maps multiple `field -> value` string pairs. Core operations: `set(key, field, value)`, `get(key, field)`, `delete(key, field) -> bool`.
  - **Level 2 (Lexicographical & Prefix Scan)**: `scan(key)` returns all valid fields formatted as `["field(value)", ...]` in strictly ascending lexicographical order of field names. `scan_by_prefix(key, prefix)` applies prefix filtering over sorted fields.
  - **Level 3 (Timestamps & Field-Level TTL Expiry)**: All operations gain `_at` timestamped variants. `set_at_with_ttl(key, field, value, timestamp, ttl)` defines a valid lifetime $[timestamp, timestamp + ttl)$. Expired records are invisibly filtered on access and lazily cleaned up.
  - **Level 4 (Snapshot Backup & Clock-Relocated Restore)**:
    - `backup(timestamp)`: Creates an isolated deep copy of all currently unexpired fields, storing each field's **Remaining TTL**: $\Delta = (ts_{set} + ttl) - timestamp$;
    - `restore(timestamp, timestamp_to_restore)`: Locates the latest backup snapshot with $ts_{backup} \le timestamp\_to\_restore$. Restores state to the live store at the new current $timestamp$. **Clock relocation**: Expired thresholds are recomputed as $new\_expiry = timestamp + \Delta$!
- **Architectural Discipline & Trap Prevention**:
  1. **Never Persist Absolute Expiry Timestamps in Backups**: Storing absolute timestamps breaks restores across time shifts; a field valid at backup time would immediately be seen as dead. Persisting relative remaining TTL $\Delta$ guarantees true clock-isolated restoration.
  2. **Deep Copy Isolation**: Backups must clone values and mappings completely; sharing dictionary references leads to silent mutation of historical snapshots by subsequent live writes.

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
    Complete 4-Level In-Memory Database Implementation
    Level 1: set, get, delete
    Level 2: scan, scan_by_prefix (lexicographical order)
    Level 3: _at operations with TTL [ts, ts + ttl)
    Level 4: backup(ts) and restore(ts, ts_to_restore) with relative TTL
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

#### 10. Todo List OOP Design & Separation of Concerns
- **Interface**: `add(entry) -> id`, `delete(id) -> bool`, `get_todo() -> List[str]`, `get_all() -> List[str]`. External predicate `check_todo(id) -> bool` returns completion status in $\mathcal{O}(1)$.
- **Architectural Discipline**:
  - `TodoList` owns sequence, ID assignment, and storage only. Completion state belongs to the external authority, preventing state drift.
  - Structure: Hash map `id_to_node` + Doubly Linked List preserving insertion order. Monotonic auto-increment counter ensures stable, non-reusable IDs.

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
    Todo List System with Separation of Concerns
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
```


#### 11. Weighted LRU Cache with Size-Bounded Eviction
- **Problem Context & Invariant Definition**:
  - Classic LRU treats each entry with unit weight $1$, bounding entry count.
  - In storage engines and GPU tensor caches, entries have **heterogeneous payload sizes (Size / Weight)**. Capacity represents total byte size / weight.
  - **Core Invariant**: $\sum_{x \in Cache} x.size \le capacity$.
- **Cascading Multi-Item Eviction**:
  - On `put(key, value, size)`, if `current_size + size > capacity`, multiple least-recently-used nodes may be sequentially evicted to accommodate the new payload.
  - **Key Update Invariant**: When updating an existing key, its previous size must be subtracted first (`current_size -= old_size`) prior to evaluating eviction, after which the new size is added and the node moved to MRU.
- **Edge Cases**:
  1. **Single item exceeds total capacity (`size > capacity`)**: Impossible to fit even if empty. If the key exists, remove it and discard insertion without breaching capacity;
  2. **Non-positive size (`size <= 0`)**: Validate and raise exception;
  3. **Exact fit (`current_size + size == capacity`)**: Loops terminate cleanly without eviction.

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
    Weighted LRU Cache with byte/size capacity limit.
    Time Complexity: get O(1), put amortized O(1)
    Space Complexity: O(N) (N = number of active keys)
    """
    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("Capacity must be non-negative")
        self.capacity = capacity
        self.current_size = 0
        self.cache: Dict[str, DLinkedNode] = {}
        # Sentinel dummy nodes
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
        
        if size > self.capacity:
            if key in self.cache:
                old = self.cache.pop(key)
                self._remove_node(old)
                self.current_size -= old.size
            return

        if key in self.cache:
            node = self.cache[key]
            self.current_size -= node.size
            node.val = value
            node.size = size
            self._move_to_head(node)
        else:
            node = DLinkedNode(key, value, size)
            self.cache[key] = node
            self._add_to_head(node)

        self.current_size += size

        # Cascade eviction
        while self.current_size > self.capacity and self.tail.prev is not self.head:
            victim = self._pop_tail()
            if victim:
                del self.cache[victim.key]
                self.current_size -= victim.size
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

### 2. Reverse Linked List In-Place via Three Pointers

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 02</span>
  <span class="review-card-title">Reverse Linked List In-Place via Three Pointers</span>
  <span class="review-card-tag">Three Pointers · In-Place Reversal · Subsegment Inversion · O(1) Space</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 206 · Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) — `https://leetcode.com/problems/reverse-linked-list/`
> - [LeetCode 92 · Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/) — `https://leetcode.com/problems/reverse-linked-list-ii/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Reverse Linked List (LeetCode 206 / 92)**:
> Given the head of a singly linked list, reverse the list in-place, and return the reversed list.
> Follow-up: implement both iterative three-pointer sliding and clean recursion, maintaining $\mathcal{O}(1)$ auxiliary space in iterative mode.

**Function Signature**:
```python
class ReverseListSolution:
    @staticmethod
    def reverseListIterative(head: Optional[ListNode]) -> Optional[ListNode]: ...
    @staticmethod
    def reverseListRecursive(head: Optional[ListNode]) -> Optional[ListNode]: ...
```

**Examples**:
- `head = [1, 2, 3, 4, 5]` $\implies$ `[5, 4, 3, 2, 1]`
- `head = [1, 2]` $\implies$ `[2, 1]`
- `head = []` $\implies$ `[]`

</div>

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
<div class="review-block">
<div class="review-block-label">🌐 Foundational Extensions: Palindrome Restoration & Forward-Order Subtraction</div>

#### 1. Palindrome Linked List with State Restoration (LC 234)
- **System Requirements**:
  - Determine if a singly linked list is a palindrome in $\mathcal{O}(1)$ auxiliary space.
  - **Engineering Contract (Production Follow-up)**: The function must **restore the list to its original physical structure before returning**, ensuring no mutating side-effects are observable by upstream callers.
- **Algorithm Execution**:
  1. Fast-slow pointers locate the midpoint ($slow$ arrives at mid).
  2. Reverse the second half in-place: `second_half = reverseList(slow.next)`.
  3. Compare forward and reversed halves value-by-value.
  4. **State Restoration**: Reverse the second half once more and splice back to `slow.next = reverseList(second_half)`.

```python
class PalindromeSolution:
    @staticmethod
    def isPalindrome(head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True

        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        def reverse_chain(node: Optional[ListNode]) -> Optional[ListNode]:
            prev, curr = None, node
            while curr:
                nxt = curr.next
                curr.next = prev
                prev = curr
                curr = nxt
            return prev

        second_head = reverse_chain(slow.next)

        p1, p2 = head, second_head
        is_pal = True
        while is_pal and p2:
            if p1.val != p2.val:
                is_pal = False
            p1 = p1.next
            p2 = p2.next

        # Crucial Engineering Restoration: restore original chain
        slow.next = reverse_chain(second_head)

        return is_pal
```

#### 2. Forward-Order Linked List Subtraction ($l_1 - l_2$)
- **Problem Requirements**:
  - Given two non-empty singly linked lists $l_1, l_2$ representing numbers in forward digit order (MSB First), $l_1 \ge l_2$.
  - Compute $l_1 - l_2$ in the same forward order without converting to built-in arbitrary-precision integers (length up to $10^5$).
  - Strip redundant leading zeros.
- **Algorithm**:
  - Reverse $l_1, l_2$ to align lower-order digits in $\mathcal{O}(1)$ space.
  - Perform elementary subtraction with borrow propagation: $	ext{diff} = v_1 - v_2 - borrow$.
  - Reverse result back to MSB order and trim leading zeros.

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

        reverse(r1)
        reverse(r2)

        res = reverse(dummy.next)
        while res and res.val == 0 and res.next:
            res = res.next

        return res
```

</div>
#### 3. Add Two Numbers in Forward Order (MSB First, LC 445)
- **Constraint**: Sum two forward-order linked lists $l_1, l_2$ **without reversing the input lists**.
- **Two-Stack Archetype**:
  - Push node values of $l_1$ and $l_2$ onto `stack1` and `stack2`, aligning least significant digits at the top;
  - Pop to compute digit sums with carry: $\text{total} = v_1 + v_2 + carry$;
  - **Head Insertion**: Construct output list by prepending new nodes: `new_node.next = head; head = new_node`, naturally yielding forward order without reversal.

#### 4. N-ary Tree Sum + Leaf Next Pointer (Citadel Phone Screen)
- **Three-Stage Ladder**:
  1. Tree sum via recursive DFS;
  2. Connect all leaf nodes in DFS order: maintain a rolling `prev_leaf` pointer; when `not node.children` is reached, wire `prev_leaf.next = curr; prev_leaf = curr`;
  3. $\mathcal{O}(1)$ Extra Space: Reuse pointer fields for traversal threading, eliminating recursive call stacks.

</div>
</details>

---

### 3. Merge Two Sorted Lists & Add Two Numbers

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 03</span>
  <span class="review-card-title">Merge Two Sorted Lists & Add Two Numbers</span>
  <span class="review-card-tag">Sentinel Dummy Head · Two-Pointer Merge · Low-to-High Carry Propagation · Auxiliary Space O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 21 · Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) — `https://leetcode.com/problems/merge-two-sorted-lists/`
> - [LeetCode 2 · Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) — `https://leetcode.com/problems/add-two-numbers/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Problem Statements**:
1. **Merge Two Sorted Lists (LC 21)**: Merge two sorted singly linked lists into one sorted list by splicing together the nodes of the first two lists.
2. **Add Two Numbers (LC 2)**: You are given two non-empty linked lists representing two non-negative integers. The digits are stored in **reverse order** (units digit at head). Add the two numbers and return the sum as a linked list.

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class MergeAndAddSolution:
    @staticmethod
    def mergeTwoLists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
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
        curr.next = l1 if l1 else l2
        return dummy.next

    @staticmethod
    def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
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
<div class="review-block-label">💡 Mechanism & Invariant Analysis</div>

- **Sentinel Head Uniformity**:
  A dummy head eliminates conditional checks for head allocation, allowing `curr.next = ...` to apply universally.
- **O(1) Remainder Splicing**:
  Unlike array merging, unvisited remainder nodes are linked in a single constant-time assignment `curr.next = l1 if l1 else l2`.
- **Fused Carry Loop**:
  The condition `while p1 or p2 or carry` seamlessly accommodates trailing carries exceeding input lengths without post-loop branches.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N + M)$ for merge; $\mathcal{O}(\max(N, M))$ for addition.
- **Space Complexity**: $\mathcal{O}(1)$ in-place for merge; $\mathcal{O}(1)$ auxiliary for addition.

</div>

</div>
</details>

---

### 4. Linked List Cycle & Find Duplicate Number

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 04</span>
  <span class="review-card-title">Linked List Cycle & Find Duplicate Number</span>
  <span class="review-card-tag">Floyd Tortoise and Hare · Cycle Entry Proof · Array-as-Graph Reduction · Space O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 141 · Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) — `https://leetcode.com/problems/linked-list-cycle/`
> - [LeetCode 142 · Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/) — `https://leetcode.com/problems/linked-list-cycle-ii/`
> - [LeetCode 287 · Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) — `https://leetcode.com/problems/find-the-duplicate-number/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Problem Statements**:
1. **Linked List Cycle I & II (LC 141 / 142)**: Detect if a cycle exists. If so, locate the node where the cycle begins in $\mathcal{O}(1)$ memory.
2. **Find the Duplicate Number (LC 287)**: Given an array of $n + 1$ integers in $[1, n]$, find the repeated number **without modifying the array** and using only $\mathcal{O}(1)$ extra space.

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import Optional, List

class CycleAndDuplicateSolution:
    @staticmethod
    def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return None
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None

        ptr1 = head
        ptr2 = slow
        while ptr1 != ptr2:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        return ptr1

    @staticmethod
    def findDuplicate(nums: List[int]) -> int:
        slow = nums[0]
        fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
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
<div class="review-block-label">💡 Mechanism & Mathematical Derivation</div>

- **Mathematical Derivation of Cycle Entry**:
  - Let $a$ be distance from head to cycle entrance; $b$ be distance from entrance to first collision; $C$ be cycle circumference.
  - At first collision:
    $$\text{slow distance} = a + b$$
    $$\text{fast distance} = a + b + k \cdot C \quad (k \ge 1)$$
  - Since $v_{\text{fast}} = 2 \cdot v_{\text{slow}}$:
    $$2(a + b) = a + b + k \cdot C \implies a + b = k \cdot C \implies a = (k - 1)C + (C - b)$$
  - Walking one pointer from `head` and another from `collision` at identical speed guarantees they meet at the cycle entry after distance $a$.
- **Array-to-Graph Mapping (LC 287 Invariant)**:
  - Directed edge: $i \to nums[i]$.
  - Because $nums[i] \in [1, n]$, index $0$ has in-degree $0$, serving as an authoritative head node.
  - A duplicate number means multiple indices point to the same value $\implies$ in-degree $\ge 2$, establishing a cycle whose entry point is the duplicate value.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$, meeting within $C$ steps and finding entry in $a$ steps.
- **Space Complexity**: $\mathcal{O}(1)$, auxiliary pointers only.

</div>

</div>
</details>

---

### 5. Reorder List (Midpoint Split, In-Place Reversal & Interleave)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 05</span>
  <span class="review-card-title">Reorder List (Midpoint Split, In-Place Reversal & Interleave)</span>
  <span class="review-card-tag">Fast-Slow Midpoint · In-Place Reversal · Alternating Interleave · Space O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 143 · Reorder List](https://leetcode.com/problems/reorder-list/) — `https://leetcode.com/problems/reorder-list/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Problem Statement**:
> Given the head of a singly linked-list: $L_0 \to L_1 \to \dots \to L_{n-1} \to L_n$.
> Reorder the list in-place to: $L_0 \to L_n \to L_1 \to L_{n-1} \to L_2 \to L_{n-2} \to \dots$
> Solve in $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ auxiliary space without altering node values.

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import Optional

class ReorderListSolution:
    @staticmethod
    def reorderList(head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return

        # 1. Locate midpoint and bisect
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        second_head = slow.next
        slow.next = None

        # 2. Reverse second half
        prev = None
        curr = second_head
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        p2 = prev

        # 3. Interleave halves
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

        ListNode* slow = head;
        ListNode* fast = head;
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        ListNode* second = slow->next;
        slow->next = nullptr;

        ListNode* prev = nullptr;
        ListNode* curr = second;
        while (curr) {
            ListNode* nxt = curr->next;
            curr->next = prev;
            prev = curr;
            curr = nxt;
        }

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
<div class="review-block-label">💡 Mechanism & Invariant Analysis</div>

- **Three-Step Composition**:
  Combines three classical primitive operations:
  1. Fast-slow bisection with `fast.next and fast.next.next` guarantees $len(p_1) \ge len(p_2)$;
  2. In-place reversal of the trailing sublist;
  3. Alternating zip-interleaving terminating naturally on `while p2`.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$, three linear half-passes.
- **Space Complexity**: $\mathcal{O}(1)$, auxiliary pointers only.

</div>

</div>
</details>

---

### 6. Remove Nth Node From End of List

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 06</span>
  <span class="review-card-title">Remove Nth Node From End of List</span>
  <span class="review-card-tag">Two-Pointer Fixed Offset · Sentinel Dummy Head · One-Pass Traversal · Space O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 19 · Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) — `https://leetcode.com/problems/remove-nth-node-from-end-of-list/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Problem Statement**:
> Given the head of a singly linked list, remove the $n$-th node from the end of the list and return its head.
> **Constraint**: Solve it in a single pass with $\mathcal{O}(1)$ auxiliary space.

**Function Signature**:
```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]: ...
```

**Examples**:
- `head = [1, 2, 3, 4, 5], n = 2` $\implies$ `[1, 2, 3, 5]`
- `head = [1], n = 1` $\implies$ `[]`
- `head = [1, 2], n = 1` $\implies$ `[1]`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class RemoveNthFromEndSolution:
    @staticmethod
    def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # One-pass removal of the n-th node from the end of a singly-linked list.
        # Uses a sentinel dummy head to eliminate edge-case branches for head deletion.
        dummy = ListNode(0, head)
        fast = dummy
        slow = dummy

        # 1. Advance fast pointer by n + 1 steps to create a fixed offset window
        for _ in range(n + 1):
            fast = fast.next

        # 2. Walk fast and slow synchronously until fast reaches None
        while fast is not None:
            fast = fast.next
            slow = slow.next

        # 3. slow is guaranteed to sit at the predecessor of the target node
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

        // 1. Advance fast by n + 1 steps
        for (int i = 0; i <= n; ++i) {
            fast = fast->next;
        }

        // 2. Synchronous advance
        while (fast != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }

        // 3. Unlink target node
        ListNode* target = slow->next;
        slow->next = slow->next->next;
        delete target; // Avoid heap memory leak

        return dummy.next;
    }
};
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariant Proof</div>

- **Fixed-Offset Sliding Invariant**:
  - Let $L$ denote the total length of the linked list. The dummy node is at index $0$, list nodes are at $1 \dots L$, and the terminal null is at index $L + 1$.
  - The $n$-th node from the end has forward index $L - n + 1$.
  - Its predecessor node has forward index $(L - n + 1) - 1 = L - n$.
  - Initially, `fast` steps $n + 1$ times starting from index $0$ (`dummy`), landing on index $n + 1$. The invariant distance between `fast` and `slow` is strictly $n + 1$.
  - When `fast` traverses past the tail to $L + 1$ (`fast is None`), `slow` resides at index $(L + 1) - (n + 1) = L - n$.
  - Thus, **`slow` is mathematically guaranteed to halt at the predecessor of the target node**.
- **Sentinel Head Design Utility**:
  - If removing the original head node ($n = L$), a standard traversal requires an explicit check `if n == length: return head.next`.
  - With a sentinel `dummy` node, the head is treated identically to an internal node, unifying the unlinking operation into `slow.next = slow.next.next`.
- **Interview Communication Standard**:
  - Verbalize the invariant clearly: *"I advance the fast pointer by $n+1$ steps from a dummy node so that when fast hits null, slow is guaranteed to sit exactly at the node immediately preceding the target deletion node."*

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Edge Cases</div>

- **Time Complexity**: $\mathcal{O}(L)$, strict one-pass traversal where `fast` visits $L + 1$ nodes.
- **Space Complexity**: $\mathcal{O}(1)$, auxiliary pointers only.
- **Edge Cases**:
  1. $L = 1, n = 1$: Single-node list becomes empty.
  2. $n = L$: Removing original list head.
  3. $n = 1$: Removing list tail.

</div>

</div>
</details>

---

### 7. Copy List with Random Pointer (In-Place Interleaving)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 07</span>
  <span class="review-card-title">Copy List with Random Pointer (In-Place Interleaving)</span>
  <span class="review-card-tag">In-Place Interleave · Random Pointer Projection · List Decoupling · Auxiliary Space O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 138 · Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/) — `https://leetcode.com/problems/copy-list-with-random-pointer/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Problem Statement**:
> Construct a **deep copy** of a linked list where each node contains an additional `random` pointer pointing to any node in the list or `null`.
> Solve with $\mathcal{O}(1)$ auxiliary space without hash maps.

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

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
        if not head:
            return None

        # Pass 1: Clone inline: A -> A' -> B -> B'
        curr = head
        while curr:
            copy = Node(curr.val, curr.next)
            curr.next = copy
            curr = copy.next

        # Pass 2: Link random pointers
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next

        # Pass 3: Decouple lists
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

        Node* curr = head;
        while (curr) {
            Node* copy = new Node(curr->val);
            copy->next = curr->next;
            curr->next = copy;
            curr = copy->next;
        }

        curr = head;
        while (curr) {
            if (curr->random) {
                curr->next->random = curr->random->next;
            }
            curr = curr->next->next;
        }

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
<div class="review-block-label">💡 Mechanism & Invariant Analysis</div>

- **In-Place Interleaving Invariant**:
  Embedding cloned nodes directly into the original chain creates an implicit physical mapping $\text{copy}(curr) \equiv curr.next$. Thus, random pointers are wired in $\mathcal{O}(1)$ time without extra memory:
  $$\text{copy}(curr.random) \equiv curr.random.next$$
- **Clean Decoupling**:
  Pass 3 faithfully restores the original chain while unlinking the deep clone.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$, three linear sweeps.
- **Space Complexity**: $\mathcal{O}(1)$ auxiliary space.

</div>

</div>
</details>

---

### 8. Reverse Nodes in k-Group & Structural Group Inversion

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 08</span>
  <span class="review-card-title">Reverse Nodes in k-Group & Structural Group Inversion</span>
  <span class="review-card-tag">Dummy Head · Lookahead K-Check · Sub-List Inversion · O(1) Space</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 25 · Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) — `https://leetcode.com/problems/reverse-nodes-in-k-group/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Reverse Nodes in k-Group (LeetCode 25)**:
> Given the head of a linked list, reverse the nodes of the list $k$ at a time, and return the modified list.
> $k$ is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of $k$ then left-out nodes, in the end, should remain as it is.
>
> **Industrial Follow-ups**:
> 1. What if remaining nodes (< $k$) at the tail should also be reversed?
> 2. What if groups of $k$ themselves are reversed in order while preserving relative order within each group?

**Function Signature**:
```python
class KGroupReverser:
    @staticmethod
    def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]: ...
```

**Examples**:
- `head = [1, 2, 3, 4, 5], k = 2` $\implies$ `[2, 1, 4, 3, 5]`
- `head = [1, 2, 3, 4, 5], k = 3` $\implies$ `[3, 2, 1, 4, 5]`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

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
<div class="review-block-label">🌐 Foundational Extensions: Thread-Safe Task Queue Transformation</div>

#### Concurrency Architectural Trade-off Matrix
Transforming a shared linked list via `reverseKGroup(head, k)` under concurrent reads and tail-appends requires rigorous synchronization:
1. **Global Mutex**:
   - Single lock protects entire queue during transformation.
   - Simplest correctness, but serializes all readers and destroys throughput.
2. **Reader-Writer Lock (shared_mutex)**:
   - Concurrent inspection acquires shared read lock; batch transformation takes exclusive write lock.
   - Eliminates read contention, but prolonged transformation starves waiting readers.
3. **Segmented Locking**:
   - Independent lock per $k$-sized segment. Threads transform disjoint subsegments concurrently.
   - Deadlock Prevention: Splice boundaries require holding locks on both adjacent segments. Locks must be acquired strictly in ascending memory address or logical sequence order.
4. **Copy-On-Write (COW) with Atomic Pointer Swap**:
   - Worker allocates and transforms a detached copy of the segment, then atomically publishes the new sub-head via CAS / `atomic_store`.
   - Readers remain completely lock-free; ideal for read-heavy task dispatching.

</div>

</div>
</details>

---

### 9. Insert into a Sorted Circular Linked List

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 09</span>
  <span class="review-card-title">Insert into a Sorted Circular Linked List</span>
  <span class="review-card-tag">Two-Pointer Cyclic Walk · Inflection Point Detection · Wrap-Around Boundary · Space O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **Related Links**:
> - [LeetCode 708 · Insert into a Sorted Circular Linked List](https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/) — `https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/`
> - [1point3acres Interview Problem](https://www.1point3acres.com/interview/problems/e1081044-6f41-5139-8596-3e843a348997) — Meta Phone Screen Classic

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Problem Statement**:
> Given a Circular Linked List node, which is sorted in non-descending order, write a function to insert a value `insertVal` into the list such that it remains a sorted circular list.
> The given node can be a reference to **any single node** in the list and may not necessarily be the smallest value in the circular list.
> If the list is empty (i.e., the given node is `null`), create a new single circular list and return the reference to that single node. Otherwise, return the original given node.

**Function Signature**:
```python
class Solution:
    def insert(self, head: Optional[Node], insertVal: int) -> Node: ...
```

**Examples**:
- `head = [3, 4, 1], insertVal = 2` $\implies$ `[3, 4, 1, 2]`
- `head = [], insertVal = 1` $\implies$ `[1]` (self-referencing circular node)
- `head = [1], insertVal = 0` $\implies$ `[1, 0]`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import Optional

class Node:
    def __init__(self, val: int = 0, next: Optional['Node'] = None):
        self.val = val
        self.next = next

class InsertSortedCircularListSolution:
    @staticmethod
    def insert(head: Optional[Node], insertVal: int) -> Node:
        # Insert insertVal into a sorted circular singly-linked list while preserving order.
        # head can point to any arbitrary node in the cycle.
        # Case 1: Empty list -> return single self-loop node
        if not head:
            new_node = Node(insertVal)
            new_node.next = new_node
            return new_node

        prev = head
        curr = head.next

        while True:
            # Case 2: Interior sorted segment (prev.val <= insertVal <= curr.val)
            if prev.val <= insertVal <= curr.val:
                break

            # Case 3: Reached inflection point where values wrap from max to min
            if prev.val > curr.val:
                # Value is greater than the global max OR smaller than the global min
                if insertVal >= prev.val or insertVal <= curr.val:
                    break

            prev = curr
            curr = curr.next

            # Case 4: Full cycle completed back to start (e.g., all values identical)
            if prev == head:
                break

        # Splice new node between prev and curr
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
            // Case 2: Standard interior insertion
            if (prev->val <= insertVal && insertVal <= curr->val) {
                break;
            }

            // Case 3: Inflection point wrap-around
            if (prev->val > curr->val) {
                if (insertVal >= prev->val || insertVal <= curr->val) {
                    break;
                }
            }

            prev = curr;
            curr = curr->next;

            // Case 4: Complete loop without triggering earlier branches
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
<div class="review-block-label">💡 Mechanism & 4-Way Branch Analysis</div>

- **Mathematical Topology of Sorted Cycle**:
  Because `head` may point anywhere, a sorted circular list has exactly one **inflection drop-off point** where the global maximum wraps back around to the global minimum (unless all values are identical).
  The algorithm exhaustively handles four mutually exclusive scenarios:
  1. **Empty List**:
     Create node with self-loop (`node.next = node`) and return.
  2. **Interior Ordered Segment**:
     If `prev.val <= insertVal <= curr.val`, `insertVal` belongs naturally between `prev` and `curr`.
  3. **Inflection Wrap-Around**:
     When `prev.val > curr.val`, `prev` is the global maximum and `curr` is the global minimum.
     - If `insertVal >= prev.val` (new maximum), it must be placed right after `prev`.
     - If `insertVal <= curr.val` (new minimum), it must also be placed right after `prev` (before `curr`).
  4. **Degenerate Uniform Cycle**:
     If all nodes have identical values (e.g., `[3, 3, 3]`), or traversal laps the entire list (`prev == head`) without finding an inflection, inserting at `prev` preserves correctness.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Edge Cases</div>

- **Time Complexity**: $\mathcal{O}(N)$, at most one complete pass around the circular list ($N$ nodes).
- **Space Complexity**: $\mathcal{O}(1)$, pointers only.
- **Edge Cases**:
  - `head = None` (empty list).
  - Single-node cycle (`head.next == head`).
  - Two nodes sorted `[1, 3]` inserting `2`, `0`, or `4`.
  - All elements equal `[3, 3, 3]` inserting `1` or `5`.

</div>

</div>
</details>

---

### 10. Flatten Multilevel Doubly Linked List with Empty-Node Filtering

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">LINKED LIST 10</span>
  <span class="review-card-title">Flatten Multilevel Doubly Linked List with Empty-Node Filtering</span>
  <span class="review-card-tag">Pointer Relinking · DFS Traversal · Tail Backtracking · In-Place Flattening</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 430 · Flatten a Multilevel Doubly Linked List](https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/) — `https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Flatten a Multilevel Doubly Linked List (LeetCode 430)**:
> You are given a doubly linked list, which contains nodes that have a next pointer, a previous pointer, and an additional child pointer. This child pointer may or may not point to a separate doubly linked list, also containing these special nodes.
> Flatten the list so that all the nodes appear in a single-level, doubly linked list. Nodes in the child list should appear between the parent node and the parent node's next node.
>
> **Industrial Extension**: Filter out empty or null nodes (`val is None`) during flattening and maintain bidirectional pointer integrity.

**Class & Method Signature**:
```python
class MultiLevelNode:
    def __init__(self, val=None, prev=None, next=None, child=None): ...

class MultiLevelListFlattenSolution:
    @classmethod
    def flattenAndFilterEmpty(cls, head: Optional[MultiLevelNode]) -> Optional[MultiLevelNode]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

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
<div class="review-block-label">🌐 Foundational Extensions: Recursive DFS vs Explicit Stack Iteration</div>

- **Recursive DFS with Subtree Tail Return**:
  - Define `dfs(node) -> tail` returning the terminal node of the flattened subsegment.
  - When encountering `curr.child`:
    1. Preserve `nxt = curr.next`;
    2. Recurse to retrieve `child_tail = dfs(curr.child)`;
    3. Splice forward: `curr.next = curr.child; curr.child.prev = curr`;
    4. Splice backward: If `nxt`, connect `child_tail.next = nxt; nxt.prev = child_tail`;
    5. Clear `curr.child = None`;
    6. Continue traversal from `nxt` (or `child_tail` if `nxt` was null).
- **Explicit Stack Iteration (Stack-Overflow Free)**:
  - Avoids runtime recursion depth limits via an explicit stack:
  - Traverse list; when `curr.child` is detected:
    - Push `curr.next` (if non-null) onto stack;
    - Redirect `curr.next = curr.child; curr.child.prev = curr`;
    - Nullify `curr.child = None`;
  - When reaching the end of a segment (`curr.next is None`) and the stack is non-empty, pop and splice the pending node to `curr.next`, updating its `prev`. Auxiliary space is bounded by $\mathcal{O}(\text{depth})$.

</div>

</div>
</details>

---

### 11. Randomized Container with O(1) Insert & PopRandom

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">CONTAINER 11</span>
  <span class="review-card-title">Randomized Container with O(1) Insert & PopRandom</span>
  <span class="review-card-tag">Contiguous Dynamic Array + Hash Index Map · Swap-with-Last Deletion · Uniform Random Sampling · O(1) Amortized</span>
</summary>
<div class="review-card-content">

> 🔗 **Related Links**:
> - [LeetCode 380 · Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) — `https://leetcode.com/problems/insert-delete-getrandom-o1/`
> - [LeetCode 381 · Insert Delete GetRandom O(1) - Duplicates allowed](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/) — `https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/`
> - [1point3acres Interview Problem](https://www.1point3acres.com/interview/problems/company/meta/randomized-container) — Meta Production Container Design

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Problem Statement**:
> Design a data structure that supports all of the following operations in average $\mathcal{O}(1)$ time complexity:
> 1. `insert(val)`: Inserts an item `val` to the set if not already present. Returns `true` if inserted, `false` otherwise.
> 2. `remove(val)`: Removes an item `val` from the set if present. Returns `true` if removed, `false` otherwise.
> 3. `getRandom()`: Returns a random element from the current set of elements with uniform probability.
> 4. `popRandom()` (Meta Extension): Removes and returns a random element from the container in $\mathcal{O}(1)$ time with uniform probability.

**Core Trade-off & Architectural Dilemma**:
- Standard Hash Sets support $\mathcal{O}(1)$ insertion and deletion, but their memory buckets are sparse, making uniform random selection in $\mathcal{O}(1)$ impossible.
- Contiguous dynamic arrays support $\mathcal{O}(1)$ random indexing via contiguous memory, but arbitrary index deletion incurs $\mathcal{O}(N)$ element shifting.
- **Architectural Solution**: Fuse a contiguous dynamic array with a hash index table via a **swap-with-last** deletion invariant.

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
import random
from typing import Dict, List

class RandomizedSet:
    # Standard unique-element variant (LeetCode 380 + Meta popRandom extension).
    # All operations execute in amortized O(1) time.
    def __init__(self):
        self.vals: List[int] = []               # Contiguous dynamic array for O(1) random lookup
        self.val_to_index: Dict[int, int] = {}       # Hash index map: value -> index in self.vals

    def insert(self, val: int) -> bool:
        if val in self.val_to_index:
            return False
        self.val_to_index[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val: int) -> bool:
        # Swap-with-Last Deletion:
        # Overwrites target index with the array tail, then pops tail in O(1).
        if val not in self.val_to_index:
            return False

        idx_to_remove = self.val_to_index[val]
        last_val = self.vals[-1]

        # Overwrite target slot with last element
        self.vals[idx_to_remove] = last_val
        self.val_to_index[last_val] = idx_to_remove

        # Pop trailing element and delete target mapping
        self.vals.pop()
        del self.val_to_index[val]
        return True

    def getRandom(self) -> int:
        # Sample an element uniformly at random in O(1).
        return random.choice(self.vals)

    def popRandom(self) -> int:
        # Meta Phone-Screen Extension:
        # Uniformly sample, delete, and return an element in O(1).
        if not self.vals:
            raise IndexError("popRandom from empty RandomizedSet")

        rand_idx = random.randrange(len(self.vals))
        val_to_pop = self.vals[rand_idx]
        last_val = self.vals[-1]

        # Overwrite rand_idx slot with last element
        self.vals[rand_idx] = last_val
        self.val_to_index[last_val] = rand_idx

        # Pop tail and purge index entry
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

        // Overwrite target slot
        vals[idxToRemove] = lastVal;
        valToIndex[lastVal] = idxToRemove;

        // Pop tail
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
<div class="review-block-label">💡 Mechanism & Follow-up: Duplicates Allowed (LC 381)</div>

- **Swap-with-Last Invariant**:
  - Removing an element from an arbitrary index $i$ in an array typically requires $\mathcal{O}(N)$ memory shift.
  - Because a set has no order constraint, **element ordering within the backing array carries no semantic weight**.
  - Replacing the target element at slot $i$ with `vals[-1]` followed by a tail pop converts an $\mathcal{O}(N)$ operation into an $\mathcal{O}(1)$ operation.
- **Follow-up: Handling Duplicates (LeetCode 381)**:
  - When duplicates are allowed, the probability of returning any value must be proportional to its frequency (each instance has probability $1/N$).
  - **Schema Evolution**: Replace integer index with a hash set of indices:
    $$	ext{val\_to\_indices}: 	ext{Dict}[val, 	ext{Set}[int]]$$
  - **Deletion Invariant with Duplicates**:
    1. Extract any index `idx = next(iter(val_to_indices[val]))`.
    2. Overwrite `vals[idx]` with `last_val = vals[-1]`.
    3. Update `val_to_indices[last_val]`: discard `len(vals) - 1` and add `idx`.
    4. Discard `idx` from `val_to_indices[val]`. If empty, delete key.
    5. Execute `vals.pop()`.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**:
  - `insert`: Amortized $\mathcal{O}(1)$.
  - `remove`: $\mathcal{O}(1)$ (hash map lookup, array slot overwrite, tail pop).
  - `getRandom`: $\mathcal{O}(1)$ (PRNG integer generation + direct array indexing).
  - `popRandom`: $\mathcal{O}(1)$.
- **Space Complexity**: $\mathcal{O}(N)$, linear in the total number of stored elements.

</div>

</div>
</details>

---

### 12. Independent Iterator Protocol & Shared Streaming Buffer (Python itertools.tee)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STREAM 12</span>
  <span class="review-card-title">Independent Iterator Protocol & Shared Streaming Buffer (Python itertools.tee)</span>
  <span class="review-card-tag">Iterator Protocol · Shared Singly-Linked Buffer · Multi-Cursor Chasing · O(1) Auto GC · Memory O(g + n)</span>
</summary>
<div class="review-card-content">

> 🔗 **Design Standards & Prototype**:
> - CPython standard library `itertools.tee(iterable, n=2)` core implementation
> - Industrial multi-consumer stream buffering with decoupled backpressure

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Architectural Evolution</div>

**Problem Statement**:
> Implement the behavior of Python's `itertools.tee`:
> Accept one iterator and an integer $n \ge 1$, then return $n$ independent iterators over the same source sequence.
> Each returned iterator must be able to advance independently while preserving source order.
> Transition from a naive baseline to an optimal shared-state design.

**Architecture Comparison & Memory Bottleneck**:
- **Naive Multi-Queue Baseline**:
  - Allocate an independent `collections.deque` for each of the $n$ iterators.
  - When consumer $i$ calls `next()` and its queue is empty, pull one source value and **append a duplicate copy to all $n$ queues**.
  - **Flaw**: If iterators trail across a gap of $g$ values, those queues retain $n$ distinct copies of every buffered value, consuming $\mathcal{O}(n \cdot g)$ memory!
- **Optimal Shared-State Linked Buffer**:
  - Maintain a **single shared singly-linked chain** of values: `Node(val, next)`.
  - Each consumer holds an independent forward `cursor` pointing to a node in the chain.
  - When a consumer advances past the currently materialized chain (`cursor.next is None`), it pulls from the source and appends a single new node.
  - **Garbage Collection**: Nodes possess forward-only links. Once the slowest consumer advances past a node, all references to that node drop to zero, prompting immediate $\mathcal{O}(1)$ garbage collection by Python's reference counting runtime!
  - **Auxiliary Space**: Strictly optimized from $\mathcal{O}(n \cdot g)$ down to $\mathcal{O}(g + n)$ where $g$ is the distance between fastest and slowest consumer.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementation</div>

```python
from typing import Any, Iterator, List, Optional

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
        # If cursor's next is not yet materialized, fetch from source
        if self.cursor.next is None:
            val = next(self.source)  # Raises StopIteration when exhausted
            self.cursor.next = _TeeNode(val)

        # Advance cursor forward and return stored value
        self.cursor = self.cursor.next
        return self.cursor.val

def custom_tee(iterable: Any, n: int = 2) -> List[Iterator[Any]]:
    # Production-grade Python itertools.tee implementation:
    # Shared singly-linked chain with independent cursors.
    # Auxiliary space is strictly bounded by O(g + n).
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []

    shared_source = iter(iterable)
    root = _TeeNode()  # Sentinel head
    return [_TeeIterator(root, shared_source) for _ in range(n)]

if __name__ == "__main__":
    src = iter([10, 20, 30, 40, 50])
    it1, it2, it3 = custom_tee(src, 3)

    assert next(it1) == 10
    assert next(it1) == 20
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
<div class="review-block-label">⏱️ Complexity & Communication Checklist</div>

- **Time Complexity**: Each item from the underlying source is evaluated exactly once; each `next()` invocation executes in strict $\mathcal{O}(1)$ time.
- **Space Complexity**: $\mathcal{O}(g + n)$ where $g$ is the distance between the fastest and slowest cursor.
- **Interview Communication Standard**:
  *"A naive queue-per-consumer model duplicates elements $n$ times during lagging consumption, incurring $\mathcal{O}(n \cdot g)$ memory. By using a shared singly-linked stream with forward-only node pointers, each iterator operates as an independent cursor. As soon as the slowest cursor advances, Python's native reference count for earlier nodes reaches zero, reclaiming memory in $\mathcal{O}(1)$ and yielding an optimal $\mathcal{O}(g + n)$ footprint."*

</div>

</div>
</details>

---

## Module 2: Stack & Monotonic Stack / Deque

### 13. Basic Calculator & Operator Precedence Hierarchy

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STACK 13</span>
  <span class="review-card-title">Basic Calculator & Operator Precedence Hierarchy</span>
  <span class="review-card-tag">Operator Precedence · Recursive Descent · Stack Evaluation · Parentheses Scoping</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 224 · Basic Calculator](https://leetcode.com/problems/basic-calculator/) — `https://leetcode.com/problems/basic-calculator/`
> - [LeetCode 227 · Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/) — `https://leetcode.com/problems/basic-calculator-ii/`
> - [LeetCode 772 · Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/) — `https://leetcode.com/problems/basic-calculator-iii/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Basic Calculator Hierarchy (LeetCode 224, 227, 772)**:
> Given a string expression representing an arithmetic expression containing non-negative integers, operators `+`, `-`, `*`, `/`, and parentheses `(`, `)`, evaluate the expression and return its integer value.
> Division must truncate toward zero (e.g. $-3 // 2 = -1$).
> Must handle nested parentheses, arbitrary whitespace, and multi-digit integers in strict $\mathcal{O}(n)$ time.

**Function Signature**:
```python
class ExpressionCalculator:
    @classmethod
    def calculate(cls, s: str) -> int: ...
```

**Examples**:
- `s = "3+2*2"` $\implies$ `7`
- `s = " 3/2 "` $\implies$ `1`
- `s = " 3+5 / 2 "` $\implies$ `5`
- `s = "2*(5+5*2)/3+(6/2+8)"` $\implies$ `21`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

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

</div>
</details>

---

### 14. Sliding Window Maximum & Monotonic Deque Pattern

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">DEQUE 14</span>
  <span class="review-card-title">Sliding Window Maximum & Monotonic Deque Pattern</span>
  <span class="review-card-tag">Monotonic Deque · Index Expiry Eviction · Amortized O(1) Transition</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 239 · Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) — `https://leetcode.com/problems/sliding-window-maximum/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Sliding Window Maximum (LeetCode 239)**:
> You are given an array of integers `nums`, there is a sliding window of size $k$ which is moving from the very left of the array to the very right. You can only see the $k$ numbers in the window. Each time the sliding window moves right by one position.
> Return the max sliding window.

**Function Signature**:
```python
class SlidingWindowMaxSolution:
    @classmethod
    def maxSlidingWindow(cls, nums: List[int], k: int) -> List[int]: ...
```

**Examples**:
- `nums = [1,3,-1,-3,5,3,6,7], k = 3` $\implies$ `[3, 3, 5, 5, 6, 7]`
- `nums = [1], k = 1` $\implies$ `[1]`

**Key Invariant**:
- Maintain a monotonic deque of indices with values strictly decreasing from front to back. Pop front when index falls out of $i - k$, and pop back values smaller than current element for amortized $\mathcal{O}(1)$ step time.

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

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
<div class="review-block-label">🌐 Foundational Extensions: Sliding-Window Stream Aggregation & Bounded Dedup</div>

#### 1. Message Event Aggregator with 5-Minute Window
- **System Requirements**:
  - Ingests events `(timestamp, user_id, chat_id, event_type)` where `event_type \in {message, react, end_chat}`.
  - Window is $[t - 300, t]$ (inclusive 5 minutes). Output for each event in original order:
    1. `message_count`: Number of messages in same `chat_id` within the window;
    2. `active_chat_count`: Number of active chats for current `user_id` within the window.
  - **Active Chat Invariant**:
    - At least one `react` or `end_chat` occurs for `(user_id, chat_id)` within $[t - 300, t]$;
    - **The latest state event within the window is `react`**.
- **Out-of-Order Handling & Memory Pruning**:
  - Maintain a deque of timestamps per `chat_id` for message count;
  - Maintain a map `chat_state[chat_id] -> deque[(timestamp, type)]` per `user_id`;
  - Evict entries with timestamp $< t - 300$ during window progression to keep memory strictly bounded.

#### 2. Sliding-Window Duplicate Records Detection
- **System Requirements**:
  - Ingests `(id, text, title, timestamp)` with non-decreasing timestamps. Emit any record matching earlier content within 60 seconds ($[t - 60, t]$).
  - Explicit requirement: Purge records older than 60 seconds to bound memory by window size, not stream length.
- **Dual Deque + Hash Map Eviction Mechanism**:
  - `content_to_ts: Dict[content, int]` tracks latest in-window occurrence;
  - `order_queue: deque[(timestamp, content)]` maintains chronological order;
  - On arrival of record at $t$, purge stale entries from `order_queue` where $timestamp < t - 60$. If the stored timestamp matches the popped timestamp, erase from hash map. Memory remains $\mathcal{O}(\text{window\_rate})$.

</div>

</div>
</details>

---

### 15. Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">MONOTONIC STACK 15</span>
  <span class="review-card-title">Largest Rectangle in Histogram & Monotonic Stack Sentinel Pattern</span>
  <span class="review-card-tag">Monotonic Stack · Dual Sentinels · Width Calculation · 2D Reduction</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 84 · Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) — `https://leetcode.com/problems/largest-rectangle-in-histogram/`
> - [LeetCode 85 · Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/) — `https://leetcode.com/problems/maximal-rectangle/`

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

if __name__ == "__main__":
    assert HistogramSolution.largestRectangleArea([2, 1, 5, 6, 2, 3]) == 10
    assert HistogramSolution.largestRectangleArea([2, 4]) == 4
    assert HistogramSolution.largestRectangleArea([]) == 0
    assert HistogramSolution.largestRectangleArea([2, 1, 2]) == 3
    print("✅ Card 06 (Largest Rectangle in Histogram) all tests passed!")
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

### 16. Remove Duplicate Letters via Monotonic Stack

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">MONOTONIC STACK 16</span>
  <span class="review-card-title">Remove Duplicate Letters via Monotonic Stack</span>
  <span class="review-card-tag">Monotonic Stack · Last Occurrence Map · Visited Bitset · O(N)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 316 · Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/) — `https://leetcode.com/problems/remove-duplicate-letters/`
> - [LeetCode 1081 · Smallest Subsequence of Distinct Characters](https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/) — `https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Remove Duplicate Letters (LeetCode 316 / 1081)**:
> Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.

**Function Signature**:
```python
class RemoveDuplicateLettersSolution:
    @classmethod
    def removeDuplicateLetters(cls, s: str) -> str: ...
```

**Examples**:
- `s = "bcabc"` $\implies$ `"abc"`
- `s = "cbacdcbc"` $\implies$ `"acdb"`

**Monotonic Stack Invariant**:
- Maintain a monotonic increasing stack. Pop the top element if it is lexicographically larger than the current character AND it will appear again later in the string, maintaining $\mathcal{O}(n)$ time.

</div>

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

if __name__ == "__main__":
    assert RemoveDuplicateLettersSolution.removeDuplicateLetters("bcabc") == "abc"
    assert RemoveDuplicateLettersSolution.removeDuplicateLetters("cbacdcbc") == "acdb"
    print("✅ Card 12 (Remove Duplicate Letters) all tests passed!")
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

### 17. Valid Parenthesis String with Wildcard & Concrete String Enumeration

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STACK/DFS 17</span>
  <span class="review-card-title">Valid Parenthesis String with Wildcard & Concrete String Enumeration</span>
  <span class="review-card-tag">Interval Greedy · O(N) Dual Bounds · Branch-and-Bound DFS</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 678 · Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/) — `https://leetcode.com/problems/valid-parenthesis-string/`

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
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$ for boolean validity; $\mathcal{O}(3^K)$ for enumeration with $K$ wildcards.
- **Space Complexity**: $\mathcal{O}(1)$ for validity; $\mathcal{O}(N)$ recursion depth.

</div>

</div>
</details>

---

## Module 3: Heap & Priority Queue

### 18. Find Median from Data Stream & K-Way Merge

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">HEAP 18</span>
  <span class="review-card-title">Find Median from Data Stream & K-Way Merge</span>
  <span class="review-card-tag">Dual Heaps · Balance Invariant · Lazy Eviction</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 295 · Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) — `https://leetcode.com/problems/find-median-from-data-stream/`
> - [LeetCode 23 · Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) — `https://leetcode.com/problems/merge-k-sorted-lists/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Find Median from Data Stream (LeetCode 295)**:
> The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.
> Implement the `MedianFinder` class:
> - `MedianFinder()` initializes the MedianFinder object.
> - `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
> - `double findMedian()` returns the median of all elements so far.

**Interface Definition**:
```python
class MedianFinder:
    def __init__(self): ...
    def addNum(self, num: int) -> None: ...
    def findMedian(self) -> float: ...
```

**Examples**:
- `mf = MedianFinder()`
- `mf.addNum(1); mf.addNum(2); mf.findMedian()` $\implies$ `1.5`
- `mf.addNum(3); mf.findMedian()` $\implies$ `2.0`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

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
<div class="review-block-label">💡 Mechanism Invariants & Industrial Extensions</div>

- **Dual-Heap Invariant Mathematical Proof**:
  - The incoming stream is divided into two balancing partitions: lower partition $lo$ (max-heap via inverted keys) and upper partition $hi$ (min-heap).
  - **Ordering Invariant**: $\max(lo) \le \min(hi)$.
  - **Cardinality Invariant**: $|lo| \ge |hi|$ and $|lo| - |hi| \le 1$.
  - Median extraction is strictly $\mathcal{O}(1)$: $\max(lo)$ when odd, $(\max(lo) + \min(hi)) / 2$ when even.

#### 1. Merge k Sorted Lists (LeetCode 23)
- **Two Canonical Strategies**:
  1. **Min-Heap Priority Queue**: Maintain a min-heap of size $k$ initialized with list heads. Repeatedly extract the minimum node and push its `next`. Total time $\mathcal{O}(N \log k)$, auxiliary space $\mathcal{O}(k)$;
  2. **Divide-and-Conquer Pairing**: Pairwise merge $k$ lists across $\lceil \log_2 k \rceil$ levels. Achieves $\mathcal{O}(N \log k)$ time with $\mathcal{O}(1)$ space, maximizing CPU cache locality.

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
```

#### 2. K-th Element on a Streaming Time Window under Hard Memory Bound
- **Problem Requirements & Constraints**:
  - High-frequency data stream yields $(timestamp, value)$ tuples.
  - **Query**: At timestamp $now$, return the $K$-th smallest (or largest) element among events in $[now - W, now]$.
  - **Hard Memory Bound**: Disallow unbounded historical retention; resident memory must strictly be bounded by window duration $W$ and ingress rate $R$.
- **Approach 1: Bounded Integer Domain $[V_{min}, V_{max}]$ (FIFO Queue + Bucketed Counts)**:
  - Applicable to latency telemetry ($0 \sim 1000\text{ ms}$), integer prices in ticks:
  - **Data Structure**:
    1. FIFO double-ended queue `queue: Deque[Tuple[ts, val]]` storing active window events;
    2. Frequency array `buckets: List[int]` of size $B = V_{max} - V_{min} + 1$.
  - **Complexity**:
    - Ingress `add(ts, val)`: Evict expired entries where $ts' < ts - W$, decrementing corresponding bucket counts, then append new event. Amortized $\mathcal{O}(1)$;
    - Query `find_kth(now, k)`: Linear scan over $B$ buckets computing prefix sums until cumulative count reaches $k$. Strictly $\mathcal{O}(B)$ time.
  - **Memory Bound**: $\le R \cdot W \times 16\text{ bytes} + B \times 4\text{ bytes}$, completely avoiding dynamic tree allocations.

- **Approach 2: Unbounded Real Domain (FIFO Queue + Order-Statistics / Fenwick Tree)**:
  - For float or large integer ranges, utilize a Fenwick Tree over discretized coordinates or an order-statistics balanced tree (`SortedList`).
  - Ingress and eviction require $\mathcal{O}(\log M)$ time, query requires $\mathcal{O}(\log M)$ time (via binary lifting), where $M = R \cdot W$.

```python
import collections
from typing import Optional

class StreamingWindowKthBounded:
    """
    Time-Window K-th Element Tracker for Bounded Integer Domains.
    Time Complexity: add amortized O(1), find_kth O(B) (B = max_val - min_val + 1)
    Space Complexity: O(R * W + B) strictly memory-bounded
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
        """Returns 1-indexed k-th smallest element in window [current_time - W, current_time]"""
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
        """Returns 1-indexed k-th largest element in window [current_time - W, current_time]"""
        self._evict_expired(current_time)
        if k < 1 or k > self.total_count:
            return None
        cum = 0
        for i in range(self.num_buckets - 1, -1, -1):
            cum += self.buckets[i]
            if cum >= k:
                return self.min_val + i
        return None
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

### 19. MinStack, MaxStack, Streaming Median & System Extensions

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">HEAP/SYSTEM 19</span>
  <span class="review-card-title">MinStack, MaxStack, Streaming Median & System Extensions</span>
  <span class="review-card-tag">Dual Heaps · O(1) Extremum Stack · Lazy Eviction · Streaming Extensions</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 155 · Min Stack](https://leetcode.com/problems/min-stack/) — `https://leetcode.com/problems/min-stack/`
> - [LeetCode 716 · Max Stack](https://leetcode.com/problems/max-stack/) — `https://leetcode.com/problems/max-stack/`
> - [LeetCode 295 · Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) — `https://leetcode.com/problems/find-median-from-data-stream/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **MinStack, MaxStack & Streaming Median System (LeetCode 155 / 716 / 295)**:
> Design stack structures supporting $\mathcal{O}(1)$ push, pop, top, and retrieval of the minimum (or maximum) element:
> - `MinStack`: tracks running minimum using parallel min stack in $\mathcal{O}(1)$.
> - `Streaming Median`: maintains running median over an infinite stream via dual balancing heaps in $\mathcal{O}(\log n)$ per insertion.

**Interface Definition**:
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
<div class="review-block">
<div class="review-block-label">🌐 Foundational Extensions: High-Frequency Limit Order Book (HFT LOB)</div>

#### 1. Core API & Price-Time Priority
- **API Surface Area**:
  - `add_order(side, price, qty, order_id)`: Registers a resting limit order;
  - `cancel_order(order_id)`: Removes resting order by ID in $\mathcal{O}(1)$ or $\mathcal{O}(\log P)$;
  - `best_bid()` / `best_ask()`: Top of book quotes in $\mathcal{O}(1)$;
  - `top_of_book_volume()`: Cumulative quantity resting at top of book.

#### 2. Canonical Two-Level Storage Architecture
- **Sorted Price Maps**:
  - Bids: Price-keyed map sorted in **descending** order (`std::map<Price, PriceLevel, greater>`);
  - Asks: Price-keyed map sorted in **ascending** order;
  - `best_bid()` and `best_ask()` access root elements in $\mathcal{O}(1)$.
- **PriceLevel FIFO Queue**:
  - Each price level contains a doubly linked list maintaining arrival FIFO ordering.
- **Order ID Hash Index**:
  - `order_map: Dict[order_id, OrderLocation]` stores pointers to the side, price level, and DLL node iterator.
  - Cancellation unlinks the node from its price level in $\mathcal{O}(1)$ time, pruning empty price levels from the map.

</div>

</div>
</details>

---

### 20. Tiered Priority Task Scheduler

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">HEAP/SCHEDULER 20</span>
  <span class="review-card-title">Tiered Priority Task Scheduler</span>
  <span class="review-card-tag">Merchant Min-Heap · FIFO Timestamp Queues · Active Round-Robin · VIP Weighted Allocation</span>
</summary>
<div class="review-card-content">

> 💡 **Problem Type**: Standalone Algorithm Implementation (No direct LeetCode equivalent; logic and test specifications are standalone, please run the self-contained test suite below for local verification).

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Tiered Priority Task Scheduler**:
> Design a multi-merchant fair round-robin task scheduler. Merchants have priority tiers (e.g., VIP vs Regular). Tasks arrive with timestamps.
> - `receive_task(merchant_id, task_id, is_vip, arrival_time)`
> - `process_next_task() -> tuple`
> Must dispatch tasks fairly: round-robin across active merchants, giving VIP merchants proportional quota, preserving FIFO arrival order within each merchant.

**Interface Definition**:
```python
class TieredTaskScheduler:
    def __init__(self, vip_ratio: int = 2): ...
    def receive_task(self, merchant_id: str, task_id: str, is_vip: bool, arrival_time: int) -> None: ...
    def process_next_task(self) -> Optional[Tuple[str, str, bool, int]]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

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
<div class="review-block-label">🌐 Foundational Extensions: Fault-Tolerant Work Queue & Round-Robin Scheduling</div>

#### 1. Fault-Tolerant Work Queue with Leases, Retries & DLQ
- **State Machine & API**:
  - `reserve() -> Optional[(task_id, token)]`: Transitions task from `READY` to `RESERVED`, establishing a lease `deadline = now + timeout` and issuing an opaque `version_token`;
  - `complete(task_id, token)`: Terminal transition to `COMPLETED`;
  - `fail(task_id, token)`: Increments `attempts`. Returns task to `READY` if `attempts < max_attempts`; otherwise shifts to Dead-Letter Queue (DLQ).
- **Fencing Token Invariant**:
  - If a worker holding token $v$ stalls across its lease deadline, the scheduler increments the token to $v+1$ and re-enqueues the task.
  - When the stalled worker eventually calls `complete(task_id, v)`, the stale token is rejected, preventing double-processing.
- **Tri-Store Data Structure**:
  - `ready_queue`: `deque` for $\mathcal{O}(1)$ FIFO dispatch;
  - `task_store`: `Dict[task_id, TaskRecord]` authoritative state;
  - `lease_heap`: `heapq` of `(deadline, task_id, version_token)` for $\mathcal{O}(\log N)$ expiration polling.

#### 2. Round-Robin Task Scheduler (Citadel NXT)
- **Core Loop**:
  - Deque of runnable descriptors. Each tick pops front, executes quantum slice, and re-enqueues if incomplete.
  - **Cooperative vs Preemptive**: Cooperative relies on voluntary yields; preemptive uses timer interrupts.
  - **Deficit Round Robin (DRR)**: Tracks deficit credit per queue to support variable-size tasks.

#### 3. Tiered Task Manager with TTL & Quota (CodeSignal OA)
- **Progressive Architecture**:
  - Levels 1-2: Task CRUD with auto-increment IDs, ranked search;
  - Level 3: Per-user concurrency quotas and TTL auto-expiry;
  - Level 4: Historical event-sourced reconstruction and overdue assignment reporting.

</div>
</details>

---

### 21. Timestamp Task Scheduler with Direct ID Removal

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">HEAP/SCHEDULER 21</span>
  <span class="review-card-title">Timestamp Task Scheduler with Direct ID Removal</span>
  <span class="review-card-tag">Composite Min-Heap · Lazy Eviction · Hash Version Validation</span>
</summary>
<div class="review-card-content">

> 💡 **Problem Type**: Standalone Algorithm Implementation (No direct LeetCode equivalent; logic and test specifications are standalone, please run the self-contained test suite below for local verification).

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Timestamp Task Scheduler with Direct ID Removal**:
> Design a priority task scheduler managing tasks with timestamps, priorities, and unique task IDs:
> - `addTask(task_id, priority, timestamp)`: schedules a task.
> - `removeTask(task_id)`: cancels/evicts a task directly before execution.
> - `pollNextTask(current_time)`: dispatches highest-priority task whose timestamp $\le \text{current\_time}$.
> Must use a min-heap with lazy eviction and hash map version validation for $\mathcal{O}(\log N)$ amortized operations.

**Interface Definition**:
```python
class TimestampTaskScheduler:
    def __init__(self): ...
    def addTask(self, task_id: str, priority: int, timestamp: int) -> None: ...
    def removeTask(self, task_id: str) -> bool: ...
    def pollNextTask(self, current_time: int) -> Optional[Tuple[str, int, int]]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

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

## Module 4: NeetCode Quick-Recall Decision Framework & Mental Models

Under high-pressure interview settings, candidates must map requirements to optimal algorithmic patterns within 10 seconds. The matrices below crystallize the foundational patterns across Linked Lists, Stacks, and Heaps into a standardized mental lookup framework:

### 4.1 Linked List Mental Models & 10-Second Decision Matrix

| Pattern Archetype | Trigger Keywords / Problem Characteristics | Optimal Algorithmic Archetype | Canonical Problems (NeetCode / LC) |
| :--- | :--- | :--- | :--- |
| **Fast & Slow Pointers (Floyd)** | Cycle detection, cycle entry, list midpoint, palindrome test | `slow = slow.next`, `fast = fast.next.next`; upon collision reset one pointer to head | LC 141 (Cycle Detection), LC 142 (Cycle Entry), LC 876 (Middle Node), LC 234 (Palindrome) |
| **Sentinel Dummy Head** | Head node subject to deletion, merging, or prefix insertion | `dummy = ListNode(0, head)`; unifies edge-case branches with internal nodes | LC 19 (Remove Nth), LC 21 (Merge Lists), LC 2 (Add Two Numbers), LC 86 (Partition) |
| **Three-Pointer Inversion** | Reverse list, reverse subsegment, reverse in k-groups | `nxt = curr.next; curr.next = prev; prev = curr; curr = nxt` | LC 206 (Reverse List), LC 92 (Reverse II), LC 25 (Reverse k-Group) |
| **Splicing & Pointer Interleaving** | Deep copy with random pointers, reorder alternating halves | In-place node cloning `node.next = cloneNode` followed by split; split & interleave | LC 138 (Copy Random List), LC 143 (Reorder List) |
| **Composite Hash + DLL** | Strict $\mathcal{O}(1)$ cache insertion, access, and eviction | Doubly-linked list for chronological order + hash map for direct node handles | LC 146 (LRU), LC 460 (LFU), Meta Waitlist Queue |
| **Dynamic Array Swap-with-Last** | $\mathcal{O}(1)$ insertion, deletion, and uniform random sampling | Contiguous array stores values + hash map stores indices; delete via swap with tail | LC 380 (O(1) Set), LC 381 (Duplicates Allowed), Meta Randomized Container |

---

### 4.2 Stack & Monotonic Structure Decision Matrix

| Pattern Archetype | Trigger Keywords / Problem Characteristics | Optimal Algorithmic Archetype | Canonical Problems (NeetCode / LC) |
| :--- | :--- | :--- | :--- |
| **Pair Matching & Balance** | Bracket validation, adjacent duplicate removal, palindrome pops | Stack buffers expected closing counterparts; validates and pops upon match | LC 20 (Valid Parentheses), LC 1047 (Remove Adjacent Duplicates) |
| **Monotonic Stack** | Next Greater / Smaller element, histogram maximum rectangle | Maintain monotonic stack; pop violating elements and compute bounding interval | LC 739 (Daily Temperatures), LC 496 (Next Greater), LC 84 (Largest Rectangle in Histogram) |
| **Monotonic Deque** | Sliding window dynamic extremum (Running Maximum / Minimum) | Deque preserves strict monotonic descending order; pop tail if smaller, pop head if expired | LC 239 (Sliding Window Maximum), LC 1438 |
| **Operator Precedence Stacks** | Arithmetic evaluation, nested parentheses, precedence order | Operand stack + operator stack; immediate precedence reduction + recursive subexpression | LC 150 (Evaluate RPN), LC 224 (Basic Calculator), LC 227 (Basic Calculator II) |
| **Monotonic Stack Greedy Pruning** | Smallest distinct subsequence in lexicographical order | Monotonic increasing stack + last occurrence index map + in-stack set | LC 316 / LC 1081 (Remove Duplicate Letters) |

---

### 4.3 Heap & Priority Queue Decision Matrix

| Pattern Archetype | Trigger Keywords / Problem Characteristics | Optimal Algorithmic Archetype | Canonical Problems (NeetCode / LC) |
| :--- | :--- | :--- | :--- |
| **Top-K Dynamic Tracking** | K largest or smallest elements in streaming / massive data | Maintain min-heap of size $K$ (for K largest) or max-heap; pop root when exceeding capacity | LC 215 (Kth Largest Element), LC 347 (Top K Frequent), LC 703 (Kth Largest in Stream) |
| **Multi-Way K-Merge** | Merge K sorted lists or sorted arrays | Maintain priority queue containing current heads of the $K$ sorted streams | LC 23 (Merge K Sorted Lists), LC 378 (Kth Smallest in Matrix) |
| **Dual Balancing Heaps** | Unordered stream ingestion, $\mathcal{O}(1)$ median retrieval | Max-heap (lower half) + Min-heap (upper half); balance capacity difference $\le 1$ | LC 295 (Find Median from Data Stream) |
| **Cooling Simulation Scheduler** | Task execution with cooldown penalty of $N$ cycles | Greedy max-heap selects highest-frequency task + waiting queue tracks cooldown | LC 621 (Task Scheduler), LC 355 (Design Twitter) |
