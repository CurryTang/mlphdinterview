# System Design 01D · Redis

Course Location: [[SystemDesign01C Kubernetes|01C Kubernetes]] → this note → [[SystemDesign02 Database Paradigms|02 Database]]

The core problem Redis solves is providing extremely fast, shared, mutable state. 
However, it should **not** serve as the source of truth for business facts (like transaction records or order tables) unless explicitly designed as a lock or lease store with a clear TTL (Time To Live).

Redis is an in-memory Key-Value (KV) store with optional persistence capabilities. Its primary use cases include:
- Cache
- Session management
- Rate limiting
- Distributed locks and leases
- Ephemeral counters
- Admission filter (fast reject / fast allow)

---

## 1 · Core API and Data Structures

Redis is not just for simple strings; it provides data structures that vastly outperform typical key-value stores.

### Basic Data Operations
- `GET` / `SET` / `DEL`: The fundamental operations.
- `EXPIRE`: Sets a Time To Live (TTL) on a key. For memory resources, nearly all data stored in Redis should have a defined expiration strategy.

### Concurrency and Coordination Commands
- `SET NX` (Set if Not eXists): Executes the write only if the key does not already exist. This forms the foundation of lightweight distributed locks.
- `INCR` / `DECR`: The single-threaded model ensures these increments are atomic, making them ideal for rate limiting and simple high-concurrency statistics.

### Complex Data Structures
- `HASH`: Provides field-level access, perfect for storing serialized objects, such as a user's session blob.
- `ZSET` (Sorted Set): Based on a skip list, it offers $O(\log N)$ insertion and range queries. It is ideal for real-time rankings and events sorted by timestamp.
- `LIST` / `STREAM`: Can be used as lightweight queues.
Note: Although they function as queues, a Redis List is not a highly reliable, replayable durable log like Kafka. See [[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]] for details.

---

## 2 · Cache

### Quantitative Cache Numbers: QPS & Capacity Sizing

When sizing a Redis cluster, rely on these foundational physical constants:

#### 1. Cache QPS & CPU Sizing Formula
- **Round-Trip Time (RTT)**: Intra-AZ simple GET/SET network latency is $\approx 0.2 - 1.0\text{ ms}$;
- **Single-Node Throughput**: A single instance comfortably handles **tens of thousands of QPS** (CPU and network packet bound).
- **CPU Sizing Formula**:
  $$\text{QPS} \approx N_{\text{cores}} \times \frac{1000}{t_{\text{cpu}}} \times u$$
  *Example*: 4 cores, average CPU time $t_{\text{cpu}} = 0.1\text{ ms}$, utilization target $u = 0.8$:
  $$\text{QPS} \approx 4 \times \frac{1000}{0.1} \times 0.8 = 32,000\text{ QPS}$$

#### 2. Cache Capacity: How Many Keys Fit in 1 GB RAM?
- **Theoretical Benchmark**: At $\approx 200\text{ B}$ per item, $\mathbf{1\text{ GB RAM} \approx 5,000,000\text{ keys}}$;
- **Production Conservative Rule**: Accounting for jemalloc fragmentation, pointer overhead, and $30\%$ safety headroom:
  $$\mathbf{1\text{ GB RAM} \approx 2,000,000 - 3,000,000\text{ stable keys}}$$

The most common role for Redis is an in-memory layer in front of a database. As a cache, it is merely a copy of the underlying database.
The core principle is: in the event of a cache miss, the application must safely fall back to the database.

### Cache Aside vs. Write-Through
The standard pattern is Cache Aside:

```python
def get_user(user_id):
    # 1. Attempt to fetch from cache
    user = redis.get(f"user:{user_id}")
    if user is not None:
        return deserialize(user)
    
    # 2. Cache miss, query the DB
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # 3. Write to cache with a TTL, then return
    if user is not None:
        redis.set(f"user:{user_id}", serialize(user), ex=3600)
    return user
```
For writes, the application updates the DB and deletes the Redis cache (Cache Invalidation). Strict Write-Through (synchronous writes to both DB and cache) is rarely used; invalidation is safer as it avoids race conditions during concurrent writes.

### TTL and Cache Stampede
All cached data must have a TTL. If highly accessed data expires abruptly, massive concurrent requests can penetrate the cache simultaneously, crushing the DB.
Solutions:
- **Mutex Lock**: When a miss occurs, allow only one thread to query the DB and backfill the cache, while others wait.
- **Probabilistic Expire**: Before true expiration, calculate a probability based on a random number. If triggered, one request asynchronously updates the cache in the background while still returning the stale value to the client.

### Memory Eviction Policies
Because Redis is bound by physical memory, it must decide how to discard data when the limit (`maxmemory`) is reached:
- **noeviction**: The default policy. It rejects all new writes when memory is full. This is disastrous for pure caching but is the correct safety baseline when locks must not be silently reclaimed.
- **allkeys-lru / volatile-lru**: Evicts the Least Recently Used data. The `volatile` variant only targets keys with an explicitly set TTL.
- **allkeys-lfu / volatile-lfu**: Evicts the Least Frequently Used data.

In pure cache clusters, `allkeys-lru` is common. However, in mixed clusters handling both sessions and locks, `volatile-lru` is usually preferred, alongside a strict rule that all cache writes must include a TTL. This protects critical coordination states without a TTL (like long leases) from accidental eviction.

### Invalidation vs. Short TTL
If consistency requirements are low, a short TTL is sufficient.
If high consistency is required, you must actively invalidate the cache upon DB updates. Do not attempt to "update" a specific field in the cache, as this easily causes dirty data. Deleting the key entirely is always safer.

---

## 3 · Session

In stateless API services, an opaque Session ID requires a centralized, shared store.

```text
User Request (Cookie: session_id)
  -> API Instance
      -> Redis: GET session:session_id
          -> Session Blob (user_id, roles)
```

Redis is ideal for this:
- **Fast Response**: Memory-speed reads do not delay the overall API request.
- **Limited Eviction Impact**: If session eviction occurs due to memory limits, it simply means users must log in again, or the system fetches credentials from the DB.

**Important Limitation**:
A Redis Session cannot substitute the authorization system's source of truth.
When banning a user or changing core permissions, the database record must be updated. Generally, you should delete the corresponding session in Redis in the same transaction as the DB update.

---

## 4 · Lock and Lease

Redis is frequently used in distributed systems for coordination, preventing multiple workers from executing the same long-running task simultaneously.

### SET NX PX as a Lease
A correct distributed lock implementation should act as a "lease," not a strict cross-process mutex.

```redis
SET resource_name my_random_worker_id NX PX 30000
```
- **NX**: Creates only if it does not exist.
- **PX**: Must include a timeout (e.g., 30 seconds). Without a TTL, a dead worker holding the lock permanently causes a deadlock.

### Unlocking and Fencing Tokens
- **Unlock only by the holder**: When releasing, `my_random_worker_id` must be verified. Utilizing a Lua script ensures atomicity, preventing a scenario where Worker A times out, Worker B acquires the lock, and Worker A wakes up to erroneously delete B's lock.
- **Fencing Token / Versioning**: If the critical section includes a database write, a Redis lock is fundamentally insufficient. Worker A might still commit data to the DB after its lease expires (or after a system-level pause). You must pair this with version checks or fencing tokens at the DB layer to ensure ultimate data consistency.

### The Redlock Controversy
In short: A single-instance Redis lock is a lease mechanism with limited fault tolerance.
Unless you have rigorously studied clock drift and network partition failure models, never treat it as a strongly consistent, linearizable lock across a Redis cluster.

---

## 5 · Rate Limiting

Rate limiting protects backend services from abusive traffic spikes.
Because it is memory-based, Redis updates counters incredibly fast.

- **Algorithms**: Common algorithms like the Token Bucket or Sliding Window are implemented in Redis using `INCR` combined with `EXPIRE`, or by leveraging `ZSET` to count requests within a time window.
- **Precision**: Approximate is OK for crash prevention. Not for billing.

---

## 6 · Fast reject / fast allow

A flash-sale burst at 200k/s cannot hit DB or Kafka first. Redis is the admission filter: most requests die or pass in memory; the DB only sees the worker budget. Stock truth stays in the DB. See [[SystemDesign10 Flash Sale|10 Flash sale]].

The two exits are not symmetric:

| | Meaning | If wrong |
|---|---|---|
| Fast reject | No chance: sold out, already ordered, over rate limit | Extra rejects; user retries. Safe direction |
| Fast allow | Looks like a slot is left; enqueue; return 202 | Slot is a Redis counter, not stock. DB may still fail |

```text
Purchase
  -> Redis admission
       reject -> immediate 4xx / sold out
       allow  -> MQ ack -> 202
  -> Worker decrements DB stock
```

Keys:

```text
soldout:{sale_id}                 flag; SET once, then read-only
tokens:{sale_id}                  remaining slots; SET to stock at start
ordered:{sale_id}:{user_id}       already admitted; SET NX
```

One Lua script so GET then DECR has no gap:

```lua
-- KEYS[1]=soldout  KEYS[2]=tokens  KEYS[3]=ordered
if redis.call('GET', KEYS[1]) then
  return 0          -- reject: sold out
end
if redis.call('EXISTS', KEYS[3]) == 1 then
  return 0          -- reject: already in
end
local n = redis.call('DECR', KEYS[2])
if n < 0 then
  redis.call('INCR', KEYS[2])
  redis.call('SET', KEYS[1], '1')
  return 0          -- reject: just sold out
end
redis.call('SET', KEYS[3], '1')
return 1            -- allow
```

Enqueue only on `return 1`. `ordered` can also be a standalone `SET NX`: already exists → reject, skip `tokens`.

After sell-out the API process can remember `soldout` locally and skip Redis. That is the second layer of fast reject.

If the DB txn fails (stock gone, unique key), the worker `INCR`s `tokens` and deletes `ordered`, or Redis leaks slots. If Redis crashes and loses the counter, set `tokens` low and reject extra. Do not treat it as stock.

---

## 7 · What Redis is Bad At

Knowing a component's boundaries is as important as knowing its capabilities.

- **Large Objects**: Redis processes commands in a single thread. Transmitting and processing multi-megabyte values will block all subsequent requests.
- **Ad-hoc Queries**: It does not support complex `WHERE` filters or relational queries.
- **Multi-key Transactions**: Although Redis supports Lua scripts, do not use it as a substitute for a relational database requiring complex ACID transactions.
- **Disk-backed Source of Truth**: Unless meticulously designed and tolerant of performance degradation, you must never store critical, non-recoverable business facts (such as financial ledgers) exclusively in Redis.

---

## 8 · Persistence Modes

Although Redis is an in-memory database, it offers persistence strategies for crash recovery.

| Mode | Mechanism | What is lost on crash |
|---|---|---|
| **None** | Pure memory, no disk writes. | All instance data is lost. Ideal for caches that can be rebuilt anytime. |
| **RDB** | Takes periodic memory snapshots to a disk file. | All modifications since the last successful snapshot (usually minutes of data). |
| **AOF** | Appends every write command to a sequential log. | Depends on the fsync policy. Usually loses less than 1 second of data, but recovery is slower. |

**Summary**: When using Redis as a Cache or Session store, enabling mixed persistence (AOF with RDB) is strictly for fast data recovery after a cluster crash to prevent traffic from destroying the database. It is not intended to provide accounting-level, permanent data storage.
