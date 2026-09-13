# System Design 01D · Redis

课程位置：[[SystemDesign01C Kubernetes|01C Kubernetes]] → 本篇 → [[SystemDesign02 Database Paradigms|02 数据库]]

Redis 解决的核心问题是：如何提供极快的、共享的、可变的状态。
但它**不**应该作为业务事实（如交易记录或订单表）的 Source of Truth，除非是作为具有明确 TTL（Time To Live）的锁或租约存储。

Redis 是一个基于内存的键值（KV）存储系统，具有可选的持久化能力。它的核心使用场景包括：
- 缓存（Cache）
- 会话（Session）
- 速率限制（Rate Limit）
- 分布式锁与租约（Lock / Lease）
- 短暂的计数器（Ephemeral Counters）
- 准入过滤（fast reject / fast allow）

---

## 1 · 核心 API 与数据结构

Redis 并非只支持简单的字符串，其提供的数据结构高度优于普通的 Key-Value Store。

### 基础数据操作
- `GET` / `SET` / `DEL`：最基础的键值操作。
- `EXPIRE`：为键设置存活时间（TTL）。对于内存资源而言，几乎所有存入 Redis 的数据都应当配有明确的过期策略。

### 并发与协调指令
- `SET NX`（Set if Not eXists）：仅当键不存在时才执行写入。这是实现轻量级分布式锁的基础。
- `INCR` / `DECR`：单线程模型保证了计数的原子性，适用于限流和并发极高的简易统计。

### 复杂数据结构
- `HASH`：提供字段级别的存取，非常适合存储序列化的对象，例如用户的 Session blob。
- `ZSET`（有序集合）：基于跳表实现，提供 $O(\log N)$ 复杂度的插入和范围查询，适合实时排行榜和按时间戳排序的事件集。
- `LIST` / `STREAM`：可用作轻量级队列。
需要指出，尽管可以作为队列，但 Redis List 并非像 Kafka 那样的持久化、可重放日志（Durable Log）。具体差异请参阅 [[SystemDesign06 Async Messaging Systems|06 异步消息系统]]。

---

## 2 · 缓存 (Cache)

Redis 最普遍的角色是数据库前置的内存层。作为缓存，它仅仅是底层数据库数据的一个复制品。
核心原则是：一旦发生 Cache Miss，应用必须能够安全回退并命中数据库。

### 缓存量化估算基准：QPS 与容量 Numbers

在规划 Redis 集群时，必须依托以下核心物理常数进行容量与吞吐估算：

#### 1. 缓存 QPS 估算与 CPU 物理公式
- **网络往返（RTT）**：同可用区机房内简单 GET/SET 的网络延迟约为 $\approx 0.2 - 1.0\text{ ms}$；
- **单实例吞吐上限**：单机通常能承载 **数万 QPS 级别**（受 CPU 单核性能与网卡包转发 PPS 限制）。
- **CPU 计算时间估算公式**：
  $$\text{QPS} \approx N_{\text{cores}} \times \frac{1000}{t_{\text{cpu}}} \times u$$
  *算例*：分配 4 核，平均每个请求占用 CPU 时间 $t_{\text{cpu}} = 0.1\text{ ms}$，安全利用率 $u = 0.8$：
  $$\text{QPS} \approx 4 \times \frac{1000}{0.1} \times 0.8 = 32,000\text{ QPS}$$

#### 2. 缓存容量：1 GB 内存能放多少 Key？
- **理想基准**：按单条记录及元数据约 $200\text{ B}$ 计算，$\mathbf{1\text{ GB 内存} \approx 500\text{ 万 Key}}$；
- **生产保守经验值**：考虑到内存分配器（jemalloc）碎片、数据结构指针膨胀以及预留的 $30\%$ 安全水位，实际按：
  $$\mathbf{1\text{ GB 内存} \approx 200\text{ 万} - 300\text{ 万稳定 Key}}$$

### Cache Aside 与 Write-Through
最标准的模式是 Cache Aside：

```python
def get_user(user_id):
    # 1. 尝试从缓存获取
    user = redis.get(f"user:{user_id}")
    if user is not None:
        return deserialize(user)
    
    # 2. 缓存未命中，回源到 DB
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # 3. 写入缓存并设置 TTL，然后返回
    if user is not None:
        redis.set(f"user:{user_id}", serialize(user), ex=3600)
    return user
```
写入数据时，应用更新 DB，并删除 Redis 缓存（Cache Invalidation）。不采取严格的 Write-Through（同时同步写入缓存和 DB），因为直接 Invalidate 可以避免并发写入时的竞态条件。

### TTL 与缓存击穿 (Cache Stampede)
所有缓存数据必须带有 TTL。当高热度数据的 TTL 突然到期，大量并发请求可能同时击穿缓存，对 DB 造成毁灭性打击。
应对方案：
- **互斥锁机制**：发现缓存失效时，仅允许第一个线程去查询 DB 并回填缓存，其他线程等待。
- **概率性提前过期 (Probabilistic Expire)**：在数据真正过期前，计算一个基于随机数的概率。如果随机触发，就让当前请求异步去后台更新缓存，而前端仍返回旧值。

### 内存淘汰策略 (Eviction Policies)
由于 Redis 的存储受限于物理内存，当内存触及上限（`maxmemory`）时，它必须决定如何丢弃数据：
- **noeviction**：默认策略，内存满时拒绝所有新写入。这对于纯缓存应用是灾难，但对于必须保证锁不被静默回收的场景是正确的安全底线。
- **allkeys-lru / volatile-lru**：驱逐最近最少使用（Least Recently Used）的数据。`volatile` 仅驱逐设置了 TTL 的键。
- **allkeys-lfu / volatile-lfu**：驱逐最不经常使用（Least Frequently Used）的数据。

在纯缓存集群中，常用 `allkeys-lru`。但在同时承载 Session 或锁的混合集群中，通常配置 `volatile-lru`，并强制所有缓存数据必须设置 TTL。这样可以保护那些未设 TTL 的重要协调状态（如长租约）不被意外清退。

### 失效策略 vs 短 TTL
如果是对一致性要求不高的视图数据，配置短 TTL 通常足够。
如果对一致性有较高要求，则必须在 DB 更新后主动 Invalidate。不要尝试“更新”缓存里的某个字段，这极易引发并发脏数据。直接删除对应的缓存键通常更安全可靠。

---

## 3 · 会话 (Session)

在无状态的 API 服务中，不透明的 Session ID 需要一个集中的共享存储。

```text
User Request (Cookie: session_id)
  -> API Instance
      -> Redis: GET session:session_id
          -> Session Blob (user_id, roles)
```

Redis 非常适合存储这类会话数据：
- **快速响应**：内存读取级别，不会拖慢整体 API 请求。
- **驱逐影响有限**：会话在 Redis 发生 Eviction（内存淘汰）时，仅仅意味着用户需要重新登录，或系统需要从持久化数据库重新拉取认证资料。

**重要限制**：
Redis Session 不能替代授权体系中的源头（Source of Truth）。
当封禁某个用户或变更核心权限时，数据库记录必须变更。通常在数据库更新的同一事务后，应立刻在 Redis 中删除对应的 Session。

---

## 4 · 锁与租约 (Lock / Lease)

Redis 经常被用来在分布式系统中做协调，防止多个 Worker 同时执行相同的长耗时任务。

### SET NX PX 作为租约
正确的分布式锁实现应该是一次“租约”（Lease），而非跨进程的强制互斥锁。

```redis
SET resource_name my_random_worker_id NX PX 30000
```
- **NX**：仅在不存在时创建。
- **PX**：必须带有超时时间（此例为 30 秒）。如果不带 TTL，一旦持有锁的 Worker 进程崩溃，死锁就会永久发生。

### 解锁与 Fencing Token
- **只能由持有者解锁**：释放锁时必须验证 `my_random_worker_id`。如果通过 Lua 脚本执行释放，可以确保原子性，防止 Worker A 超时后锁被释放，Worker B 获得了锁，而 A 醒来后却错误地清理了 B 的锁。
- **Fencing Token / Versioning**：如果临界区内包含向 DB 的写入，单靠 Redis 锁并不能保证绝对安全。Worker A 可能在租约到期后、甚至发生系统级停顿后，依然向 DB 提交数据。必须在 DB 层面配合 Version 检查或 Fencing Token 来保证数据最终一致性。

### Redlock 争议
单实例 Redis 锁本质上是一种容错能力有限的租约机制。
如果没有仔细阅读分布式系统中的时钟漂移和网络分区故障模型，绝对不要将其视为可以在 Redis 集群上提供强一致性的线性化锁。

---

## 5 · 速率限制 (Rate Limit)

速率限制用于保护后端服务免受恶意流量攻击。
由于 Redis 基于内存，它可以极快地更新计数。

- **算法实现**：常见的 Token Bucket（令牌桶）或 Sliding Window（滑动窗口），在 Redis 中通常使用 `INCR` 配合 `EXPIRE`，或者利用 `ZSET` 来实现时间窗口内的请求计数。
- **精度取舍**：限流是近似的。挡崩溃可以；计费不行。

---

## 6 · Fast reject / fast allow

秒杀这类突发写，200k/s 不能先打 DB 或 Kafka。Redis 做准入过滤器：大多数请求在内存里被拒绝或放行，DB 只处理 worker 预算内的那一小截。库存真相仍在 DB。见 [[SystemDesign10 Flash Sale|10 秒杀]]。

两个出口不对称：

| | 含义 | 错了会怎样 |
|---|---|---|
| Fast reject | 确定没戏：售罄、已买过、超限流 | 多拒几次，用户重试。安全方向 |
| Fast allow | 看起来还有名额，写入 MQ，返回 202 | 名额是 Redis 计数，不是库存。DB 仍可能失败 |

```text
Purchase
  -> Redis 准入
       reject -> 立刻 4xx / sold out
       allow  -> MQ ack -> 202
  -> Worker 才扣 DB 库存
```

一组键：

```text
soldout:{sale_id}                 售罄旗标，SET 之后只读
tokens:{sale_id}                  剩余名额，活动开始时 SET 为库存
ordered:{sale_id}:{user_id}       已进入过这条活动，SET NX
```

Lua 一次做完，避免 GET 后再 DECR 的缝：

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

`return 1` 才 enqueue。`ordered` 用 SET NX 也可以单独做：已存在直接拒绝，不必再碰 `tokens`。

售罄之后 API 进程可以本地记住 `soldout`，后续请求连 Redis 都不打。这是 fast reject 的第二层。

Worker 在 DB 事务失败（库存已尽、唯一键冲突）时，按需 `INCR tokens` 并把 `ordered` 删掉，否则 Redis 名额会漏。Redis 崩溃丢计数时，宁可把 `tokens` 估小、多 reject，不要当库存源。

---

## 7 · Redis 不擅长什么

了解组件的边界，与了解其能力同样重要。

- **大对象 (Large Objects)**：Redis 采用单线程处理命令。如果传输和处理数兆字节的值，会阻塞后续所有请求的执行。
- **Ad-hoc 查询**：不支持类似 SQL 的复杂关联查询。
- **多键事务作为数据库 (Multi-key Txn)**：尽管 Redis 支持 Lua 脚本，但不要把它当作支持复杂 ACID 事务的关系型数据库。
- **基于磁盘的 Source of Truth**：除非经过极度特殊的设计并容忍性能下降，否则绝不应将不可丢失的业务事实（如财务账本）存入 Redis。

---

## 8 · 持久化模式对比

虽然 Redis 是内存数据库，但它提供了多种持久化策略以应对崩溃恢复。

| 模式 | 机制 | 进程崩溃时丢失什么 |
|---|---|---|
| **None** | 完全不落盘，纯内存模式。 | 丢失实例中的全部数据。适合纯粹的可随时重建的缓存。 |
| **RDB** | 定期对内存数据做快照，写入磁盘文件。 | 丢失自上次成功快照之后的所有变更（通常是几分钟的数据）。 |
| **AOF** | 记录每条写命令，顺序追加到日志。 | 视 fsync 策略而定。通常丢失不到 1 秒的数据，但恢复速度较慢。 |

**总结**：在 Redis 作为 Cache 或 Session 组件时，开启混合持久化（AOF 结合 RDB）的核心目的是为了在集群崩溃重启后，快速恢复热点数据，避免所有流量瞬间打向脆弱的数据库。它不是为了将会计级别的记录长久固化。
