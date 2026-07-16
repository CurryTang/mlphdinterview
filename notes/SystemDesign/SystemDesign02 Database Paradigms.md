# System Design 02 · 数据库基本范式

课程位置：[[SystemDesign01B Virtualization Containers|01B 虚拟化与容器]] → 本篇 → [[SystemDesign03 Database Scaling|03 数据库扩展]]

数据库选型先看两件事：哪些业务不变量必须原子成立，系统最重要的访问路径是什么。产品名字放到后面。

```text
transaction boundary -> correctness
access pattern        -> data layout and indexes
```

“SQL 不能扩展”或“NoSQL 没有事务”都太粗。现代产品的能力有重叠，差别在默认数据模型、事务边界和扩展代价。

---

## 1 · RDBMS：先表达关系和约束

关系模型把数据放进 row 和 table，通过 primary key、foreign key、unique constraint 和 transaction 表达不变量。

```sql
BEGIN;

UPDATE accounts
SET balance = balance - 100
WHERE account_id = 1 AND balance >= 100;

UPDATE accounts
SET balance = balance + 100
WHERE account_id = 2;

COMMIT;
```

这段代码的重点不是 SQL 语法，而是两个余额变化属于同一个提交边界。任意一步失败，整个转账都不能留下半成品。

RDBMS 适合：

- entity 之间关系密集；
- 不变量经常跨 row 或 table；
- 查询方式多，未来还会变化；
- 需要成熟的 secondary index、join 和 ad-hoc query。

代价也很直接：跨节点 transaction、join 和全局 constraint 很难随 shard 数量一起扩展。

---

## 2 · NoSQL：先围绕访问路径组织数据

NoSQL 不是一种数据库。KV、document、wide-column 和 graph 的数据模型不同，但很多系统共同强调 partition-local access。

```text
GetUser(user_id)
ListOrders(user_id, created_at range)
GetFeed(viewer_id, cursor)
```

建模时先为这些读取选择 partition key 和 sort key。为了让一次请求命中单 partition，数据可能被反范式化：

```json
{
  "user_id": "u42",
  "profile": {"name": "Kai"},
  "shipping_city": "Seattle"
}
```

城市改名时，多个 document 可能要更新。读变简单，写入和一致性成本上升。

NoSQL 常见优势：

- 单 key / 单 partition 路径简单；
- 数据可以按 partition 水平分布；
- schema 对稀疏或变化字段更宽松；
- 延迟和吞吐更容易围绕固定 access pattern 规划。

它不适合拿来逃避建模。Partition key 选错后，hot key、scan 和跨 partition transaction 会一起出现。

---

## 3 · Transaction 是业务不变量的边界

ACID 可以这样记：

| 性质 | 实际问题 |
|---|---|
| Atomicity | 会不会只完成一半？ |
| Consistency | 提交后约束是否仍成立？ |
| Isolation | 并发操作会看到什么中间状态？ |
| Durability | 返回成功后，故障会不会让结果消失？ |

先写不变量，再决定 transaction 范围：

```text
order.total == sum(order_items)
payment may be captured at most once
username must be unique
inventory cannot fall below zero
```

如果这些条件必须跨多个 entity 原子成立，关系数据库或支持相应 transaction 的 distributed SQL 往往更省心。若可以拆成状态机并接受 eventual consistency，event-driven workflow 也可能合适。

### 不要把 database transaction 和 business workflow 混在一起

单库 transaction 通常在毫秒内结束。跨 payment provider、inventory、shipping 的订单流程可能持续数分钟，不能一直锁着数据库连接。

```text
local transaction
  -> write order + outbox
  -> async payment command
  -> state transition
  -> compensation when needed
```

这类流程靠 idempotency、state machine、outbox 和 compensation 维持业务一致性。消息细节见 [[SystemDesign06 Async Messaging Systems|06 异步消息系统]]。

---

## 4 · 强一致和最终一致在说什么

一致性必须绑定到具体操作。

```text
User updates profile to v2
User immediately reads profile
```

可能的 contract：

- Linearizable read：像只有一个最新副本；
- Read-your-writes：该用户至少能读到自己的 v2；
- Monotonic read：已经看到 v2 后不会退回 v1；
- Eventual consistency：没有新写入时，副本最终收敛。

同一个系统可以混用。订单确认页读 primary，公开商品页读 replica，推荐特征允许几秒旧。比起宣布“系统强一致”，逐条 API 写清 contract 更有用。

Replication lag、failover 和 RPO/RTO 见 [[SystemDesign05 Reliability Replication|05 可靠性与复制]]。

---

## 5 · 选型时问这六个问题

### 1. 基本读写单位是什么

```text
single key?
document?
partition + range?
graph traversal?
multi-row relation?
```

### 2. 最重要的 query 能否由主键或索引直接完成

如果每次 feed 都 scan 全表再过滤，换数据库名字也救不了它。先把 query 和 index 写出来。

### 3. Transaction 跨多大范围

单 row、单 document、单 partition、跨 partition，成本逐级上升。把强不变量尽量放在同一 transaction boundary 内。

### 4. 数据怎样分区

Partition key 决定 locality、并行度和热点。低基数 country code 经常不够均匀；hash(user_id) 更均衡，却让按地区扫描更麻烦。

### 5. 哪些读取允许旧

允许旧读可以使用 replica、cache 和 materialized view。权限、余额和库存扣减通常要更谨慎。

### 6. 运维复杂度放在哪里

关系模型把复杂度放在 database engine 和 query planner；access-pattern-first 模型把更多复杂度放到应用写路径、反范式化和数据修复。没有免费选项。

---

## 6 · 常见产品的设计中心

| 系统类型 | 设计中心 | 常见用途 |
|---|---|---|
| PostgreSQL / MySQL | relation、constraint、transaction | order、account、metadata |
| Dynamo-style KV / document | partition key、可预测访问路径 | profile、session、serving KV |
| Cassandra-style wide column | partition + clustering order、高写吞吐 | time series、event、timeline |
| MongoDB-style document | 聚合 document、灵活字段 | content、catalog、profile |
| Spanner-style distributed SQL | 分布式 transaction + SQL | 全球 metadata、强一致业务数据 |
| Graph database | vertex / edge traversal | fraud graph、relationship exploration |

这张表只能给起点。最终还要检查具体产品版本、transaction scope、index、region topology、backup 和团队运维能力。

---

## 7 · 几个典型判断

### 订单和支付

从 RDBMS 开始。订单状态、金额、幂等键和账务约束需要可靠 transaction。规模增长后再把 search、analytics 和 event stream 分出去。

### 用户 profile

如果主要是按 user_id 整体读取，document / KV 很自然；若 profile 与权限、组织和 billing 关系密集，RDBMS 可能更简单。

### Feed timeline

按 viewer_id 分区、按 rank_key 或 time 排序，适合 wide-column / sorted KV。Timeline 是派生索引，Post metadata 仍可放关系库或 document store。

### 日志与事件

高吞吐 append、按时间保留和 replay 更像 log system，不应硬塞进 OLTP 表。需要查询时再进入 search 或 analytical store。

### 金融账本

优先保护不可变 entry、双边平衡、唯一 transaction ID 和审计。不要因为写入量大就先牺牲 transaction semantics。

---

## 8 · 数据可以拆，但别过早拆库

一个成熟系统常常是 polyglot persistence：

```text
RDBMS          authoritative order metadata
Redis          hot cache
Object store   blobs
Event log      change propagation
Search index   text retrieval
Warehouse      analytics
```

这不代表第一天就需要六套系统。每增加一种 store，就多一套 schema、backup、权限、监控和数据修复流程。先用最简单的 source of truth，等访问模式或规模真的分化后再拆。

---

## 9 · 面试检查清单

```text
Correctness
- 哪些不变量必须原子成立？
- transaction 是单 key、单 partition 还是跨 entity？

Access
- top read/write path 是什么？
- index 和 partition key 是什么？
- 是否存在 scan、hot key 或跨 partition query？

Consistency
- 哪些 API 需要 read-your-writes？
- 哪些派生数据允许旧，允许多久？

Operations
- 数据怎样 backup 和 restore？
- schema / index 变更怎样发布？
- 团队能否运维引入的新 store？

Growth
- 先加 index、cache、replica，还是已经需要 shard？
- 分片细节见 03 数据库扩展。
```

一句话记忆：先用 transaction boundary 保护正确性，再用 access pattern 决定数据布局。数据库品牌是这两个问题之后的选择。
