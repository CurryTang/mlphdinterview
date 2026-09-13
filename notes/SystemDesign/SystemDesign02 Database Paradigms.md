# System Design 02 · 数据库

课程位置：[[SystemDesign01D Redis|01D Redis]] → 本篇 → [[SystemDesign04 Storage Systems|04 存储系统]]

数据库选型先看两件事：哪些业务不变量必须原子成立，系统最重要的访问路径是什么。产品名字放到后面。

```text
transaction boundary -> correctness
access pattern       -> data layout and indexes
```

“SQL 不能扩展”或“NoSQL 没有事务”都太粗。现代产品的能力有重叠，差别在默认数据模型、事务边界和扩展代价。

| API | 单位 | 例子 |
|---|---|---|
| SQL | row / txn | Postgres, MySQL |
| KV get/put | key | Dynamo, Redis-as-DB (usually wrong) |
| Document | doc by id | Mongo |
| Wide-column | partition + clustering | Cassandra |
| Graph | vertex/edge walk | Neo4j |

---


## 1 · RDBMS vs NoSQL


## RDBMS：先表达关系和约束

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
代价也很直接：跨节点 transaction、join 和全局 constraint 很难随 shard 数量一起扩展。


## NoSQL：先围绕访问路径组织数据

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
它不适合拿来逃避建模。Partition key 选错后，hot key、scan 和跨 partition transaction 会一起出现。


## 场景与建议

| 场景 | 推荐选型 | 原因 |
|---|---|---|
| 订单、财务账本 | RDBMS | 跨行关系密集，需要强一致事务和约束 |
| 用户 Profile | KV / Document | 按 `user_id` 整体读取，结构经常变化 |
| Feed Timeline | Sorted KV | 按 viewer_id 分区，按时间线排序 |
| 日志与事件 | Log system | 高吞吐 append，保留时间，不是 OLTP |

---


## 2 · Transaction 与并发控制

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

如果这些条件必须跨多个 entity 原子成立，关系数据库或 distributed SQL 更省心。
保证不变量通常需要并发控制：

- **Pessimistic lock**：写时阻塞读写，防止冲突，但容易产生排队和死锁。
- **Optimistic lock**：依靠版本号 (version) 允许并发，提交时验证。冲突少时效率高。
- **Unique constraint**：通过插入唯一键阻止重复，是最实用的幂等工具。键怎么选、进行中的第二发怎么办，见 [[SystemDesign01 Stateless Service|01]]。

当业务跨越多个独立节点时：

- **2PC (Two-Phase Commit)**：依赖 coordinator 和 participants 提供跨节点事务。缺点是如果 coordinator 在 prepare 阶段后死亡，participants 会阻塞。绝不应跨越不信任的网络或将 HTTP-to-Stripe 放入 2PC。
- **Saga / Compensation**：当工作流跨越不同系统，无法共享数据库事务时，通过执行反向补偿操作回滚已提交的步骤。


## Database transaction 和 business workflow 分离

单库 transaction 通常在毫秒内结束。跨 payment、inventory 的流程可能持续数分钟。

```text
local transaction
  -> write order + outbox
  -> async payment command
  -> state transition
  -> compensation when needed
```

这类流程靠 state machine 和 outbox，指向 [[SystemDesign06 Async Messaging Systems|06 异步消息系统]]。

---


## 3 · Consistency per API

一致性必须绑定到具体操作。[[SystemDesign01 Stateless Service|01]] 的幂等保住的是不变量在重试下仍成立。下面这轴是：写成功之后，哪一次读看得到。

```text
User updates profile to v2
User immediately reads profile
```

可能的 contract：

- **Linearizable read**：像只有一个最新副本；
- **Read-your-writes**：该用户至少能读到自己的 v2；
- **Monotonic read**：已经看到 v2 后不会退回 v1；
- **Eventual consistency**：没有新写入时，副本最终收敛。

同一个系统可以混用。订单确认页读 primary，公开商品页读 replica。

---

## 4 · Scalability Cube：数据库扩展的三维模型 (The AKF Scale Cube)

数据库单机架构终将遭遇物理资源的硬性天花板（CPU 核心数、内存带宽、磁盘 IOPS 与网络吞吐）。AKF 可扩展性立方体（AKF Scale Cube）将系统的横向扩展能力抽象为正交的三维几何空间 $(X, Y, Z)$：

```text
                  Y 轴：功能分区 / 业务解耦 (Functional Decomposition)
                  ▲  [按业务领域/微服务垂直拆库，解耦组织与爆炸半径]
                  │
                  │  / Z 轴：数据分区 / 水平分片 (Data Partitioning / Sharding)
                  │ /   [按 Shard Key 水平切分子集，突破单机写入与存储上限]
                  │/
  ────────────────┼────────────────────────► X 轴：添加副本 / 水平克隆 (Horizontal Duplication)
                 /│                             [单写多读/全量克隆，线性扩展读取 QPS 与高可用]
                / │
```

任何单一维度的扩展最终都会因物理或逻辑限制而遭遇瓶颈。超大规模工业级系统依赖于三轴的正交协同演进。

```database-scaling-visual
```

---

### 4.1 · X 轴扩展：添加副本与主从 / 主主复制

X 轴扩展是指在多个节点上运行完全相同的系统副本，每个数据库实例保存 100% 的全量数据镜像。X 轴的工程落地主要划分为两种截然不同的架构范式：**主从复制（Primary-Replica）** 与 **主主复制 / 多主多活（Multi-Primary / Active-Active）**。

#### 4.1.1 · 主从复制 (Primary-Replica Replication)：机制、拓扑与权衡

主从复制是工业界最成熟、应用最广泛的高可用与读扩展方案。写请求收敛于单一 Primary，读请求分散至多个 Replica。

##### 1. 核心机制与三种数据同步协议
所有修改状态的操作（`INSERT`、`UPDATE`、`DELETE`）强制定向到单点主库（Primary）。主库在本地事务提交时将变更写入预写日志（WAL / Binlog），副本节点（Replica）通过网络拉取日志并在本地重放（Relay Log Replay），以此追平主库状态。

```text
Client Write ──► Primary (Execute & Append Binlog)
                    │
                    ├─► [Network Stream] ──► Replica 1 (Relay Log -> Replay) ──► Client Read
                    └─► [Network Stream] ──► Replica 2 (Relay Log -> Replay) ──► Client Read
```

依据对数据持久性与响应延迟的不同权衡，底层分为三种同步模式：
- **异步复制（Asynchronous Replication）**：
  - *机制*：主库将事务写入本地 Binlog 并落盘后，立即向客户端返回提交成功，不等待任何副本的确认；副本异步建立长连接拉取日志。
  - *适用场景*：读多写少、写延迟极其敏感（要求 $< 5\text{ms}$）、允许突发停机时丢失少量数据的边缘业务。
  - *Tradeoff*：吞吐量最高、写入延迟最低；但若主库突发宕机，未同步到副本的日志将永久丢失（$\text{RPO} > 0$），且从库存在显著的复制延迟（Replication Lag，破坏 Read-your-writes 一致性）。
- **半同步复制（Semi-Synchronous Replication / Lossless Semi-Sync）**：
  - *机制*：主库在本地完成事务引擎提交前，挂起等待至少一个 Replica 确认收到日志并写入中继日志（Relay Log ACK）后，才向客户端返回提交成功。
  - *适用场景*：电商交易、核心结算等严禁数据丢失但能承受微小网络 RTT 延迟的 OLTP 系统。
  - *Tradeoff*：确保至少一个从库拥有与主库完全一致的 WAL 日志，主库宕机时能实现零数据丢失故障切换（$\text{RPO} \approx 0$）；但写入延迟增加了一个网络往返时间（RTT）；若所有副本网络超时，主库会自动降级为异步模式。
- **强同步/共识法定人数复制（Synchronous Quorum via Raft / Multi-Paxos）**：
  - *机制*：写入必须在预写日志层获得多数派节点（Quorum $W > N/2$）的持久化落盘确认，事务方可提交（如 TiDB / Spanner 底层的 Raft/Paxos Groups）。
  - *适用场景*：金融级核心账本、元数据协调服务（etcd, ZooKeeper）、分布式 NewSQL 存储。
  - *Tradeoff*：提供强一致性与自动化 Leader 选举切换（$\text{RPO} = 0, \text{RTO} < 5\text{s}$）；但写入尾延迟受最慢多数派节点制约，节点数必须为奇数（3 或 5 节点），整体写吞吐受网络共识协商限制。

##### 2. 适用场景与收益
- **高读写比的读密集型业务（Read-Heavy，读写比通常 $\ge 10:1$ 至 $1000:1$）**：如商品浏览、社交信息流，通过水平添加只读副本分担查询流量；
- **高可用容灾与自动故障转移（High Availability Failover）**：主库故障时，共识选举出日志最新的副本提升为新 Primary（必须配合 Fencing 隔离旧节点，详见第 5 节）；
- **报表分析与离线计算隔离**：将重度聚合分析（OLAP）、离线 ETL 或全量逻辑备份路由至独立专用只读副本，防止长事务锁表或争抢主库 Buffer Pool。

##### 3. 主从复制的核心局限与代价 (Tradeoffs)
- **零写入扩展能力（Zero Write Scalability）**：所有写操作仍汇聚至单一 Primary；副本越多，主库向所有副本分发 Binlog 的网络出口与 CPU 广播开销反而更大；
- **存储成本随副本数线性膨胀**：每个副本均需保存 100% 全量数据，总存储成本为 $O(N \times \text{Storage})$，无法解决单库海量数据存储超标问题；
- **复制延迟（Replication Lag）与读写时序断裂**：异步复制下产生过期读（Stale Read），用户更新后立即刷新页面可能看到旧数据；
- **连接池与 Buffer Pool 缓存稀释**：客户端连接池随副本数成倍放大（Little's Law：$\text{Concurrency} \approx \text{QPS} \times \text{Latency}$），读请求打散降低了单机内存 Buffer Pool 的局部性与命中率。

---

#### 4.1.2 · 主主复制 / 多主多活 (Multi-Primary / Active-Active Replication)：并发写冲突与权衡

主主复制（Multi-Primary / 双主双活）允许集群中存在多个节点同时接收写请求，各节点之间双向同步变更。

##### 1. 核心挑战：并发写写冲突 (Write-Write Conflict)
在单主架构中，锁机制在单点串行化了所有写操作；而在主主架构中，客户端 A 在节点 1 修改记录 $X$ 的同时，客户端 B 在节点 2 也修改记录 $X$。由于光速限制与网络延迟，两节点在产生冲突写入时毫不知情，直接导致数据分叉。

##### 2. 主流冲突解决与仲裁方法
- **方法一：避免冲突 —— 单写所有权与单元化架构 (Single-Writer Ownership / Cell-Based Architecture) [生产级推荐首选]**：
  - *机制*：在接入网关层依据特定维度（如用户归属地、`user_id` 取模），将每条数据的主写所有权唯一绑定到某一个特定 Primary。例如：中国区用户的所有写请求固定路由至上海主库，美洲用户路由至美东主库。两地主库之间只进行异步跨地域单向读取同步。
  - *优点*：**从源头上消除了并发写冲突**，完全不需要跨地域分布式锁；
  - *代价*：跨地域漫游用户写入需要跨洋 RPC 转发到其归属地主库；若某地域发生灾难，所有权切换交接协议极其复杂。
- **方法二：最后写入胜出 (Last-Write-Wins / LWW via NTP or HLC)**：
  - *机制*：每个写操作携带物理时间戳或混合逻辑时钟（Hybrid Logical Clock），冲突时以时间戳最大者覆盖旧数据。
  - *致命缺陷*：跨节点物理时钟存在漂移（Clock Skew），极易发生**有效业务写入被静默覆盖丢弃（Silent Data Loss）**，绝对禁止在金融与交易账本中使用。
- **方法三：无冲突复制数据类型 (CRDTs - Conflict-free Replicated Data Types)**：
  - *机制*：利用半格代数特性（满足交换律、结合律与幂等性），使数据包无论以何种顺序到达各节点，最终状态自然收敛一致。如基于增量的计数器（PN-Counter）、只增集合（OR-Set）、协同文档编辑。
  - *优点*：零协调开销，在完全网络分区下仍可无锁并发写入；
  - *代价*：只适用于特定代数数据结构，无法表达复杂的业务外键约束、行级悲观锁或关系型跨表事务。
- **方法四：应用层冲突合并与向量时钟 (Vector Clock & Application-Level Merge)**：
  - *机制*：存储层通过向量时钟捕获并发因果关系。检测到无法确定先后的并发分支时，同时保留两个冲突版本（Siblings），交由应用层在下一次读取时由业务逻辑手工合并（如购物车合并商品列表）。
  - *代价*：应用层代码复杂度剧烈膨胀；长时间未解决的冲突版本会耗尽存储与网络带宽。

##### 3. 主主复制的工程 Tradeoff 总结

| 评估维度 | 主从复制 (Primary-Replica) | 主主复制 (Multi-Primary Active-Active) |
|---|---|---|
| **写入入口** | 严格单一写入点（Single-Writer） | 多个并发写入点（Multi-Writer） |
| **写并发冲突** | 无冲突（由主库本地行锁串行化） | **必然存在冲突**，需依赖单写归属、LWW 或 CRDT 仲裁 |
| **事务强一致性** | 原生单机 ACID 事务保证 | **无法保障跨节点事务**；极易发生写写冲突与脏读 |
| **就近写入延迟** | 异地写必须承受跨地域长 RTT | **极佳**：就近写入本地机房，毫秒级快速返回 |
| **自增主键处理** | 简单单调自增（Auto-Increment） | 必须配置不同步长/偏移量（如 Node 1: 1,3,5; Node 2: 2,4,6）或 UUID |
| **数据同步机制** | 单向复制，拓扑简单 | **双向环形复制**；必须通过 `server-id` 拦截回环死循环 |
| **生产适用性** | 适合 95% 以上的标准 OLTP 业务 | 仅适合全球多活（结合单元化单写）或特定边缘协同系统 |

---

### 4.2 · Y 轴扩展：功能分区与业务解耦 (Functional Decomposition / Vertical Splitting)

Y 轴扩展是指沿着业务功能职责、领域限界上下文（Bounded Context）将单一庞大的巨石数据库（Monolithic Database）进行垂直拆分，形成各自独立的业务领域数据库。

#### 1. 核心机制
依据领域驱动设计（DDD），将原本共享同一数据库实例和连接池的单体大库，拆分为物理上互不干扰的专用数据库：
$$\text{Monolithic DB} \longrightarrow \text{User DB} \oplus \text{Order DB} \oplus \text{Inventory DB} \oplus \text{Payment DB}$$
每个功能数据库由对应的微服务独立拥有（Database-per-Service 模式）。禁止任何跨服务的直接跨库 SQL 访问，所有交互必须通过抽象良好的 RPC、REST API 或异步事件总线（Event-Driven Architecture）进行。

#### 2. 适用场景
- **业务读写负载特征与 SLA 诉求极度分化**：例如，商品目录库（Catalog DB）是高并发只读，适合大量二级缓存与全文索引；而支付结算库（Payment DB）数据量适中但要求严格的行级悲观锁与 ACID 强一致；日志审计库则为高吞吐只追加写。垂直拆分允许为各业务量身定制数据库引擎与参数；
- **研发组织规模扩大与康威定律（Conway's Law）**：当开发团队扩张至数百人，多团队同时对单体库执行 DDL 模式变更（Schema Migration）会导致发布锁表、部署冲突和职责不清。按业务域拆库后，各团队实现独立迭代、独立升级与独立扩展；
- **故障爆炸半径最小化（Blast Radius Containment）**：非核心功能（如用户评论、积分商场、运营打卡活动）发生慢查询、死锁或连接池耗尽时，故障严格局限在自身功能库内，绝不影响核心下单、支付与认证链路。

#### 3. Tradeoff 与工程代价
- **丧失单机 ACID 本地强事务**：跨业务域的状态变更（如“用户下单 $\to$ 扣减库存 $\to$ 扣减账户余额”）无法再通过单一 `BEGIN ... COMMIT` 保障原子性。必须引入分布式事务解决方案：
  - 两阶段提交（2PC / XA）：强一致但网络往返多、锁持有时间长、性能极差且协调者存在单点阻塞风险；
  - 最终一致性（Saga 模式 / 本地消息表 + Transactional Outbox）：通过状态机推进正向流程，失败时执行反向补偿操作。系统的研发、测试与对账复杂度急剧升高；
- **丧失关系型 SQL JOIN 与全局外键约束**：无法跨库执行关联查询。
  - *应对策略*：在应用层多次请求汇聚组装（Application-Side Join）；或在数据写入时引入反范式冗余字段；或通过 CDC（如 Debezium）将各库数据流式汇总至读端物化视图（CQRS / Elasticsearch / ClickHouse）；
- **基础设施碎片化与运维复杂度激增**：数据库实例、参数调优、备份恢复（RPO/RTO 指标管理）、监控报警与连接池配置的数量呈倍数增长；
- **单模块物理瓶颈不可避免**：如果某一单一功能模块（如电商的订单库，或社交平台的消息库）自身的数据量达到百亿级、写入 QPS 突破数万，仅靠 Y 轴垂直拆分已无能为力，必须进一步结合 Z 轴。

---

### 4.3 · Z 轴扩展：数据分区与水平分片 (Data Partitioning / Horizontal Sharding)

Z 轴扩展是指保持数据模式（Schema）完全相同，依据特定的数据属性（分片键，Shard Key / Partition Key），通过确定性的路由算法将同构数据集水平切分为 $K$ 个独立的物理分区（Shard $0 \dots K-1$）。

#### 1. 核心机制
每个 Shard 仅持有全量数据的 $1/K$，并独立承载约 $1/K$ 的读写吞吐。请求通过路由代理层（Routing Proxy，如 Vitess、ShardingSphere 或分布式数据库存储网关）解析 SQL、提取分片键并精确定位物理节点：

```text
                               ┌──► Shard 0 (Hold keys: hash(key) % 4 == 0)
Client Query ──► Query Router ──┼──► Shard 1 (Hold keys: hash(key) % 4 == 1)
   (with Shard Key)            ├──► Shard 2 (Hold keys: hash(key) % 4 == 2)
                               └──► Shard 3 (Hold keys: hash(key) % 4 == 3)
```

- **分片路由策略**：
  - **哈希分片（Hash-Based）**：通过 `hash(shard_key) % N` 或一致性哈希环分布数据。分布最均匀，但丢失范围查询能力；
  - **范围分片（Range-Based）**：按时间（如 `created_at`）或数值区间（如 `id: [1, 10000000]`）切片。利于范围扫描，但新数据的追加写入极易集中在最新区间，引发热点写入倾斜；
  - **目录映射（Directory / Lookup Table）**：维护集中式路由元数据映射表。极其灵活，但元数据服务本身成为新的网络跳步与可用性单点。

#### 2. 适用场景
- **写入 QPS 突破单节点硬件极限**：当写入吞吐达到数万至数十万 QPS，单主库的 CPU、写锁争用和磁盘 IOPS 彻底耗尽时，Z 轴是水平线性扩展写吞吐的唯一解法；
- **数据存储总量突破单机物理容量与索引深度**：单表数据量突破千万至数亿行，B+ 树索引层级从 3 层加深至 4~5 层，单次查询引发额外随机磁盘 I/O；单机 SSD 无法承受 PB 级数据，且全量数据备份/恢复窗口（RTO）超出业务容忍极限；
- **多租户隔离（Multi-Tenant SaaS）**：按租户 ID（`tenant_id`）分区，大客户独占单分片避免“喧闹邻居（Noisy Neighbor）”，小客户共享分片池；
- **地理数据主权与物理网络延迟优化（Data Sovereignty & Geo-Routing）**：按用户地理区域（`region_id`）分区，欧盟用户数据强制物理驻留在法兰克福机房以符合 GDPR 合规要求，同时实现用户就近读写降低光纤 RTT。

#### 3. Tradeoff 与工程代价
- **分片键硬绑定与查询维度坍缩（Shard Key Lock-in）**：
  - **点查（Point Query）高效**：携带分片键的查询（`WHERE user_id = 1024`）可实现 $O(1)$ 精确单片寻址；
  - **无分片键查询退化为全分片广播（Scatter-Gather）**：若按非分片键查询（如 `WHERE phone_number = ?` 或商家查全平台订单 `WHERE merchant_id = ?`），路由代理必须向全量 $K$ 个 Shard 并发广播请求，并在内存中做归并排序（Merge Sort / Limit / Offset）。整体响应延迟被最慢的一个长尾 Shard 决定（Tail Latency Amplification）；
- **跨分片分布式事务（Cross-Shard 2PC）吞吐悬崖**：跨多个 Shard 的原子写操作需要通过两阶段提交协议协调。网络往返翻倍、行锁在事务执行期间跨节点锁定，导致数据库吞吐量相比单分片写入发生断崖式下跌（通常下降 1~2 个数量级）；
- **数据倾斜与热点分片（Data Skew & Hotspots）**：真实世界数据并非均匀分布。社交网络中的超级网红、电商大促中的头部爆品，会导致对应分片成为系统的整体吞吐瓶颈；
- **再平衡与动态扩容代价（Resharding / Rebalancing Overhead）**：当分片集群从 $N$ 个扩展至 $2N$ 个时，需要通过一致性哈希或按段迁移庞大的历史数据。涉及跨网络数据搬迁、双写比对校验、路由无缝切换，工程复杂度极高。

---

### 4.4 · 三轴协同架构 ($X \times Y \times Z$ Synergy) 与全维度决策阵列

#### 1. 工业级真实落地形态
在大型工业级分布式系统架构中，扩展从来不是单选，而是沿着 $X, Y, Z$ 三轴正交结合的立体拓扑：

```text
                       [ Ingress Traffic / API Gateway ]
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │  Y 轴：功能分区 (Microservice / Bounded Context)      │
            ▼                                                     ▼
     [ User Service ]                                     [ Order Service ]
            │                                                     │
            │ (单库容量适中)                                        ▼ (超大规模吞吐)
            │                                     ┌───────────────────────────────┐
            │                                     │ Z 轴：数据分区 (user_id % 2)   │
            │                                     └───────┬───────────────┬───────┘
            │                                             │               │
            │                                   Shard 0   ▼               ▼   Shard 1
            ▼                                  ┌────────────────┐ ┌────────────────┐
  ┌──────────────────┐                         │ X 轴：主从高可用│ │ X 轴：主从高可用│
  │ X 轴：主从读写分离│                         │ Primary (写)   │ │ Primary (写)   │
  │ Primary (写)     │                         │ ├─ Replica (读)│ │ ├─ Replica (读)│
  │ └─ Replica (读)  │                         │ └─ Replica (读)│ │ └─ Replica (读)│
  └──────────────────┘                         └────────────────┘ └────────────────┘
```

1. **第一步（Y 轴）**：先按业务领域垂直拆分单体架构，将订单、用户、库存、支付拆为独立数据库，隔离团队协作与故障爆炸半径；
2. **第二步（Z 轴）**：针对少数核心超大容量与高频写入域（如订单系统），沿分片键水平切分为物理 Shards，打破单机写 IOPS 与磁盘容量上限；
3. **第三步（X 轴）**：对每一个 Shard 内部，配置一个 Primary 与多个只读 Replica，利用读写分离承载高并发读取，并提供自动故障转移（Failover）。

#### 2. 全维度权衡与选型决策矩阵

| 评估维度 | X 轴：添加副本 (Replication) | Y 轴：功能分区 (Decomposition) | Z 轴：数据分区 (Sharding) |
|---|---|---|---|
| **物理本质** | 全量克隆（Clone 100% 数据） | 垂直拆分（按业务边界拆分表/库） | 水平切片（按分片键切分同构数据） |
| **解决的核心瓶颈** | 读 QPS 瓶颈、单点故障（SLA 可用性） | 团队开发冲突、多业务负载特征分化、故障爆炸半径 | 单机写入 IOPS 极限、单表超大行数与磁盘容量上限 |
| **写入扩展能力** | **无法扩展**（单点主写或写入受最慢节点制约） | **间接分流**（不同业务写入打入不同实例） | **线性扩展**（写入打散至 $K$ 个独立分片） |
| **存储容量扩展能力**| **无法扩展**（每个节点存储 100% 全量数据） | **按业务隔离**（单个业务内仍受单机容量限制） | **线性扩展**（每个分片仅存储 $1/K$ 数据） |
| **对事务模型的影响** | 保持单机 ACID；但异步复制下存在复制延迟（破坏读写一致性） | **丧失单机事务**；需降级为分布式最终一致性（Saga / Outbox） | **跨分片事务开销剧增**；单分片内保 ACID，跨分片需 2PC |
| **对查询模型的影响** | 无影响；支持完整 SQL、JOIN 与二级索引 | **丧失跨库 JOIN**；需应用层聚合或构建 CQRS 物化视图 | **分片键点查极快**；无分片键查询退化为全分片广播（Scatter-Gather） |
| **主要工程代价** | 数据同步延迟、过期读、Buffer Pool 缓存稀释 | 分布式事务复杂度、对账系统建设、实例碎片化运维 | 分片键绑定、长尾延迟放大、数据倾斜、动态扩容再平衡 |
| **引入时机** | 读写比高（$\ge 10:1$）、需要多机容灾兜底时首先引入 | 团队扩大、单体库耦合严重、不同模块业务特征分化时引入 | 单机硬件写饱和、单表破千万/亿级且索引失效时引入 |

---

### 4.5 · 数据依赖性（Data Dependency）驱动的架构选型体系

在分布式存储演进中，**数据的内在依赖性（Data Dependency）与不变量范围（Invariant Scope）直接决定了架构扩展的物理边界**。离开依赖性谈分片属于无源之水：

#### 1. 强事务与参照完整性依赖 (Strong Transactional Invariants)
- **依赖特征**：多个实体之间存在强数学约束或不变量（如银行转账的资金借贷守恒 $\Delta A + \Delta B = 0$、外键级联约束、单行库存不能扣减为负）。此类依赖要求物理层面的**强原子性（Atomicity）与强隔离性（Snapshot Isolation / Serializable）**。
- **匹配架构解法**：
  - 首选单机高规格 RDBMS 或原生分布式事务数据库（Google Spanner、TiDB、CockroachDB）；
  - 若在微服务体系下拆库（Y 轴），必须将强事务降级为**分布式最终一致性**：采用本地事务写业务表同时记录本地事务发件箱（Transactional Outbox），后台异步轮询投递至消息队列，结合 **Saga 状态机与冲正补偿（Compensating Transactions）**。
- **绝对规避的陷阱**：在未解耦强依赖前，直接按 `user_id` 水平分片（Z 轴）。转账操作若发生在不同 Shard 的用户之间，将被迫在每笔交易中执行跨分片两阶段提交（2PC），导致行锁长时间跨网络锁定，吞吐暴跌至单机 5% 以下。

#### 2. 实体亲和聚合依赖 (Entity Colocation Affinity)
- **依赖特征**：子实体的生命周期完全隶属于父实体（DDD 聚合根），且业务查询中 95% 以上都是同时加载或修改父子数据。例如：订单主表 `orders` 与订单明细 `order_items`、用户主信息 `users` 与用户偏好 `user_preferences`。
- **匹配架构解法**：
  - **共享分片键（Colocated Sharding / Affinity Routing）**：子实体必须显式冗余父实体的分片键（如 `order_items` 表虽然有 `item_id`，但强制增加 `order_id` 并以此作为分片键）。路由代理将属于同一订单的所有项严格路由至同一物理 Shard；
  - **文档级嵌套（Document Store）**：在 MongoDB 中直接将 items 作为子数组内嵌在 order 根文档中，物理落盘于同一磁盘连续块。
- **核心收益**：同一聚合根内的读写、更新与联表查询全部在单 Shard 内完成，**保留单片原生 ACID 事务与单机物理 JOIN**，彻底消灭跨网络分布式事务开销。

#### 3. 弱依赖与派生分析流 (Weak / Analytics / Eventual Consistency)
- **依赖特征**：高频写入与读统计，但业务主链路对统计精度的瞬时绝对一致性不敏感，允许秒级数据延迟与最终收敛。例如：短链的访问计数、点赞数、商品全网浏览量、全站热搜排行。
- **匹配架构解法**：
  - **读写物理彻底解耦（CQRS 架构）**：主事务路径只负责状态更新或仅异步向 Kafka 投递一条事件消息，立即向客户端返回成功（200/202/302）；
  - 下游消费端采用**滑动微批（Micro-Batching）**方式聚合计数，批量刷入只读物化视图（Redis 缓存或 ClickHouse OLAP 分析数仓）。
- **绝对规避的陷阱**：在下单或重定向的主事务链路上，同步执行 `UPDATE items SET view_count = view_count + 1`，行级悲观锁与磁盘写入争用将直接压垮核心交易主库。

#### 4. 无状态 / 静态配置独立依赖 (Stateless / Static Configuration)
- **依赖特征**：只读不写，全站所有业务微服务与所有分片高频访问。例如：国家行政区划字典、汇率基准表、平台全局费率规则。
- **匹配架构解法**：
  - **全分片广播复制（Broadcast Replication）**：在每个物理 Shard 库中均保留一份完整的静态字典表副本；
  - **客户端本地进程内缓存（In-Memory L1 Cache）**：服务节点启动时全量加载至本地内存（如 Caffeine / Guava Cache），字典更新时通过广播事件刷新。

---

### 4.6 · 分片键（Sharding Key）设计的 6 大核心考量因子与生产级避坑指南

分片键（Sharding Key）是水平分片架构的定海神针。分片键选错往往意味着需要对数十亿条线上历史数据进行毁灭性的重分片迁移。设计分片键必须严格遵循以下 6 大核心因子：

#### 1. 高基数（Cardinality）—— 取值离散空间
- **法则**：分片键的可能取值数量（基数）必须远大于分片节点数（例如数千万至数亿级取值，如 `user_id`, `UUID`）。
- **避坑**：绝对禁止使用状态字段（如 `order_status: [0, 1, 2]`）或低基数枚举（如 `gender`, `country_code`）作为分片键。分片算法会将数千万条数据死死绑定在少数几个分片上，引发灾难性**数据倾斜（Data Skew）**，其余分片空转。

#### 2. 散列均匀度与热点规避（Uniformity & Hotspot Avoidance）
- **法则**：数据写入必须在所有分片上呈现高度伪随机均匀分布。
- **避坑（时间戳与自增主键陷阱）**：
  - 严禁按纯自增 ID 或时间戳（`created_at`）进行范围分片。当前分钟或当前秒的所有新增写入都会100%集中打向最新区间所在的单个 Shard，造成单节点磁盘 IOPS 和 CPU 彻底打满（**Write Hotspot**），而历史分片写入为零；
  - 工业级做法：采用高质量哈希散列（如 `MurmurHash3(key) % N`）。对于出现“头部超级大 V”或“爆款商品”的场景，在分片键后加入随机加盐后缀（`key + "_" + random(0, M)`）将单点流量打散至多个影子节点。

#### 3. 核心业务查询覆盖率（Query Filter Alignment）
- **法则**：**系统中超过 85%~90% 的高频核心 SQL 必须在 `WHERE` 条件中显式包含分片键**。
- **避坑（全分片广播 Scatter-Gather 性能陷阱）**：
  - 查询带分片键时，路由网关执行精确单片定向（Point Query，$O(1)$ 复杂度）；
  - 查询若缺少分片键，路由代理必须向全集群 $K$ 个 Shard 发起并发网络请求（**Scatter-Gather**），并在网关内存中将各 Shard 返回的子集执行归并排序、去重和翻页截断。响应时间取决于最慢的一个长尾节点（Tail Latency 恶化），极易导致网关内存溢出（OOM）与各分片连接池耗尽。

#### 4. 实体亲和共存度（Colocation & Transaction Affinity）
- **法则**：强关联的父子实体必须共享相同分片键。
- **工程实践**：`orders` 与 `order_items`、`users` 与 `user_addresses` 均以主实体 ID（`user_id` 或 `order_id`）为分片键。数据强行调度在同一机器实例上，使得两表关联的更新能天然复用单库行锁和单机事务日志，**彻底杜绝跨网络两阶段提交（2PC）**。

#### 5. 生命周期不可变性（Immutability）
- **法则**：**分片键一旦生成落盘，必须在数据生命周期内永久不可变**。
- **避坑**：绝不能用可能发生变更的业务字段（如 `phone_number`, `email`, `department_id`）作为分片键。修改分片键在底层意味着：跨网络分布式事务执行“原 Shard 物理删除 + 目标 Shard 物理插入”，在并发场景下极易引发死锁、分布式事务超时与数据静默丢失。

#### 6. 多维度查询支持机制（Multi-Dimensional Query Strategies）
现实业务往往不可避免地存在相互冲突的双向查询诉求（例如：买家查自己订单，卖家查店铺订单）。此时单一分片键无法兼顾，必须采用以下工业级设计模式：

```text
                               [ 客户端写订单请求 ]
                                        │
                                        ▼
                   ┌────────────────────────────────────────┐
                   │ 买家主库集群 (Shard by hash(buyer_id))    │
                   │ (order_id 尾部嵌入 buyer_id 基因哈希)    │
                   └────────────────────┬───────────────────┘
                                        │ (MySQL Binlog / WAL)
                                        ▼
                             [ CDC 实时同步 (Debezium) ]
                                        │
            ┌───────────────────────────┴───────────────────────────┐
            ▼                                                       ▼
┌──────────────────────────────────────┐ ┌──────────────────────────────────────┐
│ 卖家异构只读分片库 (Shard by seller_id) │ │ Elasticsearch / ClickHouse 全文与分析库│
│ (承载卖家端多条件复杂订单查询)        │ │ (承载运营后台多维筛选与大宽表报表)    │
└──────────────────────────────────────┘ └──────────────────────────────────────┘
```

- **模式 A：基因分片法 (Gene Sharding)**：
  - 生成 `order_id` 时，提取 `hash(buyer_id)` 的末尾 4~6 位二进制作为 `order_id` 的固定后缀；
  - 买家查询携带 `buyer_id`，直达对应 Shard；根据 `order_id` 查询时，提取其末尾几位哈希，即可精确推导其所在 Shard，实现**无需广播的双键单片直达**。
- **模式 B：CDC 驱动异构读模型 (CQRS Shadow Table)**：
  - 主交易库严格按核心买家维度切片，承载高并发 OLTP 写入；
  - 依赖变更数据捕获（CDC）异步流式抽取 Binlog，自动同步至按 `seller_id` 分片的异构只读库，或直接导入 Elasticsearch 搜索引擎，彻底实现读写分离与维度解耦。

---

### 4.7 · 分片后的查询路由：Scatter-Gather 的物理本质与 4 大替代优化方案

#### 1. 核心判定：分片之后是不是必须要用 Scatter-Gather？
**答案是否定的：分片之后绝不是必须要用 Scatter-Gather！**

在良好的水平分片设计中，**绝大多数常规高频请求都应该完全避免 Scatter-Gather**：
- **单片直达点查 (Point Query，理想状态)**：只要查询条件中包含分片键（如 `WHERE user_id = 1024`），路由代理层通过对分片键进行哈希或范围计算，可以直接将请求精准定向到某一个唯一的 Shard 节点。查询时间复杂度为 $O(1)$，延迟与单机数据库完全相同。
- **Scatter-Gather 的触发条件**：**当且仅当**查询条件中**完全缺少分片键**（如按手机号查用户 `WHERE phone_number = ?`、商家按自身 ID 查全站订单 `WHERE seller_id = ?`），或者需要执行跨全集群的**全局聚合计算**（如 `COUNT(*)`、全局大盘统计）与**全局深度分页**（`ORDER BY created_at LIMIT 10000, 20`）且没有任何辅助索引时，系统才会被迫退化为 Scatter-Gather。

```text
【单片精准点查 (Point Query)】               【全分片广播 (Scatter-Gather)】
Client (with Shard Key: user_id=42)         Client (Missing Shard Key: phone='138...')
          │                                           │
          ▼                                           ▼
    Query Router                                Query Router (Fan-out to ALL Shards)
          │                                      ┌────┼────┬────┐
          │ (Direct O(1) Route)                  ▼    ▼    ▼    ▼
          ▼                                    [S0] [S1] [S2] [S3]
       Shard 2                                   └────┴────┼────┘
          │                                                ▼
          ▼                                         Merge Sort & Limit
     Fast Result                                   (Tail Latency Risk!)
```

#### 2. 为什么 Scatter-Gather 是分布式系统的性能杀手？(Tradeoff 剖析)
1. **尾延迟严重放大 (Tail Latency Amplification)**：
   - 广播查询发向全部 $K$ 个 Shard，整体响应耗时由**最慢的那一个节点（Straggler）**决定：
     $$T_{\text{total}} = \max(T_0, T_1, \dots, T_{K-1})$$
   - 假设单台数据库节点出现 GC 停顿或磁盘抖动的概率仅为 $p = 1\%$。当系统分片数 $K = 64$ 时，客户端请求遇到抖动的概率飙升至：
     $$P(\text{Tail Jitter}) = 1 - (1 - p)^K = 1 - (1 - 0.01)^{64} \approx 47.44\%$$
     近一半的广播请求会遭遇长尾阻塞！
2. **路由网关内存爆炸与网络风暴 (Deep Pagination OOM)**：
   - 若执行 `SELECT * FROM orders ORDER BY create_time LIMIT 10000, 20`，路由代理不能只向每个分片拉取 20 条，而必须命令所有 $K$ 个分片在本地分别取出前 10,020 条数据；
   - 代理层网络和内存中必须接收 $K \times 10,020$ 条完整数据，并在内存中执行全局多路归并排序后丢弃前 10,000 条。高并发下网关极易直接发生 Full GC 或 OOM 崩溃。
3. **连接池资源瞬时枯竭**：
   - 一次查询瞬间霸占集群中每一个 Shard 的数据库连接，整个数据库集群的并发吞吐能力直接被折损为原本的 $1/K$。

---

#### 3. 替代与规避 Scatter-Gather 的 4 大生产级解决方案

为了彻底消灭或降级 Scatter-Gather，工业界演化出了以下 4 大核心架构解法：

```text
                           [ 非分片键查询请求 ]
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   [ 方案 1: 全局二级索引 GSI ]  [ 方案 2: 基因分片算法 ]  [ 方案 3: CQRS 异构读模型 ]
   按 phone_number 分片查出      order_id 尾部嵌入基因      CDC 流式写入 Elasticsearch
   对应的 user_id，再单片点查    提取后直接定位单 Shard    或商家维度影子表，多维检索
```

##### 方案一：全局二级索引表 (Global Secondary Index / GSI & Reverse Lookup Table)
- **实现机制**：
  - 单独构建一张轻量级的全局二级索引表。索引表本身也是分片的，但以业务要查的非主键字段（如 `phone_number` 或 `email`）作为分片键，Value 只保存对应实体的主分片键（如 `user_id`）；
  - **查询两步走（两次单片点查）**：
    1. 客户端查 GSI 表：`SELECT user_id FROM phone_user_index WHERE phone = '138...'` $\to$ 路由到 GSI 分片，耗时 $< 2\text{ms}$ 返回 `user_id = 42`；
    2. 客户端查主表：`SELECT * FROM users WHERE user_id = 42` $\to$ 路由到用户主 Shard，耗时 $< 2\text{ms}$ 返回完整详情。
- **适用场景**：通过非分片键做高频单点等值检索（如手机号/邮箱登录、第三方交易流水号反查）。
- **Tradeoff**：
  - 增加了一次网络 RTT（可通过网关层缓存热点映射优化）；
  - 写入时产生了分布式双写（主表 + 索引表）。若追求强一致必须引入分布式事务，通常采用异步最终一致性，存在毫秒级的索引同步延迟。

##### 方案二：基因分片算法 (Gene Sharding / Embedded Hash Routing)
- **实现机制**：
  - 将第二维高频查询键的散列特征，在生成实体主键 ID 时直接嵌入为该 ID 的固定后缀；
  - 例如电商订单系统：取 `hash(buyer_id)` 的最后 6 位二进制（假设为 `101100`），强制将其作为全局唯一 `order_id` 生成算法的低 6 位二进制；
  - **双向无广播单片直达**：
    1. 按买家查订单：`WHERE buyer_id = ?`，按 `hash(buyer_id) % 64` 直达 Shard 44；
    2. 按订单号查订单：`WHERE order_id = ?`，提取 `order_id` 末尾 6 位（`101100` 即 44），**无需任何额外索引，直接反推出 Shard 44**！
- **适用场景**：主实体与子实体、或主外键强关联的双维度超高频单点直达查询。
- **Tradeoff**：
  - 对 ID 生成算法有强侵入性，且只能解决固定的两两关联维度；
  - 无法解决多字段动态组合筛选（如既按状态又按金额又按创建时间的复杂查询）。

##### 方案三：异构读模型与 CDC 异步流式物化 (CQRS / Shadow Shards / Elasticsearch)
- **实现机制**：
  - **写入与读取物理分离**：主交易库只按核心事务维度（如 `buyer_id`）切片，承载高并发 OLTP 写入；
  - 利用变更数据捕获（CDC，如 Debezium / Flink CDC）监听 MySQL Binlog，异步将数据流式同步到专为查询优化的异构读存储：
    - *卖家维度异构只读分片库*：重新按 `seller_id` 作为分片键构建一套关系型影子表，承载商家后台的订单高频查看；
    - *搜索引擎 (Elasticsearch)*：将数据打平为 JSON 文档存入 ES，利用倒排索引承载任意多字段模糊组合检索；
    - *列式分析数仓 (ClickHouse)*：承载跨分片的大宽表聚合与 BI 统计报表。
- **适用场景**：后台管理系统的多条件组合筛选、商家工作台报表、全量聚合查询。
- **Tradeoff**：
  - 读端存在毫秒至秒级的最终一致性延迟（Eventual Consistency Lag）；
  - 架构拓扑大幅复杂化，需要保障 CDC 数据链路的精确一次（Exactly-Once）与高可用运维。

##### 方案四：算子下推与流式增量预计算 (Pushdown Aggregation & Stream Precomputation)
- **实现机制**：
  - **算子下推 (Pushdown Aggregation)**：若不可避免要做跨分片聚合（如 `COUNT(*)`），路由网关绝不拉取明细，而是将聚合函数下推至各个 Shard 本地执行 `SELECT COUNT(*)`，网关仅收集 $K$ 个整型数值并做加和（Tree-based Hierarchical Merge）；
  - **流式增量预计算**：通过 Flink / Spark Streaming 在数据写入时实时计算全局统计值，维护在全局 Redis 计数器或 HyperLogLog 中，业务读取时直接 $O(1)$ 读缓存；
  - **游标深度分页 (Cursor-based Pagination)**：彻底废弃基于物理偏移量的 `OFFSET`，强制改用基于排序键的游标分页（`WHERE (create_time, id) < (?, ?) ORDER BY create_time DESC, id DESC LIMIT 20`），使各 Shard 本地只需执行精确点查与有限扫描。
- **适用场景**：全站排行榜、大盘数据看板、超大规模列表无限滚动加载。
- **Tradeoff**：
  - 丢失实时事务一致性；
  - 游标分页不支持用户随机跳页（无法直接跳转第 500 页），只支持下一页顺序滚动。

---

#### 4. 替代 Scatter-Gather 的 4 大架构方案全维度对比决策表

| 解决方案 | 核心机制 | 查询时间复杂度 | 数据一致性 | 适用场景 | 核心代价与权衡 (Tradeoffs) |
|---|---|---|---|---|---|
| **全局二级索引 (GSI)** | 独立分片索引表记录映射 | 2 次点查 $O(1) + O(1)$ | 最终一致性（或 2PC 强一致） | 单非分片键的高频单点等值检索（手机/邮箱登录） | 额外 1 次网络 RTT；写入双写放大；需维护反向索引表 |
| **基因分片法 (Gene Sharding)** | ID 后缀嵌入关联分片特征 | 1 次点查 $O(1)$ | 强一致（落盘于同一分片） | 订单号与买家 ID 双维度高频查询 | 侵入 ID 生成规则；仅能支持一对维度，无法泛化组合检索 |
| **CQRS 异构读模型 (CDC+ES)** | CDC 异步抽取 Binlog 构建读存储 | 多维倒排查询 $O(\log N)$ | 最终一致性（秒级延迟） | 商家多条件复杂筛选、全站模糊搜索、管理后台 | 读端短暂不可见；系统组件增多，CDC 流水线运维成本高 |
| **算子下推与流式预计算** | 聚合算子单片本地执行 + 游标分页 | 分层合并 $O(K)$ 或缓存 $O(1)$ | 统计近似或流式最终收敛 | 全站统计报表、实时排行榜、深度数据流分页 | 必须禁止 `OFFSET` 随机跳页；统计指标微弱滞后 |

---

## 5 · 容灾与高可用：Survive a Failure

冗余不是“多开几台机器”这么简单。高可用架构必须在网络分区、硬件故障与脑裂风险下维持系统的不变量。

### 5.1 · 故障域划分 (Failure Domains)
```text
process
  < machine
  < rack / power domain
  < availability zone (AZ)
  < region
```
系统设计前必须明确 SLA 目标：**系统需要活过哪一级物理故障？**
- 单机故障：同机架热备即可接管；
- 机房级断电/光缆切断：必须依赖跨可用区（Multi-AZ）复制；
- 区域级自然灾害：必须构建跨地域容灾（Multi-Region）。

### 5.2 · 故障转移与隔离栅 (Failover & Fencing)
将一个 Replica 安全提升为新 Primary 是分布式系统中最危险的操作之一：
```text
1. 故障探测器 (Failure Detector) 怀疑 Primary 不可用 (心跳超时 Timeout ≠ 真实死亡)
2. 达到连续失联判定阈值 (Lease Expiry)
3. 共识机制选出 Binlog / WAL 数据最新的 Replica
4. 对旧 Primary 执行强隔离栅 (Fencing / STONITH)
5. 提升新 Primary，递增全局 Epoch / Term 版本号
6. 更新路由层映射 (Router Configuration Cutover)
7. 恢复线上写流量
```

> [!IMPORTANT]
> **Fencing（隔离栅）是不变量的绝对底线**：旧 Primary 可能仅仅是遭遇了 Full GC 停顿或单向网络分区。若未对其执行强隔离（如基于 ZooKeeper/etcd 租约失效、存储层撤销写权限、或 IPMI 远程强制断电 STONITH），旧 Primary 恢复后继续接收写请求，将直接引发灾难性的**双主脑裂（Split-Brain）**，导致数据静默损坏与无法自动合并的分叉。

### 5.3 · 容灾部署拓扑
- **Active-Passive（主备模式）**：

| 备机模式 (Standby) | 平时状态 | 故障切换速度 (RTO) | 基础设施成本 |
|---|---|---|---|
| **Cold Standby (冷备)** | 仅保留定期全量快照与部署自动化脚本，实例未运行 | 小时级至天级 | 极低 |
| **Warm Standby (温备)** | 实例运行且持续同步日志，但规格较小或未预热缓存 | 分钟级至十分钟级 | 中等 |
| **Hot Standby (热备)** | 具备完整计算/存储规格，数据实时同步且连接池常驻 | 秒级自动切换 | 高 (100% 冗余) |

- **Active-Active（双活 / 多活）**：
  - 允许多个集群同时接收写流量。同一行记录（Row）的并发写入是核心难点，跨地域合并冲突代价极高（LWW 最后写入胜出易丢数据，CRDT 仅适用于特定增量场景）。工业界生产实践通常采用**单写归属（Single-Writer Ownership / 基于用户 ID 按地域划分数据所有权）**，避免跨洋分布式锁；
- **法定人数共识 Quorum（N/W/R）**：
  - 配置写入节点数 $W$、读取节点数 $R$ 与副本总数 $N$。满足 $W + R > N$ 时确保读写集合必然重叠。但需注意：Quorum 本身仅保证能读到最新版本号，无法自动消除并发写冲突，更不自动提供严格可线性化（Linearizability），需结合 Raft/Paxos 状态机。

### 5.4 · 副本不是备份 (Replication != Backup)
- **核心本质差异**：复制（Replication）保障的是服务的高可用（Availability），能够秒级抵御硬件停机；但复制会以毫秒级速度将人为误删（`DROP TABLE`）、程序 Bug 或脏数据广播到所有副本。
- **备份（Backup）保障的是可恢复性（Durability & Recoverability）**：通过定期冷快照（Snapshot）与不可变的增量 WAL 日志流，提供 Point-in-Time 任意时间点回滚能力。
- **核心容灾指标**：
  - **RPO (Recovery Point Objective)**：系统发生灾难时容忍丢失的最大数据时间窗口（例如：$\text{RPO} \le 1\text{ min}$）；
  - **RTO (Recovery Time Objective)**：系统从故障发生到恢复对外服务所花费的最大时间（例如：$\text{RTO} \le 5\text{ min}$）。

---

## 6 · 架构选型与权衡决策矩阵

| 业务核心诉求 | 推荐首选架构方案 | 核心工程代价与需明确的折中 |
|---|---|---|
| **读密集型（Read-Heavy）** | Primary + 多 Read Replicas (X 轴) | 容忍复制延迟（Replication Lag），写后读需路由回主库 |
| **写密集型（Write-Heavy）** | 水平分片 Sharding (Z 轴) | 承受跨分片查询、跨分片 2PC 事务与分布式 JOIN 复杂度 |
| **单行不变量（Single-Row）** | 分布式 KV / Document Store | 放弃复杂关系模型，需应用层维护数据完整性 |
| **跨实体复杂事务（Cross-Row）**| 关系型数据库 (RDBMS) / Spanner | 单机伸缩受限，需承担分布式事务延迟与死锁风险 |
| **多团队敏捷与异构负载** | 功能垂直拆库 (Y 轴) | 丧失跨库本地 ACID，需实现 Saga / Outbox 补偿对账 |
| **全球低延迟多活写入** | 单元化分区所有权 (Geo-Partition Ownership) | 必须严格划分数据归属，严禁跨地域并发冲突写入 |

---

## 7 · 一手资料

- [PostgreSQL Documentation: High Availability, Load Balancing, and Replication](https://www.postgresql.org/docs/current/high-availability.html)
- [MySQL 8.0 Reference Manual: Replication](https://dev.mysql.com/doc/refman/8.0/en/replication.html)
- [MongoDB Manual: Sharding Architecture](https://www.mongodb.com/docs/manual/sharding/)
- [Dynamo: Amazon's Highly Available Key-value Store (SOSP 2007)](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
- [Cassandra - A Decentralized Structured Storage System (LADIS 2009)](https://www.cs.cornell.edu/projects/ladis2009/papers/lakshman-ladis2009.pdf)
- [Spanner: Google's Globally-Distributed Database (OSDI 2012)](https://static.googleusercontent.com/media/research.google.com/en//archive/spanner-osdi2012.pdf)
- [Vitess: Scalable Database Clustering System for MySQL](https://vitess.io/)
