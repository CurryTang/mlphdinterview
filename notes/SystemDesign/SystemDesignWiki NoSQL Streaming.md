# Wiki · NoSQL + Streaming (NoSQL 变更流与事件驱动计算)

Wiki 词条归属：[[SystemDesign00 Overview|00 系统设计全局蓝图]] → [[SystemDesignWiki NoSQL Streaming|Wiki · NoSQL + Streaming]]

**NoSQL + Streaming（NoSQL 数据库与流处理结合）** 是现代云原生与高并发架构中最关键的数据流模式。它通过将 NoSQL 存储的变更捕获（CDC）与流式计算引擎相结合，实现了**单一事实来源下的高吞吐持久化与亚秒级实时事件驱动**。

---

## 1 · 核心设计理念与架构模式

### 1.1 · 消除双写困境：内建变更数据捕获（Built-in CDC）
在微服务架构中，同时向数据库写入业务状态并向消息队列发送事件（双写）极易因网络抖动造成数据不一致。
NoSQL + Streaming 范式彻底摒弃应用层双写：
- **单一写入路径**：业务服务只发起一次对 NoSQL 数据库的持久化写入（例如 DynamoDB `PutItem` 或 MongoDB `updateOne`）。
- **引擎内原生 CDC 捕获**：NoSQL 存储引擎内部将物理事务提交日志（如 DynamoDB Storage Nodes Log、MongoDB Oplog、Cassandra CommitLog、Redis Streams WAL）无损且保序地转换为可供流式消费的**变更事件流（Change Stream）**。

```text
[Business Service]
        │
        ▼ (Single Direct Write)
┌──────────────────────────────────────┐
│  NoSQL Primary Store                 │
│  (e.g., DynamoDB / MongoDB / Scylla) │
│  ┌────────────────────────────────┐  │
│  │ Primary Key Indexed Documents  │  │
│  └────────────────────────────────┘  │
└──────────────────┬───────────────────┘
                   │
                   ▼ (Built-in Sharded Log Stream)
           [Change Stream] (DynamoDB Streams / Oplog)
                   │
                   ├───────────────────────┬───────────────────────┐
                   ▼                       ▼                       ▼
           [Search Index Sync]     [Cache Invalidation]    [Real-time Analytics]
             (Elasticsearch)             (Redis)              (Flink / OLAP)
```

### 1.2 · 流表二象性（Stream-Table Duality / Kappa 架构）
- **Table 是状态在某一时间截面的静态快照**；
- **Stream 是导致状态不断演进的所有有序变更事件序列**：
  $$\text{Table} = \int \text{Stream} \, dt, \quad \text{Stream} = \frac{d(\text{Table})}{dt}$$
- **前后镜像全记录（OldImage vs NewImage）**：流事件天然附带修改前快照（`OldImage`）与修改后快照（`NewImage`），使得下游消费者能够精准识别哪些字段发生变化、计算增量差异，而无需回查主库。
- **单分片绝对保序**：针对同一个主键（Partition Key）的所有修改事件，在变更流中保持严格的 FIFO 时序。

### 1.3 · 物化视图与异构存储多写（Polyglot Persistence via CQRS）
利用 NoSQL + Streaming 可轻松构建 CQRS（命令查询职责分离）架构：
- **写优化主库**：NoSQL 主库只针对高频并发写入、主键查询和事务隔离进行极致优化。
- **异构读模型物化**：流处理 Worker（AWS Lambda、Flink、Kafka Connect）消费变更流，异步将最新数据同步到专门的查询引擎中：
  - 同步到 **Elasticsearch** 支持多维度分词与组合检索；
  - 同步到 **Redis** 维护热点用户的高性能只读视图；
  - 同步到 **ClickHouse / Snowflake** 进行实时数仓分析与聚合报表计算。

---

## 2 · 吞吐量级与性能基准（QPS & Latency）

NoSQL 变更流通常与底层数据库的分片结构（Partitions / Shards）按 1:1 动态绑定，具备近乎线性的扩展能力：

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   NoSQL + Streaming 典型性能阶梯                        │
├─────────────────────────┬──────────────────────┬───────────────────────┤
│ 组件与流技术            │ 变更流 QPS 承载能力   │ 流传播延迟 (Latency)  │
├─────────────────────────┼──────────────────────┼───────────────────────┤
│ AWS DynamoDB Streams    │ 100,000 - 1,000,000+ │ 50 - 200 毫秒 (P99<1s)│
│ MongoDB Change Streams  │ 20,000 - 50,000 /RS  │ 10 - 50 毫秒          │
│ Redis Streams           │ 100,000 - 500,000/node│ < 1 毫秒 (微秒级)    │
│ ScyllaDB / Cassandra CDC│ 500,000 - 2,000,000+ │ 20 - 100 毫秒         │
└─────────────────────────┴──────────────────────┴───────────────────────┘
```

- **DynamoDB Streams**：流分片（Stream Shard）随着 DynamoDB 物理分区的分裂（Split）自动平行扩容，可平滑支撑百万级持续写入 QPS，且对主库读写吞吐零资源干扰；事件在流中默认保留 24 小时。
- **MongoDB Change Streams**：直接监听 WiredTiger 存储引擎的 `local.oplog.rs` 集合，读取不触发物理磁盘寻道；单 Replica Set 支持几万 QPS。
- **Redis Streams (`XADD` / `XREADGROUP`)**：纯单线程内存追加写，单节点可达数十万 QPS，提供持久化 Consumer Group 与 `ACK / XPENDING` 机制，微秒级延迟；受限于单机物理内存。

---

## 3 · 适用场景与面试系统设计选型

### 3.1 · 最佳适用场景
1. **真实数据源驱动的精准缓存淘汰（Cache Eviction）**：
   - 彻底解决“先更新 DB 再删缓存”或“先删缓存再写 DB”的并发脏读难题。
   - 应用程序只管修改 NoSQL，由流消费者根据变更流中的主键精准调用 `redis.del(key)`，杜绝缓存不一致。
2. **异构检索物化视图（Polyglot Search Sync）**：
   - 电商商品系统主数据落入 DynamoDB/MongoDB；
   - 变更流异步投递至 Elasticsearch，自动增量维护商品全文检索倒排索引。
3. **跨地域全球分布式多活（Global Tables Active-Active Replication）**：
   - DynamoDB Global Tables 底层核心引擎即由 DynamoDB Streams 驱动，在不同 Region 之间双向流转变更，结合时间戳（Last-Writer-Wins）仲裁并发冲突。
4. **实时风控与反作弊流式聚合（Real-Time Fraud Detection）**：
   - 用户交易流水落入 NoSQL，变更流即刻进入 Apache Flink，在 5 分钟滑动窗口内计算刷卡频次与金额方差，异常时实时切断账户权限。

### 3.2 · 架构禁忌与反模式（When NOT to Use）
- **禁忌 1：纯瞬态计算调度**：若只是派发无需主数据持久化的计算任务（如无状态爬虫调度、离线转码触发），应直接使用 **Message Queue**，无需先在 NoSQL 落盘一条假记录再触发变更流。
- **禁忌 2：跨实体全序保证**：NoSQL 变更流通常只在**单个分区键或文档 ID 级别严格保序**。如果业务强烈要求跨不同实体的绝对物理全局全钟时序（Total Order），应依赖外部协调器或分布式追加日志（如 Kafka 单分区）。
