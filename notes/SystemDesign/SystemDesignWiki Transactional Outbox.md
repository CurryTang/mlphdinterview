# Wiki · Transactional Outbox (事务发件箱模式)

所属类别：Wiki 模式库 · 知识点
关联模式：[[SystemDesign06 Async Messaging Systems|消息队列]] · [[SystemDesign01 Stateless Service|幂等性保障]] · [[SystemDesign11 Notification System|通知系统实战]]

---

## 1 · 核心痛点：双写困境 (Dual-Write Hazard)

在分布式或事件驱动架构中，服务常常既要更新数据库状态，又要将事件投递至外部系统（Kafka、RabbitMQ 或 Webhook）。由于跨系统调用无法天然共享单机事务，直接双写存在不可避免的故障窗口：

* **先写 DB，后发 MQ**：若 DB 事务提交成功，但在发 MQ 时发生网络超时或进程宕机，外部系统将永远丢失该事件，导致系统状态永久不一致。
* **先发 MQ，后写 DB**：若 MQ 消息发送成功，但 DB 由于约束冲突、死锁或网络异常发生事务回滚，外部消费者却已经消费了该“虚假事件”（幽灵数据）。
* **分布式事务 (2PC / XA) 的局限**：协议复杂度极高、两阶段提交锁定数据时间长导致吞吐断崖式下跌，且绝大多数现代消息队列（如 Kafka）并不原生支持跨 DB 的 XA 强一致事务。

---

## 2 · 核心思想：借力本地单机 ACID 事务

Transactional Outbox 模式放弃在应用层协调“两套独立系统”，而是将外部交互退化为**单数据库内部事务**：

```text
[Business Service]
        │
        ▼ (Local ACID Transaction)
┌───────────────────────────────────────┐
│  Database                             │
│  ┌─────────────────┐ ┌──────────────┐ │
│  │ Business Table  │ │ Outbox Table │ │
│  │ (e.g. Order/Job)│ │ (Event Queue)│ │
│  └─────────────────┘ └──────────────┘ │
└───────────────────────────────────────┘
                     │
             (Asynchronous Read)
                     ▼
           [Message Relay / CDC]
                     │
                     ▼
          [Event Bus (Kafka/Pulsar)]
```

1. **同库同事务写入**：在数据库中除业务表外，额外维护一张 `outbox` 表。应用在执行业务变更的同时，将对应的事件 Payload 插入到 `outbox` 表中。两者被严格包裹在同一个**本地数据库事务 (Local ACID Transaction)** 内。
2. **原子持久化**：只要数据库事务提交，业务变更与事件记录要么全部成功落盘，要么全部回滚，彻底消除中间状态。
3. **异步中继派发**：由独立的中继组件（Message Relay）异步读取 `outbox` 表内容，并推送到真实的消息总线中。

---

## 3 · 两种中继派发机制与权衡 (Relay Mechanisms)

从 `outbox` 表搬运事件至消息中间件，业界有两种主流路径：

| 维度 | 轮询发布器 (Polling Publisher) | 事务日志挖掘 (Log Tailing / CDC) |
|---|---|---|
| **运作机制** | 后台 Worker 定期执行 SQL：<br>`SELECT * FROM outbox WHERE status = 'PENDING' ORDER BY created_at LIMIT 100 FOR UPDATE`，投递成功后更新状态或物理删除。 | 利用变更数据捕获工具（如 Debezium、Canal）直接监听数据库底层的提交日志（MySQL Binlog、PostgreSQL WAL）。捕获到 `outbox` 变更即转发 MQ。 |
| **数据库资源开销** | 高。频繁全表/索引扫描、加行锁，高并发下严重争抢主库连接池与 CPU。 | 极低。直接流式读取顺序磁盘日志，不向数据库主引擎下发查询与锁。 |
| **派发延迟** | 秒级（取决于轮询调度间隔与批次大小）。 | 毫秒级（几乎与事务提交实时同步）。 |
| **基础设施依赖** | 低。普通应用代码调度器即可，数据库中立。 | 较高。需部署与维护 CDC 集群（如 Kafka Connect / Debezium），依赖底层日志格式。 |
| **生产选型建议** | 小型低频系统、原型验证或单机批处理。 | **高并发与现代微服务生产环境的绝对首选标准**。 |

---

## 4 · 交付保证与系统边界代价 (Guarantees & Costs)

1. **交付语义：至少一次 (At-least-once Delivery)**
   * 由于网络抖动、MQ 确认超时或 Message Relay 重启，同一个事件可能被重复推入 Event Bus。
   * **强制约束**：下游消费者必须实现严格的**消费幂等性 (Consumer Idempotency)**。消费者通常基于 `event_id` 或 `biz_id + version` 在缓存或唯一索引表中进行防重校验。
2. **时序保障 (Ordering Guarantees)**
   * 为保证单实体事件的顺序性，Outbox 中继将事件投递到 Kafka 时，必须指定实体唯一标识（如 `order_id`）作为 **Partition Key**，确保同一实体的变更落在同一个 Partition，由下游消费者单线程串行处理。
3. **Outbox 表存储治理与表膨胀**
   * 若采用 Polling 模式且保留历史事件，必须配置按时间分区的归档脚本或物理定时清理任务（Purge Worker）。
   * 若采用 CDC 模式，可设计写入 `outbox` 后即由轻量后台任务通过主键范围截断删除（或在 Debezium 发送后自动删除），防止表数据无休止膨胀导致索引性能劣化。
