# System Design 06 · 消息队列

课程位置：[[SystemDesign04 Storage Systems|04 存储系统]] → 本篇 → [[SystemDesign09 Consistent Hashing|09 一致性哈希]]

异步架构重新安排了责任转移的时间点。它并不保证一个函数本身执行得更快。核心问题是：如果在当前进程立刻崩溃，在哪个时刻能够确定任务已被持久接管？

```async-messaging-architecture-visual
```

## 1 · 核心 API 与两个确认

消息系统的高层 API 通常围绕投递和消费展开：
- **Produce**：将消息写入系统。
- **Consume**：从系统中获取消息。
- **Ack (Acknowledge)**：确认消息处理成功。
- **Nack / Requeue**：处理失败，要求系统退回并重试。
- **Seek / Commit Offset**：定位或提交当前的消费进度。

系统中存在两个截然不同的确认（Ack）：

```text
Client -> API: 发起请求
API -> Queue: 将任务入队
Queue -> API: Broker 确认 (Durable Ack)
API -> Client: 202 Accepted 并返回 job_id

Queue -> Worker: 派发任务
Worker -> DB: 执行业务写入
Worker -> Queue: Consumer 确认 (Worker Ack)
```

1. **Broker 确认**：Broker 接收到消息并持久化后，向 API 确认。这标志着消息系统正式接管任务。如果 API 在此之前就向客户端返回成功，那就是 Fire-and-Hope。
2. **Consumer 确认**：消费者完成业务副作用（例如写入 DB）后向 Broker 确认，表示该消息可以删除或推进位置。

## 2 · 拓扑模式与语义分类

分析异步场景时，需要分清拓扑结构和投递语义：

```text
1 to 1 (Point to Point):   Sender -> Queue -> One logical receiver
1 to N (Fan-out):          Publisher -> Topic -> Subscriber A / B / C
N to 1 (Fan-in):           Producers -> Collector -> Stream processor
N to N (Event Backbone):   Producers -> Event Bus/Log -> Consumer groups
```

即使拓扑都是 1-to-1，背后的语义也可能完全不同：可能是指定了唯一的接收者，可能是让任意 Worker 抢占执行，也可能是双向的 Request/Reply 模式。

### Queue、Pub/Sub 与 Log 语义对比

| 模型 | 特性 | 消费机制 |
|---|---|---|
| **Queue** (队列) | Competing Consumers 模式。一条消息只会被一个消费者成功处理。 | 多个 Worker 竞争抢占同一队列中的消息。 |
| **Pub/Sub** (发布/订阅) | Fan-out 模式。每一个逻辑订阅者（Subscriber）都能获得消息的完整拷贝。 | 每个订阅者各自获取副本，订阅者内部的 Worker 之间再竞争。 |
| **Partitioned Log** (日志) | 追加写入，独立于消费进度保留历史，支持多租户重放（Replay）。 | Kafka 等系统通过 Consumer Group 决定语义：同组是 Queue，不同组是 Pub/Sub。 |

Topic 只是一个名字或逻辑归属，并不直接决定底层的分发语义。

## 3 · Kafka 作为 Partitioned Log

Kafka 没有采用传统的 Queue 模式，而是将核心抽象换成了可保留、可定位、可重放的追加日志（Partitioned Log）。

- **Topic、Partition 与 Offset**：Producer 根据 Key 路由消息到特定的 Partition。Partition 内部保证追加顺序。Consumer 消费并保存自己所在的 Offset。消息的保留期（Retention）独立于消费进度。
- **并行度限制**：在传统的 Consumer Group 中，一个 Partition 同时只能被组内的一个 Consumer 实例独占消费，因此最大有效并行度 ≤ Partition 数量。
- **复制与容错**：每个 Partition 具有一个 Leader 和多个 Replicas。`acks` 参数与 `min.insync.replicas` 配置项共同决定了消息不丢失的法定人数（Quorum）。
- **优势场景**：跨系统数据管道（CDC）、历史重放（Replay）、多租户流式分析、以及需要按 Key 保证局部顺序的场景。
- **弱点与陷阱**：单条任务的细粒度确认（Per-task ack）、复杂的基于 Header 的路由规则不是其强项。如果遇到永远无法反序列化的 Poison Record，可能会彻底卡住要求严格按 Offset 推进的 Consumer。

## 4 · 投递、幂等与顺序

### 投递语义
- **At-most-once**：先推进 Offset，再处理。如果处理时崩溃，消息丢失。
- **At-least-once**：处理成功后再 Ack。若由于网络隔离造成 Ack 失败，系统会重发，导致重复。
- **Exactly-once effect**：精确一次效果。需要注意，Kafka 内部的事务保证无法涵盖外部系统。真正的 Exactly-once 需要依赖 Consumer 端对外部系统操作的幂等性。

### 幂等 (Idempotency)
幂等键应当是具备业务唯一性的标识（如 `order_id` 与 `status` 组合），绝不能使用 Payload Body 的 Hash。否则，当用户发起两笔金额、商品完全独立的订单时，Hash 查重会错误地丢弃第二笔订单。表怎么建、进行中怎么挡，见 [[SystemDesign01 Stateless Service|01]]。

### 局部有序
全局有序意味着必须将所有流量压到单一节点排队处理。实践中几乎总是妥协为**局部有序**。通过设置相同的 `partition_key`（例如 `user_id`），保证同一用户的操作进入同一 Partition 并按顺序处理。

### 死信队列 (DLQ)
无限重试会引发 Retry Storm。发生永久失败（如数据校验不过）的消息应进入死信队列 (DLQ)，以便保留原始 `event_id`、失败原因和堆栈，供后续人工审查（Inspect）和工具重放（Replay）。

## 5 · Outbox 模式与事件总线架构选型

### 5.1 什么是 Transactional Outbox 模式？

#### 双写困境 (The Dual-Write Problem)
在微服务与分布式系统中，持久化业务状态（写数据库）与异步发布事件（写消息队列）无法在缺乏重型分布式事务（如 2PC/XA）的情况下直接保持原子性：

```text
方案 A (先写 DB，再发 MQ):
1. 数据库本地事务提交: UPDATE orders SET status = 'PAID'
2. 调用 Broker API: publish(OrderPaidEvent)
--> 故障点: 若第 2 步因网络抖动、Broker 宕机或应用进程 OOM 崩溃，数据库变更已持久化，而下游永远丢失该事件，产生永久数据不一致。

方案 B (先发 MQ，再写 DB):
1. 调用 Broker API: publish(OrderPaidEvent)
2. 数据库本地事务提交: UPDATE orders SET status = 'PAID'
--> 故障点: 若第 2 步因唯一键冲突、死锁回滚或网络超时导致事务失败，下游系统已接收并消费了从未真正生效的“幽灵事件”。
```

分布式 2PC 协议因两阶段同步阻塞、极高延迟、单点协调者风险以及云原生中间件普遍不支持 XA，在高并发系统中已被基本弃用。

#### Outbox 核心机制
**Transactional Outbox 模式**通过降维解决双写问题：将“跨网络发布事件”降级为“单机数据库事务内的一次本地表插入”。在同一个本地 ACID 事务中完成状态修改与事件持久化：

```sql
BEGIN;

-- 1. 更新业务聚合根状态
UPDATE orders 
SET status = 'PAID', updated_at = now() 
WHERE order_id = :order_id AND status = 'PENDING';

-- 2. 在同一事务中插入事件记录到 Outbox 表
INSERT INTO outbox_events (
    event_id, 
    aggregate_type, 
    aggregate_id, 
    event_type, 
    payload, 
    created_at
) VALUES (
    :event_id, 
    'order', 
    :order_id, 
    'order.paid', 
    :payload_json, 
    now()
);

COMMIT;
```

利用单机数据库的原子性（Atomicity）与持久性（Durability），保证领域状态变更与待发送事件**要么同时持久化成功，要么同时回滚**，彻底消除分布式双写的窗口不一致。

#### Outbox Relay 中继投递机制
事件落盘至 Outbox 表后，需要外部中继（Relay）将事件发布给真正的订阅者或消息系统。主要有两种实现机制：

1. **轮询发布者 (Polling Publisher)**：
   后台定时 Worker 使用长轮询或定时任务扫描 Outbox 表，拉取未发送事件并投递给 Broker，发送成功后标记状态或删除行。
   - *优点*：纯 SQL 逻辑，易于理解与调试，无需额外依赖外部专用中间件。
   - *缺点*：高频轮询对数据库造成额外读压力；拉取间隔引入投递延迟；大批量轮询容易引发索引争抢。
2. **基于日志的变更数据捕获 (Log-based Change Data Capture, CDC)**：
   利用 CDC 工具（如 Debezium Outbox Event Router）直接监听并解析数据库底层追加的事务日志（如 MySQL Binlog、PostgreSQL WAL），流式提取对 `outbox_events` 表的行插入操作并即时转发至 Kafka 等流平台。
   - *优点*：零轮询 SQL 开销，延迟可达毫秒级，解耦应用层运行时，完全不侵入主库读写事务。
   - *缺点*：需部署并维护 Kafka Connect / Debezium 等管道基础设施，日志格式变更时存在运维与升级成本。

**投递语义规范**：无论通过 Polling 还是 CDC，Relay 在面临网络抖动、Broker 超时或节点重启时均会发起重试。因此 Outbox 模式天然只保证 **At-Least-Once（至少一次）** 交付，**下游消费者必须根据 `event_id` 或业务幂等键严格实现去重幂等**。

---

### 5.2 事件总线两大范式：DB as bus vs Queue / log as bus

在确定了由 Outbox 产生可靠事件后，事件的传递与消费总线存在两种经典架构范式：

| 方案 | 设计与适用场景 | 代价 |
|---|---|---|
| **DB as bus** | 业务事务内插入 Outbox；消费者维护持久游标，或以短事务领取任务并写入租约，提交后才执行外部 I/O | 需要清理历史、到期索引与消费者进度；多个逻辑订阅者要独立游标 / 任务行，不能共用一个消费标记 |
| **Queue / log as bus** | Outbox relay / CDC 将 event_id 发布到 Kafka；Router 消费、匹配订阅并生成 Delivery；保留原始事件供恢复 | 引入 Broker、复制和 lag 运维；relay 可能重复发布，Router 必须幂等 |

---

### 5.3 深入剖析方案一：DB as bus（数据库作为事件总线）

在单体应用、低吞吐或团队不想运维独立 MQ 集群的场景下，直接将关系型数据库作为事件总线与任务分发平台是务实的选择。

#### 核心实现机制

1. **基于租约与抢占的任务领取 (Job Lease Pattern)**：
   多个消费者并发争抢消费未处理任务时，必须使用短事务和排他行锁跳过已锁定行：
   ```sql
   BEGIN;

   -- 1. 抢占一批就绪任务，使用 SKIP LOCKED 避免消费者之间的锁等待
   SELECT event_id, payload 
   FROM outbox_events 
   WHERE status = 'READY' AND available_at <= now()
   ORDER BY available_at, event_id
   FOR UPDATE SKIP LOCKED 
   LIMIT 10;

   -- 2. 将其标记为执行中，并设置超时租约 (Lease Expiration)
   UPDATE outbox_events 
   SET status = 'RUNNING', 
       lease_expires_at = now() + INTERVAL '30 seconds', 
       attempts = attempts + 1 
   WHERE event_id = ANY(:claimed_ids);

   COMMIT;
   ```
   > **关键系统规范：提交后才执行外部 I/O**。
   > 绝对严禁在数据库事务内部发起外部 HTTP 请求、第三方 API 调用或耗时计算。若在持锁事务内发生慢网络等待，会导致数据库行锁长久不释放、连接池耗尽并引发全局雪崩。正确做法是短事务快速提交租约后，在事务外部执行 I/O，执行完毕再开启一个新短事务更新 `status = 'COMPLETED'`。若 Worker 中途崩溃，租约到期后该行自动被其他 Worker 重新抢占。

2. **基于持久游标的单调追加流 (Durable Cursor Stream)**：
   若事件只读不删，表保持纯追加。消费者在独立的元数据表记录每个订阅组当前成功消费的最大 `last_read_id`（自增主键），类似于数据库内的轻量 Offset：
   ```sql
   SELECT event_id, event_type, payload 
   FROM outbox_events 
   WHERE event_id > :last_read_id 
   ORDER BY event_id ASC 
   LIMIT 100;
   ```

#### 固有代价与现实瓶颈

- **历史清理与表膨胀 (Table Bloat)**：关系型数据库存储引擎并非为吞吐队列设计。频繁的 `UPDATE` 和 `DELETE` 会在 PostgreSQL 中生成大量 Dead Tuples，引发表膨胀、索引分裂与 Autovacuum 剧烈抖动。系统必须实现基于日期的分区裁剪（Partition Drop）或批量物理归档机制。
- **多订阅者拓展复杂度高**：在传统 MQ 中，一条消息可以通过不同 Consumer Group 天然广播给多个下游。而在 DB as bus 架构中，若有 5 个独立的订阅微服务，表中的一行无法简单通过单一的 `status` 字段满足所有消费者的进度标记。必须：
  - 方案 A：为每个订阅者维护独立的游标表（仅适用于事件持久不删的流式消费）；
  - 方案 B：在发布事件时，笛卡尔积插入 5 行独立的任务记录（写放大严重）。
- **容量与吞吐天花板**：单机数据库的事务与连接数有限，一般适合处理 QPS < 1,000~2,000 的任务派发。当并发吞吐量继续上升时，必须将总线能力迁移至独立的消息中间件。

---

### 5.4 深入剖析方案二：Queue / log as bus（消息队列 / 分区日志作为事件总线）

当微服务数量众多、吞吐量进入万级以上、或存在复杂多租户广播与历史数据重放需求时，应采用分区日志 / 消息队列作为中枢总线。

#### 核心实现机制

```text
[业务微服务] 
    │  (单机 ACID 事务)
    ├──► 业务表 (orders)
    └──► Outbox 表 (outbox_events)
            │
            ▼ (CDC 解析 Binlog / WAL)
     [Debezium / Relay]
            │
            ▼ (At-Least-Once 发布)
    [Kafka Partitioned Log]  (Topic: events.orders, Key: order_id)
            │
            ▼ (批量拉取消费)
     [Event Router] 
            │
            ├──► 匹配 Subscription 过滤规则
            ├──► 生成 Delivery 记录并持久化
            └──► 分发至下游微服务 / Webhook Worker
```

1. **CDC 驱动的管道化解耦**：
   Outbox 表仅作为事件在源数据库的“暂存缓冲区”，CDC 引擎在毫秒级将事件抽取并以 Key-Value 形式投递至 Kafka。业务数据库连接与外部通信彻底解耦。
2. **Router 与 Delivery 状态机解耦**：
   Router 负责从 Kafka 集中拉取事件，匹配多订阅者的订阅过滤条件（如 `event_type == 'order.paid' && amount > 1000`）。匹配成功后为每个订阅者生成一条 Delivery 任务，并放入下游工作队列。
3. **保留原始事件与历史回放 (Replayability)**：
   Kafka 的追加日志支持基于时间戳或 Offset 的持久化保留（如保留 7 天）。当下游微服务发布新特性需要回填历史全量事件，或者下游出现 Bug 修复后需重新计算时，可直接通过 `Seek(Offset)` 从过去任意位置重放事件流，完全不影响上游业务主库。

#### 固有代价与运维挑战

- **分布式中间件运维负担**：引入 Kafka/RocketMQ 意味着需要维护 Broker 集群、高可用选举、磁盘容量水位、以及多副本同步配置（如 `acks=all` 和 `min.insync.replicas=2`）。
- **Lag 监控与堆积治理**：必须建立完善的 `Consumer Lag` 和 `Oldest unacked age` 告警体系。当突发流量涌入或消费者发生 GC 暂停时，需要具备动态扩容 Partition 和消费者实例的能力。
- **端到端幂等必须强制落地**：分布式环境下网络分区、Broker 主从切换、Relay 重发以及 Router 重试均会引入重复数据。下游消费者必须在业务侧实现幂等（如使用分布式锁校验、数据库唯一索引 `ON CONFLICT DO NOTHING` 或状态机条件更新 `WHERE status = 'PREV'`）。

## 6 · 怎么选

| 系统 | 适用场景 | 核心机制 |
|---|---|---|
| **RabbitMQ** | 细粒度任务处理、复杂路由分发 | Exchange 路由，Queue 消费后即删 |
| **Kafka** | 海量数据管道、多次回放与流计算 | Partitioned Log，Offset 追踪，顺序 I/O |
| **Managed Queue** | 云原生环境下的默认首选 | API 驱动，免运维自动扩容 (SQS / PubSub) |
| **DB Table** | 单体架构或小规模服务 | 本地事务保证，`SKIP LOCKED` 避免竞争 |

决策表（6 行文本块）：

```text
1. Do you need strict history replay or stream analytics? -> Kafka
2. Do you need complex routing and per-message ack for tasks? -> RabbitMQ
3. Can you just use a cloud managed API? -> SQS / PubSub
4. Is your scale small? -> Database table queue (SKIP LOCKED)
```

## 7 · Event Bus 与 Router 水平扩缩容设计

**Event Bus** 是多对多事件分发路由器的系统抽象。它包含事件接入（Ingest）、规则匹配（Match）和可靠投递（Durable Delivery），并在架构上将**控制面**（租户注册 Webhook URL、事件订阅过滤规则）与**数据面**（高吞吐事件流转与分发）彻底解耦。

在分布式消息架构中，Event Router 的核心职责是从消息中间件（如 Kafka/Pulsar）拉取原始事件，匹配各租户的订阅配置，并将单条事件扇出（Fan-out）为多个下游微服务调用或外部 Webhook 的 Delivery 任务。

```text
Event Bus / Kafka ──► [Event Router] ──► 匹配 Subscription 规则 ──► Fan-out 生成多个 Delivery
```

---

### 7.1 Router 扩缩容的反模式与两层哈希间接层设计

#### 常见反模式：直接取模哈希
为实现多台 Router 实例的负载均衡，一种直觉做法是：
$$\text{target\_router} = \text{hash}(\text{tenant\_id}) \pmod{\text{router\_count}}$$

**致命缺陷（Rehashing Storm）**：
当 Router 实例从 4 台水平扩容到 5 台时，由于模数发生改变（从 4 变为 5），数学上将有 $\frac{4}{5} = 80\%$ 的 `tenant_id` 映射到全新的机器上。
这会导致海量租户瞬间发生机器漂移：
1. 实例内存中的局部缓存（订阅规则、TLS 连接池、限流令牌桶）大面积失效；
2. 正在进行中的批量发送被迫中断并在新机器上重发，引发广播风暴与大量重复投递。

#### 正确架构：两层哈希间接层（Fixed Partition Indirection）
工业级系统通过引入**固定分区（Fixed Partition）作为解耦间接层**，将“数据哈希空间”与“弹性计算实例”分离：

```text
tenant_id
   │
   ▼  (① 稳定的哈希算法，如 Murmur3 / CityHash)
partition_id   (固定数量 M，例如预分配 256 或 512 个物理分区)
   │
   ▼  (② 动态路由分配器 / Kafka Consumer Group Assignment)
Router 实例 (如 Router-1, Router-2, ...)
```

1. **第一层：数据到分区的映射恒定不变（Data -> Partition）**：
   $$\text{partition\_id} = \text{murmur3}(\text{tenant\_id}) \pmod M \quad (M = 256 \text{ 固定不变})$$
   同一租户的事件始终写入固定的 Partition，保证单租户内部事件的严格局部保序（FIFO）。
2. **第二层：分区到计算实例的弹性映射（Partition -> Router Instance）**：
   由分布式流引擎（如 Kafka Consumer Group）负责分区的动态分配与均衡：

**扩容演进示例**：
- **初始状态（4 台 Router，256 个 Partition）**：
  - `Router-1`：分配 P0 ~ P63（共 64 个）
  - `Router-2`：分配 P64 ~ P127（共 64 个）
  - `Router-3`：分配 P128 ~ P191（共 64 个）
  - `Router-4`：分配 P192 ~ P255（共 64 个）
- **触发水平扩容（Kubernetes 弹性伸缩）**：
  ```bash
  kubectl scale deployment webhook-router --replicas=8
  ```
- **扩容后状态（8 台 Router）**：
  - `Router-1` ~ `Router-8`：每台均匀分担 32 个 Partition。

**核心架构红利**：
`tenant_id -> partition_id` 的归属**零改变**。从 4 台增加到 8 台时，仅有恰好一半的分区发生所有权交接（Partition Handover），原机器上保留的 32 个分区缓存与状态完全不受影响，系统抖动降至理论最低限度。

---

### 7.2 Consumer Group Rebalance 与平滑交接协议 (Graceful Handover)

在 Kafka 架构中，Router 扩缩容本质上由 Consumer Group 的 Rebalance 机制驱动。为了防止交接过程中的数据堆积与丢件，必须实现严格的生命周期交接协议：

```text
[旧 Owner (Router-1)]                         [新 Owner (Router-5)]
        │                                              │
 1. 收到 Revoke 分区通知                                  │
 2. 暂停拉取 (Pause fetching)                            │
 3. 等待在途 Delivery 落盘 (Flush in-flight)              │
 4. 提交已完成 Offset (Commit completed offset)         │
 5. 释放分区所有权 ─────────────────────────────────────► 6. 收到 Assign 分区通知
                                                       7. 从 Broker 读取 Committed Offset
                                                       8. 恢复流消费 (Resume stream ingestion)
```

1. **旧节点交接协议 (Revocation Protocol)**：
   - 收到 Kafka `ConsumerRebalanceListener.onPartitionsRevoked()` 回调；
   - 立即对将被回收的 Partition 执行 `pause()`，停止摄入新数据；
   - 将内存中已经完成匹配但未入库的 Delivery 任务全部持久化（Flush in-flight progress）；
   - 同步提交最新的 Offset；
   - 确保外部网络调用均处于安全断点后，正式释放分区所有权。
2. **新节点接管协议 (Assignment Protocol)**：
   - 收到 `onPartitionsAssigned()` 回调；
   - 从 Broker 读取由上一个 Owner 提交的精确 Committed Offset；
   - 从该位点恢复拉取与扇出，继续推进消费。

在主流 Kafka 客户端中，推荐开启 **Cooperative Sticky Assignor**（渐进式协同再均衡），避免旧版本 Eager 协议中“全员暂停消费（Stop-The-World）”的缺陷，实现分区的平滑按需迁移。

---

### 7.3 防范脑裂与僵尸节点：Fencing Token 与 ownership_epoch

即便实现了上述交接协议，分布式系统仍然面临**假死与脑裂（Split-Brain）**的致命威胁。

#### 经典故障场景：僵尸节点写穿 (Zombie Node)
1. **Router-1 突发长 GC 停顿**（Full GC 停顿 30 秒）或发生短暂网络假死；
2. Kafka Broker 因长时间未收到心跳包，判定 Router-1 死亡，触发 Consumer Group Rebalance；
3. Broker 将原本属于 Router-1 的 `Partition-42` 重新分配给健康的新节点 `Router-2`；
4. `Router-2` 从上次提交的位点正常恢复，向数据库插入 Delivery 记录并向外部系统分发；
5. **故障点**：Router-1 从长 GC 停顿中苏醒！其自身进程并不知道已被集群驱逐，继续执行停顿前读入的一批消息，并向数据库提交 Delivery 写入！
6. **灾难后果**：Router-1 与 Router-2 同时作为同一 Partition 的 Owner 写入，产生并发脑裂与重复任务爆炸。

#### 解决方案：数据库层面的 Fencing Token (ownership_epoch)
通过在数据库持久化层引入**单调递增的代数（Epoch / Fencing Token）**与乐观锁校验，从根本上解决僵尸节点问题：

```text
Partition 租约与进度表 (partition_leases):
┌──────────────┬───────────────┬─────────────────┬──────────────────────┐
│ partition_id │ current_owner │ ownership_epoch │ last_committed_offset│
├──────────────┼───────────────┼─────────────────┼──────────────────────┤
│ 42           │ router-2      │ 108             │ 948210               │
└──────────────┴───────────────┴─────────────────┴──────────────────────┘
```

1. **接管递增 Epoch**：
   当新节点 `Router-2` 接管 `Partition-42` 时，在数据库事务中原子递增该分区的代数：
   ```sql
   UPDATE partition_leases 
   SET current_owner = 'router-2', 
       ownership_epoch = ownership_epoch + 1, 
       updated_at = now()
   WHERE partition_id = 42;
   ```
   此时 `Partition-42` 的 Epoch 从 `107` 跃升至 `108`。
2. **受检写入 (Fenced Writes)**：
   任何 Router 在向数据库插入该 Partition 派生的 Delivery 记录或更新 Offset 进度时，SQL 必须附带当前持有的 `epoch` 约束条件：
   ```sql
   INSERT INTO deliveries (delivery_id, partition_id, event_id, target_url, status)
   SELECT :delivery_id, 42, :event_id, :target_url, 'PENDING'
   WHERE EXISTS (
       SELECT 1 FROM partition_leases 
       WHERE partition_id = 42 AND ownership_epoch = :current_epoch
   );
   ```
3. **僵尸节点自然失效**：
   当僵尸节点 `Router-1` 苏醒并试图提交时，其请求携带旧的 `epoch = 107`。数据库校验条件不满足，写入受阻（更新/插入 0 行），彻底阻断由于假死引起的跨代脏写。

---

### 7.4 Webhook 可靠投递最佳实践

作为 Event Router 下游最常见的端点，外部 Webhook 属于不可信网络环境，必须实施以下系统防护：

1. **HMAC 防篡改签名**：
   在 HTTP 请求头中附加使用租户密钥生成的 HMAC-SHA256 签名（如 `X-Hub-Signature-256: sha256=...` 或 `Stripe-Signature: t=1614...,v1=...`）。包含时间戳以防御重放攻击（Replay Attack）。
2. **指数退避与抖动 (Exponential Backoff with Full Jitter)**：
   对 5xx 错误或网络超时实施渐进重试：
   $$T_{\text{wait}} = \min(T_{\max}, T_{\text{base}} \times 2^{\text{attempt}}) \times \text{Uniform}(0.5, 1.5)$$
   防止大批失败 Webhook 同时重试造成重试风暴（Thundering Herd）。
3. **租户舱壁隔离 (Tenant Bulkhead Isolation)**：
   若某租户的自建 Webhook 服务彻底宕机或响应延迟飙升至 30 秒，不能让其堵塞公共 Worker 线程池。必须按租户维度隔离消费并发度或独立队列分流，保障健康租户的事件交付不受慢租户影响。

## 8 · 观测指标与一手资料

**关键指标**：
- **Oldest unacked age**（最旧未确认消息延迟）：比只看 CPU 利用率更能反映系统是否落后。
- **Consumer lag**、**DLQ rate**（死信产生率）。

一手资料：
- [RabbitMQ: Consumer Acknowledgements and Publisher Confirms](https://www.rabbitmq.com/docs/confirms)
- [RabbitMQ: Quorum Queues](https://www.rabbitmq.com/docs/quorum-queues)
- [RabbitMQ: Native AMQP 1.0 and AMQP history](https://www.rabbitmq.com/blog/2024/08/05/native-amqp)
- [OASIS AMQP 1.0 Standard](https://www.oasis-open.org/standard/amqp/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Kafka: a Distributed Messaging System for Log Processing, NetDB 2011](https://www.odbms.org/2011/01/kafka-a-distributed-messaging-system-for-log-processing/)
- [Debezium Outbox Event Router](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html)
- [PostgreSQL SELECT: SKIP LOCKED](https://www.postgresql.org/docs/current/sql-select.html)
- [Amazon EventBridge: Event buses](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-event-bus.html)
- [GitHub webhook best practices](https://docs.github.com/en/webhooks/using-webhooks/best-practices-for-using-webhooks)
- [Stripe webhook best practices](https://docs.stripe.com/webhooks)
